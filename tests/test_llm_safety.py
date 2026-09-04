"""Phase 4 LLM narration tests — fully deterministic (ScriptedLLM only).

No live model, API key, or network is required for this suite.
"""

import json
import random
import sys
import tempfile
import unittest
import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import generate_data as gd  # noqa: E402
from agent.graph import (  # noqa: E402
    _analyze,
    _gather_evidence,
    _load_case,
    _recommend,
    investigate,
)
from agent.llm import DisabledLLM, ScriptedLLM  # noqa: E402
from agent.narration import build_llm_evidence_bundle, parse_narrative  # noqa: E402
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import InvestigationInput  # noqa: E402
from agent.state import InvestigationState, initial_state  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

DATA_FILES = ("customers.csv", "transactions.csv", "payment_attempts.csv", "payment_failures.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LLMSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "data"
        gd.write_dataset(gd.generate_dataset(random.Random(42), 800), cls.data_dir)
        cls.model_path = str(REPO_ROOT / "models" / "baseline" / "model.json")
        _X, _y, meta = _get_features(cls.data_dir)
        cls.case_ids = meta["transaction_id"].tolist()[:6]
        cls.baseline = {
            transaction_id: investigate(
                InvestigationInput(
                    transaction_id=transaction_id, data_dir=str(cls.data_dir), model_path=cls.model_path
                ),
                llm_client=DisabledLLM(),
            )
            for transaction_id in cls.case_ids
        }
        cls.attempts = pd.read_csv(cls.data_dir / "payment_attempts.csv", dtype=str)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _baseline(self, transaction_id):
        return self.baseline[transaction_id]

    def _narrative_json(self, result, **overrides):
        probability = result.prediction.probability if result.prediction else 0.0
        sources = [e.source for e in result.evidence if e.payload is not None]
        payload = {
            "summary": (
                f"Failed payment {result.transaction_id} under investigation; the deterministic "
                f"recommendation is {result.recommendation.action}."
            ),
            "key_findings": [
                {
                    "statement": "The initial failure and transaction details are supplied in evidence.",
                    "evidence_references": ["failure_details", "transaction_details"],
                }
            ],
            "supporting_evidence": sources,
            "uncertainty": "Single-case evidence; the recovery probability is a model estimate.",
            "prediction_interpretation": (
                f"The supplied XGBoost recovery probability is {probability:.3f}."
            ),
            "recommended_action": result.recommendation.action,
            "confidence": 0.7,
            "evidence_references": sources,
        }
        payload.update(overrides)
        return json.dumps(payload)

    def _investigate_with(self, transaction_id, response):
        client = ScriptedLLM([response])
        result = investigate(
            InvestigationInput(
                transaction_id=transaction_id, data_dir=str(self.data_dir), model_path=self.model_path
            ),
            llm_client=client,
        )
        return result, client

    def test_input_bundle_safety(self):
        result = self._baseline(self.case_ids[0])
        bundle = build_llm_evidence_bundle(
            transaction_id=result.transaction_id,
            prediction=result.prediction,
            evidence=result.evidence,
            findings=result.findings,
            risk_flags=result.risk_flags,
            deterministic_recommendation=result.recommendation.action,
        )
        serialized = bundle.model_dump_json()
        self.assertNotIn("recovery_outcome", serialized)
        self.assertNotIn("model_path", serialized)
        self.assertNotIn("data_dir", serialized)
        self.assertNotIn('"split"', serialized)
        future_attempts = self.attempts.loc[
            (self.attempts["transaction_id"] == result.transaction_id)
            & (self.attempts["attempt_number"] != "1"),
            "attempt_id",
        ]
        for attempt_id in future_attempts:
            self.assertNotIn(attempt_id, serialized)
        self.assertEqual(bundle.allowed_actions, ["RETRY", "REVIEW", "IGNORE"])

    def test_llm_receives_only_sanitized_prompt(self):
        response = self._narrative_json(self._baseline(self.case_ids[0]))
        _result, client = self._investigate_with(self.case_ids[0], response)
        self.assertEqual(len(client.requests), 1)
        prompt = client.requests[0].user_prompt
        self.assertNotIn("recovery_outcome", prompt)
        self.assertNotIn("model_path", prompt)

    def test_valid_narrative_attached_as_advisory(self):
        baseline = self._baseline(self.case_ids[0])
        result, _client = self._investigate_with(self.case_ids[0], self._narrative_json(baseline))
        self.assertIsNotNone(result.llm_review)
        self.assertTrue(result.llm_review.advisory_only)
        self.assertTrue(result.llm_review.agrees_with_deterministic)
        self.assertEqual(result.recommendation.action, baseline.recommendation.action)
        self.assertEqual(
            result.prediction.probability,
            baseline.prediction.probability,
            "the LLM must not change the XGBoost probability",
        )
        self.assertTrue(result.llm_review.validation["grounding_ok"])

    def test_malformed_text_falls_back_safely(self):
        baseline = self._baseline(self.case_ids[0])
        result, _client = self._investigate_with(self.case_ids[0], "I am sorry, I cannot answer that.")
        self.assertIsNone(result.llm_review)
        self.assertTrue(any("llm narration unavailable" in error for error in result.errors))
        self.assertEqual(result.recommendation.action, baseline.recommendation.action)
        self.assertEqual(result.findings, baseline.findings)
        self.assertEqual(result.evidence, baseline.evidence)

    def test_out_of_vocabulary_action_rejected(self):
        baseline = self._baseline(self.case_ids[0])
        response = self._narrative_json(baseline, recommended_action="EXECUTE_PAYMENT")
        result, _client = self._investigate_with(self.case_ids[0], response)
        self.assertIsNone(result.llm_review)
        self.assertTrue(any("llm narration unavailable" in error for error in result.errors))
        self.assertEqual(result.recommendation.action, baseline.recommendation.action)

    def test_probability_mismatch_is_recorded_not_adopted(self):
        baseline = self._baseline(self.case_ids[0])
        false_probability = min(round(baseline.prediction.probability + 0.3, 3), 0.99)
        response = self._narrative_json(
            baseline,
            prediction_interpretation=f"The recovery probability is {false_probability:.3f}.",
        )
        result, _client = self._investigate_with(self.case_ids[0], response)
        self.assertIsNotNone(result.llm_review)
        self.assertFalse(result.llm_review.validation["grounding_ok"])
        self.assertTrue(
            any("probability" in violation for violation in result.llm_review.validation["violations"])
        )
        self.assertEqual(result.prediction.probability, baseline.prediction.probability)

    def test_unsupplied_reference_flagged(self):
        baseline = self._baseline(self.case_ids[0])
        response = self._narrative_json(baseline, evidence_references=["audit_logs"])
        result, _client = self._investigate_with(self.case_ids[0], response)
        self.assertIsNotNone(result.llm_review)
        self.assertFalse(result.llm_review.validation["grounding_ok"])
        self.assertTrue(
            any("unsupplied evidence" in violation for violation in result.llm_review.validation["violations"])
        )

    def test_invented_identifier_flagged(self):
        baseline = self._baseline(self.case_ids[0])
        response = self._narrative_json(
            baseline,
            summary=f"Related transaction TXN-9999999 failed last week. {baseline.recommendation.action} suggested.",
        )
        result, _client = self._investigate_with(self.case_ids[0], response)
        self.assertIsNotNone(result.llm_review)
        self.assertTrue(
            any(
                "unsupplied identifier" in violation
                for violation in result.llm_review.validation["violations"]
            )
        )

    def test_contradictory_failure_reason_flagged(self):
        baseline = self._baseline(self.case_ids[0])
        supplied_reason = next(
            e.payload.failure_reason for e in baseline.evidence if e.source == "failure_details"
        )
        other_reason = next(r for r in gd.FAILURE_REASON_CODES if r != supplied_reason)
        response = self._narrative_json(
            baseline,
            key_findings=[
                {
                    "statement": f"The initial failure reason was {other_reason}.",
                    "evidence_references": ["failure_details"],
                }
            ],
        )
        result, _client = self._investigate_with(self.case_ids[0], response)
        self.assertIsNotNone(result.llm_review)
        self.assertTrue(
            any(
                "failure reason" in violation
                for violation in result.llm_review.validation["violations"]
            )
        )

    def test_grounding_scope_documents_interpretive_limitation(self):
        baseline = self._baseline(self.case_ids[0])
        result, _client = self._investigate_with(self.case_ids[0], self._narrative_json(baseline))
        scope = result.llm_review.validation["grounding_scope"]
        self.assertIn("mechanical", scope)
        self.assertIn("not all semantic", scope)

    def test_disagreement_recorded_and_deterministic_action_preserved(self):
        for transaction_id in self.case_ids:
            baseline = self._baseline(transaction_id)
            different = next(a for a in ("RETRY", "REVIEW", "IGNORE") if a != baseline.recommendation.action)
            response = self._narrative_json(baseline, recommended_action=different)
            result, _client = self._investigate_with(transaction_id, response)
            self.assertIsNotNone(result.llm_review)
            self.assertFalse(result.llm_review.agrees_with_deterministic)
            self.assertEqual(result.llm_review.llm_recommendation, different)
            self.assertEqual(result.llm_review.deterministic_recommendation, baseline.recommendation.action)
            self.assertEqual(result.recommendation.action, baseline.recommendation.action)

    def test_fallback_on_llm_exception(self):
        baseline = self._baseline(self.case_ids[1])
        result, _client = self._investigate_with(self.case_ids[1], TimeoutError("llm timed out"))
        self.assertIsNone(result.llm_review)
        self.assertTrue(any("llm narration unavailable" in error for error in result.errors))
        self.assertEqual(result.recommendation.action, baseline.recommendation.action)

    def test_fallback_on_disabled_llm(self):
        baseline = self._baseline(self.case_ids[1])
        result = investigate(
            InvestigationInput(
                transaction_id=self.case_ids[1], data_dir=str(self.data_dir), model_path=self.model_path
            ),
            llm_client=DisabledLLM(),
        )
        self.assertIsNone(result.llm_review)
        self.assertTrue(any("llm disabled" in error for error in result.errors))
        self.assertEqual(result.recommendation.action, baseline.recommendation.action)

    def test_no_llm_path_equivalent_to_phase3_graph(self):
        """The no-LLM run must match the Phase 3 graph exactly (the LLM nodes
        reduce to a single recorded error note)."""
        for transaction_id in self.case_ids[:3]:
            investigation_input = InvestigationInput(
                transaction_id=transaction_id, data_dir=str(self.data_dir), model_path=self.model_path
            )
            phase3_graph = StateGraph(InvestigationState)
            phase3_graph.add_node("load_case", _load_case)
            phase3_graph.add_node("gather_evidence", _gather_evidence)
            phase3_graph.add_node("analyze", _analyze)
            phase3_graph.add_node("recommend", _recommend)
            phase3_graph.add_edge(START, "load_case")
            phase3_graph.add_edge("load_case", "gather_evidence")
            phase3_graph.add_edge("gather_evidence", "analyze")
            phase3_graph.add_edge("analyze", "recommend")
            phase3_graph.add_edge("recommend", END)
            phase3_result = phase3_graph.compile().invoke(initial_state(investigation_input))["result"]
            phase4_result = investigate(investigation_input, llm_client=DisabledLLM())
            self.assertIsNone(phase3_result.llm_review)
            self.assertIsNone(phase4_result.llm_review)
            base = phase4_result.model_dump()
            reference = phase3_result.model_dump()
            narration_note = base["errors"][-1] if base["errors"] else None
            self.assertIn("llm narration unavailable", narration_note or "")
            base["errors"] = base["errors"][:-1]
            self.assertEqual(base, reference)

    def test_data_files_byte_identical_after_llm_runs(self):
        before = {name: _sha256(self.data_dir / name) for name in DATA_FILES}
        for transaction_id in self.case_ids:
            self._investigate_with(transaction_id, self._narrative_json(self._baseline(transaction_id)))
        after = {name: _sha256(self.data_dir / name) for name in DATA_FILES}
        self.assertEqual(before, after)

    def test_parse_narrative_handles_fenced_json(self):
        baseline = self._baseline(self.case_ids[2])
        fenced = "```json\n" + self._narrative_json(baseline) + "\n```"
        narrative, error = parse_narrative(fenced)
        self.assertIsNone(error)
        self.assertIsNotNone(narrative)


if __name__ == "__main__":
    unittest.main()
