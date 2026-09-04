"""Phase 5 policy tests: deterministic, authoritative, fail-closed."""

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import generate_data as gd  # noqa: E402
from agent.graph import investigate  # noqa: E402
from agent.llm import DisabledLLM, ScriptedLLM  # noqa: E402
from agent.policy import (  # noqa: E402
    C1_INVALID_PROBABILITY,
    C2_INVALID_ATTEMPT_METADATA,
    C3_CONTRADICTORY_CASE_IDENTITY,
    DEFAULT_POLICY_CONFIG,
    REQUIRED_EVIDENCE_SOURCES,
    build_policy_request,
    consistency_violations,
    evaluate_policy,
)
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import (  # noqa: E402
    AttemptRecord,
    FailureRecord,
    InvestigationInput,
    PolicyConfig,
    PolicyRequest,
    TransactionDetails,
)
from datetime import datetime

DATA_FILES = ("customers.csv", "transactions.csv", "payment_attempts.csv", "payment_failures.csv")
EVALUATED_AT = datetime(2024, 1, 1, 12, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(**overrides) -> PolicyRequest:
    base = dict(
        requested_action="RETRY",
        recovery_probability=0.60,
        failure_reason="insufficient_funds",
        known_attempt_count=1,
        amount=100.00,
        prior_failed_count=1,
        prior_recovered_count=0,
        required_evidence_present=True,
        present_evidence_sources=list(REQUIRED_EVIDENCE_SOURCES),
        consistency_violations=[],
    )
    base.update(overrides)
    return PolicyRequest(**base)


class PolicyUnitTest(unittest.TestCase):
    """Pure evaluate_policy tests on constructed requests."""

    def test_retry_allowed_case(self):
        evaluation = evaluate_policy(_request(), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "ALLOWED")
        self.assertEqual(evaluation.final_action, "RETRY")
        self.assertTrue(evaluation.execution_authorized)
        self.assertEqual(evaluation.reason_codes, ["retry_allowed_within_policy"])

    def test_retry_denied_below_probability_floor(self):
        evaluation = evaluate_policy(_request(recovery_probability=0.20), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("retry_probability_below_policy_floor", evaluation.reason_codes)

    def test_denied_retry_routes_to_review(self):
        evaluation = evaluate_policy(_request(recovery_probability=0.20), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.final_action, "REVIEW")
        self.assertFalse(evaluation.execution_authorized)

    def test_review_requested_is_allowed_safe_path(self):
        evaluation = evaluate_policy(_request(requested_action="REVIEW"), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "ALLOWED")
        self.assertEqual(evaluation.final_action, "REVIEW")
        self.assertFalse(evaluation.execution_authorized)
        self.assertEqual(evaluation.reason_codes, ["review_is_safe_path"])

    def test_ignore_requested_is_allowed_no_action(self):
        evaluation = evaluate_policy(_request(requested_action="IGNORE"), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "ALLOWED")
        self.assertEqual(evaluation.final_action, "IGNORE")
        self.assertFalse(evaluation.execution_authorized)
        self.assertEqual(evaluation.reason_codes, ["ignore_requires_no_authorization"])

    def test_retry_cap_enforcement(self):
        at_cap = evaluate_policy(_request(known_attempt_count=4), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(at_cap.policy_decision, "DENIED")
        self.assertIn("retry_cap_exceeded", at_cap.reason_codes)
        below_cap = evaluate_policy(_request(known_attempt_count=3), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(below_cap.policy_decision, "ALLOWED")

    def test_high_value_guardrail(self):
        at_threshold = evaluate_policy(_request(amount=5000.00), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(at_threshold.policy_decision, "DENIED")
        self.assertIn("high_value_auto_retry_prohibited", at_threshold.reason_codes)
        self.assertEqual(at_threshold.final_action, "REVIEW")
        below = evaluate_policy(_request(amount=4999.99), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(below.policy_decision, "ALLOWED")

    def test_non_retryable_failure_guardrail(self):
        evaluation = evaluate_policy(_request(failure_reason="expired_card"), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("non_retryable_failure_reason", evaluation.reason_codes)
        self.assertNotIn("inconsistent_evidence", evaluation.reason_codes)

    def test_risk_review_hold_guardrail(self):
        evaluation = evaluate_policy(_request(failure_reason="risk_review_hold"), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("risk_review_hold_requires_human_review", evaluation.reason_codes)
        self.assertNotIn("non_retryable_failure_reason", evaluation.reason_codes)

    def test_missing_evidence_fails_closed(self):
        evaluation = evaluate_policy(_request(required_evidence_present=False), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("missing_required_evidence", evaluation.reason_codes)
        missing_prediction = evaluate_policy(_request(recovery_probability=None), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(missing_prediction.policy_decision, "DENIED")

    def test_all_guardrails_collected_not_first_match(self):
        evaluation = evaluate_policy(
            _request(recovery_probability=0.10, known_attempt_count=4, amount=9000.00),
            DEFAULT_POLICY_CONFIG,
            EVALUATED_AT,
        )
        self.assertIn("retry_cap_exceeded", evaluation.reason_codes)
        self.assertIn("high_value_auto_retry_prohibited", evaluation.reason_codes)
        self.assertIn("retry_probability_below_policy_floor", evaluation.reason_codes)
        self.assertEqual(len(evaluation.reason_codes), 3)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValidationError):
            PolicyConfig(policy_version="x", max_total_attempts=0)
        with self.assertRaises(ValidationError):
            PolicyConfig(policy_version="x", retry_probability_floor=1.5)

    def test_deterministic_repeated_evaluation(self):
        first = evaluate_policy(_request(), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        second = evaluate_policy(_request(), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(first, second)

    def test_policy_version_and_config_snapshot(self):
        config = PolicyConfig(policy_version="p5.test", max_total_attempts=2, retry_probability_floor=0.9)
        evaluation = evaluate_policy(_request(recovery_probability=0.60), config, EVALUATED_AT)
        self.assertEqual(evaluation.policy_version, "p5.test")
        self.assertEqual(evaluation.config_snapshot, config)
        self.assertEqual(evaluation.policy_decision, "DENIED")  # raised floor denies

    def test_kill_switch_denies_all_retries(self):
        config = PolicyConfig(policy_version="p5.kill", auto_retry_enabled=False)
        evaluation = evaluate_policy(_request(), config, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("auto_retry_disabled", evaluation.reason_codes)
        self.assertEqual(evaluation.final_action, "REVIEW")

    def test_invalid_requested_action_fails_closed(self):
        # Schema-level fail-closed: an out-of-vocabulary action cannot even
        # construct a PolicyRequest.
        with self.assertRaises(ValidationError):
            PolicyRequest(requested_action="EXECUTE_PAYMENT")
        # Defensive branch (bypassing validation): the policy denies, records
        # invalid_requested_action, and records the safe coerced action.
        forged = PolicyRequest.model_construct(
            requested_action="EXECUTE_PAYMENT",
            recovery_probability=0.6,
            failure_reason="insufficient_funds",
            known_attempt_count=1,
            amount=100.0,
            required_evidence_present=True,
            consistency_violations=[],
        )
        evaluation = evaluate_policy(forged, DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("invalid_requested_action", evaluation.reason_codes)
        self.assertFalse(evaluation.execution_authorized)
        self.assertEqual(evaluation.final_action, "REVIEW")


class ConsistencyCheckTests(unittest.TestCase):
    """C1/C2/C3 — precise, mechanical, and distinct from policy violations."""

    def _attempt(self, **overrides):
        base = dict(
            attempt_id="ATT-0000001",
            transaction_id="TXN-0000001",
            attempt_number=1,
            attempted_at=datetime(2024, 1, 1, 12, 0),
            status="failed",
            payment_method="card",
        )
        base.update(overrides)
        return AttemptRecord(**base)

    def _failure(self, **overrides):
        base = dict(
            failure_id="FAIL-0000001",
            attempt_id="ATT-0000001",
            transaction_id="TXN-0000001",
            customer_id="CUST-000001",
            failed_at=datetime(2024, 1, 1, 12, 0),
            failure_reason="insufficient_funds",
            processor_response_code="51",
        )
        base.update(overrides)
        return FailureRecord(**base)

    def _details(self, **overrides):
        base = dict(
            transaction_id="TXN-0000001",
            customer_id="CUST-000001",
            created_at=datetime(2024, 1, 1, 11, 0),
            amount=100.0,
            currency="USD",
            payment_method="card",
        )
        base.update(overrides)
        return TransactionDetails(**base)

    def test_c1_invalid_probability(self):
        violations = consistency_violations("TXN-0000001", 1.5, self._attempt(), self._failure(), self._details())
        self.assertIn(C1_INVALID_PROBABILITY, violations)
        clean = consistency_violations("TXN-0000001", 0.5, self._attempt(), self._failure(), self._details())
        self.assertNotIn(C1_INVALID_PROBABILITY, clean)

    def test_c2_non_initial_attempt_metadata(self):
        violations = consistency_violations(
            "TXN-0000001", 0.5, self._attempt(attempt_number=2), self._failure(), self._details()
        )
        self.assertIn(C2_INVALID_ATTEMPT_METADATA, violations)
        violations = consistency_violations(
            "TXN-0000001", 0.5, self._attempt(), self._failure(attempt_id="ATT-0000009"), self._details()
        )
        self.assertIn(C2_INVALID_ATTEMPT_METADATA, violations)

    def test_c3_contradictory_case_identity(self):
        violations = consistency_violations(
            "TXN-0000001", 0.5, self._attempt(transaction_id="TXN-0000002"), self._failure(), self._details()
        )
        self.assertIn(C3_CONTRADICTORY_CASE_IDENTITY, violations)
        violations = consistency_violations(
            "TXN-0000001", 0.5, self._attempt(), self._failure(transaction_id="TXN-0000002"), self._details()
        )
        self.assertIn(C3_CONTRADICTORY_CASE_IDENTITY, violations)
        violations = consistency_violations(
            "TXN-0000001", 0.5, self._attempt(), self._failure(customer_id="CUST-999999"), self._details()
        )
        self.assertIn(C3_CONTRADICTORY_CASE_IDENTITY, violations)

    def test_c3_mismatch_propagates_to_policy(self):
        evaluation = evaluate_policy(
            _request(consistency_violations=[C3_CONTRADICTORY_CASE_IDENTITY]), DEFAULT_POLICY_CONFIG, EVALUATED_AT
        )
        self.assertEqual(evaluation.policy_decision, "DENIED")
        self.assertIn("inconsistent_evidence", evaluation.reason_codes)

    def test_expired_card_is_policy_violation_not_inconsistency(self):
        violations = consistency_violations("TXN-0000001", 0.5, self._attempt(), self._failure(), self._details())
        self.assertEqual(violations, [])
        evaluation = evaluate_policy(_request(failure_reason="expired_card"), DEFAULT_POLICY_CONFIG, EVALUATED_AT)
        self.assertIn("non_retryable_failure_reason", evaluation.reason_codes)
        self.assertNotIn("inconsistent_evidence", evaluation.reason_codes)


class PolicyGraphIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "data"
        gd.write_dataset(gd.generate_dataset(random.Random(42), 800), cls.data_dir)
        cls.model_path = str(REPO_ROOT / "models" / "baseline" / "model.json")
        _X, _y, meta = _get_features(cls.data_dir)
        cls.case_ids = meta.loc[meta["split"] == "test", "transaction_id"].tolist()[:25]
        cls.results = {
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

    def test_every_result_carries_policy_evaluation(self):
        for result in self.results.values():
            self.assertIsNotNone(result.policy_evaluation)
            self.assertEqual(result.policy_evaluation.requested_action, result.recommendation.action)
            self.assertEqual(
                result.policy_evaluation.evaluated_at, result.prediction.prediction_time
            )
            self.assertEqual(result.policy_evaluation.policy_version, DEFAULT_POLICY_CONFIG.policy_version)
            self.assertEqual(result.policy_evaluation.config_snapshot, DEFAULT_POLICY_CONFIG)

    def test_allowed_retry_via_graph(self):
        allowed = [r for r in self.results.values() if r.policy_evaluation.execution_authorized]
        self.assertTrue(allowed, "expected at least one policy-authorized RETRY in the sample")
        for result in allowed:
            self.assertEqual(result.recommendation.action, "RETRY")
            self.assertEqual(result.policy_evaluation.policy_decision, "ALLOWED")
            self.assertEqual(result.policy_evaluation.final_action, "RETRY")

    def test_gate_matrix(self):
        for result in self.results.values():
            policy = result.policy_evaluation
            action = result.recommendation.action
            if action == "RETRY" and policy.policy_decision == "ALLOWED":
                self.assertEqual(policy.final_action, "RETRY")
                self.assertTrue(policy.execution_authorized)
            elif action == "RETRY":
                self.assertEqual(policy.final_action, "REVIEW")
                self.assertFalse(policy.execution_authorized)
            elif action == "REVIEW":
                self.assertEqual(policy.policy_decision, "ALLOWED")
                self.assertEqual(policy.final_action, "REVIEW")
                self.assertFalse(policy.execution_authorized)
            else:
                self.assertEqual(policy.policy_decision, "ALLOWED")
                self.assertEqual(policy.final_action, "IGNORE")
                self.assertFalse(policy.execution_authorized)

    def test_allowed_does_not_imply_authorized(self):
        non_retry_allowed = [
            r.policy_evaluation
            for r in self.results.values()
            if r.policy_evaluation.policy_decision == "ALLOWED" and r.recommendation.action != "RETRY"
        ]
        self.assertTrue(non_retry_allowed)
        for policy in non_retry_allowed:
            self.assertFalse(policy.execution_authorized)

    def test_llm_disagreement_cannot_change_policy(self):
        transaction_id = self.case_ids[0]
        baseline = self.results[transaction_id]
        different = next(a for a in ("RETRY", "REVIEW", "IGNORE") if a != baseline.recommendation.action)
        probability = baseline.prediction.probability
        sources = [e.source for e in baseline.evidence if e.payload is not None]
        narrative = json.dumps(
            {
                "summary": f"Investigation of {transaction_id} suggests {different}.",
                "key_findings": [
                    {"statement": "Evidence supplied.", "evidence_references": ["failure_details"]}
                ],
                "supporting_evidence": sources,
                "uncertainty": "Model estimate.",
                "prediction_interpretation": f"The supplied recovery probability is {probability:.3f}.",
                "recommended_action": different,
                "confidence": 0.8,
                "evidence_references": sources,
            }
        )
        with_llm = investigate(
            InvestigationInput(
                transaction_id=transaction_id, data_dir=str(self.data_dir), model_path=self.model_path
            ),
            llm_client=ScriptedLLM([narrative]),
        )
        self.assertIsNotNone(with_llm.llm_review)
        self.assertFalse(with_llm.llm_review.agrees_with_deterministic)
        self.assertEqual(with_llm.policy_evaluation, baseline.policy_evaluation)

    def test_known_attempt_count_ignores_future_attempts(self):
        """Regression: future retries (attempts >= 2 in the dataset) must not
        affect known_attempt_count or the retry-cap decision."""
        multi_attempt_cases = []
        counts = self.attempts.groupby("transaction_id").size()
        for transaction_id in self.case_ids:
            if counts.get(transaction_id, 0) >= 3:
                multi_attempt_cases.append(transaction_id)
        self.assertTrue(multi_attempt_cases, "expected multi-attempt cases in the sample")
        for transaction_id in multi_attempt_cases:
            result = self.results[transaction_id]
            future_attempts = self.attempts.loc[
                (self.attempts["transaction_id"] == transaction_id)
                & (self.attempts["attempt_number"] != "1"),
                "attempt_id",
            ]
            self.assertTrue(len(future_attempts) >= 2, transaction_id)
            request = build_policy_request(
                {
                    "evidence": result.evidence,
                    "result": result,
                    "prediction": result.prediction,
                }
            )
            self.assertEqual(request.known_attempt_count, 1, transaction_id)
            serialized = request.model_dump_json()
            for attempt_id in future_attempts:
                self.assertNotIn(attempt_id, serialized)
            # The cap decision equals the decision for a case with no history
            # of future attempts: known_attempt_count=1 never exceeds the cap.
            self.assertNotIn("retry_cap_exceeded", result.policy_evaluation.reason_codes)
            recomputed = evaluate_policy(
                request, DEFAULT_POLICY_CONFIG, evaluated_at=result.prediction.prediction_time
            )
            self.assertEqual(recomputed, result.policy_evaluation)

    def test_data_files_byte_identical_after_policy_runs(self):
        before = {name: _sha256(self.data_dir / name) for name in DATA_FILES}
        for transaction_id in self.case_ids:
            investigate(
                InvestigationInput(
                    transaction_id=transaction_id, data_dir=str(self.data_dir), model_path=self.model_path
                ),
                llm_client=DisabledLLM(),
            )
        after = {name: _sha256(self.data_dir / name) for name in DATA_FILES}
        self.assertEqual(before, after)

    def test_no_execution_capability(self):
        self.assertEqual(
            set(_get_policy_evaluation_field_names()),
            {
                "requested_action",
                "policy_decision",
                "final_action",
                "reason_codes",
                "explanation",
                "applicable_guardrails",
                "policy_version",
                "config_snapshot",
                "execution_authorized",
                "evaluated_at",
            }
        )


def _get_policy_evaluation_field_names():
    from agent.schemas import PolicyEvaluation

    return set(PolicyEvaluation.model_fields.keys())


if __name__ == "__main__":
    unittest.main()
