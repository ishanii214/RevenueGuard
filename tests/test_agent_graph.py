"""End-to-end tests for the Phase 3 LangGraph investigation workflow."""

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import generate_data as gd  # noqa: E402
from agent.graph import (  # noqa: E402
    IGNORE_PROBABILITY_THRESHOLD,
    RETRY_PROBABILITY_THRESHOLD,
    investigate,
)
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import InvestigationInput, InvestigationResult  # noqa: E402

DATA_FILES = ("customers.csv", "transactions.csv", "payment_attempts.csv", "payment_failures.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_action(probability, failure_reason):
    if probability is None or failure_reason is None:
        return "REVIEW"
    if failure_reason in gd.AUTO_RETRY_REASONS and probability >= RETRY_PROBABILITY_THRESHOLD:
        return "RETRY"
    if probability < IGNORE_PROBABILITY_THRESHOLD:
        return "IGNORE"
    return "REVIEW"


class InvestigationGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "data"
        gd.write_dataset(gd.generate_dataset(random.Random(42), 800), cls.data_dir)
        cls.model_path = str(REPO_ROOT / "models" / "baseline" / "model.json")
        _X, _y, meta = _get_features(cls.data_dir)
        cls.case_ids = meta["transaction_id"].tolist()[:12]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _investigate(self, transaction_id):
        return investigate(
            InvestigationInput(transaction_id=transaction_id, data_dir=str(self.data_dir), model_path=self.model_path)
        )

    def test_investigation_returns_valid_typed_result(self):
        result = self._investigate(self.case_ids[0])
        self.assertIsInstance(result, InvestigationResult)
        self.assertEqual(result.transaction_id, self.case_ids[0])
        sources = [e.source for e in result.evidence]
        self.assertEqual(len(sources), len(set(sources)))
        self.assertEqual(
            set(sources),
            {
                "transaction_details",
                "customer_profile",
                "initial_attempt",
                "failure_details",
                "customer_history",
                "recovery_history",
                "recovery_prediction",
            },
        )
        self.assertTrue(result.findings)
        self.assertTrue(result.recommendation.rationale)

    def test_recommendation_follows_rule_table(self):
        actions_seen = set()
        for transaction_id in self.case_ids:
            result = self._investigate(transaction_id)
            failure = next(e for e in result.evidence if e.source == "failure_details")
            reason = failure.payload.failure_reason if failure.payload else None
            probability = result.prediction.probability if result.prediction else None
            expected = _expected_action(probability, reason)
            self.assertEqual(result.recommendation.action, expected, transaction_id)
            actions_seen.add(expected)
        self.assertTrue(actions_seen & {"RETRY", "REVIEW", "IGNORE"})

    def test_policy_check_required_and_no_execution(self):
        expected_keys = set(InvestigationResult.model_fields.keys())
        for transaction_id in self.case_ids:
            result = self._investigate(transaction_id)
            self.assertTrue(result.recommendation.policy_check_required)
            self.assertEqual(set(result.model_dump().keys()), expected_keys)
            serialized = result.model_dump_json()
            self.assertNotIn("executed", serialized.lower())
            # The Phase 5 policy gate (`execution_authorized`) expresses
            # authorization semantics only; the investigation itself never
            # executes anything and no past-tense execution fields exist.
            parsed = json.loads(serialized)
            self.assertNotIn("executed", json.dumps(list(parsed.keys())).lower())
            self.assertNotIn("payment_status", serialized.lower())

    def test_leakage_no_future_data_in_evidence(self):
        import pandas as pd

        attempts = pd.read_csv(self.data_dir / "payment_attempts.csv", dtype=str)
        for transaction_id in self.case_ids:
            result = self._investigate(transaction_id)
            serialized = result.model_dump_json()
            future_attempts = attempts.loc[
                (attempts["transaction_id"] == transaction_id)
                & (attempts["attempt_number"] != "1"),
                "attempt_id",
            ]
            for attempt_id in future_attempts:
                self.assertNotIn(attempt_id, serialized)
            self.assertNotIn("recovery_outcome", serialized)
            if result.prediction is not None:
                for evidence in result.evidence:
                    self.assertEqual(evidence.as_of, result.prediction.prediction_time)
            attempt_evidence = next(e for e in result.evidence if e.source == "initial_attempt")
            self.assertEqual(attempt_evidence.payload.attempt_number, 1)

    def test_unknown_transaction_is_handled_gracefully(self):
        result = self._investigate("TXN-9999999")
        self.assertIn("not found", " ".join(result.errors))
        self.assertTrue(all(e.payload is None for e in result.evidence if e.source == "transaction_details"))
        self.assertEqual(result.recommendation.action, "REVIEW")

    def test_deterministic_repeat_investigations(self):
        first = self._investigate(self.case_ids[1]).model_dump_json()
        second = self._investigate(self.case_ids[1]).model_dump_json()
        self.assertEqual(first, second)

    def test_data_files_byte_identical_after_investigations(self):
        before = {name: _sha256(self.data_dir / name) for name in DATA_FILES}
        for transaction_id in self.case_ids:
            self._investigate(transaction_id)
        after = {name: _sha256(self.data_dir / name) for name in DATA_FILES}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
