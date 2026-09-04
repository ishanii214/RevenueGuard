"""Tests for the Phase 3 deterministic investigation tools."""

import random
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import generate_data as gd  # noqa: E402
from agent.data_repository import CaseRepository, get_repository  # noqa: E402
from agent.prediction import get_recovery_prediction  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _parse(series):
    return pd.to_datetime(series, format=TIMESTAMP_FORMAT)


class InvestigationToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "data"
        gd.write_dataset(gd.generate_dataset(random.Random(42), 800), cls.data_dir)
        cls.repo = get_repository(cls.data_dir)
        cls.transactions = pd.read_csv(cls.data_dir / "transactions.csv", dtype=str)
        cls.attempts = pd.read_csv(cls.data_dir / "payment_attempts.csv", dtype=str)
        cls.failures = pd.read_csv(cls.data_dir / "payment_failures.csv", dtype=str)
        cls.attempts["attempted_at"] = _parse(cls.attempts["attempted_at"])
        cls.failures["failed_at"] = _parse(cls.failures["failed_at"])
        cls.model_path = REPO_ROOT / "models" / "baseline" / "model.json"
        failed = cls.transactions.loc[cls.transactions["status"] == "failed", "transaction_id"]
        cls.sample_failed_id = failed.iloc[0]
        cls.sample_row = cls.transactions.set_index("transaction_id").loc[cls.sample_failed_id]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_get_transaction_without_label(self):
        details = self.repo.get_transaction(self.sample_failed_id)
        self.assertIsNotNone(details)
        data = details.model_dump()
        self.assertNotIn("recovery_outcome", data)
        self.assertEqual(details.transaction_id, self.sample_failed_id)
        self.assertEqual(details.customer_id, self.sample_row["customer_id"])
        self.assertAlmostEqual(details.amount, float(self.sample_row["amount"]))

    def test_get_transaction_unknown_id(self):
        self.assertIsNone(self.repo.get_transaction("TXN-9999999"))
        self.assertIsNone(self.repo.get_transaction("not-an-id"))

    def test_get_customer(self):
        profile = self.repo.get_customer(self.sample_row["customer_id"])
        self.assertIsNotNone(profile)
        self.assertEqual(profile.customer_id, self.sample_row["customer_id"])
        self.assertIn(profile.customer_segment, gd.SEGMENTS)
        self.assertIsNone(self.repo.get_customer("CUST-999999"))

    def test_get_initial_attempt_is_attempt_one(self):
        attempt = self.repo.get_initial_attempt(self.sample_failed_id)
        self.assertIsNotNone(attempt)
        self.assertEqual(attempt.attempt_number, 1)
        self.assertEqual(attempt.transaction_id, self.sample_failed_id)
        expected = self.attempts.set_index("transaction_id").loc[self.sample_failed_id]
        expected = expected[expected["attempt_number"] == "1"].iloc[0]
        self.assertEqual(attempt.attempt_id, expected["attempt_id"])
        self.assertEqual(attempt.status, "failed")
        self.assertIsNone(self.repo.get_initial_attempt("TXN-9999999"))

    def test_get_initial_failure_matches_initial_attempt(self):
        failure = self.repo.get_initial_failure(self.sample_failed_id)
        attempt = self.repo.get_initial_attempt(self.sample_failed_id)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.attempt_id, attempt.attempt_id)
        self.assertEqual(failure.transaction_id, self.sample_failed_id)
        self.assertIn(failure.failure_reason, gd.FAILURE_REASON_CODES)
        self.assertEqual(failure.processor_response_code, gd.FAILURE_REASON_CODES[failure.failure_reason])
        self.assertIsNone(self.repo.get_initial_failure("TXN-9999999"))

    def test_customer_history_matches_independent_asof_recomputation(self):
        first_fail = self.attempts.loc[self.attempts["status"] == "failed"].groupby("transaction_id")["attempted_at"].min()
        first_success = (
            self.attempts.loc[self.attempts["status"] == "succeeded"].groupby("transaction_id")["attempted_at"].min()
        )
        txn = self.transactions.copy()
        txn["first_fail_time"] = txn["transaction_id"].map(first_fail)
        txn["first_success_time"] = txn["transaction_id"].map(first_success)
        txn["created_at"] = _parse(txn["created_at"])

        prediction = get_recovery_prediction(self.sample_failed_id, self.data_dir, self.model_path)
        self.assertIsNotNone(prediction)
        as_of = prediction.prediction_time

        history = self.repo.get_customer_history(
            self.sample_row["customer_id"], as_of, exclude_transaction_id=self.sample_failed_id
        )
        priors = txn[
            (txn["customer_id"] == self.sample_row["customer_id"])
            & (txn["transaction_id"] != self.sample_failed_id)
            & (txn["created_at"] < as_of)
        ]
        self.assertEqual(len(history.entries), len(priors))
        for _, prior in priors.iterrows():
            entry = next(e for e in history.entries if e.transaction_id == prior["transaction_id"])
            fail_known = pd.notna(prior["first_fail_time"]) and prior["first_fail_time"] < as_of
            success_known = pd.notna(prior["first_success_time"]) and prior["first_success_time"] < as_of
            if fail_known:
                expected = "recovered" if success_known else "failed_pending"
            elif success_known:
                expected = "completed"
            else:
                expected = "unknown"
            self.assertEqual(entry.known_outcome, expected, entry.transaction_id)

    def test_recovery_history_contains_only_known_recoveries(self):
        prediction = get_recovery_prediction(self.sample_failed_id, self.data_dir, self.model_path)
        recovery = self.repo.get_recovery_history(
            self.sample_row["customer_id"], prediction.prediction_time, exclude_transaction_id=self.sample_failed_id
        )
        self.assertEqual(recovery.known_recovered_count, len(recovery.entries))
        for entry in recovery.entries:
            self.assertEqual(entry.known_outcome, "recovered")

    def test_prediction_consumes_phase2_artifacts(self):
        prediction = get_recovery_prediction(self.sample_failed_id, self.data_dir, self.model_path)
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction.probability, 0.0)
        self.assertLessEqual(prediction.probability, 1.0)
        self.assertEqual(prediction.model_path, str(self.model_path))

        import xgboost as xgb

        from agent.prediction import _get_features

        X, _y, meta = _get_features(self.data_dir)
        row = meta.index[meta["transaction_id"] == self.sample_failed_id][0]
        booster = xgb.Booster()
        booster.load_model(str(self.model_path))
        expected = float(booster.predict(xgb.DMatrix(X.iloc[[row]]))[0])
        self.assertAlmostEqual(prediction.probability, expected, places=12)

    def test_prediction_none_for_non_failed_or_missing_model(self):
        completed = self.transactions.loc[self.transactions["status"] == "completed", "transaction_id"].iloc[0]
        self.assertIsNone(get_recovery_prediction(completed, self.data_dir, self.model_path))
        self.assertIsNone(
            get_recovery_prediction(self.sample_failed_id, self.data_dir, self.data_dir / "missing.json")
        )

    def test_repository_cache_returns_same_instance(self):
        self.assertIs(get_repository(self.data_dir), self.repo)


if __name__ == "__main__":
    unittest.main()
