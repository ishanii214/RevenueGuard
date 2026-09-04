"""Feature and leakage tests for the Phase 2 recovery baseline."""

import random
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import features as ft  # noqa: E402
import generate_data as gd  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _parse(series):
    return pd.to_datetime(series, format=TIMESTAMP_FORMAT)


class FeatureBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        data_dir = Path(cls._tmp.name)
        gd.write_dataset(gd.generate_dataset(random.Random(42), 1200), data_dir)
        cls.X, cls.y, cls.meta = ft.build_features(data_dir)
        cls.transactions = pd.read_csv(data_dir / "transactions.csv", dtype=str)
        cls.transactions["created_at"] = _parse(cls.transactions["created_at"])
        cls.attempts = pd.read_csv(data_dir / "payment_attempts.csv", dtype=str)
        cls.attempts["attempted_at"] = _parse(cls.attempts["attempted_at"])
        cls.failures = pd.read_csv(data_dir / "payment_failures.csv", dtype=str)
        cls.failures["failed_at"] = _parse(cls.failures["failed_at"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_rows_are_failed_transactions_with_correct_labels(self):
        failed_ids = set(self.transactions.loc[self.transactions["status"] == "failed", "transaction_id"])
        self.assertEqual(set(self.meta["transaction_id"]), failed_ids)
        expected = (self.meta["recovery_outcome"] != "unrecovered").astype(int)
        self.assertTrue((self.y.to_numpy() == expected.to_numpy()).all())

    def test_split_is_chronological_and_non_overlapping(self):
        splits = self.meta["split"].tolist()
        n = len(splits)
        n_train = int(n * ft.TRAIN_FRACTION)
        n_val = int(n * ft.VALIDATION_FRACTION)
        self.assertEqual(splits, ["train"] * n_train + ["validation"] * n_val + ["test"] * (n - n_train - n_val))
        train_time = self.meta.loc[self.meta["split"] == "train", "prediction_time"]
        val_time = self.meta.loc[self.meta["split"] == "validation", "prediction_time"]
        test_time = self.meta.loc[self.meta["split"] == "test", "prediction_time"]
        self.assertLessEqual(train_time.max(), val_time.min())
        self.assertLessEqual(val_time.max(), test_time.min())
        ids = self.meta["transaction_id"]
        self.assertEqual(ids.nunique(), len(ids))

    def test_history_matches_independent_asof_recomputation(self):
        """THE leakage test: brute-force recomputation of the temporal rule."""
        first_fail = (
            self.attempts.loc[self.attempts["status"] == "failed"]
            .groupby("transaction_id")["attempted_at"]
            .min()
        )
        first_success = (
            self.attempts.loc[self.attempts["status"] == "succeeded"]
            .groupby("transaction_id")["attempted_at"]
            .min()
        )
        txn = self.transactions.copy()
        txn["first_fail_time"] = txn["transaction_id"].map(first_fail)
        txn["first_success_time"] = txn["transaction_id"].map(first_success)
        by_customer = {cid: frame for cid, frame in txn.groupby("customer_id")}
        feature_rows = self.meta.reset_index(drop=True)

        for position in range(0, len(feature_rows), 25):
            row = feature_rows.iloc[position]
            customer_txns = by_customer[row["customer_id"]]
            asof = row["prediction_time"]
            prior = customer_txns[
                (customer_txns["transaction_id"] != row["transaction_id"])
                & (customer_txns["created_at"] < asof)
            ]
            failed_known = int((prior["first_fail_time"] < asof).fillna(False).sum())
            recovered_known = int((prior["first_success_time"] < asof).fillna(False).sum())
            self.assertEqual(int(self.X.iloc[position]["prior_txn_count"]), len(prior), row["transaction_id"])
            self.assertEqual(
                int(self.X.iloc[position]["prior_failed_count"]), failed_known, row["transaction_id"]
            )
            self.assertEqual(
                int(self.X.iloc[position]["prior_recovered_count"]), recovered_known, row["transaction_id"]
            )
            expected_rate = recovered_known / failed_known if failed_known else np.nan
            actual_rate = self.X.iloc[position]["prior_recovery_rate"]
            if np.isnan(expected_rate):
                self.assertTrue(np.isnan(actual_rate))
            else:
                self.assertAlmostEqual(float(actual_rate), expected_rate, places=12)

    def test_later_recoveries_excluded_from_history(self):
        """Targeted temporal-rule check on a handcrafted timeline."""
        transactions = pd.DataFrame(
            {
                "transaction_id": ["TXN-0000001", "TXN-0000002", "TXN-0000003"],
                "customer_id": ["CUST-000001"] * 3,
                "created_at": _parse(
                    pd.Series(
                        ["2024-01-01T10:00:00", "2024-01-10T10:00:00", "2024-02-01T10:00:00"]
                    )
                ),
                "amount": [100.0, 100.0, 100.0],
                "currency": ["USD"] * 3,
                "payment_method": ["card"] * 3,
                "status": ["failed"] * 3,
                "recovery_outcome": ["recovered_review", "unrecovered", "unrecovered"],
            }
        )
        attempts = pd.DataFrame(
            {
                "attempt_id": ["ATT-0000001", "ATT-0000002", "ATT-0000003", "ATT-0000004", "ATT-0000005"],
                "transaction_id": ["TXN-0000001", "TXN-0000001", "TXN-0000001", "TXN-0000002", "TXN-0000003"],
                "attempt_number": ["1", "2", "3", "1", "1"],
                "attempted_at": _parse(
                    pd.Series(
                        [
                            "2024-01-01T10:05:00",
                            "2024-01-04T10:05:00",
                            "2024-01-20T09:00:00",
                            "2024-01-10T10:05:00",
                            "2024-02-01T10:05:00",
                        ]
                    )
                ),
                "status": ["failed", "failed", "succeeded", "failed", "failed"],
                "payment_method": ["card"] * 5,
            }
        )
        # TXN-0000001 recovered on 2024-01-20, i.e. AFTER TXN-0000002's
        # prediction point (2024-01-10) but BEFORE TXN-0000003's (2024-02-01).
        asof_times = pd.Series(
            {
                "TXN-0000001": datetime(2024, 1, 1, 10, 5),
                "TXN-0000002": datetime(2024, 1, 10, 10, 5),
                "TXN-0000003": datetime(2024, 2, 1, 10, 5),
            }
        )
        history = ft.compute_asof_history(transactions, attempts, asof_times)
        self.assertEqual(int(history.loc["TXN-0000002", "prior_txn_count"]), 1)
        self.assertEqual(int(history.loc["TXN-0000002", "prior_failed_count"]), 1)
        self.assertEqual(int(history.loc["TXN-0000002", "prior_recovered_count"]), 0)
        # Known-failed but not yet recovered -> recovery rate is 0/1 = 0.0.
        self.assertEqual(float(history.loc["TXN-0000002", "prior_recovery_rate"]), 0.0)
        self.assertEqual(int(history.loc["TXN-0000003", "prior_txn_count"]), 2)
        self.assertEqual(int(history.loc["TXN-0000003", "prior_failed_count"]), 2)
        self.assertEqual(int(history.loc["TXN-0000003", "prior_recovered_count"]), 1)
        self.assertEqual(float(history.loc["TXN-0000003", "prior_recovery_rate"]), 0.5)

    def test_no_forbidden_columns_and_fixed_order(self):
        self.assertEqual(list(self.X.columns), ft.FEATURE_COLUMNS)
        for forbidden in ft.FORBIDDEN_FEATURE_COLUMNS:
            self.assertNotIn(forbidden, self.X.columns)
        self.assertNotIn("recovery_outcome", self.X.columns)

    def test_nan_only_in_by_design_columns(self):
        nan_counts = self.X.isna().sum()
        unexpected = nan_counts[nan_counts > 0].index.tolist()
        for column in unexpected:
            self.assertIn(column, ft.BY_DESIGN_NAN_COLUMNS)

    def test_initial_attempt_features_come_only_from_attempt_one(self):
        initial = self.attempts.loc[self.attempts["attempt_number"] == "1"].set_index("transaction_id")
        initial_failures = self.failures.loc[
            self.failures["attempt_id"].isin(initial["attempt_id"])
        ].set_index("transaction_id")
        created = self.transactions.set_index("transaction_id")["created_at"]
        for position in range(0, len(self.meta), 25):
            txn_id = self.meta.iloc[position]["transaction_id"]
            expected_minutes = (
                initial.loc[txn_id, "attempted_at"] - created.loc[txn_id]
            ).total_seconds() / 60.0
            self.assertAlmostEqual(
                float(self.X.iloc[position]["time_to_first_attempt_minutes"]),
                expected_minutes,
                places=6,
            )
            self.assertEqual(
                self.meta.iloc[position]["failure_reason"], initial_failures.loc[txn_id, "failure_reason"]
            )


if __name__ == "__main__":
    unittest.main()
