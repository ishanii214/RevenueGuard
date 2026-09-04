"""Data-quality tests for the synthetic failed-payment dataset (Phase 1).

The tests generate a fresh dataset into a temporary directory with a fixed
seed, so they validate the generator itself and do not depend on the CSVs
committed under data/.
"""

import csv
import random
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_data as gd  # noqa: E402


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ts(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")


class DatasetQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        out = Path(cls._tmp.name)
        gd.write_dataset(gd.generate_dataset(random.Random(42), 1000), out)
        cls.customers = _load(out / "customers.csv")
        cls.transactions = _load(out / "transactions.csv")
        cls.attempts = _load(out / "payment_attempts.csv")
        cls.failures = _load(out / "payment_failures.csv")
        cls.customers_by_id = {row["customer_id"]: row for row in cls.customers}
        cls.transactions_by_id = {row["transaction_id"]: row for row in cls.transactions}
        cls.attempts_by_id = {row["attempt_id"]: row for row in cls.attempts}
        cls.failures_by_attempt = {row["attempt_id"]: row for row in cls.failures}
        cls.attempts_by_txn = {}
        for attempt in cls.attempts:
            cls.attempts_by_txn.setdefault(attempt["transaction_id"], []).append(attempt)
        for attempt_list in cls.attempts_by_txn.values():
            attempt_list.sort(key=lambda row: int(row["attempt_number"]))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_referential_integrity(self):
        for txn in self.transactions:
            self.assertIn(txn["customer_id"], self.customers_by_id)
        for attempt in self.attempts:
            self.assertIn(attempt["transaction_id"], self.transactions_by_id)
        for failure in self.failures:
            self.assertIn(failure["attempt_id"], self.attempts_by_id)
            self.assertIn(failure["transaction_id"], self.transactions_by_id)
            self.assertIn(failure["customer_id"], self.customers_by_id)
            attempt = self.attempts_by_id[failure["attempt_id"]]
            self.assertEqual(attempt["transaction_id"], failure["transaction_id"])
            txn = self.transactions_by_id[failure["transaction_id"]]
            self.assertEqual(txn["customer_id"], failure["customer_id"])

    def test_ids_valid_and_unique(self):
        specs = (
            (r"^CUST-\d{6}$", self.customers, "customer_id"),
            (r"^TXN-\d{7}$", self.transactions, "transaction_id"),
            (r"^ATT-\d{7}$", self.attempts, "attempt_id"),
            (r"^FAIL-\d{7}$", self.failures, "failure_id"),
        )
        for pattern, rows, column in specs:
            ids = [row[column] for row in rows]
            self.assertTrue(all(__import__("re").match(pattern, value) for value in ids), pattern)
            self.assertEqual(len(ids), len(set(ids)), f"duplicate {column}")
        failed_attempts = [a for a in self.attempts if a["status"] == "failed"]
        self.assertEqual(len(failed_attempts), len(self.failures))
        self.assertEqual(len(self.failures_by_attempt), len(self.failures))
        for failure in self.failures:
            self.assertEqual(self.attempts_by_id[failure["attempt_id"]]["status"], "failed")

    def test_valid_statuses_and_enums(self):
        for txn in self.transactions:
            self.assertIn(txn["status"], gd.TXN_STATUSES)
            self.assertIn(txn["recovery_outcome"], gd.RECOVERY_OUTCOMES)
            self.assertIn(txn["payment_method"], gd.PAYMENT_METHODS)
            self.assertIn(txn["currency"], set(gd.COUNTRY_CURRENCY.values()))
            if txn["status"] == "completed":
                self.assertEqual(txn["recovery_outcome"], "not_applicable")
            else:
                self.assertIn(txn["recovery_outcome"], ("recovered_retry", "recovered_review", "unrecovered"))
        for attempt in self.attempts:
            self.assertIn(attempt["status"], gd.ATTEMPT_STATUSES)
            self.assertIn(attempt["payment_method"], gd.PAYMENT_METHODS)

    def test_valid_failure_reasons_and_codes(self):
        for failure in self.failures:
            self.assertIn(failure["failure_reason"], gd.FAILURE_REASON_CODES)
            self.assertEqual(
                failure["processor_response_code"], gd.FAILURE_REASON_CODES[failure["failure_reason"]]
            )
            attempt = self.attempts_by_id[failure["attempt_id"]]
            self.assertEqual(failure["failed_at"], attempt["attempted_at"])

    def test_attempt_sequences(self):
        for txn_id, attempt_list in self.attempts_by_txn.items():
            numbers = [int(a["attempt_number"]) for a in attempt_list]
            self.assertEqual(numbers, list(range(1, len(attempt_list) + 1)), txn_id)
            times = [_ts(a["attempted_at"]) for a in attempt_list]
            for previous, current in zip(times, times[1:]):
                self.assertLess(previous, current, txn_id)
            self.assertGreaterEqual(times[0], _ts(self.transactions_by_id[txn_id]["created_at"]))

    def test_outcome_consistency(self):
        for txn in self.transactions:
            attempt_list = self.attempts_by_txn.get(txn["transaction_id"], [])
            self.assertGreaterEqual(len(attempt_list), 1, txn["transaction_id"])
            if txn["status"] == "completed":
                self.assertEqual(len(attempt_list), 1)
                self.assertEqual(attempt_list[0]["status"], "succeeded")
                self.assertNotIn(attempt_list[0]["attempt_id"], self.failures_by_attempt)
            elif txn["recovery_outcome"] == "unrecovered":
                self.assertGreaterEqual(len(attempt_list), 1)
                self.assertTrue(all(a["status"] == "failed" for a in attempt_list), txn["transaction_id"])
            else:
                self.assertGreaterEqual(len(attempt_list), 2)
                successes = [a for a in attempt_list if a["status"] == "succeeded"]
                self.assertEqual(len(successes), 1, txn["transaction_id"])
                self.assertEqual(attempt_list[-1]["status"], "succeeded", txn["transaction_id"])

    def test_retry_realism_caps(self):
        for txn in self.transactions:
            if txn["status"] != "failed":
                continue
            attempt_list = self.attempts_by_txn[txn["transaction_id"]]
            reasons = {
                self.failures_by_attempt[a["attempt_id"]]["failure_reason"]
                for a in attempt_list
                if a["attempt_id"] in self.failures_by_attempt
            }
            self.assertLessEqual(len(attempt_list), 4, txn["transaction_id"])
            if reasons & gd.NON_RETRYABLE_REASONS:
                self.assertLessEqual(len(attempt_list), 3, txn["transaction_id"])
            if float(txn["amount"]) >= gd.HIGH_VALUE_THRESHOLD:
                self.assertLessEqual(len(attempt_list), 2, txn["transaction_id"])
            for attempt in attempt_list:
                if attempt["payment_method"] != txn["payment_method"]:
                    self.assertEqual(attempt["status"], "succeeded")
                    self.assertTrue(reasons & gd.NON_RETRYABLE_REASONS, txn["transaction_id"])

    def test_chronology_across_tables(self):
        for txn in self.transactions:
            signup = self.customers_by_id[txn["customer_id"]]["signup_date"]
            self.assertLess(signup, txn["created_at"][:10], txn["transaction_id"])

    def test_no_missing_values(self):
        specs = (
            (self.customers, gd.CUSTOMERS_HEADER),
            (self.transactions, gd.TRANSACTIONS_HEADER),
            (self.attempts, gd.ATTEMPTS_HEADER),
            (self.failures, gd.FAILURES_HEADER),
        )
        for rows, header in specs:
            for row in rows:
                for column in header:
                    self.assertIsNotNone(row[column], column)
                    self.assertNotEqual(row[column].strip(), "", f"empty {column}")

    def test_distributions_not_degenerate(self):
        self.assertGreaterEqual(len({row["customer_segment"] for row in self.customers}), 3)
        failed = [t for t in self.transactions if t["status"] == "failed"]
        self.assertGreater(len(failed), 0)
        self.assertGreaterEqual(
            len({t["recovery_outcome"] for t in failed}),
            3,
            "recovered_retry, recovered_review and unrecovered must all occur",
        )
        reason_counts = Counter(f["failure_reason"] for f in self.failures)
        self.assertGreaterEqual(len(reason_counts), 5)


if __name__ == "__main__":
    unittest.main()
