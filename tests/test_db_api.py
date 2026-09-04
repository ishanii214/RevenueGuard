"""Phase 6 gated PostgreSQL integration tests.

These tests run ONLY when REVENUEGUARD_TEST_DATABASE_URL is configured and
should point at a disposable database — the suite applies the schema and
seeds it. When the variable is absent the entire class is skipped and this
is reported honestly; no DB-dependent result is ever faked.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from pandas.testing import assert_frame_equal

REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import backend.db as db  # noqa: E402
import features  # noqa: E402
from agent.data_repository import CaseRepository  # noqa: E402
from agent.graph import investigate  # noqa: E402
from agent.prediction import _get_features, get_recovery_prediction  # noqa: E402
from agent.schemas import InvestigationInput  # noqa: E402
from backend.app import create_app  # noqa: E402
from backend.repository import build_postgres_repository, case_summary  # noqa: E402
from backend.service import RevenueGuardService  # noqa: E402

DB_URL = os.environ.get("REVENUEGUARD_TEST_DATABASE_URL")
DATA_FILES = ("customers.csv", "transactions.csv", "payment_attempts.csv", "payment_failures.csv")


@unittest.skipUnless(DB_URL, "REVENUEGUARD_TEST_DATABASE_URL not configured; PostgreSQL integration suite skipped")
class PostgresIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_url = DB_URL
        with db.connect(cls.db_url) as conn:
            cls.seed_counts = db.seed_from_csv(conn, "data")
        cls.frames_db = db.load_frames(db.connect(cls.db_url))
        cls.frames_csv = features.load_tables("data")
        cls.repo_db = build_postgres_repository(cls.frames_db)
        cls.repo_csv = CaseRepository("data")
        cls.service = RevenueGuardService(database_url=cls.db_url)
        cls.app = TestClient(create_app(cls.service))
        _X, _y, meta = _get_features("data")
        cls.case_ids = meta.loc[meta["split"] == "test", "transaction_id"].tolist()[:25]
        cls.attempts = pd.read_csv("data/payment_attempts.csv", dtype=str)

    def _db_mode_result(self, transaction_id):
        return self.service.investigate_transaction(transaction_id, use_database=True)

    def test_seeding_deterministic_and_matches_csv(self):
        self.assertEqual(
            self.seed_counts,
            {
                "customers": len(self.frames_csv[0]),
                "transactions": len(self.frames_csv[1]),
                "payment_attempts": len(self.frames_csv[2]),
                "payment_failures": len(self.frames_csv[3]),
            },
        )
        with db.connect(self.db_url) as conn:
            reseed_counts = db.seed_from_csv(conn, "data")
        self.assertEqual(self.seed_counts, reseed_counts)

    def test_schema_constraints_exist(self):
        with db.connect(self.db_url) as conn:
            constraints = conn.execute(
                """
                SELECT conrelid::regclass::text, contype, conname
                FROM pg_constraint
                WHERE conrelid IN (
                    'customers'::regclass, 'transactions'::regclass,
                    'payment_attempts'::regclass, 'payment_failures'::regclass,
                    'investigation_results'::regclass
                )
                """
            ).fetchall()
        by_table = {}
        for table, contype, _name in constraints:
            by_table.setdefault(table, []).append(contype)
        for table in (
            "customers",
            "transactions",
            "payment_attempts",
            "payment_failures",
            "investigation_results",
        ):
            self.assertIn("p", by_table.get(table, []), f"{table} missing primary key")
        self.assertIn("f", by_table.get("transactions", []))
        self.assertIn("f", by_table.get("payment_attempts", []))
        self.assertIn("f", by_table.get("payment_failures", []))
        unique_attempts = conn_unique_attempt_ids = [
            _name
            for _table, contype, _name in constraints
            if _table == "payment_failures" and contype == "u"
        ]
        self.assertTrue(unique_attempts, "payment_failures.attempt_id must be UNIQUE")

    def test_frame_parity_with_csv(self):
        for db_frame, csv_frame in zip(self.frames_db, self.frames_csv):
            assert_frame_equal(db_frame, csv_frame)

    def test_repository_parity_with_csv(self):
        for transaction_id in self.case_ids[:10]:
            self.assertEqual(
                self.repo_db.get_transaction(transaction_id),
                self.repo_csv.get_transaction(transaction_id),
            )
            attempt_db = self.repo_db.get_initial_attempt(transaction_id)
            attempt_csv = self.repo_csv.get_initial_attempt(transaction_id)
            self.assertEqual(attempt_db, attempt_csv)
            self.assertEqual(
                self.repo_db.get_initial_failure(transaction_id),
                self.repo_csv.get_initial_failure(transaction_id),
            )
            customer_id = attempt_db.customer_id
            prediction_time = attempt_db.attempted_at
            self.assertEqual(
                self.repo_db.get_customer_history(customer_id, prediction_time, exclude_transaction_id=transaction_id),
                self.repo_csv.get_customer_history(customer_id, prediction_time, exclude_transaction_id=transaction_id),
            )
            self.assertEqual(
                self.repo_db.get_recovery_history(customer_id, prediction_time, exclude_transaction_id=transaction_id),
                self.repo_csv.get_recovery_history(customer_id, prediction_time, exclude_transaction_id=transaction_id),
            )

    def test_db_mode_investigation_parity_with_csv_mode(self):
        for transaction_id in self.case_ids[:10]:
            csv_result = investigate(
                InvestigationInput(transaction_id=transaction_id, data_dir="data"), llm_client=None
            )
            db_result = self._db_mode_result(transaction_id)["result"]
            self.assertEqual(db_result, csv_result, transaction_id)

    def test_future_attempt_exclusion_in_db_mode(self):
        counts = self.attempts.groupby("transaction_id").size()
        multi = [t for t in self.case_ids if counts.get(t, 0) >= 3]
        self.assertTrue(multi)
        for transaction_id in multi:
            result = self._db_mode_result(transaction_id)["result"]
            policy_request_json = json.dumps(result.policy_evaluation.model_dump())
            future = self.attempts.loc[
                (self.attempts["transaction_id"] == transaction_id)
                & (self.attempts["attempt_number"] != "1"),
                "attempt_id",
            ]
            for attempt_id in future:
                self.assertNotIn(attempt_id, policy_request_json)
            self.assertNotIn("retry_cap_exceeded", result.policy_evaluation.reason_codes)

    def test_bounded_loading_single_load_across_investigations(self):
        db.reset_stats()
        for transaction_id in self.case_ids[:5]:
            self._db_mode_result(transaction_id)
        self.assertEqual(db.STATS["frame_loads"], 1, "frames must load once, not per investigation")
        self.assertGreaterEqual(db.STATS["fingerprint_checks"], 5)
        # A reseed changes the fingerprint and forces exactly one reload.
        with db.connect(self.db_url) as conn:
            db.seed_from_csv(conn, "data")
        for transaction_id in self.case_ids[:5]:
            self._db_mode_result(transaction_id)
        self.assertEqual(db.STATS["frame_loads"], 2)

    def test_timestamp_separation(self):
        transaction_id = self.case_ids[0]
        first = self._db_mode_result(transaction_id)
        second = self._db_mode_result(transaction_id)
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(first["prediction_time"], second["prediction_time"])
        self.assertGreaterEqual(second["investigated_at"], first["investigated_at"])
        with db.connect(self.db_url) as conn:
            row = db.fetch_result_snapshot(conn, transaction_id)
        self.assertEqual(row["prediction_time"], first["prediction_time"])
        self.assertEqual(row["investigated_at"], second["investigated_at"])
        stored = row["result"]
        self.assertEqual(stored["prediction"]["prediction_time"] if stored["prediction"] else None,
                         first["prediction_time"].isoformat() if first["prediction_time"] else None)

    def test_persistence_roundtrip_and_snapshot_fields(self):
        transaction_id = self.case_ids[0]
        snapshot = self._db_mode_result(transaction_id)
        with db.connect(self.db_url) as conn:
            row = db.fetch_result_snapshot(conn, transaction_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["recommendation"], snapshot["result"].recommendation.action)
        self.assertEqual(row["policy_decision"], snapshot["result"].policy_evaluation.policy_decision)
        self.assertEqual(row["final_action"], snapshot["result"].policy_evaluation.final_action)
        self.assertEqual(
            row["execution_authorized"], snapshot["result"].policy_evaluation.execution_authorized
        )
        self.assertEqual(row["policy_version"], snapshot["result"].policy_evaluation.policy_version)
        self.assertIn("llm_review", row["result"])

    def test_api_integration_and_safety(self):
        transaction_id = self.case_ids[0]
        health = self.app.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["database"])
        listing = self.app.get("/cases?limit=5&offset=0")
        self.assertEqual(listing.status_code, 200)
        body = listing.json()
        self.assertEqual(len(body["items"]), 5)
        self.assertGreater(body["total"], 0)
        self.assertNotIn("recovery_outcome", listing.text.lower())
        detail = self.app.get(f"/cases/{transaction_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn("recovery_outcome", detail.text.lower())
        start = self.app.post(f"/cases/{transaction_id}/investigation", json={"use_database": True})
        self.assertEqual(start.status_code, 200)
        self.assertNotIn("recovery_outcome", start.text.lower())
        fetched = self.app.get(f"/cases/{transaction_id}/investigation")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["result"], start.json()["result"])
        self.assertEqual(self.app.get("/cases/TXN-9999999").status_code, 404)
        self.assertEqual(self.app.post("/cases/TXN-9999999/investigation", json={}).status_code, 404)
        self.assertEqual(
            self.app.get("/cases/TXN-9999999/investigation").status_code, 404
        )
        self.assertEqual(self.app.get("/cases?limit=-1").status_code, 422)

    def test_policy_authorization_semantics_unchanged(self):
        for result in (self._db_mode_result(t)["result"] for t in self.case_ids[:10]):
            policy = result.policy_evaluation
            if result.recommendation.action == "RETRY" and policy.policy_decision == "ALLOWED":
                self.assertTrue(policy.execution_authorized)
            else:
                self.assertFalse(policy.execution_authorized)


if __name__ == "__main__":
    unittest.main()
