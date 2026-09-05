"""Phase 6 backend unit tests — meaningful coverage without PostgreSQL."""

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import backend.db as db  # noqa: E402
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import InvestigationResult  # noqa: E402
from backend.app import (  # noqa: E402
    CaseDetail,
    CaseSummary,
    InvestigationResponse,
    MetricsSummaryResponse,
    create_app,
)
from backend.service import RevenueGuardService  # noqa: E402

UNREACHABLE_DB = "postgresql://revenueguard:revenueguard@localhost:1/revenueguard"


def _schema_mentions_key(model, needle: str) -> bool:
    """Recursively check a model's JSON schema (including nested $defs) for a
    top-level property name."""

    def walk(node) -> bool:
        if isinstance(node, dict):
            if needle in node.get("properties", {}):
                return True
            return any(walk(value) for value in node.values())
        if isinstance(node, list):
            return any(walk(item) for item in node)
        return False

    schema = model.model_json_schema()
    if walk(schema):
        return True
    return any(walk(definition) for definition in schema.get("$defs", {}).values())


class ImportBoundaryTests(unittest.TestCase):
    def test_agent_package_never_imports_backend(self):
        agent_dir = REPO_ROOT / "agent"
        for source_file in agent_dir.glob("*.py"):
            source = source_file.read_text(encoding="utf-8")
            self.assertNotIn("import backend", source, source_file.name)
            self.assertNotIn("from backend", source, source_file.name)

    def test_app_does_not_import_ml_agent_or_policy_directly(self):
        source = (REPO_ROOT / "backend" / "app.py").read_text(encoding="utf-8")
        for forbidden in ("xgboost", "langgraph", "agent.policy", "agent.graph", "features"):
            self.assertNotIn(forbidden, source)


class ResponseModelSafetyTests(unittest.TestCase):
    def test_case_models_exclude_recovery_outcome(self):
        for model in (CaseSummary, CaseDetail, InvestigationResponse):
            self.assertFalse(
                _schema_mentions_key(model, "recovery_outcome"),
                f"{model.__name__} must not expose recovery_outcome",
            )

    def test_investigation_response_model_shape(self):
        fields = set(InvestigationResponse.model_fields)
        self.assertEqual(
            fields,
            {"transaction_id", "prediction_time", "investigated_at", "result"},
        )

    def test_result_model_has_no_execution_fields(self):
        fields = set(InvestigationResult.model_fields)
        self.assertNotIn("executed", fields)
        self.assertNotIn("payment_status", fields)
        self.assertNotIn("authorized", fields)
        self.assertIn("policy_evaluation", fields)
        self.assertIn("llm_review", fields)


class RouteTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(RevenueGuardService(database_url=None))

    def test_exact_endpoint_surface(self):
        documentation_routes = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}
        routes = {}
        for route in self.app.routes:
            if hasattr(route, "methods") and route.path not in documentation_routes:
                routes.setdefault(route.path, set()).update(route.methods - {"HEAD", "OPTIONS"})
        self.assertEqual(
            routes,
            {
                "/health": {"GET"},
                "/cases": {"GET"},
                "/cases/{transaction_id}": {"GET"},
                "/cases/{transaction_id}/investigation": {"GET", "POST"},
                "/metrics/summary": {"GET"},
            },
        )

    def test_no_execution_or_mutation_endpoints(self):
        for route in self.app.routes:
            if hasattr(route, "methods"):
                self.assertNotIn("DELETE", route.methods)
                self.assertNotIn("PUT", route.methods)
                self.assertNotIn("execute", route.path.lower())


class ApiBehaviorWithoutDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RevenueGuardService(database_url=None)
        cls.client = TestClient(create_app(cls.service))
        _X, _y, meta = _get_features("data")
        cls.real_case_id = meta.loc[meta["split"] == "test", "transaction_id"].iloc[0]

    def test_malformed_pagination_rejected_before_db(self):
        for query in ("?limit=-1", "?limit=0", "?offset=-1", "?limit=99999"):
            response = self.client.get(f"/cases{query}")
            self.assertEqual(response.status_code, 422, query)

    def test_health_reports_degraded_when_database_unreachable(self):
        app = create_app(RevenueGuardService(database_url=UNREACHABLE_DB))
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["database"])

    def test_case_reads_report_503_without_database(self):
        app = create_app(RevenueGuardService(database_url=UNREACHABLE_DB))
        client = TestClient(app)
        self.assertEqual(client.get("/cases").status_code, 503)
        self.assertEqual(client.get(f"/cases/{self.real_case_id}").status_code, 503)
        self.assertEqual(client.get(f"/cases/{self.real_case_id}/investigation").status_code, 503)

    def test_csv_mode_investigation_works_without_database(self):
        response = self.client.post(f"/cases/{self.real_case_id}/investigation", json={})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("recovery_outcome", json.dumps(payload).lower())
        self.assertIsNone(payload["result"]["llm_review"])
        self.assertIsNotNone(payload["result"]["policy_evaluation"])
        self.assertIn(payload["result"]["recommendation"]["action"], ("RETRY", "REVIEW", "IGNORE"))

    def test_unknown_case_returns_404(self):
        response = self.client.post("/cases/TXN-9999999/investigation", json={})
        self.assertEqual(response.status_code, 404)

    def test_metrics_summary_503_without_database(self):
        app = create_app(RevenueGuardService(database_url=UNREACHABLE_DB))
        response = TestClient(app).get("/metrics/summary")
        self.assertEqual(response.status_code, 503)

    def test_metrics_response_model_excludes_recovery_outcome(self):
        self.assertFalse(
            _schema_mentions_key(MetricsSummaryResponse, "recovery_outcome"),
            "MetricsSummaryResponse must not expose recovery_outcome",
        )
        fields = set(MetricsSummaryResponse.model_fields)
        self.assertEqual(
            fields,
            {
                "failed_transactions",
                "investigated_cases",
                "recommendations",
                "final_actions",
                "policy_decisions",
                "execution_authorized_count",
            },
        )


class CorsTests(unittest.TestCase):
    def _client(self, origins=None):
        import os

        if origins is None:
            environ = {}
        else:
            environ = {"REVENUEGUARD_CORS_ORIGINS": origins}
        previous = os.environ.pop("REVENUEGUARD_CORS_ORIGINS", None)
        os.environ.update(environ)
        try:
            app = create_app(RevenueGuardService(database_url=None))
        finally:
            if previous is None:
                os.environ.pop("REVENUEGUARD_CORS_ORIGINS", None)
            else:
                os.environ["REVENUEGUARD_CORS_ORIGINS"] = previous
        return TestClient(app)

    def test_preflight_allowed_for_default_local_origin(self):
        client = self._client()
        response = client.options(
            "/cases",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")

    def test_preflight_rejected_for_unknown_origin(self):
        client = self._client()
        response = client.options(
            "/cases",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
        )
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_configurable_origins_from_environment(self):
        client = self._client(origins="https://dashboard.example,http://localhost:5173")
        response = client.options(
            "/cases",
            headers={"Origin": "https://dashboard.example", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://dashboard.example")


class ConnectUrlResolutionTests(unittest.TestCase):
    """Regression tests for db.connect() URL resolution.

    Regression context: the connect() parameter was originally named
    ``database_url``, shadowing the module-level database_url() helper and
    raising ``TypeError: 'str' object is not callable`` at call time.
    """

    def _capture_psycopg_connect(self):
        captured = {}

        def fake_psycopg_connect(url, *args, **kwargs):
            captured["url"] = url
            return mock.MagicMock()

        return captured, fake_psycopg_connect

    def test_connect_passes_explicit_url_to_psycopg(self):
        captured, fake = self._capture_psycopg_connect()
        url = "postgresql://user:pw@localhost:5432/revenueguard"
        with mock.patch.object(db.psycopg, "connect", side_effect=fake):
            db.connect(url)
        self.assertEqual(captured["url"], url)

    def test_connect_resolves_url_from_environment(self):
        captured, fake = self._capture_psycopg_connect()
        env_url = "postgresql://env-user@localhost:5432/env-db"
        with mock.patch.dict(os.environ, {"DATABASE_URL": env_url}):
            with mock.patch.object(db.psycopg, "connect", side_effect=fake):
                db.connect()
        self.assertEqual(captured["url"], env_url)

    def test_connect_without_configuration_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                db.connect()


class TimestampSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RevenueGuardService(database_url=None)
        from agent.data_repository import get_repository

        failed_ids = get_repository("data").failed_transaction_ids
        cls.result = cls.service.investigate_transaction(failed_ids[0])["result"]

    def test_snapshot_varies_only_in_investigated_at(self):
        first = RevenueGuardService._snapshot_row(self.result, datetime(2026, 1, 1, tzinfo=timezone.utc))
        second = RevenueGuardService._snapshot_row(self.result, datetime(2027, 1, 1, tzinfo=timezone.utc))
        self.assertNotEqual(first["investigated_at"], second["investigated_at"])
        self.assertEqual(first["prediction_time"], second["prediction_time"])
        first_only = dict(first)
        second_only = dict(second)
        first_only.pop("investigated_at")
        second_only.pop("investigated_at")
        self.assertEqual(first_only, second_only)

    def test_policy_evaluated_at_is_the_prediction_point(self):
        policy = self.result.policy_evaluation
        self.assertEqual(policy.evaluated_at, self.result.prediction.prediction_time)

    def test_investigated_at_not_part_of_result_payload(self):
        self.assertNotIn("investigated_at", self.result.model_dump())


class InternalKeyExposureRegressionTests(unittest.TestCase):
    """Regression: FastAPI investigation responses must never contain
    ``model_path`` or ``recovery_outcome`` at ANY nesting depth.

    Bug context: the FastAPI layer originally returned the raw
    InvestigationResult, exposing ``prediction.model_path`` and the
    ``recovery_prediction`` evidence payload's ``model_path``. The MCP layer
    stripped them; the API did not. Responses are now recursively stripped at
    the response-model serialization boundary.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(RevenueGuardService(database_url=None)))
        from agent.data_repository import get_repository

        cls.failed_id = get_repository("data").failed_transaction_ids[0]

    @staticmethod
    def _contains_key(value, needle):
        if isinstance(value, dict):
            if needle in value:
                return True
            return any(InternalKeyExposureRegressionTests._contains_key(v, needle) for v in value.values())
        if isinstance(value, list):
            return any(InternalKeyExposureRegressionTests._contains_key(v, needle) for v in value)
        return False

    def test_post_investigation_response_has_no_internal_keys(self):
        response = self.client.post(f"/cases/{self.failed_id}/investigation", json={})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(self._contains_key(body, "model_path"))
        self.assertFalse(self._contains_key(body, "recovery_outcome"))

    def test_prediction_payload_content_survives_stripping(self):
        response = self.client.post(f"/cases/{self.failed_id}/investigation", json={})
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertIn("probability", result["prediction"])
        self.assertIn("prediction_time", result["prediction"])
        prediction_evidence = next(
            e for e in result["evidence"] if e["source"] == "recovery_prediction"
        )
        self.assertIn("probability", prediction_evidence["payload"])
        self.assertNotIn("model_path", prediction_evidence["payload"])

    def test_recursive_strip_helper_removes_keys_at_all_depths(self):
        from backend.app import _strip_internal_keys

        dirty = {
            "model_path": "models/baseline/model.json",
            "nested": {
                "recovery_outcome": "recovered",
                "list": [{"model_path": "x", "keep": 1}, "scalar"],
            },
            "keep": {"probability": 0.42},
        }
        clean = _strip_internal_keys(dirty)
        self.assertEqual(
            clean,
            {"nested": {"list": [{"keep": 1}, "scalar"]}, "keep": {"probability": 0.42}},
        )
        self.assertFalse(self._contains_key(clean, "model_path"))
        self.assertFalse(self._contains_key(clean, "recovery_outcome"))


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
