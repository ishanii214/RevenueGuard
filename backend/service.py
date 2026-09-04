"""Application service layer (Phase 6).

The service layer is the only place where graph execution, persistence, and
case queries meet. FastAPI (backend/app.py) calls these functions; it never
touches XGBoost, LangGraph, policy logic, or SQL directly.

Timestamp semantics (strictly separated):
- ``prediction_time`` — the temporal cutoff used by the existing ML/agent/
  policy logic; identical to Phase 1-5 behavior.
- ``investigated_at`` — operational wall-clock metadata stamped onto the
  persistence row only, AFTER the graph returns. It never influences
  feature engineering, evidence retrieval, history availability, or policy
  evaluation.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
for entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from agent.data_repository import CaseRepository, get_repository  # noqa: E402
from agent.graph import investigate  # noqa: E402
from agent.llm import DisabledLLM, make_llm_from_env  # noqa: E402
from agent.prediction import get_recovery_prediction  # noqa: E402
from agent.schemas import InvestigationInput, InvestigationResult  # noqa: E402

import backend.db as db  # noqa: E402
from backend.repository import build_postgres_repository, case_summary  # noqa: E402


class CaseNotFound(Exception):
    pass


class DatabaseUnavailable(Exception):
    pass


class RevenueGuardService:
    def __init__(
        self,
        database_url: str | None = None,
        data_dir: str = "data",
        model_path: str = "models/baseline/model.json",
    ):
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.data_dir = data_dir
        self.model_path = model_path

    def _connect(self):
        if not self.database_url:
            raise DatabaseUnavailable("DATABASE_URL is not configured")
        try:
            return db.connect(self.database_url)
        except Exception as exc:  # connection/credential failures → uniform 503 signal
            raise DatabaseUnavailable(f"database connection failed: {type(exc).__name__}") from exc

    def health(self) -> dict:
        database_ok = False
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            database_ok = True
        except Exception:
            database_ok = False
        return {
            "status": "ok" if database_ok else "degraded",
            "database": database_ok,
            "model_artifact": (REPO_ROOT / self.model_path).exists(),
        }

    def _require_case(self, transaction_id: str, use_database: bool):
        if use_database:
            with self._connect() as conn:
                row = db.fetch_case_row(conn, transaction_id)
            if row is None:
                raise CaseNotFound(transaction_id)
            return
        if get_repository(self.data_dir).get_transaction(transaction_id) is None:
            raise CaseNotFound(transaction_id)

    def investigate_transaction(
        self, transaction_id: str, use_database: bool = False, use_llm: bool = False
    ) -> dict:
        """Run the full pipeline for one case and persist the snapshot.

        DB mode (``use_database=True``) uses the fingerprint-cached
        PostgreSQL frames as the data source for evidence and features;
        results are identical to CSV mode (parity-tested). Persistence
        happens whenever the database is configured, in either mode.
        """
        self._require_case(transaction_id, use_database)
        llm_client = make_llm_from_env() if use_llm else DisabledLLM()
        if use_database:
            with self._connect() as conn:
                frames = db.get_frames(conn)
            prediction = get_recovery_prediction(
                transaction_id, model_path=self.model_path, frames=frames
            )
            repository = build_postgres_repository(frames)
            repository_factory = lambda data_dir: repository  # noqa: E731
            investigation_input = InvestigationInput(
                transaction_id=transaction_id,
                data_dir=self.data_dir,
                model_path=self.model_path,
                prediction=prediction,
            )
        else:
            repository_factory = None
            investigation_input = InvestigationInput(
                transaction_id=transaction_id,
                data_dir=self.data_dir,
                model_path=self.model_path,
            )
        result: InvestigationResult = investigate(
            investigation_input, llm_client=llm_client, repository_factory=repository_factory
        )
        investigated_at = datetime.now(timezone.utc)
        snapshot = self._snapshot_row(result, investigated_at)
        if self.database_url:
            with self._connect() as conn:
                db.save_result_snapshot(conn, snapshot)
        return {
            "transaction_id": transaction_id,
            "result": result,
            "prediction_time": result.prediction.prediction_time if result.prediction else None,
            "investigated_at": investigated_at,
        }

    @staticmethod
    def _snapshot_row(result: InvestigationResult, investigated_at: datetime) -> dict:
        """Build the current-result persistence row (D4). The only field
        derived from the operational clock is ``investigated_at`` itself."""
        policy = result.policy_evaluation
        return {
            "transaction_id": result.transaction_id,
            "prediction_time": result.prediction.prediction_time if result.prediction else None,
            "investigated_at": investigated_at,
            "recommendation": result.recommendation.action,
            "policy_decision": policy.policy_decision if policy else "DENIED",
            "final_action": policy.final_action if policy else "REVIEW",
            "execution_authorized": policy.execution_authorized if policy else False,
            "policy_version": policy.policy_version if policy else "none",
            "result": result.model_dump(mode="json"),
        }

    def list_cases(self, limit: int = 50, offset: int = 0) -> dict:
        with self._connect() as conn:
            rows, total = db.list_case_rows(conn, limit, offset)
        return {"items": [case_summary(row) for row in rows], "total": total, "limit": limit, "offset": offset}

    def get_case(self, transaction_id: str) -> dict:
        with self._connect() as conn:
            row = db.fetch_case_row(conn, transaction_id)
        if row is None:
            raise CaseNotFound(transaction_id)
        return case_summary(row)

    def get_investigation(self, transaction_id: str) -> dict:
        with self._connect() as conn:
            row = db.fetch_result_snapshot(conn, transaction_id)
        if row is None:
            raise CaseNotFound(transaction_id)
        return {
            "transaction_id": row["transaction_id"],
            "result": InvestigationResult.model_validate(row["result"]),
            "prediction_time": row["prediction_time"],
            "investigated_at": row["investigated_at"],
        }

    def metrics_summary(self) -> dict:
        """Real persisted aggregates only; no derived or mocked values."""
        with self._connect() as conn:
            return db.metrics_summary(conn)
