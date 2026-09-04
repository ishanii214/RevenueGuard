"""RevenueGuard FastAPI application (Phase 6 + Phase 7 additions).

Minimal, production-oriented surface. All handlers delegate to
``backend.service.RevenueGuardService``; no XGBoost, LangGraph, policy, or
SQL logic lives here. Case payloads are allowlisted projections that never
include ``recovery_outcome``. There are deliberately no execution endpoints.

Phase 7 additions (both approved): configurable CORS for the dashboard
(``REVENUEGUARD_CORS_ORIGINS``, default the local Vite origin) and
``GET /metrics/summary`` serving only real persisted aggregates.
"""

import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.schemas import InvestigationResult

from backend.repository import CASE_SUMMARY_FIELDS
from backend.service import CaseNotFound, DatabaseUnavailable, RevenueGuardService

DEFAULT_CORS_ORIGIN = "http://localhost:5173"


class CaseSummary(BaseModel):
    transaction_id: str
    customer_id: str
    created_at: datetime
    amount: float
    currency: str
    payment_method: str
    status: str


class CaseDetail(CaseSummary):
    pass


class CaseListResponse(BaseModel):
    items: list[CaseSummary]
    total: int
    limit: int
    offset: int


class InvestigationResponse(BaseModel):
    transaction_id: str
    prediction_time: datetime | None
    investigated_at: datetime
    result: InvestigationResult


class StartInvestigationRequest(BaseModel):
    use_llm: bool = False
    use_database: bool = False


class HealthResponse(BaseModel):
    status: str
    database: bool
    model_artifact: bool


class MetricsSummaryResponse(BaseModel):
    """Real persisted aggregates only (Phase 7 decision B)."""

    failed_transactions: int = Field(ge=0)
    investigated_cases: int = Field(ge=0)
    recommendations: dict[str, int]
    final_actions: dict[str, int]
    policy_decisions: dict[str, int]
    execution_authorized_count: int = Field(ge=0)


def _allowed_cors_origins() -> list[str]:
    raw = (os.environ.get("REVENUEGUARD_CORS_ORIGINS") or DEFAULT_CORS_ORIGIN).strip()
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(service: RevenueGuardService | None = None) -> FastAPI:
    service = service or RevenueGuardService()
    app = FastAPI(title="RevenueGuard API", version="0.7.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        # Informational readiness: always 200, with `status` and `database`
        # fields conveying degradation. Connection failures inside the
        # service are reported as database=False, not server errors.
        return service.health()

    @app.get("/cases", response_model=CaseListResponse)
    def list_cases(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        try:
            return service.list_cases(limit=limit, offset=offset)
        except DatabaseUnavailable:
            raise HTTPException(status_code=503, detail="database unavailable")

    @app.get("/cases/{transaction_id}", response_model=CaseDetail)
    def get_case(transaction_id: str):
        try:
            return service.get_case(transaction_id)
        except DatabaseUnavailable:
            raise HTTPException(status_code=503, detail="database unavailable")
        except CaseNotFound:
            raise HTTPException(status_code=404, detail="case not found")

    @app.post("/cases/{transaction_id}/investigation", response_model=InvestigationResponse)
    def start_investigation(transaction_id: str, request: StartInvestigationRequest):
        try:
            return service.investigate_transaction(
                transaction_id,
                use_database=request.use_database,
                use_llm=request.use_llm,
            )
        except CaseNotFound:
            raise HTTPException(status_code=404, detail="case not found")

    @app.get("/cases/{transaction_id}/investigation", response_model=InvestigationResponse)
    def get_investigation(transaction_id: str):
        try:
            return service.get_investigation(transaction_id)
        except DatabaseUnavailable:
            raise HTTPException(status_code=503, detail="database unavailable")
        except CaseNotFound:
            raise HTTPException(status_code=404, detail="investigation not found")

    @app.get("/metrics/summary", response_model=MetricsSummaryResponse)
    def metrics_summary():
        try:
            return service.metrics_summary()
        except DatabaseUnavailable:
            raise HTTPException(status_code=503, detail="database unavailable")

    return app


app = create_app()
