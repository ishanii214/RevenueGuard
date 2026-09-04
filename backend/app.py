"""RevenueGuard FastAPI application (Phase 6).

Minimal, production-oriented surface. All handlers delegate to
``backend.service.RevenueGuardService``; no XGBoost, LangGraph, policy, or
SQL logic lives here. Case payloads are allowlisted projections that never
include ``recovery_outcome``. There are deliberately no execution endpoints.
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from agent.schemas import InvestigationResult

from backend.repository import CASE_SUMMARY_FIELDS
from backend.service import CaseNotFound, DatabaseUnavailable, RevenueGuardService


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


def create_app(service: RevenueGuardService | None = None) -> FastAPI:
    service = service or RevenueGuardService()
    app = FastAPI(title="RevenueGuard API", version="0.6.0")

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

    return app


app = create_app()
