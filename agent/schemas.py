"""Typed schemas for the Phase 3 investigation layer.

Phase 3 boundary (binding for every schema here): recommendations are
investigative recommendations, not financially authorized decisions. The
agent must never execute or authorize a financial action; the later
deterministic policy/guardrail layer controls whether any action is permitted.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

RecommendationAction = Literal["RETRY", "REVIEW", "IGNORE"]
HistoryOutcome = Literal["completed", "recovered", "failed_pending", "unknown"]

EvidenceSource = Literal[
    "transaction_details",
    "customer_profile",
    "initial_attempt",
    "failure_details",
    "customer_history",
    "recovery_history",
    "recovery_prediction",
]

_RETRY = "RETRY"
_REVIEW = "REVIEW"
_IGNORE = "IGNORE"

# Heuristic defaults for the deterministic recommendation mapping. They are
# NOT tuned and NOT the financial policy engine (arrives in a later phase).
RETRY_PROBABILITY_THRESHOLD = 0.5
IGNORE_PROBABILITY_THRESHOLD = 0.2


class RecoveryPrediction(BaseModel):
    transaction_id: str
    probability: float = Field(ge=0.0, le=1.0)
    model_path: str
    prediction_time: datetime


class InvestigationInput(BaseModel):
    transaction_id: str
    data_dir: str = "data"
    model_path: str = "models/baseline/model.json"
    prediction: RecoveryPrediction | None = None


class TransactionDetails(BaseModel):
    """The failed transaction itself. Deliberately has NO recovery_outcome
    field: the label is structurally unrepresentable in investigation
    evidence."""

    source: Literal["transaction_details"] = "transaction_details"
    transaction_id: str
    customer_id: str
    created_at: datetime
    amount: float
    currency: str
    payment_method: str


class CustomerProfile(BaseModel):
    source: Literal["customer_profile"] = "customer_profile"
    customer_id: str
    signup_date: date
    customer_segment: str
    country: str
    preferred_payment_method: str


class AttemptRecord(BaseModel):
    source: Literal["initial_attempt"] = "initial_attempt"
    attempt_id: str
    transaction_id: str
    attempt_number: int
    attempted_at: datetime
    status: str
    payment_method: str


class FailureRecord(BaseModel):
    source: Literal["failure_details"] = "failure_details"
    failure_id: str
    attempt_id: str
    transaction_id: str
    customer_id: str
    failed_at: datetime
    failure_reason: str
    processor_response_code: str


class HistoryEntry(BaseModel):
    transaction_id: str
    created_at: datetime
    amount: float
    currency: str
    payment_method: str
    known_outcome: HistoryOutcome


class CustomerHistory(BaseModel):
    source: Literal["customer_history"] = "customer_history"
    customer_id: str
    as_of: datetime
    entries: list[HistoryEntry]


class RecoveryHistory(BaseModel):
    source: Literal["recovery_history"] = "recovery_history"
    customer_id: str
    as_of: datetime
    known_recovered_count: int
    entries: list[HistoryEntry]


class PredictionRecord(BaseModel):
    source: Literal["recovery_prediction"] = "recovery_prediction"
    probability: float
    model_path: str
    prediction_time: datetime
    note: str = "XGBoost baseline (Phase 2 artifact); available before investigation"


EvidencePayload = Annotated[
    Union[
        TransactionDetails,
        CustomerProfile,
        AttemptRecord,
        FailureRecord,
        CustomerHistory,
        RecoveryHistory,
        PredictionRecord,
    ],
    Field(discriminator="source"),
]


class Evidence(BaseModel):
    source: EvidenceSource
    as_of: datetime
    payload: EvidencePayload | None = None
    missing_reason: str | None = None

    @model_validator(mode="after")
    def _payload_xor_missing(self):
        if self.payload is None and self.missing_reason is None:
            raise ValueError("evidence requires payload or missing_reason")
        if self.payload is not None and self.missing_reason is not None:
            raise ValueError("evidence cannot have both payload and missing_reason")
        return self


class Finding(BaseModel):
    statement: str
    based_on: list[EvidenceSource]


class Recommendation(BaseModel):
    action: RecommendationAction
    rationale: str
    contributing_factors: list[str]
    policy_check_required: Literal[True] = True


class InvestigationResult(BaseModel):
    transaction_id: str
    prediction: RecoveryPrediction | None
    evidence: list[Evidence]
    findings: list[Finding]
    risk_flags: list[str]
    recommendation: Recommendation
    errors: list[str]
    llm_review: "LLMReview | None" = None


class NarrativeKeyFinding(BaseModel):
    statement: str
    evidence_references: list[str] = Field(default_factory=list)


class InvestigationNarrative(BaseModel):
    """Typed LLM narration output. Advisory only: the deterministic Phase 3
    recommendation remains authoritative, and this schema deliberately has no
    fields capable of expressing execution or authorization."""

    summary: str
    key_findings: list[NarrativeKeyFinding]
    supporting_evidence: list[str] = Field(default_factory=list)
    uncertainty: str
    prediction_interpretation: str
    recommended_action: RecommendationAction
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_references: list[str] = Field(default_factory=list)


class LLMReview(BaseModel):
    narrative: InvestigationNarrative
    validation: dict
    advisory_only: Literal[True] = True
    deterministic_recommendation: RecommendationAction
    llm_recommendation: RecommendationAction
    agrees_with_deterministic: bool


InvestigationResult.model_rebuild()
