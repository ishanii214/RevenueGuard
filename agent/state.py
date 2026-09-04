"""LangGraph investigation state (Phase 3).

Nodes are pure functions over this state; no wall-clock values enter it.
"""

from typing import TypedDict

from agent.schemas import (
    Evidence,
    Finding,
    InvestigationInput,
    InvestigationResult,
    RecoveryPrediction,
)


class InvestigationState(TypedDict, total=False):
    investigation_input: InvestigationInput
    prediction: RecoveryPrediction | None
    evidence: list[Evidence]
    findings: list[Finding]
    risk_flags: list[str]
    result: InvestigationResult | None
    errors: list[str]


def initial_state(investigation_input: InvestigationInput) -> InvestigationState:
    return {
        "investigation_input": investigation_input,
        "prediction": None,
        "evidence": [],
        "findings": [],
        "risk_flags": [],
        "result": None,
        "errors": [],
    }
