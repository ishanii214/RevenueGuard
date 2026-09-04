"""LangGraph investigation state (Phases 3 + 4).

Nodes are pure functions over this state; no wall-clock values enter it.
The Phase 4 keys (llm_narrative, llm_validation) carry advisory LLM output
only; the deterministic recommendation never depends on them.
"""

from typing import TypedDict

from agent.schemas import (
    Evidence,
    Finding,
    InvestigationInput,
    InvestigationNarrative,
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
    llm_narrative: InvestigationNarrative | None
    llm_validation: dict


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
