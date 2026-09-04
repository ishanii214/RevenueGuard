"""LangGraph investigation workflow (Phase 3, fully deterministic).

START -> load_case -> gather_evidence -> analyze -> recommend -> END

Every node is a pure function of the investigation state. There is no LLM in
this phase: reasoning is encoded as transparent, rule-based evidence synthesis.
The graph is LLM-ready — a narration/reasoning node can be added later without
changing schemas, tools, or state.

Boundary: this workflow produces investigative recommendations only. It never
executes or authorizes a financial action; the later deterministic
policy/guardrail layer controls whether any action is permitted.
"""

import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_data as gd  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from agent.data_repository import get_repository  # noqa: E402
from agent.prediction import get_recovery_prediction  # noqa: E402
from agent.schemas import (  # noqa: E402
    Finding,
    InvestigationInput,
    InvestigationResult,
    Recommendation,
    RETRY_PROBABILITY_THRESHOLD,
    IGNORE_PROBABILITY_THRESHOLD,
)
from agent.state import InvestigationState, initial_state  # noqa: E402
from agent.tools import build_evidence  # noqa: E402

_EPOCH = datetime(1970, 1, 1)
_GRAPH = None


def _as_of(state: InvestigationState) -> datetime:
    """The prediction point: the initial failure timestamp when known.

    Falls back to transaction creation time, then to a fixed epoch sentinel —
    never to wall-clock time — so investigations stay deterministic.
    """
    prediction = state.get("prediction")
    if prediction is not None:
        return prediction.prediction_time
    repo = get_repository(state["investigation_input"].data_dir)
    attempt = repo.get_initial_attempt(state["investigation_input"].transaction_id)
    if attempt is not None:
        return attempt.attempted_at
    details = repo.get_transaction(state["investigation_input"].transaction_id)
    if details is not None:
        return details.created_at
    return _EPOCH


def _load_case(state: InvestigationState) -> dict:
    investigation_input = state["investigation_input"]
    errors = list(state.get("errors", []))
    prediction = investigation_input.prediction
    if prediction is None:
        prediction = get_recovery_prediction(
            investigation_input.transaction_id,
            investigation_input.data_dir,
            investigation_input.model_path,
        )
        if prediction is None:
            errors.append(f"recovery prediction unavailable for {investigation_input.transaction_id}")
    repo = get_repository(investigation_input.data_dir)
    if repo.get_transaction(investigation_input.transaction_id) is None:
        errors.append(f"transaction {investigation_input.transaction_id} not found")
    return {"prediction": prediction, "errors": errors}


def _gather_evidence(state: InvestigationState) -> dict:
    investigation_input = state["investigation_input"]
    repo = get_repository(investigation_input.data_dir)
    details = repo.get_transaction(investigation_input.transaction_id)
    evidence = build_evidence(
        repo,
        investigation_input.transaction_id,
        details.customer_id if details else None,
        _as_of(state),
        state.get("prediction"),
    )
    return {"evidence": evidence}


def _analyze(state: InvestigationState) -> dict:
    evidence = {item.source: item for item in state["evidence"]}
    prediction = state.get("prediction")
    findings: list[Finding] = []
    risk_flags: list[str] = []

    transaction_payload = evidence["transaction_details"].payload
    if transaction_payload is not None:
        findings.append(
            Finding(
                statement=(
                    f"Failed transaction of {transaction_payload.amount:.2f} "
                    f"{transaction_payload.currency} via {transaction_payload.payment_method}."
                ),
                based_on=["transaction_details"],
            )
        )
        if transaction_payload.amount >= gd.HIGH_VALUE_THRESHOLD:
            risk_flags.append("high_value")

    failure_payload = evidence["failure_details"].payload
    if failure_payload is not None:
        if failure_payload.failure_reason in gd.AUTO_RETRY_REASONS:
            retry_class = "auto-retryable"
        elif failure_payload.failure_reason in gd.NON_RETRYABLE_REASONS:
            retry_class = "not retryable on the same payment method"
        else:
            retry_class = "not auto-retryable (risk hold)"
        findings.append(
            Finding(
                statement=(
                    f"Initial failure reason '{failure_payload.failure_reason}' "
                    f"(processor code {failure_payload.processor_response_code}) — {retry_class}."
                ),
                based_on=["failure_details"],
            )
        )
        if failure_payload.failure_reason in gd.NON_RETRYABLE_REASONS:
            risk_flags.append("non_retryable_reason")
        if failure_payload.failure_reason == "risk_review_hold":
            risk_flags.append("risk_review_hold")

    if prediction is not None:
        findings.append(
            Finding(
                statement=f"XGBoost recovery probability {prediction.probability:.3f}.",
                based_on=["recovery_prediction"],
            )
        )

    history_payload = evidence["customer_history"].payload
    if history_payload is not None:
        pending = sum(1 for entry in history_payload.entries if entry.known_outcome == "failed_pending")
        recovered = sum(1 for entry in history_payload.entries if entry.known_outcome == "recovered")
        findings.append(
            Finding(
                statement=(
                    f"Customer history as of the prediction point: {len(history_payload.entries)} prior "
                    f"transactions, {pending} failed and not yet recovered, {recovered} known recoveries."
                ),
                based_on=["customer_history"],
            )
        )
        if not history_payload.entries:
            risk_flags.append("no_prior_history")

    missing = sorted(item.source for item in state["evidence"] if item.payload is None)
    if missing:
        risk_flags.append("incomplete_evidence")
        findings.append(
            Finding(statement="Evidence unavailable: " + ", ".join(missing) + ".", based_on=[])
        )

    return {"findings": findings, "risk_flags": risk_flags}


def _recommend(state: InvestigationState) -> dict:
    evidence = {item.source: item for item in state["evidence"]}
    prediction = state.get("prediction")
    failure_payload = evidence["failure_details"].payload
    probability = prediction.probability if prediction is not None else None
    reason = failure_payload.failure_reason if failure_payload is not None else None
    auto_retryable = reason in gd.AUTO_RETRY_REASONS if reason is not None else False

    contributing_factors: list[str] = []
    if probability is not None:
        contributing_factors.append(f"recovery_probability={probability:.3f}")
    if reason is not None:
        contributing_factors.append(
            f"failure_reason={reason}" + (" (auto-retryable)" if auto_retryable else "")
        )
    contributing_factors.extend(state.get("risk_flags", []))

    if probability is None or reason is None:
        action = "REVIEW"
        rationale = "Investigation incomplete (missing prediction or failure details); manual review required."
    elif auto_retryable and probability >= RETRY_PROBABILITY_THRESHOLD:
        action = "RETRY"
        rationale = (
            f"Auto-retryable failure reason with recovery probability {probability:.3f} "
            f">= {RETRY_PROBABILITY_THRESHOLD}; recommend automated retry pending policy approval."
        )
    elif probability < IGNORE_PROBABILITY_THRESHOLD:
        action = "IGNORE"
        rationale = (
            f"Recovery probability {probability:.3f} below {IGNORE_PROBABILITY_THRESHOLD}; "
            "further recovery effort not justified by current evidence."
        )
    else:
        action = "REVIEW"
        rationale = (
            f"Recovery probability {probability:.3f} is in the ambiguous band or the failure reason "
            "is not auto-retryable; recommend human review."
        )

    result = InvestigationResult(
        transaction_id=state["investigation_input"].transaction_id,
        prediction=prediction,
        evidence=state["evidence"],
        findings=state["findings"],
        risk_flags=state.get("risk_flags", []),
        recommendation=Recommendation(
            action=action,
            rationale=rationale,
            contributing_factors=contributing_factors,
        ),
        errors=state.get("errors", []),
    )
    return {"result": result}


def build_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("load_case", _load_case)
    graph.add_node("gather_evidence", _gather_evidence)
    graph.add_node("analyze", _analyze)
    graph.add_node("recommend", _recommend)
    graph.add_edge(START, "load_case")
    graph.add_edge("load_case", "gather_evidence")
    graph.add_edge("gather_evidence", "analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


def investigate(investigation_input: InvestigationInput) -> InvestigationResult:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final_state = _GRAPH.invoke(initial_state(investigation_input))
    return final_state["result"]
