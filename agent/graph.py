"""LangGraph investigation workflow (Phases 3 + 4).

START -> load_case -> gather_evidence -> analyze -> llm_reason ->
validate_llm_output -> recommend -> END

The Phase 3 core (load/gather/analyze/recommend) remains fully deterministic.
The Phase 4 nodes add optional, advisory LLM narration: they receive only a
sanitized evidence bundle, their output is strongly typed and mechanically
grounding-checked, and they can never override the deterministic
recommendation or execute/authorize any financial action. When the LLM is
disabled, misconfigured, unreachable, or returns malformed output, the
workflow falls back to exact Phase 3 behavior.

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
from agent.llm import DisabledLLM, LLMRequest, make_llm_from_env  # noqa: E402
from agent.narration import (  # noqa: E402
    GROUNDING_SCOPE,
    build_llm_evidence_bundle,
    check_grounding,
    parse_narrative,
)
from agent.prediction import get_recovery_prediction  # noqa: E402
from agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402
from agent.schemas import (  # noqa: E402
    Finding,
    InvestigationInput,
    InvestigationResult,
    LLMReview,
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


def _deterministic_action(state: InvestigationState) -> str:
    """The authoritative recommendation rule (Phase 3, unchanged)."""
    evidence = {item.source: item for item in state["evidence"]}
    prediction = state.get("prediction")
    failure_payload = evidence["failure_details"].payload
    probability = prediction.probability if prediction is not None else None
    reason = failure_payload.failure_reason if failure_payload is not None else None
    if probability is None or reason is None:
        return "REVIEW"
    if reason in gd.AUTO_RETRY_REASONS and probability >= RETRY_PROBABILITY_THRESHOLD:
        return "RETRY"
    if probability < IGNORE_PROBABILITY_THRESHOLD:
        return "IGNORE"
    return "REVIEW"


def _make_llm_reason(llm_client):
    client = llm_client if llm_client is not None else make_llm_from_env()
    disabled = isinstance(client, DisabledLLM)

    def _llm_reason(state: InvestigationState) -> dict:
        if disabled:
            return {
                "llm_narrative": None,
                "llm_validation": {
                    "ok": False,
                    "error": "llm disabled",
                    "provider": "disabled",
                    "model": "",
                    "latency_ms": 0,
                },
            }
        investigation_input = state["investigation_input"]
        bundle = build_llm_evidence_bundle(
            transaction_id=investigation_input.transaction_id,
            prediction=state.get("prediction"),
            evidence=state["evidence"],
            findings=state["findings"],
            risk_flags=state.get("risk_flags", []),
            deterministic_recommendation=_deterministic_action(state),
        )
        request = LLMRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT_TEMPLATE.format(
                bundle_json=bundle.model_dump_json(),
                deterministic_action=bundle.deterministic_recommendation,
            ),
        )
        response = client.complete(request)
        base = {
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
        }
        if not response.ok:
            return {"llm_narrative": None, "llm_validation": {**base, "ok": False, "error": response.error}}
        narrative, parse_error = parse_narrative(response.text)
        if narrative is None:
            return {"llm_narrative": None, "llm_validation": {**base, "ok": False, "error": parse_error}}
        return {"llm_narrative": narrative, "llm_validation": {**base, "ok": True, "error": None}}

    return _llm_reason


def _validate_llm_output(state: InvestigationState) -> dict:
    """Schema + mechanical grounding validation of the advisory narrative.

    Malformed or schema-invalid output is rejected entirely (narrative None).
    Grounding violations are recorded but do not reject the narrative — the
    advisory object carries the violations for human review. Mechanical
    checks cover hard factual claims only (see narration.GROUNDING_SCOPE).
    """
    narrative = state.get("llm_narrative")
    validation = dict(state.get("llm_validation") or {})
    if narrative is None:
        return {"llm_validation": validation}
    investigation_input = state["investigation_input"]
    bundle = build_llm_evidence_bundle(
        transaction_id=investigation_input.transaction_id,
        prediction=state.get("prediction"),
        evidence=state["evidence"],
        findings=state["findings"],
        risk_flags=state.get("risk_flags", []),
        deterministic_recommendation=_deterministic_action(state),
    )
    violations = check_grounding(narrative, bundle)
    validation["grounding_ok"] = not violations
    validation["violations"] = violations
    validation["grounding_scope"] = GROUNDING_SCOPE
    return {"llm_validation": validation}


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

    narrative = state.get("llm_narrative")
    validation = state.get("llm_validation") or {}
    llm_review = None
    errors = list(state.get("errors", []))
    if narrative is not None:
        llm_review = LLMReview(
            narrative=narrative,
            validation=validation,
            advisory_only=True,
            deterministic_recommendation=action,
            llm_recommendation=narrative.recommended_action,
            agrees_with_deterministic=(narrative.recommended_action == action),
        )
    elif validation.get("error"):
        errors.append(f"llm narration unavailable: {validation['error']}")

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
        errors=errors,
        llm_review=llm_review,
    )
    return {"result": result}


def build_graph(llm_client=None):
    graph = StateGraph(InvestigationState)
    graph.add_node("load_case", _load_case)
    graph.add_node("gather_evidence", _gather_evidence)
    graph.add_node("analyze", _analyze)
    graph.add_node("llm_reason", _make_llm_reason(llm_client))
    graph.add_node("validate_llm_output", _validate_llm_output)
    graph.add_node("recommend", _recommend)
    graph.add_edge(START, "load_case")
    graph.add_edge("load_case", "gather_evidence")
    graph.add_edge("gather_evidence", "analyze")
    graph.add_edge("analyze", "llm_reason")
    graph.add_edge("llm_reason", "validate_llm_output")
    graph.add_edge("validate_llm_output", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


def investigate(investigation_input: InvestigationInput, llm_client=None) -> InvestigationResult:
    """Run one investigation. ``llm_client=None`` resolves the client from the
    environment (DisabledLLM when unconfigured — the deterministic default);
    pass an explicit client (or DisabledLLM()) to control narration."""
    global _GRAPH
    if llm_client is None:
        if _GRAPH is None:
            _GRAPH = build_graph(None)
        graph = _GRAPH
    else:
        graph = build_graph(llm_client)
    final_state = graph.invoke(initial_state(investigation_input))
    return final_state["result"]
