"""Deterministic financial policy / guardrails (Phase 5).

The policy layer answers a different question from the investigation:
the investigation recommends what the evidence suggests; the policy decides
whether that action is permitted under financial controls. It is:

- deterministic and pure (no I/O, no wall clock, no RNG — ``evaluated_at``
  is the case prediction point);
- authoritative for action eligibility (``final_action``);
- fail-closed (missing, invalid, or inconsistent information never yields an
  allowed financial action);
- structurally isolated from the LLM (it reads only the deterministic
  recommendation and deterministic evidence — never ``llm_review``).

Authorization semantics (the gate matrix — see agent.schemas.PolicyEvaluation):

    requested_action | policy_decision | final_action | execution_authorized
    RETRY            | ALLOWED         | RETRY        | True
    RETRY            | DENIED          | REVIEW       | False
    REVIEW           | ALLOWED         | REVIEW       | False
    IGNORE           | ALLOWED         | IGNORE       | False

``policy_decision="ALLOWED"`` by itself does NOT authorize a financial action;
``execution_authorized`` is the gate and is True only for RETRY + ALLOWED.
REVIEW and IGNORE are policy-allowed because they execute no financial action.

Nothing in this module executes, triggers, or simulates any payment.
"""

import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_data as gd  # noqa: E402

from agent.schemas import (  # noqa: E402
    AttemptRecord,
    FailureRecord,
    PolicyConfig,
    PolicyEvaluation,
    PolicyRequest,
    TransactionDetails,
)

DEFAULT_POLICY_CONFIG = PolicyConfig(
    policy_version="p5.v1",
    auto_retry_enabled=True,
    max_total_attempts=4,
    retry_probability_floor=0.30,
    high_value_threshold=5000.00,
)

REQUIRED_EVIDENCE_SOURCES = (
    "transaction_details",
    "customer_profile",
    "initial_attempt",
    "failure_details",
)

# Consistency checks (exactly three, per the approved definition):
C1_INVALID_PROBABILITY = "C1_invalid_probability"
C2_INVALID_ATTEMPT_METADATA = "C2_invalid_attempt_metadata"
C3_CONTRADICTORY_CASE_IDENTITY = "C3_contradictory_case_identity"
CONSISTENCY_CHECKS = (C1_INVALID_PROBABILITY, C2_INVALID_ATTEMPT_METADATA, C3_CONTRADICTORY_CASE_IDENTITY)

RETRY_GUARDRAILS = (
    "auto_retry_enabled",
    "required_evidence",
    "evidence_consistency",
    "retryable_reason",
    "risk_hold_review",
    "high_value_review",
    "retry_cap",
    "probability_floor",
)
SAFE_PATH_GUARDRAILS = ("safe_path_no_execution",)

_DENY_EXPLANATIONS = {
    "auto_retry_disabled": "auto-retry is disabled by policy configuration",
    "missing_required_evidence": "required investigation evidence is missing (fail closed)",
    "inconsistent_evidence": "evidence consistency checks failed (C1/C2/C3)",
    "non_retryable_failure_reason": "the failure reason cannot succeed on the same payment method",
    "risk_review_hold_requires_human_review": "risk-review holds require human disposition",
    "high_value_auto_retry_prohibited": "the transaction amount meets the high-value threshold",
    "retry_cap_exceeded": "the known attempt count has reached the policy cap",
    "retry_probability_below_policy_floor": "the recovery probability is below the policy floor or unavailable",
    "invalid_requested_action": "the requested action is outside the permitted vocabulary",
}


def consistency_violations(
    transaction_id: str,
    prediction_probability: float | None,
    attempt: AttemptRecord | None,
    failure: FailureRecord | None,
    details: TransactionDetails | None,
) -> list[str]:
    """Mechanical C1/C2/C3 consistency checks (see the approved definition).

    C1 invalid probability — recovery probability present but outside [0, 1].
    C2 invalid/non-initial attempt metadata — attempt_number != 1, or the
       failure does not reference the initial attempt.
    C3 contradictory case identity — attempt/failure reference a different
       transaction, or the failure's customer differs from the transaction's.

    Normal policy violations (e.g. expired_card + RETRY) are NEVER classified
    here — they remain ``non_retryable_failure_reason``.
    """
    violations: list[str] = []
    if prediction_probability is not None and not (0.0 <= prediction_probability <= 1.0):
        violations.append(C1_INVALID_PROBABILITY)
    if attempt is not None and attempt.attempt_number != 1:
        violations.append(C2_INVALID_ATTEMPT_METADATA)
    if attempt is not None and failure is not None and failure.attempt_id != attempt.attempt_id:
        violations.append(C2_INVALID_ATTEMPT_METADATA)
    if attempt is not None and attempt.transaction_id != transaction_id:
        violations.append(C3_CONTRADICTORY_CASE_IDENTITY)
    if failure is not None and failure.transaction_id != transaction_id:
        violations.append(C3_CONTRADICTORY_CASE_IDENTITY)
    if details is not None and failure is not None and failure.customer_id != details.customer_id:
        violations.append(C3_CONTRADICTORY_CASE_IDENTITY)
    return violations


def build_policy_request(state) -> PolicyRequest:
    """Deterministic, point-in-time-safe projection of the investigated case.

    ``known_attempt_count`` counts only attempts known at the case prediction
    timestamp (Phase 2/3 temporal rule). The repository structurally exposes
    only attempt 1 as the initial attempt, so the count is 1 when initial
    attempt evidence exists and 0 otherwise — future retries (attempts >= 2,
    which exist in the dataset) are never counted here. The field exists so
    the future simulated-intervention loop can re-evaluate the cap with
    updated, legitimately-known counts.
    """
    evidence = {item.source: item for item in state["evidence"]}
    result = state["result"]
    prediction = state.get("prediction")

    def _payload(source: str):
        item = evidence.get(source)
        return item.payload if item is not None else None

    details = _payload("transaction_details")
    attempt = _payload("initial_attempt")
    failure = _payload("failure_details")
    history = _payload("customer_history")

    prior_failed = None
    prior_recovered = None
    if history is not None:
        prior_failed = sum(1 for entry in history.entries if entry.known_outcome in ("failed_pending", "recovered"))
        prior_recovered = sum(1 for entry in history.entries if entry.known_outcome == "recovered")

    present_sources = [source for source in evidence if evidence[source].payload is not None]
    return PolicyRequest(
        requested_action=result.recommendation.action,
        recovery_probability=prediction.probability if prediction is not None else None,
        failure_reason=failure.failure_reason if failure is not None else None,
        known_attempt_count=1 if attempt is not None else 0,
        amount=details.amount if details is not None else None,
        prior_failed_count=prior_failed,
        prior_recovered_count=prior_recovered,
        required_evidence_present=all(
            source in evidence and evidence[source].payload is not None for source in REQUIRED_EVIDENCE_SOURCES
        ),
        present_evidence_sources=present_sources,
        consistency_violations=consistency_violations(
            transaction_id=result.transaction_id,
            prediction_probability=prediction.probability if prediction is not None else None,
            attempt=attempt,
            failure=failure,
            details=details,
        ),
    )


class _RetryEvaluation(BaseModel):
    reason_codes: list[str]
    decision: str
    final_action: str
    execution_authorized: bool


def _evaluate_retry(request: PolicyRequest, config: PolicyConfig) -> _RetryEvaluation:
    """Evaluate all RETRY guardrails and collect every violated one."""
    codes: list[str] = []

    if not config.auto_retry_enabled:
        codes.append("auto_retry_disabled")
    if not request.required_evidence_present:
        codes.append("missing_required_evidence")
    if request.consistency_violations:
        codes.append("inconsistent_evidence")

    if request.failure_reason is None:
        if request.required_evidence_present:
            # Contradiction: evidence marked present but no reason available.
            codes.append("missing_required_evidence")
    elif request.failure_reason == "risk_review_hold":
        codes.append("risk_review_hold_requires_human_review")
    elif request.failure_reason not in gd.AUTO_RETRY_REASONS:
        codes.append("non_retryable_failure_reason")

    if request.known_attempt_count is None:
        codes.append("missing_required_evidence")
    elif request.known_attempt_count >= config.max_total_attempts:
        codes.append("retry_cap_exceeded")

    if request.amount is None:
        codes.append("missing_required_evidence")
    elif request.amount >= config.high_value_threshold:
        codes.append("high_value_auto_retry_prohibited")

    if request.recovery_probability is None:
        codes.append("retry_probability_below_policy_floor")  # fail closed: unavailable is unfavourable
    elif C1_INVALID_PROBABILITY not in request.consistency_violations and (
        request.recovery_probability < config.retry_probability_floor
    ):
        codes.append("retry_probability_below_policy_floor")

    denied = bool(codes)
    return _RetryEvaluation(
        reason_codes=codes if denied else ["retry_allowed_within_policy"],
        decision="DENIED" if denied else "ALLOWED",
        final_action="REVIEW" if denied else "RETRY",
        execution_authorized=not denied,
    )


def evaluate_policy(
    request: PolicyRequest, config: PolicyConfig, evaluated_at: datetime
) -> PolicyEvaluation:
    """Evaluate a policy request deterministically. Pure function."""
    if request.requested_action == "REVIEW":
        return PolicyEvaluation(
            requested_action="REVIEW",
            policy_decision="ALLOWED",
            final_action="REVIEW",
            reason_codes=["review_is_safe_path"],
            explanation=(
                f"Routing to human review is a safe path that executes no financial action; "
                f"policy {config.policy_version} permits it unconditionally."
            ),
            applicable_guardrails=list(SAFE_PATH_GUARDRAILS),
            policy_version=config.policy_version,
            config_snapshot=config,
            execution_authorized=False,
            evaluated_at=evaluated_at,
        )
    if request.requested_action == "IGNORE":
        return PolicyEvaluation(
            requested_action="IGNORE",
            policy_decision="ALLOWED",
            final_action="IGNORE",
            reason_codes=["ignore_requires_no_authorization"],
            explanation=(
                f"Taking no action executes no financial action; policy {config.policy_version} "
                f"requires no authorization for it."
            ),
            applicable_guardrails=list(SAFE_PATH_GUARDRAILS),
            policy_version=config.policy_version,
            config_snapshot=config,
            execution_authorized=False,
            evaluated_at=evaluated_at,
        )
    if request.requested_action == "RETRY":
        outcome = _evaluate_retry(request, config)
        if outcome.decision == "ALLOWED":
            explanation = (
                f"Auto-retry permitted by policy {config.policy_version}: failure reason "
                f"'{request.failure_reason}' is auto-retryable, recovery probability "
                f"{request.recovery_probability:.3f} >= floor {config.retry_probability_floor}, "
                f"known attempt count {request.known_attempt_count} < cap {config.max_total_attempts}, "
                f"and amount {request.amount:.2f} is below the high-value threshold "
                f"{config.high_value_threshold:.2f}."
            )
        else:
            details = "; ".join(_DENY_EXPLANATIONS[code] for code in outcome.reason_codes)
            explanation = (
                f"Auto-retry denied by policy {config.policy_version}: {details}. "
                f"Case routed to human review."
            )
        return PolicyEvaluation(
            requested_action="RETRY",
            policy_decision=outcome.decision,
            final_action=outcome.final_action,
            reason_codes=outcome.reason_codes,
            explanation=explanation,
            applicable_guardrails=list(RETRY_GUARDRAILS),
            policy_version=config.policy_version,
            config_snapshot=config,
            execution_authorized=outcome.execution_authorized,
            evaluated_at=evaluated_at,
        )
    # Defensive branch: the Pydantic schema makes this unreachable, but the
    # policy still fails closed for any out-of-vocabulary action. The output
    # schema refuses to represent an invalid action, so the safe coerced
    # action (REVIEW) is recorded alongside the invalid_requested_action code.
    return PolicyEvaluation(
        requested_action="REVIEW",
        policy_decision="DENIED",
        final_action="REVIEW",
        reason_codes=["invalid_requested_action"],
        explanation=(
            f"The requested action is outside the permitted vocabulary; policy "
            f"{config.policy_version} denies it and routes the case to human review."
        ),
        applicable_guardrails=list(SAFE_PATH_GUARDRAILS),
        policy_version=config.policy_version,
        config_snapshot=config,
        execution_authorized=False,
        evaluated_at=evaluated_at,
    )
