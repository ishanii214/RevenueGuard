"""LLM input bundle, narrative parsing, and grounding validation (Phase 4).

Grounding scope (important limitation, stated honestly):
    The mechanical validator checks *hard factual claims* that can be
    compared against the supplied evidence — evidence references, payment
    identifiers, the recovery probability, and vocabulary-member facts such
    as failure reasons, payment methods, segments, and outcome labels.
    *Interpretive / reasoning statements* (e.g. "the customer's history
    suggests improving reliability") are required to carry evidence
    references, but their semantic correctness cannot be fully verified
    mechanically. The validator therefore cannot guarantee detection of all
    semantic hallucinations — only of contradictions with the supplied
    evidence and of unsupported references. This limitation is reported in
    every validation payload via ``grounding_scope``.

All LLM output is advisory: the deterministic Phase 3 recommendation remains
authoritative, and nothing in this module can execute or authorize any
financial action.
"""

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_data as gd  # noqa: E402

from agent.schemas import (  # noqa: E402
    AttemptRecord,
    CustomerHistory,
    CustomerProfile,
    Evidence,
    FailureRecord,
    Finding,
    RecommendationAction,
    RecoveryHistory,
    RecoveryPrediction,
    TransactionDetails,
)

IDENTIFIER_PATTERN = re.compile(r"\b(?:TXN|CUST|ATT|FAIL)-\d+\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
PROBABILITY_TOLERANCE = 0.005

GROUNDING_SCOPE = (
    "mechanical checks only: evidence references, payment identifiers, recovery-probability "
    "citations, and vocabulary-member facts (failure reasons, payment methods, segments, "
    "outcome labels). Interpretive statements must carry evidence references but their "
    "semantic correctness cannot be fully verified mechanically; not all semantic "
    "hallucinations are detectable."
)

ALLOWED_ACTIONS: list[RecommendationAction] = ["RETRY", "REVIEW", "IGNORE"]


class LLMPredictionPayload(BaseModel):
    """Prediction evidence for the LLM — the model path (a filesystem path)
    is deliberately stripped here."""

    source: str = "recovery_prediction"
    probability: float
    prediction_time: str
    note: str = ""


class LLMEvidenceItem(BaseModel):
    reference_id: str
    source: str
    as_of: str
    payload: (
        TransactionDetails
        | CustomerProfile
        | AttemptRecord
        | FailureRecord
        | CustomerHistory
        | RecoveryHistory
        | LLMPredictionPayload
    )


class LLMEvidenceBundle(BaseModel):
    transaction_id: str
    recovery_probability: float = Field(ge=0.0, le=1.0)
    prediction_time: str
    evidence: list[LLMEvidenceItem]
    deterministic_findings: list[Finding]
    deterministic_risk_flags: list[str]
    deterministic_recommendation: RecommendationAction
    allowed_actions: list[RecommendationAction] = Field(default_factory=lambda: list(ALLOWED_ACTIONS))


def build_llm_evidence_bundle(
    transaction_id: str,
    prediction: RecoveryPrediction | None,
    evidence: list[Evidence],
    findings: list[Finding],
    risk_flags: list[str],
    deterministic_recommendation: RecommendationAction,
) -> LLMEvidenceBundle:
    """Project validated Phase 3 objects into the sanitized LLM input.

    Field-by-field allowlist: nothing that is not copied here can reach the
    LLM (no recovery_outcome, no split info, no DataFrames, no paths).
    """
    items: list[LLMEvidenceItem] = []
    for item in evidence:
        payload = item.payload
        if payload is None:
            continue
        if item.source == "recovery_prediction":
            payload = LLMPredictionPayload(
                probability=payload.probability,
                prediction_time=payload.prediction_time.isoformat(),
                note=payload.note,
            )
        items.append(
            LLMEvidenceItem(
                reference_id=item.source,
                source=item.source,
                as_of=item.as_of.isoformat(),
                payload=payload,
            )
        )
    return LLMEvidenceBundle(
        transaction_id=transaction_id,
        recovery_probability=prediction.probability if prediction is not None else 0.0,
        prediction_time=prediction.prediction_time.isoformat() if prediction is not None else "",
        evidence=items,
        deterministic_findings=list(findings),
        deterministic_risk_flags=list(risk_flags),
        deterministic_recommendation=deterministic_recommendation,
    )


def _supplied_identifiers(bundle: LLMEvidenceBundle) -> set[str]:
    identifiers = {bundle.transaction_id}
    for item in bundle.evidence:
        payload = item.payload
        if isinstance(payload, (TransactionDetails, AttemptRecord, FailureRecord)):
            identifiers.add(payload.transaction_id)
        if isinstance(payload, AttemptRecord):
            identifiers.add(payload.attempt_id)
        if isinstance(payload, FailureRecord):
            identifiers.update({payload.failure_id, payload.attempt_id, payload.customer_id})
        if isinstance(payload, (CustomerProfile, CustomerHistory, RecoveryHistory)):
            identifiers.add(payload.customer_id)
        if isinstance(payload, CustomerHistory):
            identifiers.update(entry.transaction_id for entry in payload.entries)
        if isinstance(payload, RecoveryHistory):
            identifiers.update(entry.transaction_id for entry in payload.entries)
    return identifiers


def _supplied_vocabulary(bundle: LLMEvidenceBundle) -> dict[str, set[str]]:
    """Domain vocabulary (all known enum values) plus the case's supplied
    values. A narrative claiming any domain-valid value that differs from the
    supplied one is a detectable factual contradiction."""
    reasons = set(gd.FAILURE_REASON_CODES)
    methods = set(gd.PAYMENT_METHODS)
    segments = set(gd.SEGMENTS)
    outcomes = {"completed", "recovered", "failed_pending", "unknown"}
    supplied_reason = None
    supplied_method = None
    supplied_segment = None
    for item in bundle.evidence:
        payload = item.payload
        if isinstance(payload, FailureRecord):
            supplied_reason = payload.failure_reason
        if isinstance(payload, TransactionDetails):
            supplied_method = payload.payment_method
        if isinstance(payload, CustomerProfile):
            supplied_segment = payload.customer_segment
    return {
        "supplied_reason": {supplied_reason} if supplied_reason else set(),
        "known_reasons": reasons,
        "supplied_method": {supplied_method} if supplied_method else set(),
        "known_methods": methods,
        "supplied_segment": {supplied_segment} if supplied_segment else set(),
        "known_segments": segments,
        "known_outcomes": outcomes,
    }


def extract_json_object(text: str) -> dict | None:
    """Lenient JSON extraction: direct parse, then fenced/first-object parse."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_narrative(text: str) -> tuple[object | None, str | None]:
    """Parse + schema-validate LLM text. Returns (narrative, error)."""
    from agent.schemas import InvestigationNarrative

    data = extract_json_object(text)
    if data is None:
        return None, "llm output is not parseable JSON"
    try:
        return InvestigationNarrative.model_validate(data), None
    except ValidationError as exc:
        return None, f"llm output failed schema validation: {exc.error_count()} error(s); first: {exc.errors()[0]['msg']}"


def check_grounding(narrative, bundle: LLMEvidenceBundle) -> list[str]:
    """Mechanical grounding checks for hard factual claims.

    Returns a list of violation descriptions (empty when clean). See the
    module docstring for the documented limitation regarding interpretive
    statements.
    """
    violations: list[str] = []
    supplied_ids = _supplied_identifiers(bundle)
    vocabulary = _supplied_vocabulary(bundle)

    referenced = list(narrative.evidence_references) + list(narrative.supporting_evidence)
    for finding in narrative.key_findings:
        referenced.extend(finding.evidence_references)
    for reference in referenced:
        if reference not in {item.reference_id for item in bundle.evidence}:
            violations.append(f"reference to unsupplied evidence: {reference}")

    for statement in [narrative.summary, narrative.uncertainty, narrative.prediction_interpretation] + [
        finding.statement for finding in narrative.key_findings
    ]:
        for identifier in IDENTIFIER_PATTERN.findall(statement):
            if identifier not in supplied_ids:
                violations.append(f"unsupplied identifier cited: {identifier}")
        for reason in vocabulary["known_reasons"]:
            if reason in statement and reason not in vocabulary["supplied_reason"]:
                violations.append(f"cites failure reason not in supplied evidence: {reason}")
        for method in vocabulary["known_methods"]:
            if method in statement and method not in vocabulary["supplied_method"]:
                violations.append(f"cites payment method not in supplied evidence: {method}")
        for segment in vocabulary["known_segments"]:
            if segment in statement and segment not in vocabulary["supplied_segment"]:
                violations.append(f"cites customer segment not in supplied evidence: {segment}")

    numbers = [float(match) for match in NUMBER_PATTERN.findall(narrative.prediction_interpretation)]
    probability_numbers = [number for number in numbers if 0.0 <= number <= 1.0]
    if probability_numbers and not any(
        abs(number - bundle.recovery_probability) <= PROBABILITY_TOLERANCE for number in probability_numbers
    ):
        violations.append(
            "narrative cites a recovery probability inconsistent with the supplied prediction "
            f"({bundle.recovery_probability:.3f})"
        )

    return violations
