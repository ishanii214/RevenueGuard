"""Deterministic investigation tools (Phase 3).

Each tool wraps one ``CaseRepository`` accessor and returns typed ``Evidence``
stamped with the case prediction point. Unknown records become evidence with
a ``missing_reason`` instead of exceptions. Tools read only; they never
mutate data and never execute financial actions.
"""

from datetime import datetime

from agent.data_repository import CaseRepository
from agent.schemas import (
    Evidence,
    EvidenceSource,
    PredictionRecord,
    RecoveryPrediction,
)

EVIDENCE_ORDER: tuple[EvidenceSource, ...] = (
    "transaction_details",
    "customer_profile",
    "initial_attempt",
    "failure_details",
    "customer_history",
    "recovery_history",
    "recovery_prediction",
)


def build_evidence(
    repo: CaseRepository,
    transaction_id: str,
    customer_id: str | None,
    as_of: datetime,
    prediction: RecoveryPrediction | None,
) -> list[Evidence]:
    """Collect evidence from every tool in a fixed, deterministic order."""
    details = repo.get_transaction(transaction_id)
    if customer_id is None and details is not None:
        customer_id = details.customer_id

    evidence: list[Evidence] = []

    evidence.append(
        Evidence(source="transaction_details", as_of=as_of, payload=details)
        if details is not None
        else Evidence(source="transaction_details", as_of=as_of, missing_reason="transaction not found")
    )

    if customer_id is None:
        evidence.append(
            Evidence(source="customer_profile", as_of=as_of, missing_reason="customer unknown (transaction not found)")
        )
    else:
        profile = repo.get_customer(customer_id)
        evidence.append(
            Evidence(source="customer_profile", as_of=as_of, payload=profile)
            if profile is not None
            else Evidence(source="customer_profile", as_of=as_of, missing_reason="customer not found")
        )

    attempt = repo.get_initial_attempt(transaction_id)
    evidence.append(
        Evidence(source="initial_attempt", as_of=as_of, payload=attempt)
        if attempt is not None
        else Evidence(source="initial_attempt", as_of=as_of, missing_reason="initial attempt not found")
    )

    failure = repo.get_initial_failure(transaction_id)
    evidence.append(
        Evidence(source="failure_details", as_of=as_of, payload=failure)
        if failure is not None
        else Evidence(source="failure_details", as_of=as_of, missing_reason="initial failure not found")
    )

    if customer_id is None:
        evidence.append(
            Evidence(source="customer_history", as_of=as_of, missing_reason="customer unknown (transaction not found)")
        )
        evidence.append(
            Evidence(source="recovery_history", as_of=as_of, missing_reason="customer unknown (transaction not found)")
        )
    else:
        history = repo.get_customer_history(customer_id, as_of, exclude_transaction_id=transaction_id)
        evidence.append(Evidence(source="customer_history", as_of=as_of, payload=history))
        recovery = repo.get_recovery_history(customer_id, as_of, exclude_transaction_id=transaction_id)
        evidence.append(Evidence(source="recovery_history", as_of=as_of, payload=recovery))

    if prediction is not None:
        evidence.append(
            Evidence(
                source="recovery_prediction",
                as_of=as_of,
                payload=PredictionRecord(
                    probability=prediction.probability,
                    model_path=prediction.model_path,
                    prediction_time=prediction.prediction_time,
                ),
            )
        )
    else:
        evidence.append(
            Evidence(source="recovery_prediction", as_of=as_of, missing_reason="recovery prediction unavailable")
        )

    return evidence
