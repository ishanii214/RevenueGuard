"""PostgreSQL-backed repository and result store (Phase 6).

``PostgresCaseRepository`` is a ``CaseRepository`` built from DB-loaded
frames via ``CaseRepository.from_frames`` — the temporal as-of semantics are
the shared agent-package implementation, not SQL re-implementations.

Case projections (``case_summary``/``case_detail``) are allowlisted: the
``recovery_outcome`` column exists in the database but is structurally
absent from every API-facing case payload.
"""

from agent.data_repository import CaseRepository

CASE_SUMMARY_FIELDS = (
    "transaction_id",
    "customer_id",
    "created_at",
    "amount",
    "currency",
    "payment_method",
    "status",
)


class PostgresCaseRepository(CaseRepository):
    """CSV-compatible case repository backed by PostgreSQL-loaded frames."""


def build_postgres_repository(frames) -> PostgresCaseRepository:
    return PostgresCaseRepository.from_frames(frames)


def case_summary(row: dict) -> dict:
    """Allowlisted projection — never includes recovery_outcome."""
    return {field: row[field] for field in CASE_SUMMARY_FIELDS}
