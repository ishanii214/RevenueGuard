"""Point-in-time data access for the investigation layer (Phase 3).

Every accessor is as-of the case prediction point (the initial failure
timestamp of the investigated transaction). Structural safeguards:

- ``get_transaction`` never exposes ``recovery_outcome`` (the label).
- ``get_initial_attempt`` / ``get_initial_failure`` read only attempt 1
  records; attempts/failures >= 2 are unreachable through this API.
- History tools classify prior transactions with the Phase 2 temporal
  availability rule: an outcome event (first failure / first success) counts
  only if it happened before ``as_of``. A prior payment that recovered after
  ``as_of`` is reported as ``failed_pending``.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from features import load_tables  # noqa: E402

from agent.schemas import (  # noqa: E402
    AttemptRecord,
    CustomerHistory,
    CustomerProfile,
    FailureRecord,
    HistoryEntry,
    RecoveryHistory,
    TransactionDetails,
)

_REPO_CACHE: dict[str, "CaseRepository"] = {}


def get_repository(data_dir: str) -> "CaseRepository":
    key = str(Path(data_dir).resolve())
    if key not in _REPO_CACHE:
        _REPO_CACHE[key] = CaseRepository(data_dir)
    return _REPO_CACHE[key]


class CaseRepository:
    def __init__(self, data_dir):
        self.customers, self.transactions, self.attempts, self.failures = load_tables(data_dir)

        self._transactions_by_id = self.transactions.set_index("transaction_id", drop=False)
        self._customers_by_id = self.customers.set_index("customer_id", drop=False)
        self._initial_attempts = (
            self.attempts.loc[self.attempts["attempt_number"] == "1"]
            .set_index("transaction_id", drop=False)
        )
        self._initial_failures = self.failures.loc[
            self.failures["attempt_id"].isin(self._initial_attempts["attempt_id"])
        ].set_index("transaction_id", drop=False)

        failed = self.attempts.loc[self.attempts["status"] == "failed"]
        succeeded = self.attempts.loc[self.attempts["status"] == "succeeded"]
        self._first_fail = failed.groupby("transaction_id")["attempted_at"].min()
        self._first_success = succeeded.groupby("transaction_id")["attempted_at"].min()
        txn = self.transactions[
            ["transaction_id", "customer_id", "created_at", "amount", "currency", "payment_method"]
        ].copy()
        txn["first_fail_time"] = txn["transaction_id"].map(self._first_fail)
        txn["first_success_time"] = txn["transaction_id"].map(self._first_success)
        self._transactions_by_customer = {
            customer_id: frame.sort_values("created_at", kind="mergesort")
            for customer_id, frame in txn.groupby("customer_id")
        }

    @property
    def failed_transaction_ids(self) -> list[str]:
        return self.transactions.loc[self.transactions["status"] == "failed", "transaction_id"].tolist()

    def get_transaction(self, transaction_id: str) -> TransactionDetails | None:
        if transaction_id not in self._transactions_by_id.index:
            return None
        row = self._transactions_by_id.loc[transaction_id]
        return TransactionDetails(
            transaction_id=row["transaction_id"],
            customer_id=row["customer_id"],
            created_at=row["created_at"],
            amount=float(row["amount"]),
            currency=row["currency"],
            payment_method=row["payment_method"],
        )

    def get_customer(self, customer_id: str) -> CustomerProfile | None:
        if customer_id not in self._customers_by_id.index:
            return None
        row = self._customers_by_id.loc[customer_id]
        return CustomerProfile(
            customer_id=row["customer_id"],
            signup_date=datetime.strptime(row["signup_date"], "%Y-%m-%d").date(),
            customer_segment=row["customer_segment"],
            country=row["country"],
            preferred_payment_method=row["preferred_payment_method"],
        )

    def get_initial_attempt(self, transaction_id: str) -> AttemptRecord | None:
        if transaction_id not in self._initial_attempts.index:
            return None
        row = self._initial_attempts.loc[transaction_id]
        return AttemptRecord(
            attempt_id=row["attempt_id"],
            transaction_id=row["transaction_id"],
            attempt_number=int(row["attempt_number"]),
            attempted_at=row["attempted_at"],
            status=row["status"],
            payment_method=row["payment_method"],
        )

    def get_initial_failure(self, transaction_id: str) -> FailureRecord | None:
        if transaction_id not in self._initial_failures.index:
            return None
        row = self._initial_failures.loc[transaction_id]
        return FailureRecord(
            failure_id=row["failure_id"],
            attempt_id=row["attempt_id"],
            transaction_id=row["transaction_id"],
            customer_id=row["customer_id"],
            failed_at=row["failed_at"],
            failure_reason=row["failure_reason"],
            processor_response_code=row["processor_response_code"],
        )

    def _classify_outcome(self, row, as_of: datetime) -> str:
        first_fail = row["first_fail_time"]
        first_success = row["first_success_time"]
        fail_known = pd.notna(first_fail) and first_fail < as_of
        success_known = pd.notna(first_success) and first_success < as_of
        if fail_known:
            return "recovered" if success_known else "failed_pending"
        if success_known:
            return "completed"
        return "unknown"

    def _history_entries(self, customer_id: str, as_of: datetime, exclude_transaction_id: str | None):
        if customer_id not in self._transactions_by_customer:
            return []
        frame = self._transactions_by_customer[customer_id]
        entries = []
        for _, row in frame.iterrows():
            if exclude_transaction_id is not None and row["transaction_id"] == exclude_transaction_id:
                continue
            if row["created_at"] >= as_of:
                continue
            entries.append(
                HistoryEntry(
                    transaction_id=row["transaction_id"],
                    created_at=row["created_at"],
                    amount=float(row["amount"]),
                    currency=row["currency"],
                    payment_method=row["payment_method"],
                    known_outcome=self._classify_outcome(row, as_of),
                )
            )
        return entries

    def get_customer_history(
        self, customer_id: str, as_of: datetime, exclude_transaction_id: str | None = None
    ) -> CustomerHistory:
        return CustomerHistory(
            customer_id=customer_id,
            as_of=as_of,
            entries=self._history_entries(customer_id, as_of, exclude_transaction_id),
        )

    def get_recovery_history(
        self, customer_id: str, as_of: datetime, exclude_transaction_id: str | None = None
    ) -> RecoveryHistory:
        entries = [
            entry
            for entry in self._history_entries(customer_id, as_of, exclude_transaction_id)
            if entry.known_outcome == "recovered"
        ]
        return RecoveryHistory(
            customer_id=customer_id,
            as_of=as_of,
            known_recovered_count=len(entries),
            entries=entries,
        )


__all__ = [
    "CaseRepository",
    "get_repository",
]
