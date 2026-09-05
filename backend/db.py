"""PostgreSQL data access for the RevenueGuard backend (Phase 6).

Loading strategy (D1 — bounded, cached):
    The Phase 2 feature builder's contract requires complete-dataset frames
    (chronological splits + cross-customer point-in-time history), so DB-mode
    investigations consume whole-table frames. To keep that bounded, frames
    are loaded **once per dataset state** behind a fingerprint cache: a
    single aggregate query (row count + max timestamp per table) validates
    the cache before reuse; a mismatch (reseed) reloads. Repeated
    investigations therefore do not re-query the tables (query-count tested).

    Per-request API reads (case list/detail/results) use single-row or
    LIMIT/OFFSET SQL — never full-table loads.

All temporal semantics (as-of filters, history classification) live in the
agent package's shared code, fed by these frames — none are re-implemented
in SQL. The only wall-clock value in this module is the operational
``investigated_at`` stamp supplied by the service layer.
"""

import json
import os
import threading
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg

BACKEND_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BACKEND_DIR / "schema.sql"

STATS = {"frame_loads": 0, "fingerprint_checks": 0, "row_queries": 0}

TABLE_COLUMNS = {
    "customers": [
        "customer_id",
        "signup_date",
        "customer_segment",
        "country",
        "preferred_payment_method",
    ],
    "transactions": [
        "transaction_id",
        "customer_id",
        "created_at",
        "amount",
        "currency",
        "payment_method",
        "status",
        "recovery_outcome",
    ],
    "payment_attempts": [
        "attempt_id",
        "transaction_id",
        "attempt_number",
        "attempted_at",
        "status",
        "payment_method",
    ],
    "payment_failures": [
        "failure_id",
        "attempt_id",
        "transaction_id",
        "customer_id",
        "failed_at",
        "failure_reason",
        "processor_response_code",
    ],
}

CASE_SUMMARY_COLUMNS = (
    "transaction_id",
    "customer_id",
    "created_at",
    "amount",
    "currency",
    "payment_method",
    "status",
)


def database_url(override: str | None = None) -> str:
    url = override or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return url


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(database_url(url))


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _date_str(value):
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def _float(value):
    return float(value) if isinstance(value, Decimal) else value


def _int_str(value):
    return str(int(value)) if value is not None else value


def _load_table(conn, table: str, order_column: str, casts: dict) -> pd.DataFrame:
    columns = TABLE_COLUMNS[table]
    query = f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order_column}"
    rows = conn.execute(query).fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    for column, cast in casts.items():
        frame[column] = frame[column].map(cast)
    return frame


def load_frames(conn) -> tuple:
    """Load the four domain tables as pandas frames with dtypes identical to
    ``features.load_tables`` (str identifiers/enums, naive datetimes,
    float amounts)."""
    STATS["frame_loads"] += 1
    customers = _load_table(conn, "customers", "customer_id", {"signup_date": _date_str})
    transactions = _load_table(conn, "transactions", "transaction_id", {"amount": _float})
    attempts = _load_table(conn, "payment_attempts", "attempt_id", {"attempt_number": _int_str})
    failures = _load_table(conn, "payment_failures", "failure_id", {})
    return (customers, transactions, attempts, failures)


def dataset_fingerprint(conn) -> tuple:
    """Cheap cache-validity fingerprint: one row, eight aggregates."""
    STATS["fingerprint_checks"] += 1
    row = conn.execute(
        """
        SELECT (SELECT count(*) FROM customers),
               (SELECT max(signup_date) FROM customers),
               (SELECT count(*) FROM transactions),
               (SELECT max(created_at) FROM transactions),
               (SELECT count(*) FROM payment_attempts),
               (SELECT max(attempted_at) FROM payment_attempts),
               (SELECT count(*) FROM payment_failures),
               (SELECT max(failed_at) FROM payment_failures)
        """
    ).fetchone()
    return tuple(row)


class FrameCache:
    """Load-once, fingerprint-cached frames (D1)."""

    def __init__(self):
        self._frames: tuple | None = None
        self._fingerprint: tuple | None = None
        self._lock = threading.Lock()

    def get_frames(self, conn) -> tuple:
        fingerprint = dataset_fingerprint(conn)
        with self._lock:
            if self._frames is not None and fingerprint == self._fingerprint:
                return self._frames
            frames = load_frames(conn)
            self._frames = frames
            self._fingerprint = fingerprint
            return frames

    def invalidate(self) -> None:
        with self._lock:
            self._frames = None
            self._fingerprint = None


_FRAME_CACHE = FrameCache()


def get_frames(conn) -> tuple:
    return _FRAME_CACHE.get_frames(conn)


def reset_stats() -> None:
    for key in STATS:
        STATS[key] = 0


def fetch_case_row(conn, transaction_id: str) -> dict | None:
    """Single-row bounded read; allowlisted columns only (never recovery_outcome)."""
    STATS["row_queries"] += 1
    columns = ", ".join(CASE_SUMMARY_COLUMNS)
    row = conn.execute(
        f"SELECT {columns} FROM transactions WHERE transaction_id = %s AND status = 'failed'",
        (transaction_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(CASE_SUMMARY_COLUMNS, row))


def list_case_rows(conn, limit: int, offset: int) -> tuple[list[dict], int]:
    """Bounded page of failed-payment cases plus the total count."""
    STATS["row_queries"] += 1
    columns = ", ".join(CASE_SUMMARY_COLUMNS)
    rows = conn.execute(
        f"SELECT {columns} FROM transactions WHERE status = 'failed' "
        f"ORDER BY transaction_id LIMIT %s OFFSET %s",
        (limit, offset),
    ).fetchall()
    total = conn.execute("SELECT count(*) FROM transactions WHERE status = 'failed'").fetchone()[0]
    return [dict(zip(CASE_SUMMARY_COLUMNS, row)) for row in rows], int(total)


def save_result_snapshot(conn, snapshot: dict) -> None:
    """Upsert the current-result snapshot (D4: one row per transaction)."""
    conn.execute(
        """
        INSERT INTO investigation_results (
            transaction_id, prediction_time, investigated_at, recommendation,
            policy_decision, final_action, execution_authorized, policy_version, result
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO UPDATE SET
            prediction_time = EXCLUDED.prediction_time,
            investigated_at = EXCLUDED.investigated_at,
            recommendation = EXCLUDED.recommendation,
            policy_decision = EXCLUDED.policy_decision,
            final_action = EXCLUDED.final_action,
            execution_authorized = EXCLUDED.execution_authorized,
            policy_version = EXCLUDED.policy_version,
            result = EXCLUDED.result
        """,
        (
            snapshot["transaction_id"],
            snapshot["prediction_time"],
            snapshot["investigated_at"],
            snapshot["recommendation"],
            snapshot["policy_decision"],
            snapshot["final_action"],
            snapshot["execution_authorized"],
            snapshot["policy_version"],
            json.dumps(snapshot["result"]),
        ),
    )


def fetch_result_snapshot(conn, transaction_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT transaction_id, prediction_time, investigated_at, recommendation,
               policy_decision, final_action, execution_authorized, policy_version, result
        FROM investigation_results WHERE transaction_id = %s
        """,
        (transaction_id,),
    ).fetchone()
    if row is None:
        return None
    keys = (
        "transaction_id",
        "prediction_time",
        "investigated_at",
        "recommendation",
        "policy_decision",
        "final_action",
        "execution_authorized",
        "policy_version",
        "result",
    )
    return dict(zip(keys, row))


def metrics_summary(conn) -> dict:
    """Real, persisted aggregates only (Phase 7 decision B).

    ``failed_transactions`` follows the actual schema semantics:
    transactions.status == 'failed' (status ∈ {completed, failed}, see the
    Phase 1 dataset schema). All other counts aggregate the persisted
    investigation_results snapshots.
    """
    failed = conn.execute("SELECT count(*) FROM transactions WHERE status = 'failed'").fetchone()[0]
    investigated = conn.execute("SELECT count(*) FROM investigation_results").fetchone()[0]
    recommendations = {
        str(row[0]): int(row[1])
        for row in conn.execute("SELECT recommendation, count(*) FROM investigation_results GROUP BY recommendation")
    }
    final_actions = {
        str(row[0]): int(row[1])
        for row in conn.execute("SELECT final_action, count(*) FROM investigation_results GROUP BY final_action")
    }
    policy_decisions = {
        str(row[0]): int(row[1])
        for row in conn.execute("SELECT policy_decision, count(*) FROM investigation_results GROUP BY policy_decision")
    }
    authorized = conn.execute(
        "SELECT count(*) FROM investigation_results WHERE execution_authorized"
    ).fetchone()[0]
    return {
        "failed_transactions": int(failed),
        "investigated_cases": int(investigated),
        "recommendations": recommendations,
        "final_actions": final_actions,
        "policy_decisions": policy_decisions,
        "execution_authorized_count": int(authorized),
    }


def seed_from_csv(conn, data_dir: str = "data") -> dict:
    """Deterministically seed the database from the committed CSVs.

    Applies the schema, truncates in FK-safe order, inserts rows in ID order
    (identical to generation order), and verifies row counts against the
    source frames. Re-running produces the same database contents.
    """
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from features import load_tables

    apply_schema(conn)
    frames = load_tables(data_dir)
    tables = (
        ("payment_failures", "payment_failures", frames[3]),
        ("payment_attempts", "payment_attempts", frames[2]),
        ("transactions", "transactions", frames[1]),
        ("customers", "customers", frames[0]),
    )
    for name, _source, frame in tables:
        conn.execute(f"TRUNCATE {name} CASCADE")
    for name, _source, frame in tables:
        columns = TABLE_COLUMNS[name]
        placeholders = ", ".join(["%s"] * len(columns))
        rows = list(frame.itertuples(index=False, name=None))
        with conn.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO {name} ({', '.join(columns)}) VALUES ({placeholders})", rows
            )
    counts = {}
    for name, source, frame in tables:
        actual = conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        if actual != len(frame):
            raise RuntimeError(f"seed verification failed for {name}: {actual} != {len(frame)}")
        counts[name] = actual
    conn.commit()
    return counts
