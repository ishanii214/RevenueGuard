"""Seed the RevenueGuard PostgreSQL database from the committed CSVs (Phase 6).

Deterministic and idempotent: applies the schema, truncates in FK-safe
order, inserts rows in ID order (identical to generation order), and
verifies row counts against the source frames.

Usage:
    python scripts/seed_database.py --database-url postgresql://... [--data-dir data]
    # or with DATABASE_URL set in the environment
    python scripts/seed_database.py
"""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(BACKEND_DIR), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import backend.db as db  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seed the RevenueGuard database from CSVs.")
    parser.add_argument("--database-url", default=None, help="defaults to the DATABASE_URL env var")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)

    try:
        database_url = db.database_url(args.database_url)
    except RuntimeError as exc:
        parser.error(str(exc))

    with db.connect(database_url) as conn:
        counts = db.seed_from_csv(conn, args.data_dir)
    print("seeded:", ", ".join(f"{table}={count}" for table, count in counts.items()))


if __name__ == "__main__":
    main()
