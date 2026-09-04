"""Generate the RevenueGuard synthetic failed-payment dataset (Phase 1).

All randomness flows through a single ``random.Random`` instance seeded from
the ``--seed`` argument, rows are written in ID order, and CSVs use LF line
endings, so the output is deterministic and byte-identical across platforms.

Usage:
    python scripts/generate_data.py [--seed 42] [--num-customers 4000] [--output-dir data]

No real customer or payment data is used anywhere; every record is synthetic.
"""

import argparse
import csv
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED_DEFAULT = 42
NUM_CUSTOMERS_DEFAULT = 4000
OUTPUT_DIR_DEFAULT = "data"

DATA_START = date(2019, 1, 1)
SIGNUP_END = date(2024, 6, 30)
DATA_END = date(2025, 6, 30)

HIGH_VALUE_THRESHOLD = 5000.00

SEGMENTS = ("retail", "smb", "enterprise")
SEGMENT_WEIGHTS = (60, 30, 10)
COUNTRIES = ("US", "GB", "DE", "IN", "CA", "AU")
COUNTRY_WEIGHTS = (40, 15, 10, 20, 10, 5)
COUNTRY_CURRENCY = {
    "US": "USD",
    "GB": "GBP",
    "DE": "EUR",
    "IN": "INR",
    "CA": "CAD",
    "AU": "AUD",
}
PAYMENT_METHODS = ("card", "bank_transfer", "digital_wallet")
METHOD_WEIGHTS = (65, 10, 25)

# (mu, sigma) of the underlying normal for lognormal amounts.
AMOUNT_LOGNORMAL = {
    "retail": (3.8, 0.8),
    "smb": (6.1, 0.9),
    "enterprise": (8.1, 1.0),
}

TXN_STATUSES = ("completed", "failed")
RECOVERY_OUTCOMES = ("not_applicable", "recovered_retry", "recovered_review", "unrecovered")
ATTEMPT_STATUSES = ("succeeded", "failed")

FAILURE_REASON_CODES = {
    "insufficient_funds": "51",
    "expired_card": "54",
    "invalid_payment_method": "14",
    "temporary_network_error": "91",
    "bank_unavailable": "96",
    "payment_method_limit_exceeded": "61",
    "risk_review_hold": "57",
}
AUTO_RETRY_REASONS = frozenset(
    {"insufficient_funds", "temporary_network_error", "bank_unavailable", "payment_method_limit_exceeded"}
)
NON_RETRYABLE_REASONS = frozenset({"expired_card", "invalid_payment_method"})
RETRY_SUCCESS_BASE = {
    "insufficient_funds": 0.55,
    "temporary_network_error": 0.75,
    "bank_unavailable": 0.70,
    "payment_method_limit_exceeded": 0.45,
}
REVIEW_SUCCESS_BASE = 0.35
RETRY_DECAY = 0.75

CUSTOMERS_HEADER = [
    "customer_id",
    "signup_date",
    "customer_segment",
    "country",
    "preferred_payment_method",
]
TRANSACTIONS_HEADER = [
    "transaction_id",
    "customer_id",
    "created_at",
    "amount",
    "currency",
    "payment_method",
    "status",
    "recovery_outcome",
]
ATTEMPTS_HEADER = [
    "attempt_id",
    "transaction_id",
    "attempt_number",
    "attempted_at",
    "status",
    "payment_method",
]
FAILURES_HEADER = [
    "failure_id",
    "attempt_id",
    "transaction_id",
    "customer_id",
    "failed_at",
    "failure_reason",
    "processor_response_code",
]

TABLES = (
    ("customers.csv", CUSTOMERS_HEADER, "customers"),
    ("transactions.csv", TRANSACTIONS_HEADER, "transactions"),
    ("payment_attempts.csv", ATTEMPTS_HEADER, "payment_attempts"),
    ("payment_failures.csv", FAILURES_HEADER, "payment_failures"),
)


def _weighted(rng, population, weights):
    return rng.choices(population, weights=weights, k=1)[0]


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _latent_reliability(rng):
    """Latent per-customer payment reliability in (0, 1); never exported."""
    u = rng.random()
    if u < 0.70:
        return rng.betavariate(6.0, 2.0)
    if u < 0.90:
        return rng.betavariate(2.0, 2.0)
    return rng.betavariate(1.5, 4.0)


def generate_customers(rng, num_customers):
    rows = []
    for i in range(1, num_customers + 1):
        signup = date.fromordinal(rng.randint(DATA_START.toordinal(), SIGNUP_END.toordinal()))
        rows.append(
            {
                "customer_id": f"CUST-{i:06d}",
                "signup_date": signup.isoformat(),
                "customer_segment": _weighted(rng, SEGMENTS, SEGMENT_WEIGHTS),
                "country": _weighted(rng, COUNTRIES, COUNTRY_WEIGHTS),
                "preferred_payment_method": _weighted(rng, PAYMENT_METHODS, METHOD_WEIGHTS),
                "_signup_dt": datetime.combine(signup, datetime.min.time()),
                "_reliability": _latent_reliability(rng),
            }
        )
    return rows


def _transaction_datetimes(rng, signup_dt, count):
    dts = []
    cur = signup_dt + timedelta(days=rng.randint(1, 60), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    while len(dts) < count and cur.date() <= DATA_END:
        dts.append(cur)
        cur += timedelta(days=rng.randint(10, 120))
    return dts


def _amount(rng, segment):
    mu, sigma = AMOUNT_LOGNORMAL[segment]
    return max(1.0, rng.lognormvariate(mu, sigma))


def _failure_reason(rng, reliability, amount):
    weights = {
        "insufficient_funds": 0.30 + 0.25 * (1.0 - reliability),
        "temporary_network_error": 0.18,
        "bank_unavailable": 0.10,
        "expired_card": 0.14,
        "invalid_payment_method": 0.10,
        "payment_method_limit_exceeded": 0.08 + (0.10 if amount >= HIGH_VALUE_THRESHOLD else 0.0),
        "risk_review_hold": 0.04 + (0.10 if amount >= HIGH_VALUE_THRESHOLD else 0.0),
    }
    return _weighted(rng, list(weights), list(weights.values()))


def generate_dataset(rng, num_customers):
    customers = generate_customers(rng, num_customers)
    transactions = []
    attempts = []
    failures = []
    txn_no = 0
    att_no = 0
    fail_no = 0

    def add_attempt(txn_id, number, when, status, method):
        nonlocal att_no
        att_no += 1
        attempts.append(
            {
                "attempt_id": f"ATT-{att_no:07d}",
                "transaction_id": txn_id,
                "attempt_number": number,
                "attempted_at": _iso(when),
                "status": status,
                "payment_method": method,
            }
        )
        return attempts[-1]

    def add_failure(txn_id, customer_id, attempt_row, reason, when):
        nonlocal fail_no
        fail_no += 1
        failures.append(
            {
                "failure_id": f"FAIL-{fail_no:07d}",
                "attempt_id": attempt_row["attempt_id"],
                "transaction_id": txn_id,
                "customer_id": customer_id,
                "failed_at": _iso(when),
                "failure_reason": reason,
                "processor_response_code": FAILURE_REASON_CODES[reason],
            }
        )

    for cust in customers:
        rel = cust["_reliability"]
        for created in _transaction_datetimes(rng, cust["_signup_dt"], rng.randint(2, 7)):
            txn_no += 1
            txn_id = f"TXN-{txn_no:07d}"
            amount = round(_amount(rng, cust["customer_segment"]), 2)
            method = (
                cust["preferred_payment_method"]
                if rng.random() < 0.85
                else _weighted(rng, PAYMENT_METHODS, METHOD_WEIGHTS)
            )
            p_fail = 0.03 + 0.22 * (1.0 - rel)

            if rng.random() >= p_fail:
                when = created + timedelta(minutes=rng.randint(1, 180))
                add_attempt(txn_id, 1, when, "succeeded", method)
                transactions.append(
                    {
                        "transaction_id": txn_id,
                        "customer_id": cust["customer_id"],
                        "created_at": _iso(created),
                        "amount": f"{amount:.2f}",
                        "currency": COUNTRY_CURRENCY[cust["country"]],
                        "payment_method": method,
                        "status": "completed",
                        "recovery_outcome": "not_applicable",
                    }
                )
                continue

            reason = _failure_reason(rng, rel, amount)
            last_dt = created + timedelta(minutes=rng.randint(1, 180))
            first = add_attempt(txn_id, 1, last_dt, "failed", method)
            add_failure(txn_id, cust["customer_id"], first, reason, last_dt)
            outcome = "unrecovered"
            high_value = amount >= HIGH_VALUE_THRESHOLD

            if not high_value and reason in AUTO_RETRY_REASONS:
                base = RETRY_SUCCESS_BASE[reason]
                n = 1
                for retry_no in (2, 3):
                    n = retry_no
                    when = last_dt + timedelta(days=rng.randint(1, 3) if retry_no == 2 else rng.randint(3, 10))
                    if rng.random() < base * rel * (RETRY_DECAY ** (retry_no - 1)):
                        add_attempt(txn_id, retry_no, when, "succeeded", method)
                        outcome = "recovered_retry"
                        break
                    add_attempt(txn_id, retry_no, when, "failed", method)
                    add_failure(txn_id, cust["customer_id"], attempts[-1], reason, when)
                    last_dt = when
                if outcome == "unrecovered" and rng.random() < 0.5 * REVIEW_SUCCESS_BASE * rel:
                    when = last_dt + timedelta(days=rng.randint(5, 25))
                    add_attempt(txn_id, n + 1, when, "succeeded", method)
                    outcome = "recovered_review"
            elif not high_value and reason in NON_RETRYABLE_REASONS:
                # A retry on the same expired/invalid method cannot succeed.
                when = last_dt + timedelta(days=rng.randint(1, 3))
                add_attempt(txn_id, 2, when, "failed", method)
                add_failure(txn_id, cust["customer_id"], attempts[-1], reason, when)
                last_dt = when
                if rng.random() < REVIEW_SUCCESS_BASE * rel:
                    when = last_dt + timedelta(days=rng.randint(5, 25))
                    success_method = (
                        rng.choice([m for m in PAYMENT_METHODS if m != method]) if rng.random() < 0.8 else method
                    )
                    add_attempt(txn_id, 3, when, "succeeded", success_method)
                    outcome = "recovered_review"
            else:
                # High-value transactions and risk holds go straight to review.
                if rng.random() < REVIEW_SUCCESS_BASE * rel:
                    when = last_dt + timedelta(days=rng.randint(5, 25))
                    add_attempt(txn_id, 2, when, "succeeded", method)
                    outcome = "recovered_review"

            transactions.append(
                {
                    "transaction_id": txn_id,
                    "customer_id": cust["customer_id"],
                    "created_at": _iso(created),
                    "amount": f"{amount:.2f}",
                    "currency": COUNTRY_CURRENCY[cust["country"]],
                    "payment_method": method,
                    "status": "failed",
                    "recovery_outcome": outcome,
                }
            )

    public_customers = [{k: v for k, v in row.items() if not k.startswith("_")} for row in customers]
    return {
        "customers": public_customers,
        "transactions": transactions,
        "payment_attempts": attempts,
        "payment_failures": failures,
    }


def write_dataset(dataset, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, header, key in TABLES:
        with open(out / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            for row in dataset[key]:
                writer.writerow([row[column] for column in header])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the RevenueGuard synthetic dataset.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--num-customers", type=int, default=NUM_CUSTOMERS_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    dataset = generate_dataset(rng, args.num_customers)
    write_dataset(dataset, args.output_dir)
    for filename, _header, key in TABLES:
        print(f"{filename}: {len(dataset[key])} rows")


if __name__ == "__main__":
    main()
