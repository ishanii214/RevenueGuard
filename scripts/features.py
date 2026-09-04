"""Point-in-time feature engineering for the XGBoost recovery baseline (Phase 2).

Prediction point: the timestamp of a failed transaction's initial
(attempt 1) failed attempt.

Temporal availability rule: a prior transaction of the same customer may only
contribute outcome-derived information if that outcome was actually known by
the prediction point. Using only the attempts table:

- a prior transaction is *known-failed* if its earliest failed attempt
  happened before the prediction point;
- a prior transaction is *known-recovered* if its earliest succeeded attempt
  happened before the prediction point. A payment that recovered only AFTER
  the prediction point therefore counts as failed-but-not-recovered there.

No feature reads ``recovery_outcome``, attempts with ``attempt_number >= 2``
of the current transaction, or failure rows other than the initial one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_data as gd  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

HIGH_VALUE_THRESHOLD = gd.HIGH_VALUE_THRESHOLD

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "is_high_value",
    "created_hour",
    "created_weekday",
    "created_month",
    "customer_tenure_days",
    "method_mismatch",
    "time_to_first_attempt_minutes",
    "prior_txn_count",
    "prior_failed_count",
    "prior_recovered_count",
    "prior_recovery_rate",
    "has_prior_history",
]

# Currency duplicates country information and processor_response_code is a
# 1:1 mapping of failure_reason; both are intentionally not encoded.
ONEHOT_SPEC = (
    ("payment_method", list(gd.PAYMENT_METHODS)),
    ("customer_segment", list(gd.SEGMENTS)),
    ("country", list(gd.COUNTRIES)),
    ("preferred_payment_method", list(gd.PAYMENT_METHODS)),
    ("failure_reason", list(gd.FAILURE_REASON_CODES)),
)

FEATURE_COLUMNS = NUMERIC_FEATURES + [
    f"{name}_{level}" for name, levels in ONEHOT_SPEC for level in levels
]

BY_DESIGN_NAN_COLUMNS = ("prior_recovery_rate",)

META_COLUMNS = [
    "transaction_id",
    "customer_id",
    "amount",
    "currency",
    "payment_method",
    "failure_reason",
    "recovery_outcome",
    "prediction_time",
    "split",
]

FORBIDDEN_FEATURE_COLUMNS = ("recovery_outcome",)


def load_tables(data_dir):
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers.csv", dtype=str)
    transactions = pd.read_csv(data_dir / "transactions.csv", dtype=str)
    attempts = pd.read_csv(data_dir / "payment_attempts.csv", dtype=str)
    failures = pd.read_csv(data_dir / "payment_failures.csv", dtype=str)
    for frame, columns in (
        (transactions, ("created_at",)),
        (attempts, ("attempted_at",)),
        (failures, ("failed_at",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], format=TIMESTAMP_FORMAT)
    transactions["amount"] = transactions["amount"].astype(float)
    return customers, transactions, attempts, failures


def compute_asof_history(transactions, attempts, asof_times):
    """Point-in-time history features for every transaction.

    ``asof_times`` maps transaction_id -> the instant at which this
    transaction's history is observed (for failed transactions this is the
    initial-failure timestamp). A prior transaction counts only if it was
    created before that instant, and its outcome contributes only if the
    outcome's event (first failure / first success) also happened before it.

    Returns a DataFrame indexed by transaction_id with columns
    prior_txn_count, prior_failed_count, prior_recovered_count and
    prior_recovery_rate, honouring the temporal availability rule.
    """
    first_fail = (
        attempts.loc[attempts["status"] == "failed"]
        .groupby("transaction_id")["attempted_at"]
        .min()
    )
    first_success = (
        attempts.loc[attempts["status"] == "succeeded"]
        .groupby("transaction_id")["attempted_at"]
        .min()
    )
    txn = transactions[["transaction_id", "customer_id", "created_at"]].copy()
    txn["first_fail_time"] = txn["transaction_id"].map(first_fail)
    txn["first_success_time"] = txn["transaction_id"].map(first_success)

    by_customer = {
        customer_id: frame.sort_values("created_at", kind="mergesort")
        for customer_id, frame in txn.groupby("customer_id")
    }

    rows = {}
    for _, current in txn.iterrows():
        customer_id = current["customer_id"]
        asof = asof_times[current["transaction_id"]]
        prior_count = 0
        failed_known = 0
        recovered_known = 0
        for _, prior in by_customer[customer_id].iterrows():
            if prior["transaction_id"] == current["transaction_id"]:
                continue
            if prior["created_at"] >= asof:
                continue
            prior_count += 1
            if pd.notna(prior["first_fail_time"]) and prior["first_fail_time"] < asof:
                failed_known += 1
            if pd.notna(prior["first_success_time"]) and prior["first_success_time"] < asof:
                recovered_known += 1
        rate = recovered_known / failed_known if failed_known > 0 else np.nan
        rows[current["transaction_id"]] = {
            "prior_txn_count": prior_count,
            "prior_failed_count": failed_known,
            "prior_recovered_count": recovered_known,
            "prior_recovery_rate": rate,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _add_onehot(frame):
    for name, levels in ONEHOT_SPEC:
        categories = pd.Categorical(frame[name], categories=levels)
        dummies = pd.get_dummies(categories, prefix=name, dtype=int)
        for level in levels:
            column = f"{name}_{level}"
            frame[column] = dummies[column].to_numpy() if column in dummies.columns else 0
    return frame


def build_features(data_dir):
    """Build the point-in-time design matrix, labels and metadata.

    Returns (X, y, meta) where X has FEATURE_COLUMNS in fixed order, y is the
    recovery label for failed transactions, and meta carries identifiers,
    business fields and the chronological split assignment.
    """
    customers, transactions, attempts, failures = load_tables(data_dir)

    initial = attempts.loc[attempts["attempt_number"] == "1"].set_index("transaction_id")
    initial_failure = failures.loc[
        failures["attempt_id"].isin(initial["attempt_id"])
    ].set_index("transaction_id")

    failed = transactions.loc[transactions["status"] == "failed"].copy()
    failed["prediction_time"] = initial.loc[failed["transaction_id"], "attempted_at"].values
    failed["failure_reason"] = initial_failure.loc[failed["transaction_id"], "failure_reason"].values
    failed["first_attempt_at"] = failed["prediction_time"]

    profile = customers.set_index("customer_id")
    failed["customer_segment"] = profile.loc[failed["customer_id"], "customer_segment"].values
    failed["country"] = profile.loc[failed["customer_id"], "country"].values
    failed["preferred_payment_method"] = profile.loc[
        failed["customer_id"], "preferred_payment_method"
    ].values
    failed["signup_date"] = pd.to_datetime(
        profile.loc[failed["customer_id"], "signup_date"].values, format="%Y-%m-%d"
    )

    asof_times = transactions.set_index("transaction_id")["created_at"].copy()
    asof_times.update(
        pd.Series(failed["prediction_time"].values, index=failed["transaction_id"])
    )
    history = compute_asof_history(transactions, attempts, asof_times)
    for column in ("prior_txn_count", "prior_failed_count", "prior_recovered_count", "prior_recovery_rate"):
        failed[column] = failed["transaction_id"].map(history[column])

    created = failed["created_at"]
    prediction_time = failed["prediction_time"]
    failed["log_amount"] = np.log1p(failed["amount"])
    failed["is_high_value"] = (failed["amount"] >= HIGH_VALUE_THRESHOLD).astype(int)
    failed["created_hour"] = created.dt.hour
    failed["created_weekday"] = created.dt.weekday
    failed["created_month"] = created.dt.month
    failed["customer_tenure_days"] = (
        prediction_time.dt.normalize() - failed["signup_date"]
    ).dt.days
    failed["method_mismatch"] = (
        failed["payment_method"] != failed["preferred_payment_method"]
    ).astype(int)
    failed["time_to_first_attempt_minutes"] = (
        prediction_time - created
    ).dt.total_seconds() / 60.0
    failed["has_prior_history"] = (failed["prior_txn_count"] > 0).astype(int)

    failed = failed.sort_values(["prediction_time", "transaction_id"], kind="mergesort").reset_index(drop=True)
    n_total = len(failed)
    n_train = int(n_total * TRAIN_FRACTION)
    n_validation = int(n_total * VALIDATION_FRACTION)
    failed["split"] = (
        ["train"] * n_train
        + ["validation"] * n_validation
        + ["test"] * (n_total - n_train - n_validation)
    )

    failed = _add_onehot(failed)

    y = (failed["recovery_outcome"] != "unrecovered").astype(int)
    meta = failed[META_COLUMNS].copy()
    X = failed[FEATURE_COLUMNS].copy()
    return X, y, meta


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    X, y, meta = build_features(data_dir)
    print(f"rows={len(X)} features={X.shape[1]} positives={int(y.sum())}")
