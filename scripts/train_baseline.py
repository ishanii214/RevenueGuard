"""Train and evaluate the XGBoost recovery baseline (Phase 2).

Trains on information available at the prediction point (see features.py for
the temporal availability rule), compares against simple baselines, and writes
reproducible artifacts. No wall-clock timestamps enter the artifacts.
"""

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn import metrics as skm

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import features as ft  # noqa: E402
import generate_data as gd  # noqa: E402

SEED_DEFAULT = 42
INTERVENTION_COST = 2.00

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
}


def train_model(X_train, y_train, X_val, y_val, seed, n_estimators=400, early_stopping_rounds=40, scale_pos_weight=None):
    params = dict(MODEL_PARAMS)
    params["random_state"] = seed
    params["n_jobs"] = 4
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _safe_roc_auc(y_true, y_score):
    try:
        return float(skm.roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def classification_metrics(y_true, y_score, y_pred):
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": float(skm.average_precision_score(y_true, y_score)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positives": int(np.asarray(y_true).sum()),
    }


def tune_threshold(y_true, y_score):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        f1 = skm.f1_score(y_true, (np.asarray(y_score) >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return best_threshold, best_f1


def business_evaluation(y_true, y_pred, amounts, cost=INTERVENTION_COST):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts, dtype=float)
    interventions = int(y_pred.sum())
    true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
    captured = float(amounts[(y_pred == 1) & (y_true == 1)].sum())
    total_recoverable = float(amounts[y_true == 1].sum())
    return {
        "interventions": interventions,
        "true_positives": true_positives,
        "captured_recovered_value": round(captured, 2),
        "total_recoverable_value": round(total_recoverable, 2),
        "recovered_value_coverage": (captured / total_recoverable) if total_recoverable > 0 else 0.0,
        "recovery_rate_among_predicted_positives": (true_positives / interventions) if interventions else 0.0,
        "net_value": round(captured - cost * interventions, 2),
    }


def _baseline_predictions(test_meta):
    majority_score = np.zeros(len(test_meta), dtype=float)
    majority_pred = np.zeros(len(test_meta), dtype=int)
    retryable = test_meta["failure_reason"].isin(gd.AUTO_RETRY_REASONS).to_numpy()
    rule_score = retryable.astype(float)
    rule_pred = retryable.astype(int)
    return (majority_score, majority_pred), (rule_score, rule_pred)


def run_training(data_dir="data", output_dir="models/baseline", seed=SEED_DEFAULT):
    X, y, meta = ft.build_features(data_dir)
    train_mask = (meta["split"] == "train").to_numpy()
    val_mask = (meta["split"] == "validation").to_numpy()
    test_mask = (meta["split"] == "test").to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_meta = meta.loc[test_mask].reset_index(drop=True)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    model_base = train_model(X_train, y_train, X_val, y_val, seed, scale_pos_weight=None)
    model_spw = train_model(X_train, y_train, X_val, y_val, seed, scale_pos_weight=n_neg / n_pos)

    val_pr_base = float(skm.average_precision_score(y_val, model_base.predict_proba(X_val)[:, 1]))
    val_pr_spw = float(skm.average_precision_score(y_val, model_spw.predict_proba(X_val)[:, 1]))
    if val_pr_spw > val_pr_base:
        model, selected_variant = model_spw, "scale_pos_weight"
    else:
        model, selected_variant = model_base, "base"

    val_score = model.predict_proba(X_val)[:, 1]
    tuned_threshold, tuned_f1_val = tune_threshold(y_val, val_score)

    test_score = model.predict_proba(X_test)[:, 1]
    test_pred_tuned = (test_score >= tuned_threshold).astype(int)
    test_pred_default = (test_score >= 0.5).astype(int)

    model_metrics_tuned = classification_metrics(y_test, test_score, test_pred_tuned)
    model_metrics_default = classification_metrics(y_test, test_score, test_pred_default)

    (maj_score, maj_pred), (rule_score, rule_pred) = _baseline_predictions(test_meta)
    majority_metrics = classification_metrics(y_test, maj_score, maj_pred)
    rule_metrics = classification_metrics(y_test, rule_score, rule_pred)

    amounts = test_meta["amount"].to_numpy(dtype=float)
    business_model = business_evaluation(y_test, test_pred_tuned, amounts)
    business_all = business_evaluation(y_test, np.ones_like(test_pred_tuned), amounts)
    business_none = business_evaluation(y_test, np.zeros_like(test_pred_tuned), amounts)

    booster = model.get_booster()
    importance = []
    for column in ft.FEATURE_COLUMNS:
        importance.append(
            {
                "feature": column,
                "gain": float(booster.get_score(importance_type="gain").get(column, 0.0)),
                "weight": int(booster.get_score(importance_type="weight").get(column, 0)),
                "cover": float(booster.get_score(importance_type="cover").get(column, 0.0)),
            }
        )
    importance.sort(key=lambda row: row["gain"], reverse=True)

    def _iso(series):
        return series.max().isoformat() if len(series) else None

    metrics_report = {
        "seed": seed,
        "config": {**MODEL_PARAMS, "n_estimators": 400, "early_stopping_rounds": 40},
        "best_iteration": int(model.best_iteration),
        "selected_variant": selected_variant,
        "variant_validation_pr_auc": {"base": val_pr_base, "scale_pos_weight": val_pr_spw},
        "scale_pos_weight_value": round(n_neg / n_pos, 4) if n_pos else None,
        "tuned_threshold": tuned_threshold,
        "tuned_threshold_validation_f1": tuned_f1_val,
        "split": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "rule": "chronological by prediction_time (70/15/15), ties by transaction_id",
            "max_train_prediction_time": _iso(meta.loc[train_mask, "prediction_time"]),
            "min_validation_prediction_time": _iso(meta.loc[val_mask, "prediction_time"]),
            "max_validation_prediction_time": _iso(meta.loc[val_mask, "prediction_time"]),
            "min_test_prediction_time": _iso(meta.loc[test_mask, "prediction_time"]),
        },
        "label_positive_rate": {
            "train": round(n_pos / max(n_pos + n_neg, 1), 4),
            "validation": round(float(y_val.mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "model_test_metrics_tuned_threshold": model_metrics_tuned,
        "model_test_metrics_default_threshold": model_metrics_default,
        "baseline_majority_test_metrics": majority_metrics,
        "baseline_retry_rule_test_metrics": rule_metrics,
        "business_evaluation": {
            "assumptions": {
                "cost_per_intervention": INTERVENTION_COST,
                "note": "Synthetic assumption for baseline comparison only, not real economics.",
            },
            "model_at_tuned_threshold": business_model,
            "intervene_on_all": business_all,
            "intervene_on_none": business_none,
        },
        "feature_importance_top15": importance[:15],
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "failed_transactions": int(len(meta)),
            "positives": int(y.sum()),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "model.json")
    with open(output_dir / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics_report, f, indent=2, sort_keys=True, default=float)
    with open(output_dir / "feature_importance.csv", "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["feature", "gain", "weight", "cover"])
        for row in importance:
            writer.writerow([row["feature"], row["gain"], row["weight"], row["cover"]])
    with open(output_dir / "predictions_test.csv", "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["transaction_id", "y_true", "y_score", "y_pred", "amount", "failure_reason"])
        for idx in range(len(test_meta)):
            writer.writerow(
                [
                    test_meta["transaction_id"].iloc[idx],
                    int(y_test.iloc[idx]),
                    f"{test_score[idx]:.6f}",
                    int(test_pred_tuned[idx]),
                    f"{amounts[idx]:.2f}",
                    test_meta["failure_reason"].iloc[idx],
                ]
            )

    print(
        f"selected_variant={selected_variant} best_iteration={model.best_iteration} "
        f"threshold={tuned_threshold:.2f}"
    )
    print(
        f"test: roc_auc={model_metrics_tuned['roc_auc']:.4f} pr_auc={model_metrics_tuned['pr_auc']:.4f} "
        f"precision={model_metrics_tuned['precision']:.4f} recall={model_metrics_tuned['recall']:.4f} "
        f"f1={model_metrics_tuned['f1']:.4f}"
    )
    print(
        f"baselines: majority roc_auc={majority_metrics['roc_auc']:.4f} f1={majority_metrics['f1']:.4f} | "
        f"retry-rule roc_auc={rule_metrics['roc_auc']:.4f} f1={rule_metrics['f1']:.4f}"
    )
    return metrics_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the RevenueGuard XGBoost recovery baseline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models/baseline")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args(argv)
    run_training(args.data_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
