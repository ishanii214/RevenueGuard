"""Training reproducibility and end-to-end smoke tests for Phase 2."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_baseline as tb  # noqa: E402
import generate_data as gd  # noqa: E402
import features as ft  # noqa: E402


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "data"
        gd.write_dataset(gd.generate_dataset(random.Random(42), 800), cls.data_dir)
        cls.X, cls.y, cls.meta = ft.build_features(cls.data_dir)
        train_mask = (cls.meta["split"] == "train").to_numpy()
        val_mask = (cls.meta["split"] == "validation").to_numpy()
        cls.X_train, cls.y_train = cls.X[train_mask], cls.y[train_mask]
        cls.X_val, cls.y_val = cls.X[val_mask], cls.y[val_mask]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_training_is_reproducible(self):
        first = tb.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            seed=42, n_estimators=15, early_stopping_rounds=None,
        )
        second = tb.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            seed=42, n_estimators=15, early_stopping_rounds=None,
        )
        scores_first = first.predict_proba(self.X_val)[:, 1]
        scores_second = second.predict_proba(self.X_val)[:, 1]
        self.assertTrue(np.array_equal(scores_first, scores_second))

    def test_end_to_end_smoke(self):
        out_dir = Path(self._tmp.name) / "models"
        report = tb.run_training(self.data_dir, out_dir, seed=42)
        for name in ("model.json", "metrics.json", "feature_importance.csv", "predictions_test.csv"):
            self.assertTrue((out_dir / name).exists(), name)
        self.assertEqual(report["split"]["train"] + report["split"]["validation"] + report["split"]["test"], len(self.meta))
        roc = report["model_test_metrics_tuned_threshold"]["roc_auc"]
        self.assertGreater(roc, 0.5)
        self.assertLessEqual(roc, 1.0)
        for key in (
            "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix",
        ):
            self.assertIn(key, report["model_test_metrics_tuned_threshold"])
        self.assertIn("recovered_value_coverage", report["business_evaluation"]["model_at_tuned_threshold"])


if __name__ == "__main__":
    unittest.main()
