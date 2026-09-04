"""Determinism tests for the synthetic dataset generator (Phase 1)."""

import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_data as gd  # noqa: E402

CSV_NAMES = ("customers.csv", "transactions.csv", "payment_attempts.csv", "payment_failures.csv")


class DeterminismTests(unittest.TestCase):
    def _generate(self, seed, num_customers):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        gd.write_dataset(gd.generate_dataset(random.Random(seed), num_customers), Path(tmp.name))
        return {name: (Path(tmp.name) / name).read_bytes() for name in CSV_NAMES}

    def test_same_seed_produces_identical_files(self):
        first = self._generate(42, 500)
        second = self._generate(42, 500)
        for name in CSV_NAMES:
            self.assertEqual(first[name], second[name], f"{name} must be byte-identical for the same seed")

    def test_different_seed_changes_output(self):
        first = self._generate(42, 500)
        second = self._generate(43, 500)
        differing = [name for name in CSV_NAMES if first[name] != second[name]]
        self.assertEqual(len(differing), len(CSV_NAMES), "every table must change with a new seed")


if __name__ == "__main__":
    unittest.main()
