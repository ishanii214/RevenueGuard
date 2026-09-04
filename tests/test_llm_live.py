"""Optional live-LLM integration tests (Phase 4).

These run ONLY when both LLM_BASE_URL and LLM_MODEL are configured in the
environment. They are excluded from the normal test suite: live model output
can be nondeterministic, so these tests assert only structural safety
invariants, never exact narrative content.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from agent.graph import investigate  # noqa: E402
from agent.llm import make_llm_from_env  # noqa: E402
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import InvestigationInput  # noqa: E402

_LIVE_CONFIGURED = bool(os.environ.get("LLM_BASE_URL")) and bool(os.environ.get("LLM_MODEL"))


@unittest.skipUnless(_LIVE_CONFIGURED, "LLM_BASE_URL/LLM_MODEL not configured; live test skipped")
class LiveLLMTests(unittest.TestCase):
    def test_live_narration_smoke(self):
        _X, _y, meta = _get_features("data")
        transaction_id = meta.loc[meta["split"] == "test", "transaction_id"].iloc[0]
        result = investigate(
            InvestigationInput(transaction_id=transaction_id, data_dir="data"),
            llm_client=make_llm_from_env(),
        )
        baseline_result = investigate(
            InvestigationInput(transaction_id=transaction_id, data_dir="data"), llm_client=None
        )
        self.assertEqual(result.recommendation.action, baseline_result.recommendation.action)
        self.assertAlmostEqual(
            result.prediction.probability, baseline_result.prediction.probability, places=12
        )
        if result.llm_review is None:
            self.skipTest(f"live LLM unavailable: {result.errors}")
        self.assertTrue(result.llm_review.advisory_only)
        self.assertIn(
            result.llm_review.llm_recommendation, ("RETRY", "REVIEW", "IGNORE")
        )
        self.assertEqual(
            result.llm_review.deterministic_recommendation, result.recommendation.action
        )


if __name__ == "__main__":
    unittest.main()
