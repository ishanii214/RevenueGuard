"""Run RevenueGuard investigations from the command line (Phase 3).

Examples:
    python scripts/run_investigation.py --transaction-id TXN-0012345
    python scripts/run_investigation.py --split test --limit 5 --output investigations.jsonl

The workflow is deterministic: identical inputs produce identical results.
It only produces investigative recommendations; no financial action is
executed or authorized here.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.graph import investigate  # noqa: E402
from agent.llm import DisabledLLM  # noqa: E402
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import InvestigationInput  # noqa: E402


def _select_transaction_ids(args):
    if args.transaction_id:
        return [args.transaction_id]
    _X, _y, meta = _get_features(args.data_dir)
    frame = meta if args.split is None else meta[meta["split"] == args.split]
    return frame["transaction_id"].tolist()[: args.limit] if args.limit else frame["transaction_id"].tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run RevenueGuard investigations.")
    parser.add_argument("--transaction-id", help="investigate a single transaction")
    parser.add_argument("--split", choices=("train", "validation", "test"), help="investigate a whole split")
    parser.add_argument("--limit", type=int, default=None, help="max cases when using --split")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/baseline/model.json")
    parser.add_argument("--output", default=None, help="write JSON Lines to this path instead of stdout")
    parser.add_argument(
        "--llm",
        dest="llm_mode",
        action="store_const",
        const="auto",
        default="auto",
        help="use the LLM configured via LLM_BASE_URL/LLM_MODEL env vars (default; disabled when unset)",
    )
    parser.add_argument("--no-llm", dest="llm_mode", action="store_const", const="off", help="disable LLM narration")
    args = parser.parse_args(argv)

    transaction_ids = _select_transaction_ids(args)
    if not transaction_ids:
        parser.error("no transactions selected")

    llm_client = DisabledLLM() if args.llm_mode == "off" else None

    lines = []
    action_counts = {"RETRY": 0, "REVIEW": 0, "IGNORE": 0}
    for transaction_id in transaction_ids:
        result = investigate(
            InvestigationInput(
                transaction_id=transaction_id,
                data_dir=args.data_dir,
                model_path=args.model_path,
            ),
            llm_client=llm_client,
        )
        action_counts[result.recommendation.action] += 1
        lines.append(result.model_dump_json(indent=None))

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"wrote {len(lines)} investigation results to {args.output}")
    else:
        for line in lines:
            print(json.dumps(json.loads(line), indent=2))

    print(
        f"summary: {len(lines)} investigated | "
        f"RETRY={action_counts['RETRY']} REVIEW={action_counts['REVIEW']} IGNORE={action_counts['IGNORE']}"
    )


if __name__ == "__main__":
    main()
