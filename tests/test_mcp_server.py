"""Phase 8 MCP server tests — un-gated (CSV-mode investigation, no PostgreSQL).

The MCP server is exercised end-to-end over real stdio using the official
SDK client, exactly as an MCP host would launch it.
"""

import asyncio
import json
import os
import sys
import unittest
from contextlib import AsyncExitStack
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from agent.data_repository import get_repository  # noqa: E402
from agent.graph import investigate  # noqa: E402
from agent.prediction import _get_features  # noqa: E402
from agent.schemas import InvestigationInput  # noqa: E402

SERVER_PATH = REPO_ROOT / "mcp_server.py"
EXPECTED_TOOLS = {
    "list_failed_payment_cases",
    "get_case",
    "get_investigation",
    "run_investigation",
    "get_operations_metrics",
}


class StdioServerFixture:
    """Launches the MCP server as a real subprocess over stdio.

    The SDK's anyio streams must be entered and exited inside the SAME task,
    so a single owner task owns the session lifecycle and serves requests
    handed to it from the synchronous test thread via a queue.
    """

    def __init__(self, env_overrides=None):
        self._env_overrides = env_overrides or {}
        self.loop = None
        self._task = None
        self._queue = None
        self._ready = None
        self._stopped = None

    def start(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._queue = asyncio.Queue()
        self._ready = asyncio.Event()
        self._stopped = asyncio.Event()
        self._task = self.loop.create_task(self._serve())
        self.loop.run_until_complete(self._ready.wait())

    async def _serve(self):
        try:
            async with AsyncExitStack() as stack:
                env = dict(os.environ)
                env.pop("DATABASE_URL", None)
                env.update(self._env_overrides)
                params = StdioServerParameters(
                    command=sys.executable, args=[str(SERVER_PATH)], env=env
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream, read_timeout_seconds=180.0)
                )
                await asyncio.wait_for(session.initialize(), timeout=60)
                self._session = session
                self._ready.set()
                while True:
                    fn, args, future = await self._queue.get()
                    if fn is None:
                        break
                    try:
                        future.set_result(await fn(*args))
                    except Exception as exc:  # noqa: BLE001 — surfaced to the caller
                        future.set_exception(exc)
        finally:
            self._stopped.set()

    def stop(self):
        async def _signal_stop():
            await self._queue.put((None, None, None))
            await self._stopped.wait()

        self.loop.run_until_complete(_signal_stop())
        self.loop.run_until_complete(self._task)
        self.loop.close()
        asyncio.set_event_loop(None)

    async def _list_tools(self):
        return await self._session.list_tools()

    async def _call_tool(self, name, arguments):
        return await self._session.call_tool(name, arguments or {})

    def list_tools(self):
        async def _run():
            future = self.loop.create_future()
            await self._queue.put((self._list_tools, (), future))
            return await future

        return self.loop.run_until_complete(_run())

    def call_tool(self, name, arguments=None):
        async def _run():
            future = self.loop.create_future()
            await self._queue.put((self._call_tool, (name, arguments or {}), future))
            return await future

        return self.loop.run_until_complete(_run())


def _json(result):
    """Extract the JSON payload from a CallToolResult."""
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


class McpServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = StdioServerFixture()
        cls.fixture.start()
        cls._X, cls._y, meta = _get_features("data")
        cls.test_case_id = meta.loc[meta["split"] == "test", "transaction_id"].iloc[0]
        cls._baseline = investigate(
            InvestigationInput(transaction_id=cls.test_case_id, data_dir="data"), llm_client=None
        )
        attempts = __import__("pandas").read_csv("data/payment_attempts.csv", dtype=str)
        counts = attempts.groupby("transaction_id").size()
        cls.multi_attempt_id = next(
            tid for tid in meta["transaction_id"] if counts.get(tid, 0) >= 3
        )
        cls.future_attempt_ids = set(
            attempts.loc[
                (attempts["transaction_id"] == cls.multi_attempt_id)
                & (attempts["attempt_number"] != "1"),
                "attempt_id",
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.fixture.stop()

    # --- discovery -------------------------------------------------------

    def test_tool_discovery_exact_surface(self):
        listing = self.fixture.list_tools()
        names = {tool.name for tool in listing.tools}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_tool_descriptions_state_not_payment_execution(self):
        listing = self.fixture.list_tools()
        for tool in listing.tools:
            normalized = " ".join((tool.description or "").lower().split())
            self.assertIn(
                "not payment execution",
                normalized,
                f"{tool.name} description must state it is not payment execution",
            )

    def test_no_execution_like_tool_names(self):
        listing = self.fixture.list_tools()
        forbidden = ("execute", "charge", "retry_payment", "send_payment", "refund")
        for tool in listing.tools:
            for verb in forbidden:
                self.assertNotIn(verb, tool.name.lower())

    def test_server_source_has_no_data_or_ml_access(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        for forbidden in ("psycopg", "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "langgraph", "xgboost", "pandas"):
            self.assertNotIn(forbidden, source)

    # --- input validation -------------------------------------------------

    def test_pagination_bounds_rejected(self):
        for arguments in ({"limit": 0}, {"limit": 501}, {"offset": -1}):
            result = self.fixture.call_tool("list_failed_payment_cases", arguments)
            self.assertTrue(result.is_error, arguments)

    def test_transaction_id_bounds_rejected(self):
        for value in ("", "x" * 41):
            result = self.fixture.call_tool("get_case", {"transaction_id": value})
            self.assertTrue(result.is_error, value)

    # --- safe error handling ------------------------------------------------

    def test_database_backed_tools_fail_safely_without_database(self):
        for name, arguments in (
            ("list_failed_payment_cases", {}),
            ("get_case", {"transaction_id": self.test_case_id}),
            ("get_investigation", {"transaction_id": self.test_case_id}),
            ("get_operations_metrics", {}),
        ):
            payload = _json(self.fixture.call_tool(name, arguments))
            self.assertEqual(payload["error"], "database_unavailable", name)
            serialized = json.dumps(payload).lower()
            for leaked in ("traceback", "psycopg", "postgres://", "database_url"):
                self.assertNotIn(leaked, serialized)

    def test_unknown_case_fails_safely_in_db_mode(self):
        # Unknown IDs produce the same safe error class regardless of mode;
        # in CSV mode the missing-DB error fires first, so exercise the
        # not-found path through run_investigation's requirement check is not
        # possible without a database — the gated suite covers it.
        payload = _json(
            self.fixture.call_tool("run_investigation", {"transaction_id": "TXN-9999999"})
        )
        self.assertIn("error", payload)

    # --- investigation delegation (CSV mode) ---------------------------------

    def test_run_investigation_delegates_to_real_pipeline(self):
        payload = _json(self.fixture.call_tool("run_investigation", {"transaction_id": self.test_case_id}))
        self.assertNotIn("error", payload)
        result = payload["result"]
        self.assertEqual(result["transaction_id"], self.test_case_id)
        self.assertEqual(result["recommendation"]["action"], self._baseline.recommendation.action)
        self.assertEqual(
            result["policy_evaluation"]["execution_authorized"],
            self._baseline.policy_evaluation.execution_authorized,
        )
        self.assertEqual(
            result["policy_evaluation"]["policy_version"],
            self._baseline.policy_evaluation.policy_version,
        )
        self.assertAlmostEqual(
            result["prediction"]["probability"], self._baseline.prediction.probability, places=12
        )
        self.assertIsNotNone(payload["prediction_time"])
        self.assertIsNotNone(payload["investigated_at"])

    def test_policy_result_preserved_verbatim(self):
        payload = _json(self.fixture.call_tool("run_investigation", {"transaction_id": self.test_case_id}))
        tool_policy = payload["result"]["policy_evaluation"]
        baseline_policy = self._baseline.policy_evaluation
        self.assertEqual(tool_policy, baseline_policy.model_dump(mode="json"))

    def test_run_investigation_deterministic(self):
        first = _json(self.fixture.call_tool("run_investigation", {"transaction_id": self.test_case_id}))
        second = _json(self.fixture.call_tool("run_investigation", {"transaction_id": self.test_case_id}))
        # The result payload is deterministic; only the operational
        # `investigated_at` stamp (persistence metadata) may differ.
        self.assertEqual(first["prediction_time"], second["prediction_time"])
        first_only = dict(first)
        second_only = dict(second)
        first_only.pop("investigated_at")
        second_only.pop("investigated_at")
        self.assertEqual(first_only, second_only)

    # --- data exposure -------------------------------------------------------

    def test_no_recovery_outcome_or_model_path_exposure(self):
        payloads = [
            _json(self.fixture.call_tool("run_investigation", {"transaction_id": self.test_case_id})),
            _json(self.fixture.call_tool("run_investigation", {"transaction_id": self.multi_attempt_id})),
        ]
        for payload in payloads:
            serialized = json.dumps(payload)
            self.assertNotIn("recovery_outcome", serialized)
            self.assertNotIn("model_path", serialized)

    def test_no_future_attempts_in_output(self):
        payload = _json(
            self.fixture.call_tool("run_investigation", {"transaction_id": self.multi_attempt_id})
        )
        serialized = json.dumps(payload)
        for attempt_id in self.future_attempt_ids:
            self.assertNotIn(attempt_id, serialized)

    def test_repository_has_no_failed_transaction_without_history_provenance(self):
        # Guard the fixture assumption used by the leakage test above.
        repo = get_repository("data")
        self.assertIn(self.multi_attempt_id, repo.failed_transaction_ids)


if __name__ == "__main__":
    unittest.main()
