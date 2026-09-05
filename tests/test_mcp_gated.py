"""Phase 8 MCP server tests — gated on PostgreSQL (REVENUEGUARD_TEST_DATABASE_URL).

Covers the database-backed MCP tools end-to-end over stdio: bounded case
listing, case/investigation retrieval, DB-mode investigation, and real
metrics. Skipped honestly when the test database is not configured.
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

import backend.db as db  # noqa: E402
from agent.prediction import _get_features  # noqa: E402

DB_URL = os.environ.get("REVENUEGUARD_TEST_DATABASE_URL")
SERVER_PATH = REPO_ROOT / "mcp_server.py"


def _json(result):
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


@unittest.skipUnless(DB_URL, "REVENUEGUARD_TEST_DATABASE_URL not configured; gated MCP tests skipped")
class McpDatabaseToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _Fixture(DB_URL)
        cls.fixture.start()
        with db.connect(DB_URL) as conn:
            cls.seed_counts = db.seed_from_csv(conn, "data")
        _X, _y, meta = _get_features("data")
        cls.case_ids = meta.loc[meta["split"] == "test", "transaction_id"].tolist()[:10]

    @classmethod
    def tearDownClass(cls):
        cls.fixture.stop()

    def test_list_cases_bounded_pagination(self):
        for limit, offset in ((5, 0), (3, 5)):
            payload = _json(
                self.fixture.call_tool(
                    "list_failed_payment_cases", {"limit": limit, "offset": offset}
                )
            )
            self.assertNotIn("error", payload)
            self.assertEqual(len(payload["items"]), limit)
            self.assertGreaterEqual(payload["total"], payload["limit"] + payload["offset"])
        payload = _json(
            self.fixture.call_tool("list_failed_payment_cases", {"limit": 500, "offset": 0})
        )
        self.assertLessEqual(len(payload["items"]), 500)

    def test_get_case_allowlisted_projection(self):
        payload = _json(self.fixture.call_tool("get_case", {"transaction_id": self.case_ids[0]}))
        self.assertNotIn("error", payload)
        self.assertEqual(
            set(payload),
            {"transaction_id", "customer_id", "created_at", "amount", "currency", "payment_method", "status"},
        )

    def test_unknown_case_safe_error(self):
        payload = _json(self.fixture.call_tool("get_case", {"transaction_id": "TXN-9999999"}))
        self.assertEqual(payload["error"], "case_not_found")

    def test_db_mode_run_and_get_investigation_roundtrip(self):
        transaction_id = self.case_ids[0]
        ran = _json(
            self.fixture.call_tool("run_investigation", {"transaction_id": transaction_id})
        )
        self.assertNotIn("error", ran)
        fetched = _json(
            self.fixture.call_tool("get_investigation", {"transaction_id": transaction_id})
        )
        self.assertNotIn("error", fetched)
        self.assertEqual(fetched["result"], ran["result"])
        serialized = json.dumps(fetched)
        self.assertNotIn("recovery_outcome", serialized)
        self.assertNotIn("model_path", serialized)

    def test_metrics_match_persisted_rows(self):
        payload = _json(self.fixture.call_tool("get_operations_metrics", {}))
        with db.connect(DB_URL) as conn:
            expected_failed = conn.execute(
                "SELECT count(*) FROM transactions WHERE status = 'failed'"
            ).fetchone()[0]
            expected_investigated = conn.execute(
                "SELECT count(*) FROM investigation_results"
            ).fetchone()[0]
        self.assertEqual(payload["failed_transactions"], expected_failed)
        self.assertEqual(payload["investigated_cases"], expected_investigated)
        self.assertEqual(sum(payload["recommendations"].values()), expected_investigated)


class _Fixture:
    def __init__(self, database_url: str):
        self.loop = None
        self.session = None
        self._stack = None
        self.database_url = database_url

    def start(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start())

    async def _start(self):
        self._stack = AsyncExitStack()
        env = dict(os.environ)
        env["DATABASE_URL"] = self.database_url
        env["PYTHONIOENCODING"] = "utf-8"
        params = StdioServerParameters(
            command=sys.executable, args=[str(SERVER_PATH)], env=env
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream, read_timeout_seconds=300.0)
        )
        await self.session.initialize()

    def stop(self):
        try:
            self.loop.run_until_complete(self._stack.aclose())
        finally:
            self.loop.close()
            asyncio.set_event_loop(None)

    def call_tool(self, name, arguments=None):
        async def _run():
            return await self.session.call_tool(name, arguments or {})

        return self.loop.run_until_complete(_run())


if __name__ == "__main__":
    unittest.main()
