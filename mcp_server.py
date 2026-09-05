"""RevenueGuard MCP server (Phase 8) — safe, read-only investigation interface.

Exposes a minimal set of domain-level tools to MCP-compatible AI clients.
The server is a thin interface over the EXISTING service layer
(``backend.service.RevenueGuardService``): it contains no SQL, no database
access, no ML code, no investigation logic, and no policy logic.

Safety boundary (enforced here and by the underlying architecture):
- Tools are read-only or investigation-triggering only.
- Results are investigative recommendations, NOT payment execution.
  RevenueGuard cannot execute, retry, or authorize any payment.
- ``execution_authorized`` is passed through verbatim from the deterministic
  policy layer; the MCP layer never computes or overrides it.
- ``recovery_outcome`` is never exposed (structurally absent from
  InvestigationResult); ``prediction.model_path`` is stripped from outputs.
- Tool errors are safe and generic; no stack traces, credentials, paths, or
  environment details are returned.

Run with stdio transport (the standard MCP host pattern):

    python mcp_server.py
"""

import functools
import sys
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

REPO_ROOT = Path(__file__).resolve().parent
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from agent.schemas import InvestigationResult  # noqa: E402
from backend.service import CaseNotFound, DatabaseUnavailable, RevenueGuardService  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

NO_PAYMENT_EXECUTION_NOTE = (
    "Results are investigative recommendations only - NOT payment execution. "
    "RevenueGuard cannot execute, retry, or authorize any payment."
)

TransactionId = Annotated[str, Field(min_length=1, max_length=40, description="Transaction identifier, e.g. TXN-0016861")]
PageSize = Annotated[int, Field(ge=1, le=500, description="Maximum number of cases to return (1-500)")]
PageOffset = Annotated[int, Field(ge=0, description="Number of cases to skip")]


def public_investigation_payload(response: dict) -> dict:
    """API-compatible investigation payload with internal fields stripped.

    Removes ``model_path`` (an internal model artifact path) from the
    top-level prediction and from the ``recovery_prediction`` evidence
    payload. Everything else — including the authoritative policy
    evaluation and the advisory LLM review — is passed through verbatim.
    """
    payload = dict(response)
    result = payload.get("result")
    if isinstance(result, InvestigationResult):
        dumped = result.model_dump(mode="json")
        prediction = dumped.get("prediction")
        if isinstance(prediction, dict):
            prediction.pop("model_path", None)
        for evidence in dumped.get("evidence") or []:
            evidence_payload = evidence.get("payload") if isinstance(evidence, dict) else None
            if isinstance(evidence_payload, dict) and evidence_payload.get("source") == "recovery_prediction":
                evidence_payload.pop("model_path", None)
        payload["result"] = dumped
    return payload


def _safe(tool_fn):
    """Convert service exceptions into safe, generic tool errors."""

    @functools.wraps(tool_fn)
    def wrapper(*args, **kwargs):
        try:
            return tool_fn(*args, **kwargs)
        except CaseNotFound:
            return {
                "error": "case_not_found",
                "message": "The requested case or investigation does not exist.",
            }
        except DatabaseUnavailable:
            return {
                "error": "database_unavailable",
                "message": "This tool requires the RevenueGuard database, which is not configured or unreachable.",
            }
        except Exception:
            return {
                "error": "tool_failed",
                "message": "The tool failed. No internal details are available.",
            }

    return wrapper


def create_mcp_server(service: RevenueGuardService | None = None) -> MCPServer:
    """Build the RevenueGuard MCP server around an existing service instance."""
    svc = service if service is not None else RevenueGuardService()
    server = MCPServer(
        name="RevenueGuard",
        instructions=(
            "RevenueGuard investigates failed payments: XGBoost predicts recovery "
            "probability, a deterministic investigation gathers point-in-time evidence, "
            "an optional LLM adds advisory narration, and a deterministic financial "
            "policy layer decides whether the recommended action is permitted. "
            + NO_PAYMENT_EXECUTION_NOTE
        ),
    )

    @server.tool()
    @_safe
    def list_failed_payment_cases(limit: PageSize = 25, offset: PageOffset = 0) -> dict:
        """List failed-payment cases with bounded pagination (read-only).

        Returns allowlisted case summaries: transaction_id, customer_id,
        created_at, amount, currency, payment_method, status. Use this to
        discover transaction IDs before investigating. This tool performs no
        financial action and its results are not payment execution.
        """
        return svc.list_cases(limit=limit, offset=offset)

    @server.tool()
    @_safe
    def get_case(transaction_id: TransactionId) -> dict:
        """Retrieve one failed-payment case (read-only).

        Returns the same allowlisted projection as the list tool for a single
        transaction. This does NOT run an investigation; use
        run_investigation or get_investigation for the full result. This tool
        performs no financial action and its results are not payment
        execution.
        """
        return svc.get_case(transaction_id.strip())

    @server.tool()
    @_safe
    def get_investigation(transaction_id: TransactionId) -> dict:
        """Retrieve the existing investigation result for a case (read-only).

        Returns prediction, point-in-time evidence, the deterministic
        recommendation, the authoritative policy decision, and — if enabled —
        the advisory LLM narration. Does not re-run the investigation. The
        result is an investigative recommendation, not payment execution.
        """
        return public_investigation_payload(svc.get_investigation(transaction_id.strip()))

    @server.tool()
    @_safe
    def run_investigation(
        transaction_id: TransactionId,
        use_llm: Annotated[
            bool, Field(description="Include advisory LLM narration (non-authoritative)")
        ] = False,
    ) -> dict:
        """Run the full RevenueGuard investigation pipeline for one case.

        Produces the XGBoost recovery prediction, the deterministic
        investigation with point-in-time evidence, the deterministic
        recommendation, and the authoritative financial policy decision.
        This triggers analysis only — it never executes a payment; results
        are investigative recommendations, not payment execution.
        """
        use_database = svc.database_url is not None
        return public_investigation_payload(
            svc.investigate_transaction(
                transaction_id.strip(), use_database=use_database, use_llm=use_llm
            )
        )

    @server.tool()
    @_safe
    def get_operations_metrics() -> dict:
        """Retrieve real operational aggregates (read-only).

        Counts of failed transactions, investigated cases, recommendations,
        final actions, policy decisions, and policy-authorized retries. All
        values come from persisted data; nothing is derived or estimated.
        This tool performs no financial action and its results are not
        payment execution.
        """
        return svc.metrics_summary()

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()  # stdio transport (default)
