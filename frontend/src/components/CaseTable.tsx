import { Link } from "react-router-dom";
import type { CaseSummary, InvestigationResponse } from "../api/types";
import { formatDateTime, formatMoney, formatProbability } from "../lib/format";

interface CaseListPanelProps {
  cases: CaseSummary[];
  /** Persisted investigation per transaction id; null = not investigated. */
  investigations: Record<string, InvestigationResponse | null>;
  selectedId: string | null;
  onSelect: (transactionId: string) => void;
}

/**
 * Dense analyst case list. Only fields that actually exist in the API
 * responses are rendered; probability/recommendation/policy appear only when
 * a persisted investigation exists. The transaction id stays a real link so
 * deep links and browser history keep working; clicking the row selects it
 * into the investigation panel without navigating.
 */
export function CaseListPanel({ cases, investigations, selectedId, onSelect }: CaseListPanelProps) {
  if (cases.length === 0) {
    return <p className="case-list__empty">No cases on this page.</p>;
  }
  return (
    <div className="case-list">
      {cases.map((item) => {
        const result = investigations[item.transaction_id]?.result ?? null;
        const policy = result?.policy_evaluation ?? null;
        return (
          <div
            key={item.transaction_id}
            className="case-row"
            data-selected={item.transaction_id === selectedId}
            onClick={() => onSelect(item.transaction_id)}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(item.transaction_id);
              }
            }}
          >
            <div className="case-row__top">
              <Link
                to={`/cases/${encodeURIComponent(item.transaction_id)}`}
                className="case-row__id"
                onClick={(event) => event.stopPropagation()}
              >
                {item.transaction_id}
              </Link>
              <span className="case-row__amount">{formatMoney(item.amount, item.currency)}</span>
            </div>
            <div className="case-row__meta">
              <span>{item.payment_method}</span>
              <span>{item.customer_id}</span>
              <span>{formatDateTime(item.created_at)}</span>
              <span className="case-row__status">{item.status}</span>
            </div>
            {(result?.prediction || result?.recommendation || policy) && (
              <div className="case-row__signals">
                {result?.prediction && (
                  <span className="case-row__prob">
                    {formatProbability(result.prediction.probability)}{" "}
                    <span className="case-row__prob-label">recovery est.</span>
                  </span>
                )}
                {result && (
                  <span className={`badge badge--${result.recommendation.action.toLowerCase()}`}>
                    {result.recommendation.action}
                  </span>
                )}
                {policy && (
                  <span className={`badge badge--${policy.policy_decision.toLowerCase()}`}>
                    {policy.policy_decision}
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

interface PaginationProps {
  page: number;
  total: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

/** Server-side pagination controls (bounded limit/offset requests). */
export function Pagination({ page, total, pageSize, onPageChange }: PaginationProps) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const first = (page - 1) * pageSize + 1;
  const last = Math.min(total, page * pageSize);
  return (
    <nav className="pagination" aria-label="Case list pagination">
      <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        ← Previous
      </button>
      <span className="pagination__status">
        {total === 0 ? "0 cases" : `${first}–${last} of ${total} cases`} · Page {page} of {pageCount}
      </span>
      <button type="button" disabled={page >= pageCount} onClick={() => onPageChange(page + 1)}>
        Next →
      </button>
    </nav>
  );
}

