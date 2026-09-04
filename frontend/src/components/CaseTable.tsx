import { Link } from "react-router-dom";
import type { CaseSummary } from "../api/types";
import { formatDateTime, formatMoney } from "../lib/format";

interface CaseTableProps {
  cases: CaseSummary[];
}

/** Failed-payment case list. Row click navigates to the case detail view. */
export function CaseTable({ cases }: CaseTableProps) {
  return (
    <table className="case-table">
      <caption className="sr-only">Failed payment cases</caption>
      <thead>
        <tr>
          <th scope="col">Transaction</th>
          <th scope="col">Created</th>
          <th scope="col">Amount</th>
          <th scope="col">Method</th>
          <th scope="col">Customer</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {cases.map((item) => (
          <tr key={item.transaction_id}>
            <td data-label="Transaction">
              <Link to={`/cases/${encodeURIComponent(item.transaction_id)}`} className="case-table__link">
                {item.transaction_id}
              </Link>
            </td>
            <td data-label="Created">{formatDateTime(item.created_at)}</td>
            <td data-label="Amount">{formatMoney(item.amount, item.currency)}</td>
            <td data-label="Method">{item.payment_method}</td>
            <td data-label="Customer">{item.customer_id}</td>
            <td data-label="Status">{item.status}</td>
          </tr>
        ))}
      </tbody>
    </table>
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
  return (
    <nav className="pagination" aria-label="Case list pagination">
      <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        ← Previous
      </button>
      <span>
        Page {page} of {pageCount} ({total} cases)
      </span>
      <button type="button" disabled={page >= pageCount} onClick={() => onPageChange(page + 1)}>
        Next →
      </button>
    </nav>
  );
}
