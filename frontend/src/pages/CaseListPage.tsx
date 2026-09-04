import { useState } from "react";
import { CaseTable, Pagination } from "../components/CaseTable";
import { StatusMessage } from "../components/StatusMessage";
import { PAGE_SIZE, useCases } from "../hooks/useCases";

/** Failed-payment case list with bounded server-side pagination. */
export function CaseListPage() {
  const [page, setPage] = useState(1);
  const { data, loading, error } = useCases(page);
  return (
    <section aria-labelledby="cases-heading">
      <h2 id="cases-heading">Failed payment cases</h2>
      <StatusMessage
        loading={loading}
        error={error}
        empty={data !== null && data.items.length === 0}
        emptyMessage="No failed-payment cases available."
      >
        {data && (
          <>
            <CaseTable cases={data.items} />
            <Pagination page={page} total={data.total} pageSize={PAGE_SIZE} onPageChange={setPage} />
          </>
        )}
      </StatusMessage>
    </section>
  );
}
