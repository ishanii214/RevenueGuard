import { useCallback, useState } from "react";
import { useParams } from "react-router-dom";
import { CaseListPanel, Pagination } from "../components/CaseTable";
import { InvestigationPanel } from "../components/InvestigationPanel";
import { MetricsStrip } from "../components/MetricsStrip";
import { StatusMessage } from "../components/StatusMessage";
import { PAGE_SIZE, useCaseListInvestigations, useCases } from "../hooks/useCases";

/**
 * Two-pane analyst workspace: failed-payment case list on the left, the
 * selected case's investigation report on the right. Both `/` and
 * `/cases/:transactionId` render this workspace — a deep link pre-selects the
 * case, and selecting a row updates the panel without navigation.
 */
export function CaseListPage() {
  const { transactionId } = useParams();
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [listRefreshTick, setListRefreshTick] = useState(0);
  const activeId = transactionId ?? selectedId;
  const { data, loading, error } = useCases(page);
  const investigations = useCaseListInvestigations(
    data?.items.map((item) => item.transaction_id) ?? [],
    listRefreshTick,
  );

  const onSelect = useCallback((id: string) => {
    setSelectedId(id);
  }, []);

  const onInvestigated = useCallback(() => {
    setListRefreshTick((tick) => tick + 1);
  }, []);

  return (
    <section aria-labelledby="cases-heading">
      <div className="page-heading">
        <div>
          <h2 id="cases-heading">Failed payment cases</h2>
          <p className="page-heading__hint">
            Prediction → investigation → recommendation → policy → final action. No financial
            actions are executed.
          </p>
        </div>
      </div>

      <MetricsStrip />

      <div className="workspace">
        <div className="workspace__list">
          <StatusMessage
            loading={loading}
            error={error}
            empty={data !== null && data.items.length === 0}
            emptyMessage="No failed-payment cases available."
          >
            {data && (
              <div className="card panel-card">
                <header className="panel-card__header">
                  <h3>Failed payments</h3>
                  <span className="panel-card__count">{data.total} total</span>
                </header>
                <CaseListPanel
                  cases={data.items}
                  investigations={investigations}
                  selectedId={activeId}
                  onSelect={onSelect}
                />
                <Pagination page={page} total={data.total} pageSize={PAGE_SIZE} onPageChange={setPage} />
              </div>
            )}
          </StatusMessage>
        </div>

        <div className="workspace__detail">
          {activeId ? (
            <InvestigationPanel transactionId={activeId} onInvestigated={onInvestigated} />
          ) : (
            <div className="card panel-card workspace__placeholder">
              <h3>No case selected</h3>
              <p className="text-muted">
                Select a failed payment from the list to load its prediction, point-in-time
                evidence, deterministic recommendation, and financial policy decision.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

