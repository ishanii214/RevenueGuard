import { useMetrics } from "../hooks/useCases";

const ACTION_ORDER = ["RETRY", "REVIEW", "IGNORE"] as const;

/**
 * Compact KPI/summary strip for the Cases workspace. Every value comes from
 * the real `GET /metrics/summary` aggregates; when metrics are unavailable
 * the strip degrades to "—" instead of inventing numbers.
 */
export function MetricsStrip() {
  const { data } = useMetrics();
  const kpis = [
    {
      label: "Failed transactions",
      value: data?.failed_transactions ?? null,
      caption: "status = failed",
    },
    {
      label: "Investigated cases",
      value: data?.investigated_cases ?? null,
      caption: "persisted investigation snapshots",
    },
    {
      label: "Policy-authorized retries",
      value: data?.execution_authorized_count ?? null,
      caption: "execution permitted by policy",
    },
  ];
  return (
    <div className="kpi-strip" aria-label="Operations summary">
      {kpis.map((kpi) => (
        <div className="kpi-card" key={kpi.label}>
          <span className="kpi-card__label">{kpi.label}</span>
          <span className="kpi-card__value">{kpi.value ?? "—"}</span>
          <span className="kpi-card__caption">{kpi.caption}</span>
        </div>
      ))}
      <div className="kpi-card kpi-card--wide">
        <span className="kpi-card__label">Final actions (persisted)</span>
        <div className="kpi-card__chips">
          {ACTION_ORDER.map((action) => (
            <span key={action} className={`badge badge--${action.toLowerCase()}`}>
              {action} · {data?.final_actions?.[action] ?? "—"}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
