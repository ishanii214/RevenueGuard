import { StatusMessage } from "../components/StatusMessage";
import { useMetrics } from "../hooks/useCases";

const ACTION_ORDER = ["RETRY", "REVIEW", "IGNORE"] as const;
const DECISION_ORDER = ["ALLOWED", "DENIED"] as const;

/**
 * Metrics built ONLY from the real `GET /metrics/summary` aggregates.
 * Nothing is derived, estimated, or mocked in the frontend.
 */
export function MetricsPage() {
  const { data, loading, error } = useMetrics();
  return (
    <section aria-labelledby="metrics-heading">
      <h2 id="metrics-heading">Operations metrics</h2>
      <StatusMessage
        loading={loading}
        error={error}
        empty={false}
        notFoundMessage="Metrics are not available."
      >
        {data ? (
          <>
            <div className="metrics-grid">
              <div className="card metric-card">
                <span className="metric-card__value">{data.failed_transactions}</span>
                <span className="metric-card__label">Failed transactions (status = failed)</span>
              </div>
              <div className="card metric-card">
                <span className="metric-card__value">{data.investigated_cases}</span>
                <span className="metric-card__label">Investigated cases</span>
              </div>
              <div className="card metric-card">
                <span className="metric-card__value">{data.execution_authorized_count}</span>
                <span className="metric-card__label">Policy-authorized retries</span>
              </div>
            </div>
            <div className="metrics-grid">
              <div className="card metric-card">
                <h3>Recommendations</h3>
                <ul className="metric-card__breakdown">
                  {ACTION_ORDER.map((action) => (
                    <li key={action}>
                      {action}: {data.recommendations[action] ?? 0}
                    </li>
                  ))}
                </ul>
              </div>
              <div className="card metric-card">
                <h3>Final actions</h3>
                <ul className="metric-card__breakdown">
                  {ACTION_ORDER.map((action) => (
                    <li key={action}>
                      {action}: {data.final_actions[action] ?? 0}
                    </li>
                  ))}
                </ul>
              </div>
              <div className="card metric-card">
                <h3>Policy decisions</h3>
                <ul className="metric-card__breakdown">
                  {DECISION_ORDER.map((decision) => (
                    <li key={decision}>
                      {decision}: {data.policy_decisions[decision] ?? 0}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        ) : (
          <div className="status status--empty" role="status">
            <p>
              Metrics require the backend database. Start PostgreSQL, seed it
              (python scripts/seed_database.py), and run the API with DATABASE_URL set.
            </p>
          </div>
        )}
      </StatusMessage>
    </section>
  );
}
