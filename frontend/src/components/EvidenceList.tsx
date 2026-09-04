import type { Evidence, EvidencePayload } from "../api/types";
import { formatDateTime, formatMoney } from "../lib/format";

function rows(payload: EvidencePayload): [string, string][] {
  switch (payload.source) {
    case "transaction_details":
      return [
        ["Transaction", payload.transaction_id],
        ["Amount", formatMoney(payload.amount, payload.currency)],
        ["Payment method", payload.payment_method],
        ["Created", formatDateTime(payload.created_at)],
        ["Customer", payload.customer_id],
      ];
    case "customer_profile":
      return [
        ["Customer", payload.customer_id],
        ["Segment", payload.customer_segment],
        ["Country", payload.country],
        ["Signup date", payload.signup_date],
        ["Preferred method", payload.preferred_payment_method],
      ];
    case "initial_attempt":
      return [
        ["Attempt", `${payload.attempt_id} (#${payload.attempt_number})`],
        ["Status", payload.status],
        ["Attempted at", formatDateTime(payload.attempted_at)],
        ["Method", payload.payment_method],
      ];
    case "failure_details":
      return [
        ["Failure", payload.failure_id],
        ["Reason", payload.failure_reason],
        ["Processor code", payload.processor_response_code],
        ["Failed at", formatDateTime(payload.failed_at)],
      ];
    case "customer_history":
      return [["Prior transactions (as of prediction point)", String(payload.entries.length)]];
    case "recovery_history":
      return [["Known recoveries (as of prediction point)", String(payload.known_recovered_count)]];
    case "recovery_prediction":
      return [
        ["XGBoost recovery probability", `${(payload.probability * 100).toFixed(1)}%`],
        ["Prediction time", formatDateTime(payload.prediction_time)],
      ];
  }
}

const SOURCE_LABELS: Record<string, string> = {
  transaction_details: "Transaction details",
  customer_profile: "Customer profile",
  initial_attempt: "Initial payment attempt",
  failure_details: "Failure details",
  customer_history: "Customer transaction history",
  recovery_history: "Recovery history",
  recovery_prediction: "Recovery prediction",
};

const OUTCOME_LABELS: Record<string, string> = {
  completed: "completed",
  recovered: "recovered",
  failed_pending: "failed — not yet recovered at prediction point",
  unknown: "outcome not yet known",
};

function HistoryEntries({ entries }: { entries: { transaction_id: string; amount: number; currency: string; known_outcome: string }[] }) {
  if (entries.length === 0) {
    return <p className="text-muted">No prior transactions.</p>;
  }
  return (
    <ul className="evidence__entries">
      {entries.map((entry) => (
        <li key={entry.transaction_id}>
          {entry.transaction_id}: {formatMoney(entry.amount, entry.currency)} — {OUTCOME_LABELS[entry.known_outcome] ?? entry.known_outcome}
        </li>
      ))}
    </ul>
  );
}

/** Readable evidence rendering, one card per evidence source. */
export function EvidenceList({ evidence }: { evidence: Evidence[] }) {
  return (
    <section className="card" aria-labelledby="evidence-heading">
      <h3 id="evidence-heading">Investigation evidence</h3>
      <p className="text-muted">
        Point-in-time evidence as of the prediction point. Nothing after that moment is visible.
      </p>
      <div className="evidence-grid">
        {evidence.map((item) => (
          <article key={item.source} className="evidence-card">
            <h4>{SOURCE_LABELS[item.source] ?? item.source}</h4>
            {item.payload === null ? (
              <p className="text-muted">Evidence not available: {item.missing_reason}</p>
            ) : item.payload.source === "customer_history" || item.payload.source === "recovery_history" ? (
              <>
                {rows(item.payload).map(([key, value]) => (
                  <div className="kv" key={key}>
                    <span className="kv__key">{key}</span>
                    <span className="kv__value">{value}</span>
                  </div>
                ))}
                <HistoryEntries entries={item.payload.entries} />
              </>
            ) : (
              rows(item.payload).map(([key, value]) => (
                <div className="kv" key={key}>
                  <span className="kv__key">{key}</span>
                  <span className="kv__value">{value}</span>
                </div>
              ))
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
