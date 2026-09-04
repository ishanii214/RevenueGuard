import type { PolicyEvaluation, RecommendationAction } from "../api/types";
import { formatDateTime } from "../lib/format";

interface PolicyDecisionCardProps {
  policy: PolicyEvaluation;
  requestedAction: RecommendationAction;
}

/**
 * Renders the authoritative Phase 5 policy decision. `execution_authorized`
 * is displayed only as the backend policy result — the frontend never
 * derives authorization.
 */
export function PolicyDecisionCard({ policy, requestedAction }: PolicyDecisionCardProps) {
  const override =
    requestedAction === "RETRY" && policy.policy_decision === "DENIED" && policy.final_action === "REVIEW";
  return (
    <section className="card policy-card" aria-labelledby="policy-heading">
      {override && (
        <div className="guardrail-banner" role="alert">
          <strong>Guardrail override:</strong> automatic retry (RETRY) was denied by financial
          policy — the case was routed to human review (REVIEW). Reason codes:{" "}
          {policy.reason_codes.join(", ")}.
        </div>
      )}
      <h3 id="policy-heading">Financial policy decision</h3>
      <div className="policy-card__grid">
        <div className="kv">
          <span className="kv__key">Requested action</span>
          <span className="kv__value">{policy.requested_action}</span>
        </div>
        <div className="kv">
          <span className="kv__key">Policy decision</span>
          <span className={`badge ${policy.policy_decision === "ALLOWED" ? "badge--allowed" : "badge--denied"}`}>
            {policy.policy_decision}
          </span>
        </div>
        <div className="kv">
          <span className="kv__key">Final action</span>
          <span className="kv__value">{policy.final_action}</span>
        </div>
        <div className="kv">
          <span className="kv__key">Execution authorized (from policy)</span>
          <span className={`kv__value ${policy.execution_authorized ? "text-authorized" : "text-muted"}`}>
            {policy.execution_authorized ? "Yes" : "No"}
          </span>
        </div>
        <div className="kv">
          <span className="kv__key">Policy version</span>
          <span className="kv__value">{policy.policy_version}</span>
        </div>
        <div className="kv">
          <span className="kv__key">Evaluated at (prediction point)</span>
          <span className="kv__value">{formatDateTime(policy.evaluated_at)}</span>
        </div>
      </div>
      <h4>Reason codes</h4>
      <ul className="chip-list">
        {policy.reason_codes.map((code) => (
          <li key={code} className="chip">
            {code}
          </li>
        ))}
      </ul>
      <h4>Applicable guardrails</h4>
      <ul className="chip-list">
        {policy.applicable_guardrails.map((guardrail) => (
          <li key={guardrail} className="chip chip--muted">
            {guardrail}
          </li>
        ))}
      </ul>
      <p className="policy-card__explanation">{policy.explanation}</p>
    </section>
  );
}
