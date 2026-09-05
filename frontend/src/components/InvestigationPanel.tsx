import { useCallback, useState } from "react";
import { EvidenceList } from "./EvidenceList";
import { FindingsList, RiskFlags } from "./FindingsList";
import { LlmAdvisoryCard } from "./LlmAdvisoryCard";
import { PipelineStages } from "./PipelineStages";
import { PolicyDecisionCard } from "./PolicyDecisionCard";
import { ProbabilityBadge } from "./ProbabilityBadge";
import { RecommendationBadge } from "./RecommendationBadge";
import { StatusMessage } from "./StatusMessage";
import { useCaseDetail, useInvestigationRunner } from "../hooks/useCases";
import { formatDateTime, formatMoney } from "../lib/format";

interface InvestigationPanelProps {
  transactionId: string;
  /** Called after a new investigation has been persisted (list badges refresh). */
  onInvestigated?: () => void;
}

/**
 * Structured analyst report for one selected case. The decision layers are
 * always visually and verbally separated:
 *   prediction = XGBoost model estimate · recommendation = deterministic
 *   investigation result · policy decision = authoritative financial
 *   guardrail · execution authorized = whether policy permits the simulated
 *   action. The ONLY mutating action is "Run investigation"; no payment
 *   execution exists anywhere in this UI.
 */
export function InvestigationPanel({ transactionId, onInvestigated }: InvestigationPanelProps) {
  const detail = useCaseDetail(transactionId);
  const [useLlm, setUseLlm] = useState(false);
  const runner = useInvestigationRunner(transactionId, detail.reload);
  const result = detail.investigation?.result ?? null;

  const onRun = useCallback(() => {
    void runner.run(useLlm).then((ran) => {
      if (ran) {
        onInvestigated?.();
      }
    });
  }, [runner, useLlm, onInvestigated]);

  const policy = result?.policy_evaluation ?? null;

  return (
    <StatusMessage
      loading={detail.loading}
      error={detail.error}
      empty={false}
      notFoundMessage="This case does not exist in the failed-payment list."
    >
      {detail.caseData && (
        <div className="panel-stack">
          <section className="card" aria-label={`Case ${detail.caseData.transaction_id}`}>
            <span className="section-eyebrow">Selected case</span>
            <h3>{detail.caseData.transaction_id}</h3>
            <div className="case-header__facts">
              <div className="kv">
                <span className="kv__key">Amount</span>
                <span className="kv__value">
                  {formatMoney(detail.caseData.amount, detail.caseData.currency)}
                </span>
              </div>
              <div className="kv">
                <span className="kv__key">Payment method</span>
                <span className="kv__value">{detail.caseData.payment_method}</span>
              </div>
              <div className="kv">
                <span className="kv__key">Customer</span>
                <span className="kv__value">{detail.caseData.customer_id}</span>
              </div>
              <div className="kv">
                <span className="kv__key">Created</span>
                <span className="kv__value">{formatDateTime(detail.caseData.created_at)}</span>
              </div>
              <div className="kv">
                <span className="kv__key">Status</span>
                <span className="kv__value">{detail.caseData.status}</span>
              </div>
            </div>
          </section>

          <section className="card" aria-label="Run investigation">
            <span className="section-eyebrow">Investigation</span>
            {detail.investigationMissing ? (
              <p className="text-muted">
                This case has not been investigated yet. Run an investigation to produce
                evidence, a recommendation, and a policy decision.
              </p>
            ) : (
              detail.investigation && (
                <p className="text-muted">
                  Last investigated: {formatDateTime(detail.investigation.investigated_at)} (operational
                  time). Prediction point: {formatDateTime(detail.investigation.prediction_time)}.
                </p>
              )
            )}
            <div className="run-panel__controls">
              <button type="button" className="btn btn--primary" onClick={onRun} disabled={runner.running}>
                {runner.running ? "Investigating…" : "Run investigation"}
              </button>
              <label className="run-panel__llm-toggle">
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(event) => setUseLlm(event.target.checked)}
                />
                Include advisory LLM narration (non-authoritative)
              </label>
            </div>
            {runner.error && (
              <p className="status--error" role="alert">
                The investigation could not be started. Please try again.
              </p>
            )}
          </section>

          {result && (
            <>
              {result.prediction && (
                <section className="card" aria-labelledby="prediction-heading">
                  <span className="section-eyebrow">01 · Prediction — model estimate</span>
                  <h3 id="prediction-heading">Recovery probability</h3>
                  <p>
                    <ProbabilityBadge probability={result.prediction.probability} />
                  </p>
                  <p className="text-inline-note">
                    XGBoost estimate at the prediction point (
                    {formatDateTime(result.prediction.prediction_time)}). This is a model output, not a
                    decision.
                  </p>
                </section>
              )}

              <PipelineStages result={result} />

              <EvidenceList evidence={result.evidence} />
              <FindingsList findings={result.findings} />
              <RiskFlags flags={result.risk_flags} />

              <section className="card" aria-labelledby="recommendation-heading">
                <span className="section-eyebrow">02 · Deterministic recommendation</span>
                <h3 id="recommendation-heading">Investigation recommendation</h3>
                <p>
                  <RecommendationBadge action={result.recommendation.action} />
                </p>
                <p>{result.recommendation.rationale}</p>
                {result.recommendation.contributing_factors.length > 0 && (
                  <ul className="chip-list">
                    {result.recommendation.contributing_factors.map((factor) => (
                      <li key={factor} className="chip chip--muted">
                        {factor}
                      </li>
                    ))}
                  </ul>
                )}
                <p className="text-inline-note">
                  Produced by the deterministic investigation workflow — it does not authorize any
                  financial action; the policy layer decides that.
                </p>
              </section>

              {policy && (
                <>
                  <section className="card" aria-labelledby="policy-heading">
                    <span className="section-eyebrow">03 · Financial policy — authoritative guardrail</span>
                    <h3 id="policy-heading">Policy decision</h3>
                    <PolicyDecisionCard policy={policy} requestedAction={result.recommendation.action} />
                  </section>

                  <section className="card" aria-labelledby="final-action-heading">
                    <span className="section-eyebrow">04 · Final action &amp; execution</span>
                    <h3 id="final-action-heading">Outcome</h3>
                    <div className="policy-card__grid">
                      <div className="kv">
                        <span className="kv__key">Final action</span>
                        <span className={`badge badge--${policy.final_action.toLowerCase()}`}>
                          {policy.final_action}
                        </span>
                      </div>
                      <div className="kv">
                        <span className="kv__key">Execution authorized (from policy)</span>
                        <span
                          className={`kv__value ${policy.execution_authorized ? "text-authorized" : "text-muted"}`}
                        >
                          {policy.execution_authorized ? "Yes" : "No"}
                        </span>
                      </div>
                    </div>
                    <p className="text-inline-note">
                      Execution authorization means the deterministic policy permits the simulated
                      recovery action. No payment is executed by this application.
                    </p>
                  </section>
                </>
              )}

              {result.llm_review && <LlmAdvisoryCard review={result.llm_review} />}
            </>
          )}
        </div>
      )}
    </StatusMessage>
  );
}
