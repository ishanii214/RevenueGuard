import { useCallback, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { EvidenceList } from "../components/EvidenceList";
import { FindingsList, RiskFlags } from "../components/FindingsList";
import { LlmAdvisoryCard } from "../components/LlmAdvisoryCard";
import { PipelineStages } from "../components/PipelineStages";
import { PolicyDecisionCard } from "../components/PolicyDecisionCard";
import { ProbabilityBadge } from "../components/ProbabilityBadge";
import { RecommendationBadge } from "../components/RecommendationBadge";
import { StatusMessage } from "../components/StatusMessage";
import { useCaseDetail, useInvestigationRunner } from "../hooks/useCases";
import { formatDateTime, formatMoney } from "../lib/format";

/**
 * Case detail: prediction → investigation/evidence → recommendation →
 * policy → final action. The ONLY mutating action is "Run investigation".
 */
export function CaseDetailPage() {
  const { transactionId = "" } = useParams();
  const detail = useCaseDetail(transactionId);
  const [useLlm, setUseLlm] = useState(false);
  const runner = useInvestigationRunner(transactionId, detail.reload);
  const reload = useCallback(() => detail.reload(), [detail]);
  const result = detail.investigation?.result ?? null;

  const onRun = useCallback(() => {
    void runner.run(useLlm).then(() => reload());
  }, [runner, useLlm, reload]);

  return (
    <section aria-labelledby="case-heading">
      <p>
        <Link to="/">← All cases</Link>
      </p>
      <StatusMessage
        loading={detail.loading}
        error={detail.error}
        empty={false}
        notFoundMessage="This case does not exist in the failed-payment list."
      >
        {detail.caseData && (
          <>
            <div className="card case-header">
              <h2 id="case-heading">
                Case {detail.caseData.transaction_id}
              </h2>
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
            </div>

            <PipelineStages result={result} />

            <div className="card run-panel">
              <h3>Investigation</h3>
              {detail.investigationMissing ? (
                <p className="text-muted">
                  This case has not been investigated yet. Run an investigation to produce
                  evidence, a recommendation, and a policy decision.
                </p>
              ) : (
                detail.investigation && (
                  <p className="text-muted">
                    Last investigated: {formatDateTime(detail.investigation.investigated_at)}{" "}
                    (operational time). Prediction point:{" "}
                    {formatDateTime(detail.investigation.prediction_time)}.
                  </p>
                )
              )}
              <div className="run-panel__controls">
                <button type="button" onClick={onRun} disabled={runner.running}>
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
            </div>

            {result && (
              <>
                <div className="card recommendation-card" aria-labelledby="recommendation-heading">
                  <h3 id="recommendation-heading">Deterministic recommendation</h3>
                  <p>
                    <RecommendationBadge action={result.recommendation.action} />{" "}
                    {result.prediction && <ProbabilityBadge probability={result.prediction.probability} />}
                  </p>
                  <p>{result.recommendation.rationale}</p>
                  <ul className="chip-list">
                    {result.recommendation.contributing_factors.map((factor) => (
                      <li key={factor} className="chip chip--muted">
                        {factor}
                      </li>
                    ))}
                  </ul>
                  <p className="text-muted">
                    Policy check required: {result.recommendation.policy_check_required ? "yes" : "no"} —
                    the policy layer decides whether any action is permitted.
                  </p>
                </div>

                {result.policy_evaluation && (
                  <PolicyDecisionCard policy={result.policy_evaluation} requestedAction={result.recommendation.action} />
                )}

                <EvidenceList evidence={result.evidence} />
                <FindingsList findings={result.findings} />
                <RiskFlags flags={result.risk_flags} />
                {result.llm_review && <LlmAdvisoryCard review={result.llm_review} />}
              </>
            )}
          </>
        )}
      </StatusMessage>
    </section>
  );
}
