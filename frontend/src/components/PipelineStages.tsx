import type { InvestigationResult } from "../api/types";
import { formatProbability } from "../lib/format";

interface PipelineStagesProps {
  result: InvestigationResult | null;
}

/**
 * The five pipeline stages, rendered verbatim from a single API result.
 * The frontend derives nothing: every value comes from the backend.
 */
export function PipelineStages({ result }: PipelineStagesProps) {
  const policy = result?.policy_evaluation ?? null;
  const stages: { label: string; value: string; tone?: string }[] = result
    ? [
        { label: "1 · Prediction (XGBoost)", value: result.prediction ? formatProbability(result.prediction.probability) : "unavailable" },
        { label: "2 · Investigation", value: `${result.evidence.length} evidence items` },
        { label: "3 · Recommendation", value: result.recommendation.action },
        { label: "4 · Policy decision", value: policy ? policy.policy_decision : "pending" },
        {
          label: "5 · Final action",
          value: policy ? policy.final_action : "pending",
          tone: policy ? (policy.execution_authorized ? "authorized" : "not-authorized") : undefined,
        },
      ]
    : [];
  return (
    <ol className="pipeline" aria-label="Investigation pipeline">
      {stages.map((stage) => (
        <li key={stage.label} className={`pipeline__stage${stage.tone ? ` pipeline__stage--${stage.tone}` : ""}`}>
          <span className="pipeline__label">{stage.label}</span>
          <span className="pipeline__value">{stage.value}</span>
        </li>
      ))}
    </ol>
  );
}
