import { formatProbability } from "../lib/format";

interface ProbabilityBadgeProps {
  probability: number;
}

/** XGBoost recovery probability — always labeled as an estimate. */
export function ProbabilityBadge({ probability }: ProbabilityBadgeProps) {
  return (
    <span className="probability-badge" title="XGBoost model estimate of recovery likelihood">
      <span className="probability-badge__value">{formatProbability(probability)}</span>
      <span className="probability-badge__label">recovery probability — model estimate, not a guarantee</span>
    </span>
  );
}
