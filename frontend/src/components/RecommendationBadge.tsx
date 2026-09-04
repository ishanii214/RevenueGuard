import type { RecommendationAction } from "../api/types";

interface RecommendationBadgeProps {
  action: RecommendationAction;
}

const LABELS: Record<RecommendationAction, string> = {
  RETRY: "RETRY",
  REVIEW: "REVIEW",
  IGNORE: "IGNORE",
};

/** Investigative recommendation badge (Phase 3 deterministic output). */
export function RecommendationBadge({ action }: RecommendationBadgeProps) {
  return <span className={`badge badge--${action.toLowerCase()}`}>{LABELS[action]}</span>;
}
