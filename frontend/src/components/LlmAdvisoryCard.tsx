import type { LLMReview } from "../api/types";

interface LlmAdvisoryCardProps {
  review: LLMReview;
}

/**
 * Advisory LLM narration. Always visibly labeled non-authoritative, and when
 * the LLM disagrees with the deterministic recommendation the UI states that
 * the deterministic recommendation remains authoritative.
 */
export function LlmAdvisoryCard({ review }: LlmAdvisoryCardProps) {
  const narrative = review.narrative;
  const validation = review.validation as {
    grounding_ok?: boolean;
    violations?: string[];
    grounding_scope?: string;
  };
  return (
    <section className="card llm-card" aria-labelledby="llm-heading">
      <h3 id="llm-heading">
        LLM narration <span className="badge badge--advisory">ADVISORY — non-authoritative</span>
      </h3>
      {review.agrees_with_deterministic ? (
        <p className="llm-card__agreement text-muted">
          The LLM suggestion ({review.llm_recommendation}) matches the deterministic recommendation.
        </p>
      ) : (
        <div className="disagreement-banner" role="status">
          The LLM suggested <strong>{review.llm_recommendation}</strong>. This is advisory only —{" "}
          the deterministic recommendation (<strong>{review.deterministic_recommendation}</strong>)
          remains authoritative.
        </div>
      )}
      <p className="llm-card__summary">{narrative.summary}</p>
      <h4>Key findings</h4>
      <ul className="llm-card__findings">
        {narrative.key_findings.map((finding, index) => (
          <li key={index}>
            {finding.statement}
            {finding.evidence_references.length > 0 && (
              <span className="text-muted"> [{finding.evidence_references.join(", ")}]</span>
            )}
          </li>
        ))}
      </ul>
      <h4>Uncertainty</h4>
      <p>{narrative.uncertainty}</p>
      <h4>Prediction interpretation</h4>
      <p>{narrative.prediction_interpretation}</p>
      <p className="text-muted">Confidence: {(narrative.confidence * 100).toFixed(0)}%</p>
      {validation?.grounding_ok === false && validation.violations && validation.violations.length > 0 && (
        <div className="grounding-warnings">
          <h4>Grounding warnings</h4>
          <ul>
            {validation.violations.map((violation, index) => (
              <li key={index}>{violation}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
