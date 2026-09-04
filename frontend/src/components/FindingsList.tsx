import type { Finding } from "../api/types";

/** Deterministic findings from the investigation, verbatim from the API. */
export function FindingsList({ findings }: { findings: Finding[] }) {
  return (
    <section className="card" aria-labelledby="findings-heading">
      <h3 id="findings-heading">Findings</h3>
      <ul className="findings-list">
        {findings.map((finding, index) => (
          <li key={index}>
            {finding.statement}
            {finding.based_on.length > 0 && (
              <span className="text-muted"> — based on: {finding.based_on.join(", ")}</span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Deterministic risk flags, verbatim from the API. */
export function RiskFlags({ flags }: { flags: string[] }) {
  if (flags.length === 0) {
    return null;
  }
  return (
    <section className="card" aria-labelledby="risks-heading">
      <h3 id="risks-heading">Risk flags</h3>
      <ul className="chip-list">
        {flags.map((flag) => (
          <li key={flag} className="chip chip--warning">
            {flag}
          </li>
        ))}
      </ul>
    </section>
  );
}
