import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PolicyDecisionCard } from "./PolicyDecisionCard";
import type { PolicyEvaluation, RecommendationAction } from "../api/types";

function policy(overrides: Partial<PolicyEvaluation>): PolicyEvaluation {
  return {
    requested_action: "RETRY",
    policy_decision: "ALLOWED",
    final_action: "RETRY",
    reason_codes: ["retry_allowed_within_policy"],
    explanation: "Auto-retry permitted by policy p5.v1.",
    applicable_guardrails: ["retry_cap", "probability_floor"],
    policy_version: "p5.v1",
    config_snapshot: {
      policy_version: "p5.v1",
      auto_retry_enabled: true,
      max_total_attempts: 4,
      retry_probability_floor: 0.3,
      high_value_threshold: 5000,
    },
    execution_authorized: false,
    evaluated_at: "2024-01-01T12:00:00",
    ...overrides,
  };
}

describe("PolicyDecisionCard", () => {
  it("renders an allowed RETRY with execution authorized from policy", () => {
    render(
      <PolicyDecisionCard
        policy={policy({ policy_decision: "ALLOWED", final_action: "RETRY", execution_authorized: true })}
        requestedAction="RETRY"
      />,
    );
    expect(screen.getByText("ALLOWED")).toBeInTheDocument();
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getAllByText("RETRY").length).toBe(2); // requested action + final action
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the guardrail override banner when RETRY is denied to REVIEW", () => {
    render(
      <PolicyDecisionCard
        policy={policy({
          policy_decision: "DENIED",
          final_action: "REVIEW",
          execution_authorized: false,
          reason_codes: ["high_value_auto_retry_prohibited"],
        })}
        requestedAction="RETRY"
      />,
    );
    const banner = screen.getByRole("alert");
    expect(banner).toHaveTextContent("Guardrail override");
    expect(banner).toHaveTextContent("routed to human review");
    expect(banner).toHaveTextContent("high_value_auto_retry_prohibited");
    expect(screen.getByText("DENIED")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("does not show the banner for a requested REVIEW", () => {
    render(
      <PolicyDecisionCard
        policy={policy({
          requested_action: "REVIEW",
          policy_decision: "ALLOWED",
          final_action: "REVIEW",
          reason_codes: ["review_is_safe_path"],
        })}
        requestedAction={"REVIEW" as RecommendationAction}
      />,
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("review_is_safe_path")).toBeInTheDocument();
  });

  it("renders reason codes and explanation verbatim from the backend", () => {
    render(<PolicyDecisionCard policy={policy({})} requestedAction="RETRY" />);
    expect(screen.getByText("retry_allowed_within_policy")).toBeInTheDocument();
    expect(screen.getByText("Auto-retry permitted by policy p5.v1.")).toBeInTheDocument();
  });
});
