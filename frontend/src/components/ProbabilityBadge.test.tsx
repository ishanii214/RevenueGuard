import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProbabilityBadge } from "./ProbabilityBadge";
import { RecommendationBadge } from "./RecommendationBadge";

describe("ProbabilityBadge", () => {
  it("renders the percentage and the estimate disclaimer", () => {
    render(<ProbabilityBadge probability={0.504} />);
    expect(screen.getByText("50.4%")).toBeInTheDocument();
    expect(screen.getByText(/model estimate, not a guarantee/i)).toBeInTheDocument();
  });
});

describe("RecommendationBadge", () => {
  it.each(["RETRY", "REVIEW", "IGNORE"] as const)("renders %s", (action) => {
    render(<RecommendationBadge action={action} />);
    expect(screen.getByText(action)).toBeInTheDocument();
  });
});
