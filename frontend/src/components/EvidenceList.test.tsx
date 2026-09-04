import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvidenceList } from "./EvidenceList";
import type { Evidence } from "../api/types";

const failureEvidence: Evidence = {
  source: "failure_details",
  as_of: "2024-06-01T10:05:00",
  payload: {
    source: "failure_details",
    failure_id: "FAIL-0000001",
    attempt_id: "ATT-0000001",
    transaction_id: "TXN-0000001",
    customer_id: "CUST-000001",
    failed_at: "2024-06-01T10:05:00",
    failure_reason: "insufficient_funds",
    processor_response_code: "51",
  },
  missing_reason: null,
};

const missingEvidence: Evidence = {
  source: "customer_profile",
  as_of: "2024-06-01T10:05:00",
  payload: null,
  missing_reason: "customer not found",
};

const historyEvidence: Evidence = {
  source: "customer_history",
  as_of: "2024-06-01T10:05:00",
  payload: {
    source: "customer_history",
    customer_id: "CUST-000001",
    as_of: "2024-06-01T10:05:00",
    entries: [
      {
        transaction_id: "TXN-0000099",
        created_at: "2024-05-01T10:00:00",
        amount: 120,
        currency: "USD",
        payment_method: "card",
        known_outcome: "failed_pending",
      },
    ],
  },
  missing_reason: null,
};

describe("EvidenceList", () => {
  it("renders typed payload values for failure details", () => {
    render(<EvidenceList evidence={[failureEvidence]} />);
    expect(screen.getByText("Failure details")).toBeInTheDocument();
    expect(screen.getByText("insufficient_funds")).toBeInTheDocument();
    expect(screen.getByText("51")).toBeInTheDocument();
  });

  it("renders the missing-evidence path with the backend reason", () => {
    render(<EvidenceList evidence={[missingEvidence]} />);
    expect(screen.getByText(/Evidence not available: customer not found/)).toBeInTheDocument();
  });

  it("labels pending prior outcomes honestly", () => {
    render(<EvidenceList evidence={[historyEvidence]} />);
    expect(screen.getByText(/failed — not yet recovered at prediction point/)).toBeInTheDocument();
  });

  it("never renders recovery_outcome-like labels", () => {
    render(<EvidenceList evidence={[failureEvidence, historyEvidence]} />);
    expect(screen.queryByText(/recovery_outcome/i)).not.toBeInTheDocument();
  });
});
