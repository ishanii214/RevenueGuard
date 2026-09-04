import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { CaseListPage } from "./CaseListPage";
import type { CaseListResponse } from "../api/types";

function listResponse(overrides: Partial<CaseListResponse> = {}): CaseListResponse {
  return {
    items: [
      {
        transaction_id: "TXN-0000001",
        customer_id: "CUST-000001",
        created_at: "2024-06-01T10:00:00",
        amount: 36.09,
        currency: "GBP",
        payment_method: "digital_wallet",
        status: "failed",
      },
    ],
    total: 1,
    limit: 25,
    offset: 0,
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <CaseListPage />
    </MemoryRouter>,
  );
}

describe("CaseListPage", () => {
  it("renders cases in a table with a link to the detail view", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify(listResponse()), { status: 200, headers: { "Content-Type": "application/json" } }),
    ));
    renderPage();
    expect(await screen.findByText("TXN-0000001")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "TXN-0000001" })).toHaveAttribute("href", "/cases/TXN-0000001");
  });

  it("shows the empty state when there are no cases", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify(listResponse({ items: [], total: 0 })), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));
    renderPage();
    expect(await screen.findByText("No failed-payment cases available.")).toBeInTheDocument();
  });

  it("shows the backend-unavailable error state on 503", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "database unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    ));
    renderPage();
    expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  });

  it("sends bounded pagination parameters for page 2", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(listResponse({ total: 30 })), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    renderPage();
    await screen.findByText("TXN-0000001");
    expect(fetchMock).toHaveBeenCalledWith("/api/cases?limit=25&offset=0", undefined);
  });
});
