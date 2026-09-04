import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./client";
import { ApiError, ApiUnavailableError, NotFoundError } from "./ApiError";

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("sends bounded limit/offset on listCases", async () => {
    const fetchMock = mockFetch(200, { items: [], total: 0, limit: 25, offset: 50 });
    vi.stubGlobal("fetch", fetchMock);
    await api.listCases(25, 50);
    expect(fetchMock).toHaveBeenCalledWith("/api/cases?limit=25&offset=50", undefined);
  });

  it("posts a JSON body on startInvestigation", async () => {
    const fetchMock = mockFetch(200, { transaction_id: "TXN-0000001" });
    vi.stubGlobal("fetch", fetchMock);
    await api.startInvestigation("TXN-0000001", { use_llm: true, use_database: false });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/cases/TXN-0000001/investigation");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ use_llm: true, use_database: false });
  });

  it("encodes transaction ids", async () => {
    const fetchMock = mockFetch(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.getCase("TXN-000001/..");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/cases/TXN-000001%2F..");
  });

  it("maps 404 to NotFoundError", async () => {
    vi.stubGlobal("fetch", mockFetch(404, { detail: "case not found" }));
    await expect(api.getCase("TXN-9999999")).rejects.toBeInstanceOf(NotFoundError);
  });

  it("maps 503 to ApiUnavailableError", async () => {
    vi.stubGlobal("fetch", mockFetch(503, { detail: "database unavailable" }));
    await expect(api.listCases(25, 0)).rejects.toBeInstanceOf(ApiUnavailableError);
  });

  it("maps network failures to ApiUnavailableError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.health()).rejects.toBeInstanceOf(ApiUnavailableError);
  });

  it("maps other errors to ApiError with status", async () => {
    vi.stubGlobal("fetch", mockFetch(500, { detail: "boom" }));
    await expect(api.metricsSummary()).rejects.toMatchObject({ status: 500 });
    await expect(api.metricsSummary()).rejects.toBeInstanceOf(ApiError);
  });

  it("parses JSON responses", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(200, {
        failed_transactions: 3,
        investigated_cases: 2,
        recommendations: { RETRY: 1, REVIEW: 1 },
        final_actions: { RETRY: 1, REVIEW: 1 },
        policy_decisions: { ALLOWED: 1, DENIED: 1 },
        execution_authorized_count: 1,
      }),
    );
    const metrics = await api.metricsSummary();
    expect(metrics.failed_transactions).toBe(3);
    expect(metrics.recommendations.RETRY).toBe(1);
  });
});
