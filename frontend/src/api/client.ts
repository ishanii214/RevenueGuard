/**
 * Isolated, fully typed API client for the RevenueGuard FastAPI backend.
 * The ONLY module in the frontend that talks to the network. Base URL comes
 * from VITE_API_BASE_URL (or the dev proxy "/api"); nothing environment-
 * specific is hardcoded.
 */

import { ApiError, ApiUnavailableError, NotFoundError } from "./ApiError";
import type {
  CaseListResponse,
  CaseSummary,
  HealthResponse,
  InvestigationResponse,
  MetricsSummaryResponse,
  StartInvestigationRequest,
} from "./types";

const BASE_URL: string = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    // Network-level failure (backend down, CORS block, DNS, ...)
    throw new ApiUnavailableError();
  }
  if (response.status === 404) {
    throw new NotFoundError(path);
  }
  if (response.status === 503) {
    throw new ApiUnavailableError();
  }
  if (!response.ok) {
    throw new ApiError(response.status, `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  listCases(limit: number, offset: number): Promise<CaseListResponse> {
    return request<CaseListResponse>(`/cases?limit=${limit}&offset=${offset}`);
  },

  getCase(transactionId: string): Promise<CaseSummary> {
    return request<CaseSummary>(`/cases/${encodeURIComponent(transactionId)}`);
  },

  startInvestigation(
    transactionId: string,
    body: StartInvestigationRequest,
  ): Promise<InvestigationResponse> {
    return request<InvestigationResponse>(`/cases/${encodeURIComponent(transactionId)}/investigation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  },

  getInvestigation(transactionId: string): Promise<InvestigationResponse> {
    return request<InvestigationResponse>(`/cases/${encodeURIComponent(transactionId)}/investigation`);
  },

  metricsSummary(): Promise<MetricsSummaryResponse> {
    return request<MetricsSummaryResponse>("/metrics/summary");
  },
};
