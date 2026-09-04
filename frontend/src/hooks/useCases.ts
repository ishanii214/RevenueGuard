import { useAsync } from "./useAsync";
import { api } from "../api/client";
import type { CaseListResponse, CaseSummary, InvestigationResponse, MetricsSummaryResponse } from "../api/types";
import { NotFoundError } from "../api/ApiError";
import { useCallback, useState } from "react";

export const PAGE_SIZE = 25;

export function useHealth() {
  return useAsync(() => api.health(), []);
}

export function useCases(page: number) {
  const offset = (page - 1) * PAGE_SIZE;
  return useAsync<CaseListResponse>(() => api.listCases(PAGE_SIZE, offset), [page]);
}

export interface CaseDetailState {
  caseData: CaseSummary | null;
  investigation: InvestigationResponse | null;
  investigationMissing: boolean;
  loading: boolean;
  error: Error | null;
  reload: () => void;
}

/** Loads the case and its persisted investigation (if any). */
export function useCaseDetail(transactionId: string): CaseDetailState {
  const state = useAsync<{ caseData: CaseSummary; investigation: InvestigationResponse | null }>(() => {
    return api.getCase(transactionId).then((caseData) =>
      api
        .getInvestigation(transactionId)
        .then((investigation) => ({ caseData, investigation }))
        .catch((error: Error) => {
          if (error instanceof NotFoundError) {
            return { caseData, investigation: null };
          }
          throw error;
        }),
    );
  }, [transactionId]);
  return {
    caseData: state.data?.caseData ?? null,
    investigation: state.data?.investigation ?? null,
    investigationMissing: state.data !== null && state.data.investigation === null,
    loading: state.loading,
    error: state.error,
    reload: state.reload,
  };
}

export interface InvestigationRunState {
  running: boolean;
  error: Error | null;
  run: (useLlm: boolean) => Promise<InvestigationResponse | null>;
}

/** The ONLY mutating user action in Phase 7: triggering an investigation. */
export function useInvestigationRunner(
  transactionId: string,
  onCompleted: () => void,
): InvestigationRunState {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const run = useCallback(
    async (useLlm: boolean) => {
      setRunning(true);
      setError(null);
      try {
        const response = await api.startInvestigation(transactionId, {
          use_llm: useLlm,
          use_database: false,
        });
        onCompleted();
        return response;
      } catch (err) {
        setError(err as Error);
        return null;
      } finally {
        setRunning(false);
      }
    },
    [transactionId, onCompleted],
  );
  return { running, error, run };
}

export function useMetrics() {
  return useAsync<MetricsSummaryResponse>(() => api.metricsSummary(), []);
}
