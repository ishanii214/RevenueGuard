/**
 * Domain types mirroring the actual Phase 6/7 FastAPI response models
 * (backend/app.py + agent/schemas.py) field for field. No invented fields;
 * `recovery_outcome` is structurally absent because the backend never
 * exposes it. `model_path` exists in API responses but must never be
 * rendered (internal artifact path).
 */

export type RecommendationAction = "RETRY" | "REVIEW" | "IGNORE";
export type PolicyDecision = "ALLOWED" | "DENIED";

export interface CaseSummary {
  transaction_id: string;
  customer_id: string;
  created_at: string;
  amount: number;
  currency: string;
  payment_method: string;
  status: string;
}

export interface CaseListResponse {
  items: CaseSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface HealthResponse {
  status: string;
  database: boolean;
  model_artifact: boolean;
}

export interface StartInvestigationRequest {
  use_llm: boolean;
  use_database: boolean;
}

// --- Investigation result (mirrors agent.schemas.InvestigationResult) ---

export interface RecoveryPrediction {
  transaction_id: string;
  probability: number;
  model_path: string; // present in API; NEVER rendered by the UI
  prediction_time: string;
}

export interface TransactionDetailsPayload {
  source: "transaction_details";
  transaction_id: string;
  customer_id: string;
  created_at: string;
  amount: number;
  currency: string;
  payment_method: string;
}

export interface CustomerProfilePayload {
  source: "customer_profile";
  customer_id: string;
  signup_date: string;
  customer_segment: string;
  country: string;
  preferred_payment_method: string;
}

export interface AttemptRecordPayload {
  source: "initial_attempt";
  attempt_id: string;
  transaction_id: string;
  attempt_number: number;
  attempted_at: string;
  status: string;
  payment_method: string;
}

export interface FailureRecordPayload {
  source: "failure_details";
  failure_id: string;
  attempt_id: string;
  transaction_id: string;
  customer_id: string;
  failed_at: string;
  failure_reason: string;
  processor_response_code: string;
}

export type KnownOutcome = "completed" | "recovered" | "failed_pending" | "unknown";

export interface HistoryEntry {
  transaction_id: string;
  created_at: string;
  amount: number;
  currency: string;
  payment_method: string;
  known_outcome: KnownOutcome;
}

export interface CustomerHistoryPayload {
  source: "customer_history";
  customer_id: string;
  as_of: string;
  entries: HistoryEntry[];
}

export interface RecoveryHistoryPayload {
  source: "recovery_history";
  customer_id: string;
  as_of: string;
  known_recovered_count: number;
  entries: HistoryEntry[];
}

export interface PredictionRecordPayload {
  source: "recovery_prediction";
  probability: number;
  prediction_time: string;
  note: string;
}

export type EvidencePayload =
  | TransactionDetailsPayload
  | CustomerProfilePayload
  | AttemptRecordPayload
  | FailureRecordPayload
  | CustomerHistoryPayload
  | RecoveryHistoryPayload
  | PredictionRecordPayload;

export interface Evidence {
  source: string;
  as_of: string;
  payload: EvidencePayload | null;
  missing_reason: string | null;
}

export interface Finding {
  statement: string;
  based_on: string[];
}

export interface Recommendation {
  action: RecommendationAction;
  rationale: string;
  contributing_factors: string[];
  policy_check_required: true;
}

export interface NarrativeKeyFinding {
  statement: string;
  evidence_references: string[];
}

export interface InvestigationNarrative {
  summary: string;
  key_findings: NarrativeKeyFinding[];
  supporting_evidence: string[];
  uncertainty: string;
  prediction_interpretation: string;
  recommended_action: RecommendationAction;
  confidence: number;
  evidence_references: string[];
}

export interface LLMReview {
  narrative: InvestigationNarrative;
  validation: Record<string, unknown>;
  advisory_only: true;
  deterministic_recommendation: RecommendationAction;
  llm_recommendation: RecommendationAction;
  agrees_with_deterministic: boolean;
}

export interface PolicyConfigSnapshot {
  policy_version: string;
  auto_retry_enabled: boolean;
  max_total_attempts: number;
  retry_probability_floor: number;
  high_value_threshold: number;
}

export interface PolicyEvaluation {
  requested_action: RecommendationAction;
  policy_decision: PolicyDecision;
  final_action: RecommendationAction;
  reason_codes: string[];
  explanation: string;
  applicable_guardrails: string[];
  policy_version: string;
  config_snapshot: PolicyConfigSnapshot;
  execution_authorized: boolean;
  evaluated_at: string;
}

export interface InvestigationResult {
  transaction_id: string;
  prediction: RecoveryPrediction | null;
  evidence: Evidence[];
  findings: Finding[];
  risk_flags: string[];
  recommendation: Recommendation;
  errors: string[];
  llm_review: LLMReview | null;
  policy_evaluation: PolicyEvaluation | null;
}

export interface InvestigationResponse {
  transaction_id: string;
  prediction_time: string | null;
  investigated_at: string;
  result: InvestigationResult;
}

// --- Metrics (mirrors backend.app.MetricsSummaryResponse; real data only) ---

export interface MetricsSummaryResponse {
  failed_transactions: number;
  investigated_cases: number;
  recommendations: Record<string, number>;
  final_actions: Record<string, number>;
  policy_decisions: Record<string, number>;
  execution_authorized_count: number;
}
