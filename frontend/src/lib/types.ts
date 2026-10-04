/** Shapes of the static JSON written by scripts/export_dashboard_data.py. */

export interface WindowMetric {
  model: string;
  window: "35-49" | "35-42" | "43-49";
  n: number;
  illicit: number;
  prevalence: number | null;
  pr_auc: number | null;
  roc_auc: number | null;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  threshold: number | null;
}

export interface StepPerformance {
  step: number;
  prevalence: number | null;
  n_labeled: number;
  illicit: number;
  xgboost: number | null;
  graphsage: number | null;
  rgcn: number | null;
  hgt: number | null;
  /** Fewer than 10 illicit labels: per-step PR-AUC is noise. */
  low_positives: boolean;
}

export interface DriftDiagnosis {
  adversarial_auc_train_vs_drift: number;
  median_top15_ks: number;
  top15_features: string[];
  in_window_cv_pr_auc_drift: number;
  transferred_pr_auc_drift: number;
  covariate_drift_material: boolean;
  concept_drift_implicated: boolean;
  verdict: string;
}

export interface PerformanceData {
  windows: WindowMetric[];
  steps: StepPerformance[];
  drift_diagnosis: DriftDiagnosis;
  sources: string[];
}

export type TriggerAction = "NO_ACTION" | "RECALIBRATE_ONLY" | "RETRAIN";

export interface MonitoringStep {
  step: number;
  n_labeled: number;
  n_illicit: number;
  prevalence: number | null;
  rolling_prevalence: number | null;
  prevalence_rel_change_pct: number | null;
  local_drift_sig_pct: number | null;
  adversarial_auc: number | null;
  pr_auc: number | null;
  frozen_f1: number | null;
  adaptive_f1: number | null;
  bayes_f1: number | null;
  oracle_f1: number | null;
  action: TriggerAction;
  severity: string;
  primary_reason: string;
  prevalence_status: string;
  triad_status: string;
}

export interface CachedDecision {
  time_step: number;
  action: TriggerAction;
  severity: string;
  primary_reason: string;
  reasons: string[];
  component_statuses: Record<string, string>;
  label_delay_steps: number;
  label_delay_assumption: string;
  channels: ChannelStatus[];
}

export type ChannelState = "OK" | "WARNING" | "CRITICAL" | "UNAVAILABLE" | "NOT_EVALUATED" | (string & {});

export interface ChannelStatus {
  name: string;
  kind: "label_free" | "label_dependent" | string;
  status: ChannelState;
  value: number | null;
  threshold: number | null;
  can_trigger_critical: boolean;
}

export type LagSafeChannelKey = "score_shift" | "performance" | "prevalence" | "feature_shift";

/** One row of the lag-safe decision table (backtest report section 10.2). */
export interface LagSafeRow {
  step: number;
  action: TriggerAction;
  severity: string;
  label_delay_steps: number;
  perf_steps_read: number[];
  worst_f1: number | null;
  score_psi: number | null;
  channels: Record<LagSafeChannelKey, { status: ChannelState; value: number | null }>;
  critical_channels: string[];
}

export interface MonitoringData {
  metadata: { phase: string; model: string; test_window: string; generated: string };
  latest_decision: CachedDecision;
  steps: MonitoringStep[];
  lag_safe: LagSafeRow[];
  summary: {
    n_steps: number;
    action_counts: Record<string, number>;
    drift_window_mean_frozen_f1: number | null;
    drift_window_mean_adaptive_f1: number | null;
    drift_window_mean_oracle_f1: number | null;
    pre_drift_mean_frozen_f1: number | null;
  };
  sources: string[];
}

export interface SampleTx {
  tx_id: string;
  time_step: number;
  y_true: 0 | 1;
  recorded_score: number;
  title: string;
  features: Record<string, number>;
}

export interface SamplesData {
  threshold: number;
  n_features: number;
  samples: SampleTx[];
  sources: string[];
}

/** Live API responses (src/api/schemas.py). */
export interface ConfidenceContext {
  model_reliability: string;
  regime: string;
  reason: string;
  historical_pr_auc: number | null;
  historical_f1: number | null;
  recommendation: string;
}

export interface PredictionOutput {
  tx_id: string | number | null;
  time_step: number;
  probability: number;
  binary_classification: 0 | 1;
  threshold: number;
  label: string;
  confidence_context: ConfidenceContext;
}

export interface HealthResponse {
  status: string;
  model_name: string;
  model_type: string;
  n_features: number;
  operating_threshold: number;
  startup_validations_passed: boolean;
  using_fallback_metrics: boolean;
  metrics_fallback_reason: string | null;
}

export interface MonitoringStatusResponse {
  latest_monitored_step: number;
  trigger_action: TriggerAction;
  severity: string;
  primary_reason: string;
  requires_retraining: boolean;
  requires_action: boolean;
  component_statuses: Record<string, string>;
  all_reasons: string[];
  /** Added with the lag-safe monitoring pass; absent on older deployments. */
  label_delay_steps?: number | null;
  label_delay_assumption?: string | null;
  channel_breakdown?: Record<string, Omit<ChannelStatus, "name">> | null;
}

export type Strategy = "static" | "expanding";

export interface WalkforwardStep {
  step: number;
  window: string;
  n_labeled: number;
  n_illicit: number;
  prevalence: number | null;
  noise: boolean;
}

export interface WalkforwardPerStep {
  strategy: Strategy;
  lag: number;
  step: number;
  pr_auc: number | null;
  /** Validation-selected decision threshold in force at this step. */
  threshold: number | null;
  flagged: number;
  flag_rate: number | null;
}

export interface WalkforwardPooled {
  strategy: Strategy;
  lag: number;
  window: "35-42" | "43-49";
  n: number;
  n_illicit: number;
  pr_auc: number | null;
  ci_lo: number | null;
  ci_hi: number | null;
  retrains_total: number;
}

export interface WalkforwardRetrain {
  strategy: Strategy;
  lag: number;
  step: number;
  retrained: boolean;
  trained_through: number;
}

export interface TriageStep {
  strategy: Strategy;
  lag: number;
  step: number;
  k: number;
  flagged: number;
  tp: number;
  positives: number;
  precision: number | null;
  recall: number | null;
  /** min(K, positives) / positives: the best any ranker could do at this step. */
  ceiling_recall: number | null;
}

export interface TriagePooled {
  strategy: Strategy;
  lag: number;
  window: "35-42" | "43-49";
  k: number;
  alerts: number;
  tp: number;
  positives: number;
  precision: number;
  recall: number;
  max_recall: number;
}

export interface ValidationThresholdPooled {
  strategy: Strategy;
  lag: number;
  window: "35-42" | "43-49";
  n_steps: number;
  alerts_total: number;
  alerts_per_step: number | null;
  precision: number | null;
  recall: number | null;
  steps_with_zero_alerts: number;
}

export interface WalkforwardData {
  metadata: {
    strategies: Strategy[];
    lags: number[];
    ks: number[];
    windows: string[];
    noise_min_positives: number;
    ci: string;
  };
  steps: WalkforwardStep[];
  per_step: WalkforwardPerStep[];
  pooled: WalkforwardPooled[];
  retrains: WalkforwardRetrain[];
  triage: { steps: TriageStep[]; pooled: TriagePooled[]; validation_threshold: ValidationThresholdPooled[] };
  sources: string[];
}
