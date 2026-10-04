export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const signedPct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}%`;

export const sci = (v: number): string => (v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(3) : v.toFixed(4));

export const ACTION_LABEL: Record<string, string> = {
  NO_ACTION: "No action",
  RECALIBRATE_ONLY: "Recalibrate",
  RETRAIN: "Retrain",
};

export const DRIFT_START = 43;

export const CHANNEL_LABEL: Record<string, string> = {
  score_shift: "Score shift",
  performance: "Performance",
  prevalence: "Prevalence",
  feature_shift: "Feature shift",
  triad: "Triad drift",
  adversarial: "Adversarial",
};

/** Status badge tone for a monitoring channel state. */
export const channelTone = (status: string): "crit" | "warn" | "ok" | "neutral" =>
  status === "CRITICAL" ? "crit" : status === "WARNING" ? "warn" : status === "OK" ? "ok" : "neutral";
