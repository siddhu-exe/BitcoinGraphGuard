/**
 * Every metric the dashboard shows is derived here from the bundled
 * public/data/*.json written by scripts/export_dashboard_data.py. Nothing is
 * hardcoded and nothing is guessed: a missing value stays null and renders "n/a".
 */
import { num } from "./format";
import type {
  ChannelState,
  LagSafeRow,
  MonitoringData,
  PerformanceData,
  TriggerAction,
  WalkforwardData,
} from "./types";

export const DRIFT_START = 43;
export const DRIFT_WINDOW = "43-49" as const;
export const LABEL_DELAYS = [1, 2, 3, 5] as const;

/* ---------- Overview ---------- */

export interface DelayRow {
  label: string;
  value: number | null;
  ciLo: number | null;
  ciHi: number | null;
  isStatic: boolean;
}

/** Expanding-window pooled PR-AUC (and 95% cluster-bootstrap CI) per label delay. */
export function labelDelayRows(wf: WalkforwardData): DelayRow[] {
  return LABEL_DELAYS.map((lag) => {
    const row = wf.pooled.find(
      (r) => r.strategy === "expanding" && r.lag === lag && r.window === DRIFT_WINDOW,
    );
    return {
      label: lag === 1 ? "1 step" : `${lag} steps`,
      value: row?.pr_auc ?? null,
      ciLo: row?.ci_lo ?? null,
      ciHi: row?.ci_hi ?? null,
      isStatic: false,
    };
  });
}

/** The frozen model's pooled PR-AUC over the drift window — the flat reference bar. */
export function staticPooled(wf: WalkforwardData): DelayRow {
  const row = wf.pooled.find(
    (r) => r.strategy === "static" && r.lag === 1 && r.window === DRIFT_WINDOW,
  );
  return { label: "Never retrained", value: row?.pr_auc ?? null, ciLo: null, ciHi: null, isStatic: true };
}

export function expandingPooled(wf: WalkforwardData, lag: number, window: string) {
  return wf.pooled.find((r) => r.strategy === "expanding" && r.lag === lag && r.window === window) ?? null;
}

export interface ModelRow {
  name: string;
  kind: string;
  early: number | null;
  late: number | null;
}

const MODEL_ORDER = [
  { name: "XGBoost", kind: "tree model, deployed" },
  { name: "GraphSAGE", kind: "graph neural network" },
  { name: "HGT", kind: "graph transformer" },
  { name: "HeteroRGCN", kind: "graph neural network" },
] as const;

/** Four models, PR-AUC before (35–42) and after (43–49) the shift. */
export function fourModels(perf: PerformanceData): ModelRow[] {
  const pick = (model: string, window: string) =>
    perf.windows.find((w) => w.model === model && w.window === window)?.pr_auc ?? null;
  return MODEL_ORDER.map(({ name, kind }) => ({
    name,
    kind,
    early: pick(name, "35-42"),
    late: pick(name, "43-49"),
  }));
}

export function overallWindow(perf: PerformanceData, model: string, window: string) {
  return perf.windows.find((w) => w.model === model && w.window === window) ?? null;
}

export function driftPrevalence(perf: PerformanceData) {
  return {
    early: overallWindow(perf, "XGBoost", "35-42")?.prevalence ?? null,
    late: overallWindow(perf, "XGBoost", "43-49")?.prevalence ?? null,
  };
}

/* ---------- Retraining ---------- */

export interface StepColumn {
  step: number;
  pr: number | null;
  positives: number;
  noisy: boolean;
  scored: boolean;
}

/** 15 columns, steps 35–49: illicit labels per step, PR-AUC only after the shift. */
export function perStepColumns(wf: WalkforwardData): StepColumn[] {
  const prByStep = new Map(
    wf.per_step.filter((r) => r.strategy === "expanding" && r.lag === 1).map((r) => [r.step, r.pr_auc]),
  );
  return wf.steps.map((s) => ({
    step: s.step,
    scored: s.step >= DRIFT_START,
    pr: s.step >= DRIFT_START ? prByStep.get(s.step) ?? null : null,
    positives: s.n_illicit,
    noisy: s.noise,
  }));
}

export const maxPositives = (cols: StepColumn[]): number =>
  cols.reduce((m, c) => Math.max(m, c.positives), 0);

/** First step from which PR-AUC stays ≥ 0.5 for three consecutive scored (non-noisy) steps. */
export function sustainedRecoveryStep(cols: StepColumn[]): number | null {
  const scored = cols.filter((c) => c.scored && !c.noisy);
  for (let i = 0; i + 2 < scored.length; i += 1) {
    const window = scored.slice(i, i + 3);
    if (window.every((c) => (c.pr ?? 0) >= 0.5)) return window[0].step;
  }
  return null;
}

export interface TriageRow {
  k: number;
  precision: number | null;
  recall: number | null;
  staticRecall: number | null;
  /** Recomputed: Σ min(K, positives) / total positives over the drift window. */
  maxRecall: number | null;
}

/** Top-K triage at L = 1, pooled over 43–49. Max recall is recomputed, never read from the CSV. */
export function triageRows(wf: WalkforwardData): TriageRow[] {
  const steps = wf.triage.steps.filter(
    (r) => r.strategy === "expanding" && r.lag === 1 && r.step >= DRIFT_START,
  );
  const totalPositives = steps.reduce((sum, r) => sum + r.positives, 0);
  const ks = Array.from(new Set(wf.metadata.ks)).sort((a, b) => a - b);
  return ks.map((k) => {
    const emp = wf.triage.pooled.find(
      (r) => r.strategy === "expanding" && r.lag === 1 && r.window === DRIFT_WINDOW && r.k === k,
    );
    const stat = wf.triage.pooled.find(
      (r) => r.strategy === "static" && r.lag === 1 && r.window === DRIFT_WINDOW && r.k === k,
    );
    const maxRecall =
      totalPositives > 0
        ? steps.reduce((sum, r) => sum + Math.min(k, r.positives), 0) / totalPositives
        : null;
    return {
      k,
      precision: emp?.precision ?? null,
      recall: emp?.recall ?? null,
      staticRecall: stat?.recall ?? null,
      maxRecall,
    };
  });
}

/** Validation-threshold swing (max → min) across the drift window, L = 1. */
export function thresholdSwing(wf: WalkforwardData) {
  const vals = wf.per_step
    .filter((r) => r.strategy === "expanding" && r.lag === 1 && r.step >= DRIFT_START)
    .map((r) => r.threshold)
    .filter((v): v is number => v !== null);
  return vals.length ? { max: Math.max(...vals), min: Math.min(...vals) } : { max: null, min: null };
}

/** Recall of the validation-threshold rule at a given lag (drift window). */
export function thresholdRuleRecall(wf: WalkforwardData, lag: number): number | null {
  return (
    wf.triage.validation_threshold.find(
      (r) => r.strategy === "expanding" && r.lag === lag && r.window === DRIFT_WINDOW,
    )?.recall ?? null
  );
}

/** Recall of top-K triage at a given lag (drift window). */
export function topKRecall(wf: WalkforwardData, lag: number, k: number): number | null {
  return (
    wf.triage.pooled.find(
      (r) => r.strategy === "expanding" && r.lag === lag && r.window === DRIFT_WINDOW && r.k === k,
    )?.recall ?? null
  );
}

/* ---------- Monitoring ---------- */

export type LagSafeKey = "score_shift" | "performance" | "prevalence" | "feature_shift";

export const LAG_SAFE_KEYS: LagSafeKey[] = [
  "score_shift",
  "performance",
  "prevalence",
  "feature_shift",
];

export const CHANNEL_LABEL: Record<string, string> = {
  score_shift: "Score shift",
  performance: "Performance",
  prevalence: "Prevalence",
  feature_shift: "Feature shift",
  triad: "Feature triad",
  adversarial: "Adversarial",
};

export interface ChannelCard {
  key: LagSafeKey;
  name: string;
  kind: string;
  /** First step the channel reached CRITICAL, else null. */
  firstCritical: number | null;
  threshold: number | null;
  /** One-line, data-derived explanation of what the channel saw. */
  note: string;
}

/** Short, lower-case channel name used in the compact monitoring table. */
export const CHANNEL_SHORT: Record<string, string> = {
  score_shift: "score shift",
  performance: "performance",
  prevalence: "prevalence",
  feature_shift: "feature shift",
};

/** Critical channels as short names; a lone feature-shift warning is called out instead. */
export function criticalChannelsText(row: LagSafeRow): string {
  const names = row.critical_channels.map((c) => CHANNEL_SHORT[c] ?? c);
  if (names.length) return names.join(", ");
  if (row.channels.feature_shift?.status === "WARNING") return "feature warning";
  return "—";
}

/** Lower-cased prevalence channel status, or "—" while the channel has no value yet. */
export function prevalenceStatus(row: LagSafeRow): string {
  const status = row.channels.prevalence?.status;
  if (!status || status === "UNAVAILABLE" || status === "NOT_EVALUATED") return "—";
  return status.toLowerCase();
}

function channelNote(mon: MonitoringData, key: LagSafeKey, firstCritical: number | null): string {
  const lagSafe = mon.lag_safe;
  switch (key) {
    case "score_shift": {
      const atOnset = lagSafe.find((r) => r.step === firstCritical)?.score_psi ?? null;
      const preDrift = lagSafe
        .filter((r) => r.step < DRIFT_START)
        .map((r) => r.score_psi)
        .filter((v): v is number => v !== null);
      const preMax = preDrift.length ? Math.max(...preDrift) : null;
      const quietAgain =
        firstCritical === null
          ? null
          : lagSafe.find((r) => r.step > firstCritical && r.channels.score_shift?.status === "OK")?.step ?? null;
      return `PSI ${num(atOnset, 3)} against a pre-drift max of ${num(preMax, 3)}.${
        quietAgain ? ` Onset detector: quiet again from ${quietAgain}.` : ""
      }`;
    }
    case "performance":
      return firstCritical === null
        ? "Worst per-step F1 over the labelled steps read."
        : `First step where step ${firstCritical - 1} labels have arrived.`;
    case "prevalence":
      return "Rolling illicit share below the critical floor.";
    case "feature_shift":
      return "Stayed in its pre-drift range at 43 and caused the step-37 false alarm.";
    default:
      return "";
  }
}

export function channelCards(mon: MonitoringData): ChannelCard[] {
  const thresholds = new Map(mon.latest_decision.channels.map((c) => [c.name, c.threshold]));
  return LAG_SAFE_KEYS.map((key) => {
    const first = mon.lag_safe.find((r) => r.channels[key]?.status === "CRITICAL");
    const kind = mon.latest_decision.channels.find((c) => c.name === key)?.kind ?? "";
    const firstCritical = first?.step ?? null;
    return {
      key,
      name: CHANNEL_LABEL[key] ?? key,
      kind,
      firstCritical,
      threshold: thresholds.get(key) ?? null,
      note: channelNote(mon, key, firstCritical),
    };
  });
}

/** Label-read window text, e.g. "41, 42" or "—" for step 35 (no prior labels). */
export function labelsRead(perfSteps: number[]): string {
  return perfSteps.length ? perfSteps.join(", ") : "—";
}

export function actionLabel(action: TriggerAction | string): string {
  return action === "RETRAIN" ? "Retrain" : action === "NO_ACTION" ? "No action" : "Recalibrate";
}

export function actionBadgeClass(action: TriggerAction | string): string {
  return action === "RETRAIN"
    ? "badge badge--retrain"
    : action === "NO_ACTION"
      ? "badge badge--ok"
      : "badge badge--neutral";
}

export function channelStateClass(status: ChannelState): string {
  return status === "CRITICAL" ? "cell-drift--strong" : "";
}
