import type { WalkforwardData } from "./types";

/** step -> illicit labels, for steps with too few positives for a per-step metric to mean anything. */
export const noiseMap = (wf: WalkforwardData): Record<number, number> =>
  Object.fromEntries(wf.steps.filter((s) => s.noise).map((s) => [s.step, s.n_illicit]));
