/** Number formatting. Missing values render as "n/a" — never a guessed number. */

export const NA = "n/a";

export const num = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? NA : v.toFixed(digits);

/** Two-decimal percent, e.g. 0.0916 → "9.16%". */
export const pct = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? NA : `${(v * 100).toFixed(digits)}%`;

/** Drop trailing zeros: 0.810 → "0.81", 0.0750 → "0.075". */
export const trim = (v: number | null | undefined, maxDp = 3): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  const s = v.toFixed(maxDp);
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
};

/** Probability for the Try It card: tiny scores stay readable. */
export const prob = (v: number | null | undefined): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  if (v === 0) return "0";
  if (v < 1e-3) return v.toFixed(7).replace(/0+$/, "").replace(/\.$/, "");
  return v.toFixed(4);
};

/** "43–49" (en dash). */
export const span = (a: number | string, b: number | string): string => `${a}–${b}`;
