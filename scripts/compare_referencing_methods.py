"""
Compare four re-referencing methods for monotone/cumulative features (Phase 8a).

Motivation
----------
Aggregate_feature_10/43/8 are secular-trend (cumulative) features: their per-step
median is a near-perfect monotone ramp (Spearman rho = 1.0). A static-reference PSI
therefore reports a large, permanent "drift" at every future step regardless of
regime. This script evaluates four alternative re-referencing schemes on the real
Elliptic++ per-step distributions and writes a comparison artifact:

  1. M1  differenced reference  - step-over-step delta of a robust location statistic
                                  and of the per-step quantile vector, compared with
                                  the reference delta distribution from steps 1-34.
  2. M2  rolling reference      - each step vs the pooled preceding N=10 steps.
  3. M3  detrended residuals    - linear/log-linear trend fitted on the per-step
                                  reference median (1-34); PSI and standardized
                                  location shift of the residuals.
  4. M4  percentile-shape       - within-step robust quantile shape (range-normalised,
                                  magnitude-free) vs the reference shape pool.

Run:
    python scripts/compare_referencing_methods.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.feature_drift import calculate_psi

TRIAD = ["Aggregate_feature_10", "Aggregate_feature_43", "Aggregate_feature_8"]
CACHE = "results/monitoring/features_cache.joblib"
OUT_CSV = "results/monitoring/referencing_method_comparison.csv"

# Method 1 reference-delta pool uses these robust per-step quantiles.
DELTA_QUANTILES = np.linspace(5, 95, 10)
# Method 4 shape signature uses these within-step quantiles.
SHAPE_QUANTILES = np.linspace(5, 95, 11)


def _step_quantiles(df: pd.DataFrame, feat: str, qs: np.ndarray) -> np.ndarray:
    v = df[feat].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return np.percentile(v, qs) if len(v) else None


def _step_median(df: pd.DataFrame, feat: str) -> float:
    v = df[feat].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v))


def method1_differenced(all_steps: dict, feat: str) -> dict:
    """Rate-of-change monitor: delta of the step median vs the reference delta pool."""
    ref_steps = sorted(s for s in all_steps if s <= 34)
    deltas = np.array([_step_median(all_steps[s], feat) - _step_median(all_steps[s - 1], feat)
                       for s in ref_steps if s - 1 in all_steps])
    mu, sd = float(deltas.mean()), float(deltas.std())
    ref_pool = []
    for s in ref_steps:
        if s - 1 not in all_steps:
            continue
        a = _step_quantiles(all_steps[s], feat, DELTA_QUANTILES)
        b = _step_quantiles(all_steps[s - 1], feat, DELTA_QUANTILES)
        if a is not None and b is not None:
            ref_pool.append(a - b)
    ref_pool = np.concatenate(ref_pool)
    out = {}
    for t in range(35, 50):
        dlt = _step_median(all_steps[t], feat) - _step_median(all_steps[t - 1], feat)
        z = (dlt - mu) / sd if sd > 0 else 0.0
        a = _step_quantiles(all_steps[t], feat, DELTA_QUANTILES)
        b = _step_quantiles(all_steps[t - 1], feat, DELTA_QUANTILES)
        out[t] = {"z": float(z), "psi": float(calculate_psi(ref_pool, a - b))}
    return out


def method2_rolling(all_steps: dict, feat: str, window: int = 10) -> dict:
    """Trailing rolling reference: each step vs the pooled preceding `window` steps."""
    out = {}
    for t in range(35, 50):
        win = [all_steps[s][feat].to_numpy(dtype=float) for s in range(t - window, t) if s in all_steps]
        pool = np.concatenate([w[np.isfinite(w)] for w in win])
        out[t] = {"psi": float(calculate_psi(pool, all_steps[t][feat].to_numpy(dtype=float)))}
    return out


def method3_detrended(all_steps: dict, feat: str) -> dict:
    """Detrended residuals: trend on per-step median (1-34), monitor residuals."""
    xs = np.arange(1, 35, dtype=float)
    ys = np.array([_step_median(all_steps[s], feat) for s in xs])
    coef = np.polyfit(xs, ys, 1)
    trend_type = "linear"
    pred_lin = np.polyval(coef, xs)
    rmse_lin = float(np.sqrt(np.mean((ys - pred_lin) ** 2)))
    if np.all(ys > 0):
        coef_log = np.polyfit(xs, np.log(ys), 1)
        pred_log = np.exp(np.polyval(coef_log, xs))
        if float(np.sqrt(np.mean((ys - pred_log) ** 2))) < rmse_lin:
            coef, trend_type = coef_log, "log-linear"
    def trend(step):
        return np.exp(np.polyval(coef, step)) if trend_type == "log-linear" else np.polyval(coef, step)
    ref_res = np.concatenate([all_steps[s][feat].to_numpy(dtype=float) - trend(s) for s in xs])
    ref_res = ref_res[np.isfinite(ref_res)]
    mu, sd = float(ref_res.mean()), float(ref_res.std())
    out = {}
    for t in range(35, 50):
        r = all_steps[t][feat].to_numpy(dtype=float)
        r = r[np.isfinite(r)] - trend(t)
        out[t] = {
            "psi": float(calculate_psi(ref_res, r)),
            "resid_z": float((r.mean() - mu) / sd) if sd > 0 else 0.0,
            "drift_score": float(abs(r.mean() - mu) / sd) if sd > 0 else 0.0,
        }
    out["_trend_type"] = trend_type
    return out


def method4_percentile_shape(all_steps: dict, feat: str) -> dict:
    """Within-step robust quantile shape (magnitude-free) vs the reference pool."""
    def shape(s):
        q = _step_quantiles(all_steps[s], feat, SHAPE_QUANTILES)
        if q is None or q[-1] - q[0] <= 0:
            return None
        return (q - q[0]) / (q[-1] - q[0])
    ref = [shape(s) for s in range(1, 35)]
    ref = [x for x in ref if x is not None]
    pool = np.concatenate(ref)
    out = {}
    for t in range(35, 50):
        sh = shape(t)
        out[t] = {"psi": float(calculate_psi(pool, sh)) if sh is not None else float("nan")}
    return out


def load_all_steps() -> dict:
    import joblib
    d = joblib.load(CACHE)
    all_steps = {int(s): g for s, g in d["reference_df"].groupby("Time step")}
    for s, g in d["test_steps_dict"].items():
        if len(g):
            all_steps[int(s)] = g
    return all_steps


def main() -> None:
    all_steps = load_all_steps()
    rows = []
    for feat in TRIAD:
        m1 = method1_differenced(all_steps, feat)
        m2 = method2_rolling(all_steps, feat)
        m3 = method3_detrended(all_steps, feat)
        m4 = method4_percentile_shape(all_steps, feat)
        for t in range(35, 50):
            rows.append({
                "feature": feat,
                "time_step": t,
                "trend_type": m3["_trend_type"],
                "m1_delta_z": m1[t]["z"],
                "m1_delta_psi": m1[t]["psi"],
                "m2_rolling_psi": m2[t]["psi"],
                "m3_resid_psi": m3[t]["psi"],
                "m3_resid_z": m3[t]["resid_z"],
                "m3_drift_score": m3[t]["drift_score"],
                "m4_shape_psi": m4[t]["psi"],
            })
    df = pd.DataFrame(rows)
    Path("results/monitoring").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = {}
    for method in ["m1_delta_z", "m2_rolling_psi", "m3_resid_psi", "m4_shape_psi"]:
        grouped = df.groupby("time_step")[method]
        if method == "m1_delta_z":
            stat = grouped.apply(lambda x: float(np.nanmean(np.abs(x))))
        else:
            stat = grouped.mean()
        stable = stat.loc[35:41]
        drift = stat.loc[42:49]
        summary[method] = {
            "stable_mean": float(stable.mean()), "stable_max": float(stable.max()),
            "drift_mean": float(drift.mean()), "drift_max": float(drift.max()),
        }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
