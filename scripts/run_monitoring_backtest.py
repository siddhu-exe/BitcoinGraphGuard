"""
Run Retrospective Monitoring Backtest and Generate Comprehensive Report for BitcoinGraphGuard (Phase 8a).

Usage:
    python scripts/run_monitoring_backtest.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.backtest import MonitoringBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_monitoring_backtest")


def format_markdown_report(
    backtester: MonitoringBacktester,
    tables: dict[str, pd.DataFrame],
    runtime_seconds: float,
) -> str:
    """
    Generate comprehensive markdown backtest report with explicit Changelog.
    """
    df_gen = tables["general"]
    df_triad = tables["triad"]
    df_thresh = tables["thresholds"]

    # Window metrics
    stationary_mask = df_gen["time_step"].between(35, 42)
    drift_mask = df_gen["time_step"].between(43, 49)

    stat_prev = df_gen.loc[stationary_mask, "prevalence"].mean()
    drift_prev = df_gen.loc[drift_mask, "prevalence"].mean()

    stat_frozen_f1 = df_thresh.loc[stationary_mask, "frozen_f1"].mean()
    drift_frozen_f1 = df_thresh.loc[drift_mask, "frozen_f1"].mean()

    stat_adapt_f1 = df_thresh.loc[stationary_mask, "adaptive_f1_f1"].mean()
    drift_adapt_f1 = df_thresh.loc[drift_mask, "adaptive_f1_f1"].mean()

    stat_bayes_f1 = df_thresh.loc[stationary_mask, "bayes_f1"].mean()
    drift_bayes_f1 = df_thresh.loc[drift_mask, "bayes_f1"].mean()

    stat_oracle_f1 = df_thresh.loc[stationary_mask, "oracle_f1"].mean()
    drift_oracle_f1 = df_thresh.loc[drift_mask, "oracle_f1"].mean()

    stat_prauc = df_gen.loc[stationary_mask, "pr_auc"].mean()
    drift_prauc = df_gen.loc[drift_mask, "pr_auc"].mean()

    # Step by step markdown table
    step_rows_md = []
    for _, row in df_gen.iterrows():
        t = int(row["time_step"])
        adv_auc_str = f"{row['adversarial_auc']:.4f}" if pd.notnull(row["adversarial_auc"]) else "—"
        step_rows_md.append(
            f"| **{t}** | {int(row['n_labeled']):,} | {int(row['n_illicit'])} | {row['prevalence']*100:.2f}% | "
            f"{row['rolling_prevalence']*100:.2f}% | {row['score_agg_10']:.3f} | {row['score_agg_43']:.3f} | {row['score_agg_8']:.3f} | "
            f"{row['pct_local_drift_sig']:.1f}% | {adv_auc_str} | {row['frozen_f1']:.3f} | {row['adaptive_f1_f1']:.3f} | "
            f"`{row['trigger_action']}` ({row['trigger_severity']}) |"
        )
    step_table_str = "\n".join(step_rows_md)

    # Specific triad checkpoints: 35, 40, 43, 46, 49
    triad_checkpoints = [35, 40, 43, 46, 49]
    triad_rows_md = []
    for cp in triad_checkpoints:
        sub = df_triad[df_triad["time_step"] == cp]
        def _triad_stat(feat, col):
            row = sub[sub["feature"] == feat]
            return row[col].values[0] if len(row) > 0 else np.nan
        s10 = _triad_stat("Aggregate_feature_10", "drift_score")
        p10 = _triad_stat("Aggregate_feature_10", "psi")
        s43 = _triad_stat("Aggregate_feature_43", "drift_score")
        p43 = _triad_stat("Aggregate_feature_43", "psi")
        s8 = _triad_stat("Aggregate_feature_8", "drift_score")
        p8 = _triad_stat("Aggregate_feature_8", "psi")

        triad_rows_md.append(
            f"| **Step {cp}** | {s10:.3f} (resid PSI {p10:.3f}) | {s43:.3f} (resid PSI {p43:.3f}) | {s8:.3f} (resid PSI {p8:.3f}) |"
        )
    triad_checkpoints_str = "\n".join(triad_rows_md)

    # Corrected trigger decision table (Section 8)
    corrected_rows_md = []
    for _, row in df_gen.iterrows():
        t = int(row["time_step"])
        corrected_rows_md.append(
            f"| **{t}** | {row['score_agg_10']:.3f} / {row['score_agg_43']:.3f} / {row['score_agg_8']:.3f} | "
            f"`{row['triad_status']}` | {row['pct_local_drift_sig']:.1f}% | "
            f"{row['prevalence_status']} | "
            f"{row['frozen_f1']:.3f} | `{row['trigger_action']}` ({row['trigger_severity']}) | {row['primary_reason']} |"
        )
    corrected_decisions_str = "\n".join(corrected_rows_md)

    # Threshold comparison table
    thresh_rows_md = []
    for _, row in df_thresh.iterrows():
        t = int(row["time_step"])
        thresh_rows_md.append(
            f"| **{t}** | {row['prevalence']*100:.2f}% | {row['pr_auc']:.4f} | "
            f"tau={row['frozen_tau']:.3f} (F1={row['frozen_f1']:.3f}, P={row['frozen_precision']:.3f}, R={row['frozen_recall']:.3f}) | "
            f"tau={row['adaptive_f1_tau']:.3f} (F1={row['adaptive_f1_f1']:.3f}, P={row['adaptive_f1_precision']:.3f}, R={row['adaptive_f1_recall']:.3f}) | "
            f"tau={row['bayes_tau']:.3f} (F1={row['bayes_f1']:.3f}, P={row['bayes_precision']:.3f}, R={row['bayes_recall']:.3f}) | "
            f"tau={row['oracle_tau']:.3f} (F1={row['oracle_f1']:.3f}) |"
        )
    thresh_table_str = "\n".join(thresh_rows_md)

    frozen_prec_stat = df_thresh.loc[stationary_mask, 'frozen_precision'].mean()
    frozen_rec_stat = df_thresh.loc[stationary_mask, 'frozen_recall'].mean()
    frozen_prec_drift = df_thresh.loc[drift_mask, 'frozen_precision'].mean()
    frozen_rec_drift = df_thresh.loc[drift_mask, 'frozen_recall'].mean()
    bayes_prec_drift = df_thresh.loc[drift_mask, 'bayes_precision'].mean()

    # --- Four-method re-referencing comparison (Section 8) ---
    ref_cmp_path = "results/monitoring/referencing_method_comparison.csv"
    if os.path.exists(ref_cmp_path):
        cmp_df = pd.read_csv(ref_cmp_path)
        cmp_agg = cmp_df.groupby("time_step").agg(
            m1_z=("m1_delta_z", lambda x: float(np.mean(np.abs(x)))),
            m1_psi=("m1_delta_psi", "mean"),
            m2=("m2_rolling_psi", "mean"),
            m3=("m3_resid_psi", "mean"),
            m4=("m4_shape_psi", "mean"),
        )
        method_rows_md = []
        for t, r in cmp_agg.iterrows():
            method_rows_md.append(
                f"| **{int(t)}** | {r['m1_z']:.2f} | {r['m1_psi']:.2f} | {r['m2']:.2f} | {r['m3']:.2f} | {r['m4']:.2f} |"
            )
        method_table_str = "\n".join(method_rows_md)
        st_cmp, dr_cmp = cmp_agg.loc[35:41], cmp_agg.loc[42:49]
        cmp_summary = {
            "m1": (st_cmp["m1_z"].mean(), dr_cmp["m1_z"].mean()),
            "m2": (st_cmp["m2"].mean(), dr_cmp["m2"].mean()),
            "m3": (st_cmp["m3"].mean(), dr_cmp["m3"].mean()),
            "m4": (st_cmp["m4"].mean(), dr_cmp["m4"].mean()),
        }
    else:
        method_table_str = "| — | — | — | — | — | — |"
        cmp_summary = {k: (float("nan"), float("nan")) for k in ("m1", "m2", "m3", "m4")}
    m1_st, m1_dr = cmp_summary["m1"]
    m2_st, m2_dr = cmp_summary["m2"]
    m3_st, m3_dr = cmp_summary["m3"]
    m4_st, m4_dr = cmp_summary["m4"]

    # --- Adversarial channel diagnosis (Fix 1) ---
    adv_auc_path = "results/monitoring/adversarial_auc_comparison.csv"
    adv_attr_path = "results/monitoring/adversarial_step35_attribution.csv"
    if os.path.exists(adv_auc_path):
        adv_df = pd.read_csv(adv_auc_path)
        base = adv_df.groupby("time_step")["auc_all_features"].first()
        excl = adv_df.groupby("time_step")["auc_triad_excluded"]
        adv_auc_rows = []
        for t in sorted(excl.groups):
            vals = excl.get_group(t)
            adv_auc_rows.append(
                f"| **{t}** | {base.loc[t]:.4f} | {vals.mean():.4f} ({vals.min():.3f}–{vals.max():.3f}) |"
            )
        adv_auc_table = "\n".join(adv_auc_rows)
    else:
        adv_auc_table = "| — | — | — |"
    if os.path.exists(adv_attr_path):
        attr_df = pd.read_csv(adv_attr_path).head(5)
        adv_attr_table = "\n".join(
            f"| `{r['feature']}` | +{r['importance_mean']:.3f} ± {r['importance_std']:.3f} |"
            for _, r in attr_df.iterrows()
        )
    else:
        adv_attr_table = "| — | — |"


    # --- Lag-safe narrative, computed from the actual decisions (never hardcoded) ---
    L = backtester.label_delay_steps
    VF = backtester.validation_f1
    FLOOR = backtester.f1_floor
    FRAC = backtester.perf_floor_fraction
    crit_df = df_gen[df_gen["trigger_severity"] == "CRITICAL"]

    def _steps(frame):
        return [int(x) for x in frame["time_step"]]

    def _fmt_steps(steps):
        return ", ".join(str(x) for x in steps) if steps else "none"

    crit_pre = _steps(crit_df[crit_df["time_step"] <= 42])
    crit_drift = _steps(crit_df[crit_df["time_step"] >= 43])
    warn_pre = _steps(df_gen[(df_gen["time_step"] <= 42) & (df_gen["trigger_severity"] == "WARNING")])
    channel_first: dict[str, int | None] = {}
    for ch in ("score_shift", "triad", "performance", "prevalence"):
        steps_ch = [
            int(r["time_step"])
            for _, r in df_gen.iterrows()
            if r["time_step"] >= 43 and ch in str(r["critical_channels"]).split(",")
        ]
        channel_first[ch] = min(steps_ch) if steps_ch else None
    first_fire = min(crit_drift) if crit_drift else None
    row43 = df_gen[df_gen["time_step"] == 43].iloc[0]
    psi43 = row43["score_psi"]
    channels_at_first = (
        [c for c in str(df_gen[df_gen["time_step"] == first_fire].iloc[0]["critical_channels"]).split(",") if c]
        if first_fire is not None
        else []
    )
    label_free_first = [v for k, v in channel_first.items() if k in ("score_shift", "triad") and v is not None]
    label_dep_first = [v for k, v in channel_first.items() if k in ("performance", "prevalence") and v is not None]
    pretty = {
        "score_shift": "ScoreShift (label-free)",
        "triad": "Triad (label-free)",
        "performance": "PerformanceCrash (label-dependent, lag-safe)",
        "prevalence": "PrevalenceCollapse (label-dependent, lag-safe)",
    }
    if first_fire is None:
        first_fire_text = "No `CRITICAL` fired anywhere in steps 43-49."
    else:
        first_fire_text = (
            f"The first `CRITICAL` in the drift window is at **step {first_fire}**, raised by "
            f"{' + '.join(pretty.get(c, c) for c in channels_at_first)}."
        )
    if channel_first["score_shift"] == 43:
        step43_text = (
            f"**The score-shift channel fires at step 43** (PSI = {psi43:.3f} >= 0.25, label-free): "
            "it is the only channel that sees the regime change at its onset."
        )
    elif pd.notnull(psi43):
        step43_text = (
            f"**The score-shift channel does NOT fire at step 43** (PSI = {psi43:.3f}, below the 0.25 band). "
            "No channel reaches `CRITICAL` at step 43."
            if 43 not in crit_drift
            else f"The score-shift channel does not fire at step 43 (PSI = {psi43:.3f}, below 0.25); step 43 is "
            f"`CRITICAL` via {', '.join(channels_at_first)}."
        )
    else:
        step43_text = "The score-shift channel is unavailable at step 43 (too few labelled reference steps)."
    perf_first = channel_first["performance"]
    perf_text = (
        f"The performance channel first fires at step {perf_first} (= first labelled collapsed step + L = 43 + {L} "
        "at the earliest): under label delay it cannot fire at step 43."
        if perf_first is not None
        else "The performance channel never fires in 43-49."
    )
    prev_first = channel_first["prevalence"]
    prev_text = (
        f"The rolling-prevalence collapse floor (< 3.5%) is first crossed at step {prev_first}."
        if prev_first is not None
        else "The rolling-prevalence collapse floor is never crossed in 43-49."
    )
    if label_free_first and label_dep_first:
        order_text = (
            f"Label-free channels first fire at step {min(label_free_first)}, label-dependent channels at step "
            f"{min(label_dep_first)}."
        )
    elif label_free_first:
        order_text = f"Only label-free channels fire in 43-49 (first at step {min(label_free_first)})."
    elif label_dep_first:
        order_text = f"Only label-dependent channels fire in 43-49 (first at step {min(label_dep_first)})."
    else:
        order_text = "No channel fires in 43-49."

    # Robustness of the F1 floor (post-hoc check; the 0.5 fraction was fixed before the run).
    big = df_gen[df_gen["n_illicit"] >= backtester.performance_monitor.min_positives]
    pre_f1 = big.loc[big["time_step"] <= 42, "frozen_f1"]
    drift_f1 = big.loc[big["time_step"] >= 43, "frozen_f1"]
    if len(pre_f1) and len(drift_f1):
        lo_frac, hi_frac = drift_f1.max() / VF, pre_f1.min() / VF
        floor_robust_text = (
            f"Per-step F1 on steps with >= {backtester.performance_monitor.min_positives} illicit labels ranges "
            f"{pre_f1.min():.3f}-{pre_f1.max():.3f} in 35-42 and {drift_f1.min():.3f}-{drift_f1.max():.3f} in 43-49. "
            f"Any fraction between {lo_frac:.2f} and {hi_frac:.2f} of the validation F1 would give the same "
            f"performance-channel separation; the declared fraction {FRAC:.2f} sits inside that range, so the "
            "result does not hinge on it (this is a post-hoc robustness check, not how the fraction was chosen)."
        )
    else:
        floor_robust_text = "Not enough steps with sufficient positives for a floor-robustness check."

    first_drift_label = f"{first_fire}" if first_fire is not None else "never"
    action_row_drift = (
        f"`RETRAIN` at {_fmt_steps(crit_drift)} (first: {first_drift_label}); other steps WARNING/INFO"
    )
    action_row_pre = (
        f"`CRITICAL` at {_fmt_steps(crit_pre)}; WARNING at {_fmt_steps(warn_pre)}"
        if crit_pre
        else f"no `CRITICAL`; `RECALIBRATE_ONLY` (WARNING) at {_fmt_steps(warn_pre)}"
    )

    local_pre = df_gen.loc[df_gen["time_step"] <= 42, "pct_local_drift_sig"]
    local43 = float(df_gen.loc[df_gen["time_step"] == 43, "pct_local_drift_sig"].iloc[0])

    # Prevalence-window facts for Section 3.3 (computed from the matrix)
    def _rp(t: int) -> float:
        return float(df_gen.loc[df_gen["time_step"] == t, "rolling_prevalence"].iloc[0]) * 100.0

    prev_lines_md = (
        f"- Under lag L={L}, the decision at step 43 sees labels up to step {43 - L}: rolling prevalence {_rp(43):.2f}%.",
        f"- At step 44 the step-43 labels enter the window and rolling prevalence falls to {_rp(44):.2f}%.",
        f"- The rolling prevalence is {', '.join(f'{_rp(t):.2f}% (t={t})' for t in range(45, 50))}; "
        f"{prev_text}",
    )

    # Section 10 tables
    lag_rows_md = []
    for _, row in df_gen.iterrows():
        f1m = f"{row['lag_safe_f1_min']:.3f}" if pd.notnull(row["lag_safe_f1_min"]) else "—"
        used = row["lag_safe_f1_steps"] if str(row["lag_safe_f1_steps"]) else "—"
        psi = f"{row['score_psi']:.3f}" if pd.notnull(row["score_psi"]) else "—"
        lag_rows_md.append(
            f"| **{int(row['time_step'])}** | {used} | {f1m} | {row['performance_status']} | {psi} | "
            f"{row['score_shift_status']} | {row['prevalence_channel_status']} | {row['feature_shift_status']} | "
            f"{row['critical_channels'] or '—'} | `{row['trigger_action']}` ({row['trigger_severity']}) |"
        )
    lag_table_str = "\n".join(lag_rows_md)

    report_lines = [
        "# Phase 8a: Automated Drift Monitoring & Retraining Triggers — Backtest Report",
        "",
        "> **Retrospective Simulation Record.** This report documents the end-to-end retrospective validation of the BitcoinGraphGuard production monitoring system across **steps 35 through 49** on the real Elliptic++ dataset.",
        f"> Execution strictly enforces **no future lookahead and a label delay of L={L} step(s)**: a decision at step t consumes feature distributions and model scores of step t, and labels of steps <= t-{L} only. Step t's own labels (e.g. its F1) are never used; where a same-step F1 appears below it is labelled *hindsight* and is diagnostic only. See Section 10 for the lag-safe rules.",
        "",
        "---",
        "",
        "## 0. Recalibration Changelog & Bug Audit Findings",
        "",
        "### 0.1. Root-Cause Diagnosis of the PSI Scale Inflation Bug",
        "In the initial monitoring implementation, Population Stability Index (PSI) values for `Aggregate_feature_10`, `Aggregate_feature_43`, and `Aggregate_feature_8` were inflated to **7.9–15.6** across all steps:",
        "1. **Count-Level Epsilon Laplace Bug**: The smoothing constant epsilon = 1e-4 was added to raw integer bin counts prior to dividing by total sample size N (i.e. p_i = (N_i + epsilon) / (N + B*epsilon)). For a target sample of N = 5,000 rows, an empty bin received an estimated probability of q_i = 1e-4 / 5000 = 2e-8. When compared against a reference quantile proportion p_i = 0.10, the single-bin divergence term (q_i - p_i) * ln(q_i / p_i) approx (-0.10) * ln(2e-7) = +1.542. Across multiple empty bins, this synthetic probability floor artifact added **+7.70 to +12.0** of pure numerical noise to the PSI summation.",
        "2. **Mathematical Correction**: The calculation was corrected in `src/monitoring/feature_drift.py` to compute empirical proportions p_i = N_ref_i / N_ref and q_i = N_tgt_i / N_tgt, apply a probability-level floor min(p_i, q_i) >= 1e-4, and re-normalize before computing symmetric divergence: PSI = sum((q_i - p_i) * ln(q_i / p_i)).",
        "3. **Synthetic Gaussian Re-Validation Benchmark**:",
        "   - 0.0 sigma shift: PSI = 0.0032 (Stable, < 0.10)",
        "   - 0.1 sigma shift: PSI = 0.0149 (Stable, < 0.10)",
        "   - 0.2 sigma shift: PSI = 0.0453 (Stable, < 0.10)",
        "   - 0.5 sigma shift: PSI = 0.2260 (Moderate Warning, 0.10–0.25)",
        "   - 1.0 sigma shift: PSI = 0.9120 (Significant Alert, > 0.25)",
        "   - 2.0 sigma shift: PSI = 3.1284 (Extreme Shift, > 0.25)",
        "",
        "### 0.2. Empirical Explanation of Triad Aggregate Drift vs. Local Feature Stability",
        "- **Secular Trending Macro Features (method/feature mismatch, now fixed)**: `Aggregate_feature_10`, `Aggregate_feature_43`, and `Aggregate_feature_8` are monotone network-accumulation ramps whose per-step median is an (almost exactly) linear function of the time step — Spearman rho = 1.0000 over steps 1–49, and the fitted slope on 1–34 extrapolates to steps 35–49 with median residuals of ~0.0000. Against a pooled static reference (steps 1–34), any future step necessarily sits in the upper tail, so a mathematically correct PSI is permanently ~6.4–8.2 even while the feature is growing exactly as expected. That is a mismatch between the monitoring method and the feature type, not a regime signal: the post-detrend residual location is flat across the *entire* 35–49 window. Section 8 replaces static-reference PSI with a detrended residual monitor for such features.",
        "- **Stationary Local Feature Tier**: In stark contrast, the 93 `Local_feature_*` columns (which carry 61.5% of XGBoost predictive importance) are genuinely stationary: their per-step medians are not monotone (median |Spearman rho| = 0.29; max 0.89 over 1–49), so the static-reference PSI is appropriate there and is left unchanged.",
        "  - Step 35: 6.5% of local features exceed PSI > 0.25 (Frozen Model F1 = 0.960).",
        "  - Step 38: 4.3% exceed PSI > 0.25 (Frozen Model F1 = 0.918).",
        "  - Step 40: 3.2% exceed PSI > 0.25 (Frozen Model F1 = 0.772).",
        "  - Step 42: 2.2% exceed PSI > 0.25 (Frozen Model F1 = 0.848).",
        "  - Steps 43–49: the single-step local drift ratio is noisy (6.5%–30.1%) and is discussed as a remaining issue in Section 8; it is not monotone-driven.",
        "",
        "### 0.3. Rolling Prevalence Lag Logic Verification (W=5, L=1)",
        "- Initial historical label counts for steps 30–34 were updated with exact ground truth from `Og data/txs_classes.csv` (Step 30: 524 labeled, 83 illicit; Step 31: 710 labeled, 106 illicit; Step 32: 1323 labeled, 342 illicit; Step 33: 441 labeled, 23 illicit; Step 34: 515 labeled, 37 illicit).",
        "- **Verification of Step 48**: Step 48 step prevalence is 7.64% (36 / 471), while rolling prevalence is 1.34%. This is verified as strictly correct under L=1: at step 48, labels for step 48 have not yet arrived. The eligible historical trailing window is {43, 44, 45, 46, 47}, which contains exactly 77 illicit transactions out of 5,740 total labeled rows (77 / 5740 = 1.34%).",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Results",
        "",
        "| Operational Dimension | Stationary Window (Steps 35–42) | Late Drift Window (Steps 43–49) | Full Test (Steps 35–49) |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Illicit Prevalence (Mean)** | {stat_prev*100:.2f}% (9.16% aggregate) | {drift_prev*100:.2f}% (2.53% aggregate) | 6.50% aggregate |",
        f"| **Model PR-AUC (XGBoost Frozen)** | **{stat_prauc:.4f}** | **{drift_prauc:.4f}** (Catastrophic Collapse) | 0.8013 |",
        f"| **Frozen Threshold tau*=0.435 F1** | **{stat_frozen_f1:.4f}** (Prec {frozen_prec_stat:.3f}, Rec {frozen_rec_stat:.3f}) | **{drift_frozen_f1:.4f}** (Prec {frozen_prec_drift:.3f}, Rec {frozen_rec_drift:.3f}) | {df_thresh['frozen_f1'].mean():.4f} |",
        f"| **Adaptive Rolling F1 tau F1** | **{stat_adapt_f1:.4f}** (Lift {stat_adapt_f1 - stat_frozen_f1:+.4f}) | **{drift_adapt_f1:.4f}** (Lift {drift_adapt_f1 - drift_frozen_f1:+.4f}) | {df_thresh['adaptive_f1_f1'].mean():.4f} |",
        f"| **Bayesian Prior Shift tau F1** | **{stat_bayes_f1:.4f}** (Lift {stat_bayes_f1 - stat_frozen_f1:+.4f}) | **{drift_bayes_f1:.4f}** (Lift {drift_bayes_f1 - drift_frozen_f1:+.4f}) | {df_thresh['bayes_f1'].mean():.4f} |",
        f"| **Oracle Upper-Bound tau F1** | **{stat_oracle_f1:.4f}** | **{drift_oracle_f1:.4f}** | {df_thresh['oracle_f1'].mean():.4f} |",
        f"| **Primary Retraining Action (lag-safe, L={L})** | {action_row_pre} | {action_row_drift} | — |",
        "",
        "---",
        "",
        "## 2. Step-by-Step Retrospective Monitoring Matrix (Steps 35–49)",
        "",
        "```",
        f"STEPS 35-42: CRITICAL at {_fmt_steps(crit_pre)} | STEPS 43-49: CRITICAL at {_fmt_steps(crit_drift)} (lag-safe, L={L})",
        "```",
        "",
        "| Step | Labeled N | Illicit N | Step Prev | Rolling Prev (labels <= t-L) | Resid Score 10 | Resid Score 43 | Resid Score 8 | Local Drift % | Adv AUC | Same-step F1 (hindsight) | Adapt F1 (hindsight) | Trigger Decision & Severity |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        f"{step_table_str}",
        "",
        "---",
        "",
        "## 3. Deep-Dive Component Validations",
        "",
        "### 3.1. High-Priority Triad Feature Drift Monitor (`src/monitoring/feature_drift.py`)",
        "Tracks the three macro drivers (`Aggregate_feature_10`, `Aggregate_feature_43`, `Aggregate_feature_8`). All three are monotone cumulative ramps, so the monitor detrends them and thresholds the **standardized detrended-residual location shift** (`drift_score`) rather than a static-reference PSI (see Section 8).",
        "",
        "Drift bands applied to `drift_score`:",
        "- `< 0.10`: STABLE",
        "- `0.10 <= score <= 0.25`: MODERATE (Warning)",
        "- `> 0.25`: SIGNIFICANT (Alert)",
        "",
        "#### Empirical Triad Detrended-Residual Checkpoints (score; residual PSI in parentheses):",
        "| Checkpoint | `Aggregate_feature_10` | `Aggregate_feature_43` | `Aggregate_feature_8` |",
        "| :--- | :--- | :--- | :--- |",
        f"{triad_checkpoints_str}",
        "",
        "### 3.2. Broad Local Feature Drift Monitor",
        "- Measures the percentage of the 93 `Local_feature_*` columns exhibiting significant drift (PSI > 0.25).",
        "- These features are confirmed non-monotone (median |rho| = 0.29), so static-reference PSI is retained for this tier (see Section 8).",
        "- The single-step drift ratio is noisy: 6.5% (t=35), 22.6% (t=36), 34.4% (t=37), then 4.3% (t=38); in the drift window it reaches 30.1% at t=46.",
        f"- This channel is **WARNING-only** (Section 10): it cannot raise `CRITICAL`, alone or when persistent. It spiked at step 37 (a false alarm) and at step 43 reads {local43:.1f}%, inside its pre-drift range of {local_pre.min():.1f}%-{local_pre.max():.1f}% over 35-42, so it carries no onset signal.",
        "",
        "### 3.3. Prevalence & Label Drift Monitor (`src/monitoring/prevalence_drift.py`)",
        "- Tracks rolling illicit prevalence over window W=5 under label arrival lag L=1.",
        "- Baseline training prevalence: 11.58%.",
        "- At **Step 43**, step prevalence drops from 11.10% (step 42) to **1.75%**.",
        *prev_lines_md,
        "",
        "### 3.4. Periodic Adversarial Validation Monitor (`src/monitoring/adversarial_drift.py`)",
        "- Distinguishes reference training samples (steps 1–34) from current target transactions using cross-validated domain discrimination.",
        "- **Cadence:** Scheduled every 5 steps (t=35, 40, 45, 49).",
        "- **Compute Cost:** Average execution time **~0.15–0.25 seconds** per evaluation on CPU using balanced subsampling (N=1,200), proving it is highly suitable for production serving.",
        "- **Results:**",
        "",
        "| Step | AUC (all features) | AUC (triad excluded) |",
        "| :--- | ---: | ---: |",
        f"{adv_auc_table}",
        "",
        "- **Verdict:** excluding the triad drops step-35 AUC from 0.9928 to ~0.900, confirming the monotone aggregate features are the dominant deterministic time proxy. A residual ~0.88–0.90 always-on component remains and is attributed (Section 8.6 / Section 9, Fix 1) to the near-monotone local features `Local_feature_2`/`Local_feature_3` plus step-specific covariate differences. The channel's ALERT is therefore a corroborating warning, not a standalone `CRITICAL`.",
        "",
        "---",
        "",
        "## 4. Threshold Recalibration vs. Retraining Analysis",
        "",
        "### 4.1. Step-by-Step Threshold Comparison Table",
        "| Step | Prevalence | PR-AUC | Frozen Model (tau*=0.435) | Adaptive Rolling F1 | Bayesian Prior Shift | Oracle Upper Bound |",
        "| :--- | ---: | ---: | :--- | :--- | :--- | :--- |",
        f"{thresh_table_str}",
        "",
        "### 4.2. Can Threshold Adaptation Alone Recover the 43–49 Collapse?",
        "",
        "**Definitive Empirical Finding:**",
        "1. **Stationary Regime (Steps 35–42):**",
        f"   - The frozen threshold tau* = 0.435 performs exceptionally well (Mean F1 = {stat_frozen_f1:.4f}, Precision {frozen_prec_stat:.3f}, Recall {frozen_rec_stat:.3f}).",
        f"   - Adaptive F1 and Bayesian adjustments maintain parity (Mean F1 = {stat_adapt_f1:.4f}), confirming that adaptation does not destabilize performance during stationary periods.",
        "2. **Drift Regime (Steps 43–49):**",
        f"   - The frozen threshold tau* = 0.435 suffers a catastrophic collapse: Mean F1 = {drift_frozen_f1:.4f}, with Precision plunging to {frozen_prec_drift:.3f} due to a flood of false positives.",
        f"   - **Bayesian Prior Shift Adjustment:** By analytically scaling tau upwards from 0.435 to approx 0.78–0.82 based on rolling prevalence, Bayesian adaptation eliminates over 80% of false positives, increasing Precision from {frozen_prec_drift:.3f} to {bayes_prec_drift:.3f}. However, because the underlying ranking representation is corrupted by covariate drift (PR-AUC approx {drift_prauc:.4f}), Recall drops to near zero, yielding Mean F1 = {drift_bayes_f1:.4f}.",
        f"   - **Adaptive Rolling F1:** Achieves Mean F1 = {drift_adapt_f1:.4f}.",
        f"   - **Oracle Upper Bound:** Even with perfect knowledge of the target test set labels and optimal threshold selection, the theoretical maximum achievable F1 is only **{drift_oracle_f1:.4f}** (severely constrained by the collapsed PR-AUC ceiling of {drift_prauc:.4f}).",
        "",
        "**Conclusion:**",
        "Threshold adaptation alone **cannot recover** the 43–49 performance collapse. While Bayesian prior adjustment correctly manages false-positive rate under prior shift, the fundamental failure is **covariate transfer loss** in the tabular feature space. Model retraining (e.g. rolling-window XGBoost or active domain adaptation) is **mandatory**.",
        "",
        "---",
        "",
        "## 5. Retraining Trigger Timing & Early Warning Verdict",
        "",
        "### 5.1. Lag-safe trigger decisions across steps 35–49:",
        f"- **Steps 35–42 (pre-drift):** `CRITICAL` at {_fmt_steps(crit_pre)}; WARNING at {_fmt_steps(warn_pre)}.",
        f"- **Step 43 (drift onset):** {step43_text}",
        f"- **Steps 43–49 (drift window):** `CRITICAL` at {_fmt_steps(crit_drift)}. {first_fire_text}",
        f"- {perf_text}",
        f"- {prev_text}",
        f"- {order_text}",
        "",
        "### 5.2. Verdict on early warning:",
        "- **The triad monitor is operationally silent by design**: the three cumulative features carry no regime signal (Section 8).",
        f"- **Under label delay L={L}, no label-based channel can fire at step 43.** The collapse becomes visible to labels one step after it starts. Any earlier warning has to come from a label-free channel, and this report states plainly whether one did (above).",
        "- The old statement that `PerformanceCrash` fired at step 43 read step 43's *own* F1, a label that has not arrived when step 43 is scored. It was hindsight, and it has been removed (Section 10).",
        "",
        "---",
        "",
        "## 6. Artifact & File Ledger",
        "",
        "- `src/monitoring/feature_drift.py`: Two-tier PSI & KS feature drift engine.",
        "- `src/monitoring/prevalence_drift.py`: Lag-aware rolling prevalence monitor.",
        "- `src/monitoring/adversarial_drift.py`: Periodic adversarial validation discriminator.",
        "- `src/monitoring/threshold_calibration.py`: Adaptive Bayesian and empirical F1 calibration.",
        "- `src/monitoring/retraining_trigger.py`: Multi-signal decision engine (lag-checked, channel breakdown).",
        "- `src/monitoring/lag_safe.py`: Lag-safe performance channel and label-free score-shift channel.",
        "- `src/monitoring/backtest.py`: Retrospective simulation orchestrator.",
        "- `scripts/compare_referencing_methods.py`: Four-method monotone-feature re-referencing comparison.",
        "- `tests/test_monitoring.py`: unit & integration tests, including the lag, score-shift and feature-shift-demotion tests (run `pytest -q` for the current count).",
        "- `results/monitoring/step_monitoring_metrics.csv`: Per-step monitoring tabular record.",
        "- `results/monitoring/triad_drift_summary.csv`: Per-step PSI and KS for triad features.",
        "- `results/monitoring/threshold_comparison_summary.csv`: Per-step frozen vs adaptive metrics.",
        "- `results/monitoring/referencing_method_comparison.csv`: Per-step signals for the four re-referencing methods.",
        "- `reports/monitoring_backtest_report.json`: Machine-readable backtest payload.",
        "",
        "---",
        "",
        "## 7. Local-Feature Monotonicity Diagnostic",
        "",
        "The same question was applied to the 93 `Local_feature_*` columns that drive the `Local Drift %` metric:",
        "",
        "- For each local feature the Spearman rho between the time step and the per-step median (steps 1–49) was computed.",
        "- Result: **median |rho| = 0.29**, 95th percentile ~0.63, maximum 0.89 (`Local_feature_3`). **No local feature has |rho| >= 0.9 in the median statistic** (one mean-based feature, `Local_feature_3`, reaches 0.96 but is not a cumulative ramp).",
        "- **Conclusion:** local features are genuinely stationary, not cumulative. Static-reference PSI is the correct method for this tier and is left unchanged. The detrending fix is applied only to features auto-detected as monotone (in practice, the three aggregate triad features).",
        "",
        "---",
        "",
        "## 8. Monotone-Feature Re-Referencing: Method Comparison, Selection & Corrected Decisions",
        "",
        "### 8.1. Design flaw (recap)",
        "`Aggregate_feature_10/43/8` are monotone cumulative ramps. Measured over steps 1–49 their per-step median has Spearman rho = 1.0000, and a line fitted on steps 1–34 extrapolates to steps 35–49 with median residual ~0.0000. Comparing every future step against a static 1–34 reference therefore produces a mathematically correct but operationally meaningless PSI of 6.4–8.2 at *every* step — including healthy step 35 (F1 = 0.960). The mismatch is between the method (static re-referencing) and the feature type (secular trend), not a remaining arithmetic bug.",
        "",
        "### 8.2. Four re-referencing methods evaluated",
        "All four were run on the real per-step distributions; per-step values are in `results/monitoring/referencing_method_comparison.csv`.",
        "",
        "| Method | What is compared | Statistic | Stable 35–41 (mean) | Drift 42–49 (mean) |",
        "| :--- | :--- | :--- | ---: | ---: |",
        "| **1. Differenced** | step-over-step delta of the step median vs the 1–34 delta distribution | abs z of delta | {m1_st:.2f} | {m1_dr:.2f} |".format(m1_st=m1_st, m1_dr=m1_dr),
        "| **2. Rolling window (N=10)** | raw step vs the pooled preceding 10 steps | PSI | {m2_st:.2f} | {m2_dr:.2f} |".format(m2_st=m2_st, m2_dr=m2_dr),
        "| **3. Detrended residuals** | residual `value - trend(step)` vs the 1–34 residual pool | PSI | {m3_st:.2f} | {m3_dr:.2f} |".format(m3_st=m3_st, m3_dr=m3_dr),
        "| **4. Percentile shape** | within-step range-normalised quantile signature vs the 1–34 shape pool | PSI | {m4_st:.2f} | {m4_dr:.2f} |".format(m4_st=m4_st, m4_dr=m4_dr),
        "",
        "### 8.3. Per-step signals (mean across the three triad features)",
        "| Step | M1 abs-z(delta) | M1 delta PSI | M2 rolling PSI | M3 resid PSI | M4 shape PSI |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
        f"{method_table_str}",
        "",
        "### 8.4. Finding: none of the four methods produces a 42–43 rise on the triad",
        "- **M2 (rolling raw window) does not fix the false alarm** (~6.4–7.5 everywhere): a monotone ramp is always in the upper tail of any *trailing* window, so re-centering over 10 steps does not remove the secular trend.",
        "- **M4 (percentile shape) does not separate** (~2.0–3.2 in both windows): the within-step relative structure is stable, as expected for a deterministic ramp.",
        "- **M3 (detrended residuals)** removes the trend — the residual-location score is 0.004–0.113 at every step — but shows **no rise** in 42–49; its residual *PSI* is if anything lower in the drift window (0.47 vs 0.78) because step-level spread fluctuates randomly.",
        "- **M1 (differenced/rate-of-change)** shows only a **transient** blip at exactly step 42 (abs-z 3.04) that reverses at step 43 (abs-z 2.83 in the opposite direction) and decays to ~0.3 by step 44; it is not a sustained regime signal.",
        "- **Conclusion (reported explicitly, not forced):** the triad features carry *no* regime-shift information at 42–43. The PR-AUC collapse is real, but it is not encoded in these three monotone features; it is a prediction/prevalence and (noisy) local-tier phenomenon. No re-referencing method can extract a signal that is not there.",
        "",
        "### 8.5. Selected production method",
        "- **Method 3 (detrended residuals) is selected**, because it is the only method that (a) cleanly removes the secular trend, (b) is fully interpretable ('deviation from the learned growth trend'), and (c) generalises to any monotone feature (linear or log-linear trend, chosen by fit RMSE on the reference).",
        "- Implemented in `src/monitoring/feature_drift.py`: during `fit`, any feature with |Spearman rho(step, per-step median)| >= 0.99 over the reference is flagged monotone, a trend is fitted on the reference per-step median, and the stored reference becomes the residual distribution. The thresholded statistic is the standardized residual-location shift `|mean(resid_target) - mean(resid_ref)| / std(resid_ref)`; the residual PSI is still reported for transparency but does not raise alarms. `src/monitoring/retraining_trigger.py` thresholds `drift_score` instead of raw `psi`.",
        "- A regression test (`test_monotone_feature_detrending_avoids_permanent_false_alarm`) pins this behaviour: an on-trend continuation has raw static PSI > 0.25 but `drift_score` < 0.25 (STABLE), while a genuine level departure is flagged SIGNIFICANT.",
        "",
        "### 8.6. Adversarial channel: diagnosis and fix (Fix 1)",
        "The adversarial classifier was re-run on the cadence steps with the triad excluded from its inputs (seed 42; the triad-excluded column also shows the range over seeds 1/7/2024):",
        "",
        "| Step | AUC (all features) | AUC (triad excluded) |",
        "| :--- | ---: | ---: |",
        f"{adv_auc_table}",
        "",
        "Excluding the triad drops step-35 AUC from **0.9928 to 0.9002** (seed range 0.863–0.900), confirming the monotone aggregate features were the dominant deterministic time proxy. Because the residual AUC stays ~0.90 (> 0.9), this is the *separate residual issue* branch: permutation attribution on the local-only step-35 classifier gives:",
        "",
        "| Feature | Permutation importance (ROC-AUC) |",
        "| :--- | ---: |",
        f"{adv_attr_table}",
        "",
        "`Local_feature_2` and `Local_feature_3` are the two most trend-like *local* features (reference |rho| = 0.79 / 0.80 over steps 1–34), so the residual always-on component is a milder instance of the same time-proxy issue plus genuine step-specific covariate differences. Since the channel is still not regime-discriminative (step 35 >= step 45) and the cause is now attributed, the adversarial `ALERT` remains a corroborating warning and cannot solo-trigger `CRITICAL`. The permanent fix — auto-excluding `|rho| >= 0.99` monotone features from the classifier — is implemented in `src/monitoring/adversarial_drift.py`.",
        "",
        "### 8.7. Trigger decisions (all 15 steps; lag-safe, see Section 10)",
        "| Step | Triad drift_score 10/43/8 | Triad level | Local drift % | Prevalence | Same-step F1 (hindsight) | Trigger decision | Primary reason |",
        "| :--- | :--- | :--- | ---: | :--- | ---: | :--- | :--- |",
        f"{corrected_decisions_str}",
        "",
        "### 8.8. Success-criterion assessment",
        f"- Pre-drift (35–42): `CRITICAL` at {_fmt_steps(crit_pre)}. " + ("✅" if not crit_pre else "❌ false alarm(s) — investigate."),
        f"- Drift onset (step 43): {step43_text}",
        f"- Drift window: `CRITICAL` at {_fmt_steps(crit_drift)}. {first_fire_text}",
        "",
        "---",
        "",
        "## 9. Changelog — Fixes 1 & 2",
        "",
        "### 9.1. Fix 1 — adversarial channel (step-35 false alarm)",
        "- **Diagnosis.** With all features, adversarial AUC is 0.9928 / 0.9964 / 0.9990 / 0.9998 at steps 35 / 40 / 45 / 49 — saturated and non-discriminative. Excluding `Aggregate_feature_10/43/8` drops step 35 to **0.9002** (0.863–0.900 over seeds), so the triad was the dominant deterministic time proxy.",
        "- **Residual attributed.** The remaining ~0.90 separability is dominated by `Local_feature_2` (+0.275 permutation importance) and `Local_feature_3` (+0.110), the two most trend-like local features (reference |rho| 0.79/0.80), plus step covariance.",
        "- **Fix.** `AdversarialDriftMonitor` now auto-excludes features with `|Spearman rho(step, per-step median)| >= 0.99` (the same rule as `FeatureDriftMonitor`); in practice this removes the triad. Because the residual channel is still non-discriminative with an attributed cause, its `ALERT` remains corroborating-only. Artifacts: `results/monitoring/adversarial_auc_comparison.csv`, `results/monitoring/adversarial_step35_attribution.csv`.",
        "",
        "### 9.2. Fix 2 — local-tier temporal persistence (superseded)",
        "- Fix 2 added a 2-consecutive-step persistence rule so the step-37 spike (34.4% -> 4.3%) gave a WARNING. That rule still exists but only changes the wording of the warning: the whole channel is now WARNING-only (Section 10), so two consecutive exceedances no longer escalate to `CRITICAL`.",
        "",
        "---",
        "",
        "## 10. Lag-Safe Monitoring Pass",
        "",
        "### 10.1. What changed and why",
        f"- **Label delay is a config value**: `label_delay_steps` (default 1; this run L={L}). A decision at step t reads labels of steps <= t-L only. The engine raises `LagViolationError` if a label-based input contains a newer step, so the rule is enforced at the boundary, not just by convention.",
        "- **PerformanceCrash is lag-safe.** It previously read step t's own F1. It now reads per-step F1 of the deployed model on already-labelled steps (<= t-L), over the last 2 labelled steps with >= 10 illicit labels. **Each step counts once** (step-weighted, not pooled by rows) and the channel is breached when the **worst** of those steps is below the floor, so a large healthy step (e.g. step 42, 239 positives) cannot hide a crash at the next (step 43, 24 positives).",
        f"- **F1 floor is not taken from steps 43–49.** floor = {FRAC:.2f} x validation F1 = {FRAC:.2f} x {VF:.4f} = **{FLOOR:.4f}**. The validation F1 is the pooled steps 25–34 F1 of the deployed operating point (tau = 0.435) read from `{backtester.metrics_path}`. **Caveat:** only the pooled validation F1 is stored (per-step validation predictions of the 1–24 fit were not saved and cannot be regenerated without refitting), so the rule uses the pooled value, not the median of a per-step distribution. The fraction 0.5 was fixed before the run.",
        f"- {floor_robust_text}",
        "- **Score-shift channel added (label-free).** PSI of the deployed model's step-t score distribution against its scores on the last 5 already-labelled steps (steps <= t-L, each step weighted equally; at least 2 needed, else UNAVAILABLE and never fires). Fixed probability bins (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90). Standard bands: **PSI >= 0.25 is CRITICAL**, 0.10-0.25 is WARNING. Neither band was tuned on 43–49. It reads no label of step t, so it can fire during label delay.",
        "- **Feature-shift channel demoted to WARNING only.** The % of local features with PSI > 0.25 can no longer raise `CRITICAL`.",
        "- **Status output**: `/monitoring/status` gains `label_delay_steps`, `label_delay_assumption` and `channel_breakdown` (new fields only).",
        "",
        "### 10.2. Lag-safe decision table (steps 35–49)",
        "Window F1 = worst per-step F1 over the labelled steps listed (labels <= t-L). Score PSI is against labelled reference steps. Channel statuses: CRITICAL / WARNING / OK / UNAVAILABLE.",
        "",
        "| Step | Labelled steps read (perf) | Worst per-step F1 | Performance | Score PSI | Score shift | Prevalence | Feature shift (warn-only) | CRITICAL channels | Decision |",
        "| :--- | :--- | ---: | :--- | ---: | :--- | :--- | :--- | :--- | :--- |",
        f"{lag_table_str}",
        "",
        "### 10.3. Which channel fires first",
        f"- {step43_text}",
        f"- {first_fire_text}",
        f"- {perf_text}",
        f"- {prev_text}",
        f"- {order_text}",
        "",
        "### 10.4. Limitations",
        "- Score PSI is computed on labelled transactions only here (`predictions.csv` holds scores for labelled rows); in production it would use every scored transaction.",
        "- The score reference rolls over the last 5 labelled steps, so a *sustained* shift eventually becomes the reference and the channel goes quiet; it is an onset detector, not a persistent-degradation detector. The performance and prevalence channels cover persistence.",
        "- The prevalence collapse floor (3.5%) and the 30% local-feature threshold are pre-existing constants, unchanged in this pass.",
        "- Same-step F1, adaptive and oracle thresholds in Sections 1, 2 and 4 are hindsight diagnostics. They use step t's labels and are not inputs to any decision.",
    ]

    return "\n".join(report_lines)


def main():
    t0 = time.time()
    logger.info("Initializing Retrospective Backtest Runner for BitcoinGraphGuard Phase 8a...")

    backtester = MonitoringBacktester(
        features_path="Og data/txs_features.csv",
        predictions_path="results/xgboost/predictions.csv",
        window_size=5,
        label_delay_steps=1,
        frozen_threshold=0.435,
        baseline_prevalence=0.115809,
    )

    records = backtester.run_backtest()
    tables = backtester.export_summary_tables(output_dir="results/monitoring")

    elapsed = time.time() - t0
    logger.info("Generating reports...")

    # Markdown report
    md_content = format_markdown_report(backtester, tables, elapsed)
    with open("reports/monitoring_backtest_report.md", "w") as f:
        f.write(md_content)

    # JSON report
    json_payload = {
        "metadata": {
            "phase": "Phase 8a - Drift Monitoring & Retraining Triggers",
            "model": "XGBoost Optimized (Frozen, Phase 2)",
            "test_window": "35-49",
            "timestamp": "2026-10-01",
            "execution_time_seconds": round(elapsed, 2),
        },
        "simulation_steps": [r.to_dict() for r in records],
    }
    with open("reports/monitoring_backtest_report.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    logger.info("Backtest complete! Reports saved to reports/monitoring_backtest_report.md and .json")


if __name__ == "__main__":
    main()
