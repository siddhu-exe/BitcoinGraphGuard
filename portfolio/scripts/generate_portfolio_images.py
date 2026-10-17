#!/usr/bin/env python3
"""Generate dark-themed portfolio charts for BitcoinGraphGuard.

Every number is read from saved artifacts in results/ or copied from README.md
(which cites results/*/metrics.json). Run from anywhere with ./bit/bin/python.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(exist_ok=True)

BG, CARD, EDGE = "#0B0E14", "#161A22", "#2A2E37"
TXT, MUTED = "#F3F4F6", "#9CA3AF"
AMBER, MINT, STEEL, RED = "#F7931A", "#2DD4BF", "#7B96B2", "#EF4444"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": TXT, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": EDGE,
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=EDGE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def title(fig, main: str, sub: str) -> None:
    fig.text(0.07, 0.97, main, fontsize=17, fontweight="bold", va="top")
    fig.text(0.07, 0.915, sub, fontsize=10.5, color=MUTED, va="top")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.text(0.4, 8.0, "BitcoinGraphGuard: from Elliptic++ to a monitored fraud API",
            fontsize=18, fontweight="bold")
    ax.text(0.4, 7.55, "Heavy ML runs on Colab/Kaggle; committed artifacts feed a "
            "tabular-only service and a lag-safe monitor", fontsize=11, color=MUTED)

    def box(x, y, w, h, head, lines, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=CARD, ec=col, lw=1.8))
        ax.text(x + w / 2, y + h - 0.35, head, ha="center", fontsize=11.5,
                fontweight="bold", color=col)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.85 - i * 0.37, ln, ha="center", fontsize=9, color=TXT)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))

    box(0.4, 4.3, 3.1, 2.4, "Elliptic++ (raw)", ["203,769 transactions", "822,942 wallets",
        "4 edge types, 49 steps", "read-only, gitignored"], STEEL)
    box(4.2, 4.3, 3.6, 2.4, "Colab notebooks", ["XGBoost, GraphSAGE", "HeteroRGCN, HGT",
        "SHAP / Captum / ablation", "drift diagnosis, walk-forward"], AMBER)
    box(8.5, 4.3, 3.3, 2.4, "Saved artifacts", ["results/ and reports/", "frozen XGBoost model",
        "every quoted metric", "traces to a file here"], MINT)
    box(12.5, 4.3, 3.1, 2.4, "FastAPI service", ["/predict, /batch_predict", "/health, /monitoring/status",
        "tau* = 0.435", "confidence_context per step"], AMBER)
    for a, b in ((3.5, 4.2), (7.8, 8.5), (11.8, 12.5)):
        arrow(a, 5.5, b, 5.5)

    box(8.5, 0.6, 3.3, 2.6, "Lag-safe monitor", ["reads labels <= t - L only", "score PSI (label-free)",
        "performance + prevalence", "LagViolationError enforces"], RED)
    box(12.5, 0.6, 3.1, 2.6, "React dashboard", ["Overview, Monitoring", "Retraining, Try it",
        "Vite + TypeScript", "Vercel / Render"], STEEL)
    arrow(10.15, 4.3, 10.15, 3.2)
    arrow(11.8, 1.9, 12.5, 1.9)
    arrow(14.05, 4.3, 14.05, 3.2)
    ax.text(0.4, 2.4, "Serving deliberately excludes PyTorch, PyG and MLflow:\nonly the tabular "
            "XGBoost model is served.", fontsize=10.5, color=MUTED, va="center")
    save(fig, "01_system_architecture")


def model_comparison() -> None:
    models = ["XGBoost (served)", "GraphSAGE", "HGT", "2-layer MLP", "HeteroRGCN",
              "Logistic reg.", "Prevalence"]
    pr = [0.8013, 0.6209, 0.4861, 0.4768, 0.4682, 0.2917, 0.0650]
    cols = [AMBER] + [STEEL] * 5 + [EDGE]
    fig, ax = plt.subplots(figsize=(12, 6))
    style(ax)
    bars = ax.barh(models[::-1], pr[::-1], color=cols[::-1], height=0.62)
    for b, v in zip(bars, pr[::-1]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=10.5)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("Test PR-AUC, steps 35-49 (pooled)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=EDGE, lw=0.6)
    fig.subplots_adjust(top=0.82)
    title(fig, "A tuned XGBoost beats every graph model tried",
          "Same frozen temporal protocol: fit 1-24, validate 25-34, test 35-49. "
          "Graph models were tuned less deeply.")
    save(fig, "02_model_comparison")


def regime_shift() -> None:
    d = pd.read_csv(ROOT / "results/temporal_inductive/per_step_metrics.csv")
    d = d[d.model == "XGBoost (frozen)"]
    fig, ax = plt.subplots(figsize=(12.5, 6))
    style(ax)
    ax.axvspan(42.5, 49.5, color=RED, alpha=0.10)
    ax.axvline(42.5, color=RED, ls="--", lw=1.2)
    ax.plot(d.time_step, d.pr_auc, color=AMBER, lw=2.4, marker="o", label="PR-AUC (per step)")
    ax.plot(d.time_step, d.prevalence, color=MINT, lw=1.8, marker="s", ms=4,
            label="Illicit prevalence (per step)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time step")
    ax.text(43, 0.62, "Drift window\n43-49", color=RED, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45))
    fig.subplots_adjust(top=0.82)
    title(fig, "XGBoost collapses at step 43",
          "Frozen model at tau* = 0.435. Pooled PR-AUC 0.9215 (35-42) vs 0.0427 (43-49). "
          "Late steps hold few illicit labels, so per-step values are noisy.")
    save(fig, "03_regime_shift_per_step")


def retraining_lag() -> None:
    d = pd.read_csv(ROOT / "results/walkforward/lag_curve.csv")
    x = range(len(d))
    fig, ax = plt.subplots(figsize=(11, 6))
    style(ax)
    w = 0.36
    ax.bar([i - w / 2 for i in x], d.static_pr_auc, w, color=EDGE, label="Static (frozen)")
    ax.bar([i + w / 2 for i in x], d.expanding_pr_auc, w, color=AMBER, label="Expanding retrain")
    err = [d.expanding_pr_auc - d.expanding_ci_lo, d.expanding_ci_hi - d.expanding_pr_auc]
    ax.errorbar([i + w / 2 for i in x], d.expanding_pr_auc, yerr=err, fmt="none",
                ecolor=TXT, capsize=4, lw=1.2)
    for i, v in zip(x, d.expanding_pr_auc):
        ax.text(i + w / 2 + 0.05, v + 0.03, f"{v:.2f}", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"L = {int(v)}" for v in d.lag])
    ax.axhline(0.50, color=MINT, ls=":", lw=1.2)
    ax.text(3.45, 0.51, "restored = 0.50", color=MINT, fontsize=9, ha="right")
    ax.set_ylabel("PR-AUC, steps 43-49 (pooled)")
    ax.legend(frameon=False)
    fig.subplots_adjust(top=0.82)
    title(fig, "Retraining helps only if labels arrive fast",
          "Walk-forward under label delay L. Bars: pooled PR-AUC; whiskers: 95% cluster-bootstrap "
          "CI over steps (wide: 7 steps, 169 positives).")
    save(fig, "04_retraining_vs_label_delay")


def shap_top() -> None:
    s = pd.read_csv(ROOT / "results/explainability/xgb_shap_global.csv").head(15)[::-1]
    cols = [AMBER if f.startswith("Local") else STEEL for f in s.feature]
    fig, ax = plt.subplots(figsize=(11, 7))
    style(ax)
    ax.barh(s.feature, s.mean_abs_shap, color=cols, height=0.65)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=EDGE, lw=0.6)
    ax.set_xlabel("Mean |SHAP|")
    ax.barh([], [])
    ax.text(0.98, 0.06, "amber = Local_feature_*\nsteel = Aggregate_feature_*", transform=ax.transAxes,
            ha="right", fontsize=10, color=MUTED)
    fig.subplots_adjust(top=0.86)
    title(fig, "What the served model relies on",
          "Top 15 of 165 features by mean |SHAP|; Local_feature_53 dominates.")
    save(fig, "05_xgboost_shap_top_features")


def signals() -> None:
    d = pd.read_csv(ROOT / "results/monitoring/signal_evaluation.csv")
    d = d[d.detected_43_49].sort_values("detection_lead_vs_43")
    d["fa"] = d.false_alarm_rate_35_42 * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    style(ax)
    cols = [MINT if f < 20 else RED for f in d.fa]
    ax.scatter(d.detection_lead_vs_43, d.fa, s=260, c=cols, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r.signal, (r.detection_lead_vs_43, r.fa), xytext=(9, 9),
                    textcoords="offset points", fontsize=9.5)
    ax.set_xlabel("Detection lead vs step 43 (steps late; 0 = on time)")
    ax.set_ylabel("False-alarm rate on healthy steps 35-42 (%)")
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-8, 110)
    fig.subplots_adjust(top=0.82)
    title(fig, "Which monitoring signals were worth listening to",
          "Low and left is best. Signals that never fired in 43-49 are omitted; "
          "retraining trigger and prevalence alarm too often to act on alone.")
    save(fig, "06_monitoring_signal_tradeoff")


if __name__ == "__main__":
    architecture()
    model_comparison()
    regime_shift()
    retraining_lag()
    shap_top()
    signals()
