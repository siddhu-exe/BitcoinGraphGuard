"""Generate the portfolio-site card images for BitcoinGraphGuard.

Writes to portfolio/site-card/:
  01_label_delay_recovery.png   1600x900
  02_before_after_step43.png    1600x900
  03_system_architecture.png    1600x900
  thumbnail.png                 1200x630

Every number is read from saved artifacts:
  results/walkforward/lag_curve.csv     (label-delay chart, thumbnail)
  results/hgt/temporal_metrics.csv      (four-model windows, thumbnail)
Architecture counts (203,769 / 822,942 / ~4.4M) come from docs/EDA.md.

Run with: ./bit/bin/python scripts/generate_site_card_images.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "portfolio" / "site-card"
OUT.mkdir(parents=True, exist_ok=True)

BG, INK, MUTED, GRID = "#F7F6F2", "#14161A", "#6B6F76", "#E3E1DA"
BLUE, RED, PALE = "#1F3FBF", "#B4371F", "#C9CCD3"

for path in Path.home().glob(".local/share/fonts/**/Inter-*.ttf"):
    font_manager.fontManager.addfont(str(path))
FAMILY = (
    "Inter"
    if any(f.name == "Inter" for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)
plt.rcParams.update(
    {
        "font.family": FAMILY,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def new_fig(w: int, h: int):
    return plt.figure(figsize=(w / 100, h / 100), dpi=100)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=100)  # no bbox_inches: keeps exact pixel size
    plt.close(fig)
    print("wrote", OUT / name)


def header(fig, main: str, sub: str) -> None:
    fig.text(0.055, 0.935, main, fontsize=30, fontweight="bold", va="top")
    fig.text(0.055, 0.855, sub, fontsize=15, color=MUTED, va="top")


def style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=14)


def label_delay() -> None:
    d = pd.read_csv(ROOT / "results/walkforward/lag_curve.csv").sort_values("lag")
    fig = new_fig(1600, 900)
    header(
        fig,
        "Retraining recovers only if labels arrive fast",
        "Pooled PR-AUC on steps 43-49 by label delay L. Whiskers: 95% cluster-bootstrap CI "
        "over steps (7 steps, 169 positives).",
    )
    ax = fig.add_axes([0.075, 0.12, 0.89, 0.65])
    style(ax)
    xs = range(len(d))
    w = 0.34
    ax.bar([x - w / 2 for x in xs], d.static_pr_auc, w, color=PALE, label="Frozen model")
    ax.bar([x + w / 2 for x in xs], d.expanding_pr_auc, w, color=BLUE, label="Expanding retrain")
    yerr = [d.expanding_pr_auc - d.expanding_ci_lo, d.expanding_ci_hi - d.expanding_pr_auc]
    ax.errorbar([x + w / 2 for x in xs], d.expanding_pr_auc, yerr=yerr, fmt="none",
                ecolor=INK, elinewidth=1.6, capsize=6)
    for x, v in zip(xs, d.expanding_pr_auc):
        ax.text(x + w / 2 + 0.05, v + 0.015, f"{v:.2f}", fontsize=17, fontweight="bold",
                color=BLUE, ha="left", va="bottom")
    for x, v in zip(xs, d.static_pr_auc):
        ax.text(x - w / 2, v + 0.015, f"{v:.2f}", fontsize=13, color=MUTED, ha="center")
    ax.axhline(0.5, color=MUTED, lw=1.4, ls=(0, (4, 4)))
    ax.text(len(d) - 0.55, 0.512, "restored = 0.50", color=MUTED, fontsize=13, ha="right")
    ax.set_xticks(list(xs), [f"L = {int(v)}" for v in d.lag])
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC, steps 43-49", fontsize=14)
    ax.legend(frameon=False, fontsize=14, loc="upper right", bbox_to_anchor=(1, 0.95))
    save(fig, "01_label_delay_recovery.png")


def models_table() -> pd.DataFrame:
    t = pd.read_csv(ROOT / "results/hgt/temporal_metrics.csv").set_index("steps")
    early, late = t.loc["35-42"], t.loc["43-49"]
    cols = [
        ("XGBoost", "xgboost_pr_auc"),
        ("GraphSAGE", "graphsage_base_pr_auc"),
        ("HeteroRGCN", "rgcn_pr_auc"),
        ("HGT", "hgt_pr_auc"),
    ]
    return pd.DataFrame(
        {"model": [n for n, _ in cols],
         "before": [float(early[c]) for _, c in cols],
         "after": [float(late[c]) for _, c in cols]}
    )


def before_after() -> None:
    m = models_table()
    fig = new_fig(1600, 900)
    header(
        fig,
        "All four models collapse after step 43",
        "Test PR-AUC on steps 35-42 versus 43-49; illicit prevalence falls from 9.16% to 2.53%.",
    )
    ax = fig.add_axes([0.075, 0.12, 0.89, 0.65])
    style(ax)
    xs = range(len(m))
    w = 0.34
    ax.bar([x - w / 2 for x in xs], m.before, w, color=BLUE, label="Steps 35-42")
    ax.bar([x + w / 2 for x in xs], m.after, w, color=RED, label="Steps 43-49 (drift)")
    for x, b, a in zip(xs, m.before, m.after):
        ax.text(x - w / 2, b + 0.015, f"{b:.2f}", fontsize=16, fontweight="bold",
                color=BLUE, ha="center")
        ax.text(x + w / 2, a + 0.015, f"{a:.2f}", fontsize=16, fontweight="bold",
                color=RED, ha="center")
    ax.set_xticks(list(xs), list(m.model))
    ax.tick_params(axis="x", labelsize=16, labelcolor=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC", fontsize=14)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    save(fig, "02_before_after_step43.png")


def architecture() -> None:
    fig = new_fig(1600, 900)
    header(
        fig,
        "From Elliptic++ to a monitored fraud API",
        "Heavy ML runs on Colab/Kaggle; committed artifacts feed a tabular-only service and a "
        "lag-safe monitor.",
    )
    ax = fig.add_axes([0, 0, 1, 0.8])
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 720)
    ax.axis("off")

    def box(x, y, w, h, head, lines, accent):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                    fc="white", ec=GRID, lw=2))
        ax.add_patch(FancyBboxPatch((x, y + h - 8), w, 8, boxstyle="square,pad=0",
                                    fc=accent, ec=accent))
        ax.text(x + 22, y + h - 40, head, fontsize=17, fontweight="bold", va="center")
        for i, line in enumerate(lines):
            ax.text(x + 22, y + h - 82 - 30 * i, line, fontsize=13.5, color=MUTED, va="center")

    def arrow(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 2})

    w, h, y = 250, 250, 380
    xs = [88 + i * 294 for i in range(5)]
    box(xs[0], y, w, h, "Elliptic++ data",
        ["203,769 transactions", "822,942 wallets", "~4.4M edges, 4 types", "49 time steps"], MUTED)
    box(xs[1], y, w, h, "Four models",
        ["XGBoost", "GraphSAGE", "HeteroRGCN", "HGT"], BLUE)
    box(xs[2], y, w, h, "Lag-safe monitor",
        ["Score shift (label-free)", "Performance, prevalence", "Labels <= t - L only"], RED)
    box(xs[3], y, w, h, "FastAPI service",
        ["Frozen XGBoost model", "confidence_context", "Docker + CI gates"], BLUE)
    box(xs[4], y, w, h, "Dashboard",
        ["React + TypeScript", "Live on Vercel", "Reads API + artifacts"], BLUE)
    for i in range(4):
        arrow(xs[i] + w + 8, xs[i + 1] - 8, y + h / 2)

    ax.text(88, 270, "Protocol", fontsize=16, fontweight="bold")
    ax.text(88, 225, "Temporal splits only: fit steps 1-24, validate 25-34, refit 1-34, test 35-49. "
            "Class 3 (unknown) is never used as supervision.", fontsize=14, color=MUTED)
    ax.text(88, 170, "Walk-forward retraining feeds back into the monitor: it recovers performance "
            "only when labels arrive within a step or two.", fontsize=14, color=MUTED)
    save(fig, "03_system_architecture.png")


def thumbnail() -> None:
    lag = pd.read_csv(ROOT / "results/walkforward/lag_curve.csv").sort_values("lag")
    m = models_table()
    xgb = m.set_index("model").loc["XGBoost"]
    full = pd.read_csv(ROOT / "results/hgt/temporal_metrics.csv").set_index("steps").loc["35-49"]

    fig = new_fig(1200, 630)
    fig.text(0.05, 0.90, "BitcoinGraphGuard", fontsize=40, fontweight="bold", va="top")
    fig.text(0.05, 0.765, "A Bitcoin fraud detector that knows when to stop trusting itself",
             fontsize=18, color=MUTED, va="top")
    tiles = [
        (BLUE, f"{float(full['xgboost_pr_auc']):.2f} vs {float(full['graphsage_base_pr_auc']):.2f}",
         "XGBoost beat the best graph model (PR-AUC, steps 35-49)"),
        (RED, f"{xgb.before:.2f} → {xgb.after:.2f}",
         "XGBoost PR-AUC before and after step 43; all four models collapse"),
        (BLUE, "  \u00b7  ".join(f"{v:.2f}" for v in lag.expanding_pr_auc),
         "Retraining PR-AUC at label delay L = " + ", ".join(str(int(v)) for v in lag.lag)),
    ]
    tw, th, gap, x0, y0 = 350, 250, 20, 0.05 * 1200, 70
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")
    ax.patch.set_alpha(0)
    for i, (col, big, small) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        ax.add_patch(FancyBboxPatch((x, y0), tw, th, boxstyle="round,pad=0,rounding_size=14",
                                    fc="white", ec=GRID, lw=2))
        ax.add_patch(FancyBboxPatch((x, y0 + th - 8), tw, 8, boxstyle="square,pad=0",
                                    fc=col, ec=col))
        ax.text(x + 22, y0 + th - 85, big, fontsize=29 if i < 2 else 17.5, fontweight="bold",
                color=col, va="center")
        ax.text(x + 22, y0 + th - 130, _wrap(small, 30), fontsize=14, color=MUTED, va="top",
                linespacing=1.5)
    ax.text(1200 * 0.05, 32, "Real Elliptic++ data · temporal splits · lag-safe monitor "
            "· live dashboard", fontsize=13, color=MUTED, va="center")
    save(fig, "thumbnail.png")


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    lines.append(cur)
    return "\n".join(lines)


if __name__ == "__main__":
    label_delay()
    before_after()
    architecture()
    thumbnail()
