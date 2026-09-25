#!/usr/bin/env python3
"""Render Phase 2 EDA figures from ``reports/eda/`` aggregates.

This script never touches the raw dataset: it reads only the aggregated JSON and
CSV tables produced by :mod:`scripts.eda_phase2`. Use the non-interactive Agg
backend so it is safe to run headless on the laptop.

Usage::

    MPLCONFIGDIR=/tmp/mplconfig ./bit/bin/python scripts/plot_phase2_eda.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = REPO_ROOT / "reports" / "eda"
TABLE_DIR = EDA_DIR / "tables"
PLOT_DIR = EDA_DIR / "plots"


def plot_temporal(frame: pd.DataFrame, unit: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(frame["time_step"], frame["total"], color="#8ecae6", label="total")
    ax1.bar(frame["time_step"], frame["labeled"], color="#219ebc", label="labeled (1/2)")
    ax1.set_ylabel("count")
    ax1.set_title(f"{unit} activity by time step")
    ax1.legend(loc="upper right")
    ax2.plot(frame["time_step"], frame["illicit_pct_of_labeled"], marker="o", color="#d62828")
    ax2.set_xlabel("time step (1-49)")
    ax2.set_ylabel("illicit % of labeled")
    ax2.set_title("illicit share among labeled samples")
    ax2.set_xticks(np.arange(1, 50, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_class_balance(labels: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, block) in zip(axes, labels.items()):
        counts = block["class_counts"]
        keys = ["1", "2", "3"]
        values = [counts.get(k, 0) for k in keys]
        ax.bar(["illicit", "licit", "unknown"], values, color=["#d62828", "#2a9d8f", "#adb5bd"])
        ax.set_yscale("log")
        ax.set_title(f"{name} class counts (log scale)")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_degree(grid: dict, title: str, path: Path, sample_key: str = "out_degree") -> None:
    fig, axes = plt.subplots(1, len(grid), figsize=(4.5 * len(grid), 4.5))
    if len(grid) == 1:
        axes = [axes]
    for ax, (name, block) in zip(axes, grid.items()):
        histogram = block[sample_key]["degree_histogram"]
        degrees = np.array([int(k) for k in histogram])
        counts = np.array([histogram[k] for k in histogram], dtype=float)
        positive = degrees > 0
        ax.scatter(degrees[positive], counts[positive], s=6, alpha=0.6, color="#023047")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name.replace("_edgelist", ""))
        ax.set_xlabel("out-degree")
        ax.set_ylabel("nodes")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_correlation(name: str, path: Path, top: int = 20) -> None:
    frame = pd.read_csv(TABLE_DIR / f"{name}_feature_summary.csv")
    frame = frame.dropna(subset=["corr_with_illicit"])
    frame["abs_corr"] = frame["corr_with_illicit"].abs()
    frame = frame.sort_values("abs_corr", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(frame["corr_with_illicit"] >= 0, "#2a9d8f", "#e76f51")
    ax.barh(frame["feature"], frame["corr_with_illicit"], color=colors)
    ax.set_xlabel("point-biserial correlation with illicit (class 1 vs 2)")
    ax.set_title(f"Top {top} {name} features by |correlation with illicit|")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_split(split: dict, path: Path) -> None:
    metrics = split["candidate_windows"]
    units = list(metrics)
    fig, axes = plt.subplots(1, len(units), figsize=(6 * len(units), 4.5))
    if len(units) == 1:
        axes = [axes]
    for ax, unit in zip(axes, units):
        windows = metrics[unit]
        names = list(windows)
        labeled = [windows[n]["labeled"] for n in names]
        illicit = [windows[n]["illicit"] for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, labeled, width=0.4, label="labeled", color="#219ebc")
        ax.bar(x + 0.2, illicit, width=0.4, label="illicit", color="#d62828")
        pct = [windows[n]["illicit_pct_of_labeled"] for n in names]
        for xi, (v, p) in enumerate(zip(illicit, pct)):
            ax.text(xi + 0.2, v, f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{unit} label availability by candidate window")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=EDA_DIR / "phase2_eda.json")
    args = parser.parse_args(argv)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.json.read_text(encoding="utf-8"))

    plot_temporal(pd.read_csv(TABLE_DIR / "txs_temporal_by_step.csv"), "Transactions", PLOT_DIR / "txs_temporal.png")
    plot_temporal(pd.read_csv(TABLE_DIR / "wallets_temporal_by_step.csv"), "Wallet snapshots", PLOT_DIR / "wallets_temporal.png")
    plot_class_balance(data["labels"], PLOT_DIR / "class_balance.png")
    plot_degree(
        {k: data["graph"][k] for k in ("txs_edgelist", "AddrTx_edgelist", "TxAddr_edgelist", "AddrAddr_edgelist")},
        "Out-degree distribution (log-log)",
        PLOT_DIR / "degree_distributions.png",
    )
    plot_feature_correlation("txs", PLOT_DIR / "txs_feature_correlation_with_illicit.png")
    plot_feature_correlation("wallets", PLOT_DIR / "wallets_feature_correlation_with_illicit.png")
    plot_split(data["split_investigation"], PLOT_DIR / "split_windows.png")
    print("Plots written to", PLOT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
