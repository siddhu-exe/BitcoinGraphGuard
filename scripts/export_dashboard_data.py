#!/usr/bin/env python3
"""Export static JSON for the BitcoinGraphGuard dashboard (frontend/public/data).

Read-only over ``results/`` and ``reports/``: nothing in those directories is
modified. Every number written is copied or arithmetically derived from a saved
artifact; the provenance of each block is recorded under ``sources``.

Standard library only and streaming for the large raw CSV, so it is laptop-safe.

Usage:
    python scripts/export_dashboard_data.py
    python scripts/export_dashboard_data.py --out frontend/public/data
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
WALKFORWARD = RESULTS / "walkforward"
RAW_FEATURES = ROOT / "Og data" / "txs_features.csv"
TEST_FIXTURE = ROOT / "tests" / "test_api.py"
FIXTURE_NAME = "SAMPLE_TX_1813992_FEATURES"

MODEL_FEATURES = [f"Local_feature_{i}" for i in range(1, 94)] + [
    f"Aggregate_feature_{i}" for i in range(1, 73)
]
THRESHOLD = 0.435
DRIFT_START = 43

# Walk-forward export scope: the two policies the dashboard compares, the label delays
# that were run for both, the evaluation windows and the top-K alert budgets.
WF_STRATEGIES = ("static", "expanding")
WF_LAGS = (1, 2, 3, 5)
WF_WINDOWS = {"35-42": range(35, 43), "43-49": range(43, 50)}
WF_KS = (20, 50, 100)
NOISE_MIN_POSITIVES = 10  # steps with fewer illicit labels are flagged noisy (verdict.md)
TOL = 1e-9


def num(value: str | None) -> float | None:
    """Parse a CSV cell to float; empty cells become None (never NaN)."""
    if value is None or value.strip() == "":
        return None
    out = float(value)
    return None if out != out else out


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a whole (small) CSV into a list of dict rows."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_performance() -> dict[str, Any]:
    """Model comparison windows, per-step PR-AUC, prevalence and drift verdict."""
    windows: list[dict[str, Any]] = []

    # XGBoost / GraphSAGE / HeteroRGCN frozen benchmark rows.
    for row in read_csv(RESULTS / "temporal_inductive" / "model_comparison.csv"):
        if row["regime"] != "frozen_benchmark":
            continue
        windows.append(
            {
                "model": row["model"].replace(" (frozen)", ""),
                "window": row["period"].strip(),
                "n": int(row["n"]),
                "illicit": int(row["illicit"]),
                "prevalence": num(row["prevalence"]),
                "pr_auc": num(row["pr_auc"]),
                "roc_auc": num(row["roc_auc"]),
                "f1": num(row["f1"]),
                "precision": num(row["precision"]),
                "recall": num(row["recall"]),
                "threshold": num(row["threshold"]),
            }
        )

    # HGT lives in its own artifact.
    hgt_label = {"35-49 (Primary)": "35-49", "35-42 (Early Test)": "35-42", "43-49 (Recent Drift)": "43-49"}
    for row in read_csv(RESULTS / "hgt" / "temporal_metrics.csv"):
        windows.append(
            {
                "model": "HGT",
                "window": hgt_label[row["window"]],
                "n": int(row["labeled"]),
                "illicit": int(row["illicit"]),
                "prevalence": num(row["prevalence"]),
                "pr_auc": num(row["hgt_pr_auc"]),
                "roc_auc": num(row["hgt_roc_auc"]),
                "f1": num(row["hgt_f1"]),
                "precision": num(row["hgt_precision"]),
                "recall": num(row["hgt_recall"]),
                "threshold": None,
            }
        )

    # Per-step PR-AUC.
    per_step: dict[int, dict[str, Any]] = {}
    key = {"XGBoost (frozen)": "xgboost", "GraphSAGE (frozen)": "graphsage", "HeteroRGCN (frozen)": "rgcn"}
    for row in read_csv(RESULTS / "temporal_inductive" / "per_step_metrics.csv"):
        step = int(row["time_step"])
        entry = per_step.setdefault(step, {"step": step})
        entry[key[row["model"]]] = num(row["pr_auc"])
        entry["prevalence"] = num(row["prevalence"])
        entry["n_labeled"] = int(row["n"])
        entry["illicit"] = int(row["illicit"])
    for row in read_csv(RESULTS / "hgt" / "errors_by_step.csv"):
        per_step[int(row["time_step"])]["hgt"] = num(row["pr_auc"])
    # Flag steps with too few illicit labels for a per-step PR-AUC to mean anything. The flag
    # comes from the walk-forward harness; the label count must agree with this file's own.
    noise: dict[int, bool] = {}
    for row in read_csv(WALKFORWARD / "per_step_metrics.csv"):
        if row["strategy"] != "static" or row["lag"] != "1":
            continue
        step = int(row["step"])
        if int(row["n_illicit"]) != per_step[step]["illicit"]:
            sys.exit(f"step {step}: illicit count differs between temporal_inductive and walkforward")
        noise[step] = row["noise_flag"] != ""
        if noise[step] != (int(row["n_illicit"]) < NOISE_MIN_POSITIVES):
            sys.exit(f"step {step}: noise_flag disagrees with the < {NOISE_MIN_POSITIVES} positives rule")
    for step, entry in per_step.items():
        entry["low_positives"] = noise[step]
    steps = [per_step[s] for s in sorted(per_step)]

    verdict = json.loads((RESULTS / "xgboost_v2" / "drift_verdict.json").read_text(encoding="utf-8"))

    return {
        "windows": windows,
        "steps": steps,
        "drift_diagnosis": {
            "adversarial_auc_train_vs_drift": verdict["adversarial_auc_train_vs_drift"],
            "median_top15_ks": verdict["median_top15_ks"],
            "top15_features": verdict["top15_features"],
            "in_window_cv_pr_auc_drift": verdict["in_window_cv_pr_auc_drift"],
            "transferred_pr_auc_drift": verdict["phase2_transferred_pr_auc_drift"],
            "covariate_drift_material": verdict["covariate_drift_material"],
            "concept_drift_implicated": verdict["concept_drift_implicated"],
            "verdict": verdict["verdict"],
        },
        "sources": [
            "results/temporal_inductive/model_comparison.csv",
            "results/temporal_inductive/per_step_metrics.csv",
            "results/hgt/temporal_metrics.csv",
            "results/hgt/errors_by_step.csv",
            "results/xgboost_v2/drift_verdict.json",
            "results/walkforward/per_step_metrics.csv (low_positives flag)",
        ],
    }


# Channels shown in the lag-safe table, in display order (report keys).
LAG_SAFE_CHANNELS = ("score_shift", "performance", "prevalence", "feature_shift")


def channel_list(channels: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a decision's ``channels`` block (same shape as /monitoring/status channel_breakdown)."""
    return [
        {
            "name": name,
            "kind": c["kind"],
            "status": c["status"],
            "value": c["value"],
            "threshold": c["threshold"],
            "can_trigger_critical": c["can_trigger_critical"],
        }
        for name, c in channels.items()
    ]


def build_lag_safe(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-step lag-safe decision table (report section 10.2) from the backtest JSON."""
    rows: list[dict[str, Any]] = []
    for sim in report["simulation_steps"]:
        d = sim["decision"]
        ch = d["channels"]
        lsp = sim["lag_safe_performance"]
        critical = [n for n, c in ch.items() if c["status"] == "CRITICAL"]
        if (d["action"] == "RETRAIN") != bool(critical):
            sys.exit(f"step {sim['time_step']}: action {d['action']} inconsistent with critical channels {critical}")
        rows.append(
            {
                "step": sim["time_step"],
                "action": d["action"],
                "severity": d["severity"],
                "label_delay_steps": d["label_delay_steps"],
                "perf_steps_read": lsp["steps_used"],
                "worst_f1": lsp["f1_min"] if lsp["available"] else None,
                "score_psi": ch["score_shift"]["value"],
                "channels": {n: {"status": ch[n]["status"], "value": ch[n]["value"]} for n in LAG_SAFE_CHANNELS},
                "critical_channels": critical,
            }
        )
    return rows


def build_monitoring() -> dict[str, Any]:
    """Per-step monitoring metrics plus the latest decision from the backtest report."""
    report = json.loads((REPORTS / "monitoring_backtest_report.json").read_text(encoding="utf-8"))
    report_steps = {int(s["time_step"]): s for s in report["simulation_steps"]}

    rows = read_csv(RESULTS / "monitoring" / "step_monitoring_metrics.csv")
    steps: list[dict[str, Any]] = []
    for row in rows:
        step = int(row["time_step"])
        rep = report_steps.get(step)
        if rep is None:
            sys.exit(f"step {step} in step_monitoring_metrics.csv is missing from the backtest report")
        decision = rep["decision"]
        if decision["action"] != row["trigger_action"]:
            sys.exit(f"step {step}: CSV action {row['trigger_action']} != report action {decision['action']}")
        prev = rep.get("prevalence_drift") or {}
        steps.append(
            {
                "step": step,
                "n_labeled": int(row["n_labeled"]),
                "n_illicit": int(row["n_illicit"]),
                "prevalence": num(row["prevalence"]),
                "rolling_prevalence": num(row["rolling_prevalence"]),
                "prevalence_rel_change_pct": prev.get("relative_change_pct"),
                "local_drift_sig_pct": num(row["pct_local_drift_sig"]),
                "adversarial_auc": num(row["adversarial_auc"]),
                "pr_auc": num(row["pr_auc"]),
                "frozen_f1": num(row["frozen_f1"]),
                "adaptive_f1": num(row["adaptive_f1_f1"]),
                "bayes_f1": num(row["bayes_f1"]),
                "oracle_f1": num(row["oracle_f1"]),
                "action": decision["action"],
                "severity": decision["severity"],
                "primary_reason": decision["primary_reason"],
                "prevalence_status": row["prevalence_status"],
                "triad_status": row["triad_status"],
            }
        )
    steps.sort(key=lambda s: s["step"])

    latest_raw = report["simulation_steps"][-1]["decision"]
    latest = {
        "time_step": latest_raw["time_step"],
        "action": latest_raw["action"],
        "severity": latest_raw["severity"],
        "primary_reason": latest_raw["primary_reason"],
        "reasons": latest_raw["reasons"],
        "component_statuses": latest_raw["component_statuses"],
        "label_delay_steps": latest_raw["label_delay_steps"],
        "label_delay_assumption": latest_raw["label_delay_assumption"],
        "channels": channel_list(latest_raw["channels"]),
    }

    drift_steps = [s for s in steps if s["step"] >= DRIFT_START]

    def mean(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    action_counts: dict[str, int] = {}
    for s in steps:
        action_counts[s["action"]] = action_counts.get(s["action"], 0) + 1

    return {
        "metadata": {
            "phase": report["metadata"]["phase"],
            "model": report["metadata"]["model"],
            "test_window": report["metadata"]["test_window"],
            "generated": report["metadata"]["timestamp"],
        },
        "latest_decision": latest,
        "steps": steps,
        "lag_safe": build_lag_safe(report),
        "summary": {
            "n_steps": len(steps),
            "action_counts": action_counts,
            "drift_window_mean_frozen_f1": mean([s["frozen_f1"] for s in drift_steps]),
            "drift_window_mean_adaptive_f1": mean([s["adaptive_f1"] for s in drift_steps]),
            "drift_window_mean_oracle_f1": mean([s["oracle_f1"] for s in drift_steps]),
            "pre_drift_mean_frozen_f1": mean([s["frozen_f1"] for s in steps if s["step"] < DRIFT_START]),
        },
        "sources": [
            "reports/monitoring_backtest_report.json",
            "results/monitoring/step_monitoring_metrics.csv",
            "reports/monitoring_backtest_report.json (decision.channels, lag_safe_performance)",
        ],
    }


def close(a: float | None, b: float | None) -> bool:
    """True when two optional floats agree to TOL (both None counts as agreeing)."""
    return a is None and b is None or (a is not None and b is not None and abs(a - b) <= TOL)


def build_walkforward() -> dict[str, Any]:
    """Walk-forward retraining results: per-step, pooled (cluster CI), top-K triage and retrain log.

    Only ``static`` and ``expanding`` are exported. The top-K ceiling is recomputed here as
    sum over steps of min(K, positives_in_step) / total_positives; the window-level
    ``max_possible_*`` columns of alert_budget.csv are deliberately never read.
    """
    keep = lambda r: r["strategy"] in WF_STRATEGIES and int(r["lag"]) in WF_LAGS  # noqa: E731

    # --- per-step PR-AUC / validation threshold -------------------------------------------
    per_step: list[dict[str, Any]] = []
    step_info: dict[int, dict[str, Any]] = {}
    for row in filter(keep, read_csv(WALKFORWARD / "per_step_metrics.csv")):
        step = int(row["step"])
        info = {
            "step": step,
            "window": row["window"],
            "n_labeled": int(row["n_labeled"]),
            "n_illicit": int(row["n_illicit"]),
            "prevalence": num(row["prevalence"]),
            "noise": row["noise_flag"] != "",
        }
        if step_info.setdefault(step, info) != info:
            sys.exit(f"step {step}: label counts differ between strategies/lags in per_step_metrics.csv")
        per_step.append(
            {
                "strategy": row["strategy"],
                "lag": int(row["lag"]),
                "step": step,
                "pr_auc": num(row["pr_auc"]),
                "threshold": num(row["threshold"]),
                "flagged": int(row["flagged"]),
                "flag_rate": num(row["flag_rate"]),
            }
        )
    steps = [step_info[s] for s in sorted(step_info)]
    positives = {s["step"]: s["n_illicit"] for s in steps}

    # --- pooled PR-AUC with cluster-bootstrap CI ------------------------------------------
    pooled: list[dict[str, Any]] = []
    for row in filter(keep, read_csv(WALKFORWARD / "pooled_metrics.csv")):
        if row["ci_method"] != "cluster":
            sys.exit(f"pooled_metrics.csv: expected cluster CIs, got {row['ci_method']}")
        if row["window"] not in WF_WINDOWS:
            continue
        pooled.append(
            {
                "strategy": row["strategy"],
                "lag": int(row["lag"]),
                "window": row["window"],
                "n": int(row["n"]),
                "n_illicit": int(row["n_illicit"]),
                "pr_auc": num(row["pr_auc"]),
                "ci_lo": num(row["pr_auc_ci_lo"]),
                "ci_hi": num(row["pr_auc_ci_hi"]),
                "retrains_total": int(row["retrains_total"]),
            }
        )

    # --- retrain log -----------------------------------------------------------------------
    retrains: list[dict[str, Any]] = []
    for row in filter(keep, read_csv(WALKFORWARD / "retrain_log.csv")):
        if row["label_arrival_ok"] != "True":
            sys.exit(f"retrain_log.csv: label arrival violated for {row['strategy']} L={row['lag']} step {row['step']}")
        retrains.append(
            {
                "strategy": row["strategy"],
                "lag": int(row["lag"]),
                "step": int(row["step"]),
                "retrained": row["retrained"] == "True",
                "trained_through": int(row["trained_through"]),
            }
        )

    # --- top-K triage ----------------------------------------------------------------------
    triage_steps: list[dict[str, Any]] = []
    for row in filter(keep, read_csv(WALKFORWARD / "alert_budget.csv")):
        if row["level"] != "step" or int(row["K"]) not in WF_KS:
            continue
        step, k = int(row["step"]), int(row["K"])
        pos = int(row["n_illicit"])
        if pos != positives[step]:
            sys.exit(f"alert_budget.csv step {step}: positives {pos} != per_step_metrics {positives[step]}")
        tp, flagged = int(row["tp"]), int(row["flagged"])
        triage_steps.append(
            {
                "strategy": row["strategy"],
                "lag": int(row["lag"]),
                "step": step,
                "k": k,
                "flagged": flagged,
                "tp": tp,
                "positives": pos,
                "precision": tp / flagged if flagged else None,
                "recall": tp / pos if pos else None,
                # Best a perfect ranker could do this step with K alerts.
                "ceiling_recall": min(k, pos) / pos if pos else None,
            }
        )

    rules = {
        (r["strategy"], int(r["lag"]), r["window"], r["rule"]): r
        for r in filter(keep, read_csv(WALKFORWARD / "operating_rules.csv"))
    }
    triage_pooled: list[dict[str, Any]] = []
    for strategy in WF_STRATEGIES:
        for lag in WF_LAGS:
            for window, window_steps in WF_WINDOWS.items():
                total_pos = sum(positives[s] for s in window_steps)
                for k in WF_KS:
                    rows = [
                        t
                        for t in triage_steps
                        if t["strategy"] == strategy and t["lag"] == lag and t["k"] == k and t["step"] in window_steps
                    ]
                    if len(rows) != len(window_steps):
                        sys.exit(f"alert_budget.csv missing steps for {strategy} L={lag} {window} K={k}")
                    tp = sum(t["tp"] for t in rows)
                    alerts = sum(t["flagged"] for t in rows)
                    ceiling = sum(min(k, t["positives"]) for t in rows) / total_pos
                    entry = {
                        "strategy": strategy,
                        "lag": lag,
                        "window": window,
                        "k": k,
                        "alerts": alerts,
                        "tp": tp,
                        "positives": total_pos,
                        "precision": tp / alerts,
                        "recall": tp / total_pos,
                        "max_recall": ceiling,
                    }
                    triage_pooled.append(entry)
                    # Tie the recomputation to the harness's own pooled operating-rule table.
                    ref = rules[(strategy, lag, window, f"top{k}")]
                    if not (close(entry["precision"], num(ref["precision"])) and close(entry["recall"], num(ref["recall"]))):
                        sys.exit(f"top-{k} pooled precision/recall for {strategy} L={lag} {window} disagrees with operating_rules.csv")
                    if not close(ceiling, num(ref["max_possible_recall"])):
                        print(
                            f"note: max recall for {strategy} L={lag} {window} K={k} differs from "
                            f"operating_rules.csv ({ceiling:.6f} vs {ref['max_possible_recall']}); using the recomputed value",
                            file=sys.stderr,
                        )

    validation = []
    for (strategy, lag, window, rule), r in sorted(rules.items()):
        if rule != "validation_threshold" or window not in WF_WINDOWS:
            continue
        validation.append(
            {
                "strategy": strategy,
                "lag": lag,
                "window": window,
                "n_steps": int(r["n_steps"]),
                "alerts_total": int(r["alerts_total"]),
                "alerts_per_step": num(r["alerts_per_step"]),
                "precision": num(r["precision"]),
                "recall": num(r["recall"]),
                "steps_with_zero_alerts": int(r["steps_with_zero_alerts"]),
            }
        )

    return {
        "metadata": {
            "strategies": list(WF_STRATEGIES),
            "lags": list(WF_LAGS),
            "ks": list(WF_KS),
            "windows": list(WF_WINDOWS),
            "noise_min_positives": NOISE_MIN_POSITIVES,
            "ci": "95% cluster bootstrap over time steps",
        },
        "steps": steps,
        "per_step": per_step,
        "pooled": pooled,
        "retrains": retrains,
        "triage": {"steps": triage_steps, "pooled": triage_pooled, "validation_threshold": validation},
        "sources": [
            "results/walkforward/per_step_metrics.csv",
            "results/walkforward/pooled_metrics.csv",
            "results/walkforward/operating_rules.csv",
            "results/walkforward/alert_budget.csv (step-level rows only; max recall recomputed)",
            "results/walkforward/retrain_log.csv",
        ],
    }


def load_fixture_features() -> dict[str, float]:
    """Extract the canonical benchmark feature dict from the API test fixture."""
    tree = ast.parse(TEST_FIXTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == FIXTURE_NAME for t in node.targets
        ):
            feats = ast.literal_eval(node.value)
            if len(feats) != len(MODEL_FEATURES):
                sys.exit(f"{FIXTURE_NAME} has {len(feats)} features, expected {len(MODEL_FEATURES)}")
            return {k: float(feats[k]) for k in MODEL_FEATURES}
    sys.exit(f"{FIXTURE_NAME} not found in {TEST_FIXTURE}")


def pick_sample_rows() -> list[dict[str, Any]]:
    """Choose three real test-set rows (with recorded labels/scores) from predictions.csv."""
    wanted: dict[str, dict[str, Any]] = {}
    for row in read_csv(RESULTS / "xgboost" / "predictions.csv"):
        step = int(row["time_step"])
        label = int(row["y_true"])
        score = float(row["xgboost_score"])
        tx = row["txId"]
        if tx == "1813992" and "licit_benchmark" not in wanted:
            wanted["licit_benchmark"] = {"tx_id": tx, "time_step": step, "y_true": label, "recorded_score": score,
                                         "title": "Licit, step 35 (frozen-model benchmark)"}
        elif label == 1 and step < DRIFT_START and score >= THRESHOLD and "illicit_pre" not in wanted:
            wanted["illicit_pre"] = {"tx_id": tx, "time_step": step, "y_true": label, "recorded_score": score,
                                     "title": f"Illicit, step {step} (caught)"}
        elif label == 1 and step >= DRIFT_START and score < THRESHOLD and "illicit_drift" not in wanted:
            wanted["illicit_drift"] = {"tx_id": tx, "time_step": step, "y_true": label, "recorded_score": score,
                                       "title": f"Illicit, step {step} (missed in drift window)"}
    order = ["licit_benchmark", "illicit_pre", "illicit_drift"]
    return [wanted[k] for k in order if k in wanted]


def stream_features(tx_ids: set[str]) -> dict[str, dict[str, float]]:
    """Stream the raw CSV once, keeping only the requested rows (memory-safe)."""
    found: dict[str, dict[str, float]] = {}
    if not RAW_FEATURES.exists():
        return found
    with open(RAW_FEATURES, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["txId"] in tx_ids:
                found[row["txId"]] = {k: float(row[k]) for k in MODEL_FEATURES}
                if len(found) == len(tx_ids):
                    break
    return found


def build_samples() -> dict[str, Any]:
    """Bundle real labelled test rows (165 features + time_step) for the Try It page."""
    fixture = load_fixture_features()
    picks = pick_sample_rows()
    if not picks or picks[0]["tx_id"] != "1813992":
        sys.exit("benchmark tx 1813992 not found in results/xgboost/predictions.csv")

    raw = stream_features({p["tx_id"] for p in picks[1:]})
    samples: list[dict[str, Any]] = []
    for p in picks:
        if p["tx_id"] == "1813992":
            feats = fixture
        elif p["tx_id"] in raw:
            feats = raw[p["tx_id"]]
        else:
            print(f"warning: raw features for tx {p['tx_id']} unavailable (Og data missing); skipped", file=sys.stderr)
            continue
        samples.append({**p, "features": feats})
    return {
        "threshold": THRESHOLD,
        "n_features": len(MODEL_FEATURES),
        "samples": samples,
        "sources": [
            "results/xgboost/predictions.csv",
            "tests/test_api.py (SAMPLE_TX_1813992_FEATURES)",
            "Og data/txs_features.csv (read-only, streamed)",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON (NaN/Inf rejected) with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  ({len(text) / 1024:.1f} KB)")


def main() -> int:
    """Build all dashboard JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "frontend" / "public" / "data"))
    args = parser.parse_args()
    out = Path(args.out)

    write_json(out / "performance.json", build_performance())
    write_json(out / "monitoring.json", build_monitoring())
    write_json(out / "walkforward.json", build_walkforward())
    write_json(out / "samples.json", build_samples())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
