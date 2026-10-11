"""
Retrospective evaluation of monitoring signals for the Phase 8a backtest.

The monitoring engine emits a status per channel per step, but the backtest report
never answers the question a reviewer actually asks: *which alarms were worth
listening to?* This module answers it with two metrics per signal, computed over
steps 35-49:

* **false-alarm rate** on steps 35-42, the window where the model is healthy, so
  any alert there is a false alarm; and
* **detection lead/lag** relative to step 43, the first drift step, i.e. the first
  step in 43-49 where the signal alerts minus 43.

Signals that read labels are marked ``label_dependent``; under a one-step label
delay they cannot fire at step 43 by construction. Two label-free score channels
(KS and predicted-positive rate) are evaluated from the committed prediction file
so that a fair comparison exists.

All thresholds below are fixed constants (not tuned on 43-49) and the evaluation
uses artifacts already exported by the backtest, so it is deterministic and cheap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.feature_drift import calculate_ks

#: Statuses that count as "the channel alerted".
ALERT_STATES = {"MODERATE", "WARNING", "ALERT", "CRITICAL", "SIGNIFICANT"}
#: Statuses that mean "the channel had nothing to say" and are excluded from rates.
UNAVAILABLE_STATES = {"", "UNAVAILABLE", "NOT_EVALUATED"}

EARLY_WINDOW = range(35, 43)
DRIFT_WINDOW = range(43, 50)
DRIFT_START = 43

#: Adversarial validation is WARNING-only in the engine; this is its alert band.
ADVERSARIAL_ALERT_AUC = 0.85
#: Fixed score-channel thresholds, declared before evaluation on 43-49.
KS_ALERT = 0.10
POSITIVE_RATE_LOW = 0.5
POSITIVE_RATE_HIGH = 2.0


def _status_alert(value: Any) -> bool:
    """True when a channel status counts as an alert."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().upper() in ALERT_STATES


def _status_evaluated(value: Any) -> bool:
    """True when a channel status is a real observation (not blank/unavailable)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().upper() not in UNAVAILABLE_STATES


#: column -> (human name, label dependent). Adversarial is handled separately.
STATUS_SIGNALS: list[dict[str, Any]] = [
    {
        "signal": "triad_feature_drift",
        "column": "triad_status",
        "label_dependent": False,
    },
    {
        "signal": "local_feature_shift",
        "column": "feature_shift_status",
        "label_dependent": False,
    },
    {
        "signal": "score_shift_psi",
        "column": "score_shift_status",
        "label_dependent": False,
    },
    {
        "signal": "rolling_prevalence",
        "column": "prevalence_channel_status",
        "label_dependent": True,
    },
    {
        "signal": "performance_crash",
        "column": "performance_status",
        "label_dependent": True,
    },
    {
        "signal": "retraining_trigger",
        "column": "trigger_severity",
        "label_dependent": True,
    },
]


def _summarize(
    alerts: Mapping[int, bool], evaluated: Sequence[int], label_dependent: bool
) -> dict[str, Any]:
    """Reduce a per-step alert map to the false-alarm-rate / detection-lead summary."""
    early = [s for s in evaluated if s in EARLY_WINDOW]
    drift = [s for s in evaluated if s in DRIFT_WINDOW]
    early_alerts = [s for s in early if alerts.get(s)]
    drift_alerts = [s for s in drift if alerts.get(s)]
    first_drift_alert: int | None = drift_alerts[0] if drift_alerts else None
    return {
        "label_dependent": label_dependent,
        "n_evaluated_35_42": len(early),
        "alerts_35_42": len(early_alerts),
        "false_alarm_rate_35_42": (len(early_alerts) / len(early)) if early else None,
        "n_evaluated_43_49": len(drift),
        "alerts_43_49": len(drift_alerts),
        "first_alert_step_43_49": first_drift_alert,
        "detection_lead_vs_43": (first_drift_alert - DRIFT_START)
        if first_drift_alert
        else None,
        "detected_43_49": bool(drift_alerts),
    }


def evaluate_signals(step_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-signal false-alarm rate and detection lead from exported step metrics.

    ``step_metrics`` is ``results/monitoring/step_monitoring_metrics.csv``: one row per
    step 35-49 with a status column per channel.
    """
    df = step_metrics.copy()
    df["time_step"] = df["time_step"].astype(int)
    rows: list[dict[str, Any]] = []

    for spec in STATUS_SIGNALS:
        col = spec["column"]
        if col not in df.columns:
            continue
        alerts = {
            int(r.time_step): _status_alert(getattr(r, col)) for r in df.itertuples()
        }
        evaluated = sorted(
            int(r.time_step)
            for r in df.itertuples()
            if _status_evaluated(getattr(r, col))
        )
        rows.append(
            {
                "signal": spec["signal"],
                **_summarize(alerts, evaluated, spec["label_dependent"]),
            }
        )

    if "adversarial_auc" in df.columns:
        vals = df[["time_step", "adversarial_auc"]].dropna()
        alerts = {
            int(r.time_step): float(r.adversarial_auc) >= ADVERSARIAL_ALERT_AUC
            for r in vals.itertuples()
        }
        evaluated = sorted(int(r.time_step) for r in vals.itertuples())
        rows.append(
            {"signal": "adversarial_validation", **_summarize(alerts, evaluated, False)}
        )

    return pd.DataFrame(rows)


def evaluate_score_channels(
    predictions: pd.DataFrame,
    threshold: float = 0.435,
    label_delay_steps: int = 1,
    reference_steps: int = 5,
    score_column: str = "xgboost_score",
) -> pd.DataFrame:
    """Evaluate two label-free score channels (KS and predicted-positive rate).

    Reference for step ``t`` is the last ``reference_steps`` steps whose labels would
    already be available, i.e. steps ``<= t - label_delay_steps``. Only scores are used,
    so no label from step ``t`` (or later) enters the computation.
    """
    data = predictions[["time_step", score_column]].dropna().copy()
    data["time_step"] = data["time_step"].astype(int)
    by_step = {
        int(s): g[score_column].to_numpy(dtype=float)
        for s, g in data.groupby("time_step")
    }

    per_step: list[dict[str, Any]] = []
    for t in range(35, 50):
        if t not in by_step:
            continue
        usable = [
            s for s in by_step if s <= t - label_delay_steps and s >= min(by_step)
        ]
        ref_steps = sorted(usable)[-reference_steps:]
        if not ref_steps:
            continue
        ref_scores = np.concatenate([by_step[s] for s in ref_steps])
        cur_scores = by_step[t]
        ks, _ = calculate_ks(ref_scores, cur_scores)
        ref_rate = float(np.mean(ref_scores >= threshold))
        cur_rate = float(np.mean(cur_scores >= threshold))
        ratio = (
            (cur_rate / ref_rate) if ref_rate > 0 else np.inf if cur_rate > 0 else 1.0
        )
        per_step.append(
            {
                "time_step": t,
                "ks": float(ks),
                "positive_rate": cur_rate,
                "reference_positive_rate": ref_rate,
                "positive_rate_ratio": ratio,
                "ks_alert": bool(ks >= KS_ALERT),
                "positive_rate_alert": bool(
                    ratio < POSITIVE_RATE_LOW or ratio > POSITIVE_RATE_HIGH
                ),
            }
        )

    frame = pd.DataFrame(per_step)
    rows: list[dict[str, Any]] = []
    for name, col in (
        ("score_ks", "ks_alert"),
        ("score_positive_rate", "positive_rate_alert"),
    ):
        alerts = {int(r.time_step): bool(getattr(r, col)) for r in frame.itertuples()}
        evaluated = sorted(int(r.time_step) for r in frame.itertuples())
        rows.append({"signal": name, **_summarize(alerts, evaluated, False)})
    return pd.DataFrame(rows)


def combine_signal_tables(
    status_table: pd.DataFrame, score_table: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate the status-channel and score-channel summaries in report order."""
    return pd.concat([status_table, score_table], ignore_index=True)


def format_signal_evaluation_markdown(table: pd.DataFrame) -> str:
    """Render the signal-evaluation table and a plain 'what did not work' paragraph."""
    lines = [
        "### Signal evaluation: which alarms were worth listening to",
        "",
        "False-alarm rate is computed on steps 35-42 (model healthy); detection lead is the first",
        "step in 43-49 where the signal alerted minus 43 (0 = on time at the drift step, positive = late).",
        "Label-dependent channels cannot alert at step 43 under the one-step label delay.",
        "",
        "| Signal | Labels | False alarms 35-42 | First alert 43-49 | Lead vs 43 | Alerts 43-49 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in table.itertuples():
        far = (
            "n/a"
            if r.false_alarm_rate_35_42 is None
            else f"{r.alerts_35_42}/{r.n_evaluated_35_42} ({r.false_alarm_rate_35_42:.0%})"
        )
        first = (
            "never"
            if pd.isna(r.first_alert_step_43_49)
            else str(int(r.first_alert_step_43_49))
        )
        lead = (
            "n/a"
            if pd.isna(r.detection_lead_vs_43)
            else f"{int(r.detection_lead_vs_43):+d}"
        )
        lines.append(
            f"| `{r.signal}` | {'yes' if r.label_dependent else 'no'} | {far} | {first} | {lead} | {r.alerts_43_49} |"
        )

    noisy = table[(table["false_alarm_rate_35_42"].fillna(0) >= 0.5)]
    missed = table[~table["detected_43_49"]]
    lines += ["", "**What did NOT work**", ""]
    if noisy.empty and missed.empty:
        lines.append(
            "- No signal was both noisy in 35-42 and blind in 43-49 in this backtest."
        )
    else:
        for r in noisy.itertuples():
            lines.append(
                f"- `{r.signal}` alerted on {r.alerts_35_42}/{r.n_evaluated_35_42} healthy steps "
                f"({r.false_alarm_rate_35_42:.0%}) - its alarms cannot be acted on alone."
            )
        for r in missed.itertuples():
            lines.append(
                f"- `{r.signal}` never alerted in 43-49 - it carries no onset signal here."
            )
    lines.append("")
    lines.append(
        "This is one regime change on one dataset; the label-free score channels use fixed, "
        "pre-declared thresholds and are not tuned on 43-49."
    )
    return "\n".join(lines)


def load_and_evaluate(
    step_metrics_path: str = "results/monitoring/step_monitoring_metrics.csv",
    predictions_path: str = "results/xgboost/predictions.csv",
    threshold: float = 0.435,
) -> pd.DataFrame:
    """Load both artifacts and return the combined signal-evaluation table."""
    step_metrics = pd.read_csv(step_metrics_path)
    status_table = evaluate_signals(step_metrics)
    try:
        predictions = pd.read_csv(
            predictions_path, usecols=["time_step", "xgboost_score"]
        )
        score_table = evaluate_score_channels(predictions, threshold=threshold)
    except (FileNotFoundError, ValueError):
        score_table = pd.DataFrame(columns=status_table.columns)
    return combine_signal_tables(status_table, score_table)
