"""
Prevalence and Label Drift Monitor Module for BitcoinGraphGuard.

Tracks rolling illicit class prevalence over sliding temporal windows under
realistic label-arrival-lag constraints (no future lookahead). Detects abrupt
regime shifts (such as the 9.16% -> 2.53% collapse at step 43).

Baseline Training Reference (steps 1-34):
- Total labeled: 29,894
- Total illicit: 3,462
- Baseline prevalence: 11.5809% (~0.1158)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class PrevalenceDriftLevel(str, Enum):
    """Prevalence drift severity classification."""

    STABLE = "STABLE"
    WARNING = "WARNING"
    ALERT = "ALERT"


@dataclass
class StepLabelStats:
    """Aggregated label counts for a single time step."""

    time_step: int
    n_labeled: int
    n_illicit: int
    n_licit: int

    @property
    def prevalence(self) -> float:
        """Prevalence of illicit class in this step."""
        if self.n_labeled == 0:
            return 0.0
        return float(self.n_illicit / self.n_labeled)


@dataclass
class PrevalenceDriftReport:
    """Prevalence drift assessment report at a given time step."""

    time_step: int
    step_prevalence: float
    rolling_prevalence: float
    baseline_prevalence: float
    relative_change: float  # (rolling - baseline) / baseline
    absolute_change: float  # rolling - baseline
    window_size: int
    available_steps_in_window: List[int]
    total_labeled_in_window: int
    total_illicit_in_window: int
    drift_level: PrevalenceDriftLevel
    label_lag: int
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "step_prevalence": round(self.step_prevalence, 4),
            "rolling_prevalence": round(self.rolling_prevalence, 4),
            "baseline_prevalence": round(self.baseline_prevalence, 4),
            "relative_change_pct": round(self.relative_change * 100.0, 2),
            "absolute_change": round(self.absolute_change, 4),
            "window_size": self.window_size,
            "available_steps_in_window": self.available_steps_in_window,
            "total_labeled_in_window": self.total_labeled_in_window,
            "total_illicit_in_window": self.total_illicit_in_window,
            "drift_level": self.drift_level.value,
            "label_lag": self.label_lag,
            "reasons": self.reasons,
        }


class PrevalenceDriftMonitor:
    """
    Rolling Prevalence Drift Monitor with temporal lag safety.

    Parameters
    ----------
    baseline_prevalence : float, default=0.1158
        Historical training illicit class prevalence (steps 1-34: 11.58%).
    window_size : int, default=5
        Number of most recent labeled time steps to include in rolling estimation.
    label_lag : int, default=1
        Number of steps delay before ground-truth labels arrive.
        lag=1 means when evaluating at step t, labels up to step t-1 are available.
        lag=0 means labels for step t arrive synchronously at evaluation time.
    rel_warning_threshold : float, default=0.30
        Relative deviation threshold (|p_roll - p_base| / p_base) for WARNING (30% drop/rise).
    rel_alert_threshold : float, default=0.60
        Relative deviation threshold for ALERT (60% drop/rise, e.g. < 4.6% or > 18.5%).
    abs_collapse_threshold : float, default=0.035
        Absolute floor below which prevalence is classified as ALERT regime collapse.
    """

    DEFAULT_BASELINE_PREVALENCE = 0.115809

    def __init__(
        self,
        baseline_prevalence: float = DEFAULT_BASELINE_PREVALENCE,
        window_size: int = 5,
        label_lag: int = 1,
        rel_warning_threshold: float = 0.30,
        rel_alert_threshold: float = 0.60,
        abs_collapse_threshold: float = 0.035,
    ):
        self.baseline_prevalence = baseline_prevalence
        self.window_size = window_size
        self.label_lag = label_lag
        self.rel_warning_threshold = rel_warning_threshold
        self.rel_alert_threshold = rel_alert_threshold
        self.abs_collapse_threshold = abs_collapse_threshold

        # Historical record of all seen step label statistics
        self.history: Dict[int, StepLabelStats] = {}
        # Rolling buffer of StepLabelStats for fast access
        self.sliding_window: Deque[StepLabelStats] = deque(maxlen=window_size)

    def record_step_labels(
        self,
        time_step: int,
        y_true_or_counts: Union[np.ndarray, List[int], Tuple[int, int]],
    ) -> StepLabelStats:
        """
        Record labeled data arrival for a time step.

        Parameters
        ----------
        time_step : int
            Time step identifier.
        y_true_or_counts : array-like of binary labels (0=licit, 1=illicit)
                           OR tuple of (n_labeled, n_illicit).

        Returns
        -------
        StepLabelStats
        """
        if isinstance(y_true_or_counts, tuple) and len(y_true_or_counts) == 2:
            n_labeled, n_illicit = y_true_or_counts
            n_licit = n_labeled - n_illicit
        else:
            arr = np.asarray(y_true_or_counts, dtype=int)
            # Filter out unknown if class 3 is present
            arr = arr[arr != 3]
            # Convert class 2 to 0 if needed (1=illicit, 2=licit -> 1=illicit, 0=licit)
            binary = np.where(arr == 1, 1, 0)
            n_labeled = len(binary)
            n_illicit = int(np.sum(binary == 1))
            n_licit = int(np.sum(binary == 0))

        stats = StepLabelStats(
            time_step=time_step,
            n_labeled=n_labeled,
            n_illicit=n_illicit,
            n_licit=n_licit,
        )
        self.history[time_step] = stats
        return stats

    def evaluate_drift(
        self,
        current_time_step: int,
        current_step_labels: Optional[Union[np.ndarray, List[int], Tuple[int, int]]] = None,
    ) -> PrevalenceDriftReport:
        """
        Evaluate prevalence drift at current_time_step respecting label lag.

        If label_lag=0 and current_step_labels is provided, it is recorded first.
        If label_lag=1, labels available are strictly from steps <= current_time_step - 1.

        Parameters
        ----------
        current_time_step : int
            The active time step being processed.
        current_step_labels : Optional label input for current step.

        Returns
        -------
        PrevalenceDriftReport
        """
        if current_step_labels is not None and self.label_lag == 0:
            self.record_step_labels(current_time_step, current_step_labels)

        # Determine the maximum available step under label lag
        max_available_step = current_time_step - self.label_lag
        eligible_steps = [
            s for s in sorted(self.history.keys())
            if s <= max_available_step and s > max_available_step - self.window_size
        ]

        # Current step's own prevalence (if already recorded, else 0.0 or from history)
        step_stat = self.history.get(current_time_step)
        step_prevalence = step_stat.prevalence if step_stat else 0.0

        if not eligible_steps:
            # Not enough historical labels yet; default to baseline
            return PrevalenceDriftReport(
                time_step=current_time_step,
                step_prevalence=step_prevalence,
                rolling_prevalence=self.baseline_prevalence,
                baseline_prevalence=self.baseline_prevalence,
                relative_change=0.0,
                absolute_change=0.0,
                window_size=self.window_size,
                available_steps_in_window=[],
                total_labeled_in_window=0,
                total_illicit_in_window=0,
                drift_level=PrevalenceDriftLevel.STABLE,
                label_lag=self.label_lag,
                reasons=["Insufficient historical labeled steps under current lag."],
            )

        total_labeled = sum(self.history[s].n_labeled for s in eligible_steps)
        total_illicit = sum(self.history[s].n_illicit for s in eligible_steps)
        rolling_prev = total_illicit / max(1, total_labeled)

        rel_change = (rolling_prev - self.baseline_prevalence) / self.baseline_prevalence
        abs_change = rolling_prev - self.baseline_prevalence

        reasons: List[str] = []
        # Check drift severity
        if rolling_prev <= self.abs_collapse_threshold:
            drift_level = PrevalenceDriftLevel.ALERT
            reasons.append(
                f"PrevalenceCollapse: Rolling prevalence ({rolling_prev:.4f}) dropped below absolute critical floor ({self.abs_collapse_threshold:.4f})."
            )
        elif abs(rel_change) >= self.rel_alert_threshold:
            drift_level = PrevalenceDriftLevel.ALERT
            reasons.append(
                f"PrevalenceAlert: Rolling prevalence ({rolling_prev:.4f}) deviated by {rel_change * 100.0:+.1f}% from baseline ({self.baseline_prevalence:.4f}) [alert threshold: {self.rel_alert_threshold * 100:.0f}%]."
            )
        elif abs(rel_change) >= self.rel_warning_threshold:
            drift_level = PrevalenceDriftLevel.WARNING
            reasons.append(
                f"PrevalenceWarning: Rolling prevalence ({rolling_prev:.4f}) deviated by {rel_change * 100.0:+.1f}% from baseline ({self.baseline_prevalence:.4f}) [warning threshold: {self.rel_warning_threshold * 100:.0f}%]."
            )
        else:
            drift_level = PrevalenceDriftLevel.STABLE

        return PrevalenceDriftReport(
            time_step=current_time_step,
            step_prevalence=step_prevalence,
            rolling_prevalence=rolling_prev,
            baseline_prevalence=self.baseline_prevalence,
            relative_change=rel_change,
            absolute_change=abs_change,
            window_size=self.window_size,
            available_steps_in_window=eligible_steps,
            total_labeled_in_window=total_labeled,
            total_illicit_in_window=total_illicit,
            drift_level=drift_level,
            label_lag=self.label_lag,
            reasons=reasons,
        )
