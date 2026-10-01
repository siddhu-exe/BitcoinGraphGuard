"""
Threshold Recalibration Module for BitcoinGraphGuard.

Addresses decision threshold miscalibration under severe label prevalence collapse
and temporal covariate shift. Compares static frozen thresholds (tau* = 0.435)
against adaptive recalibration strategies:
1. Rolling-Window Empirical F1-Optimal Threshold: Recomputed from recent labeled window [t-W, t-1].
2. Bayesian Prior Shift Adjustment (Saerens et al. / Elkan Odds-Ratio Adjustment):
   Analytically shifts decision threshold based on rolling vs baseline class prevalence.
3. Oracle Optimal Threshold: Upper-bound F1 threshold computed with perfect target knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class CalibrationMethod(str, Enum):
    """Threshold recalibration methodology."""

    FROZEN = "FROZEN"
    ROLLING_F1 = "ROLLING_F1"
    BAYESIAN_PRIOR = "BAYESIAN_PRIOR"
    ORACLE = "ORACLE"


@dataclass
class StepEvaluationMetrics:
    """Evaluation metrics for a single time step under a specific threshold."""

    time_step: int
    threshold: float
    method: CalibrationMethod
    n_total: int
    n_illicit: int
    n_licit: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "threshold": round(self.threshold, 4),
            "method": self.method.value,
            "n_total": self.n_total,
            "n_illicit": self.n_illicit,
            "n_licit": self.n_licit,
            "prevalence": round(self.prevalence, 4),
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


@dataclass
class ThresholdComparisonResult:
    """Side-by-side comparison of frozen vs adaptive thresholds."""

    time_step: int
    prevalence: float
    pr_auc: float
    frozen_metrics: StepEvaluationMetrics
    adaptive_f1_metrics: StepEvaluationMetrics
    adaptive_bayes_metrics: StepEvaluationMetrics
    oracle_metrics: StepEvaluationMetrics
    f1_lift_adaptive_f1: float
    f1_lift_bayes: float
    f1_lift_oracle: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "prevalence": round(self.prevalence, 4),
            "pr_auc": round(self.pr_auc, 4),
            "frozen_tau": round(self.frozen_metrics.threshold, 4),
            "frozen_f1": round(self.frozen_metrics.f1, 4),
            "frozen_precision": round(self.frozen_metrics.precision, 4),
            "frozen_recall": round(self.frozen_metrics.recall, 4),
            "frozen_tp": self.frozen_metrics.tp,
            "frozen_fp": self.frozen_metrics.fp,
            "adaptive_f1_tau": round(self.adaptive_f1_metrics.threshold, 4),
            "adaptive_f1_f1": round(self.adaptive_f1_metrics.f1, 4),
            "adaptive_f1_precision": round(self.adaptive_f1_metrics.precision, 4),
            "adaptive_f1_recall": round(self.adaptive_f1_metrics.recall, 4),
            "adaptive_f1_tp": self.adaptive_f1_metrics.tp,
            "adaptive_f1_fp": self.adaptive_f1_metrics.fp,
            "bayes_tau": round(self.adaptive_bayes_metrics.threshold, 4),
            "bayes_f1": round(self.adaptive_bayes_metrics.f1, 4),
            "bayes_precision": round(self.adaptive_bayes_metrics.precision, 4),
            "bayes_recall": round(self.adaptive_bayes_metrics.recall, 4),
            "bayes_tp": self.adaptive_bayes_metrics.tp,
            "bayes_fp": self.adaptive_bayes_metrics.fp,
            "oracle_tau": round(self.oracle_metrics.threshold, 4),
            "oracle_f1": round(self.oracle_metrics.f1, 4),
            "oracle_precision": round(self.oracle_metrics.precision, 4),
            "oracle_recall": round(self.oracle_metrics.recall, 4),
            "f1_lift_adaptive_f1": round(self.f1_lift_adaptive_f1, 4),
            "f1_lift_bayes": round(self.f1_lift_bayes, 4),
        }


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve, handling single-class edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    p, r, _ = precision_recall_curve(y_true, y_score)
    # Area via trapezoidal integration on (r, p)
    # Note: precision_recall_curve returns r in descending order
    return float(-np.trapezoid(p, r)) if hasattr(np, "trapezoid") else float(np.trapz(p, r[::-1]))


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    time_step: int,
    method: CalibrationMethod,
) -> StepEvaluationMetrics:
    """Compute complete metric suite for given predictions and threshold."""
    y_true_clean = np.asarray(y_true, dtype=int)
    y_score_clean = np.asarray(y_score, dtype=float)

    n_total = len(y_true_clean)
    n_illicit = int(np.sum(y_true_clean == 1))
    n_licit = int(np.sum(y_true_clean == 0))
    prev = float(n_illicit / max(1, n_total))

    y_pred = (y_score_clean >= threshold).astype(int)

    # PR-AUC and ROC-AUC
    if n_illicit > 0 and n_licit > 0:
        pr_auc = compute_pr_auc(y_true_clean, y_score_clean)
        try:
            roc_auc = float(roc_auc_score(y_true_clean, y_score_clean))
        except Exception:
            roc_auc = 0.5
    else:
        pr_auc = 0.0
        roc_auc = 0.5

    # Precision, Recall, F1
    if np.sum(y_pred) > 0 and n_illicit > 0:
        prec = float(precision_score(y_true_clean, y_pred, zero_division=0))
        rec = float(recall_score(y_true_clean, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_clean, y_pred, zero_division=0))
    else:
        prec = 0.0
        rec = 0.0
        f1 = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    return StepEvaluationMetrics(
        time_step=time_step,
        threshold=threshold,
        method=method,
        n_total=n_total,
        n_illicit=n_illicit,
        n_licit=n_licit,
        prevalence=prev,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_thresh: float = 0.01,
    max_thresh: float = 0.99,
    step: float = 0.005,
    fallback_threshold: float = 0.435,
) -> Tuple[float, float]:
    """
    Find decision threshold maximizing F1 score.

    Returns
    -------
    Tuple[float, float]
        (best_threshold, best_f1)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)

    if len(np.unique(y_t)) < 2 or np.sum(y_t == 1) == 0:
        return fallback_threshold, 0.0

    thresholds = np.arange(min_thresh, max_thresh + step, step)
    best_thresh = fallback_threshold
    best_f1 = -1.0

    for tau in thresholds:
        preds = (y_s >= tau).astype(int)
        score = f1_score(y_t, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(tau)

    return best_thresh, float(max(0.0, best_f1))


def bayesian_prior_shift_threshold(
    base_threshold: float,
    base_prevalence: float,
    target_prevalence: float,
) -> float:
    """
    Adjust decision threshold for raw model scores under prior probability shift.

    When class prior shifts from p0 (training) to pt (target regime), the threshold
    tau_adj applied to raw model score s(x) to maintain the original decision boundary
    is derived from Bayes rule:

        odds(tau_adj) = odds(tau_0) * [odds(p_0) / odds(p_t)]
        tau_adj = odds(tau_adj) / (1 + odds(tau_adj))

    When prevalence collapses (pt << p0), odds ratio > 1, so the threshold increases
    (e.g. 0.435 -> ~0.795), filtering out massive false positives.

    Parameters
    ----------
    base_threshold : float
        Original baseline decision threshold (e.g. 0.435).
    base_prevalence : float
        Baseline training prevalence (e.g. 0.1158).
    target_prevalence : float
        Estimated current rolling prevalence (e.g. 0.0253).

    Returns
    -------
    float
        Analytically recalibrated threshold in [0.01, 0.99].
    """
    p0 = float(np.clip(base_prevalence, 1e-4, 1.0 - 1e-4))
    pt = float(np.clip(target_prevalence, 1e-4, 1.0 - 1e-4))
    tau0 = float(np.clip(base_threshold, 1e-4, 1.0 - 1e-4))

    odds_tau0 = tau0 / (1.0 - tau0)
    odds_p0 = p0 / (1.0 - p0)
    odds_pt = pt / (1.0 - pt)

    # Ratio of training odds to target odds
    prior_adjustment_ratio = odds_p0 / odds_pt
    odds_tau_adj = odds_tau0 * prior_adjustment_ratio

    tau_adj = odds_tau_adj / (1.0 + odds_tau_adj)
    return float(np.clip(tau_adj, 0.01, 0.99))


class ThresholdCalibrator:
    """
    Adaptive Threshold Manager tracking rolling historical predictions.

    Parameters
    ----------
    frozen_threshold : float, default=0.435
        Frozen production baseline threshold (Phase 2 validation F1-optimal).
    baseline_prevalence : float, default=0.1158
        Training baseline prevalence.
    window_size : int, default=5
        Number of steps in historical rolling window.
    label_lag : int, default=1
        Delay before ground truth labels arrive for recalibration.
    """

    def __init__(
        self,
        frozen_threshold: float = 0.435,
        baseline_prevalence: float = 0.115809,
        window_size: int = 5,
        label_lag: int = 1,
    ):
        self.frozen_threshold = frozen_threshold
        self.baseline_prevalence = baseline_prevalence
        self.window_size = window_size
        self.label_lag = label_lag

        # History of past step predictions: step -> (y_true, y_score)
        self.history_preds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def record_step_predictions(
        self,
        time_step: int,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> None:
        """Store predictions and true labels for a step."""
        self.history_preds[time_step] = (
            np.asarray(y_true, dtype=int),
            np.asarray(y_score, dtype=float),
        )

    def compute_adaptive_thresholds(
        self,
        current_time_step: int,
        rolling_prevalence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute adaptive thresholds available at current_time_step.

        Returns
        -------
        Tuple[float, float]
            (adaptive_f1_tau, bayesian_prior_tau)
        """
        max_available_step = current_time_step - self.label_lag
        eligible_steps = [
            s for s in sorted(self.history_preds.keys())
            if s <= max_available_step and s > max_available_step - self.window_size
        ]

        if not eligible_steps:
            # Fallback to frozen baseline threshold
            return self.frozen_threshold, self.frozen_threshold

        # 1. Rolling F1 Threshold
        y_true_roll = np.concatenate([self.history_preds[s][0] for s in eligible_steps])
        y_score_roll = np.concatenate([self.history_preds[s][1] for s in eligible_steps])

        f1_tau, _ = find_optimal_f1_threshold(
            y_true=y_true_roll,
            y_score=y_score_roll,
            fallback_threshold=self.frozen_threshold,
        )

        # 2. Bayesian Prior Adjusted Threshold
        if rolling_prevalence is None:
            n_pos = np.sum(y_true_roll == 1)
            target_prev = n_pos / max(1, len(y_true_roll))
        else:
            target_prev = rolling_prevalence

        bayes_tau = bayesian_prior_shift_threshold(
            base_threshold=self.frozen_threshold,
            base_prevalence=self.baseline_prevalence,
            target_prevalence=target_prev,
        )

        return f1_tau, bayes_tau

    def evaluate_and_compare(
        self,
        current_time_step: int,
        y_true_current: np.ndarray,
        y_score_current: np.ndarray,
        rolling_prevalence: Optional[float] = None,
    ) -> ThresholdComparisonResult:
        """
        Perform complete retrospective threshold evaluation and comparison for current step.
        """
        y_true = np.asarray(y_true_current, dtype=int)
        y_score = np.asarray(y_score_current, dtype=float)

        # Compute adaptive thresholds using ONLY historical data available before current step
        f1_tau, bayes_tau = self.compute_adaptive_thresholds(
            current_time_step=current_time_step,
            rolling_prevalence=rolling_prevalence,
        )

        # Oracle threshold (cheating upper bound using current step's true labels)
        oracle_tau, _ = find_optimal_f1_threshold(
            y_true=y_true,
            y_score=y_score,
            fallback_threshold=self.frozen_threshold,
        )

        # Evaluate all configurations on current step
        frozen_m = evaluate_predictions(
            y_true, y_score, self.frozen_threshold, current_time_step, CalibrationMethod.FROZEN
        )
        adapt_f1_m = evaluate_predictions(
            y_true, y_score, f1_tau, current_time_step, CalibrationMethod.ROLLING_F1
        )
        bayes_m = evaluate_predictions(
            y_true, y_score, bayes_tau, current_time_step, CalibrationMethod.BAYESIAN_PRIOR
        )
        oracle_m = evaluate_predictions(
            y_true, y_score, oracle_tau, current_time_step, CalibrationMethod.ORACLE
        )

        # Record current step into history for future steps
        self.record_step_predictions(current_time_step, y_true, y_score)

        return ThresholdComparisonResult(
            time_step=current_time_step,
            prevalence=frozen_m.prevalence,
            pr_auc=frozen_m.pr_auc,
            frozen_metrics=frozen_m,
            adaptive_f1_metrics=adapt_f1_m,
            adaptive_bayes_metrics=bayes_m,
            oracle_metrics=oracle_m,
            f1_lift_adaptive_f1=adapt_f1_m.f1 - frozen_m.f1,
            f1_lift_bayes=bayes_m.f1 - frozen_m.f1,
            f1_lift_oracle=oracle_m.f1 - frozen_m.f1,
        )
