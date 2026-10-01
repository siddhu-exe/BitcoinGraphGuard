"""
Feature Drift Monitor Module for BitcoinGraphGuard.

Implements a two-tier feature drift monitoring architecture:
1. High-Priority Tier: Individual PSI & KS tracking for confirmed macro-drift drivers
   (Aggregate_feature_10, Aggregate_feature_43, Aggregate_feature_8).
2. Broad Tier: Multivariate summary monitoring across all 93 Local_feature_* columns
   measuring the percentage of features drifting beyond critical thresholds.

Standard PSI Bands:
- PSI < 0.10: Stable / Negligible Drift
- 0.10 <= PSI <= 0.25: Moderate Drift (Warning)
- PSI > 0.25: Significant Drift (Action Required)

Monotonic / cumulative feature handling (design note)
----------------------------------------------------
Static-reference PSI is only valid for *stationary* features. It is NOT valid for
secular-trend (cumulative / network-growth) features such as Aggregate_feature_10/43/8,
whose per-step location is a monotone function of the time step (Spearman rho ~= 1.0).
For such features any future step necessarily sits in the upper tail of a reference
built from earlier steps, so a static-reference PSI reports a large, permanent
"drift" even while the feature is growing exactly as expected. That is a mismatch
between the monitoring method and the feature type, not evidence of a regime shift.

For features detected as monotone during ``fit`` the monitor therefore removes the
secular trend (linear / log-linear fit on the per-step reference median) and monitors
the *detrended residuals* instead of raw levels (Method 3 in the Phase 8a review).
The thresholded score in that case is the standardized residual-location shift
``|mean(residual_target) - mean(residual_ref)| / std(residual_ref)``; the distributional
``psi`` is still reported for transparency but is not used to raise alarms, because it
reacts to within-step spread differences rather than to an abnormal trend departure.
Do not "simplify" this back to a raw static-reference PSI for monotone features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftLevel(str, Enum):
    """PSI-based drift severity classification."""

    STABLE = "STABLE"
    MODERATE = "MODERATE"
    SIGNIFICANT = "SIGNIFICANT"

    @classmethod
    def from_psi(
        cls, psi: float, warn_thresh: float = 0.10, alert_thresh: float = 0.25
    ) -> DriftLevel:
        """Classify PSI into standard drift bands."""
        if psi < warn_thresh:
            return cls.STABLE
        if psi <= alert_thresh:
            return cls.MODERATE
        return cls.SIGNIFICANT


def calculate_psi(
    reference: Union[np.ndarray, pd.Series, List[float]],
    target: Union[np.ndarray, pd.Series, List[float]],
    num_bins: int = 10,
    min_prob: float = 1e-4,
    bin_edges: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and target samples.

    Uses proportion-level clipping to prevent empty bin probability singularities:
    p_i = clip(N_ref_i / N_ref, min_prob, 1.0)
    q_i = clip(N_tgt_i / N_tgt, min_prob, 1.0)
    PSI = sum((q_i - p_i) * ln(q_i / p_i))

    Parameters
    ----------
    reference : array-like
        Baseline/reference feature distribution (e.g. training steps 1-34).
    target : array-like
        Incoming/current feature distribution (e.g. test step t).
    num_bins : int, default=10
        Number of quantile bins to divide the reference distribution into.
    min_prob : float, default=1e-4
        Probability floor applied to empirical proportions to avoid log(0) singularity.
    bin_edges : Optional[np.ndarray], default=None
        Precomputed bin edges. If None, edges are computed from reference quantiles.

    Returns
    -------
    float
        Computed Population Stability Index (PSI).
    """
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)

    # Clean NaNs and infs
    ref = ref[np.isfinite(ref)]
    tgt = tgt[np.isfinite(tgt)]

    if len(ref) == 0 or len(tgt) == 0:
        return 0.0

    # Constant feature check
    if np.all(ref == ref[0]) and np.all(tgt == tgt[0]):
        return 0.0 if ref[0] == tgt[0] else float("inf")

    if bin_edges is None:
        # Generate quantile-based bins from reference
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(ref, quantiles)
        bin_edges = np.unique(bin_edges)

        # Fallback to equal-width bins if duplicate quantiles collapsed the bins
        if len(bin_edges) < 2:
            min_v, max_v = float(np.min(ref)), float(np.max(ref))
            if min_v == max_v:
                min_v -= 0.5
                max_v += 0.5
            bin_edges = np.linspace(min_v, max_v, num_bins + 1)

    # Extend boundaries to cover target values
    bin_edges = bin_edges.copy()
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate empirical histograms
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt, bins=bin_edges)

    # Convert to empirical proportions with probability-level floor
    n_ref = len(ref)
    n_tgt = len(tgt)

    p_ref = np.clip(ref_counts / n_ref, min_prob, 1.0)
    q_tgt = np.clip(tgt_counts / n_tgt, min_prob, 1.0)

    # Re-normalize
    p_ref = p_ref / np.sum(p_ref)
    q_tgt = q_tgt / np.sum(q_tgt)

    # PSI formula: sum((q - p) * ln(q / p))
    psi = np.sum((q_tgt - p_ref) * np.log(q_tgt / p_ref))
    return float(max(0.0, psi))


def calculate_ks(
    reference: Union[np.ndarray, pd.Series, List[float]],
    target: Union[np.ndarray, pd.Series, List[float]],
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov 2-sample statistic and p-value.

    Parameters
    ----------
    reference : array-like
        Baseline reference sample.
    target : array-like
        Target sample.

    Returns
    -------
    Tuple[float, float]
        (ks_statistic, p_value)
    """
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)

    ref = ref[np.isfinite(ref)]
    tgt = tgt[np.isfinite(tgt)]

    if len(ref) == 0 or len(tgt) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(ref, tgt)
    return float(res.statistic), float(res.pvalue)


@dataclass
class SingleFeatureDrift:
    """Metrics for a single monitored feature.

    Attributes
    ----------
    psi : float
        Population Stability Index between the reference and the target on the
        *monitored scale*. For monotone features the monitored scale is the
        detrended residual, so ``psi`` describes residual-distribution shape.
    drift_score : float
        The value the monitor actually thresholds. Equals ``psi`` for stationary
        features; for monotone/detrended features it is the standardized residual
        location shift (see the module docstring).
    method : str
        ``"static"`` or ``"detrended_<linear|log-linear>_residual"``.
    """

    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    level: DriftLevel
    ref_mean: float
    target_mean: float
    ref_std: float
    target_std: float
    method: str = "static"
    drift_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "method": self.method,
            "psi": round(self.psi, 4),
            "drift_score": round(self.drift_score, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_p_value": float(self.ks_p_value),
            "level": self.level.value,
            "ref_mean": round(self.ref_mean, 4),
            "target_mean": round(self.target_mean, 4),
            "ref_std": round(self.ref_std, 4),
            "target_std": round(self.target_std, 4),
        }


@dataclass
class FeatureDriftReport:
    """Comprehensive drift report for a single time step."""

    time_step: int
    n_samples: int
    triad_metrics: Dict[str, SingleFeatureDrift]
    triad_drift_level: DriftLevel
    local_metrics: Dict[str, SingleFeatureDrift]
    pct_local_drifted_moderate: float
    pct_local_drifted_significant: float
    local_drift_level: DriftLevel
    overall_drift_level: DriftLevel
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "n_samples": self.n_samples,
            "triad_drift_level": self.triad_drift_level.value,
            "triad_metrics": {k: v.to_dict() for k, v in self.triad_metrics.items()},
            "pct_local_drifted_moderate": round(self.pct_local_drifted_moderate, 4),
            "pct_local_drifted_significant": round(self.pct_local_drifted_significant, 4),
            "local_drift_level": self.local_drift_level.value,
            "overall_drift_level": self.overall_drift_level.value,
            "reasons": self.reasons,
        }


class FeatureDriftMonitor:
    """
    Two-Tier Feature Drift Monitor for Bitcoin transaction features.

    Tier 1 (High-Priority): Focal aggregate triad features
    ('Aggregate_feature_10', 'Aggregate_feature_43', 'Aggregate_feature_8').
    Tier 2 (Broad): All 93 'Local_feature_*' transaction features.
    """

    DEFAULT_TRIAD = [
        "Aggregate_feature_10",
        "Aggregate_feature_43",
        "Aggregate_feature_8",
    ]

    def __init__(
        self,
        triad_features: Optional[List[str]] = None,
        local_features: Optional[List[str]] = None,
        psi_warn_threshold: float = 0.10,
        psi_alert_threshold: float = 0.25,
        ks_alert_threshold: float = 0.30,
        broad_moderate_pct_threshold: float = 20.0,
        broad_significant_pct_threshold: float = 40.0,
        num_bins: int = 10,
        monotonic_rho_threshold: float = 0.99,
        min_reference_steps: int = 10,
    ):
        self.triad_features = triad_features or list(self.DEFAULT_TRIAD)
        self.local_features = local_features or [f"Local_feature_{i}" for i in range(1, 94)]
        self.all_monitored_features = sorted(list(set(self.triad_features + self.local_features)))

        self.psi_warn_threshold = psi_warn_threshold
        self.psi_alert_threshold = psi_alert_threshold
        self.ks_alert_threshold = ks_alert_threshold
        self.broad_moderate_pct_threshold = broad_moderate_pct_threshold
        self.broad_significant_pct_threshold = broad_significant_pct_threshold
        self.num_bins = num_bins
        # Threshold on |Spearman rho(step, per-step median)| above which a feature
        # is treated as secular-trend (cumulative) and detrended before monitoring.
        self.monotonic_rho_threshold = monotonic_rho_threshold
        self.min_reference_steps = min_reference_steps

        # Precomputed reference distribution artifacts
        self.reference_samples_: Dict[str, np.ndarray] = {}
        self.reference_bin_edges_: Dict[str, np.ndarray] = {}
        self.reference_stats_: Dict[str, Tuple[float, float]] = {}  # (mean, std)
        # Monotone-feature trend artifacts: {feature: (coef, trend_type)}
        self.trend_params_: Dict[str, Tuple[np.ndarray, str]] = {}
        self.feature_methods_: Dict[str, str] = {}
        self.monotonic_rho_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, reference_df: pd.DataFrame) -> FeatureDriftMonitor:
        """
        Fit baseline reference distribution from training data (steps 1-34).

        Features whose per-step reference median is monotone in the time step
        (|Spearman rho| >= ``monotonic_rho_threshold``) are flagged as
        cumulative/secular-trend features. A trend is fitted on their per-step
        reference median and the stored reference distribution becomes the
        residual distribution ``value - trend(step)``. All other features keep a
        raw static reference distribution.

        Parameters
        ----------
        reference_df : pd.DataFrame
            DataFrame containing reference features.
        """
        logger.info("Fitting FeatureDriftMonitor on reference data of shape %s", reference_df.shape)
        self.reference_samples_.clear()
        self.reference_bin_edges_.clear()
        self.reference_stats_.clear()
        self.trend_params_.clear()
        self.feature_methods_.clear()
        self.monotonic_rho_.clear()

        has_step = "Time step" in reference_df.columns

        for feat in self.all_monitored_features:
            if feat not in reference_df.columns:
                logger.warning("Feature %s not present in reference DataFrame", feat)
                continue

            sub = reference_df[["Time step", feat]].dropna() if has_step else reference_df[[feat]].dropna()
            vals = sub[feat].to_numpy(dtype=float)
            if len(vals) == 0:
                continue

            method = "static"

            # Detect secular-trend (cumulative) features and detrend them.
            if has_step:
                med = sub.groupby("Time step")[feat].median()
                med = med[np.isfinite(med.to_numpy(dtype=float))]
                if len(med) >= self.min_reference_steps and np.unique(med.to_numpy(dtype=float)).size > 1:
                    step_idx = med.index.to_numpy(dtype=float)
                    med_vals = med.to_numpy(dtype=float)
                    rho = stats.spearmanr(step_idx, med_vals)[0]
                    if np.isfinite(rho) and abs(rho) >= self.monotonic_rho_threshold:
                        # Fit both linear and (when defined) log-linear trends on the
                        # per-step median and keep whichever fits the reference better.
                        coef = np.polyfit(step_idx, med_vals, 1)
                        trend_type = "linear"
                        pred_lin = np.polyval(coef, step_idx)
                        rmse_lin = float(np.sqrt(np.mean((med_vals - pred_lin) ** 2)))
                        if np.all(med_vals > 0):
                            coef_log = np.polyfit(step_idx, np.log(med_vals), 1)
                            pred_log = np.exp(np.polyval(coef_log, step_idx))
                            rmse_log = float(np.sqrt(np.mean((med_vals - pred_log) ** 2)))
                            if rmse_log < rmse_lin:
                                coef, trend_type = coef_log, "log-linear"
                        if trend_type == "log-linear":
                            trend_vals = np.exp(np.polyval(coef, sub["Time step"].to_numpy(dtype=float)))
                        else:
                            trend_vals = np.polyval(coef, sub["Time step"].to_numpy(dtype=float))
                        vals = vals - trend_vals
                        self.trend_params_[feat] = (coef, trend_type)
                        self.monotonic_rho_[feat] = float(rho)
                        method = f"detrended_{trend_type}_residual"
                        logger.info(
                            "Feature %s flagged as monotone (rho=%.4f); monitoring detrended %s residuals.",
                            feat, rho, trend_type,
                        )

            self.feature_methods_[feat] = method

            # Store compact subsample if huge, or full values
            self.reference_samples_[feat] = vals
            self.reference_stats_[feat] = (float(np.mean(vals)), float(np.std(vals)))

            # Precompute reference bin edges
            quantiles = np.linspace(0, 100, self.num_bins + 1)
            edges = np.percentile(vals, quantiles)
            edges = np.unique(edges)
            if len(edges) < 2:
                min_v, max_v = float(np.min(vals)), float(np.max(vals))
                if min_v == max_v:
                    min_v -= 0.5
                    max_v += 0.5
                edges = np.linspace(min_v, max_v, self.num_bins + 1)
            self.reference_bin_edges_[feat] = edges

        self.is_fitted = True
        return self

    def score_feature(
        self,
        feat_name: str,
        target_vals: Union[np.ndarray, pd.Series, List[float]],
        time_step: Optional[int] = None,
    ) -> SingleFeatureDrift:
        """Score drift for a single feature against reference.

        For features flagged as monotone during ``fit`` the target is first
        detrended (``value - trend(time_step)``) and the drift score becomes the
        standardized residual-location shift. ``time_step`` is required for such
        features. Stationary features are scored exactly as before.
        """
        if not self.is_fitted or feat_name not in self.reference_samples_:
            raise ValueError(f"Monitor is not fitted for feature: {feat_name}")

        ref_vals = self.reference_samples_[feat_name]
        bin_edges = self.reference_bin_edges_.get(feat_name)
        ref_mean, ref_std = self.reference_stats_[feat_name]
        method = self.feature_methods_.get(feat_name, "static")

        tgt = np.asarray(target_vals, dtype=float)

        if feat_name in self.trend_params_:
            if time_step is None:
                raise ValueError(
                    f"time_step is required to detrend monotone feature {feat_name}"
                )
            coef, trend_type = self.trend_params_[feat_name]
            if trend_type == "log-linear":
                trend_value = float(np.exp(np.polyval(coef, time_step)))
            else:
                trend_value = float(np.polyval(coef, time_step))
            tgt = tgt - trend_value

        tgt_clean = tgt[np.isfinite(tgt)]
        tgt_mean = float(np.mean(tgt_clean)) if len(tgt_clean) > 0 else 0.0
        tgt_std = float(np.std(tgt_clean)) if len(tgt_clean) > 0 else 0.0

        psi = calculate_psi(
            reference=ref_vals,
            target=tgt,
            num_bins=self.num_bins,
            bin_edges=bin_edges,
        )
        ks_stat, ks_pval = calculate_ks(reference=ref_vals, target=tgt)

        # Monotone/detrended features are thresholded on a robust location shift
        # rather than distributional PSI, so that within-step spread noise does
        # not masquerade as a regime change. See module docstring.
        if feat_name in self.trend_params_ and ref_std > 0:
            drift_score = abs(tgt_mean - ref_mean) / ref_std
        else:
            drift_score = psi

        level = DriftLevel.from_psi(
            drift_score,
            warn_thresh=self.psi_warn_threshold,
            alert_thresh=self.psi_alert_threshold,
        )

        return SingleFeatureDrift(
            feature_name=feat_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_p_value=ks_pval,
            level=level,
            ref_mean=ref_mean,
            target_mean=tgt_mean,
            ref_std=ref_std,
            target_std=tgt_std,
            method=method,
            drift_score=float(drift_score),
        )

    def evaluate_step(
        self,
        step_df: pd.DataFrame,
        time_step: int,
    ) -> FeatureDriftReport:
        """
        Evaluate full two-tier feature drift for a single time step.

        Parameters
        ----------
        step_df : pd.DataFrame
            DataFrame containing transactions for this time step.
        time_step : int
            Current time step identifier.

        Returns
        -------
        FeatureDriftReport
            Structured report with triad and local drift analytics.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureDriftMonitor must be fitted before evaluating steps.")

        n_samples = len(step_df)
        reasons: List[str] = []

        # 1. High-Priority Triad Evaluation
        triad_metrics: Dict[str, SingleFeatureDrift] = {}
        triad_levels: List[DriftLevel] = []

        for feat in self.triad_features:
            if feat in step_df.columns:
                m = self.score_feature(feat, step_df[feat], time_step=time_step)
                triad_metrics[feat] = m
                triad_levels.append(m.level)
                if m.level == DriftLevel.SIGNIFICANT:
                    reasons.append(
                        f"HighPriorityTriad: {feat} severe drift (score={m.drift_score:.3f} > {self.psi_alert_threshold}, "
                        f"method={m.method}, PSI={m.psi:.3f}, KS={m.ks_statistic:.3f})"
                    )
                elif m.level == DriftLevel.MODERATE:
                    reasons.append(
                        f"HighPriorityTriad: {feat} moderate drift (score={m.drift_score:.3f} >= "
                        f"{self.psi_warn_threshold}, method={m.method})"
                    )

        if DriftLevel.SIGNIFICANT in triad_levels:
            triad_drift_level = DriftLevel.SIGNIFICANT
        elif DriftLevel.MODERATE in triad_levels:
            triad_drift_level = DriftLevel.MODERATE
        else:
            triad_drift_level = DriftLevel.STABLE

        # 2. Broad Tier (Local Features) Evaluation
        local_metrics: Dict[str, SingleFeatureDrift] = {}
        local_mod_count = 0
        local_sig_count = 0
        n_local_evaluated = 0

        for feat in self.local_features:
            if feat in step_df.columns:
                m = self.score_feature(feat, step_df[feat], time_step=time_step)
                local_metrics[feat] = m
                n_local_evaluated += 1
                if m.level == DriftLevel.SIGNIFICANT:
                    local_sig_count += 1
                    local_mod_count += 1
                elif m.level == DriftLevel.MODERATE:
                    local_mod_count += 1

        pct_local_mod = (local_mod_count / max(1, n_local_evaluated)) * 100.0
        pct_local_sig = (local_sig_count / max(1, n_local_evaluated)) * 100.0

        if pct_local_sig >= self.broad_significant_pct_threshold:
            local_drift_level = DriftLevel.SIGNIFICANT
            reasons.append(
                f"BroadLocalTier: {pct_local_sig:.1f}% local features in significant drift (>= {self.broad_significant_pct_threshold}%)"
            )
        elif pct_local_mod >= self.broad_moderate_pct_threshold:
            local_drift_level = DriftLevel.MODERATE
            reasons.append(
                f"BroadLocalTier: {pct_local_mod:.1f}% local features in moderate drift (>= {self.broad_moderate_pct_threshold}%)"
            )
        else:
            local_drift_level = DriftLevel.STABLE

        # 3. Overall Feature Drift Level
        if (
            triad_drift_level == DriftLevel.SIGNIFICANT
            or local_drift_level == DriftLevel.SIGNIFICANT
        ):
            overall_drift_level = DriftLevel.SIGNIFICANT
        elif (
            triad_drift_level == DriftLevel.MODERATE
            or local_drift_level == DriftLevel.MODERATE
        ):
            overall_drift_level = DriftLevel.MODERATE
        else:
            overall_drift_level = DriftLevel.STABLE

        return FeatureDriftReport(
            time_step=time_step,
            n_samples=n_samples,
            triad_metrics=triad_metrics,
            triad_drift_level=triad_drift_level,
            local_metrics=local_metrics,
            pct_local_drifted_moderate=pct_local_mod,
            pct_local_drifted_significant=pct_local_sig,
            local_drift_level=local_drift_level,
            overall_drift_level=overall_drift_level,
            reasons=reasons,
        )
