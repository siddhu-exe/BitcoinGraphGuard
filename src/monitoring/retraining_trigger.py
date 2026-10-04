"""
Retraining Trigger Engine Module for BitcoinGraphGuard.

Synthesizes multiple monitoring telemetry streams (high-priority triad feature drift,
broad local feature drift, rolling label prevalence collapse, adversarial AUC,
and threshold calibration) into an explicit, auditable production decision:
- RETRAIN: Retrain XGBoost over a rolling temporal window (e.g. W=20 steps).
- RECALIBRATE_ONLY: Update decision threshold (tau_t) via Bayesian prior or empirical F1 adaptation.
- NO_ACTION: Continue production inference with current model and threshold.

Channels and who may raise CRITICAL
-----------------------------------
Label-dependent (read labels of steps <= t - label_delay_steps ONLY):
- performance (lag-safe per-step F1 floor)            -> CRITICAL
- prevalence collapse (rolling label prevalence)      -> CRITICAL
Label-free (usable during label delay):
- score shift (PSI of the deployed model's scores)    -> CRITICAL at PSI >= 0.25, WARNING at >= 0.10
- triad detrended aggregate drift                     -> CRITICAL
- feature shift (% local features with PSI > 0.25)    -> WARNING only, never CRITICAL
- adversarial validation                              -> WARNING only (saturated channel)

A label from step t never influences the decision at step t: label-based inputs are
rejected with :class:`LagViolationError` if they contain a newer step than t - L.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.monitoring.adversarial_drift import AdversarialDriftLevel, AdversarialDriftReport
from src.monitoring.feature_drift import DriftLevel, FeatureDriftReport
from src.monitoring.lag_safe import (
    PSI_CRITICAL,
    PSI_WARNING,
    LagViolationError,
    PerformanceWindowReport,
    ScoreShiftReport,
)
from src.monitoring.prevalence_drift import PrevalenceDriftLevel, PrevalenceDriftReport
from src.monitoring.threshold_calibration import ThresholdComparisonResult

logger = logging.getLogger(__name__)


class TriggerAction(str, Enum):
    """Production action recommended by the monitoring system."""

    NO_ACTION = "NO_ACTION"
    RECALIBRATE_ONLY = "RECALIBRATE_ONLY"
    RETRAIN = "RETRAIN"


class AlertSeverity(str, Enum):
    """Alert severity tier."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RetrainingDecision:
    """Auditable monitoring decision output for a single time step."""

    time_step: int
    action: TriggerAction
    severity: AlertSeverity
    primary_reason: str
    reasons: List[str] = field(default_factory=list)
    component_statuses: Dict[str, str] = field(default_factory=dict)
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    # Additive fields: label-delay assumption and per-channel breakdown.
    label_delay_steps: int = 1
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def label_delay_assumption(self) -> str:
        L = self.label_delay_steps
        return (
            f"Labels of step s become usable at step s+{L} (label_delay_steps={L}); "
            f"label-based channels read steps <= t-{L} only."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "action": self.action.value,
            "severity": self.severity.value,
            "primary_reason": self.primary_reason,
            "reasons": self.reasons,
            "component_statuses": self.component_statuses,
            "telemetry_summary": self.telemetry_summary,
            "label_delay_steps": self.label_delay_steps,
            "label_delay_assumption": self.label_delay_assumption,
            "channels": self.channels,
        }


def _channel(
    kind: str,
    status: str,
    value: Optional[float] = None,
    threshold: Optional[float] = None,
    can_trigger_critical: bool = True,
    detail: str = "",
) -> Dict[str, Any]:
    """One entry of the per-channel breakdown. status: CRITICAL | WARNING | OK | UNAVAILABLE | NOT_EVALUATED."""
    return {
        "kind": kind,
        "status": status,
        "value": None if value is None or math.isnan(value) else round(float(value), 4),
        "threshold": threshold,
        "can_trigger_critical": can_trigger_critical,
        "detail": detail,
    }


class RetrainingTriggerEngine:
    """
    Multi-Signal Retraining Decision Engine.

    Parameters
    ----------
    triad_psi_retrain_threshold : float, default=0.25
        Detrended drift-score threshold on high-priority triad features.
    triad_psi_extreme_threshold : float, default=1.0
        Extreme triad drift score where even a single feature mandates retraining.
    broad_local_drift_retrain_pct : float, default=30.0
        Percentage of 93 local features with PSI > 0.25 at which the feature-shift
        channel raises a WARNING. It can never raise CRITICAL on its own.
    adversarial_auc_retrain_threshold : float, default=0.85
        Adversarial AUC at which a corroborating (WARNING-only) alert is raised.
    prevalence_collapse_threshold : float, default=0.035
        Absolute rolling prevalence floor indicating regime collapse (label-based).
    local_persistence_steps : int, default=2
        Consecutive steps above the local-drift threshold that mark the feature-shift
        warning as persistent. Persistence changes the wording, not the severity.
    label_delay_steps : int, default=1
        L. Label-based inputs may contain only steps <= t - L; anything newer raises
        :class:`LagViolationError`.
    score_psi_warning_threshold, score_psi_critical_threshold : float
        Standard PSI bands (0.10 / 0.25) for the score-shift channel.

    The engine is stateful across calls (local-tier streak) and must be reset between
    independent backtests via :meth:`reset`.
    """

    def __init__(
        self,
        triad_psi_retrain_threshold: float = 0.25,
        triad_psi_extreme_threshold: float = 1.0,
        broad_local_drift_retrain_pct: float = 30.0,
        adversarial_auc_retrain_threshold: float = 0.85,
        prevalence_collapse_threshold: float = 0.035,
        local_persistence_steps: int = 2,
        label_delay_steps: int = 1,
        score_psi_warning_threshold: float = PSI_WARNING,
        score_psi_critical_threshold: float = PSI_CRITICAL,
    ):
        self.triad_psi_retrain_threshold = triad_psi_retrain_threshold
        self.triad_psi_extreme_threshold = triad_psi_extreme_threshold
        self.broad_local_drift_retrain_pct = broad_local_drift_retrain_pct
        self.adversarial_auc_retrain_threshold = adversarial_auc_retrain_threshold
        self.prevalence_collapse_threshold = prevalence_collapse_threshold
        self.local_persistence_steps = local_persistence_steps
        self.label_delay_steps = label_delay_steps
        self.score_psi_warning_threshold = score_psi_warning_threshold
        self.score_psi_critical_threshold = score_psi_critical_threshold

        # Sequential state for the local-tier persistence check.
        self._local_prev_pct: Optional[float] = None
        self._local_streak: int = 0

    def reset(self) -> None:
        """Reset sequential state (call once before a new backtest run)."""
        self._local_prev_pct = None
        self._local_streak = 0

    def _check_lag(self, channel: str, time_step: int, newest_label_step: Optional[int]) -> None:
        limit = time_step - self.label_delay_steps
        if newest_label_step is not None and newest_label_step > limit:
            raise LagViolationError(
                f"{channel}: input uses labels of step {newest_label_step}, but with "
                f"label_delay_steps={self.label_delay_steps} a decision at step {time_step} "
                f"may read steps <= {limit} only."
            )

    def evaluate(
        self,
        time_step: int,
        feature_report: Optional[FeatureDriftReport] = None,
        prevalence_report: Optional[PrevalenceDriftReport] = None,
        adversarial_report: Optional[AdversarialDriftReport] = None,
        threshold_report: Optional[ThresholdComparisonResult] = None,
        performance_report: Optional[PerformanceWindowReport] = None,
        score_shift_report: Optional[ScoreShiftReport] = None,
    ) -> RetrainingDecision:
        """
        Evaluate current monitoring telemetry and produce an auditable decision.

        Decision rules:
        1. CRITICAL RETRAIN (any of):
           - triad detrended drift score >= 1.0, or >= 0.25 on two features
           - score-shift PSI >= 0.25 (label-free)
           - lag-safe PerformanceCrash: worst of the last labelled steps' F1 < floor
           - rolling prevalence (labels <= t-L) <= collapse floor
        2. WARNING RECALIBRATE_ONLY: prevalence warning/alert, moderate triad drift,
           score-shift PSI in [0.10, 0.25), feature-shift (local %) >= 30%,
           adversarial AUC alerts.
        3. INFO NO_ACTION: everything stable or unavailable.

        ``threshold_report`` is accepted for same-step telemetry only. Its F1 uses the
        labels of step t, which have not arrived under any label delay, so it is never
        used for a decision; pass ``performance_report`` for the performance channel.
        """
        retrain_reasons: List[str] = []
        recalibrate_reasons: List[str] = []

        statuses: Dict[str, str] = {}
        telemetry: Dict[str, Any] = {"time_step": time_step, "label_delay_steps": self.label_delay_steps}
        channels: Dict[str, Dict[str, Any]] = {}

        # ---- Feature drift: triad (label-free, may be CRITICAL) and local tier (WARNING only)
        if feature_report is not None:
            statuses["feature_triad"] = feature_report.triad_drift_level.value
            statuses["feature_local"] = feature_report.local_drift_level.value
            statuses["feature_overall"] = feature_report.overall_drift_level.value

            telemetry["triad_psi"] = {k: v.psi for k, v in feature_report.triad_metrics.items()}
            telemetry["pct_local_drifted_sig"] = feature_report.pct_local_drifted_significant
            telemetry["pct_local_drifted_mod"] = feature_report.pct_local_drifted_moderate

            # NOTE: drift_score (detrended residual shift), not raw static-reference PSI,
            # which is permanently inflated for monotone cumulative features.
            scores = {k: v.drift_score for k, v in feature_report.triad_metrics.items()}
            n_triad_sig = sum(1 for v in scores.values() if v >= self.triad_psi_retrain_threshold)
            n_triad_extreme = sum(1 for v in scores.values() if v >= self.triad_psi_extreme_threshold)
            worst_triad = max(scores.values()) if scores else float("nan")

            if n_triad_extreme >= 1:
                worst_feat = max(scores.items(), key=lambda x: x[1])
                retrain_reasons.append(
                    f"ConcentratedMacroDrift: {worst_feat[0]} drift_score={worst_feat[1]:.2f} "
                    f">= {self.triad_psi_extreme_threshold:.2f} (extreme detrended aggregate shift)."
                )
                triad_status = "CRITICAL"
            elif n_triad_sig >= 2:
                retrain_reasons.append(
                    f"ConcentratedMacroDrift: {n_triad_sig} of 3 triad features in significant drift "
                    f"(drift_score >= {self.triad_psi_retrain_threshold:.2f})."
                )
                triad_status = "CRITICAL"
            elif feature_report.triad_drift_level == DriftLevel.MODERATE:
                recalibrate_reasons.append(
                    "FeatureDriftWarning: Triad features show moderate detrended-drift score in [0.10, 0.25]."
                )
                triad_status = "WARNING"
            else:
                triad_status = "OK"
            channels["triad"] = _channel(
                "label_free", triad_status, worst_triad, self.triad_psi_retrain_threshold,
                detail="worst detrended drift score across triad features",
            )

            # Feature-shift channel: WARNING only. A single-step ratio spiked at step 37 (false
            # alarm) and sat inside its pre-drift range at step 43, so it cannot carry CRITICAL.
            pct_local = feature_report.pct_local_drifted_significant
            if pct_local >= self.broad_local_drift_retrain_pct:
                if (
                    self._local_prev_pct is not None
                    and self._local_prev_pct >= self.broad_local_drift_retrain_pct
                ):
                    self._local_streak += 1
                else:
                    self._local_streak = 1
            else:
                self._local_streak = 0
            self._local_prev_pct = pct_local
            telemetry["local_drift_streak"] = self._local_streak

            if pct_local >= self.broad_local_drift_retrain_pct:
                persistence = (
                    f"for {self._local_streak} consecutive steps"
                    if self._local_streak >= self.local_persistence_steps
                    else "for a single step"
                )
                recalibrate_reasons.append(
                    f"DiffuseMicroDriftWarning: {pct_local:.1f}% of local features in significant drift "
                    f"{persistence} (>= {self.broad_local_drift_retrain_pct:.0f}%). "
                    "Feature-shift is a WARNING-only channel and cannot raise CRITICAL."
                )
                local_status = "WARNING"
            else:
                local_status = "OK"
            channels["feature_shift"] = _channel(
                "label_free", local_status, pct_local, self.broad_local_drift_retrain_pct,
                can_trigger_critical=False,
                detail=f"% of local features with PSI > 0.25; streak={self._local_streak}",
            )
        else:
            channels["triad"] = _channel("label_free", "NOT_EVALUATED")
            channels["feature_shift"] = _channel("label_free", "NOT_EVALUATED", can_trigger_critical=False)

        # ---- Score shift (label-free, may be CRITICAL)
        if score_shift_report is not None:
            statuses["score_shift"] = score_shift_report.level
            telemetry["score_psi"] = None if not score_shift_report.available else score_shift_report.psi
            telemetry["score_reference_steps"] = score_shift_report.reference_steps
            if not score_shift_report.available:
                channels["score_shift"] = _channel(
                    "label_free", "UNAVAILABLE", None, self.score_psi_critical_threshold,
                    detail=score_shift_report.note,
                )
            else:
                psi = score_shift_report.psi
                if psi >= self.score_psi_critical_threshold:
                    status = "CRITICAL"
                    retrain_reasons.append(
                        f"ScoreShift: PSI of the deployed model's score distribution = {psi:.3f} "
                        f">= {self.score_psi_critical_threshold:.2f} against labelled steps "
                        f"{score_shift_report.reference_steps} (label-free)."
                    )
                elif psi >= self.score_psi_warning_threshold:
                    status = "WARNING"
                    recalibrate_reasons.append(
                        f"ScoreShiftWarning: score PSI = {psi:.3f} in "
                        f"[{self.score_psi_warning_threshold:.2f}, {self.score_psi_critical_threshold:.2f})."
                    )
                else:
                    status = "OK"
                channels["score_shift"] = _channel(
                    "label_free", status, psi, self.score_psi_critical_threshold,
                    detail=f"reference steps {score_shift_report.reference_steps}",
                )
        else:
            channels["score_shift"] = _channel("label_free", "NOT_EVALUATED", None, self.score_psi_critical_threshold)

        # ---- Performance (label-dependent, lag-safe, may be CRITICAL)
        if performance_report is not None:
            self._check_lag("performance", time_step, performance_report.max_label_step_used)
            statuses["performance"] = (
                "UNAVAILABLE" if not performance_report.available
                else ("CRITICAL" if performance_report.breached else "OK")
            )
            telemetry["lag_safe_f1_min"] = None if not performance_report.available else performance_report.f1_min
            telemetry["lag_safe_f1_steps"] = performance_report.steps_used
            telemetry["f1_floor"] = performance_report.f1_floor
            if not performance_report.available:
                channels["performance"] = _channel(
                    "label_dependent", "UNAVAILABLE", None, performance_report.f1_floor,
                    detail=performance_report.note,
                )
            else:
                if performance_report.breached:
                    retrain_reasons.append(
                        f"PerformanceCrash: worst per-step F1 over labelled steps "
                        f"{performance_report.steps_used} = {performance_report.f1_min:.3f} "
                        f"< floor {performance_report.f1_floor:.3f} (labels <= step "
                        f"{performance_report.max_label_step_used}, label delay "
                        f"{self.label_delay_steps})."
                    )
                channels["performance"] = _channel(
                    "label_dependent",
                    "CRITICAL" if performance_report.breached else "OK",
                    performance_report.f1_min, performance_report.f1_floor,
                    detail=f"per-step F1 {performance_report.to_dict()['per_step_f1']}",
                )
        else:
            channels["performance"] = _channel("label_dependent", "NOT_EVALUATED")

        # ---- Prevalence (label-dependent, lag-checked, may be CRITICAL on collapse)
        if prevalence_report is not None:
            self._check_lag(
                "prevalence", time_step,
                max(prevalence_report.available_steps_in_window, default=None),
            )
            statuses["prevalence"] = prevalence_report.drift_level.value
            telemetry["rolling_prevalence"] = prevalence_report.rolling_prevalence
            telemetry["prevalence_rel_change_pct"] = prevalence_report.relative_change * 100.0
            rp = prevalence_report.rolling_prevalence
            if rp <= self.prevalence_collapse_threshold:
                retrain_reasons.append(
                    f"PrevalenceRegimeCollapse: Rolling illicit prevalence ({rp:.4f}, labels <= step "
                    f"{time_step - self.label_delay_steps}) plummeted below critical floor "
                    f"({self.prevalence_collapse_threshold:.4f})."
                )
                prev_status = "CRITICAL"
            elif prevalence_report.drift_level == PrevalenceDriftLevel.ALERT:
                recalibrate_reasons.append(
                    f"PrevalenceAlert: Rolling prevalence ({rp:.4f}) shifted by "
                    f"{prevalence_report.relative_change * 100.0:+.1f}% vs baseline."
                )
                prev_status = "WARNING"
            elif prevalence_report.drift_level == PrevalenceDriftLevel.WARNING:
                recalibrate_reasons.append(
                    f"PrevalenceWarning: Rolling prevalence ({rp:.4f}) shifted by "
                    f"{prevalence_report.relative_change * 100.0:+.1f}% vs baseline."
                )
                prev_status = "WARNING"
            else:
                prev_status = "OK"
            channels["prevalence"] = _channel(
                "label_dependent", prev_status, rp, self.prevalence_collapse_threshold,
                detail=f"window steps {prevalence_report.available_steps_in_window}",
            )
        else:
            channels["prevalence"] = _channel("label_dependent", "NOT_EVALUATED", None, self.prevalence_collapse_threshold)

        # ---- Adversarial (label-free, corroborating WARNING only)
        if adversarial_report is not None:
            statuses["adversarial"] = adversarial_report.drift_level.value
            telemetry["adversarial_auc"] = adversarial_report.adversarial_auc
            adv_status = "OK"
            if adversarial_report.adversarial_auc >= self.adversarial_auc_retrain_threshold:
                # Saturated, non-discriminating channel (~0.99 AUC at every step, healthy or not):
                # corroborating warning only, never a standalone trigger.
                recalibrate_reasons.append(
                    f"AdversarialOODAlert: Domain classifier AUC={adversarial_report.adversarial_auc:.4f} >= "
                    f"{self.adversarial_auc_retrain_threshold:.2f} (saturated domain separability; "
                    "corroborating signal only, not a standalone retrain trigger)."
                )
                adv_status = "WARNING"
            elif adversarial_report.drift_level == AdversarialDriftLevel.MODERATE:
                recalibrate_reasons.append(
                    f"AdversarialWarning: Domain classifier AUC={adversarial_report.adversarial_auc:.4f} "
                    "shows moderate divergence."
                )
                adv_status = "WARNING"
            channels["adversarial"] = _channel(
                "label_free", adv_status, adversarial_report.adversarial_auc,
                self.adversarial_auc_retrain_threshold, can_trigger_critical=False,
            )
        else:
            channels["adversarial"] = _channel(
                "label_free", "NOT_EVALUATED", None, self.adversarial_auc_retrain_threshold,
                can_trigger_critical=False,
            )

        # ---- Same-step diagnostics (hindsight, NOT used for any decision)
        if threshold_report is not None:
            telemetry["same_step_frozen_f1_diagnostic_only"] = threshold_report.frozen_metrics.f1
            telemetry["same_step_pr_auc_diagnostic_only"] = threshold_report.pr_auc

        # ---- Synthesize
        if retrain_reasons:
            action = TriggerAction.RETRAIN
            severity = AlertSeverity.CRITICAL
            primary = retrain_reasons[0]
            all_reasons = retrain_reasons + recalibrate_reasons
        elif recalibrate_reasons:
            action = TriggerAction.RECALIBRATE_ONLY
            severity = AlertSeverity.WARNING
            primary = recalibrate_reasons[0]
            all_reasons = recalibrate_reasons
        else:
            action = TriggerAction.NO_ACTION
            severity = AlertSeverity.INFO
            primary = "All monitoring channels within stable operational envelope."
            all_reasons = ["Nominal feature distributions, stable prevalence, and calibrated thresholds."]

        return RetrainingDecision(
            time_step=time_step,
            action=action,
            severity=severity,
            primary_reason=primary,
            reasons=all_reasons,
            component_statuses=statuses,
            telemetry_summary=telemetry,
            label_delay_steps=self.label_delay_steps,
            channels=channels,
        )
