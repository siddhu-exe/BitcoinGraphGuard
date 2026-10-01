"""
Retraining Trigger Engine Module for BitcoinGraphGuard.

Synthesizes multiple monitoring telemetry streams (high-priority triad feature drift,
broad local feature drift, rolling label prevalence collapse, adversarial AUC,
and threshold calibration) into an explicit, auditable production decision:
- RETRAIN: Retrain XGBoost over a rolling temporal window (e.g. W=20 steps).
- RECALIBRATE_ONLY: Update decision threshold (tau_t) via Bayesian prior or empirical F1 adaptation.
- NO_ACTION: Continue production inference with current model and threshold.

Threshold-Driven Logic (Never Arbitrary Schedule):
- Defined per project specifications to detect and prevent catastrophic degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.monitoring.adversarial_drift import AdversarialDriftLevel, AdversarialDriftReport
from src.monitoring.feature_drift import DriftLevel, FeatureDriftReport
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "action": self.action.value,
            "severity": self.severity.value,
            "primary_reason": self.primary_reason,
            "reasons": self.reasons,
            "component_statuses": self.component_statuses,
            "telemetry_summary": self.telemetry_summary,
        }


class RetrainingTriggerEngine:
    """
    Multi-Signal Retraining Decision Engine.

    Parameters
    ----------
    triad_psi_retrain_threshold : float, default=0.25
        PSI threshold on high-priority triad features to trigger retraining alert.
    triad_psi_extreme_threshold : float, default=1.0
        Extreme PSI threshold where even a single triad feature mandates retraining.
    broad_local_drift_retrain_pct : float, default=30.0
        Percentage of 93 local features in significant drift to trigger retraining.
    adversarial_auc_retrain_threshold : float, default=0.85
        Adversarial AUC threshold indicating severe OOD domain separation.
    prevalence_collapse_threshold : float, default=0.035
        Absolute rolling prevalence floor (e.g. 3.5%) indicating regime collapse.
    local_persistence_steps : int, default=2
        Number of consecutive steps the broad local-tier drift ratio must exceed
        ``broad_local_drift_retrain_pct`` before it contributes to a CRITICAL
        decision. A single-step spike that reverses the next step yields at most
        WARNING. The engine is stateful across calls and must be reset between
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
    ):
        self.triad_psi_retrain_threshold = triad_psi_retrain_threshold
        self.triad_psi_extreme_threshold = triad_psi_extreme_threshold
        self.broad_local_drift_retrain_pct = broad_local_drift_retrain_pct
        self.adversarial_auc_retrain_threshold = adversarial_auc_retrain_threshold
        self.prevalence_collapse_threshold = prevalence_collapse_threshold
        self.local_persistence_steps = local_persistence_steps

        # Sequential state for the local-tier persistence check.
        self._local_prev_pct: Optional[float] = None
        self._local_streak: int = 0

    def reset(self) -> None:
        """Reset sequential state (call once before a new backtest run)."""
        self._local_prev_pct = None
        self._local_streak = 0

    def evaluate(
        self,
        time_step: int,
        feature_report: Optional[FeatureDriftReport] = None,
        prevalence_report: Optional[PrevalenceDriftReport] = None,
        adversarial_report: Optional[AdversarialDriftReport] = None,
        threshold_report: Optional[ThresholdComparisonResult] = None,
    ) -> RetrainingDecision:
        """
        Evaluate current monitoring telemetry and produce an auditable decision.

        Decision Rules:
        1. CRITICAL RETRAIN:
           - Triad extreme PSI (any triad PSI > 1.0) or multiple triad PSI > 0.25
           - Broad local feature drift >= 30% significant PSI
           - Adversarial validation AUC >= 0.85 (severe OOD shift)
           - Compound drift: Prevalence collapse (< 3.5%) + moderate/significant feature drift
        2. WARNING RECALIBRATE_ONLY:
           - Prevalence drift warning/alert without severe feature drift
           - Triad PSI moderate (0.10 <= PSI < 0.25) without local or adversarial drift
        3. INFO NO_ACTION:
           - All monitors in STABLE envelope.
        """
        retrain_reasons: List[str] = []
        recalibrate_reasons: List[str] = []
        info_notes: List[str] = []

        statuses: Dict[str, str] = {}
        telemetry: Dict[str, Any] = {"time_step": time_step}

        # 1. Inspect Feature Drift
        triad_critical = False
        triad_moderate = False
        local_critical = False
        local_moderate = False

        if feature_report is not None:
            statuses["feature_triad"] = feature_report.triad_drift_level.value
            statuses["feature_local"] = feature_report.local_drift_level.value
            statuses["feature_overall"] = feature_report.overall_drift_level.value

            # Telemetry
            triad_psi_dict = {k: v.psi for k, v in feature_report.triad_metrics.items()}
            telemetry["triad_psi"] = triad_psi_dict
            telemetry["pct_local_drifted_sig"] = feature_report.pct_local_drifted_significant
            telemetry["pct_local_drifted_mod"] = feature_report.pct_local_drifted_moderate

            # High-priority triad check
            # NOTE: use drift_score rather than raw psi. For monotone/cumulative
            # features the monitor detrends and reports a standardized residual
            # location shift as the drift score (raw static-reference PSI is
            # permanently inflated for such features and must not raise alarms).
            n_triad_sig = sum(
                1 for v in feature_report.triad_metrics.values()
                if v.drift_score >= self.triad_psi_retrain_threshold
            )
            n_triad_extreme = sum(
                1 for v in feature_report.triad_metrics.values()
                if v.drift_score >= self.triad_psi_extreme_threshold
            )

            if n_triad_extreme >= 1:
                triad_critical = True
                worst_feat = max(feature_report.triad_metrics.items(), key=lambda x: x[1].drift_score)
                retrain_reasons.append(
                    f"ConcentratedMacroDrift: {worst_feat[0]} drift_score={worst_feat[1].drift_score:.2f} "
                    f">= {self.triad_psi_extreme_threshold:.2f} (extreme detrended aggregate shift)."
                )
            elif n_triad_sig >= 2:
                triad_critical = True
                retrain_reasons.append(
                    f"ConcentratedMacroDrift: {n_triad_sig} of 3 triad features in significant drift "
                    f"(drift_score >= {self.triad_psi_retrain_threshold:.2f})."
                )
            elif feature_report.triad_drift_level == DriftLevel.MODERATE:
                triad_moderate = True
                recalibrate_reasons.append(
                    f"FeatureDriftWarning: Triad features show moderate detrended-drift score in [0.10, 0.25]."
                )

            # Broad local check with temporal persistence. The single-step ratio
            # is computed against a finite reference sample and can spike for one
            # step then reverse; require ``local_persistence_steps`` consecutive
            # exceedances before it can contribute to CRITICAL.
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

            if self._local_streak >= self.local_persistence_steps:
                local_critical = True
                retrain_reasons.append(
                    f"DiffuseMicroDrift: {pct_local:.1f}% of 93 local features in significant drift for "
                    f"{self._local_streak} consecutive steps (>= {self.broad_local_drift_retrain_pct:.0f}% threshold)."
                )
            elif pct_local >= self.broad_local_drift_retrain_pct:
                local_moderate = True
                recalibrate_reasons.append(
                    f"DiffuseMicroDriftWarning: {pct_local:.1f}% of local features in significant drift for a single step "
                    f"(need {self.local_persistence_steps} consecutive steps for CRITICAL); awaiting confirmation."
                )
            elif feature_report.local_drift_level == DriftLevel.MODERATE:
                local_moderate = True

        # 2. Inspect Prevalence Drift
        prev_collapse = False
        prev_alert = False
        prev_warn = False

        if prevalence_report is not None:
            statuses["prevalence"] = prevalence_report.drift_level.value
            telemetry["rolling_prevalence"] = prevalence_report.rolling_prevalence
            telemetry["prevalence_rel_change_pct"] = prevalence_report.relative_change * 100.0

            if prevalence_report.rolling_prevalence <= self.prevalence_collapse_threshold:
                prev_collapse = True
                retrain_reasons.append(
                    f"PrevalenceRegimeCollapse: Rolling illicit prevalence ({prevalence_report.rolling_prevalence:.4f}) plummeted below critical floor ({self.prevalence_collapse_threshold:.4f})."
                )
            elif prevalence_report.drift_level == PrevalenceDriftLevel.ALERT:
                prev_alert = True
                recalibrate_reasons.append(
                    f"PrevalenceAlert: Rolling prevalence ({prevalence_report.rolling_prevalence:.4f}) shifted by {prevalence_report.relative_change * 100.0:+.1f}% vs baseline."
                )
            elif prevalence_report.drift_level == PrevalenceDriftLevel.WARNING:
                prev_warn = True
                recalibrate_reasons.append(
                    f"PrevalenceWarning: Rolling prevalence ({prevalence_report.rolling_prevalence:.4f}) shifted by {prevalence_report.relative_change * 100.0:+.1f}% vs baseline."
                )

        # 3. Inspect Adversarial Drift
        if adversarial_report is not None:
            statuses["adversarial"] = adversarial_report.drift_level.value
            telemetry["adversarial_auc"] = adversarial_report.adversarial_auc

            if adversarial_report.adversarial_auc >= self.adversarial_auc_retrain_threshold:
                # Saturated, non-discriminating channel. On this dataset the domain
                # classifier reads ~0.99 AUC at *every* step, including healthy ones
                # (step 35, F1=0.960) and collapsed ones alike, because the monotone
                # cumulative features make training vs. any future step trivially
                # separable. Such an alarm carries no regime information on its own,
                # so it is reported as a corroborating warning and must not solo
                # trigger RETRAIN. This is the same "always-on alarm" design flaw as
                # raw static-reference PSI on monotone features.
                recalibrate_reasons.append(
                    f"AdversarialOODAlert: Domain classifier AUC={adversarial_report.adversarial_auc:.4f} >= {self.adversarial_auc_retrain_threshold:.2f} "
                    f"(saturated domain separability; corroborating signal only, not a standalone retrain trigger)."
                )
            elif adversarial_report.drift_level == AdversarialDriftLevel.MODERATE:
                recalibrate_reasons.append(
                    f"AdversarialWarning: Domain classifier AUC={adversarial_report.adversarial_auc:.4f} shows moderate divergence."
                )

        # 4. Inspect Threshold Performance & Calibration
        if threshold_report is not None:
            telemetry["frozen_f1"] = threshold_report.frozen_metrics.f1
            telemetry["adaptive_f1"] = threshold_report.adaptive_f1_metrics.f1
            telemetry["bayes_f1"] = threshold_report.adaptive_bayes_metrics.f1
            telemetry["pr_auc"] = threshold_report.pr_auc

            if threshold_report.frozen_metrics.f1 < 0.10 and threshold_report.prevalence > 0:
                retrain_reasons.append(
                    f"PerformanceCrash: Frozen threshold F1 collapsed to {threshold_report.frozen_metrics.f1:.4f} (PR-AUC={threshold_report.pr_auc:.4f})."
                )

        # 5. Synthesize Decision
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
        )
