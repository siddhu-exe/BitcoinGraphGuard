"""
Monitoring package for BitcoinGraphGuard.

Provides automated feature drift monitoring, rolling prevalence tracking,
periodic adversarial domain validation, adaptive threshold calibration,
and auditable retraining triggers.
"""

from src.monitoring.adversarial_drift import (
    AdversarialDriftLevel,
    AdversarialDriftMonitor,
    AdversarialDriftReport,
)
from src.monitoring.feature_drift import (
    DriftLevel,
    FeatureDriftMonitor,
    FeatureDriftReport,
    SingleFeatureDrift,
    calculate_ks,
    calculate_psi,
)
from src.monitoring.prevalence_drift import (
    PrevalenceDriftLevel,
    PrevalenceDriftMonitor,
    PrevalenceDriftReport,
    StepLabelStats,
)
from src.monitoring.retraining_trigger import (
    AlertSeverity,
    RetrainingDecision,
    RetrainingTriggerEngine,
    TriggerAction,
)
from src.monitoring.threshold_calibration import (
    CalibrationMethod,
    StepEvaluationMetrics,
    ThresholdCalibrator,
    ThresholdComparisonResult,
    bayesian_prior_shift_threshold,
    compute_pr_auc,
    evaluate_predictions,
    find_optimal_f1_threshold,
)

__all__ = [
    "DriftLevel",
    "SingleFeatureDrift",
    "FeatureDriftReport",
    "FeatureDriftMonitor",
    "calculate_psi",
    "calculate_ks",
    "PrevalenceDriftLevel",
    "StepLabelStats",
    "PrevalenceDriftReport",
    "PrevalenceDriftMonitor",
    "AdversarialDriftLevel",
    "AdversarialDriftReport",
    "AdversarialDriftMonitor",
    "CalibrationMethod",
    "StepEvaluationMetrics",
    "ThresholdComparisonResult",
    "ThresholdCalibrator",
    "compute_pr_auc",
    "evaluate_predictions",
    "find_optimal_f1_threshold",
    "bayesian_prior_shift_threshold",
    "TriggerAction",
    "AlertSeverity",
    "RetrainingDecision",
    "RetrainingTriggerEngine",
]
