"""
Unit and Integration Test Suite for BitcoinGraphGuard Monitoring System.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.adversarial_drift import (
    AdversarialDriftLevel,
    AdversarialDriftMonitor,
)
from src.monitoring.feature_drift import (
    DriftLevel,
    FeatureDriftMonitor,
    FeatureDriftReport,
    calculate_ks,
    calculate_psi,
)
from src.monitoring.prevalence_drift import (
    PrevalenceDriftLevel,
    PrevalenceDriftMonitor,
    PrevalenceDriftReport,
)
from src.monitoring.retraining_trigger import (
    AlertSeverity,
    RetrainingTriggerEngine,
    TriggerAction,
)
from src.monitoring.threshold_calibration import (
    CalibrationMethod,
    ThresholdCalibrator,
    bayesian_prior_shift_threshold,
    compute_pr_auc,
    evaluate_predictions,
    find_optimal_f1_threshold,
)


# =====================================================================
# 1. Feature Drift Monitor Tests
# =====================================================================

def test_psi_synthetic_gaussian_calibration():
    """Verify calibrated PSI against known Gaussian distribution shifts."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 5000)

    # 0.0 sigma shift -> PSI should be < 0.02 (negligible/stable)
    tgt_0 = np.random.normal(0.0, 1, 5000)
    psi_0 = calculate_psi(ref, tgt_0)
    assert psi_0 < 0.02, f"Expected PSI < 0.02 for 0.0 sigma, got {psi_0}"

    # 0.5 sigma shift -> PSI in moderate/early-warning range (~0.15 - 0.30)
    tgt_05 = np.random.normal(0.5, 1, 5000)
    psi_05 = calculate_psi(ref, tgt_05)
    assert 0.15 <= psi_05 <= 0.35, f"Expected PSI ~0.20-0.30 for 0.5 sigma, got {psi_05}"

    # 1.0 sigma shift -> PSI in significant range (~0.70 - 1.10)
    tgt_10 = np.random.normal(1.0, 1, 5000)
    psi_10 = calculate_psi(ref, tgt_10)
    assert 0.70 <= psi_10 <= 1.20, f"Expected PSI ~0.80-1.00 for 1.0 sigma, got {psi_10}"

    # 2.0 sigma shift -> PSI in strong shift range (~2.5 - 4.0)
    tgt_20 = np.random.normal(2.0, 1, 5000)
    psi_20 = calculate_psi(ref, tgt_20)
    assert 2.5 <= psi_20 <= 4.5, f"Expected PSI ~3.0 for 2.0 sigma, got {psi_20}"


def test_psi_identical_distributions():
    np.random.seed(42)
    sample_a = np.random.normal(0, 1, 1000)
    sample_b = np.random.normal(0, 1, 1000)
    psi = calculate_psi(sample_a, sample_b, num_bins=10)
    assert psi < 0.05, f"Expected near-zero PSI for identical distributions, got {psi}"


def test_psi_shifted_distribution():
    np.random.seed(42)
    sample_ref = np.random.normal(0, 1, 1000)
    sample_shifted = np.random.normal(3, 1, 1000)
    psi = calculate_psi(sample_ref, sample_shifted, num_bins=10)
    assert psi > 0.25, f"Expected high PSI (>0.25) for 3-sigma shifted distribution, got {psi}"
    level = DriftLevel.from_psi(psi)
    assert level == DriftLevel.SIGNIFICANT


def test_psi_constant_and_edge_cases():
    ref_const = np.ones(500)
    tgt_const = np.ones(500)
    assert calculate_psi(ref_const, tgt_const) == 0.0

    # With NaNs
    ref_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 100)
    tgt_nan = np.array([1.0, 2.0, 3.0, np.nan, 5.0] * 100)
    psi_nan = calculate_psi(ref_nan, tgt_nan)
    assert np.isfinite(psi_nan)


def test_ks_statistic():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    tgt = np.random.normal(2, 1, 1000)
    stat, pval = calculate_ks(ref, tgt)
    assert stat > 0.5
    assert pval < 1e-10


def test_feature_drift_monitor_two_tier():
    np.random.seed(42)
    n_ref = 2000
    n_tgt = 500

    triad_cols = ["Aggregate_feature_10", "Aggregate_feature_43", "Aggregate_feature_8"]
    local_cols = [f"Local_feature_{i}" for i in range(1, 94)]

    # Build mock reference (steps 1-34)
    ref_data = {col: np.random.normal(0, 1, n_ref) for col in triad_cols + local_cols}
    ref_df = pd.DataFrame(ref_data)

    monitor = FeatureDriftMonitor(
        triad_features=triad_cols,
        local_features=local_cols,
        psi_warn_threshold=0.10,
        psi_alert_threshold=0.25,
    )
    monitor.fit(ref_df)
    assert monitor.is_fitted

    # 1. Stationary test step
    stat_data = {col: np.random.normal(0, 1, n_tgt) for col in triad_cols + local_cols}
    stat_df = pd.DataFrame(stat_data)
    stat_report = monitor.evaluate_step(stat_df, time_step=35)

    assert stat_report.triad_drift_level == DriftLevel.STABLE
    assert stat_report.local_drift_level == DriftLevel.STABLE
    assert stat_report.overall_drift_level == DriftLevel.STABLE
    assert stat_report.pct_local_drifted_significant < 10.0

    # 2. Shifted test step (macro drift in triad)
    drift_data = {col: np.random.normal(0, 1, n_tgt) for col in local_cols}
    for col in triad_cols:
        drift_data[col] = np.random.normal(2.5, 1, n_tgt)  # Severe shift
    drift_df = pd.DataFrame(drift_data)
    drift_report = monitor.evaluate_step(drift_df, time_step=43)

    assert drift_report.triad_drift_level == DriftLevel.SIGNIFICANT
    assert drift_report.overall_drift_level == DriftLevel.SIGNIFICANT
    assert len(drift_report.reasons) > 0


# =====================================================================
# 2. Prevalence Drift Monitor Tests
# =====================================================================

def test_prevalence_monitor_lag_and_window():
    baseline_prev = 0.1158
    monitor = PrevalenceDriftMonitor(
        baseline_prevalence=baseline_prev,
        window_size=3,
        label_lag=1,
        rel_warning_threshold=0.30,
        rel_alert_threshold=0.60,
        abs_collapse_threshold=0.035,
    )

    # Feed stationary steps 32, 33, 34
    monitor.record_step_labels(32, (1000, 115))  # 11.5%
    monitor.record_step_labels(33, (1000, 120))  # 12.0%
    monitor.record_step_labels(34, (1000, 110))  # 11.0%

    # Evaluate at step 35 (lag=1 means steps 32..34 available)
    report_35 = monitor.evaluate_drift(current_time_step=35)
    assert report_35.drift_level == PrevalenceDriftLevel.STABLE
    assert 0.10 <= report_35.rolling_prevalence <= 0.13
    assert report_35.available_steps_in_window == [32, 33, 34]

    # Now step 43 collapses to 1.75%
    monitor.record_step_labels(41, (1000, 100))
    monitor.record_step_labels(42, (1000, 110))
    monitor.record_step_labels(43, (1000, 17))   # 1.7%

    # At step 44 (lag=1: labels for 41, 42, 43 available)
    report_44 = monitor.evaluate_drift(current_time_step=44)
    # Rolling prevalence is (100+110+17)/3000 = 7.56% -> relative drop > 30%
    assert report_44.drift_level in (PrevalenceDriftLevel.WARNING, PrevalenceDriftLevel.ALERT)

    # After 3 low steps (43, 44, 45 all ~2%)
    monitor.record_step_labels(44, (1000, 15))
    monitor.record_step_labels(45, (1000, 10))

    # At step 46 (lag=1: steps 43, 44, 45 available)
    report_46 = monitor.evaluate_drift(current_time_step=46)
    assert report_46.drift_level == PrevalenceDriftLevel.ALERT
    assert report_46.rolling_prevalence < 0.035
    assert any("PrevalenceCollapse" in r or "PrevalenceAlert" in r for r in report_46.reasons)


# =====================================================================
# 3. Adversarial Drift Monitor Tests
# =====================================================================

def test_adversarial_drift_monitor():
    np.random.seed(42)
    features = [f"feat_{i}" for i in range(10)]

    ref_df = pd.DataFrame(np.random.normal(0, 1, (500, 10)), columns=features)
    tgt_stable_df = pd.DataFrame(np.random.normal(0, 1, (200, 10)), columns=features)
    tgt_shifted_df = pd.DataFrame(np.random.normal(2, 1, (200, 10)), columns=features)

    monitor = AdversarialDriftMonitor(
        features=features,
        subsample_size=300,
        n_folds=2,
        cadence_steps=5,
        warn_threshold=0.75,
        alert_threshold=0.85,
    )
    monitor.fit_reference(ref_df)

    # Test cadence checking
    assert monitor.should_run_at_step(35)
    assert not monitor.should_run_at_step(36)
    assert monitor.should_run_at_step(40)

    # Stable report
    report_stable = monitor.evaluate_step(tgt_stable_df, time_step=35)
    assert report_stable.adversarial_auc < 0.70
    assert report_stable.drift_level == AdversarialDriftLevel.STABLE

    # Shifted report
    report_shifted = monitor.evaluate_step(tgt_shifted_df, time_step=40)
    assert report_shifted.adversarial_auc > 0.85
    assert report_shifted.drift_level == AdversarialDriftLevel.ALERT


# =====================================================================
# 4. Threshold Calibration Tests
# =====================================================================

def test_bayesian_prior_shift_formula():
    # Base prevalence = 10%, base threshold = 0.50
    # Target prevalence drops to 2.5% (4x drop) -> odds ratio drops -> threshold increases
    tau_adj = bayesian_prior_shift_threshold(
        base_threshold=0.435,
        base_prevalence=0.1158,
        target_prevalence=0.0253,
    )
    assert tau_adj > 0.435, f"Expected higher threshold under prevalence drop, got {tau_adj}"
    assert 0.60 < tau_adj < 0.90


def test_optimal_f1_threshold():
    np.random.seed(42)
    # Synthetic scores: positives centered at 0.7, negatives at 0.3
    y_true = np.array([0] * 900 + [1] * 100)
    y_score = np.concatenate([np.random.beta(2, 5, 900), np.random.beta(5, 2, 100)])

    best_tau, best_f1 = find_optimal_f1_threshold(y_true, y_score)
    assert 0.30 <= best_tau <= 0.75
    assert best_f1 > 0.50


def test_threshold_calibrator_history_and_eval():
    calibrator = ThresholdCalibrator(
        frozen_threshold=0.435,
        baseline_prevalence=0.1158,
        window_size=3,
        label_lag=1,
    )

    # Step 35
    y_t35 = np.array([0] * 900 + [1] * 100)
    y_s35 = np.random.uniform(0, 1, 1000)
    res35 = calibrator.evaluate_and_compare(35, y_t35, y_s35)
    assert res35.frozen_metrics.threshold == 0.435
    assert res35.adaptive_f1_metrics.threshold == 0.435  # First step fallback

    # Step 36
    y_t36 = np.array([0] * 900 + [1] * 100)
    y_s36 = np.random.uniform(0, 1, 1000)
    res36 = calibrator.evaluate_and_compare(36, y_t36, y_s36)
    assert res36.time_step == 36


# =====================================================================
# 5. Retraining Trigger Tests
# =====================================================================

def test_retraining_trigger_decisions():
    engine = RetrainingTriggerEngine(
        triad_psi_retrain_threshold=0.25,
        triad_psi_extreme_threshold=1.0,
        broad_local_drift_retrain_pct=30.0,
        adversarial_auc_retrain_threshold=0.85,
        prevalence_collapse_threshold=0.035,
    )

    # 1. All stable -> NO_ACTION
    dec_stable = engine.evaluate(time_step=35)
    assert dec_stable.action == TriggerAction.NO_ACTION
    assert dec_stable.severity == AlertSeverity.INFO

    # 2. Moderate prevalence drop alone -> RECALIBRATE_ONLY
    prev_report_warn = PrevalenceDriftReport(
        time_step=38,
        step_prevalence=0.07,
        rolling_prevalence=0.075,
        baseline_prevalence=0.1158,
        relative_change=-0.35,
        absolute_change=-0.04,
        window_size=5,
        available_steps_in_window=[34, 35, 36, 37],
        total_labeled_in_window=4000,
        total_illicit_in_window=300,
        drift_level=PrevalenceDriftLevel.WARNING,
        label_lag=1,
        reasons=["PrevalenceWarning: -35%"],
    )
    dec_recal = engine.evaluate(time_step=38, prevalence_report=prev_report_warn)
    assert dec_recal.action == TriggerAction.RECALIBRATE_ONLY
    assert dec_recal.severity == AlertSeverity.WARNING

    # 3. Severe Macro Triad Drift -> RETRAIN
    feat_report_critical = FeatureDriftReport(
        time_step=43,
        n_samples=1370,
        triad_metrics={
            "Aggregate_feature_10": None,  # Mock
        },
        triad_drift_level=DriftLevel.SIGNIFICANT,
        local_metrics={},
        pct_local_drifted_moderate=15.0,
        pct_local_drifted_significant=5.0,
        local_drift_level=DriftLevel.STABLE,
        overall_drift_level=DriftLevel.SIGNIFICANT,
        reasons=["HighPriorityTriad: Aggregate_feature_43 PSI=11.84"],
    )
    # Give mock with extreme PSI
    from src.monitoring.feature_drift import SingleFeatureDrift
    feat_report_critical.triad_metrics = {
        "Aggregate_feature_43": SingleFeatureDrift(
            feature_name="Aggregate_feature_43",
            psi=11.84,
            ks_statistic=0.99,
            ks_p_value=0.0,
            level=DriftLevel.SIGNIFICANT,
            ref_mean=-0.51,
            target_mean=1.30,
            ref_std=0.8,
            target_std=0.8,
            method="detrended_linear_residual",
            drift_score=11.84,
        )
    }

    dec_retrain = engine.evaluate(time_step=43, feature_report=feat_report_critical)
    assert dec_retrain.action == TriggerAction.RETRAIN
    assert dec_retrain.severity == AlertSeverity.CRITICAL
    assert "ConcentratedMacroDrift" in dec_retrain.primary_reason


def test_monotone_feature_detrending_avoids_permanent_false_alarm():
    """Cumulative features on their expected trend must not read as drift."""
    np.random.seed(0)
    steps = np.arange(1, 35)
    rows = []
    for s in steps:
        n = 200
        # Monotone secular ramp (+0.05/step) plus small noise.
        monotone = 0.05 * s + np.random.normal(0, 0.01, n)
        # Genuinely stationary feature.
        stationary = np.random.normal(0, 1, n)
        rows.append(pd.DataFrame({"Time step": s, "Aggregate_feature_10": monotone, "Local_feature_1": stationary}))
    ref_df = pd.concat(rows, ignore_index=True)

    monitor = FeatureDriftMonitor(
        triad_features=["Aggregate_feature_10"],
        local_features=["Local_feature_1"],
        num_bins=10,
    )
    monitor.fit(ref_df)
    assert monitor.feature_methods_["Aggregate_feature_10"].startswith("detrended")
    assert monitor.feature_methods_["Local_feature_1"] == "static"

    def make_step(step, level_offset):
        n = 500
        monotone = 0.05 * step + level_offset + np.random.normal(0, 0.01, n)
        stationary = np.random.normal(0, 1, n)
        return pd.DataFrame({"Aggregate_feature_10": monotone, "Local_feature_1": stationary})

    # On-trend continuation: raw static PSI would be huge, detrended score ~0.
    on_trend_step = make_step(40, 0.0)
    # Naive static-reference comparison false-alarms on the secular trend alone.
    assert calculate_psi(ref_df["Aggregate_feature_10"].to_numpy(), on_trend_step["Aggregate_feature_10"].to_numpy()) > 0.25
    on_trend = monitor.evaluate_step(on_trend_step, time_step=40)
    m = on_trend.triad_metrics["Aggregate_feature_10"]
    assert m.drift_score < 0.25
    assert m.level in (DriftLevel.STABLE, DriftLevel.MODERATE)

    # Genuine level departure from the trend must be flagged.
    off_trend = monitor.evaluate_step(make_step(40, 1.0), time_step=40)
    m_off = off_trend.triad_metrics["Aggregate_feature_10"]
    assert m_off.drift_score > 0.25
    assert m_off.level == DriftLevel.SIGNIFICANT


def test_local_tier_persistence_suppresses_single_step_spike():
    """A one-step local-tier spike must not solo-trigger CRITICAL (the step-37 case)."""
    engine = RetrainingTriggerEngine()
    engine.reset()

    def _report(step: int, pct: float) -> FeatureDriftReport:
        return FeatureDriftReport(
            time_step=step,
            n_samples=1000,
            triad_metrics={},
            triad_drift_level=DriftLevel.STABLE,
            local_metrics={},
            pct_local_drifted_moderate=pct,
            pct_local_drifted_significant=pct,
            local_drift_level=DriftLevel.STABLE,
            overall_drift_level=DriftLevel.STABLE,
            reasons=[],
        )

    # Step 37 spikes to 34.4% with the previous step below threshold -> WARNING only.
    d37 = engine.evaluate(time_step=37, feature_report=_report(37, 34.4))
    assert d37.action == TriggerAction.RECALIBRATE_ONLY
    assert d37.severity == AlertSeverity.WARNING

    # Step 38 reverses to 4.3% -> back to NO_ACTION, never CRITICAL.
    d38 = engine.evaluate(time_step=38, feature_report=_report(38, 4.3))
    assert d38.action == TriggerAction.NO_ACTION

    # Two consecutive exceedances still escalate to CRITICAL.
    engine.reset()
    engine.evaluate(time_step=40, feature_report=_report(40, 31.0))
    d41 = engine.evaluate(time_step=41, feature_report=_report(41, 33.0))
    assert d41.action == TriggerAction.RETRAIN
    assert d41.severity == AlertSeverity.CRITICAL
