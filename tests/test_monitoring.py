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
from src.monitoring.lag_safe import (
    LabelledScoreHistory,
    LagSafePerformanceMonitor,
    LagViolationError,
    PerformanceWindowReport,
    ScoreShiftMonitor,
    derive_f1_floor,
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


def _local_report(step: int, pct: float) -> FeatureDriftReport:
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


def test_feature_shift_is_warning_only_never_critical():
    """The local-feature PSI channel is demoted: single-step or persistent, it is WARNING at most."""
    engine = RetrainingTriggerEngine()
    engine.reset()

    # Step 37 case: one spike, then reversal.
    d37 = engine.evaluate(time_step=37, feature_report=_local_report(37, 34.4))
    assert d37.action == TriggerAction.RECALIBRATE_ONLY
    assert d37.severity == AlertSeverity.WARNING
    d38 = engine.evaluate(time_step=38, feature_report=_local_report(38, 4.3))
    assert d38.action == TriggerAction.NO_ACTION

    # Even 5 consecutive exceedances (streak well past local_persistence_steps) stay WARNING.
    engine.reset()
    last = None
    for step in range(40, 45):
        last = engine.evaluate(time_step=step, feature_report=_local_report(step, 90.0))
        assert last.severity == AlertSeverity.WARNING
        assert last.action == TriggerAction.RECALIBRATE_ONLY
    assert last.telemetry_summary["local_drift_streak"] == 5
    assert "consecutive steps" in last.primary_reason
    ch = last.channels["feature_shift"]
    assert ch["status"] == "WARNING"
    assert ch["can_trigger_critical"] is False
    assert ch["kind"] == "label_free"


# =====================================================================
# 6. Lag-safe performance channel
# =====================================================================


def _step_data(n_pos: int, good: bool, n_neg: int = 200):
    """Labels + scores: a perfect model (F1 = 1.0) or a collapsed one (nothing flagged, F1 = 0)."""
    y = np.array([1] * n_pos + [0] * n_neg)
    if good:
        sc = np.where(y == 1, 0.9, 0.05)
    else:
        sc = np.full(len(y), 0.05)
    return y, sc.astype(float)


def test_derive_f1_floor_is_fraction_of_validation_f1():
    assert derive_f1_floor(0.9605, 0.5) == pytest.approx(0.48025)
    with pytest.raises(ValueError):
        derive_f1_floor(0.0)
    with pytest.raises(ValueError):
        derive_f1_floor(0.9, fraction=1.5)


def test_label_from_step_t_never_influences_decision_at_step_t():
    """Recording a catastrophic step t before evaluating t must change nothing; t+L sees it."""
    hist = LabelledScoreHistory(label_delay_steps=1)
    mon = LagSafePerformanceMonitor(hist, f1_floor=0.48, window_steps=2, min_positives=10)
    for step in (40, 41):
        hist.record(step, *_step_data(20, good=True))

    before = mon.evaluate(42)
    hist.record(42, *_step_data(20, good=False))  # step 42's own labels: F1 = 0
    after = mon.evaluate(42)
    assert after.to_dict() == before.to_dict()
    assert not after.breached
    assert 42 not in after.steps_used
    assert after.max_label_step_used == 41 <= 42 - 1

    # One step later (L = 1) the crash is visible and fires.
    at_43 = mon.evaluate(43)
    assert at_43.breached
    assert 42 in at_43.steps_used and at_43.f1_min == 0.0


def test_label_delay_is_configurable():
    """With L = 3 a crash at step 42 stays invisible until step 45."""
    hist = LabelledScoreHistory(label_delay_steps=3)
    mon = LagSafePerformanceMonitor(hist, f1_floor=0.48, window_steps=2, min_positives=10)
    for step in (39, 40, 41):
        hist.record(step, *_step_data(20, good=True))
    hist.record(42, *_step_data(20, good=False))
    assert not mon.evaluate(44).breached  # newest usable step is 41
    assert mon.evaluate(44).max_label_step_used == 41
    assert mon.evaluate(45).breached      # step 42 labels arrive at 45


def test_performance_window_is_step_weighted_not_pooled():
    """A big healthy step must not hide a crash at the next, small step."""
    hist = LabelledScoreHistory(label_delay_steps=1)
    mon = LagSafePerformanceMonitor(hist, f1_floor=0.48, window_steps=2, min_positives=10)
    hist.record(42, *_step_data(239, good=True, n_neg=1900))
    hist.record(43, *_step_data(24, good=False, n_neg=1350))

    # Pooled F1 over both steps would be well above the floor ...
    y = np.concatenate([hist.get(42)[0], hist.get(43)[0]])
    sc = np.concatenate([hist.get(42)[1], hist.get(43)[1]])
    assert LagSafePerformanceMonitor._f1(y, sc, 0.435) > 0.85
    # ... but the step-weighted window sees the crash.
    rep = mon.evaluate(44)
    assert rep.breached
    assert rep.f1_min == 0.0
    assert rep.per_step_f1[42] == 1.0


def test_performance_ignores_low_positive_steps_and_reports_unavailable():
    hist = LabelledScoreHistory(label_delay_steps=1)
    mon = LagSafePerformanceMonitor(hist, f1_floor=0.48, window_steps=2, min_positives=10)
    assert not mon.evaluate(35).available  # nothing labelled yet
    hist.record(45, *_step_data(5, good=False))  # 5 positives: too noisy to judge
    rep = mon.evaluate(46)
    assert not rep.available and not rep.breached


def test_engine_rejects_label_inputs_newer_than_t_minus_l():
    engine = RetrainingTriggerEngine(label_delay_steps=1)
    leaky = PerformanceWindowReport(
        time_step=43, label_delay_steps=1, max_label_step_allowed=42,
        steps_used=[43], per_step_f1={43: 0.0}, per_step_positives={43: 24},
        f1_floor=0.48, available=True, breached=True, f1_min=0.0, f1_mean=0.0,
    )
    with pytest.raises(LagViolationError):
        engine.evaluate(time_step=43, performance_report=leaky)

    leaky_prev = PrevalenceDriftReport(
        time_step=43, step_prevalence=0.0175, rolling_prevalence=0.01,
        baseline_prevalence=0.1158, relative_change=-0.9, absolute_change=-0.1,
        window_size=5, available_steps_in_window=[39, 40, 41, 42, 43],
        total_labeled_in_window=5000, total_illicit_in_window=50,
        drift_level=PrevalenceDriftLevel.ALERT, label_lag=0,
    )
    with pytest.raises(LagViolationError):
        engine.evaluate(time_step=43, prevalence_report=leaky_prev)


def test_engine_performance_crash_is_critical_with_lag_safe_report():
    hist = LabelledScoreHistory(label_delay_steps=1)
    mon = LagSafePerformanceMonitor(hist, f1_floor=0.48, window_steps=2, min_positives=10)
    hist.record(42, *_step_data(20, good=True))
    hist.record(43, *_step_data(20, good=False))
    engine = RetrainingTriggerEngine(label_delay_steps=1)

    d43 = engine.evaluate(time_step=43, performance_report=mon.evaluate(43))
    assert d43.action == TriggerAction.NO_ACTION  # step 43's own crash is not visible at 43
    d44 = engine.evaluate(time_step=44, performance_report=mon.evaluate(44))
    assert d44.action == TriggerAction.RETRAIN
    assert d44.primary_reason.startswith("PerformanceCrash")
    assert d44.channels["performance"]["kind"] == "label_dependent"
    assert d44.to_dict()["label_delay_steps"] == 1


# =====================================================================
# 7. Score-shift channel (label-free)
# =====================================================================


def _score_history(n_steps: int = 5, seed: int = 0) -> LabelledScoreHistory:
    rng = np.random.default_rng(seed)
    hist = LabelledScoreHistory(label_delay_steps=1)
    for step in range(35, 35 + n_steps):
        y = (rng.random(1000) < 0.09).astype(int)
        hist.record(step, y, rng.beta(1, 12, 1000))  # healthy: mass near 0
    return hist


def test_score_shift_stable_distribution_does_not_fire():
    hist = _score_history()
    rng = np.random.default_rng(1)
    rep = ScoreShiftMonitor(hist).evaluate(40, rng.beta(1, 12, 1000))
    assert rep.available and rep.psi < 0.10
    assert rep.level == "STABLE"
    dec = RetrainingTriggerEngine().evaluate(time_step=40, score_shift_report=rep)
    assert dec.action == TriggerAction.NO_ACTION


def test_score_shift_fires_critical_at_standard_025_band_without_labels():
    hist = _score_history()
    rng = np.random.default_rng(2)
    shifted = rng.beta(4, 3, 1000)  # scores migrate toward the middle/high range
    rep = ScoreShiftMonitor(hist).evaluate(40, shifted)
    assert rep.psi >= 0.25 and rep.level == "CRITICAL"
    assert rep.critical_threshold == 0.25

    dec = RetrainingTriggerEngine().evaluate(time_step=40, score_shift_report=rep)
    assert dec.action == TriggerAction.RETRAIN
    assert dec.severity == AlertSeverity.CRITICAL
    assert dec.primary_reason.startswith("ScoreShift")
    ch = dec.channels["score_shift"]
    assert ch["kind"] == "label_free" and ch["status"] == "CRITICAL" and ch["threshold"] == 0.25


def test_score_shift_warning_band_is_recalibrate_only():
    engine = RetrainingTriggerEngine()
    rep = ScoreShiftMonitor(_score_history()).evaluate(40, np.random.default_rng(3).beta(1, 12, 1000))
    rep.psi = 0.15  # inside [0.10, 0.25)
    dec = engine.evaluate(time_step=40, score_shift_report=rep)
    assert dec.action == TriggerAction.RECALIBRATE_ONLY
    assert dec.channels["score_shift"]["status"] == "WARNING"


def test_score_shift_reference_uses_labelled_steps_only_and_needs_two():
    hist = LabelledScoreHistory(label_delay_steps=1)
    mon = ScoreShiftMonitor(hist, reference_steps=5, min_reference_steps=2)
    scores = np.random.default_rng(4).beta(1, 12, 500)
    hist.record(35, np.zeros(500, dtype=int), scores)
    assert not mon.evaluate(36, scores).available  # only one labelled step
    hist.record(36, np.zeros(500, dtype=int), scores)
    rep37 = mon.evaluate(37, scores)
    assert rep37.available and rep37.reference_steps == [35, 36]
    # Step 37's own record is invisible at 37 (strictly <= t - L), so the reference is unchanged.
    hist.record(37, np.zeros(500, dtype=int), np.full(500, 0.95))
    assert mon.evaluate(37, scores).reference_steps == [35, 36]
    # Unavailable never fires.
    dec = RetrainingTriggerEngine().evaluate(
        time_step=36, score_shift_report=mon.evaluate(36, scores)
    )
    assert dec.action == TriggerAction.NO_ACTION
    assert dec.channels["score_shift"]["status"] == "UNAVAILABLE"
