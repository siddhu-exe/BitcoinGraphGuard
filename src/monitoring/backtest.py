"""
Retrospective Backtest Simulation Module for BitcoinGraphGuard.

Simulates a live production deployment of the monitoring system walking step-by-step
from step 35 to 49 over real Elliptic++ data under strict no-future-lookahead constraints:
- Ingests streaming feature batches per step.
- Computes two-tier feature drift (Triad Aggregate 10, 43, 8 + 93 Broad Local features).
- Every label-based input (rolling prevalence, performance F1) reads labels of steps <= t - L only
  (``label_delay_steps``, default 1); step t's own labels are recorded AFTER its decision.
- Adds a label-free score-shift channel (PSI of the frozen model's scores vs labelled steps).
- Executes periodic adversarial validation on scheduled cadence (every 5 steps).
- Compares frozen threshold (tau*=0.435) against adaptive F1 and Bayesian recalibration.
- Synthesizes all streams into an auditable Retraining Decision per step.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.monitoring.adversarial_drift import AdversarialDriftMonitor, AdversarialDriftReport
from src.monitoring.feature_drift import FeatureDriftMonitor, FeatureDriftReport
from src.monitoring.lag_safe import (
    LabelledScoreHistory,
    LagSafePerformanceMonitor,
    PerformanceWindowReport,
    ScoreShiftMonitor,
    ScoreShiftReport,
    derive_f1_floor,
)
from src.monitoring.prevalence_drift import PrevalenceDriftMonitor, PrevalenceDriftReport
from src.monitoring.retraining_trigger import (
    AlertSeverity,
    RetrainingDecision,
    RetrainingTriggerEngine,
    TriggerAction,
)
from src.monitoring.threshold_calibration import (
    StepEvaluationMetrics,
    ThresholdCalibrator,
    ThresholdComparisonResult,
)

logger = logging.getLogger(__name__)


@dataclass
class StepSimulationRecord:
    """Full monitoring snapshot for a single backtest time step."""

    time_step: int
    n_transactions: int
    n_labeled: int
    n_illicit: int
    prevalence: float
    feature_report: FeatureDriftReport
    prevalence_report: PrevalenceDriftReport
    adversarial_report: Optional[AdversarialDriftReport]
    threshold_comparison: ThresholdComparisonResult
    decision: RetrainingDecision
    performance_report: Optional[PerformanceWindowReport] = None
    score_shift_report: Optional[ScoreShiftReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "n_transactions": self.n_transactions,
            "n_labeled": self.n_labeled,
            "n_illicit": self.n_illicit,
            "prevalence": round(self.prevalence, 4),
            "feature_drift": self.feature_report.to_dict(),
            "prevalence_drift": self.prevalence_report.to_dict(),
            "adversarial_drift": self.adversarial_report.to_dict() if self.adversarial_report else None,
            "threshold_comparison": self.threshold_comparison.to_dict(),
            "decision": self.decision.to_dict(),
            "lag_safe_performance": self.performance_report.to_dict() if self.performance_report else None,
            "score_shift": self.score_shift_report.to_dict() if self.score_shift_report else None,
        }


class MonitoringBacktester:
    """
    Orchestrates the step-by-step retrospective monitoring simulation.
    """

    def __init__(
        self,
        features_path: str = "Og data/txs_features.csv",
        predictions_path: str = "results/xgboost/predictions.csv",
        label_counts_path: str = "results/temporal_inductive/per_step_label_counts.csv",
        window_size: int = 5,
        label_delay_steps: int = 1,
        frozen_threshold: float = 0.435,
        baseline_prevalence: float = 0.115809,
        metrics_path: str = "results/xgboost/metrics.json",
        perf_floor_fraction: float = 0.5,
        perf_window_steps: int = 2,
        perf_min_positives: int = 10,
        score_reference_steps: int = 5,
    ):
        self.features_path = features_path
        self.predictions_path = predictions_path
        self.label_counts_path = label_counts_path
        self.window_size = window_size
        self.label_delay_steps = label_delay_steps
        self.metrics_path = metrics_path
        self.frozen_threshold = frozen_threshold
        self.baseline_prevalence = baseline_prevalence

        # Initialize monitoring sub-components
        self.feature_monitor = FeatureDriftMonitor()
        self.prevalence_monitor = PrevalenceDriftMonitor(
            baseline_prevalence=baseline_prevalence,
            window_size=window_size,
            label_lag=label_delay_steps,
        )
        self.adversarial_monitor = AdversarialDriftMonitor(
            cadence_steps=5,
            subsample_size=1200,
            n_folds=3,
        )
        self.threshold_calibrator = ThresholdCalibrator(
            frozen_threshold=frozen_threshold,
            baseline_prevalence=baseline_prevalence,
            window_size=window_size,
            label_lag=label_delay_steps,
        )
        self.trigger_engine = RetrainingTriggerEngine(label_delay_steps=label_delay_steps)

        # Lag-safe channels. The F1 floor is a fixed fraction of the validation-period F1 of the
        # deployed operating point (read from the Phase 2 metrics artifact); nothing from steps
        # 43-49 enters it. Fails loudly if the artifact is missing (no silent fallback).
        self.validation_f1 = self.load_validation_f1(metrics_path)
        self.perf_floor_fraction = perf_floor_fraction
        self.f1_floor = derive_f1_floor(self.validation_f1, perf_floor_fraction)
        self.label_history = LabelledScoreHistory(label_delay_steps=label_delay_steps)
        self.performance_monitor = LagSafePerformanceMonitor(
            self.label_history,
            f1_floor=self.f1_floor,
            threshold=frozen_threshold,
            window_steps=perf_window_steps,
            min_positives=perf_min_positives,
        )
        self.score_shift_monitor = ScoreShiftMonitor(
            self.label_history, reference_steps=score_reference_steps
        )

        self.simulation_records: List[StepSimulationRecord] = []

    @staticmethod
    def load_validation_f1(metrics_path: str, model_key: str = "xgboost_optimized") -> float:
        """Validation-window (steps 25-34) F1 of the deployed operating point, from the Phase 2 artifact.

        Only the pooled validation F1 is stored (per-step validation F1 is not saved), so the
        floor is a fraction of the pooled value. Reported as such in the backtest report.
        """
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return float(metrics["operating_points"][model_key]["validation_f1"])

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], pd.DataFrame]:
        """
        Stream features from Og data/txs_features.csv in chunks to minimize memory footprint.

        Returns
        -------
        Tuple[reference_df, test_steps_dict, predictions_df]
        """
        logger.info("Loading predictions from %s", self.predictions_path)
        pred_df = pd.read_csv(self.predictions_path)

        cols_to_read = ["txId", "Time step"] + self.feature_monitor.all_monitored_features

        cache_path = "results/monitoring/features_cache.joblib"
        if os.path.exists(cache_path):
            logger.info("Loading cached feature datasets from %s", cache_path)
            import joblib
            cached_data = joblib.load(cache_path)
            return cached_data["reference_df"], cached_data["test_steps_dict"], pred_df

        logger.info("Streaming feature data from %s...", self.features_path)
        ref_chunks: List[pd.DataFrame] = []
        test_step_chunks: Dict[int, List[pd.DataFrame]] = {step: [] for step in range(35, 50)}

        # Stream chunk by chunk (chunksize 50,000 ~ 70MB memory)
        chunks = pd.read_csv(
            self.features_path,
            usecols=cols_to_read,
            chunksize=50000,
        )

        for chunk in chunks:
            # Training reference (steps 1-34)
            tr_mask = chunk["Time step"] <= 34
            if tr_mask.any():
                # Subsample reference chunk to avoid memory bloat
                sub = chunk[tr_mask]
                if len(sub) > 2000:
                    sub = sub.sample(n=2000, random_state=42)
                ref_chunks.append(sub)

            # Test steps (35-49)
            for step in range(35, 50):
                st_mask = chunk["Time step"] == step
                if st_mask.any():
                    test_step_chunks[step].append(chunk[st_mask])

        reference_df = pd.concat(ref_chunks, ignore_index=True)
        test_steps_dict = {
            step: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            for step, dfs in test_step_chunks.items()
        }

        logger.info(
            "Reference dataset assembled: %d rows. Test steps 35-49 prepared.",
            len(reference_df),
        )

        try:
            import joblib
            os.makedirs("results/monitoring", exist_ok=True)
            joblib.dump({"reference_df": reference_df, "test_steps_dict": test_steps_dict}, cache_path)
            logger.info("Cached features saved to %s", cache_path)
        except Exception as e:
            logger.warning("Could not cache features: %s", e)
        return reference_df, test_steps_dict, pred_df

    def initialize_history(self, reference_df: pd.DataFrame) -> None:
        """Fit baseline references and prime historical label counts."""
        logger.info("Fitting feature drift monitor on reference distribution...")
        self.feature_monitor.fit(reference_df)

        logger.info("Fitting adversarial monitor reference buffer...")
        self.adversarial_monitor.fit_reference(reference_df)

        # Prime historical label counts for steps 30-34
        # Derived from exact ground-truth training labels (Og data/txs_classes.csv)
        historical_counts = {
            30: (524, 83),
            31: (710, 106),
            32: (1323, 342),
            33: (441, 23),
            34: (515, 37),
        }
        for s, counts in historical_counts.items():
            self.prevalence_monitor.record_step_labels(s, counts)

    def run_backtest(self) -> List[StepSimulationRecord]:
        """
        Execute retrospective backtest simulation over test steps 35 through 49.
        """
        t0 = time.time()
        ref_df, test_steps_dict, pred_df = self.load_and_prepare_data()
        self.initialize_history(ref_df)

        self.simulation_records.clear()
        self.trigger_engine.reset()
        self.label_history = LabelledScoreHistory(label_delay_steps=self.label_delay_steps)
        self.performance_monitor.history = self.label_history
        self.score_shift_monitor.history = self.label_history
        logger.info("Starting step-by-step retrospective monitoring loop (steps 35-49)...")

        for time_step in range(35, 50):
            step_feats = test_steps_dict[time_step]
            step_preds = pred_df[pred_df["time_step"] == time_step].copy()

            n_tx = len(step_feats)
            n_labeled = len(step_preds)
            n_illicit = int((step_preds["y_true"] == 1).sum()) if n_labeled > 0 else 0
            prev = float(n_illicit / max(1, n_labeled))

            # 1. Feature Drift Evaluation
            feat_report = self.feature_monitor.evaluate_step(step_feats, time_step=time_step)

            # 2. Prevalence Drift Evaluation (Lag-Aware)
            # Reads labels of steps <= t - L only; step t's labels are recorded after the decision.
            prev_report = self.prevalence_monitor.evaluate_drift(current_time_step=time_step)

            # 2b. Lag-safe performance (labels <= t - L) and score shift (no labels of step t).
            scores_t = step_preds["xgboost_score"].to_numpy()
            perf_report = self.performance_monitor.evaluate(time_step)
            shift_report = self.score_shift_monitor.evaluate(time_step, scores_t)

            # 3. Adversarial Validation (Scheduled on Cadence)
            adv_report = None
            if self.adversarial_monitor.should_run_at_step(time_step):
                adv_report = self.adversarial_monitor.evaluate_step(step_feats, time_step=time_step)

            # 4. Threshold Calibration & Evaluation
            thresh_result = self.threshold_calibrator.evaluate_and_compare(
                current_time_step=time_step,
                y_true_current=step_preds["y_true"].to_numpy(),
                y_score_current=step_preds["xgboost_score"].to_numpy(),
                rolling_prevalence=prev_report.rolling_prevalence,
            )

            # 5. Retraining Trigger Decision
            decision = self.trigger_engine.evaluate(
                time_step=time_step,
                feature_report=feat_report,
                prevalence_report=prev_report,
                adversarial_report=adv_report,
                threshold_report=thresh_result,
                performance_report=perf_report,
                score_shift_report=shift_report,
            )

            # Record step t's labels/scores for FUTURE steps only (visible from t + L).
            self.prevalence_monitor.record_step_labels(time_step, (n_labeled, n_illicit))
            self.label_history.record(time_step, step_preds["y_true"].to_numpy(), scores_t)

            record = StepSimulationRecord(
                time_step=time_step,
                n_transactions=n_tx,
                n_labeled=n_labeled,
                n_illicit=n_illicit,
                prevalence=prev,
                feature_report=feat_report,
                prevalence_report=prev_report,
                adversarial_report=adv_report,
                threshold_comparison=thresh_result,
                decision=decision,
                performance_report=perf_report,
                score_shift_report=shift_report,
            )
            self.simulation_records.append(record)

            logger.info(
                "Step %d complete: Action=%s (%s), Triad Level=%s, Prev=%.3f, Frozen F1=%.3f, Adapt F1=%.3f",
                time_step,
                decision.action.value,
                decision.severity.value,
                feat_report.triad_drift_level.value,
                prev,
                thresh_result.frozen_metrics.f1,
                thresh_result.adaptive_f1_metrics.f1,
            )

        elapsed = time.time() - t0
        logger.info("Backtest finished in %.2f seconds.", elapsed)
        return self.simulation_records

    def export_summary_tables(self, output_dir: str = "results/monitoring") -> Dict[str, pd.DataFrame]:
        """
        Export tabular dataframes summarizing the backtest simulation.
        """
        os.makedirs(output_dir, exist_ok=True)

        rows_general = []
        rows_triad = []
        rows_thresholds = []

        for r in self.simulation_records:
            t = r.time_step
            # General step metrics
            f10 = r.feature_report.triad_metrics.get("Aggregate_feature_10")
            f43 = r.feature_report.triad_metrics.get("Aggregate_feature_43")
            f8 = r.feature_report.triad_metrics.get("Aggregate_feature_8")

            rows_general.append({
                "time_step": t,
                "n_tx": r.n_transactions,
                "n_labeled": r.n_labeled,
                "n_illicit": r.n_illicit,
                "prevalence": r.prevalence,
                "rolling_prevalence": r.prevalence_report.rolling_prevalence,
                "prevalence_status": r.prevalence_report.drift_level.value,
                "triad_status": r.feature_report.triad_drift_level.value,
                "psi_agg_10": f10.psi if f10 else np.nan,
                "psi_agg_43": f43.psi if f43 else np.nan,
                "psi_agg_8": f8.psi if f8 else np.nan,
                "score_agg_10": f10.drift_score if f10 else np.nan,
                "score_agg_43": f43.drift_score if f43 else np.nan,
                "score_agg_8": f8.drift_score if f8 else np.nan,
                "triad_method": f10.method if f10 else "",
                "pct_local_drift_sig": r.feature_report.pct_local_drifted_significant,
                "adversarial_auc": r.adversarial_report.adversarial_auc if r.adversarial_report else np.nan,
                "pr_auc": r.threshold_comparison.pr_auc,
                "frozen_f1": r.threshold_comparison.frozen_metrics.f1,
                "adaptive_f1_f1": r.threshold_comparison.adaptive_f1_metrics.f1,
                "bayes_f1": r.threshold_comparison.adaptive_bayes_metrics.f1,
                "oracle_f1": r.threshold_comparison.oracle_metrics.f1,
                "lag_safe_f1_min": r.performance_report.f1_min if r.performance_report else np.nan,
                "lag_safe_f1_steps": ",".join(str(x) for x in r.performance_report.steps_used) if r.performance_report else "",
                "score_psi": r.score_shift_report.psi if r.score_shift_report else np.nan,
                "performance_status": r.decision.channels.get("performance", {}).get("status", ""),
                "score_shift_status": r.decision.channels.get("score_shift", {}).get("status", ""),
                "prevalence_channel_status": r.decision.channels.get("prevalence", {}).get("status", ""),
                "feature_shift_status": r.decision.channels.get("feature_shift", {}).get("status", ""),
                "critical_channels": ",".join(
                    k for k, v in r.decision.channels.items() if v.get("status") == "CRITICAL"
                ),
                "trigger_action": r.decision.action.value,
                "trigger_severity": r.decision.severity.value,
                "primary_reason": r.decision.primary_reason,
            })

            # Triad breakdown
            for feat_name, sf in r.feature_report.triad_metrics.items():
                rows_triad.append({
                    "time_step": t,
                    "feature": feat_name,
                    "method": sf.method,
                    "psi": sf.psi,
                    "drift_score": sf.drift_score,
                    "ks_stat": sf.ks_statistic,
                    "ks_pval": sf.ks_p_value,
                    "level": sf.level.value,
                    "target_mean": sf.target_mean,
                    "ref_mean": sf.ref_mean,
                })

            # Threshold comparisons
            tc = r.threshold_comparison
            rows_thresholds.append(tc.to_dict())

        df_general = pd.DataFrame(rows_general)
        df_triad = pd.DataFrame(rows_triad)
        df_thresholds = pd.DataFrame(rows_thresholds)

        df_general.to_csv(os.path.join(output_dir, "step_monitoring_metrics.csv"), index=False)
        df_triad.to_csv(os.path.join(output_dir, "triad_drift_summary.csv"), index=False)
        df_thresholds.to_csv(os.path.join(output_dir, "threshold_comparison_summary.csv"), index=False)

        return {
            "general": df_general,
            "triad": df_triad,
            "thresholds": df_thresholds,
        }
