"""
Model Loader and Inference Engine for BitcoinGraphGuard.

Loads the frozen production XGBoost model, feature definitions, and operating threshold
at startup. Provides input validation, probability scoring, binary classification,
and context-aware reliability assessments based on empirical regime benchmarks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.api.schemas import (
    MODEL_FEATURES,
    ConfidenceContext,
    PredictionOutput,
    TransactionInput,
)

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_MODEL_PATH = Path("results/xgboost/xgb_model_optimized.json")
DEFAULT_FEATURE_LIST_PATH = Path("results/xgboost/selected_features.json")
DEFAULT_METRICS_PATH = Path("results/temporal_inductive/per_step_metrics.csv")
DEFAULT_THRESHOLD = 0.435  # tau* operating threshold selected in Phase 2


class ModelLoadError(RuntimeError):
    """Raised when model artifacts cannot be loaded or validated."""


class ModelContainer:
    """
    Singleton container holding the loaded XGBoost model and configuration.

    Guarantees:
    - Loaded once at startup, not per-request.
    - Verified feature names, order, and count (exactly 165).
    - Verified inference execution with dummy input.
    - Historical reliability lookup mapping time steps to benchmark metrics.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        feature_list_path: str | Path = DEFAULT_FEATURE_LIST_PATH,
        metrics_path: str | Path | None = DEFAULT_METRICS_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_list_path = Path(feature_list_path)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.threshold = threshold

        self.model: xgb.Booster | None = None
        self.features: list[str] = []
        self.load_timestamp: str | None = None
        self.startup_validations_passed: bool = False
        self.step_metrics_lookup: dict[int, dict[str, float]] = {}
        self.using_fallback_metrics: bool = False
        self.metrics_fallback_reason: str = ""

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load and strictly validate all serving artifacts."""
        logger.info("Initializing BitcoinGraphGuard ModelContainer...")

        # 1. Validate and load feature list
        if not self.feature_list_path.exists():
            # Fallback check
            alt_path = Path("results/xgboost_v2/feature_list.json")
            if alt_path.exists():
                self.feature_list_path = alt_path
            else:
                raise ModelLoadError(
                    f"Feature list artifact not found at {self.feature_list_path} or {alt_path}"
                )

        try:
            with open(self.feature_list_path, "r", encoding="utf-8") as f:
                feature_data = json.load(f)
            if "features" in feature_data:
                self.features = feature_data["features"]
            elif "model_features" in feature_data:
                self.features = feature_data["model_features"]
            else:
                raise ValueError(
                    "Could not find 'features' or 'model_features' in JSON artifact."
                )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to parse feature list from {self.feature_list_path}: {e}"
            ) from e

        if len(self.features) != 165:
            raise ModelLoadError(
                f"Feature count mismatch: expected 165 features, got {len(self.features)} in {self.feature_list_path}"
            )

        if self.features != MODEL_FEATURES:
            # Check if set matches but order differs, or if distinct
            if set(self.features) != set(MODEL_FEATURES):
                missing = set(MODEL_FEATURES) - set(self.features)
                extra = set(self.features) - set(MODEL_FEATURES)
                raise ModelLoadError(
                    f"Feature set mismatch: missing={missing}, extra={extra}"
                )
            raise ModelLoadError(
                "Feature list order mismatch: the artifact lists the canonical 165 "
                "features in a different order than MODEL_FEATURES. Inference vectors "
                "are emitted in MODEL_FEATURES order, so using the loaded order would "
                "silently map values into the wrong DMatrix columns. Regenerate the "
                "feature artifact in canonical order."
            )

        # 2. Validate and load XGBoost model artifact
        if not self.model_path.exists():
            # Fallback check
            alt_model = Path("results/xgboost/xgb_model_baseline.json")
            if alt_model.exists():
                self.model_path = alt_model
            else:
                raise ModelLoadError(f"Model artifact not found at {self.model_path}")

        try:
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load XGBoost model from {self.model_path}: {e}"
            ) from e

        # 3. Model sanity check: test inference on a dummy 165-feature vector
        try:
            dummy_matrix = xgb.DMatrix(
                np.zeros((1, 165), dtype=np.float32),
                feature_names=self.features,
            )
            dummy_pred = self.model.predict(dummy_matrix)
            if len(dummy_pred) != 1 or not (0.0 <= float(dummy_pred[0]) <= 1.0):
                raise ValueError(f"Unexpected dummy prediction output: {dummy_pred}")
        except Exception as e:
            raise ModelLoadError(f"Model smoke test failed on dummy input: {e}") from e

        # 4. Load historical per-step metrics lookup if available
        self._load_step_metrics()

        self.load_timestamp = datetime.now(timezone.utc).isoformat()
        self.startup_validations_passed = True
        logger.info(
            "ModelContainer initialized successfully. Model: %s, Features: %d, Threshold: %.3f",
            self.model_path.name,
            len(self.features),
            self.threshold,
        )

    def _load_step_metrics(self) -> None:
        """Load empirical per-step metrics from Phase 5 benchmark artifacts."""
        reason: str | None = None

        if self.metrics_path is None:
            reason = "metrics artifact path is not configured"
        elif not self.metrics_path.exists():
            reason = f"file not found: {self.metrics_path}"
        else:
            try:
                import pandas as pd

                df = pd.read_csv(self.metrics_path)
                required_columns = {
                    "model",
                    "time_step",
                    "pr_auc",
                    "f1",
                    "precision",
                    "recall",
                }
                missing_columns = required_columns - set(df.columns)
                if missing_columns:
                    reason = (
                        f"schema mismatch in {self.metrics_path}: missing columns "
                        f"{sorted(missing_columns)}"
                    )
                else:
                    xgb_df = df[
                        df["model"].str.contains("XGBoost", case=False, na=False)
                    ]
                    if xgb_df.empty:
                        reason = f"schema mismatch in {self.metrics_path}: no XGBoost rows found"
                    else:
                        for _, row in xgb_df.iterrows():
                            step = int(row["time_step"])
                            self.step_metrics_lookup[step] = {
                                "pr_auc": float(row["pr_auc"])
                                if not pd.isna(row["pr_auc"])
                                else 0.0,
                                "f1": float(row["f1"])
                                if not pd.isna(row["f1"])
                                else 0.0,
                                "precision": float(row["precision"])
                                if not pd.isna(row["precision"])
                                else 0.0,
                                "recall": float(row["recall"])
                                if not pd.isna(row["recall"])
                                else 0.0,
                            }
            except Exception as e:  # noqa: BLE001 - fall back to verified benchmarks
                reason = f"parse error reading {self.metrics_path}: {e}"

        if reason is not None:
            self.using_fallback_metrics = True
            self.metrics_fallback_reason = reason
            logger.warning(
                "Per-step metrics unavailable (%s). Using hardcoded Phase 8a "
                "benchmark values for confidence_context instead of live per-step "
                "metrics from %s. Set / fix the metrics artifact to clear this.",
                reason,
                self.metrics_path,
            )

        # Fallback / verified benchmarks if file not loaded
        if self.using_fallback_metrics or not self.step_metrics_lookup:
            self.using_fallback_metrics = True
            if not self.metrics_fallback_reason:
                self.metrics_fallback_reason = "no per-step metrics loaded"
            # Benchmark values from reports/monitoring_backtest_report.md
            benchmarks = {
                35: {"pr_auc": 0.9934, "f1": 0.9596},
                36: {"pr_auc": 0.9946, "f1": 0.8354},
                37: {"pr_auc": 0.8893, "f1": 0.8116},
                38: {"pr_auc": 0.9532, "f1": 0.9182},
                39: {"pr_auc": 0.9608, "f1": 0.9102},
                40: {"pr_auc": 0.7678, "f1": 0.7716},
                41: {"pr_auc": 0.9641, "f1": 0.9442},
                42: {"pr_auc": 0.8935, "f1": 0.8477},
                43: {"pr_auc": 0.0365, "f1": 0.0000},
                44: {"pr_auc": 0.0372, "f1": 0.0727},
                45: {"pr_auc": 0.0070, "f1": 0.0000},
                46: {"pr_auc": 0.0925, "f1": 0.1333},
                47: {"pr_auc": 0.0476, "f1": 0.0000},
                48: {"pr_auc": 0.1810, "f1": 0.0500},
                49: {"pr_auc": 0.1823, "f1": 0.0282},
            }
            self.step_metrics_lookup.update(benchmarks)

    def get_confidence_context(self, time_step: int) -> ConfidenceContext:
        """
        Evaluate and return model reliability and regime context for a given time step.

        Regime Classification:
        - Steps 1..34: Historical training and validation period.
        - Steps 35..42: Stationary test regime. Historically reliable (PR-AUC 0.77-0.99, F1 0.77-0.96).
        - Steps 43..49: Severe drift & regime collapse. Degraded performance (PR-AUC 0.04-0.18, F1 < 0.13).
        - Steps > 49 or < 1: Live unmonitored / future step. Reliability unknown pending streaming telemetry.
        """
        metrics = self.step_metrics_lookup.get(time_step, {})
        pr_auc = metrics.get("pr_auc")
        f1 = metrics.get("f1")

        if 1 <= time_step <= 34:
            return ConfidenceContext(
                model_reliability="training_reference",
                regime="historical_training_window",
                reason=(
                    f"Time step {time_step} is within the historical training/validation window (steps 1-34). "
                    "Model parameters and decision threshold tau*=0.435 were derived from this period."
                ),
                historical_pr_auc=pr_auc,
                historical_f1=f1,
                recommendation="Historical reference data; expected high in-sample consistency.",
            )
        elif 35 <= time_step <= 42:
            return ConfidenceContext(
                model_reliability="reliable",
                regime="stationary_test_window",
                reason=(
                    f"Time step {time_step} falls within the stationary test envelope (steps 35-42). "
                    f"Historical performance on this regime is high (PR-AUC: {pr_auc:.4f}, F1: {f1:.4f})."
                    if pr_auc is not None and f1 is not None
                    else f"Time step {time_step} is in the verified stationary test window (steps 35-42)."
                ),
                historical_pr_auc=pr_auc,
                historical_f1=f1,
                recommendation="Operational inference reliable; predictions carry high historical fidelity.",
            )
        elif 43 <= time_step <= 49:
            return ConfidenceContext(
                model_reliability="degraded",
                regime="drift_collapse_regime",
                reason=(
                    f"Time step {time_step} falls in the documented regime collapse window (steps 43-49). "
                    "Severe covariate drift in aggregate features and prevalence drop (to 2.53%) cause "
                    f"severe ranking breakdown (historical PR-AUC: {pr_auc:.4f}, F1: {f1:.4f})."
                    if pr_auc is not None and f1 is not None
                    else f"Time step {time_step} is in the documented 43-49 regime shift and collapse."
                ),
                historical_pr_auc=pr_auc,
                historical_f1=f1,
                recommendation=(
                    "CRITICAL WARNING: Predictions in steps 43-49 operate under severe distribution shift. "
                    "High false positive / false negative risk. Manual investigation or model retraining mandated."
                ),
            )
        else:
            return ConfidenceContext(
                model_reliability="unknown",
                regime="live_unmonitored_step",
                reason=(
                    f"Time step {time_step} is outside the historical benchmark dataset range (steps 1-49). "
                    "Operational reliability is UNKNOWN pending live feature drift monitoring and label arrival."
                ),
                historical_pr_auc=None,
                historical_f1=None,
                recommendation=(
                    "Poll GET /monitoring/status to inspect real-time drift metrics and retraining triggers."
                ),
            )

    def predict_single(self, transaction: TransactionInput) -> PredictionOutput:
        """Predict illicit probability and classification for a single transaction."""
        if not self.model:
            raise ModelLoadError("Model is not initialized.")

        vector = transaction.to_feature_vector()
        feature_matrix = np.array([vector], dtype=np.float32)

        dmatrix = xgb.DMatrix(feature_matrix, feature_names=self.features)
        raw_prob = float(self.model.predict(dmatrix)[0])
        # Clamp probability strictly to [0.0, 1.0]
        prob = max(0.0, min(1.0, raw_prob))

        binary_class = 1 if prob >= self.threshold else 0
        label = "illicit" if binary_class == 1 else "licit"
        confidence_ctx = self.get_confidence_context(transaction.time_step)

        return PredictionOutput(
            tx_id=transaction.tx_id,
            time_step=transaction.time_step,
            probability=round(prob, 8),
            binary_classification=binary_class,
            threshold=self.threshold,
            label=label,
            confidence_context=confidence_ctx,
        )

    def predict_batch(
        self, transactions: list[TransactionInput]
    ) -> list[PredictionOutput]:
        """Predict illicit probabilities and classifications for a batch of transactions."""
        if not self.model:
            raise ModelLoadError("Model is not initialized.")

        if not transactions:
            return []

        vectors = [tx.to_feature_vector() for tx in transactions]
        feature_matrix = np.array(vectors, dtype=np.float32)

        dmatrix = xgb.DMatrix(feature_matrix, feature_names=self.features)
        raw_probs = self.model.predict(dmatrix)

        results: list[PredictionOutput] = []
        for tx, raw_prob in zip(transactions, raw_probs):
            prob = max(0.0, min(1.0, float(raw_prob)))
            binary_class = 1 if prob >= self.threshold else 0
            label = "illicit" if binary_class == 1 else "licit"
            confidence_ctx = self.get_confidence_context(tx.time_step)

            results.append(
                PredictionOutput(
                    tx_id=tx.tx_id,
                    time_step=tx.time_step,
                    probability=round(prob, 8),
                    binary_classification=binary_class,
                    threshold=self.threshold,
                    label=label,
                    confidence_context=confidence_ctx,
                )
            )

        return results


# Global container instance
_model_container: ModelContainer | None = None


def get_model_container() -> ModelContainer:
    """Access or lazily initialize the singleton ModelContainer."""
    global _model_container
    if _model_container is None:
        _model_container = ModelContainer()
    return _model_container
