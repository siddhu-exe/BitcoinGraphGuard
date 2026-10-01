"""
Pydantic Schemas for BitcoinGraphGuard Inference API.

Defines strict type models, feature schemas, input validation, and output responses
for the frozen production XGBoost model serving layer.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 165 Frozen Production Model Features: 93 Local + 72 Aggregate
LOCAL_FEATURES: list[str] = [f"Local_feature_{i}" for i in range(1, 94)]
AGGREGATE_FEATURES: list[str] = [f"Aggregate_feature_{i}" for i in range(1, 73)]
MODEL_FEATURES: list[str] = LOCAL_FEATURES + AGGREGATE_FEATURES

assert len(MODEL_FEATURES) == 165, f"Expected 165 features, got {len(MODEL_FEATURES)}"


class ConfidenceContext(BaseModel):
    """
    Model reliability and operational regime context for a given prediction.

    Ensures the serving API is explicitly self-aware of its historical reliability boundary,
    specifically the documented 43-49 regime shift and collapse.
    """

    model_reliability: str = Field(
        ...,
        description="Reliability status: 'reliable', 'degraded', 'unknown', or 'training_reference'",
        examples=["reliable", "degraded", "unknown"],
    )
    regime: str = Field(
        ...,
        description="Operational regime description",
        examples=[
            "stationary_test_window",
            "drift_collapse_regime",
            "live_unmonitored_step",
        ],
    )
    reason: str = Field(
        ...,
        description="Explanatory rationale for model reliability status",
    )
    historical_pr_auc: float | None = Field(
        None,
        description="Historical benchmark PR-AUC at this time step if available",
    )
    historical_f1: float | None = Field(
        None,
        description="Historical benchmark F1 score at tau*=0.435 if available",
    )
    recommendation: str = Field(
        ...,
        description="Operational action or caution recommendation",
    )


# Dynamically create TransactionFeatures schema with all 165 typed fields
# to provide OpenAPI schema generation while keeping code clean.
def _create_transaction_model() -> type[BaseModel]:
    """Generate dynamic Pydantic model with explicit 165 feature fields."""
    fields: dict[str, Any] = {
        "tx_id": (int | str | None, Field(None, description="Transaction ID")),
        "time_step": (
            int,
            Field(
                ...,
                alias="Time step",
                description="Time step of the transaction (1..49 or future live step)",
            ),
        ),
        "features": (
            dict[str, float] | None,
            Field(
                None,
                description="Optional dictionary mapping feature names to float values (as an alternative to flat fields)",
            ),
        ),
    }

    # Add all 165 features as optional in definition so both flat and dict formats are accepted
    for feat in MODEL_FEATURES:
        fields[feat] = (
            float | None,
            Field(None, description=f"Feature value for {feat}"),
        )

    return type(
        "TransactionInputBase",
        (BaseModel,),
        {
            "__annotations__": {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items()},
            "model_config": ConfigDict(
                populate_by_name=True,
                extra="ignore",
                json_schema_extra={
                    "example": {
                        "tx_id": "1813992",
                        "time_step": 35,
                        **{feat: 0.0 for feat in MODEL_FEATURES[:5]},
                        "Local_feature_6": -0.123,
                        "Aggregate_feature_1": 0.456,
                    }
                },
            ),
        },
    )


TransactionInputBase = _create_transaction_model()


class TransactionInput(TransactionInputBase):
    """
    Transaction input model for inference.

    Validates:
    - Exactly 165 features provided (via flat fields or `features` dict).
    - No NaN, Inf, or string non-numeric values in features.
    - Time step is a valid integer.
    """

    @model_validator(mode="before")
    @classmethod
    def validate_and_extract_features(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError(  # noqa: TRY004 - surfaced as a 422 validation error
                "Input data must be a dictionary/JSON object"
            )

        # Handle time step alias
        step = data.get("time_step") if "time_step" in data else data.get("Time step")
        if step is None:
            raise ValueError("Field 'time_step' (or 'Time step') is required.")

        try:
            int(step)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid time_step '{step}': must be an integer.")

        # Extract feature vector
        flat_missing = []
        nan_or_inf_features = []
        invalid_type_features = []
        feature_dict: dict[str, float] = {}

        nested_features = data.get("features")

        for feat in MODEL_FEATURES:
            val = None
            if (
                nested_features
                and isinstance(nested_features, dict)
                and feat in nested_features
            ):
                val = nested_features[feat]
            elif feat in data:
                val = data[feat]

            if val is None:
                flat_missing.append(feat)
                continue

            try:
                f_val = float(val)
                if math.isnan(f_val) or math.isinf(f_val):
                    nan_or_inf_features.append(feat)
                else:
                    feature_dict[feat] = f_val
            except (ValueError, TypeError):
                invalid_type_features.append(feat)

        errors = []
        if flat_missing:
            sample_missing = flat_missing[:5]
            more_str = (
                f" and {len(flat_missing) - 5} more" if len(flat_missing) > 5 else ""
            )
            errors.append(
                f"Missing {len(flat_missing)} required feature(s): {', '.join(sample_missing)}{more_str}"
            )
        if nan_or_inf_features:
            sample_nan = nan_or_inf_features[:5]
            more_str = (
                f" and {len(nan_or_inf_features) - 5} more"
                if len(nan_or_inf_features) > 5
                else ""
            )
            errors.append(
                f"Feature(s) contain NaN or Inf values: {', '.join(sample_nan)}{more_str}"
            )
        if invalid_type_features:
            sample_invalid = invalid_type_features[:5]
            more_str = (
                f" and {len(invalid_type_features) - 5} more"
                if len(invalid_type_features) > 5
                else ""
            )
            errors.append(
                f"Feature(s) have non-numeric values: {', '.join(sample_invalid)}{more_str}"
            )

        if errors:
            raise ValueError("; ".join(errors))

        # Store validated feature vector in data dictionary
        data["features"] = feature_dict
        return data

    def to_feature_vector(self) -> list[float]:
        """Return the ordered 165-element float vector matching model contract."""
        feat_dict = self.features or {}
        return [
            float(feat_dict.get(k, getattr(self, k, 0.0) or 0.0))
            for k in MODEL_FEATURES
        ]


class BatchTransactionInput(BaseModel):
    """Batch input schema with safety guardrail on maximum batch size."""

    transactions: list[TransactionInput] = Field(
        ...,
        description="List of transactions for batch inference",
    )

    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v: list[TransactionInput]) -> list[TransactionInput]:
        if not v:
            raise ValueError("Batch cannot be empty.")
        # Configurable batch limit to protect ~5.6GB laptop memory constraint
        max_batch = 10000
        if len(v) > max_batch:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum allowed batch limit of {max_batch} transactions."
            )
        return v


class PredictionOutput(BaseModel):
    """Inference output schema for a single transaction prediction."""

    tx_id: int | str | None = Field(None, description="Transaction identifier")
    time_step: int = Field(..., description="Time step of the transaction")
    probability: float = Field(
        ..., description="Predicted probability of illicit class (class 1)"
    )
    binary_classification: int = Field(
        ...,
        description="Binary prediction (1 = illicit, 0 = licit) at operating threshold tau*",
    )
    threshold: float = Field(
        ..., description="Operating decision threshold tau* applied"
    )
    label: str = Field(..., description="Classification label ('illicit' or 'licit')")
    confidence_context: ConfidenceContext = Field(
        ...,
        description="Self-aware operational reliability and regime context",
    )


class BatchPredictionOutput(BaseModel):
    """Inference output schema for batch predictions."""

    total_transactions: int = Field(..., description="Number of transactions processed")
    threshold: float = Field(
        ..., description="Operating threshold tau* applied across all predictions"
    )
    predictions: list[PredictionOutput] = Field(
        ..., description="List of per-transaction predictions"
    )


class HealthResponse(BaseModel):
    """System health and artifact load validation status."""

    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    model_name: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Architecture type (e.g. XGBoost)")
    n_features: int = Field(..., description="Number of model features expected (165)")
    operating_threshold: float = Field(
        ..., description="Frozen operating threshold tau*"
    )
    startup_timestamp: str = Field(
        ..., description="UTC timestamp when model was loaded"
    )
    startup_validations_passed: bool = Field(
        ..., description="Whether all startup sanity checks passed"
    )
    artifact_paths: dict[str, str] = Field(..., description="Paths to loaded artifacts")


class MonitoringStatusResponse(BaseModel):
    """
    Live retraining and drift monitoring status from the RetrainingTriggerEngine.

    Exposes the latest monitoring evaluation so dashboards and on-call engineers
    can immediately identify if the serving model is operating in a degraded regime.
    """

    latest_monitored_step: int = Field(
        ..., description="Most recent time step evaluated by monitoring"
    )
    trigger_action: str = Field(
        ...,
        description="Recommended action: 'NO_ACTION', 'RECALIBRATE_ONLY', or 'RETRAIN'",
    )
    severity: str = Field(
        ...,
        description="Alert severity tier: 'INFO', 'WARNING', or 'CRITICAL'",
    )
    primary_reason: str = Field(
        ..., description="Primary attributed explanation for the decision"
    )
    requires_retraining: bool = Field(
        ..., description="Whether urgent model retraining is mandated"
    )
    requires_action: bool = Field(
        ..., description="Whether either recalibration or retraining is required"
    )
    component_statuses: dict[str, str] = Field(
        ...,
        description="Per-channel monitoring health status (e.g. triad, prevalence, performance)",
    )
    telemetry_summary: dict[str, Any] = Field(
        ...,
        description="Summary numerical telemetry (prevalence, F1, PR-AUC, PSI)",
    )
    all_reasons: list[str] = Field(
        default_factory=list,
        description="All firing condition reasons for the latest step",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Evaluation query timestamp",
    )
