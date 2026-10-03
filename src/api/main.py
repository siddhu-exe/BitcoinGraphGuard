"""
FastAPI Serving Application for BitcoinGraphGuard.

Provides high-throughput, memory-safe inference endpoints for the frozen production
XGBoost model (165 features, tau*=0.435) with context-aware reliability reporting
and active retraining monitoring integration.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.logging_utils import get_prediction_logger
from src.api.model_loader import ModelLoadError, get_model_container
from src.api.schemas import (
    BatchPredictionOutput,
    BatchTransactionInput,
    HealthResponse,
    MonitoringStatusResponse,
    PredictionOutput,
    TransactionInput,
)
from src.monitoring.retraining_trigger import (
    AlertSeverity,
    RetrainingDecision,
    TriggerAction,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bitcoingraphguard.api")

# Max batch size limit to protect machine memory (~5.6GB RAM constraint)
# A batch of 10,000 transactions with 165 float32 columns is ~6.6 MB in memory,
# ensuring inference runs well under 100 MB RAM overhead.
DEFAULT_MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10000"))

# Artifact holding the latest sequential monitoring decision produced by the
# Phase 8a backtest. Environment-overridable so tests can point at fixtures.
MONITORING_REPORT_PATH = Path(
    os.getenv("MONITORING_REPORT_PATH", "reports/monitoring_backtest_report.json")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: load artifacts once at startup."""
    logger.info("Starting BitcoinGraphGuard API service...")
    try:
        container = get_model_container()
        logger.info(
            "Production model loaded successfully. Model features: %d, Threshold: %.3f",
            len(container.features),
            container.threshold,
        )
    except Exception as e:
        logger.critical("Fatal: Failed to load model artifacts during startup: %s", e)
        raise

    # /monitoring/status reads this artifact lazily; fail at boot instead of
    # serving a permanent 503 if it was left out of the image.
    try:
        decision = load_latest_monitoring_decision()
        logger.info(
            "Monitoring report loaded: latest step %d, action %s",
            decision.time_step,
            decision.action.value,
        )
    except Exception as e:
        logger.critical("Fatal: Monitoring report unusable during startup: %s", e)
        raise

    yield

    logger.info("Shutting down BitcoinGraphGuard API service.")


app = FastAPI(
    title="BitcoinGraphGuard Inference Service",
    description=(
        "Production-grade tabular inference service for Bitcoin fraud detection on the Elliptic++ dataset. "
        "Serves the frozen XGBoost baseline (165 features, PR-AUC 0.8013 on steps 35-49) with "
        "built-in regime self-awareness (flagging the 43-49 covariate collapse) and live retraining telemetry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Phase 9 frontend / dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_timing_header(request: Request, call_next):
    """Attach unique request ID and calculate request duration."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom 422 handler that returns precise, actionable diagnostics naming failed features.
    """
    error_details = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_details.append(f"Field '{loc}': {msg}")

    detail_message = "; ".join(error_details)
    logger.warning("Validation error on %s: %s", request.url.path, detail_message)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed. Ensure exactly 165 numeric features are provided without NaN or Inf.",
            "details": error_details,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    """Handle model artifact unavailability."""
    logger.error("Model unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Service Unavailable",
            "message": f"Model inference engine is not ready: {exc}",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.get("/", tags=["Info"])
async def root():
    """Service metadata and routing overview."""
    return {
        "service": "BitcoinGraphGuard Inference Service",
        "version": "1.0.0",
        "phase": "Phase 8b - FastAPI Inference Service + Docker",
        "docs_url": "/docs",
        "health_url": "/health",
        "monitoring_url": "/monitoring/status",
        "predict_url": "/predict",
        "batch_predict_url": "/batch_predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint returning artifact versions, feature schema, and threshold.
    """
    try:
        container = get_model_container()
        return HealthResponse(
            status="healthy" if container.startup_validations_passed else "degraded",
            model_name="XGBoost Optimized (Frozen Phase 2)",
            model_type="xgboost.Booster",
            n_features=len(container.features),
            operating_threshold=container.threshold,
            startup_timestamp=container.load_timestamp or "unknown",
            startup_validations_passed=container.startup_validations_passed,
            using_fallback_metrics=container.using_fallback_metrics,
            metrics_fallback_reason=container.metrics_fallback_reason or None,
            artifact_paths={
                "model_path": str(container.model_path),
                "feature_list_path": str(container.feature_list_path),
                "metrics_path": str(container.metrics_path)
                if container.metrics_path
                else "none",
            },
        )
    except Exception as e:  # noqa: BLE001 - health probe must never crash the process
        logger.error("Health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {e}",
        )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict_single(
    transaction: TransactionInput,
    request: Request,
):
    """
    Score a single Bitcoin transaction for illicit activity.

    Parameters:
    - time_step: Integer time step (1..49 or live).
    - 165 features: Local_feature_1..93 and Aggregate_feature_1..72 (flat or inside `features` dict).

    Returns:
    - probability: Illicit probability score.
    - binary_classification: 1 (illicit) or 0 (licit) evaluated at tau*=0.435.
    - confidence_context: Operational reliability assessment (explicitly flags 43-49 regime collapse).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    container = get_model_container()

    start_t = time.perf_counter()
    output = container.predict_single(transaction)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    # Structured JSONL request logging
    pred_logger = get_prediction_logger()
    pred_logger.log_prediction(
        endpoint="/predict",
        request_id=request_id,
        tx_id=output.tx_id,
        time_step=output.time_step,
        features=transaction.to_feature_vector(),
        probability=output.probability,
        binary_classification=output.binary_classification,
        threshold=output.threshold,
        model_reliability=output.confidence_context.model_reliability,
        regime=output.confidence_context.regime,
        latency_ms=latency_ms,
    )

    return output


@app.post("/batch_predict", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(
    request: Request,
    payload: Annotated[BatchTransactionInput | list[TransactionInput], Body()],
):
    """
    Score a batch of Bitcoin transactions for illicit activity with memory safety guardrails.

    Accepts either `{"transactions": [...]}` or a raw list `[...]`.
    Enforces maximum batch size (default: 10,000) to protect memory constraints.
    """
    request_id = (
        getattr(request.state, "request_id", str(uuid.uuid4()))
        if request
        else str(uuid.uuid4())
    )
    container = get_model_container()

    transactions: list[TransactionInput] = (
        payload.transactions if isinstance(payload, BatchTransactionInput) else payload
    )

    if not transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction batch cannot be empty.",
        )

    if len(transactions) > DEFAULT_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size {len(transactions)} exceeds maximum allowed batch limit "
                f"of {DEFAULT_MAX_BATCH_SIZE} transactions (memory safety protection)."
            ),
        )

    start_t = time.perf_counter()
    predictions = container.predict_batch(transactions)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    # Structured batch logging
    pred_logger = get_prediction_logger()
    log_records = [
        {
            "tx_id": p.tx_id,
            "time_step": p.time_step,
            "input_feature_hash": pred_logger.compute_feature_hash(
                tx.to_feature_vector()
            ),
            "probability": p.probability,
            "binary_classification": p.binary_classification,
            "threshold": p.threshold,
            "model_reliability": p.confidence_context.model_reliability,
            "regime": p.confidence_context.regime,
        }
        for tx, p in zip(transactions, predictions)
    ]
    pred_logger.log_batch(
        endpoint="/batch_predict",
        request_id=request_id,
        records=log_records,
        latency_ms=latency_ms,
    )

    return BatchPredictionOutput(
        total_transactions=len(predictions),
        threshold=container.threshold,
        predictions=predictions,
    )


def load_latest_monitoring_decision(
    report_path: Path | None = None,
) -> RetrainingDecision:
    """
    Load the most recent monitoring decision produced by the Phase 8a backtest.

    The persisted decision is rehydrated through the monitoring engine's own
    ``RetrainingDecision``/``TriggerAction``/``AlertSeverity`` types so the
    serving layer never re-implements or hardcodes trigger logic. The stored
    decision is authoritative because ``RetrainingTriggerEngine`` is stateful
    across steps (local-tier persistence streak), so it cannot be faithfully
    recomputed in isolation from a single saved step.
    """
    path = Path(report_path) if report_path is not None else MONITORING_REPORT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Monitoring report artifact not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("simulation_steps") or []
    if not steps:
        raise ValueError(f"Monitoring report at {path} contains no simulation steps")

    latest = steps[-1]
    decision = latest.get("decision")
    if not decision:
        raise ValueError(f"Latest step in {path} is missing its 'decision' block")

    return RetrainingDecision(
        time_step=int(decision.get("time_step", latest.get("time_step"))),
        action=TriggerAction(decision["action"]),
        severity=AlertSeverity(decision["severity"]),
        primary_reason=decision.get("primary_reason", ""),
        reasons=list(decision.get("reasons", [])),
        component_statuses=dict(decision.get("component_statuses", {})),
        telemetry_summary=dict(decision.get("telemetry_summary", {})),
    )


@app.get(
    "/monitoring/status", response_model=MonitoringStatusResponse, tags=["Monitoring"]
)
async def get_monitoring_status():
    """
    Expose the latest retraining trigger decision and drift monitoring health.

    Reads the most recent decision from the Phase 8a monitoring backtest (via the
    canonical monitoring trigger types) so dashboards and on-call engineers can
    immediately detect when the serving model has entered a degraded regime.
    """
    try:
        decision = load_latest_monitoring_decision()
    except (OSError, ValueError, KeyError) as e:
        logger.error("Monitoring status unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Monitoring status unavailable: {e}",
        )

    return MonitoringStatusResponse(
        latest_monitored_step=decision.time_step,
        trigger_action=decision.action.value,
        severity=decision.severity.value,
        primary_reason=decision.primary_reason,
        requires_retraining=decision.action == TriggerAction.RETRAIN,
        requires_action=decision.action
        in (TriggerAction.RETRAIN, TriggerAction.RECALIBRATE_ONLY),
        component_statuses=decision.component_statuses,
        telemetry_summary=decision.telemetry_summary,
        all_reasons=decision.reasons,
    )
