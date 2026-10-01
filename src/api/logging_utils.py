"""
Structured Prediction Logger for BitcoinGraphGuard Serving API.

Logs every inference request to a newline-delimited JSON (JSONL) file.
Captures timestamps, input hashes, time steps, output probabilities, decisions,
and reliability regimes for downstream audit, drift monitoring, and Phase 9 dashboard consumption.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))


class StructuredPredictionLogger:
    """
    Thread-safe structured prediction logger emitting JSON lines.

    Schema:
    - timestamp: ISO 8601 UTC timestamp
    - request_id: Unique UUID identifying request batch
    - endpoint: API endpoint invoked (/predict, /batch_predict)
    - tx_id: Transaction identifier if provided
    - time_step: Transaction time step
    - input_feature_hash: SHA-256 hash of feature float values for data integrity
    - probability: Illicit probability score
    - binary_classification: Binary class (0 or 1)
    - threshold: Decision threshold applied (tau*=0.435)
    - model_reliability: Self-aware reliability state (reliable, degraded, unknown)
    - regime: Operational regime name
    - latency_ms: Inference execution latency in milliseconds
    """

    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create log directory if it does not exist."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001 - logging must never break inference
            logger.error("Failed to create log directory for %s: %s", self.log_path, e)

    @staticmethod
    def compute_feature_hash(features: list[float]) -> str:
        """Compute deterministic SHA-256 hash of the 165-feature vector."""
        byte_data = json.dumps([round(f, 6) for f in features]).encode("utf-8")
        return hashlib.sha256(byte_data).hexdigest()[:16]

    def log_prediction(
        self,
        endpoint: str,
        request_id: str,
        tx_id: int | str | None,
        time_step: int,
        features: list[float],
        probability: float,
        binary_classification: int,
        threshold: float,
        model_reliability: str,
        regime: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Log a single prediction event to JSONL file."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "tx_id": str(tx_id) if tx_id is not None else None,
            "time_step": time_step,
            "input_feature_hash": self.compute_feature_hash(features),
            "probability": probability,
            "binary_classification": binary_classification,
            "threshold": threshold,
            "model_reliability": model_reliability,
            "regime": regime,
            "latency_ms": round(latency_ms, 3),
        }

        line = json.dumps(record) + "\n"

        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:  # noqa: BLE001 - logging must never break inference
                logger.error(
                    "Failed to write prediction log entry to %s: %s", self.log_path, e
                )

    def log_batch(
        self,
        endpoint: str,
        request_id: str,
        records: list[dict[str, Any]],
        latency_ms: float = 0.0,
    ) -> None:
        """Log a batch of prediction events atomically."""
        if not records:
            return

        ts = datetime.now(timezone.utc).isoformat()
        lines = []
        for r in records:
            entry = {
                "timestamp": ts,
                "request_id": request_id,
                "endpoint": endpoint,
                "tx_id": str(r.get("tx_id")) if r.get("tx_id") is not None else None,
                "time_step": r.get("time_step"),
                "input_feature_hash": r.get("input_feature_hash", ""),
                "probability": r.get("probability"),
                "binary_classification": r.get("binary_classification"),
                "threshold": r.get("threshold"),
                "model_reliability": r.get("model_reliability"),
                "regime": r.get("regime"),
                "latency_ms": round(latency_ms, 3),
            }
            lines.append(json.dumps(entry) + "\n")

        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.writelines(lines)
            except Exception as e:  # noqa: BLE001 - logging must never break inference
                logger.error(
                    "Failed to write batch prediction logs to %s: %s", self.log_path, e
                )


# Global singleton logger
_prediction_logger: StructuredPredictionLogger | None = None


def get_prediction_logger() -> StructuredPredictionLogger:
    """Access or lazily initialize the singleton prediction logger."""
    global _prediction_logger
    if _prediction_logger is None:
        _prediction_logger = StructuredPredictionLogger()
    return _prediction_logger
