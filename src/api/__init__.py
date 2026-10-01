"""
Inference and Serving API package for BitcoinGraphGuard.
"""

from src.api.main import app
from src.api.model_loader import ModelContainer, get_model_container
from src.api.schemas import (
    BatchPredictionOutput,
    BatchTransactionInput,
    ConfidenceContext,
    HealthResponse,
    MonitoringStatusResponse,
    PredictionOutput,
    TransactionInput,
)

__all__ = [
    "BatchPredictionOutput",
    "BatchTransactionInput",
    "ConfidenceContext",
    "HealthResponse",
    "ModelContainer",
    "MonitoringStatusResponse",
    "PredictionOutput",
    "TransactionInput",
    "app",
    "get_model_container",
]
