# ==============================================================================
# BitcoinGraphGuard Inference Service — Multi-Stage Production Dockerfile
#
# Constraints & Guidelines:
# - Minimal CPU-only tabular serving container (XGBoost + FastAPI).
# - Strictly excludes PyTorch, PyG, MLflow, and heavy training dependencies.
# - Multi-stage build for small image size.
# - Non-root execution for security.
# - Built-in health check against GET /health.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Dependencies
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install minimal build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install lightweight serving requirements
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serving.txt

# ------------------------------------------------------------------------------
# Stage 2: Final Lightweight Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

WORKDIR /app

# Install curl for Docker healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PREDICTION_LOG_PATH=/app/logs/predictions.jsonl \
    MAX_BATCH_SIZE=10000

# Create dedicated non-root user and directories
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /app/logs /app/results /app/reports && \
    chown -R appuser:appgroup /app

# Copy application source code and necessary production artifacts
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup results/ /app/results/
COPY --chown=appuser:appgroup reports/ /app/reports/

USER appuser

EXPOSE 8000

# Health check configured against GET /health
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI application via Uvicorn with single worker to respect RAM limits
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
