# BitcoinGraphGuard — Inference Service & Serving Layer (Phase 8b)

Production-grade tabular inference service and containerized serving layer for Bitcoin illicit transaction detection on the **Elliptic++** dataset.

The service serves the frozen **XGBoost Optimized baseline** (165 features: 93 Local + 72 Aggregate) at the pre-registered decision threshold $\tau^* = 0.435$, augmented with **context-aware operational reliability assessments** and active integration with the Phase 8a drift monitoring engine.

---

## ⚠️ Critical Known Limitation & Operational Reliability Boundary

> **REGIME DEGRADATION NOTICE:**
> The frozen production model achieves a high **PR-AUC of 0.8013** (Precision 0.827, Recall 0.738, F1 0.780) across steps 35–49 aggregate and **0.9271 PR-AUC** (F1 0.875) on the stationary steps 35–42.
>
> However, starting at **Time step 43**, the underlying network undergoes an **irreducible covariate regime shift** and illicit prevalence collapses from 9.16% to 2.53%. In the **43–49 window**, the frozen model experiences a **catastrophic performance breakdown** (PR-AUC drops to ~0.04–0.18, F1 = 0.0406).
>
> **Empirical Truth:** Retrospective drift investigations (Phase 2b/8a) proved that this collapse **cannot be recovered by threshold adaptation alone** (Bayesian adaptation yields F1 0.0456; theoretical oracle ceiling is only 0.1324 due to corrupted ranking representation).
>
> **How the API Protects Users:** Every prediction returned by `POST /predict` and `POST /batch_predict` includes a mandatory `confidence_context` block:
> - **Steps 35–42:** `model_reliability: "reliable"` (`stationary_test_window`).
> - **Steps 43–49:** `model_reliability: "degraded"` (`drift_collapse_regime`) with explicit warnings that model output is unreliable.
> - **Steps 50+ (Live):** `model_reliability: "unknown"` (`live_unmonitored_step`), directing operators to query `GET /monitoring/status`.

---

## Architecture & System Overview

```text
Incoming Transaction Request (165 features + Time step)
   │
   ├──▶ Request Validation & Feature Schema Check (Strict 165-feature contract, NaN/Inf reject)
   │
   ├──▶ XGBoost Inference (xgb.Booster, frozen tau* = 0.435)
   │
   ├──▶ Regime & Reliability Context Lookup (per-step empirical benchmark mapping)
   │
   ├──▶ Structured Prediction Audit Logging (JSONL -> logs/predictions.jsonl)
   │
   └──▶ Response Output (Probability, Binary Class, Threshold, Confidence Context)
```

---

## API Endpoints

### 1. `POST /predict`
Scores a single transaction. Accepts 165 features as flat fields or in a nested `features` dictionary, alongside the transaction's `time_step`.

#### Request Example
```json
{
  "tx_id": "1813992",
  "time_step": 35,
  "Local_feature_1": 0.105155,
  "Local_feature_2": 3.567918,
  "...": 0.0,
  "Local_feature_93": -0.348348,
  "Aggregate_feature_1": -0.169584,
  "...": 0.0,
  "Aggregate_feature_72": 0.250279
}
```

#### Response Example
```json
{
  "tx_id": "1813992",
  "time_step": 35,
  "probability": 0.00001719,
  "binary_classification": 0,
  "threshold": 0.435,
  "label": "licit",
  "confidence_context": {
    "model_reliability": "reliable",
    "regime": "stationary_test_window",
    "reason": "Time step 35 falls within the stationary test envelope (steps 35-42). Historical performance on this regime is high (PR-AUC: 0.9934, F1: 0.9596).",
    "historical_pr_auc": 0.9934,
    "historical_f1": 0.9596,
    "recommendation": "Operational inference reliable; predictions carry high historical fidelity."
  }
}
```

---

### 2. `POST /batch_predict`
Batch scoring endpoint for high throughput. Accepts a JSON array or `{"transactions": [...]}`. Protected by a configurable batch limit (`MAX_BATCH_SIZE=10000`) to respect the machine's ~5.6 GB RAM constraint.

#### Response Example
```json
{
  "total_transactions": 2,
  "threshold": 0.435,
  "predictions": [
    {
      "tx_id": "tx_step_35",
      "time_step": 35,
      "probability": 0.00001719,
      "binary_classification": 0,
      "threshold": 0.435,
      "label": "licit",
      "confidence_context": {
        "model_reliability": "reliable",
        "regime": "stationary_test_window",
        "recommendation": "Operational inference reliable; predictions carry high historical fidelity."
      }
    },
    {
      "tx_id": "tx_step_43",
      "time_step": 43,
      "probability": 0.01250000,
      "binary_classification": 0,
      "threshold": 0.435,
      "label": "licit",
      "confidence_context": {
        "model_reliability": "degraded",
        "regime": "drift_collapse_regime",
        "recommendation": "CRITICAL WARNING: Predictions in steps 43-49 operate under severe distribution shift. High false positive / false negative risk. Manual investigation or model retraining mandated."
      }
    }
  ]
}
```

---

### 3. `GET /monitoring/status`
Exposes the latest retraining-trigger state by reading the Phase 8a monitoring artifact
(`reports/monitoring_backtest_report.json`) and rehydrating it through the canonical
`RetrainingDecision` / `TriggerAction` / `AlertSeverity` types from
`src/monitoring/retraining_trigger.py`. No trigger logic is re-implemented or hardcoded in the
serving layer. If the artifact is missing or malformed the endpoint returns `503` rather than
inventing a status.

#### Response Example (Reflecting Active Drift Regime)
```json
{
  "latest_monitored_step": 49,
  "trigger_action": "RETRAIN",
  "severity": "CRITICAL",
  "primary_reason": "PrevalenceRegimeCollapse: Rolling illicit prevalence (0.0184) plummeted below critical floor (0.0350).",
  "requires_retraining": true,
  "requires_action": true,
  "component_statuses": {
    "feature_triad": "STABLE",
    "feature_local": "MODERATE",
    "feature_overall": "MODERATE",
    "prevalence": "ALERT",
    "adversarial": "ALERT"
  },
  "telemetry_summary": {
    "time_step": 49,
    "triad_psi": {
      "Aggregate_feature_10": 0.2251,
      "Aggregate_feature_43": 0.1808,
      "Aggregate_feature_8": 0.3897
    },
    "pct_local_drifted_sig": 11.827956989247312,
    "pct_local_drifted_mod": 45.16129032258064,
    "local_drift_streak": 0,
    "rolling_prevalence": 0.018384631274530057,
    "prevalence_rel_change_pct": -84.12504099462905,
    "adversarial_auc": 0.87056875,
    "frozen_f1": 0.028169014084507043,
    "adaptive_f1": 0.027777777777777776,
    "bayes_f1": 0.03389830508474576,
    "pr_auc": 0.18225106911311845
  },
  "all_reasons": [
    "PrevalenceRegimeCollapse: Rolling illicit prevalence (0.0184) plummeted below critical floor (0.0350).",
    "PerformanceCrash: Frozen threshold F1 collapsed to 0.0282 (PR-AUC=0.1823).",
    "AdversarialOODAlert: Domain classifier AUC=0.8706 >= 0.85 (saturated domain separability; corroborating signal only, not a standalone retrain trigger)."
  ],
  "timestamp": "2026-10-01T20:00:00.000000+00:00"
}
```

---

### 4. `GET /health`
Returns service status, model artifact paths, feature count (165), threshold, and startup validation checks.

```json
{
  "status": "healthy",
  "model_name": "XGBoost Optimized (Frozen Phase 2)",
  "model_type": "xgboost.Booster",
  "n_features": 165,
  "operating_threshold": 0.435,
  "startup_timestamp": "2026-10-01T20:00:00.000000+00:00",
  "startup_validations_passed": true,
  "artifact_paths": {
    "model_path": "results/xgboost/xgb_model_optimized.json",
    "feature_list_path": "results/xgboost/selected_features.json",
    "metrics_path": "results/temporal_inductive/per_step_metrics.csv"
  }
}
```

---

## Structured Prediction Logging

Every inference call emits a structured JSON line to `logs/predictions.jsonl`:

```json
{
  "timestamp": "2026-10-01T20:00:00.000000+00:00",
  "request_id": "84c8a417-640a-4286-98ec-246e6b47c0ea",
  "endpoint": "/predict",
  "tx_id": "1813992",
  "time_step": 35,
  "input_feature_hash": "a1b2c3d4e5f67890",
  "probability": 0.00001719,
  "binary_classification": 0,
  "threshold": 0.435,
  "model_reliability": "reliable",
  "regime": "stationary_test_window",
  "latency_ms": 2.41
}
```

This structured audit trail provides tamper-evident tracking and directly feeds the Phase 9 monitoring dashboard.

---

## How to Run

### 1. Local Development (Laptop)

```bash
# Activate environment
source .venv/bin/activate

# Install serving dependencies
uv pip install -r requirements-serving.txt pytest

# Run FastAPI server with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Access interactive Swagger API documentation
open http://localhost:8000/docs
```

### 2. Docker & Docker-Compose (Production)

The Docker configuration uses a lightweight, multi-stage build that excludes PyTorch, CUDA, PyG, and heavy MLflow packages to maintain a minimal container footprint:

```bash
# Build and run the containerized API with resource limits
docker compose up --build -d

# Verify container status & healthcheck
docker compose ps

# View live application logs
docker compose logs -f api

# Test the health endpoint
curl -s http://localhost:8000/health | jq .

# Stop the container
docker compose down
```

### Resource Allocation
- Memory limit: `1536M` (1.5 GB), memory reservation: `512M`.
- CPU limit: `2.0` cores.
- Designed strictly to run safely within the laptop's ~5.6 GB RAM ceiling.

---

## Running the Test Suite

```bash
# Run API test suite
pytest tests/test_api.py -v

# Run full project test suite with coverage
pytest --cov=src tests/
```
