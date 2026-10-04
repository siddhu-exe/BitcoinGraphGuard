"""
Unit and Integration Tests for BitcoinGraphGuard FastAPI Serving Layer.

Tests:
1. Model loading, artifact validation, and health checks.
2. End-to-end consistency check: /predict on known-good test set transactions matches recorded benchmark scores.
3. Self-aware confidence context: verifies reliable (35-42), degraded (43-49), unknown (50+), and reference (1-34) regimes.
4. Input validation and error handling: rejects missing features, NaN/Inf, invalid types with 422 naming failed features.
5. Batch prediction throughput and memory safety guardrails (max batch size).
6. Live monitoring integration: GET /monitoring/status correctly exposes RETRAIN (CRITICAL) state.
7. Structured JSONL prediction logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.logging_utils import StructuredPredictionLogger
from src.api.main import app
from src.api.model_loader import ModelContainer, ModelLoadError, get_model_container
from src.api.schemas import MODEL_FEATURES


class SyncASGIClient:
    """Synchronous httpx client that drives the ASGI app in-process.

    Starlette's ``TestClient`` relies on
    ``anyio.from_thread.start_blocking_portal``, which deadlocks on Python 3.14
    in this project's environment. ``httpx.ASGITransport`` exercises the same
    ASGI app without a blocking portal and exposes the ``.get``/``.post``
    surface used by this suite.
    """

    __test__ = False  # Tell pytest this is a helper, not a test class.

    def __init__(self, app: Any) -> None:
        self._app = app

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Run one request to completion on a fresh event loop."""
        return asyncio.run(self._arequest(method, url, **kwargs))

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    async def _arequest(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)


# Alias keeps the existing ``client: TestClient`` annotations meaningful.
TestClient = SyncASGIClient

# Sample known-good transaction from test set (txId: 1813992, time_step: 35)
# Recorded benchmark score in results/xgboost/predictions.csv: 1.7188735e-05
SAMPLE_TX_1813992_FEATURES = {
    "Local_feature_1": 0.1051558351511294,
    "Local_feature_2": 3.567918580259372,
    "Local_feature_3": 2.128587252854236,
    "Local_feature_4": 2.129155651434577,
    "Local_feature_5": 0.8692266168710974,
    "Local_feature_6": 2.2351406705800065,
    "Local_feature_7": 0.2427123366398294,
    "Local_feature_8": -0.162415288177796,
    "Local_feature_9": -0.111938407754468,
    "Local_feature_10": 0.0967859699131057,
    "Local_feature_11": -0.1567768162595724,
    "Local_feature_12": 0.9993121004435106,
    "Local_feature_13": 1.7504519236367715,
    "Local_feature_14": 0.8882987864148069,
    "Local_feature_15": -0.0132816148700588,
    "Local_feature_16": -0.057347413469842,
    "Local_feature_17": 0.0558181295495354,
    "Local_feature_18": -0.1187926051485286,
    "Local_feature_19": -0.162543962757623,
    "Local_feature_20": -0.1919742751704946,
    "Local_feature_21": -0.1507470377046298,
    "Local_feature_22": -0.1397313370721692,
    "Local_feature_23": -0.1487076574221477,
    "Local_feature_24": -0.0800101541120838,
    "Local_feature_25": -0.1556162810711981,
    "Local_feature_26": 0.6516081526156264,
    "Local_feature_27": 1.5131002246399583,
    "Local_feature_28": -0.1397331368195247,
    "Local_feature_29": -0.1487029428702339,
    "Local_feature_30": -0.0800096204127154,
    "Local_feature_31": -0.1556162462222343,
    "Local_feature_32": 0.6516945083662262,
    "Local_feature_33": 1.5131046809967257,
    "Local_feature_34": -0.0246688306562535,
    "Local_feature_35": -0.0312723904866303,
    "Local_feature_36": -0.0230451563960962,
    "Local_feature_37": -0.0262146551774309,
    "Local_feature_38": 0.0014278137097094,
    "Local_feature_39": 0.0014826437872997,
    "Local_feature_40": -0.2272154464478222,
    "Local_feature_41": -0.2378361603743357,
    "Local_feature_42": -0.0728020304924843,
    "Local_feature_43": -0.2346984961348917,
    "Local_feature_44": 0.9315851212174608,
    "Local_feature_45": 0.9859333934126446,
    "Local_feature_46": -0.2272033238276858,
    "Local_feature_47": -0.2419165865408378,
    "Local_feature_48": -0.0948356646872554,
    "Local_feature_49": -0.2356886639171357,
    "Local_feature_50": 0.8918945579660855,
    "Local_feature_51": 0.9454264587398654,
    "Local_feature_52": -0.4136712980599087,
    "Local_feature_53": 0.159652687976383,
    "Local_feature_54": 0.2542872023970234,
    "Local_feature_55": -0.3932619547284197,
    "Local_feature_56": 0.4771716818317059,
    "Local_feature_57": 0.2639489697535097,
    "Local_feature_58": -0.0391510397700154,
    "Local_feature_59": -0.172477895407373,
    "Local_feature_60": -0.1630355437025884,
    "Local_feature_61": -0.1609103723470971,
    "Local_feature_62": -0.2145555103003116,
    "Local_feature_63": -0.2696925001552488,
    "Local_feature_64": -0.0391463242980219,
    "Local_feature_65": -0.1724668221561377,
    "Local_feature_66": -0.1630242055308584,
    "Local_feature_67": -0.1609034861477888,
    "Local_feature_68": -0.2145247931165385,
    "Local_feature_69": -0.2696391264460032,
    "Local_feature_70": -0.0170316758803496,
    "Local_feature_71": -0.0300202313177692,
    "Local_feature_72": -0.0176382968121671,
    "Local_feature_73": -0.015070332814139,
    "Local_feature_74": -0.6897875035336272,
    "Local_feature_75": -0.6678923079379219,
    "Local_feature_76": -0.0954026892728408,
    "Local_feature_77": -0.2575446153294355,
    "Local_feature_78": -0.2488493260357874,
    "Local_feature_79": -0.2625834054797074,
    "Local_feature_80": -0.3947263486632724,
    "Local_feature_81": -0.4361160200772773,
    "Local_feature_82": -0.0590130547854711,
    "Local_feature_83": -0.2606514405502226,
    "Local_feature_84": -0.2545216476773478,
    "Local_feature_85": -0.25855437523039,
    "Local_feature_86": -0.3223755796110325,
    "Local_feature_87": -0.4321907156416882,
    "Local_feature_88": -0.2935671031979225,
    "Local_feature_89": 1.1686129427945966,
    "Local_feature_90": 0.1577788247526858,
    "Local_feature_91": 0.1526423902730215,
    "Local_feature_92": -0.3727860563562097,
    "Local_feature_93": -0.3483485513219919,
    "Aggregate_feature_1": -0.1695842028977804,
    "Aggregate_feature_2": 1.403454545580869,
    "Aggregate_feature_3": 1.272129256003659,
    "Aggregate_feature_4": -0.0272214804409713,
    "Aggregate_feature_5": -0.4422571297062452,
    "Aggregate_feature_6": -0.0985797774012211,
    "Aggregate_feature_7": 0.7363768166109619,
    "Aggregate_feature_8": 0.7374214503521257,
    "Aggregate_feature_9": -0.0827551587750555,
    "Aggregate_feature_10": 0.7393715044689905,
    "Aggregate_feature_11": 0.6090560782613939,
    "Aggregate_feature_12": 0.6985657807710832,
    "Aggregate_feature_13": -0.1408704596223426,
    "Aggregate_feature_14": 0.1833417405796316,
    "Aggregate_feature_15": 0.1766396441321852,
    "Aggregate_feature_16": -0.0920491177010026,
    "Aggregate_feature_17": 0.6597681052097889,
    "Aggregate_feature_18": 1.0269645023853171,
    "Aggregate_feature_19": -1.0963356939258175,
    "Aggregate_feature_20": 1.3011283510936915,
    "Aggregate_feature_21": 1.9988717957074849,
    "Aggregate_feature_22": 0.6766607554267099,
    "Aggregate_feature_23": 0.3441105169755871,
    "Aggregate_feature_24": 0.4367536148635979,
    "Aggregate_feature_25": -0.1164246038907396,
    "Aggregate_feature_26": 0.6174617341292103,
    "Aggregate_feature_27": 0.4294204465778772,
    "Aggregate_feature_28": -0.027549385272111,
    "Aggregate_feature_29": 0.3673209467693854,
    "Aggregate_feature_30": 0.8458543942983812,
    "Aggregate_feature_31": -0.0931447270142624,
    "Aggregate_feature_32": -0.1209992215861581,
    "Aggregate_feature_33": -0.0834212293685824,
    "Aggregate_feature_34": -0.1228706013779709,
    "Aggregate_feature_35": 1.4064593794837252,
    "Aggregate_feature_36": 0.9565559031029552,
    "Aggregate_feature_37": -0.1247353278306089,
    "Aggregate_feature_38": -0.0507606680764395,
    "Aggregate_feature_39": -0.1279836561354381,
    "Aggregate_feature_40": -0.1791688127149777,
    "Aggregate_feature_41": 0.1511966406390993,
    "Aggregate_feature_42": -0.0032698974619527,
    "Aggregate_feature_43": 0.6794028221389465,
    "Aggregate_feature_44": 1.802177463729156,
    "Aggregate_feature_45": 0.7460293913474197,
    "Aggregate_feature_46": 0.7900320799598203,
    "Aggregate_feature_47": -0.1179614224910244,
    "Aggregate_feature_48": -0.0321314893578808,
    "Aggregate_feature_49": -0.172889448019655,
    "Aggregate_feature_50": 2.0699478022992963,
    "Aggregate_feature_51": 0.66164167425406,
    "Aggregate_feature_52": 0.4304883108158572,
    "Aggregate_feature_53": 0.3883714191761521,
    "Aggregate_feature_54": 0.4588404363651816,
    "Aggregate_feature_55": 0.6601173970186788,
    "Aggregate_feature_56": 1.1924206753212654,
    "Aggregate_feature_57": -0.2522260492272348,
    "Aggregate_feature_58": 1.0024142677427172,
    "Aggregate_feature_59": 0.5177736756280918,
    "Aggregate_feature_60": 0.5314344939312184,
    "Aggregate_feature_61": -0.2168143606184322,
    "Aggregate_feature_62": 0.6035813381827653,
    "Aggregate_feature_63": 0.3796216886966277,
    "Aggregate_feature_64": 0.3855893515213265,
    "Aggregate_feature_65": 0.2689873322909921,
    "Aggregate_feature_66": 0.1829519336740336,
    "Aggregate_feature_67": -0.098888736679456,
    "Aggregate_feature_68": 0.0470810477654827,
    "Aggregate_feature_69": -0.0473032378827808,
    "Aggregate_feature_70": -0.1172656076084601,
    "Aggregate_feature_71": 0.0523369512886388,
    "Aggregate_feature_72": 0.2502795121531035,
}


@pytest.fixture
def client():
    """Create an in-process ASGI test client with the model pre-loaded."""
    get_model_container()
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200, 165 features, and passed startup validations."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["n_features"] == 165
    assert data["operating_threshold"] == 0.435
    assert data["startup_validations_passed"] is True
    assert "model_path" in data["artifact_paths"]


def test_root_endpoint(client: TestClient):
    """Verify / returns service overview."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "BitcoinGraphGuard Inference Service"
    assert "/predict" in data["predict_url"]


def test_predict_known_good_transaction_reproduces_benchmark(client: TestClient):
    """
    End-to-end consistency check:
    Verify that txId 1813992 at step 35 returns the exact recorded benchmark score
    (1.7188735e-05) within floating point tolerance.
    """
    payload = {
        "tx_id": "1813992",
        "time_step": 35,
        **SAMPLE_TX_1813992_FEATURES,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["tx_id"] == "1813992"
    assert data["time_step"] == 35
    assert data["binary_classification"] == 0
    assert data["label"] == "licit"
    assert data["threshold"] == 0.435

    # Benchmark score is 1.7188735e-05
    expected_prob = 1.7188735e-05
    assert data["probability"] == pytest.approx(expected_prob, rel=1e-3, abs=1e-6)

    # Reliability on step 35 must be "reliable" (stationary window)
    ctx = data["confidence_context"]
    assert ctx["model_reliability"] == "reliable"
    assert ctx["regime"] == "stationary_test_window"
    assert ctx["historical_pr_auc"] is not None
    assert ctx["historical_pr_auc"] > 0.90


def test_confidence_context_regimes(client: TestClient):
    """
    Verify self-aware confidence context across all operational regimes:
    - Step 36: Reliable (stationary test window)
    - Step 43: Degraded (regime collapse onset)
    - Step 47: Degraded (drift window)
    - Step 50: Unknown (live unmonitored future step)
    - Step 20: Training reference (historical training period)
    """
    # 1. Reliable Step (Step 36)
    p36 = {"time_step": 36, **SAMPLE_TX_1813992_FEATURES}
    r36 = client.post("/predict", json=p36).json()
    assert r36["confidence_context"]["model_reliability"] == "reliable"
    assert r36["confidence_context"]["regime"] == "stationary_test_window"

    # 2. Degraded Step (Step 43 - documented regime collapse)
    p43 = {"time_step": 43, **SAMPLE_TX_1813992_FEATURES}
    r43 = client.post("/predict", json=p43).json()
    assert r43["confidence_context"]["model_reliability"] == "degraded"
    assert r43["confidence_context"]["regime"] == "drift_collapse_regime"
    assert "WARNING" in r43["confidence_context"]["recommendation"].upper()

    # 3. Degraded Step (Step 47)
    p47 = {"time_step": 47, **SAMPLE_TX_1813992_FEATURES}
    r47 = client.post("/predict", json=p47).json()
    assert r47["confidence_context"]["model_reliability"] == "degraded"
    assert r47["confidence_context"]["regime"] == "drift_collapse_regime"

    # 4. Unknown Step (Live step 50+)
    p50 = {"time_step": 50, **SAMPLE_TX_1813992_FEATURES}
    r50 = client.post("/predict", json=p50).json()
    assert r50["confidence_context"]["model_reliability"] == "unknown"
    assert r50["confidence_context"]["regime"] == "live_unmonitored_step"
    assert r50["confidence_context"]["historical_pr_auc"] is None

    # 5. Training Reference Step (Step 20)
    p20 = {"time_step": 20, **SAMPLE_TX_1813992_FEATURES}
    r20 = client.post("/predict", json=p20).json()
    assert r20["confidence_context"]["model_reliability"] == "training_reference"
    assert r20["confidence_context"]["regime"] == "historical_training_window"


def test_input_validation_missing_feature_fails_fast(client: TestClient):
    """Verify that omitting a feature returns 422 naming the missing feature."""
    incomplete_features = {
        k: v
        for k, v in SAMPLE_TX_1813992_FEATURES.items()
        if k != "Aggregate_feature_10"
    }
    payload = {
        "time_step": 35,
        **incomplete_features,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "Aggregate_feature_10" in str(data)


def test_input_validation_nan_or_inf_fails_fast(client: TestClient):
    """Verify that NaN or Inf feature values return 422 naming the offending feature."""
    # Test NaN string representation which is common in JSON deserialization
    nan_features = dict(SAMPLE_TX_1813992_FEATURES)
    nan_features["Local_feature_5"] = "NaN"

    payload = {
        "time_step": 35,
        "features": nan_features,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "Local_feature_5" in str(data)
    assert "NaN or Inf" in str(data)

    # Test Inf string representation
    inf_features = dict(SAMPLE_TX_1813992_FEATURES)
    inf_features["Aggregate_feature_10"] = "Infinity"
    inf_payload = {
        "time_step": 35,
        "features": inf_features,
    }
    response_inf = client.post("/predict", json=inf_payload)
    assert response_inf.status_code == 422
    assert "Aggregate_feature_10" in str(response_inf.json())


def test_input_validation_non_numeric_type_fails_fast(client: TestClient):
    """Verify that non-numeric values return 422."""
    bad_features = dict(SAMPLE_TX_1813992_FEATURES)
    bad_features["Local_feature_1"] = "invalid_string"

    payload = {
        "time_step": 35,
        "features": bad_features,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_nested_features_dict_format(client: TestClient):
    """Verify that passing features inside a nested 'features' dict works identically."""
    payload = {
        "tx_id": "nested_test_1",
        "time_step": 35,
        "features": SAMPLE_TX_1813992_FEATURES,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == "nested_test_1"
    assert data["probability"] == pytest.approx(1.7188735e-05, rel=1e-3, abs=1e-6)


def test_batch_predict_endpoint(client: TestClient):
    """Verify /batch_predict processes multiple transactions and returns ordered outputs."""
    batch_payload = {
        "transactions": [
            {"tx_id": "tx_1", "time_step": 35, **SAMPLE_TX_1813992_FEATURES},
            {"tx_id": "tx_2", "time_step": 43, **SAMPLE_TX_1813992_FEATURES},
            {"tx_id": "tx_3", "time_step": 50, **SAMPLE_TX_1813992_FEATURES},
        ]
    }

    response = client.post("/batch_predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_transactions"] == 3
    assert len(data["predictions"]) == 3
    assert (
        data["predictions"][0]["confidence_context"]["model_reliability"] == "reliable"
    )
    assert (
        data["predictions"][1]["confidence_context"]["model_reliability"] == "degraded"
    )
    assert (
        data["predictions"][2]["confidence_context"]["model_reliability"] == "unknown"
    )


def test_batch_predict_empty_and_max_limit_guardrails(client: TestClient, monkeypatch):
    """Verify empty batch and oversized batch guardrails protect memory limits."""
    # 1. Empty batch
    r_empty = client.post("/batch_predict", json={"transactions": []})
    assert r_empty.status_code == 422 or r_empty.status_code == 400

    # 2. Oversized batch limit
    monkeypatch.setattr("src.api.main.DEFAULT_MAX_BATCH_SIZE", 2)
    oversized = {
        "transactions": [
            {"tx_id": f"tx_{i}", "time_step": 35, **SAMPLE_TX_1813992_FEATURES}
            for i in range(3)
        ]
    }
    r_oversized = client.post("/batch_predict", json=oversized)
    assert r_oversized.status_code == 400
    assert "exceeds maximum allowed batch limit" in r_oversized.json()["detail"]


def test_monitoring_status_endpoint(client: TestClient):
    """
    Verify GET /monitoring/status reflects the ACTUAL latest backtest decision.

    The expected values are read from reports/monitoring_backtest_report.json
    (step 49: RETRAIN / CRITICAL) rather than hardcoded, so this test only passes
    when the endpoint genuinely parses the monitoring artifact through the
    canonical RetrainingDecision types.
    """
    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "monitoring_backtest_report.json"
    )
    latest_decision = json.loads(report_path.read_text(encoding="utf-8"))[
        "simulation_steps"
    ][-1]["decision"]

    response = client.get("/monitoring/status")
    assert response.status_code == 200
    data = response.json()

    assert data["latest_monitored_step"] == latest_decision["time_step"] == 49
    assert data["trigger_action"] == latest_decision["action"] == "RETRAIN"
    assert data["severity"] == latest_decision["severity"] == "CRITICAL"
    assert data["primary_reason"] == latest_decision["primary_reason"]
    assert data["requires_retraining"] is True
    assert data["requires_action"] is True
    assert any(
        k in data["primary_reason"]
        for k in ("PrevalenceRegimeCollapse", "PerformanceCrash", "ScoreShift")
    )
    # Full artifacts must round-trip unchanged (no hardcoded placeholders).
    assert data["component_statuses"] == latest_decision["component_statuses"]
    assert data["telemetry_summary"] == latest_decision["telemetry_summary"]
    assert data["all_reasons"] == latest_decision["reasons"]
    # Real engine component labels; guards against the old hardcoded keys.
    assert "feature_triad" in data["component_statuses"]
    assert "triad_feature_drift" not in data["component_statuses"]
    assert "rolling_prevalence" in data["telemetry_summary"]
    # Additive lag-safe fields (new fields only; nothing above was removed).
    assert data["label_delay_steps"] == latest_decision["label_delay_steps"]
    assert str(data["label_delay_steps"]) in data["label_delay_assumption"]
    assert data["channel_breakdown"] == latest_decision["channels"]
    assert {"performance", "prevalence", "score_shift", "feature_shift"} <= set(
        data["channel_breakdown"]
    )
    assert data["channel_breakdown"]["feature_shift"]["can_trigger_critical"] is False


def test_monitoring_status_unavailable_returns_503(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing monitoring artifact must surface as 503, never fabricated data."""
    monkeypatch.setattr(
        "src.api.main.MONITORING_REPORT_PATH", tmp_path / "does_not_exist.json"
    )
    response = client.get("/monitoring/status")
    assert response.status_code == 503
    assert "Monitoring status unavailable" in response.json()["detail"]


def test_structured_prediction_logging(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify that prediction requests append structured JSONL records."""
    log_file = tmp_path / "test_predictions.jsonl"
    test_logger = StructuredPredictionLogger(log_path=log_file)

    # Point the singleton logger (owned by logging_utils) at the temp file.
    monkeypatch.setattr("src.api.logging_utils._prediction_logger", test_logger)

    payload = {
        "tx_id": "audit_log_tx_123",
        "time_step": 35,
        **SAMPLE_TX_1813992_FEATURES,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1

    last_record = json.loads(lines[-1])
    assert last_record["endpoint"] == "/predict"
    assert last_record["tx_id"] == "audit_log_tx_123"
    assert last_record["time_step"] == 35
    assert last_record["binary_classification"] == 0
    assert last_record["threshold"] == 0.435
    assert last_record["model_reliability"] == "reliable"
    assert last_record["input_feature_hash"] != ""
    assert "timestamp" in last_record


# ---------------------------------------------------------------------------
# Task 2: per-step metrics fallback must be loud, not silent
# ---------------------------------------------------------------------------

_PER_STEP_METRICS_HEADER = (
    "time_step,model,n,illicit,licit,prevalence,pr_auc,roc_auc,threshold,"
    "precision,recall,f1,tp,fp,tn,fn\n"
)


def _write_metrics_csv(path: Path, pr_auc: float, f1: float) -> None:
    """Write a minimal but schema-valid per-step metrics CSV with one XGBoost row."""
    row = (
        f"35,XGBoost (frozen),100,10,90,0.1,{pr_auc},0.99,0.435,"
        f"0.5,0.5,{f1},10,10,80,0\n"
    )
    path.write_text(_PER_STEP_METRICS_HEADER + row, encoding="utf-8")


def test_health_no_fallback_when_metrics_csv_present(client, tmp_path, monkeypatch):
    """Valid CSV -> using_fallback_metrics=false and context values from the CSV."""
    metrics_csv = tmp_path / "per_step_metrics.csv"
    _write_metrics_csv(metrics_csv, pr_auc=0.424242, f1=0.111111)

    container = ModelContainer(metrics_path=metrics_csv)
    assert container.using_fallback_metrics is False
    ctx = container.get_confidence_context(35)
    assert ctx.historical_pr_auc == pytest.approx(0.424242)
    assert ctx.historical_f1 == pytest.approx(0.111111)

    from src.api import model_loader

    monkeypatch.setattr(model_loader, "_model_container", container)
    data = client.get("/health").json()
    assert data["using_fallback_metrics"] is False
    assert data["metrics_fallback_reason"] is None


def test_health_fallback_when_metrics_missing_logs_warning(
    client, tmp_path, monkeypatch, caplog
):
    """Missing CSV -> using_fallback_metrics=true AND an explicit WARNING is logged."""
    missing_csv = tmp_path / "does_not_exist.csv"
    assert not missing_csv.exists()

    with caplog.at_level(logging.WARNING, logger="src.api.model_loader"):
        container = ModelContainer(metrics_path=missing_csv)

    assert container.using_fallback_metrics is True
    assert "file not found" in container.metrics_fallback_reason

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "hardcoded Phase 8a" in msg and "file not found" in msg for msg in warnings
    ), f"Expected loud fallback warning, got: {warnings}"

    from src.api import model_loader

    monkeypatch.setattr(model_loader, "_model_container", container)
    data = client.get("/health").json()
    assert data["using_fallback_metrics"] is True
    assert "file not found" in (data["metrics_fallback_reason"] or "")


def test_startup_rejects_wrong_feature_names_not_just_count(tmp_path):
    """165 keys but wrong names must fail startup (name validation, not count)."""
    bogus = tmp_path / "features.json"
    bogus.write_text(
        json.dumps({"features": [f"Wrong_feature_{i}" for i in range(1, 166)]}),
        encoding="utf-8",
    )

    with pytest.raises(ModelLoadError, match="Feature set mismatch"):
        ModelContainer(feature_list_path=bogus)


# ---------------------------------------------------------------------------
# Task 3: feature values must map by NAME, never by JSON key order
# ---------------------------------------------------------------------------


def test_feature_values_map_by_name_not_json_key_order(client):
    """Shuffling JSON key order must not change predictions; swapped values do."""
    base = dict(SAMPLE_TX_1813992_FEATURES)
    swapped = dict(base)
    swapped["Local_feature_1"], swapped["Local_feature_2"] = (
        base["Local_feature_2"],
        base["Local_feature_1"],
    )

    base_prob = client.post("/predict", json={"time_step": 35, **base}).json()[
        "probability"
    ]
    swapped_prob = client.post("/predict", json={"time_step": 35, **swapped}).json()[
        "probability"
    ]

    # Values are honored per named column: swapping them changes the score.
    assert swapped_prob != pytest.approx(base_prob, rel=1e-6, abs=1e-12)

    # Same named values, arbitrary JSON insertion order -> identical prediction.
    items = list(swapped.items())
    random.Random(0).shuffle(items)
    shuffled_prob = client.post(
        "/predict", json={"time_step": 35, **dict(items)}
    ).json()["probability"]
    assert shuffled_prob == pytest.approx(swapped_prob, rel=1e-9, abs=1e-12)


def test_misspelled_feature_name_rejected_and_named(client):
    """165 keys with one typo -> 422 naming the unrecognized key, no silent default."""
    typo = dict(SAMPLE_TX_1813992_FEATURES)
    typo["Local_feature_1a"] = typo.pop("Local_feature_1")

    response = client.post("/predict", json={"time_step": 35, **typo})
    assert response.status_code == 422
    body = json.dumps(response.json())
    assert "Local_feature_1a" in body
    assert "Local_feature_1" in body
    assert "Unrecognized" in body


def test_startup_rejects_correct_names_in_wrong_order(tmp_path):
    """165 canonical names in wrong order must fail startup, not silently misalign."""
    reordered = tmp_path / "features.json"
    reordered.write_text(
        json.dumps({"features": list(reversed(MODEL_FEATURES))}),
        encoding="utf-8",
    )

    with pytest.raises(ModelLoadError, match="order mismatch"):
        ModelContainer(feature_list_path=reordered)
