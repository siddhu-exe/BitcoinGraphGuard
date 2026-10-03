# Deploying the inference API to Render

The service is the CPU-only XGBoost API in `src/api/` (see `Dockerfile`). Render builds the image
from this repo, so every runtime artifact must be committed: `results/xgboost/xgb_model_optimized.json`,
`results/xgboost/selected_features.json`, `results/temporal_inductive/per_step_metrics.csv` and
`reports/monitoring_backtest_report.json`. All four are tracked in git today.

## Steps

1. Push the repo (including `render.yaml`) to GitHub.
2. In the Render dashboard: **New > Blueprint**, pick the repo, and apply. Render reads `render.yaml`
   (Docker runtime, free plan, health check path `/health`, no secrets).
   - Without a Blueprint: **New > Web Service**, runtime **Docker**, plan **Free**, Health Check Path `/health`.
3. Wait for the first build and deploy. Render sets `PORT`; the container binds `0.0.0.0:$PORT`.
4. Smoke test (replace the host):
   ```bash
   curl https://<service>.onrender.com/health
   curl https://<service>.onrender.com/monitoring/status
   ```
   Then `POST /predict` with a 165-feature body (see `tests/test_api.py::SAMPLE_TX_1813992_FEATURES`);
   tx 1813992 at step 35 must return about `1.7188735e-05`.

If an artifact is missing from the image the service now exits at startup (model, feature list, and
monitoring report are all checked), so a bad deploy shows as a failed deploy rather than a running
service returning errors.

## Free-tier caveats

- **Cold start:** the service spins down after inactivity; the next request waits about 50 s while it
  restarts. Model load itself is under a second. Health checks do not keep it awake.
- **512 MB RAM:** idle after startup is about 140 MiB. A 10,000-row `/batch_predict` peaked at about
  511 MiB locally, which is the entire limit, so `render.yaml` sets `MAX_BATCH_SIZE=1000`. Keep batches
  small. The free plan also has very little CPU: a 1,000-row batch took ~7 s on a laptop-class CPU.
- **Ephemeral disk:** `logs/predictions.jsonl` is lost on every redeploy or restart. Treat it as
  debug output, not an audit trail. Writes never crash inference if the directory is unavailable.
- **Monitoring is a snapshot:** `/monitoring/status` serves the committed backtest report (latest step
  49), not live telemetry. Refresh it by re-running `scripts/run_monitoring_backtest.py` and committing
  `reports/monitoring_backtest_report.json`.
- **Reliability caveat:** steps 43+ are in the drift regime where the frozen model is unreliable; the
  API flags this in each response's `confidence_context`.

## Local parity

```bash
docker build -t bitcoingraphguard:latest .
docker run --rm -e PORT=9123 -p 9123:9123 bitcoingraphguard:latest   # same port mechanism as Render
docker compose up                                                    # still serves on 8000
```
