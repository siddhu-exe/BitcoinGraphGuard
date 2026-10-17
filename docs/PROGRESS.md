# Progress

Current status of BitcoinGraphGuard. Per-phase detail lives in the linked docs; older
running notes are in the git history of this file.

## Status

Everything through modeling, explainability, drift diagnosis, monitoring, walk-forward
retraining, the serving API, Docker, CI and deployment is **done**. What is genuinely open is
listed under [Open work](#open-work).

| Phase | Work | Status | Record |
| :--- | :--- | :--- | :--- |
| 1 | Dataset verification and EDA | Done | `DATA_INVENTORY.md`, `EDA.md` |
| 2 / 2b | XGBoost baseline and drift study | Done | `XGBOOST.md`, `XGBOOST_V2.md` |
| 3 / 3b | GraphSAGE and cross-step variant | Done | `GRAPHSAGE.md`, `GRAPHSAGE_V2.md` |
| 4 | HeteroRGCN | Done | `HETEROGENEOUS_GNN.md` |
| 5 | Temporal and inductive evaluation | Done | `TEMPORAL_INDUCTIVE_EVALUATION.md` |
| 6 | Explainability (SHAP, Captum, relation ablation) | Done | `EXPLAINABILITY.md` |
| 7 | Heterogeneous Graph Transformer | Done | `HGT.md` |
| 7b | OOD diagnosis and leak audit | Done | `OOD_DIAGNOSIS.md` |
| 8a | Lag-safe drift monitoring and backtest | Done | `reports/monitoring_backtest_report.md`, `reports/monitoring_signal_evaluation.md` |
| 8b | FastAPI inference service and Docker | Done | `src/api/README.md`, `Dockerfile` |
| 8c | GitHub Actions CI | Done | `CI_CD.md` |
| 8d | Deployment (Render API, Vercel dashboard) | Done | `DEPLOY.md`, `render.yaml`, `frontend/vercel.json` |
| 9a | Walk-forward retraining under label delay | Done | `WALKFORWARD.md`, `results/walkforward/` |
| 9 | MLflow and DVC | Not started | `IMPROVEMENT_PLAN.md` |

## Headline results (all traceable to `results/` and `reports/`)

- Test PR-AUC on steps 35-49 (pooled): XGBoost 0.8013, GraphSAGE 0.6209, HGT 0.4861,
  HeteroRGCN 0.4682.
- Drift window 43-49 PR-AUC: XGBoost 0.0427, GraphSAGE 0.0505, HeteroRGCN 0.0550, HGT 0.0386.
- Adversarial AUC between steps 1-34 and 43-49 is 1.0000; 5-fold CV inside 43-49 reaches PR-AUC
  0.9199.
- Walk-forward expanding retraining, steps 43-49 PR-AUC by label delay: L=1 0.6949, L=2 0.3878,
  L=3 0.1377, L=5 0.1050 (frozen model 0.0406 at L=1). Verdict `PARTIAL`.
- Deployed model: the frozen XGBoost at threshold 0.435 behind FastAPI, with 49 pytest tests and
  three CI workflows.

## Open work

Source of truth: `IMPROVEMENT_PLAN.md`.

- **Phase 9: MLflow and DVC.** Planned only; there is no `dvc.yaml`.
- **T9:** a real basis for `confidence_context` (it is a step lookup today) and an `input_ood_score`.
- **T10:** a raw-transaction-ID feature pipeline (the API needs the 165 precomputed features).
- **T11:** `top_contributions` per prediction via SHAP `pred_contribs`.
- Rigor extras (ask before running): bootstrap CIs for the headline model table, GNN seed loops, fair
  GNN tuning, and the Phase 2b focal-loss re-run on Colab.

## Rules that still apply

- Temporal splits only; report both 35-49 and 43-49; PR-AUC is the primary metric.
- Class 3 is unknown, never licit; no leakage; no fabricated numbers.
- Raw data in `Og data/` is read-only. Heavy ML runs on Colab/Kaggle, not the laptop.
- Frozen artifacts (`results/xgboost/**`, `results/xgboost_v2/**`,
  `results/temporal_inductive/per_step_metrics.csv`) are never modified.
