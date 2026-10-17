# Plan

Checkbox tracker. Every phase is complete except MLflow/DVC and the engineering items listed under
"Open". Heavy ML ran in notebooks on Kaggle / Google Colab; the laptop handled engineering,
documentation and serving. See `ARCHITECTURE.md`.

## Notebook map

| Notebook | Phase |
| :--- | :--- |
| `notebooks/01_eda.ipynb` | 1: EDA |
| `notebooks/02_xgboost.ipynb`, `02_xgboost_v2.ipynb` | 2, 2b: XGBoost baseline and drift study |
| `notebooks/03_graphsage.ipynb` | 3, 3b: GraphSAGE |
| `notebooks/04_heterogeneous_gnn.ipynb` | 4: HeteroRGCN |
| `notebooks/05_temporal_inductive_evaluation.ipynb` | 5: temporal and inductive evaluation |
| `notebooks/06_explainability.ipynb` | 6: explainability |
| `notebooks/07_hgt.ipynb` | 7: HGT |
| `notebooks/08_ood_diagnosis.ipynb`, `08b_ood_permutation_check.ipynb` | 7b: OOD diagnosis |
| `notebooks/09_walkforward_retraining.ipynb` | 9a: walk-forward retraining |

## Done

- [x] **Phase 1, data and EDA.** Dataset verified (0 structural failures); EDA run on Colab.
- [x] **Phase 2, XGBoost.** Baseline and optimized model; frozen at PR-AUC 0.8013 (steps 35-49).
- [x] **Phase 2b, XGBoost drift study.** v2 reference 0.8070 (35-49), 0.0483 (43-49).
- [x] **Phase 3, GraphSAGE** (0.6209) and **3b** cross-step variant (0.6216).
- [x] **Phase 4, HeteroRGCN** over all four edge types (0.4682).
- [x] **Phase 5, temporal and inductive evaluation.**
- [x] **Phase 6, explainability.** SHAP, Captum/GNNExplainer, relation ablation.
- [x] **Phase 7, HGT** (0.4861).
- [x] **Phase 7b, OOD diagnosis.** Adversarial AUC 1.0000, leak audit, permutation check.
- [x] **Phase 8a, lag-safe monitoring.** Backtest, signal evaluation, retraining trigger.
- [x] **Phase 8b, FastAPI service and Docker.** Frozen XGBoost at tau* = 0.435; 49 pytest tests.
- [x] **Phase 8c, CI.** Tests plus scoped lint, model-consistency gates, Docker build.
- [x] **Phase 8d, deployment.** API on Render (`render.yaml`), dashboard on Vercel.
- [x] **Dashboard.** React + Vite + TypeScript, reading the API and exported artifacts.
- [x] **Phase 9a, walk-forward retraining** (`WALKFORWARD.md`). Verdict `PARTIAL`.
- [x] **API hardening (T12 subset).** CORS allow-list from `ALLOWED_ORIGINS`, xgboost pinned to
  3.4.1, slim Docker image, `model_version` on `/health`.

## Open

- [ ] **Phase 9, MLflow and DVC.** Experiment tracking and data/pipeline versioning.
- [ ] **T9.** Basis for `confidence_context` beyond a step lookup; `input_ood_score`.
- [ ] **T10.** Raw-transaction-ID feature pipeline.
- [ ] **T11.** Per-prediction `top_contributions` from SHAP `pred_contribs`.
- [ ] **Rigor extras.** Bootstrap CIs for the model table, GNN seed loops, fair GNN tuning, Phase 2b
  focal-loss re-run.

Detail and priorities: `IMPROVEMENT_PLAN.md`.
