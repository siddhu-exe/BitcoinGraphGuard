# BitcoinGraphGuard: Project Timeline (Roadmap)

A dependency-ordered roadmap with no calendar dates. Each phase is listed once, with its
objective, outcome and status. `PLAN.md` is the checkbox tracker and `PROGRESS.md` the current status.

**Status legend:** `Complete` · `Not started`.

## Phase 1: Dataset and EDA (Complete)

Confirm the Elliptic++ data is complete and understand it before modelling. Streaming verification of
all nine CSVs, then `01_eda.ipynb` on Colab. Outcome: two corrections (347,569 duplicate wallet rows;
965 address-less transactions), 9.25:1 / 17.60:1 label imbalance, 49 single-step transaction
components, and the temporal protocol: fit 1-24, validate 25-34, refit 1-34, test 35-49. Docs:
`DATA_INVENTORY.md`, `EDA.md`.

## Phase 2: XGBoost baseline and drift study (Complete)

A strong tabular baseline first. Optimized XGBoost reaches test PR-AUC 0.8013 (ROC-AUC 0.9281) on
steps 35-49, and 0.0427 on steps 43-49 (2.53% prevalence). The tree budget was not the constraint.
Phase 2b (`XGBOOST_V2.md`) studied the drift window and is the reference for the monitoring work.
Docs: `XGBOOST.md`, `XGBOOST_V2.md`.

## Phase 3: GraphSAGE (Complete)

Homogeneous GNN on the transaction graph. PR-AUC 0.6209 against a 2-layer MLP twin at 0.4768.
Transaction edges are 100% intra-step, so a homogeneous GNN cannot carry information across time
steps. Phase 3b tested cross-step and lag features (0.6216). Docs: `GRAPHSAGE.md`, `GRAPHSAGE_V2.md`.

## Phase 4: Heterogeneous GNN, RGCN (Complete)

Transactions and wallets over all four edge types. PR-AUC 0.4682 on steps 35-49 (0.0550 on 43-49).
Doc: `HETEROGENEOUS_GNN.md`.

## Phase 5: Temporal and inductive evaluation (Complete)

Per-step evaluation, drift diagnostics, static versus expanding retraining, seen versus unseen
address context. Located the collapse at step 43. Doc: `TEMPORAL_INDUCTIVE_EVALUATION.md`.

## Phase 6: Explainability (Complete)

SHAP for XGBoost (61.5% of attribution on local features), Captum/GNNExplainer and relation
ablation for the graph models. Doc: `EXPLAINABILITY.md`.

## Phase 7: Heterogeneous Graph Transformer (Complete)

HGT reaches PR-AUC 0.4861 on steps 35-49 and 0.0386 on 43-49, in the same collapse band as the
other models. Doc: `HGT.md`.

## Phase 7b: OOD diagnosis and leak audit (Complete)

Adversarial AUC 1.0000 between steps 1-34 and 43-49, while 5-fold CV inside 43-49 reaches PR-AUC
0.9199: the signal is present and the transfer fails. A permutation check and leak audit complete the
diagnosis. Doc: `OOD_DIAGNOSIS.md`.

## Phase 8: Monitoring, serving and deployment (Complete)

- **8a, monitoring.** Lag-safe drift engines (a decision at step t reads labels of steps <= t - L
  only), signal evaluation and backtest. Reports: `reports/monitoring_backtest_report.md`,
  `reports/monitoring_signal_evaluation.md`.
- **8b, API and Docker.** FastAPI service for the frozen XGBoost model with a per-step
  `confidence_context`; CPU-only image. Doc: `src/api/README.md`.
- **8c, CI.** Tests with a scoped lint gate, frozen-model consistency gates, Docker build.
  Doc: `CI_CD.md`.
- **8d, deployment and dashboard.** API on Render, React dashboard on Vercel. Doc: `DEPLOY.md`.

## Phase 9a: Walk-forward retraining under label delay (Complete)

Retraining recovers performance only when labels arrive fast. Steps 43-49 PR-AUC with expanding
retraining: 0.6949 at L=1, 0.3878 at L=2, 0.1377 at L=3, 0.1050 at L=5. Verdict `PARTIAL`.
Doc: `WALKFORWARD.md`.

## Phase 9: MLOps, MLflow and DVC (Not started)

Experiment tracking and dataset/pipeline versioning. Completion criterion: any experiment can be
reproduced from versioned inputs.

## Remaining engineering items (Not started)

`confidence_context` basis and `input_ood_score` (T9), a raw-transaction-ID feature pipeline (T10),
per-prediction SHAP contributions (T11). Detail in `IMPROVEMENT_PLAN.md`.
