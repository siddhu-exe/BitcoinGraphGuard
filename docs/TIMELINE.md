# BitcoinGraphGuard — Project Timeline (Roadmap)

The complete project roadmap, in nine phases. Each phase lists its objective, tasks,
expected output, and completion criteria.

**No dates are given.** This is a dependency-ordered roadmap, not a calendar. Phases run in
order, though light Phase 1/2 work may overlap.

**Status legend:** `Completed` · `In progress` · `Not started`.

> **Relationship to other docs:** `PLAN.md` is the checkbox tracker for the same work.
> A few phase titles differ slightly between the two files (for example `PLAN.md`
> "Temporal & Inductive Evaluation" corresponds to this file's Phase 4 "Robust Evaluation"),
> but they describe the same phases. This file carries the detailed objectives.

## Phase 1 — Dataset & EDA

**Environment:** Laptop (verification) + `notebooks/01_eda.ipynb` on Kaggle/Colab (EDA) ·
**Status:** Complete (dataset verified; `notebooks/01_eda.ipynb` executed on Colab 2026-09-25,
results recorded in `EDA.md`)

**Objective:** Confirm the Elliptic++ dataset is complete, correct, and understood before
any modelling begins.

**Tasks**

* Dataset verification (files, rows, columns, IDs, edges, labels)
* Data inventory and data-quality report
* Temporal EDA across the 49 time steps
* Class imbalance analysis for transactions and wallets
* Graph statistics (degree distributions, connected structure, edge-type counts)
* Feature understanding (what each feature group means)

**Output:** `DATA_INVENTORY.md`, a reproducible verification script, and
`notebooks/01_eda.ipynb` (run 2026-09-25) describing the dataset's temporal and graph structure;
findings in `EDA.md`, artifacts in `eda/`.

**Outcome:** two mandatory corrections (347,569 duplicate wallet rows; 965 address-less
transactions with blank domain columns), 9.25:1 / 17.60:1 label imbalance, non-stationary
prevalence across the 49 steps, a transaction graph split into 49 single-time-step components, and
an adopted temporal protocol of train 1–34 / validation 25–34 / test 35–49.

**Completion criteria:** Every file verified; statistics recorded without fabrication;
temporal and class-imbalance behaviour documented **by the executed notebook**; no open
questions about column meaning.

**Already completed:** dataset verification, data inventory, verification script, and the
first pass of temporal/class/graph/feature EDA (run locally before the notebook architecture
was adopted; evidence in `reports/eda/`).
**Done:** `notebooks/01_eda.ipynb` executed end to end on Google Colab on 2026-09-25 and its
results recorded in `docs/EDA.md` (artifacts in `eda/`), giving the phase a standalone,
reproducible source of truth.

## Phase 2 — Classical Baseline (XGBoost)

**Environment:** `notebooks/02_xgboost.ipynb` on Kaggle/Colab; laptop for MLflow configuration ·
**Status:** Complete (baseline and optimisation pass executed on Colab 2026-09-25; artifacts in `xgboost/`)

**Done:** `notebooks/02_xgboost.ipynb` executed end to end on Google Colab on 2026-09-25 in two
passes, artifacts in `xgboost/`, 56 of 57 checks passing (the one failure is a mis-specified check,
not a result). Prevalence baseline PR-AUC 0.0650, Logistic Regression 0.2917, XGBoost baseline
**0.8007** (ROC-AUC 0.9317) and optimised XGBoost **0.8013** (ROC-AUC 0.9281) on test 35–49.
Sub-window PR-AUC 0.9215 on 35–42 but 0.0427 on 43–49 (2.53% prevalence). Recorded in
`docs/XGBOOST.md`.

**Objective:** Establish a strong classical ML baseline before any GNN is introduced.

**Tasks**

* Build the classical ML dataset with temporal splits (train 1–34 / validation 25–34 / test 35–49)
* Run an explicit feature leakage audit before training
* Train the prevalence baseline, Logistic Regression and the XGBoost baseline on Kaggle/Colab
* Baseline evaluation with PR-AUC, precision, recall, F1, confusion matrix and temporal sub-windows
* MLflow experiment tracking

**Expected output:** A reproducible XGBoost baseline with tracked metrics and a documented
feature set, recorded in `docs/XGBOOST.md` from the run artifacts.

**Tree-budget question settled 2026-09-25:** the 500-tree run consumed its whole budget (499 trees
kept, best validation PR-AUC on the second-to-last permitted round), which is why the cap was
re-tested. The revision lifted the budget to 1500 trees with patience 100, ran a seeded 20-trial
randomised search scored on validation PR-AUC only, ablated `has_addresses` against a
pre-registered margin and tested one label-free fit-window redundancy reduction. Result: 582 trees
kept, so early stopping ended training rather than the cap, and the primary metric moved by +0.0006
PR-AUC while ROC-AUC fell 0.0036 and F1 fell 0.0052. **The baseline is not capacity-limited**; the
features and the 43–49 regime are the limits.

**Completion criteria:** Baseline results recorded; evaluation protocol matches the temporal
split rules; frozen benchmark established (PR-AUC 0.8013).

## Phase 3 — Homogeneous Graph Baseline (GraphSAGE)

**Environment:** `notebooks/03_graphsage.ipynb` on Kaggle/Colab ·
**Status:** Complete (executed on Colab 2026-09-25; artifacts in `GraphSage/`; report in `docs/GRAPHSAGE.md`)

**Objective:** Establish a homogeneous GNN baseline on the transaction-to-transaction graph
(`txs_edgelist.csv`) and test whether graph neighborhood aggregation improves over the frozen
XGBoost baseline (PR-AUC 0.8013).

**Tasks**

* Author `notebooks/03_graphsage.ipynb` with self-contained fallback for environments without PyG
* Author `docs/GRAPHSAGE.md` detailing architecture, temporal protocol, and ablation design
* Empirically verify intra-step edge confinement across all 234,355 transaction edges (100% intra-step)
* Implement 2-layer GraphSAGE (mean aggregation, hidden dim 128, dropout 0.3)
* Implement architectural twin 2-layer MLP baseline for controlled neural ablation
* Enforce leakage-free normalization (`StandardScaler` fitted strictly on historical splits)
* Apply `nn.BCEWithLogitsLoss(pos_weight=...)` for class imbalance (~9.64 on fit, ~7.63 on train)
* Implement validation PR-AUC early stopping and F1-maximizing operating threshold
* Decompose test evaluation across degree buckets and sub-windows
* Implement automated assertion suite and artifact export to `GraphSage/`
* Execute notebook remotely on Colab/Kaggle and record findings

**Outcome:** GraphSAGE achieves **PR-AUC 0.6209 / ROC-AUC 0.9044 / F1 0.5945** on test 35–49 against
2-layer MLP **0.4768 / 0.8912 / 0.5759** (+0.1442 absolute lift over neural baseline) and frozen XGBoost
**0.8013 / 0.9281 / 0.7818**. Intra-step edge confinement (100% intra-step) explains why homogeneous
GNNs cannot bridge temporal steps. Ready for Phase 4.

**Completion criteria:** GraphSAGE and MLP evaluated under the identical temporal protocol as
XGBoost; intra-step edge confinement verified; breakdown recorded;
artifacts exported to `GraphSage/`.

## Phase 4 — Heterogeneous Graph Deep Learning (RGCN / HGT)

**Environment:** `notebooks/04_heterogeneous_gnn.ipynb` on Kaggle/Colab ·
**Status:** Complete — executed on Google Colab (CUDA GPU) 2026-09-25, artifacts in
`results/heterogeneous_gnn/`, 46/46 checks passing; record in `docs/HETEROGENEOUS_GNN.md`

**Objective:** Learn from heterogeneous graph structure (transactions + wallets/addresses across
all 4 edge types: `AddrTx`, `TxAddr`, `AddrAddr`, `txs_edgelist`) to bridge isolated transactions
and capture multi-step temporal flow.

**Tasks**

* Build the heterogeneous graph (transaction + wallet nodes, four edge types)
* Implement and train RGCN (heterogeneous GNN)
* Evaluate HGT only if RGCN results justify it
* Hyperparameter experiments and ablations
* Benchmark against frozen XGBoost (0.8013) and GraphSAGE

**Expected output:** Trained heterogeneous GNN models with tracked configurations and metrics,
comparable against XGBoost and GraphSAGE.

**Completion criteria:** RGCN (and optionally HGT) trained and evaluated under the same temporal
protocol; any HGT use is explicitly justified.

## Phase 5 — Robust Evaluation

**Environment:** `notebooks/05_temporal_inductive_evaluation.ipynb` on Kaggle/Colab ·
**Status:** Notebook authored, awaiting remote run (no ML executed on the laptop)

**Objective:** Determine whether the models generalise over time and to unseen actors — and whether the
heterogeneous graph's temporal bridge actually helps — instead of reporting a single aggregate score.

**Tasks**

* Per-step evaluation of the frozen XGBoost/GraphSAGE/RGCN models for steps 35–49
* Label-free graph, address-activity and feature drift diagnostics per step
* Static (1–34) vs expanding (1..t-1) RGCN retraining, plus one pre-registered rolling window (W=20)
* Seen vs unseen address context and historical `T→A→T` path breakdowns
* Frozen vs adaptive (vs labelled oracle) threshold analysis, with PR-AUC kept primary
* Programmatic verification of the temporal rule: features/edges `<= t`, labels `< t`

**Expected output:** A temporal and inductive evaluation report (`docs/TEMPORAL_INDUCTIVE_EVALUATION.md`)
plus the tables and figures in `temporal_inductive/`.

**Completion criteria:** Temporal degradation is measured per step and located; static vs adaptive
regimes are compared; inductive and historical-path breakdowns are reported with sample sizes; the
evaluation is leakage-free and every check passes.

## Phase 5 — Explainability

**Environment:** `notebooks/06_explainability.ipynb` on Kaggle/Colab ·
**Status:** Not started

**Objective:** Explain individual predictions so results are trustworthy for fraud
investigation.

**Tasks**

* Apply GNNExplainer (or equivalent) to selected predictions
* Analyse representative fraud cases
* Analyse false positives (innocent wallets flagged)
* Analyse false negatives (missed fraud)
* Document explainability limitations

**Expected output:** Explanation artifacts (influential nodes/edges/subgraphs) and a
written case analysis.

**Completion criteria:** Selected fraud, false-positive, and false-negative cases are
explained and their limits stated.

## Phase 6 — MLOps

**Environment:** Laptop · **Status:** Not started

**Objective:** Make the project reproducible, trackable, and monitorable.

**Tasks**

* DVC for dataset/pipeline versioning
* MLflow for experiment and model tracking
* Model registry / artifact management
* Graph drift monitoring (degree distribution, communities, subgraphs)
* Model drift monitoring (performance over time)
* Define retraining triggers and thresholds

**Expected output:** Versioned data and pipelines, tracked experiments, and a documented
drift and retraining policy.

**Completion criteria:** Any experiment can be reproduced from versioned inputs, and
retraining is driven by defined thresholds.

## Phase 7 — Production System

**Environment:** Laptop · **Status:** Not started

**Objective:** Expose the trained model as a real inference service.

**Tasks**

* FastAPI inference service (model loading, endpoints, request validation)
* Logging and error handling
* Dockerize the service
* CI/CD with GitHub Actions
* Model validation before deployment
* Deployment

**Expected output:** A containerized, tested inference API with an automated pipeline.

**Completion criteria:** Service runs in Docker, passes automated checks, and is validated
and deployed. Until then the system is not production-ready.

## Phase 8 — Monitoring & Dashboard

**Environment:** Laptop · **Status:** Not started

**Objective:** Give a live view of model and graph health.

**Tasks**

* Model metrics display
* Drift views (graph and model)
* Temporal performance views
* Prediction monitoring
* Explainability views

**Expected output:** A monitoring dashboard backed by tracked experiments.

**Completion criteria:** Dashboard reflects real tracked data and is validated against
MLflow/DVC records.

## Phase 9 — Finalization

**Environment:** `notebooks/07_final_evaluation.ipynb` on Kaggle/Colab for final runs;
laptop for docs and packaging · **Status:** Not started

**Objective:** Freeze the final system and document it end to end.

**Tasks**

* Final reproducible experiments
* Documentation (experiments, limitations, architecture)
* Architecture diagrams
* Numbers-first README
* Deployment validation
* Project demonstration

**Expected output:** A frozen final model/version with complete documentation and a
reproducible end-to-end demonstration.

**Completion criteria:** Final results reproduce from versioned inputs; documentation is
complete and accurate; deployment is validated.
