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
**Status:** In progress (EDA done locally; notebook consolidation pending)

**Objective:** Confirm the Elliptic++ dataset is complete, correct, and understood before
any modelling begins.

**Tasks**

* Dataset verification (files, rows, columns, IDs, edges, labels)
* Data inventory and data-quality report
* Temporal EDA across the 49 time steps
* Class imbalance analysis for transactions and wallets
* Graph statistics (degree distributions, connected structure, edge-type counts)
* Feature understanding (what each feature group means)

**Expected output:** `DATA_INVENTORY.md`, a reproducible verification script, and
`notebooks/01_eda.ipynb` describing the dataset's temporal and graph structure.

**Completion criteria:** Every file verified; statistics recorded without fabrication;
temporal and class-imbalance behaviour documented; no open questions about column meaning.

**Already completed:** dataset verification, data inventory, verification script, and
temporal/class/graph/feature EDA (recorded in `docs/EDA.md`).
**Still to do:** consolidate the EDA into `notebooks/01_eda.ipynb` so it runs standalone
on Kaggle/Colab.

## Phase 2 — Baseline

**Environment:** `notebooks/02_xgboost.ipynb` and `notebooks/03_graphsage.ipynb` on
Kaggle/Colab; laptop for MLflow configuration · **Status:** Not started

**Objective:** Establish a strong classical ML baseline before any GNN is introduced.

**Tasks**

* Graph feature engineering (degrees, transaction statistics, neighbourhood aggregates)
* Build the classical ML dataset with temporal splits
* Train the XGBoost baseline on Kaggle/Colab
* Baseline evaluation with PR-AUC, precision, recall, F1, confusion matrix
* MLflow experiment tracking

**Expected output:** A reproducible XGBoost baseline with tracked metrics and a documented
feature set.

**Completion criteria:** Baseline results recorded in MLflow; evaluation protocol matches
the temporal split rules; no GNN claim is made without a working baseline.

## Phase 3 — Graph Deep Learning

**Environment:** `notebooks/04_heterogeneous_gnn.ipynb` on Kaggle/Colab ·
**Status:** Not started

**Objective:** Learn from graph structure with GNNs, starting simple and escalating only
when justified.

**Tasks**

* Build the heterogeneous graph (transaction + wallet nodes, four edge types)
* Implement and train GraphSAGE (homogeneous baseline)
* Implement and train RGCN (heterogeneous)
* Evaluate HGT only if RGCN results justify it
* Hyperparameter experiments

**Expected output:** Trained GNN models with tracked configurations and metrics, comparable
against the XGBoost baseline.

**Completion criteria:** GraphSAGE and RGCN are trained and evaluated under the same
protocol as XGBoost; any HGT use is explicitly justified.

## Phase 4 — Robust Evaluation

**Environment:** `notebooks/05_temporal_inductive_evaluation.ipynb` on Kaggle/Colab ·
**Status:** Not started

**Objective:** Prove the models generalize over time and to unseen actors, not just on a
random split.

**Tasks**

* Temporal split evaluation (train on past, test on future)
* Inductive evaluation on unseen transactions/wallets
* Ablation studies (edge types, features, sampling, imbalance handling)
* Error analysis across time steps and node types

**Expected output:** A temporal and inductive evaluation report plus ablation results.

**Completion criteria:** Temporal degradation is measured and documented; ablations show
which components help; evaluation is leakage-free.

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
