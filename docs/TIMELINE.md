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

## Phase 2 — Baseline

**Environment:** `notebooks/02_xgboost.ipynb` and `notebooks/03_graphsage.ipynb` on
Kaggle/Colab; laptop for MLflow configuration · **Status:** XGBoost executed 2026-09-25 on Colab and
awaiting review sign-off; GraphSAGE not started

**Done:** `notebooks/02_xgboost.ipynb` executed end to end on Google Colab on 2026-09-25, artifacts
in `xgboost/`, all 39 checks passing. Prevalence baseline PR-AUC 0.0650, Logistic Regression 0.2917,
XGBoost **0.8007** (ROC-AUC 0.9317) on test 35–49. Sub-window PR-AUC 0.9211 on 35–42 but 0.0423 on
43–49 (2.53% prevalence). Recorded in `docs/XGBOOST.md`.

**Objective:** Establish a strong classical ML baseline before any GNN is introduced.

**Tasks**

* Build the classical ML dataset with temporal splits (train 1–34 / validation 25–34 / test 35–49)
* Run an explicit feature leakage audit before training
* Train the prevalence baseline, Logistic Regression and the XGBoost baseline on Kaggle/Colab
* Baseline evaluation with PR-AUC, precision, recall, F1, confusion matrix and temporal sub-windows
* MLflow experiment tracking

**Expected output:** A reproducible XGBoost baseline with tracked metrics and a documented
feature set, recorded in `docs/XGBOOST.md` from the run artifacts.

**Open item before the baseline is treated as final:** the 500-tree cap was binding — 499 trees
kept, best validation PR-AUC at the last tree — so the reported metrics are a floor.

**Completion criteria:** Baseline results recorded in MLflow; evaluation protocol matches
the temporal split rules; no GNN claim is made without a working baseline.

**Scope note.** Graph features (degrees, component statistics, neighbourhood aggregates) are
deliberately *not* part of the Phase 2 feature set: the EDA computed them over the full transductive
graph including future steps, so using them here would leak. Graph feature engineering belongs to
`03_graphsage.ipynb`, where every feature must be step-bounded and audited the same way.

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
