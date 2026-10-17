# BitcoinGraphGuard — Overview

Everything about the project at a glance. For depth, follow the links in the docs map at
the end.

## What It Is

**BitcoinGraphGuard** is a temporal heterogeneous graph-based Bitcoin fraud detection system
built on the real **Elliptic++** dataset. It detects illicit activity at the **transaction**
and **wallet/address** level using classical ML, graph neural networks, drift monitoring,
and a deployed inference service.

**Status:** The modelling, drift diagnosis, monitoring, walk-forward retraining, serving API,
Docker image, CI and deployment are complete. The open items are MLflow/DVC and a few API
improvements (see `PROGRESS.md`). Live demo: https://bitcoin-graph-delta.vercel.app

**Three findings** (steps 35-49 pooled unless stated; sources in `results/` and `reports/`):

1. XGBoost beat every graph model: PR-AUC 0.8013 against 0.6209 for the best GNN (GraphSAGE).
2. All four models collapse after step 43: XGBoost goes from 0.9215 on steps 35-42 to 0.0427 on
   43-49.
3. Retraining recovers only with fast labels: expanding-window retraining reaches 0.6949 PR-AUC on
   steps 43-49 at a one-step label delay, 0.3878 at two, 0.1377 at three and 0.1050 at five
   (frozen model: 0.0406 at one).

## The Problem

Detecting fraud on Bitcoin is hard because:

* Labels are scarce and severely imbalanced (illicit is ~2% of activity).
* Addresses are pseudonymous and fraudsters imitate normal behaviour.
* Fraud patterns change over time (concept drift).
* Signal lives in relationships (money flow between transactions and wallets).
* The graph is large (~1.03M nodes, ~4.42M edges) and the laptop is small (~5.6 GB RAM).
* Labels arrive late, so evaluation must respect time.

## Dataset at a Glance

Real Elliptic++ data across **49 time steps** (about two weeks each, per Weber et al., 2019).

| Item | Count |
| :--- | :--- |
| Transaction nodes | 203,769 |
| Wallet / address nodes | 822,942 |
| Transaction → transaction edges | 234,355 |
| Wallet → transaction edges | 477,117 |
| Transaction → wallet edges | 837,124 |
| Wallet → wallet edges | 2,868,964 |
| Unique nodes (total) | ~1.03M |
| Heterogeneous directed edges (total) | ~4.42M |
| Raw dataset size | ~2.1 GB |

**Labels:** `1 = illicit`, `2 = licit`, `3 = unknown`.

| Node type | Illicit | Licit | Unknown |
| :--- | ---: | ---: | ---: |
| Transactions | 4,545 | 42,019 | 157,205 |
| Wallets | 14,266 | 251,088 | 557,588 |

**Unknown is not licit.** Class 3 means "unlabeled" and must never be used as a legitimate
example.

## Approach

Model transactions and wallets as one heterogeneous temporal graph, then combine:
classical ML · graph ML · deep learning · temporal evaluation · inductive evaluation ·
explainability · MLOps.

## Models

| Model | Test PR-AUC (35-49) | Drift-window PR-AUC (43-49) |
| :--- | ---: | ---: |
| XGBoost (deployed) | 0.8013 | 0.0427 |
| GraphSAGE | 0.6209 | 0.0505 |
| HGT | 0.4861 | 0.0386 |
| HeteroRGCN | 0.4682 | 0.0550 |

All four were trained and evaluated under the same temporal protocol. The graph models were tuned
less deeply than XGBoost, so the gap is a result for this protocol, not a general claim.

## Evaluation

* **Temporal split** — train on earlier time steps, test on later unseen ones.
* **Inductive** — evaluate on transactions/wallets never seen in training.
* **Metrics** — PR-AUC (primary), precision, recall, F1, confusion matrix. Not accuracy.
* **Temporal degradation** and **ablation studies** to show what actually matters.

## Explainability

Investigators need to know *why* a wallet was flagged. SHAP explains the XGBoost model
and Captum/GNNExplainer plus relation ablation explain the graph models (`EXPLAINABILITY.md`).

## Compute Strategy

Two environments. This is a hard rule.

| Laptop — development & documentation only | Kaggle / Google Colab — ALL ML execution (in notebooks) |
| :--- | :--- |
| Code writing & review, Git | Dataset loading & ML data processing |
| Documentation, system design | **EDA** & feature engineering |
| Notebook code review, config, tests | Graph construction for training |
| FastAPI, Docker, CI/CD | **XGBoost**, GraphSAGE, RGCN/HGT training |
| MLOps engineering, monitoring, dashboard | Hyperparameter tuning, ablations |
| Documentation | Temporal/inductive evaluation, error analysis, GNNExplainer, final evaluation |

**No ML runs on the laptop** — not training, and not "lightweight" EDA or feature
engineering. The ML work lives in one notebook per phase (`notebooks/01_eda.ipynb` through
`notebooks/09_walkforward_retraining.ipynb`), and notebooks run standalone on Colab/Kaggle.
Application/serving logic lives in `src/`, and remote runs export artifacts back.

## Serving and MLOps

Done: **FastAPI** serving the frozen XGBoost model, **Docker** packaging, **GitHub Actions** CI
(tests, model-consistency gates, Docker build), lag-safe **drift monitoring** and a deployed React
dashboard. Not started: **MLflow** experiment tracking and **DVC** versioning.

## Key Rules

* No synthetic replacement dataset.
* No temporal leakage — no future information in features or graph construction.
* No random split for the primary temporal evaluation.
* No fabricated metrics or dataset statistics.
* Unknown ≠ licit.
* Preserve the raw dataset (`../Og data/` is read-only).
* Do not unnecessarily change the project direction.

## Documents Map

| File | What it covers |
| :--- | :--- |
| `PROJECT_CONTEXT.md` | Read first — full project context for humans and AI agents |
| `ARCHITECTURE.md` | Technical architecture: data, ML, training, evaluation, serving |
| `TIMELINE.md` | Phase roadmap with objectives, outcomes and status |
| `DATA_INVENTORY.md` | Verified dataset statistics and data-quality report |
| `OBJECTIVES.md` | Goal, requirements, definition of done |
| `PLAN.md` | Checkbox tracker for the roadmap |
| `PROGRESS.md` | Current status and open work |
| `WALKFORWARD.md` | Walk-forward retraining under label delay |
| `IMPROVEMENT_PLAN.md` | What remains to be done |
| `../AGENTS.md` / `../CLAUDE.md` | Operating rules for AI agents and contributors |
| `../README.md` | Entry point and setup instructions |
