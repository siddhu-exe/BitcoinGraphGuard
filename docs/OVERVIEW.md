# BitcoinGraphGuard — Overview

Everything about the project at a glance. For depth, follow the links in the docs map at
the end.

## What It Is

**BitcoinGraphGuard** is a temporal heterogeneous graph-based Bitcoin fraud detection system
built on the real **Elliptic++** dataset. It detects illicit activity at the **transaction**
and **wallet/address** level using classical ML, graph neural networks, and production-grade
MLOps.

**Status:** Dataset verified and EDA complete (see `EDA.md`). No models trained yet and no
experimental results yet.

## The Problem

Detecting fraud on Bitcoin is hard because:

* Labels are scarce and severely imbalanced (illicit is ~2% of activity).
* Addresses are pseudonymous and fraudsters imitate normal behaviour.
* Fraud patterns change over time (concept drift).
* Signal lives in relationships (money flow between transactions and wallets).
* The graph is large (~1.03M nodes, ~4.42M edges) and the laptop is small (~5.6 GB RAM).
* Labels arrive late, so evaluation must respect time.

## Dataset at a Glance

Real Elliptic++ data across **49 time steps** (roughly two-week windows).

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

## Models (planned progression)

1. **XGBoost** — classical baseline on engineered graph features.
2. **GraphSAGE** — homogeneous GNN baseline.
3. **RGCN** — heterogeneous GNN that respects the four edge types.
4. **HGT** — only if RGCN results justify it.

Each model must beat the previous one *under the same evaluation protocol* before any
improvement is claimed.

## Evaluation

* **Temporal split** — train on earlier time steps, test on later unseen ones.
* **Inductive** — evaluate on transactions/wallets never seen in training.
* **Metrics** — PR-AUC (primary), precision, recall, F1, confusion matrix. Not accuracy.
* **Temporal degradation** and **ablation studies** to show what actually matters.

## Explainability

Investigators need to know *why* a wallet was flagged. GNNExplainer (or equivalent)
surfaces the influential nodes, edges, and subgraphs, and is applied to real fraud cases,
false positives, and false negatives.

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
`notebooks/07_final_evaluation.ipynb`), and notebooks must run standalone on Colab/Kaggle.
Application/serving logic lives in `src/`, and remote runs export artifacts back.

## MLOps

**DVC** (data/pipeline versioning) · **MLflow** (experiments & model metadata) ·
**model artifacts** (reproducible checkpoints) · **drift monitoring** (model quality +
graph structure) · **retraining triggers** (threshold-driven, never arbitrary) ·
**FastAPI** (serving) · **Docker** (packaging) · **CI/CD** (GitHub Actions).

The system is **not production-ready** until deployment (Phase 7) is actually completed.

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
| `TIMELINE.md` | Nine-phase roadmap with objectives and completion criteria |
| `DATA_INVENTORY.md` | Verified dataset statistics and data-quality report |
| `OBJECTIVES.md` | Goal, requirements, definition of done |
| `PLAN.md` | Checkbox tracker for the roadmap |
| `PROGRESS.md` | Current status and next actions |
| `../AGENTS.md` / `../CLAUDE.md` | Operating rules for AI agents and contributors |
| `../README.md` | Entry point and setup instructions |
