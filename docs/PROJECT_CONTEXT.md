# BitcoinGraphGuard — Project Context

> **Read this first.** This is the single file an AI agent (or a new contributor) should
> read to quickly understand the whole project.

## Project Name

**BitcoinGraphGuard**

## One-Line Description

A Bitcoin fraud detector that knows when to stop trusting itself: a temporal heterogeneous
graph-based fraud detection system built on the real Elliptic++ dataset, with drift monitoring, a
serving API and a live dashboard (https://bitcoin-graph-delta.vercel.app).

## Problem

The system tries to detect **illicit Bitcoin activity** — fraud, scams, darknet market
payments, ransomware, and money laundering — at two levels:

* **Transaction level:** is this individual Bitcoin transaction illicit?
* **Wallet/address level:** is this actor (wallet) involved in illicit activity?

Bitcoin fraud detection is difficult because:

* **Labels are scarce and very imbalanced.** Almost all activity is normal, and only a
  small fraction is labeled illicit.
* **Actors hide.** Addresses are pseudonymous, and illicit actors deliberately imitate
  normal transaction patterns.
* **Behaviour changes over time.** Fraud patterns evolve, so a model trained on the past
  may degrade on the future.
* **Data is relational.** Money flows through chains of transactions and wallets, so
  isolated row-by-row models miss the most useful signal.
* **Data is large.** The graph has millions of nodes and edges and cannot be processed
  naively on a small laptop.
* **Labels arrive late.** A transaction's label may only be known after the fact, which is
  why evaluation must respect time.

## Dataset

The project uses **Elliptic++**, a real Bitcoin dataset of transactions, wallets/addresses,
and the relationships between them, spread across **49 time steps** (each step is about two
weeks of blockchain activity, per Weber et al., 2019).

Verified statistics (see `DATA_INVENTORY.md`):

| Item | Count |
| :--- | :--- |
| Transaction nodes | 203,769 |
| Wallet / address nodes | 822,942 |
| Transaction → transaction edges | 234,355 |
| Wallet → transaction edges (inputs) | 477,117 |
| Transaction → wallet edges (outputs) | 837,124 |
| Wallet → wallet edges | 2,868,964 |
| Unique nodes (total) | ~1.03M |
| Heterogeneous directed edges (total) | ~4.42M |
| Temporal steps | 49 |
| Raw dataset size | ~2.1 GB |

**Classes.** Labels are `1 = illicit`, `2 = licit`, `3 = unknown`.

* Transactions: illicit **4,545**, licit **42,019**, unknown **157,205**.
* Wallets: illicit **14,266**, licit **251,088**, unknown **557,588**.

This is **severe class imbalance**: illicit transactions are ~2.2% of all transactions.
**Unknown is not licit** — class 3 means "not labeled", and must never be treated as a
legitimate example.

## Core Idea

Model Bitcoin transactions and wallets as one **heterogeneous temporal graph**, then combine
several complementary approaches:

* **Classical ML** — a strong tabular baseline on engineered graph features.
* **Graph ML / deep learning** — GNNs that learn from transaction and wallet relationships.
* **Temporal evaluation** — train on earlier time steps, test on later unseen ones.
* **Inductive evaluation** — measure performance on transactions/wallets never seen in
  training.
* **Explainability** — show *why* the model flagged something, using graph structure.
* **MLOps** — track, version, serve, monitor, and retrain the system reproducibly.

The point is not just a high score. It is a realistic, leakage-free, explainable,
production-oriented fraud detection pipeline.

## Models

All four were trained and evaluated under one temporal protocol (fit 1-24, validate 25-34, refit
1-34, test 35-49):

| Model | Test PR-AUC (35-49) | Drift-window PR-AUC (43-49) |
| :--- | ---: | ---: |
| XGBoost, optimized (deployed) | 0.8013 | 0.0427 |
| GraphSAGE | 0.6209 | 0.0505 |
| HGT | 0.4861 | 0.0386 |
| HeteroRGCN | 0.4682 | 0.0550 |

The graph models were tuned less deeply than XGBoost, so the ranking is a result for this protocol,
not a general claim about graph methods.

## Evaluation

Evaluation is designed to reflect real deployment, not a random leaderboard:

* **Temporal split** — train on earlier time steps, evaluate on later unseen time steps.
* **Inductive evaluation** — evaluate on transactions/wallets not present during training.
* **Metrics** — PR-AUC (primary), precision, recall, F1, and confusion matrices. Accuracy is
  not used as the main metric because of the class imbalance.
* **Temporal degradation** — measure how much performance drops over time.
* **Ablation studies** — remove or change components to see what actually helps.

## Explainability

For fraud detection, a prediction alone is not enough. Investigators must understand *why*
the model flagged an address or transaction. GNN explainability (e.g. GNNExplainer) is used
to identify the influential nodes, edges, and subgraphs behind selected predictions, and to
study **false positives** (innocent wallets flagged) and **false negatives** (missed fraud).

## Serving, monitoring and MLOps

* **FastAPI** serves the frozen XGBoost model (threshold 0.435); each prediction carries a
  `confidence_context` for its time step. **Done.**
* **Docker** packages the service and **GitHub Actions** runs tests, frozen-model consistency gates
  and the image build. **Done.**
* **Lag-safe drift monitoring** and a retraining trigger: a decision at step t reads labels of
  steps <= t - L only. **Done.**
* **Walk-forward retraining** under label delay (`WALKFORWARD.md`). **Done.**
* **Deployment:** API on Render, React dashboard on Vercel. **Done.**
* **MLflow / DVC:** experiment tracking and data versioning. **Not started.**

## Compute Strategy

Work is split across two environments. This is a hard rule, not a preference.

* **Laptop = development and documentation only.** Writing and reviewing code, Git
  management, documentation, system design, reviewing notebook code, backend/FastAPI,
  Docker, MLOps engineering code, CI/CD, dashboard/frontend, and project configuration.
* **Kaggle / Google Colab = ALL ML execution, inside notebooks.** Dataset loading, ML data
  processing, EDA, feature engineering, graph construction, XGBoost, GraphSAGE, RGCN, HGT,
  hyperparameter tuning, ablations, temporal/inductive evaluation, error analysis,
  GNNExplainer, final evaluation, and model-artifact generation.

**Do not execute ML work on the laptop** — including EDA, feature engineering, and training.
The ML work lives in one notebook per phase (`notebooks/01_eda.ipynb` through
`notebooks/09_walkforward_retraining.ipynb`), and notebooks must run standalone on Colab/Kaggle.
Application and serving logic lives in `src/`. See `ARCHITECTURE.md`.

## Current Status

Everything through modeling, explainability, drift diagnosis, monitoring, walk-forward retraining,
the API, Docker, CI and deployment is **done** (`PROGRESS.md`, `PLAN.md`).

Key results (sources in `results/` and `reports/`):

* Test PR-AUC on steps 35-49: XGBoost 0.8013, best graph model (GraphSAGE) 0.6209.
* All four models collapse after step 43: XGBoost 0.9215 on steps 35-42, 0.0427 on 43-49.
* Adversarial validation separates steps 1-34 from 43-49 perfectly (AUC 1.0000), yet 5-fold CV
  inside 43-49 reaches PR-AUC 0.9199, so the signal is present and the transfer fails.
* Expanding-window retraining on steps 43-49: PR-AUC 0.6949 at label delay L=1, 0.3878 at L=2,
  0.1377 at L=3, 0.1050 at L=5 (frozen model 0.0406 at L=1). Verdict `PARTIAL`.
* Monitor (L=1): the label-free score-shift channel fires at step 43, the performance channel at
  step 44, and no critical alarm fires during steps 35-42. Validated on one drift event.

**Open work:** MLflow/DVC (Phase 9), `confidence_context` basis and `input_ood_score` (T9), a raw
transaction-ID feature pipeline (T10), per-prediction SHAP contributions (T11). See
`IMPROVEMENT_PLAN.md`.

**Limitations:** one drift event, real label delay unknown, XGBoost is the deployed model, and the
literature notes split-leakage concerns on Elliptic.

## Important Rules

* Do **not** replace Elliptic++ with a synthetic or toy dataset for primary evaluation.
* Do **not** introduce temporal leakage — no future information in training features or
  graph construction.
* Do **not** use a random split for the primary temporal evaluation.
* Do **not** fabricate metrics, results, or dataset statistics.
* **Unknown ≠ licit.** Class 3 is unlabeled, not legitimate.
* **Preserve the raw dataset.** Never modify, overwrite, or move files in `../Og data/`.
* Do **not** unnecessarily change the project direction.
