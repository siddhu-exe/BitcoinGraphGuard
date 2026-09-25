# BitcoinGraphGuard — Project Context

> **Read this first.** This is the single file an AI agent (or a new contributor) should
> read to quickly understand the whole project.

## Project Name

**BitcoinGraphGuard**

## One-Line Description

A temporal heterogeneous graph-based Bitcoin fraud detection system built using the real
Elliptic++ dataset.

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
and the relationships between them, spread across **49 time steps** (each step is roughly a
two-week window of blockchain activity).

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

The planned progression, from simplest to most complex:

1. **XGBoost baseline** — classical gradient boosting on engineered transaction features.
   Establishes how far non-graph ML gets and gives a fair reference point.
2. **GraphSAGE** — a homogeneous graph neural network baseline. Learns from neighbourhood
   structure and shows how much graph structure adds over XGBoost.
3. **RGCN** (Relational GCN) — a heterogeneous GNN that respects different edge types
   (tx→tx, wallet→tx, tx→wallet, wallet→wallet).
4. **HGT** (Heterogeneous Graph Transformer) — evaluated **only if justified** as a
   stronger alternative to RGCN.

No later model is claimed to be better until it is actually trained and evaluated.

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

## MLOps

* **DVC** — version datasets and pipelines.
* **MLflow** — track experiments, parameters, metrics, and model metadata.
* **Model artifacts** — save checkpoints and metadata reproducibly.
* **Drift monitoring** — watch both model performance and graph structure
  (degree distribution, communities, temporal degradation).
* **Retraining triggers** — retraining is driven by defined thresholds, not guesswork.
* **FastAPI** — serve model inference.
* **Docker** — package production services.
* **CI/CD** — automated checks with GitHub Actions.

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
`notebooks/07_final_evaluation.ipynb`), and notebooks must run standalone on Colab/Kaggle.
Application and serving logic lives in `src/`. See `ARCHITECTURE.md`.

## Current Status

Dataset verification is **complete**; exploratory data analysis is **authored but not yet
executed** (see `PROGRESS.md` and `EDA.md`):

* **Done:** Elliptic++ downloaded and fully verified with a streaming, low-memory verifier
  (`../scripts/verify_dataset.py`); data inventory in `DATA_INVENTORY.md`; a first pass of
  temporal, class-imbalance, graph and feature EDA run locally (`reports/eda/`);
  `notebooks/01_eda.ipynb` authored as the canonical EDA entrypoint (`../scripts/eda_phase2.py`
  is the superseded reference implementation of the same logic); `requirements.txt` created;
  implementation architecture and compute split defined.
* **Not done yet:** executing `notebooks/01_eda.ipynb` on Colab/Kaggle and recording the
  results in `EDA.md`; the remaining phase notebooks; DVC setup; the `src/` project structure;
  MLflow tracking; and the Kaggle/Colab environment. `EDA.md` stays pending until that first
  notebook run has been reviewed.

No models have been trained and **no experimental results exist yet**.

## Important Rules

* Do **not** replace Elliptic++ with a synthetic or toy dataset for primary evaluation.
* Do **not** introduce temporal leakage — no future information in training features or
  graph construction.
* Do **not** use a random split for the primary temporal evaluation.
* Do **not** fabricate metrics, results, or dataset statistics.
* **Unknown ≠ licit.** Class 3 is unlabeled, not legitimate.
* **Preserve the raw dataset.** Never modify, overwrite, or move files in `../Og data/`.
* Do **not** unnecessarily change the project direction.
