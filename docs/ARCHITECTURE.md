# BitcoinGraphGuard — Technical Architecture

This is the detailed architecture for BitcoinGraphGuard: what the system is made of, where
each part runs, and how data and models move through it. It does not change the project
objective. For a fast summary see `PROJECT_CONTEXT.md`; for the phased plan see
`TIMELINE.md`.

## System Overview

```text
Elliptic++
    ↓
Data Verification
    ↓
EDA / Feature Engineering
    ↓
Graph Construction
    ↓
Model Training on Kaggle/Colab
    ↓
XGBoost → GraphSAGE → RGCN/HGT
    ↓
Evaluation
    ↓
Model Artifacts
    ↓
MLflow / DVC
    ↓
FastAPI
    ↓
Docker
    ↓
CI/CD
    ↓
Monitoring / Dashboard
```

Everything above *Model Training* happens on the laptop. Training and heavy evaluation
happen on Kaggle/Google Colab. Everything from *Model Artifacts* onward happens back on the
laptop.

## Data Architecture

The dataset is a **heterogeneous graph**: two node types connected by four directed edge
types. "Heterogeneous" means the graph intentionally contains more than one kind of node
and edge.

**Node types**

* **Transaction nodes** (`txId`) — 203,769 nodes. Each has 182 features
  (`txs_features.csv` is 184 columns: `txId`, `Time step`, and 182 features) plus a class
  label (`1` illicit, `2` licit, `3` unknown).
* **Wallet / address nodes** (`address`) — 822,942 nodes. Each has 55 behavioural features
  plus a class label.

**Edge types**

* **Transaction → transaction** (`txs_edgelist.csv`) — 234,355 edges. Bitcoin flow between
  transactions.
* **Wallet → transaction** (`AddrTx_edgelist.csv`) — 477,117 edges. Addresses that provided
  inputs to a transaction.
* **Transaction → wallet** (`TxAddr_edgelist.csv`) — 837,124 edges. Addresses that received
  outputs from a transaction.
* **Wallet → wallet** (`AddrAddr_edgelist.csv`) — 2,868,964 edges. Direct address-to-address
  flow, including multi-edges (repeated payments) and self-loops.

**Time**

All nodes carry a `Time step` in `1..49`. Time is part of the data model, not an
afterthought: graph construction and features for a given step must not depend on later
steps.

**Verified scale:** ~1.03M unique nodes and ~4.42M heterogeneous directed edges across
~2.1 GB of raw CSVs. Full numbers: `DATA_INVENTORY.md`.

## ML Architecture

Each model exists for a specific reason and is compared under the same evaluation protocol.

| Model | Role | Why it exists |
| :--- | :--- | :--- |
| **XGBoost** | Classical ML baseline | Measures what engineered tabular features achieve without graph learning. Cheap, strong, and interpretable via feature importance. |
| **GraphSAGE** | Homogeneous GNN baseline | Tests whether neighbourhood structure improves on XGBoost. Samples neighbour subgraphs, so it scales. |
| **RGCN** | Heterogeneous GNN | Respects distinct edge types (tx→tx, wallet→tx, tx→wallet, wallet→wallet) instead of flattening them into one relation. |
| **HGT** | Optional heterogeneous transformer | Evaluated only if RGCN results justify a more expressive (and more expensive) architecture. |

Baselines come first: no GNN improvement is claimed before the classical baseline exists.
Models are compared with identical data splits, metrics, and evaluation protocol.

## Training Architecture

### Local (laptop)

* Lightweight EDA and dataset inspection
* Feature engineering and preprocessing scripts
* Graph construction and schema design
* Code development, review, and configuration
* Testing and validation
* Notebook orchestration code kept in `src/` so it is reusable

The laptop is memory-constrained (~5.6 GB RAM), so local work must stream and chunk data.

### Kaggle / Google Colab

All model training and heavy experimentation happens remotely, because it needs GPU and
more memory than the laptop has:

* **XGBoost training** (yes, this is remote too)
* GraphSAGE training
* RGCN / HGT training
* Hyperparameter tuning
* Heavy experiments and ablations
* Explainability runs (GNNExplainer)
* Final training and evaluation

Notebooks are the **training interface**, not the application. Reusable logic — data
loaders, feature builders, graph construction, model definitions, metrics, and training
loops — lives in project source so the notebook only orchestrates it.

## Evaluation Architecture

**Temporal evaluation.** Split by `Time step`, never randomly. Train on earlier steps,
validate on a later step, and test on the latest unseen steps. This mirrors deployment and
prevents lookahead bias.

**Inductive evaluation.** Evaluate on transactions and wallets that were never seen during
training, to measure how the model generalizes to genuinely new actors.

**Metrics.** PR-AUC is primary given the severe class imbalance, supported by precision,
recall, F1, and confusion matrices. Accuracy alone is not acceptable.

**Degradation and ablations.** Measure how performance changes across time steps, and run
ablations to attribute gains to specific components (edge types, features, sampling,
imbalance handling).

## MLOps Architecture

* **DVC** — datasets and pipeline versioning, so experiments are reproducible.
* **MLflow** — experiment tracking: parameters, metrics, artifacts, and model metadata.
* **Model artifacts** — checkpoints, predictions, metrics, and configs exported from remote
  runs and stored reproducibly.
* **Monitoring** — two layers: model performance (quality over time) and graph-structural
  drift (degree distribution, communities, subgraph changes).
* **Retraining** — triggered by defined thresholds (e.g. performance drop, drift exceeded),
  never by arbitrary decisions.

## Serving Architecture

Trained artifacts flow back to the laptop, where the serving path is built:

```text
Exported model artifact → model loading / inference module → FastAPI endpoints
    → request validation → logging & error handling → Docker image → CI/CD → deployment
```

The model is validated before deployment. Until the deployment step is actually completed,
the system is **not** considered production-ready.

## Resource Constraints

The graph contains **>1M nodes and >4M heterogeneous edges**, which does not fit
comfortably in the laptop's ~5.6 GB of RAM. Therefore:

* Local CSV and graph processing must use streaming/chunked readers (`chunksize`,
  generators) and explicit memory cleanup (`del`, `gc.collect()`).
* GNN training must use **mini-batch / neighbour sampling**
  (`NeighborLoader` / `HeteroNeighborLoader`) rather than loading the full graph.
* Heavy training is offloaded to Kaggle/Colab; the laptop is not a training machine.

## Artifact Handoff

Training environments are ephemeral and must **export, not host**:

* Model checkpoints / weights
* Predictions on validation, test, and future time steps
* Evaluation metrics (PR-AUC, precision, recall, F1, confusion matrices)
* Run configuration and hyperparameters
* Explanation outputs (subgraphs, node/edge importances)

These return to the laptop for MLflow/DVC tracking, FastAPI serving, Docker packaging,
monitoring, and the dashboard.

## Data & Modeling Guardrails

* **Unknown labels are not legitimate.** Classes 1/2 are labeled; class 3 is unknown and
  must never be treated as licit.
* **Preserve temporal structure.** Chronological splits only for the primary evaluation.
* **Avoid temporal leakage.** No future time step may enter training features or graph
  construction.
* **Preserve heterogeneity.** Do not silently collapse the graph into a homogeneous graph;
  homogeneous models (e.g. GraphSAGE) are baselines, not replacements.
* **Preserve the raw dataset.** Never modify, overwrite, or move files in `../Og data/`.
