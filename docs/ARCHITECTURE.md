# BitcoinGraphGuard — Technical Architecture

This is the detailed architecture for BitcoinGraphGuard: what the system is made of, where
each part runs, and how data and models move through it. It does not change the project
objective. For a fast summary see `PROJECT_CONTEXT.md`; for the phased plan see
`TIMELINE.md`.

## System Overview

```text
Elliptic++
    ↓
Data Verification (laptop)
    ↓
[Kaggle / Colab — notebooks/]
  01 EDA → 02 XGBoost → 03 GraphSAGE → 04 Heterogeneous GNN
    → 05 Temporal & Inductive Evaluation
    → 06 Explainability → 07 Final Evaluation
    ↓
Model Artifacts + Metrics
    ↓
[Laptop — engineering]
  MLflow / DVC → FastAPI → Docker → CI/CD → Monitoring / Dashboard
```

Dataset verification stays on the laptop. **All ML execution — EDA, feature engineering,
graph construction, training, evaluation, and explainability — happens in notebooks on
Kaggle/Google Colab.** Everything from *Model Artifacts* onward happens back on the laptop:
tracking, serving, packaging, and monitoring.

## Notebook Architecture

Notebooks are the executable implementation of the ML work and are run in Google Colab or
Kaggle. There is **one notebook per major project phase** — not one per small task.

| Notebook | Phase | Contains |
| :--- | :--- | :--- |
| `notebooks/01_eda.ipynb` | Exploratory data analysis | Dataset loading, validation for EDA, temporal analysis, class-imbalance analysis, graph statistics, feature analysis, visualizations, conclusions |
| `notebooks/02_xgboost.ipynb` | Classical baseline | Leakage audit, temporal split construction, prevalence baseline, Logistic Regression, XGBoost training, PR-AUC/precision/recall/F1 evaluation, temporal windows, error and feature-importance analysis |
| `notebooks/03_graphsage.ipynb` | Homogeneous GNN baseline | Graph construction, GraphSAGE training with neighbour sampling, comparison to XGBoost |
| `notebooks/04_heterogeneous_gnn.ipynb` | Heterogeneous GNN | Heterogeneous graph build, RGCN (and HGT only if justified), tuning, ablations |
| `notebooks/05_temporal_inductive_evaluation.ipynb` | Robust evaluation | Temporal degradation, inductive evaluation on unseen nodes, per-step error analysis |
| `notebooks/06_explainability.ipynb` | Explainability | GNNExplainer on selected fraud, false-positive, and false-negative cases |
| `notebooks/07_final_evaluation.ipynb` | Finalization | Frozen final runs, final metrics, artifact export |

Adjust the count only for a genuine reason. A notebook may hold many related steps
internally; splitting a phase across several notebooks is not allowed.

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

## Experimentation Architecture

### Laptop — development only

* Writing and reviewing code, including reviewing notebook code
* Git repository management and project configuration
* Documentation and system design
* Backend development: FastAPI, inference modules, Docker, CI/CD
* MLOps engineering code, dashboard/frontend, and test suites
* Lightweight dataset inspection and verification scripts (not ML analysis)

The laptop never executes ML: no expensive data processing, feature engineering, model
training, GNN training, or experiments. It is memory-constrained (~5.6 GB RAM), so even
inspection work must stream and chunk data.

### Kaggle / Google Colab — all ML execution (in notebooks)

Every ML step runs remotely in the phase notebooks, because it needs GPU and more memory
than the laptop has:

* Dataset loading and data processing required for ML
* EDA and feature engineering
* Graph construction required for training
* XGBoost, GraphSAGE, RGCN, and HGT training
* Hyperparameter tuning
* Ablations, temporal and inductive evaluation, error analysis
* Explainability runs (GNNExplainer)
* Final evaluation and model-artifact generation

Notebooks are self-contained: the code the user copies into Colab/Kaggle must actually run
there. Application and serving logic (model loading, FastAPI, inference) lives in `src/`;
ML experimentation implementation lives in the notebook. Reusable helpers may be factored
out where genuinely useful, but the notebook must not depend on laptop-only paths.

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

Temporal and inductive evaluation live in
`notebooks/05_temporal_inductive_evaluation.ipynb`; ablations run alongside the model they
belong to (e.g. `notebooks/04_heterogeneous_gnn.ipynb`). Evaluation executes in Colab/Kaggle,
not on the laptop.

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

* CSV and graph processing in notebooks must use streaming/chunked readers (`chunksize`,
  generators) and explicit memory cleanup (`del`, `gc.collect()`).
* GNN training must use **mini-batch / neighbour sampling**
  (`NeighborLoader` / `HeteroNeighborLoader`) rather than loading the full graph.
* All ML execution is offloaded to Kaggle/Colab; the laptop is not a compute machine.

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
