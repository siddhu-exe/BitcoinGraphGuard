# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BitcoinGraphGuard** is an end-to-end temporal heterogeneous graph-based Bitcoin fraud detection system built on the **Elliptic++** dataset. It detects illicit activity across 49 discrete time steps by combining tabular baselines (XGBoost), graph neural networks (GraphSAGE, HeteroRGCN, HGT), explainability (SHAP, GNNExplainer/Captum, relation ablation), drift monitoring, and a containerized FastAPI inference service.

## Core Architecture & Graph Topology

### Dataset & Graph Structure (`Og data/`)
The raw dataset comprises 49 contiguous time steps, ~1.03M unique nodes, and ~4.42M directed edges:
- **Transactions (`txId`)**: 203,769 nodes. Features include 165 non-domain features (93 `Local_feature_*`, 72 `Aggregate_feature_*`) plus 17 domain columns. Labels: 4,545 illicit (class 1), 42,019 licit (class 2), 157,205 unknown (class 3).
- **Wallets/Addresses (`address`)**: 822,942 nodes, 55 features per snapshot. Labels: 14,266 illicit, 251,088 licit, 557,588 unknown.
- **Heterogeneous Relations (4 directed edge types)**:
  1. `txs_edgelist.csv` (`tx_to_tx`): 234,355 directed edges. **100% intra-step** — zero cross-step transaction edges.
  2. `AddrTx_edgelist.csv` (`addr_to_tx`): 477,117 directed edges.
  3. `TxAddr_edgelist.csv` (`tx_to_addr`): 837,124 directed edges.
  4. `AddrAddr_edgelist.csv` (`addr_to_addr`): 2,784,344 directed edges.

*Note: Raw data in `Og data/` is gitignored and read-only. Never modify, move, or commit raw files.*

### Crucial Structural & Drift Discoveries
- **Cross-Temporal Message Passing**: Because transaction-to-transaction edges are 100% intra-step, homogeneous GNNs (GraphSAGE) cannot propagate temporal information across time steps. Cross-temporal message passing requires wallet/address nodes.
- **Abrupt Regime Shift at Step 43**: Illicit transaction prevalence abruptly drops from 9–11% (steps 1–42) to **2.53%** (steps 43–49), causing severe metric degradation across all models.
- **Covariate Drift Mechanism**: Adversarial validation yields AUC 1.0000; top drifting features are almost entirely `Aggregate_feature_*` (median KS 0.5336). In-window 5-fold CV on 43–49 achieves 0.9199 PR-AUC, confirming signal is present but distribution transfer fails.
- **Explainability Insight**: XGBoost relies 61.5% on local features (`Local_feature_53` dominant). RGCN relies heavily on recurrent address pathways (`addr_to_tx`, `tx_to_addr`), making it brittle when transactions shift to unseen addresses in the drift window.

## Compute Strategy: Laptop vs Colab/Kaggle

- **Laptop (5.6 GB RAM / Intel Core i3 4-thread)**: Development, documentation, git, code reviews, notebook generator authoring, FastAPI serving (`src/`), testing (`tests/`), Docker, MLOps configuration.
- **Kaggle / Google Colab (GPU / High-RAM)**: ALL ML and data processing — dataset loading, EDA, feature engineering, graph construction, training (XGBoost, GraphSAGE, RGCN, HGT), hyperparameter tuning, GNNExplainer, evaluation, artifact generation.
- **No heavy ML on laptop**: Never run full graph loads or model training locally. Local CSV inspections must use memory-safe streaming/chunking.

## Common Development Commands

### Environment Setup & Linting
```bash
# Create and activate local environment
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Lint and format code & notebooks
ruff check . && ruff format .

# Memory-safe dataset verification (laptop-safe streaming)
python scripts/verify_dataset.py
```

### Notebook Generation
Notebooks are programmatically authored via reviewable generator scripts:
```bash
python scripts/generate_notebook_02_v2.py    # Generates notebooks/02_xgboost_v2.ipynb
python scripts/generate_notebook_03_v2.py    # Generates notebooks/03_graphsage.ipynb
python scripts/generate_notebook_07.py       # Generates notebooks/07_hgt.ipynb
python scripts/generate_notebook_08.py       # Generates notebooks/08_ood_diagnosis.ipynb
python scripts/generate_notebook_08b.py      # Generates notebooks/08b_ood_permutation_check.ipynb
```

### Testing (Pytest)
```bash
# Run entire test suite (once src/ and tests/ are created)
pytest -q

# Run single test file or specific test function
pytest tests/test_api.py
pytest tests/test_api.py::test_predict_endpoint

# Run with test coverage
pytest --cov=src tests/
```

### Serving & MLOps (Phases 8–9)
```bash
# FastAPI local development
uvicorn src.api.main:app --reload --port 8000

# Docker build
docker build -t bitcoingraphguard:latest .

# MLflow UI & DVC tracking
mlflow ui --port 5000
dvc repro
```

## Phase & Benchmark Status

| Phase | Model / Analysis | Status | Test PR-AUC (35–49) | Drift PR-AUC (43–49) | Primary Documentation / Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | EDA & Verification | ✅ Complete | N/A | N/A | `docs/EDA.md`, `results/eda/` |
| **Phase 2** | XGBoost Baseline | ✅ Complete | **0.8013** | 0.0427 | `docs/XGBOOST.md`, `results/xgboost/` |
| **Phase 2b**| XGBoost Drift Study | ✅ Complete | 0.8070 | 0.0483 | `docs/XGBOOST_V2.md`, `results/xgboost_v2/` |
| **Phase 3** | GraphSAGE Baseline | ✅ Complete | 0.6209 | 0.0504 | `docs/GRAPHSAGE.md`, `results/GraphSage/` |
| **Phase 3b**| GraphSAGE v2 Cross-Step | ✅ Complete | 0.6216 | 0.0505 | `docs/GRAPHSAGE_V2.md`, `results/graphsage_v2/` |
| **Phase 4** | HeteroRGCN | ✅ Complete | 0.4682 | 0.0550 | `docs/HETEROGENEOUS_GNN.md`, `results/heterogeneous_gnn/` |
| **Phase 5** | Temporal & Inductive Eval | ✅ Complete | Analysis | Analysis | `docs/TEMPORAL_INDUCTIVE_EVALUATION.md`, `results/temporal_inductive/` |
| **Phase 6** | Explainability (SHAP/Captum)| ✅ Complete | Analysis | Analysis | `docs/EXPLAINABILITY.md`, `results/explainability/` |
| **Phase 7** | Hetero Graph Transformer | ✅ Complete | 0.4861 | 0.0386 | `docs/HGT.md`, `results/hgt/` |
| **Phase 7b**| OOD Diagnosis & Leak Audit | ✅ Complete | Adversarial AUC 1.0000 | Locals Only AUC 0.9885 | `docs/OOD_DIAGNOSIS.md`, `results/ood_diagnosis/` (inc. `permutation_check/`) |
| **Phase 8–9**| Serving & MLOps | 🔲 Planned | — | — | `docs/ARCHITECTURE.md`, `docs/PLAN.md` |

## Strict Carry-Forward Constraints & Evaluation Rules

1. **Temporal Splits Only**: Fit steps 1–24, Validation 25–34 (or late slice 33–34), Train/Refit 1–34, Test 35–49. Never use random splits.
2. **Sub-Window Reporting**: Always report metrics on both aggregate test (35–49) and the late drift window (43–49).
3. **Class 3 is Unknown**: Class 3 is unlabeled background data. Never treat class 3 as licit (class 2) and never use it as positive/negative training supervision.
4. **Primary Metric**: Due to extreme class imbalance (6.5% illicit in test, 2.53% in drift window), **PR-AUC** is the primary evaluation metric. Complement with Precision, Recall, F1 (illicit class), and ROC-AUC. Never report accuracy alone.
5. **No Data Leakage**: Feature scaling (`StandardScaler`) must be fit strictly on training steps ($\le 34$). Address snapshot aggregation and neighborhood sampling must strictly enforce zero future lookahead ($\le t$).
6. **Wallet Snapshot Deduplication**: Deduplicate wallet snapshots to distinct `(address, time step)` pairs before partitioning.
7. **965 Blank-Domain Transactions**: Exactly 965 transactions lack address linkages; handle missing domain values explicitly without naive zero-coercion.
8. **No Metric Fabrication**: Every quoted dataset statistic or metric must be traceable to saved artifact files in `results/` or `reports/`.

## Code Style & Development Conventions

- Python 3.10–3.12, 4-space indentation; `ruff` for linting and formatting.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Type hints on functions; docstrings on public callables.
- Notebooks: One notebook per phase; clear markdown cells explaining rationale, results, and architectural decisions. Avoid hardcoded personal paths.
- Keep `CLAUDE.md` and `AGENTS.md` synchronized when architectural or workflow guidelines change.
