# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BitcoinGraphGuard** is an end-to-end Bitcoin fraud detection system built on the **Elliptic++** dataset. It models Bitcoin transactions and actor/wallet relationships as a temporal heterogeneous graph to identify illicit activity, backed by classical ML baselines, GNNs (GraphSAGE, RGCN/HGT), explainability (GNNExplainer), drift monitoring, and a production FastAPI inference service.

## Core Dataset Architecture (`Og data/`)

The raw dataset in `Og data/` comprises temporal graph and tabular data across 49 discrete time steps:
- **Transactions (`txId`)**: `txs_features.csv` (165 features + time step), `txs_classes.csv` (1=illicit, 2=licit, 3=unknown), `txs_edgelist.csv` (`txId1 -> txId2`).
- **Wallets/Addresses (`address`)**: `wallets_features.csv`, `wallets_classes.csv`, `wallets_features_classes_combined.csv`.
- **Heterogeneous Edges**:
  - `AddrTx_edgelist.csv` (`input_address -> txId`)
  - `TxAddr_edgelist.csv` (`txId -> output_address`)
  - `AddrAddr_edgelist.csv` (`input_address -> output_address`)
  - `txs_edgelist.csv` (`txId1 -> txId2`)

*Note: Raw data in `Og data/` is gitignored; do not commit large CSVs or raw data files.*

## Development & Environment Commands

Python 3.10+ / `uv` is recommended for dependency and environment management.

### Environment & Dependencies
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"      # or uv pip install -r requirements.txt
```

### Testing
```bash
pytest                          # Run entire test suite
pytest tests/unit/              # Run unit tests only
pytest tests/test_graph.py -k "test_temporal_split"  # Run single test
pytest --cov=src tests/         # Run tests with coverage
```

### Code Quality & Formatting
```bash
ruff check .                    # Lint check
ruff check --fix .              # Lint autofix
ruff format .                   # Code formatting
```

### MLOps & Services
```bash
mlflow ui --port 5000           # Launch MLflow tracking UI
dvc repro                       # Run DVC data/training pipeline
dvc status                      # Check DVC pipeline status
uvicorn src.api.main:app --reload --port 8000  # Run FastAPI inference service
docker build -t bitcoingraphguard:latest .     # Build container
```

## Hardware & Execution Constraints (Low-Resource Environment)

**CRITICAL**: This laptop has limited compute resources (**5.6 GB RAM**, ~3.0 GB available, Intel Core i3 4-thread CPU) and cannot execute heavy, unconstrained tasks.
- **No Heavy / Full-Graph In-Memory Loading**: Never load the full raw 2.1 GB dataset or the complete 1M-node / 4.4M-edge graph in memory at once.
- **Streaming & Chunked Processing**: Always use streaming iterators, generator pipelines, or chunked processing (`chunksize`, sqlite/duckdb streaming).
- **Mini-Batch Graph Learning**: Use sub-graph sampling and mini-batch loaders (`NeighborLoader` / `HeteroNeighborLoader`) for GNN models.
- **Resource Discipline**: Limit parallel worker threads (max 2 workers), explicitly release large variables, and call `gc.collect()` to prevent system freezing and OOM kills.

## Critical Modeling & Evaluation Rules

1. **Temporal Splits Only**: Never use random train/test splits. Always split chronologically by `Time step` (e.g., train on early time steps, evaluate on later unseen time steps) to prevent temporal data leakage and evaluate real-world inductive generalization.
2. **Heterogeneous Graph Structure**: Preserve both node types (`transaction`, `address/wallet`) and their directional relationships (`AddrTx`, `TxAddr`, `txs_edgelist`). Do not collapse the problem into a simple homogeneous graph unless specifically running a comparative baseline.
3. **Class Imbalance & Metrics**: Class 1 (illicit) is heavily underrepresented. Prioritize **PR-AUC**, **Precision**, **Recall**, **F1 (illicit class)**, and confusion matrices. Never evaluate using accuracy alone.
4. **Baselines First**: Establish strong tabular (XGBoost) and homogeneous graph (GraphSAGE) baselines before benchmarking heterogeneous GNN architectures (RGCN, HGT).
5. **No Synthetic Shortcuts**: Do not replace Elliptic++ data with synthetic or toy graph structures for primary evaluation.

## Task Completion Report

At the end of every non-trivial task or phase, report:
- **What changed**: Summary of changes made
- **Files changed**: List of created/modified files
- **Tests/checks run**: Commands executed and their status
- **Results**: Quantitative metrics or validation output
- **Important assumptions**: Architectural or domain assumptions made
- **Next recommended step**: Immediate next action in `PLAN.md`
