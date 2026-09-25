# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BitcoinGraphGuard** is an end-to-end Bitcoin fraud detection system built on the **Elliptic++** dataset. It models Bitcoin transactions and actor/wallet relationships as a temporal heterogeneous graph to identify illicit activity, backed by classical ML baselines, GNNs (GraphSAGE, RGCN/HGT), explainability (GNNExplainer), drift monitoring, and a production FastAPI inference service.

## Core Dataset Architecture (`Og data/`)

The raw dataset in `Og data/` comprises temporal graph and tabular data across 49 discrete time steps:
- **Transactions (`txId`)**: `txs_features.csv` is 184 columns = `txId` + `Time step` + 182 features (93 `Local_feature_*`, 72 `Aggregate_feature_*`, 17 domain features). `txs_classes.csv` (1=illicit, 2=licit, 3=unknown), `txs_edgelist.csv` (`txId1 -> txId2`).
- **Wallets/Addresses (`address`)**: `wallets_features.csv`, `wallets_classes.csv`, `wallets_features_classes_combined.csv`.
- **Heterogeneous Edges**:
  - `AddrTx_edgelist.csv` (`input_address -> txId`)
  - `TxAddr_edgelist.csv` (`txId -> output_address`)
  - `AddrAddr_edgelist.csv` (`input_address -> output_address`)
  - `txs_edgelist.csv` (`txId1 -> txId2`)

*Note: Raw data in `Og data/` is gitignored; do not commit large CSVs or raw data files.*

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `Og data/` | Raw Elliptic++ CSVs — gitignored, **read-only**; never modified, moved or committed |
| `notebooks/` | One notebook per ML phase, run on Colab/Kaggle: `01_eda.ipynb` … `07_final_evaluation.ipynb` |
| `docs/` | Project, architecture, dataset, roadmap, status and phase documents (including `EDA.md`) |
| `scripts/` | Repository tooling: `verify_dataset.py`; `eda_phase2.py` is superseded by notebook 01 |
| `reports/` | Verification and EDA evidence (JSON / CSV / PNG) |
| `src/`, `tests/` | Application/serving code and tests — planned, not created yet |

## Development & Environment Commands

Python 3.10+ / `uv` is recommended for dependency and environment management.

### Environment & Dependencies
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
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

## Compute Strategy: Laptop vs Kaggle/Colab

Work is split across two environments; full details in `docs/ARCHITECTURE.md`.

- **Laptop (development & documentation only)**: writing/reviewing code, Git management, documentation, system design, notebook code review, FastAPI backend, Docker, MLOps engineering code, CI/CD, dashboard/frontend, and project configuration.
- **Kaggle / Google Colab (ALL ML execution, in notebooks)**: dataset loading, ML data processing, EDA, feature engineering, graph construction, XGBoost, GraphSAGE, RGCN, HGT, hyperparameter tuning, ablations, temporal/inductive evaluation, error analysis, GNNExplainer, final evaluation, and model-artifact generation.
- **Do not execute ML on the laptop** — no data processing, feature engineering, training, or experiments locally, including "lightweight" EDA.
- **One notebook per phase** under `notebooks/` (`01_eda.ipynb` … `07_final_evaluation.ipynb`). Notebooks are the executable ML implementation and must run standalone on Colab/Kaggle; application/serving logic lives in `src/`.
- Remote runs export checkpoints, predictions, metrics, and explanation outputs back to the laptop for tracking, serving, and monitoring.

When asked for the next ML phase: provide the phase notebook, assume it runs on Colab/Kaggle,
keep it inside the single appropriate notebook, write human-like code, explain important
decisions, and wait for real results before designing the next phase.

## Hardware & Execution Constraints (Low-Resource Environment)

**CRITICAL**: This laptop has limited compute resources (**5.6 GB RAM**, ~3.0 GB available, Intel Core i3 4-thread CPU) and cannot execute heavy, unconstrained tasks.
- **No Heavy / Full-Graph In-Memory Loading**: Never load the full raw 2.1 GB dataset or the complete 1M-node / 4.4M-edge graph in memory at once.
- **Streaming & Chunked Processing**: Always use streaming iterators, generator pipelines, or chunked processing (`chunksize`, sqlite/duckdb streaming).
- **Mini-Batch Graph Learning**: Use sub-graph sampling and mini-batch loaders (`NeighborLoader` / `HeteroNeighborLoader`) for GNN models.
- **Resource Discipline**: Limit parallel worker threads (max 2 workers), explicitly release large variables, and call `gc.collect()` to prevent system freezing and OOM kills.

## Coding Style & Naming Conventions

* Python 3.10–3.12, 4-space indentation; `ruff` is the linter and formatter.
* `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, lowercase module names; type hints where practical, one-line docstrings on public callables.
* Notebooks: readable, logically ordered code; markdown cells explain what is being done, why, what the result means and what decision follows; no comments on obvious syntax; no hardcoded personal paths; no leftover exploratory cells.
* **Production-quality code**: all committed code must be production quality — clear, readable, typed where practical, error-handled, tested, and free of dead code, debug leftovers, and hardcoded values. Notebook/prototype code does not belong in project source.

## Testing Guidelines

* `pytest`, tests under `tests/`, files named `test_*.py`, functions named `test_<behaviour>`, mirroring the `src/` layout. Use small synthetic fixtures — never the raw Elliptic++ files.
* No test suite exists yet; tests arrive with `src/`. Until then, correctness is evidenced by the notebook's own checks (`checks.csv`) and by unit-checking notebook helper logic on tiny synthetic arrays outside the repository. Coverage thresholds are not enforced yet.

## Commit & Pull Request Guidelines

* Commit history uses short, scoped, imperative subjects — for example `dataset reverification`, `docs: add dataset verification details`, `fix(frontend): gate the UI on a backend readiness probe`. Prefer `type(scope): summary` (`feat`, `fix`, `docs`, `chore`, `test`) and add a body covering what and why when the change is not self-evident.
* Pull requests should describe the change and its motivation, point at the relevant phase document (`docs/PLAN.md`, `docs/PROGRESS.md`, `docs/EDA.md`), list the commands run and their results, note any documentation updated, and include figures or metrics for EDA/ML changes.
* Never include raw data, model artifacts, credentials or other large generated files in a commit or PR. Keep each PR focused on a single phase or concern.

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
- **Next recommended step**: Immediate next action in `docs/PLAN.md`
