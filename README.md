# BitcoinGraphGuard

An end-to-end Bitcoin fraud-detection system built on the real **Elliptic++** dataset.
Transactions and wallet/addresses are modeled as a **temporal heterogeneous graph**,
combining classical ML baselines, graph neural networks, explainability, drift
monitoring, and a containerized inference service.

> Status: dataset verified and EDA complete; no model results yet. See `docs/PROGRESS.md`,
> `docs/EDA.md`, and `docs/PLAN.md`.

## Architecture at a Glance

BitcoinGraphGuard runs across **two environments**. Full detail: `docs/ARCHITECTURE.md`.

- **Laptop — development & documentation only**: code writing/review, Git, documentation,
  system design, notebook code review, FastAPI, Docker, MLOps engineering, CI/CD,
  dashboard, configuration.
- **Kaggle / Google Colab — ALL ML execution, in notebooks**: dataset loading, ML data
  processing, EDA, feature engineering, graph construction, XGBoost, GraphSAGE, RGCN/HGT,
  tuning, ablations, temporal/inductive evaluation, error analysis, GNNExplainer, final
  evaluation.

**No ML runs on the laptop** — not training, and not "lightweight" EDA or feature
engineering. ML work lives in one notebook per phase (`notebooks/01_eda.ipynb` through
`07_final_evaluation.ipynb`) and runs standalone on Colab/Kaggle. Application/serving logic
lives in `src/`; remote runs export checkpoints, predictions, metrics, and explanations back
to the laptop for tracking, serving, and monitoring.

```text
Elliptic++ → [Laptop: Verification]
   → [Kaggle/Colab notebooks: 01 EDA → 02 XGBoost → 03 GraphSAGE → 04 Heterogeneous GNN
      → 05 Temporal & Inductive → 06 Explainability → 07 Final Evaluation]
   → Model Artifacts + Metrics → [Laptop: MLflow/DVC → FastAPI → Docker → CI/CD → Monitoring → Dashboard → Deployment]
```

## Dataset

Verified Elliptic++ inventory (`docs/DATA_INVENTORY.md`): 203,769 transaction nodes,
822,942 wallet/address nodes, 49 time steps, ~1.03M unique nodes and ~4.42M heterogeneous
directed edges, ~2.1 GB raw. Transaction labels: illicit 4,545 / licit 42,019 / unknown
157,205. Wallet labels: illicit 14,266 / licit 251,088 / unknown 557,588.

Unknown labels are **not** legitimate. Temporal order is preserved and never randomly
split for the primary evaluation. Raw data lives in `Og data/` and is gitignored.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `docs/OVERVIEW.md` | One-glance project overview |
| `docs/PROJECT_CONTEXT.md` | Start here — full project context for humans and AI agents |
| `docs/ARCHITECTURE.md` | Technical architecture: data, ML, training, evaluation, serving |
| `docs/TIMELINE.md` | Nine-phase roadmap with objectives and completion criteria |
| `docs/DATA_INVENTORY.md` | Verified dataset inventory and data-quality report |
| `docs/EDA.md` | Phase 2 exploratory data analysis findings |
| `notebooks/` | One notebook per ML phase; runs on Kaggle/Colab (not yet created) |
| `scripts/verify_dataset.py` | Reproducible, memory-safe dataset verification |
| `Og data/` | Raw Elliptic++ CSVs (gitignored, read-only) |
| `docs/PLAN.md`, `docs/PROGRESS.md`, `docs/OBJECTIVES.md` | Roadmap, status, objective |
| `AGENTS.md`, `CLAUDE.md` | Contributor/agent operating rules |
| `requirements.txt` | Tiered Python dependencies |
| `scripts/` | Memory-safe verification helper (`eda_phase2.py` superseded by notebook 01) |
| `src/`, `tests/` | Not created yet (planned) |

## Getting Started

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/verify_dataset.py    # memory-safe dataset verification
```

Heavy dependencies (`torch`, `torch-geometric`) install in the Colab/Kaggle runtime, not on
the laptop. All ML execution — EDA onward — happens in the phase notebooks.

## Documentation Index

- `docs/OVERVIEW.md` — one-glance project overview
- `docs/PROJECT_CONTEXT.md` — read-first project context
- `docs/ARCHITECTURE.md` — implementation architecture and compute split
- `docs/TIMELINE.md` — detailed nine-phase roadmap
- `docs/DATA_INVENTORY.md` — dataset facts, quality, graph relationships
- `docs/OBJECTIVES.md` — goal, requirements, definition of done
- `docs/PLAN.md` — phased roadmap with execution environments
- `docs/PROGRESS.md` — current status and next actions
