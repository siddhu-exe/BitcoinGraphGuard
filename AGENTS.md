# Repository Guidelines

Contributor and agent guide for **BitcoinGraphGuard** — temporal heterogeneous graph-based Bitcoin
fraud detection on the real **Elliptic++** dataset. Read `docs/PROJECT_CONTEXT.md` for project
context and `docs/ARCHITECTURE.md` for the full design.

## Project Structure & Module Organization

| Path | Purpose |
| :--- | :--- |
| `Og data/` | Raw Elliptic++ CSVs (path contains a space — always quote it) — gitignored, **read-only**; never modified, moved or committed |
| `notebooks/` | One notebook per ML phase, run on Colab/Kaggle: `01_eda.ipynb` … `07_final_evaluation.ipynb` |
| `docs/` | Project, architecture, dataset, roadmap, status and phase documents (`EDA.md` records the Phase 1 run) |
| `eda/` | Canonical Phase 1 EDA run artifacts exported from `notebooks/01_eda.ipynb`: `eda_digest.txt`, `checks.csv`, `eda_summary.json`, table CSVs, figures. **Source of truth for EDA numbers.** |
| `reports/` | Verification evidence (`phase1_verification.json`) and `reports/eda/` from the superseded pre-notebook streaming pass |
| `scripts/` | Repository tooling: `verify_dataset.py` (Phase 1 verifier); `eda_phase2.py` + `plot_phase2_eda.py` are superseded by notebook 01 and kept only as reference |
| `src/`, `tests/` | Application and serving code plus tests — planned, not created yet |
| `CLAUDE.md` | Claude Code mirror of this file — keep the two in sync when either changes |

## Build, Test, and Development Commands

```bash
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
python scripts/verify_dataset.py     # streaming, memory-safe dataset verification (laptop-safe)
MPLCONFIGDIR=/tmp/mplconfig ./bit/bin/python scripts/plot_phase2_eda.py  # reference-only plot render
ruff check . && ruff format .        # lint and format (notebooks included)
pytest -q                            # test suite (once src/ and tests/ exist)
pytest --cov=src tests/              # coverage run
mlflow ui --port 5000                # experiment tracking UI (Phase 2/6)
dvc repro                            # versioned data/training pipeline (Phase 6)
uvicorn src.api.main:app --reload    # FastAPI inference service (Phase 7)
```

`.venv/` and `bit/` are gitignored local virtualenvs — never referenced from committed code, never
committed. `bit/` is an existing environment for the streaming scripts; `requirements.txt` in tiers is
the declared dependency contract.

## Compute & Notebook Architecture (critical)

* **Laptop = development and documentation only**: writing and reviewing code, Git, docs, system
  design, notebook review, FastAPI/Docker, MLOps engineering code, CI/CD, dashboard, configuration.
* **Kaggle / Google Colab = all ML execution, inside notebooks**: dataset loading, ML data
  processing, EDA, feature engineering, graph construction, XGBoost, GraphSAGE, RGCN/HGT, tuning,
  ablations, temporal and inductive evaluation, error analysis, GNNExplainer, final evaluation and
  model-artifact generation.
* **No ML on the laptop** — not training, and not "lightweight" EDA or feature engineering. Local
  inspection of raw data uses streaming, chunked, single-core, low-priority passes only.
* **One notebook per phase**, never one per small step. A notebook holds the whole phase
  internally: `01_eda.ipynb` covers loading, validation, temporal/class/graph/feature analysis,
  visualisations and conclusions.
* Notebooks run standalone on Colab/Kaggle, resolve paths from one config cell, set seeds, and
  stream large files in chunks. Application and serving logic belongs in `src/`.

## Coding Style & Naming Conventions

* Python 3.10–3.12, 4-space indentation; `ruff` is the linter and formatter.
* `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for
  constants, lowercase module names; type hints where practical, one-line docstrings on public
  callables.
* Notebooks: readable, logically ordered code; markdown cells explain what is being done, why, what
  the result means and what decision follows; no comments on obvious syntax; no hardcoded personal
  paths; no leftover exploratory cells.
* Committed code is production quality — no dead code, debug leftovers or hardcoded values.

## Testing Guidelines

* `pytest`, tests under `tests/`, files named `test_*.py`, functions named `test_<behaviour>`,
  mirroring the `src/` layout. Use small synthetic fixtures — never the raw Elliptic++ files.
* No test suite exists yet; tests arrive with `src/`. Until then, correctness is evidenced by the
  notebook's own checks (`checks.csv`) and by unit-checking notebook helper logic on tiny synthetic
  arrays outside the repository. Coverage thresholds are not enforced yet.

## Data & ML Guardrails

* **Unknown is not licit**: class 3 means "unlabeled" and never becomes a training example.
* No data leakage: no random split for the primary temporal evaluation, no future information in
  training features or graph construction, no duplicate rows leaking across a split boundary.
* Baselines before GNNs; PR-AUC, precision, recall, F1 and confusion matrices over accuracy, under
  one identical evaluation protocol when models are compared.
* Never fabricate metrics, results or dataset statistics; never substitute synthetic data for
  Elliptic++ in the primary evaluation; report results only after the experiment has run.
* Every quoted dataset number must be traceable to a saved artifact — `eda/` or
  `reports/phase1_verification.json` — not to memory or expectation. If a number is not in an
  artifact, re-derive it or omit it.
* Never commit raw data, model artifacts, credentials or large generated files.

## Current Phase State

Phase 1 is complete: dataset verification (`reports/phase1_verification.json`) and exploratory data
analysis (`notebooks/01_eda.ipynb`, artifacts in `eda/`, write-up in `docs/EDA.md`). Phase 2 is
**executed and awaiting review sign-off**: `notebooks/02_xgboost.ipynb` ran on Google Colab on
2026-09-25, artifacts in `xgboost/`, results in `docs/XGBOOST.md`. Protocol: fit 1–24 / validation
25–34 / refit 1–34 / test 35–49, on the 166 transaction features. Read `docs/PROGRESS.md` before
starting any phase work; it carries the open decisions and known issues that later phases depend on.

**Do not start `notebooks/03_graphsage.ipynb` until Phase 2 has been signed off.**

Carry-forward constraints from Phase 1 that must not be silently reversed:

* Deduplicate wallet snapshots to `(address, time step)` before any split.
* Decide explicitly how to handle the 965 blank-domain, address-less transactions; never treat
  blanks as `0`.
* Keep one of the two near-duplicate representations of the 17 transaction domain columns.
* The `43–49` window is a secondary drift window, not the primary test set.

Carry-forward constraints from Phase 2:

* The XGBoost numbers are a floor, not a ceiling: the 500-tree cap was binding (499 trees kept, best
  validation PR-AUC at the last tree), so the tree-count sweep must be re-run before the baseline is
  called final.
* Every GNN result must be reported on both `35–49` and `43–49`. An aggregate-only improvement would
  hide the regime where the baseline collapses.
* `has_addresses` can never mark a positive example — all 4,545 illicit transactions have an address
  link — so it is a one-sided licit indicator, not evidence of fraud.

## Working Protocol: ML Phases

When asked for the next ML phase: provide the notebook for that phase, assume it runs in Google
Colab or Kaggle, keep the implementation inside the single appropriate notebook, explain important
decisions alongside the code, and wait for the real results before designing the next phase.
Proceed **one notebook/phase at a time**.

## Task Completion Report

End every task with: what changed · files changed · tests/checks run · results · important
assumptions · remaining issues · next recommended step.
