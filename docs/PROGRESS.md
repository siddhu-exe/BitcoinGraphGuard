# Progress

## Current Status

**Phase 1 — dataset verification: COMPLETE.** On 2026-09-22 all nine
raw Elliptic++ CSVs and every cross-entity relationship were independently re-verified
with a streaming, low-priority verifier: 0 structural failures, peak RSS 1.07 GB,
457 s runtime. Evidence: `reports/phase1_verification.json`; full report:
`docs/DATA_INVENTORY.md`.

Two previous claims were corrected: `txs_features.csv` has 16,405 blank cells (not 0),
and `wallets_features.csv` contains 347,569 exact duplicate rows (so 1,268,260 raw rows =
920,691 distinct `(address, time step)` snapshots).

**Phase 1 — exploratory data analysis (`notebooks/01_eda.ipynb`): authored, execution pending.** A local
streaming EDA ran on 2026-09-22 (`scripts/eda_phase2.py`: full pipeline **297 s**, peak RSS
**879 MB**, no model trained, no split finalized); evidence: `reports/eda/phase2_eda.json`.
Under the current architecture that pipeline is **superseded**: `notebooks/01_eda.ipynb` is
now the canonical, executable EDA entrypoint, and the script's logic is the reference
implementation carried into it. The notebook is authored and validated (lint-clean, helper
logic unit-checked on tiny synthetic arrays) but has **not been executed**, so the EDA phase
is deliberately **not** marked complete and `docs/EDA.md` stays pending until the notebook
has been run and reviewed on Colab/Kaggle.

Headline findings from the superseded local run (recorded reference evidence, to be
reconfirmed by the notebook): activity spans 49 steps with ~7× burstiness; illicit share of
labeled transactions ranges **0.28%–35.97%** by step; the address graph is a single
component of 822,935 nodes while the transaction graph fragments into **49 components**;
the combined address↔transaction graph has **965 singleton components = the 965
address-less transactions**; wallet duplicate rows re-confirmed exactly (920,691 distinct
snapshots); and the natural test window (steps 43–49) has far lower transaction illicit
prevalence (2.53%) than training (11.58%), so the split and metric protocol are
deliberately left open.

The project direction is fixed around **BitcoinGraphGuard**, using the real Elliptic++
dataset for temporal heterogeneous graph-based Bitcoin fraud detection.

## Compute Strategy

Work is split across two environments (see `ARCHITECTURE.md`):

* **Laptop** — development and documentation only: code writing/review, Git, docs, system
  design, notebook review, FastAPI, Docker, MLOps engineering code, CI/CD, dashboard, and
  project configuration.
* **Kaggle / Google Colab** — ALL ML execution, inside notebooks: dataset loading, ML data
  processing, EDA, feature engineering, graph construction, XGBoost, GraphSAGE, RGCN/HGT,
  tuning, ablations, temporal/inductive evaluation, error analysis, GNNExplainer, and final
  evaluation.

No ML runs on the laptop — not training, and not "lightweight" EDA or feature engineering.
The ML work is one notebook per phase (`notebooks/01_eda.ipynb` through
`notebooks/07_final_evaluation.ipynb`), and remote environments export checkpoints,
predictions, metrics, and explanation outputs back for local tracking and serving.

## Verified Dataset Facts (2026-09-22)

* **Nodes:** 203,769 transactions (0 duplicate `txId`) + 822,942 wallets (0 duplicate
  addresses) = 1,026,711 unique nodes.
* **Time steps:** 49 contiguous steps (1–49) for both node types.
* **Edges (unique):** `txs` 234,355; `AddrTx` 477,117; `TxAddr` 837,124; `AddrAddr`
  2,784,344. Total 4,417,560 directed edges (4,332,940 unique).
* **Labels:** transactions illicit 4,545 / licit 42,019 / unknown 157,205; wallets
  illicit 14,266 / licit 251,088 / unknown 557,588. **Unknown labels are not
  legitimate.**
* **Cross-entity integrity:** every transaction/ID and wallet/address reconciles across
  features, classes and edge lists; 0 edge endpoints outside the node universes; 202,804
  transactions have address links and 965 do not.
* **Data-quality findings:** `txs_features` has 16,405 blank cells (17 domain columns ×
  the 965 address-less transactions); `wallets_features`/combined have 347,569 exact
  duplicate rows (920,691 distinct snapshots); `AddrAddr` has 84,620 repeated pairs and
  45,981 self-loops. Temporal order is preserved and never randomly split.

## Completed

* [x] Project scope defined
* [x] Elliptic++ selected as the primary dataset
* [x] Project architecture defined
* [x] Nine-phase development strategy defined (see `PLAN.md`)
* [x] ML baseline, graph deep learning, temporal/inductive evaluation, explainability,
      graph-drift monitoring, and MLOps strategies defined
* [x] Download Elliptic++ raw dataset
* [x] Rewrite `scripts/verify_dataset.py` as a streaming, low-memory, CLI-driven verifier
* [x] **Full independent re-verification of all 9 raw CSVs and all cross-references**
      (`passed: true`, 0 failures)
* [x] Correct the inventory: txs blank cells and wallet duplicate rows documented
* [x] Finalize `docs/DATA_INVENTORY.md` with verified facts, assumptions and unresolved
      items
* [x] Define implementation architecture and laptop vs Kaggle/Colab compute split
      (`ARCHITECTURE.md`)
* [x] Create `requirements.txt` (user-managed install pending)
* [x] Build streaming, resumable EDA tooling (`scripts/eda_phase2.py`,
      `scripts/plot_phase2_eda.py`)
* [x] Author `notebooks/01_eda.ipynb` — the canonical EDA entrypoint (single notebook for
      the phase; streams both large feature files, re-derives the Phase 1 inventory, and
      emits tables, figures, a digest and a machine-readable summary)
* [x] **Temporal structure analysis** across all 49 steps for transactions and wallet
      snapshots
* [x] **Class-imbalance analysis** (overall and per time step; unknown never treated as
      licit)
* [x] **Graph-level EDA** (degrees, components, isolated nodes, multi-edges, self-loops,
      hubs for all four edge types)
* [x] **Feature analysis** (missingness, constants, skew, class correlation, redundancy,
      early/late drift)
* [x] **Temporal-split investigation** over candidate windows (split not finalized)
* [x] Record the first-pass EDA findings as evidence (`reports/eda/`); `docs/EDA.md` now waits
      for the notebook run before it is treated as final

## Current Task

* [ ] Install project Python dependencies from `requirements.txt` (user-managed)
* [ ] **Execute `notebooks/01_eda.ipynb` on Colab/Kaggle and record the results in
      `docs/EDA.md`** — the EDA phase is not complete until this run has been reviewed
* [ ] Configure DVC and project directory structure (`src/`)
* [ ] Initialize MLflow tracking
* [ ] Prepare the Kaggle/Colab notebook environment (one notebook per phase)
* [ ] Decide the transaction feature set (drop the 17 domain columns that duplicate
      `Local_feature_*`; mask/flag the 965 address-less transactions)

## Next

* Confirm whether `txs_edgelist` components map one-to-one onto time steps — answered by
  §8 of `notebooks/01_eda.ipynb` (`txs_component_time_span.csv`) once it is executed
* Build the memory-aware project data pipeline; deduplicate wallet snapshots to
  `(address, time step)` before any split
* Design the heterogeneous graph schema (mask/flag the 965 address-less transactions)
* Prepare the XGBoost classical baseline before claiming GNN improvements
* Finalize the temporal split and metric protocol once the baseline is defined (Phase 3);
  account for the 11.58% → 2.53% transaction illicit-prevalence shift

## Known Issues

* Wallet feature files must be deduplicated by `(address, time step)` before modeling;
  otherwise snapshot counts are inflated and identical rows can leak across splits.
* The 17 blank transaction domain columns need an explicit masking/imputation decision in
  Phase 3 (do not treat blanks as 0).
* The 17 transaction domain columns are near-perfect copies of `Local_feature_*`
  (r ≈ 1.0); keep only one representation.
* Temporal class prevalence is non-stationary; the candidate test window 43–49 has only
  169 illicit transactions (2.53% of labeled). Split/metric protocol deliberately open.
* Whether `txs_edgelist`'s 49 components correspond exactly to the 49 time steps is
  strongly suggested but not yet confirmed; the notebook reports it as a component×time-step
  purity table rather than leaving it implicit.
* Final heterogeneous graph schema is pending Phase 3 design.
* Final deployment infrastructure will be decided after the inference pipeline is
  implemented.
* Remote training environments (Kaggle/Colab) are not yet set up or documented as
  runnable.

## Important Rules

* Do not introduce synthetic data to replace Elliptic++.
* Do not randomly split temporal data for the primary evaluation.
* Do not introduce data leakage (including duplicate-row leakage across splits).
* Do not report experimental metrics before running the experiment.
* Do not replace the heterogeneous graph objective with a simpler unrelated approach.
* Do not execute ML on the laptop at all; all ML work runs in notebooks on Kaggle/Colab.
  Even local inspection must use streaming, chunked, low-priority, one-core passes.
