# Progress

## Current Status

**Phase 1 (data & foundation) — dataset verification COMPLETE.** On 2026-09-22 all nine
raw Elliptic++ CSVs and every cross-entity relationship were independently re-verified
with a streaming, low-priority verifier: 0 structural failures, peak RSS 1.07 GB,
457 s runtime. Evidence: `reports/phase1_verification.json`; full report:
`docs/DATA_INVENTORY.md`.

Two previous claims were corrected: `txs_features.csv` has 16,405 blank cells (not 0),
and `wallets_features.csv` contains 347,569 exact duplicate rows (so 1,268,260 raw rows =
920,691 distinct `(address, time step)` snapshots).

The project direction is fixed around **BitcoinGraphGuard**, using the real Elliptic++
dataset for temporal heterogeneous graph-based Bitcoin fraud detection.

## Compute Strategy

Work is split across two environments (see `ARCHITECTURE.md`):

* **Laptop** — engineering: code, data inspection, lightweight EDA/preprocessing, graph
  schema design, experiment configuration, testing, FastAPI, MLflow/DVC, Docker, CI/CD,
  monitoring, dashboard, documentation.
* **Kaggle / Google Colab** — all model training and heavy experiments (XGBoost,
  GraphSAGE, RGCN/HGT, tuning, ablations, temporal/inductive evaluation, GNNExplainer).

No project model is trained on the laptop. Remote environments export checkpoints,
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

## Current Task

* [ ] Install project Python dependencies from `requirements.txt` (user-managed)
* [ ] Configure DVC and project directory structure (`src/`)
* [ ] Initialize MLflow tracking
* [ ] Prepare Kaggle/Colab training environment and notebook entrypoints

## Next

* Phase 2: temporal and class-imbalance EDA on the verified inventory
* Build the memory-aware project data pipeline; deduplicate wallet snapshots to
  `(address, time step)` before any split
* Design the heterogeneous graph schema (mask/flag the 965 address-less transactions)
* Prepare the XGBoost classical baseline before claiming GNN improvements

## Known Issues

* Wallet feature files must be deduplicated by `(address, time step)` before modeling;
  otherwise snapshot counts are inflated and identical rows can leak across splits.
* The 17 blank transaction domain columns need an explicit masking/imputation decision in
  Phase 2 (do not treat blanks as 0).
* Final heterogeneous graph schema is pending Phase 2 design.
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
* Do not run heavy, unconstrained jobs on the laptop; use streaming, chunked,
  low-priority, one-core passes or a remote machine. Project models are never trained on
  the laptop.
