# Elliptic++ Dataset Inventory & Verification Report

**Dataset Version:** Elliptic++ (Transactions + Wallets/Addresses Heterogeneous Temporal Graph)
**Initial Verification:** 2026-09-21
**Full Streamed Re-verification:** 2026-09-22
**Verification Mode:** pure streaming; no raw file is loaded into RAM (single core, low priority)
**Implementation Architecture:** `ARCHITECTURE.md` (laptop vs Kaggle/Colab compute split)

> **Status: Phase 1 dataset verification is COMPLETE.** All nine raw CSVs and all
> cross-entity relationships were independently re-verified on 2026-09-22 by
> `scripts/verify_dataset.py`. The run passed every structural assertion
> (`passed: true`, 0 failures), peak RSS **1.07 GB**, elapsed **457 s**. Machine-readable
> evidence: `reports/phase1_verification.json`.
>
> **Two prior claims were corrected** (details in §6):
> 1. `txs_features.csv` is **not** free of missing values — it has 16,405 blank cells.
> 2. `wallets_features.csv` is **not** one row per (address, time step) — it has 347,569
>    **exact duplicate rows**, so 1,268,260 raw rows collapse to 920,691 distinct
>    (address, time step) snapshots.

---

## 1. Dataset Location & Directory Structure

```
/home/siddharth/Desktop/Projects/projects/BitcoinGraphGuard/Og data/
```
Total raw size: **2,206,089,537 bytes (~2.21 GB / ~2.05 GiB)** across **9 CSV files**.
Raw data is read-only and gitignored; it was not modified.

---

## 2. File Inventory & Specifications

All rows/columns below were re-verified on 2026-09-22 (streaming row count + header parse).

| File Name | Category | Size | Rows (raw) | Columns | Key Identifiers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `txs_features.csv` | Transactions | 662.60 MB | 203,769 | 184 | `txId`, `Time step`, 182 features |
| `txs_classes.csv` | Transactions | 2.25 MB | 203,769 | 2 | `txId`, `class` |
| `txs_edgelist.csv` | Tx Graph | 4.26 MB | 234,355 | 2 | `txId1` -> `txId2` |
| `wallets_features.csv` | Wallets | 578.37 MB | 1,268,260 | 57 | `address`, `Time step`, 55 features |
| `wallets_classes.csv` | Wallets | 29.01 MB | 822,942 | 2 | `address`, `class` |
| `wallets_features_classes_combined.csv` | Wallets | 580.79 MB | 1,268,260 | 58 | `address`, `Time step`, `class`, 55 features |
| `AddrTx_edgelist.csv` | Bipartite Edge | 20.26 MB | 477,117 | 2 | `input_address` -> `txId` |
| `TxAddr_edgelist.csv` | Bipartite Edge | 35.00 MB | 837,124 | 2 | `txId` -> `output_address` |
| `AddrAddr_edgelist.csv` | Wallet Graph | 191.34 MB | 2,868,964 | 2 | `input_address` -> `output_address` |

---

## 3. Node & Label Statistics

### 3.1 Transactions (`txId`)
* **Rows / unique IDs:** 203,769 / 203,769 — 0 duplicates, 0 blank IDs.
* **Time steps:** 49 discrete, contiguous time steps (min 1, max 49).
* **Features (182):** 93 `Local_feature_*` + 72 `Aggregate_feature_*` + 17 domain features
  (`in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input_addresses`,
  `num_output_addresses`, 5 `in_BTC_*`, 5 `out_BTC_*`).
* **Missing values:** 16,405 blank cells — every one of those 17 domain columns is blank
  for exactly the **965 transactions that have no input/output address links** (verified
  as an exact set match against the complement of the `AddrTx` transaction set). All 93
  `Local_feature_*` and 72 `Aggregate_feature_*` columns are complete. 0 non-numeric, 0 NaN.
* **Labels (`txs_classes.csv`):** 1 (illicit) 4,545 · 2 (licit) 42,019 · 3 (unknown)
  157,205. Labeled total 46,564 (22.85%).

### 3.2 Wallets / Addresses (`address`)
* **Unique addresses:** 822,942 (in `wallets_classes.csv`, `wallets_features.csv`, and the
  combined file) — 0 duplicates, 0 missing, all three sets identical.
* **Raw feature rows:** 1,268,260 — but only **920,691 distinct (address, time step)
  snapshots**; **347,569 rows are exact duplicates** (see §6).
* **Time steps:** 49 discrete, contiguous (min 1, max 49).
* **Features (55):** activity metrics (BTC transacted/sent/received, fees, block spans,
  address reuse, temporal gaps); 0 blank cells, 0 non-numeric, 0 NaN.
* **`num_timesteps_appeared_in` consistency:** equals the number of distinct time steps for
  **all 822,942 addresses** (0 mismatches), and is internally consistent within each
  address (0 conflicts).
* **Labels (`wallets_classes.csv`):** 1 (illicit) 14,266 · 2 (licit) 251,088 · 3 (unknown)
  557,588. Labeled total 265,354 (32.24%).
* **Combined file:** exactly `wallets_features` + the `class` column — same row count,
  same 347,569 duplicates, 0 addresses missing from `wallets_classes`, 0 class mismatches,
  0 `(address, time step)` pairs absent from `wallets_features`.

---

## 4. Edge & Graph Topology

Uniqueness is defined on the exact parsed `(source, destination)` pair.

| Edge List | Edges | Unique | Duplicates | Self-loops | Unique Src | Unique Dst | Unique Nodes | Endpoints outside universe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `txs_edgelist.csv` | 234,355 | 234,355 | 0 | 0 | 166,345 | 148,447 | 203,769 | 0 |
| `AddrTx_edgelist.csv` | 477,117 | 477,117 | 0 | 0 | 400,212 | 202,804 | 603,016 | 0 |
| `TxAddr_edgelist.csv` | 837,124 | 837,124 | 0 | 0 | 202,804 | 641,043 | 843,847 | 0 |
| `AddrAddr_edgelist.csv` | 2,868,964 | 2,784,344 | 84,620 | 45,981 | 400,212 | 641,043 | 822,942 | 0 |

* **Total heterogeneous directed edges:** 4,417,560 (4,332,940 unique).
* **Total unique nodes:** 203,769 transactions + 822,942 wallets = **1,026,711**.
* 0 blank endpoints and 0 malformed rows in every edge list.
* `AddrAddr_edgelist.csv` alone touches all 822,942 wallets; its 84,620 duplicates are
  repeated payments between the same address pair and its 45,981 self-loops are
  address-to-itself (change/self-transfer) edges.

---

## 5. Cross-Entity Alignment (all verified 2026-09-22)

| Relationship | Result |
| :--- | :--- |
| `txs_features` IDs vs `txs_classes` IDs | 203,769 ↔ 203,769, 0 missing either way |
| `txs_edgelist` nodes vs `txs_features` IDs | 203,769 ↔ 203,769, 0 missing |
| Transactions with address links | 202,804 (`AddrTx` destinations == `TxAddr` sources, 0 diff) |
| Transactions without address links | 965 — exactly the 965 rows with blank domain features |
| `wallets_features` addresses vs `wallets_classes` | 822,942 ↔ 822,942, 0 missing either way |
| Addresses in `AddrTx` ∪ `TxAddr` ∪ `AddrAddr` | 822,942 — all in `wallets_features`, 0 outside |
| Wallets with no incident edge | 0 |

---

## 6. Data-Quality Findings & Corrections

1. **CORRECTION — `txs_features.csv` missing values.** The prior "0 missing values" claim
   was wrong. There are **16,405 blank cells**: the 17 value/domain columns are blank for
   exactly the 965 transactions with no address links. (Structurally clean otherwise:
   0 malformed rows, 0 non-numeric, 0 NaN.)

2. **CORRECTION — `wallets_features.csv` duplicate rows.** The prior description ("one row
   per active time step") was wrong. There are 1,268,260 raw rows but **920,691 distinct
   (address, time step) snapshots**; **347,569 rows are exact duplicates**. Verified by a
   one-pass BLAKE2b row-hash check: all 347,569 extra rows are **byte-identical**, with
   **0 conflicting values**. Duplication affects 234,355 addresses; the most duplicated
   address has 1,471 identical rows spanning 27 distinct time steps. The same duplication
   exists in `wallets_features_classes_combined.csv`.

3. **Repeated pairs / self-loops in `AddrAddr_edgelist.csv`:** 84,620 repeated
   `(input_address, output_address)` pairs (multi-edges across transactions) and 45,981
   self-loops. These are expected multi-edge/self-transfer structure, not corruption.

4. **No missing values** in either wallet feature file: 0 blank cells, 0 NaN.

---

## 7. Modeling & Architectural Implications

1. **Deduplicate wallets before graph construction.** Collapse `wallets_features` (and the
   combined file) to unique `(address, Time step)` rows first; duplicate rows would
   otherwise inflate node-snapshot counts and can leak identical rows across a temporal
   split. `num_timesteps_appeared_in` already matches the distinct-step counts, so the
   deduplicated table is the intended representation.

2. **Handle the 965 address-less transactions explicitly.** Their 17 domain features are
   blank and they have no address links. Do not treat blanks as 0; decide on masking or a
   dedicated "no-address" indicator when the feature set is fixed
   (`notebooks/02_xgboost.ipynb`).

3. **Heterogeneous graph (PyG / DGL):** `transaction` = 203,769 nodes (182 features);
   `wallet` = 822,942 nodes (55 features); edge types
   `('wallet','sends_to','transaction')` 477,117 · `('transaction','receives_to','wallet')`
   837,124 · `('transaction','flows_to','transaction')` 234,355 ·
   `('wallet','transacts_with','wallet')` 2,784,344 unique.

4. **Severe class imbalance:** illicit transactions ≈2.23% of all (9.76% of labeled);
   illicit wallets ≈1.73% of all (5.38% of labeled). Use PR-AUC, F1 (illicit), recall@
   precision, confusion-matrix analysis, and cost-sensitive weighting.

5. **Temporal split:** 49 chronological steps; train on historical steps and evaluate on
   future unseen steps. Never randomly split temporal graph data.

6. **Resource constraints:** ~1.03M nodes / ~4.42M edges. Use mini-batch sampling
   (`NeighborLoader` / `HeteroNeighborLoader`) and streaming pipelines. GNN training runs
   on Kaggle/Colab, never the laptop.

---

## 8. Verification Status & Reproduction

`scripts/verify_dataset.py` is a streaming, CLI-driven verifier (standard library only):
per-column blank/non-numeric/NaN audit, exact parsed-pair deduplication, cross-reference
stage, `--low-priority` (`os.nice(19)`), opt-in `--hash`. It releases each stage's working
set with `del` + `gc.collect()`.

Final 2026-09-22 run (single core, low priority):

| Metric | Value |
| :--- | :--- |
| Elapsed | 457 s |
| Peak RSS | 1,071.6 MB |
| Structural failures | 0 (`passed: true`) |
| Quality findings | `txs_features` blanks; wallet duplicate `(address, time step)` pairs |
| Evidence file | `reports/phase1_verification.json` |

Reproduce:

```bash
nice -n 19 python3 -u scripts/verify_dataset.py --low-priority \
    --json reports/phase1_verification.json
```

---

## 9. Assumptions & Unresolved Items

* **Assumption:** the wallet-row duplication is a property of the published Elliptic++
  file as downloaded; it was not compared against an independent upstream copy/SHA. The
  bytes were not modified locally.
* **Assumption:** `AddrAddr` duplicates are multi-edges and self-loops are intra-address
  transfers; the source transaction IDs behind them were not decomposed in this phase.
* **Unresolved:** the direction/semantics of `txs_edgelist` (`txId1` -> `txId2`) were not
  re-derived from raw Bitcoin; the project uses the published edge direction.
* Class 3 (`unknown`) labels are **not** legitimate/negative; they are excluded from
  supervised labels unless an experiment explicitly defines otherwise.
