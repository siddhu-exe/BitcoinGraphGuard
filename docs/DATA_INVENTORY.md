# Elliptic++ Dataset Inventory & Verification Report

**Dataset Version:** Elliptic++ (Transactions + Wallets/Addresses Heterogeneous Temporal Graph)  
**Verification Date:** 2026-09-21  
**Verification Mode:** Pure Streaming / Zero Memory Leak (compatible with low-resource environments)
**Implementation Architecture:** `ARCHITECTURE.md` (laptop vs Kaggle/Colab compute split)

---

## 1. Dataset Location & Directory Structure

The dataset is located in the local directory:
```
/home/siddharth/Desktop/Projects/projects/BitcoinGraphGuard/Og data/
```
Total Raw Size: **~2.10 GB** across **9 CSV files**.

---

## 2. File Inventory & Specifications

| File Name | Category | Format | Size | Rows (Data) | Columns | Key Identifiers / Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `txs_features.csv` | Transactions | CSV | 662.60 MB | 203,769 | 184 | `txId`, `Time step`, Features (182) |
| `txs_classes.csv` | Transactions | CSV | 2.25 MB | 203,769 | 2 | `txId`, `class` |
| `txs_edgelist.csv` | Tx Graph | CSV | 4.26 MB | 234,355 | 2 | `txId1` -> `txId2` |
| `wallets_features.csv` | Wallets | CSV | 578.37 MB | 1,268,260 | 57 | `address`, `Time step`, Features (55) |
| `wallets_classes.csv` | Wallets | CSV | 29.01 MB | 822,942 | 2 | `address`, `class` |
| `wallets_features_classes_combined.csv` | Wallets | CSV | 580.79 MB | 1,268,260 | 58 | `address`, `Time step`, `class`, Features (55) |
| `AddrTx_edgelist.csv` | Bipartite Edge | CSV | 20.26 MB | 477,117 | 2 | `input_address` -> `txId` |
| `TxAddr_edgelist.csv` | Bipartite Edge | CSV | 35.00 MB | 837,124 | 2 | `txId` -> `output_address` |
| `AddrAddr_edgelist.csv` | Wallet Graph | CSV | 191.34 MB | 2,868,964 | 2 | `input_address` -> `output_address` |

---

## 3. Node & Label Statistics

### 3.1 Transactions (`txId`)
* **Total Rows:** 203,769
* **Unique Transaction IDs:** 203,769 (0 duplicates, 0 missing IDs)
* **Time Steps:** 49 discrete time steps (min: 1, max: 49)
* **Features (182 total):**
  - `Local_feature_1` to `Local_feature_93` (93 local transaction features)
  - `Aggregate_feature_1` to `Aggregate_feature_72` (72 aggregated 1-hop neighborhood features)
  - 17 explicit domain features: `in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, `in_BTC_min`, `in_BTC_max`, `in_BTC_mean`, `in_BTC_median`, `in_BTC_total`, `out_BTC_min`, `out_BTC_max`, `out_BTC_mean`, `out_BTC_median`, `out_BTC_total`.
* **Label Distribution (`txs_classes.csv`):**
  - **Class 1 (Illicit):** 4,545 (2.23%)
  - **Class 2 (Licit):** 42,019 (20.62%)
  - **Class 3 (Unknown / Unlabeled):** 157,205 (77.15%)
  - *Labeled Total (1 + 2):* 46,564 (22.85%)

### 3.2 Wallets / Addresses (`address`)
* **Unique Wallet Addresses:** 822,942
* **Temporal Snapshot Rows in Features:** 1,268,260 (addresses active across multiple time steps appear once per active time step)
* **Time Steps:** 49 discrete time steps (min: 1, max: 49)
* **Features (55 total):** Activity metrics including transacted BTC, fees, sent/received totals, block spans, address reuse counts, and temporal gaps.
* **Label Distribution (`wallets_classes.csv`):**
  - **Class 1 (Illicit):** 14,266 (1.73%)
  - **Class 2 (Licit):** 251,088 (30.51%)
  - **Class 3 (Unknown / Unlabeled):** 557,588 (67.76%)
  - *Labeled Total (1 + 2):* 265,354 (32.24%)

---

## 4. Edge & Graph Topology

| Edge Relationship | Source Node Type | Destination Node Type | Total Edges | Unique Edges | Duplicate Edges | Self-Loops | Unique Sources | Unique Destinations | Total Nodes Involved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `txs_edgelist.csv` | Transaction | Transaction | 234,355 | 234,355 | 0 | 0 | 166,345 | 148,447 | 203,769 |
| `AddrTx_edgelist.csv` | Address/Wallet | Transaction | 477,117 | 477,117 | 0 | 0 | 400,212 | 202,804 | 603,016 |
| `TxAddr_edgelist.csv` | Transaction | Address/Wallet | 837,124 | 837,124 | 0 | 0 | 202,804 | 641,043 | 843,847 |
| `AddrAddr_edgelist.csv` | Address/Wallet | Address/Wallet | 2,868,964 | 2,784,344 | 84,620 | 45,981 | 400,212 | 641,043 | 822,942 |

---

## 5. Cross-Entity Alignment & Data Integrity

1. **Transaction Integrity:**
   - 100% match between `txs_features.csv` (203,769 txIds) and `txs_classes.csv` (203,769 txIds).
   - All 203,769 transactions appear in `txs_edgelist.csv`.
   - 202,804 transactions appear in `AddrTx` and `TxAddr` (965 transactions do not have input/output address mappings, representing coinbase or aggregated transactions).

2. **Wallet Address Integrity:**
   - 100% match between `wallets_classes.csv` (822,942 addresses) and unique addresses in `wallets_features.csv`.
   - All 400,212 input addresses from `AddrTx` and `AddrAddr` are present in `wallets_features`.
  - All 641,043 output addresses from `TxAddr` and `AddrAddr` are present in `wallets_features`.

3. **Data Quality Summary:**
   - **Missing Values:** 0 missing values or broken rows in transaction files.
   - **Duplicates:**
     - 0 duplicate transaction IDs.
     - 0 duplicate edge definitions in `txs_edgelist`, `AddrTx_edgelist`, and `TxAddr_edgelist`.
     - `AddrAddr_edgelist` contains 84,620 multi-edges (repeated payments between same wallet pairs across different transactions) and 45,981 self-loops (change addresses / self-transfers).

---

## 6. Key Modeling & Architectural Implications

1. **Heterogeneous Graph Formulation (PyG / DGL):**
   - **Node Types:**
     - `transaction`: 203,769 nodes, feature dimension = 182
     - `wallet`: 822,942 nodes, feature dimension = 55
   - **Edge Types (Canonical Directional):**
     - `('wallet', 'sends_to', 'transaction')`: 477,117 edges
     - `('transaction', 'receives_to', 'wallet')`: 837,124 edges
     - `('transaction', 'flows_to', 'transaction')`: 234,355 edges
     - `('wallet', 'transacts_with', 'wallet')`: 2,784,344 unique edges

2. **Severe Class Imbalance:**
   - Illicit transactions account for **2.23%** of all transactions (and only 9.76% of labeled transactions).
   - Illicit wallets account for **1.73%** of all wallets (and 5.38% of labeled wallets).
   - **Metric Focus:** PR-AUC, F1 (Minority/Illicit), Recall@Precision, Cost-sensitive loss weighting.

3. **Temporal Dynamics:**
   - 49 chronological time steps (~2-week intervals per time step in Bitcoin blockchain).
   - Evaluation protocol must strictly train on historical time steps (e.g. steps 1–34) and evaluate on future unseen steps (e.g. steps 35–49) to prevent lookahead bias.

4. **Resource Constraints & Optimization:**
   - Total graph representation contains ~1.03M unique nodes (203,769 transactions + 822,942 wallets) and ~4.42M heterogeneous directed edges.
   - For a machine with 5.6 GB RAM, graph batching (NeighborLoader / HeteroNeighborLoader / mini-batch sampling) and streaming chunk pipelines are required rather than full-graph in-memory training.
   - Heavy GNN training runs on Kaggle/Google Colab, not on the laptop (`ARCHITECTURE.md`).

---

## 7. Verification Status (2026-09-21)

Re-confirmed by re-running `../scripts/verify_dataset.py` (memory-safe revision) on the
small/medium files, with peak RSS documented per run:

| Check | Independent re-run result | Status |
| :--- | :--- | :--- |
| `txs_edgelist.csv` counts | 234,355 edges, 0 dup, 0 self-loops, 203,769 nodes | Verified (82 MB peak) |
| `txs_classes.csv` labels | 203,769 rows, 0 dup, {1: 4,545, 2: 42,019, 3: 157,205}, 0 missing | Verified |
| `wallets_classes.csv` labels | 822,942 rows, 0 dup, {1: 14,266, 2: 251,088, 3: 557,588}, 0 missing | Verified |
| `AddrTx_edgelist.csv` counts | 477,117 edges, 0 dup, 0 self-loops, 400,212 src / 202,804 dst | Verified |
| `TxAddr_edgelist.csv` counts | 837,124 edges, 0 dup, 0 self-loops, 202,804 src / 641,043 dst | Verified |
| Column counts | `txs_features`=184, `wallets_features`=57, `wallets_features_classes_combined`=58 | Verified (headers) |
| `txs_features.csv` rows/time steps | 203,769 rows, 49 time steps | From prior run; not yet re-run (663 MB) |
| `wallets_features.csv` rows | 1,268,260 rows, 49 time steps | From prior run; not yet re-run (579 MB) |
| `wallets_features_classes_combined.csv` rows | 1,268,260 rows | From prior run; not yet re-run (581 MB) |
| `AddrAddr_edgelist.csv` counts | 2,868,964 edges, 2,784,344 unique, 84,620 dup, 45,981 self-loops | From prior run; not yet re-run (192 MB) |

**Note:** the four large files are marked "not yet re-run" because a full 2.1 GB pass was
intentionally skipped on this low-resource laptop. Re-run after dependency setup with:

```bash
source .venv/bin/activate
python scripts/verify_dataset.py
```
