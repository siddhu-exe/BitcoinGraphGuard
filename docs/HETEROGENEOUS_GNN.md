# Heterogeneous Graph Deep Learning (RGCN) — Phase 4 Execution Record & Methodology

> **Phase 4 Record.** This document records the methodology, architecture, leakage controls, and experimental benchmark for the **Heterogeneous Graph Neural Network (RGCN)** baseline in **BitcoinGraphGuard**.
> It evaluates multi-relational message passing across the bipartite graph of 203,769 transactions and 822,942 wallet addresses, analyzes how address nodes bridge the 100% intra-step edge confinement of homogeneous transaction graphs, and establishes the benchmark progression against the frozen baselines.
>
> Executable implementation: `notebooks/04_heterogeneous_gnn.ipynb` (authored for Google Colab / Kaggle GPU execution; artifacts exported to `heterogeneous_gnn/`).

---

## 1. Executive Summary & Headline Progression

Phase 4 introduces multi-relational heterogeneous graph learning to the BitcoinGraphGuard progression:

$$\text{Prevalence Baseline} \longrightarrow \text{Logistic Regression} \longrightarrow \text{MLP (Neural Baseline)} \longrightarrow \text{GraphSAGE (Homogeneous)} \longrightarrow \text{XGBoost (Frozen)} \longrightarrow \mathbf{\text{RGCN (Heterogeneous)}}$$

The phase evaluates whether incorporating **822,942 wallet address nodes** and **4,183,205 address-level directed edges** (`AddrTx`, `TxAddr`, `AddrAddr`) provides predictive information beyond transaction-only features and the homogeneous GraphSAGE baseline.

### Frozen Benchmark Progression on Primary Test 35–49 (16,670 nodes, 1,083 illicit, 6.50% prevalence):

| Model / Architecture | Heterogeneous Relations | Test PR-AUC | PR-AUC Lift over Base | Test ROC-AUC | Precision | Recall | Test F1 | Operating Threshold $\tau^*$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prevalence Baseline** | None | 0.0650 | $1.00\times$ | 0.5000 | 0.0650 | 1.0000 | 0.1220 | 0.065 |
| **Logistic Regression** | None | 0.2917 | $4.49\times$ | 0.8828 | 0.3547 | 0.5734 | 0.4385 | 0.655 |
| **2-Layer MLP (Neural Baseline)** | None | 0.4768 | $7.34\times$ | 0.8912 | 0.5964 | 0.5568 | 0.5759 | 0.710 |
| **2-Layer GraphSAGE (Homogeneous)**| `txs_edgelist` only (100% intra-step) | 0.6209 | $9.56\times$ | 0.9044 | 0.6991 | 0.5171 | 0.5945 | 0.830 |
| **XGBoost Baseline (500 trees)** | Tabular Neighborhood Aggregates | 0.8007 | $12.32\times$ | 0.9317 | 0.8258 | 0.7516 | 0.7870 | 0.515 |
| **XGBoost Optimized (Frozen)** | Tabular Neighborhood Aggregates | **0.8013** | $\mathbf{12.33\times}$ | **0.9281** | **0.7960** | **0.7682** | **0.7818** | **0.435** |
| **2-Layer HeteroRGCN** | `tx_to_tx`, `addr_to_tx`, `tx_to_addr`, `addr_to_addr` | **0.4682** | **$7.21\times$** | **0.8946** | **0.6241** | **0.4598** | **0.5295** | **0.795** |

*All operating thresholds $\tau^*$ are derived strictly on the validation set (steps 25–34) by F1 maximization and frozen prior to test evaluation.*

---

## 2. Core Research Questions & Motivation

### 1. Does Heterogeneous Message Passing Bridge Temporal Confinement?
- **Homogeneous Limitation (Phase 3):** All 234,355 edges in `txs_edgelist.csv` are 100% intra-step ($t_u = t_v$). Homogeneous GNNs cannot propagate representations across time steps.
- **Heterogeneous Hypothesis:** Wallet addresses persist across multiple time steps. A transaction in step $t_1$ outputs funds to wallet $A$ (`TxAddr`), which later funds transaction $t_2$ (`AddrTx`). Multi-hop heterogeneous convolutions enable cross-temporal flow propagation ($T_1 \to A \to T_2$).

### 2. Can RGCN Outperform Tabular Gradient Boosting?
- XGBoost ingests 72 pre-aggregated tabular neighborhood features (`Aggregate_feature_*`), reaching PR-AUC 0.8013.
- RGCN learns explicit relational embeddings over the actual 4.42M-edge graph topology, testing whether relational message passing can surpass hand-crafted tabular aggregations.

### 3. Does Heterogeneity Mitigate the Steps 43–49 Temporal Collapse?
- In steps 43–49 (2.53% prevalence), both XGBoost (0.0427) and GraphSAGE (0.0504) collapse under severe temporal regime shift.
- We evaluate whether persistent wallet representations provide structural memory that stabilizes predictions across the drift window.

---

## 3. Heterogeneous Graph Schema & Diagnostics

The heterogeneous graph is constructed from the official Elliptic++ dataset:

```text
                  +-------------------------+
                  |  Wallets / Addresses    |
                  |     (822,942 nodes)     |
                  +-------------------------+
                     /       |           ^
        AddrTx      /        | AddrAddr  |    TxAddr
    (477,117 edges)/         | (2.87M)   | (837,124 edges)
                  v          v           |
          +------------------------------------+
          |           Transactions             | <--- txs_edgelist (234,355 edges)
          |         (203,769 nodes)            |
          +------------------------------------+
```

### 3.1. Entity & Relation Inventory:

| Entity / Relation | Source Type | Target Type | Count | Feature Dimension | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Transaction Nodes** | `transaction` | — | 203,769 | 165 | 93 Local + 72 Aggregate features |
| **Address Nodes** | `address` | — | 822,942 | 55 | Historical wallet snapshot features |
| **`tx_to_tx`** | `transaction` | `transaction` | 234,355 | — | Direct transaction spending chain |
| **`addr_to_tx`** | `address` | `transaction` | 477,117 | — | Wallet input funding (`AddrTx`) |
| **`tx_to_addr`** | `transaction` | `address` | 837,124 | — | Transaction output deposit (`TxAddr`) |
| **`addr_to_addr`** | `address` | `address` | 2,868,964 | — | Direct wallet-to-wallet transfer (`AddrAddr`) |
| **Total Nodes** | — | — | **1,026,711** | — | Unique blockchain entities |
| **Total Directed Edges** | — | — | **4,417,560** | — | Multi-relational connectivity |

---

## 4. Relational Graph Convolutional Network (RGCN) Architecture

### 4.1. Mathematical Formulation
For node $i \in V$, the $(l+1)$-th layer representation is computed by aggregating neighbor messages across all active relation types $r \in \mathcal{R}$:

$$h_i^{(l+1)} = \sigma\left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{|\mathcal{N}_i^r|} W_r^{(l)} h_j^{(l)} \right)$$

Where:
* $W_0^{(l)}$ is the self-transformation weight matrix.
* $W_r^{(l)}$ is the relation-specific transformation matrix for edge type $r$.
* $\mathcal{N}_i^r$ is the neighborhood of node $i$ under relation $r$.
* Mean normalization $\frac{1}{|\mathcal{N}_i^r|}$ is applied per relation.

### 4.2. Layer Specifications:
* **Input Dimensions:**
  - Transaction ($X_{\text{tx}}$): $\mathbb{R}^{203,769 \times 165}$
  - Address ($X_{\text{addr}}$): $\mathbb{R}^{822,942 \times 55}$
* **Layer 1 (HeteroRGCN Conv):**
  - Linear self-transforms: $\text{tx} \to 128$, $\text{addr} \to 128$
  - Relation transforms:
    - $W_{\text{tx}\to\text{tx}}^{(1)}: 165 \to 128$
    - $W_{\text{addr}\to\text{tx}}^{(1)}: 55 \to 128$
    - $W_{\text{tx}\to\text{addr}}^{(1)}: 165 \to 128$
    - $W_{\text{addr}\to\text{addr}}^{(1)}: 55 \to 128$
  - Activation: ReLU, Dropout: 0.3
* **Layer 2 (HeteroRGCN Conv):**
  - Linear self-transforms: $\text{tx} \to 1$, $\text{addr} \to 128$
  - Relation transforms:
    - $W_{\text{tx}\to\text{tx}}^{(2)}: 128 \to 1$
    - $W_{\text{addr}\to\text{tx}}^{(2)}: 128 \to 1$
    - $W_{\text{tx}\to\text{addr}}^{(2)}: 128 \to 128$
    - $W_{\text{addr}\to\text{addr}}^{(2)}: 128 \to 128$
  - Output: 1 logit per transaction node for binary classification.

---

## 5. Leakage Controls & Temporal Split Protocol

### 5.1. Strict Invariants
1. **Target:** Primary transaction classification objective (illicit vs licit). Class 3 (unknown) is strictly excluded from loss computation.
2. **No Wallet Label Leakage:** Wallet classes in `wallets_classes.csv` are **NEVER** used as model features or supervision signals.
3. **No Future Lookahead in Address Features:** Wallet snapshot features are aggregated strictly using historical appearances ($t \le 24$ for selection, $t \le 34$ for refit).
4. **Historical StandardScaler:** `StandardScaler` fitted strictly on historical training nodes.
5. **Class Imbalance Loss Weighting:** `nn.BCEWithLogitsLoss(pos_weight=...)` computed strictly on historical training labels ($w_{\text{pos}} \approx 9.64$ on fit 1–24; $w_{\text{pos}} \approx 7.63$ on train 1–34).
6. **Operating Threshold Selection:** $\tau^*$ selected strictly by F1 maximization on validation period (25–34) and frozen before test evaluation.

---

## 6. Empirical Results & Scientific Findings

### 6.1. Primary Multi-Relational Ablation Analysis

| Architecture | Relations Used | Test PR-AUC | Test ROC-AUC | Test F1 | Lift over MLP Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2-Layer MLP Baseline** | None (Features Only) | 0.4768 | 0.8912 | 0.5759 | $+0.0000$ (Reference) |
| **HeteroRGCN (4 Relations)** | `tx_to_tx`, `addr_to_tx`, `tx_to_addr`, `addr_to_addr` | 0.4682 | 0.8946 | 0.5295 | $-0.0086$ ($-1.8\%$) |
| **GraphSAGE (Homogeneous)** | `tx_to_tx` only (100% intra-step) | 0.6209 | 0.9044 | 0.5945 | $+0.1441$ ($+30.2\%$) |
| **XGBoost Optimized (Frozen)** | Tabular Neighborhood Aggregates | **0.8013** | **0.9281** | **0.7818** | $\mathbf{+0.3245}$ ($\mathbf{+68.1\%}$) |

### 6.2. Scientific Insights & Root Cause Analysis

1. **Uniform Relational Aggregation Causes Message Dilution (Over-Smoothing):**
   - In 2-layer HeteroRGCN, wallet address nodes receive messages from all connected wallets via `AddrAddr` (2,868,964 edges, representing **65% of all graph edges**).
   - Because mean aggregation assigns equal weight to every incoming relation without edge-type attention or adaptive gating, dense wallet-to-wallet transfers diffuse the sharp, localized transaction signals.
   - Homogeneous GraphSAGE aggregates strictly across the 234,355 direct `tx_to_tx` edges, preserving local topological transaction patterns and achieving PR-AUC **0.6209** vs RGCN's **0.4682**.

2. **Cross-Temporal Flow Bridge Dynamics:**
   - As verified in `hetero_graph_diagnostics.csv`, **61,487 wallet addresses** are active across multiple time steps, successfully bridging **10,812 test transactions** to historical training transactions via 2-hop paths ($T_{\text{train}} \to A \to T_{\text{test}}$).
   - While this structural bridge exists, simple static mean aggregation mixes historical wallet states with noisy background transactions, demonstrating that multi-relational graphs require attention mechanisms (such as HGT) or temporal edge weighting to filter informative cross-step paths.

3. **Superiority of Gradient-Boosted Tabular Ensembles:**
   - XGBoost achieves PR-AUC **0.8013** because its 72 pre-aggregated tabular neighborhood features (`Aggregate_feature_*`) provide non-linear, tree-partitioned representations of local subgraphs without suffering from message-passing over-smoothing or high relational variance.

---

## 7. Temporal Sub-Window & Degradation Analysis

| Temporal Window | Time Steps | Labeled Nodes | Illicit Count | Illicit Prevalence | XGBoost PR-AUC | GraphSAGE PR-AUC | RGCN PR-AUC | RGCN Precision | RGCN Recall | RGCN F1 | RGCN TP | RGCN FP | RGCN FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Test** | 35–49 | 16,670 | 1,083 | 6.50% | **0.8013** | 0.6209 | 0.4682 | 0.6241 | 0.4598 | 0.5295 | 498 | 300 | 585 |
| **Early Test** | 35–42 | 9,983 | 914 | 9.16% | **0.9215** | 0.7346 | 0.6083 | 0.7347 | 0.5394 | 0.6221 | 493 | 178 | 421 |
| **Recent Drift** | 43–49 | 6,687 | 169 | 2.53% | 0.0427 | 0.0504 | **0.0550** | 0.0394 | 0.0296 | 0.0338 | 5 | 122 | 164 |

### Key Temporal Findings:
1. **Window 35–42 (Stationary Regime):** RGCN performs solidly with PR-AUC **0.6083** and F1 **0.6221** (Precision 73.5%, Recall 53.9%), correctly identifying 493 illicit transactions.
2. **Window 43–49 (Drift Regime):** Under severe temporal regime shift and reduced illicit prevalence (2.53%), RGCN scores PR-AUC **0.0550**, outperforming GraphSAGE (**0.0504**) and XGBoost (**0.0427**). The persistent wallet connectivity provides a marginal structural stabilizer across the temporal shift, though all models degrade significantly.

---

## 8. Execution Quality & Leakage Audit

The automated assertion suite in `results/heterogeneous_gnn/checks.csv` confirmed **46 of 46 checks passing** (`ok = True`):
- **Temporal Isolation:** Zero train/test overlap; `StandardScaler` fitted strictly on historical nodes (`fit 1–24` for validation tuning, `train 1–34` for refit).
- **Zero Label Leakage:** Wallet class labels from `wallets_classes.csv` were completely excluded from features and loss computation.
- **Historical Address Snapshots:** Wallet features aggregated strictly using historical appearances ($t \le 24$ for selection, $t \le 34$ for refit) without future lookahead.
- **Class Weighting:** $w_{\text{pos}} = 9.6381$ (fit 1–24) and $w_{\text{pos}} = 7.6349$ (train 1–34) computed strictly from training labels.
- **Operating Thresholds:** $\tau^* = 0.795$ derived strictly by validation F1 maximization on steps 25–34 and frozen prior to test evaluation.
- **Confusion Matrix Closure:** Arithmetic closed exactly ($498 + 300 + 15,287 + 585 = 16,670$).

---

## 9. Artifact Inventory (`results/heterogeneous_gnn/`)

The remote execution produced all required artifacts in `results/heterogeneous_gnn/`:

```text
results/heterogeneous_gnn/
├── rgcn_model.pt                    # Trained PyTorch state dict for HeteroRGCN (544 KB)
├── model_comparison.csv             # Primary benchmark table across all 6 models
├── temporal_metrics.csv             # Sub-window metrics (35-49, 35-42, 43-49)
├── ablation_results.csv             # Multi-relational architectural ablation table
├── errors_by_step.csv               # Temporal per-step confusion metrics
├── predictions.csv                  # Test predictions for all 16,670 nodes
├── split_windows.csv                # Split summary table
├── hetero_graph_diagnostics.csv     # Heterogeneous graph statistics and flow bridges
├── leakage_audit.csv                # 6-point leakage audit table
├── feature_list.json                # 165 selected transaction features
├── rgcn_hyperparameters.json        # Architecture and training hyperparameters
├── metrics.json                     # Machine-readable evaluation payload
├── checks.csv                       # 46 automated sanity assertions (46/46 passing)
├── heterogeneous_gnn_digest.txt     # One-page run digest
└── figures/
    ├── learning_curves.png          # Loss and validation PR-AUC curves
    ├── test_pr_curve_and_roc.png    # Precision-Recall and ROC curves
    ├── temporal_pr_auc.png          # Temporal sub-window comparison
    ├── hetero_ablation.png          # Ablation lift over MLP and GraphSAGE
    └── error_analysis.png           # Errors per time step and score distributions
```

---

## 10. Transition to Phase 5 (Temporal & Inductive Robustness)

### Key Takeaways from Phase 4:
1. **Heterogeneous Benchmark Firmly Established:** Relational RGCN achieves **PR-AUC 0.4682 / ROC-AUC 0.8946 / F1 0.5295** on Test 35–49.
2. **Empirical Understanding of Graph Learning Bottlenecks:** Simple relational mean aggregation over dense wallet connections introduces noise and over-smoothing; homogeneous GraphSAGE (0.6209) remains the superior GNN architecture for local transaction message passing, while XGBoost (0.8013) sets the benchmark ceiling.
3. **Temporal Adaptation Imperative:** All models suffer in the late drift window (steps 43–49). Phase 5 will comprehensively evaluate temporal inductive generalization, rolling retraining windows, and inductive robustness across unseen transaction and wallet distributions.

**Status: Phase 4 is COMPLETE (46/46 checks passing). The project is fully ready to advance to Phase 5 (`notebooks/05_temporal_inductive_evaluation.ipynb`).**
