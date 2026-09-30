# Phase 7: Heterogeneous Graph Transformer (HGT) — Methodology & Benchmark Specification

> **Phase 7 Record.** This document records the methodology, architecture, leakage controls, and experimental benchmark for the **Heterogeneous Graph Transformer (HGT)** in **BitcoinGraphGuard**.
> It investigates whether multi-head relational and neighborhood attention mechanisms (Hu et al., *WWW 2020*) can resolve the message dilution observed in HeteroRGCN and close the performance gap to XGBoost on the Bitcoin heterogeneous graph.
>
> Executable implementation: `notebooks/07_hgt.ipynb` (authored via `scripts/generate_notebook_07.py` for Google Colab / Kaggle GPU execution; artifacts exported to `results/hgt/`).

---

## 1. Executive Summary & Progression

Phase 7 introduces learned relational attention across the bipartite graph of 203,769 transactions and 822,942 wallet addresses:

$$\text{Prevalence (0.0650)} \longrightarrow \text{Logistic Regression (0.2917)} \longrightarrow \text{MLP (0.4768)} \longrightarrow \text{HeteroRGCN (0.4682)} \longrightarrow \text{GraphSAGE (0.6216)} \longrightarrow \mathbf{\text{XGBoost (0.8013)}} \longleftrightarrow \mathbf{\text{HGT}}$$

### Unified Benchmark Progression on Primary Test 35–49 (16,670 nodes, 1,083 illicit, 6.50% prevalence):

| Model / Architecture | Graph Topology / Relations | Test PR-AUC | Test ROC-AUC | Precision | Recall | Test F1 | Operating Threshold $\tau^*$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prevalence Baseline** | None | 0.0650 | 0.5000 | 0.0650 | 1.0000 | 0.1220 | 0.065 |
| **Logistic Regression** | None (165 Tabular Features) | 0.2917 | 0.8828 | 0.3547 | 0.5734 | 0.4385 | 0.655 |
| **2-Layer MLP Baseline** | None (165 Tabular Features) | 0.4768 | 0.8912 | 0.5964 | 0.5568 | 0.5759 | 0.710 |
| **HeteroRGCN (Phase 4)** | 4 Relations (`tx_to_tx`, `addr_to_tx`, `tx_to_addr`, `addr_to_addr`) | 0.4682 | 0.8946 | 0.6241 | 0.4598 | 0.5295 | 0.795 |
| **GraphSAGE Baseline (Phase 3)** | `txs_edgelist` only (100% intra-step) | 0.6216 | 0.9044 | 0.6991 | 0.5171 | 0.5945 | 0.830 |
| **GraphSAGE O4 (Phase 3b)** | Cross-Step + Lag + Pos-Weight BCE | 0.6001 | 0.8993 | 0.6715 | 0.5291 | 0.5919 | 0.815 |
| **XGBoost Baseline (Phase 2)** | 165 Features (72 Tabular Graph Aggregates) | 0.8007 | 0.9317 | 0.8258 | 0.7516 | 0.7870 | 0.515 |
| **XGBoost Optimized (Phase 2)** | 165 Features (72 Tabular Graph Aggregates) | **0.8013** | **0.9281** | **0.7960** | **0.7682** | **0.7818** | **0.435** |
| **HeteroHGT (Phase 7)** | 4 Relations (Learned Multi-Head Relational Attention) | *Evaluating* | *Evaluating* | *Evaluating* | *Evaluating* | *Evaluating* | *Val-F1* |

---

## 2. Core Research Hypotheses

### Hypothesis 1: Relational Dilution vs. Dynamic Attention (Primary Test 35–49)
- **Problem in RGCN (Phase 4 & 6):** Uniform mean aggregation assigns identical weight to every incoming edge within a relation. Because `AddrAddr` accounts for 65% of all edges (2,868,964 edges), high-degree exchange hot wallets and mining pools dilute sharp fraud signals into background noise, causing RGCN (0.4682) to underperform even the simple homogeneous GraphSAGE (0.6216).
- **HGT Mechanism:** HGT replaces uniform mean aggregation with **multi-head mutual attention**, parameterizing attention by source node type $\tau(s)$, target node type $\tau(t)$, and relation type $\phi(e)$. This allows the model to learn relation scaling scalars ($\mu_{\text{rel}}$ / `a_rel`) and downweight high-degree reused hub addresses.

### Hypothesis 2: Late Temporal Drift Window (Steps 43–49 — The Decisive Test)
- In steps 43–49, illicit transaction prevalence abruptly drops from 9.16% to **2.53%**, and all previous models collapsed identically to $\approx 0.04 - 0.055$ PR-AUC (XGBoost 0.0427, GraphSAGE 0.0504, RGCN 0.0550).
- **The Decisive Test:** If HGT improves performance on steps 35–42 but still collapses to $\sim 0.05$ on 43–49, this empirically proves that the late drift failure is **not an aggregation-weighting flaw**, but an **irreducible out-of-distribution regime shift** requiring continuous retraining or active supervision.

---

## 3. Heterogeneous Graph Transformer Architecture

### 3.1. Mathematical Formulation
For a node pair $(s, t)$ connected by edge $e = (s, \phi(e), t)$:

1. **Heterogeneous Mutual Attention**:
   $$\text{Attention}(s, e, t) = \text{Softmax}_{j \in \mathcal{N}(t)}\left( \bigoplus_{h=1}^H \frac{K^h(s) W_{\phi(e)}^{\text{ATT}} {Q^h(t)}^T}{\sqrt{d}} \cdot \mu_{\langle \tau(s), \phi(e), \tau(t) \rangle} \right)$$
   Where $Q^h(t) = \text{Linear}_{\tau(t)}^Q(h_t)$, $K^h(s) = \text{Linear}_{\tau(s)}^K(h_s)$, and $\mu$ is a relation scaling parameter.

2. **Heterogeneous Message Passing**:
   $$\text{Message}(s, e, t) = \bigoplus_{h=1}^H \left( \text{Linear}_{\tau(s)}^V(h_s) W_{\phi(e)}^{\text{MSG}} \right)$$

3. **Target Aggregation & Residual Connection**:
   $$h_t^{(l+1)} = \text{GELU}\left( \text{Linear}_{\tau(t)}^{\text{up}}\left( \sum_{s \in \mathcal{N}(t)} \text{Attention}(s, e, t) \cdot \text{Message}(s, e, t) \right) + h_t^{(l)} \right)$$

### 3.2. Architecture Specifications
* **Input Projections**: Linear transforms for `tx` ($165 \to 64$) and `addr` ($55 \to 64$).
* **Convolutions**: 2 layers of `HGTConv(in_channels=64, out_channels=64, metadata=data.metadata(), heads=4, group='sum')`.
* **Classifier**: Linear layer on transaction embeddings ($64 \to 1$ logit).
* **Regularization**: Dropout $p = 0.3$, GELU non-linearities, AdamW optimizer (`lr=0.001`, `weight_decay=1e-4`).
* **Scalability**: Mini-batch neighborhood sampling using `HeteroNeighborLoader` (`batch_size=1024`, `num_neighbors=[10, 10]`).

---

## 4. Strict Leakage Controls & Temporal Protocol

1. **Strict Chronological Splits**:
   - **Fit / Feature Selection Window**: Steps 1–24 (29,894 labeled transactions).
   - **Validation Window**: Steps 25–34 (9,091 labeled transactions).
   - **Full Historical Refit Window**: Steps 1–34 (38,985 labeled transactions).
   - **Primary Test Window**: Steps 35–49 (16,670 labeled transactions; 1,083 illicit).
2. **Zero Lookahead in Address Snapshots**: Address feature aggregation incorporates historical snapshots strictly up to the training cutoff ($t \le 24$ for selection, $t \le 34$ for refit).
3. **Class-Weighted BCE Loss**: `pos_weight` derived strictly from historical training labels ($w_{\text{pos}} \approx 9.64$ on fit 1–24; $w_{\text{pos}} \approx 7.63$ on train 1–34).
4. **Operating Threshold Freezing**: $\tau^*$ derived strictly by F1 maximization on the validation window (25–34) and frozen before test evaluation.

---

## 5. Direct Mechanistic Attention Analysis on True Positives

To verify whether HGT actually solves the dilution problem:
1. **Relational Scalars (`a_rel`)**: We log the learned scaling scalars across all 4 relation types to verify if HGT prioritizes direct transaction flow (`tx_to_tx`, `addr_to_tx`) over dense background wallet transfers (`addr_to_addr`).
2. **True Positive Edge Attention**: For sampled test True Positives, we compute exact edge attention scores $\alpha_{u \to v}$ on incoming `AddrAddr` and `AddrTx` edges and correlate them with source address degree.
3. **Hypothesis Verification**: We check if high-degree exchange hubs (in-degree $> 50$) receive lower attention weights than low-degree specific wallet addresses.

---

## 6. Artifact Inventory (`results/hgt/`)

```text
results/hgt/
├── hgt_model.pt                     # Trained PyTorch state dict for HGT
├── model_comparison.csv             # Full benchmark comparison against all 8 baselines
├── temporal_metrics.csv             # 3-window breakdown (35-49, 35-42, 43-49)
├── inductive_metrics.csv            # Seen vs Unseen address context metrics
├── hgt_attention_weights.csv        # Learned relational scaling parameters (a_rel)
├── tp_addr_attention.csv            # True positive edge attention vs address degree
├── relation_ablation.csv            # Full graph vs drop AddrAddr vs drop TxTx
├── errors_by_step.csv               # Temporal per-step confusion metrics
├── predictions.csv                  # Test predictions for all 16,670 nodes
├── hgt_hyperparameters.json         # Architecture and training configuration
├── metrics.json                     # Machine-readable evaluation payload
├── checks.csv                       # Automated assertion suite
├── digest.txt                       # Executive summary run digest
└── figures/
    ├── learning_curves.png          # Training loss and validation PR-AUC curves
    ├── test_pr_roc.png              # Precision-Recall & ROC curves
    ├── temporal_pr_auc.png          # Temporal sub-window comparison bar chart
    ├── attention_weights.png        # Learned relational scalars and attention vs degree
    └── error_analysis.png           # Errors per time step and score distributions
```

---

## 7. MLOps Transition & Production Model Selection (Phase 8)

Phase 7 provides the final evidence for the production serving architecture in Phase 8:
- **Primary Production Model**: **XGBoost Optimized (Frozen, 0.8013 PR-AUC)** is selected as the primary real-time scoring engine in the FastAPI service (`src/api/`), due to its tabular superiority, $12.33\times$ lift over prevalence, and sub-millisecond inference latency without multi-hop graph overhead.
- **Structural Auxiliary Model**: HeteroHGT serves as a structural explainer and attention-weighted cluster diagnostic module.
- **Drift Monitoring**: Automated drift detection (KS tests, adversarial validation triggers) is prioritized in MLOps monitoring.
