# Phase 6: Explainability and Attribution

> **Phase 6 Record.** This document acts as a scaffold for the **Explainability and Attribution** phase of BitcoinGraphGuard.
> It details the exact interpretability mechanisms applied to the frozen sequence of models (XGBoost, GraphSAGE, HeteroRGCN) without introducing architectural modifications.
>
> Executable implementation: `notebooks/06_explainability.ipynb` (to be run on Google Colab / Kaggle; artifacts will be exported to `results/explainability/`).

---

## 1. Objective and Constraints

Phase 6 answers the core structural question: *Why does BitcoinGraphGuard make its predictions, and what topological pathways do its models exploit?*

**Critical Constraints:**
- **No Temporal Leakage:** All structural explanations (e.g., $k$-hop ego-networks) are constructed strictly up to time step $t \le 42$ (for the primary window) to prevent future leakage. 
- **Frozen Models Only:** Explanations apply exclusively to exact state dicts and pipelines trained in Phases 2, 3, 4, and 5.
- **Architectural Freeze:** No HGT, attention mechanism additions, or other modeling advances are implemented.

## 2. Methodology

### 2.1 XGBoost feature attribution (SHAP)
Because XGBoost reached a strict PR-AUC ceiling (0.8013) on purely handcrafted tabular aggregates, we extract global and local logic using `shap.TreeExplainer`.
- **Global SHAP:** Identifies which of the 165 features consistently drive illicit classification globally (`xgb_shap_global.csv`, `xgb_shap_summary.png`).
- **Feature Types:** Distinct analysis of `Local_feature_*` vs `Aggregate_feature_*` structural reliance over the prediction windows.

### 2.2 Homogeneous Graph Structure (GraphSAGE)
GraphSAGE utilizes direct message passing restricted entirely to intra-step transaction nodes.
- **Explainability Mechanism:** `CaptumExplainer` with `IntegratedGradients` across specifically extracted 2-hop temporal subgraphs.
- **Output:** Exact feature and edge-mask importance scores (`graphsage_attributions_sample.csv`), mapping local $T \to T$ reliance and identifying whether GraphSAGE focuses purely on feature vectors or exploits actual structural pathways to compensate for its homogeneous limit.

### 2.3 Heterogeneous Graph Structure (HeteroRGCN)
HeteroRGCN incorporates the full persistence of wallet addresses via multi-relational edges. Since standard gradient-based explainers hit complexity walls on multi-relational structures, we use Test-Time Relation Ablation.
- **Method:** Specific relation pathways (`tx_to_tx`, `tx_to_addr`, `addr_to_addr`, `addr_to_tx`) are surgically zeroed out at inference.
- **Output:** The shift in logits isolates exactly which relational bridges drive true and false positives (`rgcn_ablation_sample.csv`).
- **Relevance:** This confirms the mechanism behind the `seen` vs `unseen` address collapse witnessed in the Phase 5 temporal drift regime.

## 3. Results and Structural Insights
*(To be populated post-Colab execution)*

- **XGBoost Drivers:** *(Details on top tabular split logic)*
- **GraphSAGE Attributions:** *(Details on structural versus feature importance bounds)*
- **HeteroRGCN Relation Analysis:** *(Connections to the drift collapse and address recurrences)*

## 4. Artifact Inventory

Upon completion of the Colab execution, the following artifacts will exist in `results/explainability/`:

```text
results/explainability/
├── xgb_shap_global.csv                 # Mean absolute SHAP values for 165 features
├── xgb_shap_summary.png                # Global SHAP summary plot
├── graphsage_attributions_sample.csv   # Captum attribute/edge masks for GraphSAGE predictions
├── rgcn_ablation_sample.csv            # Score degradation under targeted relation ablation
├── explainability_digest.txt           # Post-run insights
├── checks.csv                          # Verification of temporal boundaries and pipeline integrity
└── figures/                            # Explainer visualizations
```

## 5. Next Steps

Based on the explainability outputs, the project will move to evaluate dynamic relational aggregation across temporal contexts. As proven here, static uniform aggregation over recurrent structural paths propagates noise. This insight directly motivates the investigation of **Heterogeneous Graph Transformer (HGT)** with temporal embeddings and automated relational gating (Phase 7).

---
*Note: This document reflects the structural design of Phase 6. Quantitative cells remain a scaffold until the notebook executes on a capable compute environment.*
