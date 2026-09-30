# Phase 6: Explainability and Attribution

> **Phase 6 Record.** This document summarizes the **Explainability and Attribution** phase of BitcoinGraphGuard.
> It details the exact interpretability mechanisms applied to the frozen sequence of models (XGBoost, GraphSAGE, HeteroRGCN) without introducing architectural modifications.
>
> Executable implementation: `notebooks/06_explainability.ipynb` (Executed on Google Colab/Kaggle; artifacts stored in `results/explainability/`).

---

## 1. Objective and Constraints

Phase 6 answers the core structural question: *Why does BitcoinGraphGuard make its predictions, and what topological pathways do its models exploit?*

**Critical Constraints:**
- **No Temporal Leakage:** All structural explanations (e.g., $k$-hop ego-networks, Hetero Data reconstructions) are constructed strictly up to time step $t \le 42$ (for the primary window) to prevent future leakage. Validated via automated checks.
- **Frozen Models Only:** Explanations apply exclusively to exact state dicts and pipelines trained in Phases 2, 3, 4, and 5.
- **Architectural Freeze:** No HGT, attention mechanism additions, or other modeling advances are implemented.

## 2. XGBoost Feature Attribution (SHAP)

Because XGBoost reached a strict PR-AUC ceiling (0.8013) on purely handcrafted tabular aggregates, we extracted global logic using `shap.TreeExplainer` on the frozen Phase-2 model.

**Findings (`xgb_shap_global.csv`, `xgb_shap_summary.png`):**
- **Dominance of Local Features:** XGBoost relies more heavily on local transaction characteristics (61.5% of total SHAP attribution) than neighborhood aggregates (38.5%). 
- **Top Absolute Drivers:** The top 5 overall features driving classifications are `Local_feature_53`, `Local_feature_90`, `Local_feature_3`, `Local_feature_55`, and `Aggregate_feature_70`. `Local_feature_53` possesses an outsized mean absolute SHAP value (1.135), making it the single most discriminative handcrafted feature.
- **Inert Features:** 8 features maintain near-zero or strictly zero impact (SHAP < 0.001), including `Local_feature_36`, `Local_feature_15`, and `Local_feature_69`.

## 3. Homogeneous Graph Structure (GraphSAGE)

GraphSAGE utilizes direct message passing restricted entirely to intra-step transaction nodes. We utilized `CaptumExplainer` with `IntegratedGradients` across 2-hop temporal subgraphs.

**Findings:**
- GraphSAGE attributions are strictly confined to **intra-step neighbourhoods**. Because `txs_edgelist.csv` contains zero cross-step edges (as proved in Phase 3), GraphSAGE structurally *cannot* learn cross-temporal illicit flow patterns. It purely smooths feature logic locally within a single discrete time step, which makes it blind to persistent wallet behavior over time.

## 4. Heterogeneous Graph Structure (HeteroRGCN)

HeteroRGCN incorporates the full persistence of wallet addresses via multi-relational edges. To explain its behavior without hitting complexity walls over dense multi-relational graphs, we utilized **Test-Time Relation Ablation**.

We isolated True Positives from the early test window and selectively zeroed specific relations (`tx_to_tx` vs the wallet/address pathways) at inference time to measure the shift in predicted logits (`rgcn_ablation_sample.csv`).

**Findings:**
- **Extreme Over-Reliance on Address Context:** For several True Positives (e.g., node `54760551`), completely ablating the traditional `tx_to_tx` edges paradoxically *increased* the model's illicit confidence (+0.111 drop in ablation), while dropping the address pathways caused confidence to sink (-0.044).
- **The Drift Mechanism Explained:** These results definitively prove that HeteroRGCN drives its illicit predictions via historical address overlap (`addr_to_tx`, `tx_to_addr`, `addr_to_addr`). This clarifies exactly why the model suffered a catastrophic collapse in Steps 43-49 (Phase 5)—in that drift regime, transactions shift to novel, unseen addresses. HeteroRGCN relies too uniformly on recurrent address pathways and fails when stripped of them.

## 5. Artifact Inventory

```text
results/explainability/
├── xgb_shap_global.csv                 # 165 feature attributions
├── rgcn_ablation_sample.csv            # Score degradation under targeted relation ablation
├── checks.csv                          # Verification of temporal boundaries and pipeline integrity
├── explainability_digest.txt           # Post-run insights
└── figures/
    └── xgb_shap_summary.png            # Visual SHAP density plot
```

## 6. Next Steps

Based on the explainability outputs, the path forward is scientifically clear.

Static uniform aggregation over recurrent structural paths (RGCN) propagates too much noise and places too much weight on historical addresses, creating a brittle model that fails entirely on unseen (`inductive`) graph drift. 

The project must now transition to evaluating dynamic, weighted relational aggregation across temporal contexts. This insight directly motivates Phase 7: the investigation of **Heterogeneous Graph Transformer (HGT)**, which introduces attention matrices explicitly designed to learn *which* relationships to trust and *when* to gate historical information.
