# Diagnostic Follow-up Verdict: Permutation Importance & Collinear Redundancy

## Verdict: (b) CONCENTRATION CONFIRMED — Genuine Localized Aggregate Drift

### Key Empirical Findings:
1. **Baseline Adversarial AUC**: 5-Fold Mean AUC on 72 Aggregate Features = **1.0000 +/- 0.0000**.
2. **Attribution Share Redistribution**:
   - **Gain Top 3 Share**: **91.0%** (Gini = 0.9510)
   - **SHAP Top 3 Share**: **78.5%** (Gini = 0.9322)
   - **Permutation Top 3 Share**: **99.9%** (Gini = 0.9761)
3. **Discrepancy Analysis**: Identified **2** features over-credited by greedy tree splits and **4** collinear sister features under-credited despite carrying redundant causal drift signal.
4. **Grouped Block Permutation Performance Drops**:
   - **Focal Triad (`Aggregate_feature_10, Aggregate_feature_43, Aggregate_feature_8`)**: $\Delta \text{AUC} = 0.30550$
   - **Top Correlated Triad (`Aggregate_feature_7, Aggregate_feature_46, Aggregate_feature_44`)**: $\Delta \text{AUC} = 0.00000$
   - **Random Aggregate Triad Baseline (N=15)**: $\Delta \text{AUC} = 0.00000$

### Scientific Conclusion & Reconciliation with Locals-Only Drift:
Permutation importance confirms that Aggregate_feature_10, 43, and 8 genuinely dominate separability within aggregate features (Permutation Top 3 share = 99.9%), with non-focal sister features failing to replicate the drop. The GBDT splitting artifact claim was incorrect: aggregate drift is genuinely concentrated in these 3 features, while local transaction drift is separately diffuse (two distinct drift mechanisms).

Combined with the Phase 7b finding that Local Features alone achieve $\text{AUC} = 0.9885$, this audit confirms that the temporal regime shift between training (1–34) and drift (43–49) is a systemic, multi-dimensional distribution shift, mandating automated drift monitoring and adaptive retraining in Phase 8.
