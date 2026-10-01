# Out-of-Distribution (OOD) Diagnosis & Leak-Feature Audit — Methodology & Specification

> **Diagnostic Pass Record.** This document records the methodology, statistical formulations, leak criteria, and robustness benchmarks for the diagnostic audit executed in `notebooks/08_ood_diagnosis.ipynb`.
> It investigates whether the adversarial validation $\text{AUC} = 1.0000$ observed between training (steps 1–34) and late drift (steps 43–49) represents genuine, distributed out-of-distribution regime shift or an artifact of mechanical leak features.

---

## 1. Research Context & Problem Statement

Across Phases 1 through 7, four architecturally distinct models exhibited an identical collapse in the late temporal test window (**steps 43–49**, where illicit prevalence drops from 9.16% to 2.53%):

$$\text{XGBoost (0.8013 $\to$ 0.0427)} \quad|\quad \text{GraphSAGE (0.6216 $\to$ 0.0505)} \quad|\quad \text{HeteroRGCN (0.4682 $\to$ 0.0550)} \quad|\quad \text{HeteroHGT (0.4861 $\to$ 0.0386)}$$

In Phase 2b (`docs/XGBOOST_V2.md`), an adversarial validation classifier distinguishing training transactions (steps 1–34) from drift transactions (steps 43–49) achieved **$\text{AUC} = 1.0000$**, with median top-15 Kolmogorov-Smirnov distance $0.5336$, while an in-window 5-fold cross-validation on 43–49 preserved high discriminatory power ($\text{PR-AUC} = 0.9199$).

### The Decisive Audit Objective
Adversarial validation separability ($\text{AUC} = 1.0000$) can arise from two mutually exclusive phenomena:
1. **Hypothesis A (Distributed OOD Regime Shift)**: Covariate shift is broadly distributed across dozens of topological and transactional features, reflecting genuine macroeconomic and behavioral shifts across Bitcoin network regimes.
2. **Hypothesis B (Mechanical Leak Artifact)**: A small set of 1–3 features trivially encode temporal position (e.g., cumulative transaction counters, running aggregate volume, unnormalized sequence IDs), creating an artificial $\text{AUC} = 1.0000$ separation without representing real behavioral drift.

---

## 2. Six-Step Audit Protocol

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        6-STEP OOD DIAGNOSTIC AUDIT WORKFLOW                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Full Attribution Re-Run   │ Retrain adversarial classifier (1-34 vs 43-49).         │
│                              │ Extract BOTH Gain and TreeExplainer SHAP for Top 20.    │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ 2. Leak-Feature Audit        │ Test Top 20 for: (a) Spearman rank corr vs Time step,   │
│                              │ (b) Cumulative/running naming patterns,                 │
│                              │ (c) Variance ratio signature (η² = σ²_between / σ²_tot).│
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ 3. Concentration Test        │ Measure Top 1, Top 3, Top 10 Gain & SHAP shares; Gini.  │
│                              │ Evaluate localized vs. distributed attribution.         │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ 4. Leave-One-Out Robustness  │ Re-run adversarial model after dropping Top 1, 3, 5, 10,│
│                              │ Top 20, Top 15 KS, and all Aggregate features.          │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ 5. Production Model Overlap  │ Cross-check adversarial drift features vs. frozen       │
│                              │ XGBoost fraud classifier features (4-quadrant mapping). │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ 6. Per-Step Granularity      │ Compute adversarial AUC for each individual step 35..49 │
│                              │ against train (test uniform vs late-onset shift).       │
└──────────────────────────────┴─────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations & Leak Diagnostic Criteria

### 3.1. Adversarial Validation Formulation
Let $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, 0)\}_{i=1}^{N_0}$ for $t_i \in [1, 34]$ and $\mathcal{D}_{\text{drift}} = \{(\mathbf{x}_j, 1)\}_{j=1}^{N_1}$ for $t_j \in [43, 49]$ where $N_0 = N_1 = 6,687$ (balanced stratified subsampling). An XGBoost classifier $f_{\text{adv}}(\mathbf{x}) \in [0, 1]$ is trained under 5-fold Stratified Cross-Validation:
$$\mathcal{L}_{\text{adv}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log f_{\text{adv}}(\mathbf{x}_i) + (1 - y_i) \log(1 - f_{\text{adv}}(\mathbf{x}_i)) \right]$$

### 3.2. Attribution Extraction (Gain vs. SHAP)
1. **Gain Share**:
   $$\text{GainShare}_k = \frac{\sum_{\text{trees}} \text{Gain}(f_k)}{\sum_{j=1}^{165} \sum_{\text{trees}} \text{Gain}(f_j)} \times 100\%$$
2. **TreeExplainer SHAP Share**:
   $$\text{SHAPShare}_k = \frac{\frac{1}{N}\sum_{i=1}^N |\phi_k(\mathbf{x}_i)|}{\sum_{j=1}^{165} \frac{1}{N}\sum_{i=1}^N |\phi_j(\mathbf{x}_i)|} \times 100\%$$

### 3.3. De Facto Timestamp Leak Signature
For each feature $x_k$:
1. **Spearman Rank Correlation**: $\rho_s(x_k, \text{Time step})$.
2. **Within-Step vs. Between-Step Variance Decomposition**:
   $$\bar{\sigma}^2_{\text{within}} = \frac{1}{T} \sum_{t=1}^T \text{Var}(x_k \mid \text{step}=t)$$
   $$\sigma^2_{\text{between}} = \text{Var}_t(\mathbb{E}[x_k \mid \text{step}=t])$$
   $$\eta^2 = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{total}}} = \frac{\sigma^2_{\text{between}}}{\bar{\sigma}^2_{\text{within}} + \sigma^2_{\text{between}}}$$

**Classification Decision Boundary:**
- **Mechanical Time Proxy (Leak)**: $\eta^2 \ge 0.90 \land |\rho_s| \ge 0.90$ with $\bar{\sigma}^2_{\text{within}} \approx 0$.
- **Strong Secular Drift**: $\eta^2 \ge 0.60 \lor |\rho_s| \ge 0.60$ with substantial intra-step variance.
- **Behavioral Covariate Drift**: $\eta^2 < 0.60 \land |\rho_s| < 0.60$ (domain shift driven by non-monotonic distribution divergence).

---

## 4. Decision Rule & Verdict Criteria

The notebook's final execution cell programmatically assigns one of three verdicts:

| Verdict Code | Title | Mathematical Condition | Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| `(a) CONFIRMED` | **Distributed OOD Regime Shift** | $\text{AUC}_{\text{DropTop20}} \ge 0.90 \land N_{\text{TimeProxies}} = 0 \land \text{Top3Share} < 70\%$ | Adversarial separability is broadly distributed across dozens of features; the 43–49 collapse is an irreducible regime shift requiring continuous retraining / active domain adaptation. |
| `(b) PARTIALLY CONFIRMED` | **Multi-Feature Covariate Shift with Specific Concentration** | $\text{AUC}_{\text{DropTop3}} \ge 0.85 \land (\text{Top3Share} \ge 50\% \lor \text{LocalsOnly AUC} < 0.80)$ | Adversarial AUC remains high, but drift is heavily concentrated within the 72 `Aggregate_feature_*` block rather than the 93 local transaction metrics. |
| `(c) NOT CONFIRMED` | **Mechanical Leak / Trivial Single-Feature Concentration** | $\text{AUC}_{\text{DropTop3}} < 0.80 \lor N_{\text{TimeProxies}} \ge 5 \lor \text{Top3Share} \ge 70\%$ | Separability collapses upon removing 1–3 top features; the $\text{AUC}=1.0000$ result is an artifact of cumulative time leakage. |

---

## 5. Artifact Inventory

```text
results/ood_diagnosis/
├── adversarial_importances.csv   # Gain & SHAP importances and percentage shares for all 165 features
├── leak_audit.csv                # Spearman rho, eta² variance ratio, and leak classification for Top 20
├── concentration_metrics.json    # Cumulative Top 1, 3, 5, 10, 20 shares and Gini index
├── leave_one_out_results.csv     # 5-fold AUC across 9 progressive removal configurations
├── model_overlap.csv             # 4-quadrant cross-check between fraud importance and drift importance
├── per_step_adversarial_auc.csv  # 5-fold AUC for individual test steps 35 through 49 vs train 1–34
├── verdict.json                  # Machine-readable diagnostic verdict payload
├── verdict.md                    # Rendered markdown verdict summarizing empirical conclusions
├── checks.csv                    # Automated verification suite (30+ assertions)
├── digest.txt                    # Executive summary run digest
├── figures/
│   ├── adversarial_top20_importance.png  # Barplot of Top 20 features by Gain and SHAP share
│   ├── leak_feature_signatures.png       # Scatterplot of eta² vs Spearman rho with bubble size by gain
│   ├── leave_one_out_auc.png             # Barplot of adversarial AUC across feature removal suites
│   ├── fraud_vs_adversarial_overlap.png  # Log-scale scatterplot of Fraud Gain vs Drift Gain
│   └── per_step_adversarial_auc.png      # Temporal line plot of adversarial AUC across steps 35–49
└── permutation_check/            # Phase 7b Follow-up: Permutation Importance & Correlation Audit
    ├── permutation_importances.csv       # Out-of-fold permutation AUC drops for all 72 aggregate features
    ├── importance_method_comparison.csv  # Direct Gain vs SHAP vs Permutation comparison & discrepancy flags
    ├── correlation_matrix.csv            # 72x72 pairwise Pearson correlation matrix on adversarial dataset
    ├── grouped_permutation_results.csv   # Joint block permutation AUC drops (Focal vs Correlated vs Random)
    ├── checks.csv                        # Sanity checks for permutation baseline reproduction & bounds
    ├── digest.txt                        # Executive summary digest for permutation follow-up
    ├── verdict.json                      # Programmatic verdict payload for permutation check
    ├── verdict.md                        # Rendered verdict markdown
    └── figures/
        ├── permutation_importances_top25.png      # Top 25 aggregate features by causal AUC drop
        ├── permutation_vs_gain_shap_comparison.png# Tri-method comparison across top 15 features
        ├── attribution_concentration_curves.png   # Lorenz cumulative attribution share curves
        ├── correlation_heatmap_top_features.png   # Correlation heatmap of top 20 drifting aggregates
        └── grouped_permutation_auc_drops.png      # Barplot of grouped block permutation AUC drops
```

---

## 6. Empirical Results & Findings

The diagnostic audit notebook `notebooks/08_ood_diagnosis.ipynb` was executed on Google Colab, exporting validated artifacts to `results/ood_diagnosis/`.

### 6.1. Adversarial Separability & Feature Attribution
The 5-fold cross-validated adversarial classifier distinguishing training transactions (steps 1–34, $N_0 = 6,687$) from late drift transactions (steps 43–49, $N_1 = 6,687$) achieved:
$$\text{Mean Adversarial AUC} = \mathbf{1.0000 \pm 0.0000}$$

Feature attribution displays extreme concentration in the top aggregate features when all 165 features are available:
- **Top 1 Feature (`Aggregate_feature_10`)**: 36.54% Gain Share, 14.29% SHAP Share
- **Top 3 Features (`Aggregate_feature_10, 43, 8`)**: **95.16%** Gain Share, **80.32%** SHAP Share
- **Top 10 Features**: 98.38% Gain Share, 93.19% SHAP Share
- **Top 20 Features**: 99.03% Gain Share, 94.78% SHAP Share
- **Gini Index**: Gain $\text{Gini} = 0.9770$, SHAP $\text{Gini} = 0.9688$

### 6.2. Leak-Feature Signature Audit (Top 20 Drifting Features)

| Rank | Feature | Group | Gain Share (%) | SHAP Share (%) | Spearman $\rho_s$ | Within-Step Var $\bar{\sigma}^2_w$ | $\eta^2$ Ratio | Leak Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Aggregate_feature_10` | Aggregate | 36.54% | 14.29% | +0.9965 | 0.0152 | 0.8561 | Strong Trending Drift |
| 2 | `Aggregate_feature_43` | Aggregate | 30.47% | 33.48% | +0.9949 | 0.1670 | 0.7237 | Strong Trending Drift |
| 3 | `Aggregate_feature_8` | Aggregate | 28.15% | 32.56% | +0.9976 | 0.0118 | 0.8589 | Strong Trending Drift |
| 4 | `Aggregate_feature_7` | Aggregate | 2.40% | 9.52% | +0.9950 | 0.0333 | 0.8409 | Strong Trending Drift |
| 5 | `Aggregate_feature_44` | Aggregate | 0.20% | 0.44% | +0.9019 | 0.3674 | 0.5434 | Strong Trending Drift |
| 6 | `Aggregate_feature_46` | Aggregate | 0.20% | 1.87% | +0.9704 | 0.2062 | 0.6888 | Strong Trending Drift |
| 7 | `Local_feature_2` | Local | 0.14% | 0.37% | +0.6309 | 0.9076 | 0.0418 | Strong Trending Drift |
| 8 | `Aggregate_feature_55` | Aggregate | 0.11% | 0.23% | +0.5282 | 0.7320 | 0.2579 | Behavioral Covariate Drift |
| 9 | `Local_feature_88` | Local | 0.09% | 0.34% | +0.0198 | 0.9877 | 0.0086 | Behavioral Covariate Drift |
| 10 | `Aggregate_feature_31` | Aggregate | 0.08% | 0.10% | +0.0703 | 0.8292 | 0.0168 | Behavioral Covariate Drift |
| 11 | `Aggregate_feature_56` | Aggregate | 0.08% | 0.15% | +0.4155 | 0.7219 | 0.1858 | Behavioral Covariate Drift |
| 12 | `Aggregate_feature_37` | Aggregate | 0.08% | 0.07% | -0.0943 | 0.9907 | 0.0125 | Behavioral Covariate Drift |
| 13 | `Local_feature_53` | Local | 0.07% | 0.33% | -0.0021 | 0.9704 | 0.0213 | Behavioral Covariate Drift |
| 14 | `Aggregate_feature_50` | Aggregate | 0.07% | 0.16% | +0.4205 | 0.8304 | 0.1685 | Behavioral Covariate Drift |
| 15 | `Local_feature_49` | Local | 0.07% | 0.01% | -0.0477 | 1.0457 | 0.0308 | Behavioral Covariate Drift |
| 16 | `Local_feature_9` | Local | 0.06% | 0.05% | -0.0811 | 1.0102 | 0.0219 | Behavioral Covariate Drift |
| 17 | `Aggregate_feature_49` | Aggregate | 0.06% | 0.77% | +0.5330 | 1.0954 | 0.0342 | Behavioral Covariate Drift |
| 18 | `Aggregate_feature_9` | Aggregate | 0.06% | 0.02% | -0.0967 | 0.9736 | 0.0361 | Behavioral Covariate Drift |
| 19 | `Aggregate_feature_67` | Aggregate | 0.05% | 0.01% | +0.0185 | 1.3296 | 0.0039 | Behavioral Covariate Drift |
| 20 | `Local_feature_89` | Local | 0.05% | 0.02% | -0.1200 | 0.9435 | 0.0337 | Behavioral Covariate Drift |

*Audit Finding:* **0 out of 20 features** met the mechanical leak signature ($\eta^2 \ge 0.90 \land \bar{\sigma}^2_w \approx 0$). Although top aggregate features exhibit high monotonic correlation with time ($\rho_s > 0.99$), they maintain substantial within-step variance ($\bar{\sigma}^2_w > 0.01$), representing real macroeconomic secular growth rather than unnormalized sequence counters.

### 6.3. Leave-One-Out (LOO) Robustness Benchmark
To test whether separability collapses when the dominant trending features are excluded, 9 progressive removal configurations were tested:

| Configuration | Features Remaining | 5-Fold Adversarial AUC | Std Dev | $\Delta$ AUC vs Full |
| :--- | :--- | :--- | :--- | :--- |
| **All 165 Features (Full)** | 165 | **1.0000** | $\pm 0.0000$ | 0.0000 |
| **Drop Top 1** | 164 | **1.0000** | $\pm 0.0000$ | 0.0000 |
| **Drop Top 3** | 162 | **1.0000** | $\pm 0.0000$ | -0.0000 |
| **Drop Top 5** | 160 | **1.0000** | $\pm 0.0000$ | -0.0000 |
| **Drop Top 10** | 155 | **0.9909** | $\pm 0.0009$ | -0.0091 |
| **Drop Top 20** | 145 | **0.9904** | $\pm 0.0015$ | -0.0096 |
| **Drop Top 15 KS-Drifted** | 150 | **0.9834** | $\pm 0.0021$ | -0.0166 |
| **Local Features Only (93 feats)** | 93 | **0.9885** | $\pm 0.0026$ | -0.0115 |
| **Aggregate Features Only (72 feats)**| 72 | **1.0000** | $\pm 0.0000$ | 0.0000 |

*Decisive Discovery:* Adversarial separability **does not collapse** upon dropping the top features. Even when dropping the Top 20 drift features or stripping away all 72 aggregate features entirely (Local Features Only), adversarial AUC remains **0.9885 – 0.9904**.

### 6.4. Fraud Classifier vs. Drift Classifier Overlap

| Dimension | Fraud Classifier (Phase 2 Frozen XGBoost) | Drift Classifier (Adversarial XGBoost) |
| :--- | :--- | :--- |
| **Primary Domain** | Dominated by **Local Features** (61.5% SHAP mass) | Dominated by **Aggregate Features** (99.0% Gain mass) |
| **Top 1 Feature** | `Local_feature_53` (8.61% Gain share) | `Aggregate_feature_10` (36.54% Gain share) |
| **Top 3 Features** | `Local_feature_53` (8.6%), `Local_feature_46` (7.9%), `Local_feature_5` (6.4%) | `Aggregate_feature_10` (36.5%), `Aggregate_feature_43` (30.5%), `Aggregate_feature_8` (28.2%) |
| **Top 20 Overlap** | Only **4 of 20** features overlap (`Local_feature_53`, `Aggregate_feature_7`, `Aggregate_feature_43`, `Aggregate_feature_10`) | — |

### 6.5. Per-Step Granular Adversarial AUC vs. Train (1–34)

| Time Step $t$ | Evaluation Subgroup | Labeled $N$ | Adversarial AUC (Mean $\pm$ Std) | Illicit Prevalence |
| :--- | :--- | :--- | :--- | :--- |
| **Step 35** | Early Test (Stationary) | 1,341 | $1.0000 \pm 0.0000$ | 9.47% |
| **Step 36** | Early Test (Stationary) | 1,708 | $1.0000 \pm 0.0000$ | 9.25% |
| **Step 37** | Early Test (Stationary) | 498 | $1.0000 \pm 0.0000$ | 10.84% |
| **Step 38** | Early Test (Stationary) | 756 | $1.0000 \pm 0.0000$ | 10.45% |
| **Step 39** | Early Test (Stationary) | 1,183 | $1.0000 \pm 0.0000$ | 9.04% |
| **Step 40** | Early Test (Stationary) | 1,211 | $1.0000 \pm 0.0000$ | 9.50% |
| **Step 41** | Early Test (Stationary) | 1,132 | $1.0000 \pm 0.0000$ | 8.83% |
| **Step 42** | Early Test (Stationary) | 2,154 | $1.0000 \pm 0.0000$ | 8.26% |
| **Step 43** | Late Test (Drift Onset) | 1,370 | $0.9996 \pm 0.0007$ | **2.55%** |
| **Step 44** | Late Test (Drift Window) | 1,591 | $1.0000 \pm 0.0000$ | 2.58% |
| **Step 45** | Late Test (Drift Window) | 1,221 | $0.9996 \pm 0.0008$ | 2.54% |
| **Step 46** | Late Test (Drift Window) | 712 | $1.0000 \pm 0.0000$ | 2.39% |
| **Step 47** | Late Test (Drift Window) | 846 | $1.0000 \pm 0.0000$ | 2.60% |
| **Step 48** | Late Test (Drift Window) | 471 | $1.0000 \pm 0.0000$ | 2.55% |
| **Step 49** | Late Test (Drift Window) | 476 | $1.0000 \pm 0.0000$ | 2.52% |

### 6.6. Permutation Importance & Collinear Redundancy Audit (Notebook 08b)

To resolve whether the concentration in `Aggregate_feature_10, 43, 8` was an artifact of greedy tree splitting or a genuine localized driver, `notebooks/08b_ood_permutation_check.ipynb` executed an out-of-fold permutation importance and grouped block ablation suite (5 folds $\times$ 10 shuffle repeats = 50 evaluations per feature) restricted to the 72 aggregate features.

#### Tri-Method Attribution Comparison (Top 15 Aggregate Features)

| Perm Rank | Gain Rank | SHAP Rank | Feature | Mean AUC Drop ($\mu_{\Delta}$) | Perm Share (%) | Gain Share (%) | SHAP Share (%) | Discrepancy Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2 | 2 | `Aggregate_feature_43` | **0.001068** $\pm 0.00049$ | **64.70%** | 26.96% | 30.58% | `CONCORDANT_DOMINANT` |
| **2** | 3 | 1 | `Aggregate_feature_8` | **0.000579** $\pm 0.00056$ | **35.07%** | 26.78% | 33.54% | `CONCORDANT_DOMINANT` |
| **3** | 1 | 3 | `Aggregate_feature_10` | 0.000002 $\pm 0.00000$ | 0.11% | **37.30%** | 14.37% | `OVERCREDITED_BY_SPLIT` |
| **4** | 7 | 9 | `Aggregate_feature_44` | $5.93 \times 10^{-7}$ | 0.04% | 0.35% | 0.75% | `CONCORDANT_MINOR` |
| **5** | 4 | 4 | `Aggregate_feature_7` | $4.58 \times 10^{-7}$ | 0.03% | 4.28% | 9.57% | `CONCORDANT_DOMINANT` |
| **6** | 11 | 17 | `Aggregate_feature_50` | $2.46 \times 10^{-7}$ | 0.01% | 0.08% | 0.24% | `UNDERCREDITED_BY_SPLIT` |
| **7** | 14 | 6 | `Aggregate_feature_52` | $1.68 \times 10^{-7}$ | 0.01% | 0.06% | 1.06% | `UNDERCREDITED_BY_SPLIT` |
| **8** | 16 | 8 | `Aggregate_feature_49` | $1.12 \times 10^{-7}$ | 0.01% | 0.05% | 0.88% | `UNDERCREDITED_BY_SPLIT` |
| **9** | 9 | 14 | `Aggregate_feature_55` | $1.12 \times 10^{-7}$ | 0.01% | 0.09% | 0.27% | `CONCORDANT_MINOR` |
| **10** | 15 | 13 | `Aggregate_feature_56` | $1.01 \times 10^{-7}$ | 0.01% | 0.06% | 0.31% | `UNDERCREDITED_BY_SPLIT` |
| **11** | 20 | 7 | `Aggregate_feature_58` | $8.94 \times 10^{-8}$ | 0.01% | 0.04% | 0.94% | `CONCORDANT_MINOR` |
| **12** | 8 | 25 | `Aggregate_feature_37` | $7.83 \times 10^{-8}$ | 0.00% | 0.10% | 0.08% | `CONCORDANT_MINOR` |
| **13** | 5 | 5 | `Aggregate_feature_46` | $1.12 \times 10^{-8}$ | 0.00% | 2.19% | 4.07% | `OVERCREDITED_BY_SPLIT` |
| **14** | 6 | 23 | `Aggregate_feature_45` | $\approx 0$ | 0.00% | 1.05% | 0.11% | `CONCORDANT_MINOR` |
| **15** | 31 | 22 | `Aggregate_feature_64` | $\approx 0$ | 0.00% | 0.01% | 0.16% | `CONCORDANT_MINOR` |

*Attribution Concentration Summary:*
- **Top 3 Features Share**: Gain **91.03%** $\to$ SHAP **78.49%** $\to$ Permutation **99.88%**
- **Gini Concentration**: Gain **0.9510** $\to$ SHAP **0.9322** $\to$ Permutation **0.9761**
- **Spearman Rank Correlation**: Gain vs. Permutation $\rho = 0.8883$ ($p = 2.1 \times 10^{-24}$); SHAP vs. Permutation $\rho = 0.8812$.

#### Grouped Block Permutation Ablation Results

| Feature Block Configuration | Features in Block | Mean Adversarial AUC Drop | Std Dev | Permuted 5-Fold AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Unpermuted 72 Aggregates)** | All 72 | 0.00000 | $\pm 0.0000$ | **$1.0000 \pm 0.0000$** |
| **Block A: Focal Triad $\{10, 43, 8\}$** | `Aggregate_feature_10, 43, 8` | **0.30550** | $\pm 0.0375$ | **$0.6945 \pm 0.0375$** |
| **Block B: Top Correlated Sister Triad** | `Aggregate_feature_7, 46, 44` | **0.00000** | $\pm 0.0000$ | **$1.0000 \pm 0.0000$** |
| **Block C: Top-3 Permutation Triad** | `Aggregate_feature_43, 8, 10` | **0.30227** | $\pm 0.0376$ | **$0.6977 \pm 0.0376$** |
| **Block D: Random Triads (N=15 Sample Mean)**| 15 sampled random triplets | **0.00000** | $\pm 0.0000$ | **$1.0000 \pm 0.0000$** |
| **Block E: Combined Top 6 (Focal + Correlated)**| `Aggregate_feature_10, 43, 8, 7, 46, 44` | **0.47254** | $\pm 0.0099$ | **$0.5275 \pm 0.0099$** |

*Definitive Grouped Permutation Finding:*
Shuffling the Focal Triad $\{10, 43, 8\}$ degrades adversarial AUC by **$\Delta \text{AUC} = 0.3055$** (down to 0.6945), whereas shuffling non-focal sister features ($\{7, 46, 44\}$) or random triplets produces **$\Delta \text{AUC} = 0.0000$**. When the top 6 features are permuted jointly, separability collapses completely to **$\text{AUC} = 0.5275$** (near chance).

---

## 7. Scientific Synthesis & Root-Cause Resolution

### 7.1. Definitive Verdict: Two Decoupled Drift Mechanisms

The permutation check notebook programmatically returned:
$$\mathbf{\text{Verdict: (b) CONCENTRATION CONFIRMED — Genuine Localized Aggregate Drift}}$$

This result definitively revises the initial "GBDT splitting artifact" conjecture:
1. **Aggregate Feature Subspace is Genuinely Concentrated**:
   Within the 72 aggregate features, causal attribution is not diffuse. Rather, it is genuinely localized in **`Aggregate_feature_43, 8, 10`** (accounting for **99.88%** of causal permutation loss and driving a 0.3055 AUC collapse in block permutation). Tree splits over-credited `Aggregate_feature_10` (37.3% Gain vs 0.11% single Permutation) because of greedy race conditions, but collectively the triad carries nearly all aggregate drift information.
2. **Two Decoupled Drift Mechanisms Operating Simultaneously**:
   The reason overall dataset drift cannot be solved by dropping 1–3 features is that drift operates via **two distinct, decoupled mechanisms**:
   - **Mechanism 1 (Macro Aggregate Secular Drift)**: Extreme, localized secular drift in graph-aggregated transaction volumes and neighbor counters (`Aggregate_feature_43, 8, 10`).
   - **Mechanism 2 (Micro Local Behavioral Drift)**: Broadly distributed behavioral drift across the 93 local transaction metrics. Local features alone achieve **$\text{AUC} = 0.9885 \pm 0.0026$** completely independently of all 72 aggregate features.
3. **No Mechanical Timestamp Leak**:
   Zero features have zero within-step variance ($\bar{\sigma}^2_w > 0$). Both aggregate and local shifts reflect real network evolution across Bitcoin transaction history.

### 7.2. Why Performance Collapses Specifically at Step 43
The per-step adversarial audit reveals that while covariate drift progresses continuously from step 35 to 49 ($\text{AUC} \approx 1.0$), the models (XGBoost, GraphSAGE, RGCN, HGT) maintain high PR-AUC (0.73–0.92) across steps 35–42, but experience a cliff-edge collapse at step 43 (0.038–0.055).

This proves that **covariate drift alone does not cause the collapse**; rather, the collapse is triggered by the interaction between **continuous covariate drift** and a **violent 4x drop in illicit class prevalence** at step 43 ($9.16\% \to 2.53\%$). Because static models fit thresholds $\tau^*$ on high-prevalence training data (steps 1–34), their uncalibrated risk scores generate massive false positives on the sparse late-regime distribution.

---

## 8. Strategic Directives for Phase 8 (MLOps) & Phase 9 (Serving)

1. **Production Engine Finalization**: Finalize **XGBoost Optimized (Frozen, PR-AUC 0.8013)** as the primary transaction scoring engine.
2. **Dual-Layer Covariate Drift Monitoring**:
   - **Aggregate Layer (Univariate / KS & PSI)**: Track primary aggregate drivers `Aggregate_feature_43, 8, 10` and sister cluster `Aggregate_feature_7, 46, 44`.
   - **Local Layer (Multivariate / Centroid PSI)**: Track key fraud drivers `Local_feature_53, 46, 5` to detect behavioral drift.
3. **Dynamic Prior & Threshold Adaptation**: Provide real-time Bayesian prior adjustment / rolling threshold calibration ($\tau_t$) based on sliding-window background prevalence estimation.
4. **Rolling Retraining Triggers**: Implement an MLOps pipeline trigger that retrains XGBoost over a rolling temporal window ($W=20$ steps) whenever adversarial separability or PSI crosses critical drift thresholds.
