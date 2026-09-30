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

## 5. Artifact Inventory (`results/ood_diagnosis/`)

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
└── figures/
    ├── adversarial_top20_importance.png  # Barplot of Top 20 features by Gain and SHAP share
    ├── leak_feature_signatures.png       # Scatterplot of eta² vs Spearman rho with bubble size by gain
    ├── leave_one_out_auc.png             # Barplot of adversarial AUC across feature removal suites
    ├── fraud_vs_adversarial_overlap.png  # Log-scale scatterplot of Fraud Gain vs Drift Gain
    └── per_step_adversarial_auc.png      # Temporal line plot of adversarial AUC across steps 35–49
```
