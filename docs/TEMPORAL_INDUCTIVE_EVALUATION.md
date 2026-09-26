# Temporal & Inductive Evaluation — Phase 5 Record & Methodology

> **Phase 5 Record.** This document records the methodology, protocols, and experimental analysis for the **Temporal & Inductive Evaluation** in **BitcoinGraphGuard**.
> It evaluates models under temporal degradation, retraining regimes, and inductive generalization criteria across time steps 35–49.
>
> Executable implementation: `notebooks/05_temporal_inductive_evaluation.ipynb` 

---

## 1. Executive Summary

Phase 5 evaluates whether the heterogeneous graph representation's primary value is inductive generalization across continuous temporal shifts, investigating four distinct hypotheses related to temporal deterioration and adaptation.

- **Objective:** Evaluate XGBoost, GraphSAGE, and HeteroRGCN on temporal degradation, rolling/expanding window retraining stability, and cross-temporal structural generalization.
- **Protocol:** Analyzes non-stationary test windows $T \in [35, 49]$ highlighting the dramatic regime drift at $T \ge 43$ (illicit prevalence collapse from 9.1% to 2.5%).

---

## 2. Core Research Questions (Q1-Q4)

### Q1: Temporal Degradation
Tracking metrics iteratively per-step uncovers structural weaknesses over the drift window. As verified in previous phases, all models severely degrade step 43 onwards.
- **Hypothesis:** Temporal distribution shift fundamentally alters transaction patterns in steps 43-49, rendering static historical models obsolete.

### Q2: Static vs Continuous Retraining Regimes
Evaluates whether degradation is curable simply by updating the model weights over time. 
- **Regimes tested:**
  - **Static:** Train on $T_{1-34}$, predict $T_t$.
  - **Expanding:** Train on $T_{1..t-1}$, predict $T_t$.
  - **Rolling Window:** Train on $T_{t-W..t-1}$ (where $W \in \{10, 20\}$), predict $T_t$.

### Q3: Inductive Generalization (Structural Bridges)
Classifies nodes functionally:
- **Transductive-like (Seen Wallets):** Transactions interacting with highly recurrent address endpoints seen in $T \le 34$.
- **Inductive (Unseen Wallets):** Transactions engaging exclusively with newly created addresses.
- **Singletons:** Address-less nodes.

### Q4: Calibration (Threshold Instability)
Did models functionally forget how to separate distributions (AUC collapse), or simply become uncalibrated on raw probabilities? 
Benchmarking:
- **Static Validation $\tau^*$:** Fixed from step 25-34 validation.
- **Adaptive $\tau_t$**: Derived from historical window bounds.
- **Oracle $\tau_t^*$**: The post-hoc best threshold derived on the actual test step $t$ distribution.

---

## 3. Results Overview & Artifact Inventory

The full Phase 5 notebook generated corresponding evaluation tables, metrics JSONs, arrays, and visualizations assessing calibration vs discriminative deterioration across continuous learning loops. (See complete numbers in `temporal_inductive/`).

```text
temporal_inductive/
├── per_step_metrics.csv       # Output metrics per step T (35-49)
├── static_vs_expanding.csv    # Retraining evaluation results
├── rolling_window.csv         # Window size comparison results
├── inductive_metrics.csv      # Unseen/Seen subpopulation breakdown
├── historical_path_metrics.csv# Structural flow 2-step metrics
├── threshold_analysis.csv     # Oracle vs Static threshold evaluation
├── checks.csv                 # Automated validation suite results
├── metrics.json               # Exported telemetry
├── temporal_inductive_digest.txt # Quick summary
└── figures/                   # 6 Core plots for temporal evaluation
```

## 4. Status Check

**Phase 5 Complete.** The models have now been baselined against real-world incremental learning and zero-shot temporal scenarios. The environment is clear to move to Phase 6 Explainability and Phase 7 MLOps.
