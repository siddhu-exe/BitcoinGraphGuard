# temporal_inductive — Phase 5 artifacts

Produced by `notebooks/05_temporal_inductive_evaluation.ipynb` (run on Google Colab / Kaggle).
Random seed 42; frozen 35-49 benchmark retained at XGBoost 0.8013 / GraphSAGE 0.6209 / HeteroRGCN 0.4682.

| File | Contents |
| :--- | :--- |
| `per_step_metrics.csv` | Frozen XGBoost/GraphSAGE/RGCN metrics for every step 35-49 |
| `per_step_label_counts.csv` | Labelled, illicit, licit counts and prevalence per step |
| `drift_metrics.csv` | Per-step graph, address-activity, connectivity and feature-drift diagnostics |
| `drift_performance_association.csv` | Spearman association of each drift metric with per-step PR-AUC |
| `static_vs_expanding.csv` | Per-step static vs expanding-window RGCN metrics |
| `expanding_windows.csv` | Exact training window and positive count for every expanding refit |
| `rolling_window.csv` | Rolling-window (W=20) variant, if executed |
| `inductive_metrics.csv` | RGCN performance by seen/unseen/no-address context |
| `inductive_diagnostics_per_step.csv` | Expanding-regime per-step breakdown by seen/unseen context |
| `inductive_context_by_step.csv` | Per-step address-context composition |
| `historical_path_metrics.csv` | Performance split by historical T->A->T path availability |
| `historical_path_availability.csv` | Per-step historical-path counts |
| `threshold_analysis.csv` | Frozen vs adaptive (vs oracle) thresholds per step |
| `model_comparison.csv` | Frozen benchmark plus Phase 5 regimes, for 35-49 / 35-42 / 43-49 |
| `temporal_audit.csv` | Training window and graph cut-off for every model evaluated at every step |
| `rgcn_temporal_predictions.csv` | Per-transaction scores for every regime |
| `reproducibility.json` | Seed, policies, versions, hardware, checkpoint hash |
| `checks.csv` | Automated validation checks (all must pass) |
| `figures/` | Question-driven figures (per-step PR-AUC, prevalence, drift, regimes, inductive, thresholds) |

Frozen benchmark references are never overwritten by adaptive results; they answer different questions.
