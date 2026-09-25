# Plan

Each phase is annotated with its **execution environment** and, where ML is involved, the
notebook that implements it. All ML execution — EDA, feature engineering, training,
evaluation, explainability — runs in notebooks on **Kaggle / Google Colab**; the laptop
handles engineering, documentation, and MLOps only. See `ARCHITECTURE.md`.

## Notebook Map

| Notebook | Phase |
| :--- | :--- |
| `notebooks/01_eda.ipynb` | Phase 1 — EDA |
| `notebooks/02_xgboost.ipynb` | Phase 2 — XGBoost baseline |
| `notebooks/03_graphsage.ipynb` | Phase 2 — GraphSAGE baseline |
| `notebooks/04_heterogeneous_gnn.ipynb` | Phase 3 — RGCN / HGT |
| `notebooks/05_temporal_inductive_evaluation.ipynb` | Phase 4 — Temporal & inductive evaluation |
| `notebooks/06_explainability.ipynb` | Phase 5 — Explainability |
| `notebooks/07_final_evaluation.ipynb` | Phase 9 — Final evaluation |

One notebook per phase; do not create a notebook per small step.

## Phase 1 — Data & Foundation

*Environment: Laptop for repo/environment setup; notebook `01_eda.ipynb` on Kaggle/Colab for EDA.*

* [ ] Create project structure
* [ ] Configure Python environment (in progress — `requirements.txt` created, user-managed install)
* [ ] Install and pin dependencies
* [x] Download and verify Elliptic++ dataset
* [x] Inspect dataset structure and metadata
* [x] Validate transaction and actor/wallet data
* [x] Identify missing values, duplicates, invalid relationships, and label distribution
* [x] Analyze temporal structure (notebook, 2026-09-25; see `docs/EDA.md`)
* [x] Analyze class imbalance (notebook, 2026-09-25; see `docs/EDA.md`)
* [x] Perform graph-level EDA (notebook, 2026-09-25; see `docs/EDA.md`)
* [x] Author the Phase 1 EDA notebook (`notebooks/01_eda.ipynb`)
* [x] Execute `notebooks/01_eda.ipynb` on Colab and record the results in `docs/EDA.md`
      (run 2026-09-25, artifacts in `eda/`)
* [ ] Configure DVC
* [ ] Establish reproducibility and configuration system

## Phase 2 — Baselines

*Environment: Kaggle/Colab notebooks `02_xgboost.ipynb` and `03_graphsage.ipynb`; laptop only for MLflow configuration.*

*Status: executed 2026-09-25 on Colab in two passes and awaiting review sign-off before GraphSAGE
is designed. The first pass is the recorded 500-tree baseline; the second lifts the budget to 1500
trees (582 kept, so the cap no longer binds) and adds a controlled 20-trial randomised search, a
`has_addresses` ablation and one label-free redundancy ablation. `notebooks/02_xgboost.ipynb`
covers the leakage audit, the prevalence baseline, Logistic Regression and XGBoost on the 165
non-domain transaction features, evaluated fit 1–24 / validation 25–34 / refit 1–34 / test 35–49.
It deliberately uses no graph statistics (degrees, components, hub ranks), because the EDA computed
those over the full transductive graph including future steps; graph features belong to the GNN
phases and must be step-bounded there. GraphSAGE stays in `03_graphsage.ipynb`.*

Measured on the test period 35–49 (16,670 labeled, 1,083 illicit): prevalence baseline PR-AUC
0.0650, Logistic Regression 0.2917, XGBoost baseline 0.8007 (ROC-AUC 0.9317), optimised XGBoost
0.8013 (ROC-AUC 0.9281). Retuning therefore bought +0.0006 PR-AUC while ROC-AUC and F1 fell: the
tree budget was not the constraint. Sub-window PR-AUC is 0.9215 on 35–42 but 0.0427 on 43–49, where
prevalence drops to 2.53%. `has_addresses` was dropped after scoring +0.000933 against a
pre-registered +0.005 margin. Record: `docs/XGBOOST.md`.

* [x] Build the Phase 2 dataset: 165 non-domain transaction features (`has_addresses` tested and
      dropped by the pre-registered ablation)
* [x] Run the feature leakage audit (`xgboost/leakage_audit.csv`, `xgboost/checks.csv`)
* [x] Train the prevalence baseline, Logistic Regression and XGBoost baseline (remote)
* [x] Establish fraud-detection evaluation metrics (PR-AUC primary; ROC-AUC, P, R, F1, confusion)
* [x] Implement the temporal evaluation protocol and sub-window reporting
* [x] Revise the notebook for the binding tree cap (1500 trees, patience 100), randomised
      search, `has_addresses` ablation and redundancy ablation
* [x] Re-execute the revised notebook on Colab/Kaggle (2026-09-25) and record the optimised result
      alongside the 0.8007 baseline rather than replacing it
* [ ] Compare XGBoost and GraphSAGE
* [ ] Configure MLflow experiment tracking
* [ ] Engineer step-bounded transaction-level graph features (Phase 3, not here)

## Phase 3 — Heterogeneous Graph Deep Learning

*Environment: Kaggle/Colab notebook `04_heterogeneous_gnn.ipynb` (graph construction and training).*

* [ ] Construct heterogeneous transaction + actor/wallet graph
* [ ] Define node and edge types
* [ ] Build graph preprocessing pipeline
* [ ] Implement RGCN (remote)
* [ ] Evaluate HGT as an alternative if justified (remote)
* [ ] Tune model using validation data (remote)
* [ ] Handle class imbalance
* [ ] Compare against XGBoost and GraphSAGE
* [ ] Run ablation studies (remote)
* [ ] Analyze model errors

## Phase 4 — Temporal & Inductive Evaluation

*Environment: Kaggle/Colab notebook `05_temporal_inductive_evaluation.ipynb`.*

* [ ] Train using historical time steps
* [ ] Evaluate on later unseen time steps
* [ ] Measure temporal performance degradation
* [ ] Evaluate unseen transactions/wallets
* [ ] Measure inductive generalization
* [ ] Analyze performance across individual time steps
* [ ] Document temporal failure modes

## Phase 5 — Explainability

*Environment: Kaggle/Colab notebook `06_explainability.ipynb` (GNNExplainer).*

* [ ] Select representative fraud predictions
* [ ] Select false-positive cases
* [ ] Select false-negative cases
* [ ] Apply GNNExplainer or equivalent method
* [ ] Identify influential nodes and edges
* [ ] Visualize important graph structures
* [ ] Document explainability limitations

## Phase 6 — MLOps & Monitoring

*Environment: Laptop.*

* [ ] Track experiments with MLflow
* [ ] Version datasets and pipelines with DVC
* [ ] Track model artifacts
* [ ] Define model-quality monitoring
* [ ] Define graph-structural drift metrics
* [ ] Monitor degree-distribution changes
* [ ] Monitor graph/community/subgraph changes
* [ ] Monitor temporal performance degradation
* [ ] Define retraining thresholds
* [ ] Implement retraining trigger logic

## Phase 7 — ML System & Deployment

*Environment: Laptop.*

* [ ] Design production inference architecture
* [ ] Implement model loading and inference service
* [ ] Build FastAPI endpoints
* [ ] Add request validation
* [ ] Add logging and error handling
* [ ] Dockerize inference service
* [ ] Create CI/CD pipeline with GitHub Actions
* [ ] Add automated tests
* [ ] Add model validation checks
* [ ] Deploy inference service

## Phase 8 — Monitoring & Dashboard

*Environment: Laptop.*

* [ ] Build model monitoring dashboard
* [ ] Display fraud predictions and model metrics
* [ ] Display temporal performance
* [ ] Display graph drift
* [ ] Display model drift
* [ ] Display retraining status
* [ ] Add representative explainability visualizations
* [ ] Validate dashboard against tracked experiments

## Phase 9 — Finalization

*Environment: Kaggle/Colab notebook `07_final_evaluation.ipynb` for final runs; laptop for documentation and packaging.*

* [ ] Run final reproducible experiments
* [ ] Freeze final model/version
* [ ] Record final metrics
* [ ] Document architecture
* [ ] Document experiments and ablations
* [ ] Document limitations
* [ ] Document deployment architecture
* [ ] Write numbers-first README
* [ ] Prepare project demonstration
* [ ] Final end-to-end validation
