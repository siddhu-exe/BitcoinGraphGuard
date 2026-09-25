# Plan

Each phase is annotated with its **execution environment** and, where ML is involved, the
notebook that implements it. All ML execution — EDA, feature engineering, training,
evaluation, explainability — runs in notebooks on **Kaggle / Google Colab**; the laptop
handles engineering, documentation, and MLOps only. See `ARCHITECTURE.md`.

## Notebook Map

| Notebook | Phase |
| :--- | :--- |
| `notebooks/01_eda.ipynb` | Phase 1 — EDA |
| `notebooks/02_xgboost.ipynb` | Phase 2 — Classical Baseline (XGBoost) |
| `notebooks/03_graphsage.ipynb` | Phase 3 — Homogeneous Graph Baseline (GraphSAGE) |
| `notebooks/04_heterogeneous_gnn.ipynb` | Phase 4 — Heterogeneous GNN (RGCN / HGT) |
| `notebooks/05_temporal_inductive_evaluation.ipynb` | Phase 5 — Temporal & Inductive Evaluation |
| `notebooks/06_explainability.ipynb` | Phase 6 — Explainability |
| `notebooks/07_final_evaluation.ipynb` | Phase 9 — Final Evaluation |

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

## Phase 2 — Classical Baselines (XGBoost)

*Environment: Kaggle/Colab notebook `02_xgboost.ipynb`; laptop for MLflow configuration.*

*Status: COMPLETE (executed 2026-09-25 on Colab in two passes). The first pass is the recorded 500-tree
baseline; the second lifts the budget to 1500 trees (582 kept, so the cap no longer binds) and adds a
controlled 20-trial randomised search, a `has_addresses` ablation and one label-free redundancy
ablation. `notebooks/02_xgboost.ipynb` covers the leakage audit, the prevalence baseline, Logistic
Regression and XGBoost on the 165 non-domain transaction features, evaluated fit 1–24 / validation
25–34 / refit 1–34 / test 35–49.*

Measured on the test period 35–49 (16,670 labeled, 1,083 illicit): prevalence baseline PR-AUC
0.0650, Logistic Regression 0.2917, XGBoost baseline 0.8007 (ROC-AUC 0.9317), optimised XGBoost
0.8013 (ROC-AUC 0.9281). Retuning bought +0.0006 PR-AUC while ROC-AUC and F1 fell: the tree budget
was not the constraint and the tabular features set the performance ceiling. Sub-window PR-AUC is
0.9215 on 35–42 but 0.0427 on 43–49, where prevalence drops to 2.53%. `has_addresses` was dropped
after scoring +0.000933 against a pre-registered +0.005 margin. Record: `docs/XGBOOST.md`.

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
* [ ] Configure MLflow experiment tracking

## Phase 3 — Homogeneous Graph Baseline (GraphSAGE)

*Environment: Kaggle/Colab notebook `03_graphsage.ipynb` (graph construction, training, ablation).*

*Status: COMPLETE (executed 2026-09-25 on Colab with CUDA GPU; artifacts in `GraphSage/`). Evaluates
homogeneous transaction graph (`txs_edgelist.csv`) with 165 features under fit 1–24 / validation 25–34 /
train 1–34 / test 35–49. GraphSAGE reaches **PR-AUC 0.6209 / ROC-AUC 0.9044 / F1 0.5945** vs 2-layer MLP
**0.4768 / 0.8912 / 0.5759** (+0.1442 lift over neural baseline) and frozen XGBoost **0.8013 / 0.9281 / 0.7818**.
Intra-step edge confinement empirically verified (100% intra-step, 0 cross-step), explaining why homogeneous
GNNs cannot bridge temporal steps without wallet nodes. Full record: `docs/GRAPHSAGE.md`.*

* [x] Author `notebooks/03_graphsage.ipynb` — standalone Colab/Kaggle notebook for GraphSAGE & MLP
* [x] Author `docs/GRAPHSAGE.md` — design, structural verification, ablation, and artifact specification
* [x] Empirically verify intra-step edge confinement (234,355 intra-step, 0 cross-step)
* [x] Formulate architectural twin MLP baseline for controlled graph ablation
* [x] Implement leakage-free feature scaling (`StandardScaler` on fit/train only) and historical `pos_weight`
* [x] Implement validation PR-AUC early stopping (steps 25–34) and F1-maximizing operating threshold
* [x] Implement connected (`degree >= 1`, 100%) vs isolated (`degree == 0`, 0%) test decomposition
* [x] Implement automated sanity assertion suite (`checks.csv`) and artifact export
* [x] Execute `notebooks/03_graphsage.ipynb` on Google Colab (GPU runtime)
* [x] Export artifacts to `GraphSage/` and review results against frozen XGBoost baseline (0.8013)
* [x] Populate experimental results in `docs/GRAPHSAGE.md`

## Phase 4 — Heterogeneous Graph Deep Learning (RGCN / HGT)

*Environment: Kaggle/Colab notebook `04_heterogeneous_gnn.ipynb` (graph construction and training).*

*Do not start Phase 4 until Phase 3 artifacts are exported and reviewed.*

* [ ] Construct heterogeneous transaction + actor/wallet graph (`AddrTx`, `TxAddr`, `AddrAddr`, `txs_edgelist`)
* [ ] Define node and edge types
* [ ] Build graph preprocessing pipeline
* [ ] Implement RGCN (remote)
* [ ] Evaluate HGT as an alternative if justified (remote)
* [ ] Tune model using validation data (remote)
* [ ] Handle class imbalance
* [ ] Compare against XGBoost (0.8013) and GraphSAGE
* [ ] Run ablation studies (remote)
* [ ] Analyze model errors

## Phase 5 — Temporal & Inductive Evaluation

*Environment: Kaggle/Colab notebook `05_temporal_inductive_evaluation.ipynb`.*

* [ ] Train using historical time steps
* [ ] Evaluate on later unseen time steps
* [ ] Measure temporal performance degradation
* [ ] Evaluate unseen transactions/wallets
* [ ] Measure inductive generalization
* [ ] Analyze performance across individual time steps
* [ ] Document temporal failure modes

## Phase 6 — Explainability

*Environment: Kaggle/Colab notebook `06_explainability.ipynb` (GNNExplainer).*

* [ ] Select representative fraud predictions
* [ ] Select false-positive cases
* [ ] Select false-negative cases
* [ ] Apply GNNExplainer or equivalent method
* [ ] Identify influential nodes and edges
* [ ] Visualize important graph structures
* [ ] Document explainability limitations

## Phase 7 — MLOps & Monitoring

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

## Phase 8 — ML System & Deployment

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

## Phase 9 — Monitoring & Dashboard

*Environment: Laptop.*

* [ ] Build model monitoring dashboard
* [ ] Display fraud predictions and model metrics
* [ ] Display temporal performance
* [ ] Display graph drift
* [ ] Display model drift
* [ ] Display retraining status
* [ ] Add representative explainability visualizations
* [ ] Validate dashboard against tracked experiments

## Phase 10 — Finalization

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
