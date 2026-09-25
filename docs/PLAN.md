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
* [x] Analyze temporal structure (first pass run locally; see `docs/EDA.md`)
* [x] Analyze class imbalance (first pass run locally; see `docs/EDA.md`)
* [x] Perform graph-level EDA (first pass run locally; see `docs/EDA.md`)
* [x] Author the Phase 1 EDA notebook (`notebooks/01_eda.ipynb`)
* [ ] Execute `notebooks/01_eda.ipynb` on Colab/Kaggle and record the results in `docs/EDA.md`
* [ ] Configure DVC
* [ ] Establish reproducibility and configuration system

## Phase 2 — Baselines

*Environment: Kaggle/Colab notebooks `02_xgboost.ipynb` and `03_graphsage.ipynb`; laptop only for MLflow configuration.*

* [ ] Engineer transaction-level graph features
* [ ] Build classical ML dataset
* [ ] Train XGBoost baseline (remote)
* [ ] Establish fraud-detection evaluation metrics
* [ ] Implement temporal evaluation protocol
* [ ] Implement GraphSAGE baseline (remote)
* [ ] Compare XGBoost and GraphSAGE
* [ ] Configure MLflow experiment tracking

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
