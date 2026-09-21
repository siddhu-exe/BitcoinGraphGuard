# Plan

## Phase 1 — Data & Foundation

* [ ] Create project structure
* [ ] Configure Python environment
* [ ] Install and pin dependencies
* [ ] Download and verify Elliptic++ dataset
* [ ] Inspect dataset structure and metadata
* [ ] Validate transaction and actor/wallet data
* [ ] Identify missing values, duplicates, invalid relationships, and label distribution
* [ ] Analyze temporal structure
* [ ] Analyze class imbalance
* [ ] Perform graph-level EDA
* [ ] Configure DVC
* [ ] Establish reproducibility and configuration system

## Phase 2 — Baselines

* [ ] Engineer transaction-level graph features
* [ ] Build classical ML dataset
* [ ] Train XGBoost baseline
* [ ] Establish fraud-detection evaluation metrics
* [ ] Implement temporal evaluation protocol
* [ ] Implement GraphSAGE baseline
* [ ] Compare XGBoost and GraphSAGE
* [ ] Configure MLflow experiment tracking

## Phase 3 — Heterogeneous Graph Deep Learning

* [ ] Construct heterogeneous transaction + actor/wallet graph
* [ ] Define node and edge types
* [ ] Build graph preprocessing pipeline
* [ ] Implement RGCN
* [ ] Evaluate HGT as an alternative if justified
* [ ] Tune model using validation data
* [ ] Handle class imbalance
* [ ] Compare against XGBoost and GraphSAGE
* [ ] Run ablation studies
* [ ] Analyze model errors

## Phase 4 — Temporal & Inductive Evaluation

* [ ] Train using historical time steps
* [ ] Evaluate on later unseen time steps
* [ ] Measure temporal performance degradation
* [ ] Evaluate unseen transactions/wallets
* [ ] Measure inductive generalization
* [ ] Analyze performance across individual time steps
* [ ] Document temporal failure modes

## Phase 5 — Explainability

* [ ] Select representative fraud predictions
* [ ] Select false-positive cases
* [ ] Select false-negative cases
* [ ] Apply GNNExplainer or equivalent method
* [ ] Identify influential nodes and edges
* [ ] Visualize important graph structures
* [ ] Document explainability limitations

## Phase 6 — MLOps & Monitoring

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

* [ ] Build model monitoring dashboard
* [ ] Display fraud predictions and model metrics
* [ ] Display temporal performance
* [ ] Display graph drift
* [ ] Display model drift
* [ ] Display retraining status
* [ ] Add representative explainability visualizations
* [ ] Validate dashboard against tracked experiments

## Phase 9 — Finalization

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
