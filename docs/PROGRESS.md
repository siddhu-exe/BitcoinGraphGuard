# Progress

## Current Status

Project initialization and dataset acquisition complete; Phase 1 data engineering ongoing
on the laptop.

The project direction is fixed around **BitcoinGraphGuard**, using the real Elliptic++ dataset for temporal heterogeneous graph-based Bitcoin fraud detection.

## Compute Strategy

Work is split across two environments (see `ARCHITECTURE.md`):

* **Laptop** — engineering: code, data inspection, lightweight EDA/preprocessing, graph
  schema design, experiment configuration, testing, FastAPI, MLflow/DVC, Docker, CI/CD,
  monitoring, dashboard, documentation.
* **Kaggle / Google Colab** — all model training and heavy experiments (XGBoost,
  GraphSAGE, RGCN/HGT, tuning, ablations, temporal/inductive evaluation, GNNExplainer).

No project model is trained on the laptop. Remote environments export checkpoints,
predictions, metrics, and explanation outputs back for local tracking and serving.

## Verified Dataset Facts

From `DATA_INVENTORY.md`: 203,769 transaction nodes, 822,942 wallet/address nodes,
49 time steps, ~1.03M unique nodes, ~4.42M heterogeneous directed edges
(234,355 tx→tx; 477,117 addr→tx; 837,124 tx→addr; 2,868,964 addr→addr), raw data ~2.1 GB.

Labels — transactions: illicit 4,545 / licit 42,019 / unknown 157,205. Wallets:
illicit 14,266 / licit 251,088 / unknown 557,588. **Unknown labels are not legitimate.**

## Completed

* [x] Project scope defined
* [x] Elliptic++ selected as the primary dataset
* [x] Project architecture defined
* [x] Nine-phase development strategy defined (see `PLAN.md`)
* [x] ML baseline strategy defined
* [x] Graph deep learning strategy defined
* [x] Temporal evaluation strategy defined
* [x] Inductive evaluation strategy defined
* [x] Explainability strategy defined
* [x] Graph drift monitoring strategy defined
* [x] MLOps strategy defined
* [x] Download and verify Elliptic++ raw dataset
* [x] Inspect dataset structure and record statistics (203,769 txs, 822,942 wallets, 4 edge types, 49 time steps)
* [x] Verify data quality, missing values, duplicates, and integrity constraints
* [x] Complete Data Inventory report (`DATA_INVENTORY.md`) and reproducible script (`../scripts/verify_dataset.py`)
* [x] Define implementation architecture and laptop vs Kaggle/Colab compute split (`ARCHITECTURE.md`)
* [x] Create `requirements.txt` (user-managed install pending)

## Current Task

* [ ] Install project Python dependencies from `requirements.txt` (user-managed)
* [ ] Initial temporal and class-imbalance EDA
* [ ] Configure DVC and project directory structure (`src/`)
* [ ] Initialize MLflow tracking
* [ ] Prepare Kaggle/Colab training environment and notebook entrypoints

## Next

* Build the project data pipeline (laptop, memory-aware)
* Design the heterogeneous graph schema and preprocessing
* Set up remote training environment on Kaggle/Colab
* Begin Phase 2 feature engineering and XGBoost baseline preparation

## Known Issues

* Dataset has not yet been fully integrated.
* Final heterogeneous graph schema is pending dataset inspection.
* Model architecture will be finalized after understanding the available features and relationships.
* Final deployment infrastructure will be decided after the inference pipeline is implemented.
* Remote training environments (Kaggle/Colab) are not yet set up or documented as runnable.
* Four large CSVs have not been re-verified under the memory-safe inspection script (see `DATA_INVENTORY.md` §7).

## Important Rules

* Do not introduce synthetic data to replace Elliptic++.
* Do not randomly split temporal data for the primary evaluation.
* Do not introduce data leakage.
* Do not report experimental metrics before running the experiment.
* Do not replace the heterogeneous graph objective with a simpler unrelated approach.
