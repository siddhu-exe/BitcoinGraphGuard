# Progress

## Current Status

Project initialization and dataset acquisition.

The project direction is fixed around **BitcoinGraphGuard**, using the real Elliptic++ dataset for temporal heterogeneous graph-based Bitcoin fraud detection.

## Completed

* [x] Project scope defined
* [x] Elliptic++ selected as the primary dataset
* [x] Project architecture defined
* [x] Six-phase development strategy defined
* [x] ML baseline strategy defined
* [x] Graph deep learning strategy defined
* [x] Temporal evaluation strategy defined
* [x] Inductive evaluation strategy defined
* [x] Explainability strategy defined
* [x] Graph drift monitoring strategy defined
* [x] MLOps strategy defined

## Current Task

* [ ] Download Elliptic++ dataset
* [ ] Verify downloaded files
* [ ] Inspect dataset structure
* [ ] Record dataset statistics
* [ ] Set up project environment
* [ ] Initialize DVC
* [ ] Initialize MLflow experiment tracking

## Next

* Validate transaction dataset
* Validate actor/wallet dataset
* Understand all node and edge relationships
* Perform initial temporal and class-imbalance EDA
* Build the project data pipeline

## Known Issues

* Dataset has not yet been fully integrated.
* Final heterogeneous graph schema is pending dataset inspection.
* Model architecture will be finalized after understanding the available features and relationships.
* Final deployment infrastructure will be decided after the inference pipeline is implemented.

## Important Rules

* Do not introduce synthetic data to replace Elliptic++.
* Do not randomly split temporal data for the primary evaluation.
* Do not introduce data leakage.
* Do not report experimental metrics before running the experiment.
* Do not replace the heterogeneous graph objective with a simpler unrelated approach.
