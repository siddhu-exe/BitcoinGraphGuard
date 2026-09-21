# Agent Rules

## General

* Read the existing project structure and relevant code before making changes.
* Make small, focused, reversible changes.
* Do not rewrite working code unnecessarily.
* Follow the existing project structure and coding conventions.
* Keep data processing, training, evaluation, inference, and monitoring clearly separated.
* Run relevant tests and validation checks after changes.
* Document important assumptions and design decisions.
* Never expose, hardcode, or commit secrets, credentials, API keys, or private data.
* Never commit raw datasets, model artifacts, credentials, or other large/generated files unless explicitly required.
* Do not introduce data leakage, especially across temporal train/test boundaries.
* Do not randomly split temporal graph data when evaluating temporal generalization.
* Preserve reproducibility through fixed seeds, configuration files, experiment tracking, and versioned data.
* Do not fabricate metrics, experimental results, or dataset statistics.
* Clearly distinguish experimental results from assumptions or expected outcomes.

## Hardware & Low-Resource Constraints (CRITICAL)

* **Low Hardware Specs**: This laptop has limited memory (5.6 GB total RAM, ~3.0 GB available) and an older dual-core/4-thread Intel Core i3 CPU.
* **No Heavy Unconstrained Tasks**: Never run heavy full-graph in-memory jobs, massive concurrent multiprocessing pools, or unconstrained training runs that can freeze the machine or trigger OOM errors.
* **Streaming & Chunking Mandate**: Always process raw CSVs and graph data using chunked readers (`chunksize`), streaming iterators, or generator pipelines.
* **Mini-Batching**: For GNNs and graph operations, use mini-batch sampling (`NeighborLoader`, `HeteroNeighborLoader`) instead of loading or training on the entire 1M-node / 4.4M-edge graph in RAM at once.
* **Memory Management**: Explicitly delete large transient objects and invoke garbage collection (`gc.collect()`) after memory-intensive processing steps.

## Machine Learning Rules

* Establish classical ML baselines before claiming improvements from GNNs.
* Use appropriate metrics for severe class imbalance.
* Prioritize PR-AUC, precision, recall, F1, and confusion-matrix analysis over accuracy alone.
* Keep training, validation, and test data strictly separated.
* Use the temporal split defined by the project unless an experiment explicitly requires another split.
* Record model configuration and experiment parameters.
* Compare models using the same evaluation protocol.

## Graph / Deep Learning Rules

* Preserve transaction and actor/wallet relationships when constructing the heterogeneous graph.
* Do not silently convert the complete problem into a simple homogeneous graph.
* Clearly distinguish transductive and inductive experiments.
* Prevent information from future time steps from entering training features or graph construction.
* Track graph-specific preprocessing and transformations.
* Save model checkpoints and experiment metadata reproducibly.

## MLOps Rules

* Use DVC for dataset/version management where appropriate.
* Use MLflow for experiment tracking and model metadata.
* Keep training and inference environments reproducible.
* Validate models before deployment.
* Monitor both model performance and graph-structural drift.
* Retraining must be triggered by defined evidence or thresholds, not arbitrary decisions.
* Dockerize production services.
* Keep CI/CD checks automated where practical.

## Task Completion Report

At the end of every task, report:

* What changed
* Files changed
* Tests/checks run
* Results
* Important assumptions
* Remaining issues
* Next recommended step
