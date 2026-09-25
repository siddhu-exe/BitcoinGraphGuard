# Agent Rules

## General

* Read the existing project structure and relevant code before making changes.
* Make small, focused, reversible changes.
* Do not rewrite working code unnecessarily.
* Follow the existing project structure and coding conventions.
* **Production-quality code**: all committed code must be production quality — clear, readable, typed where practical, error-handled, tested, and free of dead code, debug leftovers, and hardcoded values.
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

## Compute & Training Environments (CRITICAL)

BitcoinGraphGuard uses **two environments**. See `docs/ARCHITECTURE.md` for the full
pipeline and artifact handoff.

* **Laptop = development and documentation only.** Writing and reviewing code, managing
  the Git repository, editing documentation, system design, reviewing notebook code,
  backend/FastAPI, Docker, MLOps engineering code, CI/CD, dashboard/frontend, and project
  configuration.
* **Kaggle / Google Colab = ALL ML execution.** Dataset loading, ML data processing, EDA,
  feature engineering, XGBoost, graph construction for training, GraphSAGE, RGCN, HGT,
  hyperparameter tuning, ablations, temporal/inductive evaluation, error analysis,
  GNNExplainer, final evaluation, and model-artifact generation all run remotely in
  notebooks.
* **Do not execute ML work on the laptop.** No expensive data processing, feature
  engineering, model training, GNN training, or experiments locally — not even XGBoost or
  "lightweight" EDA.
* **Export artifacts back.** Checkpoints, predictions, metrics, configurations, and
  explanation outputs are downloaded and consumed locally by MLflow/DVC, the API, and the
  dashboard.

## Notebook Architecture (CRITICAL)

Notebooks are the **executable implementation** of the ML work and run in Google Colab or
Kaggle. One notebook = one major project phase; never create a notebook per small step.

```text
notebooks/
├── 01_eda.ipynb
├── 02_xgboost.ipynb
├── 03_graphsage.ipynb
├── 04_heterogeneous_gnn.ipynb
├── 05_temporal_inductive_evaluation.ipynb
├── 06_explainability.ipynb
└── 07_final_evaluation.ipynb
```

* Adjust the count only for a genuine reason; never split one phase across several
  notebooks.
* A notebook holds many related steps internally — e.g. `01_eda.ipynb` covers loading,
  validation, temporal/class/graph/feature analysis, visualizations, and conclusions.
* Notebooks must run standalone on Colab/Kaggle. Application and serving logic belongs in
  `src/`; ML experimentation implementation belongs in the notebook.

## Notebook Quality & Reproducibility

* Write notebooks the way a real data scientist would: readable, logically ordered,
  well commented where it adds value, modular where useful, reasonably concise, and
  reproducible. Avoid auto-generated boilerplate.
* Do not comment obvious Python syntax. Use markdown cells to explain what is being done,
  why, what the result means, and what decision follows from it.
* Do not blindly run every possible analysis — each notebook tells a coherent story.
* Define random seeds, record configuration, identify the dataset, and never hardcode
  personal paths. Paths must work on Colab/Kaggle or be configurable in one cell at the top.
* Save metrics and required artifacts; integrate MLflow/DVC where they add real value
  rather than for appearance.
* Process data with chunked readers, memory-efficient structures, GPU use, mini-batch
  training, neighbour sampling, and explicit cleanup. Never fabricate synthetic or
  artificial data to make an experiment easier.

## Working Protocol: ML Phases (CRITICAL)

When the user asks for the next ML phase:

1. Provide the notebook implementation for that phase.
2. Assume the user runs it in Google Colab/Kaggle.
3. Never ask the user to run ML code on the laptop.
4. Keep the implementation inside the single appropriate phase notebook.
5. Do not create unnecessary additional notebooks.
6. Write understandable, human-like code.
7. Explain important decisions before or alongside the code.
8. Wait for the actual results before designing the next phase.

Proceed **one notebook/phase at a time**.

## Hardware & Low-Resource Constraints (CRITICAL)

* **Low Hardware Specs**: This laptop has limited memory (5.6 GB total RAM, ~3.0 GB available) and an older dual-core/4-thread Intel Core i3 CPU.
* **No ML on the Laptop**: The laptop never executes data processing, feature engineering, training, or experiments; those run in Colab/Kaggle notebooks.
* **No Heavy Unconstrained Tasks**: Never run heavy full-graph in-memory jobs, massive concurrent multiprocessing pools, or unconstrained training runs that can freeze the machine or trigger OOM errors.
* **Streaming & Chunking Mandate**: Always process raw CSVs and graph data using chunked readers (`chunksize`), streaming iterators, or generator pipelines — in notebooks as well.
* **Mini-Batching**: For GNNs and graph operations, use mini-batch sampling (`NeighborLoader`, `HeteroNeighborLoader`) instead of loading the entire ~1.03M-node / ~4.42M-edge graph into RAM at once.
* **Memory Management**: Explicitly delete large transient objects and invoke garbage collection (`gc.collect()`) after memory-intensive processing steps.

## Machine Learning Rules

* Establish classical ML baselines before claiming improvements from GNNs.
* Use appropriate metrics for severe class imbalance.
* Prioritize PR-AUC, precision, recall, F1, and confusion-matrix analysis over accuracy alone.
* Keep training, validation, and test data strictly separated.
* Use the temporal split defined by the project unless an experiment explicitly requires another split.
* Record model configuration and experiment parameters.
* Compare models using the same evaluation protocol.
* Never introduce data leakage: do not randomly split the primary temporal experiment, do
  not let future time steps construct historical training features, and do not use unknown
  labels as licit.
* Report actual results only after the experiment has been run.

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
