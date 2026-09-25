# Project Objective

## Goal

Build **BitcoinGraphGuard**, an end-to-end real-world Bitcoin fraud detection system using the **Elliptic++ dataset**.

The system will model Bitcoin transactions and actor/wallet relationships as a heterogeneous temporal graph and use machine learning and deep learning to identify potentially illicit activity.

The project must demonstrate the complete workflow of a production-oriented Data Science project, from data exploration and modeling through explainability, monitoring, deployment, and MLOps.

## Requirements

* Use the real **Elliptic++** Bitcoin transaction and actor/wallet dataset.
* Perform temporal and graph-based exploratory data analysis.
* Build a strong classical ML baseline using engineered graph features.
* Implement a GraphSAGE baseline.
* Implement a heterogeneous graph model using RGCN or HGT.
* Preserve transaction and actor/wallet relationships.
* Use a real temporal evaluation setup rather than a random split.
* Evaluate inductive generalization to previously unseen transactions/wallets.
* Handle severe class imbalance appropriately.
* Evaluate using PR-AUC, precision, recall, F1, confusion matrix, and relevant fraud-detection metrics.
* Perform model ablation experiments.
* Use GNNExplainer or an equivalent graph explanation method for selected predictions.
* Monitor graph-structural drift across temporal periods.
* Monitor model-performance degradation across time.
* Define evidence-based retraining triggers.
* Track experiments and models using MLflow.
* Version important datasets and artifacts using DVC where appropriate.
* Provide reproducible training and evaluation pipelines.
* Expose model inference through a FastAPI service.
* Containerize the system with Docker.
* Implement CI/CD using GitHub Actions.
* Build a monitoring/dashboard layer.
* Document the complete system, experiments, results, limitations, and production considerations.
* Deliver all ML experimentation as one notebook per project phase under `notebooks/`,
  runnable directly in Google Colab or Kaggle.

## Compute Strategy

The objective is delivered across **two environments**; full details are in
`ARCHITECTURE.md`.

* **Laptop — development & documentation only**: repository management, code development
  and review, documentation, system design, reviewing notebook code, FastAPI backend,
  Docker, MLOps engineering code, CI/CD, dashboard, and configuration.
* **Kaggle / Google Colab — ALL ML execution, inside notebooks**: dataset loading, ML data
  processing, EDA, feature engineering, graph construction, XGBoost, GraphSAGE, RGCN/HGT,
  hyperparameter tuning, ablations, temporal and inductive evaluation, error analysis,
  GNNExplainer, final evaluation, and model-artifact generation.

ML work must **not** run on the laptop — no data processing, feature engineering, training,
or experiments. Each phase has exactly one notebook (`notebooks/01_eda.ipynb` …
`notebooks/07_final_evaluation.ipynb`) that runs standalone on Colab/Kaggle; application and
serving logic belongs in `src/`, and trained artifacts/metrics are exported back to the
laptop for MLflow/DVC tracking and serving.

## Definition of Done

* [ ] Elliptic++ dataset integrated and validated
* [ ] Data quality and temporal EDA completed
* [ ] Transaction graph constructed
* [ ] Actor/wallet graph constructed
* [ ] Heterogeneous graph representation implemented
* [ ] XGBoost baseline implemented
* [ ] GraphSAGE baseline implemented
* [ ] RGCN/HGT model implemented
* [ ] Temporal evaluation completed
* [ ] Inductive evaluation completed
* [ ] Class imbalance strategy evaluated
* [ ] Ablation studies completed
* [ ] Explainability analysis completed
* [ ] Graph drift monitoring implemented
* [ ] Model drift monitoring implemented
* [ ] Retraining trigger defined
* [ ] MLflow experiment tracking implemented
* [ ] DVC data/version management implemented where appropriate
* [ ] FastAPI inference service implemented
* [ ] Docker setup completed
* [ ] CI/CD pipeline implemented
* [ ] Monitoring/dashboard completed
* [ ] Tests and validation completed
* [ ] Documentation completed
* [ ] Deployment completed
* [ ] Final results and limitations documented
* [ ] Training environments (Kaggle/Colab) documented and reproducible from the phase notebooks
* [ ] Phase notebooks implemented under `notebooks/` (one per phase, standalone on Colab/Kaggle)
