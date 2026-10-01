"""
Diagnose the adversarial drift channel's step-35 false alarm (Phase 8a, Fix 1).

Runs the adversarial domain classifier on the same cadence steps as the backtest
(35, 40, 45, 49), with and without the near-deterministic monotone triad
(Aggregate_feature_10/43/8), and attributes the residual separability with
permutation importance on the lightweight logistic classifier.

Artifacts
---------
results/monitoring/adversarial_auc_comparison.csv
results/monitoring/adversarial_step35_attribution.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.adversarial_drift import AdversarialDriftMonitor

LOCAL = [f"Local_feature_{i}" for i in range(1, 94)]
TRIAD = ["Aggregate_feature_10", "Aggregate_feature_43", "Aggregate_feature_8"]
STEPS = [35, 40, 45, 49]
SEEDS = [42, 1, 7, 2024]
OUT_AUC = Path("results/monitoring/adversarial_auc_comparison.csv")
OUT_ATTR = Path("results/monitoring/adversarial_step35_attribution.csv")


def _run(cache, feats, seed=42, subsample=1200):
    monitor = AdversarialDriftMonitor(
        features=feats, subsample_size=subsample, n_folds=3, random_state=seed,
        auto_exclude_monotone=False,
    )
    monitor.fit_reference(cache["reference_df"])
    return {t: monitor.evaluate_step(cache["test_steps_dict"][t], time_step=t).adversarial_auc for t in STEPS}


def _permutation(cache, t, feats, seed=42, subsample=1200):
    ref = cache["reference_df"][feats].dropna().sample(n=subsample, random_state=seed)
    tgt = cache["test_steps_dict"][t][feats].dropna().sample(n=subsample, random_state=seed)
    X = np.vstack([ref.to_numpy(np.float32), tgt.to_numpy(np.float32)])
    y = np.concatenate([np.zeros(len(ref)), np.ones(len(tgt))])
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xs, y)
    imp = permutation_importance(clf, Xs, y, n_repeats=5, random_state=seed, scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1]
    return [
        {"feature": feats[i], "importance_mean": float(imp.importances_mean[i]),
         "importance_std": float(imp.importances_std[i])}
        for i in order
    ]


def main() -> None:
    import joblib

    cache = joblib.load("results/monitoring/features_cache.joblib")

    rows = []
    all_auc = _run(cache, None)
    no_triad_auc = _run(cache, LOCAL)
    for t in STEPS:
        rows.append({"time_step": t, "auc_all_features": all_auc[t], "auc_triad_excluded": no_triad_auc[t]})
    for seed in SEEDS[1:]:
        seed_auc = _run(cache, LOCAL, seed=seed)
        for t in STEPS:
            rows.append({"time_step": t, "auc_all_features": np.nan, "auc_triad_excluded": seed_auc[t]})
    df = pd.DataFrame(rows)
    OUT_AUC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_AUC, index=False)

    attr = pd.DataFrame(_permutation(cache, 35, LOCAL))
    attr.insert(0, "time_step", 35)
    attr.to_csv(OUT_ATTR, index=False)

    print(df.to_string(index=False))
    print("\nStep-35 attribution (local-only), top 8:")
    print(attr.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
