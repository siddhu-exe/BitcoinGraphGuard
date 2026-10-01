"""
Adversarial Drift Monitor Module for BitcoinGraphGuard.

Productionizes the adversarial validation diagnostic from Phase 7b (OOD Diagnosis)
into a lightweight, periodic domain separability detector.

Distinguishes reference training transactions (steps 1-34) from incoming target
transactions (step t or sliding window [t-W+1, t]) using a fast classifier
(e.g., Logistic Regression or shallow tree/ensemble) and reports cross-validated
or out-of-fold ROC-AUC.

Cadence & Compute Cost:
- Adversarial validation is higher cost than univariate PSI/KS.
- Recommended production cadence: Every 5 steps (e.g. steps 35, 40, 45, 49)
  or conditionally triggered whenever high-priority feature PSI flags a warning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdversarialDriftLevel(str, Enum):
    """Adversarial validation domain separation severity."""

    STABLE = "STABLE"
    MODERATE = "MODERATE"
    ALERT = "ALERT"

    @classmethod
    def from_auc(
        cls, auc: float, warn_thresh: float = 0.75, alert_thresh: float = 0.85
    ) -> AdversarialDriftLevel:
        if auc < warn_thresh:
            return cls.STABLE
        if auc < alert_thresh:
            return cls.MODERATE
        return cls.ALERT


@dataclass
class AdversarialDriftReport:
    """Report on adversarial validation domain separability."""

    time_step: int
    adversarial_auc: float
    auc_std: float
    drift_level: AdversarialDriftLevel
    n_reference_samples: int
    n_target_samples: int
    n_folds: int
    execution_time_seconds: float
    model_type: str
    feature_names: List[str]
    is_periodic_run: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "adversarial_auc": round(self.adversarial_auc, 4),
            "auc_std": round(self.auc_std, 4),
            "drift_level": self.drift_level.value,
            "n_reference_samples": self.n_reference_samples,
            "n_target_samples": self.n_target_samples,
            "n_folds": self.n_folds,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "model_type": self.model_type,
            "is_periodic_run": self.is_periodic_run,
            "reasons": self.reasons,
        }


class AdversarialDriftMonitor:
    """
    Lightweight periodic adversarial domain discriminator.

    Parameters
    ----------
    features : Optional[List[str]]
        Features to include. If None, uses all common columns between ref and target.
    model_type : str, default='logistic'
        Classifier type: 'logistic' (fast linear) or 'fast_tree'.
    subsample_size : int, default=1500
        Maximum number of reference and target samples to balance compute on laptop.
    n_folds : int, default=3
        Number of cross-validation folds.
    cadence_steps : int, default=5
        Recommended periodicity (run every N steps).
    warn_threshold : float, default=0.75
        ROC-AUC threshold for moderate domain shift.
    alert_threshold : float, default=0.85
        ROC-AUC threshold for severe domain shift.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        model_type: str = "logistic",
        subsample_size: int = 1500,
        n_folds: int = 3,
        cadence_steps: int = 5,
        warn_threshold: float = 0.75,
        alert_threshold: float = 0.85,
        random_state: int = 42,
        auto_exclude_monotone: bool = True,
        monotonic_rho_threshold: float = 0.99,
    ):
        self.features = features
        self.model_type = model_type
        self.subsample_size = subsample_size
        self.n_folds = n_folds
        self.cadence_steps = cadence_steps
        self.warn_threshold = warn_threshold
        self.alert_threshold = alert_threshold
        self.random_state = random_state
        # Near-deterministic monotone features (|Spearman rho(step, per-step median)|
        # >= threshold) are excluded from the domain classifier: they encode the time
        # step directly (deterministic time proxy), so they make training vs. any
        # future step trivially separable regardless of regime. Same detection rule
        # as FeatureDriftMonitor.
        self.auto_exclude_monotone = auto_exclude_monotone
        self.monotonic_rho_threshold = monotonic_rho_threshold

        self.reference_data_: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False
        self.excluded_monotone_features_: List[str] = []

    def fit_reference(self, reference_df: pd.DataFrame) -> AdversarialDriftMonitor:
        """
        Store reference training distribution (steps 1-34).

        Subsamples if necessary to maintain fast evaluation on resource-constrained hardware.
        """
        feats = self.features or [
            c for c in reference_df.columns
            if c.startswith("Local_feature_") or c.startswith("Aggregate_feature_")
        ]

        # Drop near-deterministic monotone features (deterministic time proxies).
        self.excluded_monotone_features_ = []
        if self.auto_exclude_monotone and "Time step" in reference_df.columns and len(feats) > 1:
            from scipy import stats as _stats

            kept: List[str] = []
            for feat in feats:
                sub = reference_df[["Time step", feat]].dropna()
                med = sub.groupby("Time step")[feat].median()
                med = med[np.isfinite(med.to_numpy(dtype=float))]
                if len(med) >= 10 and np.unique(med.to_numpy(dtype=float)).size > 1:
                    rho = _stats.spearmanr(
                        med.index.to_numpy(dtype=float), med.to_numpy(dtype=float)
                    )[0]
                    if np.isfinite(rho) and abs(rho) >= self.monotonic_rho_threshold:
                        self.excluded_monotone_features_.append(feat)
                        continue
                kept.append(feat)
            if self.excluded_monotone_features_:
                logger.info(
                    "Excluding %d near-deterministic monotone feature(s) from adversarial classifier: %s",
                    len(self.excluded_monotone_features_),
                    ", ".join(self.excluded_monotone_features_),
                )
            feats = kept

        self.features = feats

        clean_df = reference_df[feats].dropna()
        if len(clean_df) > self.subsample_size:
            self.reference_data_ = clean_df.sample(
                n=self.subsample_size, random_state=self.random_state
            )
        else:
            self.reference_data_ = clean_df.copy()

        self.is_fitted = True
        return self

    def should_run_at_step(self, time_step: int, force: bool = False) -> bool:
        """Check if adversarial validation should execute at given time step."""
        if force:
            return True
        # Run at step 35, 40, 45, 49 or multiples of cadence
        return (time_step % self.cadence_steps == 0) or (time_step == 35) or (time_step == 49)

    def evaluate_step(
        self,
        target_df: pd.DataFrame,
        time_step: int,
        force_run: bool = False,
    ) -> AdversarialDriftReport:
        """
        Run adversarial domain discrimination between reference and target.

        Parameters
        ----------
        target_df : pd.DataFrame
            Target transactions for current step or window.
        time_step : int
            Current time step.
        force_run : bool, default=False
            Override cadence schedule.

        Returns
        -------
        AdversarialDriftReport
        """
        if not self.is_fitted or self.reference_data_ is None:
            raise RuntimeError("AdversarialDriftMonitor must be fitted with reference data first.")

        t0 = time.time()
        is_scheduled = self.should_run_at_step(time_step, force=force_run)

        feats = [f for f in self.features if f in target_df.columns and f in self.reference_data_.columns]
        target_clean = target_df[feats].dropna()

        if len(target_clean) > self.subsample_size:
            target_sub = target_clean.sample(n=self.subsample_size, random_state=self.random_state)
        else:
            target_sub = target_clean.copy()

        n_ref = len(self.reference_data_)
        n_tgt = len(target_sub)

        if n_tgt < 10:
            elapsed = time.time() - t0
            return AdversarialDriftReport(
                time_step=time_step,
                adversarial_auc=0.5,
                auc_std=0.0,
                drift_level=AdversarialDriftLevel.STABLE,
                n_reference_samples=n_ref,
                n_target_samples=n_tgt,
                n_folds=self.n_folds,
                execution_time_seconds=elapsed,
                model_type=self.model_type,
                feature_names=feats,
                is_periodic_run=is_scheduled,
                reasons=["Insufficient target sample size for adversarial validation."],
            )

        # Build combined dataset: 0 = reference, 1 = target
        X_ref = self.reference_data_[feats].to_numpy(dtype=np.float32)
        X_tgt = target_sub[feats].to_numpy(dtype=np.float32)

        y_ref = np.zeros(n_ref, dtype=int)
        y_tgt = np.ones(n_tgt, dtype=int)

        X = np.vstack([X_ref, X_tgt])
        y = np.concatenate([y_ref, y_tgt])

        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_aucs: List[float] = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)

            clf = LogisticRegression(
                max_iter=150,
                C=1.0,
                solver="lbfgs",
                random_state=self.random_state,
            )
            clf.fit(X_tr_scaled, y_tr)
            probs = clf.predict_proba(X_va_scaled)[:, 1]

            score = float(roc_auc_score(y_val, probs))
            fold_aucs.append(score)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        elapsed = time.time() - t0

        drift_level = AdversarialDriftLevel.from_auc(
            mean_auc,
            warn_thresh=self.warn_threshold,
            alert_thresh=self.alert_threshold,
        )

        reasons: List[str] = []
        if drift_level == AdversarialDriftLevel.ALERT:
            reasons.append(
                f"AdversarialAlert: Domain separability AUC={mean_auc:.4f} >= {self.alert_threshold:.2f} (severe OOD covariate shift from training)."
            )
        elif drift_level == AdversarialDriftLevel.MODERATE:
            reasons.append(
                f"AdversarialWarning: Domain separability AUC={mean_auc:.4f} >= {self.warn_threshold:.2f} (moderate domain divergence)."
            )

        return AdversarialDriftReport(
            time_step=time_step,
            adversarial_auc=mean_auc,
            auc_std=std_auc,
            drift_level=drift_level,
            n_reference_samples=n_ref,
            n_target_samples=n_tgt,
            n_folds=self.n_folds,
            execution_time_seconds=elapsed,
            model_type=self.model_type,
            feature_names=feats,
            is_periodic_run=is_scheduled,
            reasons=reasons,
        )
