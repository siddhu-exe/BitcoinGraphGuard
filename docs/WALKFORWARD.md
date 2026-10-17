# Walk-forward retraining under label delay (Phase 9a)

Question: after the step-43 regime shift, can retraining recover the model, and how much does the
delay before labels arrive limit that recovery?

**Verdict: (b) PARTIAL.** Retraining recovers performance when labels arrive within one step. It does
not reach the pre-registered "restored" level (PR-AUC 0.50) at both a one-step and a three-step delay,
and the gain shrinks quickly as the delay grows.

Everything below is taken from `results/walkforward/` (`verdict.md`, `digest.txt`, `lag_curve.csv`,
`pooled_metrics.csv`, `sustained_recovery.csv`, `operating_rules.csv`, `threshold_instability.csv`,
`monitor_concordance.csv`, `checks.csv`). The generator is `scripts/generate_notebook_09.py`, which
writes `notebooks/09_walkforward_retraining.ipynb` (run on Colab). 97 of 97 internal checks passed.

## 1. Protocol

- **Walk-forward.** For each test step `t` in 35-49 the model used to score `t` is chosen and trained
  using only labels that have already arrived.
- **Label delay `L`.** Labels of step `s` become available at step `s + L`. A model scoring step `t`
  may use labels from steps `<= t - L` only. The harness records `max_label_step_used` and
  `label_arrival_ok` for every retrain (`retrain_log.csv`).
- **Delays run.** Main `L = 1`; sensitivity `L = 3`; lag curve `L = 1, 2, 3, 5` (static and expanding
  only at `L = 2` and `L = 5`).
- **No leakage.** Features and labels never come from steps after `t`; the static model at `L = 3` is
  trained through step 32, not 34, to respect label arrival. Class 3 (unknown) is never used as
  supervision.
- **Metric.** PR-AUC, pooled over the labeled transactions in the window. Reported separately for
  steps 35-42 and the drift window 43-49 (169 illicit labels; steps 45 and 46 hold only 5 and 2).
- **Intervals.** 95% cluster bootstrap over time steps (1,000 draws, paired across strategies). The
  row-level bootstrap treats rows inside a step as independent and is exported alongside for comparison
  only.
- **Pre-registered thresholds** (unchanged during the study): "restored" = PR-AUC 0.50, "material gain"
  = 0.15, non-inferiority margin 0.02. In-window oracle (uses 43-49's own labels, not deployable) = 0.9199.
- **Models fitted:** 87 unique, plus one `xgboost_v2` control that ties the harness to the earlier
  split (measured 0.7922 on 35-49 vs 0.8011 documented).

## 2. Strategies

| Strategy | What it does |
| :--- | :--- |
| `static` | Frozen model trained on steps 1-34 (no retraining) |
| `expanding` | Retrain every step on all labels that have arrived |
| `rolling_w5`, `rolling_w10` | Retrain on the latest 5 or 10 steps of arrived labels |
| `recency_exp_hl6` | Expanding window with exponential recency weights, half-life 6 steps |
| `monitor_triggered` | Retrain only when a lag-safe re-implementation of the monitoring rules fires |
| `expanding_no_triad` | Expanding, with the three aggregate "triad" features dropped |

`expanding` is the highest-ranked qualifying policy at `L = 1`.

## 3. Label-delay table

Pooled PR-AUC on steps 43-49, expanding window vs the frozen model, with 95% cluster-bootstrap CIs
(`lag_curve.csv`):

| Label delay L | Static (frozen) | Expanding retrain | Gain vs static | Sustained recovery |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0406 [0.0248, 0.0943] | **0.6949** [0.3559, 0.9120] | +0.6543 [+0.3257, +0.8217] | step 47 (+4) |
| 2 | 0.0415 [0.0246, 0.0965] | 0.3878 [0.1771, 0.7184] | +0.3463 [+0.1483, +0.6320] | not reached |
| 3 | 0.0270 [0.0164, 0.0592] | 0.1377 [0.1063, 0.4949] | +0.1107 [+0.0842, +0.4322] | not reached |
| 5 | 0.0398 | 0.1050 [0.0489, 0.3614] | +0.0652 [+0.0179, +0.2805] | not reached |

Never quote the `L = 1` value alone: the gain falls from +0.65 to +0.07 as the delay goes from one step
to five. The same strategies at `L = 1` and `L = 3` are in `verdict.md`; at `L = 1` every retraining
strategy improves on the frozen model by +0.41 to +0.65 PR-AUC on 43-49, with the cluster lower bound
above zero.

On steps 35-42 retraining costs nothing: `expanding` scores 0.9271 vs 0.9199 for `static` at `L = 1`.

### Sustained recovery

"Sustained recovery" is the first step `t >= 43` with PR-AUC >= 0.5 that holds for the next two
evaluated steps. Steps 45 and 46 (5 and 2 illicit labels) are excluded as noise. For `expanding` at
`L = 1` the PR-AUC on the evaluated steps is 0.34 (43), 0.38 (44), 0.60 (47), 0.89 (48), 0.99 (49), so
recovery is sustained **from step 47, four steps after onset**. The frozen model reaches 0.04, 0.04,
0.05, 0.20, 0.27 on the same steps (`sustained_recovery.csv`). `monitor_triggered` does not reach a
sustained recovery (it retrained 5 times, against 14 for the others).

## 4. Top-K triage

A fixed probability threshold behaves badly under the shift (section 5), so the study's primary
operating rule is **top-K alerts per step**, K = 20, 50, 100. The best recall any K can reach is

```
max_possible_recall = sum over steps of min(K, positives_s) / total positives
```

On steps 43-49 (positives per step 24, 24, 5, 2, 22, 36, 56; 169 in total) that ceiling is 0.633 for
K = 20 (107/169) and 0.964 for K = 50 (163/169). Precision and recall pooled over 43-49
(`operating_rules.csv`):

| Strategy | L | Rule | Alerts/step | Precision | Recall |
| :--- | ---: | :--- | ---: | ---: | ---: |
| static | 1 | top-20 | 20 | 0.029 | 0.024 |
| static | 1 | top-50 | 50 | 0.043 | 0.089 |
| expanding | 1 | top-20 | 20 | 0.507 | 0.420 |
| expanding | 1 | top-50 | 50 | 0.391 | 0.811 |
| expanding | 1 | top-100 | 100 | 0.223 | 0.923 |
| expanding | 3 | top-20 | 20 | 0.357 | 0.296 |
| expanding | 3 | top-50 | 50 | 0.297 | 0.615 |
| expanding | 3 | top-100 | 100 | 0.193 | 0.799 |

> **Known error in `alert_budget.csv`.** The window-level `max_possible_recall` and
> `max_possible_precision` columns (`level == "window"`) are wrong: they apply the per-step formula to
> window totals, so surplus alert capacity at a quiet step offsets a busy one and the ceilings are
> overstated. The correct ceiling takes the minimum **per step** and then sums, as above. Per-step rows
> of `alert_budget.csv`, `operating_rules.csv`, `verdict.md` and `digest.txt` are correct. See
> `results/walkforward/README.md`.

## 5. Threshold instability

The validation-selected threshold moves a lot once the data shifts. For `expanding` at `L = 1` the
threshold chosen at steps 43-49 is 0.810, 0.620, 0.360, 0.360, 0.360, 0.140, 0.075, so the share of
rows flagged runs from 0.000 at step 43 to 0.153 at step 49, and one step produces no alert at all
(`threshold_instability.csv`). The frozen model keeps one threshold (0.715) and flags 0.1% to 2.7% of
rows per step. At the validation threshold `expanding` reaches F1 0.5275 at `L = 1` against a per-step
oracle ceiling of 0.7301 (a ceiling, not a result).

Hold-out PR-AUC is also a poor guide to the next step during the shift: for `expanding` at `L = 1` on
43-49 the Pearson correlation between hold-out and realised PR-AUC is -0.57 excluding the noise steps
and -0.12 on all seven steps. That is a descriptive finding about drift, not a defect in the selection
code.

## 6. Monitor concordance

The `monitor_triggered` strategy re-implements the monitoring rules lag-safely. At `L = 1` over 43-49 it
agrees with the earlier Phase 8a rules on 5 of 7 steps; the first fire is at step 43 (label-free
channel) and the first label-dependent fire is at step 44. At `L = 3` no label-dependent channel fires
in 43-49 (`monitor_concordance_summary.csv`). The score-shift PSI used by the deployed monitor at step
43 is in `reports/monitoring_backtest_report.json`, not in this study.

## 7. Limitations

- One regime change, seven drift steps and 169 positives. Steps 45 and 46 are noise.
- Cluster intervals over seven steps are approximate and wide (for example 0.3559 to 0.9120 at `L = 1`).
- Extra delays `L = 2` and `L = 5` cover `static` and `expanding` only and are bootstrapped on 43-49
  only.
- The real label delay in production is unknown; the study shows what each delay would cost.
- The hold-out rule differs from `xgboost_v2`'s fixed 1-24 / 33-34 split; `v2_exact_control` ties the two.
- The `monitor_triggered` strategy is a re-implementation; the production engine was not imported, and
  its label-free channels have not been validated outside this dataset.
- Serving `expanding` needs a retraining job, a new model artifact and an updated consistency gate. It
  is not an API-only change, and the deployed model is still the frozen XGBoost.
