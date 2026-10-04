## Verdict: (b) PARTIAL

Intervals are 95% **cluster bootstrap over time steps** (the row-level interval is exported alongside in `pooled_metrics.csv` and `paired_deltas.csv`).

| Strategy | L | pooled 35-42 PR-AUC [95% CI] | pooled 43-49 PR-AUC [95% CI] | delta vs static on 43-49 | retrains | sustained recovery (PR-AUC >= 0.50 for 3 evaluated steps) |
| :--- | ---: | :--- | :--- | :--- | ---: | :--- |
| `static` | 1 | 0.9199 [0.8632, 0.9682] | 0.0406 [0.0248, 0.0943] | - | 0 | not reached |
| `expanding` | 1 | 0.9271 [0.8674, 0.9720] | 0.6949 [0.3559, 0.9120] | +0.6543 [+0.3257, +0.8217] | 14 | step 47 (+4) |
| `rolling_w5` | 1 | 0.9206 [0.8630, 0.9653] | 0.4528 [0.3087, 0.7034] | +0.4123 [+0.2791, +0.6227] | 14 | step 47 (+4) |
| `rolling_w10` | 1 | 0.9374 [0.8768, 0.9729] | 0.5988 [0.3002, 0.9058] | +0.5582 [+0.2702, +0.8152] | 14 | step 47 (+4) |
| `recency_exp_hl6` | 1 | 0.9388 [0.8811, 0.9766] | 0.6553 [0.3383, 0.8963] | +0.6147 [+0.3118, +0.8028] | 14 | step 47 (+4) |
| `monitor_triggered` | 1 | 0.8931 [0.8273, 0.9565] | 0.6355 [0.2670, 0.8844] | +0.5949 [+0.2419, +0.7859] | 5 | not reached |
| `expanding_no_triad` | 1 | 0.9308 [0.8689, 0.9735] | 0.6745 [0.2999, 0.9121] | +0.6339 [+0.2711, +0.8243] | 14 | step 47 (+4) |
| `static` | 3 | 0.9083 [0.8442, 0.9576] | 0.0270 [0.0164, 0.0592] | - | 0 | not reached |
| `expanding` | 3 | 0.9048 [0.8541, 0.9557] | 0.1377 [0.1063, 0.4949] | +0.1107 [+0.0842, +0.4322] | 14 | not reached |
| `rolling_w5` | 3 | 0.8870 [0.8078, 0.9530] | 0.0982 [0.0868, 0.2617] | +0.0712 [+0.0603, +0.2075] | 14 | not reached |
| `rolling_w10` | 3 | 0.8812 [0.8224, 0.9401] | 0.1599 [0.1152, 0.3381] | +0.1329 [+0.0917, +0.2874] | 14 | step 47 (+4) |
| `recency_exp_hl6` | 3 | 0.9046 [0.8424, 0.9532] | 0.1108 [0.0857, 0.2308] | +0.0838 [+0.0578, +0.2002] | 14 | not reached |
| `monitor_triggered` | 3 | 0.9048 [0.8479, 0.9576] | 0.1109 [0.0767, 0.4069] | +0.0839 [+0.0522, +0.3589] | 3 | not reached |
| `expanding_no_triad` | 3 | 0.9066 [0.8558, 0.9545] | 0.1521 [0.1027, 0.4476] | +0.1251 [+0.0812, +0.3923] | 14 | not reached |

Pre-registered thresholds: restored = 0.50, material gain = 0.15. In-window oracle (uses 43-49's own labels, not deployable) = 0.9199.
`expanding` reaches the restored level at both lags: **False**. Cells clearing the material-gain bar with a paired lower bound above 0: **[('expanding', 1), ('rolling_w5', 1), ('rolling_w10', 1), ('rolling_w10', 3), ('recency_exp_hl6', 1), ('monitor_triggered', 1), ('expanding_no_triad', 1), ('expanding_no_triad', 3)]**.
Sustained recovery is the first step t >= 43 with PR-AUC >= 0.50 that stays there for the next 2 evaluated steps; steps [45, 46] (fewer than 10 illicit labels) are skipped, so the evaluated steps are [43, 44, 47, 48, 49] and the last possible start is 47.

### Lag curve (static vs expanding, pooled 43-49)

| L | static PR-AUC | expanding PR-AUC [95% CI] | delta vs static [95% CI] | expanding sustained recovery |
| ---: | ---: | :--- | :--- | :--- |
| 1 | 0.0406 | 0.6949 [0.3559, 0.9120] | +0.6543 [+0.3257, +0.8217] | step 47 (+4) |
| 2 | 0.0415 | 0.3878 [0.1771, 0.7184] | +0.3463 [+0.1483, +0.6320] | not reached |
| 3 | 0.0270 | 0.1377 [0.1063, 0.4949] | +0.1107 [+0.0842, +0.4322] | not reached |
| 5 | 0.0398 | 0.1050 [0.0489, 0.3614] | +0.0652 [+0.0179, +0.2805] | not reached |

`expanding` first falls below the restored level at L = 2 (descriptive; it is not part of the verdict rule).

### Why (measured, not inferred)

* `expanding` at L=1: mean hold-out PR-AUC over decision steps 43-49 is 0.4990, mean realised per-step PR-AUC is 0.5448. Across those steps the Pearson correlation between hold-out and realised PR-AUC is -0.57 excluding the noise steps (n = 5), -0.12 on all steps (n = 7) and -0.89 at L=3. This is a descriptive finding about drift (the hold-out is the latest arrived labels, whose relationship to the next step is itself shifting), not a defect in the selection code.
* By step 49 the expanding model holds 4489 positives, of which only 113 come from the post-43 regime.
* Triad share of split gain in `expanding`: 0.029 (steps 35-42) vs 0.010 (steps 43-49).
* Removing the triad changes pooled PR-AUC by +0.0037 (35-42) and -0.0205 (43-49) at L=1; paired 95% interval on 43-49: [-0.0683, +0.0329].

### Operating rule: top-K alerts per step (primary) vs the validation threshold

Pooled over 43-49, labeled transactions only. Top-K is the primary mode: it fixes the analyst workload; the validation threshold is the comparison and its alert volume is whatever the threshold happens to give.

| Strategy (L=1) | Rule | alerts/step | precision | recall | F1 | max possible recall | steps with zero alerts | recall 35-42 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static` | top20 | 20.0 | 0.029 | 0.024 | 0.026 | 0.633 | 0 | 0.175 |
| `static` | top50 | 50.0 | 0.043 | 0.089 | 0.058 | 0.964 | 0 | 0.397 |
| `static` | top100 | 100.0 | 0.089 | 0.367 | 0.143 | 1.000 | 0 | 0.682 |
| `static` | validation_threshold | 13.0 | 0.033 | 0.018 | 0.023 | - | 0 | 0.859 |
| `expanding` | top20 | 20.0 | 0.507 | 0.420 | 0.460 | 0.633 | 0 | 0.175 |
| `expanding` | top50 | 50.0 | 0.391 | 0.811 | 0.528 | 0.964 | 0 | 0.402 |
| `expanding` | top100 | 100.0 | 0.223 | 0.923 | 0.359 | 1.000 | 0 | 0.689 |
| `expanding` | validation_threshold | 17.3 | 0.752 | 0.538 | 0.628 | - | 1 | 0.859 |

### Finding: validation-threshold instability

* `expanding` at L=1, steps 43-49, threshold per step: 43:0.810, 44:0.620, 45:0.360, 46:0.360, 47:0.360, 48:0.140, 49:0.075 (range 0.075 to 0.810, 5 distinct values); share of rows flagged per step: 43:0.000, 44:0.005, 45:0.001, 46:0.001, 47:0.006, 48:0.070, 49:0.153; steps with zero alerts: 1.
* `expanding` at L=3, 43-49: threshold range 0.360 to 0.810; steps with zero alerts: 0.
* `static` keeps one threshold (0.715) and flags 0.001 to 0.027 of rows per step over 43-49; steps with zero alerts: 0.
* Per-step values for every strategy and lag are in `threshold_instability.csv`.

### Monitor-triggered vs Phase 8a (per-step agreement, `monitor_concordance.csv`)

| L | scope | steps agreeing | Phase 8a fires missed | lag-safe fires Phase 8a did not make | first fire in 43-49 (lag-safe / label-free / label-dependent / Phase 8a) |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 1 | 43-49 | 5 of 7 | 2 | 0 | 43 / 43 / 44 / 43 |
| 1 | 35-49 | 12 of 15 | 2 | 1 | 43 / 43 / 44 / 43 |
| 3 | 43-49 | 3 of 7 | 4 | 0 | 43 / 43 / - / 43 |
| 3 | 35-49 | 10 of 15 | 4 | 1 | 43 / 43 / - / 43 |

### Freeze for the dashboard and API: `expanding`

`expanding` is the highest-ranked qualifying retraining policy at L=1. Serving it needs a retraining job, a new model artifact and an updated consistency gate; it is not an API-only change.

**Dashboard should show**

* A triage panel as the primary operating mode: precision@K and recall@K for K = 20, 50, 100 per step, drawn against the maximum possible value for that step's positive count.
* The validation threshold as a secondary indicator only, shown per step with the share of rows it flagged, because its per-step instability is measured above.
* The label-delay state: the newest step whose labels have arrived, and the model's trained-through step.
* A noise badge on any step with fewer than 10 illicit labels; no per-step PR-AUC headline.
* The existing `confidence_context` banner (reliable / degraded / unknown) and rolling prevalence.
* A model-freshness widget (trained-through step, steps since retrain) and the recovery curve.
* The human-review queue stays, because recovery is not uniform across strategies and lags.

### Limits of this evidence

* 43-49 holds 169 positives across one regime change; steps 45 and 46 hold 5 and 2 and are not interpreted.
* Intervals are a cluster bootstrap over time steps; with only 7 steps in 43-49 they are approximate and wide. The row-level bootstrap treats rows inside a step as independent and is reported only for comparison.
* Strategy 5 is a lag-safe re-implementation of the Phase 8a rules; the production engine was not imported. Its label-free channels use constants declared before the run (not tuned on 43-49) and have not been validated outside this dataset.
* Extra lags L = 2, 5 were run for static and expanding only and are bootstrapped on 43-49 only.
* The hold-out rule differs from `xgboost_v2`'s fixed 1-24 / 33-34 split; `v2_exact_control` ties the harness to it.
* The L=3 `static` model is trained through step 32, not 34, to respect label arrival.