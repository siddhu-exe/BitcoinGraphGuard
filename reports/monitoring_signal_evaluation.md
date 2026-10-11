### Signal evaluation: which alarms were worth listening to

False-alarm rate is computed on steps 35-42 (model healthy); detection lead is the first
step in 43-49 where the signal alerted minus 43 (0 = on time at the drift step, positive = late).
Label-dependent channels cannot alert at step 43 under the one-step label delay.

| Signal | Labels | False alarms 35-42 | First alert 43-49 | Lead vs 43 | Alerts 43-49 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `triad_feature_drift` | no | 1/8 (12%) | never | n/a | 0 |
| `local_feature_shift` | no | 1/8 (12%) | 46 | +3 | 1 |
| `score_shift_psi` | no | 0/6 (0%) | 43 | +0 | 5 |
| `rolling_prevalence` | yes | 4/8 (50%) | 45 | +2 | 5 |
| `performance_crash` | yes | 0/7 (0%) | 44 | +1 | 6 |
| `retraining_trigger` | yes | 7/8 (88%) | 43 | +0 | 7 |
| `adversarial_validation` | no | 2/2 (100%) | 49 | +6 | 1 |
| `score_ks` | no | 2/7 (29%) | 44 | +1 | 5 |
| `score_positive_rate` | no | 1/7 (14%) | 43 | +0 | 6 |

**What did NOT work**

- `rolling_prevalence` alerted on 4/8 healthy steps (50%) - its alarms cannot be acted on alone.
- `retraining_trigger` alerted on 7/8 healthy steps (88%) - its alarms cannot be acted on alone.
- `adversarial_validation` alerted on 2/2 healthy steps (100%) - its alarms cannot be acted on alone.
- `triad_feature_drift` never alerted in 43-49 - it carries no onset signal here.

This is one regime change on one dataset; the label-free score channels use fixed, pre-declared thresholds and are not tuned on 43-49.
