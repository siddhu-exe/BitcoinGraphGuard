# XGBoost Baseline (Phase 2)

**Phase:** 2 — classical supervised baseline
**Implementation:** `notebooks/02_xgboost.ipynb` (Google Colab / Kaggle — never the laptop)
**Status:** notebook authored, executed end to end on Google Colab on **2026-09-25**, and its
results recorded below straight from the exported artifacts in `xgboost/`. Phase 2 awaits review
sign-off; GraphSAGE has not been started.
**Last updated:** 2026-09-25

> **Provenance of the numbers.** Every figure in this document is read from
> `xgboost/metrics.json`, `xgboost/temporal_metrics.csv`, `xgboost/model_comparison.csv`,
> `xgboost/feature_importance.csv` and `xgboost/xgboost_digest.txt`, all produced by that run
> (xgboost 3.4.1). The test-period metrics were independently recomputed from
> `xgboost/predictions.csv` and reproduce exactly. The copy of the notebook committed to this
> repository carries no stored outputs, so the exported artifacts are the record of the run.

## Objective

XGBoost is the first serious supervised model in the project, and its job is to set a bar rather
than to win.

The model progression is `simple baseline -> logistic regression -> XGBoost -> GraphSAGE ->
RGCN/HGT`. Everything after XGBoost claims to extract value from **graph structure**. That claim is
only meaningful against a leakage-safe, reproducible number for what the **transaction features
alone** achieve. If GraphSAGE does not beat this baseline under an identical temporal protocol, the
added complexity is not paying for itself.

The notebook therefore also contains the two steps below XGBoost — a trivial prevalence baseline
and a Logistic Regression — so the contribution of nonlinearity is separated from the contribution
of the features themselves.

## Data & Features

Real Elliptic++ transactions only. No synthetic data is used for any reported result.

**Population.** Class 1 (illicit) and class 2 (licit). Class 3 means *unlabeled* and is never a
training or evaluation example — unknown is not licit.

**Features (166).**

| Group | Count | Used |
| :--- | ---: | :--- |
| `Local_feature_*` | 93 | yes |
| `Aggregate_feature_*` | 72 | yes |
| 17 domain columns | 17 | **no** |
| `has_addresses` | 1 | yes |

**Why the domain block is excluded.** The EDA showed all 17 domain columns have a
`Local_feature_*` counterpart at `|r| ≈ 0.98 – 1.00` (`eda/txs_domain_redundancy.csv`); they are
re-expressions rather than independent evidence. They are also the only columns carrying missing
values: 17 × 965 = 16,405 blank cells, on exactly the 965 transactions with no address links.
Including them would force a masking decision for data that adds nothing the `Local_feature_*`
block does not already carry. The blanks are therefore never imputed as `0` and never need to be.

**Why `has_addresses` is added.** It records whether a transaction appears in `AddrTx` or `TxAddr`.
That is a fact about the transaction's own inputs and outputs, known at prediction time, and it
lets the model treat the 965 address-less transactions as a distinct population. It is *not* a
graph statistic: no degree, no component size, no neighbour aggregate is used anywhere.

**No feature scaling and no `log1p`.** Trees split on order, so monotone rescaling cannot change a
split. The EDA measured skewness up to 148 and heavy mass at zero on the fee/degree columns; that
is a reason to prefer a tree, not a reason to transform. Any transformation would be introduced as
a measured experiment, and it was not.

## Leakage Prevention

The notebook implements a leakage audit as a table plus assertions, not as prose. Each feature
group is checked against six questions: available at prediction time, uses future steps, uses
labels, uses test data, computed on the complete graph, aggregates future observations. The table
is saved to `xgboost/leakage_audit.csv`.

Structural decisions that follow from it:

* **The 17 domain columns are dropped at parse time** (`usecols`), so they never enter memory as
  features.
* **`txId` and `Time step` are never features.** `Time step` defines the split; as a feature it
  would hand the model the experiment design.
* **No graph statistics.** The EDA computed degrees, components and hubs over the whole
  transductive graph, including future edges and test nodes; those columns are excluded by
  construction. The address edge lists are read for membership only, which is why `has_addresses`
  is admissible and a degree is not.
* **No target encoding, no label-derived feature.**
* **Preprocessing statistics are fitted on the fitting window only.** The Logistic Regression
  pipeline's imputer and scaler see only the rows it trains on, then are applied unchanged to
  validation and test.
* **Assertions, not assumptions.** The notebook checks that the excluded columns are absent from
  the feature matrix, and that no single feature separates the labels implausibly well on the
  fitting window — a near-perfect single-feature AUC would be the signature of a leaked column.

The provenance caveat is stated rather than hidden: the 165 Local/Aggregate columns come from the
published dataset, and their step-boundedness is assumed, not re-derived here. The recommendation
for the GNN phases is that every graph feature be explicitly step-bounded and audited the same way.

## Experimental Setup

| Window | Steps | Role |
| :--- | :--- | :--- |
| fit | 1–24 | early stopping and threshold selection |
| validation | 25–34 | historical validation |
| train | 1–34 | final model refit |
| test | 35–49 | final evaluation, untouched until section 9 |
| test_early | 35–42 | temporal sub-window |
| test_drift | 43–49 | secondary recent-drift window |

**Why fit stops at 24.** The adopted protocol names 25–34 as validation, and 25–34 lies *inside*
the training period. Training on 1–34 while early-stopping on 25–34 would stop on rows the model had
already seen, and a threshold chosen there would be optimistic. The model is therefore first fitted
on 1–24 and validated on 25–34 to select the number of trees and the operating threshold; it is then
refitted on the full 1–34 and the frozen threshold is applied to the test period.

`35–42` and `43–49` partition the test period exactly, so both are directly comparable to `35–49`.
`43–49` was rejected as the primary test in the EDA (169 illicit transactions make a noisy
estimate) but it is the low-prevalence regime a deployed model would meet, so it is reported.

Known prevalence shift, from the EDA: 11.58% illicit among labeled transactions in 1–34, 6.50% in
35–49, 2.53% in 43–49, and 19.77% on the 25–34 validation window. The last of these is a caveat,
not a detail — see Limitations.

## Models

**1. Simple baseline.** A constant score equal to the training prevalence, plus a flag-everything
detector. A constant score has ROC-AUC 0.5 and a PR-AUC equal to the illicit prevalence *of the
window it is evaluated on* — 6.50% on the test period. The baseline therefore gets harder to beat as
prevalence falls, which is the correct behaviour.

**2. Logistic Regression.** Median imputation, `StandardScaler`, `class_weight="balanced"`,
`max_iter=2000`. Answers how much signal the 166 features carry without tree interactions. Fitted on
the fitting window, with the pipeline applied unchanged to validation and test.

**3. XGBoost.** `learning_rate=0.05`, `max_depth=6`, `min_child_weight=5`, `subsample=0.8`,
`colsample_bytree=0.8`, `reg_lambda=1.0`, `tree_method="hist"`, `eval_metric="aucpr"`, up to 500
trees with 50-round early stopping on the validation window, `random_state=42`.
`scale_pos_weight = negatives/positives` computed on the training window only — it changes the
gradient weighting, never the data. No resampling, and the test distribution is left exactly as the
world produced it. The surviving tree count from the selection run is reused for the refit on 1–34.
The full hyperparameter record is written to `xgboost/xgb_hyperparameters.json`.

In the executed run the selection model kept **499 of the 500 permitted trees**: the best validation
PR-AUC was reached at the last tree tried, so early stopping never actually fired and the cap, not
the data, ended training. The 500-tree budget is therefore binding and the model is under-trained —
see Limitations.

**Operating point.** Threshold-independent metrics (PR-AUC, ROC-AUC) are reported first. The
precision/recall/F1 operating point uses the threshold that maximises F1 on the validation window,
frozen, then applied to the test period. It is never tuned against test results or test prevalence.
A second threshold derived from training prevalence is reported as a sensitivity check because the
validation window's illicit rate (19.77%) is far from the test period's (6.50%).

## Results

Executed on Google Colab, 2026-09-25 (xgboost 3.4.1). The test period 35–49 holds 16,670 labeled
transactions, 1,083 of them illicit — a prevalence of **6.4967%**.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Threshold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prevalence (constant score) | 0.0650 | 0.5000 | 0.0650 | 1.0000 | 0.1220 | 0.1158 |
| Majority class (all licit) | 0.0650 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.01 |
| Logistic Regression | 0.2917 | 0.8828 | 0.2715 | 0.6473 | 0.3825 | 0.880 |
| XGBoost | **0.8007** | **0.9317** | **0.8400** | 0.7368 | **0.7850** | 0.515 |

Confusion matrices at the frozen operating point:

| Model | TP | FP | FN | TN |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 701 | 1,881 | 382 | 13,706 |
| XGBoost | 798 | 152 | 285 | 15,435 |

Every value is read from `xgboost/metrics.json` and `xgboost/model_comparison.csv`, and was
independently recomputed from `xgboost/predictions.csv`; the recomputation matches to four decimals.

**Lift over the trivial baseline.** A constant score has PR-AUC equal to the prevalence of the
window it is scored on, so on test it is 0.0650 and ROC-AUC is exactly 0.500 — the baseline carries
no ranking information. XGBoost lifts PR-AUC to 0.8007, a **12.32×** multiple of prevalence;
Logistic Regression reaches 0.2917, a **4.49×** multiple. Both models add real signal, and the gap
between them is the value of nonlinear interactions over the same 166 features.

**The operating point.** Threshold-independent metrics come first — PR-AUC is the primary metric,
with ROC-AUC as a secondary view. The precision/recall/F1 row above is then taken at a **single
frozen threshold**: F1-maximising on validation 25–34 under the selection model fitted on 1–24
(validation F1 0.9546). It was fixed before the test period was scored, and never re-derived from
test results or test prevalence. A training-prevalence threshold of 0.1158 is reported in
`metrics.json` as a sensitivity check only.

**The asymmetry.** At the frozen point XGBoost makes 285 false negatives against 152 false
positives — roughly 1.9 missed illicit transfers per false alarm — and recall is 0.7368, so about
one illicit transaction in four goes unflagged. For fraud investigation that is the more expensive
side of the trade, and it is a policy choice, not a property of the model: lowering the threshold
trades false negatives for false positives along the same PR curve. The trade-off should be set
from a stated cost structure, not from a convenient flag rate.

## Temporal Results

Read from `xgboost/temporal_metrics.csv`. The two sub-windows partition the test period exactly
(9,983 + 6,687 = 16,670). Logistic Regression is shown alongside because the ranking changes here.

| Window | Steps | Labeled | Illicit | Prevalence | XGB PR-AUC | XGB ROC-AUC | LR PR-AUC | LR ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | 35–49 | 16,670 | 1,083 | 0.0650 | 0.8007 | 0.9317 | 0.2917 | 0.8828 |
| Early | 35–42 | 9,983 | 914 | 0.0916 | **0.9211** | 0.9729 | 0.4230 | 0.9081 |
| Recent | 43–49 | 6,687 | 169 | 0.0253 | **0.0423** | 0.6886 | 0.0547 | 0.7543 |

At the same frozen threshold of 0.515, XGBoost's precision / recall / F1 are 0.9179 / 0.8687 /
0.8926 on 35–42, then collapse to 0.0471 / 0.0237 / 0.0315 on 43–49 (4 true positives out of 169).

**The 43–49 window is the result to carry forward.** Prevalence there is 2.53%, roughly a quarter
of the 35–42 rate, and XGBoost's PR-AUC falls to 0.0423 — a lift of only **1.67×** over that
window's own prevalence, against 10.1× on 35–42 and 12.3× on the full test period. ROC-AUC drops
from the high 0.97s to 0.689. Those two metrics are threshold-free, so the loss is in the ranking
itself, not merely in where the cut sits.

Two caveats stated rather than glossed. The operating point was frozen on a window with 19.77%
prevalence, so part of the precision/recall collapse on 43–49 is mis-calibration rather than lost
signal — which is exactly why the threshold-free numbers are the ones quoted above. And Logistic
Regression, despite being far weaker on the full period, edges XGBoost out on both threshold-free
metrics in 43–49 (PR-AUC 0.0547 vs 0.0423, ROC-AUC 0.7543 vs 0.6886). Both models are close to
useless in that regime and the gap is well within what a single split can resolve, but it does mean
the nonlinearity advantage does not survive the regime shift.

This is a **temporal-generalization measurement, not drift monitoring**. Formal drift detection and
retraining triggers belong to Phase 6.

## Error Analysis

From `xgboost/errors_by_step.csv`, `xgboost/errors_by_address_linkage.csv` and
`xgboost/error_analysis.json`. Transaction identifiers are masked in the notebook, and only
aggregates are reported here.

**Errors are concentrated in the recent window.** The per-step table shows 35–42 working well
(per-step precision 0.72–0.96, recall 0.68–1.00) and 43–49 failing: steps 43, 45, 47 and 48 each
produce **zero** true positives at 0/24, 0/5, 0/22 and 0/36 respectively, and step 49 finds 1 of 56.
False positives continue at a low but non-zero rate across those same steps, so the model is not
silent there — it is ranking the wrong transactions.

**The two error types look different.** The 285 false negatives have a median XGBoost score of
**0.0024**: they are confidently missed, not marginally missed. The 152 false positives have a
median score of **0.7115**, much closer to the 0.515 boundary, so they are the ones a threshold
change would move first. Because the misses are not stacked just below the cut, buying recall by
lowering the threshold costs false alarms at a worse rate than the median FP score alone suggests.

**Address-less transactions.** 324 test transactions have no address links. All 324 are licit, so
recall is undefined for the group and none of them are flagged. That is not a property of the test
window: across the whole dataset **every one of the 4,545 illicit transactions has at least one
address link**, while the 965 address-less transactions split 519 licit / 446 unknown. So
`has_addresses = 0` can never mark a positive example — it is a weak, one-sided licit signal, and
there is no address-less fraud in the labeled data to analyse. The feature remains legitimate (it
describes the transaction's own inputs and outputs, is known at prediction time and is not derived
from labels), but it carries no positive evidence and should not be expected to do work.

## Feature Importance

From `xgboost/feature_importance.csv`. Gain is XGBoost's own split-gain attribution; permutation
importance is the mean drop in validation PR-AUC when a feature is shuffled (4,000 validation rows,
5 repeats). **Neither is a causal statement** — a feature can rank highly through correlation,
through cutting off a convenient subpopulation, or because another feature redundantly covers it.

Top by gain: `Local_feature_53` (534.5), `Local_feature_40` (320.2), `Local_feature_46` (314.6),
`Aggregate_feature_7` (275.9), `Local_feature_90` (275.9). Top by permutation: `Local_feature_53`
(0.000538), `Local_feature_2` (0.000117), `Local_feature_3` (0.000054), `Aggregate_feature_13`
(0.000043), `Local_feature_59` (0.000042).

**Agreement with the EDA.** `Local_feature_53` ranks first under both measures, and the EDA had
independently found it the single strongest point-biserial correlate of the illicit label at
`r = −0.26` — weak and diffuse. The top-5 gain list is dominated by `Local_feature_*` with two
`Aggregate_feature_*` entries, which matches the EDA's redundancy finding that the domain block is
a re-expression of the Local block. The model is exploiting the same spread-out signal the EDA
described, not some hidden strong feature.

**Read the permutation column with care.** Its values are tiny in absolute terms, 55 of the 166
features score exactly 0.0, and many ranks are tied. That is the expected behaviour of permutation
importance on a zero-inflated, strongly correlated feature set: shuffling one member of a redundant
pair barely moves the predictions, so the measured drop understates that feature's contribution.
`Local_feature_40` shows the disagreement outright — rank 2 by gain, rank 108 by permutation. The
rows were also drawn as a contiguous 4,000-row slice of the validation window rather than a random
sample, which adds a mild time bias. For this feature set, gain is the more stable of the two
columns; permutation importance is a weak tie-breaker.

## Conclusion

Executed 2026-09-25 against the real Elliptic++ dataset. The six questions, answered from the
artifacts:

1. **Does XGBoost beat the prevalence baseline?** Yes, decisively. PR-AUC 0.8007 against a 0.0650
   constant-score baseline — a 12.3× lift — with ROC-AUC 0.9317 against 0.500.
2. **Does it beat Logistic Regression?** Yes, by a wide margin: PR-AUC 0.8007 vs 0.2917 and F1 0.785
   vs 0.383 on identical features and an identical protocol. The 166 transaction features carry
   substantial *nonlinear* structure, not merely linear signal. That matters for the next phase:
   the headroom GraphSAGE must find is not simply "add nonlinearity".
3. **Does it survive the 43–49 low-prevalence regime?** No. PR-AUC 0.9211 on 35–42 falls to 0.0423
   on 43–49 — a lift of only 1.67× over that window's 2.53% prevalence — and Logistic Regression
   edges XGBoost out there on both threshold-free metrics. The aggregate test number is carried by
   the earlier part of the period.
4. **Which error dominates?** False negatives: 285 against 152 false positives, and they are
   confident misses (median score 0.0024). Recall is 0.7368, so roughly one illicit transaction in
   four is missed. For fraud that is the more expensive direction, but the threshold should be moved
   deliberately against a stated cost structure, not toward a target flag rate.
5. **Do the important features agree with the EDA?** Yes. `Local_feature_53` tops both gain and
   permutation importance, and it is the EDA's strongest — still weak — correlate at `r = −0.26`;
   the Local block dominates the Aggregate block, consistent with the measured domain/local
   redundancy.
6. **Do address-less transactions behave differently?** There is nothing to measure. Every illicit
   transaction in the dataset has at least one address link, so the 965 address-less transactions
   are never positive in the labeled data.

**What this establishes for the GNN phases.** A graph-free model over 166 tabular features reaches
**PR-AUC 0.8007 / ROC-AUC 0.9317** on the primary test period 35–49 at a validation-frozen
threshold. That is the number GraphSAGE, and then RGCN/HGT, must beat under the identical protocol:
same windows, same population, same threshold discipline, same metric implementation. Two things
make that bar informative rather than decorative — the margin over Logistic Regression shows the
features are already being exploited well, and the 43–49 collapse marks the regime where a
structure-aware model has the most room to differ. Any GNN result must therefore be reported both
on 35–49 and on 43–49; an aggregate-only improvement would hide exactly the failure this baseline
exposes.

## Limitations

* **The tree budget was binding, so XGBoost is under-trained.** The selection run kept 499 of a
  500-tree cap, meaning the best validation PR-AUC sat at the final tree and the 50-round patience
  never triggered. The notebook's check "early stopping stopped before the tree cap" passes only
  because 499 < 500; it should be read as a warning, not a confirmation. The reported 0.8007 is a
  floor rather than a ceiling, and a re-run with a larger cap (or an explicit
  learning-rate / tree-count sweep) should be recorded before this number is treated as final.
* **`has_addresses` cannot contribute positive evidence.** All 4,545 illicit transactions have at
  least one address link, so `has_addresses = 0` is only ever associated with licit or unlabeled
  rows. The feature is admissible, but it is a one-sided licit indicator and carries no information
  about illicit activity; it also makes the address-less subpopulation unanalysable rather than
  merely small.
* **No graph structure at all.** The model scores each transaction in isolation. The EDA's finding
  that transaction components are confined to a single time step — so cross-step signal travels
  through address nodes — describes exactly what a per-row model cannot use: money-flow chains,
  addresses shared between licit and illicit transactions, and multi-hop neighbourhoods.
* **Threshold transfer across prevalence shifts.** The operating point is selected on a window with
  19.77% illicit prevalence and applied to a period with 6.50%. It must be re-derived on data that
  matches the deployment regime, and never on the test set.
* **Unlabeled data is discarded, not used.** 157,205 transactions are class 3. Excluding them is
  required for a clean supervised baseline, but it throws away the majority of the graph, and
  semi-supervised or contrastive approaches are not explored here.
* **Transductive protocol.** Both periods come from the same 49-step window; nothing here measures
  generalization to unseen nodes, which is Phase 4's concern.
* **Single split, no significance testing.** Differences between models are point estimates on one
  temporal split; no confidence intervals are computed, so small gaps should not be over-read.
* **`Aggregate_feature_*` provenance is assumed.** Their contents are not re-derived from the raw
  blockchain data in this notebook, so their step-boundedness is taken on trust rather than proven.
* **No calibration.** Probabilities are used for ranking. Nothing here establishes that a score of
  0.7 means a 70% chance of illicit activity.
