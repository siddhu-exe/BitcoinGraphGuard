# XGBoost Baseline (Phase 2)

**Phase:** 2 — classical supervised baseline
**Implementation:** `notebooks/02_xgboost.ipynb` (Google Colab / Kaggle — never the laptop)
**Status:** notebook executed end to end on Google Colab on **2026-09-25**, in two passes: the
original 500-tree baseline, then a controlled optimisation pass that lifted the tree budget, ran a
seeded 20-trial randomised search and ablated two feature decisions. Both passes are recorded below;
the original result is kept as the historical reference and is not restated as current. Phase 2
awaits review sign-off; GraphSAGE has not been started.
**Last updated:** 2026-09-25

> **Provenance of the numbers.** Every figure in this document is read from
> `xgboost/metrics.json`, `xgboost/temporal_metrics.csv`, `xgboost/model_comparison.csv`,
> `xgboost/feature_importance.csv`, `xgboost/optimization_results.csv`,
> `xgboost/selected_hyperparameters.json` and `xgboost/xgboost_digest.txt`, all produced by the
> optimisation pass (xgboost 3.4.1). The test-period metrics were independently recomputed from
> `xgboost/predictions.csv` and reproduce exactly. The copy of the notebook committed to this
> repository carries no stored outputs, so the exported artifacts are the record of the run.

> **Follow-up: Phase 2b (2026-09-29).** The 43–49 collapse recorded below was investigated in
> `notebooks/02_xgboost_v2.ipynb` — late low-prevalence validation slice, class-weight and objective
> sweep, feature-drift audit, recency weighting and a feature-provenance ledger — with artifacts in
> `results/xgboost_v2/` and the full record in `docs/XGBOOST_V2.md`. It changed nothing here: the
> frozen Phase 2 numbers below remain the tabular reference, and Phase 2b's best configuration moved
> 43–49 by only **+0.0056 PR-AUC** (0.0427 → 0.0483) while confirming the cause as covariate drift.

> **Optimisation pass executed (2026-09-25).** The first run's 500-tree budget ran out before its
> optimum, so 0.8007 was a floor rather than a converged estimate. The notebook now lifts the budget
> to 1500 trees with 100 rounds of patience, runs a seeded 20-trial randomised search scored **on
> validation PR-AUC only**, ablates `has_addresses` against a pre-registered margin, and tests one
> label-free, fit-window-only redundancy reduction. The result: the tree budget was **not** what
> limited test performance. PR-AUC moved 0.8007 → 0.8013 (+0.0006), ROC-AUC fell 0.9317 → 0.9281
> and F1 fell 0.7850 → 0.7799. Read the two XGBoost rows as one baseline and a re-tuning of it.

## Objective

XGBoost is the first serious supervised model in the project, and its job is to set a bar rather
than to win.

The model progression is `simple baseline → logistic regression → XGBoost → GraphSAGE →
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

**Feature set: 165 columns.**

| Group | Count | Used |
| :--- | ---: | :--- |
| `Local_feature_*` | 93 | yes |
| `Aggregate_feature_*` | 72 | yes |
| 17 domain columns | 17 | **no** |
| `has_addresses` | 1 | **no** — dropped by the ablation below |

**Why the domain block is excluded.** The EDA showed all 17 domain columns have a
`Local_feature_*` counterpart at `|r| ≈ 0.98 – 1.00` (`eda/txs_domain_redundancy.csv`); they are
re-expressions rather than independent evidence. They are also the only columns carrying missing
values: 17 × 965 = 16,405 blank cells, on exactly the 965 transactions with no address links.
Including them would force a masking decision for data that adds nothing the `Local_feature_*`
block does not already carry. The blanks are therefore never imputed as `0` and never need to be.

**`has_addresses`: built, tested, dropped.** The column records whether a transaction appears in
`AddrTx` or `TxAddr` — a fact about its own inputs and outputs, known at prediction time, and not a
graph statistic. It was ablated against a pre-registered adoption margin of +0.005 validation
PR-AUC and scored **+0.000933** (165 features 0.98713 with 582 trees, 166 features 0.98807 with 840
trees; `metrics.json:ablation_has_addresses`). That is below the margin, so the flag is **not** in
the final model and the cleaner 165-column set is used. The reason was known before the ablation
ran: all 4,545 illicit transactions have at least one address link, so `has_addresses = 0` can only
ever mark licit or unlabeled rows. It is a one-sided licit indicator, it cannot contribute positive
evidence, and it makes the address-less subpopulation unanalysable rather than merely small.

**No feature scaling and no `log1p`.** Trees split on order, so monotone rescaling cannot change a
split. The EDA measured skewness up to 148 and heavy mass at zero on the fee/degree columns; that
is a reason to prefer a tree, not a reason to transform. Any transformation would be introduced as
a measured experiment, and none was.

**One redundancy ablation, label-free.** EDA reported substantial redundancy between features. One
controlled test compared the 165-column set against a 116-column set built by dropping, within the
fit window only, one member of every pair above `|r| = 0.98` (`metrics.json:ablation_redundancy`).
Validation PR-AUC moved **+0.000255** — inside the same pre-registered margin — so the full set is
retained. The reduction deliberately does **not** use `eda/txs_feature_summary.csv:corr_with_illicit`,
which was computed over all 203,769 rows and would have imported test-period label information.

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
`max_iter=2000`. Answers how much signal the 165 features carry without tree interactions. Fitted on
the fitting window, with the pipeline applied unchanged to validation and test.

**3. XGBoost, original configuration (historical reference).** `learning_rate=0.05`, `max_depth=6`,
`min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0`, `reg_lambda=1.0`,
`tree_method="hist"`, `eval_metric="aucpr"`, up to 500 trees with 50-round early stopping,
`random_state=42`. In this run it kept **499 of the 500 permitted trees** and its best validation
PR-AUC sat on the second-to-last round, i.e. training consumed the entire budget instead of
converging (`metrics.json:xgboost_reference`, best iteration 498). This is the run behind the
recorded 0.8007.

**4. XGBoost, optimised configuration.** The reference config was re-searched over a seeded,
bounded space — `learning_rate` {0.02, 0.05, 0.1}, `max_depth` {3, 4, 5, 6, 8},
`min_child_weight` {1, 3, 5, 10}, `subsample` {0.7, 0.85, 1.0}, `colsample_bytree` {0.6, 0.8, 1.0},
`gamma` {0, 0.1, 0.5, 1} — with `sklearn.ParameterSampler`, `random_state=42`, **20 trials**, ranked
by validation PR-AUC on 25–34 and by nothing else. Selection metric, search space, trial count, seed
and the full per-trial table are in `selected_hyperparameters.json` and `optimization_results.csv`.
The winner: `learning_rate=0.05`, `max_depth=6`, `min_child_weight=3`, `subsample=0.7`,
`colsample_bytree=0.6`, `gamma=0.5`, `reg_lambda=1.0`, budget **1500 trees** with **100 rounds of
patience**. It kept **582 trees** (best iteration 581), so the budget no longer binds: training
stopped because the validation metric stopped improving, with more than 800 trees still unused.

**What the search did and did not buy.** Validation PR-AUC barely moved across the entire search —
0.9801 at the worst trial to 0.9881 at the best, against 0.9861 for the reference
(`optimization_results.csv`). The validation window is close to saturated for this feature set, so
the search mostly redistributed capacity rather than finding a better fit, and the test numbers
follow. One wrinkle stated rather than hidden: the search evaluated candidates with the
`has_addresses` column present (166 features, the best trial's 840 trees), while the final model
uses the 165-column set chosen by the ablation, so the final model's own validation PR-AUC is
0.98713 rather than the search's 0.98807.

**`scale_pos_weight`.** `negatives/positives` computed on the fitting window only — 9.638 on 1–24,
7.635 on the 1–34 refit. It changes the gradient weighting, never the data. No resampling, no
oversampling, and the test distribution is left exactly as the world produced it. Every search trial
used the fit-window value.

**Operating point.** Threshold-independent metrics (PR-AUC, ROC-AUC) are reported first. The
precision/recall/F1 operating point uses the threshold that maximises F1 on the validation window
(0.435 for the optimised model, validation F1 0.9605), frozen, then applied to the test period. It is
never tuned against test results or test prevalence.

## Results

Executed on Google Colab, 2026-09-25 (xgboost 3.4.1). The test period 35–49 holds 16,670 labeled
transactions, 1,083 of them illicit — a prevalence of **6.4967%**.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Threshold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prevalence (constant score) | 0.0650 | 0.5000 | 0.0650 | 1.0000 | 0.1220 | 0.1158 |
| Majority class (all licit) | 0.0650 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.01 |
| Logistic Regression | 0.2917 | 0.8828 | 0.2715 | 0.6473 | 0.3825 | 0.880 |
| XGBoost baseline (500-tree cap) | 0.8007 | **0.9317** | **0.8400** | 0.7368 | **0.7850** | 0.515 |
| XGBoost optimised (1500-tree budget) | **0.8013** | 0.9281 | 0.8271 | **0.7378** | 0.7799 | 0.435 |

Confusion matrices at each model's frozen operating point:

| Model | TP | FP | FN | TN |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 701 | 1,881 | 382 | 13,706 |
| XGBoost baseline | 798 | 152 | 285 | 15,435 |
| XGBoost optimised | 799 | 167 | 284 | 15,420 |

Every value is read from `xgboost/metrics.json` and `xgboost/model_comparison.csv`, and was
independently recomputed from `xgboost/predictions.csv`; the recomputation matches to four decimals,
the confusion matrices close to 16,670 rows, and the labels match `score >= threshold` exactly.

**The optimisation is not an improvement worth claiming.** PR-AUC rises by **+0.0006**, which is
noise on 1,083 positive examples, while ROC-AUC falls by 0.0036 and F1 by 0.0052. The optimised
model moves one transaction from false negative to true positive and adds 15 false positives. The
useful conclusion is not that it is better: it is that **the 500-tree budget was not the binding
constraint on test performance.** Validation PR-AUC was already 0.9861 at the cap and 0.9881 at the
best of 20 trials, and that 0.002 of validation headroom moved the test metric by 0.0006. The
baseline is limited by the features and the temporal shift, not by model capacity — which is
exactly the question this pass was run to answer.

**Lift over the trivial baseline.** A constant score has PR-AUC equal to the prevalence of the
window it is scored on, so on test it is 0.0650 and ROC-AUC is exactly 0.500 — the baseline carries
no ranking information. The optimised model lifts PR-AUC to 0.8013, a **12.33×** multiple of
prevalence; Logistic Regression reaches 0.2917, a **4.49×** multiple. Both models add real signal,
and the gap between them is the value of nonlinear interactions over the same features.

**The operating point.** Threshold-independent metrics come first — PR-AUC is the primary metric,
with ROC-AUC as a secondary view. The precision/recall/F1 rows are taken at a **single frozen
threshold per model**, F1-maximising on validation 25–34 under the selection fit on 1–24, fixed
before the test period was scored and never re-derived from test results or test prevalence. The
optimised threshold (0.435) is lower than the baseline's (0.515), and that is the whole of its
recall change: +0.0010. A training-prevalence threshold of 0.1158 is reported in `metrics.json` as a
sensitivity check only.

**The asymmetry.** At the frozen point the optimised model makes 284 false negatives against 167
false positives — roughly 1.7 missed illicit transfers per false alarm — and recall is 0.7378, so
about one illicit transaction in four goes unflagged. For fraud investigation that is the more
expensive side of the trade, and it is a policy choice, not a property of the model: lowering the
threshold trades false negatives for false positives along the same PR curve. The trade-off should
be set from a stated cost structure, not from a convenient flag rate.

## Temporal Results

Read from `xgboost/temporal_metrics.csv`. The two sub-windows partition the test period exactly
(9,983 + 6,687 = 16,670).

| Window | Steps | Illicit | Prevalence | XGB opt PR-AUC | XGB base PR-AUC | XGB opt ROC-AUC | XGB base ROC-AUC | LR PR-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | 35–49 | 1,083 | 0.0650 | **0.8013** | 0.8007 | 0.9281 | 0.9317 | 0.2917 |
| Early | 35–42 | 914 | 0.0916 | **0.9215** | 0.9211 | 0.9710 | 0.9729 | 0.4230 |
| Recent | 43–49 | 169 | 0.0253 | **0.0427** | 0.0423 | 0.6839 | 0.6886 | 0.0547 |

At its own frozen threshold of 0.435, the optimised model's precision / recall / F1 are 0.9074 /
0.8687 / 0.8876 on 35–42, then collapse to 0.0549 / 0.0296 / 0.0385 on 43–49 (5 true positives out
of 169).

**The 43–49 window is the result to carry forward.** Prevalence there is 2.53%, roughly a quarter of
the 35–42 rate, and PR-AUC falls to 0.0427 — a lift of only **1.69×** over that window's own
prevalence, against 10.06× on 35–42 and 12.33× on the full test period. ROC-AUC drops from 0.9710 to
0.6839. Both metrics are threshold-free, so the loss is in the ranking itself, not merely in where
the cut sits. Re-tuning did not change this: 0.0423 → 0.0427 on 169 positives is the same answer
twice.

Two caveats stated rather than glossed. The operating point was frozen on a window with 19.77%
prevalence, so part of the precision/recall collapse on 43–49 is mis-calibration rather than lost
signal — which is why the threshold-free numbers are the ones quoted. And Logistic Regression,
despite being far weaker on the full period, still edges both XGBoost models out on PR-AUC in 43–49
(0.0547 vs 0.0427) while losing on ROC-AUC (0.7543 vs 0.6839). Both are close to useless in that
regime and the gap is within what one split can resolve, but it does mean the nonlinearity advantage
does not survive the regime shift.

This is a **temporal-generalization measurement, not drift monitoring**. Formal drift detection and
retraining triggers belong to Phase 6.

## Error Analysis

From `xgboost/errors_by_step.csv` and `xgboost/error_analysis.json` for the optimised model
(false negatives 284, false positives 167). Transaction identifiers are masked in the notebook, and
only aggregates are reported here.

**Errors are concentrated in the recent window.** The per-step table shows 35–42 working well
(per-step precision 0.72–0.97, recall 0.68–1.00) and 43–49 failing: steps 43, 45 and 47 each produce
**zero** true positives (0/24, 0/5 and 0/22), step 48 finds 1 of 36 and step 49 finds 1 of 56. False
positives continue at a low but non-zero rate across those same steps (8, 16, 4, 3 and 14), so the
model is not silent there — it is ranking the wrong transactions. Of the 284 false negatives, 120
fall in 35–42 and 164 in 43–49, even though 43–49 holds only 169 illicit transactions: the model
misses almost everything in the recent window.

**The two error types look different.** The 284 false negatives have a median optimised score of
**0.0020**: they are confidently missed, not marginally missed. The 167 false positives have a median
score of **0.6248**, much closer to the 0.435 boundary, so they are the ones a threshold change
would move first. Because the misses are not stacked just below the cut, buying recall by lowering
the threshold costs false alarms at a worse rate than the median FP score alone suggests.

**Address-less transactions.** The earlier `errors_by_address_linkage.csv` is no longer produced.
The notebook now runs that breakdown only if `has_addresses` survives the ablation, and it did not —
the group has no positive examples to measure (all 324 address-less rows in the test period are
licit, and across the whole dataset all 4,545 illicit transactions have an address link, while the
965 address-less transactions split 519 licit / 446 unknown). Repeating a structurally vacuous
analysis would add a table, not evidence. `xgboost/predictions.csv` still carries the flag, so the
subgroup can be reconstructed if a later phase needs it.

## Feature Importance

From `xgboost/feature_importance.csv` for the optimised model (165 features). Gain is XGBoost's own
split-gain attribution; permutation importance is the mean drop in validation PR-AUC when a feature
is shuffled (contiguous 4,000-row validation slice, 5 repeats). **Neither is a causal statement** —
a feature can rank highly through correlation, through cutting off a convenient subpopulation, or
because another feature redundantly covers it.

Top by gain: `Local_feature_53` (366.3), `Local_feature_14` (255.7), `Local_feature_46` (228.0),
`Local_feature_55` (227.3), `Local_feature_5` (224.1). Top by permutation: `Local_feature_53`
(1.07e-4), `Local_feature_2` (9.97e-5), `Local_feature_3` (1.94e-5), `Local_feature_79` (5.56e-6),
`Local_feature_16` (4.17e-6).

**Agreement with the EDA.** `Local_feature_53` ranks first under both measures, and the EDA had
independently found it the single strongest point-biserial correlate of the illicit label at
`r = −0.26` — weak and diffuse. 16 of the top 20 by gain are `Local_feature_*`, which matches the
EDA's finding that the domain block is a re-expression of the Local block. The model is exploiting
the same spread-out signal the EDA described, not some hidden strong feature. Re-tuning reshuffled
the tail of the gain ranking but left the head intact.

**Read the permutation column with care.** Its values are tiny in absolute terms, **90 of the 165
features score exactly 0.0**, and many ranks are tied. That is the expected behaviour of permutation
importance on a zero-inflated, strongly correlated feature set: shuffling one member of a redundant
pair barely moves the predictions, so the measured drop understates that feature's contribution.
`Local_feature_14` shows the disagreement outright — rank 2 by gain, tied at rank 59 by permutation.
The rows were also drawn as a contiguous slice of the validation window rather than a random sample,
which adds a mild time bias. For this feature set, **gain is the primary view**; permutation
importance is a weak tie-breaker.

## Conclusion

The optimisation pass was run to answer one question — was 0.8007 limited by the tree budget? — and
the answer, in the order the questions were asked:

1. **Did lifting the tree budget raise validation PR-AUC?** Barely. The reference reached 0.9861
   with its 500-tree budget exhausted; the best of 20 tuned trials reached 0.9881 on the same
   window, and the final 165-feature model scored 0.98713. Validation is close to saturated for
   these features.
2. **Did it beat 0.8007 on test?** Not meaningfully. 0.8013 against 0.8007 — +0.0006 PR-AUC, with
   ROC-AUC down 0.0036 and F1 down 0.0052, on a point estimate from a single split. The
   hyperparameters changed; the performance did not.
3. **Did 35–42 improve?** Marginally: 0.9211 → 0.9215 PR-AUC, with ROC-AUC down (0.9729 → 0.9710)
   and F1 down (0.8926 → 0.8876) at the frozen threshold.
4. **What happened in 43–49?** Nothing changed: 0.0423 → 0.0427 PR-AUC, ROC-AUC down to 0.6839. The
   regime collapse is a property of the data and protocol, not of the model configuration.
5. **Did `has_addresses` help?** It moved validation PR-AUC by +0.000933, below the pre-registered
   +0.005 margin, so it was **dropped**. It could not have done more: no illicit transaction is
   address-less, so it can only ever be a licit-side flag.
6. **Which hyperparameters were selected?** `learning_rate=0.05`, `max_depth=6`,
   `min_child_weight=3`, `subsample=0.7`, `colsample_bytree=0.6`, `gamma=0.5`, `reg_lambda=1.0`,
   budget 1500 trees, patience 100, seed 42 — chosen by validation PR-AUC over 20 randomised trials
   (`selected_hyperparameters.json`).
7. **Is the model still tree-cap limited?** No. 582 trees were kept out of 1500, with training
   stopping on early stopping rather than on the budget.
8. **Is the baseline strong enough to freeze?** Yes. It is leakage-audited, reproducible from
   recorded seeds and artifacts, and now shown to sit at the ceiling of these features rather than
   at the ceiling of the tree count. The GNN bar is **PR-AUC 0.8013 / ROC-AUC 0.9281** on 35–49, and
   the honest pair of XGBoost rows is 0.8007 / 0.8013 — a difference too small to attribute to
   anything but noise.
9. **What still has to be fixed before GraphSAGE?** The `43–49` collapse must be reported for every
   GNN result alongside `35–49`; the operating point must be re-derived for the deployment
   prevalence rather than inherited from a 19.77% validation window; and calibration remains
   unmeasured. Nothing here blocks Phase 3.

**Does the baseline beat Logistic Regression?** Yes, by a wide margin: PR-AUC 0.8013 vs 0.2917 and
F1 0.7799 vs 0.3825 on identical features and an identical protocol. The 165 transaction features
carry substantial *nonlinear* structure. That matters for what follows: the headroom GraphSAGE must
find is not simply "add nonlinearity", because a tuned tree ensemble over the same features is
already at 0.80.

**What this establishes for the GNN phases.** A graph-free model over 165 tabular features reaches
**PR-AUC 0.8013 / ROC-AUC 0.9281** on 35–49, and **0.0427 / 0.6839** on 43–49. Those are the numbers
GraphSAGE, and then RGCN/HGT, must beat under the identical protocol: same windows, same population,
same threshold discipline, same metric implementation. Two things make that bar informative rather
than decorative — the wide margin over Logistic Regression shows the features are already being
exploited well, and the 43–49 collapse marks the regime where a structure-aware model has the most
room to differ. Any GNN result must therefore be reported on both windows; an aggregate-only
improvement would hide exactly the failure this baseline exposes.

## Limitations

* **The tree budget was the wrong suspect.** Lifting the cap from 500 to 1500 trees and retuning
  moved the primary test metric by +0.0006 PR-AUC. The original 0.8007 was not capacity-limited; it
  is the level these 165 features support under this protocol. The follow-on risk is the opposite
  of the one Phase 2 started with: further tabular tuning is now clearly low-yield, and any effort
  spent there is effort not spent on structure.
* **Optimisation on a saturated validation signal.** Validation PR-AUC spanned 0.9801–0.9881 across
  the search while test PR-AUC barely moved, so validation stops being a useful discriminator
  between candidates well before it stops being computable. Model selection on that window is
  therefore only weakly informative, and the 20-trial budget was spent confirming a ceiling rather
  than finding a better model. A re-search would need a different selection signal.
* **Hyperparameters were searched with `has_addresses` present.** The search ran on the 166-column
  set and the final model uses the 165-column set, so the selected configuration was tuned on a
  feature set it does not use and its own validation PR-AUC (0.98713) is below the search's best
  (0.98807). The effect is small but it is a real inconsistency in the protocol.
* **The 43–49 collapse is unresolved.** PR-AUC 0.0427 on 2.53% prevalence, ROC-AUC 0.6839, 5 true
  positives from 169, and 164 of the 284 false negatives. Re-tuning did not touch it. No causal
  explanation is offered here; it is a measurement to carry into the graph phases.
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
  temporal split; no confidence intervals are computed, and the +0.0006 PR-AUC gap between the two
  XGBoost configurations is smaller than the resolution of the sample. It is not evidence of
  improvement.
* **`Aggregate_feature_*` provenance is assumed.** Their contents are not re-derived from the raw
  blockchain data in this notebook, so their step-boundedness is taken on trust rather than proven.
* **No calibration.** Probabilities are used for ranking. Nothing here establishes that a score of
  0.7 means a 70% chance of illicit activity.
* **One recorded check is mis-specified in the checkpoint.** `xgboost/checks.csv` was produced
  before the tree-cap diagnostic was corrected: the row "reference xgb: the first run's tree cap was
  binding (early stopping never fired)" reports `ok = False` because the notebook compared
  `early_stopping_fired` against the wrong expected value. The finding itself stands — the reference
  run consumed its whole 500-round budget — and the notebook now records it as
  `ran_out_of_budget = True`. 56 of the 57 recorded checks pass.
