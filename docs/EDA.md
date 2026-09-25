# Elliptic++ Exploratory Data Analysis (Phase 2)

**Dataset:** Elliptic++ (transactions + wallets/addresses heterogeneous temporal graph)
**Analysis date:** 2026-09-22
**Scope:** temporal structure, class imbalance, graph structure, feature distributions,
temporal-split investigation. **No model was trained and no split was finalized.**
**Tooling:** `scripts/eda_phase2.py` (streaming), `scripts/plot_phase2_eda.py` (figures)
**Evidence:** `reports/eda/phase2_eda.json`, `reports/eda/tables/*.csv`,
`reports/eda/plots/*.png`

> **Status: Phase 2 EDA complete for all four requested areas.** Every number below was
> produced by the streaming EDA tooling against the raw files; nothing is copied from the
> Phase 1 inventory without re-derivation. Confidence labels are given in §8.

## 0. How the analysis was run (constraints honoured)

* The raw dataset in `Og data/` was opened **read-only** and never modified.
* Every stage uses `pandas.read_csv(..., chunksize=50000)`; no CSV is fully loaded.
* Wallet rows are de-duplicated on `(address, Time step)` **within and across chunks**,
  so wallet activity/features describe 920,691 distinct snapshots, not 1,268,260 raw rows.
* Runs were single-core at `nice -n 19`. Full-pipeline wall time **297 s**, peak RSS
  **879 MB** (a later wallet-only re-run: 159 s, 644 MB). Both are below the 1.07 GB of the
  Phase 1 verifier, so this is a light, laptop-safe workload.
* Missing values are counted from the raw stream; class `3` (unknown) is tracked
  separately and is **never** folded into licit. Here "labeled" means class `1` or `2`.

Reproduce everything:

```bash
MPLCONFIGDIR=/tmp/mplconfig nice -n 19 ./bit/bin/python -u scripts/eda_phase2.py \
    --low-priority --json reports/eda/phase2_eda.json
MPLCONFIGDIR=/tmp/mplconfig ./bit/bin/python scripts/plot_phase2_eda.py
# resume/update a subset later without re-reading everything:
./bit/bin/python scripts/eda_phase2.py --stage wallets --stage split --resume \
    --json reports/eda/phase2_eda.json
```

## 1. Temporal structure (49 time steps)

The 49 contiguous steps (1–49) are preserved for both node types. Per-step tables:
`reports/eda/tables/txs_temporal_by_step.csv`, `wallets_temporal_by_step.csv`.

**Transactions** — 203,769 rows, all carrying class 1/2/3.

* Activity is bursty: peaks at step 1 (**7,880** transactions) and troughs at step 27
  (**1,089**) — a ~7.2× spread.
* Labeled (1/2) share per step ranges from ~15.6% (step 9) to ~32.0% (step 44).
* Illicit share among labeled swings from **0.28%** (step 46) to **35.97%** (step 13).
* Mid-dataset spike steps (illicit % of labeled): 9 (31.9%), 13 (36.0%), 20 (28.9%),
  28 (29.9%), 32 (25.9%).
* The late window 43–46 is unusually clean: illicit share 1.75%, 1.51%, 0.41%, 0.28%
  (only 55 illicit transactions across those four steps), then it rebounds to 11.8% at 49.

| step | total | labeled | illicit | licit | unknown | illicit % of labeled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7,880 | 2,147 | 17 | 2,130 | 5,733 | 0.79% |
| 9 | 4,996 | 778 | 248 | 530 | 4,218 | 31.88% |
| 13 | 4,528 | 809 | 291 | 518 | 3,719 | 35.97% |
| 20 | 4,291 | 900 | 260 | 640 | 3,391 | 28.89% |
| 27 | 1,089 | 206 | 24 | 182 | 883 | 11.65% |
| 28 | 1,653 | 284 | 85 | 199 | 1,369 | 29.93% |
| 32 | 4,525 | 1,323 | 342 | 981 | 3,202 | 25.85% |
| 42 | 7,140 | 2,154 | 239 | 1,915 | 4,986 | 11.10% |
| 43 | 5,063 | 1,370 | 24 | 1,346 | 3,693 | 1.75% |
| 44 | 4,975 | 1,591 | 24 | 1,567 | 3,384 | 1.51% |
| 45 | 5,598 | 1,221 | 5 | 1,216 | 4,377 | 0.41% |
| 46 | 3,519 | 712 | 2 | 710 | 2,807 | 0.28% |
| 49 | 2,454 | 476 | 56 | 420 | 1,978 | 11.76% |

**Wallet snapshots** — 920,691 distinct `(address, Time step)` snapshots (after removing
the 347,569 exact duplicate rows).

* Peak step 1 (**34,853** snapshots), trough step 28 (**4,940**) — a ~7.1× spread.
* Illicit share among labeled snapshots ranges **0.35%** (step 45) to **27.65%** (step 26).
* Snapshot count is *not* dominated by new addresses only; the same addresses recur.

**Temporal feature drift.** Streaming early (1–34) vs late (35–49) means show sign flips
in several transaction `Aggregate_feature_*` columns (e.g. Aggregate_feature_8
≈ −0.58 → +1.17, Aggregate_feature_10 ≈ −0.58 → +1.17, Aggregate_feature_43
≈ −0.53 → +1.07). Wallet fee features also rise (fees_median ≈ 0.005 → 0.021). This is
real covariate shift that a temporal evaluation must expect; see §5.

## 2. Class imbalance

Overall (distinct entities): see `reports/eda/phase2_eda.json` → `labels`.

| Entity | total | illicit | licit | unknown | labeled | illicit % of all | illicit % of labeled | illicit:licit |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transactions | 203,769 | 4,545 | 42,019 | 157,205 | 46,564 (22.85%) | 2.23% | 9.76% | 1 : 9.25 |
| Wallets (addresses) | 822,942 | 14,266 | 251,088 | 557,588 | 265,354 (32.24%) | 1.73% | 5.38% | 1 : 17.60 |

At the **snapshot** level (wallet rows de-duplicated): 920,691 snapshots, 291,419 labeled,
14,720 illicit — 5.05% of labeled snapshots.

* Imbalance is severe but **not constant over time**: the illicit share of labeled
  transactions is ~12× higher in the middle steps than at step 46 (§1). Any single global
  prevalence figure hides this.
* Class `3` is large (77.1% of transactions, 67.8% of wallets) and must be treated as
  *unknown*, never as a negative class.
* The effective positive count is small: 4,545 illicit transactions total, and only 169 in
  the candidate test window 43–49.

## 3. Graph structure

Full block: `reports/eda/phase2_eda.json` → `graph`. Degree summaries below use
out-degree as the illustrative direction; the JSON contains full in/out/total histograms.

| Edge list | Edges | Unique pairs | Repeated | Self-loops | Recip. pairs | Unique src | Unique dst | Isolated (in universe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `txs_edgelist` (tx→tx) | 234,355 | 234,355 | 0 | 0 | 0 | 166,345 | 148,447 | 0 / 203,769 |
| `AddrTx_edgelist` (addr→tx) | 477,117 | 477,117 | 0 | 0 | 0 | 400,212 | 202,804 | 423,695 / 1,026,711 |
| `TxAddr_edgelist` (tx→addr) | 837,124 | 837,124 | 0 | 0 | 0 | 202,804 | 641,043 | 182,864 / 1,026,711 |
| `AddrAddr_edgelist` (addr→addr) | 2,868,964 | 2,784,344 | 84,620 | 45,981 | 57,874 | 400,212 | 641,043 | 0 / 822,942 |

Notes on the "isolated" column: bipartite stages use the combined tx+wallet node space
(1,026,711), so a node is "isolated" for that edge type if it has no edge of that type.
For `AddrAddr` the universe is wallets only, and **0 wallets are isolated** across all
address edge types combined (matches Phase 1).

**Degree distributions are heavy-tailed** (`reports/eda/plots/degree_distributions.png`):

| Graph | out mean | out p99 | out max | in mean | in p99 | in max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| tx→tx | 1.150 | 5 | 472 | 1.150 | 9 | 284 |
| addr→tx | 0.465 | 2 | 1,453 | 0.465 | 5 | 653 |
| tx→addr | 0.815 | 8 | 13,107 | 0.815 | 4 | 1,471 |
| addr→addr | 3.486 | 42 | 37,835 | 3.486 | 46 | 12,019 |

* Top hubs (by total degree): address `1GX28yLjVWux7ws4UQ9FB4MnLH4UKTPK2z` (out 37,835),
  address `1N52wHoVR79PMDishab2XmRHsbekCdGquK` (in 12,019, out 3,307),
  address `17A16QmavnUfCW11DAApiJxp7ARnxN5pGX` (out 13,816). Largest transaction:
  `txId 2984918` (out 472), `43388675` (in 284).
* **Components (weak connectivity):**
  * `txs_edgelist`: **49 components**, largest 7,880, zero singletons. The component count
    equals the number of time steps and the largest component equals the step-1
    transaction count — strongly suggesting the transaction graph splits by time step
    (see §8, still to be confirmed by joining component labels to `Time step`).
  * `AddrAddr`: **2 components**, largest 822,935 (99.999%); the address graph is almost
    fully connected.
  * Combined address↔transaction graph (`AddrTx` + `TxAddr`, undirected): **967
    components**, largest 1,025,738 (99.905%), and **965 singleton components — exactly
    the 965 address-less transactions** from Phase 1. This is an independent structural
    confirmation of that data-quality finding.
* Multi-edges/self-loops in `AddrAddr` are confirmed (84,620 repeated pairs, 45,981
  self-loops, 57,874 reciprocal pairs); they are legitimate multi-edge structure, not
  corruption.

## 4. Feature analysis

Full per-column tables: `reports/eda/tables/txs_feature_summary.csv` (182 features) and
`wallets_feature_summary.csv` (55 features). Correlations:
`txs_feature_correlation.csv`, `wallets_feature_correlation.csv`.

**Composition.**
* Transactions: 93 `Local_feature_*` + 72 `Aggregate_feature_*` + 17 domain features.
* Wallets: 55 behavioural features (BTC, fees, block spans, address reuse, timing).

**Modelling caveat on method.** Full-stream accumulators (count, missing, mean, std, min,
max, zero/negative counts, class-conditional moments, early/late means) use every row.
Quantiles, skewness and pairwise correlations use one bounded **systematic sample**
(stride 7 → ~29k transaction rows; stride 36 → ~26k deduplicated wallet snapshots), because
storing every column of every row would exceed the laptop's memory budget. Sample-based
values are distributional summaries, not exact population quantiles.

**Missing / constant / near-constant.**
* Transactions: no constant and no near-constant (≥99% zero) features. Missing values are
  exactly the 17 domain columns × the 965 address-less transactions (16,405 blanks) —
  independent agreement with Phase 1.
* Wallets: **0 missing cells**, no constant and no near-constant features.

**Class discriminability (point-biserial correlation with illicit vs licit).**
* Transactions: max |r| ≈ **0.271**. Strongest: `Local_feature_53` (−0.271),
  `Local_feature_89` (−0.234), `Local_feature_55` (−0.233), `Local_feature_90` (−0.227),
  `Aggregate_feature_49` (+0.196), `Aggregate_feature_57` (−0.191). 31 of 182 features
  reach |r| ≥ 0.10; the remaining `Aggregate_feature_*` are mostly near-zero
  (e.g. Aggregate_feature_18/60/36/59 ≈ 0.000).
* Wallets: max |r| ≈ **0.191** (`first_sent_block`); only 2 of 55 features reach
  |r| ≥ 0.10 (`first_sent_block` +0.191, `fees_max` +0.120). Wallet labels are only weakly
  linearly separable from these features.
* No single feature is close to sufficient; signals are weak and will likely need
  interaction/graph structure — consistent with the project's baseline-then-GNN plan.

**Redundancy / collinearity (important for feature selection).**
* The 17 transaction domain features are near-perfect copies of `Local_feature_*` columns:
  `Local_feature_1 ~ total_BTC` r=1.000, `Local_feature_1 ~ out_BTC_total` r=1.000,
  `Local_feature_2 ~ fees` r=1.000, `Local_feature_4 ~ num_input_addresses` r=1.000,
  `Local_feature_5 ~ num_output_addresses` r=1.000, `Local_feature_9 ~ in_BTC_max` r=1.000,
  `Local_feature_19 ~ out_BTC_mean` r=1.000. This redundancy (and the 16,405 blanks)
  supports using the `Local_feature_*` block and deriving the address-less indicator rather
  than duplicating the domain block.
* Wallet features contain near-duplicate groups:
  `lifetime_in_blocks ~ blocks_btwn_txs_total` r=1.000, and mean/median/total clusters
  (`btc_received_mean ~ btc_received_median` r=0.999, `fees_mean ~ fees_median` r=0.997,
  `blocks_btwn_*_mean ~ *_median` r≈0.99).

**Skewness.** Extremely right-skewed: transactions `Local_feature_34/37/35` skew
≈ 149/149/143, `Local_feature_15` ≈ 107; wallets `btc_received_min` ≈ 133,
`num_addr_transacted_multiple` ≈ 117. Log/robust scaling will be needed.

## 5. Temporal-split investigation

Candidate windows are computed from the verified per-step tables (investigation only — **no
split is finalized**). Full table: `reports/eda/phase2_eda.json` → `split_investigation`;
figure `reports/eda/plots/split_windows.png`.

| Window | Steps | Txs total | Txs labeled | Txs illicit | Txs illicit % | Wallet snapshots | Snap. labeled | Snap. illicit | Snap. illicit % |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train (proposed) | 1–34 | 136,265 | 29,894 | 3,462 | 11.58% | 601,271 | 193,663 | 9,697 | 5.01% |
| val candidate | 35–42 | 37,820 | 9,983 | 914 | 9.16% | 179,913 | 49,391 | 2,848 | 5.77% |
| test candidate | 43–49 | 29,684 | 6,687 | 169 | 2.53% | 139,507 | 48,365 | 2,175 | 4.50% |
| alt val | 35–40 | 25,338 | 6,697 | 559 | 8.35% | 169,849 | 44,423 | 4,456 | 10.03% |
| alt test | 41–49 | 42,166 | 9,973 | 524 | 5.25% | 255,522 | 77,382 | 4,220 | 5.45% |
| alt test | 35–49 | 67,504 | 16,670 | 1,083 | 6.50% | 425,371 | 121,805 | 8,676 | 7.12% |

**Findings that constrain the choice.**

1. The proposed train end (step 34) is reasonable: it keeps 76% of all illicit
   transactions (3,462 of 4,545) in training.
2. The natural test window 43–49 has a **much lower illicit prevalence for transactions**
   (2.53%) than training (11.58%) — a ~4.6× prior shift, driven by steps 43–46 (55 illicit
   total). Metrics such as PR-AUC will not be comparable to training performance without
   accounting for this, and 169 positives is a small test set.
3. A later-start alternative (test 41–49) yields more positives (524) at 5.25% prevalence;
   a wider test (35–49) yields 1,083 positives but removes the step-35–42 validation
   period.
4. Wallet snapshot prevalence is more stable than transaction prevalence
   (train 5.01% vs test 4.50%), but wallet labels are per-address and reused across
   snapshots, so snapshot-level metrics must be reported carefully.

**Recommendation:** do **not** finalize the split yet. Prefer
`train 1–34 / validation 35–42 / test 43–49` only if the low-prevalence test window is
accepted as the realistic deployment condition; otherwise evaluate on `41–49` (or `35–49`)
as a secondary protocol and always report prevalence alongside PR-AUC, plus per-step
breakdowns. This decision belongs to Phase 3 (baseline construction), once the metric
protocol is fixed.

## 6. Data-quality findings reconfirmed

* Wallet duplication is real and exact: 1,268,260 raw rows → **920,691** distinct
  `(address, Time step)` snapshots; **347,569** duplicate rows. A diagnostic confirmed
  347,566 of those duplicates are adjacent (within a single 50k-row chunk) — a useful
  detail for anyone writing their own dedup logic.
* `txs_features`: 16,405 blank cells confined to the 17 domain columns of the 965
  address-less transactions; zero non-numeric, zero NaN elsewhere.
* No constant or near-constant features were found in either node type.
* The 965 address-less transactions reappear as the 965 singleton components of the
  combined address↔transaction graph — three independent signals agree.

## 7. Implications for modelling

1. **Deduplicate wallets to `(address, Time step)` before any split** (prevents inflated
   counts and duplicate-row leakage across temporal boundaries).
2. **Do not treat class 3 (unknown) as licit**, and do not treat the 965 blanks as 0; mask
   them and/or add a binary "has_address_links" indicator.
3. **Use PR-AUC / recall at fixed precision** as primary metrics, but report them *per
   time window and overall* because prevalence shifts strongly (11.6% → 2.5%
   transaction illicit share).
4. **Drop or de-duplicate the 17 domain columns** (`Local_feature_*` already encode them at
   r≈1.0) — a concrete, defensible feature-selection step.
5. **Expect heavy tails**: scale/transform skewed features; consider degree and
   neighbourhood aggregates as graph features.
6. **Preserve heterogeneity**: wallets, transactions and the four edge types are
   structurally very different (address graph is one giant component; transaction graph
   fragments into ~49 near-time-step components).
7. **Time-step disconnection of `txs_edgelist`** (if confirmed) means transaction-only
   message passing cannot cross time steps; RGCN/HGT must rely on address links for
   cross-step signal. Confirm before Phase 3 graph design.

## 8. Verified facts, assumptions, unresolved

**Verified (independently re-derived in this phase).**
* Class counts, labeled/unknown shares, and per-step breakdowns for transactions and
  wallets (§1, §2).
* Edge counts, uniqueness, self-loops, reciprocal pairs, degree summaries, component
  counts, and isolated-node counts for all four edge lists (§3).
* 0 edge endpoints outside the node universes; combined graph singletons = the 965
  address-less transactions (§3).
* 0 missing wallet cells; transaction missingness confined to 17 columns × 965 rows (§4).
* No constant/near-constant features; quantified redundancy and skew (§4).
* Candidate-window label availability (§5).

**Assumptions.**
* Sample-based quantiles/skew/correlations (strides 7 and 36) are representative
  distributional summaries; full-stream means/stds are exact.
* Duplicate wallet rows are a property of the published download; the raw file was not
  compared to an upstream SHA and was not modified.
* `AddrAddr` repeated pairs/self-loops are legitimate multi-edge and self-transfer
  structure.

**Unresolved.**
* Whether `txs_edgelist` components correspond exactly to time steps (count and size match,
  but component labels were not joined to `Time step`).
* Whether the 347,569 wallet duplicates originate upstream or during curation.
* The precise semantic meaning of many anonymised `Local_feature_*`/`Aggregate_feature_*`
  columns (names are obscured in Elliptic++).
* The final train/validation/test boundary and metric protocol (deliberately deferred).

## 9. Artifacts produced

| Path | Contents |
| :--- | :--- |
| `scripts/eda_phase2.py` | Streaming staged EDA CLI (`labels`, `txs`, `wallets`, `graph_*`, `split`; `--resume`) |
| `scripts/plot_phase2_eda.py` | Figure generation from the aggregates |
| `reports/eda/phase2_eda.json` | Full machine-readable results |
| `reports/eda/tables/*.csv` | Per-step activity and per-feature summary/correlation tables |
| `reports/eda/plots/*.png` | Temporal activity, class imbalance, degree distributions, feature correlations, split windows |
