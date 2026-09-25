# Elliptic++ Exploratory Data Analysis (Phase 1)

**Phase:** 1 — exploratory data analysis
**Implementation:** `notebooks/01_eda.ipynb` (Google Colab / Kaggle — never the laptop)
**Status:** executed end to end on Google Colab on 2026-09-25. Every result-dependent section below
reports that run; artifacts are in `eda/` (`eda_digest.txt`, `checks.csv`, `eda_summary.json`, the
table CSVs and the figures).
**Last updated:** 2026-09-25

> **Provenance of the numbers.** Every figure below is read from the saved run artifacts, never from
> expectation. Facts independently verified in Phase 1 are labelled `verified`; run-derived numbers
> name the file they come from.

## Objective

Establish, before any model is built, the facts that the later phases depend on:

* the real size, dimensionality and internal consistency of Elliptic++ (entities, edges, time steps, labels);
* the data-quality corrections that must be applied or explicitly handled;
* how severely imbalanced the labels are, and how that imbalance moves across the 49 time steps;
* which transaction and wallet features carry signal, which are redundant, and which drift over time;
* the structure of the four edge types (degrees, hubs, multi-edges, self-loops, components);
* whether the proposed temporal split (train 1–34, test 35–49) is workable.

This phase trains nothing, finalises no split and selects no features. Feature engineering,
graph construction and the temporal protocol are decided in later phases **from these results**.

## Dataset

Nine raw CSVs (~2.1 GB) in `Og data/` (gitignored, read-only). `wallets_features_classes_combined.csv`
is deliberately not read by the notebook — it is `wallets_features` plus the `class` column.

**Verified in Phase 1** (`docs/DATA_INVENTORY.md`, `reports/phase1_verification.json`):

| Item | Value (verified) |
| :--- | :--- |
| Transaction nodes / features | 203,769 / 182 (93 `Local_feature_*`, 72 `Aggregate_feature_*`, 17 domain) |
| Wallet-address nodes / features | 822,942 / 55 |
| Time steps | 49 contiguous (1–49) |
| `txs_edgelist` edges | 234,355 |
| `AddrTx_edgelist` edges | 477,117 |
| `TxAddr_edgelist` edges | 837,124 |
| `AddrAddr_edgelist` edges | 2,868,964 |
| Transaction labels | illicit 4,545 / licit 42,019 / unknown 157,205 |
| Wallet labels | illicit 14,266 / licit 251,088 / unknown 557,588 |

Re-verified by the notebook: every count above reproduced exactly (`overview.csv`,
`edge_inventory.csv`, `file_inventory.csv`, and the `transactions:` / `wallets:` / `edges:` rows of
`checks.csv`), including 203,769 transaction nodes, 822,942 wallet addresses, 920,691 distinct
wallet snapshots and all four edge-row counts. No check produced a `!!` warning.

## Data Quality Findings

Established in Phase 1 and re-derived by the notebook (§4 of `notebooks/01_eda.ipynb`):

* `txs_features.csv` has **16,405 blank cells**, confined to the 17 domain columns of the
  **965 transactions with no address links**; all 93 `Local_feature_*` and 72 `Aggregate_feature_*`
  columns are complete.
* `wallets_features.csv` contains **347,569 exact duplicate rows**: 1,268,260 raw rows collapse to
  **920,691 distinct `(address, time step)` snapshots**. The notebook reports the correction rather
  than silently absorbing it.
* `AddrAddr_edgelist.csv` has 84,620 repeated `(input, output)` pairs and 45,981 self-loops —
  multi-edges and self-transfers, not corruption.
* Zero edge endpoints fall outside the entity universes.

Confirmed by the run:

* The 965 blank-domain rows are **exactly** the address-less transactions — three independent
  signals agree (`checks.csv`: `blank-domain rows == address-less transactions`,
  `transactions: with at least one address link`, `combined address+transaction graph: singleton
  components`). 202,804 of 203,769 transactions have at least one address link.
* Missingness is confined to the 17 domain columns: 17 × 965 = 16,405 blank cells, no other column
  has a single missing value (`txs_feature_summary.csv`).
* **No constant and no near-constant column** in either entity (0 of 182, 0 of 55), no infinite
  values, and no non-numeric feature column.
* No duplicate or blank IDs; every transaction and wallet row has a class label; label values are a
  subset of `{1, 2, 3}`; no edge endpoint falls outside its entity universe.
* `wallets_features.csv` is 1,268,260 rows / 606 MB and is the only file needing correction before
  use; the raw file was not modified.

## Class Imbalance

Transactions and wallets are labelled separately, and `unknown` (class 3) is never treated as licit.
The notebook reports three views — transactions, wallet addresses, and de-duplicated wallet
snapshots — as counts *and* as shares of all records and of labeled records only.

From `class_distribution.csv` and `class_balance.png` — `unknown` is excluded from every
"share of labeled" figure and never counted as licit:

| Entity | Records | illicit | licit | unknown | labeled % | illicit % of all | illicit % of labeled | licit:illicit |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transactions | 203,769 | 4,545 | 42,019 | 157,205 | 22.85% | 2.23% | 9.76% | 9.25 : 1 |
| Wallet addresses | 822,942 | 14,266 | 251,088 | 557,588 | 32.24% | 1.73% | 5.38% | 17.60 : 1 |
| Wallet snapshots | 920,691 | 14,720 | 276,699 | 629,272 | 31.65% | 1.60% | 5.05% | 18.80 : 1 |

The imbalance is severe and differs by entity: ~9:1 for transactions, ~18:1 for addresses. 77% of
transactions and 68% of addresses are unlabeled, so supervision is a minority-of-a-minority problem
and accuracy is uninformative. Wallet labels are attached per address, so the snapshot view slightly
inflates the illicit count (14,720 snapshots vs 14,266 addresses) and must not be reported as if the
snapshots were independent samples.

## Temporal Findings

Per-step activity, label coverage and illicit share for transactions (`txs_temporal_by_step.csv`)
and wallet snapshots (`wallets_temporal_by_step.csv`), summarised in `temporal_summary.csv` and
plotted in `temporal_activity.png` (§6 of the notebook). The notebook also quantifies early/late
feature drift (steps 1–34 vs 35–49).

From `txs_temporal_by_step.csv`, `wallets_temporal_by_step.csv`, `temporal_summary.csv` and
`temporal_activity.png`:

| Per step | Transactions | Wallet snapshots |
| :--- | :--- | :--- |
| Records | 1,089 – 7,880 | 4,940 – 34,853 |
| Labeled | 206 – 2,154 (11.2% – 42.9% of the step) | 798 – 19,540 (10.4% – 59.7%) |
| Illicit | 2 – 342 | 11 – 1,106 |
| Illicit % of labeled | 0.28% – 35.97% (median 10.25%) | 0.35% – 27.65% (median 4.59%) |

Activity is bursty and prevalence is emphatically **non-stationary**. The illicit share of labeled
transactions climbs to 20–36% through the middle of the history (peaking at step 32), then collapses
to 0.4% around step 45 and recovers to 11.8% at step 49. No step has zero illicit transactions, so
per-step evaluation is possible everywhere; label coverage also swings by a factor of four
(11.2%–42.9%), which is a second reason a single global metric would be misleading. Wallet snapshots
move less violently (median 4.6%) but still span 0.35%–27.65%.

## Feature Findings

From §7 of the notebook: strongest associations with `illicit` (point-biserial `r`,
`txs_feature_summary.csv` / `wallets_feature_summary.csv`), distribution shape (range, quantiles,
skew, mass at zero), redundancy among features, and early/late drift. Figure:
`feature_signal_and_drift.png`.

* **Signal is weak and diffuse.** The strongest point-biserial correlation with `illicit` is
  `Local_feature_53` at `r = −0.26`; the next four are `−0.23 … −0.19`. Only 86 of 182 transaction
  features reach `|r| ≥ 0.05`, and all correlations are now bounded in [−1, 1]. Wallet features are
  weaker still: `first_sent_block` (+0.19), `fees_max` (+0.12), everything else ≤ 0.08. No single
  feature is a usable detector — this is an argument for the graph structure, not for a feature hunt.
* **Extreme skew.** Transaction features reach a skewness of 148.6 and wallet features 132.8, with
  heavy mass at zero on the fee/degree columns. Monotone transforms or tree/NN-native robustness are
  required; means alone are not interpretable.
* **The domain block is redundant.** All 17 transaction domain columns have a `Local_feature_*`
  counterpart at `|r| ≈ 0.98 – 1.00` (`txs_domain_redundancy.csv`): `total_BTC`/`in_BTC_total`/
  `out_BTC_total` ↦ `Local_feature_1`, `size`/`num_input_addresses` ↦ `Local_feature_4`, and so on.
  They are re-expressions, not independent evidence. Overall: 78 transaction feature pairs and 8
  wallet feature pairs at `|r| ≥ 0.99`.
* **Some features drift hard between the two periods.** Standardised early(1–34)→late(35–49) drift
  reaches ~1.7σ for `Aggregate_feature_8`/`10`/`7`, and `first_block_appeared_in` (1.71σ) /
  `last_block_appeared_in` (1.65σ) / `fees_*` (~0.6σ) for wallets. These are time-correlated columns,
  which is exactly the profiling a temporal split needs to survive.
* Figure: `feature_signal_and_drift.png`.

No feature is selected, transformed or imputed in this phase.

## Graph Findings

From §8 of the notebook, per edge type (`graph_summary.csv`, `graph_hubs.csv`, `degree_ccdf.png`):
edge counts, unique pairs, repeated pairs, self-loops, reciprocal edges, in/out/total degree
statistics, isolated nodes, top hub nodes and weakly-connected components.

Two structural questions the run must settle:

1. the address↔transaction edges as one combined bipartite graph — its component structure, and
   whether the singleton components are the 965 address-less transactions (`txs_component_time_span.csv`);
2. whether transaction-graph components are confined to single time steps. If they are,
   transaction-only message passing can never cross a time-step boundary, which directly shapes the
   RGCN/HGT design.

From `graph_summary.csv`, `graph_hubs.csv`, `degree_ccdf.png` and `txs_component_time_span.csv`:

| Edge type | Rows | Unique pairs | Self-loops | Reciprocal | Components | Largest | Mean / median / p99 / max degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `tx → tx` | 234,355 | 234,355 | 0 | 0 | 49 | 7,880 (3.9%) | 2.30 / 2 / 13 / 473 |
| `addr → tx` | 477,117 | 477,117 | 0 | 0 | 570,478 | 16,514 | 0.93 / 1 / 6 / 1,453 |
| `tx → addr` | 837,124 | 837,124 | 0 | 0 | 295,027 | 420,138 | 1.63 / 1 / 10 / 13,107 |
| `addr → addr` | 2,868,964 | 2,784,344 | 45,981 | 57,874 | 2 | 822,935 (99.999%) | 6.97 / 2 / 74 / 37,841 |

Both structural questions are answered:

1. The address↔transaction edges form one combined graph whose **965 singleton components are exactly
   the 965 address-less transactions**. Nothing else is disconnected.
2. **All 49 transaction-graph components are confined to a single time step** (49/49,
   `txs_component_time_span.csv`). Transaction-only message passing therefore *cannot* cross a
   time-step boundary, so any cross-step signal must travel through address nodes.

Other structure worth carrying forward: the address graph is one giant component (822,935 of 822,942
nodes in 2 components total) plus its 45,981 self-loops and 84,620 multi-edges — the network is
extremely dense at the head (`1GX28yLjVWux7ws4UQ9FB4MnLH4UKTPK2z` has out-degree 37,835) and sparse
in the tail (median degree 2). The `tx → addr` direction exposes a single transaction with 13,107
output addresses.

## Temporal Evaluation Decision

The proposed protocol is train on steps 1–34 and test on steps 35–49. It is **investigated, not
adopted**, in §9 (`split_windows.csv`, `split_windows.png`). For each candidate window the notebook
reports total, labeled, illicit, licit and unknown counts, and the illicit share of labeled records,
for transactions and wallet snapshots — including validation windows carved out of the *historical*
period so the future period stays untouched.

Measured windows (`split_windows.csv`), transactions / wallet snapshots:

| Window | Steps | Labeled | Illicit | Illicit % of labeled |
| :--- | :--- | ---: | ---: | ---: |
| Train (proposed) | 1–34 | 29,894 / 193,663 | 3,462 / 9,697 | 11.58% / 5.01% |
| Test (proposed) | 35–49 | 16,670 / 97,756 | 1,083 / 5,023 | 6.50% / 5.14% |
| Val in history | 25–34 | 6,288 / 40,373 | 1,243 / 4,263 | 19.77% / 10.56% |
| Val (alt) | 35–42 | 9,983 / 49,391 | 914 / 2,848 | 9.16% / 5.77% |
| Test (alt) | 43–49 | 6,687 / 48,365 | 169 / 2,175 | 2.53% / 4.50% |
| Val (alt) | 35–40 | 6,697 / 33,994 | 559 / 2,255 | 8.35% / 6.63% |
| Test (alt) | 41–49 | 9,973 / 63,762 | 524 / 2,768 | 5.25% / 4.34% |

**Decision (adopted for phase 2):**

* **Primary benchmark: train 1–34, test 35–49.** It matches the published Elliptic protocol and keeps
  the largest usable future sample (16,670 labeled transactions, 1,083 illicit). The alternative
  `43–49` window is rejected as the *primary* test: 169 illicit transactions put the illicit share at
  2.53% with roughly ±15% relative uncertainty, which is too noisy to support a headline number. It
  is retained as a secondary "recent-drift" window only.
* **Validation is carved from history (25–34), never from the future.** Steps 35–49 are read exactly
  once, at the end. The cost is a prevalence mismatch between validation (19.77%) and test (6.50%),
  so thresholds are not tuned to hit a target rate; PR-AUC and per-window reporting carry the
  comparison instead.
* **The low-prevalence future period is accepted as the deployment condition.** Prevalence is
  non-stationary by nature here (see Temporal Findings), so the drop from 11.58% to 6.50% is treated
  as signal to measure, not a defect to correct by resampling the test set.
* **Wallet results are reported per address, not per snapshot.** Snapshots inherit their address's
  label, so per-snapshot metrics are pseudo-replicated. The snapshot view is used for temporal and
  graph descriptions only.
* **Metric protocol:** PR-AUC is primary; precision, recall and F1 are reported at a fixed operating
  point; both are broken down per temporal window rather than aggregated, because a single global
  number would average across a 0.4% and a 36% regime.

## Key Findings

From `eda_digest.txt` and the sections above:

* The dataset is exactly as inventoried and internally clean: 203,769 transactions, 822,942
  addresses, 920,691 address-time snapshots, 49 steps, four edge types. No `!!` warnings.
* The only mandatory correction is the 347,569 duplicate wallet rows (1,268,260 → 920,691
  snapshots). The only "missing data" is 965 address-less transactions with blank domain columns —
  a real property of those rows, not a loading bug.
* Labels are badly imbalanced and mostly absent: 9.76% illicit among labeled transactions (9.25:1)
  and 5.38% among labeled addresses (17.60:1), with 77%/68% unknown.
* Prevalence is non-stationary across the 49 steps (0.28%–35.97% of labeled transactions), and
  activity is bursty.
* Feature signal is weak and diffuse — the best single correlation is `|r| = 0.26` — and the 17
  domain columns are redundant re-expressions of `Local_feature_*` at `|r| ≈ 0.98–1.00`.
* The transaction graph is 49 disconnected, single-time-step components; the address graph is one
  giant component; the 965 singletons in the combined graph are exactly the address-less
  transactions.

## Modeling Implications

The EDA exists to settle these decisions; they are resolved once the results above are read:

* **XGBoost features** — use the 165 non-domain transaction features, keep the domain block only if
  it is useful for interpretation (it duplicates `Local_feature_*` at `|r| ≈ 1` and adds nothing),
  and add an explicit `has_addresses`/missing indicator for the 965 address-less transactions rather
  than letting blanks become `0`. Apply monotone transforms (e.g. `log1p`) to the fee/degree columns
  before linear models; skew reaches 148.
* **Class imbalance** — expect ~9:1 (transactions) and ~18:1 (addresses). Plan on class weighting or
  `scale_pos_weight` rather than resampling, evaluate with PR-AUC plus precision/recall at a fixed
  operating point, and never report accuracy. Do not tune a threshold to match the historical
  prevalence — it does not hold in the test period.
* **Graph construction** — collapse wallets to `(address, time step)` snapshots; keep all four edge
  types (a homogeneous projection would discard the `addr → tx` / `tx → addr` structure the EDA shows
  to be the only time-crossing path); decide explicitly whether to keep the 45,981 self-loops and
  84,620 multi-edges; use mini-batch neighbour sampling, since the address graph is one component of
  822,935 nodes with a degree-37,841 head.
* **Temporal splitting** — the windows in *Temporal Evaluation Decision* above, with evaluation
  reported per window because prevalence moves by an order of magnitude across the 49 steps.
* **GNN design** — transaction edges alone cannot carry information across time steps (49/49
  components are single-step), so a transaction-only GraphSAGE model cannot learn temporal structure
  beyond per-step features. Cross-step signal must flow through address nodes. This is the concrete
  justification for moving to RGCN/HGT after the GraphSAGE baseline, rather than assuming it.

## Limitations

* No model is trained in this phase, so nothing here speaks to achievable detection performance.
* Feature semantics are unknown: most `Local_feature_*` / `Aggregate_feature_*` names are obscured
  in Elliptic++, so associations are statistical, not explanatory.
* Duplicate wallet rows are assumed to be a property of the published download; the raw file was not
  compared against an upstream copy or hash.
* `unknown` labels are excluded from supervised statistics; their true class composition cannot be
  recovered, so the measured illicit share is a labelled-sample share, not a population share.
* Component and degree statistics are transductive descriptions of the full graph; they are not yet
  evidence about inductive generalisation.
* Distributional summaries for quantiles, skew and correlations come from a bounded systematic
  sample (~25k rows); means, standard deviations and missingness are exact whole-file values.
* Correlations are computed on labeled records only; they say nothing about the unlabeled majority.
* The temporal split decision rests on labeled-sample counts, so the test period's true prevalence is
  unknown and may differ from the measured 6.50%.
* Node degrees are computed over the full, transductive graph including future time steps. They are a
  description of the dataset, not a feature recipe — feeding them into a temporal model would leak.

## Provenance

* **Run record:** `notebooks/01_eda.ipynb` executed end to end on Google Colab, 2026-09-25
  (python 3.13.15, pandas 2.2.3, numpy 2.1.3, seed 42, `chunk_size` 50,000 — `eda/config.json`).
  Artifacts were exported back to `eda/`. 31 distinct checks, all passing. `checks.csv` contains 7
  duplicated rows at the end because a check-appending cell was re-run inside the same kernel; the
  values are identical and consistent, and the duplication is an artifact of the session, not of the
  data. The underlying statistics are unaffected — totals re-sum to 203,769 transactions and 4,545
  illicit.
* Data and outputs live in one Drive folder (`DRIVE_ROOT` in the notebook's setup cell,
  `/content/drive/MyDrive/Projects/Bitcoin Graph` on Colab): raw CSVs in `Dataset/`, this
  notebook's artifacts in `eda/`.
* Phase 1 verification (streaming, low-memory): `scripts/verify_dataset.py`, evidence in
  `reports/phase1_verification.json`, report in `docs/DATA_INVENTORY.md` — 0 structural failures.
* Canonical EDA entrypoint: `notebooks/01_eda.ipynb` (this document's source of truth).
* Superseded tooling: `scripts/eda_phase2.py` and `scripts/plot_phase2_eda.py` produced a local
  streaming EDA on 2026-09-22 (`reports/eda/`) under the pre-notebook architecture. Those recorded
  results remain valid reference evidence, but the notebook is now the canonical entrypoint and this
  document is populated from it.
