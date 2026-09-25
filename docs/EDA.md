# Elliptic++ Exploratory Data Analysis (Phase 1)

**Phase:** 1 — exploratory data analysis
**Implementation:** `notebooks/01_eda.ipynb` (Google Colab / Kaggle — never the laptop)
**Status:** the notebook is authored and validated but **has not been executed yet**, so every
result-dependent section below is **pending**.
**Last updated:** 2026-09-25

> **How to complete this document.** Run `notebooks/01_eda.ipynb` end to end (`Run all`) on
> Colab or Kaggle, then replace each *pending* marker with the numbers printed in
> `eda_digest.txt`, `checks.csv`, `eda_summary.json` and the tables listed under
> *Dataset*. Nothing here is invented: independently verified facts are labelled `verified`,
> and anything that depends on the run stays `pending` until the run exists.

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

Re-verified by the notebook: `overview.csv`, `edge_inventory.csv`, `file_inventory.csv`, and the
`transactions:` / `wallets:` / `edges:` rows of `checks.csv`. **Result: pending.**

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

Still to be reported by the run: whether the 965 blank-domain transactions are *exactly* the
address-less transactions (three independent signals should agree), constant/near-constant columns,
infinite or impossible values, and any column-level surprises. **Result: pending.**

## Class Imbalance

Transactions and wallets are labelled separately, and `unknown` (class 3) is never treated as licit.
The notebook reports three views — transactions, wallet addresses, and de-duplicated wallet
snapshots — as counts *and* as shares of all records and of labeled records only.

Verified label counts are in the table above; the resulting shares, the licit-to-illicit ratio, and
the snapshot-level view come from `class_distribution.csv` and figure `class_balance.png`.
**Result: pending.**

## Temporal Findings

Per-step activity, label coverage and illicit share for transactions (`txs_temporal_by_step.csv`)
and wallet snapshots (`wallets_temporal_by_step.csv`), summarised in `temporal_summary.csv` and
plotted in `temporal_activity.png` (§6 of the notebook). The notebook also quantifies early/late
feature drift (steps 1–34 vs 35–49).

Questions this section must answer after the run: how bursty is activity across the 49 steps, how
much of each step is labeled at all, how far the illicit share of labeled records swings step to
step, and whether the swings are large enough to make a single global metric misleading.
**Result: pending.**

## Feature Findings

From §7 of the notebook: strongest associations with `illicit` (point-biserial `r`,
`txs_feature_summary.csv` / `wallets_feature_summary.csv`), distribution shape (range, quantiles,
skew, mass at zero), redundancy among features, and early/late drift. Figure:
`feature_signal_and_drift.png`.

One specific question is already framed: are the 17 transaction domain columns redundant copies of
the anonymous `Local_feature_*` block? The notebook computes the closest `Local_feature_*`
counterpart for each domain column in `txs_domain_redundancy.csv`. **Result: pending.**

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

**Result: pending.**

## Temporal Evaluation Decision

The proposed protocol is train on steps 1–34 and test on steps 35–49. It is **investigated, not
adopted**, in §9 (`split_windows.csv`, `split_windows.png`). For each candidate window the notebook
reports total, labeled, illicit, licit and unknown counts, and the illicit share of labeled records,
for transactions and wallet snapshots — including validation windows carved out of the *historical*
period so the future period stays untouched.

**Decision: pending the run.** When the notebook has been executed, record here:

* the chosen train / validation / test windows and the reason;
* whether validation is carved from history rather than from the future, and why;
* whether the low-prevalence future window is accepted as the realistic deployment condition;
* the reporting unit for wallet results (per address or per snapshot), since wallet labels are
  reused across snapshots;
* the metric protocol (PR-AUC primary, precision/recall/F1 at a fixed operating point, reported per
  temporal window).

## Key Findings

**Pending.** Populate from `eda_digest.txt` (the notebook prints and saves its own digest at the end
of the analysis) plus the figures. Any `!!` line in `checks.csv` is a finding that must be explained
before a model is trained on this data.

## Modeling Implications

The EDA exists to settle these decisions; they are resolved once the results above are read:

* **XGBoost features** — keep all 182 transaction features or drop the 17 domain duplicates; how to
  represent the 965 address-less transactions (drop, or mask the domain columns and add an explicit
  indicator — blanks must never be read as `0`).
* **Class imbalance** — weighting versus resampling, and the metric protocol, given the measured
  illicit share and how much it moves between time steps.
* **Graph construction** — deduplicate wallets to `(address, time step)`; keep the graph
  heterogeneous across all four edge types; use mini-batch neighbour sampling because the address
  component is effectively one giant component.
* **Temporal splitting** — the windows chosen in the section above, with evaluation reported per
  time window because prevalence is non-stationary.
* **GNN design** — if the transaction graph is time-step-aligned, cross-step signal must come from
  the address links; this decides whether RGCN/HGT can rely on transaction edges alone.

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

## Provenance

* Phase 1 verification (streaming, low-memory): `scripts/verify_dataset.py`, evidence in
  `reports/phase1_verification.json`, report in `docs/DATA_INVENTORY.md` — 0 structural failures.
* Canonical EDA entrypoint: `notebooks/01_eda.ipynb` (this document's source of truth).
* Superseded tooling: `scripts/eda_phase2.py` and `scripts/plot_phase2_eda.py` produced a local
  streaming EDA on 2026-09-22 (`reports/eda/`) under the pre-notebook architecture. Those recorded
  results remain valid reference evidence, but the notebook is now the canonical entrypoint and this
  document is populated from it.
