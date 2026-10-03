# IMPROVEMENT_PLAN.md — Task Spec for LLM Agents

> **Audience:** an LLM coding agent (Claude Code, Codex, etc.) working in this repo.
> **Purpose:** turn BitcoinGraphGuard from a "claims to be production" project into an honest,
> rigorous, resume-grade project. Work through the tasks **in order**. Each task lists
> *why*, *what to change*, *how*, and *done when*.
> **Source:** senior-DS review dated 2026-10-04. All numbers below were read from files in
> `results/` and `reports/` — re-verify before quoting.

---

## 0. Read this first — hard rules (do not violate)

1. **Read `CLAUDE.md` / `AGENTS.md` before anything.** All carry-forward constraints there still apply
   (temporal splits, class 3 = unknown, PR-AUC primary, no leakage, no metric fabrication).
2. **No ML on the laptop.** Training, data loading, SHAP, bootstraps over predictions with the raw
   dataset → write notebook generator code (`scripts/generate_notebook_XX.py`) that the user runs on
   Colab/Kaggle. Small bootstraps over committed `results/**/predictions.csv` are OK only if the user
   approves running them.
3. **The user runs commands themselves.** Give the exact command; the user pastes the output; you verify.
   Do not execute scripts, training, or Docker on your own.
4. **Do not modify frozen artifacts:** `results/xgboost/**`, `results/xgboost_v2/**`,
   `results/temporal_inductive/per_step_metrics.csv`. Changing them trips the CI model-consistency gate
   (`.github/workflows/model-consistency-check.yml`, tx 1813992 @ step 35 must score 1.7188735e-05).
   New experiments write to **new** folders (e.g. `results/walkforward/`, `results/rigor/`).
5. **Never invent numbers.** Every metric written in docs must point to an artifact file path.
6. **Never touch `Og data/`** (raw, gitignored, read-only).
7. Keep `CLAUDE.md` and `AGENTS.md` in sync if workflow rules change.
8. Scoped CI lint must stay green:
   `ruff check --force-exclude --config .github/ruff-ci.toml src/api src/monitoring tests`

---

## 1. Project in one paragraph (context)

BitcoinGraphGuard detects illicit Bitcoin transactions on the **Elliptic++** dataset (203,769 txs,
822,942 addresses, 49 time steps). Models: XGBoost (best, test 35–49 PR-AUC **0.8013**), GraphSAGE
(0.6216), HeteroRGCN (0.4682), HGT (0.4861). All models collapse in steps 43–49 (XGBoost PR-AUC
**0.0427** vs prevalence **0.0253**, recall 5/169 at τ=0.435). A FastAPI service (`src/api/`) serves the
frozen XGBoost; `src/monitoring/` holds drift monitors and a retraining trigger, validated by a
retrospective backtest (`reports/monitoring_backtest_report.md`). A React frontend (`frontend/`, untracked)
and a walk-forward retraining notebook (`scripts/generate_notebook_09.py`, untracked) are in progress.

## 2. Core problems the tasks fix

| ID | Problem | Evidence |
|----|---------|----------|
| P1 | API needs 165 anonymised Elliptic features → unusable on any real transaction | `src/api/schemas.py` (`MODEL_FEATURES`) |
| P2 | Model is near-random on the most recent window | `results/xgboost/temporal_metrics.csv` row 43-49 |
| P3 | `confidence_context` is a lookup on client-supplied `time_step`, not a model-based confidence | `src/api/model_loader.py:254` |
| P4 | Step-43 collapse presented as a "discovery"; it is documented in Weber et al. 2019 | `CLAUDE.md`, `README.md` |
| P5 | Validation window saturated (all 20 trials 0.957–0.988) → tuning can't discriminate | `results/xgboost/optimization_results.csv` |
| P6 | Contradictory numbers across reports (pooled vs per-step mean unlabeled; narrative ≠ table) | `reports/monitoring_backtest_report.md` lines 42–43, 106–108 vs table in §2 |
| P7 | No confidence intervals / seeds; per-step metrics on 2–5 positives | step 45: 5 illicit, step 46: 2 illicit |
| P8 | Label-free drift monitors don't separate good (35–42) vs bad (43–49) steps | adversarial AUC 0.993/0.996 (good) vs 0.999 (bad); trigger fires RECALIBRATE_ONLY on 7/8 good steps |
| P9 | GNNs likely under-tuned; comparison not fair | RGCN < GraphSAGE; HGT F1 0.31 |
| P10 | Engineering gaps | CORS `*` + credentials; `xgboost>=2.0.0` unpinned (trained on 3.4.1); Docker copies all of `results/`; no auth |
| P11 | Doc sprawl, stale README, hype language | 19 files in `docs/`; README says `src/` "planned", references nonexistent `07_final_evaluation.ipynb` |

---

## 3. Tasks (in priority order)

### TIER 1 — Fix the story

#### T1. Fix contradictory numbers in the monitoring report (P6)
- **Why:** Contradictions destroy reviewer trust faster than anything.
- **What:**
  - `reports/monitoring_backtest_report.md` §1 table: 43–49 PR-AUC `0.0834` and full-test F1 `0.4855` are
    per-step **means**; `results/xgboost/metrics.json` gives **pooled** 0.0427 / 0.7799.
  - §3.3 says rolling prevalence 10.42% / 8.21% / "1.15–1.34% at steps 45–47", but the §2 table shows
    10.24% / 8.11% / 6.91%, 5.46%, 4.17% for 45–47 (1.34% is at step 48).
- **How:**
  1. Find where the report text is produced: `scripts/run_monitoring_backtest.py` and `src/monitoring/backtest.py`.
  2. Make every aggregate column explicitly labelled `pooled` or `mean of per-step`; show both for PR-AUC and F1.
  3. Make §3.3 narrative values derive from the same computed data as the §2 table (no hand-typed numbers).
  4. Give the user the command `python scripts/run_monitoring_backtest.py` to regenerate; verify the output.
- **Done when:** every number in the report appears once with one meaning; narrative = table; label "pooled"/"mean" on every aggregate; `pytest -q` passes.

#### T2. Finish walk-forward retraining (Phase 9a) — the most valuable result
- **Why:** Answers the real question: *does realistic retraining under label delay recover the 43–49 collapse?*
- **What:** `scripts/generate_notebook_09.py` → `notebooks/09_walkforward_retraining.ipynb` (both untracked).
- **How:**
  1. Review the generator; protocol must be: to predict step t, train only on labels from steps ≤ t − L.
  2. Run for L ∈ {1, 2, 4, 8}. Also compare: frozen model, expanding window, sliding window (e.g. last 10 steps), recency-weighted.
  3. Outputs to `walkforward/` → user copies to `results/walkforward/`: per-step PR-AUC, pooled PR-AUC for 35–42 and 43–49, plot "PR-AUC vs label delay".
  4. Include bootstrap CIs (see T4).
  5. User runs on Colab; you write `docs/WALKFORWARD.md` from the artifacts only.
- **Done when:** `results/walkforward/` committed; doc states clearly whether retraining recovers 43–49 and at which L it stops helping.

#### T3. Honest monitoring evaluation (P8)
- **Why:** Current report claims monitoring works; evidence shows only label-based signals detect the collapse.
- **What:** new section in the monitoring report + new artifact `results/monitoring/signal_evaluation.csv`.
- **How:**
  1. For each signal (triad residual drift, local-feature PSI ratio, adversarial AUC, rolling prevalence, rolling F1, trigger decision) compute over steps 35–49:
     - **false-alarm rate** on steps 35–42 (model is fine there),
     - **detection lead/lag** relative to step 43 (first step it alerts in 43–49 minus 43),
     - whether it needs labels (yes/no).
  2. Add **label-free** candidates in `src/monitoring/` (new module, with tests): predicted-score distribution shift (KS between reference-window scores and step-t scores), predicted-positive rate at τ. Evaluate the same way.
  3. State plainly in the report which signals failed.
- **Done when:** table exists, a "What did NOT work" paragraph exists, new monitors have unit tests in `tests/test_monitoring.py`.

#### T4. Reframe README and positioning (P2, P4)
- **What:** rewrite `README.md` top section.
- **How:**
  - New pitch: *"Why fraud models collapse under regime shift: diagnosis on Elliptic++ and a retraining policy under realistic label delay."*
  - Results table showing **both** 35–42 and 43–49 with prevalence alongside PR-AUC.
  - Cite Weber et al. 2019 ("Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", KDD workshop) for the step-43 dark-market shutdown, and Elmougy & Liu 2023 (Elliptic++, KDD) for baselines. Do not claim step-43 as a discovery. Ask the user to confirm the exact published numbers before adding a baseline row — do not quote numbers from memory.
  - Add a **Limitations** section near the top: anonymised features (P1), 2016–17-era data, drift-window failure (P2), API reliability is a step lookup (P3).
  - Remove stale lines (`src/`, `tests/` "planned"; `07_final_evaluation.ipynb`).
  - Update the matching wording in `CLAUDE.md` and `AGENTS.md` ("Crucial … Discoveries" → "Known structural facts (see Weber et al. 2019)").
- **Done when:** README is accurate against the repo, ≤ ~150 lines, no hype words ("catastrophic", "proving", "production-grade").

### TIER 2 — Rigor

#### T5. Confidence intervals and seeds (P7)
- **How:**
  1. Add a small module (e.g. `scripts/bootstrap_metrics.py`, stdlib + numpy + sklearn) that reads any `results/**/predictions.csv` and outputs bootstrap 95% CIs (1,000 resamples, stratified by class, resample within window) for PR-AUC and F1 on 35–49, 35–42, 43–49. Output to `results/rigor/bootstrap_ci.csv`. Ask the user before running it.
  2. Add a `n_illicit` column next to every per-step metric; mark steps with < 10 positives as "too few positives — not interpretable".
  3. GNN seeds: add a seed loop (3–5 seeds) to the GNN notebook generators; report mean ± std. User runs on Colab.
- **Done when:** every headline PR-AUC in README has a CI.

#### T6. Label metrics as pooled vs per-step mean everywhere
- Grep `docs/` and `reports/` for PR-AUC/F1 aggregates; add the label. Same rule as T1.

#### T7. Analyst-capacity metrics
- Add precision@k and recall@k per step (k = 50, 100, 200) to the bootstrap script output. Fraud teams work with fixed review capacity; this matters more than a single τ.

#### T8. Fair GNN comparison (P9) — optional, do after T1–T7
- Either: proper tuning budget (≥ 30 trials) + seeds for RGCN/HGT, or collapse Phases 3/4/7 into one "GNN comparison" doc with an honest note that the GNNs are under-tuned.
- Do not overwrite existing `results/heterogeneous_gnn/` or `results/hgt/`; write to new folders.

### TIER 3 — Make it usable

#### T9. Fix `confidence_context` (P3)
- **How (in `src/api/`):**
  1. Rename the field meaning in the API docs: it is a *historical regime lookup*, not a confidence. Add `"basis": "historical_step_lookup"` to the response.
  2. Add a real per-input signal: e.g. an **input OOD score** — distance of the input from the training distribution (fraction of features outside the training 1st–99th percentile, computed from a small committed reference-quantile JSON generated on Colab). Return it as `input_ood_score`.
  3. Tests in `tests/test_api.py`.
  4. Do not change the model or threshold (CI gate).
- **Done when:** response says how "reliability" was derived; OOD score is tested.

#### T10. Raw-blockchain feature pipeline (P1) — the biggest differentiator
- **Goal:** a model that accepts a **real txid**.
- **How:**
  1. Define ~15–30 features you can compute from public tx data: n_inputs, n_outputs, total in/out value, fee, fee rate, value entropy across outputs, address reuse count, round-number outputs, input-address age, etc.
  2. Problem: you need these for Elliptic txIds. Elliptic++ tx IDs are not public tx hashes in all releases — **verify first** whether a txId→hash mapping is available. If not, document that and train on whatever raw-computable features Elliptic++ wallet/address tables allow, or use another labelled public source. Ask the user before choosing.
  3. Separate model folder `results/raw_features_model/`; separate endpoint `POST /predict_txid`. Keep the frozen model untouched.
  4. Fetching tx data: use a public block explorer API with caching and rate limiting; handle failures explicitly.
- **Done when:** a demo can score a real txid end-to-end, with its honest (likely lower) PR-AUC reported.

#### T11. Per-prediction explanation
- Return top-5 SHAP contributions per prediction (`xgboost` `pred_contribs=True` on the Booster — no `shap` dependency needed). Add field `top_contributions` to `PredictionOutput` in `src/api/schemas.py`. Test it.

#### T12. Engineering fixes (P10)
- `src/api/main.py`: CORS → explicit origin list from env var `CORS_ORIGINS`; set `allow_credentials=False` unless needed.
- `requirements-serving.txt`: pin `xgboost==3.4.1` (version used for training per `results/xgboost/selected_hyperparameters.json`).
- `Dockerfile`: copy only `results/xgboost/xgb_model_optimized.json`, `results/xgboost/selected_features.json`, `results/temporal_inductive/per_step_metrics.csv`, `reports/monitoring_backtest_report.json`. Check `.github/scripts/verify_container_prediction.py` and `docker-build.yml` still pass.
- Optional API-key auth via header `X-API-Key` (env `API_KEY`; disabled if unset) + simple rate limit.
- `time_step`: document that in a live setting it must come from the ingestion pipeline, not the caller.
- Phase 9 MLflow: log model version + artifact hash; expose `model_version` in `/health`.
- **Done when:** `pytest -q` green, scoped ruff green, Docker CI green.

### TIER 4 — Presentation

#### T13. Consolidate docs (P11)
- Target structure: `README.md`, `docs/METHODOLOGY.md` (data, splits, leakage controls, models), `docs/FINDINGS.md` (results + drift diagnosis + monitoring evaluation + walk-forward), `docs/SERVING.md` (API, Docker, CI), `docs/archive/` (move per-phase docs there, don't delete).
- Remove/merge `PLAN.md`, `PROGRESS.md`, `TIMELINE.md`, `OBJECTIVES.md`, `OVERVIEW.md`, `PROJECT_CONTEXT.md` into the above. Update every path reference in `CLAUDE.md`, `AGENTS.md`, README.
- Remove hype wording across docs and reports.

#### T14. Commit untracked work cleanly
- `frontend/` (ensure `node_modules/` is gitignored), `scripts/export_dashboard_data.py`, notebook 09 + generator. Separate commits per concern. Ask the user before committing.

---

## 4. Definition of done (whole plan)

- [ ] No contradictory numbers anywhere; every aggregate labelled pooled/mean.
- [ ] Walk-forward results committed and documented.
- [ ] Monitoring report has a signal-evaluation table and a "what failed" section.
- [ ] README: honest pitch, limitations, citations, CIs, both windows.
- [ ] API: reliability basis explicit, OOD score, SHAP top-5, pinned deps, slim Docker, fixed CORS.
- [ ] (Stretch) real-txid scoring endpoint.
- [ ] Docs consolidated to ~4 files + archive.
- [ ] `pytest -q`, scoped ruff, and all three CI workflows green.
