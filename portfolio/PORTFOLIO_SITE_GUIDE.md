# BitcoinGraphGuard → Portfolio Site: Hand-off Guide (for an LLM / developer)

Everything needed to add **BitcoinGraphGuard** as a new book on the Siddharth Dongardive portfolio site
(`/home/siddharth/Desktop/Projects/projects/Portfolio`) lives in this folder. Read top to bottom, then follow the
site's own guides: `Portfolio/docs/ADDING_PROJECTS.md` (shelf/book), `Portfolio/docs/CHAPTER_PAGE.md`
(gallery/page), `Portfolio/docs/PROJECT_GUIDE.md` (architecture). LegalRAG's entry in `Portfolio/backend/data.py`
is the worked example to copy.

Files here: `images/` (6 charts, §5) and `scripts/generate_portfolio_images.py` (regenerates them from
`results/`; run `./bit/bin/python portfolio/scripts/generate_portfolio_images.py`).

> **Accuracy rule.** Every number below traces to a file in `results/` or `reports/` (cited inline). Use these
> numbers and wording; avoid the claims in §7.

---

## 1. Steps to add the project

1. Copy images (§5) into `Portfolio/frontend/public/images/bitcoingraphguard/`.
2. In `Portfolio/backend/data.py`: add the tuple to `_CATEGORIES` (§3.1), a `_DETAILS["BitcoinGraphGuard"]` entry
   with `gallery_overrides` (§3.2, §5.3), and `_HEADER_META["BitcoinGraphGuard"]` (§3.3).
3. Mirror it in `Portfolio/frontend/src/data/fallbackProjects.js`, and add the content file
   `frontend/src/data/projects/bitcoingraphguard.js` the way the other projects do (see CHAPTER_PAGE.md).
4. Restart the backend, check `GET /api/projects/bitcoingraphguard` for the slug, open the shelf, click the
   book, then **View Full Project**; check each gallery section (1 image = full width, 2 = side by side,
   3+ = scroll row).

---

## 2. Project facts (single source of truth for copy)

| Field | Value |
|---|---|
| Name | BitcoinGraphGuard |
| One-liner | A Bitcoin fraud detector that knows when to stop trusting itself, built on real Elliptic++ data (203,769 transactions, 822,942 wallets, ~4.4M edges, 49 time steps). |
| Role framing | **Data Scientist / ML Engineer** (trains and compares models, builds serving + monitoring). |
| GitHub | https://github.com/siddhu-exe/BitcoinGraphGuard |
| Live demo | https://bitcoin-graph-delta.vercel.app (React dashboard on Vercel; the API runs on Render via `render.yaml`) |
| Year | 2026 |
| Dataset | Elliptic++: 203,769 transactions, 822,942 wallets, 4 directed edge types (~4.4M edges: 4,417,560 listed rows, `docs/EDA.md`), 49 time steps |
| Labels | Transactions: 4,545 illicit, 42,019 licit, 157,205 unknown. Class 3 (unknown) is never used as supervision. |
| Protocol | Fit 1-24, validate 25-34, refit 1-34, test 35-49 (16,670 labeled tx, 1,083 illicit). No random splits. |
| Primary metric | PR-AUC (illicit prevalence 6.5% in test, 2.53% in steps 43-49) |
| Tests | 49 pytest tests (19 API, 30 monitoring); 3 CI workflows (tests + scoped ruff, model-consistency gates, Docker build) |

### Tech stack (chips, ~10)
`Python`, `XGBoost`, `PyTorch Geometric`, `FastAPI`, `SHAP`, `Captum`, `React + TypeScript`, `Vite`, `Docker`,
`GitHub Actions`. Extras: `Pydantic`, `Recharts`, `pytest`, `Render`, `Vercel`, `Google Colab / Kaggle`.

### How it works
1. **Benchmark** XGBoost, GraphSAGE, HeteroRGCN, HGT (plus MLP, logistic regression, prevalence baseline) under one
   frozen temporal protocol.
2. **Diagnose** why every model collapses from step 43: adversarial validation, leak audit, permutation check.
3. **Explain** with SHAP (XGBoost), Captum/GNNExplainer and relation ablation (graph models).
4. **Test retraining** walk-forward under label delay L (labels from steps <= t - L only).
5. **Serve** the frozen XGBoost model (tau* = 0.435) via FastAPI; each prediction carries a `confidence_context`
   (`reliable` for steps 35-42, `degraded` for 43-49, `unknown` after).
6. **Monitor** lag-safe: nine candidate drift signals evaluated for false-alarm rate and detection lead;
   `LagViolationError` blocks any decision that would read a not-yet-available label.

### Engineering highlights (true, defensible)
- **Evaluation discipline is the differentiator**: temporal splits only, scaler fit on training steps, class 3 never
  treated as licit, every quoted metric traceable to a saved artifact, PR-AUC primary with both aggregate and
  drift-window reporting.
- **Lag-safe monitoring**: a decision at step t reads labels of steps <= t - L only; step t's own F1 is hindsight
  and is never a decision input. Enforced in code, covered by tests.
- **Honest signal evaluation**: of nine monitoring signals, the report lists which ones did *not* work
  (e.g. retraining trigger alarmed on 7 of 8 healthy steps) (`reports/monitoring_signal_evaluation.md`).
- **Frozen-model integrity gates in CI**: reproduces tx 1813992 @ step 35 = 1.7188735e-05, checks the 165 feature
  names and order, and fails if metrics fall back silently (`using_fallback_metrics` is exposed on `/health`).
- **Compute split by design**: all heavy ML on Colab/Kaggle via generated notebooks; the laptop (5.6 GB RAM)
  only serves, tests and documents. Serving excludes PyTorch/PyG/MLflow.

### What was built
1. Data and graph pipeline: transactions and wallets joined by four directed edge types, split strictly by time.
2. Four models (XGBoost, GraphSAGE, HeteroRGCN, HGT) under one leakage-free temporal protocol.
3. A lag-safe drift monitor (a decision at step t uses only labels available by then).
4. Walk-forward retraining studied under label delay.
5. FastAPI service, Docker image and GitHub Actions CI.
6. Live dashboard: https://bitcoin-graph-delta.vercel.app

### Three findings (plain words, one headline number each)
1. **XGBoost beat every graph model.** Test PR-AUC 0.80 (0.8013) vs 0.62 (0.6209) for the best graph model, GraphSAGE.
2. **All four models collapsed after step 43.** XGBoost went from 0.92 PR-AUC on steps 35-42 to 0.04 on 43-49 (0.80 pooled over 35-49). The graph models also end at 0.04-0.06 on 43-49: GraphSAGE 0.0505, HeteroRGCN 0.0550, HGT 0.0386.
3. **Retraining recovers only if labels arrive fast.** Steps 43-49 PR-AUC with expanding-window retraining: **0.695 at 1-step label delay, 0.388 at 2, 0.138 at 3, 0.105 at 5**; the frozen model scores 0.041 (L=1). Always quote the delays together.

### Monitor (lag-safe, label delay L=1; `reports/monitoring_backtest_report.md`, `results/walkforward/monitor_concordance.csv`)
- The label-free **score-shift channel fires at step 43**, the first drift step (score PSI 0.609 in the deployed monitor, `reports/monitoring_backtest_report.json`; the 0.25 critical level).
- The label-dependent **performance channel fires at step 44**, because step-43 labels only arrive one step later.
- **No CRITICAL alarm during steps 35-42.** One WARNING-level feature-shift alert occurs at step 37.
- Validated on **one** drift event only.

### Triage (`results/walkforward/operating_rules.csv`, `results/walkforward/README.md`)
Instead of one fixed probability threshold, analysts review the top-K scored transactions per step (K = 20, 50, 100). The best possible recall for a given K is
`sum over steps of min(K, positives_s) / total positives`. On steps 43-49 (169 illicit): **K=20 caps recall at 0.633** (107/169), K=50 at 0.964 (163/169). With expanding-window retraining at L=1, top-20 reaches recall 0.420 and top-50 0.811; at L=3 it is 0.296 and 0.615. The frozen model gets 0.024 / 0.089 (L=1). Do not use the window rows of `alert_budget.csv` (known error).

### Limitations
- One drift event (step 43) and 7 drift steps with 169 illicit labels, so the intervals are wide.
- Real-world label delay is unknown; results are shown for L = 1, 2, 3, 5.
- The deployed model is XGBoost; the graph models are benchmarked, not served.
- The API needs the 165 precomputed features and cannot score a raw transaction hash.

### Key results (use these exact numbers)

Test PR-AUC, steps 35-49, pooled (`results/*/metrics.json`, README):

| Model | PR-AUC | ROC-AUC | F1 (illicit) |
|---|---|---|---|
| **XGBoost, optimized (served)** | **0.8013** | 0.9281 | 0.7818 |
| GraphSAGE | 0.6209 | 0.9044 | 0.5945 |
| HGT | 0.4861 | 0.8937 | 0.3106 |
| 2-layer MLP | 0.4768 | 0.8912 | 0.5759 |
| HeteroRGCN | 0.4682 | 0.8946 | 0.5295 |
| Logistic regression | 0.2917 | 0.8828 | 0.4385 |
| Prevalence baseline | 0.0650 | 0.5000 | 0.1220 |

Regime shift, XGBoost at tau* = 0.435: PR-AUC **0.9215** on steps 35-42 (prevalence 9.16%) vs **0.0427** on 43-49
(prevalence 2.53%); recall 794/914 vs 5/169.

Drift diagnosis (`docs/OOD_DIAGNOSIS.md`): adversarial AUC 1.0000 between steps 1-34 and 43-49, yet 5-fold CV
*inside* 43-49 reaches PR-AUC 0.9199, so the signal is present and the distribution transfer fails.

Retraining under label delay, steps 43-49 pooled PR-AUC [95% cluster-bootstrap CI over steps]
(`results/walkforward/verdict.md`, verdict `PARTIAL`):

| Policy | L | PR-AUC |
|---|---|---|
| Static (frozen) | 1 | 0.0406 [0.0248, 0.0943] |
| Expanding window | 1 | 0.6949 [0.3559, 0.9120] |
| Expanding window | 2 | 0.3878 [0.1771, 0.7184] |
| Expanding window | 3 | 0.1377 [0.1063, 0.4949] |
| Expanding window | 5 | 0.1050 [0.0489, 0.3614] |

Explainability: XGBoost relies **61.5%** on local features (`Local_feature_53` is the top SHAP feature);
HeteroRGCN relies heavily on recurrent address pathways (`addr_to_tx`, `tx_to_addr`), making it brittle when
transactions move to unseen addresses.

**Honest headline findings** (keep them; they make the case study credible):
- A well-tuned XGBoost beat every graph model under the same protocol. Graph models were tuned less deeply, so
  this is a result for this protocol, not a general claim about graph methods.
- The step-43 drop is already documented by the original Elliptic work (Weber et al., 2019). The contribution is
  measuring it on every model and diagnosing it as covariate drift rather than a vanished signal.
- Retraining recovers performance only if labels arrive quickly: +0.65 PR-AUC at a one-step delay (0.695 vs 0.041), but only +0.11 at three (0.138 vs 0.027). Never quote one without the other.

---

## 3. Ready-to-paste data for `backend/data.py`

Adapt names to what the file actually uses (check allowed `tag` values and the `_CATEGORIES` layout first; the
LegalRAG entry at ~line 127/194 shows the format).

### 3.1 `_CATEGORIES` tuple
```python
("BitcoinGraphGuard",
 ["Python", "XGBoost", "PyTorch Geometric", "FastAPI", "SHAP", "Captum", "React", "TypeScript", "Docker", "GitHub Actions"],
 "Temporal fraud detection on the Elliptic++ Bitcoin graph: tabular and graph models compared on a strict "
 "train-on-past protocol, the step-43 drift diagnosed, and a drift-aware API with lag-safe monitoring.")
```
Suggested category: the existing Machine Learning one.

### 3.2 `_DETAILS["BitcoinGraphGuard"]`
```python
"BitcoinGraphGuard": {
    "subtitle": "Fraud detection that is tested on the future, and knows when it stops working.",
    "description": (
        "BitcoinGraphGuard benchmarks XGBoost, GraphSAGE, HeteroRGCN and a heterogeneous graph transformer on the "
        "Elliptic++ Bitcoin dataset (203,769 transactions, 822,942 wallets, 49 time steps) using temporal splits "
        "only. A tuned XGBoost scores 0.80 PR-AUC on steps 35-49 and beats every graph model, but every model "
        "collapses after step 43. Adversarial validation shows covariate drift, not a vanished signal, and "
        "walk-forward retraining recovers it only when labels arrive quickly. The best model is served through "
        "a FastAPI service that labels each prediction with how reliable that time step has been, alongside a "
        "lag-safe drift monitor and a React dashboard."
    ),
    "problem": "Fraud models trained on the past quietly fail when the world changes, and labels arrive too late to notice.",
    "data_text": "Elliptic++: 203,769 Bitcoin transactions and 822,942 wallets over 49 time steps; 157,205 transactions are unlabeled and never used as supervision.",
    "metric": "0.80",
    "metric_label": "Test PR-AUC (XGBoost, steps 35-49)",
    "reflection": "The model was fine until the data changed. Knowing when to distrust it, and how fast labels arrive, mattered more than a fancier architecture.",
    "published": True,
    "sort_order": 2,            # choose per site; renumbering shifts the others
    "tag": "ML",
    "spineColor": "#7A4A12",    # dark bitcoin amber; contrast is automatic
    "result": "0.80 PR-AUC · drift diagnosed · lag-safe monitoring",
    "gallery_overrides": { ... },   # see §5.3
},
```

### 3.3 `_HEADER_META`
```python
"BitcoinGraphGuard": {"emoji": "🛡️", "year": 2026},   # or "₿" if it renders cleanly
```

### 3.4 Links
`links.github = "https://github.com/siddhu-exe/BitcoinGraphGuard"`; add `links.demo` only once the dashboard URL
is confirmed.

### 3.5 `fallbackProjects.js`
Mirror title, tech, description, `sortOrder`, `tag`, `published`, `spineColor`, `result`, links and gallery paths.

---

## 4. Chapter copy (five fixed chapters)

| key | Heading | Suggested subtitle |
|---|---|---|
| architecture | Architecture | Notebooks on Colab produce saved artifacts; a tabular-only API and a lag-safe monitor consume them. |
| pipeline | Data Pipeline | A 49-step temporal graph of transactions and wallets, split strictly by time with no leakage. |
| ml | Machine Learning | Four model families under one protocol, why XGBoost won, and what it relies on. |
| platform | Platform | A FastAPI service with per-step confidence, a drift monitor, and a React dashboard. |
| results | Results | The step-43 collapse, the retraining recovery curve, and which alarms were worth trusting. |

---

## 5. Images

All files in `portfolio/images/` (PNG, 200 DPI). Copy to `Portfolio/frontend/public/images/bitcoingraphguard/` and
reference as `/images/bitcoingraphguard/<file>`.

### 5.1 Catalogue

| File | What it shows | Source |
|---|---|---|
| `01_system_architecture.png` | Dataset → notebooks → artifacts → API / monitor → dashboard | drawn |
| `02_model_comparison.png` | Test PR-AUC for 7 models | README / `results/*/metrics.json` |
| `03_regime_shift_per_step.png` | Per-step XGBoost PR-AUC and prevalence, drift window shaded | `results/temporal_inductive/per_step_metrics.csv` |
| `04_retraining_vs_label_delay.png` | Static vs expanding retrain by label delay, with CIs | `results/walkforward/lag_curve.csv` |
| `05_xgboost_shap_top_features.png` | Top 15 SHAP features | `results/explainability/xgb_shap_global.csv` |
| `06_monitoring_signal_tradeoff.png` | False-alarm rate vs detection lead per monitoring signal | `results/monitoring/signal_evaluation.csv` |

**Missing:** real dashboard screenshots. None exist in the repo yet. Run `cd frontend && npm run dev`, capture the
Overview, Monitoring, Retraining and Try-it pages, save them as `07_ui_overview.png` etc. here, and use them for
the `platform` slot and `thumbnail`.

### 5.2 Where each is used

| Site slot | Image(s) |
|---|---|
| `thumbnail` | dashboard Overview screenshot once captured; until then `03_regime_shift_per_step.png` |
| `architecture` (1) | `01_system_architecture.png` |
| `pipeline` (1) | `03_regime_shift_per_step.png` (the data's defining property) |
| `ml` (3, scroll row) | `02_model_comparison.png`, `05_xgboost_shap_top_features.png`, `04_retraining_vs_label_delay.png` |
| `platform` (1-2) | dashboard screenshots; fallback `06_monitoring_signal_tradeoff.png` |
| `results` (1) | `04_retraining_vs_label_delay.png` or `06_monitoring_signal_tradeoff.png` |

### 5.3 `gallery_overrides`
```python
"gallery_overrides": {
    "architecture": ["/images/bitcoingraphguard/01_system_architecture.png"],
    "pipeline": ["/images/bitcoingraphguard/03_regime_shift_per_step.png"],
    "ml": [
        "/images/bitcoingraphguard/02_model_comparison.png",
        "/images/bitcoingraphguard/05_xgboost_shap_top_features.png",
        "/images/bitcoingraphguard/04_retraining_vs_label_delay.png",
    ],
    "platform": ["/images/bitcoingraphguard/06_monitoring_signal_tradeoff.png"],  # swap for UI screenshots
    "results": ["/images/bitcoingraphguard/04_retraining_vs_label_delay.png"],
},
```

Copy command (from the repo root):
```bash
mkdir -p ../Portfolio/frontend/public/images/bitcoingraphguard
cp portfolio/images/*.png ../Portfolio/frontend/public/images/bitcoingraphguard/
```

---

## 6. Optional wording

- **Shelf `result` line**: `0.80 PR-AUC · drift diagnosed · lag-safe monitoring`
- **Three highlight bullets**:
  - Temporal benchmark of XGBoost, GraphSAGE, HeteroRGCN and HGT on 203k transactions and 823k wallets; tuned XGBoost wins at 0.8013 PR-AUC.
  - Diagnosed the step-43 collapse as covariate drift (adversarial AUC 1.0000) and measured how label delay limits retraining recovery (0.695 at L=1, 0.388 at L=2, 0.138 at L=3, 0.105 at L=5).
  - FastAPI service with per-step confidence context, lag-safe drift monitor, CI consistency gates and a React dashboard.
- **Resume bullet**: Built an end-to-end temporal fraud-detection system on Elliptic++ (203k transactions, 823k wallets): benchmarked four model families under leakage-free temporal splits, diagnosed a regime shift as covariate drift, and shipped a FastAPI service with lag-safe drift monitoring and 49 tests.

---

## 7. Claims to avoid

- ❌ "Discovered" the step-43 drop; it is in Weber et al. 2019. Say "measured and diagnosed".
- ❌ "Graph models don't work" or "XGBoost is better than GNNs" in general; GNNs were tuned less.
- ❌ "Production-grade", "catastrophic", "proving" (project framing rule).
- ❌ Quoting an aggregate metric without saying whether it is pooled or a mean of per-step values. The numbers
  here are **pooled** unless stated.
- ❌ "Detects fraud in real time on raw transactions": the API needs the 165 precomputed Elliptic++ features and
  cannot score a raw transaction hash.
- ❌ Presenting `confidence_context` as a probability of being right; it is a historical regime label per step.
- ❌ Treating the retraining result as precise: 7 steps and 169 positives give wide intervals (see CIs).
- ❌ Using the window-level ceilings in `results/walkforward/alert_budget.csv`; its README marks them wrong.
- ❌ Claiming MLflow/DVC: MLOps tooling is planned only.
- ⚠️ Monitoring thresholds were validated on one dataset and one regime change.
- ⚠️ "49 two-week steps": the two-week step length is not stated in any repo artifact; say "49 time steps".
- ⚠️ Edge total: ~4.4M counts raw listed rows (4,417,560, `docs/EDA.md`). The wallet-to-wallet list has 2,868,964 rows but
  2,784,344 unique pairs; say "~4.4M edge rows" if challenged.
- ❌ "Irreducible drift": the in-window 5-fold CV score (0.9199) shows the signal is learnable.
- ❌ Saying the monitor fires at step 43 through performance: the score-shift channel fires at 43, performance at 44.
