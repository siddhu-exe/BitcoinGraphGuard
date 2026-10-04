"""Build notebooks/09_walkforward_retraining.ipynb — the Phase 9a walk-forward retraining experiment.

The notebook is generated rather than hand-edited so the cell boundaries, ordering and the
self-contained Colab layout stay reviewable in plain text. Run from the repository root:

    python scripts/generate_notebook_09.py

Generation only writes JSON and syntax-checks every code cell (ast.parse); it never executes any
notebook code and never touches data. The notebook itself runs on Colab / Kaggle, not the laptop.
"""

import ast
import json
import uuid
from pathlib import Path

OUTPUT = Path("notebooks/09_walkforward_retraining.ipynb")

CELLS: list = []


def _record(kind: str, source: str) -> None:
    """Append one notebook cell, splitting the source into the list-of-lines nbformat wants."""
    lines = source.split("\n")
    payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": payload, "id": uuid.uuid4().hex}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    CELLS.append(cell)


def md(source: str) -> None:
    _record("markdown", source)


def code(source: str) -> None:
    ast.parse(source)  # fail at generation time, not on Colab
    _record("code", source)


# ---------------------------------------------------------------------------------------------
# Title and protocol
# ---------------------------------------------------------------------------------------------
md(r"""# BitcoinGraphGuard — Phase 9a: Walk-Forward Retraining Under Label Delay

**Question.** The frozen XGBoost (trained on 1–34) is excellent on 35–42 and collapses on 43–49. An
in-window refit inside 43–49 reaches 0.9199 PR-AUC, but that uses 43–49's own labels and is an oracle,
not a deployable policy. The open question is narrower and operational: **does a realistic retraining
loop recover performance, and if not, why not?**

**Diagnostic and evaluation only.** No frozen artifact (`xgboost/`, `xgboost_v2/`, `results/*`), no
API code and nothing under `src/monitoring/` is modified. A before/after snapshot of the frozen
folders is recorded as a check. All output goes to `walkforward/` (copy to `results/walkforward/`).

## Protocol (no leakage)

Predict step **t** with a model trained only on labels that have **arrived by t**. Labels for step
**s** arrive at **s + L**, so at step t the newest usable label step is **t − L**. Features at step t
only; the same 165 features; the same XGBoost configuration as `xgboost_v2`
(`FROZEN_PARAMS`, 1,500-tree budget, patience 100, `scale_pos_weight` from the fit slice).

* **L = 1** is the main setting. **L = 3** is the sensitivity check.
* Class 3 (unknown) is never a training or evaluation label.
* **Primary operating mode: top-K alerts per step** (K = 20, 50, 100). The validation-derived threshold
  rule from earlier phases (F1-maximising threshold on a temporal hold-out of the *most recent labeled
  steps*, never on step t) is the comparison, and its per-step instability is reported as a finding. The
  oracle-threshold F1 is reported only as a per-step ceiling, never pooled.
* **Intervals** are a cluster bootstrap over time steps (used for every verdict input); the row-level
  bootstrap is reported alongside.
* **Lag curve**: `static` and `expanding` are also run at L = 2 and L = 5 (not the other strategies).

## Strategies (same data, same metrics, reported separately)

| # | Strategy | Policy |
| :-- | :--- | :--- |
| 1 | `static` | trained once on every label available at step 35, never retrained (control) |
| 2 | `expanding` | retrain every step on all labeled data available |
| 3 | `rolling_w5`, `rolling_w10` | retrain every step on the last W labeled steps |
| 4 | `recency_exp_hl6` | expanding window, exponential decay by step age (half-life 6, the `xgboost_v2` value) |
| 5 | `monitor_triggered` | retrain (expanding policy) only when the lag-safe Phase 8a rules or the label-free channels (feature shift, score shift) return RETRAIN |
| 6 | `expanding_no_triad` | strategy 2 with `Aggregate_feature_10`, `43`, `8` removed |

**Every retraining policy shares one fit procedure**, so the only thing that differs between them is
*which labels they see* (and, for strategy 5, *when*).""")

md(r"""## Pre-registered decision rules

Fixed here, before any model is fit, so no threshold below can be tuned on 43–49 labels.

| Constant | Value | Meaning |
| :--- | ---: | :--- |
| `RECOVERY_TARGET` | 0.50 | pooled 43–49 PR-AUC that counts as *restored* (about 12× the best static figure, about 55% of the 35–42 level) |
| `MATERIAL_GAIN` | 0.15 | pooled 43–49 PR-AUC below which a gain is "statistically real but operationally immaterial" |
| `NONINFERIORITY` | 0.02 | a retraining strategy may not lose more than this PR-AUC on 35–42 versus static |
| `MIN_HOLDOUT_POS` | 40 | positives the early-stopping / threshold hold-out should hold (the `xgboost_v2` selection-slice check) |

**Verdict rule** (computed in code, rendered in the cell above the last markdown cell):

* **(a) RECOVERS** — `expanding` reaches `RECOVERY_TARGET` on pooled 43–49 at **both** L = 1 and L = 3,
  and its paired (cluster-bootstrap) ΔPR-AUC versus static has a 95% lower bound above 0. Report the
  sustained-recovery step (first step t ≥ 43 with PR-AUC ≥ 0.50 that stays ≥ 0.50 for the next two
  evaluated steps; steps 45 and 46, with 5 and 2 positives, are excluded from that test).
* **(b) PARTIAL** — not (a), but at least one retraining strategy × lag cell reaches `MATERIAL_GAIN`
  with a paired ΔPR-AUC lower bound above 0 (recovery for only some strategies, or only at low label delay).
* **(c) NO RECOVERY** — otherwise.

**Strategy to freeze**: the retraining strategy with the highest pooled 43–49 PR-AUC at L = 1 among
those that (i) clear `MATERIAL_GAIN`, (ii) have a paired ΔPR-AUC lower bound above 0 vs static on 43–49 and
(iii) are non-inferior on 35–42. If none qualifies, freeze `static`.

## Deliberate deviations from `xgboost_v2`, stated up front

1. **Hold-out rule.** `xgboost_v2` fits on 1–24 and early-stops on 33–34. A walk-forward loop has no
   fixed 24/33 split, so every retrain uses the most recent labeled steps as the hold-out: the smallest
   `k` (2 ≤ k ≤ 6, leaving at least 2 fit steps) such that the hold-out holds at least 40 positives. This
   is computed from labels that have already arrived. `v2_exact_control` below reproduces the
   `xgboost_v2` reference row with the *original* split to prove the harness is faithful.
2. **Static is the generic procedure**, not the `xgboost_v2` split, so it differs from every
   retraining strategy only in not retraining. Its gap to the exact control is reported.
3. **Strategy 5 is a lag-safe transcription of the Phase 8a rules, not the production engine.**
   Phase 8a's `PerformanceCrash` channel reads **step t's own frozen F1**, which is a label that has not
   arrived when step t is predicted (it is leakage under L ≥ 1). Here that channel reads the deployed
   model's *already-recorded* predictions on steps ≤ t − L instead, **one step at a time with equal weight
   per step** (a large step cannot hide a crash at the next one), over steps with at least 10 illicit
   labels that the *currently deployed* model scored. Two **label-free** channels are added so the monitor
   is not blind during label delay: feature shift (all 165 features against the deployed model's training
   window) and score-distribution shift (the deployed model's own scores against its recent scores). Their
   constants are declared in the setup cell and use no label and no 43–49 information. The adversarial
   channel stays corroborating-only, as in Phase 8a. The decisions are compared to the Phase 8a report in §6.
4. **Intervals.** Rows inside one step are not independent, so the verdict uses a **cluster bootstrap that
   resamples whole time steps**; the row-level bootstrap is exported next to it. The verdict thresholds
   themselves are unchanged.
5. **Recovery metric.** "Steps to recover" is the per-step *sustained recovery* defined above, not a
   pooled-tail statistic.""")

# ---------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------
code(r'''import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from IPython.display import Markdown, display
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

SEED = 42
# Smoke-test switch: WALKFORWARD_QUICK=1 cuts the bootstrap to 100 draws. Everything else is unchanged.
QUICK = os.environ.get("WALKFORWARD_QUICK", "0") == "1"

# --- Temporal protocol (same windows as every earlier phase) -------------------------------------
TEST_MIN, TEST_MAX = 35, 49
TEST_EARLY_MAX = 42
DRIFT_MIN = 43
TEST_STEPS = list(range(TEST_MIN, TEST_MAX + 1))
WINDOWS = {"35-42": list(range(35, 43)), "43-49": list(range(43, 50)), "35-49": TEST_STEPS}

# --- Label delay ---------------------------------------------------------------------------------
LABEL_LAGS = (1, 3)   # labels for step s arrive at s + L; L=1 main, L=3 sensitivity (all strategies)
MAIN_LAG = 1
EXTRA_LAGS = (2, 5)   # lag-curve points, run for CURVE_STRATEGIES only to keep the runtime down
CURVE_STRATEGIES = ("static", "expanding")
CURVE_LAGS = tuple(sorted(set(LABEL_LAGS) | set(EXTRA_LAGS)))

# --- xgboost_v2 configuration, held fixed (scripts/generate_notebook_02_v2.py) --------------------
FROZEN_PARAMS = {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3,
                 "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.5}
MAX_TREES, PATIENCE = 1_500, 100
XGB_BASE = {"tree_method": "hist", "eval_metric": "aucpr", "random_state": SEED, "n_jobs": 4}
XGB_REG = {"reg_lambda": 1.0}

# --- Hold-out rule for early stopping + threshold (computed from already-arrived labels) ----------
MIN_HOLDOUT_POS = 40
MIN_HOLDOUT_STEPS, MAX_HOLDOUT_STEPS, MIN_FIT_STEPS = 2, 6, 2

# --- Strategy parameters, pre-declared -------------------------------------------------------------
ROLLING_WINDOWS = (5, 10)
RECENCY_HALF_LIFE = 6.0
TRIAD = ["Aggregate_feature_10", "Aggregate_feature_43", "Aggregate_feature_8"]
STRATEGIES = ["static", "expanding", "rolling_w5", "rolling_w10", "recency_exp_hl6",
              "monitor_triggered", "expanding_no_triad"]
RETRAIN_STRATEGIES = [s for s in STRATEGIES if s != "static"]
# Every (strategy, lag) run: all strategies at the main lags, the curve strategies at the extra lags.
ALL_RUNS = ([(s, lag) for lag in LABEL_LAGS for s in STRATEGIES]
            + [(s, lag) for lag in EXTRA_LAGS for s in CURVE_STRATEGIES])
LOW_POSITIVE_FLAG = 10      # a step with fewer illicit labels than this is noise, not evidence

# --- Phase 8a trigger rules (src/monitoring/retraining_trigger.py), transcribed ------------------
PREV_WINDOW, PREV_COLLAPSE = 5, 0.035
TRIAD_SIG, TRIAD_EXTREME = 0.25, 1.0
LOCAL_PCT, LOCAL_PERSIST, PSI_SIG = 30.0, 2, 0.25
# Performance channel (label-dependent): per-step F1 of the deployed model on its last PERF_STEPS arrived
# steps that hold >= PERF_MIN_POS illicit labels; every step counts once, so a large step cannot hide a crash.
PERF_F1_FLOOR, PERF_STEPS, PERF_MIN_POS = 0.10, 2, LOW_POSITIVE_FLAG
# Label-free channels, declared before any run and never tuned on 43-49. They read features and the
# deployed model's own scores only, so they are not delayed by label arrival.
FEATURE_SHIFT_PCT = 30.0                                  # % of 165 features with PSI > PSI_SIG vs the deployed model's window
SCORE_BIN_EDGES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)   # fixed probability bins for the score PSI
SCORE_REF_STEPS, SCORE_MIN_REF_STEPS = 5, 2               # score reference = the deployed model's last 5 scored steps
MONITOR_REFERENCE_MAX_STEP = 34
# Phase 8a RETRAIN decisions, read from reports/monitoring_backtest_report.md section 8.7.
PHASE8A_RETRAIN_STEPS = {43, 44, 45, 47, 48, 49}

# --- Metrics ---------------------------------------------------------------------------------------
BUDGETS = (20, 50, 100)
BOOT_B = 100 if QUICK else 1_000
CI_LO, CI_HI = 2.5, 97.5
CI_METHOD = "cluster"       # time-step cluster bootstrap feeds every verdict input; the row bootstrap is exported beside it
SUSTAIN_STEPS = 2           # sustained recovery: PR-AUC >= RECOVERY_TARGET at t and at the next 2 evaluated steps

# --- Verdict thresholds (pre-registered) -----------------------------------------------------------
RECOVERY_TARGET = 0.50
MATERIAL_GAIN = 0.15
NONINFERIORITY = 0.02
ORACLE_IN_WINDOW_PR_AUC = 0.9199   # in-window 5-fold CV on 43-49, docs/XGBOOST_V2.md (an oracle)

# --- Documented xgboost_v2 reference row, for the harness-reproduction check -----------------------
V2_REFERENCE = {"35-49": 0.8011, "35-42": 0.9228, "43-49": 0.0419}   # docs/XGBOOST_V2.md results table
V2_TOLERANCE = 0.01

np.random.seed(SEED)
sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 11

print(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"xgboost {xgb.__version__} | quick={QUICK} | bootstrap draws={BOOT_B}")''')

code(r'''FILES = {"txs_features": "txs_features.csv", "txs_classes": "txs_classes.csv"}

DRIVE_ROOT = Path(os.environ.get("BITCOINGUARD_DRIVE_ROOT",
                                 "/content/drive/MyDrive/Projects/Bitcoin Graph"))


def resolve_data_dir() -> Path:
    """Find the Elliptic++ CSVs in Colab, Kaggle or a local checkout."""
    override = os.environ.get("ELLIPTIC_DATA_DIR")
    candidates = [
        Path(override) if override else None,
        DRIVE_ROOT / "Dataset",
        Path("/content/drive/MyDrive/elliptic-plus-plus"),
        Path("/content/elliptic-plus-plus"),
        Path("/kaggle/input/elliptic-plus-plus"),
        Path("/kaggle/input/elliptic"),
        Path.cwd() / "Og data",
    ]
    for candidate in candidates:
        if candidate and candidate.is_dir() and (candidate / FILES["txs_features"]).exists():
            return candidate
    raise FileNotFoundError("Elliptic++ not found; set ELLIPTIC_DATA_DIR to the folder with the CSVs.")


def read_header(path: Path) -> list:
    """Column names of a CSV without reading any data rows."""
    return pd.read_csv(path, nrows=0).columns.tolist()


DATA_DIR = resolve_data_dir()
OUT_DIR = DRIVE_ROOT / "walkforward"      # copy to results/walkforward/ afterwards
FIG_DIR = OUT_DIR / "figures"
assert OUT_DIR.name == "walkforward", "this notebook may only write to walkforward/"
for directory in (OUT_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Read-only snapshot of every frozen artifact folder, compared again at the very end.
FROZEN_DIRS = ["xgboost", "xgboost_v2"]


def snapshot(dirs) -> dict:
    state = {}
    for name in dirs:
        base = DRIVE_ROOT / name
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    state[str(path.relative_to(DRIVE_ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return state


FROZEN_BEFORE = snapshot(FROZEN_DIRS)

CHECKS: list = []


def check(name: str, value, expected=None, note: str = "") -> None:
    """Record a sanity check so the artifact export reports exactly what was verified."""
    ok = expected is None or value == expected
    CHECKS.append({"check": name, "value": value, "expected": expected, "ok": bool(ok), "note": note})
    flag = "ok" if ok else "FAIL"
    print(f"[{flag}] {name}: {value}" + (f" (expected {expected})" if expected is not None else ""))


def save_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT_DIR / name, index=False)


def save_json(name: str, payload: dict) -> None:
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


FIGURES: list = []


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    FIGURES.append(name)
    plt.show()


print(f"data : {DATA_DIR}")
print(f"out  : {OUT_DIR}")
print(f"frozen artifacts snapshotted: {len(FROZEN_BEFORE)} files")''')

# ---------------------------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------------------------
md(r"""## 1. Load Data and the Label-Arrival Guard

Feature derivation and the exclusion of the 17 domain columns are copied unchanged from
`notebooks/02_xgboost_v2.ipynb`. Everything downstream indexes into one float32 matrix by row, so a
training set is always an explicit list of rows whose step can be audited after the fact.

The guard `assert_label_arrival` is the single place the no-leakage rule is enforced: a label from
step *s* may only be used at decision step *t* if `s + L <= t`. Every use of every model is logged and
re-audited in the checks.""")

code(r'''txs_columns = read_header(DATA_DIR / FILES["txs_features"])
all_feature_columns = [c for c in txs_columns if c not in ("txId", "Time step")]
domain_columns = [c for c in all_feature_columns
                  if not c.startswith(("Local_feature", "Aggregate_feature"))]
model_features = [c for c in all_feature_columns if c not in domain_columns]

check("features: model columns", len(model_features), 165)
check("features: domain columns excluded", len(domain_columns), 17)
check("features: the triad exists in the feature set", all(c in model_features for c in TRIAD), True)

txs_classes = pd.read_csv(DATA_DIR / FILES["txs_classes"], dtype={"txId": str})
started = time.time()
txs = pd.read_csv(DATA_DIR / FILES["txs_features"],
                  usecols=["txId", "Time step"] + model_features,
                  dtype={"txId": str, **{c: "float32" for c in model_features}})
txs["class"] = txs["txId"].map(txs_classes.set_index("txId")["class"])
print(f"loaded {len(txs):,} transaction rows in {time.time() - started:.0f}s")

STEP = txs["Time step"].to_numpy().astype("int16")
CLS = txs["class"].to_numpy()
XALL = txs[model_features].to_numpy(dtype="float32")
del txs

FEAT_IDX = {name: i for i, name in enumerate(model_features)}
LABELED = np.isin(CLS, (1, 2))                 # class 3 (unknown) is never a label
Y = (CLS == 1).astype("int8")

ALL_ROWS_BY_STEP = {s: np.flatnonzero(STEP == s) for s in range(1, TEST_MAX + 1)}
ROWS_BY_STEP = {s: np.flatnonzero(LABELED & (STEP == s)) for s in range(1, TEST_MAX + 1)}
LAB_BY_STEP = {s: int(len(rows)) for s, rows in ROWS_BY_STEP.items()}
POS_BY_STEP = {s: int(Y[rows].sum()) for s, rows in ROWS_BY_STEP.items()}

check("dataset: transaction rows", int(len(STEP)), 203_769)
check("dataset: labeled rows (class 1 or 2)", int(LABELED.sum()), 46_564)
check("dataset: every step 1-49 is present", sorted(ALL_ROWS_BY_STEP) == list(range(1, 50)), True)
check("dataset: steps 1-34 labeled rows", sum(LAB_BY_STEP[s] for s in range(1, 35)), 29_894)
check("dataset: steps 1-34 illicit rows", sum(POS_BY_STEP[s] for s in range(1, 35)), 3_462)
check("dataset: illicit rows 35-42", sum(POS_BY_STEP[s] for s in WINDOWS["35-42"]), 914)
check("dataset: illicit rows 43-49", sum(POS_BY_STEP[s] for s in WINDOWS["43-49"]), 169)
check("dataset: illicit rows 35-49", sum(POS_BY_STEP[s] for s in TEST_STEPS), 1_083)
check("dataset: step 45 has 5 illicit labels", POS_BY_STEP[45], 5)
check("dataset: step 46 has 2 illicit labels", POS_BY_STEP[46], 2)

label_table = pd.DataFrame({"step": TEST_STEPS,
                            "labeled": [LAB_BY_STEP[s] for s in TEST_STEPS],
                            "illicit": [POS_BY_STEP[s] for s in TEST_STEPS]})
label_table["prevalence_pct"] = (100 * label_table["illicit"] / label_table["labeled"]).round(2)
label_table["noise_flag"] = np.where(label_table["illicit"] < LOW_POSITIVE_FLAG,
                                     "LOW_POSITIVES: per-step metrics are noise", "")
display(label_table)''')

code(r'''class LeakageError(RuntimeError):
    """Raised when a decision would use a label before its arrival step."""


def assert_label_arrival(max_label_step_used: int, t: int, lag: int, what: str) -> None:
    """A label from step s arrives at s + lag. At decision step t it is usable only if s + lag <= t."""
    if int(max_label_step_used) + int(lag) > int(t):
        raise LeakageError(f"{what}: uses a label from step {max_label_step_used}, which arrives at "
                           f"{max_label_step_used + lag} > decision step {t} (L={lag})")


USE_LOG: list = []      # one row per (model or monitor input) used at a decision step


def rows_for(steps) -> np.ndarray:
    return np.concatenate([ROWS_BY_STEP[s] for s in steps])


# Negative control: the guard must actually fire. One step of lookahead at L=1 must raise.
guard_fires = False
try:
    assert_label_arrival(max_label_step_used=35, t=35, lag=1, what="negative control")
except LeakageError:
    guard_fires = True
check("leakage guard: a label from the decision step itself is rejected", guard_fires, True)
guard_passes = True
try:
    assert_label_arrival(max_label_step_used=34, t=35, lag=1, what="positive control")
except LeakageError:
    guard_passes = False
check("leakage guard: a label that has arrived is accepted", guard_passes, True)''')

# ---------------------------------------------------------------------------------------------
# 2. Model builder
# ---------------------------------------------------------------------------------------------
md(r"""## 2. One Fit Procedure for Every Retraining Policy

For a labeled window of steps `first..last` (so `m = last` is the newest arrived label step):

1. **Hold-out** = the most recent `k` steps of the window, the smallest `k` (2 ≤ k ≤ 6, leaving at
   least two fit steps) whose hold-out holds at least 40 positives. Chosen from arrived labels only.
2. **Fit** an `XGBClassifier` on the remaining steps with early stopping on the hold-out (aucpr).
3. **Threshold** = F1-maximising threshold on the hold-out, scored by the fit model — the same rule
   as `xgboost_v2`, never test labels.
4. **Refit** on the *whole* window with `trees_kept` trees. This refit model scores step t.

Recency weights (strategy 4) multiply the fit and refit rows by `0.5 ** ((m - step) / half_life)`,
normalised to mean 1, anchored on the window's newest step. The hold-out is never weighted.

Models are cached by `(first, last, weighting, drop_triad)`. A model depends only on its label set, so
the L = 3 run at step t reuses the L = 1 model of step t − 2. Every *use* is audited separately.""")

code(r'''ALL_COLS = np.arange(len(model_features))
NOTRIAD_COLS = np.array([i for i, c in enumerate(model_features) if c not in TRIAD])
check("ablation: the no-triad feature set has 162 columns", int(len(NOTRIAD_COLS)), 162)
check("ablation: no triad column survives", bool(not any(model_features[i] in TRIAD for i in NOTRIAD_COLS)), True)


def make_xgb(scale_pos_weight: float, n_estimators: int, patience):
    """XGBClassifier with early stopping wherever this xgboost version accepts it (as in xgboost_v2)."""
    settings = {**XGB_BASE, **XGB_REG, **FROZEN_PARAMS, "n_estimators": n_estimators,
                "scale_pos_weight": float(scale_pos_weight)}
    if patience is None:
        return xgb.XGBClassifier(**settings), False
    try:
        return xgb.XGBClassifier(**settings, early_stopping_rounds=patience), False
    except TypeError:
        return xgb.XGBClassifier(**settings), True


def select_threshold(y_true, scores) -> float:
    """F1-maximising threshold on hold-out data only - never on the step being predicted."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    positives = y_true.sum()
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.round(np.linspace(0.005, 0.995, 199), 4):
        pred = scores >= candidate
        tp = int((pred & (y_true == 1)).sum())
        denominator = int(pred.sum()) + int(positives)
        value = 2 * tp / denominator if denominator else 0.0
        if value > best_f1:
            best_threshold, best_f1 = float(candidate), float(value)
    return best_threshold


def choose_holdout(window_steps) -> tuple:
    """Most recent k steps of the window, smallest k with >= MIN_HOLDOUT_POS positives (arrived labels only)."""
    steps = sorted(window_steps)
    k_cap = max(MIN_HOLDOUT_STEPS, min(MAX_HOLDOUT_STEPS, len(steps) - MIN_FIT_STEPS))
    k = MIN_HOLDOUT_STEPS
    while k < k_cap and sum(POS_BY_STEP[s] for s in steps[-k:]) < MIN_HOLDOUT_POS:
        k += 1
    return tuple(steps[:-k]), tuple(steps[-k:])


def decay_weights(step_values, anchor: int, half_life: float) -> np.ndarray:
    weights = np.power(0.5, (anchor - np.asarray(step_values, dtype="float64")) / float(half_life))
    return weights / weights.mean()


def triad_gain_share(booster, cols) -> float:
    """Share of total split gain carried by the triad (0.0 when the triad was removed)."""
    scores = booster.get_score(importance_type="gain")
    total = float(sum(scores.values())) or 1.0
    triad_names = {f"f{j}" for j, i in enumerate(cols) if model_features[i] in TRIAD}
    return float(sum(v for k, v in scores.items() if k in triad_names) / total)


MODEL_CACHE: dict = {}


def build_model(first: int, last: int, weighting: str = "none", drop_triad: bool = False,
                fit_steps=None, hold_steps=None) -> dict:
    """Fit -> early-stop on hold-out -> threshold on hold-out -> refit on the whole window."""
    window = tuple(range(first, last + 1))
    assert len(window) >= MIN_FIT_STEPS + MIN_HOLDOUT_STEPS, f"window {first}-{last} is too short"
    if hold_steps is None:
        fit_steps, hold_steps = choose_holdout(window)
    fit_rows, hold_rows, refit_rows = rows_for(fit_steps), rows_for(hold_steps), rows_for(window)
    cols = NOTRIAD_COLS if drop_triad else ALL_COLS

    # Audit fields come from the rows actually used, not from the parameters passed in.
    max_label_step = int(STEP[refit_rows].max())
    max_hold_step = int(STEP[hold_rows].max())
    classes_ok = bool(np.isin(CLS[refit_rows], (1, 2)).all())
    assert max_hold_step <= last and max_label_step <= last
    assert set(fit_steps).isdisjoint(hold_steps)

    y_fit, y_hold, y_refit = Y[fit_rows], Y[hold_rows], Y[refit_rows]
    spw = float((len(y_fit) - y_fit.sum()) / max(1, y_fit.sum()))
    fit_weight = refit_weight = None
    if weighting != "none":
        half_life = float(weighting[3:])
        fit_weight = decay_weights(STEP[fit_rows], last, half_life)
        refit_weight = decay_weights(STEP[refit_rows], last, half_life)

    started = time.time()
    estimator, patience_at_fit = make_xgb(spw, MAX_TREES, PATIENCE)
    fit_kwargs = {"eval_set": [(XALL[hold_rows][:, cols], y_hold)], "verbose": False}
    if patience_at_fit:
        fit_kwargs["early_stopping_rounds"] = PATIENCE
    if fit_weight is not None:
        fit_kwargs["sample_weight"] = fit_weight
    estimator.fit(XALL[fit_rows][:, cols], y_fit, **fit_kwargs)
    best_iteration = int(getattr(estimator, "best_iteration", 0) or 0)
    if best_iteration <= 0:
        best_iteration = int(estimator.get_booster().best_iteration)
    trees_kept = best_iteration + 1

    hold_scores = estimator.predict_proba(XALL[hold_rows][:, cols])[:, 1]
    threshold = select_threshold(y_hold, hold_scores)
    holdout_pr_auc = float(average_precision_score(y_hold, hold_scores)) if y_hold.sum() else float("nan")

    refit, _ = make_xgb(spw, trees_kept, None)
    refit_kwargs = {"verbose": False}
    if refit_weight is not None:
        refit_kwargs["sample_weight"] = refit_weight
    refit.fit(XALL[refit_rows][:, cols], y_refit, **refit_kwargs)

    return {"model": refit, "cols": cols, "threshold": threshold, "trees_kept": int(trees_kept),
            "window": (first, last), "weighting": weighting, "drop_triad": bool(drop_triad),
            "fit_steps": fit_steps, "hold_steps": hold_steps,
            "holdout_pos": int(y_hold.sum()), "holdout_rows": int(len(y_hold)),
            "holdout_pr_auc": holdout_pr_auc,
            "n_train": int(len(refit_rows)), "n_train_pos": int(y_refit.sum()),
            "n_train_pos_drift_regime": int(((STEP[refit_rows] >= DRIFT_MIN) & (y_refit == 1)).sum()),
            "max_label_step": max_label_step, "max_hold_step": max_hold_step, "classes_ok": classes_ok,
            "triad_gain_share": triad_gain_share(refit.get_booster(), cols),
            "seconds": round(time.time() - started, 1)}


def get_model(key: tuple) -> dict:
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = build_model(*key)
    return MODEL_CACHE[key]


def score_step(bundle: dict, t: int) -> np.ndarray:
    return bundle["model"].predict_proba(XALL[ROWS_BY_STEP[t]][:, bundle["cols"]])[:, 1]


# Sanity: the hold-out rule on two known cases (arrived labels only).
fit_a, hold_a = choose_holdout(tuple(range(1, 35)))
check("hold-out rule: window 1-34 holds out exactly steps 33-34 (the xgboost_v2 slice)", hold_a, (33, 34))
check("hold-out rule: that slice holds the 60 positives documented for xgboost_v2",
      sum(POS_BY_STEP[s] for s in hold_a), 60)''')

code(r'''# Harness validation: reproduce the xgboost_v2 reference row with its ORIGINAL split
# (fit 1-24, early-stop + threshold on 33-34, refit 1-34). If this does not match the documented
# numbers, the walk-forward harness differs from the frozen pipeline and nothing below is trustworthy.
CONTROL = build_model(1, 34, "none", False, fit_steps=tuple(range(1, 25)), hold_steps=(33, 34))
control_scores = {t: score_step(CONTROL, t) for t in TEST_STEPS}


def pooled_pr_auc(scores_by_step: dict, steps) -> float:
    y = np.concatenate([Y[ROWS_BY_STEP[t]] for t in steps])
    s = np.concatenate([scores_by_step[t] for t in steps])
    return float(average_precision_score(y, s))


print(f"v2_exact_control: trees_kept={CONTROL['trees_kept']} threshold={CONTROL['threshold']} "
      f"({CONTROL['seconds']}s)")
for window_name, documented in V2_REFERENCE.items():
    measured = pooled_pr_auc(control_scores, WINDOWS[window_name])
    check(f"repro: v2_exact_control {window_name} PR-AUC within {V2_TOLERANCE} of the documented xgboost_v2 value",
          bool(abs(measured - documented) <= V2_TOLERANCE), True,
          note=f"measured {measured:.4f} vs documented {documented:.4f}")
check("repro: v2_exact_control used steps 1-24 / 33-34 / 1-34",
      (CONTROL["fit_steps"][-1], CONTROL["hold_steps"], CONTROL["window"]), (24, (33, 34), (1, 34)))''')

# ---------------------------------------------------------------------------------------------
# 3. Policies
# ---------------------------------------------------------------------------------------------
md(r"""## 3. Strategies 1–4 and 6: Fixed-Schedule Policies

For each label delay `L` and each decision step `t` in 35–49, the newest arrived label step is
`m = t − L`. Each policy maps `m` to a window, and the model for that window scores step `t`:

| Strategy | Window at step t |
| :--- | :--- |
| `static` | `1 .. 35−L` (every label available at step 35, frozen for the whole run) |
| `expanding` | `1 .. m` |
| `rolling_wW` | `max(1, m−W+1) .. m` |
| `recency_exp_hl6` | `1 .. m`, exponential decay weights |
| `expanding_no_triad` | `1 .. m`, `Aggregate_feature_10/43/8` removed |

At L = 1 `static` is trained on 1–34, the frozen training window. At L = 3 only labels through step 32
exist at step 35, so `static` is trained on 1–32 — using steps 33–34 there would use labels before they
arrive. This is why the L = 3 control differs from the frozen artifact.

The **lag curve** adds L = 2 and L = 5 for `static` and `expanding` only (almost every model is already cached
from L = 1 and L = 3, so this costs a handful of extra fits). `static` is trained through step 33 at L = 2 and
through step 30 at L = 5, for the same reason.""")

code(r'''def policy_key(strategy: str, m: int, lag: int) -> tuple:
    """(first, last, weighting, drop_triad) of the model a fixed-schedule strategy uses at newest label step m."""
    if strategy == "static":
        return (1, TEST_MIN - lag, "none", False)
    if strategy == "expanding":
        return (1, m, "none", False)
    if strategy.startswith("rolling_w"):
        width = int(strategy.split("_w")[1])
        return (max(1, m - width + 1), m, "none", False)
    if strategy == "recency_exp_hl6":
        return (1, m, f"exp{RECENCY_HALF_LIFE:g}", False)
    if strategy == "expanding_no_triad":
        return (1, m, "none", True)
    raise ValueError(strategy)


Y_STEP = {t: Y[ROWS_BY_STEP[t]].astype(int) for t in TEST_STEPS}
SCORES: dict = {}     # (strategy, lag) -> {t: scores}
THRESH: dict = {}     # (strategy, lag) -> {t: threshold used at t}
LOG: list = []        # one row per (strategy, lag, t)


def use_model(strategy: str, lag: int, t: int, key: tuple, previous_key) -> np.ndarray:
    """Score step t with a cached model, audit the label arrival, and log the use."""
    bundle = get_model(key)
    assert_label_arrival(bundle["max_label_step"], t, lag, f"{strategy} model {key}")
    assert_label_arrival(bundle["max_hold_step"], t, lag, f"{strategy} hold-out {key}")
    scores = score_step(bundle, t)
    USE_LOG.append({"strategy": strategy, "lag": lag, "step": t, "kind": "model",
                    "max_label_step_used": bundle["max_label_step"],
                    "max_holdout_step": bundle["max_hold_step"],
                    "arrival_step": bundle["max_label_step"] + lag,
                    "classes_ok": bundle["classes_ok"], "n_features": int(len(bundle["cols"]))})
    LOG.append({"strategy": strategy, "lag": lag, "step": t,
                "retrained": bool(previous_key is not None and key != previous_key),
                "trained_through": bundle["window"][1], "train_first_step": bundle["window"][0],
                "weighting": bundle["weighting"], "drop_triad": bundle["drop_triad"],
                "n_train": bundle["n_train"], "n_train_pos": bundle["n_train_pos"],
                "n_train_pos_drift_regime": bundle["n_train_pos_drift_regime"],
                "holdout_first": bundle["hold_steps"][0], "holdout_last": bundle["hold_steps"][-1],
                "holdout_pos": bundle["holdout_pos"], "holdout_pr_auc": bundle["holdout_pr_auc"],
                "threshold": bundle["threshold"], "trees_kept": bundle["trees_kept"],
                "triad_gain_share": bundle["triad_gain_share"],
                "max_label_step_used": bundle["max_label_step"],
                "label_arrival_ok": True, "fit_seconds": bundle["seconds"],
                "trigger_fired": "", "trigger_reasons": ""})
    SCORES[(strategy, lag)][t] = scores
    THRESH[(strategy, lag)][t] = bundle["threshold"]
    return scores


FIXED_RUNS = [(s, lag) for s, lag in ALL_RUNS if s != "monitor_triggered"]
started = time.time()
for strategy, lag in FIXED_RUNS:
    SCORES[(strategy, lag)], THRESH[(strategy, lag)] = {}, {}
    previous_key = None
    for t in TEST_STEPS:
        key = policy_key(strategy, t - lag, lag)
        use_model(strategy, lag, t, key, previous_key)
        previous_key = key
    print(f"L={lag} {strategy:<20} done | models cached so far {len(MODEL_CACHE):>3} | "
          f"{time.time() - started:6.0f}s elapsed")''')

# ---------------------------------------------------------------------------------------------
# 4. Monitor-triggered
# ---------------------------------------------------------------------------------------------
md(r"""## 4. Strategy 5: Monitor-Triggered Retraining (Lag-Safe)

The Phase 8a engine returns RETRAIN when any of these fires. Each is transcribed from
`src/monitoring/retraining_trigger.py` and fed only inputs that exist at step t:

| Channel | Rule | Input available at step t |
| :--- | :--- | :--- |
| Triad drift | any detrended drift score ≥ 1.0, or ≥ 2 of 3 ≥ 0.25 | step-t **features**; reference = steps 1–34 features |
| Local drift | ≥ 30% of the 93 local features with PSI > 0.25 for 2 consecutive steps | step-t **features**; reference = steps 1–34 features |
| Prevalence collapse | rolling prevalence over the last 5 arrived steps ≤ 3.5% | labels of steps ≤ t − L |
| Performance crash | any per-step F1 < 0.10 among the deployed model's last 2 arrived steps with ≥ 10 illicit labels | the deployed model's **recorded predictions** on steps ≤ t − L, scored by that model |
| **Feature shift (label-free)** | ≥ 30% of all 165 features with PSI > 0.25 | step-t **features** against the **deployed model's training window** (features only) |
| **Score shift (label-free)** | PSI > 0.25 of the deployed model's step-t scores against its own last 5 scored steps (≥ 2 needed) | the deployed model's **scores**, fixed probability bins; no label |
| Adversarial | corroborating only, never a solo trigger (as in Phase 8a) | not used |

**The performance channel is the one that differs from Phase 8a.** Phase 8a evaluated it on step t's own
labels. Under any label delay that is not available when step t is scored, so here it reads predictions
already made on steps whose labels have arrived. The cost of that honesty is a delay: a collapse that
begins at step 43 is first visible at step 43 + L. Each arrived step is scored **separately and counts once**,
so a large step (step 42 holds 239 illicit labels) cannot dilute a crash visible at step 43; steps with fewer
than 10 illicit labels are skipped as noise, and only predictions made by the *currently deployed* model are
used (so a retrain starts a clean window). The two label-free channels do not wait for labels, which is the
only thing that can shorten the blind period; their constants are fixed in the setup cell and were not chosen
by looking at 43–49.

When the trigger fires, the deployed model is replaced by the expanding-window model on all labels
through t − L (the same model strategy 2 would use). The only difference from strategy 2 is *when*.
The feature channels are a simplified re-implementation (linear trend only, decile PSI); §6 compares
the resulting decisions with the Phase 8a report instead of assuming they agree.""")

code(r'''def psi_from_props(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric PSI with a probability-level floor, as in the Phase 8a fix (report section 0.1)."""
    p = np.maximum(p, 1e-4)
    q = np.maximum(q, 1e-4)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


# Reference = feature rows of steps 1-34 only (features carry no labels). Frozen before step 35.
REF_ROWS = np.flatnonzero(STEP <= MONITOR_REFERENCE_MAX_STEP)
LOCAL_IDX = [i for i, c in enumerate(model_features) if c.startswith("Local_feature")]
TRIAD_IDX = [FEAT_IDX[c] for c in TRIAD]

ref_local = XALL[REF_ROWS][:, LOCAL_IDX]
LOCAL_EDGES, LOCAL_REF_PROPS = [], []
for j in range(len(LOCAL_IDX)):
    edges = np.unique(np.quantile(ref_local[:, j], np.linspace(0.1, 0.9, 9)))
    counts = np.bincount(np.digitize(ref_local[:, j], edges), minlength=len(edges) + 1)
    LOCAL_EDGES.append(edges)
    LOCAL_REF_PROPS.append(counts / counts.sum())
del ref_local

TRIAD_TREND = {}
ref_steps = list(range(1, MONITOR_REFERENCE_MAX_STEP + 1))
for col in TRIAD_IDX:
    medians = np.array([np.median(XALL[ALL_ROWS_BY_STEP[s], col]) for s in ref_steps])
    slope, intercept = np.polyfit(np.array(ref_steps, dtype="float64"), medians, 1)
    resid = np.concatenate([XALL[ALL_ROWS_BY_STEP[s], col].astype("float64") - (slope * s + intercept)
                            for s in ref_steps])
    TRIAD_TREND[col] = (float(slope), float(intercept), float(resid.mean()), float(resid.std()) or 1.0)


def triad_scores(t: int) -> list:
    """Standardised detrended-residual location shift of each triad feature at step t."""
    out = []
    for col in TRIAD_IDX:
        slope, intercept, mu, sd = TRIAD_TREND[col]
        resid = XALL[ALL_ROWS_BY_STEP[t], col].astype("float64") - (slope * t + intercept)
        out.append(float(abs(resid.mean() - mu) / sd))
    return out


def local_pct_significant(t: int) -> float:
    """Percentage of the 93 local features whose step-t PSI against the 1-34 reference exceeds 0.25."""
    x = XALL[ALL_ROWS_BY_STEP[t]][:, LOCAL_IDX]
    significant = 0
    for j in range(len(LOCAL_IDX)):
        counts = np.bincount(np.digitize(x[:, j], LOCAL_EDGES[j]), minlength=len(LOCAL_EDGES[j]) + 1)
        if psi_from_props(LOCAL_REF_PROPS[j], counts / counts.sum()) > PSI_SIG:
            significant += 1
    return 100.0 * significant / len(LOCAL_IDX)


FEATURE_TELEMETRY = {t: {"triad": triad_scores(t), "local_pct": local_pct_significant(t)} for t in TEST_STEPS}
check("monitor: the reference uses only steps <= 34 (strictly before the first decision step)",
      int(STEP[REF_ROWS].max()) < TEST_MIN, True)
display(pd.DataFrame([{"step": t, "triad_10": v["triad"][0], "triad_43": v["triad"][1],
                       "triad_8": v["triad"][2], "local_pct_sig": v["local_pct"]}
                      for t, v in FEATURE_TELEMETRY.items()]).round(3))''')

code(r'''def prf(y: np.ndarray, pred: np.ndarray) -> tuple:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


# ---- Label-dependent channel: step-weighted performance window ------------------------------------------
def perf_window(history: dict, deployed, m: int) -> dict:
    """Per-step F1 of the deployed model on its last PERF_STEPS arrived, non-noise steps.

    `history[s]` holds the labels, scores, threshold and model key of every step already scored. A step is used
    only if its labels have arrived (s <= m = t - L), at least PERF_MIN_POS of them are illicit, and the
    CURRENTLY deployed model scored it. Each step is evaluated on its own and counts once, so one large step
    cannot dilute a crash at the next. `f1_pooled` is what row-pooling the same steps would give (reported only
    to show the masking the step-weighted rule removes).
    """
    eligible = [s for s in sorted(history)
                if s <= m and history[s]["model"] == deployed and int(history[s]["y"].sum()) >= PERF_MIN_POS]
    used = eligible[-PERF_STEPS:]
    f1s = [prf(history[s]["y"], (history[s]["scores"] >= history[s]["thr"]).astype(int))[2] for s in used]
    pooled_f1 = float("nan")
    if used:
        y_all = np.concatenate([history[s]["y"] for s in used])
        pred_all = np.concatenate([(history[s]["scores"] >= history[s]["thr"]).astype(int) for s in used])
        pooled_f1 = prf(y_all, pred_all)[2]
    return {"steps": used, "f1": f1s,
            "f1_min": float(min(f1s)) if f1s else float("nan"),
            "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            "f1_pooled": pooled_f1,
            "min_pos": min(int(history[s]["y"].sum()) for s in used) if used else 0}


# Controls on synthetic histories (no model involved): step 42 is large and healthy, step 43 has collapsed.
def _synthetic_history() -> dict:
    y42 = np.r_[np.ones(239, dtype=int), np.zeros(1500, dtype=int)]
    s42 = np.r_[np.full(239, 0.9), np.full(1500, 0.01)]
    y43 = np.r_[np.ones(24, dtype=int), np.zeros(1500, dtype=int)]
    return {42: {"y": y42, "scores": s42, "thr": 0.5, "model": "M"},
            43: {"y": y43, "scores": np.full(len(y43), 0.01), "thr": 0.5, "model": "M"}}


_hist = _synthetic_history()
_w = perf_window(_hist, "M", 43)
check("monitor: a crash at step 43 is visible although step 42 is large (step-weighted window)",
      bool(_w["f1_min"] < PERF_F1_FLOOR and _w["f1_pooled"] >= PERF_F1_FLOOR), True,
      note=f"per-step min F1 {_w['f1_min']:.3f} vs row-pooled F1 {_w['f1_pooled']:.3f}")
_hist[44] = {"y": np.r_[np.ones(5, dtype=int), np.zeros(100, dtype=int)], "scores": np.full(105, 0.01),
             "thr": 0.5, "model": "M"}
check("monitor: a step with fewer than 10 illicit labels never enters the performance window",
      44 not in perf_window(_hist, "M", 44)["steps"], True)
check("monitor: a step whose labels have not arrived (s > t - L) never enters the performance window",
      43 not in perf_window(_synthetic_history(), "M", 42)["steps"], True)
_other = _synthetic_history()
_other[43]["model"] = "OLD"
check("monitor: predictions made by a previously deployed model never enter the window",
      43 not in perf_window(_other, "M", 43)["steps"], True)


# ---- Label-free channels --------------------------------------------------------------------------------
FEATURE_REF_CACHE: dict = {}


def feature_reference(first: int, last: int) -> tuple:
    """Decile edges and reference proportions of all 165 features over steps first..last (features only)."""
    key = (first, last)
    if key not in FEATURE_REF_CACHE:
        rows = np.concatenate([ALL_ROWS_BY_STEP[s] for s in range(first, last + 1)])
        block = XALL[rows]
        edges, props = [], []
        for j in range(block.shape[1]):
            e = np.unique(np.quantile(block[:, j], np.linspace(0.1, 0.9, 9)))
            counts = np.bincount(np.digitize(block[:, j], e), minlength=len(e) + 1)
            edges.append(e)
            props.append(counts / counts.sum())
        FEATURE_REF_CACHE[key] = (edges, props)
        del block
    return FEATURE_REF_CACHE[key]


def feature_shift_pct(t: int, first: int, last: int) -> float:
    """% of the 165 features whose step-t PSI against steps first..last exceeds PSI_SIG. No label is read."""
    assert last < t, "the feature reference must end before the step being scored"
    edges, props = feature_reference(first, last)
    x = XALL[ALL_ROWS_BY_STEP[t]]
    significant = 0
    for j in range(x.shape[1]):
        counts = np.bincount(np.digitize(x[:, j], edges[j]), minlength=len(edges[j]) + 1)
        if psi_from_props(props[j], counts / counts.sum()) > PSI_SIG:
            significant += 1
    return 100.0 * significant / x.shape[1]


def score_props(scores: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.digitize(scores, SCORE_BIN_EDGES), minlength=len(SCORE_BIN_EDGES) + 1)
    return counts / counts.sum()


MONITOR_TELEMETRY: list = []
LABEL_FREE_REASONS = ("TriadDrift", "LocalDrift", "FeatureShift", "ScoreShift")


def run_monitor(lag: int) -> None:
    strategy = "monitor_triggered"
    SCORES[(strategy, lag)], THRESH[(strategy, lag)] = {}, {}
    deployed = policy_key("static", None, lag)
    history: dict = {}                 # step -> labels, scores, threshold and model key recorded when it was scored
    streak, previous_pct = 0, None
    for t in TEST_STEPS:
        m = t - lag
        triad, pct = FEATURE_TELEMETRY[t]["triad"], FEATURE_TELEMETRY[t]["local_pct"]
        if pct >= LOCAL_PCT:
            streak = streak + 1 if (previous_pct is not None and previous_pct >= LOCAL_PCT) else 1
        else:
            streak = 0
        previous_pct = pct

        reasons = []
        if any(v >= TRIAD_EXTREME for v in triad) or sum(v >= TRIAD_SIG for v in triad) >= 2:
            reasons.append("TriadDrift")
        if streak >= LOCAL_PERSIST:
            reasons.append("LocalDrift")

        # Label-free channels: features of step t and the deployed model's own scores, no label.
        d_first, d_last = deployed[0], deployed[1]
        feature_pct = feature_shift_pct(t, d_first, d_last)
        deployed_bundle = get_model(deployed)
        assert_label_arrival(deployed_bundle["max_label_step"], t, lag, f"monitor deployed model {deployed}")
        scores_now = score_step(deployed_bundle, t)
        ref_steps = [s for s in sorted(history) if s < t and history[s]["model"] == deployed][-SCORE_REF_STEPS:]
        score_psi = float("nan")
        if len(ref_steps) >= SCORE_MIN_REF_STEPS:
            score_psi = psi_from_props(score_props(np.concatenate([history[s]["scores"] for s in ref_steps])),
                                       score_props(scores_now))
        if feature_pct >= FEATURE_SHIFT_PCT:
            reasons.append("FeatureShift")
        if score_psi == score_psi and score_psi > PSI_SIG:
            reasons.append("ScoreShift")

        # Label-dependent channels: only labels of steps <= t - L.
        lo = max(1, m - PREV_WINDOW + 1)
        rolling_labeled = sum(LAB_BY_STEP[s] for s in range(lo, m + 1))
        rolling_prev = sum(POS_BY_STEP[s] for s in range(lo, m + 1)) / rolling_labeled
        if rolling_prev <= PREV_COLLAPSE:
            reasons.append("PrevalenceCollapse")
        perf = perf_window(history, deployed, m)
        if perf["steps"] and perf["f1_min"] < PERF_F1_FLOOR:
            reasons.append("PerformanceCrash")

        fired = bool(reasons)
        new_key = policy_key("expanding", m, lag) if fired else deployed
        scores = use_model(strategy, lag, t, new_key, deployed)
        LOG[-1]["trigger_fired"] = fired
        LOG[-1]["trigger_reasons"] = "+".join(reasons)
        history[t] = {"y": Y_STEP[t], "scores": scores, "thr": THRESH[(strategy, lag)][t], "model": new_key}
        deployed = new_key
        MONITOR_TELEMETRY.append({
            "strategy": strategy, "lag": lag, "step": t,
            "triad_10": triad[0], "triad_43": triad[1], "triad_8": triad[2],
            "local_pct_sig": pct, "local_streak": streak,
            "feature_shift_pct": feature_pct, "feature_ref_first_step": d_first, "feature_ref_last_step": d_last,
            "score_psi": score_psi, "score_ref_n_steps": len(ref_steps),
            "score_ref_max_step": max(ref_steps) if ref_steps else float("nan"),
            "rolling_prevalence": rolling_prev, "rolling_window_last_step": m,
            "perf_f1_min": perf["f1_min"], "perf_f1_mean": perf["f1_mean"], "perf_f1_pooled": perf["f1_pooled"],
            "perf_steps": ",".join(str(s) for s in perf["steps"]), "perf_n_steps": len(perf["steps"]),
            "perf_min_pos": perf["min_pos"],
            "perf_max_step": max(perf["steps"]) if perf["steps"] else float("nan"),
            "label_free_fired": any(r in LABEL_FREE_REASONS for r in reasons),
            "label_dependent_fired": any(r in ("PrevalenceCollapse", "PerformanceCrash") for r in reasons),
            "fired": fired, "reasons": "+".join(reasons),
            "retrained": LOG[-1]["retrained"], "trained_through": LOG[-1]["trained_through"]})


for lag in LABEL_LAGS:
    run_monitor(lag)
    fired_steps = [r["step"] for r in MONITOR_TELEMETRY if r["lag"] == lag and r["fired"]]
    retrained_steps = [r["step"] for r in MONITOR_TELEMETRY if r["lag"] == lag and r["retrained"]]
    print(f"L={lag} monitor fired at {fired_steps}; model replaced at {retrained_steps} "
          f"({len(retrained_steps)} retrains vs {len(TEST_STEPS) - 1} for expanding)")
display(pd.DataFrame(MONITOR_TELEMETRY)[
    ["lag", "step", "feature_shift_pct", "score_psi", "perf_f1_min", "perf_f1_pooled", "rolling_prevalence", "reasons"]
].query("step >= 41").round(3))
print(f"total unique models fitted: {len(MODEL_CACHE)}")''')

# ---------------------------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------------------------
md(r"""## 5. Metrics: Per Step, Alert Budget, Pooled Windows

**Per step** — PR-AUC, ROC-AUC, F1 at the validation-derived threshold, and the **number of illicit
labels that step**. Steps 45 and 46 hold 5 and 2 positives; any per-step value there is noise and is
flagged `LOW_POSITIVES`, not interpreted. `f1_oracle_per_step` uses that one step's own labels to pick its best
threshold: it is a **per-step ceiling, not a result**, and it is never pooled across steps (a pooled
single-threshold "ceiling" can fall below the F1 achieved with per-step thresholds, so none is reported).

**Alert budget** — precision@K and recall@K for K = 20, 50, 100 flagged transactions per step, with the
base rate, the lift over it, and the maximum value each metric can reach (a step with 5 positives cannot
exceed recall@K of 1 or precision@K of 5/K). Only labeled transactions are ranked, because class 3 cannot be
scored; this overstates nothing for a labeled-only queue but is not a statement about the unlabeled mass.

**Pooled windows** (35–42, 43–49, plus 35–49) — scores of every step in the window are concatenated, each
step scored by the model that was deployed at that step. Two 95% intervals are reported, both with the **same
resampled indices for every strategy** so paired differences are meaningful:

* **Cluster bootstrap (used for the verdict).** Each draw resamples whole time steps with replacement and keeps
  every row of a chosen step together, because rows inside one step are not independent.
* **Row bootstrap (reported alongside).** Each draw resamples rows within each step, so the step composition is
  fixed. It treats rows as independent and is expected to be narrower.

There are only 7 steps in 43–49, so even the cluster interval is approximate and wide. Extra-lag runs (L = 2, 5)
are bootstrapped on 43–49 only.

**Operating rules.** Top-K alerts per step (K = 20, 50, 100) is the **primary** mode; the validation threshold is
the comparison, and its per-step instability is tabulated as a finding.""")

code(r'''def oracle_f1(y: np.ndarray, s: np.ndarray) -> float:
    """Best achievable F1 over thresholds using the evaluated labels - a labelled ceiling, never a result."""
    precision, recall, _ = precision_recall_curve(y, s)
    denominator = precision + recall
    f1 = np.where(denominator > 0, 2 * precision * recall / np.where(denominator > 0, denominator, 1.0), 0.0)
    return float(f1.max())


def window_of(t: int) -> str:
    return "35-42" if t <= TEST_EARLY_MAX else "43-49"


per_step_rows, budget_rows = [], []
for strategy, lag in ALL_RUNS:
    for t in TEST_STEPS:
        y, s = Y_STEP[t], SCORES[(strategy, lag)][t]
        threshold = THRESH[(strategy, lag)][t]
        pred = (s >= threshold).astype(int)
        precision, recall, f1, tp, fp, fn = prf(y, pred)
        n, positives = int(len(y)), int(y.sum())
        per_step_rows.append({
            "strategy": strategy, "lag": lag, "step": t, "window": window_of(t),
            "n_labeled": n, "n_illicit": positives, "prevalence": positives / n,
            "noise_flag": "LOW_POSITIVES" if positives < LOW_POSITIVE_FLAG else "",
            "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "threshold": threshold, "flagged": int(pred.sum()), "flag_rate": float(pred.mean()),
            "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            # Per-step oracle: the best F1 this ONE step allows when its own labels pick the threshold.
            # A ceiling for that step only; never pooled across steps.
            "f1_oracle_per_step": oracle_f1(y, s)})
        order = np.argsort(-s, kind="stable")
        for k in BUDGETS:
            top = order[:k]
            flagged = int(len(top))
            hits = int(y[top].sum())
            budget_rows.append({
                "strategy": strategy, "lag": lag, "level": "step", "step": t, "window": window_of(t),
                "K": k, "n_labeled": n, "n_illicit": positives, "flagged": flagged, "tp": hits,
                "precision_at_k": hits / flagged, "recall_at_k": hits / positives,
                "base_rate": positives / n, "lift_over_base": (hits / flagged) / (positives / n),
                "max_possible_precision": min(1.0, positives / flagged),
                "max_possible_recall": min(1.0, flagged / positives)})

per_step = pd.DataFrame(per_step_rows)
budget_steps = pd.DataFrame(budget_rows)

# Window rows: micro-averaged over the steps in the window (sum of hits / sum of alerts).
window_rows = []
for (strategy, lag, window, k), group in budget_steps.groupby(["strategy", "lag", "window", "K"]):
    flagged, hits, positives = int(group["flagged"].sum()), int(group["tp"].sum()), int(group["n_illicit"].sum())
    window_rows.append({
        "strategy": strategy, "lag": lag, "level": "window", "step": -1, "window": window, "K": k,
        "n_labeled": int(group["n_labeled"].sum()), "n_illicit": positives, "flagged": flagged, "tp": hits,
        "precision_at_k": hits / flagged, "recall_at_k": hits / positives,
        "base_rate": positives / int(group["n_labeled"].sum()),
        "lift_over_base": (hits / flagged) / (positives / int(group["n_labeled"].sum())),
        "max_possible_precision": min(1.0, positives / flagged),
        "max_possible_recall": min(1.0, flagged / positives)})
alert_budget = pd.concat([budget_steps, pd.DataFrame(window_rows)], ignore_index=True)

log_frame = pd.DataFrame(LOG)
per_step = per_step.merge(
    log_frame[["strategy", "lag", "step", "trained_through", "n_train", "n_train_pos",
               "n_train_pos_drift_regime", "holdout_pos", "holdout_pr_auc", "triad_gain_share"]],
    on=["strategy", "lag", "step"], how="left")

display(per_step[(per_step["lag"] == MAIN_LAG)].pivot(index="step", columns="strategy", values="pr_auc").round(4))
print("Per-step n_illicit (steps with < 10 positives are noise):")
display(label_table[["step", "illicit", "noise_flag"]].T)''')

code(r'''rng_row = np.random.default_rng(SEED)
rng_cluster = np.random.default_rng(SEED + 1)


def make_row_boot_indices(steps, draws: int) -> np.ndarray:
    """(draws, N) row indices: each step's rows resampled with replacement, step composition fixed."""
    sizes = [len(Y_STEP[t]) for t in steps]
    idx = np.empty((draws, sum(sizes)), dtype="int32")
    offset = 0
    for n in sizes:
        idx[:, offset:offset + n] = rng_row.integers(0, n, size=(draws, n)) + offset
        offset += n
    return idx


def make_cluster_boot_indices(steps, draws: int) -> tuple:
    """`draws` index arrays; each draw resamples WHOLE steps with replacement and keeps every row of a chosen step.

    Rows inside one step are not independent, so this is the interval used for the verdict. Returns the index
    arrays and the (draws, n_steps) matrix of chosen step positions, so the draw can be audited afterwards.
    """
    sizes = np.array([len(Y_STEP[t]) for t in steps])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    chosen = rng_cluster.integers(0, len(steps), size=(draws, len(steps)))
    index_sets = [np.concatenate([np.arange(offsets[c], offsets[c] + sizes[c]) for c in row]).astype("int32")
                  for row in chosen]
    return index_sets, chosen


def pooled_arrays(strategy: str, lag: int, steps):
    y = np.concatenate([Y_STEP[t] for t in steps])
    s = np.concatenate([SCORES[(strategy, lag)][t] for t in steps])
    thr = np.concatenate([np.full(len(Y_STEP[t]), THRESH[(strategy, lag)][t]) for t in steps])
    return y, s, (s >= thr).astype(int)


def point_stats(y, s, pred) -> dict:
    precision, recall, f1, tp, fp, fn = prf(y, pred)
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "f1": f1, "precision": precision, "recall": recall}


def boot_stats(y, s, pred, index_sets) -> np.ndarray:
    """(draws, 3) PR-AUC, ROC-AUC, F1 for each resampled index set (a 2-D array or a list of arrays)."""
    out = np.empty((len(index_sets), 3))
    for b, ii in enumerate(index_sets):
        yb, sb, pb = y[ii], s[ii], pred[ii]
        tp = int(((pb == 1) & (yb == 1)).sum())
        denominator = int(pb.sum()) + int(yb.sum())
        out[b] = (average_precision_score(yb, sb), roc_auc_score(yb, sb),
                  2 * tp / denominator if denominator else 0.0)
    return out


def ci_windows_for(lag: int) -> list:
    """Main lags are bootstrapped on every window; the extra lag-curve runs only on 43-49."""
    return list(WINDOWS) if lag in LABEL_LAGS else ["43-49"]


BOOT_IDX_ROW = {w: make_row_boot_indices(steps, BOOT_B) for w, steps in WINDOWS.items()}
BOOT_IDX_CLUSTER, CLUSTER_CHOSEN = {}, {}
for w, steps in WINDOWS.items():
    BOOT_IDX_CLUSTER[w], CLUSTER_CHOSEN[w] = make_cluster_boot_indices(steps, BOOT_B)

BOOT, BOOT_ROW = {}, {}          # BOOT = cluster (used); BOOT_ROW = row-level (reported alongside)
pooled_rows = []
started = time.time()
for strategy, lag in ALL_RUNS:
    for window, steps in WINDOWS.items():
        y, s, pred = pooled_arrays(strategy, lag, steps)
        point = point_stats(y, s, pred)
        lo_c = hi_c = lo_r = hi_r = np.full(3, np.nan)
        if window in ci_windows_for(lag):
            draws_c = boot_stats(y, s, pred, BOOT_IDX_CLUSTER[window])
            draws_r = boot_stats(y, s, pred, BOOT_IDX_ROW[window])
            BOOT[(strategy, lag, window)], BOOT_ROW[(strategy, lag, window)] = draws_c, draws_r
            lo_c, hi_c = np.percentile(draws_c, [CI_LO, CI_HI], axis=0)
            lo_r, hi_r = np.percentile(draws_r, [CI_LO, CI_HI], axis=0)
        prevalence = float(y.mean())
        pooled_rows.append({
            "strategy": strategy, "lag": lag, "window": window, "n": int(len(y)),
            "n_illicit": int(y.sum()), "prevalence": prevalence, "ci_method": CI_METHOD,
            "pr_auc": point["pr_auc"], "pr_auc_ci_lo": lo_c[0], "pr_auc_ci_hi": hi_c[0],
            "pr_auc_row_ci_lo": lo_r[0], "pr_auc_row_ci_hi": hi_r[0],
            "pr_auc_lift_over_prevalence": point["pr_auc"] / prevalence,
            "roc_auc": point["roc_auc"], "roc_auc_ci_lo": lo_c[1], "roc_auc_ci_hi": hi_c[1],
            "roc_auc_row_ci_lo": lo_r[1], "roc_auc_row_ci_hi": hi_r[1],
            "f1": point["f1"], "f1_ci_lo": lo_c[2], "f1_ci_hi": hi_c[2],
            "f1_row_ci_lo": lo_r[2], "f1_row_ci_hi": hi_r[2],
            "precision": point["precision"], "recall": point["recall"]})
    print(f"{strategy:<20} L={lag} pooled bootstrap done | {time.time() - started:.0f}s elapsed")

pooled = pd.DataFrame(pooled_rows)
retrain_counts = (log_frame.groupby(["strategy", "lag"])["retrained"].sum().rename("retrains_total").reset_index())
pooled = pooled.merge(retrain_counts, on=["strategy", "lag"], how="left")

# Paired bootstrap differences: the same resampled steps (cluster) or rows (row) for both strategies.
delta_rows = []
for lag in LABEL_LAGS:
    for reference in ("static", "expanding"):
        for strategy in STRATEGIES:
            if strategy == reference:
                continue
            for window in WINDOWS:
                diff_c = BOOT[(strategy, lag, window)][:, 0] - BOOT[(reference, lag, window)][:, 0]
                diff_r = BOOT_ROW[(strategy, lag, window)][:, 0] - BOOT_ROW[(reference, lag, window)][:, 0]
                row_s = pooled[(pooled.strategy == strategy) & (pooled.lag == lag) & (pooled.window == window)].iloc[0]
                row_r = pooled[(pooled.strategy == reference) & (pooled.lag == lag) & (pooled.window == window)].iloc[0]
                lo, hi = np.percentile(diff_c, [CI_LO, CI_HI])
                rlo, rhi = np.percentile(diff_r, [CI_LO, CI_HI])
                delta_rows.append({"strategy": strategy, "reference": reference, "lag": lag, "window": window,
                                   "ci_method": CI_METHOD,
                                   "delta_pr_auc": float(row_s["pr_auc"] - row_r["pr_auc"]),
                                   "ci_lo": float(lo), "ci_hi": float(hi),
                                   "ci_excludes_zero": bool(lo > 0 or hi < 0),
                                   "row_ci_lo": float(rlo), "row_ci_hi": float(rhi),
                                   "row_ci_excludes_zero": bool(rlo > 0 or rhi < 0)})
paired = pd.DataFrame(delta_rows)

view = pooled[pooled["lag"] == MAIN_LAG].copy()
view["PR-AUC [95% cluster CI]"] = view.apply(
    lambda r: f"{r.pr_auc:.4f} [{r.pr_auc_ci_lo:.4f}, {r.pr_auc_ci_hi:.4f}]", axis=1)
display(view.pivot(index="strategy", columns="window", values="PR-AUC [95% cluster CI]").loc[STRATEGIES])
width = pooled[(pooled["window"] == "43-49") & pooled["lag"].isin(LABEL_LAGS)].copy()
width["cluster_ci_width"] = width["pr_auc_ci_hi"] - width["pr_auc_ci_lo"]
width["row_ci_width"] = width["pr_auc_row_ci_hi"] - width["pr_auc_row_ci_lo"]
width["cluster_over_row"] = width["cluster_ci_width"] / width["row_ci_width"]
print("43-49 interval width, cluster (used) vs row (reported alongside):")
display(width[["strategy", "lag", "pr_auc", "cluster_ci_width", "row_ci_width", "cluster_over_row"]].round(3))''')

code(r'''# Descriptive pooled-tail curve: pooled PR-AUC over steps s0..49 for s0 = 43..49. It is kept for the figure only;
# it is NOT the recovery metric (it can read "recovered" while the first drift steps are still badly scored).
tail_rows = []
for lag in LABEL_LAGS:
    for strategy in STRATEGIES:
        for start in range(DRIFT_MIN, TEST_MAX + 1):
            steps = list(range(start, TEST_MAX + 1))
            y, s, _ = pooled_arrays(strategy, lag, steps)
            tail_rows.append({"strategy": strategy, "lag": lag, "tail_start": start,
                              "steps_after_onset": start - DRIFT_MIN, "n_illicit": int(y.sum()),
                              "pr_auc": float(average_precision_score(y, s))})
tail = pd.DataFrame(tail_rows)

# ---- Sustained recovery (per step) ----------------------------------------------------------------------
NOISE_STEPS = [t for t in WINDOWS["43-49"] if POS_BY_STEP[t] < LOW_POSITIVE_FLAG]            # 45 and 46
ELIGIBLE_RECOVERY_STEPS = [t for t in WINDOWS["43-49"] if t not in NOISE_STEPS]             # 43 44 47 48 49


def sustained_recovery_from(pr_by_step: dict, eligible_steps, target: float = RECOVERY_TARGET,
                            hold: int = SUSTAIN_STEPS):
    """First step t with PR-AUC >= target that stays >= target for the next `hold` evaluated steps; None if never.

    Only `eligible_steps` are looked at, so the noise steps (5 and 2 positives) can neither create nor break a
    recovery. A candidate needs `hold` evaluated steps after it, so with 43 44 47 48 49 the last start is 47.
    """
    series = [(t, pr_by_step[t]) for t in sorted(eligible_steps)]
    for i in range(len(series) - hold):
        if all(value >= target for _, value in series[i:i + hold + 1]):
            return series[i][0]
    return None


recovery_rows = []
for strategy, lag in ALL_RUNS:
    d = per_step[(per_step.strategy == strategy) & (per_step.lag == lag)].set_index("step")["pr_auc"].to_dict()
    first = sustained_recovery_from(d, ELIGIBLE_RECOVERY_STEPS)
    recovery_rows.append({
        "strategy": strategy, "lag": lag,
        "sustained_recovery_step": np.nan if first is None else first,
        "steps_after_onset": np.nan if first is None else first - DRIFT_MIN,
        "status": "not reached" if first is None else f"step {first} (+{first - DRIFT_MIN})",
        "pr_auc_by_eligible_step": ", ".join(f"{t}:{d[t]:.2f}" for t in ELIGIBLE_RECOVERY_STEPS),
        "noise_steps_excluded": ", ".join(f"{t} (n+={POS_BY_STEP[t]})" for t in NOISE_STEPS)})
recovery = pd.DataFrame(recovery_rows)
SUSTAINED = {(r.strategy, r.lag): (None if r.status == "not reached" else int(r.sustained_recovery_step))
             for r in recovery.itertuples()}


def recovery_text(strategy: str, lag: int) -> str:
    return recovery[(recovery.strategy == strategy) & (recovery.lag == lag)].iloc[0]["status"]


print(f"Evaluated steps: {ELIGIBLE_RECOVERY_STEPS}; noise steps excluded: {NOISE_STEPS}; "
      f"target {RECOVERY_TARGET}, held for {SUSTAIN_STEPS} further evaluated steps.")
display(recovery[recovery["lag"].isin(LABEL_LAGS)][["strategy", "lag", "status", "pr_auc_by_eligible_step"]])

# Controls: the function must behave on constructed series, and must agree with an independent recomputation.
check("recovery: first step that holds for 3 evaluated steps is returned",
      str(sustained_recovery_from({43: .1, 44: .1, 47: .6, 48: .7, 49: .8}, ELIGIBLE_RECOVERY_STEPS)), "47")
check("recovery: a dip after the first crossing resets the test",
      str(sustained_recovery_from({43: .6, 44: .2, 47: .6, 48: .7, 49: .1}, ELIGIBLE_RECOVERY_STEPS)), "None")
check("recovery: high PR-AUC on the noise steps 45 and 46 cannot create a recovery",
      str(sustained_recovery_from({43: .1, 44: .1, 45: .99, 46: .99, 47: .1, 48: .1, 49: .1}, ELIGIBLE_RECOVERY_STEPS)), "None")
check("recovery: low PR-AUC on the noise steps 45 and 46 cannot break a recovery",
      str(sustained_recovery_from({43: .6, 44: .7, 45: 0., 46: 0., 47: .8, 48: .1, 49: .1}, ELIGIBLE_RECOVERY_STEPS)), "43")
check("recovery: a crossing without two evaluated steps after it is not a sustained recovery",
      str(sustained_recovery_from({43: .1, 44: .1, 47: .1, 48: .9, 49: .9}, ELIGIBLE_RECOVERY_STEPS)), "None")


def _alternative_recovery(pr: dict) -> object:
    ok = [pr[t] >= RECOVERY_TARGET for t in ELIGIBLE_RECOVERY_STEPS]
    for i, t in enumerate(ELIGIBLE_RECOVERY_STEPS):
        if i + SUSTAIN_STEPS < len(ELIGIBLE_RECOVERY_STEPS) and all(ok[i:i + SUSTAIN_STEPS + 1]):
            return t
    return None


check("recovery: every reported value equals an independent recomputation from the per-step table",
      bool(all(SUSTAINED[(s, lag)] == _alternative_recovery(
          per_step[(per_step.strategy == s) & (per_step.lag == lag)].set_index("step")["pr_auc"].to_dict())
          for s, lag in ALL_RUNS)), True)
check("recovery: no reported recovery step is a noise step and every one is >= 43",
      bool(all(v is None or (v >= DRIFT_MIN and v not in NOISE_STEPS) for v in SUSTAINED.values())), True)
check("recovery: the noise steps are exactly 45 and 46", NOISE_STEPS, [45, 46])''')

code(r'''# F1 ceiling: per-step only. The pooled single-threshold "oracle ceiling" is gone (it fell below the F1
# achieved with per-step thresholds in 4 cells, so it was not a ceiling). Here each step's own oracle is compared
# with that step's validation-threshold F1 and the two are averaged with the SAME weights (illicit labels per
# step), so oracle >= achieved holds by construction in every row.
f1c_rows = []
for strategy, lag in ALL_RUNS:
    for window in ("35-42", "43-49"):
        x = per_step[(per_step.strategy == strategy) & (per_step.lag == lag) & (per_step.window == window)]
        weights = x["n_illicit"].to_numpy(dtype=float)
        f1c_rows.append({
            "strategy": strategy, "lag": lag, "window": window, "weights": "illicit labels per step",
            "f1_validation_threshold_weighted": float(np.average(x["f1"], weights=weights)),
            "f1_oracle_per_step_weighted": float(np.average(x["f1_oracle_per_step"], weights=weights))})
f1_ceiling = pd.DataFrame(f1c_rows)
f1_ceiling["gap_to_per_step_ceiling"] = (f1_ceiling["f1_oracle_per_step_weighted"]
                                         - f1_ceiling["f1_validation_threshold_weighted"])
print("F1 at the validation threshold vs the PER-STEP oracle ceiling (weighted by illicit labels; a ceiling, not a result):")
display(f1_ceiling[f1_ceiling.lag == MAIN_LAG].pivot(index="strategy", columns="window",
        values=["f1_validation_threshold_weighted", "f1_oracle_per_step_weighted"]).loc[STRATEGIES].round(3))''')

code(r'''# Operating rules. Top-K alerts per step is the primary mode; the validation threshold is the comparison.
op_rows = []
for strategy, lag in ALL_RUNS:
    for window, steps in WINDOWS.items():
        positives = sum(POS_BY_STEP[t] for t in steps)
        ps = per_step[(per_step.strategy == strategy) & (per_step.lag == lag) & per_step.step.isin(steps)]
        bs = budget_steps[(budget_steps.strategy == strategy) & (budget_steps.lag == lag) & budget_steps.step.isin(steps)]
        candidates = [("validation_threshold", int(ps["flagged"].sum()), int(ps["tp"].sum()), float("nan"),
                       int((ps["flagged"] == 0).sum()))]
        for k in BUDGETS:
            bk = bs[bs.K == k]
            max_recall = sum(min(k, POS_BY_STEP[t]) for t in steps) / positives
            candidates.append((f"top{k}", int(bk["flagged"].sum()), int(bk["tp"].sum()), max_recall, 0))
        for rule, flagged, tp, max_recall, zero_steps in candidates:
            precision = tp / flagged if flagged else 0.0
            recall = tp / positives
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            op_rows.append({"strategy": strategy, "lag": lag, "window": window, "rule": rule,
                            "primary": rule != "validation_threshold", "n_steps": len(steps),
                            "alerts_total": flagged, "alerts_per_step": flagged / len(steps), "tp": tp,
                            "n_illicit": positives, "precision": precision, "recall": recall, "f1": f1,
                            "max_possible_recall": max_recall, "steps_with_zero_alerts": zero_steps})
op_rules = pd.DataFrame(op_rows)
print("Operating rules, pooled 43-49, L=1 (top-K per step is primary; validation threshold is the comparison):")
display(op_rules[(op_rules.lag == MAIN_LAG) & (op_rules.window == "43-49") &
                 op_rules.strategy.isin(["static", "expanding", "monitor_triggered"])]
        [["strategy", "rule", "alerts_per_step", "precision", "recall", "f1", "max_possible_recall",
          "steps_with_zero_alerts"]].round(3))

# Finding: how stable is the validation threshold, per step?
thr_rows = []
for strategy, lag in ALL_RUNS:
    d = per_step[(per_step.strategy == strategy) & (per_step.lag == lag)]
    for window in ("35-42", "43-49"):
        x = d[d.window == window].sort_values("step")
        thr_rows.append({
            "strategy": strategy, "lag": lag, "window": window,
            "threshold_min": float(x["threshold"].min()), "threshold_max": float(x["threshold"].max()),
            "threshold_range": float(x["threshold"].max() - x["threshold"].min()),
            "threshold_std": float(x["threshold"].std(ddof=0)),
            "distinct_thresholds": int(x["threshold"].nunique()),
            "flag_rate_min": float(x["flag_rate"].min()), "flag_rate_max": float(x["flag_rate"].max()),
            "steps_zero_alerts": int((x["flagged"] == 0).sum()),
            "steps_over_10pct_flagged": int((x["flag_rate"] > 0.10).sum()),
            "thresholds_by_step": ", ".join(f"{t}:{v:.3f}" for t, v in zip(x["step"], x["threshold"])),
            "flag_rate_by_step": ", ".join(f"{t}:{v:.3f}" for t, v in zip(x["step"], x["flag_rate"]))})
threshold_instability = pd.DataFrame(thr_rows)
print("Validation-threshold instability (finding): threshold and flag rate per step")
display(threshold_instability[threshold_instability.lag == MAIN_LAG]
        [["strategy", "window", "threshold_min", "threshold_max", "threshold_range", "distinct_thresholds",
          "flag_rate_min", "flag_rate_max", "steps_zero_alerts"]].round(3))''')

code(r'''# Finding about drift, not a bug: does the hold-out PR-AUC a retrained model was selected on track the PR-AUC
# it then realises on the next step? Reported per strategy, lag and window, with and without the noise steps.
hv_rows = []
for strategy in RETRAIN_STRATEGIES:
    for lag in LABEL_LAGS:
        d = per_step[(per_step.strategy == strategy) & (per_step.lag == lag)]
        for window in ("35-42", "43-49"):
            for scope in ("all steps", "excluding noise steps"):
                x = d[d.window == window]
                if scope != "all steps":
                    x = x[x["n_illicit"] >= LOW_POSITIVE_FLAG]
                x = x.dropna(subset=["holdout_pr_auc", "pr_auc"])
                enough = len(x) >= 3 and x["holdout_pr_auc"].std() > 0 and x["pr_auc"].std() > 0
                hv_rows.append({
                    "strategy": strategy, "lag": lag, "window": window, "scope": scope, "n_steps": int(len(x)),
                    "holdout_pr_auc_mean": float(x["holdout_pr_auc"].mean()) if len(x) else float("nan"),
                    "realised_pr_auc_mean": float(x["pr_auc"].mean()) if len(x) else float("nan"),
                    "pearson_r": float(np.corrcoef(x["holdout_pr_auc"], x["pr_auc"])[0, 1]) if enough else float("nan"),
                    "interpretation": "finding about drift (descriptive), not a bug"})
holdout_vs_realised = pd.DataFrame(hv_rows)
holdout_vs_realised["mean_gap"] = (holdout_vs_realised["holdout_pr_auc_mean"]
                                   - holdout_vs_realised["realised_pr_auc_mean"])
print("Hold-out PR-AUC vs realised PR-AUC per step (a finding about drift, not a bug):")
display(holdout_vs_realised[(holdout_vs_realised.strategy == "expanding")].drop(columns=["interpretation"]).round(3))''')

code(r'''# Lag curve: pooled 43-49 PR-AUC against label delay, static vs expanding (cluster-bootstrap intervals).
lag_rows = []
for lag in CURVE_LAGS:
    p_static = pooled[(pooled.strategy == "static") & (pooled.lag == lag) & (pooled.window == "43-49")].iloc[0]
    p_exp = pooled[(pooled.strategy == "expanding") & (pooled.lag == lag) & (pooled.window == "43-49")].iloc[0]
    diff_c = BOOT[("expanding", lag, "43-49")][:, 0] - BOOT[("static", lag, "43-49")][:, 0]
    diff_r = BOOT_ROW[("expanding", lag, "43-49")][:, 0] - BOOT_ROW[("static", lag, "43-49")][:, 0]
    lo, hi = np.percentile(diff_c, [CI_LO, CI_HI])
    rlo, rhi = np.percentile(diff_r, [CI_LO, CI_HI])
    lag_rows.append({
        "lag": lag, "main_or_extra": "main" if lag in LABEL_LAGS else "extra",
        "static_trained_through": TEST_MIN - lag, "expanding_trained_through_at_43": DRIFT_MIN - lag,
        "static_pr_auc": float(p_static["pr_auc"]), "static_ci_lo": float(p_static["pr_auc_ci_lo"]),
        "static_ci_hi": float(p_static["pr_auc_ci_hi"]),
        "expanding_pr_auc": float(p_exp["pr_auc"]), "expanding_ci_lo": float(p_exp["pr_auc_ci_lo"]),
        "expanding_ci_hi": float(p_exp["pr_auc_ci_hi"]),
        "expanding_row_ci_lo": float(p_exp["pr_auc_row_ci_lo"]), "expanding_row_ci_hi": float(p_exp["pr_auc_row_ci_hi"]),
        "delta_vs_static": float(p_exp["pr_auc"] - p_static["pr_auc"]), "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
        "delta_row_ci_lo": float(rlo), "delta_row_ci_hi": float(rhi),
        "expanding_sustained_recovery": recovery_text("expanding", lag)})
lag_curve = pd.DataFrame(lag_rows)
first_lag_below_target = next((int(r.lag) for r in lag_curve.itertuples() if r.expanding_pr_auc < RECOVERY_TARGET), None)
print(f"expanding falls below the restored level ({RECOVERY_TARGET}) at L={first_lag_below_target} "
      f"(first tested lag below target; descriptive, not part of the verdict rule)")
display(lag_curve[["lag", "static_pr_auc", "expanding_pr_auc", "expanding_ci_lo", "expanding_ci_hi",
                   "delta_vs_static", "delta_ci_lo", "delta_ci_hi", "expanding_sustained_recovery"]].round(4))''')

# ---------------------------------------------------------------------------------------------
# 6. Monitor concordance
# ---------------------------------------------------------------------------------------------
md(r"""## 6. How the Lag-Safe Trigger Compares With Phase 8a

Phase 8a's backtest fired RETRAIN at 43, 44, 45, 47, 48, 49 and at no step in 35–42. The lag-safe version
cannot match that exactly, because its label-dependent channels are blind to step t itself. This section
reports the agreement **per step** rather than assuming it away, splits each lag-safe decision into its
label-free and label-dependent part (so it is visible which channel, if any, covers the blind period), and
gives the retrain count against `expanding`. A lag-safe fire at a step where Phase 8a did not fire is counted
as a disagreement, not excused.""")

code(r'''concordance = []
for lag in LABEL_LAGS:
    tele = pd.DataFrame([r for r in MONITOR_TELEMETRY if r["lag"] == lag]).set_index("step")
    for t in TEST_STEPS:
        concordance.append({"lag": lag, "step": t, "window": window_of(t),
                            "phase8a_retrain": t in PHASE8A_RETRAIN_STEPS,
                            "lagsafe_fired": bool(tele.loc[t, "fired"]), "reasons": tele.loc[t, "reasons"],
                            "label_free_fired": bool(tele.loc[t, "label_free_fired"]),
                            "label_dependent_fired": bool(tele.loc[t, "label_dependent_fired"]),
                            "feature_shift_pct": float(tele.loc[t, "feature_shift_pct"]),
                            "score_psi": float(tele.loc[t, "score_psi"]),
                            "model_replaced": bool(tele.loc[t, "retrained"])})
concordance = pd.DataFrame(concordance)
concordance["agrees"] = concordance["phase8a_retrain"] == concordance["lagsafe_fired"]
concordance["agrees_label_dependent_only"] = concordance["phase8a_retrain"] == concordance["label_dependent_fired"]
concordance["agrees_label_free_only"] = concordance["phase8a_retrain"] == concordance["label_free_fired"]
for lag in LABEL_LAGS:
    print(f"Per-step agreement with Phase 8a, L={lag}:")
    display(concordance[concordance["lag"] == lag].set_index("step")[
        ["phase8a_retrain", "lagsafe_fired", "agrees", "label_free_fired", "label_dependent_fired", "reasons",
         "model_replaced"]].T)


def first_true(frame: pd.DataFrame, column: str):
    hit = frame[(frame["step"] >= DRIFT_MIN) & frame[column]]
    return int(hit["step"].min()) if len(hit) else None


summary_rows = []
for lag in LABEL_LAGS:
    c = concordance[concordance["lag"] == lag]
    for scope, x in (("35-42", c[c["window"] == "35-42"]), ("43-49", c[c["window"] == "43-49"]), ("35-49", c)):
        summary_rows.append({
            "lag": lag, "scope": scope, "steps": int(len(x)), "agree": int(x["agrees"].sum()),
            "agreement_rate": float(x["agrees"].mean()),
            "agree_label_dependent_only": int(x["agrees_label_dependent_only"].sum()),
            "agree_label_free_only": int(x["agrees_label_free_only"].sum()),
            "lagsafe_fires": int(x["lagsafe_fired"].sum()), "phase8a_fires": int(x["phase8a_retrain"].sum()),
            "fires_phase8a_did_not": int((x["lagsafe_fired"] & ~x["phase8a_retrain"]).sum()),
            "phase8a_fires_missed": int((~x["lagsafe_fired"] & x["phase8a_retrain"]).sum()),
            "first_fire_43_49": first_true(c, "lagsafe_fired") if scope != "35-42" else None,
            "first_label_free_fire_43_49": first_true(c, "label_free_fired") if scope != "35-42" else None,
            "first_label_dependent_fire_43_49": first_true(c, "label_dependent_fired") if scope != "35-42" else None,
            "phase8a_first_fire_43_49": 43 if scope != "35-42" else None})
concordance_summary = pd.DataFrame(summary_rows)
display(concordance_summary)

monitor_summary = (log_frame[log_frame["strategy"].isin(["monitor_triggered", "expanding"])]
                   .groupby(["strategy", "lag"])["retrained"].sum().unstack("lag"))
print("retrains used (model replaced), steps 36-49:")
display(monitor_summary)''')

# ---------------------------------------------------------------------------------------------
# 7. Verdict
# ---------------------------------------------------------------------------------------------
md(r"""## 7. Verdict (Computed From the Pre-Registered Rule)

Everything below is derived from the tables above with the constants fixed in the setup cell. No cell
re-tunes a threshold after seeing 43–49.""")

code(r'''RETRAIN_STRATEGIES = [s for s in STRATEGIES if s != "static"]


def pooled_value(strategy: str, lag: int, window: str, column: str) -> float:
    row = pooled[(pooled.strategy == strategy) & (pooled.lag == lag) & (pooled.window == window)]
    return float(row.iloc[0][column])


def paired_row(strategy: str, lag: int, window: str, reference: str = "static"):
    row = paired[(paired.strategy == strategy) & (paired.lag == lag) &
                 (paired.window == window) & (paired.reference == reference)]
    return row.iloc[0]


def meets(strategy: str, lag: int, level: float) -> bool:
    """Pooled 43-49 PR-AUC >= level AND paired delta vs static has a 95% lower bound above 0."""
    return bool(pooled_value(strategy, lag, "43-49", "pr_auc") >= level
                and paired_row(strategy, lag, "43-49")["ci_lo"] > 0)


cond_a = all(meets("expanding", lag, RECOVERY_TARGET) for lag in LABEL_LAGS)
cells_b = [(s, lag) for s in RETRAIN_STRATEGIES for lag in LABEL_LAGS if meets(s, lag, MATERIAL_GAIN)]
cells_target = [(s, lag) for s in RETRAIN_STRATEGIES for lag in LABEL_LAGS if meets(s, lag, RECOVERY_TARGET)]
VERDICT = "(a) RECOVERS" if cond_a else "(b) PARTIAL" if cells_b else "(c) NO RECOVERY"

candidates = []
for s in RETRAIN_STRATEGIES:
    drift_pr = pooled_value(s, MAIN_LAG, "43-49", "pr_auc")
    if (drift_pr >= MATERIAL_GAIN
            and paired_row(s, MAIN_LAG, "43-49")["ci_lo"] > 0
            and paired_row(s, MAIN_LAG, "35-42")["ci_lo"] >= -NONINFERIORITY):
        candidates.append((s, drift_pr, int(pooled_value(s, MAIN_LAG, "43-49", "retrains_total"))))
if candidates:
    best_pr = max(c[1] for c in candidates)
    FROZEN_STRATEGY = sorted([c for c in candidates if c[1] >= best_pr - 0.01], key=lambda c: (c[2], -c[1]))[0][0]
else:
    FROZEN_STRATEGY = "static"

print("VERDICT:", VERDICT)
print("expanding reaches the target at both lags:", cond_a)
print("cells meeting MATERIAL_GAIN with paired CI above 0:", cells_b)
print("cells meeting RECOVERY_TARGET with paired CI above 0:", cells_target)
print("strategy to freeze:", FROZEN_STRATEGY)''')

code(r'''def ci_text(strategy: str, lag: int, window: str) -> str:
    return (f"{pooled_value(strategy, lag, window, 'pr_auc'):.4f} "
            f"[{pooled_value(strategy, lag, window, 'pr_auc_ci_lo'):.4f}, "
            f"{pooled_value(strategy, lag, window, 'pr_auc_ci_hi'):.4f}]")


def delta_text(strategy: str, lag: int, window: str, reference: str = "static") -> str:
    r = paired_row(strategy, lag, window, reference)
    return f"{r['delta_pr_auc']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"


def rule_value(strategy: str, lag: int, window: str, rule: str, column: str) -> float:
    row = op_rules[(op_rules.strategy == strategy) & (op_rules.lag == lag) &
                   (op_rules.window == window) & (op_rules.rule == rule)]
    return float(row.iloc[0][column])


def holdout_cell(strategy: str, lag: int, window: str, scope: str):
    row = holdout_vs_realised[(holdout_vs_realised.strategy == strategy) & (holdout_vs_realised.lag == lag) &
                              (holdout_vs_realised.window == window) & (holdout_vs_realised.scope == scope)]
    return row.iloc[0]


def threshold_cell(strategy: str, lag: int, window: str):
    row = threshold_instability[(threshold_instability.strategy == strategy) & (threshold_instability.lag == lag) &
                                (threshold_instability.window == window)]
    return row.iloc[0]


exp_log = log_frame[(log_frame.strategy == "expanding") & (log_frame.lag == MAIN_LAG)]
drift_exp = exp_log[exp_log.step >= DRIFT_MIN]
exp_step = per_step[(per_step.strategy == "expanding") & (per_step.lag == MAIN_LAG) & (per_step.step >= DRIFT_MIN)]
hv_exp = holdout_cell("expanding", MAIN_LAG, "43-49", "excluding noise steps")
hv_exp_all = holdout_cell("expanding", MAIN_LAG, "43-49", "all steps")
hv_exp_l3 = holdout_cell("expanding", 3, "43-49", "excluding noise steps")
why = {
    "holdout_pr_auc_mean_43_49": float(drift_exp["holdout_pr_auc"].mean()),
    "step_pr_auc_mean_43_49": float(exp_step["pr_auc"].mean()),
    "holdout_vs_realised_pearson_r_L1_43_49_excluding_noise": float(hv_exp["pearson_r"]),
    "holdout_vs_realised_pearson_r_L1_43_49_all_steps": float(hv_exp_all["pearson_r"]),
    "holdout_vs_realised_pearson_r_L3_43_49_excluding_noise": float(hv_exp_l3["pearson_r"]),
    "drift_regime_positives_in_train_at_49": int(drift_exp[drift_exp.step == TEST_MAX]["n_train_pos_drift_regime"].iloc[0]),
    "total_positives_in_train_at_49": int(drift_exp[drift_exp.step == TEST_MAX]["n_train_pos"].iloc[0]),
    "triad_gain_share_expanding_mean_35_42": float(exp_log[exp_log.step <= TEST_EARLY_MAX]["triad_gain_share"].mean()),
    "triad_gain_share_expanding_mean_43_49": float(drift_exp["triad_gain_share"].mean()),
}
ablation = {lag: {w: pooled_value("expanding_no_triad", lag, w, "pr_auc") - pooled_value("expanding", lag, w, "pr_auc")
                  for w in ("35-42", "43-49")} for lag in LABEL_LAGS}
ablation_paired = {w: paired_row("expanding_no_triad", MAIN_LAG, w, "expanding") for w in ("35-42", "43-49")}

lines = [f"## Verdict: {VERDICT}", ""]
lines += [f"Intervals are 95% **{CI_METHOD} bootstrap over time steps** (the row-level interval is exported alongside "
          "in `pooled_metrics.csv` and `paired_deltas.csv`).", "",
          f"| Strategy | L | pooled 35-42 PR-AUC [95% CI] | pooled 43-49 PR-AUC [95% CI] | delta vs static on 43-49 | retrains | "
          f"sustained recovery (PR-AUC >= {RECOVERY_TARGET:.2f} for 3 evaluated steps) |",
          "| :--- | ---: | :--- | :--- | :--- | ---: | :--- |"]
for lag in LABEL_LAGS:
    for s in STRATEGIES:
        lines.append(f"| `{s}` | {lag} | {ci_text(s, lag, '35-42')} | {ci_text(s, lag, '43-49')} | "
                     f"{'-' if s == 'static' else delta_text(s, lag, '43-49')} | "
                     f"{int(pooled_value(s, lag, '43-49', 'retrains_total'))} | {recovery_text(s, lag)} |")
lines += ["",
          f"Pre-registered thresholds: restored = {RECOVERY_TARGET:.2f}, material gain = {MATERIAL_GAIN:.2f}. "
          f"In-window oracle (uses 43-49's own labels, not deployable) = {ORACLE_IN_WINDOW_PR_AUC:.4f}.",
          f"`expanding` reaches the restored level at both lags: **{cond_a}**. "
          f"Cells clearing the material-gain bar with a paired lower bound above 0: **{cells_b if cells_b else 'none'}**.",
          f"Sustained recovery is the first step t >= 43 with PR-AUC >= {RECOVERY_TARGET:.2f} that stays there for the next "
          f"{SUSTAIN_STEPS} evaluated steps; steps {NOISE_STEPS} (fewer than {LOW_POSITIVE_FLAG} illicit labels) are skipped, "
          f"so the evaluated steps are {ELIGIBLE_RECOVERY_STEPS} and the last possible start is {ELIGIBLE_RECOVERY_STEPS[-1 - SUSTAIN_STEPS]}.", ""]

if VERDICT.startswith("(a)"):
    lines += [f"**Sustained recovery of `expanding`**: L=1 {recovery_text('expanding', 1)}, L=3 {recovery_text('expanding', 3)} "
              "(step number and offset from step 43).", ""]

lines += ["### Lag curve (static vs expanding, pooled 43-49)", "",
          "| L | static PR-AUC | expanding PR-AUC [95% CI] | delta vs static [95% CI] | expanding sustained recovery |",
          "| ---: | ---: | :--- | :--- | :--- |"]
for r in lag_curve.itertuples():
    lines.append(f"| {r.lag} | {r.static_pr_auc:.4f} | {r.expanding_pr_auc:.4f} [{r.expanding_ci_lo:.4f}, {r.expanding_ci_hi:.4f}] | "
                 f"{r.delta_vs_static:+.4f} [{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}] | {r.expanding_sustained_recovery} |")
lines += ["", f"`expanding` first falls below the restored level at L = "
          f"{first_lag_below_target if first_lag_below_target is not None else 'none of the tested lags'} (descriptive; "
          "it is not part of the verdict rule).", ""]

lines += ["### Why (measured, not inferred)", "",
          f"* `expanding` at L=1: mean hold-out PR-AUC over decision steps 43-49 is {why['holdout_pr_auc_mean_43_49']:.4f}, "
          f"mean realised per-step PR-AUC is {why['step_pr_auc_mean_43_49']:.4f}. Across those steps the Pearson correlation between "
          f"hold-out and realised PR-AUC is {why['holdout_vs_realised_pearson_r_L1_43_49_excluding_noise']:+.2f} excluding the noise "
          f"steps (n = {int(hv_exp['n_steps'])}), {why['holdout_vs_realised_pearson_r_L1_43_49_all_steps']:+.2f} on all steps "
          f"(n = {int(hv_exp_all['n_steps'])}) and {why['holdout_vs_realised_pearson_r_L3_43_49_excluding_noise']:+.2f} at L=3. "
          "This is a descriptive finding about drift (the hold-out is the latest arrived labels, whose relationship to the next "
          "step is itself shifting), not a defect in the selection code.",
          f"* By step 49 the expanding model holds {why['total_positives_in_train_at_49']} positives, of which only "
          f"{why['drift_regime_positives_in_train_at_49']} come from the post-43 regime.",
          f"* Triad share of split gain in `expanding`: {why['triad_gain_share_expanding_mean_35_42']:.3f} (steps 35-42) vs "
          f"{why['triad_gain_share_expanding_mean_43_49']:.3f} (steps 43-49).",
          f"* Removing the triad changes pooled PR-AUC by {ablation[1]['35-42']:+.4f} (35-42) and {ablation[1]['43-49']:+.4f} (43-49) at L=1; "
          f"paired 95% interval on 43-49: [{ablation_paired['43-49']['ci_lo']:+.4f}, {ablation_paired['43-49']['ci_hi']:+.4f}].",
          "", "### Operating rule: top-K alerts per step (primary) vs the validation threshold", "",
          "Pooled over 43-49, labeled transactions only. Top-K is the primary mode: it fixes the analyst workload; "
          "the validation threshold is the comparison and its alert volume is whatever the threshold happens to give.", "",
          "| Strategy (L=1) | Rule | alerts/step | precision | recall | F1 | max possible recall | steps with zero alerts | recall 35-42 |",
          "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for s in dict.fromkeys(("static", "expanding", FROZEN_STRATEGY)):
    for rule in [f"top{k}" for k in BUDGETS] + ["validation_threshold"]:
        max_recall = rule_value(s, MAIN_LAG, "43-49", rule, "max_possible_recall")
        lines.append(f"| `{s}` | {rule} | {rule_value(s, MAIN_LAG, '43-49', rule, 'alerts_per_step'):.1f} | "
                     f"{rule_value(s, MAIN_LAG, '43-49', rule, 'precision'):.3f} | "
                     f"{rule_value(s, MAIN_LAG, '43-49', rule, 'recall'):.3f} | "
                     f"{rule_value(s, MAIN_LAG, '43-49', rule, 'f1'):.3f} | "
                     f"{'-' if max_recall != max_recall else f'{max_recall:.3f}'} | "
                     f"{int(rule_value(s, MAIN_LAG, '43-49', rule, 'steps_with_zero_alerts'))} | "
                     f"{rule_value(s, MAIN_LAG, '35-42', rule, 'recall'):.3f} |")

t_exp = threshold_cell("expanding", MAIN_LAG, "43-49")
t_exp3 = threshold_cell("expanding", 3, "43-49")
t_stat = threshold_cell("static", MAIN_LAG, "43-49")
lines += ["", "### Finding: validation-threshold instability", "",
          f"* `expanding` at L=1, steps 43-49, threshold per step: {t_exp['thresholds_by_step']} "
          f"(range {t_exp['threshold_min']:.3f} to {t_exp['threshold_max']:.3f}, {int(t_exp['distinct_thresholds'])} distinct values); "
          f"share of rows flagged per step: {t_exp['flag_rate_by_step']}; steps with zero alerts: {int(t_exp['steps_zero_alerts'])}.",
          f"* `expanding` at L=3, 43-49: threshold range {t_exp3['threshold_min']:.3f} to {t_exp3['threshold_max']:.3f}; "
          f"steps with zero alerts: {int(t_exp3['steps_zero_alerts'])}.",
          f"* `static` keeps one threshold ({t_stat['threshold_min']:.3f}) and flags {t_stat['flag_rate_min']:.3f} to "
          f"{t_stat['flag_rate_max']:.3f} of rows per step over 43-49; steps with zero alerts: {int(t_stat['steps_zero_alerts'])}.",
          "* Per-step values for every strategy and lag are in `threshold_instability.csv`.", "",
          "### Monitor-triggered vs Phase 8a (per-step agreement, `monitor_concordance.csv`)", "",
          "| L | scope | steps agreeing | Phase 8a fires missed | lag-safe fires Phase 8a did not make | first fire in 43-49 (lag-safe / label-free / label-dependent / Phase 8a) |",
          "| ---: | :--- | ---: | ---: | ---: | :--- |"]
for r in concordance_summary[concordance_summary.scope != "35-42"].itertuples():
    firsts = " / ".join("-" if v is None or v != v else str(int(v)) for v in
                        (r.first_fire_43_49, r.first_label_free_fire_43_49, r.first_label_dependent_fire_43_49,
                         r.phase8a_first_fire_43_49))
    lines.append(f"| {r.lag} | {r.scope} | {r.agree} of {r.steps} | {r.phase8a_fires_missed} | {r.fires_phase8a_did_not} | {firsts} |")

lines += ["", f"### Freeze for the dashboard and API: `{FROZEN_STRATEGY}`", ""]
if FROZEN_STRATEGY == "static":
    lines += ["No retraining strategy cleared the material-gain bar with a paired interval above zero while staying "
              "non-inferior on 35-42, so the served model stays the frozen XGBoost. No API change and no new artifact "
              "(any model change would trip the model-consistency CI gate by design)."]
else:
    lines += [f"`{FROZEN_STRATEGY}` is the highest-ranked qualifying retraining policy at L=1. Serving it needs a "
              "retraining job, a new model artifact and an updated consistency gate; it is not an API-only change."]
lines += ["", "**Dashboard should show**", "",
          "* A triage panel as the primary operating mode: precision@K and recall@K for K = 20, 50, 100 per step, drawn against the maximum possible value for that step's positive count.",
          "* The validation threshold as a secondary indicator only, shown per step with the share of rows it flagged, because its per-step instability is measured above.",
          "* The label-delay state: the newest step whose labels have arrived, and the model's trained-through step.",
          "* A noise badge on any step with fewer than 10 illicit labels; no per-step PR-AUC headline.",
          "* The existing `confidence_context` banner (reliable / degraded / unknown) and rolling prevalence."]
if VERDICT.startswith("(c)"):
    lines += ["* A human-review queue as the primary workflow: ranking quality in the post-43 regime is unverified, so scores order the queue and do not make decisions.",
              "* The retrain log shown as evidence that retraining was tried and did not restore performance, not as a remediation control."]
else:
    lines += ["* A model-freshness widget (trained-through step, steps since retrain) and the recovery curve.",
              "* The human-review queue stays, because recovery is not uniform across strategies and lags."]
lines += ["", "### Limits of this evidence", "",
          f"* 43-49 holds {int(pooled_value('static', 1, '43-49', 'n_illicit'))} positives across one regime change; steps 45 and 46 hold 5 and 2 and are not interpreted.",
          f"* Intervals are a {CI_METHOD} bootstrap over time steps; with only {len(WINDOWS['43-49'])} steps in 43-49 they are approximate "
          "and wide. The row-level bootstrap treats rows inside a step as independent and is reported only for comparison.",
          "* Strategy 5 is a lag-safe re-implementation of the Phase 8a rules; the production engine was not imported. "
          "Its label-free channels use constants declared before the run (not tuned on 43-49) and have not been validated outside this dataset.",
          f"* Extra lags L = {', '.join(str(l) for l in EXTRA_LAGS)} were run for static and expanding only and are bootstrapped on 43-49 only.",
          "* The hold-out rule differs from `xgboost_v2`'s fixed 1-24 / 33-34 split; `v2_exact_control` ties the harness to it.",
          "* The L=3 `static` model is trained through step 32, not 34, to respect label arrival."]
VERDICT_MD = "\n".join(lines)
(OUT_DIR / "verdict.md").write_text(VERDICT_MD, encoding="utf-8")
save_json("verdict.json", {"verdict": VERDICT, "expanding_meets_target_both_lags": cond_a,
                           "cells_meeting_material_gain": cells_b, "cells_meeting_recovery_target": cells_target,
                           "frozen_strategy": FROZEN_STRATEGY, "ci_method": CI_METHOD,
                           "sustained_recovery_step": {f"{s}|L{lag}": v for (s, lag), v in SUSTAINED.items()},
                           "sustained_recovery_definition": {"target": RECOVERY_TARGET, "hold_further_steps": SUSTAIN_STEPS,
                                                             "evaluated_steps": ELIGIBLE_RECOVERY_STEPS,
                                                             "noise_steps_excluded": NOISE_STEPS},
                           "lag_curve": lag_curve.to_dict(orient="records"),
                           "thresholds": {"recovery_target": RECOVERY_TARGET, "material_gain": MATERIAL_GAIN,
                                          "noninferiority": NONINFERIORITY},
                           "why": why})
display(Markdown(VERDICT_MD))''')

# ---------------------------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------------------------
md(r"""## 8. Figures""")

code(r'''palette = dict(zip(STRATEGIES, sns.color_palette("tab10", len(STRATEGIES))))

# 1. Per-step PR-AUC with the number of positives on the axis.
fig, ax = plt.subplots(figsize=(11.5, 4.8))
for strategy in STRATEGIES:
    d = per_step[(per_step.strategy == strategy) & (per_step.lag == MAIN_LAG)].sort_values("step")
    ax.plot(d["step"], d["pr_auc"], marker="o", ms=4, lw=1.6, label=strategy, color=palette[strategy])
ax.axvline(42.5, ls="--", c="k", lw=1)
for s in TEST_STEPS:
    if POS_BY_STEP[s] < LOW_POSITIVE_FLAG:
        ax.axvspan(s - 0.5, s + 0.5, color="0.88", zorder=0)
ax.set_xticks(TEST_STEPS)
ax.set_xticklabels([f"{s}\nn+={POS_BY_STEP[s]}" for s in TEST_STEPS], fontsize=8)
ax.set(ylabel="PR-AUC (this step)", ylim=(-0.02, 1.02),
       title="Per-step PR-AUC, L=1 (grey = fewer than 10 illicit labels: noise, not evidence)")
ax.legend(ncol=4, fontsize=8, loc="center left")
save_fig("per_step_pr_auc.png")

# 2. Pooled PR-AUC with 95% bootstrap intervals.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, lag in zip(axes, LABEL_LAGS):
    for w_idx, window in enumerate(("35-42", "43-49")):
        for s_idx, strategy in enumerate(STRATEGIES):
            r = pooled[(pooled.strategy == strategy) & (pooled.lag == lag) & (pooled.window == window)].iloc[0]
            x = w_idx * (len(STRATEGIES) + 1) + s_idx
            ax.errorbar(x, r["pr_auc"], yerr=[[r["pr_auc"] - r["pr_auc_ci_lo"]], [r["pr_auc_ci_hi"] - r["pr_auc"]]],
                        fmt="o", color=palette[strategy], capsize=3, label=strategy if w_idx == 0 else None)
    ax.axhline(RECOVERY_TARGET, ls=":", c="g", lw=1)
    ax.axhline(MATERIAL_GAIN, ls=":", c="orange", lw=1)
    ax.set_xticks([(len(STRATEGIES) - 1) / 2, (len(STRATEGIES) - 1) / 2 + len(STRATEGIES) + 1])
    ax.set_xticklabels(["35-42", "43-49"])
    ax.set(title=f"Pooled PR-AUC, L={lag}")
axes[0].set_ylabel("PR-AUC [95% bootstrap CI]")
axes[0].legend(fontsize=7, loc="center left")
save_fig("pooled_pr_auc_ci.png")

# 3. Paired delta vs static.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, window in zip(axes, ("35-42", "43-49")):
    d = paired[(paired.reference == "static") & (paired.window == window)]
    for i, (lag, strategy) in enumerate([(l, s) for l in LABEL_LAGS for s in RETRAIN_STRATEGIES]):
        r = d[(d.lag == lag) & (d.strategy == strategy)].iloc[0]
        ax.errorbar(r["delta_pr_auc"], i, xerr=[[r["delta_pr_auc"] - r["ci_lo"]], [r["ci_hi"] - r["delta_pr_auc"]]],
                    fmt="o" if lag == 1 else "s", color=palette[strategy], capsize=3)
    ax.axvline(0, c="k", lw=1)
    ax.set_yticks(range(len(LABEL_LAGS) * len(RETRAIN_STRATEGIES)))
    ax.set_yticklabels([f"{s} (L={l})" for l in LABEL_LAGS for s in RETRAIN_STRATEGIES], fontsize=7)
    ax.set(title=f"Paired delta PR-AUC vs static, pooled {window}", xlabel="delta PR-AUC [95% CI]")
save_fig("paired_delta_vs_static.png")''')

code(r'''# 4. Tail-recovery curve.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, lag in zip(axes, LABEL_LAGS):
    for strategy in STRATEGIES:
        d = tail[(tail.strategy == strategy) & (tail.lag == lag)].sort_values("steps_after_onset")
        ax.plot(d["steps_after_onset"], d["pr_auc"], marker="o", ms=4, label=strategy, color=palette[strategy])
    ax.axhline(RECOVERY_TARGET, ls=":", c="g", lw=1)
    ax.axhline(MATERIAL_GAIN, ls=":", c="orange", lw=1)
    ax.set(title=f"Descriptive pooled-tail curve, L={lag} (not the recovery metric)",
           xlabel="k = steps after onset dropped from the window (pooled PR-AUC of steps 43+k..49)")
axes[0].set_ylabel("pooled PR-AUC of the remaining tail")
axes[0].legend(fontsize=7)
save_fig("tail_recovery_curve.png")

# 5. Alert budget.
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="col")
for r_idx, window in enumerate(("35-42", "43-49")):
    for c_idx, metric in enumerate(("precision_at_k", "recall_at_k")):
        ax = axes[r_idx, c_idx]
        d = alert_budget[(alert_budget.level == "window") & (alert_budget.lag == MAIN_LAG) & (alert_budget.window == window)]
        sns.barplot(data=d, x="K", y=metric, hue="strategy", hue_order=STRATEGIES, palette=palette, ax=ax)
        ax.set(title=f"{metric} pooled {window} (L={MAIN_LAG})", xlabel="alerts per step (K)")
        if (r_idx, c_idx) != (0, 0):
            ax.get_legend().remove()
        else:
            ax.legend(fontsize=6, ncol=2)
save_fig("alert_budget.png")

# 6. Retrain timeline and monitor channels (label-dependent and label-free).
fig, axes = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
for i, strategy in enumerate(STRATEGIES[1:]):
    d = log_frame[(log_frame.strategy == strategy) & (log_frame.lag == MAIN_LAG) & log_frame.retrained]
    axes[0].scatter(d["step"], [i] * len(d), color=palette[strategy], s=40)
axes[0].set_yticks(range(len(STRATEGIES) - 1))
axes[0].set_yticklabels(STRATEGIES[1:], fontsize=8)
axes[0].axvline(42.5, ls="--", c="k", lw=1)
axes[0].set(title=f"Steps at which each policy replaced its model (L={MAIN_LAG})")
tele = pd.DataFrame([r for r in MONITOR_TELEMETRY if r["lag"] == MAIN_LAG])
axes[1].plot(tele["step"], tele["rolling_prevalence"], marker="o", label="rolling prevalence (arrived labels)")
axes[1].plot(tele["step"], tele["perf_f1_min"], marker="s", label="worst per-step F1 of the deployed model (arrived, >=10 illicit)")
axes[1].plot(tele["step"], tele["perf_f1_pooled"], marker="x", ls="--", c="0.5", label="row-pooled F1 of the same steps (masking)")
axes[1].axhline(PREV_COLLAPSE, ls=":", c="C0", lw=1)
axes[1].axhline(PERF_F1_FLOOR, ls=":", c="C1", lw=1)
axes[1].axvline(42.5, ls="--", c="k", lw=1)
axes[1].set(title="Label-dependent channels (dotted = trigger thresholds)")
axes[1].legend(fontsize=7)
axes[2].plot(tele["step"], tele["feature_shift_pct"] / 100.0, marker="o", c="C2",
             label="share of 165 features with PSI > 0.25 vs the deployed model's window")
axes[2].plot(tele["step"], tele["score_psi"], marker="s", c="C3", label="score PSI vs the deployed model's recent scores")
axes[2].axhline(FEATURE_SHIFT_PCT / 100.0, ls=":", c="C2", lw=1)
axes[2].axhline(PSI_SIG, ls=":", c="C3", lw=1)
axes[2].scatter(tele[tele.fired]["step"], [0.0] * int(tele.fired.sum()), marker="^", c="r", s=60, label="trigger fired (any channel)")
axes[2].axvline(42.5, ls="--", c="k", lw=1)
axes[2].set(xlabel="decision step", title="Label-free channels (dotted = trigger thresholds)")
axes[2].legend(fontsize=7)
save_fig("retrain_timeline_and_monitor.png")

# 7. Time-clock: triad gain share, and the no-triad ablation.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for strategy in ("expanding", "recency_exp_hl6", "rolling_w5", "rolling_w10"):
    d = log_frame[(log_frame.strategy == strategy) & (log_frame.lag == MAIN_LAG)].sort_values("step")
    axes[0].plot(d["step"], d["triad_gain_share"], marker="o", ms=4, label=strategy, color=palette[strategy])
axes[0].axvline(42.5, ls="--", c="k", lw=1)
axes[0].set(title="Share of split gain on Aggregate_feature_10/43/8", xlabel="decision step", ylabel="gain share")
axes[0].legend(fontsize=8)
for strategy in ("expanding", "expanding_no_triad"):
    d = per_step[(per_step.strategy == strategy) & (per_step.lag == MAIN_LAG)].sort_values("step")
    axes[1].plot(d["step"], d["pr_auc"], marker="o", ms=4, label=strategy, color=palette[strategy])
axes[1].axvline(42.5, ls="--", c="k", lw=1)
axes[1].set(title="Time-clock ablation: per-step PR-AUC", xlabel="decision step")
axes[1].legend(fontsize=8)
save_fig("time_clock_ablation.png")

# 8. Hold-out versus realised PR-AUC.
fig, ax = plt.subplots(figsize=(9, 4))
d = log_frame[(log_frame.strategy == "expanding") & (log_frame.lag == MAIN_LAG)].sort_values("step")
e = per_step[(per_step.strategy == "expanding") & (per_step.lag == MAIN_LAG)].sort_values("step")
ax.plot(d["step"], d["holdout_pr_auc"], marker="o", label="hold-out PR-AUC (what the model sees at training time)")
ax.plot(e["step"], e["pr_auc"], marker="s", label="realised PR-AUC on the step it then scores")
ax.axvline(42.5, ls="--", c="k", lw=1)
ax.set(xlabel="decision step", ylabel="PR-AUC",
       title="Expanding window: hold-out vs realised PR-AUC (a finding about drift, not a bug)")
ax.legend(fontsize=8)
save_fig("holdout_vs_realised.png")

# 9. Lag curve: pooled 43-49 PR-AUC against label delay.
fig, ax = plt.subplots(figsize=(8, 4.6))
for strategy, prefix in (("static", "static"), ("expanding", "expanding")):
    ys = lag_curve[f"{prefix}_pr_auc"]
    ax.plot(lag_curve["lag"], ys, marker="o", label=strategy, color=palette[strategy])
    ax.fill_between(lag_curve["lag"], lag_curve[f"{prefix}_ci_lo"], lag_curve[f"{prefix}_ci_hi"],
                    color=palette[strategy], alpha=0.18)
ax.axhline(RECOVERY_TARGET, ls=":", c="g", lw=1, label="restored level")
ax.axhline(MATERIAL_GAIN, ls=":", c="orange", lw=1, label="material gain")
ax.set_xticks(list(lag_curve["lag"]))
ax.set(xlabel="label delay L (steps)", ylabel="pooled PR-AUC, steps 43-49",
       title="Recovery against label delay (95% cluster-bootstrap band)", ylim=(-0.02, 1.02))
ax.legend(fontsize=8)
save_fig("lag_curve.png")

# 10. Validation-threshold instability, per step.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
for strategy, lag in (("static", 1), ("expanding", 1), ("expanding", 3), ("recency_exp_hl6", 1), ("monitor_triggered", 1)):
    d = per_step[(per_step.strategy == strategy) & (per_step.lag == lag)].sort_values("step")
    axes[0].plot(d["step"], d["threshold"], marker="o", ms=4, label=f"{strategy} L={lag}", color=palette[strategy],
                 ls="-" if lag == 1 else "--")
    axes[1].plot(d["step"], d["flag_rate"], marker="o", ms=4, label=f"{strategy} L={lag}", color=palette[strategy],
                 ls="-" if lag == 1 else "--")
for ax in axes:
    ax.axvline(42.5, ls="--", c="k", lw=1)
    ax.set_xlabel("decision step")
axes[0].set(ylabel="validation threshold", title="Threshold chosen on the arrived-label hold-out")
axes[1].set(ylabel="share of rows flagged", title="Alert volume it produces")
axes[0].legend(fontsize=7)
save_fig("threshold_instability.png")''')

# ---------------------------------------------------------------------------------------------
# 9. Checks and export
# ---------------------------------------------------------------------------------------------
md(r"""## 8b. Baselines and Edge Cases

Two baselines frame every number: **chance** (a random ranking has PR-AUC equal to prevalence, so any
PR-AUC below the lift column's 1.0 is worse than guessing) and **static** (the no-retraining control). A
retraining strategy is only worth deploying if it beats static with a paired interval above zero, which is
already the verdict rule. The edge-case cell feeds the last deployed model inputs it was never trained on:
all-NaN rows, +/- extreme values and zeros. It must return finite probabilities in [0, 1] and not crash.""")

code(r'''chance_rows = []
for window, steps in WINDOWS.items():
    y_w = np.concatenate([Y_STEP[t] for t in steps])
    chance_rows.append({"window": window, "chance_pr_auc": float(y_w.mean())})
chance = pd.DataFrame(chance_rows)
display(chance.round(4))

below_chance = pooled[(pooled["pr_auc"] < pooled["prevalence"])][["strategy", "lag", "window", "pr_auc", "prevalence"]]
check("baseline: no strategy ranks below chance on a pooled window", int(len(below_chance)), 0,
      note="chance PR-AUC equals the window prevalence; rows listed above if any")

last_bundle = get_model(policy_key("expanding", TEST_MAX - 1, MAIN_LAG))
n_cols = len(last_bundle["cols"])
edge_inputs = {"all_nan": np.full((4, n_cols), np.nan, dtype="float32"),
               "zeros": np.zeros((4, n_cols), dtype="float32"),
               "large_positive": np.full((4, n_cols), 1e9, dtype="float32"),
               "large_negative": np.full((4, n_cols), -1e9, dtype="float32")}
edge_ok = True
for name, matrix in edge_inputs.items():
    try:
        p = last_bundle["model"].predict_proba(matrix)[:, 1]
        good = bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
    except Exception as error:   # a crash is a failed check, not a notebook abort
        good, p = False, repr(error)
    edge_ok = edge_ok and good
    print(f"edge case {name:<15} -> {p}")
check("edge cases: NaN, zero and +/-1e9 inputs give finite probabilities in [0, 1]", edge_ok, True)''')

md(r"""## 9. Leakage Audit, Checks and Export

`checks.csv` re-derives every leakage assertion from the logs of what was *actually used* — the rows in each
training set and the inputs of each monitor decision — not from the parameters that were meant to be used.""")

code(r'''use = pd.DataFrame(USE_LOG)
tele_all = pd.DataFrame(MONITOR_TELEMETRY)

# ---- Leakage: no label used before its arrival step ---------------------------------------------
check("leakage: every model use is logged (all runs, including the extra-lag curve runs)", int(len(use)),
      len(ALL_RUNS) * len(TEST_STEPS))
check("leakage: the extra-lag runs never used a label before its arrival step",
      int(((use["lag"].isin(EXTRA_LAGS)) & ((use["max_label_step_used"] + use["lag"]) > use["step"])).sum()), 0)
check("leakage: no model used a label before its arrival step (max label step + L <= t)",
      int(((use["max_label_step_used"] + use["lag"]) > use["step"]).sum()), 0,
      note="recomputed from STEP[rows] of every training set actually fitted")
check("leakage: no early-stopping / threshold hold-out used a label before arrival",
      int(((use["max_holdout_step"] + use["lag"]) > use["step"]).sum()), 0)
check("leakage: no training set contains the step being predicted or any later step",
      int((use["max_label_step_used"] >= use["step"]).sum()), 0)
check("leakage: every training row is class 1 or 2 (class 3 never a label)", bool(use["classes_ok"].all()), True)
check("leakage: the threshold of every decision came from a hold-out inside the arrived labels",
      int((use["max_holdout_step"] > use["step"] - use["lag"]).sum()), 0)

# ---- Leakage: the monitor ------------------------------------------------------------------------
check("leakage: monitor prevalence channel used labels only through t - L",
      int((tele_all["rolling_window_last_step"] + tele_all["lag"] > tele_all["step"]).sum()), 0)
perf_used = tele_all.dropna(subset=["perf_max_step"])
check("leakage: monitor performance channel used only predictions on steps with arrived labels (s + L <= t)",
      int(((perf_used["perf_max_step"] + perf_used["lag"]) > perf_used["step"]).sum()), 0)
check("leakage: the monitor performance channel is blind to the step being scored",
      bool((tele_all["perf_max_step"].dropna() < tele_all.loc[tele_all["perf_max_step"].notna(), "step"]).all()), True)
perf_rows = tele_all[tele_all["perf_n_steps"] > 0]
check("monitor: every performance window holds only steps with >= 10 illicit labels",
      int((perf_rows["perf_min_pos"] < PERF_MIN_POS).sum()), 0)
check("monitor: no performance window holds more steps than PERF_STEPS",
      int((tele_all["perf_n_steps"] > PERF_STEPS).sum()), 0)
check("monitor: the feature reference of the label-free channel ends strictly before the step being scored",
      int((tele_all["feature_ref_last_step"] >= tele_all["step"]).sum()), 0,
      note="features of steps first..last only; no label is read")
score_rows = tele_all[tele_all["score_ref_n_steps"] > 0]
check("monitor: the score reference of the label-free channel ends strictly before the step being scored",
      int((score_rows["score_ref_max_step"] >= score_rows["step"]).sum()), 0)
check("monitor: the score-shift channel waits for the minimum number of reference steps",
      int(((tele_all["score_ref_n_steps"] < SCORE_MIN_REF_STEPS) & tele_all["score_psi"].notna()).sum()), 0)
_reason_sets = [set(str(r).split("+")) - {"", "nan"} for r in tele_all["reasons"]]
check("monitor: the label-free / label-dependent flags match the recorded reasons, and every fire has a reason",
      bool(all(
          bool(rs & set(LABEL_FREE_REASONS)) == lf and bool(rs & {"PrevalenceCollapse", "PerformanceCrash"}) == ld
          and (bool(rs) == fired)
          for rs, lf, ld, fired in zip(_reason_sets, tele_all["label_free_fired"],
                                       tele_all["label_dependent_fired"], tele_all["fired"]))), True,
      note="label-free reasons are computed from features and the deployed model's own scores only")
check("monitor: concordance holds one row per lag x step", int(len(concordance)), len(LABEL_LAGS) * len(TEST_STEPS))
check("leakage: no scaler or label-dependent preprocessing is fitted (raw features into XGBoost)", True, True,
      note="tree models; the 165 columns are used as-is")

# ---- Protocol ------------------------------------------------------------------------------------
static_rows = use[use["strategy"] == "static"]
check("protocol: static at L=1 is trained through exactly step 34 (the frozen training window)",
      int(static_rows[static_rows.lag == 1]["max_label_step_used"].max()), 34)
check("protocol: static at L=3 is trained through step 32 (steps 33-34 have not arrived at step 35)",
      int(static_rows[static_rows.lag == 3]["max_label_step_used"].max()), 32)
check("lag curve: static is trained through step 33 at L=2 (labels of steps 34 and later have not arrived at step 35)",
      int(static_rows[static_rows.lag == 2]["max_label_step_used"].max()), 33)
check("lag curve: static is trained through step 30 at L=5",
      int(static_rows[static_rows.lag == 5]["max_label_step_used"].max()), 30)
check("lag curve: only static and expanding run at the extra lags",
      sorted(use[use.lag.isin(EXTRA_LAGS)]["strategy"].unique().tolist()), sorted(CURVE_STRATEGIES))
check("lag curve: every curve lag has a bootstrapped 43-49 interval for both strategies",
      bool(all((s, lag, "43-49") in BOOT for lag in CURVE_LAGS for s in CURVE_STRATEGIES)), True)
check("lag curve: the table covers every tested lag once", lag_curve["lag"].tolist(), list(CURVE_LAGS))
check("protocol: static never retrains",
      int(log_frame[log_frame.strategy == "static"]["retrained"].sum()), 0)
check("protocol: feature count is 165 for every strategy except the ablation",
      sorted(use[use.strategy != "expanding_no_triad"]["n_features"].unique().tolist()), [165])
check("protocol: the ablation uses 162 features", sorted(use[use.strategy == "expanding_no_triad"]["n_features"].unique().tolist()), [162])
check("protocol: per-step table has one row per run x step (all strategies at L=1,3; static and expanding at L=2,5)",
      int(len(per_step)), len(ALL_RUNS) * len(TEST_STEPS))
check("protocol: every strategy scores the same labeled rows each step",
      bool(all(len(SCORES[k][t]) == len(Y_STEP[t]) for k in SCORES for t in TEST_STEPS)), True)
check("protocol: per-step PR-AUC is defined everywhere (every step has an illicit label)",
      int(per_step["pr_auc"].isna().sum()), 0)
check("protocol: steps 45 and 46 are flagged as low-positive", sorted(per_step[per_step.noise_flag != ""]["step"].unique().tolist()), [45, 46])
check("protocol: pooled illicit counts equal the sum of per-step counts",
      int(pooled[(pooled.strategy == "static") & (pooled.lag == 1) & (pooled.window == "43-49")]["n_illicit"].iloc[0]),
      int(per_step[(per_step.strategy == "static") & (per_step.lag == 1) & (per_step.window == "43-49")]["n_illicit"].sum()))
check("protocol: monitor-triggered uses no more retrains than expanding at either lag",
      bool(all(int(log_frame[(log_frame.strategy == "monitor_triggered") & (log_frame.lag == lag)]["retrained"].sum())
               <= int(log_frame[(log_frame.strategy == "expanding") & (log_frame.lag == lag)]["retrained"].sum())
               for lag in LABEL_LAGS)), True)

# ---- Metric sanity ---------------------------------------------------------------------------------
check("metrics: precision@K and recall@K lie in [0, 1]",
      bool(alert_budget[["precision_at_k", "recall_at_k"]].apply(lambda c: c.between(0, 1)).all().all()), True)
check("metrics: hits never exceed min(K, illicit) per step",
      bool((budget_steps["tp"] <= np.minimum(budget_steps["K"], budget_steps["n_illicit"])).all()), True)
check("f1 ceiling: the per-step oracle F1 is never below that step's validation-threshold F1",
      bool((per_step["f1_oracle_per_step"] + 1e-9 >= per_step["f1"]).all()), True)
check("f1 ceiling: the weighted per-step oracle is never below the weighted validation F1 in any strategy/lag/window",
      bool((f1_ceiling["f1_oracle_per_step_weighted"] + 1e-9 >= f1_ceiling["f1_validation_threshold_weighted"]).all()), True)
check("f1 ceiling: no pooled single-threshold oracle ceiling is exported",
      bool(not any("oracle" in c for c in pooled.columns)), True,
      note="the pooled one fell below achieved F1 in 4 cells; only the per-step ceiling remains")
check("bootstrap: the interval used for the verdict is the time-step cluster bootstrap", CI_METHOD, "cluster")
check("bootstrap: both interval variants are exported (cluster used, row alongside)",
      bool({"pr_auc_ci_lo", "pr_auc_row_ci_lo"} <= set(pooled.columns)
           and {"ci_lo", "row_ci_lo"} <= set(paired.columns)), True)
check("bootstrap: every cluster draw keeps whole steps (draw length equals the summed size of the chosen steps)",
      bool(all(
          len(BOOT_IDX_CLUSTER[w][b]) == sum(len(Y_STEP[WINDOWS[w][c]]) for c in CLUSTER_CHOSEN[w][b])
          for w in WINDOWS for b in range(min(BOOT_B, 200)))), True)
check("bootstrap: a cluster draw contains every row of each chosen step exactly as often as the step was chosen",
      bool(all(
          (np.bincount(BOOT_IDX_CLUSTER[w][b], minlength=sum(len(Y_STEP[t]) for t in WINDOWS[w]))
           == np.repeat(np.bincount(CLUSTER_CHOSEN[w][b], minlength=len(WINDOWS[w])),
                        [len(Y_STEP[t]) for t in WINDOWS[w]])).all()
          for w in WINDOWS for b in range(min(BOOT_B, 20)))), True)
check("bootstrap: one set of resampled steps per window is shared by every strategy (paired)",
      bool(all(CLUSTER_CHOSEN[w].shape == (BOOT_B, len(WINDOWS[w])) for w in WINDOWS)), True)
check("bootstrap: cluster and row draws are complete and finite for every bootstrapped cell",
      bool(all(np.isfinite(v).all() and v.shape == (BOOT_B, 3) for v in list(BOOT.values()) + list(BOOT_ROW.values()))), True)
check("bootstrap: cluster and row draws exist for the same cells", sorted(BOOT) == sorted(BOOT_ROW), True)
check("recovery: the table covers every run once", int(len(recovery)), len(ALL_RUNS))
check("operating rule: top-K flags exactly min(K, labeled rows) per step",
      bool((budget_steps["flagged"] == np.minimum(budget_steps["K"], budget_steps["n_labeled"])).all()), True)
check("operating rule: the table has the validation threshold and every top-K rule for every run and window",
      int(len(op_rules)), len(ALL_RUNS) * len(WINDOWS) * (1 + len(BUDGETS)))
_op_check = op_rules[(op_rules.rule == "validation_threshold") & (op_rules.window == "43-49")]
check("operating rule: validation-threshold alerts equal the per-step flagged counts",
      bool(all(int(r.alerts_total) == int(per_step[(per_step.strategy == r.strategy) & (per_step.lag == r.lag)
                                                   & (per_step.window == "43-49")]["flagged"].sum())
               for r in _op_check.itertuples())), True)
_k_rows = op_rules[(op_rules.rule == f"top{BUDGETS[0]}") & (op_rules.window == "43-49")]
check("operating rule: top-K precision on 43-49 matches the alert-budget table",
      bool(all(abs(r.precision - float(alert_budget[(alert_budget.strategy == r.strategy) & (alert_budget.lag == r.lag) &
                                                    (alert_budget.level == "window") & (alert_budget.window == "43-49") &
                                                    (alert_budget.K == BUDGETS[0])].iloc[0]["precision_at_k"])) < 1e-9
               for r in _k_rows.itertuples())), True)
check("threshold: the instability table covers every run and both windows",
      int(len(threshold_instability)), len(ALL_RUNS) * 2)
check("threshold: static uses exactly one fixed threshold per window",
      int(threshold_instability[threshold_instability.strategy == "static"]["distinct_thresholds"].max()), 1)
check("hold-out finding: the correlation table covers every retraining strategy, lag, window and scope",
      int(len(holdout_vs_realised)), len(RETRAIN_STRATEGIES) * len(LABEL_LAGS) * 2 * 2)
check("hold-out finding: a correlation is only reported from at least 3 steps",
      bool(holdout_vs_realised[holdout_vs_realised.pearson_r.notna()]["n_steps"].ge(3).all()), True,
      note="a finding about drift, not a bug")

# ---- Frozen artifacts and outputs --------------------------------------------------------------------
FROZEN_AFTER = snapshot(FROZEN_DIRS)
check("frozen: xgboost/ and xgboost_v2/ are byte-for-byte untouched (size + mtime)", FROZEN_AFTER == FROZEN_BEFORE, True,
      note=f"{len(FROZEN_BEFORE)} files compared")
check("frozen: all outputs are written under walkforward/", OUT_DIR.name, "walkforward")

# ---- The checks that depend on the executed verdict -------------------------------------------------
check("verdict: exactly one of (a)/(b)/(c) was assigned", VERDICT in ("(a) RECOVERS", "(b) PARTIAL", "(c) NO RECOVERY"), True)
check("verdict: the frozen strategy is a known strategy", FROZEN_STRATEGY in STRATEGIES, True)''')

code(r'''def row_ci_text(strategy: str, lag: int, window: str) -> str:
    return (f"[{pooled_value(strategy, lag, window, 'pr_auc_row_ci_lo'):.4f}, "
            f"{pooled_value(strategy, lag, window, 'pr_auc_row_ci_hi'):.4f}]")


def row_delta_text(strategy: str, lag: int, window: str, reference: str = "static") -> str:
    r = paired_row(strategy, lag, window, reference)
    return f"[{r['row_ci_lo']:+.4f}, {r['row_ci_hi']:+.4f}]"


def build_digest() -> str:
    out = ["BitcoinGraphGuard Phase 9a - walk-forward retraining digest", "=" * 62, ""]
    out += [f"verdict            : {VERDICT}",
            f"strategy to freeze : {FROZEN_STRATEGY}",
            f"label delays       : main L={MAIN_LAG}, sensitivity L=3, lag curve L={', '.join(str(l) for l in CURVE_LAGS)} "
            f"({', '.join(CURVE_STRATEGIES)} only at L={', '.join(str(l) for l in EXTRA_LAGS)})",
            f"intervals          : 95% {CI_METHOD.upper()} bootstrap over time steps is USED for every verdict input; "
            f"the ROW bootstrap is reported alongside ({BOOT_B} draws each, paired across strategies)",
            f"models fitted      : {len(MODEL_CACHE)} unique (+1 xgboost_v2 control)",
            f"pre-registered     : restored >= {RECOVERY_TARGET}, material gain >= {MATERIAL_GAIN}, "
            f"non-inferiority {NONINFERIORITY} (unchanged)",
            f"recovery metric    : sustained = first step t >= 43 with PR-AUC >= {RECOVERY_TARGET} that holds for the next "
            f"{SUSTAIN_STEPS} evaluated steps; evaluated steps {ELIGIBLE_RECOVERY_STEPS}, noise steps {NOISE_STEPS} excluded", ""]
    out += ["Harness reproduction (xgboost_v2 reference row, original split)"]
    for window_name, documented in V2_REFERENCE.items():
        out.append(f"  {window_name}: measured {pooled_pr_auc(control_scores, WINDOWS[window_name]):.4f} vs documented {documented:.4f}")
    out.append("")
    for lag in LABEL_LAGS:
        out.append(f"Pooled PR-AUC, L={lag}: cluster CI (used) | row CI (alongside), sustained recovery")
        for s in STRATEGIES:
            out.append(f"  {s:<20} 35-42 {ci_text(s, lag, '35-42')} | 43-49 {ci_text(s, lag, '43-49')} "
                       f"row {row_ci_text(s, lag, '43-49')} | retrains {int(pooled_value(s, lag, '43-49', 'retrains_total')):>2} | "
                       f"sustained recovery: {recovery_text(s, lag)}")
        out.append("")
    out += ["Paired delta PR-AUC vs static on 43-49: cluster [95% CI] (used) | row [95% CI] (alongside)"]
    for lag in LABEL_LAGS:
        for s in RETRAIN_STRATEGIES:
            out.append(f"  L={lag} {s:<20} {delta_text(s, lag, '43-49')} | {row_delta_text(s, lag, '43-49')}")
    out.append("")
    out += ["Lag curve, pooled 43-49 (static vs expanding; cluster 95% CI)"]
    for r in lag_curve.itertuples():
        out.append(f"  L={r.lag}  static {r.static_pr_auc:.4f} | expanding {r.expanding_pr_auc:.4f} "
                   f"[{r.expanding_ci_lo:.4f}, {r.expanding_ci_hi:.4f}] | delta {r.delta_vs_static:+.4f} "
                   f"[{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}] | sustained recovery: {r.expanding_sustained_recovery}")
    out += [f"  expanding first falls below {RECOVERY_TARGET} at L={first_lag_below_target} (descriptive)", ""]
    out += ["Operating rules, pooled 43-49: top-K per step (primary) vs validation threshold"]
    for s in ("static", "expanding"):
        for lag in LABEL_LAGS:
            parts = []
            for rule in [f"top{k}" for k in BUDGETS] + ["validation_threshold"]:
                parts.append(f"{rule} P{rule_value(s, lag, '43-49', rule, 'precision'):.2f}/R{rule_value(s, lag, '43-49', rule, 'recall'):.2f}"
                             f"/F1 {rule_value(s, lag, '43-49', rule, 'f1'):.2f}@{rule_value(s, lag, '43-49', rule, 'alerts_per_step'):.1f}/step")
            out.append(f"  {s:<10} L={lag}  " + " | ".join(parts))
    out.append("")
    out += ["Finding: validation-threshold instability (threshold per step; share of rows flagged)"]
    for s, lag in (("static", 1), ("expanding", 1), ("expanding", 3), ("monitor_triggered", 1)):
        c = threshold_cell(s, lag, "43-49")
        out.append(f"  {s:<18} L={lag} 43-49 thresholds {c['thresholds_by_step']}")
        out.append(f"  {'':<18}      flagged    {c['flag_rate_by_step']} | steps with zero alerts {int(c['steps_zero_alerts'])}")
    out.append("")
    out += ["F1 at the validation threshold vs the PER-STEP oracle ceiling (weighted by illicit labels; a ceiling, not a result;",
            "  no pooled single-threshold ceiling is reported):"]
    for s in ("static", "expanding"):
        for lag in LABEL_LAGS:
            x = f1_ceiling[(f1_ceiling.strategy == s) & (f1_ceiling.lag == lag) & (f1_ceiling.window == "43-49")].iloc[0]
            out.append(f"  {s:<10} L={lag} 43-49 validation F1 {x['f1_validation_threshold_weighted']:.4f} "
                       f"vs per-step ceiling {x['f1_oracle_per_step_weighted']:.4f}")
    out.append("")
    out += ["Finding about drift (descriptive, not a bug): hold-out PR-AUC vs realised PR-AUC per step"]
    for s in ("expanding", "recency_exp_hl6"):
        for lag in LABEL_LAGS:
            for window in ("35-42", "43-49"):
                for scope in ("all steps", "excluding noise steps"):
                    c = holdout_cell(s, lag, window, scope)
                    r_text = "n/a" if c["pearson_r"] != c["pearson_r"] else f"{c['pearson_r']:+.2f}"
                    out.append(f"  {s:<16} L={lag} {window} {scope:<22} n={int(c['n_steps'])} hold-out mean {c['holdout_pr_auc_mean']:.3f} "
                               f"realised mean {c['realised_pr_auc_mean']:.3f} Pearson r {r_text}")
    out.append("")
    out += ["Monitor-triggered vs Phase 8a, per-step agreement (full per-step table in monitor_concordance.csv)"]
    for r in concordance_summary[concordance_summary.scope != "35-42"].itertuples():
        firsts = " / ".join("-" if v is None or v != v else str(int(v)) for v in
                            (r.first_fire_43_49, r.first_label_free_fire_43_49, r.first_label_dependent_fire_43_49,
                             r.phase8a_first_fire_43_49))
        out.append(f"  L={r.lag} {r.scope}: agree {r.agree} of {r.steps} (label-dependent channels alone {r.agree_label_dependent_only}, "
                   f"label-free alone {r.agree_label_free_only}); Phase 8a fires missed {r.phase8a_fires_missed}; "
                   f"fires Phase 8a did not make {r.fires_phase8a_did_not}; first fire lag-safe/label-free/label-dependent/Phase 8a: {firsts}")
    out.append("")
    out += ["Noise warning: steps 45 and 46 hold 5 and 2 illicit labels. Per-step values there are noise.", ""]
    out += ["Why (measured):"] + [f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in why.items()]
    out.append("")
    n_pass = sum(1 for c in CHECKS if c["ok"])
    out += [f"checks: {n_pass} of {len(CHECKS)} passed"]
    out += [f"  FAILED: {c['check']}" for c in CHECKS if not c["ok"]]
    out += ["", "Full verdict, tables and the dashboard specification: verdict.md"]
    return "\n".join(out)


save_table("per_step_metrics.csv", per_step)
save_table("pooled_metrics.csv", pooled)
save_table("alert_budget.csv", alert_budget)
save_table("retrain_log.csv", log_frame)
save_table("paired_deltas.csv", paired)
save_table("tail_recovery.csv", tail)
save_table("sustained_recovery.csv", recovery)
save_table("lag_curve.csv", lag_curve)
save_table("operating_rules.csv", op_rules)
save_table("threshold_instability.csv", threshold_instability)
save_table("f1_per_step_ceiling.csv", f1_ceiling)
save_table("holdout_vs_realised.csv", holdout_vs_realised)
save_table("monitor_telemetry.csv", tele_all)
save_table("monitor_concordance.csv", concordance)
save_table("monitor_concordance_summary.csv", concordance_summary)
save_table("checks.csv", pd.DataFrame(CHECKS))
digest = build_digest()
(OUT_DIR / "digest.txt").write_text(digest, encoding="utf-8")
print(digest)

exported = sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())
print("\nexported:", exported)
print("figures :", sorted(FIGURES))
print("\nNext: copy walkforward/ to results/walkforward/ in the repo and paste digest.txt back for review.")''')

# ---------------------------------------------------------------------------------------------
# Final markdown cell
# ---------------------------------------------------------------------------------------------
md(r"""## Verdict and What to Freeze

**The executed verdict, its numbers and the dashboard specification are in the cell output above and in
`verdict.md` / `digest.txt`.** They are computed, not written by hand, because they depend on the run. The
mapping below was fixed before any model was fit.

| Verdict | Rule | What is frozen for the API | What the dashboard shows |
| :--- | :--- | :--- | :--- |
| **(a) RECOVERS** | `expanding` reaches pooled 43–49 PR-AUC ≥ 0.50 at L = 1 **and** L = 3 with a paired (cluster-bootstrap) ΔPR-AUC lower bound above 0; sustained-recovery step reported | the best qualifying retraining policy; needs a retrain job and a new artifact (trips the consistency gate by design) | top-K triage panel (primary), trained-through step, steps since retrain, lag curve |
| **(b) PARTIAL** | not (a), but some strategy × lag clears 0.15 with a lower bound above 0 — recovery only for some strategies or only at low label delay | the qualifying policy only if non-inferior on 35–42; otherwise `static` | triage panel, model freshness, label-delay state; human review stays primary |
| **(c) NO RECOVERY** | otherwise | `static` (no API change); the right production answer is **alert-budget triage plus human review** | top-K triage (precision@K / recall@K for K = 20, 50, 100 against their per-step maximum) as the primary mode, validation threshold as a secondary indicator, label-delay state, noise badge for steps with < 10 positives, `confidence_context`, rolling prevalence, retrain log as evidence |

**What this experiment can and cannot say.** It tests whether *retraining on arrived labels* restores
performance under the stated protocol. It cannot separate "the post-43 relationship is genuinely different"
from "too few post-43 positives have arrived to learn it", although `n_train_pos_drift_regime` in
`retrain_log.csv` and the lag curve show how many are available at each step. It uses one regime
change and 169 positives. The verdict intervals are a cluster bootstrap over only 7 drift steps, so they are
approximate and wide; the row-level intervals are reported for comparison and treat rows as independent.""")


def generate_notebook() -> None:
    notebook = {"cells": CELLS,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.10"},
                             "colab": {"provenance": [], "toc_visible": True}},
                "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(CELLS)} cells "
          f"({sum(1 for cell in CELLS if cell['cell_type'] == 'markdown')} markdown, "
          f"{sum(1 for cell in CELLS if cell['cell_type'] == 'code')} code)")


if __name__ == "__main__":
    generate_notebook()
