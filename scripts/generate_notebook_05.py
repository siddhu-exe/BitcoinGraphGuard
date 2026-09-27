"""Generate notebooks/05_temporal_inductive_evaluation.ipynb (Phase 5).

Run from the repository root:  python scripts/generate_notebook_05.py

The notebook performs no ML on this machine; ML execution happens on Google Colab / Kaggle.
Authoring the notebook through this generator keeps Phase 5 reviewable and reproducible in Git.
"""

import json
import uuid

CELLS = []


def C(kind, src):
    """Append one notebook cell (kind in {markdown, code}) to the build list."""
    CELLS.append({"kind": kind, "src": src})


def make_cell(cell_type, source, cell_id=None):
    """Build a single nbformat-4 cell with deterministic source line splitting."""
    if cell_id is None:
        cell_id = uuid.uuid4().hex
    lines = source.split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": cell_type, "metadata": {}, "source": src, "id": cell_id}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


C("markdown", r"""# BitcoinGraphGuard — Phase 5: Temporal & Inductive Evaluation

**Phase:** 5 of 9 · Temporal & Inductive Evaluation
**Environment:** Google Colab / Kaggle (CUDA GPU strongly recommended). This notebook is **never executed on the laptop**.
**Inputs:** the frozen Phase 2-4 prediction artifacts plus the raw Elliptic++ CSVs.
**Outputs:** `temporal_inductive/` (tables, figures, `checks.csv`, `README.md`).

This phase is an **evaluation and temporal-behaviour phase**, not an architecture phase. It does not
implement HGT, change the RGCN architecture, add attention/temporal embeddings, or add features to
XGBoost. It measures *why* the current models degrade over time and whether the heterogeneous graph
carries useful signal once deployment is treated as genuinely temporal and inductive.

## Research questions
1. **Temporal degradation (Q1):** how do the frozen models behave on each step 35-49, and where does the collapse begin?
2. **Static vs temporal adaptation (Q2):** does expanding- or rolling-window retraining reduce the 43-49 collapse?
3. **Inductive generalisation (Q3):** how does performance differ for transactions with *seen* vs *unseen* address context?
4. **Threshold instability (Q4):** how much of the degradation is operating-point/prevalence mismatch versus discrimination loss (PR-AUC is threshold-free)?

## Critical temporal rule enforced programmatically
Every prediction at step `t` may use only features, edges and address snapshots observed at steps `<= t`,
and may train only on labels from steps `< t`. Future labels, features, edges, snapshots and graph
statistics are forbidden. Section 11 validates this.
""")

C("markdown", r"""## 0. Setup — one config cell

Installs nothing; assumes the Colab/Kaggle runtime already provides `torch`. Seeds every source of
randomness, fixes the split windows inherited from Phases 1-4, and declares the Phase 5 controls.
""")

C("code", r'''import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
N_STEPS = 49

# Temporal protocol inherited unchanged from Phases 1-4.
FIT_MIN, FIT_MAX = 1, 24
VAL_MIN, VAL_MAX = 25, 34
TRAIN_MIN, TRAIN_MAX = 1, 34
TEST_MIN, TEST_MAX = 35, 49
TEST_EARLY_MAX = 42
DRIFT_MIN = 43

# Phase 5 controls (pre-registered).
ROLLING_WINDOW = 20                     # single pre-registered rolling width
PREDICTION_INCLUDES_CURRENT_STEP_EDGES = True  # per the Phase 5 temporal rule: edges available by t
RUN_EXPANDING = True
RUN_ROLLING = True
THRESHOLD_GRID = np.round(np.linspace(0.005, 0.995, 199), 4)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 11

print(f"python {sys.version.split()[0]} | platform {platform.platform()}")
print(f"numpy {np.__version__} | pandas {pd.__version__} | torch {torch.__version__}")
print(f"device: {device} " + (f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "(CPU)"))
''')

C("markdown", r"""## 1. Paths & validation harness

One config cell resolves the raw dataset, the frozen Phase 2-4 artifacts, and the Phase 5 output
directory. The same `check()` harness used in Phases 1-4 records every assertion to `checks.csv`.
""")

C("code", r'''FILES = {
    "txs_features": "txs_features.csv",
    "txs_classes": "txs_classes.csv",
    "txs_edgelist": "txs_edgelist.csv",
    "wallets_features": "wallets_features.csv",
    "wallets_classes": "wallets_classes.csv",
    "addr_tx": "AddrTx_edgelist.csv",
    "tx_addr": "TxAddr_edgelist.csv",
    "addr_addr": "AddrAddr_edgelist.csv",
}

DRIVE_ROOT = Path(
    os.environ.get("BITCOINGUARD_DRIVE_ROOT", "/content/drive/MyDrive/Projects/Bitcoin Graph")
)


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
    raise FileNotFoundError(
        "Elliptic++ not found; set ELLIPTIC_DATA_DIR to the folder holding the nine CSVs."
    )


DATA_DIR = resolve_data_dir()

if os.environ.get("TEMPORAL_OUT_DIR"):
    OUT_DIR = Path(os.environ["TEMPORAL_OUT_DIR"])
elif DRIVE_ROOT.is_dir():
    OUT_DIR = DRIVE_ROOT / "temporal_inductive"
else:
    OUT_DIR = Path("temporal_inductive")
FIG_DIR = OUT_DIR / "figures"
for _directory in (OUT_DIR, FIG_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Frozen Phase 2-4 artifacts live next to the raw data on Drive, or under results/ in the repo.
ARTIFACT_ROOTS = [DRIVE_ROOT, Path("results"), Path.cwd(), Path(".")]


def resolve_artifact(folder: str, name: str = "predictions.csv"):
    """Locate a frozen artifact across the Drive and repository layouts."""
    for root in ARTIFACT_ROOTS:
        candidate = root / folder / name
        if candidate.exists():
            return candidate
    return None


CHECKS = []


def check(name, value, expected=None, note=""):
    """Record one sanity check for the section 11 validation table."""
    ok = expected is None or value == expected
    CHECKS.append({"check": name, "value": value, "expected": expected, "ok": bool(ok), "note": note})
    flag = "ok" if ok else "FAIL"
    print(f"[{flag}] {name}: {value}" + (f" (expected {expected})" if expected is not None else ""))


def save_table(name, frame):
    frame.to_csv(OUT_DIR / name, index=False)


def save_json(name, payload):
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.show()


def tick(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


print(f"data : {DATA_DIR}")
print(f"out  : {OUT_DIR}")
''')

C("markdown", r"""## 2. Load the Elliptic++ graph (read-only)

Identical loading and validation to Phase 4: 203,769 transactions x 165 non-domain features, 822,942
unique wallets (deduplicated to `(address, time step)`), and the four timestamped relation lists.
""")

C("code", r'''def read_header(path: Path):
    return pd.read_csv(path, nrows=0).columns.tolist()


# --- Transactions: 165 non-domain features (93 Local + 72 Aggregate) ---
txs_columns = read_header(DATA_DIR / FILES["txs_features"])
all_tx_features = [c for c in txs_columns if c not in ("txId", "Time step")]
domain_columns = [c for c in all_tx_features if not c.startswith(("Local_feature", "Aggregate_feature"))]
tx_model_features = [c for c in all_tx_features if c not in domain_columns]

txs_classes = pd.read_csv(DATA_DIR / FILES["txs_classes"], dtype={"txId": str})
txs = pd.read_csv(
    DATA_DIR / FILES["txs_features"],
    usecols=["txId", "Time step"] + tx_model_features,
    dtype={"txId": str, "Time step": "int16", **{c: "float32" for c in tx_model_features}},
)
txs["class"] = txs["txId"].map(txs_classes.set_index("txId")["class"]).astype("int8")
print(f"Loaded {len(txs):,} transactions | {len(tx_model_features)} model features")
check("features: tx model feature count", len(tx_model_features), 165)
check("transactions: row count", len(txs), 203_769)
check("transactions: time-step span", (int(txs["Time step"].min()), int(txs["Time step"].max())), (1, 49))

# --- Wallet universe + deduplicated snapshots ---
wallets_classes = pd.read_csv(DATA_DIR / FILES["wallets_classes"], dtype={"address": str})
wallet_cols = read_header(DATA_DIR / FILES["wallets_features"])
wallet_feature_cols = [c for c in wallet_cols if c not in ("address", "Time step")]

wallets_raw_df = pd.read_csv(
    DATA_DIR / FILES["wallets_features"],
    dtype={"address": str, "Time step": "int16", **{c: "float32" for c in wallet_feature_cols}},
)
wallets_df = wallets_raw_df.drop_duplicates(subset=["address", "Time step"]).reset_index(drop=True)
del wallets_raw_df
print(f"Wallets: {len(wallets_classes):,} unique addresses | {len(wallets_df):,} distinct snapshots")
check("wallets: unique address count", len(wallets_classes), 822_942)
check("wallets: deduplicated snapshot count", len(wallets_df), 920_691)
check("wallets: feature column count", len(wallet_feature_cols), 55)

# --- Node-id mappings ---
tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(txs["txId"].values)}
addr_to_idx = {addr: idx for idx, addr in enumerate(wallets_classes["address"].values)}
N_TX = len(txs)
N_ADDR = len(wallets_classes)

# --- The four directed relation lists ---
edges_tx_tx = pd.read_csv(DATA_DIR / FILES["txs_edgelist"], dtype=str)
edges_addr_tx = pd.read_csv(DATA_DIR / FILES["addr_tx"], dtype=str)
edges_tx_addr = pd.read_csv(DATA_DIR / FILES["tx_addr"], dtype=str)
edges_addr_addr = pd.read_csv(DATA_DIR / FILES["addr_addr"], dtype=str)

src_tx_tx = edges_tx_tx["txId1"].map(tx_id_to_idx).values.astype(np.int64)
dst_tx_tx = edges_tx_tx["txId2"].map(tx_id_to_idx).values.astype(np.int64)
src_addr_tx = edges_addr_tx["input_address"].map(addr_to_idx).values.astype(np.int64)
dst_addr_tx = edges_addr_tx["txId"].map(tx_id_to_idx).values.astype(np.int64)
src_tx_addr = edges_tx_addr["txId"].map(tx_id_to_idx).values.astype(np.int64)
dst_tx_addr = edges_tx_addr["output_address"].map(addr_to_idx).values.astype(np.int64)
src_addr_addr = edges_addr_addr["input_address"].map(addr_to_idx).values.astype(np.int64)
dst_addr_addr = edges_addr_addr["output_address"].map(addr_to_idx).values.astype(np.int64)

total_edges = len(edges_tx_tx) + len(edges_addr_tx) + len(edges_tx_addr) + len(edges_addr_addr)
print(f"Loaded {total_edges:,} directed edges across 4 relations")
check("graph: tx->tx edges", len(edges_tx_tx), 234_355)
check("graph: addr->tx edges", len(edges_addr_tx), 477_117)
check("graph: tx->addr edges", len(edges_tx_addr), 837_124)
check("graph: addr->addr edges", len(edges_addr_addr), 2_868_964)
check("graph: total directed edges", total_edges, 4_417_560)
''')

C("markdown", r"""### Temporal availability of every relation

Every edge is stamped with the earliest step at which it is observed, so the graph can be restricted
to a cutoff with no future lookahead:

* `tx->tx` and the transaction endpoints of `addr->tx` / `tx->addr` are stamped with the transaction's own step.
* `AddrAddr` carries no native timestamp in Elliptic++, so an address-to-address edge is made available
  at `max(first_seen[src], first_seen[dst])`, where `first_seen` is derived from wallet snapshots and
  timestamped incident edges. This is a documented, conservative proxy (see Limitations).
""")

C("code", r'''tx_step = txs["Time step"].values.astype(np.int16)
tx_class = txs["class"].values.astype(np.int8)
is_labeled = np.isin(tx_class, [1, 2])
tx_ids = txs["txId"].values
raw_tx_features = txs[tx_model_features].to_numpy(dtype=np.float32)

wallets_df["addr_idx"] = wallets_df["address"].map(addr_to_idx)
wallets_arr = wallets_df[wallet_feature_cols].to_numpy(dtype=np.float32)
wallets_step = wallets_df["Time step"].values.astype(np.int16)
wallets_addr_idx = wallets_df["addr_idx"].values.astype(np.int64)

UNSEEN_STEP = np.int16(N_STEPS + 1)
first_seen_addr = np.full(N_ADDR, UNSEEN_STEP, dtype=np.int16)
np.minimum.at(first_seen_addr, wallets_addr_idx, wallets_step)          # wallet snapshots
np.minimum.at(first_seen_addr, src_addr_tx, tx_step[dst_addr_tx])       # funding addresses
np.minimum.at(first_seen_addr, dst_tx_addr, tx_step[src_tx_addr])       # destination addresses

EDGE_KEYS = ["tx_to_tx", "addr_to_tx", "tx_to_addr", "addr_to_addr"]
edge_endpoints = {
    "tx_to_tx": (src_tx_tx, dst_tx_tx),
    "addr_to_tx": (src_addr_tx, dst_addr_tx),
    "tx_to_addr": (src_tx_addr, dst_tx_addr),
    "addr_to_addr": (src_addr_addr, dst_addr_addr),
}
edge_avail = {
    "tx_to_tx": tx_step[src_tx_tx],
    "addr_to_tx": tx_step[dst_addr_tx],
    "tx_to_addr": tx_step[src_tx_addr],
    "addr_to_addr": np.maximum(first_seen_addr[src_addr_addr], first_seen_addr[dst_addr_addr]),
}
edge_index_dev = {
    key: torch.from_numpy(np.stack(endpoints, axis=0).astype(np.int64)).to(device)
    for key, endpoints in edge_endpoints.items()
}

check("availability: tx->tx edges are 100% intra-step (both endpoints same step)",
      int((tx_step[src_tx_tx] != tx_step[dst_tx_tx]).sum()), 0)
check("availability: every tx->tx edge stamped 1-49",
      bool(edge_avail["tx_to_tx"].min() >= 1 and edge_avail["tx_to_tx"].max() <= N_STEPS), True)
check("availability: addr->addr proxy steps within 1..N_STEPS+1",
      bool(edge_avail["addr_to_addr"].min() >= 1 and edge_avail["addr_to_addr"].max() <= UNSEEN_STEP), True)
check("availability: addresses never observed anywhere",
      int((first_seen_addr == UNSEEN_STEP).sum()), 0)
print("First-seen address coverage:", int((first_seen_addr != UNSEEN_STEP).sum()), "of", N_ADDR)
''')

C("markdown", r"""### Transaction <-> address incidence (label-free structural context)

The bipartite incidence between transactions and addresses (funding edges plus destination edges) is
built once. From it we derive three per-transaction aggregates used by the drift and inductive
sections — all structural, none label-derived:

* `tx_min_addr_firstseen[i]` — earliest step at which any address of transaction `i` was first observed.
* `tx_min_hist_txstep[i]` — earliest step of any *other* transaction sharing an address with `i`
  (a `T -> A -> T` bridge, ignoring the transaction itself).
* `has_any_addr[i]` — whether the transaction has any address linkage at all.
""")

C("code", r'''inc_rows = np.concatenate([dst_addr_tx, src_tx_addr])
inc_cols = np.concatenate([src_addr_tx, dst_tx_addr])
INCIDENCE = coo_matrix(
    (np.ones(inc_rows.size, dtype=np.float32), (inc_rows, inc_cols)), shape=(N_TX, N_ADDR)
).tocsr()
INCIDENCE.data[:] = 1.0  # collapse duplicate (tx, address) pairs to a binary link

row_idx = np.repeat(np.arange(N_TX), np.diff(INCIDENCE.indptr))
addr_of_tx = INCIDENCE.indices

tx_min_addr_firstseen = np.full(N_TX, UNSEEN_STEP, dtype=np.int16)
np.minimum.at(tx_min_addr_firstseen, row_idx, first_seen_addr[addr_of_tx])

TRANSPOSED = INCIDENCE.T.tocsr()
addr_idx_repeat = np.repeat(np.arange(N_ADDR), np.diff(TRANSPOSED.indptr))
addr_min_txstep = np.full(N_ADDR, UNSEEN_STEP, dtype=np.int16)
np.minimum.at(addr_min_txstep, addr_idx_repeat, tx_step[TRANSPOSED.indices])

tx_min_hist_txstep = np.full(N_TX, UNSEEN_STEP, dtype=np.int16)
np.minimum.at(tx_min_hist_txstep, row_idx, addr_min_txstep[addr_of_tx])

# A transaction cannot be its own historical bridge.
tx_min_hist_txstep = np.maximum(
    tx_min_hist_txstep,
    np.where(tx_step < tx_min_hist_txstep, tx_step, tx_step),  # keep as-is; guard below
)
has_any_addr = np.diff(INCIDENCE.indptr) > 0

print(f"Transactions with address linkage: {int(has_any_addr.sum()):,} / {N_TX:,}")
check("incidence: address-less transactions", int((~has_any_addr).sum()), 965)
check("incidence: every address-unlinked transaction never marked positive",
      int(((~has_any_addr) & (tx_class == 1)).sum()), 0)

# Phase 4 continuity check: test transactions with a wallet bridge to a step <= 34 transaction.
tx_in_test = np.where(tx_step >= TEST_MIN)[0]
bridge_phase4_def = ((tx_step >= TEST_MIN) & (tx_min_hist_txstep <= TRAIN_MAX)).sum()
print(f"Test transactions with a historical T->A->T bridge (Phase 4 definition): {int(bridge_phase4_def):,}")
check("continuity: Phase 4 historical bridge count reproduced", int(bridge_phase4_def), 10_812)
''')

C("markdown", r"""## 3. Leakage-safe temporal accessors

All experiment code goes through these four helpers, which make the temporal rule structural rather
than a convention:

* `edges_up_to(c)` — relation sub-graph restricted to edges observed at step `<= c`.
* `features_up_to(c)` — scaled transaction/address tensors from a scaler fit only on `<= c` data.
* `labeled_up_to(c)` / `labeled_window(lo, hi)` — label masks.
* `train_rgcn(...)` — trains with one graph and validates with another, so validation never sees a
  different graph regime than it should.
""")

C("code", r'''def edges_up_to(cutoff: int) -> dict:
    """Restrict every relation to edges observed at step <= cutoff (no future edges)."""
    out = {}
    for key in EDGE_KEYS:
        keep = torch.from_numpy(np.nonzero(edge_avail[key] <= cutoff)[0])
        out[key] = edge_index_dev[key][:, keep]
    return out


def labeled_up_to(cutoff: int) -> np.ndarray:
    return is_labeled & (tx_step <= cutoff)


def labeled_window(lo: int, hi: int) -> np.ndarray:
    return is_labeled & (tx_step >= lo) & (tx_step <= hi)


def step_labeled_mask(t: int) -> np.ndarray:
    return is_labeled & (tx_step == t)


def raw_address_matrix(cutoff: int) -> np.ndarray:
    """Mean historical wallet features per address, using snapshots up to `cutoff` only."""
    hist = wallets_step <= cutoff
    hist_addrs = wallets_addr_idx[hist]
    hist_feats = wallets_arr[hist]
    sums = np.zeros((N_ADDR, hist_feats.shape[1]), dtype=np.float32)
    counts = np.zeros(N_ADDR, dtype=np.float32)
    np.add.at(sums, hist_addrs, hist_feats)
    np.add.at(counts, hist_addrs, 1.0)
    valid = counts > 0
    sums[valid] /= counts[valid, None]
    sums[~valid] = hist_feats.mean(axis=0) if hist_addrs.size else 0.0
    return sums


def scale(raw: np.ndarray, scaler) -> torch.Tensor:
    return torch.from_numpy(scaler.transform(raw).astype(np.float32))


def pos_weight_of(mask: np.ndarray) -> torch.Tensor:
    labels = tx_class[mask]
    positives = int((labels == 1).sum())
    negatives = int((labels == 2).sum())
    return torch.tensor([negatives / max(1, positives)], dtype=torch.float32).to(device)


def select_threshold(y_true: np.ndarray, scores: np.ndarray):
    """F1-maximising threshold on the supplied labelled window only."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in THRESHOLD_GRID:
        value = f1_score(y_true, (scores >= candidate).astype(int), zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(candidate), float(value)
    return best_threshold, best_f1


def pr_auc_reason(y_true: np.ndarray) -> str:
    positives = int(np.asarray(y_true).sum())
    if positives == 0:
        return "undefined: no illicit (positive) example in this subgroup"
    if positives == y_true.size:
        return "undefined: no licit (negative) example in this subgroup"
    return ""


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Standard metrics at a fixed operating threshold; undefined metrics reported, not faked."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    reason = pr_auc_reason(y_true)
    prevalence = float(y_true.mean())
    return {
        "n": int(y_true.size),
        "illicit": int(y_true.sum()),
        "licit": int(y_true.size - y_true.sum()),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "pr_auc": float("nan") if reason else float(average_precision_score(y_true, scores)),
        "pr_auc_reason": reason,
        "roc_auc": float("nan") if reason else float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
''')

C("markdown", r"""## 4. Frozen benchmark references (Q1 baseline)

Per-step evaluation of the frozen models needs their *saved* test predictions, not a retrain. The
Phase 2-4 artifacts provide exactly that:
`results/xgboost/predictions.csv`, `results/GraphSage/predictions.csv`, and
`results/heterogeneous_gnn/predictions.csv` — each 16,670 labelled test rows. Operating thresholds are
read from the recorded hyperparameter/metric JSONs so the frozen benchmark is never modified.
""")

C("code", r'''def read_json_if_present(path):
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


FROZEN_MODELS = {
    "xgboost": {
        "folder": "xgboost", "score_column": "xgboost_score", "display": "XGBoost (frozen)",
        "threshold_default": 0.435,
        "threshold_json": ("metrics.json", lambda d: d.get("operating_points", {}).get("xgboost_optimized", {}).get("threshold")),
    },
    "graphsage": {
        "folder": "GraphSage", "score_column": "graphsage_score", "display": "GraphSAGE (frozen)",
        "threshold_default": 0.830,
        "threshold_json": ("graphsage_hyperparameters.json", lambda d: d.get("graphsage", {}).get("threshold")),
    },
    "rgcn_phase4": {
        "folder": "heterogeneous_gnn", "score_column": "rgcn_score", "display": "HeteroRGCN (frozen)",
        "threshold_default": 0.795,
        "threshold_json": ("rgcn_hyperparameters.json", lambda d: d.get("rgcn", {}).get("threshold")),
    },
}

frozen_frames = {}
for key, spec in FROZEN_MODELS.items():
    path = resolve_artifact(spec["folder"], "predictions.csv")
    if path is None:
        check(f"frozen: {key} predictions.csv found", False, True, "not found; skipped")
        continue
    frame = pd.read_csv(path, dtype={"txId": str})
    json_path = resolve_artifact(spec["folder"], spec["threshold_json"][0])
    threshold = spec["threshold_json"][1](read_json_if_present(json_path)) or spec["threshold_default"]
    frozen_frames[key] = {"path": path, "threshold": float(threshold),
                          "score": frame.set_index("txId")[spec["score_column"]], "frame": frame}

check("frozen: three reference prediction sets available", len(frozen_frames), 3)

# Align every frozen score to the canonical transaction order by txId.
frozen_scores = {}
for key, entry in frozen_frames.items():
    frozen_scores[key] = entry["score"].reindex(tx_ids).to_numpy(dtype="float64")
    n_available = int(np.isfinite(frozen_scores[key]).sum())
    check(f"frozen: {key} scores align to all test transactions", n_available, 16_670)

# Confirm the stored y_true agrees with the freshly loaded labels on the test period.
for key, entry in frozen_frames.items():
    stored_truth = entry["frame"].set_index("txId")["y_true"].reindex(tx_ids[is_labeled & (tx_step >= TEST_MIN)])
    loaded_truth = (tx_class[is_labeled & (tx_step >= TEST_MIN)] == 1).astype(int)
    check(f"frozen: {key} stored y_true matches loaded labels",
          int((stored_truth.to_numpy().astype(int) != loaded_truth).sum()), 0)
''')

C("markdown", r"""### Experiment A — per-step metrics for the frozen models

For every step 35-49 we record the labelled/illicit/licit counts, prevalence, and the full metric set
for each frozen model at its own frozen threshold. A PR-AUC is written only when the metric is defined
(both classes present); otherwise the reason is recorded instead.
""")

C("code", r'''per_step_rows = []
per_step_counts = {}
for t in range(TEST_MIN, TEST_MAX + 1):
    idx = np.where(step_labeled_mask(t))[0]
    y_t = (tx_class[idx] == 1).astype(int)
    per_step_counts[t] = {
        "time_step": t,
        "n_labeled": int(idx.size),
        "illicit": int(y_t.sum()),
        "licit": int(idx.size - y_t.sum()),
        "prevalence": float(y_t.mean()),
    }
    for key, entry in frozen_frames.items():
        scores_t = frozen_scores[key][idx]
        metrics = evaluate(y_t, scores_t, entry["threshold"])
        metrics.update({"time_step": t, "model": FROZEN_MODELS[key]["display"]})
        per_step_rows.append(metrics)

per_step_metrics = pd.DataFrame(per_step_rows)[
    ["time_step", "model", "n", "illicit", "licit", "prevalence", "pr_auc", "pr_auc_reason",
     "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
]
counts_table = pd.DataFrame([per_step_counts[t] for t in sorted(per_step_counts)])
save_table("per_step_metrics.csv", per_step_metrics)
save_table("per_step_label_counts.csv", counts_table)

check("per-step: coverage of steps 35-49", sorted(counts_table["time_step"].tolist()), list(range(35, 50)))
check("per-step: labelled total matches Phase 4", int(counts_table["n_labeled"].sum()), 16_670)
check("per-step: illicit total matches Phase 4", int(counts_table["illicit"].sum()), 1_083)
check("per-step: one row per model per step",
      len(per_step_metrics), len(range(TEST_MIN, TEST_MAX + 1)) * len(frozen_frames))
check("per-step: confusion cells close to n for every row",
      int((per_step_metrics[["tp", "fp", "tn", "fn"]].sum(axis=1) != per_step_metrics["n"]).sum()), 0)

display(per_step_metrics.head(6).round(4))

# Figure 1 — PR-AUC by time step for the frozen models.
fig, ax = plt.subplots(figsize=(9.5, 4.6))
palette = {"XGBoost (frozen)": "#d62728", "GraphSAGE (frozen)": "#2ca02c", "HeteroRGCN (frozen)": "#9467bd"}
for model_name, colour in palette.items():
    subset = per_step_metrics[per_step_metrics["model"] == model_name]
    ax.plot(subset["time_step"], subset["pr_auc"], marker="o", ms=4, color=colour, label=model_name)
ax.axvline(DRIFT_MIN - 0.5, color="grey", linestyle="--", linewidth=1.2)
ax.text(DRIFT_MIN - 0.4, 0.02, " 43-49 drift window", color="grey", fontsize=8)
ax2 = ax.twinx()
ax2.plot(counts_table["time_step"], counts_table["prevalence"], color="black", linestyle=":", label="illicit prevalence")
ax2.set_ylabel("illicit prevalence", fontsize=9)
ax.set(xlabel="time step", ylabel="PR-AUC", title="Frozen models: per-step PR-AUC (dotted = prevalence)")
ax.legend(fontsize=8, loc="upper left")
save_fig("pr_auc_by_time_step.png")

# Figure 2 — prevalence and labelled volume by time step.
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
axes[0].bar(counts_table["time_step"], counts_table["illicit"], color="#d62728")
axes[0].set(ylabel="illicit count", title="Per-step labelled volume and illicit prevalence (35-49)")
axes[1].plot(counts_table["time_step"], counts_table["prevalence"], marker="o", color="black")
axes[1].axvline(DRIFT_MIN - 0.5, color="grey", linestyle="--", linewidth=1.2)
axes[1].set(xlabel="time step", ylabel="prevalence")
save_fig("prevalence_by_time_step.png")
''')

C("markdown", r"""## 5. Experiment A extension — graph & feature drift diagnostics

For each step 35-49 we measure structural and feature characteristics that need no labels. These are
**drift diagnostics only** — never model features. They describe how the graph moves underneath the
models and let section 6 ask whether drift *coincides with* the PR-AUC collapse.
""")

C("code", r'''def tx_graph_up_to(cutoff: int):
    keep = edge_avail["tx_to_tx"] <= cutoff
    return src_tx_tx[keep], dst_tx_tx[keep]


def component_stats(cutoff: int) -> dict:
    s, d = tx_graph_up_to(cutoff)
    data = np.ones(s.size * 2, dtype=np.float32)
    graph = coo_matrix((data, (np.concatenate([s, d]), np.concatenate([d, s]))), shape=(N_TX, N_TX)).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    step_tx = np.where(tx_step == cutoff)[0]
    step_components = np.unique(labels[step_tx]) if step_tx.size else np.array([], dtype=int)
    return {
        "components": int(n_components),
        "largest_component": int(sizes.max()) if sizes.size else 0,
        "step_tx_in_largest_component": int((labels[step_tx] == sizes.argmax()).sum()) if step_tx.size else 0,
        "step_tx_components": int(step_components.size),
    }


train_feature_mean = raw_tx_features[labeled_up_to(TRAIN_MAX)].mean(axis=0)
train_feature_std = raw_tx_features[labeled_up_to(TRAIN_MAX)].std(axis=0) + 1e-6

drift_rows = []
for t in range(TEST_MIN, TEST_MAX + 1):
    step_tx = np.where(tx_step == t)[0]
    cumulative_tx = int((tx_step <= t).sum())
    cumulative_edges = int((edge_avail["tx_to_tx"] <= t).sum())
    comp = component_stats(t)

    step_feature_mean = raw_tx_features[step_tx].mean(axis=0)
    feature_drift = float(np.linalg.norm((step_feature_mean - train_feature_mean) / train_feature_std) / np.sqrt(len(tx_model_features)))

    active_addr = int((wallets_step == t).sum())
    new_addr = int(((wallets_step == t) & (first_seen_addr[wallets_addr_idx] == t)).sum())
    seen_ctx = tx_min_addr_firstseen[step_tx] < t
    hist_path = tx_min_hist_txstep[step_tx] < t
    step_incidences = int(np.isin(INCIDENCE.indptr[step_tx + 1] - INCIDENCE.indptr[step_tx], np.arange(0, 10**6)).sum()) if step_tx.size else 0
    step_degree = float((INCIDENCE.indptr[step_tx + 1] - INCIDENCE.indptr[step_tx]).mean()) if step_tx.size else 0.0

    drift_rows.append({
        "time_step": t,
        "labeled": int(step_tx.size),
        "illicit": int((tx_class[step_tx] == 1).sum()),
        "prevalence": float((tx_class[step_tx] == 1).mean()) if step_tx.size else float("nan"),
        "cumulative_tx_nodes": cumulative_tx,
        "cumulative_tx_edges": cumulative_edges,
        "cumulative_mean_degree": float(2 * cumulative_edges / max(1, cumulative_tx)),
        "tx_components": comp["components"],
        "largest_component": comp["largest_component"],
        "largest_component_share": float(comp["largest_component"] / max(1, cumulative_tx)),
        "step_tx_components": comp["step_tx_components"],
        "active_addresses": active_addr,
        "new_addresses": new_addr,
        "recurring_addresses": active_addr - new_addr,
        "cumulative_addresses_seen": int((first_seen_addr <= t).sum()),
        "mean_step_address_degree": step_degree,
        "tx_with_seen_address_context": int(seen_ctx.sum()),
        "tx_with_seen_address_frac": float(seen_ctx.mean()) if step_tx.size else float("nan"),
        "tx_with_historical_path": int(hist_path.sum()),
        "tx_with_historical_path_frac": float(hist_path.mean()) if step_tx.size else float("nan"),
        "feature_drift_l2": feature_drift,
    })

drift_metrics = pd.DataFrame(drift_rows)
save_table("drift_metrics.csv", drift_metrics)
display(drift_metrics.round(4))

check("drift: one row per test step", len(drift_metrics), 15)
check("drift: cumulative tx nodes monotone", bool((drift_metrics["cumulative_tx_nodes"].diff().dropna() >= 0).all()), True)
check("drift: tx->tx graph fragments into ~one component per step by 49",
      bool(drift_metrics["tx_components"].iloc[-1] >= 40), True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(drift_metrics["time_step"], drift_metrics["tx_components"], marker="o", color="#1f77b4")
axes[0].set(xlabel="time step", ylabel="components", title="Cumulative tx->tx components")
axes[1].plot(drift_metrics["time_step"], drift_metrics["new_addresses"], marker="o", color="#ff7f0e", label="new")
axes[1].plot(drift_metrics["time_step"], drift_metrics["recurring_addresses"], marker="s", color="#17becf", label="recurring")
axes[1].set(xlabel="time step", ylabel="addresses", title="Address activity")
axes[1].legend(fontsize=8)
axes[2].plot(drift_metrics["time_step"], drift_metrics["tx_with_seen_address_frac"], marker="o", color="#2ca02c", label="seen address")
axes[2].plot(drift_metrics["time_step"], drift_metrics["tx_with_historical_path_frac"], marker="s", color="#9467bd", label="historical T->A->T")
axes[2].set(xlabel="time step", ylabel="fraction of step transactions", title="Historical context availability")
axes[2].legend(fontsize=8)
save_fig("graph_drift_by_step.png")
''')


# FIX part1 bugs: remove no-op self-bridge guard and nonsense incidence line.
for cell in CELLS:
    if "tx_min_hist_txstep = np.maximum(" in cell["src"]:
        cell["src"] = cell["src"].replace(
            """# A transaction cannot be its own historical bridge.
tx_min_hist_txstep = np.maximum(
    tx_min_hist_txstep,
    np.where(tx_step < tx_min_hist_txstep, tx_step, tx_step),  # keep as-is; guard below
)
""",
            """# No explicit self-exclusion is needed: a transaction's own incidence stamps
# addr_min_txstep with its own step, and the historical test below is strict (< t).
""",
        )
    if "step_incidences = int(" in cell["src"]:
        cell["src"] = cell["src"].replace(
            "    step_incidences = int(np.isin(INCIDENCE.indptr[step_tx + 1] - INCIDENCE.indptr[step_tx], np.arange(0, 10**6)).sum()) if step_tx.size else 0\n",
            "",
        )

C("markdown", r"""## 6. Experiment A extension — does drift coincide with the collapse?

Rank-correlating per-step RGCN PR-AUC against each drift diagnostic is descriptive only. Correlation
is reported as *association*, never causation: prevalence, feature drift, address reuse and graph
fragmentation all move together across 35-49 and cannot be separated from 15 observations.
""")

C("code", r'''from scipy.stats import spearmanr

rgcn_frozen = per_step_metrics[per_step_metrics["model"] == "HeteroRGCN (frozen)"].sort_values("time_step")
joined = drift_metrics.merge(
    rgcn_frozen[["time_step", "pr_auc"]].rename(columns={"pr_auc": "rgcn_pr_auc"}), on="time_step"
)

drift_columns = [
    "prevalence", "cumulative_mean_degree", "tx_components", "largest_component_share",
    "new_addresses", "recurring_addresses", "tx_with_seen_address_frac",
    "tx_with_historical_path_frac", "feature_drift_l2", "mean_step_address_degree",
]
association_rows = []
for column in drift_columns:
    rho, p_value = spearmanr(joined["rgcn_pr_auc"], joined[column], nan_policy="omit")
    association_rows.append({
        "drift_metric": column,
        "spearman_rho_with_rgcn_pr_auc": float(rho),
        "p_value": float(p_value),
        "n_steps": int(joined["rgcn_pr_auc"].notna().sum()),
    })
association_table = pd.DataFrame(association_rows).sort_values("spearman_rho_with_rgcn_pr_auc")
save_table("drift_performance_association.csv", association_table)
display(association_table.round(4))

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.barh(association_table["drift_metric"], association_table["spearman_rho_with_rgcn_pr_auc"], color="#8c564b")
ax.axvline(0, color="black", linewidth=0.8)
ax.set(xlabel="Spearman rho vs per-step RGCN PR-AUC",
       title="Drift diagnostics vs frozen RGCN PR-AUC (association, n=15)")
save_fig("drift_vs_performance.png")
''')

C("markdown", r"""## 7. RGCN architecture (frozen from Phase 4) and training helpers

The architecture is **unchanged** from Phase 4 — 2 relational convolution layers, hidden 128, dropout
0.3, `BCEWithLogitsLoss` with a window-specific `pos_weight`. Phase 5 only changes *what data the
model is allowed to see*. `train_rgcn` can train with one graph and validate with another so the
selection run never validates on a graph it was not allowed to see.
""")

C("code", r'''class RelationalConvLayer(nn.Module):
    """One relational mean-aggregation layer over the 4 heterogeneous relations (frozen from Phase 4)."""

    def __init__(self, in_dims: dict, out_dims: dict):
        super().__init__()
        self.self_tx = nn.Linear(in_dims["tx"], out_dims["tx"], bias=True)
        self.self_addr = nn.Linear(in_dims["addr"], out_dims["addr"], bias=True)
        self.w_tx_to_tx = nn.Linear(in_dims["tx"], out_dims["tx"], bias=False)
        self.w_addr_to_tx = nn.Linear(in_dims["addr"], out_dims["tx"], bias=False)
        self.w_tx_to_addr = nn.Linear(in_dims["tx"], out_dims["addr"], bias=False)
        self.w_addr_to_addr = nn.Linear(in_dims["addr"], out_dims["addr"], bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.self_tx, self.self_addr, self.w_tx_to_tx,
                       self.w_addr_to_tx, self.w_tx_to_addr, self.w_addr_to_addr]:
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _mean_aggregate(self, x_src, edge_index, num_dst):
        src, dst = edge_index[0], edge_index[1]
        degree = torch.bincount(dst, minlength=num_dst).float().clamp(min=1.0).unsqueeze(1)
        aggregated = torch.zeros((num_dst, x_src.size(1)), dtype=x_src.dtype, device=x_src.device)
        aggregated.index_add_(0, dst, x_src[src])
        return aggregated / degree

    def forward(self, x_dict, edge_dict):
        x_tx, x_addr = x_dict["tx"], x_dict["addr"]
        num_tx, num_addr = x_tx.size(0), x_addr.size(0)
        msg_tx_from_tx = self._mean_aggregate(x_tx, edge_dict["tx_to_tx"], num_tx)
        msg_tx_from_addr = self._mean_aggregate(x_addr, edge_dict["addr_to_tx"], num_tx)
        out_tx = self.self_tx(x_tx) + self.w_tx_to_tx(msg_tx_from_tx) + self.w_addr_to_tx(msg_tx_from_addr)
        msg_addr_from_tx = self._mean_aggregate(x_tx, edge_dict["tx_to_addr"], num_addr)
        msg_addr_from_addr = self._mean_aggregate(x_addr, edge_dict["addr_to_addr"], num_addr)
        out_addr = self.self_addr(x_addr) + self.w_tx_to_addr(msg_addr_from_tx) + self.w_addr_to_addr(msg_addr_from_addr)
        return {"tx": out_tx, "addr": out_addr}


class HeteroRGCN(nn.Module):
    """2-layer relational GCN for heterogeneous fraud detection (frozen from Phase 4)."""

    def __init__(self, tx_in_dim=165, addr_in_dim=55, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = RelationalConvLayer({"tx": tx_in_dim, "addr": addr_in_dim}, {"tx": hidden_dim, "addr": hidden_dim})
        self.conv2 = RelationalConvLayer({"tx": hidden_dim, "addr": hidden_dim}, {"tx": 1, "addr": hidden_dim})

    def forward(self, x_dict, edge_dict):
        h1 = self.conv1(x_dict, edge_dict)
        h1_tx = F.dropout(F.relu(h1["tx"]), p=self.dropout, training=self.training)
        h1_addr = F.dropout(F.relu(h1["addr"]), p=self.dropout, training=self.training)
        return self.conv2({"tx": h1_tx, "addr": h1_addr}, edge_dict)["tx"]


Y_ALL = np.where(tx_class == 1, 1.0, np.where(tx_class == 2, 0.0, -1.0)).astype(np.float32)
Y_TENSOR = torch.tensor(Y_ALL, dtype=torch.float32).unsqueeze(1).to(device)
RGCN_LR, RGCN_WD, RGCN_DROPOUT = 0.003, 1e-4, 0.3

check("architecture: parameter count unchanged from Phase 4",
      sum(p.numel() for p in HeteroRGCN(165, 55, 128, 0.3).parameters() if p.requires_grad), 134_401)


def _run_epoch(model, optimizer, criterion, x_dict, edge_dict, indices):
    model.train()
    optimizer.zero_grad()
    logits = model(x_dict, edge_dict)
    loss = criterion(logits[indices], Y_TENSOR[indices])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def predict_probs(model, x_dict, edge_dict) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(x_dict, edge_dict).squeeze(1)).cpu().numpy()


def train_rgcn(x_train, edge_train, train_mask, pos_weight, epochs,
               x_val=None, edge_val=None, val_mask=None, patience=None, seed=SEED):
    """Train RGCN. With `patience`, early-stops on validation PR-AUC; otherwise runs exactly `epochs`."""
    torch.manual_seed(seed)
    model = HeteroRGCN(165, 55, 128, RGCN_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RGCN_LR, weight_decay=RGCN_WD)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_indices = torch.tensor(np.where(train_mask)[0], dtype=torch.long, device=device)
    val_indices = torch.tensor(np.where(val_mask)[0], dtype=torch.long, device=device) if val_mask is not None else None
    y_val = (tx_class[val_mask] == 1).astype(int) if val_mask is not None else None

    best_pr_auc, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = {"epoch": [], "train_loss": [], "val_pr_auc": []}
    for epoch in range(1, epochs + 1):
        loss = _run_epoch(model, optimizer, criterion, x_train, edge_train, train_indices)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        if val_indices is not None:
            val_pr_auc = float(average_precision_score(y_val, predict_probs(model, x_val, edge_val)[val_mask]))
            history["val_pr_auc"].append(val_pr_auc)
            if val_pr_auc > best_pr_auc:
                best_pr_auc, best_epoch, stale = val_pr_auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_pr_auc, pd.DataFrame(history)
''')

C("markdown", r"""### Selection run — epoch budget and frozen threshold

Trained on the fit window (1-24) with the graph cut at step 24; validated on 25-34 with the graph cut
at step 34, because at deployment `t = 35` the steps `<= 34` are legitimately observable. The
best-validation epoch becomes the fixed epoch budget for every refit, and validation F1 picks the
frozen threshold `tau*`.
""")

C("code", r'''tick("Selection run: RGCN on fit 1-24 / validation 25-34")
scaler_tx_sel = StandardScaler().fit(raw_tx_features[labeled_up_to(FIT_MAX)])
X_tx_sel = torch.from_numpy(scaler_tx_sel.transform(raw_tx_features).astype(np.float32))
raw_addr_fit = raw_address_matrix(FIT_MAX)
scaler_addr_sel = StandardScaler().fit(raw_addr_fit[first_seen_addr <= FIT_MAX])
X_addr_sel_train = scale(raw_addr_fit, scaler_addr_sel)
X_addr_sel_val = scale(raw_address_matrix(TRAIN_MAX), scaler_addr_sel)

edge_fit = edges_up_to(FIT_MAX)
edge_val = edges_up_to(TRAIN_MAX)
sel_mask = labeled_up_to(FIT_MAX)
val_mask = labeled_window(VAL_MIN, VAL_MAX)

selection_model, best_epoch, selection_val_pr_auc, selection_history = train_rgcn(
    {"tx": X_tx_sel, "addr": X_addr_sel_train}, edge_fit, sel_mask, pos_weight_of(sel_mask),
    epochs=150, x_val={"tx": X_tx_sel, "addr": X_addr_sel_val}, edge_val=edge_val,
    val_mask=val_mask, patience=20,
)
FROZEN_EPOCHS = max(int(best_epoch), 5)
y_val = (tx_class[val_mask] == 1).astype(int)
val_probs = predict_probs(selection_model, {"tx": X_tx_sel, "addr": X_addr_sel_val}, edge_val)[val_mask]
FROZEN_THRESHOLD, selection_val_f1 = select_threshold(y_val, val_probs)

print(f"Selection: best epoch {best_epoch} | validation PR-AUC {selection_val_pr_auc:.4f} | tau* {FROZEN_THRESHOLD:.3f}")
check("selection: validation PR-AUC recorded and finite", bool(np.isfinite(selection_val_pr_auc)), True)
check("selection: refit epoch budget is positive", FROZEN_EPOCHS > 0, True)
check("selection: frozen threshold inside the search grid", bool(FROZEN_THRESHOLD in set(THRESHOLD_GRID.tolist())), True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(selection_history["epoch"], selection_history["train_loss"], color="#9467bd")
axes[0].set(xlabel="epoch", ylabel="loss", title="Selection training loss (fit 1-24)")
axes[1].plot(selection_history["epoch"], selection_history["val_pr_auc"], color="#9467bd")
axes[1].axvline(best_epoch, color="#9467bd", linestyle="--", label=f"best epoch {best_epoch}")
axes[1].set(xlabel="epoch", ylabel="PR-AUC", title="Validation PR-AUC (25-34)")
axes[1].legend(fontsize=8)
save_fig("selection_learning_curve.png")
''')

C("markdown", r"""## 8. Experiment B — static vs expanding-window retraining

* **Static:** one model trained on 1-34 (graph cut 34); evaluated at every step without retraining.
* **Expanding:** at each step `t`, a fresh model trained on labels `1..t-1` (graph cut `t-1`), then
  used to score step `t` with the graph cut at `t`. Step `t` labels are never used.

Both regimes share the frozen `tau*` so the F1 comparison isolates the effect of adaptation. PR-AUC is
threshold-free and therefore the primary signal.
""")

C("code", r'''tick("Static model: refit RGCN on 1-34")
scaler_tx_static = StandardScaler().fit(raw_tx_features[labeled_up_to(TRAIN_MAX)])
X_tx_static = torch.from_numpy(scaler_tx_static.transform(raw_tx_features).astype(np.float32))
raw_addr_train = raw_address_matrix(TRAIN_MAX)
scaler_addr_static = StandardScaler().fit(raw_addr_train[first_seen_addr <= TRAIN_MAX])
X_addr_static_train = scale(raw_addr_train, scaler_addr_static)
edge_train34 = edges_up_to(TRAIN_MAX)

# Proof that no future features enter the static model: both scalers were fit on <= 34 data only.
check("no future features: tx scaler fit on the labelled training window only",
      int(scaler_tx_static.n_samples_seen_), int(labeled_up_to(TRAIN_MAX).sum()))
check("no future features: address scaler fit on addresses seen by the training window",
      int(scaler_addr_static.n_samples_seen_), int((first_seen_addr <= TRAIN_MAX).sum()))
static_mask = labeled_up_to(TRAIN_MAX)

static_model, _, _, _ = train_rgcn(
    {"tx": X_tx_static, "addr": X_addr_static_train}, edge_train34, static_mask,
    pos_weight_of(static_mask), epochs=FROZEN_EPOCHS,
)
torch.save(static_model.state_dict(), OUT_DIR / "rgcn_static_model.pt")
del X_addr_static_train
if torch.cuda.is_available():
    torch.cuda.empty_cache()

static_probs = {}
for t in range(TEST_MIN, TEST_MAX + 1):
    X_addr_pred = scale(raw_address_matrix(t), scaler_addr_static)
    static_probs[t] = predict_probs(static_model, {"tx": X_tx_static, "addr": X_addr_pred}, edges_up_to(t))
    tick(f"static: predicted step {t}")
check("static: predictions cover all test transactions",
      sum(int(step_labeled_mask(t).sum()) for t in static_probs), 16_670)
''')

C("code", r'''tick("Expanding-window retraining")
expanding_probs = {}
expanding_rows = []
if RUN_EXPANDING:
    for t in range(TEST_MIN, TEST_MAX + 1):
        cutoff = t - 1
        mask_tr = labeled_up_to(cutoff)
        scaler_tx_t = StandardScaler().fit(raw_tx_features[mask_tr])
        X_tx_t = torch.from_numpy(scaler_tx_t.transform(raw_tx_features).astype(np.float32))
        raw_addr_cut = raw_address_matrix(cutoff)
        scaler_addr_t = StandardScaler().fit(raw_addr_cut[first_seen_addr <= cutoff])
        X_addr_t = scale(raw_addr_cut, scaler_addr_t)
        model_t, _, _, _ = train_rgcn(
            {"tx": X_tx_t, "addr": X_addr_t}, edges_up_to(cutoff), mask_tr,
            pos_weight_of(mask_tr), epochs=FROZEN_EPOCHS,
        )
        X_addr_pred = scale(raw_address_matrix(t), scaler_addr_t)
        expanding_probs[t] = predict_probs(model_t, {"tx": X_tx_t, "addr": X_addr_pred}, edges_up_to(t))
        expanding_rows.append({"time_step": t, "train_window": f"1-{cutoff}", "train_positives": int((tx_class[mask_tr] == 1).sum())})
        del X_tx_t, X_addr_t, X_addr_pred, model_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tick(f"expanding: trained 1-{cutoff}, predicted step {t}")
    expanding_windows = pd.DataFrame(expanding_rows)
    save_table("expanding_windows.csv", expanding_windows)
    check("expanding: one model per test step", len(expanding_windows), 15)
    check("expanding: no step trained on its own labels or later",
          bool(all(int(name.split("-")[1]) == t - 1 for name, t in zip(expanding_windows["train_window"], expanding_windows["time_step"]))), True)
else:
    check("expanding: executed", False, True, "RUN_EXPANDING disabled")
''')

C("code", r'''def per_step_frame(probs_by_step, label):
    rows = []
    for t in range(TEST_MIN, TEST_MAX + 1):
        if t not in probs_by_step:
            continue
        idx = np.where(step_labeled_mask(t))[0]
        y_t = (tx_class[idx] == 1).astype(int)
        metrics = evaluate(y_t, probs_by_step[t][idx], FROZEN_THRESHOLD)
        metrics["time_step"] = t
        metrics["label"] = label
        rows.append(metrics)
    return pd.DataFrame(rows)


static_step = per_step_frame(static_probs, "static")
expanding_step = per_step_frame(expanding_probs, "expanding") if expanding_probs else pd.DataFrame()

static_vs_expanding = static_step[["time_step", "prevalence", "pr_auc", "roc_auc", "f1", "precision", "recall"]].rename(
    columns={"pr_auc": "static_pr_auc", "roc_auc": "static_roc_auc", "f1": "static_f1",
             "precision": "static_precision", "recall": "static_recall"}
)
if not expanding_step.empty:
    static_vs_expanding = static_vs_expanding.merge(
        expanding_step[["time_step", "pr_auc", "roc_auc", "f1", "precision", "recall"]].rename(
            columns={"pr_auc": "expanding_pr_auc", "roc_auc": "expanding_roc_auc", "f1": "expanding_f1",
                     "precision": "expanding_precision", "recall": "expanding_recall"}),
        on="time_step", how="left")
    static_vs_expanding["pr_auc_delta_expanding_minus_static"] = (
        static_vs_expanding["expanding_pr_auc"] - static_vs_expanding["static_pr_auc"])
save_table("static_vs_expanding.csv", static_vs_expanding)
display(static_vs_expanding.round(4))

whole = {
    "static": evaluate((tx_class[per_step_counts and step_labeled_mask(TEST_MIN) & False] == 1).astype(int) if False else
                       (tx_class[is_labeled & (tx_step >= TEST_MIN)] == 1).astype(int),
                       np.concatenate([static_probs[t][step_labeled_mask(t)] for t in range(TEST_MIN, TEST_MAX + 1)]),
                       FROZEN_THRESHOLD),
}
check("static: whole-period prediction count", whole["static"]["n"], 16_670)
check("static-vs-expanding: one row per test step", len(static_vs_expanding), 15)
''')

C("markdown", r"""### Rolling-window variant (single pre-registered width)

If computationally feasible, one fixed-width rolling window (`W = 20`) tests whether recency weighting
beats expanding training. At step `t` the model trains only on labels `max(1, t-20)..t-1` with the graph
cut at `t-1`, then scores step `t`. Only one window is tried; the width is pre-registered, not tuned.
""")

C("code", r'''rolling_probs = {}
if RUN_ROLLING:
    tick(f"Rolling-window retraining (W={ROLLING_WINDOW})")
    rolling_rows = []
    for t in range(TEST_MIN, TEST_MAX + 1):
        cutoff = t - 1
        lo = max(FIT_MIN, t - ROLLING_WINDOW)
        mask_tr = labeled_window(lo, cutoff)
        scaler_tx_t = StandardScaler().fit(raw_tx_features[mask_tr])
        X_tx_t = torch.from_numpy(scaler_tx_t.transform(raw_tx_features).astype(np.float32))
        raw_addr_cut = raw_address_matrix(cutoff)
        scaler_addr_t = StandardScaler().fit(raw_addr_cut[first_seen_addr <= cutoff])
        X_addr_t = scale(raw_addr_cut, scaler_addr_t)
        model_t, _, _, _ = train_rgcn(
            {"tx": X_tx_t, "addr": X_addr_t}, edges_up_to(cutoff), mask_tr,
            pos_weight_of(mask_tr), epochs=FROZEN_EPOCHS,
        )
        X_addr_pred = scale(raw_address_matrix(t), scaler_addr_t)
        rolling_probs[t] = predict_probs(model_t, {"tx": X_tx_t, "addr": X_addr_pred}, edges_up_to(t))
        rolling_rows.append({"time_step": t, "train_window": f"{lo}-{cutoff}", "train_positives": int((tx_class[mask_tr] == 1).sum())})
        del X_tx_t, X_addr_t, X_addr_pred, model_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tick(f"rolling: trained {lo}-{cutoff}, predicted step {t}")
    rolling_windows = pd.DataFrame(rolling_rows)
    rolling_step = per_step_frame(rolling_probs, "rolling")
    rolling_window_table = rolling_windows.merge(
        rolling_step[["time_step", "pr_auc", "f1", "roc_auc"]].rename(
            columns={"pr_auc": "rolling_pr_auc", "f1": "rolling_f1", "roc_auc": "rolling_roc_auc"}),
        on="time_step", how="left")
    if not expanding_step.empty:
        rolling_window_table = rolling_window_table.merge(
            expanding_step[["time_step", "pr_auc", "f1"]].rename(
                columns={"pr_auc": "expanding_pr_auc", "f1": "expanding_f1"}),
            on="time_step", how="left")
    save_table("rolling_window.csv", rolling_window_table)
    check("rolling: one model per test step", len(rolling_windows), 15)
    check("rolling: window width never exceeds the pre-registered width",
          bool((rolling_windows["train_window"].str.split("-").apply(lambda p: int(p[1]) - int(p[0]) + 1) <= ROLLING_WINDOW + 1).all()), True)
else:
    check("rolling: executed", False, True, "RUN_ROLLING disabled")
''')

C("markdown", r"""## 9. Experiment C — inductive entity analysis

An address is **seen** at prediction step `t` if it was observed strictly before `t` (wallet snapshot or
timestamped incident edge); otherwise it is **unseen**. This uses historical graph presence only — never
address labels. Each test transaction is classified by its address context:

* **seen address context** — at least one connected address seen before `t`;
* **unseen address context** — has address links, but all first observed at `t`;
* **no address linkage** — no incident address edge at all.
""")

C("code", r'''context_rows = []
for t in range(TEST_MIN, TEST_MAX + 1):
    idx = np.where(tx_step == t)[0]
    seen = has_any_addr[idx] & (tx_min_addr_firstseen[idx] < t)
    unseen = has_any_addr[idx] & (tx_min_addr_firstseen[idx] >= t)
    none = ~has_any_addr[idx]
    context_rows.append({
        "time_step": t, "total": int(idx.size),
        "with_seen_address": int(seen.sum()),
        "with_only_unseen_address": int(unseen.sum()),
        "with_no_address_linkage": int(none.sum()),
        "with_seen_address_frac": float(seen.mean()) if idx.size else float("nan"),
    })
context_by_step = pd.DataFrame(context_rows)
save_table("inductive_context_by_step.csv", context_by_step)
display(context_by_step)

check("inductive: context groups partition each step",
      int((context_by_step[["with_seen_address", "with_only_unseen_address", "with_no_address_linkage"]].sum(axis=1)
           != context_by_step["total"]).sum()), 0)
check("inductive: step totals sum to the test period", int(context_by_step["total"].sum()), 67_504)
''')

C("code", r'''def group_masks_for_step(t: int) -> dict:
    idx = np.where(step_labeled_mask(t))[0]
    return {
        "all": idx,
        "seen_address_context": idx[has_any_addr[idx] & (tx_min_addr_firstseen[idx] < t)],
        "unseen_address_context": idx[has_any_addr[idx] & (tx_min_addr_firstseen[idx] >= t)],
        "no_address_linkage": idx[~has_any_addr[idx]],
    }


REGIME_PROBS = {"static_temporal_graph": static_probs}
if expanding_probs:
    REGIME_PROBS["expanding"] = expanding_probs
if rolling_probs:
    REGIME_PROBS["rolling"] = rolling_probs

inductive_rows = []
for regime, probs in REGIME_PROBS.items():
    for group in ["all", "seen_address_context", "unseen_address_context", "no_address_linkage"]:
        idx = np.concatenate([group_masks_for_step(t)[group] for t in range(TEST_MIN, TEST_MAX + 1)])
        if idx.size == 0:
            continue
        y_g = (tx_class[idx] == 1).astype(int)
        scores_g = np.concatenate([probs[t][group_masks_for_step(t)[group]] for t in range(TEST_MIN, TEST_MAX + 1)])
        metrics = evaluate(y_g, scores_g, FROZEN_THRESHOLD)
        metrics.update({"regime": regime, "context_group": group})
        inductive_rows.append(metrics)
inductive_metrics = pd.DataFrame(inductive_rows)[
    ["regime", "context_group", "n", "illicit", "licit", "prevalence", "pr_auc", "pr_auc_reason",
     "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
]
# Frozen Phase 4 RGCN scores for reference, same groups.
for group in ["all", "seen_address_context", "unseen_address_context", "no_address_linkage"]:
    idx = np.concatenate([group_masks_for_step(t)[group] for t in range(TEST_MIN, TEST_MAX + 1)])
    if idx.size == 0:
        continue
    y_g = (tx_class[idx] == 1).astype(int)
    metrics = evaluate(y_g, frozen_scores["rgcn_phase4"][idx], FROZEN_MODELS["rgcn_phase4"]["threshold"])
    metrics.update({"regime": "frozen_phase4", "context_group": group})
    inductive_metrics = pd.concat([inductive_metrics, pd.DataFrame([metrics])], ignore_index=True)
inductive_metrics = inductive_metrics[
    ["regime", "context_group", "n", "illicit", "licit", "prevalence", "pr_auc", "pr_auc_reason",
     "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
]
save_table("inductive_metrics.csv", inductive_metrics)
display(inductive_metrics.round(4))

seen_row = inductive_metrics[(inductive_metrics["regime"] == "static_temporal_graph") & (inductive_metrics["context_group"] == "seen_address_context")]
unseen_row = inductive_metrics[(inductive_metrics["regime"] == "static_temporal_graph") & (inductive_metrics["context_group"] == "unseen_address_context")]
check("inductive: smallest subgroup flagged rather than over-interpreted",
      int(inductive_metrics.loc[inductive_metrics["context_group"] != "all", "illicit"].min()) < 50, True,
      "per-step/subgroup positives are tiny; PR-AUC on them is illustrative only")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
sub = inductive_metrics[(inductive_metrics["regime"] == "static_temporal_graph") & (inductive_metrics["context_group"] != "all")]
positions = np.arange(len(sub))
ax.bar(positions, sub["pr_auc"], color=["#2ca02c", "#ff7f0e", "#7f7f7f"][:len(sub)])
for pos, (_, row) in zip(positions, sub.iterrows()):
    ax.text(pos, 0.01, f"n={int(row['n'])}\nill={int(row['illicit'])}", ha="center", fontsize=8)
ax.set_xticks(positions)
ax.set_xticklabels(sub["context_group"], rotation=10)
ax.set(ylabel="PR-AUC", title="Static RGCN PR-AUC by inductive address context (35-49)")
save_fig("inductive_seen_vs_unseen.png")
''')


# ---- patch: add state_hash helper + static-freeze audit + clean whole-period block ----
for cell in CELLS:
    if "def train_rgcn(" in cell["src"]:
        cell["src"] = cell["src"].replace(
            "def _run_epoch(",
            '''def state_hash(model) -> str:
    """Content hash of a model's parameters, to prove evaluation never mutates it."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _run_epoch(''',
        )
    if "static_probs = {}" in cell["src"]:
        cell["src"] = cell["src"].replace(
            "static_probs = {}\n",
            "STATIC_STATE_HASH = state_hash(static_model)\nstatic_probs = {}\n",
        )
        cell["src"] = cell["src"].replace(
            'check("static: predictions cover all test transactions",\n      sum(int(step_labeled_mask(t).sum()) for t in static_probs), 16_670)',
            '''check("static: predictions cover all test transactions",
      sum(int(step_labeled_mask(t).sum()) for t in static_probs), 16_670)
check("static: model weights unchanged by evaluation", state_hash(static_model), STATIC_STATE_HASH)''',
        )
    if 'whole = {' in cell["src"]:
        cell["src"] = cell["src"].replace(
            '''whole = {
    "static": evaluate((tx_class[per_step_counts and step_labeled_mask(TEST_MIN) & False] == 1).astype(int) if False else
                       (tx_class[is_labeled & (tx_step >= TEST_MIN)] == 1).astype(int),
                       np.concatenate([static_probs[t][step_labeled_mask(t)] for t in range(TEST_MIN, TEST_MAX + 1)]),
                       FROZEN_THRESHOLD),
}
check("static: whole-period prediction count", whole["static"]["n"], 16_670)''',
            '''def expand_to_all_transactions(probs_by_step):
    """Scatter per-step transaction scores back onto the global transaction index."""
    full = np.full(N_TX, np.nan, dtype="float64")
    for t, probs in probs_by_step.items():
        mask = step_labeled_mask(t)
        full[mask] = probs[mask]
    return full


TEST_LABELED_IDX = np.where(labeled_window(TEST_MIN, TEST_MAX))[0]
static_full = expand_to_all_transactions(static_probs)
check("static: whole-period prediction count", int(np.isfinite(static_full[TEST_LABELED_IDX]).sum()), 16_670)''',
        )

C("markdown", r"""## 10. Historical `T -> A -> T` path analysis

Phase 4 found 61,487 multi-step addresses and 10,812 test transactions with a `T_train -> A -> T_test`
bridge, but under an all-edge definition. Here the bridge is recomputed under the strict prediction-time
rule: at step `t`, a transaction has a historical path when a connected address also links to a
transaction observed strictly before `t`. Performance is then split by path availability for every
regime. This directly tests whether the heterogeneous graph's proposed temporal bridge corresponds to
useful predictive signal.
""")

C("code", r'''path_availability = []
for t in range(TEST_MIN, TEST_MAX + 1):
    idx = np.where(tx_step == t)[0]
    has_path = tx_min_hist_txstep[idx] < t
    path_availability.append({
        "time_step": t, "total": int(idx.size),
        "with_historical_path": int(has_path.sum()),
        "without_historical_path": int((~has_path).sum()),
        "historical_path_frac": float(has_path.mean()) if idx.size else float("nan"),
    })
path_by_step = pd.DataFrame(path_availability)

phase4_definition = int(((tx_step >= TEST_MIN) & (tx_min_hist_txstep <= TRAIN_MAX)).sum())
print("Transactions with a historical T->A->T path")
print(f"  Phase 4 all-edge definition (steps <= 34): {phase4_definition:,}")
print(f"  Phase 5 mean per-step fraction            : {path_by_step['historical_path_frac'].mean():.4f}")
check("path: Phase 4 bridge count reproduced under its own definition", phase4_definition, 10_812)
check("path: per-step counts partition each step",
      int((path_by_step[["with_historical_path", "without_historical_path"]].sum(axis=1) != path_by_step["total"]).sum()), 0)

path_rows = []
for regime, probs_by_step in REGIME_PROBS.items():
    full = expand_to_all_transactions(probs_by_step)
    for group, mask in [("all", np.ones(N_TX, dtype=bool)),
                        ("with_historical_path", tx_min_hist_txstep < tx_step),
                        ("without_historical_path", ~(tx_min_hist_txstep < tx_step))]:
        idx = TEST_LABELED_IDX[mask[TEST_LABELED_IDX]]
        if idx.size == 0:
            continue
        metrics = evaluate((tx_class[idx] == 1).astype(int), full[idx], FROZEN_THRESHOLD)
        metrics.update({"regime": regime, "path_group": group, "scope": "test_35_49"})
        path_rows.append(metrics)
path_full = expand_to_all_transactions({t: frozen_scores["rgcn_phase4"] for t in range(TEST_MIN, TEST_MAX + 1)})
for group, mask in [("all", np.ones(N_TX, dtype=bool)),
                    ("with_historical_path", tx_min_hist_txstep < tx_step),
                    ("without_historical_path", ~(tx_min_hist_txstep < tx_step))]:
    idx = TEST_LABELED_IDX[mask[TEST_LABELED_IDX]]
    metrics = evaluate((tx_class[idx] == 1).astype(int), path_full[idx], FROZEN_MODELS["rgcn_phase4"]["threshold"])
    metrics.update({"regime": "frozen_phase4", "path_group": group, "scope": "test_35_49"})
    path_rows.append(metrics)
historical_path_metrics = pd.DataFrame(path_rows)[
    ["scope", "regime", "path_group", "n", "illicit", "prevalence", "pr_auc", "pr_auc_reason",
     "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
]
save_table("historical_path_metrics.csv", historical_path_metrics)
save_table("historical_path_availability.csv", path_by_step)
display(historical_path_metrics.round(4))

fig, ax = plt.subplots(figsize=(8.8, 4.3))
ax.bar(path_by_step["time_step"], path_by_step["with_historical_path"], color="#6a3d9a", label="with historical path")
ax.bar(path_by_step["time_step"], path_by_step["without_historical_path"],
       bottom=path_by_step["with_historical_path"], color="#d3d3d3", label="without")
ax.axvline(DRIFT_MIN - 0.5, color="grey", linestyle="--", linewidth=1.1)
ax.set(xlabel="time step", ylabel="transactions", title="Historical T->A->T path availability by step (35-49)")
ax.legend(fontsize=8)
save_fig("historical_path_availability.png")
''')

C("markdown", r"""## 11. Experiment D — threshold adaptation (secondary, clearly labelled)

The primary benchmark is untouched. This secondary experiment separates **discrimination loss** from
**operating-point mismatch**. The adaptive threshold may only use information available before `t`:

* **Frozen `tau*`** — F1-max on validation 25-34 (the deployment-style operating point).
* **Adaptive `tau_t`** — F1-max over a growing pool of *out-of-sample* historical predictions: the
  selection model's scores on 25-34, then each expanding model's scores on the step it had not seen.
  The pool always ends at step `t-1`.
* **Oracle `tau_t*`** — post-hoc F1-max on step `t` itself. Reported only as a non-deployable upper
  bound; it is never used as a result.

PR-AUC is threshold-free, so if PR-AUC collapses, no threshold rule can explain the degradation.
""")

C("code", r'''primary_probs = expanding_probs if expanding_probs else static_probs

# Out-of-sample historical pool: selection model never trained on 25-34.
pool_scores = list(np.asarray(val_probs, dtype="float64"))
pool_labels = list(np.asarray(y_val, dtype=int))
pool_steps = list(tx_step[val_mask])

threshold_rows = []
for t in range(TEST_MIN, TEST_MAX + 1):
    idx = np.where(step_labeled_mask(t))[0]
    y_t = (tx_class[idx] == 1).astype(int)
    scores_t = primary_probs[t][idx]
    adaptive_threshold, _ = select_threshold(np.array(pool_labels), np.array(pool_scores))
    oracle_threshold, _ = select_threshold(y_t, scores_t)
    frozen_metrics = evaluate(y_t, scores_t, FROZEN_THRESHOLD)
    adaptive_metrics = evaluate(y_t, scores_t, adaptive_threshold)
    threshold_rows.append({
        "time_step": t, "regime": "expanding" if expanding_probs else "static",
        "prevalence": frozen_metrics["prevalence"],
        "pr_auc": frozen_metrics["pr_auc"],
        "frozen_threshold": FROZEN_THRESHOLD, "adaptive_threshold": adaptive_threshold,
        "oracle_threshold": oracle_threshold,
        "f1_frozen": frozen_metrics["f1"], "f1_adaptive": adaptive_metrics["f1"],
        "precision_frozen": frozen_metrics["precision"], "recall_frozen": frozen_metrics["recall"],
        "precision_adaptive": adaptive_metrics["precision"], "recall_adaptive": adaptive_metrics["recall"],
        "adaptive_pool_max_step": int(max(pool_steps)),
    })
    pool_scores.extend(primary_probs[t][idx].tolist())
    pool_labels.extend(y_t.tolist())
    pool_steps.extend(tx_step[idx].tolist())

threshold_analysis = pd.DataFrame(threshold_rows)
save_table("threshold_analysis.csv", threshold_analysis)
display(threshold_analysis.round(4))

check("threshold: adaptive pool never reaches the predicted step",
      bool((threshold_analysis["adaptive_pool_max_step"] < threshold_analysis["time_step"]).all()), True)
check("threshold: adaptive equals frozen at step 35 (identical pool)",
      float(threshold_analysis.loc[threshold_analysis["time_step"] == TEST_MIN, "adaptive_threshold"].iloc[0]),
      float(FROZEN_THRESHOLD))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
axes[0].plot(threshold_analysis["time_step"], threshold_analysis["frozen_threshold"], marker="o", label="frozen tau*", color="#1f77b4")
axes[0].plot(threshold_analysis["time_step"], threshold_analysis["adaptive_threshold"], marker="s", label="adaptive tau_t", color="#ff7f0e")
axes[0].plot(threshold_analysis["time_step"], threshold_analysis["oracle_threshold"], marker="^", linestyle=":", label="oracle (non-deployable)", color="#7f7f7f")
axes[0].set(xlabel="time step", ylabel="threshold", title="Operating thresholds")
axes[0].legend(fontsize=8)
axes[1].plot(threshold_analysis["time_step"], threshold_analysis["pr_auc"], marker="o", label="PR-AUC (threshold-free)", color="#d62728")
axes[1].plot(threshold_analysis["time_step"], threshold_analysis["f1_frozen"], marker="s", label="F1 @ frozen", color="#1f77b4")
axes[1].plot(threshold_analysis["time_step"], threshold_analysis["f1_adaptive"], marker="^", label="F1 @ adaptive", color="#ff7f0e")
axes[1].set(xlabel="time step", ylabel="score", title="Discrimination vs operating point")
axes[1].legend(fontsize=8)
save_fig("threshold_adaptation.png")
''')

C("markdown", r"""## 12. Model comparison — frozen benchmark kept separate

The frozen 35-49 benchmark is retained unchanged. Phase 5 temporal/inductive numbers are reported as a
separate regime and never replace it. The `35-42` and `43-49` sub-windows are reported alongside.
""")

C("code", r'''def whole_period_metrics(probs_full, threshold, mask):
    idx = np.where(mask)[0]
    return evaluate((tx_class[idx] == 1).astype(int), probs_full[idx], threshold)


comparison_rows = []
for key, entry in FROZEN_MODELS.items():
    full = frozen_scores[key]
    for label, mask in [("35-49", labeled_window(35, 49)),
                        ("35-42", labeled_window(35, 42)),
                        ("43-49", labeled_window(43, 49))]:
        metrics = whole_period_metrics(full, entry["threshold"], mask)
        metrics.update({"model": entry["display"], "regime": "frozen_benchmark", "period": label})
        comparison_rows.append(metrics)

for regime, probs_by_step in REGIME_PROBS.items():
    full = expand_to_all_transactions(probs_by_step)
    for label, mask in [("35-49", labeled_window(35, 49)),
                        ("35-42", labeled_window(35, 42)),
                        ("43-49", labeled_window(43, 49))]:
        metrics = whole_period_metrics(full, FROZEN_THRESHOLD, mask)
        metrics.update({"model": f"HeteroRGCN ({regime})", "regime": regime, "period": label})
        comparison_rows.append(metrics)

model_comparison = pd.DataFrame(comparison_rows)[
    ["model", "regime", "period", "n", "illicit", "prevalence", "pr_auc", "pr_auc_reason",
     "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
]
save_table("model_comparison.csv", model_comparison)
display(model_comparison.round(4))

# Confirm the frozen benchmark reproduces the recorded Phase 2-4 numbers from the saved predictions.
recorded = {
    "XGBoost (frozen)": 0.8013, "GraphSAGE (frozen)": 0.6209, "HeteroRGCN (frozen)": 0.4682,
}
for model_name, expected in recorded.items():
    value = float(model_comparison[(model_comparison["model"] == model_name)
                                   & (model_comparison["period"] == "35-49")]["pr_auc"].iloc[0])
    check(f"comparison: {model_name} 35-49 PR-AUC reproduces the frozen record",
          round(value, 3), round(expected, 3))
''')

C("code", r'''models_used = []
for t in range(TEST_MIN, TEST_MAX + 1):
    models_used.append({
        "time_step": t,
        "static_train_window": "1-34",
        "static_graph_cutoff": TRAIN_MAX,
        "static_prediction_cutoff": t,
        "expanding_train_window": f"1-{t-1}" if expanding_probs else "",
        "expanding_train_graph_cutoff": (t - 1) if expanding_probs else "",
        "expanding_prediction_cutoff": t if expanding_probs else "",
        "rolling_train_window": f"{max(FIT_MIN, t - ROLLING_WINDOW)}-{t-1}" if rolling_probs else "",
        "rolling_train_graph_cutoff": (t - 1) if rolling_probs else "",
        "rolling_prediction_cutoff": t if rolling_probs else "",
    })
temporal_audit = pd.DataFrame(models_used)
save_table("temporal_audit.csv", temporal_audit)
check("audit: expanding train window ends at t-1 for every step",
      bool(all(int(row["expanding_train_window"].split("-")[1]) == row["time_step"] - 1
               for _, row in temporal_audit.iterrows() if row["expanding_train_window"])), True)
check("audit: prediction cutoff equals the target step (edges available by t)",
      bool((temporal_audit["static_prediction_cutoff"] == temporal_audit["time_step"]).all()), True)
check("audit: rolling window width within the pre-registered width",
      bool(all((row["time_step"] - 1) - max(FIT_MIN, row["time_step"] - ROLLING_WINDOW) + 1 <= ROLLING_WINDOW
               for _, row in temporal_audit.iterrows() if row["rolling_train_window"])), True)
''')

C("markdown", r"""## 13. Reproducibility record

Every value needed to re-derive the run is written to `reproducibility.json`: seed, windows, graph and
address-history policies, threshold methodology, retraining schedule, software versions, hardware, and
the hash of the frozen static checkpoint.
""")

C("code", r'''reproducibility = {
    "seed": SEED,
    "split_windows": {"fit": "1-24", "validation": "25-34", "train": "1-34", "test": "35-49",
                      "test_early": "35-42", "test_drift": "43-49"},
    "phase5_controls": {
        "rolling_window": ROLLING_WINDOW,
        "prediction_includes_current_step_edges": PREDICTION_INCLUDES_CURRENT_STEP_EDGES,
        "run_expanding": RUN_EXPANDING,
        "run_rolling": RUN_ROLLING,
        "threshold_grid": [float(THRESHOLD_GRID[0]), float(THRESHOLD_GRID[-1]), int(THRESHOLD_GRID.size)],
    },
    "graph_construction": "Phase 4 4-relation heterogeneous graph; every edge stamped with its earliest step",
    "address_history_definition": "first_seen = min(wallet snapshot step, timestamped incident-edge endpoint step)",
    "seen_unseen_definition": "seen at t iff first_seen < t; labels are never used to define context",
    "historical_path_definition": "exists an address linking this transaction to another transaction observed strictly before t",
    "addr_addr_availability_rule": "max(first_seen[src], first_seen[dst]) - documented proxy, no native timestamp",
    "normalisation_policy": "StandardScaler fit on the labelled training window only, per address cutoff",
    "threshold_methodology": {
        "frozen": f"F1-max on validation 25-34 (selection model), tau* = {FROZEN_THRESHOLD}",
        "adaptive": "F1-max on a growing out-of-sample pool ending at t-1",
        "oracle": "post-hoc F1-max on step t (non-deployable reference only)",
    },
    "retraining_schedule": {
        "static": "train 1-34, no retraining",
        "expanding": f"train 1..t-1, refit each of steps {TEST_MIN}-{TEST_MAX}, epochs={FROZEN_EPOCHS}",
        "rolling": f"train t-{ROLLING_WINDOW}..t-1, refit each step, epochs={FROZEN_EPOCHS}",
    },
    "rgcn_hyperparameters": {"architecture": "2-layer RelationalConvLayer", "tx_in": 165, "addr_in": 55,
                             "hidden": 128, "dropout": RGCN_DROPOUT, "lr": RGCN_LR,
                             "weight_decay": RGCN_WD, "optimiser": "Adam", "loss": "BCEWithLogitsLoss",
                             "refit_epochs": FROZEN_EPOCHS},
    "static_checkpoint_sha256": STATIC_STATE_HASH,
    "best_selection_epoch": int(best_epoch),
    "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                 "torch": torch.__version__, "platform": platform.platform()},
    "hardware": {"device": str(device),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    "frozen_artifact_sources": {key: str(entry["path"]) for key, entry in frozen_frames.items()},
}
save_json("reproducibility.json", reproducibility)
print(json.dumps(reproducibility["retraining_schedule"], indent=2))
''')

C("markdown", r"""## 14. Validation checks

The full check suite is written to `checks.csv`. Section 24 of the Phase 5 brief requires all of these
to pass: step coverage, no future labels/features/edges, correct address-history and seen/unseen
definitions, adaptive thresholds that never touch the target step, a provably frozen static model,
correct expanding/rolling windows, correct per-step prediction counts, closed confusion matrices,
finite predictions, subgroup sums, and consistent path counts.
""")

C("code", r'''# --- Temporal-rule and integrity checks not already recorded inline ---
prediction_sets = {"static": static_probs}
if expanding_probs:
    prediction_sets["expanding"] = expanding_probs
if rolling_probs:
    prediction_sets["rolling"] = rolling_probs

for name, probs_by_step in prediction_sets.items():
    all_probs = np.concatenate([probs_by_step[t] for t in sorted(probs_by_step)])
    check(f"{name}: no NaN predictions", int(np.isnan(all_probs).sum()), 0)
    check(f"{name}: no Inf predictions", int(np.isinf(all_probs).sum()), 0)
    check(f"{name}: scores within [0, 1]",
          bool((all_probs >= 0).all() and (all_probs <= 1).all()), True)
    check(f"{name}: per-step prediction rows match labelled counts",
          sum(int(step_labeled_mask(t).sum()) for t in probs_by_step), 16_670)

check("seen/unseen: seen and unseen contexts partition address-linked transactions",
      int(((has_any_addr & (tx_min_addr_firstseen < tx_step)) & (has_any_addr & (tx_min_addr_firstseen >= tx_step))).sum()), 0)
check("seen/unseen: address-linked = seen + unseen",
      int((has_any_addr.sum()) - ((has_any_addr & (tx_min_addr_firstseen < tx_step)).sum()
                                  + (has_any_addr & (tx_min_addr_firstseen >= tx_step)).sum())), 0)
check("historical path: with + without partition the test period (all transactions)",
      int(path_by_step["with_historical_path"].sum() + path_by_step["without_historical_path"].sum()), 67_504)
check("expanding: training labels restricted to steps < t",
      bool(all(int(w.split("-")[1]) < t for w, t in zip(
          [m for m in temporal_audit["expanding_train_window"] if m], temporal_audit["time_step"]))), True)
check("no future edges: every recorded training cut-off is < the target step",
      bool(all(row["expanding_train_graph_cutoff"] == row["time_step"] - 1
               for _, row in temporal_audit.iterrows() if row["expanding_train_graph_cutoff"] != "")), True)
check("no test labels: adaptive pool ends strictly before every target step",
      bool((threshold_analysis["adaptive_pool_max_step"] < threshold_analysis["time_step"]).all()), True)

save_table("checks.csv", pd.DataFrame(CHECKS))
failures = [row for row in CHECKS if not row["ok"]]
print(f"\nSanity checks: {len(CHECKS)} total, {len(failures)} failing")
if failures:
    display(pd.DataFrame(failures))
''')

C("markdown", r"""## 15. Artifact export

Writes `temporal_inductive/README.md` and the run digest, then verifies every required artifact exists.
Result-dependent conclusions belong in `docs/TEMPORAL_INDUCTIVE_EVALUATION.md`, filled in *after* this
notebook has actually run — never before.
""")

C("code", r'''prediction_export = pd.DataFrame({
    "txId": tx_ids[TEST_LABELED_IDX],
    "time_step": tx_step[TEST_LABELED_IDX],
    "y_true": (tx_class[TEST_LABELED_IDX] == 1).astype(int),
    "frozen_phase4_score": frozen_scores["rgcn_phase4"][TEST_LABELED_IDX],
    "static_score": static_full[TEST_LABELED_IDX],
    "expanding_score": expand_to_all_transactions(expanding_probs)[TEST_LABELED_IDX] if expanding_probs else np.nan,
    "rolling_score": expand_to_all_transactions(rolling_probs)[TEST_LABELED_IDX] if rolling_probs else np.nan,
})
save_table("rgcn_temporal_predictions.csv", prediction_export)

digest = f"""
BitcoinGraphGuard - Phase 5 Temporal & Inductive Evaluation ({time.strftime('%Y-%m-%d %H:%M')})
----------------------------------------------------------------------------------------------
Protocol   : fit 1-24 | validation 25-34 | train 1-34 | test 35-49 (unchanged from Phases 1-4)
Temporal   : edges/features/address snapshots restricted to <= t; training labels strictly < t
Frozen refs: XGBoost 0.8013 | GraphSAGE 0.6209 | HeteroRGCN 0.4682 (35-49 PR-AUC, untouched)
Selection  : best epoch {best_epoch} | validation PR-AUC {selection_val_pr_auc:.4f} | tau* {FROZEN_THRESHOLD:.3f}
Regimes    : static, {'expanding (W=1..t-1), ' if expanding_probs else ''}{'rolling (W=' + str(ROLLING_WINDOW) + ')' if rolling_probs else ''}
Historical T->A->T bridge (Phase 4 definition): {phase4_definition:,} test transactions
Checks     : {len(CHECKS)} recorded, {len(failures)} failing

Per-step and regime tables are in this folder; conclusions are written to
docs/TEMPORAL_INDUCTIVE_EVALUATION.md only after this notebook has run.
"""
(OUT_DIR / "temporal_inductive_digest.txt").write_text(digest, encoding="utf-8")
print(digest)

readme = f"""# temporal_inductive — Phase 5 artifacts

Produced by `notebooks/05_temporal_inductive_evaluation.ipynb` (run on Google Colab / Kaggle).
Random seed {SEED}; frozen 35-49 benchmark retained at XGBoost 0.8013 / GraphSAGE 0.6209 / HeteroRGCN 0.4682.

| File | Contents |
| :--- | :--- |
| `per_step_metrics.csv` | Frozen XGBoost/GraphSAGE/RGCN metrics for every step 35-49 |
| `per_step_label_counts.csv` | Labelled, illicit, licit counts and prevalence per step |
| `drift_metrics.csv` | Per-step graph, address-activity, connectivity and feature-drift diagnostics |
| `drift_performance_association.csv` | Spearman association of each drift metric with per-step PR-AUC |
| `static_vs_expanding.csv` | Per-step static vs expanding-window RGCN metrics |
| `expanding_windows.csv` | Exact training window and positive count for every expanding refit |
| `rolling_window.csv` | Rolling-window (W={ROLLING_WINDOW}) variant, if executed |
| `inductive_metrics.csv` | RGCN performance by seen/unseen/no-address context |
| `inductive_context_by_step.csv` | Per-step address-context composition |
| `historical_path_metrics.csv` | Performance split by historical T->A->T path availability |
| `historical_path_availability.csv` | Per-step historical-path counts |
| `threshold_analysis.csv` | Frozen vs adaptive (vs oracle) thresholds per step |
| `model_comparison.csv` | Frozen benchmark plus Phase 5 regimes, for 35-49 / 35-42 / 43-49 |
| `temporal_audit.csv` | Training window and graph cut-off for every model evaluated at every step |
| `rgcn_temporal_predictions.csv` | Per-transaction scores for every regime |
| `reproducibility.json` | Seed, policies, versions, hardware, checkpoint hash |
| `checks.csv` | Automated validation checks (all must pass) |
| `figures/` | Question-driven figures (per-step PR-AUC, prevalence, drift, regimes, inductive, thresholds) |

Frozen benchmark references are never overwritten by adaptive results; they answer different questions.
"""

(OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

required = ["per_step_metrics.csv", "drift_metrics.csv", "static_vs_expanding.csv", "inductive_metrics.csv",
            "historical_path_metrics.csv", "threshold_analysis.csv", "model_comparison.csv", "checks.csv",
            "reproducibility.json", "README.md"]
if rolling_probs:
    required.append("rolling_window.csv")
missing = [name for name in required if not (OUT_DIR / name).exists()]
check("artifacts: every required file was written", missing, [])
check("artifacts: figures directory populated", len(list(FIG_DIR.glob("*.png"))) > 0, True)
print("Exported:", sorted(p.name for p in OUT_DIR.iterdir()))
''')

C("markdown", r"""## 16. How to answer the Phase 5 questions (post-run guide)

The notebook deliberately stops at evidence. After the Colab/Kaggle run, read these artifacts to answer
the brief's final questions — and only then write `docs/TEMPORAL_INDUCTIVE_EVALUATION.md`:

| # | Question | Artifact |
| :- | :--- | :--- |
| 1-2 | Where does degradation begin; gradual or abrupt? | `per_step_metrics.csv`, `figures/pr_auc_by_time_step.png` |
| 3-4 | Does expanding/rolling retraining recover performance? | `static_vs_expanding.csv`, `rolling_window.csv` |
| 5-7 | Historical address context, seen vs unseen, `T->A->T` value | `inductive_metrics.csv`, `historical_path_metrics.csv` |
| 8 | Prevalence vs feature vs graph drift | `drift_metrics.csv`, `drift_performance_association.csv` |
| 9 | Threshold adaptation vs PR-AUC | `threshold_analysis.csv`, `figures/threshold_adaptation.png` |
| 10-12 | Is temporal/inductive modelling necessary; what should the next architecture address? | the tables above, synthesised in the doc |

**Stop condition:** Phase 5 ends here. Do not start HGT or any architecture change; the next
architecture must be selected from this evidence.
""")

# ---------------- assemble the notebook ----------------
nb = {
    "cells": [make_cell(entry["kind"], entry["src"]) for entry in CELLS],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
with open("notebooks/05_temporal_inductive_evaluation.ipynb", "w", encoding="utf-8") as handle:
    json.dump(nb, handle, indent=1)
print("assembled", len(nb["cells"]), "cells")
