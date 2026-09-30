"""Build notebooks/03_graphsage.ipynb — Phase 3 v2, giving homogeneous GraphSAGE cross-step signal.

The notebook is generated rather than hand-edited so the cell boundaries, ordering and the
self-contained Colab layout stay reviewable in plain text. Run from the repository root:

    python scripts/generate_notebook_03_v2.py

v2 keeps the homogeneous node space (transactions only). It adds two independent ways to give
GraphSAGE cross-step reach: leakage-safe per-transaction lag features, and a projected tx-to-tx
edge set derived through shared wallet addresses. The frozen Phase 3 artifacts in
results/GraphSage/ are left untouched; v2 writes to graphsage_v2/.
"""

import json
import uuid
from pathlib import Path

OUTPUT = Path("notebooks/03_graphsage.ipynb")

CELLS: list = []


def _record(kind: str, source: str) -> None:
    """Append one notebook cell, splitting the source into the list-of-lines nbformat wants."""
    lines = source.split("\n")
    payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell = {
        "cell_type": kind,
        "metadata": {},
        "source": payload,
        "id": uuid.uuid4().hex,
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    CELLS.append(cell)


def md(source: str) -> None:
    _record("markdown", source)


def code(source: str) -> None:
    _record("code", source)


md(r"""# BitcoinGraphGuard — Phase 3: GraphSAGE Baseline (v2, cross-step signal)

**Phase:** 3 of 9 · Homogeneous Graph Neural Network Baseline — cross-step investigation
**Environment:** Google Colab / Kaggle. This notebook is **never executed on the laptop**.
**Input:** the real Elliptic++ dataset (`txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv`, `wallets_features.csv`, `AddrTx_edgelist.csv`, `TxAddr_edgelist.csv`).
**Output:** `graphsage_v2/`. The frozen Phase 3 artifacts in `GraphSage/` are **never overwritten**.

## Why this notebook exists

The Phase 3 GraphSAGE run reached **PR-AUC 0.6209** on test 35–49, well below the frozen XGBoost
ceiling of **0.8013**. That is not a tuning problem. It is structural, and it was proven empirically:
**all 234,355 edges in `txs_edgelist.csv` are 100% intra-step.** A homogeneous GNN on that graph can
only smooth features *within* one time step, so it cannot learn cross-step patterns at all. Adding
layers, changing aggregators, or training longer cannot create temporal reach that the edge set does
not contain.

This notebook therefore gives GraphSAGE **actual cross-step signal while staying homogeneous** — one
node type (transactions), one edge type (transaction → transaction). Full heterogeneity (wallet
nodes, RGCN) belongs to `notebooks/04_heterogeneous_gnn.ipynb` and is deliberately out of scope.

## Experiments

| # | Experiment | What changes | What it tests |
| :-- | :--- | :--- | :--- |
| 1 | **Baseline control** | nothing (2-layer GraphSAGE, intra-step graph, 165 features) | reproduces the 0.6209 number |
| 2 | **Temporal feature augmentation** | + leakage-safe address-history lag features | can cross-step memory live in the *features* while edges stay intra-step? |
| 3 | **Projected multi-step graph** | + tx→tx edges derived through shared addresses (past → future) | can cross-step memory live in the *edges*? |
| 4 | **Depth control** | 3 layers instead of 2 on the original intra-step graph | negative control: depth alone cannot substitute for cross-step edges |
| 5 | **Combined (supplementary)** | lag features **and** projected graph | do the two signals compound? |

Every option reports **PR-AUC / ROC-AUC / F1** on **35–49 (full), 35–42 (early), and 43–49 (drift)**,
and is compared against the frozen XGBoost (0.8013) and the original GraphSAGE (0.6209).

### Frozen reference numbers (from `docs/GRAPHSAGE.md`, `docs/XGBOOST.md`, `docs/XGBOOST_V2.md`)

| Model | PR-AUC 35–49 | ROC-AUC 35–49 | F1 35–49 | PR-AUC 35–42 | PR-AUC 43–49 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Prevalence baseline | 0.0650 | 0.5000 | 0.1220 | — | — |
| Logistic Regression | 0.2917 | 0.8828 | 0.4385 | — | — |
| 2-layer MLP | 0.4768 | 0.8912 | 0.5759 | 0.5912 | 0.0395 |
| **GraphSAGE (Phase 3, frozen)** | **0.6209** | **0.9044** | **0.5945** | **0.7346** | **0.0504** |
| XGBoost Optimized (frozen) | **0.8013** | **0.9281** | **0.7818** | **0.9215** | **0.0427** |
| HeteroRGCN (Phase 4, frozen) | 0.4682 | 0.8946 | 0.5295 | 0.6083 | 0.0550 |

**Protocol (unchanged):** fit 1–24 / validation 25–34 / refit 1–34 / test 35–49, with 35–42 and 43–49
reported separately. Class 3 (`unknown`) is never a training example. `StandardScaler` is fit only on
historical rows and `BCEWithLogitsLoss(pos_weight=…)` is computed only on the training window.

**To run:** `Runtime → Run all`.
""")
md(r"""## 1. Setup & Environment

One config cell: seeds, device, the frozen temporal windows, and the two cross-step knobs
(`TEMPORAL_WINDOW`, `MAX_PRIOR_NEIGHBORS`). No personal paths are hardcoded.
""")

code(r"""import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

try:
    from torch_geometric.nn import SAGEConv

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("PyG not detected; using the self-contained SAGEConv fallback.")

SEED = 42

# Frozen temporal protocol
FIT_MIN, FIT_MAX = 1, 24
VAL_MIN, VAL_MAX = 25, 34
TRAIN_MIN, TRAIN_MAX = 1, 34
TEST_MIN, TEST_MAX = 35, 49
TEST_EARLY_MAX = 42
DRIFT_MIN = 43

# Cross-step signal knobs (see section 8)
TEMPORAL_WINDOW = 5       # how far back a shared address may reach (steps)
MAX_PRIOR_NEIGHBORS = 5   # cap on past neighbours contributed per membership row

# Shared training hyperparameters (identical for every option)
HIDDEN_CHANNELS = 128
DROPOUT = 0.3
MAX_EPOCHS = 150
PATIENCE = 20
LR = 5e-3
WEIGHT_DECAY = 1e-4

LABEL_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 11

print(
    f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
    f"torch {torch.__version__} | device {device} | PyG {PYG_AVAILABLE}"
)
""")
code(r'''FILES = {
    "txs_features": "txs_features.csv",
    "txs_classes": "txs_classes.csv",
    "txs_edgelist": "txs_edgelist.csv",
    "addr_tx": "AddrTx_edgelist.csv",
    "tx_addr": "TxAddr_edgelist.csv",
    "wallets_features": "wallets_features.csv",
}

# One Drive folder holds the raw dataset and every notebook's outputs.
DRIVE_ROOT = Path(
    os.environ.get(
        "BITCOINGUARD_DRIVE_ROOT", "/content/drive/MyDrive/Projects/Bitcoin Graph"
    )
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
        "Elliptic++ not found; set ELLIPTIC_DATA_DIR to the folder with the CSVs."
    )


def read_header(path: Path) -> list:
    """Column names of a CSV without reading data rows."""
    return pd.read_csv(path, nrows=0).columns.tolist()


DATA_DIR = resolve_data_dir()
# v2 artifacts go to a NEW folder; the frozen Phase 3 GraphSage/ artifacts are never touched.
OUT_DIR = DRIVE_ROOT / "graphsage_v2"
FIG_DIR = OUT_DIR / "figures"
for directory in (OUT_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CHECKS: list = []


def check(name: str, value, expected=None, note: str = "") -> None:
    """Record a sanity check for the section 12 validation report."""
    ok = expected is None or value == expected
    CHECKS.append(
        {"check": name, "value": value, "expected": expected, "ok": bool(ok), "note": note}
    )
    print(
        f"[{'ok' if ok else 'FAIL'}] {name}: {value}"
        + (f" (expected {expected})" if expected is not None else "")
    )


def check_true(name: str, condition, note: str = "") -> None:
    check(name, bool(condition), True, note)


def check_close(name: str, value, expected, tol: float = 0.02, note: str = "") -> None:
    """Assert an approximate match (used where a run is not bit-for-bit deterministic)."""
    ok = abs(float(value) - float(expected)) <= tol
    CHECKS.append(
        {
            "check": name,
            "value": float(value),
            "expected": float(expected),
            "ok": bool(ok),
            "note": note or f"tolerance +/-{tol}",
        }
    )
    print(
        f"[{'ok' if ok else 'FAIL'}] {name}: {float(value):.4f} "
        f"(expected {float(expected):.4f} +/-{tol})"
    )


def save_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT_DIR / name, index=False)


def save_json(name: str, payload: dict) -> None:
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.show()


# Frozen reference numbers, exactly as recorded in docs/. ROC-AUC / F1 are only carried where the
# source document recorded them; nothing here is inferred or fabricated.
FROZEN = {
    "Prevalence Baseline": {
        "35-49": {"pr_auc": 0.0650, "roc_auc": 0.5000, "f1": 0.1220},
        "35-42": {"pr_auc": 0.0916},
        "43-49": {"pr_auc": 0.0253},
    },
    "Logistic Regression": {
        "35-49": {"pr_auc": 0.2917, "roc_auc": 0.8828, "f1": 0.4385},
    },
    "MLP Baseline (frozen)": {
        "35-49": {"pr_auc": 0.4768, "roc_auc": 0.8912, "f1": 0.5759},
        "35-42": {"pr_auc": 0.5912, "f1": 0.6529},
        "43-49": {"pr_auc": 0.0395},
    },
    "GraphSAGE Phase 3 (frozen)": {
        "35-49": {"pr_auc": 0.6209, "roc_auc": 0.9044, "f1": 0.5945},
        "35-42": {"pr_auc": 0.7346, "f1": 0.6759},
        "43-49": {"pr_auc": 0.0504},
    },
    "XGBoost Optimized (frozen)": {
        "35-49": {"pr_auc": 0.8013, "roc_auc": 0.9281, "f1": 0.7818},
        "35-42": {"pr_auc": 0.9215},
        "43-49": {"pr_auc": 0.0427},
    },
    "HeteroRGCN Phase 4 (frozen)": {
        "35-49": {"pr_auc": 0.4682, "roc_auc": 0.8946, "f1": 0.5295},
        "35-42": {"pr_auc": 0.6083, "f1": 0.6221},
        "43-49": {"pr_auc": 0.0550},
    },
}

EXPERIMENTS: dict = {}

print(f"data : {DATA_DIR}")
print(f"out  : {OUT_DIR}")
''')
md(r"""## 2. Load Data

Raw files are opened read-only and never modified. We load the 165 primary transaction features
(93 `Local_feature_*` + 72 `Aggregate_feature_*`; the 17 domain columns are excluded at parse time
for the same collinearity/missing-value reasons as Phase 3), the labels, the transaction edge list,
and — new in v2 — the wallet/address data needed to derive cross-step signal **without** adding
wallet nodes to the graph.
""")

code(r"""txs_columns = read_header(DATA_DIR / FILES["txs_features"])
all_feature_columns = [c for c in txs_columns if c not in ("txId", "Time step")]
domain_columns = [
    c for c in all_feature_columns if not c.startswith(("Local_feature", "Aggregate_feature"))
]
model_features = [c for c in all_feature_columns if c not in domain_columns]

print(
    f"{len(all_feature_columns)} total features = {len(domain_columns)} domain (excluded) "
    f"+ {len(model_features)} model features"
)
check("features: total raw features", len(all_feature_columns), 182)
check("features: domain columns excluded", len(domain_columns), 17)
check("features: primary model features", len(model_features), 165)
check(
    "features: domain block is disjoint from model features",
    set(domain_columns).isdisjoint(model_features),
    True,
)

txs_classes = pd.read_csv(DATA_DIR / FILES["txs_classes"], dtype={"txId": str})
check("labels: transaction rows in classes CSV", len(txs_classes), 203_769)
check(
    "labels: classes are subset of {1, 2, 3}",
    set(txs_classes["class"].unique()) <= {1, 2, 3},
    True,
)
""")

code(r"""started = time.time()
txs = pd.read_csv(
    DATA_DIR / FILES["txs_features"],
    usecols=["txId", "Time step"] + model_features,
    dtype={"txId": str, "Time step": "int16", **{c: "float32" for c in model_features}},
)
txs["class"] = txs["txId"].map(txs_classes.set_index("txId")["class"]).astype("int8")

print(
    f"Loaded {len(txs):,} transactions in {time.time() - started:.1f}s | "
    f"Memory: {txs.memory_usage(deep=True).sum() / 1e6:.1f} MB"
)
check("transactions: total row count", len(txs), 203_769)
check("transactions: zero missing labels", int(txs["class"].isna().sum()), 0)
check("transactions: time step span", (int(txs["Time step"].min()), int(txs["Time step"].max())), (1, 49))
check(
    "transactions: missing values in 165 model features",
    int(txs[model_features].isna().to_numpy().sum()),
    0,
)

label_counts = {LABEL_NAMES[int(k)]: int(v) for k, v in txs["class"].value_counts().items()}
print("Transaction label distribution:", label_counts)
check("transactions: illicit count", label_counts.get("illicit"), 4_545)
check("transactions: licit count", label_counts.get("licit"), 42_019)
check("transactions: unknown count", label_counts.get("unknown"), 157_205)
""")
code(r"""# Map string txId to contiguous integer index [0, N-1]
tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(txs["txId"].values)}
node_steps = txs["Time step"].to_numpy()

# Original homogeneous transaction graph: txs_edgelist.csv only.
txs_edges_df = pd.read_csv(DATA_DIR / FILES["txs_edgelist"], dtype=str)
check("graph: raw transaction edges count", len(txs_edges_df), 234_355)
check("graph: edge source nodes in node universe", bool(txs_edges_df["txId1"].isin(tx_id_to_idx).all()), True)
check("graph: edge target nodes in node universe", bool(txs_edges_df["txId2"].isin(tx_id_to_idx).all()), True)

src_indices = txs_edges_df["txId1"].map(tx_id_to_idx).values.astype(np.int64)
dst_indices = txs_edges_df["txId2"].map(tx_id_to_idx).values.astype(np.int64)
edge_index = torch.tensor(np.stack([src_indices, dst_indices], axis=0), dtype=torch.long)
check("graph: edge_index shape", tuple(edge_index.shape), (2, 234_355))

# Empirical intra-step confinement of the original graph (the structural ceiling).
orig_intra = int((node_steps[src_indices] == node_steps[dst_indices]).sum())
orig_cross = int((node_steps[src_indices] != node_steps[dst_indices]).sum())
print(f"Original txs_edgelist: {orig_intra:,} intra-step / {orig_cross:,} cross-step edges")
check("graph: 100% of original transaction edges are intra-step", orig_intra, 234_355)
check(
    "graph: zero cross-step edges in the original graph",
    orig_cross,
    0,
    note="Proves the Phase 3 ceiling is structural, not an optimisation artefact",
)

# Degrees on the original graph (used by the ablation/diagnostics)
in_degrees = np.bincount(dst_indices, minlength=len(txs))
out_degrees = np.bincount(src_indices, minlength=len(txs))
txs["in_degree"] = in_degrees
txs["out_degree"] = out_degrees
txs["total_degree"] = in_degrees + out_degrees

graph_diag = pd.DataFrame(
    [
        {"metric": "total_nodes", "value": len(txs)},
        {"metric": "total_edges", "value": len(src_indices)},
        {"metric": "intra_step_edges", "value": orig_intra},
        {"metric": "cross_step_edges", "value": orig_cross},
        {"metric": "cross_step_fraction", "value": orig_cross / len(src_indices)},
    ]
)
save_table("graph_diagnostics.csv", graph_diag)
display(graph_diag)
""")
md(r"""## 3. Temporal Protocol & Split Masks

The protocol is identical to Phases 1–3: fit 1–24 (model selection + early stopping) / validation
25–34 (early stopping metric + threshold selection) / refit 1–34 / test 35–49, always reported with
the 35–42 and 43–49 sub-windows. Only labeled transactions (classes 1 and 2) enter the loss and the
metrics; class 3 (`unknown`) is never treated as licit. Test rows are never touched by scaling,
training or threshold selection.
""")

code(r"""step_series = txs["Time step"]
is_labeled_series = txs["class"].isin([1, 2])

mask_fit = ((step_series >= FIT_MIN) & (step_series <= FIT_MAX) & is_labeled_series).values
mask_val = ((step_series >= VAL_MIN) & (step_series <= VAL_MAX) & is_labeled_series).values
mask_train = ((step_series >= TRAIN_MIN) & (step_series <= TRAIN_MAX) & is_labeled_series).values
mask_test = ((step_series >= TEST_MIN) & (step_series <= TEST_MAX) & is_labeled_series).values

split_masks = {"fit": mask_fit, "validation": mask_val, "train": mask_train, "test": mask_test}
split_steps = {"fit": "1-24", "validation": "25-34", "train": "1-34", "test": "35-49"}

window_rows = []
for name, mask in split_masks.items():
    labels = txs.loc[mask, "class"]
    illicit_cnt = int((labels == 1).sum())
    licit_cnt = int((labels == 2).sum())
    window_rows.append(
        {
            "window": name,
            "steps": split_steps[name],
            "labeled": illicit_cnt + licit_cnt,
            "illicit": illicit_cnt,
            "licit": licit_cnt,
            "illicit_share": illicit_cnt / max(1, illicit_cnt + licit_cnt),
        }
    )
window_table = pd.DataFrame(window_rows)
save_table("split_windows.csv", window_table)
display(window_table)

check("protocol: fit labeled rows", int(window_table.loc[window_table["window"] == "fit", "labeled"].iloc[0]), 23_606)
check("protocol: validation labeled rows", int(window_table.loc[window_table["window"] == "validation", "labeled"].iloc[0]), 6_288)
check("protocol: train labeled rows", int(window_table.loc[window_table["window"] == "train", "labeled"].iloc[0]), 29_894)
check("protocol: test labeled rows", int(window_table.loc[window_table["window"] == "test", "labeled"].iloc[0]), 16_670)
check("protocol: test illicit rows", int(window_table.loc[window_table["window"] == "test", "illicit"].iloc[0]), 1_083)

# Row-level test windows (the exact masks used by every experiment's evaluation)
test_step_vals = txs.loc[mask_test, "Time step"].values
y_test = (txs.loc[mask_test, "class"] == 1).values.astype(int)
WINDOWS = {
    "35-49": np.ones(len(y_test), dtype=bool),
    "35-42": test_step_vals <= TEST_EARLY_MAX,
    "43-49": test_step_vals >= DRIFT_MIN,
}
check("protocol: 35-42 sub-window labeled rows", int(WINDOWS["35-42"].sum()), 9_983)
check("protocol: 43-49 sub-window labeled rows", int(WINDOWS["43-49"].sum()), 6_687)
check("protocol: test sub-windows partition the full window", int(WINDOWS["35-42"].sum() + WINDOWS["43-49"].sum()), len(y_test))

val_y = (txs.loc[mask_val, "class"] == 1).values.astype(int)
""")

md(r"""## 4. Leakage-Safe Feature Scaling

Both scalers are fit strictly on historical rows — `fit` (1–24) for model selection and `train`
(1–34) for the final refit — exactly as in Phase 3. Every option reuses the same scaling code path,
including the augmented feature matrix built in §7.
""")

code(r'''def scaled_matrices(raw: np.ndarray, fit_rows: np.ndarray, train_rows: np.ndarray):
    """Return (selection, final) scaled matrices; each scaler sees only its own window."""
    scaler_fit = StandardScaler().fit(raw[fit_rows])
    scaler_train = StandardScaler().fit(raw[train_rows])
    return (
        scaler_fit.transform(raw).astype("float32"),
        scaler_train.transform(raw).astype("float32"),
        scaler_fit,
        scaler_train,
    )


raw_base = txs[model_features].to_numpy(dtype="float32")
X_base_sel_np, X_base_fin_np, _, _ = scaled_matrices(raw_base, mask_fit, mask_train)
X_base_sel = torch.tensor(X_base_sel_np, dtype=torch.float32)
X_base_fin = torch.tensor(X_base_fin_np, dtype=torch.float32)

y_all = np.where(txs["class"] == 1, 1.0, np.where(txs["class"] == 2, 0.0, -1.0)).astype("float32")
y_tensor = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)

check("leakage: fit window has no test steps", int((txs.loc[mask_fit, "Time step"] > FIT_MAX).sum()), 0)
check("leakage: train window has no test steps", int((txs.loc[mask_train, "Time step"] > TRAIN_MAX).sum()), 0)
check("leakage: test window has no train steps", int((txs.loc[mask_test, "Time step"] <= TRAIN_MAX).sum()), 0)
check("leakage: base feature tensor shape", tuple(X_base_sel.shape), (203_769, 165))
check("leakage: label tensor shape", tuple(y_tensor.shape), (203_769, 1))
check("leakage: unknown labels are masked out of both training windows",
      int(((y_tensor.squeeze(1).numpy() < 0) & (mask_train | mask_fit)).sum()), 0)
''')

md(r"""## 5. Model Architecture

The homogeneous GraphSAGE is unchanged from Phase 3 — mean aggregation, 128 hidden units, dropout
0.3, one output logit — except that the layer count is now a parameter so §9's depth control and the
cross-step runs share one implementation. A self-contained sparse fallback keeps the notebook
runnable without PyG.
""")

code(r'''class SAGEConvFallback(nn.Module):
    """Self-contained SAGEConv layer with mean aggregation using PyTorch sparse operations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.zeros_(self.lin_l.bias)
        nn.init.xavier_uniform_(self.lin_r.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1.0).unsqueeze(1)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        out = out / deg
        return self.lin_l(x) + self.lin_r(out)


class GraphSAGENet(nn.Module):
    """Mean-aggregation GraphSAGE with a configurable depth (one node type, one edge type)."""

    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout: float = 0.3,
        n_layers: int = 2,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("GraphSAGE needs at least two layers")
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        convs = []
        for i in range(n_layers):
            if PYG_AVAILABLE:
                convs.append(SAGEConv(dims[i], dims[i + 1], aggr="mean"))
            else:
                convs.append(SAGEConvFallback(dims[i], dims[i + 1]))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


_probe = GraphSAGENet(in_channels=len(model_features), hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT, n_layers=2)
check("architecture: 2-layer GraphSAGE output shape", tuple(_probe(X_base_sel[:10], edge_index[:, :20]).shape), (10, 1))
_probe3 = GraphSAGENet(in_channels=len(model_features), hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT, n_layers=3)
check("architecture: 3-layer GraphSAGE output shape", tuple(_probe3(X_base_sel[:10], edge_index[:, :20]).shape), (10, 1))
print("2-layer params:", sum(p.numel() for p in _probe.parameters() if p.requires_grad))
print("3-layer params:", sum(p.numel() for p in _probe3.parameters() if p.requires_grad))
''')
md(r"""## 6. Training Pipeline

Every option shares one training path: Adam ($\eta = 0.005$, weight decay $10^{-4}$),
`BCEWithLogitsLoss(pos_weight = negatives / positives)` computed only on the training window, early
stopping on validation PR-AUC (max 150 epochs, patience 20), then a refit on 1–34 for the best epoch
count. The operating threshold is F1-maximising on validation 25–34 and is frozen before any test
evaluation.
""")

code(r'''def compute_pos_weight(mask: np.ndarray) -> torch.Tensor:
    """pos_weight = negatives / positives, strictly on the masked split."""
    labels = txs.loc[mask, "class"].values
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 2))
    return torch.tensor([negatives / max(1, positives)], dtype=torch.float32)


pos_weight_fit = compute_pos_weight(mask_fit).to(device)
pos_weight_train = compute_pos_weight(mask_train).to(device)
print(f"pos_weight (fit 1-24): {pos_weight_fit.item():.4f} | pos_weight (train 1-34): {pos_weight_train.item():.4f}")
check("training: fit pos_weight computed on fit rows only", round(pos_weight_fit.item(), 2), 9.64)
check("training: train pos_weight computed on train rows only", round(pos_weight_train.item(), 2), 7.63)


def train_model(
    model_fn,
    X_tensor: torch.Tensor,
    edge_idx: torch.Tensor,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    pos_weight: torch.Tensor,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
):
    """Train with validation PR-AUC early stopping; returns (model, best_epoch, best_val_pr_auc, history)."""
    torch.manual_seed(seed)
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x = X_tensor.to(device)
    edges = edge_idx.to(device)
    y_true = y_tensor.to(device)
    train_indices = torch.tensor(np.where(fit_mask)[0], dtype=torch.long, device=device)
    val_indices = torch.tensor(np.where(val_mask)[0], dtype=torch.long, device=device)
    y_val_np = txs.loc[val_mask, "class"].values == 1

    best_val_pr_auc, best_epoch, best_state, epochs_no_improve = -1.0, 0, None, 0
    history = {"epoch": [], "train_loss": [], "val_pr_auc": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edges)
        loss = criterion(logits[train_indices], y_true[train_indices])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x, edges)[val_indices].squeeze(1)).cpu().numpy()
        val_pr_auc = float(average_precision_score(y_val_np, val_probs))

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["val_pr_auc"].append(val_pr_auc)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_pr_auc, pd.DataFrame(history)


def refit_model(model_fn, X_tensor, edge_idx, train_mask, pos_weight, epochs, lr=LR, weight_decay=WEIGHT_DECAY, seed=SEED):
    """Refit on the full historical window (1-34) for a fixed epoch count."""
    torch.manual_seed(seed)
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x, edges, y_true = X_tensor.to(device), edge_idx.to(device), y_tensor.to(device)
    train_indices = torch.tensor(np.where(train_mask)[0], dtype=torch.long, device=device)
    model.train()
    for _ in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = criterion(model(x, edges)[train_indices], y_true[train_indices])
        loss.backward()
        optimizer.step()
    return model
''')

code(r'''def get_model_probabilities(model: nn.Module, X_tensor: torch.Tensor, edge_idx: torch.Tensor) -> np.ndarray:
    """Sigmoid probabilities for every node."""
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device), edge_idx.to(device)).squeeze(1)
    return torch.sigmoid(logits).cpu().numpy()


def select_threshold(y_true: np.ndarray, scores: np.ndarray):
    """F1-maximising threshold on validation rows only."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.round(np.linspace(0.005, 0.995, 199), 4):
        value = f1_score(y_true, (scores >= candidate).astype(int), zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(candidate), float(value)
    return best_threshold, best_f1


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """All standard metrics at one frozen operating threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        "n": int(y_true.size),
        "illicit": int(y_true.sum()),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y_true, scores)) if 0 < y_true.sum() < y_true.size else float("nan"),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "predicted_positives": int(predicted.sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
''')

code(r'''def run_experiment(key, label, X_sel, X_fin, edge_idx, n_layers=2, graph_label="", features_label="", notes=""):
    """Train, refit and evaluate one option under the frozen protocol."""
    in_channels = int(X_sel.shape[1])

    def model_fn():
        return GraphSAGENet(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=1,
            dropout=DROPOUT,
            n_layers=n_layers,
        )

    started = time.time()
    sel_model, best_epoch, best_val_pr_auc, history = train_model(
        model_fn, X_sel, edge_idx, mask_fit, mask_val, pos_weight_fit
    )
    val_probs = get_model_probabilities(sel_model, X_sel, edge_idx)[mask_val]
    threshold, val_f1 = select_threshold(val_y, val_probs)

    fin_model = refit_model(model_fn, X_fin, edge_idx, mask_train, pos_weight_train, epochs=best_epoch)
    test_probs = get_model_probabilities(fin_model, X_fin, edge_idx)[mask_test]
    elapsed = time.time() - started

    windows = {name: evaluate(y_test[m], test_probs[m], threshold) for name, m in WINDOWS.items()}
    n_params = int(sum(p.numel() for p in fin_model.parameters() if p.requires_grad))
    EXPERIMENTS[key] = {
        "key": key, "label": label, "graph": graph_label, "features": features_label, "notes": notes,
        "n_features": in_channels, "n_layers": n_layers, "n_params": n_params,
        "threshold": float(threshold), "validation_f1": float(val_f1),
        "best_epoch": int(best_epoch), "best_val_pr_auc": float(best_val_pr_auc),
        "train_seconds": float(elapsed), "history": history, "windows": windows,
        "test_probs": test_probs,
    }
    torch.save(fin_model.state_dict(), OUT_DIR / f"{key}_model.pt")

    print(f"--- {label} ---")
    print(f"features={in_channels} | layers={n_layers} | params={n_params:,} | best_epoch={best_epoch} | "
          f"val_PR-AUC={best_val_pr_auc:.4f} | tau*={threshold:.3f} | {elapsed:.1f}s")
    print("  PR-AUC  35-49={:.4f} | 35-42={:.4f} | 43-49={:.4f}".format(
        windows["35-49"]["pr_auc"], windows["35-42"]["pr_auc"], windows["43-49"]["pr_auc"]))
    return EXPERIMENTS[key]
''')
md(r"""## 7. Option 1 — Baseline Control (2-layer GraphSAGE, intra-step graph)

This is the **sanity check, not the target**: 2-layer mean-aggregation GraphSAGE on the original
`txs_edgelist.csv` graph with the 165 base features, exactly as Phase 3. It must land near
**PR-AUC 0.6209** on 35–49; if it does, every later delta is attributable to the cross-step change
and not to a change in protocol.
""")

code(r"""opt1 = run_experiment(
    "opt1_baseline",
    "O1 Baseline (2-layer GraphSAGE, intra-step)",
    X_base_sel,
    X_base_fin,
    edge_index,
    n_layers=2,
    graph_label="txs_edgelist (100% intra-step)",
    features_label="165 base features",
    notes="control reproduction of the Phase 3 0.6209 result",
)
check_close(
    "option 1: reproduces the Phase 3 baseline PR-AUC 0.6209 on 35-49",
    opt1["windows"]["35-49"]["pr_auc"],
    0.6209,
    tol=0.02,
    note="small drift expected from GPU non-determinism in the refit",
)
check_close(
    "option 1: reproduces the Phase 3 baseline ROC-AUC 0.9044 on 35-49",
    opt1["windows"]["35-49"]["roc_auc"],
    0.9044,
    tol=0.02,
)
""")
md(r"""## 8. Option 2 — Temporal Feature Augmentation (address-history lag features)

The edges stay intra-step, but the **node features themselves carry cross-step memory**. For each
transaction at step $t$ we look up every address it touches (via `AddrTx` and `TxAddr`), take that
address's most recent wallet snapshot **strictly before** $t$, and aggregate across the transaction's
addresses. Wallet snapshots are cumulative-as-of-that-step, so this is genuine prior-activity
history, and the strict `< t` cutoff is the safest reading of the "≤ t" leakage rule.

Ten wallet columns are averaged, plus two indicators (`has_prior_address_history`,
`prior_address_share`) so the model can tell "no history" from "history of zeros" — a blank is never
silently treated as a real 0. Wallet rows are deduplicated to `(address, time step)` first.
""")

code(r"""# ---- Every (transaction, address) link, input or output ----
check_true("lag: wallets_features.csv is present", (DATA_DIR / FILES["wallets_features"]).exists())
addr_tx = pd.read_csv(DATA_DIR / FILES["addr_tx"], dtype=str)
tx_addr = pd.read_csv(DATA_DIR / FILES["tx_addr"], dtype=str)
check("lag: AddrTx edge count", len(addr_tx), 477_117)
check("lag: TxAddr edge count", len(tx_addr), 837_124)

memberships = pd.concat(
    [
        addr_tx.rename(columns={"input_address": "address"})[["txId", "address"]],
        tx_addr.rename(columns={"output_address": "address"})[["txId", "address"]],
    ],
    ignore_index=True,
).drop_duplicates()
print(f"Unique (transaction, address) memberships: {len(memberships):,}")
check("lag: all memberships reference known transactions", bool(memberships["txId"].isin(tx_id_to_idx).all()), True)

tx_step = txs[["txId", "Time step"]].rename(columns={"Time step": "step"})
mem = memberships.merge(tx_step, on="txId", how="left")
check("lag: every membership resolves to a transaction step", int(mem["step"].isna().sum()), 0)
# merge_asof below requires the join key to have one identical dtype on both sides.
mem["step"] = mem["step"].astype("int32")

# ---- Wallet snapshots, deduplicated to (address, time step) as Phase 1 requires ----
WALLET_LAG_SOURCE = {
    "num_txs_as_sender": "num_txs_as_sender",
    "num_txs_as receiver": "num_txs_as_receiver",
    "total_txs": "total_txs",
    "num_timesteps_appeared_in": "num_timesteps_appeared_in",
    "btc_transacted_total": "btc_transacted_total",
    "btc_sent_total": "btc_sent_total",
    "btc_received_total": "btc_received_total",
    "fees_total": "fees_total",
    "lifetime_in_blocks": "lifetime_in_blocks",
    "transacted_w_address_total": "transacted_w_address_total",
}
wallet_columns = ["address", "Time step"] + list(WALLET_LAG_SOURCE)
wallet_dtypes = {"address": str, "Time step": "int16", **{c: "float32" for c in WALLET_LAG_SOURCE}}

parts, raw_wallet_rows = [], 0
for chunk in pd.read_csv(
    DATA_DIR / FILES["wallets_features"], usecols=wallet_columns, dtype=wallet_dtypes, chunksize=250_000
):
    raw_wallet_rows += len(chunk)
    parts.append(chunk)
wallet_snapshots = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["address", "Time step"])
del parts
gc.collect()
wallet_snapshots = (
    wallet_snapshots.rename(columns={"Time step": "step"})
    .astype({"step": "int32"})
    .sort_values("step", kind="mergesort")
    .reset_index(drop=True)
)
wallet_snapshots["_snap_step"] = wallet_snapshots["step"]
print(f"Wallet rows: {raw_wallet_rows:,} raw -> {len(wallet_snapshots):,} distinct (address, step) snapshots")
check("lag: raw wallet row count", raw_wallet_rows, 1_268_260)
check("lag: duplicates collapsed below the raw row count", bool(len(wallet_snapshots) < raw_wallet_rows), True)
check("lag: wallet snapshot step span", (int(wallet_snapshots["step"].min()), int(wallet_snapshots["step"].max())), (1, 49))

# ---- Latest strictly-prior snapshot per (transaction, address) ----
mem_sorted = mem.sort_values("step", kind="mergesort").reset_index(drop=True)
merged = pd.merge_asof(
    mem_sorted, wallet_snapshots, on="step", by="address", direction="backward", allow_exact_matches=False
)
check(
    "lag: no wallet snapshot at or after the transaction step (leakage)",
    int((merged["_snap_step"] >= merged["step"]).sum()),
    0,
    note="allow_exact_matches=False enforces a strict < t cutoff",
)

# Independent sampled brute-force verification of merge_asof. The lookup structures are built
# only for the sampled addresses so this stays cheap on Colab.
rng = np.random.default_rng(SEED)
sample_pos = rng.choice(len(merged), size=min(5000, len(merged)), replace=False)
m_addr, m_step, m_snap = merged["address"].to_numpy(), merged["step"].to_numpy(), merged["_snap_step"].to_numpy()
snap_sub = wallet_snapshots[wallet_snapshots["address"].isin(pd.unique(m_addr[sample_pos]))]
snap_steps_by_addr = {a: np.sort(g["step"].to_numpy()) for a, g in snap_sub.groupby("address")}
violations = 0
for i in sample_pos:
    arr = snap_steps_by_addr.get(m_addr[i])
    expected = np.nan
    if arr is not None:
        j = np.searchsorted(arr, m_step[i], side="left") - 1
        if j >= 0:
            expected = arr[j]
    if not ((np.isnan(expected) and np.isnan(m_snap[i])) or expected == m_snap[i]):
        violations += 1
check("lag: sampled brute force agrees merge_asof returns the latest prior snapshot", violations, 0)

# ---- Aggregate across the transaction's addresses ----
lag_src_cols = list(WALLET_LAG_SOURCE)
lag_names = [f"lag_{WALLET_LAG_SOURCE[c]}_mean" for c in lag_src_cols]
lag_agg = merged.groupby("txId")[lag_src_cols].mean()
lag_agg.columns = lag_names
history = (
    merged.assign(_has=merged["_snap_step"].notna())
    .groupby("txId")
    .agg(n_addr=("address", "size"), n_hist=("_has", "sum"))
)
lag_frame = txs[["txId"]].merge(lag_agg, on="txId", how="left").merge(history, on="txId", how="left")
lag_frame["has_prior_address_history"] = lag_frame["n_hist"].fillna(0).gt(0).astype("int8")
lag_frame["prior_address_share"] = (lag_frame["n_hist"] / lag_frame["n_addr"]).fillna(0.0).astype("float32")
lag_frame[lag_names] = lag_frame[lag_names].fillna(0.0)
lag_feature_names = lag_names + ["has_prior_address_history", "prior_address_share"]

check("lag: lag feature count", len(lag_feature_names), 12)
check("lag: lag frame aligned to node order", bool((lag_frame["txId"].values == txs["txId"].values).all()), True)
check(
    "lag: no NaN/inf in lag features",
    int((~np.isfinite(lag_frame[lag_feature_names].to_numpy(dtype="float32"))).sum()),
    0,
)
check_true("lag: some transactions have address history", int(lag_frame["has_prior_address_history"].sum()) > 0)
check_true(
    "lag: some transactions have no history (0-filled with an explicit indicator)",
    int((lag_frame["has_prior_address_history"] == 0).sum()) > 0,
)

lag_audit = pd.DataFrame(
    [
        {"metric": "membership_pairs", "value": len(memberships)},
        {"metric": "lag_features_added", "value": len(lag_feature_names)},
        {"metric": "transactions_with_history", "value": int(lag_frame["has_prior_address_history"].sum())},
        {"metric": "transactions_without_history", "value": int((lag_frame["has_prior_address_history"] == 0).sum())},
        {"metric": "raw_wallet_rows", "value": raw_wallet_rows},
        {"metric": "deduped_wallet_snapshots", "value": len(wallet_snapshots)},
        {"metric": "temporal_cutoff", "value": "snapshot step < transaction step"},
    ]
)
save_table("lag_feature_audit.csv", lag_audit)
display(lag_audit)
""")

code(r"""LAG_MATRIX = lag_frame[lag_feature_names].to_numpy(dtype="float32")
raw_aug = np.concatenate([raw_base, LAG_MATRIX], axis=1)
X_aug_sel_np, X_aug_fin_np, _, _ = scaled_matrices(raw_aug, mask_fit, mask_train)
X_aug_sel = torch.tensor(X_aug_sel_np, dtype=torch.float32)
X_aug_fin = torch.tensor(X_aug_fin_np, dtype=torch.float32)

check("lag: augmented feature count = 165 base + 12 lag", int(raw_aug.shape[1]), 177)
check("lag: augmented tensors match the node universe", tuple(X_aug_sel.shape), (203_769, 177))
check("lag: augmented matrix is finite", int((~np.isfinite(raw_aug)).sum()), 0)

save_json(
    "feature_list.json",
    {
        "base_model_features": model_features,
        "lag_feature_names": lag_feature_names,
        "wallet_lag_source_columns": WALLET_LAG_SOURCE,
        "n_base_features": len(model_features),
        "n_lag_features": len(lag_feature_names),
    },
)
""")

code(r"""opt2 = run_experiment(
    "opt2_lag_features",
    "O2 Temporal lag features (intra-step graph)",
    X_aug_sel,
    X_aug_fin,
    edge_index,
    n_layers=2,
    graph_label="txs_edgelist (100% intra-step)",
    features_label="165 base + 12 address-history lag features",
    notes="cross-step memory carried by node features; edges still intra-step",
)
""")
md(r"""## 9. Option 3 — Projected Multi-Step Graph (cross-step tx→tx edges via shared addresses)

Here the **edges** carry the cross-step signal. Two transactions that share an intermediating
address (`AddrTx` / `TxAddr`) are connected when the earlier one is at a *nearby past* step:

$$\text{edge}(A \to B) \iff \text{addr}(A) \cap \text{addr}(B) \neq \varnothing \;\wedge\; 0 < t_B - t_A \le W$$

with $W$ = `TEMPORAL_WINDOW` (5) and at most `MAX_PRIOR_NEIGHBORS` (5) past neighbours per
membership row. The node space is **still transactions only**, so the model stays the homogeneous
GraphSAGE of §5 — only the edge set gains temporal reach.

Two deliberate choices, stated plainly:

1. **Strictly past → future.** For the equal-step case ($t_A = t_B$) the pair is already reachable
   through `txs_edgelist.csv`, and adding same-step shared-address edges would confound "temporal
   reach" with "denser same-step connectivity". The construction therefore adds only strictly
   cross-step edges and reports the same-step candidates it skipped.
2. **Direction is non-decreasing in time**, so every edge satisfies $t_{src} \le t_{dst}$. Message
   passing can therefore only absorb information at or before a node's own step — the graph-level
   leakage guarantee this notebook asserts below.
""")

code(r'''# Projected tx -> tx edges through shared addresses: strictly past -> future.
mem_p = mem[["txId", "address", "step"]].copy()
mem_p["tx_idx"] = mem_p["txId"].map(tx_id_to_idx).astype(np.int64)
mem_p = mem_p.sort_values(["address", "step"], kind="mergesort").reset_index(drop=True)

proj_tx_idx = mem_p["tx_idx"].to_numpy()
proj_step = mem_p["step"].to_numpy()
grouped = mem_p.groupby("address", sort=False)

src_parts, dst_parts, same_step_slots = [], [], 0
for k in range(1, MAX_PRIOR_NEIGHBORS + 1):
    prev_idx = grouped["tx_idx"].shift(k).to_numpy()
    prev_step = grouped["step"].shift(k).to_numpy()
    has_prev = ~pd.isna(prev_step)
    same_step_slots += int((has_prev & (proj_step == prev_step)).sum())
    valid = has_prev & (proj_step > prev_step) & ((proj_step - prev_step) <= TEMPORAL_WINDOW)
    src_parts.append(prev_idx[valid].astype(np.int64))
    dst_parts.append(proj_tx_idx[valid])

proj_src = np.concatenate(src_parts)
proj_dst = np.concatenate(dst_parts)
keep = proj_src != proj_dst
projected_edges = (
    pd.DataFrame({"src": proj_src[keep], "dst": proj_dst[keep]})
    .drop_duplicates()
    .reset_index(drop=True)
)
del mem_p, grouped, src_parts, dst_parts
gc.collect()

original_edges = pd.DataFrame({"src": src_indices, "dst": dst_indices})
combined_edges = (
    pd.concat([original_edges, projected_edges], ignore_index=True)
    .drop_duplicates()
    .reset_index(drop=True)
)
edge_index_projected = torch.tensor(combined_edges[["src", "dst"]].to_numpy().T, dtype=torch.long)
print(f"Projected cross-step edges: {len(projected_edges):,} | combined graph edges: {len(combined_edges):,}")


def edge_time_profile(edges: pd.DataFrame, tag: str) -> dict:
    """Split an edge set into intra-step / cross-step / backward-in-time."""
    s = txs["Time step"].to_numpy()[edges["src"].to_numpy()]
    d = txs["Time step"].to_numpy()[edges["dst"].to_numpy()]
    intra, cross, backward = int((s == d).sum()), int((s < d).sum()), int((s > d).sum())
    return {
        "graph": tag,
        "edges": len(edges),
        "intra_step": intra,
        "cross_step": cross,
        "backward_in_time": backward,
        "cross_step_fraction": cross / max(1, len(edges)),
    }


graph_diag_projected = pd.DataFrame(
    [
        edge_time_profile(original_edges, "original txs_edgelist"),
        edge_time_profile(projected_edges, "projected shared-address"),
        edge_time_profile(combined_edges, "combined (trained on)"),
    ]
)
save_table("graph_diagnostics_projected.csv", graph_diag_projected)
display(graph_diag_projected)

combined_cross_fraction = float(
    graph_diag_projected.loc[graph_diag_projected["graph"] == "combined (trained on)", "cross_step_fraction"].iloc[0]
)
check_true("graph: projected graph contains cross-step edges (> 0%)", len(projected_edges) > 0)
check(
    "graph: every projected edge is strictly past -> future",
    int(
        (
            txs["Time step"].to_numpy()[projected_edges["src"].to_numpy()]
            >= txs["Time step"].to_numpy()[projected_edges["dst"].to_numpy()]
        ).sum()
    ),
    0,
)
check(
    "graph: no backward-in-time edge in the combined graph",
    int((txs["Time step"].to_numpy()[combined_edges["src"].to_numpy()] > txs["Time step"].to_numpy()[combined_edges["dst"].to_numpy()]).sum()),
    0,
    note="graph-level leakage guarantee: every edge satisfies t_src <= t_dst",
)
check_true("graph: combined graph cross-step fraction is > 0", combined_cross_fraction > 0)
check("graph: combined cross-step fraction (informational)", round(combined_cross_fraction, 4))
check("graph: same-step shared-address candidate slots skipped (informational)", same_step_slots)
check("graph: combined edge_index shape", tuple(edge_index_projected.shape), (2, len(combined_edges)))

# Independent check: a sampled projected edge really does share an address. Only the sampled
# endpoints' address sets are materialised.
tx_ids = txs["txId"].to_numpy()
rng = np.random.default_rng(SEED)
sample = projected_edges.sample(n=min(2000, len(projected_edges)), random_state=SEED)
sample_pairs = sample[["src", "dst"]].to_numpy()
needed_tx_ids = set(tx_ids[sample_pairs[:, 0]]) | set(tx_ids[sample_pairs[:, 1]])
addr_sets = memberships[memberships["txId"].isin(needed_tx_ids)].groupby("txId")["address"].apply(set)
shared_ok = sum(1 for src, dst in sample_pairs if addr_sets[tx_ids[src]] & addr_sets[tx_ids[dst]])
check("graph: sampled projected edges share an intermediating address", shared_ok, len(sample))
''')

code(r"""opt3 = run_experiment(
    "opt3_projected",
    "O3 Projected cross-step graph (2-layer GraphSAGE)",
    X_base_sel,
    X_base_fin,
    edge_index_projected,
    n_layers=2,
    graph_label=f"txs_edgelist + projected cross-step edges (W={TEMPORAL_WINDOW})",
    features_label="165 base features",
    notes="cross-step reach carried by the edge set; node features unchanged",
)
""")
md(r"""## 10. Option 4 — Depth Control (3-layer GraphSAGE, intra-step graph)

**This is a negative control, not a main experiment.** It is the same graph and the same features as
Option 1 with one extra SAGE layer, included only to demonstrate that **depth cannot manufacture
cross-step reach**: a 3-hop intra-step receptive field is still confined to a single time step. If
Option 4 does not move the number, the Phase 3 ceiling was never a depth problem.
""")

code(r"""opt4 = run_experiment(
    "opt4_depth3",
    "O4 Depth control (3-layer GraphSAGE, intra-step)",
    X_base_sel,
    X_base_fin,
    edge_index,
    n_layers=3,
    graph_label="txs_edgelist (100% intra-step)",
    features_label="165 base features",
    notes="negative control: deeper receptive field still confined within a single time step",
)
""")

md(r"""## 11. Option 5 — Combined: Lag Features **and** Projected Graph (supplementary)

Options 2 and 3 are deliberately isolated so each signal can be attributed. This supplementary run
combines them (177 features on the projected graph) to answer the practical question directly: if
both kinds of cross-step signal are present, does the gap to XGBoost close any further?
""")

code(r"""opt5 = run_experiment(
    "opt5_combined",
    "O5 Combined (lag features + projected graph)",
    X_aug_sel,
    X_aug_fin,
    edge_index_projected,
    n_layers=2,
    graph_label=f"txs_edgelist + projected cross-step edges (W={TEMPORAL_WINDOW})",
    features_label="165 base + 12 address-history lag features",
    notes="supplementary: both cross-step signals together",
)
""")
md(r"""## 12. Comparison — One Summary Table

Every option is reported on the three windows alongside the frozen references. `roc_auc` and `f1`
are left blank for frozen rows where the source document did not record them — nothing is inferred.
""")

code(r"""WINDOW_NAMES = ["35-49", "35-42", "43-49"]

summary_rows = []
for key, exp in EXPERIMENTS.items():
    for window in WINDOW_NAMES:
        m = exp["windows"][window]
        summary_rows.append(
            {
                "model": exp["label"], "source": "this notebook", "option": key, "window": window,
                "graph": exp["graph"], "features": exp["features"], "n_layers": exp["n_layers"],
                "n_features": exp["n_features"],
                "n": m["n"], "illicit": m["illicit"], "prevalence": m["prevalence"],
                "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"],
                "precision": m["precision"], "recall": m["recall"], "threshold": m["threshold"],
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            }
        )
for name, windows in FROZEN.items():
    for window in WINDOW_NAMES:
        d = windows.get(window, {})
        summary_rows.append(
            {
                "model": name, "source": "frozen reference", "option": "reference", "window": window,
                "graph": "n/a", "features": "n/a", "n_layers": np.nan, "n_features": np.nan,
                "n": np.nan, "illicit": np.nan, "prevalence": np.nan,
                "pr_auc": d.get("pr_auc", np.nan), "roc_auc": d.get("roc_auc", np.nan), "f1": d.get("f1", np.nan),
                "precision": np.nan, "recall": np.nan, "threshold": np.nan,
                "tp": np.nan, "fp": np.nan, "tn": np.nan, "fn": np.nan,
            }
        )
summary_long = pd.DataFrame(summary_rows)

pivots = []
for metric in ["pr_auc", "roc_auc", "f1"]:
    piv = summary_long.pivot_table(index="model", columns="window", values=metric, aggfunc="first")
    piv.columns = [f"{metric}_{c}" for c in piv.columns]
    pivots.append(piv)
summary_wide = pd.concat(pivots, axis=1).reset_index()

model_order = [exp["label"] for exp in EXPERIMENTS.values()] + list(FROZEN)
summary_wide["model"] = pd.Categorical(summary_wide["model"], categories=model_order, ordered=True)
summary_wide = summary_wide.sort_values("model").reset_index(drop=True)
ordered_columns = ["model"] + [f"{metric}_{w}" for w in WINDOW_NAMES for metric in ["pr_auc", "roc_auc", "f1"]]
summary_wide = summary_wide[ordered_columns]

save_table("experiment_summary.csv", summary_long)
save_table("experiment_summary_wide.csv", summary_wide)
display(summary_wide.round(4))

for window in WINDOW_NAMES:
    for exp in EXPERIMENTS.values():
        m = exp["windows"][window]
        check(
            f"summary: {exp['key']} {window} confusion matrix closes",
            m["tp"] + m["fp"] + m["tn"] + m["fn"],
            m["n"],
        )
check("summary: all five options are present", sorted(EXPERIMENTS) == ["opt1_baseline", "opt2_lag_features", "opt3_projected", "opt4_depth3", "opt5_combined"], True)

# --- Chart 1: PR-AUC by window for the options and the two headline references ---
plot_models = [exp["label"] for exp in EXPERIMENTS.values()] + ["GraphSAGE Phase 3 (frozen)", "XGBoost Optimized (frozen)"]


def pinned_pr_auc(model: str, window: str) -> float:
    rows = summary_long[(summary_long["model"] == model) & (summary_long["window"] == window)]
    return float(rows["pr_auc"].iloc[0]) if len(rows) else float("nan")


fig, ax = plt.subplots(figsize=(12.5, 5.2))
positions = np.arange(len(plot_models))
width = 0.26
colours = {"35-49": "#4c72b0", "35-42": "#55a868", "43-49": "#c44e52"}
for offset, window in zip([-width, 0.0, width], WINDOW_NAMES):
    values = [pinned_pr_auc(model, window) for model in plot_models]
    ax.bar(positions + offset, values, width, label=f"PR-AUC {window}", color=colours[window])
    for x, value in zip(positions + offset, values):
        if np.isfinite(value):
            ax.text(x, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(positions)
ax.set_xticklabels(plot_models, rotation=18, ha="right", fontsize=8)
ax.set(ylabel="PR-AUC", title="Cross-step signal vs the frozen references (PR-AUC by window)")
ax.legend(fontsize=8)
save_fig("experiment_pr_auc_by_window.png")

# --- Chart 2: validation PR-AUC learning curves ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for key, exp in EXPERIMENTS.items():
    history = exp["history"]
    axes[0].plot(history["epoch"], history["train_loss"], lw=1.3, label=exp["label"])
    axes[1].plot(history["epoch"], history["val_pr_auc"], lw=1.6, label=f"{exp['label']} (best={exp['best_val_pr_auc']:.4f})")
    axes[1].axvline(exp["best_epoch"], linestyle=":", alpha=0.5)
axes[0].set(xlabel="Epoch", ylabel="Train loss", title="Training loss (fit window 1-24)")
axes[1].set(xlabel="Epoch", ylabel="Validation PR-AUC", title="Validation PR-AUC (window 25-34)")
axes[0].legend(fontsize=7)
axes[1].legend(fontsize=7)
save_fig("learning_curves_v2.png")

# --- Chart 3: edge-time profile of the three graphs ---
fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = graph_diag_projected["graph"].tolist()
intra = graph_diag_projected["intra_step"].to_numpy()
cross = graph_diag_projected["cross_step"].to_numpy()
ax.bar(labels, intra, label="intra-step", color="#4c72b0")
ax.bar(labels, cross, bottom=intra, label="cross-step", color="#c44e52")
ax.set(ylabel="edges", title="Edge-time profile: intra-step vs cross-step")
ax.legend(fontsize=8)
save_fig("graph_edge_profile.png")
""")
md(r"""## 13. Artifact Export & Automated Checks

v2 writes to `graphsage_v2/` only. The frozen Phase 3 `GraphSage/` artifacts are never touched.
`checks.csv` extends the Phase 3 suite with the cross-step assertions: the projected edge list
contains cross-step edges, no edge points backward in time, and the lag features use only strictly
prior wallet snapshots.
""")

code(r"""# 1. Test predictions for every option
prediction_frame = pd.DataFrame(
    {"txId": txs.loc[mask_test, "txId"].values, "time_step": test_step_vals, "y_true": y_test}
)
for key, exp in EXPERIMENTS.items():
    prediction_frame[f"{key}__score"] = exp["test_probs"]
    prediction_frame[f"{key}__predicted"] = (exp["test_probs"] >= exp["threshold"]).astype(int)
save_table("predictions.csv", prediction_frame)

check(
    "predictions: one row per labeled test transaction",
    len(prediction_frame),
    int(mask_test.sum()),
)

# 2. Hyperparameters and full metrics payload
save_json(
    "hyperparameters_v2.json",
    {
        "seed": SEED, "hidden_channels": HIDDEN_CHANNELS, "dropout": DROPOUT,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "temporal_window": TEMPORAL_WINDOW, "max_prior_neighbors": MAX_PRIOR_NEIGHBORS,
        "protocol": {"fit": "1-24", "validation": "25-34", "train": "1-34", "test": "35-49",
                     "test_early": "35-42", "drift": "43-49"},
        "options": {
            key: {
                "label": exp["label"], "graph": exp["graph"], "features": exp["features"],
                "n_features": exp["n_features"], "n_layers": exp["n_layers"], "n_params": exp["n_params"],
                "threshold": exp["threshold"], "best_epoch": exp["best_epoch"],
                "best_val_pr_auc": exp["best_val_pr_auc"], "train_seconds": exp["train_seconds"],
                "notes": exp["notes"],
            }
            for key, exp in EXPERIMENTS.items()
        },
    },
)

metrics_payload = {
    "protocol": {"fit": "1-24", "validation": "25-34", "train": "1-34", "test": "35-49",
                 "test_early": "35-42", "drift": "43-49"},
    "frozen_reference": FROZEN,
    "graph_diagnostics": graph_diag_projected.to_dict(orient="records"),
    "lag_feature_audit": lag_audit.to_dict(orient="records"),
    "experiments": {
        key: {
            "label": exp["label"], "graph": exp["graph"], "features": exp["features"],
            "n_features": exp["n_features"], "n_layers": exp["n_layers"], "n_params": exp["n_params"],
            "threshold": exp["threshold"], "validation_f1": exp["validation_f1"],
            "best_epoch": exp["best_epoch"], "best_val_pr_auc": exp["best_val_pr_auc"],
            "train_seconds": exp["train_seconds"], "windows": exp["windows"],
        }
        for key, exp in EXPERIMENTS.items()
    },
}
save_json("metrics.json", metrics_payload)

# 3. One-page digest
digest_lines = [
    f"BitcoinGraphGuard - Phase 3 v2 GraphSAGE cross-step digest ({time.strftime('%Y-%m-%d %H:%M')})",
    "-" * 88,
    f"Protocol : fit 1-24 | validation 25-34 | refit 1-34 | test 35-49 (35-42 / 43-49 reported)",
    f"Graph    : original {len(original_edges):,} intra-step edges | projected {len(projected_edges):,} "
    f"cross-step edges | combined {len(combined_edges):,} (cross-step {combined_cross_fraction:.1%})",
    f"Lag      : {len(lag_feature_names)} features from strictly-prior address snapshots "
    f"({int(lag_frame['has_prior_address_history'].sum()):,} transactions with history)",
    "",
]
for key, exp in EXPERIMENTS.items():
    w = exp["windows"]
    digest_lines.append(
        f"{exp['label']:<52} PR-AUC 35-49={w['35-49']['pr_auc']:.4f} 35-42={w['35-42']['pr_auc']:.4f} "
        f"43-49={w['43-49']['pr_auc']:.4f} | ROC {w['35-49']['roc_auc']:.4f} | F1 {w['35-49']['f1']:.4f}"
    )
digest_lines += [
    "",
    "Frozen references: XGBoost 0.8013 (35-49) / 0.0427 (43-49); GraphSAGE Phase 3 0.6209 / 0.0504; RGCN 0.4682 / 0.0550",
]
digest = "\n".join(digest_lines)
print(digest)
(OUT_DIR / "graphsage_v2_digest.txt").write_text(digest, encoding="utf-8")

# 4. Persist the extended check suite
save_table("checks.csv", pd.DataFrame(CHECKS))
failures = [row for row in CHECKS if not row["ok"]]
print(f"\nSanity checks: {len(CHECKS)} total, {len(failures)} failures.")
if failures:
    display(pd.DataFrame(failures))
print("Exported artifacts:", sorted(p.name for p in OUT_DIR.iterdir()))
""")

md(r"""## 14. Verdict
""")

code(r'''base = EXPERIMENTS["opt1_baseline"]["windows"]
lag = EXPERIMENTS["opt2_lag_features"]["windows"]
projected = EXPERIMENTS["opt3_projected"]["windows"]
depth = EXPERIMENTS["opt4_depth3"]["windows"]
combined = EXPERIMENTS["opt5_combined"]["windows"]

XGB_FULL, XGB_DRIFT = 0.8013, 0.0427
baseline_full = base["35-49"]["pr_auc"]
gap = XGB_FULL - baseline_full
fraction_closed = (projected["35-49"]["pr_auc"] - baseline_full) / gap if gap else float("nan")


def line(name: str, windows: dict) -> str:
    return (f"{name:<46} 35-49 PR-AUC {windows['35-49']['pr_auc']:.4f} | "
            f"35-42 {windows['35-42']['pr_auc']:.4f} | 43-49 {windows['43-49']['pr_auc']:.4f}")


helped_full = projected["35-49"]["pr_auc"] > baseline_full + 0.01
helped_drift = projected["43-49"]["pr_auc"] > base["43-49"]["pr_auc"] + 0.01
lag_helped_full = lag["35-49"]["pr_auc"] > baseline_full + 0.01
combined_helped_full = combined["35-49"]["pr_auc"] > baseline_full + 0.01

verdict = f"""
### Auto-generated verdict (computed from this run's numbers)

```
{line("O1 baseline (intra-step)", base)}
{line("O2 lag features (intra-step)", lag)}
{line("O3 projected cross-step graph", projected)}
{line("O4 3-layer depth control (intra-step)", depth)}
{line("O5 combined (lag + projected)", combined)}
```

- Reproduced baseline gap to frozen XGBoost on 35-49 ({XGB_FULL:.4f}): **{gap:.4f} PR-AUC**.
- The projected cross-step graph closed **{100 * fraction_closed:.1f}%** of that gap on 35-49.
- Cross-step edges {"**did**" if helped_full else "**did not**"} lift full-test PR-AUC by more than +0.01;
  lag features {"**did**" if lag_helped_full else "**did not**"}; the combined run {"**did**" if combined_helped_full else "**did not**"}.
- On the 43-49 drift window, cross-step edges {"**did**" if helped_drift else "**did not**"} lift PR-AUC by more than +0.01
  over the baseline ({base['43-49']['pr_auc']:.4f} -> {projected['43-49']['pr_auc']:.4f}).
"""
display(Markdown(verdict))
save_json(
    "verdict.json",
    {
        "baseline_pr_auc_35_49": float(baseline_full),
        "gap_to_xgboost_35_49": float(gap),
        "fraction_of_gap_closed_35_49_projected": float(fraction_closed),
        "cross_step_helped_full": bool(helped_full),
        "cross_step_helped_drift_43_49": bool(helped_drift),
        "lag_features_helped_full": bool(lag_helped_full),
        "combined_helped_full": bool(combined_helped_full),
    },
)
''')

md(r"""## 15. Conclusions & Next Steps

### Did giving the model cross-step signal close the gap to XGBoost, and did it help on 43–49?

The auto-generated verdict above answers this from the run's own numbers; the interpretation is
mechanical and must be reported either way:

- **Full test 35–49.** If O2 and O3 sit above the reproduced 0.6209 baseline but still below 0.8013,
  cross-step signal bought real, attributable gain while leaving the tabular ceiling intact — the
  remaining headroom is then about *how* history is summarised, not *whether* it is reachable.
- **Depth is not the answer.** O4 (3 layers, intra-step) is the negative control: a deeper receptive
  field inside a single time step cannot move the number, which is what makes the O2/O3 deltas
  attributable to cross-step information rather than extra capacity.
- **Drift window 43–49.** If O3 and O5 stay at or near the low-prevalence floor there
  (~0.02–0.05 PR-AUC at 2.53% prevalence), then cross-step signal did **not** fix the 43–49 regime.

**A clean negative result is a result.** If cross-step signal alone does not lift 43–49, that removes
the last structural explanation — graph reach — from the Phase 4 narrative and strengthens the
evidence from `docs/XGBOOST_V2.md` that 43–49 is genuine **covariate drift** (transfer failure, not
missing signal), which no intra-homogeneous graph change can repair.

### Carry-forward constraints for the next phase

* Report 35–49, 35–42 **and** 43–49 for every model from here on; an aggregate-only improvement
  would hide the regime that matters.
* The projected graph adds only past → future edges. Any future graph work must preserve the
  invariant $t_{src} \le t_{dst}$, or it leaks.
* The lag features are the strictest reading of "≤ t" (strictly `< t`). Do not relax the cutoff
  without re-deriving the leakage checks.
* Re-run this notebook whenever `TEMPORAL_WINDOW` or `MAX_PRIOR_NEIGHBORS` changes; both are
  first-class hyperparameters, and the checks exist to catch a window that reaches forward.

**Status:** homogeneous cross-step investigation complete; full heterogeneity remains the job of
`notebooks/04_heterogeneous_gnn.ipynb`.
""")


def generate_notebook() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    n_md = sum(1 for cell in CELLS if cell["cell_type"] == "markdown")
    n_code = sum(1 for cell in CELLS if cell["cell_type"] == "code")
    print(f"wrote {OUTPUT} with {len(CELLS)} cells ({n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    generate_notebook()
