#!/usr/bin/env python3
"""Streaming, low-memory EDA for the raw Elliptic++ dataset (Phase 2).

This script performs the exploratory analysis that follows Phase 1 verification:
temporal activity, class imbalance, graph structure and feature distributions. It
never loads a whole CSV into memory: every stage reads the raw files with
``chunksize`` and keeps compact numpy accumulators plus one bounded systematic
sample used for quantiles and correlations.

The raw dataset is opened read-only and is never modified. Outputs are written to
``reports/eda/`` (JSON summary + CSV tables) so the same numbers can be inspected
and re-used without re-running the heavy passes.

Design notes
------------
* ``txs_features.csv`` and ``wallets_features.csv`` are read **once each**; a
  single pass produces both the per-time-step label activity and the per-feature
  distribution statistics.
* Wallet rows are de-duplicated on ``(address, Time step)`` on the fly (Phase 1
  found 347,569 exact duplicate rows), so wallet activity and feature statistics
  describe distinct snapshots rather than raw rows.
* Class labels are joined from the small ``*_classes.csv`` files. Class ``3``
  (unknown) is tracked separately and is never folded into licit.
* Graph stages stream the edge lists and use integer index arrays plus
  ``scipy.sparse`` weak-connectivity; no edge list is held as Python objects.

Usage
-----
::

    ./bit/bin/python scripts/eda_phase2.py                       # every stage
    ./bit/bin/python scripts/eda_phase2.py --stage txs
    ./bit/bin/python scripts/eda_phase2.py --stage txs --limit-chunks 3   # smoke run
    ./bit/bin/python scripts/eda_phase2.py --list-stages

Run it at low priority on the laptop::

    MPLCONFIGDIR=/tmp/mplconfig nice -n 19 ./bit/bin/python -u scripts/eda_phase2.py \
        --low-priority --json reports/eda/phase2_eda.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is expected but optional
    HAVE_SCIPY = False


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "Og data"
OUTPUT_DIR = REPO_ROOT / "reports" / "eda"
TABLE_DIR = OUTPUT_DIR / "tables"

N_TIME_STEPS = 49
STEP_COLUMN = "Time step"
TX_ID = "txId"
ADDRESS = "address"

# Raw row counts verified in Phase 1; used only to size the systematic sample.
EXPECTED_TXS_ROWS = 203_769
EXPECTED_WALLET_SNAPSHOTS = 920_691
SAMPLE_CAP = 25_000

LABEL_ILLICIT = 1
LABEL_LICIT = 2
LABEL_UNKNOWN = 3

# Candidate temporal windows investigated (not a finalized split).
TRAIN_END = 34
CANDIDATE_WINDOWS: List[Tuple[str, int, int]] = [
    ("train_1_34", 1, 34),
    ("val_35_42", 35, 42),
    ("test_43_49", 43, 49),
    ("test_35_49", 35, 49),
    ("val_35_40", 35, 40),
    ("test_41_49", 41, 49),
]

STAGE_ORDER = [
    "labels",
    "txs",
    "wallets",
    "graph_txs",
    "graph_bipartite",
    "graph_addr",
    "split",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def peak_rss_mb() -> Optional[float]:
    """Peak resident set size of this process in MiB, if available."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:  # pragma: no cover - POSIX only
        return None


def read_header(path: Path) -> List[str]:
    """Return a CSV header without reading any data rows."""
    return list(pd.read_csv(path, nrows=0).columns)


def as_python(obj: object) -> object:
    """Recursively convert numpy scalars/arrays to JSON-serialisable Python."""
    if isinstance(obj, dict):
        return {str(k): as_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return as_python(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _bincount(values: np.ndarray) -> np.ndarray:
    """Histogram of integer positions in ``1..N_TIME_STEPS`` (index 0 unused)."""
    return np.bincount(values, minlength=N_TIME_STEPS + 1).astype(np.int64)


def _label_arrays(steps: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-time-step counts split by label, computed with vectorised bincounts.

    ``labeled`` counts only classes 1/2; class 3 (unknown) is reported separately
    and is never folded into the labeled total.
    """
    illicit = _bincount(steps[labels == LABEL_ILLICIT])
    licit = _bincount(steps[labels == LABEL_LICIT])
    unknown = _bincount(steps[labels == LABEL_UNKNOWN])
    return {
        "total": _bincount(steps),
        "labeled": illicit + licit,
        "illicit": illicit,
        "licit": licit,
        "unknown": unknown,
    }


def _merge_step_counts(dest: Dict[str, np.ndarray], source: Dict[str, np.ndarray]) -> None:
    for key, values in source.items():
        dest[key] = dest.get(key, np.zeros(N_TIME_STEPS + 1, dtype=np.int64)) + values


def _step_rows(counts: Dict[str, np.ndarray], unit: str) -> List[Dict[str, object]]:
    """Convert per-time-step counter arrays into a JSON/CSV friendly table."""
    rows: List[Dict[str, object]] = []
    for step in range(1, N_TIME_STEPS + 1):
        total = int(counts["total"][step])
        labeled = int(counts["labeled"][step])
        illicit = int(counts["illicit"][step])
        rows.append(
            {
                "time_step": step,
                "unit": unit,
                "total": total,
                "labeled": labeled,
                "illicit": illicit,
                "licit": int(counts["licit"][step]),
                "unknown": int(counts["unknown"][step]),
                "illicit_pct_of_labeled": (100.0 * illicit / labeled) if labeled else None,
                "labeled_pct_of_total": (100.0 * labeled / total) if total else None,
            }
        )
    return rows


def _degree_summary(degrees: np.ndarray, unit: str) -> Dict[str, object]:
    """Summarise a degree vector (histogram + central-tendency + top buckets)."""
    degrees = degrees.astype(np.int64)
    positive = degrees[degrees > 0]
    histogram = Counter(int(d) for d in degrees)
    top = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    return {
        "unit": unit,
        "nodes": int(degrees.size),
        "nodes_with_degree_zero": int((degrees == 0).sum()),
        "mean": float(degrees.mean()) if degrees.size else None,
        "median": float(np.median(degrees)) if degrees.size else None,
        "p95": float(np.percentile(degrees, 95)) if degrees.size else None,
        "p99": float(np.percentile(degrees, 99)) if degrees.size else None,
        "max": int(degrees.max()) if degrees.size else None,
        "mean_positive": float(positive.mean()) if positive.size else None,
        "max_positive": int(positive.max()) if positive.size else None,
        "degree_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        "top_degrees": [{"degree": d, "count": c} for d, c in top],
    }


def _components(node_count: int, sources: np.ndarray, targets: np.ndarray) -> Dict[str, object]:
    """Weakly-connected component statistics over an undirected view of edges."""
    if not HAVE_SCIPY:
        return {"available": False, "reason": "scipy not installed"}
    if sources.size == 0:
        return {"available": True, "components": int(node_count), "largest_component_size": 1}
    rows = np.concatenate([sources, targets])
    cols = np.concatenate([targets, sources])
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(node_count, node_count),
    )
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return {
        "available": True,
        "components": int(n_components),
        "largest_component_size": int(sizes.max()),
        "largest_component_pct": float(100.0 * sizes.max() / node_count),
        "singleton_components": int((sizes == 1).sum()),
    }


class FeatureAccumulator:
    """Compact, memory-bounded per-column statistics for one node type.

    Full-stream accumulators cover count/missing/mean/std/min/max/zero/negative,
    illicit-vs-licit conditional moments and early-vs-late means. A bounded
    systematic sample adds quantiles, skewness and pairwise correlations.
    """

    def __init__(self, columns: Sequence[str], sample_stride: int) -> None:
        self.columns = list(columns)
        n = len(self.columns)
        self.n = np.zeros(n)
        self.s = np.zeros(n)
        self.ss = np.zeros(n)
        self.mn = np.full(n, np.inf)
        self.mx = np.full(n, -np.inf)
        self.zeros = np.zeros(n)
        self.negs = np.zeros(n)
        self.missing = np.zeros(n)
        self.cond_n = np.zeros((2, n))
        self.cond_s = np.zeros((2, n))
        self.cond_ss = np.zeros((2, n))
        self.early_n = np.zeros(n)
        self.early_s = np.zeros(n)
        self.late_n = np.zeros(n)
        self.late_s = np.zeros(n)
        self.sample_stride = max(1, int(sample_stride))
        self._row_counter = 0
        self._samples: List[np.ndarray] = []
        self._sample_labels: List[np.ndarray] = []
        self.rows = 0

    def update(self, frame: pd.DataFrame, labels: np.ndarray, steps: np.ndarray) -> None:
        values = frame[self.columns].to_numpy(dtype="float64")
        mask = ~np.isnan(values)
        filled = np.where(mask, values, 0.0)
        self.n += mask.sum(axis=0)
        self.s += filled.sum(axis=0)
        self.ss += (filled * filled).sum(axis=0)
        self.zeros += ((filled == 0.0) & mask).sum(axis=0)
        self.negs += ((filled < 0.0) & mask).sum(axis=0)
        self.missing += (~mask).sum(axis=0)
        self.mn = np.minimum(self.mn, np.where(mask, values, np.inf).min(axis=0))
        self.mx = np.maximum(self.mx, np.where(mask, values, -np.inf).max(axis=0))

        for index, label in enumerate((LABEL_ILLICIT, LABEL_LICIT)):
            rows = labels == label
            if rows.any():
                sub = values[rows]
                sub_mask = ~np.isnan(sub)
                sub_filled = np.where(sub_mask, sub, 0.0)
                self.cond_n[index] += sub_mask.sum(axis=0)
                self.cond_s[index] += sub_filled.sum(axis=0)
                self.cond_ss[index] += (sub_filled * sub_filled).sum(axis=0)

        early = steps <= TRAIN_END
        late = ~early
        if early.any():
            self.early_n += mask[early].sum(axis=0)
            self.early_s += np.where(mask[early], values[early], 0.0).sum(axis=0)
        if late.any():
            self.late_n += mask[late].sum(axis=0)
            self.late_s += np.where(mask[late], values[late], 0.0).sum(axis=0)

        row_indices = np.arange(self._row_counter, self._row_counter + len(values))
        selected = row_indices % self.sample_stride == 0
        if selected.any():
            self._samples.append(values[selected])
            self._sample_labels.append(labels[selected])
        self._row_counter += len(values)
        self.rows += len(values)

    def sample_frame(self) -> pd.DataFrame:
        if not self._samples:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(np.vstack(self._samples), columns=self.columns)

    def summary_rows(self) -> List[Dict[str, object]]:
        sample = self.sample_frame()
        rows: List[Dict[str, object]] = []
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.n > 0, self.s / np.maximum(self.n, 1), np.nan)
            var = np.where(self.n > 0, self.ss / np.maximum(self.n, 1) - mean * mean, np.nan)
        for index, column in enumerate(self.columns):
            n = self.n[index]
            finite = (
                sample[column].to_numpy(dtype="float64")
                if not sample.empty
                else np.array([])
            )
            finite = finite[np.isfinite(finite)]
            ok = finite.size > 0
            row: Dict[str, object] = {
                "feature": column,
                "count": int(n),
                "missing": int(self.missing[index]),
                "mean": float(mean[index]) if n > 0 else None,
                "std": float(math.sqrt(max(var[index], 0.0))) if n > 0 else None,
                "min": float(self.mn[index]) if n > 0 else None,
                "max": float(self.mx[index]) if n > 0 else None,
                "p01": float(np.percentile(finite, 1)) if ok else None,
                "p25": float(np.percentile(finite, 25)) if ok else None,
                "median": float(np.percentile(finite, 50)) if ok else None,
                "p75": float(np.percentile(finite, 75)) if ok else None,
                "p99": float(np.percentile(finite, 99)) if ok else None,
                "skew": float(pd.Series(finite).skew()) if ok and finite.size > 2 else None,
                "zero_fraction": float(self.zeros[index] / n) if n > 0 else None,
                "negative_count": int(self.negs[index]),
                "constant": bool(n > 0 and self.mn[index] == self.mx[index]),
                "near_constant": bool(
                    n > 0
                    and (self.mn[index] == self.mx[index] or self.zeros[index] / n >= 0.99)
                ),
                "mean_early_1_34": (
                    float(self.early_s[index] / self.early_n[index])
                    if self.early_n[index] > 0
                    else None
                ),
                "mean_late_35_49": (
                    float(self.late_s[index] / self.late_n[index])
                    if self.late_n[index] > 0
                    else None
                ),
            }
            row["corr_with_illicit"] = self._point_biserial(index)
            rows.append(row)
        return rows

    def _point_biserial(self, index: int) -> Optional[float]:
        n1, n2 = self.cond_n[0, index], self.cond_n[1, index]
        if n1 < 2 or n2 < 2:
            return None
        mean1 = self.cond_s[0, index] / n1
        mean2 = self.cond_s[1, index] / n2
        var1 = max(self.cond_ss[0, index] / n1 - mean1 * mean1, 0.0)
        var2 = max(self.cond_ss[1, index] / n2 - mean2 * mean2, 0.0)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        if pooled_var <= 0:
            return None
        return float((mean1 - mean2) / math.sqrt(pooled_var) * math.sqrt(n1 * n2) / (n1 + n2))

    def correlation_matrix(self) -> Optional[pd.DataFrame]:
        sample = self.sample_frame()
        if sample.shape[1] < 2 or sample.shape[0] < 3:
            return None
        return sample.corr(numeric_only=True)

    def reset_samples(self) -> None:
        self._samples.clear()
        self._sample_labels.clear()
        gc.collect()


def _iter_chunks(
    path: Path,
    chunk_size: int,
    usecols: Optional[Sequence[str]] = None,
    limit_chunks: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=list(usecols) if usecols else None,
        low_memory=True,
    )
    for index, chunk in enumerate(reader):
        if limit_chunks is not None and index >= limit_chunks:
            break
        yield chunk


def _write_table(name: str, rows: Sequence[Dict[str, object]]) -> str:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return str(path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_labels(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Class balance for transactions and wallets, and labeled/unknown coverage."""
    txs = pd.read_csv(RAW_DATA_DIR / "txs_classes.csv")
    wallets = pd.read_csv(RAW_DATA_DIR / "wallets_classes.csv")

    def distribution(frame: pd.DataFrame, key: str) -> Dict[str, object]:
        counts = frame["class"].value_counts().sort_index()
        total = int(len(frame))
        labeled = int(counts.get(LABEL_ILLICIT, 0) + counts.get(LABEL_LICIT, 0))
        illicit = int(counts.get(LABEL_ILLICIT, 0))
        licit = int(counts.get(LABEL_LICIT, 0))
        return {
            "rows": total,
            "unique_ids": int(frame[key].nunique()),
            "duplicate_ids": int(total - frame[key].nunique()),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "labeled": labeled,
            "labeled_pct": round(100.0 * labeled / total, 4),
            "unknown_pct": round(100.0 * int(counts.get(LABEL_UNKNOWN, 0)) / total, 4),
            "illicit_over_licit": round(illicit / licit, 4) if licit else None,
            "illicit_pct_of_all": round(100.0 * illicit / total, 4),
            "illicit_pct_of_labeled": round(100.0 * illicit / labeled, 4) if labeled else None,
        }

    result["labels"] = {
        "transactions": distribution(txs, TX_ID),
        "wallets": distribution(wallets, ADDRESS),
    }
    del txs, wallets
    gc.collect()


def _prepare_class_map(path: Path, key: str) -> Dict[object, int]:
    frame = pd.read_csv(path)
    return dict(zip(frame[key].to_numpy(), frame["class"].to_numpy()))


def stage_txs(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Single pass over ``txs_features.csv``: temporal activity + feature stats."""
    path = RAW_DATA_DIR / "txs_features.csv"
    header = read_header(path)
    feature_columns = [c for c in header if c not in (TX_ID, STEP_COLUMN)]
    class_map = _prepare_class_map(RAW_DATA_DIR / "txs_classes.csv", TX_ID)
    stride = max(1, EXPECTED_TXS_ROWS // SAMPLE_CAP)
    accumulator = FeatureAccumulator(feature_columns, sample_stride=stride)
    counts: Dict[str, np.ndarray] = {}
    rows = 0
    for chunk in _iter_chunks(path, args.chunk_size, limit_chunks=args.limit_chunks):
        steps = chunk[STEP_COLUMN].to_numpy(dtype=np.int64)
        labels = chunk[TX_ID].map(class_map).to_numpy(dtype="float64", na_value=np.nan)
        _merge_step_counts(counts, _label_arrays(steps, labels))
        accumulator.update(chunk, labels, steps)
        rows += len(chunk)
        del chunk
    temporal = _step_rows(counts, "transactions")
    result["temporal"] = result.get("temporal", {})
    result["temporal"]["transactions"] = {"rows": rows, "by_time_step": temporal}
    summary = accumulator.summary_rows()
    result["features"] = result.get("features", {})
    result["features"]["transactions"] = {
        "columns": feature_columns,
        "sample_stride": stride,
        "summary": summary,
    }
    result.setdefault("tables", []).append(_write_table("txs_temporal_by_step.csv", temporal))
    result["tables"].append(_write_table("txs_feature_summary.csv", summary))
    corr = accumulator.correlation_matrix()
    if corr is not None:
        corr.to_csv(TABLE_DIR / "txs_feature_correlation.csv")
        result["tables"].append("reports/eda/tables/txs_feature_correlation.csv")
    accumulator.reset_samples()
    del accumulator, counts, class_map, summary
    gc.collect()


def stage_wallets(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Single pass over ``wallets_features.csv`` with ``(address, step)`` dedup."""
    path = RAW_DATA_DIR / "wallets_features.csv"
    header = read_header(path)
    feature_columns = [c for c in header if c not in (ADDRESS, STEP_COLUMN)]
    class_map = _prepare_class_map(RAW_DATA_DIR / "wallets_classes.csv", ADDRESS)
    stride = max(1, EXPECTED_WALLET_SNAPSHOTS // SAMPLE_CAP)
    accumulator = FeatureAccumulator(feature_columns, sample_stride=stride)
    counts: Dict[str, np.ndarray] = {}
    seen_keys: set = set()
    seen_addresses: set = set()
    rows = 0
    distinct = 0
    duplicates = 0
    for chunk in _iter_chunks(path, args.chunk_size, limit_chunks=args.limit_chunks):
        keys = chunk[ADDRESS].astype(str) + "\x1f" + chunk[STEP_COLUMN].astype(str)
        rows += len(chunk)
        # De-duplicate against all previous chunks *and* within the chunk itself:
        # verified duplicates are usually adjacent, so a cross-chunk-only check
        # misses ~347k of them.
        new_keys = keys[~keys.isin(seen_keys)]
        first_in_chunk = ~new_keys.duplicated()
        keys_to_add = new_keys[first_in_chunk]
        if not keys_to_add.empty:
            unique = chunk.loc[keys_to_add.index].copy()
            seen_keys.update(keys_to_add.to_numpy())
            unique_steps = unique[STEP_COLUMN].to_numpy(dtype=np.int64)
            unique_labels = unique[ADDRESS].map(class_map).to_numpy(
                dtype="float64", na_value=np.nan
            )
            _merge_step_counts(counts, _label_arrays(unique_steps, unique_labels))
            seen_addresses.update(unique[ADDRESS].astype(str).to_numpy())
            accumulator.update(unique, unique_labels, unique_steps)
            distinct += len(unique)
            del unique
        duplicates += len(chunk) - int(keys_to_add.size)
        del chunk, keys
    temporal = _step_rows(counts, "wallet_snapshots")
    summary = accumulator.summary_rows()
    result["temporal"] = result.get("temporal", {})
    result["temporal"]["wallets"] = {
        "raw_rows": rows,
        "distinct_address_time_step": distinct,
        "duplicate_rows": duplicates,
        "distinct_addresses": len(seen_addresses),
        "columns": feature_columns,
        "sample_stride": stride,
        "by_time_step": temporal,
    }
    result["features"] = result.get("features", {})
    result["features"]["wallets"] = {
        "columns": feature_columns,
        "sample_stride": stride,
        "summary": summary,
    }
    result.setdefault("tables", []).append(_write_table("wallets_temporal_by_step.csv", temporal))
    result["tables"].append(_write_table("wallets_feature_summary.csv", summary))
    corr = accumulator.correlation_matrix()
    if corr is not None:
        corr.to_csv(TABLE_DIR / "wallets_feature_correlation.csv")
        result["tables"].append("reports/eda/tables/wallets_feature_correlation.csv")
    accumulator.reset_samples()
    del accumulator, counts, class_map, seen_keys, seen_addresses, summary
    gc.collect()


def _graph_degree_block(
    name: str,
    path: Path,
    source_column: str,
    target_column: str,
    universe_sources: np.ndarray,
    universe_targets: np.ndarray,
    chunk_size: int,
    limit_chunks: Optional[int],
) -> Dict[str, object]:
    """Stream an edge list and summarise degrees, multiplicity and components."""
    # Node ids can be numeric (txId) or string (address); normalise to str so the
    # two universes can be combined and looked up in one index.
    all_ids = np.unique(
        np.concatenate([universe_sources.astype(str), universe_targets.astype(str)])
    )
    index = pd.Index(all_ids)
    node_count = int(all_ids.size)
    source_index: List[np.ndarray] = []
    target_index: List[np.ndarray] = []
    edges = 0
    sources_outside = 0
    targets_outside = 0
    for chunk in _iter_chunks(path, chunk_size, usecols=[source_column, target_column], limit_chunks=limit_chunks):
        sources = chunk[source_column].astype(str).to_numpy()
        targets = chunk[target_column].astype(str).to_numpy()
        source_positions = index.get_indexer(sources)
        target_positions = index.get_indexer(targets)
        sources_outside += int((source_positions < 0).sum())
        targets_outside += int((target_positions < 0).sum())
        source_index.append(np.where(source_positions < 0, 0, source_positions).astype(np.int64))
        target_index.append(np.where(target_positions < 0, 0, target_positions).astype(np.int64))
        edges += len(chunk)
        del chunk, sources, targets, source_positions, target_positions
    src = np.concatenate(source_index) if source_index else np.array([], dtype=np.int64)
    dst = np.concatenate(target_index) if target_index else np.array([], dtype=np.int64)
    out_degree = np.bincount(src, minlength=node_count) if src.size else np.zeros(node_count, dtype=np.int64)
    in_degree = np.bincount(dst, minlength=node_count) if dst.size else np.zeros(node_count, dtype=np.int64)
    total_degree = out_degree + in_degree
    if src.size:
        keys = src * node_count + dst
        unique_keys = np.unique(keys)
        unique_pairs = int(unique_keys.size)
        self_loops = int((src == dst).sum())
        reciprocal = int(np.isin(dst * node_count + src, unique_keys).sum())
    else:
        unique_pairs = 0
        self_loops = 0
        reciprocal = 0
    top_indices = np.argsort(total_degree)[::-1][:10] if node_count else np.array([], dtype=int)
    return {
        "file": path.name,
        "edges": int(edges),
        "unique_pairs": unique_pairs,
        "repeated_pairs": int(edges - unique_pairs),
        "self_loops": self_loops,
        "reciprocal_pairs": reciprocal,
        "unique_source_nodes": int(np.unique(src).size) if src.size else 0,
        "unique_destination_nodes": int(np.unique(dst).size) if dst.size else 0,
        "node_universe_size": node_count,
        "source_nodes_outside_universe": sources_outside,
        "target_nodes_outside_universe": targets_outside,
        "isolated_universe_nodes": int((total_degree == 0).sum()),
        "out_degree": _degree_summary(out_degree, "out"),
        "in_degree": _degree_summary(in_degree, "in"),
        "total_degree": _degree_summary(total_degree, "total"),
        "top_nodes_by_total_degree": [
            {
                "node": str(all_ids[i]),
                "in_degree": int(in_degree[i]),
                "out_degree": int(out_degree[i]),
            }
            for i in top_indices
        ],
        "components": _components(node_count, src.astype(np.int32), dst.astype(np.int32)),
    }


def stage_graph_txs(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Transaction->transaction edge graph: degrees and weak components."""
    universe = pd.read_csv(RAW_DATA_DIR / "txs_classes.csv", usecols=[TX_ID])[TX_ID].to_numpy()
    result["graph"] = result.get("graph", {})
    result["graph"]["txs_edgelist"] = _graph_degree_block(
        "txs_edgelist",
        RAW_DATA_DIR / "txs_edgelist.csv",
        "txId1",
        "txId2",
        universe,
        universe,
        args.chunk_size,
        args.limit_chunks,
    )
    del universe
    gc.collect()


def stage_graph_bipartite(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Address<->transaction bipartite edges plus combined weak components."""
    wallets = pd.read_csv(RAW_DATA_DIR / "wallets_classes.csv", usecols=[ADDRESS])[ADDRESS].to_numpy()
    txs = pd.read_csv(RAW_DATA_DIR / "txs_classes.csv", usecols=[TX_ID])[TX_ID].to_numpy()
    result["graph"] = result.get("graph", {})
    result["graph"]["AddrTx_edgelist"] = _graph_degree_block(
        "AddrTx_edgelist",
        RAW_DATA_DIR / "AddrTx_edgelist.csv",
        "input_address",
        TX_ID,
        wallets,
        txs,
        args.chunk_size,
        args.limit_chunks,
    )
    result["graph"]["TxAddr_edgelist"] = _graph_degree_block(
        "TxAddr_edgelist",
        RAW_DATA_DIR / "TxAddr_edgelist.csv",
        TX_ID,
        "output_address",
        txs,
        wallets,
        args.chunk_size,
        args.limit_chunks,
    )
    # Combined undirected graph over tx (0..T-1) and wallets (T..T+W-1).
    all_nodes = np.concatenate([txs.astype(str), wallets.astype(str)])
    node_index = pd.Index(all_nodes)
    sources: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for path, source_column, target_column in (
        (RAW_DATA_DIR / "AddrTx_edgelist.csv", "input_address", TX_ID),
        (RAW_DATA_DIR / "TxAddr_edgelist.csv", TX_ID, "output_address"),
    ):
        for chunk in _iter_chunks(path, args.chunk_size, usecols=[source_column, target_column], limit_chunks=args.limit_chunks):
            sources.append(node_index.get_indexer(chunk[source_column].astype(str).to_numpy()).astype(np.int32))
            targets.append(node_index.get_indexer(chunk[target_column].astype(str).to_numpy()).astype(np.int32))
            del chunk
    src = np.concatenate(sources)
    dst = np.concatenate(targets)
    result["graph"]["combined_address_transaction_components"] = _components(len(all_nodes), src, dst)
    del wallets, txs, all_nodes, node_index, sources, targets, src, dst
    gc.collect()


def stage_graph_addr(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Address->address graph: multi-edges, self-loops, degrees, components."""
    wallets = pd.read_csv(RAW_DATA_DIR / "wallets_classes.csv", usecols=[ADDRESS])[ADDRESS].to_numpy()
    block = _graph_degree_block(
        "AddrAddr_edgelist",
        RAW_DATA_DIR / "AddrAddr_edgelist.csv",
        "input_address",
        "output_address",
        wallets,
        wallets,
        args.chunk_size,
        args.limit_chunks,
    )
    result["graph"] = result.get("graph", {})
    result["graph"]["AddrAddr_edgelist"] = block
    del wallets
    gc.collect()


def stage_split(args: argparse.Namespace, result: Dict[str, object]) -> None:
    """Summarise label availability over candidate temporal windows (not final)."""
    temporal = result.get("temporal", {})
    windows: Dict[str, object] = {}
    for unit_key, unit in (("transactions", "transactions"), ("wallets", "wallet_snapshots")):
        block = temporal.get(unit_key)
        if not block:
            continue
        by_step = {row["time_step"]: row for row in block["by_time_step"]}
        for name, start, end in CANDIDATE_WINDOWS:
            total = labeled = illicit = licit = unknown = 0
            for step in range(start, end + 1):
                row = by_step[step]
                total += row["total"]
                labeled += row["labeled"]
                illicit += row["illicit"]
                licit += row["licit"]
                unknown += row["unknown"]
            windows.setdefault(unit, {})[name] = {
                "steps": f"{start}-{end}",
                "total": total,
                "labeled": labeled,
                "illicit": illicit,
                "licit": licit,
                "unknown": unknown,
                "illicit_pct_of_labeled": round(100.0 * illicit / labeled, 4) if labeled else None,
            }
    result["split_investigation"] = {
        "note": "Investigation only; no split is finalized in Phase 2.",
        "proposed_train_end": TRAIN_END,
        "candidate_windows": windows,
    }


STAGE_FUNCTIONS = {
    "labels": stage_labels,
    "txs": stage_txs,
    "wallets": stage_wallets,
    "graph_txs": stage_graph_txs,
    "graph_bipartite": stage_graph_bipartite,
    "graph_addr": stage_graph_addr,
    "split": stage_split,
}


def run_pipeline(
    stages: Sequence[str],
    args: argparse.Namespace,
    initial: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages_requested": list(stages),
        "chunk_size": args.chunk_size,
        "limit_chunks": args.limit_chunks,
    }
    if initial:
        result.update(initial)
    result["stages_requested"] = list(stages)
    result["chunk_size"] = args.chunk_size
    result["limit_chunks"] = args.limit_chunks
    for name in stages:
        started = time.time()
        print(f"[stage] {name} ...", flush=True)
        STAGE_FUNCTIONS[name](args, result)
        print(f"[stage] {name} done in {time.time() - started:.1f}s", flush=True)
        # Checkpoint after every stage so a long run cannot lose completed work.
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(as_python(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="append", choices=STAGE_ORDER, help="run only this stage (repeatable)")
    parser.add_argument("--json", type=Path, default=OUTPUT_DIR / "phase2_eda.json", help="JSON output path")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="rows per pandas chunk")
    parser.add_argument("--limit-chunks", type=int, default=None, help="stop each chunked file after N chunks (smoke test)")
    parser.add_argument("--low-priority", action="store_true", help="run at the lowest CPU scheduling priority")
    parser.add_argument("--list-stages", action="store_true", help="list stages and exit")
    parser.add_argument("--resume", action="store_true", help="merge results into an existing --json file (skip stages already there)")
    args = parser.parse_args(argv)

    if args.list_stages:
        for stage in STAGE_ORDER:
            print(stage)
        return 0
    if args.low_priority:
        try:
            os.nice(19)
        except (AttributeError, PermissionError):  # pragma: no cover - platform dependent
            pass

    stages = args.stage if args.stage else STAGE_ORDER
    initial = None
    if args.resume and args.json.exists():
        initial = json.loads(args.json.read_text(encoding="utf-8"))
        print(f"Resuming from {args.json} (existing keys: {sorted(initial)})", flush=True)
    started = time.time()
    result = run_pipeline(stages, args, initial=initial)
    result["elapsed_seconds"] = time.time() - started
    result["peak_rss_mb"] = peak_rss_mb()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(as_python(result), indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nJSON written to {args.json}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s | peak RSS: {result['peak_rss_mb']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
