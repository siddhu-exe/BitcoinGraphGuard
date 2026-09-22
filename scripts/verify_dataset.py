#!/usr/bin/env python3
"""Streaming verification of the raw Elliptic++ dataset (Phase 1).

The script is standard-library only and never loads a whole CSV into memory. It
iterates rows with :mod:`csv`, keeps compact aggregates (counters, per-column
tallies and short-lived dedup sets of packed keys), and releases each stage's
working set with ``del`` + :func:`gc.collect` before the next stage starts.

Stages
------
``txs_classes``, ``txs_features``, ``txs_edgelist``, ``wallets_classes``,
``wallets_features``, ``wallets_features_classes_combined``, ``AddrTx_edgelist``,
``TxAddr_edgelist``, ``AddrAddr_edgelist`` and ``cross_reference``.

Usage
-----
::

    python scripts/verify_dataset.py                         # every stage
    python scripts/verify_dataset.py --stage txs_features
    python scripts/verify_dataset.py --json reports/phase1_verification.json
    python scripts/verify_dataset.py --list-stages

Exit code is ``0`` when every requested stage passes its integrity assertions and
``1`` when at least one assertion fails, so the script is CI-friendly.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - resource is POSIX only
    import resource

    def peak_rss_mb() -> Optional[float]:
        """Peak resident set size of this process in MiB, if available."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

except ImportError:  # pragma: no cover

    def peak_rss_mb() -> Optional[float]:
        return None


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "Og data"
OUTPUT_DIR = REPO_ROOT / "reports"

N_TIME_STEPS = 49
TS_COLUMN = "Time step"
KEY_SEPARATOR = "\x1f"  # ASCII unit separator; cannot appear in IDs/addresses


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def format_bytes(size: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, used as an integrity fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def csv_rows(path: Path) -> Iterator[Tuple[List[str], Iterator[List[str]]]]:
    """Open a CSV and yield ``(header, row_iterator)`` without buffering it all."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        yield header, reader


def parse_time_step(value: str) -> Optional[int]:
    """Parse a time-step cell to ``int`` or return ``None`` when malformed."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _pack_key(*parts: str) -> bytes:
    """Pack an edge/row identity into a compact bytes key for exact dedup."""
    return KEY_SEPARATOR.join(parts).encode("utf-8", "surrogatepass")


def _nonzero(mapping: Sequence[int], names: Sequence[str]) -> Dict[str, int]:
    return {names[i]: mapping[i] for i in range(len(names)) if mapping[i]}


def _time_step_report(time_steps: Counter) -> Dict[str, object]:
    return {
        "distribution": {str(k): v for k, v in sorted(time_steps.items())},
        "min": min(time_steps) if time_steps else None,
        "max": max(time_steps) if time_steps else None,
        "count": len(time_steps),
        "contiguous_1_to_49": set(time_steps) == set(range(1, N_TIME_STEPS + 1)),
    }


# ---------------------------------------------------------------------------
# Per-column value audit
# ---------------------------------------------------------------------------
@dataclass
class ColumnAudit:
    """Counts blank, non-numeric and NaN cells per column while streaming."""

    names: List[str]
    blank: List[int] = field(default_factory=list)
    non_numeric: List[int] = field(default_factory=list)
    nan_count: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        width = len(self.names)
        self.blank = [0] * width
        self.non_numeric = [0] * width
        self.nan_count = [0] * width

    def observe(self, value: str, index: int, *, numeric: bool) -> None:
        if value == "":
            self.blank[index] += 1
            return
        if not numeric:
            return
        try:
            parsed = float(value)
        except ValueError:
            self.non_numeric[index] += 1
            return
        if math.isnan(parsed):
            self.nan_count[index] += 1

    def report(self) -> Dict[str, object]:
        return {
            "blank_values": _nonzero(self.blank, self.names),
            "non_numeric_values": _nonzero(self.non_numeric, self.names),
            "nan_values": _nonzero(self.nan_count, self.names),
            "total_blank_cells": sum(self.blank),
            "total_non_numeric_cells": sum(self.non_numeric),
            "total_nan_cells": sum(self.nan_count),
        }


# ---------------------------------------------------------------------------
# Transaction stages
# ---------------------------------------------------------------------------
def verify_txs_features(artifacts: Dict[str, object]) -> Dict[str, object]:
    path = RAW_DATA_DIR / "txs_features.csv"
    with csv_rows(path) as (header, reader):
        if not header:
            raise ValueError(f"{path.name} is empty")
        ncols = len(header)
        tx_idx = header.index("txId")
        ts_idx = header.index(TS_COLUMN)
        feature_idx = [i for i in range(ncols) if i not in (tx_idx, ts_idx)]

        audit = ColumnAudit(header)
        seen: Set[str] = set()
        duplicates = 0
        malformed_rows = 0
        invalid_time_steps = 0
        rows = 0
        time_steps: Counter = Counter()

        for row in reader:
            rows += 1
            if len(row) != ncols:
                malformed_rows += 1
                continue
            tx_id = row[tx_idx]
            if tx_id in seen:
                duplicates += 1
            else:
                seen.add(tx_id)
            ts = parse_time_step(row[ts_idx])
            if ts is None:
                invalid_time_steps += 1
            else:
                time_steps[ts] += 1
            for index in feature_idx:
                audit.observe(row[index], index, numeric=True)

    artifacts["tx_universe"] = seen
    result: Dict[str, object] = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": ncols,
        "column_names": header,
        "unique_ids": len(seen),
        "duplicate_ids": duplicates,
        "malformed_rows": malformed_rows,
        "invalid_time_steps": invalid_time_steps,
        "time_steps": _time_step_report(time_steps),
    }
    result.update(audit.report())
    return result


def verify_txs_classes(artifacts: Dict[str, object]) -> Dict[str, object]:
    path = RAW_DATA_DIR / "txs_classes.csv"
    classes: Counter = Counter()
    seen: Set[str] = set()
    duplicates = 0
    blank_ids = 0
    rows = 0
    with csv_rows(path) as (header, reader):
        tx_idx = header.index("txId")
        cls_idx = header.index("class")
        for row in reader:
            rows += 1
            if len(row) != len(header):
                continue
            tx_id, label = row[tx_idx], row[cls_idx]
            if not tx_id:
                blank_ids += 1
            if tx_id in seen:
                duplicates += 1
            else:
                seen.add(tx_id)
            classes[label] += 1

    artifacts["txs_classes_ids"] = seen
    invalid = {label: count for label, count in classes.items() if label not in {"1", "2", "3"}}
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(header),
        "unique_ids": len(seen),
        "duplicate_ids": duplicates,
        "blank_ids": blank_ids,
        "class_distribution": {k: classes[k] for k in sorted(classes)},
        "invalid_class_values": invalid,
    }


# ---------------------------------------------------------------------------
# Wallet stages
# ---------------------------------------------------------------------------
def _verify_wallet_table(
    path: Path,
    *,
    class_lookup: Optional[Dict[str, str]] = None,
    reference_keys: Optional[Set[bytes]] = None,
    artifacts: Optional[Dict[str, object]] = None,
    universe_key: Optional[str] = None,
    keys_key: Optional[str] = None,
) -> Dict[str, object]:
    """Verify a wallet-level table (features-only or combined features+class)."""
    with csv_rows(path) as (header, reader):
        if not header:
            raise ValueError(f"{path.name} is empty")
        ncols = len(header)
        addr_idx = header.index("address")
        ts_idx = header.index(TS_COLUMN)
        declared_idx = header.index("num_timesteps_appeared_in")
        has_class = "class" in header
        class_idx = header.index("class") if has_class else None
        feature_idx = [
            i for i in range(ncols) if i not in (addr_idx, ts_idx) and i != class_idx
        ]

        audit = ColumnAudit(header)
        address_rows: Dict[str, int] = {}
        declared_steps: Dict[str, int] = {}
        declared_conflicts: Set[str] = set()
        seen_keys: Set[bytes] = set()
        duplicate_pairs = 0
        keys_absent_from_reference = 0
        classes: Counter = Counter()
        class_mismatches = 0
        class_missing = 0
        blank_addresses = 0
        malformed_rows = 0
        invalid_time_steps = 0
        rows = 0
        time_steps: Counter = Counter()

        for row in reader:
            rows += 1
            if len(row) != ncols:
                malformed_rows += 1
                continue
            address = row[addr_idx]
            if not address:
                blank_addresses += 1

            ts = parse_time_step(row[ts_idx])
            if ts is None:
                invalid_time_steps += 1
            else:
                time_steps[ts] += 1
                key = _pack_key(address, str(ts))
                if key in seen_keys:
                    duplicate_pairs += 1
                else:
                    seen_keys.add(key)
                if reference_keys is not None and key not in reference_keys:
                    keys_absent_from_reference += 1

            address_rows[address] = address_rows.get(address, 0) + 1

            try:
                declared = int(float(row[declared_idx]))
            except ValueError:
                declared = -1
            if address in declared_steps:
                if declared_steps[address] != declared:
                    declared_conflicts.add(address)
            else:
                declared_steps[address] = declared

            if class_idx is not None:
                label = row[class_idx]
                classes[label] += 1
                if class_lookup is not None:
                    expected = class_lookup.get(address)
                    if expected is None:
                        class_missing += 1
                    elif expected != label:
                        class_mismatches += 1

            for index in feature_idx:
                audit.observe(row[index], index, numeric=True)

    declared_mismatch = sum(
        1
        for address, count in address_rows.items()
        if address not in declared_conflicts and declared_steps.get(address) != count
    )

    if artifacts is not None:
        if universe_key:
            artifacts[universe_key] = set(address_rows)
        if keys_key:
            artifacts[keys_key] = seen_keys

    result: Dict[str, object] = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": ncols,
        "column_names": header,
        "unique_addresses": len(address_rows),
        "duplicate_address_time_step_pairs": duplicate_pairs,
        "max_rows_for_one_address": max(address_rows.values()) if address_rows else 0,
        "declared_step_conflicts": len(declared_conflicts),
        "rows_per_address_ne_declared_timesteps": declared_mismatch,
        "blank_addresses": blank_addresses,
        "malformed_rows": malformed_rows,
        "invalid_time_steps": invalid_time_steps,
        "time_steps": _time_step_report(time_steps),
    }
    if has_class:
        result["class_distribution"] = {k: classes[k] for k in sorted(classes)}
        result["addresses_missing_from_wallets_classes"] = class_missing
        result["class_mismatches_vs_wallets_classes"] = class_mismatches
    if reference_keys is not None:
        result["time_step_pairs_absent_from_wallets_features"] = keys_absent_from_reference
    result.update(audit.report())
    return result


def verify_wallets_features(artifacts: Dict[str, object]) -> Dict[str, object]:
    return _verify_wallet_table(
        RAW_DATA_DIR / "wallets_features.csv",
        artifacts=artifacts,
        universe_key="wallet_universe",
        keys_key="wallet_keys",
    )


def verify_wallets_classes(artifacts: Dict[str, object]) -> Dict[str, object]:
    path = RAW_DATA_DIR / "wallets_classes.csv"
    classes: Counter = Counter()
    seen: Set[str] = set()
    duplicates = 0
    rows = 0
    with csv_rows(path) as (header, reader):
        addr_idx = header.index("address")
        cls_idx = header.index("class")
        for row in reader:
            rows += 1
            if len(row) != len(header):
                continue
            address, label = row[addr_idx], row[cls_idx]
            if address in seen:
                duplicates += 1
            else:
                seen.add(address)
            classes[label] += 1

    artifacts["wallets_classes_ids"] = seen
    invalid = {label: count for label, count in classes.items() if label not in {"1", "2", "3"}}
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(header),
        "unique_addresses": len(seen),
        "duplicate_addresses": duplicates,
        "class_distribution": {k: classes[k] for k in sorted(classes)},
        "invalid_class_values": invalid,
    }


def verify_wallets_combined(artifacts: Dict[str, object]) -> Dict[str, object]:
    class_lookup: Dict[str, str] = {}
    with csv_rows(RAW_DATA_DIR / "wallets_classes.csv") as (header, reader):
        addr_idx = header.index("address")
        cls_idx = header.index("class")
        for row in reader:
            if len(row) == len(header):
                class_lookup[row[addr_idx]] = row[cls_idx]

    result = _verify_wallet_table(
        RAW_DATA_DIR / "wallets_features_classes_combined.csv",
        class_lookup=class_lookup,
        reference_keys=artifacts.get("wallet_keys"),  # type: ignore[arg-type]
    )
    if "wallet_keys" in artifacts:
        del artifacts["wallet_keys"]
        gc.collect()
    return result


# ---------------------------------------------------------------------------
# Edge stages
# ---------------------------------------------------------------------------
def verify_edgelist(
    filename: str,
    src_column: str,
    dst_column: str,
    *,
    src_universe: Optional[Set[str]] = None,
    dst_universe: Optional[Set[str]] = None,
) -> Dict[str, object]:
    path = RAW_DATA_DIR / filename
    edge_keys: Set[bytes] = set()
    src_set: Set[str] = set()
    dst_set: Set[str] = set()
    duplicates = 0
    self_loops = 0
    blank_endpoints = 0
    malformed_rows = 0
    src_not_in_universe = 0
    dst_not_in_universe = 0
    rows = 0

    with csv_rows(path) as (header, reader):
        if not header:
            raise ValueError(f"{path.name} is empty")
        ncols = len(header)
        src_idx = header.index(src_column)
        dst_idx = header.index(dst_column)

        for row in reader:
            rows += 1
            if len(row) != ncols:
                malformed_rows += 1
                continue
            src, dst = row[src_idx], row[dst_idx]
            if not src or not dst:
                blank_endpoints += 1
            key = _pack_key(src, dst)
            if key in edge_keys:
                duplicates += 1
            else:
                edge_keys.add(key)
            if src == dst:
                self_loops += 1
            src_set.add(src)
            dst_set.add(dst)
            if src_universe is not None and src not in src_universe:
                src_not_in_universe += 1
            if dst_universe is not None and dst not in dst_universe:
                dst_not_in_universe += 1

    result: Dict[str, object] = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": ncols,
        "column_names": header,
        "total_edges": rows - malformed_rows,
        "unique_edges": len(edge_keys),
        "duplicate_edges": duplicates,
        "self_loops": self_loops,
        "blank_endpoints": blank_endpoints,
        "malformed_rows": malformed_rows,
        "unique_sources": len(src_set),
        "unique_destinations": len(dst_set),
        "unique_nodes": len(src_set | dst_set),
        "source_nodes_not_in_universe": src_not_in_universe,
        "destination_nodes_not_in_universe": dst_not_in_universe,
        "src_universe_checked": src_universe is not None,
        "dst_universe_checked": dst_universe is not None,
    }
    del edge_keys
    result["_src_set"] = src_set
    result["_dst_set"] = dst_set
    return result


def _public(result: Dict[str, object]) -> Dict[str, object]:
    """Drop private working-set entries (prefixed with ``_``) before serialising."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


STAGE_ORDER = [
    "txs_classes",
    "txs_features",
    "txs_edgelist",
    "wallets_classes",
    "wallets_features",
    "wallets_features_classes_combined",
    "AddrTx_edgelist",
    "TxAddr_edgelist",
    "AddrAddr_edgelist",
    "cross_reference",
]


def integrity_failures(stage: str, result: Dict[str, object]) -> List[str]:
    """Return structural integrity failures for one stage's result."""
    failures: List[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(f"{stage}: {message}")

    require(result.get("malformed_rows", 0) == 0, "malformed rows present")
    require(result.get("total_non_numeric_cells", 0) == 0, "non-numeric feature cells present")
    require(result.get("total_nan_cells", 0) == 0, "NaN feature cells present")

    time_steps = result.get("time_steps")
    if isinstance(time_steps, dict) and time_steps.get("count"):
        require(bool(time_steps.get("contiguous_1_to_49")), "time steps are not contiguous 1..49")
        require(result.get("invalid_time_steps", 0) == 0, "invalid time-step values present")

    if stage in {"txs_features", "txs_classes"}:
        require(result.get("duplicate_ids", 0) == 0, "duplicate txIds present")
        require(result.get("blank_ids", 0) == 0, "blank txIds present")
    if stage == "wallets_classes":
        require(result.get("duplicate_addresses", 0) == 0, "duplicate addresses present")
    if stage in {"wallets_features", "wallets_features_classes_combined"}:
        require(result.get("blank_addresses", 0) == 0, "blank addresses present")
        require(
            result.get("declared_step_conflicts", 0) == 0,
            "num_timesteps_appeared_in conflicts within one address",
        )
    if stage == "wallets_features_classes_combined":
        for key in (
            "addresses_missing_from_wallets_classes",
            "class_mismatches_vs_wallets_classes",
            "time_step_pairs_absent_from_wallets_features",
        ):
            require(result.get(key, 0) == 0, f"{key} != 0")

    if "total_edges" in result:
        require(result.get("blank_endpoints", 0) == 0, "blank edge endpoints present")
        require(
            result.get("source_nodes_not_in_universe", 0) == 0,
            "source nodes outside the node universe",
        )
        require(
            result.get("destination_nodes_not_in_universe", 0) == 0,
            "destination nodes outside the node universe",
        )

    if stage == "cross_reference":
        for key in (
            "txs_edge_nodes_not_in_features",
            "txs_features_ids_missing_from_classes",
            "txs_classes_ids_missing_from_features",
            "wallets_features_addresses_missing_from_classes",
            "wallets_classes_addresses_missing_from_features",
            "edge_addresses_not_in_wallets_features",
            "addrtx_tx_minus_txaddr_tx",
            "txaddr_tx_minus_addrtx_tx",
        ):
            require(result.get(key, 0) == 0, f"{key} != 0")

    return failures


def run_pipeline(stages: Sequence[str], *, compute_hash: bool = False) -> Dict[str, object]:
    artifacts: Dict[str, object] = {}
    results: Dict[str, object] = {"generated_at": datetime.now(timezone.utc).isoformat()}
    failures: List[str] = []
    quality: Dict[str, object] = {}
    hash_cache: Dict[str, str] = {}

    def emit(name: str, result: Dict[str, object]) -> None:
        filename = result.get("file")
        if compute_hash and isinstance(filename, str):
            if filename not in hash_cache:
                print(f"     hashing {filename} ...", flush=True)
                hash_cache[filename] = sha256_file(RAW_DATA_DIR / filename)
            result["file_sha256"] = hash_cache[filename]
        results[name] = _public(result)
        print(f"[ok] {name}", flush=True)
        for key in (
            "rows",
            "total_edges",
            "unique_edges",
            "duplicate_edges",
            "self_loops",
            "unique_ids",
            "unique_addresses",
            "columns",
        ):
            if key in result:
                print(f"     {key}: {result[key]:,}", flush=True)
        peak = peak_rss_mb()
        if peak is not None:
            print(f"     peak_rss_mb: {peak:.1f}", flush=True)

    for stage in stages:
        t0 = time.time()
        if stage == "txs_classes":
            result = verify_txs_classes(artifacts)
        elif stage == "txs_features":
            result = verify_txs_features(artifacts)
        elif stage == "wallets_classes":
            result = verify_wallets_classes(artifacts)
        elif stage == "wallets_features":
            result = verify_wallets_features(artifacts)
        elif stage == "wallets_features_classes_combined":
            result = verify_wallets_combined(artifacts)
        elif stage in {"txs_edgelist", "AddrTx_edgelist", "TxAddr_edgelist", "AddrAddr_edgelist"}:
            spec = {
                "txs_edgelist": ("txId1", "txId2", "tx_universe", "tx_universe"),
                "AddrTx_edgelist": ("input_address", "txId", "wallet_universe", "tx_universe"),
                "TxAddr_edgelist": ("txId", "output_address", "tx_universe", "wallet_universe"),
                "AddrAddr_edgelist": (
                    "input_address",
                    "output_address",
                    "wallet_universe",
                    "wallet_universe",
                ),
            }[stage]
            src_col, dst_col, src_key, dst_key = spec
            result = verify_edgelist(
                f"{stage}.csv",
                src_col,
                dst_col,
                src_universe=artifacts.get(src_key),  # type: ignore[arg-type]
                dst_universe=artifacts.get(dst_key),  # type: ignore[arg-type]
            )
            artifacts[f"{stage}:src"] = result.pop("_src_set")
            artifacts[f"{stage}:dst"] = result.pop("_dst_set")
        elif stage == "cross_reference":
            result = cross_reference(artifacts)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unknown stage: {stage}")

        emit(stage, result)
        failures += integrity_failures(stage, result)
        finding: Dict[str, object] = {}
        if result.get("total_blank_cells"):
            finding["blank_cells"] = result["total_blank_cells"]
            finding["blank_columns"] = result.get("blank_values", {})
        if result.get("total_nan_cells"):
            finding["nan_cells"] = result["total_nan_cells"]
            finding["nan_columns"] = result.get("nan_values", {})
        if result.get("duplicate_address_time_step_pairs"):
            finding["duplicate_address_time_step_pairs"] = result["duplicate_address_time_step_pairs"]
        if result.get("rows_per_address_ne_declared_timesteps"):
            finding["rows_per_address_ne_declared_timesteps"] = result[
                "rows_per_address_ne_declared_timesteps"
            ]
        if finding:
            quality[stage] = finding
        print(f"     stage_seconds: {time.time() - t0:.1f}", flush=True)
        gc.collect()

    results["failures"] = failures
    results["quality_findings"] = quality
    results["passed"] = not failures
    results["peak_rss_mb"] = peak_rss_mb()
    return results


def cross_reference(artifacts: Dict[str, object]) -> Dict[str, object]:
    """Confirm ID relationships across node tables and edge lists."""
    if "tx_universe" not in artifacts or "wallet_universe" not in artifacts:
        return {
            "error": (
                "cross_reference requires the txs_features and wallets_features stages "
                "to have run in the same invocation"
            )
        }

    tx_universe: Set[str] = artifacts["tx_universe"]  # type: ignore[assignment]
    wallet_universe: Set[str] = artifacts["wallet_universe"]  # type: ignore[assignment]
    tx_classes: Set[str] = artifacts.get("txs_classes_ids", set())  # type: ignore[assignment]
    wallet_classes: Set[str] = artifacts.get("wallets_classes_ids", set())  # type: ignore[assignment]

    def as_set(key: str) -> Set[str]:
        return artifacts.get(key, set())  # type: ignore[return-value]

    addrtx_src = as_set("AddrTx_edgelist:src")
    addrtx_dst = as_set("AddrTx_edgelist:dst")
    txaddr_src = as_set("TxAddr_edgelist:src")
    txaddr_dst = as_set("TxAddr_edgelist:dst")
    addraddr_src = as_set("AddrAddr_edgelist:src")
    addraddr_dst = as_set("AddrAddr_edgelist:dst")
    txs_edge_nodes = as_set("txs_edgelist:src") | as_set("txs_edgelist:dst")

    edge_addresses = addrtx_src | txaddr_dst | addraddr_src | addraddr_dst
    linked_transactions = addrtx_dst | txaddr_src

    return {
        "txs_features_id_count": len(tx_universe),
        "txs_classes_id_count": len(tx_classes),
        "txs_features_ids_in_txs_classes": len(tx_universe & tx_classes),
        "txs_features_ids_missing_from_classes": len(tx_universe - tx_classes),
        "txs_classes_ids_missing_from_features": len(tx_classes - tx_universe),
        "txs_edge_nodes": len(txs_edge_nodes),
        "txs_edge_nodes_in_features": len(txs_edge_nodes & tx_universe),
        "txs_edge_nodes_not_in_features": len(txs_edge_nodes - tx_universe),
        "txs_features_not_in_txs_edgelist": len(tx_universe - txs_edge_nodes),
        "wallets_features_address_count": len(wallet_universe),
        "wallets_classes_address_count": len(wallet_classes),
        "wallets_features_addresses_in_wallets_classes": len(wallet_universe & wallet_classes),
        "wallets_features_addresses_missing_from_classes": len(wallet_universe - wallet_classes),
        "wallets_classes_addresses_missing_from_features": len(wallet_classes - wallet_universe),
        "transactions_in_addrtx": len(addrtx_dst),
        "transactions_in_txaddr": len(txaddr_src),
        "transactions_with_address_links": len(linked_transactions),
        "transactions_without_address_links": len(tx_universe - linked_transactions),
        "addrtx_tx_minus_txaddr_tx": len(addrtx_dst - txaddr_src),
        "txaddr_tx_minus_addrtx_tx": len(txaddr_src - addrtx_dst),
        "edge_addresses_total": len(edge_addresses),
        "edge_addresses_in_wallets_features": len(edge_addresses & wallet_universe),
        "edge_addresses_not_in_wallets_features": len(edge_addresses - wallet_universe),
        "wallets_features_addresses_with_no_edge": len(wallet_universe - edge_addresses),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def list_files() -> None:
    if not RAW_DATA_DIR.exists():
        print(f"Raw data directory not found: {RAW_DATA_DIR}")
        return
    print(f"Raw data: {RAW_DATA_DIR}")
    for path in sorted(RAW_DATA_DIR.iterdir()):
        if path.is_file():
            print(f"  - {path.name} ({format_bytes(path.stat().st_size)})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGE_ORDER,
        help="run only the given stage (repeatable); default runs every stage",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full result as JSON")
    parser.add_argument(
        "--hash",
        action="store_true",
        help="also compute a SHA-256 fingerprint per file (extra full read; off by default)",
    )
    parser.add_argument(
        "--low-priority",
        action="store_true",
        help="run at the lowest CPU scheduling priority (os.nice(19))",
    )
    parser.add_argument("--list-stages", action="store_true", help="list stages and exit")
    parser.add_argument("--list-files", action="store_true", help="list raw data files and exit")
    args = parser.parse_args(argv)

    if args.list_stages:
        for stage in STAGE_ORDER:
            print(stage)
        return 0
    if args.list_files:
        list_files()
        return 0

    if args.low_priority:
        try:
            os.nice(19)
        except (AttributeError, PermissionError):  # pragma: no cover - platform dependent
            pass

    stages = args.stage if args.stage else STAGE_ORDER
    print(f"Verifying Elliptic++ dataset in {RAW_DATA_DIR}", flush=True)
    started = time.time()
    results = run_pipeline(stages, compute_hash=args.hash)
    results["elapsed_seconds"] = time.time() - started

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")

    print(f"\nElapsed: {results['elapsed_seconds']:.1f}s | peak RSS: {results.get('peak_rss_mb')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
