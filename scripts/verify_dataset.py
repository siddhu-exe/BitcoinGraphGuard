#!/usr/bin/env python3
"""
Elliptic++ Dataset Verification & Inspection Script.
Uses standard library only (streaming/buffered) to ensure zero dependencies and low memory footprint.
"""

import os
import csv
import sys
import time
import gc
from collections import Counter, defaultdict

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Og data")

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def analyze_file_basic(filepath):
    """Counts lines, columns, header names, file size."""
    size = os.path.getsize(filepath)
    line_count = 0
    headers = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            line_count = 1
        except StopIteration:
            return size, 0, []

        for _ in reader:
            line_count += 1

    row_count = line_count - 1 if line_count > 0 else 0
    return size, row_count, headers

def inspect_txs_classes():
    filepath = os.path.join(RAW_DATA_DIR, "txs_classes.csv")
    print(f"\n--- Inspecting {os.path.basename(filepath)} ---")
    size = os.path.getsize(filepath)
    classes = Counter()
    unique_txs = set()
    dup_tx_count = 0
    missing_vals = 0

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tx = row['txId']
            cls = row['class']
            if not tx or not cls:
                missing_vals += 1
            if tx in unique_txs:
                dup_tx_count += 1
            else:
                unique_txs.add(tx)
            classes[cls] += 1

    print(f"Total rows: {len(unique_txs) + dup_tx_count}")
    print(f"Unique txIds: {len(unique_txs)}")
    print(f"Duplicate txIds: {dup_tx_count}")
    print(f"Class distribution: {dict(classes)}")
    print(f"Missing values: {missing_vals}")
    return {"unique_txs": unique_txs, "classes": classes, "dup_txs": dup_tx_count}

def inspect_txs_features():
    filepath = os.path.join(RAW_DATA_DIR, "txs_features.csv")
    print(f"\n--- Inspecting {os.path.basename(filepath)} ---")
    size = os.path.getsize(filepath)
    unique_txs = set()
    dup_tx_count = 0
    timesteps = Counter()
    missing_count = 0
    col_count = 0

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_count = len(headers)

        # indices
        tx_idx = headers.index("txId")
        ts_idx = headers.index("Time step")

        row_idx = 0
        for row in reader:
            row_idx += 1
            if len(row) != col_count:
                missing_count += 1
            tx = row[tx_idx]
            ts = row[ts_idx]
            if tx in unique_txs:
                dup_tx_count += 1
            else:
                unique_txs.add(tx)
            timesteps[int(ts)] += 1

    print(f"Total rows: {row_idx}")
    print(f"Total columns: {col_count}")
    print(f"Unique txIds: {len(unique_txs)}")
    print(f"Duplicate txIds: {dup_tx_count}")
    print(f"Time steps range: min={min(timesteps.keys())}, max={max(timesteps.keys())}, total_timesteps={len(timesteps)}")
    print(f"Rows with column mismatch/missing: {missing_count}")
    return {"unique_txs": unique_txs, "timesteps": timesteps, "headers": headers}

def inspect_wallets_classes():
    filepath = os.path.join(RAW_DATA_DIR, "wallets_classes.csv")
    print(f"\n--- Inspecting {os.path.basename(filepath)} ---")
    classes = Counter()
    unique_addrs = set()
    dup_addr_count = 0
    missing_vals = 0

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row['address']
            cls = row['class']
            if not addr or not cls:
                missing_vals += 1
            if addr in unique_addrs:
                dup_addr_count += 1
            else:
                unique_addrs.add(addr)
            classes[cls] += 1

    print(f"Total rows: {len(unique_addrs) + dup_addr_count}")
    print(f"Unique addresses: {len(unique_addrs)}")
    print(f"Duplicate addresses: {dup_addr_count}")
    print(f"Class distribution: {dict(classes)}")
    print(f"Missing values: {missing_vals}")
    return {"unique_addrs": unique_addrs, "classes": classes, "dup_addrs": dup_addr_count}

def inspect_wallets_features():
    filepath = os.path.join(RAW_DATA_DIR, "wallets_features.csv")
    print(f"\n--- Inspecting {os.path.basename(filepath)} ---")
    unique_addrs = set()
    dup_addr_count = 0
    timesteps = Counter()
    missing_count = 0

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_count = len(headers)
        addr_idx = headers.index("address")
        ts_idx = headers.index("Time step")

        row_idx = 0
        for row in reader:
            row_idx += 1
            if len(row) != col_count:
                missing_count += 1
            addr = row[addr_idx]
            ts = row[ts_idx]
            if addr in unique_addrs:
                dup_addr_count += 1
            else:
                unique_addrs.add(addr)
            try:
                timesteps[int(float(ts))] += 1
            except ValueError:
                pass

    print(f"Total rows: {row_idx}")
    print(f"Total columns: {col_count}")
    print(f"Unique addresses: {len(unique_addrs)}")
    print(f"Duplicate addresses (across time or rows): {dup_addr_count}")
    print(f"Time steps range: min={min(timesteps.keys()) if timesteps else 'N/A'}, max={max(timesteps.keys()) if timesteps else 'N/A'}, total_timesteps={len(timesteps)}")
    print(f"Rows with column mismatch: {missing_count}")
    return {"unique_addrs": unique_addrs, "timesteps": timesteps, "headers": headers}

def inspect_wallets_combined():
    filepath = os.path.join(RAW_DATA_DIR, "wallets_features_classes_combined.csv")
    print(f"\n--- Inspecting {os.path.basename(filepath)} ---")
    unique_addrs = set()
    classes = Counter()
    timesteps = Counter()
    row_count = 0

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_count = len(headers)
        addr_idx = headers.index("address")
        ts_idx = headers.index("Time step")
        cls_idx = headers.index("class")

        for row in reader:
            row_count += 1
            unique_addrs.add(row[addr_idx])
            classes[row[cls_idx]] += 1
            try:
                timesteps[int(float(row[ts_idx]))] += 1
            except ValueError:
                pass

    print(f"Total rows: {row_count}")
    print(f"Total columns: {col_count}")
    print(f"Unique addresses: {len(unique_addrs)}")
    print(f"Class distribution: {dict(classes)}")
    return {"headers": headers, "row_count": row_count, "col_count": col_count}

def inspect_edgelist(filename, src_col, dst_col):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    print(f"\n--- Inspecting Edge List: {filename} ---")

    edge_count = 0
    src_set = set()
    dst_set = set()
    self_loops = 0
    duplicate_edges = 0
    # Deduplicate on the raw line text (one short string per edge) instead of a
    # (u, v) tuple of freshly-built str objects. This keeps peak RAM bounded
    # for the 2.8M-edge AddrAddr_edgelist.csv on a low-memory machine.
    seen_lines = set()

    with open(filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
        headers = next(csv.reader([f.readline()]))
        src_idx = headers.index(src_col)
        dst_idx = headers.index(dst_col)

        for line in f:
            key = line.rstrip('\r\n')
            if not key:
                continue
            edge_count += 1
            if key in seen_lines:
                duplicate_edges += 1
            else:
                seen_lines.add(key)
            fields = key.split(',')
            u, v = fields[src_idx], fields[dst_idx]
            if u == v:
                self_loops += 1
            src_set.add(u)
            dst_set.add(v)

    unique_edges = len(seen_lines)
    del seen_lines
    gc.collect()

    print(f"Total edges: {edge_count}")
    print(f"Unique edges: {unique_edges}")
    print(f"Duplicate edges: {duplicate_edges}")
    print(f"Self-loops: {self_loops}")
    print(f"Unique sources: {len(src_set)}")
    print(f"Unique destinations: {len(dst_set)}")
    print(f"Total unique nodes in edge list: {len(src_set | dst_set)}")
    return {
        "filename": filename,
        "edge_count": edge_count,
        "unique_edges": unique_edges,
        "duplicate_edges": duplicate_edges,
        "self_loops": self_loops,
        "src_set": src_set,
        "dst_set": dst_set,
        "headers": headers
    }

if __name__ == "__main__":
    print("Starting Comprehensive Elliptic++ Dataset Verification...")
    t0 = time.time()

    # Verify files
    files = sorted(os.listdir(RAW_DATA_DIR))
    print(f"\nDiscovered {len(files)} files in '{RAW_DATA_DIR}':")
    for f in files:
        fpath = os.path.join(RAW_DATA_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  - {f} ({format_bytes(size)})")

    tx_cls = inspect_txs_classes()
    tx_feat = inspect_txs_features()
    w_cls = inspect_wallets_classes()
    w_feat = inspect_wallets_features()
    w_comb = inspect_wallets_combined()

    el_tx = inspect_edgelist("txs_edgelist.csv", "txId1", "txId2")
    el_addrtx = inspect_edgelist("AddrTx_edgelist.csv", "input_address", "txId")
    el_txaddr = inspect_edgelist("TxAddr_edgelist.csv", "txId", "output_address")
    el_addraddr = inspect_edgelist("AddrAddr_edgelist.csv", "input_address", "output_address")

    # Cross-referencing entities
    print("\n--- Cross-Referencing Entity Alignment ---")
    # Txs
    print(f"txs_classes txIds vs txs_features txIds match: {tx_cls['unique_txs'] == tx_feat['unique_txs']}")
    print(f"Missing in features: {len(tx_cls['unique_txs'] - tx_feat['unique_txs'])}")
    print(f"Missing in classes: {len(tx_feat['unique_txs'] - tx_cls['unique_txs'])}")

    # Txs in edge lists
    tx_in_edges = el_tx['src_set'] | el_tx['dst_set']
    print(f"Transactions in txs_edgelist: {len(tx_in_edges)}")
    print(f"txs_edgelist txIds in txs_features: {len(tx_in_edges.intersection(tx_feat['unique_txs']))} / {len(tx_in_edges)}")

    # Wallets
    print(f"wallets_classes addresses vs wallets_features addresses match: {w_cls['unique_addrs'] == w_feat['unique_addrs']}")
    print(f"Missing in wallet features: {len(w_cls['unique_addrs'] - w_feat['unique_addrs'])}")
    print(f"Missing in wallet classes: {len(w_feat['unique_addrs'] - w_cls['unique_addrs'])}")

    # Heterogeneous alignment
    print(f"AddrTx txIds in txs_features: {len(el_addrtx['dst_set'].intersection(tx_feat['unique_txs']))} / {len(el_addrtx['dst_set'])}")
    print(f"TxAddr txIds in txs_features: {len(el_txaddr['src_set'].intersection(tx_feat['unique_txs']))} / {len(el_txaddr['src_set'])}")
    print(f"AddrTx addresses in wallets_features: {len(el_addrtx['src_set'].intersection(w_feat['unique_addrs']))} / {len(el_addrtx['src_set'])}")
    print(f"TxAddr addresses in wallets_features: {len(el_txaddr['dst_set'].intersection(w_feat['unique_addrs']))} / {len(el_txaddr['dst_set'])}")
    print(f"AddrAddr inputs in wallets_features: {len(el_addraddr['src_set'].intersection(w_feat['unique_addrs']))} / {len(el_addraddr['src_set'])}")
    print(f"AddrAddr outputs in wallets_features: {len(el_addraddr['dst_set'].intersection(w_feat['unique_addrs']))} / {len(el_addraddr['dst_set'])}")

    print(f"\nVerification finished in {time.time() - t0:.2f} seconds.")
