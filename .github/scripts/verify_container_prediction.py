#!/usr/bin/env python3
"""Verify the containerized service against the frozen-model benchmark.

Posts the tx 1813992 / step 35 benchmark request (the exact payload defined in
tests/test_api.py) to a running BitcoinGraphGuard container, then asserts the
returned probability matches the recorded frozen-model value 1.7188735e-05
within tolerance, and that the classification contract is unchanged.

Standard library only: the host runner does not need serving dependencies to run
this container-level consistency check.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BENCHMARK_TX_ID = "1813992"
BENCHMARK_TIME_STEP = 35
EXPECTED_PROBABILITY = 1.7188735e-05
EXPECTED_FEATURE_COUNT = 165
REL_TOL = 1e-3
ABS_TOL = 1e-6
FIXTURE_PATH = Path("tests/test_api.py")
FIXTURE_NAME = "SAMPLE_TX_1813992_FEATURES"


def load_benchmark_features(fixture_path: Path) -> dict[str, float]:
    """Extract the canonical benchmark feature dict from the test fixture."""
    tree = ast.parse(fixture_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if FIXTURE_NAME in names:
            features = ast.literal_eval(node.value)
            if len(features) != EXPECTED_FEATURE_COUNT:
                raise SystemExit(
                    f"Expected {EXPECTED_FEATURE_COUNT} benchmark features, "
                    f"found {len(features)} in {fixture_path}"
                )
            return features
    raise SystemExit(f"Could not locate {FIXTURE_NAME} in {fixture_path}")


def request_prediction(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST the payload to /predict and return the decoded JSON body."""
    url = f"{base_url.rstrip('/')}/predict"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise SystemExit(f"POST {url} returned HTTP {response.status}")
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"::error::Could not reach {url}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Run the container benchmark consistency check; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    features = load_benchmark_features(FIXTURE_PATH)
    payload = {
        "tx_id": BENCHMARK_TX_ID,
        "time_step": BENCHMARK_TIME_STEP,
        **features,
    }
    body = request_prediction(args.base_url, payload, args.timeout)

    probability = body["probability"]
    print(f"container /predict probability = {probability!r} (expected {EXPECTED_PROBABILITY!r})")
    print(
        f"label={body.get('label')} threshold={body.get('threshold')} "
        f"binary_classification={body.get('binary_classification')}"
    )

    if not math.isclose(probability, EXPECTED_PROBABILITY, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        print(
            "::error::CONTAINER CONSISTENCY FAILURE: the running image scored the "
            f"frozen-model benchmark at {probability!r}, outside {ABS_TOL} / {REL_TOL} "
            f"of the recorded {EXPECTED_PROBABILITY!r}. The image is not serving the "
            "frozen XGBoost Optimized artifact (wrong model, wrong feature order, or "
            "a stale build).",
            file=sys.stderr,
        )
        return 1

    if body.get("binary_classification") != 0 or body.get("label") != "licit":
        print(
            "::error::The benchmark transaction was not classified as licit - the "
            f"model or threshold contract changed (got label={body.get('label')!r}).",
            file=sys.stderr,
        )
        return 1

    print("Container benchmark check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
