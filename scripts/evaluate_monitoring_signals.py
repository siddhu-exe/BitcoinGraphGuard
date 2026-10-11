"""
Evaluate the Phase 8a monitoring signals and export the signal-evaluation artifacts.

Reads the artifacts already exported by ``scripts/run_monitoring_backtest.py``:

* ``results/monitoring/step_monitoring_metrics.csv`` - one row per step 35-49 with a
  status per channel; and
* ``results/xgboost/predictions.csv`` - committed frozen-model scores used to evaluate
  the label-free score channels (KS, predicted-positive rate).

Writes ``results/monitoring/signal_evaluation.csv`` and
``reports/monitoring_signal_evaluation.md``. Deterministic, CPU-only, no training.

Usage:
    python scripts/evaluate_monitoring_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.signal_evaluation import (  # noqa: E402
    format_signal_evaluation_markdown,
    load_and_evaluate,
)


def main() -> None:
    table = load_and_evaluate()

    out_csv = Path("results/monitoring/signal_evaluation.csv")
    out_md = Path("reports/monitoring_signal_evaluation.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(out_csv, index=False)
    out_md.write_text(format_signal_evaluation_markdown(table) + "\n", encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nWrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
