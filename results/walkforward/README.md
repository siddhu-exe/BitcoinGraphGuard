# Walk-forward retraining artifacts (Phase 9a)

Narrative and verdict: `verdict.md`, `digest.txt`. Generator: `scripts/generate_notebook_09.py`
(`notebooks/09_walkforward_retraining.ipynb`).

## Known error: window-level ceilings in `alert_budget.csv`

**Do not use `max_possible_precision` or `max_possible_recall` on the `level == "window"` rows of
`alert_budget.csv`.** They are wrong. The per-step rows (`level == "step"`) are correct.

The window rows apply the per-step formula to the window totals, i.e.
`min(1, total_positives / total_flagged)` and `min(1, total_flagged / total_positives)`. That lets
surplus alert capacity at a step with few positives offset a step with many positives, so the
recall ceiling is overstated and the precision ceiling is overstated or reported as 1.0.

The correct window ceilings take the minimum **per step**, then sum:

```
max_possible_recall    = sum_over_steps( min(K, positives_s) ) / sum_over_steps( positives_s )
max_possible_precision = sum_over_steps( min(K, positives_s) ) / sum_over_steps( K )
```

(`K` alerts per step, `positives_s` = illicit labels at step `s`; use the steps in the window.)

Worked example, steps 43-49 (positives 24, 24, 5, 2, 22, 36, 56), K = 20:
`sum(min(20, p)) = 20+20+5+2+20+20+20 = 107`, so recall ceiling = 107 / 169 = **0.633** and precision
ceiling = 107 / 140 = **0.764**. `alert_budget.csv` shows 0.828 and 1.000. For K = 50 the correct values
are 163 / 169 = **0.964** and 163 / 350 = **0.466** (file: 1.000 and 0.483).

`operating_rules.csv` (`max_possible_recall`) and the tables in `verdict.md` / `digest.txt` were computed
per step and already agree with the formula above. Only `alert_budget.csv` window rows are affected,
and nothing in the verdict reads them. The generator (`generate_notebook_09.py`, the `window_rows`
block) still has the old formula and should be fixed the next time the notebook is re-run on Colab.
