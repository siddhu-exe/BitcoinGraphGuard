# CI/CD - GitHub Actions (Phase 8c)

Validation-only CI for BitcoinGraphGuard. The pipeline exists to protect the two
failure classes this project has repeatedly and specifically caught during
development, **not** to provide a generic "tests pass" checkmark:

1. **Silent miscalibration / wrong-but-plausible output** - the PSI calibration
   bug, the hardcoded `/monitoring/status` response, and the feature-order
   `DMatrix` bug all produced output that looked reasonable and was wrong.
2. **An unverified Docker artifact** - the `Dockerfile` was documented but never
   built or run for two full review cycles.

Two hard constraints shape every workflow:

- **Free-tier GitHub runners only.** No GPU, no raw `Og data/` download (the
  consistency checks score the small committed artifacts under `results/`), and
  no model training of any kind. CI is validation-only, consistent with the
  laptop-vs-Colab compute split in `AGENTS.md`; all training stays off CI.
- **Phase 8b is frozen.** `src/api/`, `src/monitoring/`, the `Dockerfile` and
  `docker-compose.yml` are not modified by this phase.

## Workflow map

| Workflow | Trigger | Gate it provides |
| :--- | :--- | :--- |
| `.github/workflows/test.yml` | push/PR touching `src/**`, `tests/**`, `requirements-serving.txt`, `pytest.ini`, `.github/ruff-ci.toml` | Full 33-test suite + scoped ruff lint/format |
| `.github/workflows/model-consistency-check.yml` | push/PR touching `src/api/**`, `src/monitoring/**`, `results/xgboost*/**`, the per-step metrics CSV, `reports/monitoring_backtest_report.json`, `tests/test_api.py` | `frozen-model-consistency`: three named silent-failure gates |
| `.github/workflows/docker-build.yml` | push to `main` (and manual dispatch) | Build, run, health-poll and benchmark the container image |

All jobs run on `ubuntu-latest` with Python 3.12 and `actions/setup-python` pip
caching. Dependencies come from `requirements-serving.txt`, which already covers
the monitoring engines (`numpy`/`pandas`/`scipy`/`scikit-learn`); there is no
separate `requirements-monitoring.txt`. `pytest` and `ruff` are dev-only tools
and are installed explicitly.

## 1. `test.yml` - suite + scoped lint

Two jobs:

- **`pytest-suite`** runs `pytest tests/ -q` (all 33 tests: 19 API + 14
  monitoring). Any failure fails the build.
- **`lint-scoped`** runs `ruff check` and `ruff format --check`, scoped to
  `src/api/`, `src/monitoring/` and `tests/` only, through
  `.github/ruff-ci.toml` (see below).

### Scoped lint policy (`.github/ruff-ci.toml`)

`ruff check .` reports 393 violations repo-wide. The 138 violations the scoped
gate must not fail on are pre-existing and live **inside** `src/monitoring/`
(133) and `tests/` (5) - not "elsewhere in the repo". Since Phase 8b froze those
files, Phase 8c tolerates that debt per file instead of rewriting it:

- `src/api/` is enforced **fully strictly** - it is clean today.
- The seven legacy modules list their exact pre-existing rule codes
  (`UP006`, `UP007`, `UP035`, `UP045`, `F401`, `F841`, `I001`, `F821`,
  `BLE001`, `RUF022`, `RUF012`, `PIE810`, `F541`, `C414`) under
  `[lint.per-file-ignores]`, plus `RUF022` for `src/monitoring/__init__.py`.
- `[format] exclude` lists the seven files that predate `ruff format`.

Any **new** violation, in any file, still fails the build; the workflow uses
`--force-exclude` so the exclusions keep working if a path is named explicitly.
Remove the entries in the dedicated lint-debt cleanup pass - do not add new ones.

## 2. `model-consistency-check.yml` - the silent-failure gate

One job, `frozen-model-consistency`, deliberately separate from the general
suite so these gates are visible rather than buried in a 33-test summary. Each
check runs as its own step with `continue-on-error`, and a final gate step fails
the job while naming exactly which check failed and the historical bug it guards.
Results are also written to the run's step summary.

| # | Check | Node IDs | Historical bug it prevents |
| :--- | :--- | :--- | :--- |
| 1 | `frozen-model-consistency` - `/predict` for tx `1813992` @ step 35 reproduces the recorded `1.7188735e-05` | `test_predict_known_good_transaction_reproduces_benchmark` | The API was once found returning a hardcoded `/monitoring/status` response instead of real telemetry; a served model must actually reproduce the recorded Phase 2 prediction. |
| 2 | Feature name/order integrity | `test_feature_values_map_by_name_not_json_key_order`, `test_startup_rejects_correct_names_in_wrong_order`, `test_startup_rejects_wrong_feature_names_not_just_count`, `test_misspelled_feature_name_rejected_and_named` | The feature-order `DMatrix` bug: values were mapped by JSON key order rather than canonical name, so swapped/reordered features produced plausible but wrong probabilities. |
| 3 | Metrics fallback visibility | `test_health_fallback_when_metrics_missing_logs_warning`, `test_health_no_fallback_when_metrics_csv_present` | The API once served hardcoded Phase 8a fallback metrics as if they were real telemetry; `using_fallback_metrics` must now track actual CSV presence/absence and log a loud warning. |

## 3. `docker-build.yml` - the Docker artifact is verified

**`docker-build`'s health-poll step exists because this project's `Dockerfile`
went two full review cycles without ever being built or run.** Steps:

1. **Build** the image from the existing (unmodified) `Dockerfile` with
   `docker/build-push-action` and `cache-from`/`cache-to: type=gha`, so repeat
   runs reuse layers instead of rebuilding from scratch.
2. **Record final image size** in the log and step summary. Recorded only - the
   workflow does not fail on size, so drift is visible in workflow history.
3. **Start** the service via `docker compose up -d --no-build`.
4. **Poll `GET /health`** every 2s for a 60s budget and fail if it never returns
   HTTP 200, dumping `docker compose ps` and logs on failure.
5. **Verify the container benchmark**: `.github/scripts/verify_container_prediction.py`
   (standard library only) posts the same tx `1813992` / step 35 request used in
   the manual verification and asserts the probability matches `1.7188735e-05`
   within `rel_tol=1e-3, abs_tol=1e-6` inside the running image - the
   container-level version of the consistency check.
6. **Tear down** with `docker compose down -v --remove-orphans` in an
   `if: always()` step, so a failed health check cannot leave an orphaned
   container on the runner.

## Required checks for merge to `main`

These cannot be configured from the repository - a maintainer must set them in
**Settings → Branches → Branch protection rules** for `main`. Recommended
required status checks:

- `pytest - full suite (33 tests)`
- `ruff - scoped lint and format`
- `frozen-model-consistency`
- `docker build - run - verify`

Recommended: require branches to be up to date before merging and require a pull
request review. Because `docker-build.yml` runs on `main` pushes, its check
appears on the head commit of a pull request; keep it required so an unbuildable
image can never reach `main` again.

## Running the same checks locally

```bash
pytest tests/ -q
ruff check --force-exclude --config .github/ruff-ci.toml src/api src/monitoring tests
ruff format --check --force-exclude --config .github/ruff-ci.toml src/api src/monitoring tests

# Container-level, mirrors docker-build.yml:
docker compose up -d --build
python3 .github/scripts/verify_container_prediction.py
docker compose down -v --remove-orphans
```

## Known limitations

- The frozen-model probability is compared with a tolerance (`rel 1e-3`,
  `abs 1e-6`) so minor float drift from dependency versions does not produce
  false failures; it still catches a wrong model or a wrong feature mapping.
- `ruff` is pinned to the version the scoped config was validated against, so
  lint results do not drift with new ruff releases.
- CI does not download `Og data/` or retrain anything; drift/retraining work
  remains a Colab/Kaggle activity.
