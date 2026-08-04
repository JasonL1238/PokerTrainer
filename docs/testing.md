# Testing

3823 tests across 196 files in `tests/`, plus `deploy/tests`. A full run is
~3.5 minutes. Run the cheapest check that can fail first.

## The ladder

```bash
# 1. the single test you affected
pytest tests/test_study_readiness.py::test_name

# 2. its file
pytest tests/test_study_readiness.py -q

# 3. the package you touched (by test file, there are no test packages)
pytest tests/test_accounting_ledger.py tests/test_accounting_properties.py -q

# 4. lint + types, scoped
ruff check poker_tracker/services
python -m mypy            # typed subset only; file list is in pyproject.toml

# 5. full validation, last
pytest -q
```

`pytest` config lives in `pyproject.toml`. `testpaths` is `["tests", "deploy/tests"]`,
so a bare `pytest` collects the container-contract tests too. `addopts = "-ra"`
prints every skip reason, because the exit gate forbids unexplained skips.

## Environment

**Nothing to set up for the default suite.** `tests/conftest.py` redirects
`POKER_DB_PATH` and `POKER_DATA_DIR` into a temp sandbox at import time, before
the first `poker_tracker` import. Two consequences:

- It will **refuse to run** with `RuntimeError` if `poker_tracker` was imported
  first — which is what loading a plugin from inside the package with
  `-p poker_tracker.<x>` does. Use `-p sq_random_order` (the root-level shim).
- Never point tests at the real `poker_tracker.db`. The suite once applied an
  irreversible migration to the operator's live database; that is what the
  redirect exists to prevent.

Optional capabilities, and what needs them:

| Capability | Needed for | How |
|---|---|---|
| TexasSolver binary | solver integration tests | `export TEXAS_SOLVER_PATH=…/console_solver`, `TEXAS_SOLVER_RESOURCE_DIR=…/resources` |
| CV extra (`torch`, `ultralytics`, `opencv`) | `cv_lab` pipeline tests, SBOM AGPL check | `pip install -r requirements-cv.txt`, plus `ultralytics` pinned only in `deploy/docker/build_python_env.sh` |
| Model weights (`*.pt`) | CV inference, model-inventory tests | **Not in any checkout** — `.gitignore` excludes `*.pt`. Provision via `deploy/provision_models.py --source …` |
| `ANTHROPIC_API_KEY` | live coaching provider | export it; tests use stubs |
| `ffmpeg` | some video paths | install separately; PyAV covers ingest validation |

Tests that need an absent capability should skip with a reason that names the
condition — `python -m poker_tracker.suite_quality skips tests` audits this and
currently reports 25 declarations, 23 explained, 0 unexplained.

## Known failures

**`tests/test_sbom.py` — 4 tests fail on any machine without the CV stack and
provisioned weights, including CI.** These are pre-existing and unrelated to
product logic. Do not "fix" them by chasing the product code.

```
test_the_agpl_dependency_the_cv_pipeline_needs_is_flagged   assert 'ultralytics' in set()
test_fail_on_review_exits_nonzero                           assert 0 == 1
test_models_are_inventoried_with_hashes                     assert []
test_cyclonedx_is_valid_json_with_the_expected_shape        'machine-learning-model' not in {'library'}
```

Cause: the SBOM reads *installed distributions* and *on-disk weights*. CI installs
only `requirements-dev.txt`, `ultralytics` is pinned in a shell script rather than
a requirements file, and `*.pt` is gitignored so no checkout ever has weights.
Reporting no AGPL component on such a machine is *correct* — the assertions
encode a provisioned developer machine. The fix (rewrite against synthetic
fixtures, and reformulate the AGPL check to read the pin rather than the
environment) is tracked but not done.

## CI

Three jobs, all on Python 3.11 / ubuntu-latest, all installing
`requirements-dev.txt` only:

| Job | Runs | Gates on |
|---|---|---|
| Quality and tests | `ruff`, `mypy`, `pytest` | test failures |
| Core-module coverage | `coverage run -m pytest` | nothing — report only, no `fail_under` |
| Repeat determinism | `suite_quality flake --passes 3` | **flakiness only** |

The determinism job exits 0 even with consistently failing tests — it prints
`CONSISTENTLY FAILING -- broken, not flaky:` and still returns success, because
it gates on `report.stable`. A green determinism job is not evidence the suite
passes.

## Writing tests

- `tests/` has its own `AGENTS.md`. Read it — some files are frozen records.
- Assert the invariant, not the symptom. When you delete code because it is
  unreachable, add the test that pins *why* it was unreachable.
- Skips need a reason naming the condition, or the skip audit fails.
