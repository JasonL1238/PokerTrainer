# Architecture

Local-first, post-session poker study platform. Streamlit UI, SQLite store,
files on disk for video and frames. No hosted deployment; the container path is
kept working but nothing is deployed.

## Entry points

| Entry point | Kind | Purpose |
|---|---|---|
| `app.py` → `main()` (line 982) | Streamlit | The whole UI. 7 workspaces; see [repository-map.md](repository-map.md#apppy-line-map) |
| `poker_tracker/solver/run_job.py` | subprocess worker | Runs one TexasSolver job; launched by `solver/jobs.py`, own process |
| `python -m poker_tracker.validation` | CLI | Fail-closed Phase 2 corpus / answer-key checks |
| `python -m poker_tracker.release_gate` | CLI | Executable release gate (`--mode fixture\|full\|container`) |
| `python -m poker_tracker.maintenance` | CLI | Audits local DB, artifacts, backups |
| `python -m poker_tracker.perf` | CLI | Perf harness (`run`, `new-baseline`, `compare`) |
| `python -m poker_tracker.suite_quality` | CLI | `skips`, `flake`, `coverage` |

## Major components

`poker_tracker/` — 104 files. Package sizes given because they predict read cost.

| Package | Lines | Responsibility |
|---|---|---|
| `persistence/` | 10.3k | SQLite store (`db.py` alone is 6.9k), models, migrations, import/export, backup |
| `ui/` | 7.4k | Streamlit-facing helpers, view models, reconstruction review, CV job wiring |
| `services/` | 6.3k | Hand accounting, study readiness, validated import, manual spot entry |
| `math/` | 5.2k | Accounting/ledger engine, analytics, equity, ICM, preflop ranges |
| `maintenance/` | 3.3k | Data health, recovery, retention, diagnostics, SBOM |
| `solver/` | 3.0k | TexasSolver adapter, job lifecycle, ranges, eligibility |
| `perf/` | 2.5k | Measurement probes and baselines (leaf — nothing imports it) |
| `release_gate/` | 2.4k | Release-gate runner and model inventory |
| `coaching/` | 1.6k | Prompt building, provider adapters, safety, solver grounding |
| `validation/` | 1.4k | Corpus split, schemas, hashing |
| `suite_quality/` | 1.3k | Skip policy, flake hunt, coverage report (leaf) |
| `safety/`, `runtime/` | 0.3k | Redaction; resource limits |

Outside the package: `cv_lab/` (57 files, 17.8k lines) is CV research tooling —
detector, card classifier, OCR, timeline builders. `tests/` is 196 files.

## Data flow

```
video file ──► ui/video_ingest (PyAV validates container + codec; hard gate)
                     │
                     ▼
            CV job (cv_lab detector + card classifier + template OCR)
                     │  writes a timeline artifact to disk
                     ▼
        services/validated_hand_import ──► SQLite (schema v20)
                     │
                     ▼
        services/hand_accounting  ── reconcile twice: with the declared
                     │               assumption, and with it withdrawn
                     ▼
        services/study_readiness  ── derived, never persisted.
                     │               13 blocker codes; `reviewed` is a
                     │               workflow label, not a verdict
                     ▼
          ┌──────────┴──────────┐
          ▼                     ▼
   solver/ (TexasSolver)   coaching/ (LLM providers)
   post-session only       post-session only, checked at the write boundary
```

Key invariants that shape the code:

- **Study readiness is derived, never stored.** Any edit demotes `reviewed`.
- **Assumption-dependent accounting blocks study until attested.** Reconciliation
  runs twice and the two results must agree, or the operator must confirm.
- **Import strips review status, warning acknowledgements, and attestations** —
  they cannot travel in JSON.
- **Coaching is checked where it is written,** not where it is requested:
  `build_coaching_response` compares the answer to its prompt, and every
  provider calls `ensure_post_session_prompt` before any network call.

## Dependency boundaries

Honest description: the package graph is **interconnected, not layered**. There
are real cycles — `persistence ↔ services ↔ math`, and `ui ↔ services`. Do not
assume a clean layering when planning a change; check the actual imports.

What does hold:

- `perf/` and `suite_quality/` are leaves. Nothing imports them; both are CLI
  entry points. Changes there cannot break the product.
- `safety/` and `runtime/` are near-leaves, imported widely but importing little.
- **`cv_lab` is imported by product code** at 6 sites — `ui/run_cv_job.py`,
  `maintenance/diagnostics.py`, `services/validated_hand_import.py` (×2),
  `perf/probes.py`, and `app.py:535`. Several are function-local imports to keep
  the CV stack optional at startup. There is *no* enforced boundary here, so a
  `cv_lab` signature change can break the product.
- `cv_lab` never imports `poker_tracker`. That direction is clean.
- `ultralytics` (AGPL) is pinned in `deploy/docker/build_python_env.sh`, not in
  any requirements file. Only `cv_lab/` imports it. Base installs and CI do not
  have it; the container does.

## Storage

SQLite for structured data (`SCHEMA_VERSION = 20`). Videos, frames, timelines,
exports and backups are **files**, never blobs in SQL, and live under
operator-owned roots resolved once from `POKER_DB_PATH` and `POKER_DATA_DIR` at
import time. That single-resolution-at-import behaviour is what lets the test
suite redirect them; see `tests/conftest.py`.
