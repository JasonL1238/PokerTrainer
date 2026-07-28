# PokerTrainer

PokerTrainer is a local-first, post-session poker study and review workspace. It organizes completed sessions and hands, runs offline video reconstruction, provides poker math and coaching tools, and keeps source confidence visible throughout review.

It never provides real-time poker assistance, live table capture, poker-client overlays, or current-hand recommendations.

The current implementation status, release gates, and remaining work live in
[PLAN.md](PLAN.md).

## Product workspace

The Streamlit application is organized around seven workflows:

- **Overview** — portfolio metrics, recent sessions, and processing jobs.
- **Sessions** — session summaries and manual hand entry.
- **Hands** — searchable cross-session hand library.
- **Study** — hand replay, recorded math, optional TexasSolver analysis,
  auditable correction, coaching reruns, notes, and review state.
- **Insights** — evidence-backed review coverage and tagged study themes.
- **Import** — completed-session video upload and offline CV reconstruction.
- **Settings** — ROI calibration, data transfer, math tools, and coaching configuration.

CV, equity, solver, and coaching output remain separately labeled by source and
confidence. The application does not turn approximate inputs into a universal
GTO score.

## Local setup

Requirements:

- Python 3.11+
- SQLite
- FFmpeg for video workflows
- Optional TexasSolver `console_solver` for solver-backed postflop review

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The base requirements support the review application. The cloud CV container additionally installs the pinned dependencies in `requirements-cv.txt`.

By default, structured data is stored in `poker_tracker.db` and file data under `data/`. Override those locations with:

```bash
export POKER_DB_PATH=/path/to/poker_tracker.db
export POKER_DATA_DIR=/path/to/data
```

## Shared-password authentication

Local development remains available without a password. Web deployments fail closed when authentication is required:

```bash
export APP_PASSWORD='use-a-long-random-secret'
export POKERTRAINER_REQUIRE_AUTH=true
streamlit run app.py
```

Secrets are read from the environment and are never stored in SQLite.

## Completed-session reconstruction

The Import workflow connects a saved video to the existing two-model CV pipeline:

1. Save and validate a completed-session video.
2. Start one detached `cv_reconstruction` job.
3. Track PID, heartbeat, progress, and safe error state in SQLite.
4. Produce a retained timeline and session export.
5. Create a consistent SQLite backup.
6. Import reconstructed hands as needs-correction drafts with confidence and provenance.
7. Either flag the hand for later debugging or correct it immediately in Study.
8. Retain before/after audit records, reconcile the ledger, and rerun coaching
   before marking the hand reviewed.

Only one local processing job can run at a time. On restart, dead or stale workers are marked failed instead of remaining stuck. SQLite uses WAL mode so the UI can continue reading while the worker writes.

Corrections are written transactionally to SQLite. Editing hand facts, players,
or actions changes CV imports to `corrected_cv`, records the original and
corrected values in `hand_corrections`, invalidates settlement, and marks prior
hand/session coaching stale without deleting it.

If the cause is not known yet, use **Flag this hand for future debugging**.
PokerTrainer saves the issue categories, description, and a snapshot of the
hand, session, players, actions, and correction history in `hand_issues`. The
Hands page provides a cross-session inbox of every unresolved issue. A later
debugging pass can download the snapshot, fix the pipeline, and record resolution
notes without losing the original evidence.

JSON export version 4 carries correction, issue, and coaching history through
backup/import workflows. Schema version 12 is additive: v10 added correction
history and review staleness, v11 added solver records, and v12 adds the
debugging issue queue. Existing rows remain intact. Startup migrates older
databases automatically; older application versions intentionally refuse to
open a newer database.

File layout:

```text
data/
  backups/       rotating pre-import SQLite backups
  cv_timelines/  retained reconstruction timelines
  exports/       generated session exports
  frames/        diagnostic extracted frames
  job_logs/      detached worker logs
  roi_previews/  ROI crop previews
  solver_runs/   TexasSolver commands, logs, and raw strategy JSON
  videos/        uploaded completed-session videos
```

Videos remain files; SQLite stores structured records and paths.

## Coaching providers

Coaching uses a configured external provider. If no key is available, coaching
stays unavailable instead of generating substitute content:

```bash
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
```

Every generated prompt passes the post-session safety check and can be inspected before use.

## TexasSolver postflop review

The Study workspace can run TexasSolver against a completed, reconciled,
heads-up NLHE cash-game postflop spot. Five- through eight-handed tables are
supported through their input ranges; the solver itself receives the two
remaining postflop ranges.

For local use, compile or install the pinned `console` implementation and
configure its absolute path:

```bash
# Apple Silicon source builds require CMake and an OpenMP runtime.
brew install cmake libomp
export TEXAS_SOLVER_PATH=/absolute/path/to/console_solver
# Optional when the installed resources directory is not beside the binary:
export TEXAS_SOLVER_RESOURCE_DIR=/absolute/path/to/resources
export POKERTRAINER_SOLVER_THREADS=4
streamlit run app.py
```

The Docker image compiles commit `42313c9c` and configures the resulting binary
automatically for both `linux/amd64` and `linux/arm64`. Hosted defaults limit
the solver to two threads, 8 GB of address space, a 30-minute run, and one
heavy CV/solver job at a time.

Each player can use an automatic estimated range, a built-in or saved premade
range, or validated custom weighted notation. Custom syntax accepts both eval7
weights such as `50%(AJs+)` and TexasSolver-style `AJs+:0.5`. Exact range
snapshots, convergence, assumptions, logs, and raw JSON are retained with the
run. Built-in ranges are study estimates, not solved preflop GTO.
The blocker-filtered adapter sends exact suit-specific combinations; the pinned
console cannot safely apply suit isomorphism to those ranges, so this optimization
is explicitly disabled and recorded in each run's assumptions.

TexasSolver output is used only as evidence for an optional AI explanation.
Raked cash results are labeled no-rake equilibrium approximations, entirely
multiway and tournament/ICM spots are excluded, and the app does not report
exact BB loss because the standard strategy JSON does not provide action EV.
Hosted execution and binary distribution remain a release gate until written
maintainer permission for that scope or documented AGPL compliance is retained
with the release.

## Data health

Run the operator audit against the configured SQLite database and data directory:

```bash
python -m poker_tracker.maintenance --restore-backups
```

The command runs SQLite structural and foreign-key checks, validates the schema
version and core schema, verifies recorded videos and review images still exist,
compares stored video sizes, and checks every retained backup.
`--restore-backups` copies each backup into an isolated temporary database for a
safe restore drill; it never replaces the live database. The audit does not
issue application-data writes, although SQLite may create or update its normal
WAL shared-memory sidecars while opening an active database. Add `--json` for
automation. The command exits nonzero when a check fails, while a fresh
installation with no backups reports a warning.

## Test

```bash
python -m pytest -q
```

The suite covers persistence, migrations, solver ranges/eligibility/CLI parsing,
the debugging issue queue, correction audit/export, Study UI mutation, coaching
safety, math, video handling, CV reconstruction, job recovery, backups, view
models, authentication, and the Streamlit product shell.

## Container

```bash
docker build --platform linux/amd64 -t pokertrainer .
docker run --rm -p 8501:8501 \
  -e APP_PASSWORD='replace-me' \
  -v pokertrainer-data:/data \
  pokertrainer
```

The image runs as a non-root user and exposes a Streamlit healthcheck. Build both `linux/amd64` and `linux/arm64` before accepting a deployment architecture.

## Deployment

- **Local-first:** run the application directly with `streamlit run app.py` or
  use the Docker command above.
- **Provider-neutral container:** the image keeps runtime configuration in
  environment variables and persistent state under `/data`.
- **Optional OCI reference:** [deploy/oci/README.md](deploy/oci/README.md)
  documents one possible Oracle Cloud setup. It is not an active deployment
  target and should only be used when deployment is explicitly requested.

## Repository guidance

Project guidance has one source per purpose:

- [AGENTS.md](AGENTS.md) contains durable engineering instructions for Codex and
  other coding agents that support the convention.
- [CLAUDE.md](CLAUDE.md) contains the same complete instructions for Claude
  Code. It must remain byte-for-byte identical to `AGENTS.md`.
- [PLAN.md](PLAN.md) contains current status, priorities, release gates, and the
  definition of done.
- [cv_lab/notes/README.md](cv_lab/notes/README.md) indexes the chronological CV
  research record. Those findings explain decisions but are not the roadmap.
- [deploy/oci/README.md](deploy/oci/README.md) is the Oracle deployment
  runbook.

There are currently no repository-local skills, custom subagents, or Codex
project settings. Add a skill only for a repeatable workflow that needs its own
instructions or scripts; ordinary project conventions belong in `AGENTS.md`.
