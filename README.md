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

Study shows one task at a time:

1. **Replay** — inspect the full saved action history in a readable vertical
   action list with expanded pot, stack, SPR, and note details. Choose any action
   to update the single table replay to that moment, including the street board,
   pot, remaining stacks, folded seats, and highlighted actor. Table stacks, pot,
   and results always show an explicit **BB** unit.
2. **Fix & confirm** — resolve blockers and open only the correction tool needed.
3. **Analyze** — use quick math, TexasSolver, coaching, or notes.

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

`POKER_DB_BUSY_TIMEOUT_MS` (default `30000`) is how long a second process — a CV
job, a solver job, or a second app start — waits for a lock instead of failing
with "database is locked". Raise it if a migration on a very large database
outlasts it.

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
hand/session coaching stale without deleting it. Deleting a hand marks its
session's coaching stale for the same reason.

Everything you declare in the Accounting reconciliation panel — the **rake
policy**, which removes chips the recording never showed; **dead money**, which
adds them; and the **pot awards**, which decide who is paid and therefore what
the hero result is — is checked by reconciling the hand twice: once with what you
declared, and once with the whole declaration withdrawn. The hand is
**assumption-dependent** when withdrawing it changes either the verdict (it stops
reconciling) or the figures (it still reconciles, but the pot, the rake, a payout
or the hero result come out different). Both halves matter: a freshly imported
hand often records none of the figures the cross-check compares, so it reconciles
under every policy while the declaration still moves the hero result shown in
every list, stat and prompt.

The pot awards are in that set because on a reconstructed hand nothing observed
them. The CV pipeline emits no settlement rows at all, so the winner of every pot
is typed in — in the same panel, in the same save, as the rake — and it is the
single input the payouts and the hero result are computed from. On a hand with no
recorded pot size and no recorded result, choosing the other winner moves the
reported result by the whole pot.

An assumption-dependent hand is blocked from study until you press **Confirm this
assumption** beside each listed declaration in Study → Summary → Accounting
reconciliation, which records the exact chip movement you are attesting to.
Nothing else clears it: not the "I confirm this hand is correct" tick, which is a
judgement about the reconstruction rather than a claim about the room, and not
the **Acknowledge** button in the Source warnings panel, which answers pipeline
codes and is never offered a settlement assumption.

Confirming is also what re-enables everything the block disabled. Until every
measured declaration is answered, the hand's derived result is left out of the
session win rate, out of the hero-result column of every list, and out of the
math facts handed to a coaching provider, and the hand is not solver-eligible;
once answered, it is published everywhere at once. A hand you entered yourself is
answered by definition and is never held back.

Because the check measures chips rather than inspecting fields, a declaration that
moves nothing is never raised: a rate with a zero cap, no-flop-no-drop on a hand
that saw no flop, or a chip unit coarser than the whole rake are all silent. And
because the confirmation carries both the measured amount and the declaration it
was measured on, it covers that declaration only — if a later correction grows the
pot the same policy applies to, or the policy changes to one that happens to
remove the same chips, you are asked again.

A separate, simpler measurement — "does this declaration move chips at all?" —
records `declared_unobserved_chips` and `declared_unobserved_rake` as an audit
note on the hand's own evidence, shown as **Declared settlement inputs**. They are
not pipeline warnings and never appear in the Source warnings panel: what the
pipeline could not prove and what you declared are different claims, and nothing
in readiness reads the second.

The exemption is for hands **you entered here**, where every figure is your own
entry: those are measured and reported but never blocked, and never withheld from
coaching, the win rate or the solver either — an ordinary room rake on a hand you
typed in is your own observation, and there is no attestation for the product to
ask you for. It is not granted by the
`manual` label, because an import payload can write that label. Every hand that
arrives through import is stamped as imported and attests to its own declared
assumptions in the database it lands in — the same reason import never lands a
`reviewed` status and never carries your acknowledgements across.

The settlement editor's **Replace observed final pot/result with the derived
ledger values** rewrites `pot_size` and `hero_bb_won` — the hand's *observed*
summary — from the reconciled ledger. It is offered because a freshly
reconstructed hand often records neither, and once you have verified the action
line and the winners the ledger is the best record there is. It is refused while
the reconciliation is not established: an unbalanced or illegal ledger, or a
declared settlement input you have not confirmed. Confirm the assumption (or
correct the declaration) and save again, and it writes; otherwise the settlement
is saved and the recorded pot and result are left exactly as they were, with the
reason named. A derived figure that rests on an unconfirmed declaration must not
become an observation, because the recorded pair is the independent evidence the
cross-check compares the ledger against.

The settlement editor's **Chip unit** rounds the rake and nothing else. It is a
room rule — a house that drops whole dollars against a 0.50 blind is ordinary —
and it no longer decides the granularity a chopped pot is divided at. That
granularity is derived from the hand's own amounts: the finest decimal place any
observed contribution or declared dead-money figure is written in, capped at one
whole chip. Indivisible chips are still real, so a 21-chip pot chopped two ways is
still pushed 11/10 in the audited `Order` column's order, and a rake share is
never charged to a pot beyond what that pot holds.

What you type in `Chip unit` **does** change derived payouts, because rounding the
rake changes the net pot every payout is drawn from. On an 80-chip pot at a
declared 50% rake, a unit of 1 charges 40 and pays the winner 40; a unit of 3
charges 39 and pays 41; a unit of 81 charges nothing and pays 80. Read it as
carefully as the rake rate itself before you press **Confirm this assumption** —
it is a declared settlement input like any other, it is measured like one, and a
value that moves chips is named and blocks the hand until you attest to the
movement. (A previous revision of this paragraph said no value you type here
changes a derived payout. That was false, and false about the one field two
adversarial rounds landed critical findings on.)

If the cause is not known yet, use **Flag this hand for future debugging**.
PokerTrainer saves the issue categories, description, and a snapshot of the
hand, session, players, actions, and correction history in `hand_issues`. The
Hands page provides a cross-session inbox of every unresolved issue. A later
debugging pass can download the snapshot, fix the pipeline, and record resolution
notes without losing the original evidence.

A hand is only presented as study-ready when every blocker clears: explicit
completion, valid and unique hero/board cards, confirmed table layout,
authoritative reconciled accounting, no open debugging issue, no unresolved
source warning, no stale coaching or solver result being shown as current, and —
for a reconstructed hand — your explicit confirmation. The Study page lists each
blocker by category with the exact action that clears it. Reconstruction
confidence is shown as a bucketed label only; no single percentage is ever
presented as proof the whole hand is correct. Partial and uncertain hands stay
fully inspectable, replayable, and correctable — they simply cannot be marked
reviewed.

Every blocker names an action you can actually take. The two stale-evidence
blockers each carry an escape hatch for the case where re-running is not
available to you: **Delete stale run** removes a solver result a correction
invalidated when the hand is no longer solver-eligible, and **Discard stale
coaching** deletes a stale retained review when no LLM provider is configured.
Importing a session marks every coaching review stale by construction, so without
that control an imported hand would sit behind a blocker naming a button you
cannot press. Neither control touches a current result. The solver control is not
drawn while a cancellation is still in flight — that blocker says to wait for the
cancellation instead.

`Reviewed` is a workflow label, not a readiness verdict, and PokerTrainer never
presents it as one: readiness is re-derived on every render, and any edit to a
hand's players, actions, or settlement returns a reviewed hand to
`needs_correction` so a promotion cannot outlive the evidence it was granted on.

JSON export version 5 carries correction, issue, coaching, and completion history
through backup/import workflows. Import still accepts versions 1-4 and gives them
safe conservative defaults. Importing a session lands every hand as
`needs_correction`, whatever review status and whatever `source_type` the payload
declares: your confirmation that a hand is correct is deliberately per-render and
never persisted, so it cannot travel in a file, and the importing operator has not
yet seen the evidence. (A genuine manual export loses its `reviewed` label too,
because it is byte-identical to a forgery of one, and the label is one tick and
one save away for the operator who now vouches for it. The v13 migration is
different: it keeps manual review statuses, because a migrated database is your
own data rather than somebody's JSON.) For the same reason, source-warning
acknowledgements do not travel either: a v5 export of a hand whose warnings you
accepted re-imports with those warnings unaccepted, so the importing operator
accepts them themselves. The codes are preserved in full — only your attestation
to them is dropped. Settlement-assumption confirmations are reset for the same
reason, and the dependence is simply re-measured and asked again. A debugging
issue you resolved re-imports **open**, with your resolution notes carried into
its description as history: resolving is an assertion that somebody looked at the
hand and fixed the thing, and the importing operator has looked at nothing.
Retained coaching travels the same way: the text of every saved review is imported
in full and marked *stale*, because a review describes the hand, ledger and winners
of the database that wrote it and nothing in the importing database can verify that
claim. Re-run coaching there to make it current. Schema
version 13 is additive: v10 added correction
history and review staleness, v11 added solver records, v12 added the debugging
issue queue, and v13 adds explicit hand completion
(`complete`/`partial`/`uncertain`/`not_applicable`) and its versioned
reconstruction evidence. Existing manual hands become `not_applicable`; existing
reconstructed hands migrate conservatively to `uncertain` and `needs_correction`
and must be re-confirmed. Existing rows remain intact. Startup migrates older
databases automatically; older application versions intentionally refuse to open
a newer database, and a version 5 export cannot be read back by an older release,
so keep a copy of any pre-upgrade export you may need to restore. A database whose
schema version stamp is missing, unreadable, behind the schema the file actually
contains, or ahead of it is refused rather than re-migrated, and the message says to restore from a
backup: re-running the version 13 migration against a live database would discard
every review confirmation recorded since it first ran. A brand-new database cannot
reach that state: the base tables, the migration chain, and the version stamp are
written in one transaction, so an interrupted first start — a power loss, an OOM
kill, a container restart, Ctrl-C — leaves an empty file the next start simply
creates again.

A consistent, self-contained backup is written to `data/backups` before any real
file database is migrated, and each snapshot is left in `journal_mode=delete` so
it can be verified and restore-drilled from a read-only or archival mount.
**Migration fails closed:** if the snapshot cannot be written — a read-only
container filesystem, an unmounted data volume, or a full disk — startup raises
and no migration runs. The error names the backup directory it could not write
and states that the database is unchanged, rather than reporting SQLite's
`unable to open database file`, which reads as database corruption. Make the
`data/` mount writable before the first start after an upgrade.

Pre-migration snapshots are **pinned**: they are named
`poker_tracker-premigration-<timestamp>.sqlite3` and the five-slot per-import
rotation never matches them, so routine snapshots cannot delete the only rollback
point for an irreversible migration. They keep their own five retention slots, so
a repeatedly failing migration — which snapshots the whole database on every
start — cannot fill the backup mount either. They are stamped at the pre-upgrade
schema version, so `data_health` verifies and restore-drills them without the
current-version comparison it applies to rotating snapshots. Delete them yourself
once the upgrade is proven.
Per-import snapshots keep the rotating `poker_tracker_<timestamp>.sqlite3` name
and the newest five are retained. Rotation matches that exact timestamped name and
nothing else, so your own files in `data/backups` are never deleted even when they
start with `poker_tracker_`. Opening a database this app refuses — one
written by a newer release, or one whose version stamp is unreadable — never
writes to the file, so an archival or read-only restore mount is safe to point at.

File layout:

```text
data/
  backups/       pinned pre-migration and rotating pre-import SQLite backups
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

### Using the integration

1. Open **Study** and choose a completed hand.
2. In **Fix & confirm**, correct any cards, positions, players, or actions and
   reconcile the chip ledger. **Show exact requirements** explains each blocker.
3. Confirm imported/reconstructed evidence after it matches the recording.
4. Open **Analyze → TexasSolver**. The app checks eligibility and explains every
   item that still needs correction.
5. Confirm the automatically selected heads-up street, pot, and effective stack.
6. Start with **Default** ranges for both players, or choose a premade/custom
   range when you have a better assumption.
7. Press **Run TexasSolver analysis**. Solver work runs in the background; use
   **Refresh** to check it or **Cancel** to stop it.
8. Review Hero's combo frequencies, convergence, assumptions, and the mapped
   recorded action. Optionally generate an AI explanation grounded in that
   saved solver evidence.

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
python -m ruff check .
python -m mypy
```

The suite covers persistence, migrations, hand completion and study readiness,
solver ranges/eligibility/CLI parsing, the debugging issue queue, correction
audit/export, Study UI mutation, coaching safety, math, video handling, CV
reconstruction, job recovery, backups, view models, authentication, and the
Streamlit product shell. It also carries dedicated readiness-bypass and
adversarial-round regressions that try to make an unproven hand look
study-ready.

One test is skipped on purpose:
`tests/test_ocr_readers.py::test_without_chip_template_chip_would_join_run` is a
negative control documenting why the chip affix exists, and it skips when the
synthetic chip glyph falls below the classifier confidence floor, because the
misread it demonstrates then does not occur. Any other skip is a real problem.

`python -m mypy` uses the narrow file list configured in `pyproject.toml`; it is
not whole-repository type checking.

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
