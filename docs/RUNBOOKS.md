# Operator runbooks

Procedures for installing, validating, diagnosing, backing up, restoring, and
debugging PokerTrainer. Each one is written to be followed without prior
context — the intent is that another engineer, or a future agent, can act from
this file alone.

Conventions used throughout:

- `$DATA` is the data root (`POKER_DATA_DIR`, default `./data`).
- `$DB` is the SQLite database (`POKER_DB_PATH`, default `./poker_tracker.db`).
- Commands are run from the repository root.
- **Nothing here writes to `$DATA/backups/` except the backup procedure itself.**
  That directory rotates on a fixed count; a stray file evicts a real restore
  point.

---

## 1. Clean local install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r requirements-cv.txt      # only if you will run reconstruction
streamlit run app.py
```

Verify before doing anything else:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
python -m poker_tracker.maintenance --json | head -40
```

A clean install has no database yet. The first launch creates it at the current
schema version; there is nothing to migrate.

---

## 2. Diagnosing models, FFmpeg, and TexasSolver

```bash
python -m poker_tracker.release_gate --mode fixture --report-dir /tmp/rg
```

The report's `environment` and `models` blocks answer all three questions at
once. Read them rather than probing individually:

| Symptom in the report | Meaning | Fix |
|---|---|---|
| `environment.ffmpeg: null` | FFmpeg is not on `PATH` | `brew install ffmpeg` (macOS) |
| `models.<role>.present: false` | Weights missing | Restore the `.pt` file under `cv_lab/models/` |
| `environment.dependencies.torch: null` | CV extras not installed | `pip install -r requirements-cv.txt` |

TexasSolver is separate and optional — see README's solver section. The base
application must remain usable without it; if it is not, that is a defect.

---

## 3. Running the release gate

```bash
# Fast, no video decoding. This is what CI runs.
python -m poker_tracker.release_gate --mode fixture --report-dir data/release_reports

# The real thing: decodes every corpus recording with the pinned weights.
POKER_VALIDATION_ROOT=/path/to/vault \
python -m poker_tracker.release_gate --mode full --report-dir data/release_reports

# The same acceptance path inside the pinned image.
python -m poker_tracker.release_gate --mode container --container-image pokertrainer:release-gate
```

**Read the exit code, not the output's tone:**

| Exit | Meaning | What to do |
|---|---|---|
| `0` | Every mandatory gate passed | Proceed |
| `1` | A product or accuracy gate failed | Read `aggregate` and `per_hand`; the product is wrong |
| `2` | Setup invalid — the run could not be performed | Fix the environment; **nothing was measured** |

Two things in the report are easy to misread and worth checking explicitly:

- **`aggregate.measured`.** When `false`, every count beside it is `null`
  rather than `0`, because zero errors over zero measurements is not a result.
  If you find yourself quoting "0 critical errors", check this field first.
- **`certification.release_certifying`.** Read `certification.executed` first:
  it lists what the run performed, and everything else in the block is computed
  from it rather than from the mode you asked for. `release_certifying` is
  `true` only when the run decoded video, loaded the pinned weights,
  reconstructed the recordings and scored the result.
  - `fixture` scores retained timelines and cannot tell a pipeline's timeline
    from one written by hand. A passing fixture report is a regression check.
  - `container` runs the **fixture** gate inside the image: the image gets the
    manifest directory and no vault. It certifies that the image reproduces the
    host's fixture verdict, and nothing about video reconstruction.
  - A `full` run that stopped before decoding anything — no vault, no FFmpeg, no
    weights — reports an empty `executed` and claims no decoding, which is why
    an exit `2` full run must never be quoted as a certified one.
- **`duration_source` on a full-mode case.** `manifest` means the case declared
  `recording.duration_s`; `probed_from_recording` means the manifest did not and
  the file was measured. A case whose length can be established neither way
  fails as setup invalid: the sampling window is unknown, and the gate does not
  choose one. Declare `duration_s` in the manifest to keep the window pinned
  rather than re-derived on every run.

Today the committed corpus exits `2` because Phase 2 has produced no answer
keys. CI asserts that it does.

---

## 4. Corpus vault setup

```bash
export POKER_VALIDATION_ROOT=/Volumes/Vault/pokertrainer
python -m poker_tracker.validation --manifest validation/clubwpt_v1.json --require-recordings
```

Rules that are not negotiable:

- Raw recordings live under the vault and are **never committed**.
- The locked test split is sealed. Do not open, tune against, or rearrange it.
- Answer keys are committed; recordings are not. The manifest carries the hash
  of both so a silent edit to either is detectable.

The seal check compares recording digests across splits, not only filenames, and
`--require-recordings` makes it read the bytes; without that flag it compares what
the manifest declares and `stats["locked_seal"]` says so. What a digest cannot
answer is named in the report and is your responsibility: a re-encoded, trimmed or
re-rendered copy of a locked recording, and a near-duplicate or adjacent segment
captured from the same source session, are the same material to a model and
different bytes to the check. Assign a whole session to one split and never split
a session across two.

To re-hash an edited answer key:

```bash
python -m poker_tracker.validation --hash-truth validation/truth/<case>.json
# paste the digest into the manifest's truth_sha256 for that case
```

---

## 5. Database migration

Migrations run automatically on `init_db()` and are strictly forward. The
application refuses a database written by a newer build rather than guessing.

```bash
# Always snapshot first. This is the same call the CV import path makes.
python - <<'PY'
from pathlib import Path
from poker_tracker.persistence.backup import backup_database
print(backup_database(Path("poker_tracker.db"), Path("data/backups")))
PY

# Then open the app (or any PokerDatabase) once to apply pending migrations.
python -c "from poker_tracker.persistence.db import PokerDatabase, SCHEMA_VERSION; \
d=PokerDatabase('poker_tracker.db'); d.init_db(); print('schema', SCHEMA_VERSION); d.close()"

python -m poker_tracker.maintenance --json | python -c "import json,sys; \
r=json.load(sys.stdin); print(r['healthy']); [print(c['name'], c['status']) for c in r['checks']]"
```

Every migration's docstring carries a MIGRATION IMPACT section stating exactly
what was added or changed and why an existing row still reads correctly. Read it
before applying one to data you care about.

---

## 6. Backup and isolated restore

Backups rotate: the newest `BACKUP_KEEP_COUNT` are retained and older ones are
evicted. A backup is only a backup once it has been restored somewhere else.

### Undoing a deletion made in the product

Deleting a session, a hand or an ROI profile writes a snapshot first, into its
own retention pool, and refuses the deletion if the snapshot cannot be written.
**Settings -> Storage & health** lists every retained snapshot with its purpose.
To roll one back:

```bash
# 1. Stop PokerTrainer.
# 2. Pick the snapshot taken before the deletion.
ls -t data/backups/poker_tracker-predelete-*.sqlite3 | head
# 3. Verify it in isolation BEFORE overwriting anything.
DRILL=$(mktemp -d)
cp data/backups/poker_tracker-predelete-session12-<stamp>.sqlite3 "$DRILL/restored.db"
python -m poker_tracker.maintenance --db "$DRILL/restored.db" --data-dir "$DRILL/data" --json
# 4. Copy it over the file POKER_DB_PATH points at, then start the app.
```

Snapshots hold rows only. Videos, frames, timelines and solver outputs live
outside SQLite and are not copied, so a snapshot restored after those files were
removed will reference artifacts that are gone; the health audit reports them.

```bash
# Verify every retained backup by restoring it into memory and checking it.
python -m poker_tracker.maintenance --restore-backups --json > /tmp/health.json
python -c "import json; r=json.load(open('/tmp/health.json')); \
print('healthy:', r['healthy']); \
[print(c['name'], c['status'], c['message']) for c in r['checks'] if c['status']!='ok']"
```

Full drill onto a clean data root — **never against the live database**:

```bash
DRILL=$(mktemp -d)
cp "$(ls -t data/backups/poker_tracker_*.sqlite3 | head -1)" "$DRILL/restored.db"
python -m poker_tracker.maintenance --db "$DRILL/restored.db" --data-dir "$DRILL/data" --json
```

Confirm after restoring: session and hand counts match expectations, the schema
version is current, foreign keys pass, and one completed hand opens in Study.
The health report's artifact check reports missing video, frame, timeline and
solver files — those live outside SQLite and are backed up separately.

### Fresh-machine recovery drill

The procedure above verifies a *file*. This one verifies a *recovery*: it
restores a chosen snapshot into a throwaway location, migrates it, and answers
whether the study history came back. Run it on the machine you would actually
recover onto, and run it before you need it.

**Bring three things.** Nothing else in this section works without all three.

| Input | Where it comes from | If it is missing | If it is stale |
|---|---|---|---|
| Environment configuration | `deploy/.env.example`, copied to `.env`, plus `POKER_DB_PATH` and `POKER_DATA_DIR` | The application starts against `./poker_tracker.db` and `./data` — an empty install that looks healthy | `POKER_DATA_DIR` pointing somewhere the artifacts are not makes every recording, frame and timeline report missing |
| Persistent data directory | Your `$DATA` mount: `videos/`, `frames/`, `cv_timelines/`, `solver/`, `job_logs/` | The drill exits `2` and verifies nothing — there is nowhere to look for artifacts | The drill reports `PARTIAL RECOVERY` and names each file, including artifacts that are present but no longer the bytes the snapshot recorded |
| Verified backup | `$DATA/backups/`, with its `.inventory.json` beside it | Nothing to restore | An older snapshot restores and migrates fine; everything recorded after it is gone |

**Mount the data directory at the same path it had on the machine that wrote
it.** Recorded artifact paths are absolute. A data directory mounted somewhere
new restores a database whose every reference dangles, and the drill will say so
rather than let you find out during a session.

From nothing to a verified application:

```bash
git clone <repo-url> pokertrainer && cd pokertrainer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp /path/to/brought/.env .env            # environment configuration
export POKER_DATA_DIR=/mnt/pokertrainer/data
export POKER_DB_PATH=/mnt/pokertrainer/poker_tracker.db

# Prove the backup recovers BEFORE putting it in place.
DRILL=$(mktemp -d)
BACKUP=$(ls -t "$POKER_DATA_DIR"/backups/*.sqlite3 | head -1)
python -m poker_tracker.maintenance.recovery \
  --backup "$BACKUP" --data-dir "$POKER_DATA_DIR" --target "$DRILL"
echo "drill exit: $?"

# Only once the drill exits 0: put the database in place and start.
cp "$BACKUP" "$POKER_DB_PATH"
python -m poker_tracker.maintenance --restore-backups
streamlit run app.py
```

**Read the exit code:**

| Exit | Verdict | Meaning |
|---|---|---|
| `0` | `RECOVERED` | The history came back and every check verified it |
| `1` | `PARTIAL` | Something is provably gone — rows, artifacts, or a hand the application cannot read |
| `1` | `UNVERIFIED` | The database restored cleanly, but no inventory accompanied the snapshot, so completeness is unproven |
| `2` | `NOT PERFORMED` | The drill refused or could not run; **nothing was checked** |

The drill will not run at all if its target overlaps `POKER_DATA_DIR`, contains
`POKER_DB_PATH`, or already holds a database. That refusal is exit `2`, not a
warning — a drill that restored a three-month-old snapshot over the live file
would destroy exactly the history it was run to protect.

Overlap is decided by file identity, not by spelling: `$DATA/drill` and
`$data/drill` are one directory on the case-insensitive filesystem macOS ships,
and the drill refuses both. It uses the same comparison retention does, and it
deliberately over-matches — on a case-sensitive filesystem, where two spellings
really could be two directories, it still refuses. A target it will not accept
costs you one `--target` argument; the error in the other direction costs the
study history.

What each failing check means:

| Check | Failing means |
|---|---|
| `restored_open` | This build cannot open the snapshot. Usually a database written by a newer build; run the drill with the matching version |
| `study_history_counts` | The restored history is empty, or short of what the inventory records. This is the failure a bare `quick_check` cannot see |
| `backup_inventory` | No inventory beside the snapshot, or one that will not parse. Take a fresh backup; `backup_database` writes the inventory itself |
| `issue_evidence` | A `hand_issues` row came back without the frozen evidence that is the reason it exists |
| `completed_hand_readback` | Rows exist that the application cannot compose into a hand. SQLite is happy; Study would not be |
| `recovered_artifacts` | Recordings, frames, timelines or solver outputs the history references are absent from the data directory, or no longer match the snapshot |

Add `--json` for a machine-readable report; `counts` and `missing_artifacts` are
the two fields worth diffing between drills.

**Known gap, reported on every run.** The inventory records artifacts but no
session or hand counts, so on a snapshot taken today the drill verifies the
artifact set and *self-reports* the totals — `backup_inventory` says so as a
warning. Until the inventory carries counts, compare the reported `counts`
against what you know the source held.

---

## 7. Failed job recovery

```bash
python -c "
from poker_tracker.persistence.db import PokerDatabase
db = PokerDatabase('poker_tracker.db'); db.init_db()
for j in db.fetch_active_jobs():
    print(j.id, j.status, j.progress_percent, repr(j.error_message))
db.close()"
```

A failed job never imports hands — that invariant is asserted in
`tests/test_job_failure_injection.py`. If a job is stuck `running` with a dead
worker, reconciliation resolves it:

```bash
python -c "
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.ui.cv_jobs import reconcile_stuck_jobs
db = PokerDatabase('poker_tracker.db'); db.init_db()
print('reconciled:', reconcile_stuck_jobs(db)); db.close()"
```

The full traceback for a pipeline crash is in `$DATA/job_logs/cv_job_<id>.log`.
The UI does not currently link to it, and the retention sweep expires it after
`POKER_RETAIN_JOB_LOGS_DAYS` — copy it out before running a sweep if you are
still diagnosing.

Then re-run reconstruction. Re-running is safe: completed imports are guarded
against duplication.

### A job that stopped on a resource bound

Three failures are configuration, not a defect, and each names the variable that
caused it in `error_message`:

| Message contains | What happened | What to do |
| --- | --- | --- |
| `exceeding its N-second wall-clock limit` | The reconstruction ran past `POKERTRAINER_CV_TIMEOUT_SECONDS`. The partial timeline and review frames were discarded; no hands were exported. | Raise the variable (60–86400) or shorten the recording, then re-run. |
| `could not be applied` | `POKERTRAINER_CV_MEMORY_GB` was set on a platform that refuses `RLIMIT_AS` — macOS does. Nothing was started and nothing was written. | Unset it for local macOS runs, or run the job in the Linux container. |
| `Reconstruction was not started: … is already running` | Another heavy CV or solver job holds the machine. Nothing was read from the recording. | Wait for it, cancel it, or reconcile it (above), then re-run. |

A timeout message that also says the process "could not be confirmed stopped"
means a pipeline child outlived the worker. Find it before re-running:

```bash
pgrep -f run_two_model_pipeline
```

The bounds are documented in README.md. `POKERTRAINER_CV_MEMORY_GB` caps address
space, not resident memory, so it has to sit well above the resident figure you
expect from PyTorch.

---

## 8. Data-health and storage audit

The same audit is available in-product at **Settings -> Storage & health**,
behind an explicit *Run health check* button, alongside the redacted diagnostics
bundle (configuration, dependency and model identity, layout support, row counts,
and the health report). Attach the bundle to a report rather than pasting
configuration by hand: it is scrubbed before it is written and reports
environment variables by name and set/unset only.

```bash
python -m poker_tracker.maintenance                      # database, artifacts, backups
python -m poker_tracker.maintenance.retention_cli        # what retention WOULD delete
python -m poker_tracker.maintenance.retention_cli --apply
```

The retention audit is a dry run by default and always prints its plan before
`--apply` acts. Five things to understand before using it:

- **A file the product still expects is never offered**, at any age. Windows only
  apply to files nothing expects.
- **"Expects" is wider than "a column names it".** Two artifact classes are
  addressed by convention rather than by a database column, and both are retained
  for as long as the job that produced them exists:
  - `cv_timelines/job_<id>_timeline.json` for a **completed** reconstruction job.
    Nothing in the product deletes a `processing_jobs` row, so a deleted timeline
    leaves that job permanently expecting a file that cannot be rebuilt: every
    remaining validated-hand import for it is blocked, and the recovery drill
    reports `PARTIAL` forever afterwards on an otherwise healthy machine.
  - The frames under `frames/cv_job_<id>/` that a timeline's states name. A
    reconstructed frame gets a database reference only once you review it, so
    asking the columns alone expired exactly the frames still waiting for review.
  A timeline that will not parse is treated as unreadable, not as empty: it still
  named frames and nothing can say which, so the whole sweep is held back and the
  run exits `3`.
- **The age shown is the file's mtime, not how long it has been unreferenced.**
  Nothing records when a row stopped pointing at a file, so a recording orphaned
  moments ago still reports its original age. This matters most for
  `--include-orphan-videos`, which is the one deletion nothing can undo.
- **The plan is a proposal, not an authorization.** `--apply` re-checks every
  path against the database immediately before unlinking it, so a CV job that
  finishes while you are reading the plan protects its frames retroactively.
  Files rescued that way are printed as `KEPT` and the run exits `3`.
- **A reference is matched by file identity, not by spelling.** On the
  case-insensitive filesystem macOS ships, `Session.MOV` on disk and
  `session.mov` in SQLite are one file; retention compares `st_dev`/`st_ino`
  rather than strings so it cannot mistake one for an orphan.

Windows are set per category, in whole days:

```bash
export POKER_RETAIN_FRAMES_DAYS=30
export POKER_RETAIN_TIMELINES_DAYS=90
export POKER_RETAIN_JOB_LOGS_DAYS=30
export POKER_RETAIN_EXPORTS_DAYS=90
export POKER_RETAIN_ROI_PREVIEWS_DAYS=30
export POKER_RETAIN_ORPHAN_VIDEOS_DAYS=365
```

**Zero and negative windows are refused.** `POKER_RETAIN_FRAMES_DAYS=0` reads as
"retain for zero days", which purges every unreferenced frame on the next
`--apply` — too destructive to sit one typo away from a correct setting, and
reachable by accident whenever a shell expands an unset variable to something
`int()` accepts. The CLI exits `2` and names the variable instead. Purging on
purpose is a real need, so it has a flag that says so:

```bash
python -m poker_tracker.maintenance.retention_cli --purge-now          # dry run
python -m poker_tracker.maintenance.retention_cli --purge-now --apply
```

`--purge-now` ignores every window and offers all unreferenced managed artifacts
regardless of age. It does not weaken anything else: referenced files are still
kept, and source recordings still need `--include-orphan-videos` on top.

Exit codes are the same in text and `--json` mode — the format decides how the
outcome is written down, never what it was:

| Code | Meaning |
| --- | --- |
| `0` | Audit only, or an apply that removed everything it offered |
| `1` | An apply attempted a deletion and it failed |
| `2` | Invalid configuration; nothing was examined |
| `3` | Refused to act — a reference source was unreadable, or the plan went stale |

Backups are deliberately outside retention's scope — `poker_tracker.persistence.backup`
owns that directory, and two components expiring one directory is how a verified
restore point disappears.

---

## 9. Docker build and run, both architectures

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t pokertrainer:local .

docker run --rm -p 8501:8501 \
  -e APP_PASSWORD="$APP_PASSWORD" \
  -e POKERTRAINER_REQUIRE_AUTH=1 \
  -v "$PWD/data:/data" -e POKER_DATA_DIR=/data \
  -v "$PWD/poker_tracker.db:/data/poker_tracker.db" -e POKER_DB_PATH=/data/poker_tracker.db \
  pokertrainer:local
```

All durable data must sit on explicit mounts. A container that writes into its
own filesystem loses everything on restart, so verify after starting that new
sessions appear under the mounted `$DATA`.

Before publishing any image, see §11.

---

## 10. Upgrade and rollback

**Upgrade:** back up (§6), pull, reinstall dependencies, launch once to migrate,
run the health check.

**Rollback:** migrations are forward-only, so rolling the *code* back below the
database's schema version makes the application refuse to open it — by design,
because a newer schema may hold data the older build would silently drop. To
roll back, restore the pre-upgrade backup alongside the older code:

```bash
cp data/backups/poker_tracker_<timestamp>.sqlite3 poker_tracker.db
git checkout <previous-tag>
```

Anything recorded after the backup is lost. Take the backup immediately before
upgrading, not the night before.

---

## 11. Licensing before distributing an image

Two components make publishing an image a licensing question. Local use is
unaffected; **distribution** is what triggers the obligation.

```bash
python -m poker_tracker.maintenance.sbom --format notices > NOTICES.txt
python -m poker_tracker.maintenance.sbom --format cyclonedx > sbom.json
python -m poker_tracker.maintenance.sbom --format notices --fail-on-review   # exits 1 while unresolved
```

- **`ultralytics` is AGPL-3.0**, and the reconstruction pipeline depends on it.
  This affects the *base* image, not only a solver-enabled one.
- **TexasSolver** is a separately-obtained native binary with its own license,
  and is a release blocker for any distributed solver-enabled image.

Resolve by obtaining a commercial license, replacing the dependency, or
satisfying the source-offer obligations — with qualified review. Nothing in this
repository is legal advice.

---

## 12. Issue-to-regression debugging workflow

This is the loop a future agent should follow when a hand is wrong.

**1. Capture, without diagnosing.** Flag the hand from Study. The evidence
snapshot freezes what you were looking at.

**2. Export a reproducible bundle.**

```python
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.services.issue_bundle import build_issue_bundle, serialize_issue_bundle

db = PokerDatabase("poker_tracker.db"); db.init_db()
open("issue.json", "w").write(serialize_issue_bundle(build_issue_bundle(db, ISSUE_ID)))
```

The bundle carries the identity — path, size, hash — of the source recording and
frames, the model hashes, and the redacted environment. It carries identifiers,
never the recording; restore that from the path and hash.

**3. Reproduce, then write the failing test first.** Choose the cheapest fixture
that can actually reproduce it:

| Bug lives in | Fixture |
|---|---|
| Reconstruction only | cached-state fixture |
| OCR or detection | cropped-frame fixture |
| Decoding, sampling, anchoring, boundaries | full-video corpus case |

**4. Register the regression and record that it failed.**

```python
from poker_tracker.services.regression_promotion import (
    promote_issue_to_regression, record_regression_observation)

case = promote_issue_to_regression(
    db, ISSUE_ID, kind="cropped_frame", fixture_path="tests/test_x.py::test_y")
record_regression_observation(db, case.id, failing_before=True)
```

**5. Fix it, then record that it passes.**

```python
record_regression_observation(
    db, case.id, passing_after=True, fixing_commit="<sha>",
    report_path="data/release_reports/release_gate_report.json")
```

**6. Close the issue.** A release-blocking issue cannot be closed until a
regression has been observed *both* failing and passing — a test that only ever
passed proves nothing, because it may not touch the defect at all.

```python
db.resolve_hand_issue(ISSUE_ID, resolution_notes="...")
```

**7. Re-run the affected corpus slice and the locked acceptance set** (§3).

---

## 13. Suite quality: skips, flakes, and coverage

Three commands behind the Phase 14 exit gate — "all mandatory suites pass
without unexplained skips or flaky reruns". They live in
`poker_tracker/suite_quality/`.

**Skips.** Every skip must be conditional and must say what the environment is
missing. The rule is enforced inside the suite by
`tests/test_suite_quality.py`; this prints the inventory a reviewer reads.

```bash
python -m poker_tracker.suite_quality skips tests deploy/tests
```

Exit 2 means a skip is unexplained. If a legitimate skip states a precondition
no vocabulary would recognise, register its reason in
`skip_policy.REVIEWED_SKIPS` with the review a reader would otherwise have to
perform — that registration *is* the explanation, and the audit reports any
registration whose skip has since gone away.

**Flakes.** The suite runs once, in order, everywhere else. This runs it
repeatedly, shuffling module order and within-module order between passes, and
names every test whose result was not the same each time.

```bash
python -m poker_tracker.suite_quality flake --passes 4 --seeds 20260801,20260802
```

Exit 2 means a test disagreed with itself. The report separates *unstable*
(passed once, failed once — the flake), *consistently failing* (broken, not
flaky), and *order dependent* (ran or skipped differently under a shuffle).
There is no rerun-until-green anywhere in this repository: rerunning until
green is how a flake becomes permanent.

To shuffle a single ordinary run — the plugin does nothing without a seed:

```bash
python -m pytest -p sq_random_order --sq-seed 20260801
```

Load `sq_random_order` (the top-level shim), never
`poker_tracker.suite_quality.random_order`. pytest imports a `-p` plugin before
any conftest, so naming a module inside the package pulls in
`poker_tracker/__init__.py` — and with it the modules that resolve the
operator's database and data directory — before `tests/conftest.py` redirects
them. A run that does that reads and migrates `<repo>/poker_tracker.db`.
`tests/conftest.py` now refuses such a run outright rather than letting it
proceed.

**Coverage.** Measured with the `coverage` library, not `pytest-cov`:
`pytest-cov` registers a plugin on install and so changes every run on the
machine.

```bash
python -m coverage run -m pytest -q
python -m coverage json -o coverage.json
python -m poker_tracker.suite_quality coverage coverage.json
```

There is no floor and no ratchet, deliberately. The output worth acting on is
the *name* of a core module the suite never executes, not a percentage; a
`fail_under` rewards executing lines instead of asserting on them.

Read the last section of that report carefully. `poker_tracker/coaching`,
`math`, `persistence` and `ui` have no `__init__.py`, and coverage's walk of
the source tree descends only into directories that have one — so a module in
them that no test imports is missing from the payload rather than sitting in it
at 0%. The report reconstructs that list from the filesystem and prints it
under "Never imported by the suite".

---

## 14. What this application must never become

Non-negotiable, and worth restating because it constrains every fix:

- No real-time poker assistance.
- No live table capture.
- No poker-client overlay.
- No current-hand recommendations.

Analysis is for completed hands and sessions only. CV workers are file-based and
detached from any client. If a change would require window capture, screen
polling, hotkeys, or a live API, it is out of scope regardless of how it is
framed.
