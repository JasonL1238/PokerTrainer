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
- **`certification.release_certifying`.** `fixture` mode is `false`: it scores
  retained timelines without decoding video or loading models, and cannot tell a
  pipeline's timeline from one written by hand. A passing fixture report is a
  regression check, not a release.

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

---

## 8. Data-health and storage audit

```bash
python -m poker_tracker.maintenance                      # database, artifacts, backups
python -m poker_tracker.maintenance.retention_cli        # what retention WOULD delete
python -m poker_tracker.maintenance.retention_cli --apply
```

The retention audit is a dry run by default and always prints its plan before
`--apply` acts. Two things to understand before using it:

- **A file the database references is never offered**, at any age. Windows only
  apply to files nothing points at.
- **The age shown is the file's mtime, not how long it has been unreferenced.**
  Nothing records when a row stopped pointing at a file, so a recording orphaned
  moments ago still reports its original age. This matters most for
  `--include-orphan-videos`, which is the one deletion nothing can undo.

Windows are set per category:

```bash
export POKER_RETAIN_FRAMES_DAYS=30
export POKER_RETAIN_TIMELINES_DAYS=90
export POKER_RETAIN_JOB_LOGS_DAYS=30
export POKER_RETAIN_EXPORTS_DAYS=90
export POKER_RETAIN_ORPHAN_VIDEOS_DAYS=365
```

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

## 13. What this application must never become

Non-negotiable, and worth restating because it constrains every fix:

- No real-time poker assistance.
- No live table capture.
- No poker-client overlay.
- No current-hand recommendations.

Analysis is for completed hands and sessions only. CV workers are file-based and
detached from any client. If a change would require window capture, screen
polling, hotkeys, or a live API, it is out of scope regardless of how it is
framed.
