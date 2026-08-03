"""Phase 4 retention: what it refuses to delete matters more than what it deletes.

The failure this suite exists to prevent is deleting the source evidence behind a
saved issue because the file happens to be old. Age is never the first question;
"does anything still point at this" is.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from poker_tracker.maintenance import retention_cli
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.services.retention import (
    DEFAULT_RETENTION_DAYS,
    NEVER_MANAGED,
    RETENTION_ENV_VARS,
    AuditedFile,
    RetentionPolicy,
    StorageAudit,
    apply_retention,
    audit_storage,
    path_identity_keys,
)
from poker_tracker.ui.video_storage import ensure_data_directories

DAY = 86400.0


@pytest.fixture
def workspace(tmp_path: Path):
    """A data root plus an initialized database, with helpers to age files."""
    data_dir = tmp_path / "data"
    paths = ensure_data_directories(data_dir)
    db = PokerDatabase(tmp_path / "test.db")
    db.init_db()
    yield db, paths
    db.close()


def _write(path: Path, *, age_days: float, content: bytes = b"x" * 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    stamp = time.time() - age_days * DAY
    os.utime(path, (stamp, stamp))
    return path


def _record_video(db, path: Path, *, filename: str | None = None) -> None:
    """Point a videos row at ``path`` exactly as an upload would."""
    db._execute(
        "INSERT INTO videos (session_id, original_filename, stored_path,"
        " file_size_bytes, content_sha256, uploaded_at, notes)"
        " VALUES (NULL, ?, ?, ?, '', '2026-01-01T00:00:00Z', '')",
        (filename or Path(path).name, str(path), 128),
    )
    db._commit()


def _filesystem_ignores_case(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.write_bytes(b"probe")
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


def _reference_frame(db, image_path) -> None:
    """Point a saved frame review at ``image_path``, as the app would.

    The review row is foreign-keyed to a real job, so the fixture builds the
    video and job behind it rather than disabling the constraint.
    """
    row = db._execute("SELECT id FROM processing_jobs LIMIT 1").fetchone()
    if row is None:
        video = db._execute(
            "INSERT INTO videos (session_id, original_filename, stored_path,"
            " file_size_bytes, content_sha256, uploaded_at, notes)"
            " VALUES (NULL, 'src.mov', '/nonexistent/src.mov', 1, '',"
            " '2026-01-01T00:00:00Z', '')"
        ).lastrowid
        job_id = db._execute(
            "INSERT INTO processing_jobs (video_id, job_type, status,"
            " progress_percent, message, error_message, created_at)"
            " VALUES (?, 'cv_reconstruction', 'completed', 100, '', '',"
            " '2026-01-01T00:00:00Z')",
            (video,),
        ).lastrowid
    else:
        job_id = row[0]
    db._execute(
        "INSERT INTO reconstruction_frame_reviews (job_id, hand_number,"
        " source_image, timestamp_seconds, status, issue_types, notes,"
        " created_at, updated_at)"
        " VALUES (?, 1, ?, 0.0, 'confirmed', '', '',"
        " '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (job_id, str(image_path)),
    )
    db._commit()


def _audit(db, paths, **kwargs):
    policy = RetentionPolicy(days=dict(DEFAULT_RETENTION_DAYS), **kwargs)
    return audit_storage(db, paths, policy)


# --- The core safety rule ---------------------------------------------------


def test_referenced_file_is_never_deletable_however_old(workspace):
    """A frame behind a saved review stays, even at ten times the window."""
    db, paths = workspace
    frame = _write(paths["frames"] / "cv_job_1" / "t000000.00.jpg", age_days=3000)
    _reference_frame(db, frame.resolve())

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == frame)
    assert entry.referenced is True
    assert entry.deletable is False
    assert "referenced" in entry.reason


def test_unreferenced_file_past_the_window_is_deletable(workspace):
    db, paths = workspace
    stale = _write(paths["frames"] / "orphan.jpg", age_days=90)
    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == stale)
    assert entry.referenced is False
    assert entry.deletable is True


def test_unreferenced_file_inside_the_window_is_kept(workspace):
    db, paths = workspace
    fresh = _write(paths["frames"] / "recent.jpg", age_days=1)
    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == fresh)
    assert entry.deletable is False
    assert "window" in entry.reason


def test_resolved_paths_match_even_through_a_symlinked_data_root(
    tmp_path: Path, workspace
):
    """A reference stored through one spelling must protect the same file."""
    db, paths = workspace
    frame = _write(paths["frames"] / "linked.jpg", age_days=500)
    link_dir = tmp_path / "alias"
    link_dir.symlink_to(paths["frames"])
    # The database records the path through the alias, the walk finds the real one.
    _reference_frame(db, link_dir / "linked.jpg")

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == frame)
    assert entry.referenced is True
    assert entry.deletable is False


# --- Source recordings ------------------------------------------------------


def test_orphan_video_is_not_offered_by_default(workspace):
    """The one artifact nothing can rebuild needs an explicit opt-in."""
    db, paths = workspace
    video = _write(paths["videos"] / "orphan.mov", age_days=5000)
    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == video)
    assert entry.deletable is False
    assert "orphan removal not requested" in entry.reason


def test_orphan_video_is_offered_on_explicit_opt_in(workspace):
    db, paths = workspace
    video = _write(paths["videos"] / "orphan.mov", age_days=5000)
    audit = _audit(db, paths, include_orphan_videos=True)
    entry = next(f for f in audit.files if f.path == video)
    assert entry.deletable is True


def test_referenced_video_survives_the_opt_in(workspace):
    db, paths = workspace
    video = _write(paths["videos"] / "kept.mov", age_days=5000)
    db._execute(
        "INSERT INTO videos (session_id, original_filename, stored_path,"
        " file_size_bytes, content_sha256, uploaded_at, notes)"
        " VALUES (NULL, ?, ?, ?, '', '2026-01-01T00:00:00Z', '')",
        ("kept.mov", str(video.resolve()), 128),
    )
    db._commit()
    audit = _audit(db, paths, include_orphan_videos=True)
    entry = next(f for f in audit.files if f.path == video)
    assert entry.deletable is False


# --- Fail-closed on an unreadable reference source --------------------------


def test_unreadable_reference_source_holds_everything_back(workspace, monkeypatch):
    """Not knowing whether a file is referenced must not mean deleting it.

    "This table holds no references" and "this table could not be queried" would
    otherwise produce the same deletion plan.
    """
    db, paths = workspace
    stale = _write(paths["frames"] / "orphan.jpg", age_days=900)

    original = db.referenced_artifact_paths
    monkeypatch.setattr(
        db,
        "referenced_artifact_paths",
        lambda: (original()[0], ["reconstruction_frame_reviews.source_image"]),
    )
    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == stale)
    assert entry.deletable is False
    assert "held back" in entry.reason
    assert audit.deletable == []
    assert audit.reclaimable_bytes == 0


def test_every_artifact_path_column_is_queryable_against_the_live_schema(workspace):
    """A column listed in ARTIFACT_PATH_COLUMNS but absent would fail silently."""
    db, _paths = workspace
    _found, unreadable = db.referenced_artifact_paths()
    assert unreadable == []


# --- Backups are somebody else's job ----------------------------------------


def test_backups_are_never_managed_by_retention(workspace):
    """Two components expiring one directory is how a restore point vanishes."""
    db, paths = workspace
    backup = _write(paths["backups"] / "poker_tracker_old.sqlite3", age_days=9000)
    audit = _audit(db, paths)
    assert all(f.path != backup for f in audit.files)
    assert "backups" in NEVER_MANAGED


# --- Deletion requires explicit confirmation --------------------------------


def test_apply_is_a_no_op_without_confirm(workspace):
    db, paths = workspace
    stale = _write(paths["frames"] / "orphan.jpg", age_days=900)
    audit = _audit(db, paths)
    assert audit.deletable

    outcome = apply_retention(audit)
    assert outcome.removed == []
    assert stale.exists()


def test_apply_removes_exactly_what_the_audit_listed(workspace):
    db, paths = workspace
    stale = _write(paths["frames"] / "orphan.jpg", age_days=900)
    fresh = _write(paths["frames"] / "recent.jpg", age_days=1)
    referenced = _write(paths["frames"] / "kept.jpg", age_days=900)
    _reference_frame(db, referenced.resolve())

    audit = _audit(db, paths)
    outcome = apply_retention(audit, confirm=True)

    assert outcome.removed == [stale]
    assert not stale.exists()
    assert fresh.exists()
    assert referenced.exists()


# --- Policy configuration ---------------------------------------------------


def test_windows_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("POKER_RETAIN_FRAMES_DAYS", "7")
    policy = RetentionPolicy.from_env()
    assert policy.window_days("frames") == 7
    # Untouched categories keep their defaults.
    assert policy.window_days("exports") == DEFAULT_RETENTION_DAYS["exports"]


@pytest.mark.parametrize("bad", ["-1", "soon", "3.5", ""])
def test_invalid_window_is_rejected_rather_than_defaulted(monkeypatch, bad):
    """A typo must not silently fall back to a window that deletes more."""
    monkeypatch.setenv("POKER_RETAIN_FRAMES_DAYS", bad)
    if bad == "":
        # Empty means "unset", which is the documented way to keep the default.
        assert RetentionPolicy.from_env().window_days("frames") == 30
        return
    with pytest.raises(ValueError):
        RetentionPolicy.from_env()


def test_every_managed_category_has_an_env_var_and_a_default():
    assert set(RETENTION_ENV_VARS) == set(DEFAULT_RETENTION_DAYS)
    assert not set(RETENTION_ENV_VARS) & NEVER_MANAGED


# --- Adversarial round 2 findings -------------------------------------------


def test_a_regression_fixture_is_never_deleted(workspace):
    """The evidence proving a closed issue stays closed must survive a sweep.

    Regression fixtures are frequently a frame or a recording under a managed
    directory, so omitting their column from the reference list let retention
    delete exactly the file the gate depends on.
    """
    from poker_tracker.persistence.models import Hand, HandIssue, Session
    from poker_tracker.services.regression_promotion import promote_issue_to_regression

    db, paths = workspace
    fixture = _write(paths["frames"] / "issue_evidence.jpg", age_days=400)
    session = db.create_session(Session(name="Regression"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    issue = db.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["cards"], description="misread")
    )
    promote_issue_to_regression(
        db, issue.id, kind="cropped_frame", fixture_path=str(fixture.resolve())
    )

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == fixture)
    assert entry.referenced is True
    assert entry.deletable is False

    # And it survives the sweep itself, not just the plan.
    outcome = apply_retention(audit, confirm=True)
    assert fixture.exists()
    assert outcome.removed == []


def test_an_orphan_video_does_not_claim_to_have_been_unused(workspace):
    """Age is the file's mtime, not how long nothing pointed at it.

    A recording orphaned one second ago by a session delete still carries
    whatever mtime it was written with, so an "unused for 600 days" claim would
    be one the data cannot support.
    """
    db, paths = workspace
    video = _write(paths["videos"] / "orphan.mov", age_days=600)
    audit = _audit(db, paths, include_orphan_videos=True)
    entry = next(f for f in audit.files if f.path == video)
    assert entry.deletable is True
    assert "does NOT mean it has been unused" in entry.reason


# --- A file the product expects is not an orphan ----------------------------
#
# Two artifact classes are addressed by convention rather than by a column, so
# `referenced_artifact_paths` cannot see either and retention expired both.


def _completed_cv_job(db) -> int:
    """A completed reconstruction job, exactly as a finished worker leaves one."""
    video = db._execute(
        "INSERT INTO videos (session_id, original_filename, stored_path,"
        " file_size_bytes, content_sha256, uploaded_at, notes)"
        " VALUES (NULL, 'rec.mov', '/nonexistent/rec.mov', 1, '',"
        " '2026-01-01T00:00:00Z', '')"
    ).lastrowid
    job_id = db._execute(
        "INSERT INTO processing_jobs (video_id, job_type, status,"
        " progress_percent, message, error_message, created_at)"
        " VALUES (?, 'cv_reconstruction', 'completed', 100, '', '',"
        " '2026-01-01T00:00:00Z')",
        (video,),
    ).lastrowid
    db._commit()
    return int(job_id)


def _write_timeline(paths, job_id: int, *, age_days: float, images=()) -> Path:
    """Write a job's timeline naming ``images``, as the pipeline does."""
    payload = {
        "hands": [],
        "states": [
            {"state_index": index, "time_s": float(index), "image": str(image)}
            for index, image in enumerate(images)
        ],
    }
    path = paths["cv_timelines"] / f"job_{job_id}_timeline.json"
    return _write(path, age_days=age_days, content=json.dumps(payload).encode())


def test_a_completed_jobs_timeline_is_never_offered_however_old(workspace):
    """A timeline nothing can rebuild must outlive its retention window.

    No column names a timeline, so the reference check called it an orphan from
    the moment it was written. Nothing in the product deletes a `processing_jobs`
    row, so a deleted timeline leaves the completed job expecting a file that can
    never come back: every remaining validated-hand import for it is blocked and
    the recovery drill reports PARTIAL forever on a healthy machine.
    """
    db, paths = workspace
    job_id = _completed_cv_job(db)
    timeline = _write_timeline(paths, job_id, age_days=3000)

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == timeline)
    assert entry.referenced is True
    assert entry.deletable is False

    apply_retention(audit, confirm=True)
    assert timeline.is_file()


def test_purging_now_still_keeps_a_completed_jobs_timeline(workspace):
    """--purge-now waives windows, never the reference rule."""
    db, paths = workspace
    job_id = _completed_cv_job(db)
    timeline = _write_timeline(paths, job_id, age_days=0)

    audit = _audit(db, paths, purge_immediately=True)
    entry = next(f for f in audit.files if f.path == timeline)
    assert entry.deletable is False
    apply_retention(audit, confirm=True)
    assert timeline.is_file()


def test_retention_protects_the_timelines_the_inventory_records(workspace):
    """Retention and the snapshot inventory must name the same file for a job.

    Two definitions of where a job's timeline lives is how one component protects
    a path the other never looks at.
    """
    from poker_tracker.persistence import backup_inventory

    db, paths = workspace
    job_id = _completed_cv_job(db)
    _write_timeline(paths, job_id, age_days=3000)

    expected = backup_inventory.timeline_paths(
        db._connection, backup_inventory.timeline_dir_for(paths["data"])
    )
    assert expected  # the fixture really did produce a completed job
    audit = _audit(db, paths)
    for stored in expected:
        entry = next(f for f in audit.files if f.path == Path(stored))
        assert entry.referenced is True, stored


def test_a_frame_the_timeline_still_names_is_not_an_orphan(workspace):
    """The frames waiting for review were exactly the ones age expired.

    A reconstructed frame acquires a column reference only when the operator
    reviews it, so asking the columns alone deleted precisely the evidence the
    remaining hands still have to be validated against -- and the loss is silent:
    nothing that reads the timeline afterwards can say the frame ever existed.
    """
    db, paths = workspace
    job_id = _completed_cv_job(db)
    frame = _write(paths["frames"] / f"cv_job_{job_id}" / "t000012.00.jpg", age_days=400)
    _write_timeline(paths, job_id, age_days=400, images=[frame.resolve()])

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == frame)
    assert entry.referenced is True
    assert entry.deletable is False

    apply_retention(audit, confirm=True)
    assert frame.is_file()


def test_a_frame_no_timeline_names_is_still_expirable(workspace):
    """The protection is what the timeline names, not the directory it sits in.

    The pipeline deletes every sampled frame no state kept, so a leftover in a
    job's frame directory is genuinely disposable. Protecting the whole directory
    would quietly turn the largest retention category off.
    """
    db, paths = workspace
    job_id = _completed_cv_job(db)
    kept = _write(paths["frames"] / f"cv_job_{job_id}" / "kept.jpg", age_days=400)
    stale = _write(paths["frames"] / f"cv_job_{job_id}" / "stale.jpg", age_days=400)
    _write_timeline(paths, job_id, age_days=400, images=[kept.resolve()])

    audit = _audit(db, paths)
    assert next(f for f in audit.files if f.path == kept).deletable is False
    assert next(f for f in audit.files if f.path == stale).deletable is True


def test_a_timeline_that_will_not_parse_holds_the_whole_sweep_back(workspace):
    """An unreadable reference source is not a source that names nothing.

    A timeline that will not parse still named frames; nothing can say which, so
    offering any of them would be a deletion no one can prove is safe.
    """
    db, paths = workspace
    job_id = _completed_cv_job(db)
    timeline = _write(
        paths["cv_timelines"] / f"job_{job_id}_timeline.json",
        age_days=400,
        content=b"{ this is not json",
    )
    frame = _write(paths["frames"] / f"cv_job_{job_id}" / "t0.jpg", age_days=400)

    audit = _audit(db, paths)
    assert audit.unreadable_references
    assert any(str(timeline) in item for item in audit.unreadable_references)
    assert audit.deletable == []
    assert next(f for f in audit.files if f.path == frame).deletable is False

    outcome = apply_retention(audit, confirm=True)
    assert outcome.removed == []
    assert frame.is_file()


def test_a_timeline_that_does_not_decode_is_unreadable_not_empty(workspace):
    """Bytes that are not UTF-8 must hold the sweep back, not crash the audit."""
    db, paths = workspace
    job_id = _completed_cv_job(db)
    _write(
        paths["cv_timelines"] / f"job_{job_id}_timeline.json",
        age_days=400,
        content=b"\xff\xfe not utf-8 at all",
    )
    _write(paths["frames"] / f"cv_job_{job_id}" / "t0.jpg", age_days=400)

    audit = _audit(db, paths)
    assert audit.unreadable_references
    assert audit.deletable == []


def test_a_timeline_without_a_states_list_is_unreadable_not_empty(workspace):
    """"No states key" and "no frames" must never produce the same answer."""
    db, paths = workspace
    job_id = _completed_cv_job(db)
    _write(
        paths["cv_timelines"] / f"job_{job_id}_timeline.json",
        age_days=400,
        content=json.dumps({"hands": []}).encode(),
    )
    _write(paths["frames"] / f"cv_job_{job_id}" / "t0.jpg", age_days=400)

    audit = _audit(db, paths)
    assert audit.unreadable_references
    assert audit.deletable == []


# --- B-2: a path is a file, not a string -----------------------------------


def test_a_reference_is_honored_through_a_different_case_spelling(workspace):
    """``Session.MOV`` on disk and ``session.mov`` in SQLite are one file.

    macOS ships a case-insensitive filesystem and ``realpath`` does not fold
    case, so a string comparison calls the recording an orphan and deletes the
    one artifact nothing can rebuild while a row still points at it.
    """
    db, paths = workspace
    if not _filesystem_ignores_case(paths["videos"]):
        pytest.skip("case-sensitive filesystem: the two spellings are two files")
    video = _write(paths["videos"] / "Session.MOV", age_days=5000)
    _record_video(db, paths["videos"] / "session.mov", filename="Session.MOV")

    audit = _audit(db, paths, include_orphan_videos=True)
    entry = next(f for f in audit.files if f.path == video)
    assert entry.referenced is True
    assert entry.deletable is False

    outcome = apply_retention(audit, confirm=True)
    assert video.exists()
    assert outcome.removed == []


def test_a_reference_is_honored_through_a_hard_link(workspace):
    """Identity is the inode. Two names for one file are not two files."""
    db, paths = workspace
    frame = _write(paths["frames"] / "real.jpg", age_days=900)
    alias = paths["frames"] / "alias.jpg"
    os.link(frame, alias)
    _reference_frame(db, alias)

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == frame)
    assert entry.referenced is True
    assert entry.deletable is False


def test_non_normalized_spellings_share_one_identity(tmp_path: Path):
    """The helper every comparison goes through, exercised directly."""
    real = tmp_path / "sub" / "file.jpg"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"x")
    detour = tmp_path / "sub" / "." / ".." / "sub" / "FILE.jpg"
    assert path_identity_keys(real) & path_identity_keys(detour)
    assert not path_identity_keys(real) & path_identity_keys(tmp_path / "other.jpg")


def test_a_relative_reference_is_honored_from_any_working_directory(
    workspace, monkeypatch, tmp_path: Path
):
    """A relative stored path means the data root, not wherever the sweep ran.

    Resolving it against the process working directory is how a frame behind a
    saved review looks like an orphan simply because the operator ran the sweep
    from somewhere else.
    """
    db, paths = workspace
    frame = _write(paths["frames"] / "relative.jpg", age_days=900)
    _reference_frame(db, Path("frames") / "relative.jpg")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    audit = _audit(db, paths)
    entry = next(f for f in audit.files if f.path == frame)
    assert entry.referenced is True
    assert entry.deletable is False


def test_a_category_directory_that_resolves_inside_backups_is_not_walked(workspace):
    """The never-managed guarantee holds even when a data root is laid out oddly."""
    db, paths = workspace
    trap = paths["backups"] / "frames"
    trap.mkdir(parents=True, exist_ok=True)
    snapshot = _write(trap / "poker_tracker_old.sqlite3", age_days=9000)

    audit = audit_storage(
        db,
        {**paths, "frames": trap},
        RetentionPolicy(days=dict(DEFAULT_RETENTION_DAYS)),
    )
    assert all(f.path != snapshot for f in audit.files)
    assert snapshot.exists()


# --- B-3: an audit is a proposal, never an authorization --------------------


def test_a_reference_created_after_the_audit_stops_the_deletion(workspace):
    """A CV job finishing while the operator reads the plan must win.

    The audit classified this frame as an orphan and it stopped being one before
    the sweep ran. Deleting on the strength of a stale plan destroys a live file.
    """
    db, paths = workspace
    orphan = _write(paths["frames"] / "orphan.jpg", age_days=900)
    audit = _audit(db, paths)
    assert [f.path for f in audit.deletable] == [orphan]

    _reference_frame(db, orphan.resolve())

    outcome = apply_retention(audit, confirm=True)
    assert orphan.exists()
    assert outcome.removed == []
    assert any("now references it" in entry for entry in outcome.skipped)


def test_a_reference_source_that_fails_after_the_audit_stops_the_deletion(
    workspace, monkeypatch
):
    """Losing the ability to check must stop the sweep, not wave it through."""
    db, paths = workspace
    orphan = _write(paths["frames"] / "orphan.jpg", age_days=900)
    audit = _audit(db, paths)
    assert audit.deletable

    monkeypatch.setattr(
        db,
        "referenced_artifact_paths",
        lambda: (set(), ["extracted_frames.image_path"]),
    )
    # Something else writes, so the check re-reads and discovers the bad source.
    _record_video(db, paths["videos"] / "unrelated.mov")

    outcome = apply_retention(audit, confirm=True)
    assert orphan.exists()
    assert outcome.removed == []
    assert any("could not read" in entry for entry in outcome.skipped)


def test_an_audit_without_a_live_reference_check_cannot_delete(tmp_path: Path):
    """A hand-built audit is a claim about the past with nothing to re-prove it."""
    victim = _write(tmp_path / "orphan.jpg", age_days=900)
    audit = StorageAudit(
        files=[
            AuditedFile(
                path=victim,
                category="frames",
                size_bytes=128,
                age_days=900.0,
                referenced=False,
                deletable=True,
                reason="unreferenced",
            )
        ]
    )
    outcome = apply_retention(audit, confirm=True)
    assert victim.exists()
    assert outcome.removed == []
    assert outcome.failures


# --- B-4: zero days is a typo, not a policy ---------------------------------


@pytest.mark.parametrize("variable", sorted(set(RETENTION_ENV_VARS.values())))
def test_a_zero_window_is_refused_for_every_category(monkeypatch, variable):
    """"Retain for zero days" purges everything, and it is one keystroke away."""
    monkeypatch.setenv(variable, "0")
    with pytest.raises(ValueError) as excinfo:
        RetentionPolicy.from_env()
    message = str(excinfo.value)
    assert variable in message
    assert "--purge-now" in message


def test_a_zero_window_is_refused_however_the_policy_is_built():
    days = dict(DEFAULT_RETENTION_DAYS)
    days["frames"] = 0
    with pytest.raises(ValueError):
        RetentionPolicy(days=days)


def test_a_zero_window_written_after_construction_is_still_refused():
    """``days`` is a plain dict, so the check has to live at the point of use too."""
    policy = RetentionPolicy(days=dict(DEFAULT_RETENTION_DAYS))
    policy.days["frames"] = 0
    with pytest.raises(ValueError):
        policy.window_days("frames")


def test_purging_now_is_available_but_has_to_be_asked_for(workspace):
    """The real operator need is met by a flag that says what it does."""
    db, paths = workspace
    fresh = _write(paths["frames"] / "recent.jpg", age_days=0.5)
    referenced = _write(paths["frames"] / "kept.jpg", age_days=0.5)
    _reference_frame(db, referenced.resolve())
    video = _write(paths["videos"] / "orphan.mov", age_days=0.5)

    policy = RetentionPolicy(
        days=dict(DEFAULT_RETENTION_DAYS), purge_immediately=True
    )
    assert policy.window_days("frames") == 0
    audit = audit_storage(db, paths, policy)

    assert next(f for f in audit.files if f.path == fresh).deletable is True
    # Purging ignores age. It does not ignore references, and it does not
    # promote a source recording past its own opt-in.
    assert next(f for f in audit.files if f.path == referenced).deletable is False
    assert next(f for f in audit.files if f.path == video).deletable is False


# --- B2-6: the output format cannot decide the exit code --------------------


def _run_cli_scenario(scenario: str, *, json_mode: bool, root: Path, monkeypatch) -> int:
    """Run the CLI once for ``scenario`` in a workspace of its own."""
    data_dir = root / "data"
    paths = ensure_data_directories(data_dir)
    db_path = root / "retention.db"
    db = PokerDatabase(db_path)
    db.init_db()
    db.close()

    argv = ["--db", str(db_path), "--data-dir", str(data_dir)]
    if json_mode:
        argv.append("--json")

    if scenario == "error":
        monkeypatch.setenv("POKER_RETAIN_FRAMES_DAYS", "0")
    if scenario in {"would-delete", "deleted", "refused", "deletion-failed"}:
        _write(paths["frames"] / "orphan.jpg", age_days=900)
    if scenario in {"deleted", "refused", "deletion-failed"}:
        argv.append("--apply")
    if scenario == "refused":
        monkeypatch.setattr(
            PokerDatabase,
            "referenced_artifact_paths",
            lambda self: (set(), ["reconstruction_frame_reviews.source_image"]),
        )
    if scenario == "deletion-failed":

        def _cannot_unlink(self, missing_ok: bool = False) -> None:
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(Path, "unlink", _cannot_unlink)

    return retention_cli.main(argv)


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("nothing-to-do", 0),
        ("would-delete", 0),
        ("deleted", 0),
        ("deletion-failed", 1),
        ("error", 2),
        ("refused", 3),
    ],
)
def test_json_and_text_report_the_same_exit_code(
    scenario, expected, tmp_path: Path, monkeypatch, capsys
):
    """A script branching on the exit code must see what the operator sees."""
    text_code = _run_cli_scenario(
        scenario, json_mode=False, root=tmp_path / f"{scenario}-text", monkeypatch=monkeypatch
    )
    capsys.readouterr()
    json_code = _run_cli_scenario(
        scenario, json_mode=True, root=tmp_path / f"{scenario}-json", monkeypatch=monkeypatch
    )
    payload = json.loads(capsys.readouterr().out)

    assert text_code == json_code == expected
    assert payload["outcome"] == scenario
    assert payload["exit_code"] == expected
