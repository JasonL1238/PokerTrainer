"""Phase 4 retention: what it refuses to delete matters more than what it deletes.

The failure this suite exists to prevent is deleting the source evidence behind a
saved issue because the file happens to be old. Age is never the first question;
"does anything still point at this" is.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.services.retention import (
    DEFAULT_RETENTION_DAYS,
    NEVER_MANAGED,
    RETENTION_ENV_VARS,
    RetentionPolicy,
    apply_retention,
    audit_storage,
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
    import os

    os.utime(path, (stamp, stamp))
    return path


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
