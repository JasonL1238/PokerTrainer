"""Removing hands and recordings from a session, and refusing to when it is unsafe.

Two controls were missing and one was misplaced. A session could list its
recordings but never unlink or delete one, so the only delete lived on Import
behind "Advanced diagnostics · legacy frame extraction" -- and wrote no rollback
snapshot, which is the one thing
``poker_tracker/persistence/AGENTS.md`` says a delete path may not do. Hands were
deletable one at a time only, so clearing a superseded import meant N snapshots
into a five-slot pool, which evicts every other rollback point in it.

The tests here are mostly about the refusals rather than the deletions. A delete
that works is easy to see; a delete that quietly races a worker, or strands a
recording no surface can reach, is not.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app as app_module
import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.backup import backups_dir_for, find_snapshots
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Hand,
    ProcessingJob,
    Session,
    VideoRecord,
)
from poker_tracker.ui.cv_artifacts import cv_job_artifact_paths

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _video(session_id: int | None, name: str, stored: Path) -> VideoRecord:
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(b"not really a recording")
    return VideoRecord(
        session_id=session_id,
        original_filename=name,
        stored_path=str(stored),
        file_size_bytes=stored.stat().st_size,
    )


def _hand(session_id: int, number: int) -> Hand:
    return Hand(
        session_id=session_id,
        hand_number=number,
        game_type="No-limit Hold'em",
        table_size=6,
        hero_position="BTN",
        hero_cards="Ah Qs",
        pot_size=20,
        source_type="cv_import",
    )


def _configure(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Detaching a recording
# ---------------------------------------------------------------------------


def test_removing_a_recording_from_a_session_keeps_the_file_and_its_rows(
    tmp_path: Path,
) -> None:
    """Detach is the reversible half. It must not be a quiet delete."""
    db = PokerDatabase(str(tmp_path / "detach.sqlite3"))
    db.init_db()
    session = db.create_session(Session(name="Tuesday"))
    assert session.id is not None
    stored = tmp_path / "videos" / "a.mp4"
    video = db.create_video(_video(session.id, "a.mp4", stored))
    assert video.id is not None

    db.update_video_session(video.id, None)

    detached = db.fetch_video(video.id)
    assert detached is not None, "detach deleted the row"
    assert detached.session_id is None
    assert stored.is_file(), "detach deleted the recording"
    assert db.fetch_videos(session.id) == []

    # And it goes back.
    db.update_video_session(video.id, session.id)
    reattached = db.fetch_videos(session.id)
    assert [item.id for item in reattached] == [video.id]
    db.close()


def test_a_detached_recording_is_still_reachable_behind_fifteen_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason detach could not ship as first written.

    The attach list showed one fifteen-row window over every recording not in this
    session, newest first. Sixteen recordings in other sessions therefore pushed an
    older unassigned one off the end of the ONLY list that can re-attach it -- and
    because both the delete control and Import reach a recording through a session,
    falling off that list left it neither attachable nor deletable while its file
    stayed on disk. A deliberate removal must not be the step that strands one.
    """
    path = tmp_path / "reachable.sqlite3"
    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Target"))
    assert session.id is not None

    # The unassigned one is created FIRST, so every other recording is newer and
    # sorts ahead of it under `uploaded_at DESC, id DESC`.
    stranded = db.create_video(_video(None, "stranded.mp4", tmp_path / "videos" / "s.mp4"))
    assert stranded.id is not None
    for index in range(16):
        other = db.create_session(Session(name=f"Other {index}"))
        db.create_video(
            _video(other.id, f"other{index}.mp4", tmp_path / "videos" / f"o{index}.mp4")
        )
    db.close()

    _configure(path, monkeypatch)
    script = tmp_path / "_panel.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"session = db.fetch_session({session.id})",
                "app_module.show_session_videos(db, session)",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not list(app.exception), [str(item) for item in app.exception]

    keys = {str(item.key) for item in app.button}
    assert f"attach_video_{stranded.id}_to_{session.id}" in keys, (
        "an unassigned recording fell off the only list that can re-attach it"
    )
    # And it can be deleted from there too, so it is never merely stuck.
    assert f"unassigned_delete_video_{stranded.id}" in keys


# ---------------------------------------------------------------------------
# Deleting a recording
# ---------------------------------------------------------------------------


def test_deleting_a_recording_snapshots_and_removes_its_job_artifacts(
    tmp_path: Path,
) -> None:
    """The artifacts no column names, and the rollback point for the rows that do.

    A job's timeline, progress file, pid file, export and log are addressed by JOB
    id and live in three directories, so nothing cascades to them and the frame
    cleanup -- which only knows the simple extraction job's ``video_<id>`` folder
    -- never saw them. Deleting one reconstruction used to leave all five behind
    with no row left that could ever name them again.
    """
    path = tmp_path / "delete-video.sqlite3"
    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Friday"))
    assert session.id is not None
    stored = tmp_path / "videos" / "friday.mp4"
    video = db.create_video(_video(session.id, "friday.mp4", stored))
    assert video.id is not None
    job = db.create_processing_job(
        ProcessingJob(job_type="cv_reconstruction", status="completed", video_id=video.id)
    )
    assert job.id is not None

    artifacts = cv_job_artifact_paths(job.id)
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")

    backups = backups_dir_for(path)
    assert find_snapshots(backups, purpose="predelete", scope=f"video{video.id}") == []

    error, snapshot = app_module.delete_video_and_artifacts(db, video.id)

    assert error is None, error
    assert snapshot is not None
    assert db.fetch_video(video.id) is None
    assert db.fetch_processing_job(job.id) is None, "the job row should cascade"
    assert not stored.exists(), "the recording file survived its own deletion"
    for artifact in artifacts:
        assert not artifact.exists(), f"{artifact.name} was left behind"

    # The rollback point holds the rows, which is the only thing it can hold.
    restored = sqlite3.connect(str(snapshot))
    try:
        assert restored.execute(
            "SELECT COUNT(*) FROM videos WHERE id = ?", (video.id,)
        ).fetchone()[0] == 1
    finally:
        restored.close()
    db.close()


def test_deleting_a_recording_spares_the_frames_surviving_hands_point_at(
    tmp_path: Path,
) -> None:
    """``actions.source_image`` points into ``frames/cv_job_<id>/``.

    Hands imported from a recording outlive the recording on purpose, so that
    directory holds evidence they still reference. Deleting it would be the one
    thing retention forbids at any age: removing a file the product still expects.
    Retention owns that folder because only it can tell a referenced frame from an
    unreferenced one.
    """
    db = PokerDatabase(str(tmp_path / "spare-frames.sqlite3"))
    db.init_db()
    session = db.create_session(Session(name="Keep"))
    assert session.id is not None
    video = db.create_video(
        _video(session.id, "keep.mp4", tmp_path / "videos" / "keep.mp4")
    )
    assert video.id is not None
    job = db.create_processing_job(
        ProcessingJob(job_type="cv_reconstruction", status="completed", video_id=video.id)
    )
    assert job.id is not None

    timeline = next(
        item for item in cv_job_artifact_paths(job.id) if item.name.endswith("_timeline.json")
    )
    cv_frames = timeline.parent.parent / "frames" / f"cv_job_{job.id}"
    cv_frames.mkdir(parents=True, exist_ok=True)
    referenced = cv_frames / "frame_000001.jpg"
    referenced.write_bytes(b"jpeg")

    error, _ = app_module.delete_video_and_artifacts(db, video.id)

    assert error is None, error
    assert referenced.is_file(), (
        "deleted a frame a surviving hand's action still names"
    )
    db.close()


def test_a_recording_is_not_deleted_when_no_rollback_point_can_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing is the whole point of taking the snapshot first."""
    db = PokerDatabase(str(tmp_path / "no-snapshot.sqlite3"))
    db.init_db()
    session = db.create_session(Session(name="Blocked"))
    assert session.id is not None
    stored = tmp_path / "videos" / "blocked.mp4"
    video = db.create_video(_video(session.id, "blocked.mp4", stored))
    assert video.id is not None

    monkeypatch.setattr(
        app_module,
        "snapshot_before_destructive",
        lambda *args, **kwargs: (None, "disk full"),
    )
    error, snapshot = app_module.delete_video_and_artifacts(db, video.id)

    assert snapshot is None
    assert error == "disk full"
    assert db.fetch_video(video.id) is not None, "deleted without a rollback point"
    assert stored.is_file()
    db.close()


def test_a_recording_whose_job_is_still_starting_up_is_refused(
    tmp_path: Path,
) -> None:
    """The launch window, where cancelling reports success and stops nothing.

    A job's pid is recorded AFTER the worker is spawned. Inside that window
    ``cancel_processing_job`` takes its falsy-pid branch, terminates nothing, and
    still returns ``cancelled``. Proceeding on that answer leaves a detached
    worker writing frames and a timeline for a recording whose rows are gone --
    and on POSIX its open descriptor on the unlinked file lets it finish. A live
    job with no pid is therefore indeterminate, and indeterminate is a refusal.
    """
    db = PokerDatabase(str(tmp_path / "starting.sqlite3"))
    db.init_db()
    session = db.create_session(Session(name="Racing"))
    assert session.id is not None
    stored = tmp_path / "videos" / "racing.mp4"
    video = db.create_video(_video(session.id, "racing.mp4", stored))
    assert video.id is not None
    db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction", status="queued", video_id=video.id, pid=None
        )
    )

    error, snapshot = app_module.delete_video_and_artifacts(db, video.id)

    assert error is not None
    assert "starting up" in error, error
    assert snapshot is not None, "the refusal still reports the point it snapshotted"
    assert db.fetch_video(video.id) is not None, "raced a worker that was still launching"
    assert stored.is_file()
    db.close()


# ---------------------------------------------------------------------------
# Deleting hands in a batch
# ---------------------------------------------------------------------------


def test_a_batch_of_hands_is_deleted_under_exactly_one_snapshot(
    tmp_path: Path,
) -> None:
    """One snapshot for the batch, not one per hand.

    ``predelete`` keeps a fixed number of snapshots and rotates the pool by class,
    not by scope, so six per-hand copies would evict every other rollback point in
    it -- including the session delete from minutes earlier. A batch is one
    deletion with one pre-state, which is exactly what the rule in
    ``snapshot_before_destructive`` distinguishes from reusing a snapshot across
    two separate deletions.
    """
    path = tmp_path / "batch.sqlite3"
    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Bulk"))
    assert session.id is not None
    hand_ids = [db.create_hand(_hand(session.id, number)).id for number in range(1, 7)]
    assert all(hand_id is not None for hand_id in hand_ids)

    backups = backups_dir_for(path)
    deleted, failures, snapshot = app_module.delete_hands_and_artifacts(
        db, [hand_id for hand_id in hand_ids if hand_id is not None], session_id=session.id
    )

    assert failures == []
    assert sorted(deleted) == sorted(hand_id for hand_id in hand_ids if hand_id is not None)
    assert snapshot is not None
    assert db.fetch_hands_by_session(session.id) == []

    batch_snapshots = find_snapshots(
        backups, purpose="predelete", scope=f"session{session.id}hands"
    )
    assert len(batch_snapshots) == 1, (
        f"six hands produced {len(batch_snapshots)} rollback points, not one"
    )

    restored = sqlite3.connect(str(batch_snapshots[0]))
    try:
        assert restored.execute(
            "SELECT COUNT(*) FROM hands WHERE session_id = ?", (session.id,)
        ).fetchone()[0] == 6, "the rollback point does not hold the deleted hands"
    finally:
        restored.close()
    db.close()


def test_a_batch_deletes_what_it_can_and_names_what_it_could_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused hand must not abort the batch silently.

    One snapshot covers every hand equally, so stopping halfway would leave a
    partial delete AND no report of which hands survived -- the operator would have
    to diff the list to find out.
    """
    db = PokerDatabase(str(tmp_path / "partial.sqlite3"))
    db.init_db()
    session = db.create_session(Session(name="Partial"))
    assert session.id is not None
    hands = [db.create_hand(_hand(session.id, number)) for number in range(1, 5)]
    stubborn = hands[1].id
    assert stubborn is not None

    real = app_module._stop_and_clear_solver_runs

    def _refuse_one(db_arg: PokerDatabase, hand_id: int) -> str | None:
        if hand_id == stubborn:
            return "The active solver could not be stopped yet."
        return real(db_arg, hand_id)

    monkeypatch.setattr(app_module, "_stop_and_clear_solver_runs", _refuse_one)

    deleted, failures, snapshot = app_module.delete_hands_and_artifacts(
        db,
        [hand.id for hand in hands if hand.id is not None],
        session_id=session.id,
    )

    assert snapshot is not None
    assert len(deleted) == 3
    assert stubborn not in deleted
    assert [hand_id for hand_id, _ in failures] == [stubborn]
    assert "could not be stopped" in failures[0][1]

    surviving = [hand.id for hand in db.fetch_hands_by_session(session.id)]
    assert surviving == [stubborn]
    db.close()


def test_no_hand_is_deleted_against_a_snapshot_that_vanished(tmp_path: Path) -> None:
    """The required ``snapshot`` keyword is read, not decorative.

    A source scan can count call sites but cannot see that one of them passed a
    rollback point which is no longer there. Checking the file makes "recoverable
    by construction" a property of the code rather than of whoever writes the next
    caller.
    """
    db = PokerDatabase(str(tmp_path / "vanished.sqlite3"))
    db.init_db()
    session = db.create_session(Session(name="Gone"))
    assert session.id is not None
    hand = db.create_hand(_hand(session.id, 1))
    assert hand.id is not None

    error = app_module._remove_hand_and_artifacts(
        db, hand.id, snapshot=tmp_path / "never-written.sqlite3"
    )

    assert error is not None
    assert "no longer on disk" in error
    assert db.fetch_hand(hand.id) is not None, "deleted against a missing rollback point"
    db.close()
