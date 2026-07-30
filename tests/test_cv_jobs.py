from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import ProcessingJob, VideoRecord
from poker_tracker.ui import cv_jobs, run_cv_job
from poker_tracker.ui.run_cv_job import BACKUP_KEEP_COUNT, backup_database
from poker_tracker.ui.video_ingest import sha256_file


def make_db(path: str = ":memory:") -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def create_synthetic_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(20):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 10 % 255, index * 5 % 255, index * 20 % 255)
        writer.write(frame)
    writer.release()
    return path


def add_video(
    db: PokerDatabase,
    path: Path,
    *,
    playable: bool = False,
    with_hash: bool = False,
) -> VideoRecord:
    if playable:
        create_synthetic_video(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"completed-session-video")
    digest = sha256_file(path) if with_hash else ""
    return db.create_video(
        VideoRecord(
            original_filename=path.name,
            stored_path=str(path),
            file_size_bytes=path.stat().st_size,
            content_sha256=digest,
        )
    )


@pytest.fixture
def videos_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "videos"
    root.mkdir()
    monkeypatch.setattr(cv_jobs, "VIDEOS_DIR", root)
    monkeypatch.setattr(run_cv_job, "VIDEOS_DIR", root)
    return root


def test_start_cv_job_launches_detached_worker_and_enforces_single_job(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=43210)

    monkeypatch.setattr(cv_jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cv_jobs, "JOB_LOGS_DIR", videos_dir.parent / "logs")
    monkeypatch.setattr(cv_jobs, "ensure_data_directories", lambda: {})

    job = cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")

    assert job.job_type == "cv_reconstruction"
    assert job.status == "running"
    assert job.pid == 43210
    assert job.heartbeat_at is not None
    assert captured["kwargs"]["start_new_session"] is True
    assert "poker_tracker.ui.run_cv_job" in captured["command"]

    with pytest.raises(cv_jobs.CVJobAlreadyRunningError):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Another")
    db.close()


def test_start_cv_job_rejects_size_mismatched_video(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    Path(video.stored_path).write_bytes(b"truncated")
    monkeypatch.setattr(cv_jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=1))

    with pytest.raises(ValueError, match="size mismatch"):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")

    assert db.fetch_active_jobs() == []
    db.close()


def test_start_cv_job_rejects_hash_mismatched_same_size(videos_dir: Path) -> None:
    db = make_db()
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    path = Path(video.stored_path)
    original = path.read_bytes()
    # Same-size mutation: flip first byte.
    mutated = bytes((original[0] ^ 0xFF,)) + original[1:]
    assert len(mutated) == len(original)
    path.write_bytes(mutated)

    with pytest.raises(ValueError, match="hash mismatch"):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")

    assert db.fetch_active_jobs() == []
    db.close()


def test_start_cv_job_rejects_path_not_matching_stored(videos_dir: Path) -> None:
    db = make_db()
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    other = create_synthetic_video(videos_dir / "other.avi")

    with pytest.raises(ValueError, match="does not match"):
        cv_jobs.start_cv_job(db, video.id, other, "Imported study")

    assert db.fetch_active_jobs() == []
    db.close()


def test_start_cv_job_rejects_out_of_root_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos_root = tmp_path / "videos"
    videos_root.mkdir()
    monkeypatch.setattr(cv_jobs, "VIDEOS_DIR", videos_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    db = make_db()
    video = add_video(db, outside / "session.avi", playable=True, with_hash=True)

    with pytest.raises(ValueError, match="outside video storage root"):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")

    assert db.fetch_active_jobs() == []
    db.close()


def test_start_cv_job_rejects_corrupt_video(videos_dir: Path) -> None:
    db = make_db()
    path = videos_dir / "corrupt.mp4"
    path.write_bytes(b"not-a-video")
    video = db.create_video(
        VideoRecord(
            original_filename=path.name,
            stored_path=str(path),
            file_size_bytes=path.stat().st_size,
            content_sha256=sha256_file(path),
        )
    )

    with pytest.raises(ValueError):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")

    assert db.fetch_active_jobs() == []
    db.close()


def test_reconcile_stuck_jobs_marks_dead_pid_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    now = datetime.now(UTC)
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="running",
            video_id=video.id,
            pid=99999,
            heartbeat_at=now,
            started_at=now,
        )
    )
    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: False)

    assert cv_jobs.reconcile_stuck_jobs(db, now=now) == [job.id]
    saved = db.fetch_processing_job(job.id)
    assert saved.status == "failed"
    assert "no longer running" in saved.error_message
    db.close()


def test_cv_setup_failure_does_not_leave_queued_job(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    monkeypatch.setattr(
        cv_jobs,
        "ensure_data_directories",
        lambda: (_ for _ in ()).throw(PermissionError("read only")),
    )

    with pytest.raises(RuntimeError, match="Could not start"):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")

    jobs = db.fetch_jobs_by_video(video.id)
    assert len(jobs) == 1
    assert jobs[0].status == "failed"
    assert db.fetch_active_jobs() == []
    db.close()


def test_reconcile_stuck_jobs_marks_stale_live_worker_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    heartbeat = datetime.now(UTC) - timedelta(minutes=20)
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="running",
            video_id=video.id,
            pid=123,
            heartbeat_at=heartbeat,
            started_at=heartbeat,
        )
    )
    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: True)
    terminated: list[int] = []
    monkeypatch.setattr(
        cv_jobs,
        "_terminate_job_group",
        lambda pid: terminated.append(pid) or True,
    )

    reconciled = cv_jobs.reconcile_stuck_jobs(
        db,
        now=datetime.now(UTC),
        stale_after=timedelta(minutes=15),
    )

    assert reconciled == [job.id]
    assert terminated == [123]
    assert "heartbeat expired" in db.fetch_processing_job(job.id).error_message
    db.close()


def test_reconcile_keeps_live_fresh_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    now = datetime.now(UTC)
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="running",
            video_id=video.id,
            pid=123,
            heartbeat_at=now,
            started_at=now,
        )
    )
    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: True)

    assert cv_jobs.reconcile_stuck_jobs(db, now=now) == []
    assert db.fetch_processing_job(job.id).status == "running"
    db.close()


def test_database_backup_is_consistent_and_rotates(tmp_path: Path) -> None:
    db_path = tmp_path / "tracker.sqlite3"
    db = make_db(str(db_path))
    db.close()
    backup_dir = tmp_path / "backups"

    for _ in range(BACKUP_KEEP_COUNT + 2):
        backup_database(db_path, backup_dir)

    backups = list(backup_dir.glob("poker_tracker_*.sqlite3"))
    assert len(backups) == BACKUP_KEEP_COUNT
    restored = PokerDatabase(backups[0])
    assert restored.schema_version() > 0
    restored.close()


def test_worker_completes_import_and_records_backup(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = videos_dir.parent / "tracker.sqlite3"
    db = make_db(str(db_path))
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    job = db.create_processing_job(
        ProcessingJob(video_id=video.id, job_type="cv_reconstruction", status="running")
    )
    db.close()
    paths = {
        "cv_timelines": videos_dir.parent / "timelines",
        "exports": videos_dir.parent / "exports",
        "backups": videos_dir.parent / "backups",
    }
    for path in paths.values():
        path.mkdir()
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda: paths)
    monkeypatch.setattr(run_cv_job, "_run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_cv_job,
        "export_timeline",
        lambda *args, **kwargs: {"cv_import_summary": {"exported_hands": 2}},
    )
    monkeypatch.setattr(run_cv_job, "import_session", lambda db, payload: SimpleNamespace(id=55))

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(video.stored_path),
        session_name="Reconstructed",
        db_path=db_path,
    )

    checked = PokerDatabase(db_path)
    saved = checked.fetch_processing_job(job.id)
    assert exit_code == 0
    assert saved.status == "completed"
    assert saved.progress_percent == 100
    assert "Imported 2 hands" in saved.message
    assert len(list(paths["backups"].glob("*.sqlite3"))) == 1
    checked.close()


def test_read_pipeline_progress_handles_live_snapshot_and_startup_race(
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "job_progress.json"

    assert run_cv_job._read_pipeline_progress(progress_path) is None

    progress_path.write_text(
        '{"stage": "frames", "current": 7, "total": 20}',
        encoding="utf-8",
    )
    assert run_cv_job._read_pipeline_progress(progress_path) == (7, 20, "frames")

    progress_path.write_text(
        '{"stage": "frames", "current": 30, "total": 20}',
        encoding="utf-8",
    )
    assert run_cv_job._read_pipeline_progress(progress_path) == (20, 20, "frames")


def test_backup_helper_is_reexported_from_run_cv_job() -> None:
    """db.py owns the migration snapshot now; the CV worker import must keep working."""
    from poker_tracker.persistence import backup as backup_module

    assert backup_database is backup_module.backup_database
    assert BACKUP_KEEP_COUNT == backup_module.BACKUP_KEEP_COUNT


def test_cancel_kills_pipeline_sidecar_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="running",
            video_id=video.id,
            pid=4242,
        )
    )
    timelines = tmp_path / "timelines"
    timelines.mkdir()
    pid_path = timelines / f"job_{job.id}_pipeline.pid"
    pid_path.write_text("7777\n", encoding="utf-8")
    monkeypatch.setattr(cv_jobs, "CV_TIMELINES_DIR", timelines)
    killed: list[int] = []

    def fake_terminate(pid):
        killed.append(pid)
        return True

    monkeypatch.setattr(cv_jobs, "_terminate_job_group", fake_terminate)

    cancelled = cv_jobs.cancel_processing_job(db, job.id)

    assert cancelled.status == "cancelled"
    assert killed == [7777, 4242]
    assert not pid_path.exists()
    db.close()


def test_cancel_processing_job_terminates_and_clears_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="running",
            video_id=video.id,
            pid=4242,
            message="Working",
        )
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        cv_jobs,
        "_terminate_job_group",
        lambda pid: terminated.append(pid) or True,
    )

    cancelled = cv_jobs.cancel_processing_job(db, job.id)

    assert cancelled.status == "cancelled"
    assert cancelled.pid is None
    assert terminated == [4242]
    assert db.fetch_active_jobs() == []
    db.close()


def test_cancel_processing_job_enters_cancelling_when_kill_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="running",
            video_id=video.id,
            pid=4242,
        )
    )
    monkeypatch.setattr(cv_jobs, "_terminate_job_group", lambda pid: False)

    pending = cv_jobs.cancel_processing_job(db, job.id)

    assert pending.status == "cancelling"
    assert pending.pid == 4242
    assert db.fetch_active_jobs()[0].id == job.id
    db.close()


def test_worker_honors_cancel_before_import(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = videos_dir.parent / "tracker.sqlite3"
    db = make_db(str(db_path))
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="cancelling",
            message="Cancelling",
        )
    )
    session_count = len(db.fetch_sessions())
    db.close()
    imported = {"called": False}

    def fake_import(db, payload):
        imported["called"] = True
        return SimpleNamespace(id=99)

    paths = {
        "cv_timelines": videos_dir.parent / "timelines",
        "exports": videos_dir.parent / "exports",
        "backups": videos_dir.parent / "backups",
    }
    for path in paths.values():
        path.mkdir()
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda: paths)
    monkeypatch.setattr(run_cv_job, "_run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_cv_job, "import_session", fake_import)

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(video.stored_path),
        session_name="Reconstructed",
        db_path=db_path,
    )

    checked = PokerDatabase(db_path)
    saved = checked.fetch_processing_job(job.id)
    assert exit_code == 1
    assert saved.status == "cancelled"
    assert imported["called"] is False
    assert len(checked.fetch_sessions()) == session_count
    checked.close()


def test_backup_failure_skips_import(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = videos_dir.parent / "tracker.sqlite3"
    db = make_db(str(db_path))
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    job = db.create_processing_job(
        ProcessingJob(video_id=video.id, job_type="cv_reconstruction", status="running")
    )
    session_count = len(db.fetch_sessions())
    db.close()
    imported = {"called": False}

    paths = {
        "cv_timelines": videos_dir.parent / "timelines",
        "exports": videos_dir.parent / "exports",
        "backups": videos_dir.parent / "backups",
    }
    for path in paths.values():
        path.mkdir()
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda: paths)
    monkeypatch.setattr(run_cv_job, "_run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_cv_job,
        "export_timeline",
        lambda *args, **kwargs: {"cv_import_summary": {"exported_hands": 2}},
    )
    monkeypatch.setattr(
        run_cv_job,
        "backup_database",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        run_cv_job,
        "import_session",
        lambda db, payload: imported.__setitem__("called", True) or SimpleNamespace(id=55),
    )

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(video.stored_path),
        session_name="Reconstructed",
        db_path=db_path,
    )

    checked = PokerDatabase(db_path)
    saved = checked.fetch_processing_job(job.id)
    assert exit_code == 1
    assert saved.status == "failed"
    assert "disk full" in saved.error_message
    assert imported["called"] is False
    assert len(checked.fetch_sessions()) == session_count
    assert saved.pid is None
    checked.close()


def test_start_cv_job_blocked_by_active_solver(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poker_tracker.persistence.models import Hand, Session, SolverRun

    db = make_db()
    session = db.create_session(Session(name="Study"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="running",
            backend_name="texassolver",
            backend_version="test",
            input_hash="abc",
            pid=99,
        )
    )
    monkeypatch.setattr(cv_jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=1))
    monkeypatch.setattr(cv_jobs, "ensure_data_directories", lambda: {})
    monkeypatch.setattr(cv_jobs, "JOB_LOGS_DIR", videos_dir.parent / "logs")

    with pytest.raises(cv_jobs.CVJobAlreadyRunningError, match="Solver run"):
        cv_jobs.start_cv_job(db, video.id, video.stored_path, "Imported study")
    db.close()


def test_reconcile_finalizes_cancelling_when_worker_dead(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    now = datetime.now(UTC)
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="cancelling",
            video_id=video.id,
            pid=99999,
            heartbeat_at=now,
            started_at=now,
        )
    )
    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: False)

    assert cv_jobs.reconcile_stuck_jobs(db, now=now) == [job.id]
    saved = db.fetch_processing_job(job.id)
    assert saved.status == "cancelled"
    assert saved.pid is None
    db.close()


def test_reconcile_keeps_cancelling_as_cancelled_when_stale_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    heartbeat = datetime.now(UTC) - timedelta(minutes=20)
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="cancelling",
            video_id=video.id,
            pid=123,
            heartbeat_at=heartbeat,
            started_at=heartbeat,
        )
    )
    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: True)
    monkeypatch.setattr(cv_jobs, "_terminate_job_group", lambda pid: True)

    assert cv_jobs.reconcile_stuck_jobs(db, now=datetime.now(UTC)) == [job.id]
    saved = db.fetch_processing_job(job.id)
    assert saved.status == "cancelled"
    assert "failed" not in saved.status
    db.close()


def test_mark_completed_does_not_override_cancelled_job(tmp_path: Path) -> None:
    from poker_tracker.ui.jobs import mark_completed

    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="cancelled",
            video_id=video.id,
            message="Cancelled",
            error_message="Cancelled by user.",
        )
    )

    saved = mark_completed(db, job.id, "should not win")
    assert saved.status == "cancelled"
    assert saved.message == "Cancelled"
    db.close()


def test_worker_completes_when_cancel_arrives_during_import(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import is atomic with job completion: late cancel must not leave drafts orphaned."""
    db_path = videos_dir.parent / "tracker.sqlite3"
    db = make_db(str(db_path))
    video = add_video(db, videos_dir / "session.avi", playable=True, with_hash=True)
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="running",
            message="Running",
        )
    )
    db.close()

    paths = {
        "cv_timelines": videos_dir.parent / "timelines",
        "exports": videos_dir.parent / "exports",
        "backups": videos_dir.parent / "backups",
    }
    for path in paths.values():
        path.mkdir()
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda: paths)
    monkeypatch.setattr(run_cv_job, "_run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_cv_job,
        "export_timeline",
        lambda *args, **kwargs: {
            "export_version": 5,
            "session": {
                "name": "Late cancel",
                "date_played": "2026-07-30",
                "platform": "",
                "stakes": "",
                "notes": "",
            },
            "hands": [],
            "cv_import_summary": {"exported_hands": 0},
        },
    )
    monkeypatch.setattr(
        run_cv_job,
        "backup_database",
        lambda *args, **kwargs: paths["backups"] / "backup.sqlite3",
    )

    real_import = run_cv_job.import_session

    def import_then_mark_cancelling(database, payload):
        database.update_processing_job(
            job.id,
            expected_statuses=("running",),
            status="cancelling",
            message="Cancelling",
        )
        return real_import(database, payload)

    monkeypatch.setattr(run_cv_job, "import_session", import_then_mark_cancelling)

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(video.stored_path),
        session_name="Late cancel",
        db_path=db_path,
    )

    checked = PokerDatabase(db_path)
    saved = checked.fetch_processing_job(job.id)
    assert exit_code == 0
    assert saved.status == "completed"
    assert "cancel arrived after import" in saved.message
    assert checked.fetch_video(video.id).session_id is not None
    checked.close()


def test_terminate_job_group_returns_false_when_pid_stays_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: True)
    monkeypatch.setattr(cv_jobs.os, "killpg", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv_jobs.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(cv_jobs.time, "sleep", lambda *_: None)

    assert cv_jobs._terminate_job_group(12345) is False


def test_reconcile_fails_stale_queued_job_without_worker(tmp_path: Path) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    created = datetime.now(UTC) - timedelta(minutes=30)
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="queued",
            video_id=video.id,
            created_at=created,
            message="Waiting for worker",
        )
    )

    reconciled = cv_jobs.reconcile_stuck_jobs(
        db,
        now=datetime.now(UTC),
        stale_after=timedelta(minutes=15),
    )

    assert reconciled == [job.id]
    assert db.fetch_processing_job(job.id).status == "failed"
    db.close()
