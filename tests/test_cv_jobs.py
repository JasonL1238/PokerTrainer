from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import ProcessingJob, VideoRecord
from poker_tracker.ui import cv_jobs
from poker_tracker.ui import run_cv_job
from poker_tracker.ui.run_cv_job import BACKUP_KEEP_COUNT, backup_database


def make_db(path: str = ":memory:") -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def add_video(db: PokerDatabase, path: Path) -> VideoRecord:
    path.write_bytes(b"completed-session-video")
    return db.create_video(
        VideoRecord(
            original_filename=path.name,
            stored_path=str(path),
            file_size_bytes=path.stat().st_size,
        )
    )


def test_start_cv_job_launches_detached_worker_and_enforces_single_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=43210)

    monkeypatch.setattr(cv_jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cv_jobs, "JOB_LOGS_DIR", tmp_path / "logs")
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


def test_reconcile_stuck_jobs_marks_dead_pid_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    now = datetime.now(timezone.utc)
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


def test_reconcile_stuck_jobs_marks_stale_live_worker_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    heartbeat = datetime.now(timezone.utc) - timedelta(minutes=20)
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

    reconciled = cv_jobs.reconcile_stuck_jobs(
        db,
        now=datetime.now(timezone.utc),
        stale_after=timedelta(minutes=15),
    )

    assert reconciled == [job.id]
    assert "heartbeat expired" in db.fetch_processing_job(job.id).error_message
    db.close()


def test_reconcile_keeps_live_fresh_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_db()
    video = add_video(db, tmp_path / "session.mp4")
    now = datetime.now(timezone.utc)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracker.sqlite3"
    db = make_db(str(db_path))
    video = add_video(db, tmp_path / "session.mp4")
    job = db.create_processing_job(
        ProcessingJob(video_id=video.id, job_type="cv_reconstruction", status="running")
    )
    db.close()
    paths = {
        "cv_timelines": tmp_path / "timelines",
        "exports": tmp_path / "exports",
        "backups": tmp_path / "backups",
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
