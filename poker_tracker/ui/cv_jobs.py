"""Launch and reconcile offline CV reconstruction subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import ProcessingJob
from poker_tracker.ui.jobs import mark_failed
from poker_tracker.ui.video_ingest import (
    assert_stored_video_matches_record,
    require_playable_video,
    resolve_stored_video_path,
)
from poker_tracker.ui.video_storage import JOB_LOGS_DIR, VIDEOS_DIR, ensure_data_directories

DEFAULT_STALE_AFTER = timedelta(minutes=15)


class CVJobAlreadyRunningError(RuntimeError):
    """Raised when the single-job policy rejects another launch."""


def start_cv_job(
    db: PokerDatabase,
    video_id: int,
    video_path: str | Path,
    session_name: str,
    target_session_id: int | None = None,
) -> ProcessingJob:
    """Queue and detach one full offline reconstruction job."""
    if not session_name.strip():
        raise ValueError("Session name is required.")

    video = db.fetch_video(video_id)
    if video is None:
        raise ValueError(f"Video #{video_id} was not found.")
    path = resolve_stored_video_path(
        video_path,
        stored_path=video.stored_path,
        videos_dir=VIDEOS_DIR,
    )
    expected_hash = (video.content_sha256 or "").strip() or None
    assert_stored_video_matches_record(
        path,
        expected_size_bytes=video.file_size_bytes,
        expected_sha256=expected_hash,
    )
    require_playable_video(path)

    with db.transaction(immediate=True):
        active = db.fetch_active_jobs()
        if active:
            current = active[0]
            raise CVJobAlreadyRunningError(
                f"Job #{current.id} is already {current.status}. "
                "Wait for it to finish or fail."
            )
        active_solver = db.fetch_active_solver_runs()
        if active_solver:
            current = active_solver[0]
            raise CVJobAlreadyRunningError(
                f"Solver run #{current.id} is already {current.status}. "
                "Wait for it to finish or cancel it."
            )
        job = db.create_processing_job(
            ProcessingJob(
                video_id=video_id,
                job_type="cv_reconstruction",
                status="queued",
                message="Waiting for worker",
            )
        )
    if job.id is None:
        raise RuntimeError("The processing job could not be saved.")

    try:
        ensure_data_directories()
        JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = JOB_LOGS_DIR / f"cv_job_{job.id}.log"
        command = [
            sys.executable,
            "-m",
            "poker_tracker.ui.run_cv_job",
            "--job-id",
            str(job.id),
            "--video",
            str(path),
            "--session-name",
            session_name.strip(),
            "--db",
            db.db_path,
        ]
        if target_session_id is not None:
            command.extend(["--target-session-id", str(target_session_id)])
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except Exception as exc:
        mark_failed(db, job.id, f"Could not start reconstruction worker: {exc}")
        raise RuntimeError("Could not start the reconstruction worker.") from exc

    now = _now()
    db.update_processing_job(
        job.id,
        status="running",
        pid=process.pid,
        heartbeat_at=now,
        started_at=now,
        progress_percent=1,
        message="Worker started",
    )
    saved = db.fetch_processing_job(job.id)
    if saved is None:
        raise RuntimeError("The launched processing job could not be reloaded.")
    return saved


def reconcile_stuck_jobs(
    db: PokerDatabase,
    *,
    stale_after: timedelta = DEFAULT_STALE_AFTER,
    now: datetime | None = None,
) -> list[int]:
    """Fail running jobs whose worker is gone or heartbeat has expired."""
    current = now or _now()
    reconciled: list[int] = []
    for job in db.fetch_running_jobs():
        if job.id is None:
            continue
        reference = job.heartbeat_at or job.started_at or job.created_at
        stale = current - _as_utc(reference) > stale_after
        dead = job.pid is None or not _pid_is_alive(job.pid)
        if stale and not dead and not _terminate_job_group(job.pid):
            continue
        if dead or stale:
            reason = "worker process is no longer running" if dead else "worker heartbeat expired"
            mark_failed(db, job.id, f"Orphaned after restart: {reason}.")
            reconciled.append(job.id)
    return reconciled


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_job_group(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    except PermissionError:
        return False
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return True
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
    except PermissionError:
        return False
    return True


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)
