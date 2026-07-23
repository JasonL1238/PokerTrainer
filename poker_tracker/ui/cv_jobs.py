"""Launch and reconcile offline CV reconstruction subprocesses."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import ProcessingJob
from poker_tracker.ui.jobs import mark_failed
from poker_tracker.ui.video_storage import JOB_LOGS_DIR, ensure_data_directories


DEFAULT_STALE_AFTER = timedelta(minutes=15)


class CVJobAlreadyRunningError(RuntimeError):
    """Raised when the single-job policy rejects another launch."""


def start_cv_job(
    db: PokerDatabase,
    video_id: int,
    video_path: str | Path,
    session_name: str,
) -> ProcessingJob:
    """Queue and detach one full offline reconstruction job."""
    active = db.fetch_active_jobs()
    if active:
        current = active[0]
        raise CVJobAlreadyRunningError(
            f"Job #{current.id} is already {current.status}. Wait for it to finish or fail."
        )
    path = Path(video_path)
    if not path.is_file():
        raise ValueError(f"Video file not found: {path}")
    if not session_name.strip():
        raise ValueError("Session name is required.")

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
    try:
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
    except OSError as exc:
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


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)
