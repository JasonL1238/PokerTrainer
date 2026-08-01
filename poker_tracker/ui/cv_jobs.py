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
from poker_tracker.ui.jobs import job_log_path, mark_cancelled, mark_failed
from poker_tracker.ui.video_ingest import (
    assert_stored_video_matches_record,
    require_playable_video,
    resolve_stored_video_path,
)
from poker_tracker.ui.video_storage import (
    CV_TIMELINES_DIR,
    VIDEOS_DIR,
    ensure_data_directories,
)

DEFAULT_STALE_AFTER = timedelta(minutes=15)
_LIVE_STATUSES = frozenset({"queued", "running", "cancelling"})


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
        log_path = job_log_path(job.id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
    saved = db.update_processing_job(
        job.id,
        expected_statuses=("queued",),
        status="running",
        pid=process.pid,
        heartbeat_at=now,
        started_at=now,
        progress_percent=1,
        message="Worker started",
    )
    if saved.status != "running" or saved.pid != process.pid:
        _terminate_job_group(process.pid)
        _terminate_pipeline_sidecar(job.id)
        refreshed = db.fetch_processing_job(job.id)
        if refreshed is not None and refreshed.status in {"cancelling", "cancelled"}:
            if refreshed.status == "cancelling":
                return mark_cancelled(
                    db,
                    job.id,
                    "Cancelled while the reconstruction worker was launching.",
                )
            return refreshed
        raise ValueError(
            "The reconstruction job changed while the worker was launching; "
            "the worker was stopped."
        )
    refreshed = db.fetch_processing_job(job.id)
    if refreshed is None:
        raise RuntimeError("The launched processing job could not be reloaded.")
    if refreshed.status in {"cancelling", "cancelled"}:
        _terminate_job_group(process.pid)
        _terminate_pipeline_sidecar(job.id)
        if refreshed.status == "cancelling":
            return mark_cancelled(
                db,
                job.id,
                "Cancelled while the reconstruction worker was launching.",
            )
        return refreshed
    return refreshed


def cancel_processing_job(db: PokerDatabase, job_id: int) -> ProcessingJob:
    """Request cancellation of one active CV/processing job."""
    job = db.fetch_processing_job(job_id)
    if job is None:
        raise ValueError("Processing job not found.")
    if job.status not in _LIVE_STATUSES:
        return job
    _terminate_pipeline_sidecar(job_id)
    if job.pid and not _terminate_job_group(job.pid):
        return db.update_processing_job(
            job_id,
            expected_statuses=("queued", "running", "cancelling"),
            status="cancelling",
            message="Cancelling",
            error_message="Cancellation requested; waiting for the worker to stop.",
        )
    return db.update_processing_job(
        job_id,
        expected_statuses=("queued", "running", "cancelling"),
        status="cancelled",
        message="Cancelled",
        error_message="Cancelled by user.",
        clear_pid=True,
        completed_at=_now(),
    )


def reconcile_stuck_jobs(
    db: PokerDatabase,
    *,
    stale_after: timedelta = DEFAULT_STALE_AFTER,
    now: datetime | None = None,
) -> list[int]:
    """Fail or finalize jobs whose worker is gone or heartbeat has expired."""
    current = now or _now()
    reconciled: list[int] = []
    for job in db.fetch_active_jobs():
        if job.id is None:
            continue
        reference = job.heartbeat_at or job.started_at or job.created_at
        stale = current - _as_utc(reference) > stale_after
        alive = _pid_is_alive(job.pid)
        if job.status == "queued" and not stale:
            continue
        if not stale and alive:
            continue
        if job.status == "cancelling":
            if alive and not _terminate_job_group(job.pid):
                continue
            mark_cancelled(
                db,
                job.id,
                "Cancelled after the reconstruction worker stopped.",
            )
            reconciled.append(job.id)
            continue
        if stale and alive and not _terminate_job_group(job.pid):
            continue
        reason = (
            "worker process is no longer running"
            if not alive
            else "worker heartbeat expired"
        )
        mark_failed(db, job.id, f"Orphaned after restart: {reason}.")
        reconciled.append(job.id)
    return reconciled


def _terminate_pipeline_sidecar(job_id: int) -> None:
    """Kill a detached pipeline child recorded next to the progress snapshot."""
    pid_path = CV_TIMELINES_DIR / f"job_{job_id}_pipeline.pid"
    try:
        raw = pid_path.read_text(encoding="utf-8").strip()
        pipeline_pid = int(raw)
    except (OSError, ValueError):
        return
    _terminate_job_group(pipeline_pid)
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_job_group(pid: int | None) -> bool:
    """Send TERM/KILL to a process group; True only when the PID is gone."""
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
    deadline = time.monotonic() + 2.0
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
            return not _pid_is_alive(pid)
        except PermissionError:
            return False
    except PermissionError:
        return False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_is_alive(pid)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)
