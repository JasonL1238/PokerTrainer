from __future__ import annotations

from datetime import datetime, timezone

from poker_tracker.db import PokerDatabase
from poker_tracker.models import ProcessingJob


def create_job(db: PokerDatabase, video_id: int, job_type: str = "frame_extraction") -> ProcessingJob:
    """Create a queued local processing job."""
    return db.create_processing_job(ProcessingJob(video_id=video_id, job_type=job_type))


def mark_running(db: PokerDatabase, job_id: int, message: str = "Running") -> None:
    db.update_processing_job(
        job_id,
        status="running",
        progress_percent=0,
        message=message,
        started_at=_now(),
    )


def update_progress(db: PokerDatabase, job_id: int, progress_percent: float, message: str) -> None:
    db.update_processing_job(
        job_id,
        progress_percent=max(0, min(100, progress_percent)),
        message=message,
    )


def mark_completed(db: PokerDatabase, job_id: int, message: str = "Completed") -> None:
    db.update_processing_job(
        job_id,
        status="completed",
        progress_percent=100,
        message=message,
        completed_at=_now(),
    )


def mark_failed(db: PokerDatabase, job_id: int, error_message: str) -> None:
    db.update_processing_job(
        job_id,
        status="failed",
        message="Failed",
        error_message=error_message,
        completed_at=_now(),
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)
