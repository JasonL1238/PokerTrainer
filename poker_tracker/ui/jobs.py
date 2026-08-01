"""Create and advance local processing jobs, and report what they left behind."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from poker_tracker.persistence.completion import parse_completion_evidence
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Hand, ProcessingJob
from poker_tracker.safety.redaction import redact_text
from poker_tracker.ui import video_storage
from poker_tracker.ui.reconstruction_review import job_id_from_hand_notes
from poker_tracker.ui.view_models import (
    JobOutcome,
    build_job_outcome,
    job_is_live,
    job_stopped_without_success,
)

# How much of the worker's stdout a failure panel is willing to show, and how
# far back it will seek to find it. A reconstruction log runs to megabytes; the
# explanation of why it stopped is at the end of it.
LOG_TAIL_LINES = 25
_LOG_TAIL_BYTES = 64 * 1024


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


def mark_completed(db: PokerDatabase, job_id: int, message: str = "Completed") -> ProcessingJob:
    return db.update_processing_job(
        job_id,
        expected_statuses=("running",),
        status="completed",
        progress_percent=100,
        message=message,
        clear_pid=True,
        completed_at=_now(),
    )


def mark_failed(db: PokerDatabase, job_id: int, error_message: str) -> ProcessingJob:
    return db.update_processing_job(
        job_id,
        expected_statuses=("queued", "running", "cancelling"),
        status="failed",
        message="Failed",
        error_message=error_message,
        clear_pid=True,
        completed_at=_now(),
    )


def mark_cancelled(
    db: PokerDatabase,
    job_id: int,
    error_message: str = "Cancelled by user.",
) -> ProcessingJob:
    return db.update_processing_job(
        job_id,
        expected_statuses=("queued", "running", "cancelling"),
        status="cancelled",
        message="Cancelled",
        error_message=error_message,
        clear_pid=True,
        completed_at=_now(),
    )


def job_log_path(job_id: int) -> Path:
    """Where the detached worker's output for this job is written.

    One definition, resolved from the job's own id, used both by the launcher
    that opens the file and by the failure panel that offers it. A second copy
    is how the single artifact that explains a failure ends up pointing at a
    path nothing ever wrote. ``video_storage`` is read through the module so a
    redirected data root reaches the reader and the writer alike.
    """
    return video_storage.JOB_LOGS_DIR / f"cv_job_{job_id}.log"


def read_job_log_tail(job_id: int, *, max_lines: int = LOG_TAIL_LINES) -> tuple[str, ...]:
    """The last lines of a job's log, already scrubbed of credentials.

    Redacted here as well as in ``build_job_outcome`` because a log tail is
    exactly where a provider key surfaces, and the raw text should not outlive
    the read that produced it.
    """
    path = job_log_path(job_id)
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _LOG_TAIL_BYTES))
            raw = handle.read()
    except OSError:
        return ()
    lines = [
        line.rstrip()
        for line in raw.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return tuple(redact_text(line) for line in lines[-max_lines:])


def hands_committed_by_job(db: PokerDatabase, job: ProcessingJob) -> list[Hand]:
    """The study hands that carry this reconstruction job's stamp.

    Attribution matches the import service: the timeline identity stamped into
    ``completion_evidence`` survives notes edits, and the notes marker covers
    hands landed before that stamp existed.
    """
    if job.id is None:
        return []
    video = db.fetch_video(job.video_id)
    if video is None or video.session_id is None:
        return []
    identity_key = _timeline_identity_key()
    return [
        hand
        for hand in db.fetch_hands_by_session(video.session_id)
        if _hand_belongs_to_job(hand, job.id, identity_key)
    ]


def committed_work_count(db: PokerDatabase, job: ProcessingJob) -> int | None:
    """How much of this job's output actually reached the database.

    ``None`` when this build cannot answer for the job type. Returning zero
    there would state "nothing was imported" without having checked, and an
    unearned reassurance is the same defect as an unearned percentage.
    """
    if job.id is None:
        return None
    if job.job_type == "cv_reconstruction":
        return len(hands_committed_by_job(db, job))
    if job.job_type == "frame_extraction":
        return sum(1 for frame in db.fetch_frames_by_video(job.video_id) if frame.job_id == job.id)
    return None


def describe_job_outcome(db: PokerDatabase, job: ProcessingJob) -> JobOutcome:
    """The one statement of a job's state that every surface renders.

    Terminal states are answered by querying what the job committed rather than
    by reading the progress figure it stopped at, and a terminal failure also
    carries the log the worker wrote, because that file is the only record of
    why it stopped and nothing else offers it.
    """
    live = job_is_live(job.status)
    # A live job's panel re-renders every second and reports its own progress;
    # scanning the destination session for committed work on every tick would
    # answer a question nobody has asked yet.
    committed = None if live else committed_work_count(db, job)
    destination = _destination_label(db, job) if committed else ""
    log_path = job_log_path(job.id) if job.id is not None else None
    stopped = job_stopped_without_success(job.status)
    exists = bool(log_path and log_path.exists())
    return build_job_outcome(
        job,
        committed_count=committed,
        destination=destination,
        log_path=str(log_path) if exists else "",
        # The path is worth offering at any time; the tail is only read once
        # there is a failure to explain.
        log_tail=read_job_log_tail(job.id) if exists and stopped and job.id is not None else (),
    )


def describe_job_outcomes(
    db: PokerDatabase, jobs: Iterable[ProcessingJob]
) -> dict[int, JobOutcome]:
    """Outcomes keyed by job id, for the list surfaces that render many at once."""
    return {job.id: describe_job_outcome(db, job) for job in jobs if job.id is not None}


def _destination_label(db: PokerDatabase, job: ProcessingJob) -> str:
    video = db.fetch_video(job.video_id)
    if video is None or video.session_id is None:
        return ""
    session = db.fetch_session(video.session_id)
    if session is None:
        return f"session #{video.session_id}"
    return f"session #{video.session_id} ({redact_text(session.name)})"


def _timeline_identity_key() -> str:
    """The evidence key the import service stamps on every hand it lands.

    Imported on use: ``poker_tracker.services.validated_hand_import`` imports
    back into ``poker_tracker.ui``, so binding it at module scope would close a
    loop between the two packages for the sake of one constant, and would put
    the CV export pipeline in the import graph of every caller of this module.
    """
    from poker_tracker.services.validated_hand_import import CV_TIMELINE_IDENTITY_KEY

    return CV_TIMELINE_IDENTITY_KEY


def _hand_belongs_to_job(hand: Hand, job_id: int, identity_key: str) -> bool:
    evidence = parse_completion_evidence(hand.completion_evidence)
    identity = evidence.extra.get(identity_key)
    if isinstance(identity, dict):
        recorded = identity.get("job_id")
        if recorded is not None:
            try:
                if int(recorded) == job_id:
                    return True
            except (TypeError, ValueError):
                pass
    return job_id_from_hand_notes(hand.notes) == job_id


def _now() -> datetime:
    return datetime.now(UTC)
