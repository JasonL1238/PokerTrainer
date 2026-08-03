"""Detached worker for completed-session video reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import export_timeline
from poker_tracker.maintenance.data_health import verify_snapshot
from poker_tracker.persistence.backup import BACKUP_KEEP_COUNT, backup_database
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Session
from poker_tracker.runtime.limits import (
    bounded_int_from_env,
    format_gb,
    memory_limit_bytes_from_env,
    memory_limiter,
)
from poker_tracker.safety.redaction import safe_error_message

# _pid_is_alive is the reconciler's own liveness predicate. Admission and
# reconciliation have to agree on what "the worker is still there" means, and
# two copies of that answer drift.
from poker_tracker.ui.cv_jobs import _pid_is_alive
from poker_tracker.ui.jobs import mark_cancelled, mark_failed, update_progress
from poker_tracker.ui.video_ingest import (
    assert_stored_video_matches_record,
    require_playable_video,
    resolve_stored_video_path,
)
from poker_tracker.ui.video_storage import VIDEOS_DIR, ensure_data_directories

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "cv_lab" / "scripts" / "pipeline" / "run_two_model_pipeline.py"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
HEARTBEAT_INTERVAL_SECONDS = 2

# The two bounds this worker is allowed to be given. Both are read here rather
# than passed down from cv_jobs.start_cv_job, because the worker is also a
# public entrypoint (`python -m poker_tracker.ui.run_cv_job`) and a bound only
# the UI path applies is not a bound.
CV_TIMEOUT_ENV_VAR = "POKERTRAINER_CV_TIMEOUT_SECONDS"
CV_MEMORY_ENV_VAR = "POKERTRAINER_CV_MEMORY_GB"
# A minute is below any real reconstruction and a day is above any recording the
# ingest limits accept, so a value outside the range is a typo, not a choice.
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 24 * 60 * 60

# The statuses a row can be in and still receive this run's outcome. The terminal
# write below carries an optimistic status guard and update_processing_job answers
# a refused guard by returning the unchanged row rather than raising, so a worker
# started on a row outside this set does an hour of CV that nothing can record.
RUNNABLE_JOB_STATUSES = frozenset({"queued", "running"})

# backup_database now lives in the persistence package so db.py can snapshot
# before a migration; re-exported here because callers and tests import it from
# this module.
__all__ = [
    "BACKUP_KEEP_COUNT",
    "CV_MEMORY_ENV_VAR",
    "CV_TIMEOUT_ENV_VAR",
    "CVJobLimits",
    "backup_database",
    "main",
    "resolve_limits",
    "run_job",
]


@dataclass(frozen=True)
class CVJobLimits:
    """The bounds one reconstruction runs under, after validation."""

    timeout_seconds: int
    memory_limit_bytes: int | None


def resolve_limits(timeout_seconds: int | None = None) -> CVJobLimits:
    """Validate both CV bounds, or raise naming the variable that is wrong.

    Called before any video is opened and before any model is loaded, so a
    misconfigured host spends nothing on work it is going to abandon. An
    explicit ``timeout_seconds`` from the CLI wins over the variable; the
    variable still has to parse, because an operator who set both and typo'd one
    should hear about it.
    """
    configured = bounded_int_from_env(
        CV_TIMEOUT_ENV_VAR,
        default=DEFAULT_TIMEOUT_SECONDS,
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=MAX_TIMEOUT_SECONDS,
    )
    return CVJobLimits(
        timeout_seconds=configured if timeout_seconds is None else timeout_seconds,
        memory_limit_bytes=memory_limit_bytes_from_env(CV_MEMORY_ENV_VAR),
    )


def _pipeline_device() -> str:
    """Prefer Apple MPS / CUDA when present; same weights, less wall time."""
    forced = (os.environ.get("POKER_CV_DEVICE") or "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


class JobCancelled(Exception):
    """Raised when the operator cancels an active reconstruction job."""


def _raise_job_cancelled(signum: int, _frame: object) -> None:
    raise JobCancelled(f"Stopped by signal {signum}.")


@contextmanager
def _cleanup_on_termination_signal() -> Iterator[None]:
    """Turn SIGTERM/SIGINT into an exception so the cleanup in ``finally`` runs.

    Cancelling a reconstruction sends SIGTERM to this worker's process group,
    and CPython's default disposition for SIGTERM exits the interpreter without
    unwinding. That skipped the one path ``_discard_partial_artifacts`` names in
    its own docstring: a cancelled run left its partial timeline and one JPEG
    per sampled second behind. Raising instead reaches the existing cancelled
    branch and the existing ``finally``.

    Best effort by design. ``cancel_processing_job`` allows two seconds before
    SIGKILL, which is ample for the unlink and the tree removal but is not a
    guarantee, and a signal handler can only be installed from the main thread
    of the main interpreter -- the suite calls ``run_job`` in-process and is
    left on the default disposition.
    """

    previous: dict[signal.Signals, _SignalHandler] = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[number] = signal.signal(number, _raise_job_cancelled)
        except (ValueError, OSError):
            continue
    try:
        yield
    finally:
        for restored, handler in previous.items():
            try:
                signal.signal(restored, handler)
            except (ValueError, OSError):
                pass


def run_job(
    *,
    job_id: int,
    video_path: Path,
    session_name: str,
    db_path: Path,
    target_session_id: int | None = None,
    timeout_seconds: int | None = None,
) -> int:
    """Run one reconstruction, cleaning up after itself even when signalled."""

    with _cleanup_on_termination_signal():
        return _run_job(
            job_id=job_id,
            video_path=video_path,
            session_name=session_name,
            db_path=db_path,
            target_session_id=target_session_id,
            timeout_seconds=timeout_seconds,
        )


def _run_job(
    *,
    job_id: int,
    video_path: Path,
    session_name: str,
    db_path: Path,
    target_session_id: int | None = None,
    timeout_seconds: int | None = None,
) -> int:
    db = PokerDatabase(db_path)
    db.init_db()
    paths = ensure_data_directories()
    timeline_path = Path(paths["cv_timelines"]) / f"job_{job_id}_timeline.json"
    progress_path = timeline_path.with_name(f"job_{job_id}_progress.json")
    export_path = Path(paths["exports"]) / f"job_{job_id}_session.json"
    frame_root = Path(paths.get("frames", timeline_path.parent / "frames"))
    frame_dir = frame_root / f"cv_job_{job_id}"
    # Only artifacts this invocation produced are ever discarded. Re-running a
    # completed job's id and failing before the pipeline starts must not take
    # the frames the earlier run's validated hands point at.
    pipeline_started = False
    reconstruction_completed = False
    try:
        # Both bounds are resolved inside the try so a bad value becomes a
        # failed job carrying the variable's name, which is what the operator
        # reads, rather than a traceback in a detached worker's log.
        limits = resolve_limits(timeout_seconds)
        deadline = time.monotonic() + limits.timeout_seconds
        _assert_not_cancelled(db, job_id)
        job = db.fetch_processing_job(job_id)
        if job is None:
            raise ValueError(f"Processing job #{job_id} was not found.")
        # Refused here, before the recording is opened and before any artifact is
        # touched. Re-running a finished job's id -- easy to do, because the worker
        # is a documented recovery entrypoint and the id is right there in the job
        # panel -- rebuilt the timeline, took another backup, overwrote the
        # finished row's progress and message with this run's heartbeats, and then
        # lost its own completion to the status guard, leaving the row reading
        # "Preparing session for validation" at 96% with no record of how many
        # hands were exported or whether the backup verified. It returned 0 while
        # doing it. A finished job is not resumable; the recording gets a new one.
        if job.status not in RUNNABLE_JOB_STATUSES:
            raise ValueError(
                f"Processing job #{job_id} is already {job.status!r}, so this "
                "worker could not record an outcome on it and nothing was read "
                "from the recording. Start a new reconstruction for this video "
                "instead of re-running a finished job."
            )
        _assert_no_foreign_heavy_job(db, job_id)
        video = db.fetch_video(job.video_id)
        if video is None:
            raise ValueError(f"Video #{job.video_id} was not found.")
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
        _heartbeat(db, job_id, 3, "Loading reconstruction models")
        # The window this run asks for, and whether it came from the recording
        # at all. When ingest could not measure a duration the bound is a
        # 24-hour ceiling, which is not the recording's length and must never be
        # reported as though it were.
        duration_measured = bool(video.duration_seconds)
        end_seconds = video.duration_seconds if duration_measured else 86_400
        window_note = (
            f"sampling window requested 0–{end_seconds:g}s at 1s"
            if duration_measured
            else (
                "sampling window requested 0–86400s at 1s because this "
                "recording's duration was never measured, so the window is a "
                "ceiling rather than the recording's length"
            )
        )
        command = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--video",
            str(path),
            "--start",
            "0",
            "--end",
            str(end_seconds),
            "--interval",
            "1",
            "--device",
            _pipeline_device(),
            "--out",
            str(timeline_path),
            "--frame-dir",
            str(frame_dir),
            "--progress-file",
            str(progress_path),
        ]
        pipeline_started = True
        _run_pipeline(command, db, job_id, deadline, progress_path, limits)

        _assert_not_cancelled(db, job_id)
        _heartbeat(db, job_id, 82, "Validating reconstructed hands")
        payload = export_timeline(
            timeline_path,
            export_path,
            session_name=session_name,
            # Incomplete segments stay in the export for evidence review. Hands
            # are not bulk-imported here; frame validation or an explicit draft
            # add lands them in the session later.
            include_incomplete=True,
        )
        _check_deadline(deadline, limits.timeout_seconds)
        _assert_not_cancelled(db, job_id)
        _heartbeat(db, job_id, 92, "Backing up study database")
        data_root = Path(paths.get("data", Path(paths["backups"]).parent))
        backup_path = backup_database(
            db_path, Path(paths["backups"]), data_dir=data_root
        )
        _heartbeat(db, job_id, 94, "Verifying the study database backup")
        backup_note = _verified_backup_note(
            backup_path, db_path=db_path, data_dir=data_root
        )
        _assert_not_cancelled(db, job_id)
        _heartbeat(db, job_id, 96, "Preparing session for validation")
        _assert_not_cancelled(db, job_id)
        # Session link + job completion in one transaction. Hands are added later
        # by validate-then-import (auto or manual draft), not on job finish.
        with db.transaction():
            current_video = db.fetch_video(job.video_id)
            if current_video is None:
                raise ValueError(f"Video #{job.video_id} was not found.")
            destination = _ensure_destination_session(
                db,
                session_name=session_name,
                target_session_id=target_session_id,
                video_session_id=current_video.session_id,
            )
            exported_count = int(
                payload.get("cv_import_summary", {}).get("exported_hands", 0) or 0
            )
            # The window is stated on the row because no other artifact carried
            # it: the timeline summary, the export summary and this message all
            # reported the same fields for a run that read the whole recording
            # and one that read part of it. It says what was REQUESTED -- the
            # sampler's own record of where it actually stopped is printed to
            # the worker's log and is not readable from here, and a coverage
            # claim this code cannot check is the thing it must not make.
            message = (
                f"Ready for validation; {exported_count} hands exported for "
                f"session #{destination.id} — add each when validated or as a "
                f"draft; {window_note}; {backup_note}"
            )
            current = db.fetch_processing_job(job_id)
            if current is not None and current.status in {"cancelling", "cancelled"}:
                message = f"{message} (cancel arrived after export)."
            video_id = current.video_id if current is not None else job.video_id
            recorded = db.update_processing_job(
                job_id,
                expected_statuses=("running", "cancelling", "cancelled"),
                status="completed",
                progress_percent=100,
                message=message,
                clear_pid=True,
                completed_at=datetime.now(UTC),
            )
            db.update_video_session(video_id, destination.id)
        reconstruction_completed = True
        if recorded.status != "completed":
            # The status guard was refused -- reconcile_stuck_jobs failed this row
            # while the export and the backup were running, and it answers a
            # refused guard by returning the row unchanged rather than raising.
            # The work is real and its artifacts are kept, but the row now belongs
            # to the reconciler and there is nowhere left to record the outcome:
            # mark_failed carries the same kind of guard and would be refused too.
            # The log is the last surface that can say so, and a zero exit code
            # here would report a success no operator could ever see.
            print(
                f"Reconstruction job #{job_id} finished but its row is "
                f"{recorded.status!r} and would not accept a completion, so the "
                f"outcome was recorded nowhere: {message}",
                file=sys.stderr,
            )
            return 1
        return 0
    except JobCancelled as exc:
        try:
            mark_cancelled(db, job_id, str(exc) or "Cancelled by user.")
        except Exception:
            pass
        return 1
    except BaseException as exc:
        safe_message = safe_error_message(exc)
        try:
            current = db.fetch_processing_job(job_id)
            if current is not None and current.status in {"cancelling", "cancelled"}:
                mark_cancelled(db, job_id, "Cancelled by user.")
            else:
                mark_failed(db, job_id, safe_message)
        except Exception:
            pass
        return 1
    finally:
        progress_path.unlink(missing_ok=True)
        if pipeline_started and not reconstruction_completed:
            _discard_partial_artifacts(timeline_path, frame_dir)
        db.close()


def _holds_the_machine(status: str, pid: int | None) -> bool:
    """Whether a live row is evidence of heavy work actually occupying the host.

    The launcher refuses on the row alone, which is right for it: it is deciding
    whether to create a second row. The worker is deciding whether a second
    heavy *process* would exist, and rows outlive processes — an unclean
    shutdown leaves 'running' rows with dead workers for the fifteen minutes
    reconcile_stuck_jobs takes to notice. Refusing on those would make the
    recovery entrypoint unusable exactly when it is needed. A queued row still
    blocks, because its worker is about to start.
    """
    return status == "queued" or (pid is not None and _pid_is_alive(pid))


def _assert_no_foreign_heavy_job(db: PokerDatabase, job_id: int) -> None:
    """Refuse to start heavy work while another heavy job already holds the host.

    ``cv_jobs.start_cv_job`` decides this before it creates the row, but the
    worker is reachable without it: ``python -m poker_tracker.ui.run_cv_job``
    is a documented recovery entrypoint and the release gate spawns the same
    pipeline. Two detectors plus a solve on the reference host is how the OOM
    killer picks which job dies, so the admission decision belongs where the
    heavy work actually starts as well as where the button is.
    """
    conflicts = [
        f"{other.job_type.replace('_', ' ')} job #{other.id} is already {other.status}"
        for other in db.fetch_active_jobs()
        if other.id != job_id and _holds_the_machine(other.status, other.pid)
    ]
    conflicts.extend(
        f"solver run #{run.id} is already {run.status}"
        for run in db.fetch_active_solver_runs()
        if _holds_the_machine(run.status, run.pid)
    )
    if not conflicts:
        return
    raise RuntimeError(
        f"Reconstruction was not started: {conflicts[0]}. Only one heavy "
        "CV/solver job runs at a time. Wait for it to finish or cancel it, then "
        "start this reconstruction again — nothing was read from the recording "
        "and no artifacts were written."
    )


def _discard_partial_artifacts(timeline_path: Path, frame_dir: Path) -> None:
    """Leave nothing behind that a later reader could mistake for a result.

    The pipeline prunes its review frames only when it reaches the end, so a run
    that was killed, timed out or cancelled leaves one JPEG per sampled frame —
    a 90-minute recording at one second is 5,400 files — plus whatever partial
    timeline was on disk. Neither is evidence of anything: the recording is
    still there and re-running rebuilds both. Only a completed job keeps its
    frames, because the export and the validated hands point at them.
    """
    try:
        timeline_path.unlink(missing_ok=True)
    except OSError:
        pass
    shutil.rmtree(frame_dir, ignore_errors=True)


def _verified_backup_note(
    backup_path: Path, *, db_path: Path, data_dir: Path
) -> str:
    """Restore the snapshot just written, and say in the job message what happened.

    A snapshot nothing ever opens is a recovery point on paper. This is the one
    place in the product that both writes one and has an operator-visible surface
    to report it on, so the drill runs here rather than waiting for someone to
    remember a maintenance flag. A failed verification does not fail the job --
    the reconstruction is finished and worth keeping -- but it must never be
    reported as a completed backup either.
    """
    verification = verify_snapshot(
        backup_path, live_database=db_path, data_dir=data_dir
    )
    if verification.status == "fail":
        detail = "; ".join(verification.details[:2]) or verification.message
        return f"backup {backup_path.name} FAILED VERIFICATION: {detail}"
    if verification.status == "warning":
        detail = "; ".join(verification.details[:2]) or verification.message
        return f"backup {backup_path.name} verified with warnings: {detail}"
    return f"backup {backup_path.name} verified"


def _run_pipeline(
    command: list[str],
    db: PokerDatabase,
    job_id: int,
    deadline: float,
    progress_path: Path,
    limits: CVJobLimits | None = None,
) -> None:
    bounds = limits or CVJobLimits(DEFAULT_TIMEOUT_SECONDS, None)
    memory_limit_bytes = bounds.memory_limit_bytes
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            start_new_session=True,
            close_fds=True,
            preexec_fn=(
                memory_limiter(memory_limit_bytes, variable=CV_MEMORY_ENV_VAR)
                if os.name == "posix" and memory_limit_bytes is not None
                else None
            ),
        )
    except subprocess.SubprocessError as exc:
        if memory_limit_bytes is None:
            raise
        # subprocess reports a hook failure only as "Exception occurred in
        # preexec_fn", so the reason the operator needs has to be restated.
        # Refusing to reconstruct is the point: a cap that was asked for and not
        # applied is how a detector plus a decoder reaches the host OOM killer
        # while the operator believes the job is bounded.
        raise RuntimeError(
            f"The {format_gb(memory_limit_bytes)} reconstruction memory cap from "
            f"{CV_MEMORY_ENV_VAR} could not be applied, so no reconstruction was "
            "started and nothing was written. macOS rejects RLIMIT_AS outright; "
            "unset the variable to reconstruct without a cap, or run the job in "
            "the Linux container where the cap holds."
        ) from exc
    pid_path = _pipeline_pid_path(progress_path)
    try:
        pid_path.write_text(str(process.pid), encoding="utf-8")
    except OSError:
        pid_path = None
    last_heartbeat = 0.0
    last_progress = 8.0
    try:
        while process.poll() is None:
            _check_deadline(deadline, bounds.timeout_seconds)
            _assert_not_cancelled(db, job_id)
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                progress = _read_pipeline_progress(progress_path)
                if progress is not None:
                    current, total, stage = progress
                    if stage == "timeline":
                        last_progress = 79.0
                        message = "Building reconstructed hand timeline"
                    else:
                        last_progress = max(
                            last_progress,
                            min(78.0, 8.0 + (70.0 * current / total)),
                        )
                        message = f"Reconstructing video · frame {current}/{total}"
                else:
                    message = "Loading models and preparing video"
                _heartbeat(db, job_id, last_progress, message)
                last_heartbeat = now
            time.sleep(1)
    except BaseException as exc:
        stopped = _terminate_process_group(process.pid)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stopped = _terminate_process_group(process.pid)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stopped = False
        if isinstance(exc, TimeoutError) and not stopped:
            # An orphaned pipeline keeps its detector, its decoder and its
            # memory, so the operator has to be told to go find it rather than
            # assume the timeout cleaned up after itself.
            raise TimeoutError(
                f"{exc} The reconstruction process (pid {process.pid}) could not "
                "be confirmed stopped and may still be running."
            ) from exc
        raise
    finally:
        if pid_path is not None:
            pid_path.unlink(missing_ok=True)
    if process.returncode:
        detail = f"Reconstruction pipeline exited with code {process.returncode}."
        if memory_limit_bytes is not None:
            # Not a claim about the cause. RLIMIT_AS surfaces as an ordinary
            # allocation failure in the child, indistinguishable from any other
            # crash from out here, so this says where to look and stops short of
            # saying what happened.
            detail += (
                f" Its address space was capped at {format_gb(memory_limit_bytes)} "
                f"by {CV_MEMORY_ENV_VAR}; check the job log before raising it."
            )
        raise RuntimeError(detail)


def _ensure_destination_session(
    db: PokerDatabase,
    *,
    session_name: str,
    target_session_id: int | None,
    video_session_id: int | None = None,
) -> Session:
    """Return the session reconstructed hands will be added into later.

    Prefers an explicit target, then an already-linked video session, then creates
    a ClubWPT-provenance session so re-running a completed job does not spawn
    duplicate empty destinations.
    """
    preferred_id = target_session_id if target_session_id is not None else video_session_id
    if preferred_id is not None:
        existing = db.fetch_session(preferred_id)
        if existing is None:
            if target_session_id is not None:
                raise ValueError(f"Session not found: {target_session_id}")
        else:
            return existing
    created = db.create_session(
        Session(
            name=session_name.strip() or "Reconstructed",
            platform="ClubWPT Gold",
            notes=(
                "Prepared by offline CV reconstruction. Hands are added after "
                "frame validation or an explicit draft add; they still require "
                "operator review before study."
            ),
        )
    )
    if created.id is None:
        raise RuntimeError("Destination session was not persisted.")
    return created


def _pipeline_pid_path(progress_path: Path) -> Path:
    return progress_path.with_name(progress_path.name.replace("_progress.json", "_pipeline.pid"))


def _read_pipeline_progress(progress_path: Path) -> tuple[int, int, str] | None:
    """Read the pipeline's atomic progress snapshot, tolerating startup races."""
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        current = max(0, int(payload["current"]))
        total = max(1, int(payload["total"]))
        stage = str(payload.get("stage", "frames"))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return min(current, total), total, stage


def _heartbeat(db: PokerDatabase, job_id: int, progress: float, message: str) -> None:
    update_progress(db, job_id, progress, message)
    db.update_processing_job(job_id, heartbeat_at=datetime.now(UTC))


def _assert_not_cancelled(db: PokerDatabase, job_id: int) -> None:
    job = db.fetch_processing_job(job_id)
    if job is None:
        raise ValueError(f"Processing job #{job_id} was not found.")
    if job.status in {"cancelling", "cancelled"}:
        raise JobCancelled("Cancelled by user.")


def _check_deadline(deadline: float, timeout_seconds: int) -> None:
    """Stop the job at its wall-clock bound, saying what the bound was.

    A terminal state has to state its outcome, not its last progress reading:
    "82%" is what the row said a moment ago, and the operator needs to know the
    run is over, which limit ended it, where that limit comes from, and that the
    partial work is gone.
    """
    if time.monotonic() > deadline:
        raise TimeoutError(
            f"Reconstruction stopped after exceeding its {timeout_seconds}-second "
            f"wall-clock limit, set by {CV_TIMEOUT_ENV_VAR}. No hands were "
            "exported and the partial timeline and review frames were discarded. "
            "Raise the limit or shorten the recording, then run it again."
        )


def _terminate_process_group(pid: int | None) -> bool:
    """Stop the pipeline and everything it forked. True only when the pid is gone."""
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
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    except PermissionError:
        return False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--target-session-id", type=int)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help=(
            f"Wall-clock bound in seconds. Defaults to {CV_TIMEOUT_ENV_VAR}, "
            f"or {DEFAULT_TIMEOUT_SECONDS} when that is unset."
        ),
    )
    args = parser.parse_args()
    return run_job(
        job_id=args.job_id,
        video_path=args.video,
        session_name=args.session_name,
        db_path=args.db,
        target_session_id=args.target_session_id,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
