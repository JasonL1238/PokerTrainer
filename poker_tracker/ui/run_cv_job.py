"""Detached worker for completed-session video reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import export_timeline
from poker_tracker.persistence.backup import BACKUP_KEEP_COUNT, backup_database
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_hands_into_session, import_session
from poker_tracker.ui.jobs import mark_cancelled, mark_completed, mark_failed, update_progress
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

# backup_database now lives in the persistence package so db.py can snapshot
# before a migration; re-exported here because callers and tests import it from
# this module.
__all__ = ["BACKUP_KEEP_COUNT", "backup_database", "run_job", "main"]


class JobCancelled(Exception):
    """Raised when the operator cancels an active reconstruction job."""


def run_job(
    *,
    job_id: int,
    video_path: Path,
    session_name: str,
    db_path: Path,
    target_session_id: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> int:
    db = PokerDatabase(db_path)
    db.init_db()
    paths = ensure_data_directories()
    timeline_path = Path(paths["cv_timelines"]) / f"job_{job_id}_timeline.json"
    progress_path = timeline_path.with_name(f"job_{job_id}_progress.json")
    export_path = Path(paths["exports"]) / f"job_{job_id}_session.json"
    frame_root = Path(paths.get("frames", timeline_path.parent / "frames"))
    deadline = time.monotonic() + timeout_seconds
    try:
        _assert_not_cancelled(db, job_id)
        job = db.fetch_processing_job(job_id)
        if job is None:
            raise ValueError(f"Processing job #{job_id} was not found.")
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
        end_seconds = video.duration_seconds if video.duration_seconds else 86_400
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
            "cpu",
            "--out",
            str(timeline_path),
            "--frame-dir",
            str(frame_root / f"cv_job_{job_id}"),
            "--progress-file",
            str(progress_path),
        ]
        _run_pipeline(command, db, job_id, deadline, progress_path)

        _assert_not_cancelled(db, job_id)
        _heartbeat(db, job_id, 82, "Validating reconstructed hands")
        payload = export_timeline(
            timeline_path,
            export_path,
            session_name=session_name,
            # Incomplete hero-preflop segments become partial draft hands under
            # the session so the operator can fill blanks and finalize them.
            # Hands where hero never played preflop are still skipped by export.
            include_incomplete=True,
        )
        _check_deadline(deadline)
        _assert_not_cancelled(db, job_id)
        _heartbeat(db, job_id, 92, "Backing up study database")
        backup_path = backup_database(db_path, Path(paths["backups"]))
        _assert_not_cancelled(db, job_id)
        _heartbeat(db, job_id, 96, "Importing reconstructed hands")
        _assert_not_cancelled(db, job_id)
        # Import + job completion + video link in one transaction. If the process
        # dies mid-import, SQLite rolls back the hands; if it commits, the job
        # cannot report "cancelled" while drafts remain (re-run would append).
        with db.transaction():
            imported = (
                import_hands_into_session(db, payload, target_session_id)
                if target_session_id is not None
                else import_session(db, payload)
            )
            exported_count = payload.get("cv_import_summary", {}).get(
                "exported_hands", 0
            )
            message = (
                f"Imported {exported_count} hands into session #{imported.id}; "
                f"backup {backup_path.name}"
            )
            current = db.fetch_processing_job(job_id)
            if current is not None and current.status in {"cancelling", "cancelled"}:
                message = f"{message} (cancel arrived after import)."
            video_id = current.video_id if current is not None else job.video_id
            db.update_processing_job(
                job_id,
                expected_statuses=("running", "cancelling", "cancelled"),
                status="completed",
                progress_percent=100,
                message=message,
                clear_pid=True,
                completed_at=datetime.now(UTC),
            )
            if db.fetch_session(imported.id) is not None:
                db.update_video_session(video_id, imported.id)
        return 0
    except JobCancelled as exc:
        try:
            mark_cancelled(db, job_id, str(exc) or "Cancelled by user.")
        except Exception:
            pass
        return 1
    except BaseException as exc:
        safe_message = str(exc).replace("\n", " ")[:500] or type(exc).__name__
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
        db.close()


def _run_pipeline(
    command: list[str],
    db: PokerDatabase,
    job_id: int,
    deadline: float,
    progress_path: Path,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        start_new_session=True,
        close_fds=True,
    )
    pid_path = _pipeline_pid_path(progress_path)
    try:
        pid_path.write_text(str(process.pid), encoding="utf-8")
    except OSError:
        pid_path = None
    last_heartbeat = 0.0
    last_progress = 8.0
    try:
        while process.poll() is None:
            _check_deadline(deadline)
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
    except BaseException:
        _terminate_process_group(process.pid)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process.pid)
            process.wait(timeout=5)
        raise
    finally:
        if pid_path is not None:
            pid_path.unlink(missing_ok=True)
    if process.returncode:
        raise RuntimeError(f"Reconstruction pipeline exited with code {process.returncode}.")


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


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("Reconstruction exceeded the configured timeout.")


def _terminate_process_group(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    except PermissionError:
        return
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    except PermissionError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--target-session-id", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
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
