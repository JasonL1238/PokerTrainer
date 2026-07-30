"""Detached worker for completed-session video reconstruction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import export_timeline
from poker_tracker.persistence.backup import BACKUP_KEEP_COUNT, backup_database
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_hands_into_session, import_session
from poker_tracker.ui.jobs import mark_completed, mark_failed, update_progress
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

        _heartbeat(db, job_id, 82, "Validating reconstructed hands")
        payload = export_timeline(
            timeline_path,
            export_path,
            session_name=session_name,
        )
        _check_deadline(deadline)
        _heartbeat(db, job_id, 92, "Backing up study database")
        backup_path = backup_database(db_path, Path(paths["backups"]))
        _heartbeat(db, job_id, 96, "Importing reconstructed hands")
        imported = (
            import_hands_into_session(db, payload, target_session_id)
            if target_session_id is not None
            else import_session(db, payload)
        )
        job = db.fetch_processing_job(job_id)
        if job is None:
            raise RuntimeError("Processing job disappeared before import completed.")
        if db.fetch_session(imported.id) is not None:
            db.update_video_session(job.video_id, imported.id)
        exported_count = payload.get("cv_import_summary", {}).get("exported_hands", 0)
        mark_completed(
            db,
            job_id,
            f"Imported {exported_count} hands into session #{imported.id}; backup {backup_path.name}",
        )
        return 0
    except BaseException as exc:
        safe_message = str(exc).replace("\n", " ")[:500] or type(exc).__name__
        try:
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
    process = subprocess.Popen(command, cwd=str(REPO_ROOT))
    last_heartbeat = 0.0
    last_progress = 8.0
    try:
        while process.poll() is None:
            _check_deadline(deadline)
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
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    if process.returncode:
        raise RuntimeError(f"Reconstruction pipeline exited with code {process.returncode}.")


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


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("Reconstruction exceeded the configured timeout.")


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
