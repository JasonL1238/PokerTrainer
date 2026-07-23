"""Detached worker for completed-session video reconstruction."""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import export_timeline
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.ui.jobs import mark_completed, mark_failed, update_progress
from poker_tracker.ui.video_storage import BACKUPS_DIR, CV_TIMELINES_DIR, ensure_data_directories


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "cv_lab" / "scripts" / "pipeline" / "run_two_model_pipeline.py"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
HEARTBEAT_INTERVAL_SECONDS = 20
BACKUP_KEEP_COUNT = 5


def run_job(
    *,
    job_id: int,
    video_path: Path,
    session_name: str,
    db_path: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> int:
    db = PokerDatabase(db_path)
    db.init_db()
    paths = ensure_data_directories()
    timeline_path = Path(paths["cv_timelines"]) / f"job_{job_id}_timeline.json"
    export_path = Path(paths["exports"]) / f"job_{job_id}_session.json"
    deadline = time.monotonic() + timeout_seconds
    try:
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        _heartbeat(db, job_id, 3, "Loading reconstruction models")
        video = db.fetch_video(db.fetch_processing_job(job_id).video_id)  # type: ignore[union-attr]
        end_seconds = video.duration_seconds if video and video.duration_seconds else 86_400
        command = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--video",
            str(video_path),
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
        ]
        _run_pipeline(command, db, job_id, deadline)

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
        imported = import_session(db, payload)
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
        db.close()


def backup_database(db_path: Path, backup_dir: Path = BACKUPS_DIR) -> Path:
    """Create a SQLite-consistent snapshot and retain the newest five."""
    if str(db_path) == ":memory:":
        raise ValueError("Cannot back up an in-memory database.")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = backup_dir / f"poker_tracker_{stamp}.sqlite3"
    with sqlite3.connect(str(db_path)) as source, sqlite3.connect(str(destination)) as target:
        source.backup(target)
    backups = sorted(backup_dir.glob("poker_tracker_*.sqlite3"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old in backups[BACKUP_KEEP_COUNT:]:
        old.unlink(missing_ok=True)
    return destination


def _run_pipeline(command: list[str], db: PokerDatabase, job_id: int, deadline: float) -> None:
    process = subprocess.Popen(command, cwd=str(REPO_ROOT))
    last_heartbeat = 0.0
    try:
        while process.poll() is None:
            _check_deadline(deadline)
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                elapsed = max(0, DEFAULT_TIMEOUT_SECONDS - int(deadline - now))
                estimated = min(78, 8 + elapsed / 60)
                _heartbeat(db, job_id, estimated, "Reconstructing completed-session video")
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


def _heartbeat(db: PokerDatabase, job_id: int, progress: float, message: str) -> None:
    update_progress(db, job_id, progress, message)
    db.update_processing_job(job_id, heartbeat_at=datetime.now(timezone.utc))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("Reconstruction exceeded the configured timeout.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    return run_job(
        job_id=args.job_id,
        video_path=args.video,
        session_name=args.session_name,
        db_path=args.db,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
