from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from poker_tracker.db import PokerDatabase
from poker_tracker.jobs import create_job, mark_completed, mark_failed, mark_running, update_progress
from poker_tracker.models import ExtractedFrame
from poker_tracker.video_storage import FRAMES_DIR, video_frame_dir


@dataclass(frozen=True)
class FrameExtractionSummary:
    video_id: int
    job_id: int
    frames_extracted: int
    output_dir: Path
    duration_seconds: float | None = None
    errors: list[str] | None = None


def extract_frames_for_video(
    db: PokerDatabase,
    video_id: int,
    *,
    frames_per_second: float | None = 2.0,
    interval_seconds: float | None = None,
    max_frames: int | None = None,
    start_time_seconds: float = 0.0,
    end_time_seconds: float | None = None,
    frames_dir: Path = FRAMES_DIR,
) -> FrameExtractionSummary:
    """Synchronously extract preview frames from a stored completed-session video."""
    video = db.fetch_video(video_id)
    if video is None:
        raise ValueError(f"Video not found: {video_id}")
    if frames_per_second is not None and frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive.")
    if interval_seconds is not None and interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    job = create_job(db, video_id)
    output_dir = video_frame_dir(video_id, frames_dir)
    capture = cv2.VideoCapture(video.stored_path)
    extracted = 0
    errors: list[str] = []
    try:
        mark_running(db, job.id, "Extracting frames")
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video.stored_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / source_fps if source_fps > 0 and total_frames > 0 else None
        interval = interval_seconds if interval_seconds is not None else 1 / (frames_per_second or 2.0)
        start = max(0.0, start_time_seconds)
        end = end_time_seconds if end_time_seconds is not None else duration
        if end is not None and end < start:
            raise ValueError("end_time_seconds must be after start_time_seconds.")

        timestamps = _target_timestamps(start, end, interval, duration, max_frames)
        total_targets = max(1, len(timestamps))
        existing_frame_indexes = {frame.frame_index for frame in db.fetch_frames_by_video(video_id)}

        for target_number, timestamp in enumerate(timestamps, start=1):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                errors.append(f"Could not read frame at {timestamp:.2f}s")
                continue
            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_index in existing_frame_indexes:
                update_progress(
                    db,
                    job.id,
                    target_number / total_targets * 100,
                    f"Skipped duplicate frame {frame_index}",
                )
                continue
            image_path = output_dir / f"frame_{frame_index:06d}_{int(timestamp * 1000):010d}ms.jpg"
            cv2.imwrite(str(image_path), frame)
            db.create_extracted_frame(
                ExtractedFrame(
                    video_id=video_id,
                    job_id=job.id,
                    timestamp_seconds=timestamp,
                    frame_index=frame_index,
                    image_path=str(image_path),
                )
            )
            existing_frame_indexes.add(frame_index)
            extracted += 1
            update_progress(
                db,
                job.id,
                target_number / total_targets * 100,
                f"Extracted {extracted} frames",
            )

        mark_completed(db, job.id, f"Extracted {extracted} frames")
        return FrameExtractionSummary(
            video_id=video_id,
            job_id=job.id,
            frames_extracted=extracted,
            output_dir=output_dir,
            duration_seconds=duration,
            errors=errors,
        )
    except Exception as exc:
        mark_failed(db, job.id, str(exc))
        raise
    finally:
        capture.release()


def delete_extracted_frames(db: PokerDatabase, video_id: int, frames_dir: Path = FRAMES_DIR) -> int:
    """Delete extracted frame files and DB records for a video."""
    frames = db.fetch_frames_by_video(video_id)
    for frame in frames:
        path = Path(frame.image_path)
        if path.exists():
            path.unlink()
    output_dir = frames_dir / f"video_{video_id}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    db.delete_frame_records_by_video(video_id)
    return len(frames)


def select_representative_frames(frames: list[ExtractedFrame], limit: int = 12) -> list[ExtractedFrame]:
    """Select evenly spaced frames for preview."""
    if limit <= 0:
        return []
    if len(frames) <= limit:
        return frames
    step = (len(frames) - 1) / (limit - 1)
    indexes = [round(index * step) for index in range(limit)]
    return [frames[index] for index in indexes]


def _target_timestamps(
    start: float,
    end: float | None,
    interval: float,
    duration: float | None,
    max_frames: int | None,
) -> list[float]:
    effective_end = end if end is not None else duration
    if effective_end is None:
        effective_end = start + interval * (max_frames or 1)
    timestamps: list[float] = []
    current = start
    while current <= effective_end + 1e-9:
        timestamps.append(round(current, 3))
        if max_frames is not None and len(timestamps) >= max_frames:
            break
        current += interval
    return timestamps


# TODO: Future ROI/CV/OCR modules should consume extracted frames from this table.
