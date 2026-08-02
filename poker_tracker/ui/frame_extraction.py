from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import ExtractedFrame
from poker_tracker.runtime.limits import bounded_int_from_env
from poker_tracker.ui.jobs import (
    create_job,
    mark_completed,
    mark_failed,
    mark_running,
    update_progress,
)
from poker_tracker.ui.video_storage import FRAMES_DIR, video_frame_dir

# Extracted frames are retained artifacts on the operator's disk, and a caller
# that passes no max_frames asks for one JPEG per sampled second of a recording
# the ingest limits allow to be 24 hours long. The ceiling bounds that in every
# direction, including a caller that asks for more than it.
FRAME_CEILING_ENV_VAR = "POKERTRAINER_CV_MAX_EXTRACTED_FRAMES"
DEFAULT_FRAME_CEILING = 2000
MAX_FRAME_CEILING = 1_000_000


def configured_frame_ceiling() -> int:
    """The retained-frame ceiling, or raise naming the variable that is wrong."""
    return bounded_int_from_env(
        FRAME_CEILING_ENV_VAR,
        default=DEFAULT_FRAME_CEILING,
        minimum=1,
        maximum=MAX_FRAME_CEILING,
    )


@dataclass(frozen=True)
class FrameExtractionSummary:
    video_id: int
    job_id: int
    frames_extracted: int
    output_dir: Path
    duration_seconds: float | None = None
    errors: list[str] | None = None
    # The two fields below exist so a truncated frame set is never handed back
    # looking like the whole recording. frame_limit is the bound that applied;
    # truncated_at_limit says the video had more to give.
    frame_limit: int | None = None
    truncated_at_limit: bool = False


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
    # Validated before the job row exists, so a misconfigured ceiling is an
    # error the caller sees now rather than a job that fails a moment later.
    ceiling = configured_frame_ceiling()
    frame_limit = ceiling if max_frames is None else min(max_frames, ceiling)
    ceiling_applied = max_frames is None or max_frames > ceiling

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

        # One past the limit, so the truncation is observed rather than assumed:
        # a video that happens to end exactly on the limit was not truncated.
        timestamps = _target_timestamps(start, end, interval, duration, frame_limit + 1)
        # Truncation is only claimed against a span the video actually reported.
        # With no readable duration and no end time there is no denominator, so
        # the limit still applies but "the recording has more" is unevidenced.
        truncated = end is not None and len(timestamps) > frame_limit
        timestamps = timestamps[:frame_limit]
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

        truncated_at_limit = truncated and ceiling_applied
        completion = f"Extracted {extracted} frames"
        if truncated_at_limit:
            completion += (
                f"; stopped at the {ceiling}-frame ceiling set by "
                f"{FRAME_CEILING_ENV_VAR}, so the recording has more"
            )
        mark_completed(db, job.id, completion)
        return FrameExtractionSummary(
            video_id=video_id,
            job_id=job.id,
            frames_extracted=extracted,
            output_dir=output_dir,
            duration_seconds=duration,
            errors=errors,
            frame_limit=frame_limit,
            truncated_at_limit=truncated_at_limit,
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
