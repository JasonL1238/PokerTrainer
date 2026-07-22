from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    error: str = ""


def extract_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read basic metadata from a stored post-session video, failing gracefully."""
    path = Path(video_path)
    if not path.exists():
        return VideoMetadata(error=f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return VideoMetadata(error=f"Could not open video: {path}")
        fps = _positive_or_none(capture.get(cv2.CAP_PROP_FPS))
        frame_count = _int_positive_or_none(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = _int_positive_or_none(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _int_positive_or_none(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = None
        if fps and frame_count:
            duration = frame_count / fps
        return VideoMetadata(
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
    finally:
        capture.release()


def _positive_or_none(value: float) -> float | None:
    return None if value <= 0 else float(value)


def _int_positive_or_none(value: float) -> int | None:
    return None if value <= 0 else int(value)
