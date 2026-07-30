"""Validate and atomically ingest completed-session video uploads.

Analysis remains post-session only: this module rejects unsafe or unreadable
recordings before they are scheduled for offline CV.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from poker_tracker.ui.video_metadata import VideoMetadata, extract_video_metadata
from poker_tracker.ui.video_storage import (
    ALLOWED_VIDEO_EXTENSIONS,
    VIDEOS_DIR,
    ensure_data_directories,
    unique_stored_video_path,
    validate_video_extension,
)

# Defaults are intentionally generous for long table-client recordings while
# still bounding decompression-bomb and accidental-huge uploads.
DEFAULT_MAX_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB
DEFAULT_MIN_BYTES = 1
DEFAULT_MIN_WIDTH = 32
DEFAULT_MIN_HEIGHT = 24
DEFAULT_MAX_WIDTH = 7680
DEFAULT_MAX_HEIGHT = 4320
DEFAULT_MIN_DURATION_SECONDS = 0.1
DEFAULT_MAX_DURATION_SECONDS = 24 * 60 * 60  # 24h
DEFAULT_MIN_FPS = 1.0
DEFAULT_MAX_FPS = 120.0
DEFAULT_MIN_FRAME_COUNT = 1

# Container/codec names reported by PyAV (lowercase).
ALLOWED_CONTAINERS = frozenset({"mp4", "mov", "avi", "matroska", "quicktime"})
ALLOWED_VIDEO_CODECS = frozenset(
    {
        "h264",
        "hevc",
        "h265",
        "mpeg4",
        "mjpeg",
        "vp8",
        "vp9",
        "av1",
    }
)


@dataclass(frozen=True)
class VideoIngestLimits:
    max_bytes: int = DEFAULT_MAX_BYTES
    min_bytes: int = DEFAULT_MIN_BYTES
    min_width: int = DEFAULT_MIN_WIDTH
    min_height: int = DEFAULT_MIN_HEIGHT
    max_width: int = DEFAULT_MAX_WIDTH
    max_height: int = DEFAULT_MAX_HEIGHT
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    min_fps: float = DEFAULT_MIN_FPS
    max_fps: float = DEFAULT_MAX_FPS
    min_frame_count: int = DEFAULT_MIN_FRAME_COUNT


@dataclass(frozen=True)
class IngestedVideo:
    path: Path
    metadata: VideoMetadata
    file_size_bytes: int
    content_sha256: str


def ingest_uploaded_video(
    source: BinaryIO,
    original_filename: str,
    videos_dir: Path = VIDEOS_DIR,
    *,
    limits: VideoIngestLimits | None = None,
) -> IngestedVideo:
    """Atomically store an upload and require it to be a playable regular video."""
    bounds = limits or VideoIngestLimits()
    validate_video_extension(original_filename)
    ensure_data_directories(videos_dir.parent)
    assert_safe_storage_dir(videos_dir)

    destination = unique_stored_video_path(original_filename, videos_dir)
    assert_safe_destination(destination, videos_dir)

    try:
        file_size = _atomic_copy(source, destination, max_bytes=bounds.max_bytes)
        assert_regular_owned_file(destination)
        if file_size < bounds.min_bytes:
            raise ValueError("Video file is empty.")
        metadata = require_playable_video(destination, limits=bounds)
        digest = sha256_file(destination)
    except Exception:
        _cleanup_failed_ingest(destination)
        raise

    return IngestedVideo(
        path=destination,
        metadata=metadata,
        file_size_bytes=file_size,
        content_sha256=digest,
    )


def require_playable_video(
    video_path: str | Path,
    *,
    limits: VideoIngestLimits | None = None,
) -> VideoMetadata:
    """Probe a stored video and raise ValueError if it is not safe to process."""
    bounds = limits or VideoIngestLimits()
    path = Path(video_path)
    assert_regular_owned_file(path)

    size = path.stat().st_size
    if size < bounds.min_bytes:
        raise ValueError("Video file is empty.")
    if size > bounds.max_bytes:
        raise ValueError(
            f"Video exceeds maximum size of {bounds.max_bytes} bytes "
            f"(got {size} bytes)."
        )

    extension = path.suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {extension or 'none'}")

    metadata = extract_video_metadata(path)
    if metadata.error:
        raise ValueError(metadata.error)

    _require_dimension("width", metadata.width, bounds.min_width, bounds.max_width)
    _require_dimension("height", metadata.height, bounds.min_height, bounds.max_height)

    if metadata.fps is None or not (bounds.min_fps <= metadata.fps <= bounds.max_fps):
        raise ValueError(
            f"Unsupported or missing FPS (need {bounds.min_fps}-{bounds.max_fps})."
        )
    if (
        metadata.frame_count is None
        or metadata.frame_count < bounds.min_frame_count
    ):
        raise ValueError("Video has no readable frames.")
    if (
        metadata.duration_seconds is None
        or metadata.duration_seconds < bounds.min_duration_seconds
        or metadata.duration_seconds > bounds.max_duration_seconds
    ):
        raise ValueError(
            "Unsupported or missing duration "
            f"(need {bounds.min_duration_seconds}-{bounds.max_duration_seconds}s)."
        )

    _require_container_and_codec(path)
    _require_decodable_frames(path, frame_count=metadata.frame_count)
    return metadata


def resolve_stored_video_path(
    video_path: str | Path,
    *,
    stored_path: str,
    videos_dir: Path = VIDEOS_DIR,
) -> Path:
    """Bind a caller path to the DB record and confine it under videos_dir."""
    caller = Path(video_path)
    recorded = Path(stored_path)
    try:
        if caller.resolve() != recorded.resolve():
            raise ValueError(
                "Video path does not match the stored recording path for this video."
            )
    except FileNotFoundError as exc:
        raise ValueError(f"Video file not found: {video_path}") from exc
    assert_safe_destination(recorded, videos_dir)
    assert_regular_owned_file(recorded)
    return recorded


def assert_stored_video_matches_record(
    video_path: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str | None = None,
) -> None:
    """Fail closed when a stored recording is missing, replaced, or truncated."""
    path = Path(video_path)
    if not path.exists():
        raise ValueError(f"Video file not found: {path}")
    assert_regular_owned_file(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size_bytes:
        raise ValueError(
            f"Video size mismatch for {path.name}: expected "
            f"{expected_size_bytes} bytes, found {actual_size} bytes."
        )
    if expected_sha256 is not None:
        digest = sha256_file(path)
        if digest != expected_sha256.lower():
            raise ValueError(f"Video content hash mismatch for {path.name}.")


def assert_safe_storage_dir(videos_dir: Path) -> None:
    """Reject storage roots that are missing or not writable directories."""
    path = videos_dir.expanduser()
    if path.exists() and path.is_symlink():
        raise ValueError(f"Video storage directory must not be a symlink: {path}")
    if not path.exists():
        raise ValueError(f"Video storage directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Video storage path is not a directory: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        raise ValueError(f"Video storage directory is not writable: {path}")


def assert_safe_destination(destination: Path, videos_dir: Path) -> None:
    """Ensure the destination stays inside the videos root and is not a link."""
    videos_root = videos_dir.resolve()
    # Destination may not exist yet; resolve parent and append name.
    resolved_parent = destination.parent.resolve()
    candidate = resolved_parent / destination.name
    try:
        candidate.relative_to(videos_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing path outside video storage root: {destination}"
        ) from exc
    if destination.exists():
        assert_regular_owned_file(destination)


def assert_regular_owned_file(path: Path) -> None:
    """Reject symlinks, hardlinks, and non-regular files."""
    if path.is_symlink():
        raise ValueError(f"Symlinked videos are not allowed: {path}")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Video file not found: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Video path is not a regular file: {path}")
    if info.st_nlink > 1:
        raise ValueError(f"Hard-linked videos are not allowed: {path}")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: BinaryIO, destination: Path, *, max_bytes: int) -> int:
    """Write to a temp file in the same directory, then replace into place."""
    videos_dir = destination.parent
    fd, temp_name = tempfile.mkstemp(
        prefix=".upload_",
        suffix=destination.suffix.lower() or ".part",
        dir=str(videos_dir),
    )
    temp_path: Path | None = Path(temp_name)
    written = 0
    try:
        with os.fdopen(fd, "wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"Video exceeds maximum size of {max_bytes} bytes."
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written < DEFAULT_MIN_BYTES:
            raise ValueError("Video file is empty.")
        # Refuse to clobber an unexpected pre-existing destination.
        if destination.exists():
            raise ValueError(f"Destination already exists: {destination}")
        os.replace(temp_path, destination)
        temp_path = None  # replaced; skip cleanup
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return written


def _require_dimension(
    label: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> None:
    if value is None or value < minimum or value > maximum:
        raise ValueError(
            f"Unsupported or missing {label} "
            f"(need {minimum}-{maximum}px)."
        )


def _require_container_and_codec(path: Path) -> None:
    try:
        import av
    except ImportError as exc:
        raise ValueError(
            "PyAV is required to validate video container/codec before processing."
        ) from exc

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001 - surface any demux failure as reject
        raise ValueError(f"Unsupported or corrupt video container: {path.name}") from exc
    try:
        format_name = (container.format.name or "").lower()
        containers = {part.strip() for part in format_name.split(",") if part.strip()}
        if not containers.intersection(ALLOWED_CONTAINERS):
            raise ValueError(
                f"Unsupported video container '{format_name or 'unknown'}'."
            )
        video_streams = list(container.streams.video)
        if not video_streams:
            raise ValueError("Video has no video stream.")
        codec = (video_streams[0].codec_context.name or "").lower()
        if codec not in ALLOWED_VIDEO_CODECS:
            raise ValueError(f"Unsupported video codec '{codec or 'unknown'}'.")
    finally:
        container.close()


def _require_decodable_frames(path: Path, *, frame_count: int | None) -> None:
    """Decode first/middle/near-end samples so truncated files fail closed.

    OpenCV often over-reports frame_count on screen recordings, so the exact
    last index can fail even when the file is healthy. Near-end sampling walks
    backward from the claimed end until one frame decodes.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        total = int(frame_count or 0)
        if not _read_frame_at(capture, 0, total):
            raise ValueError(f"Could not decode frames from video near index 0: {path}")
        if total <= 1:
            return
        mid = max(0, total // 2)
        if not _read_frame_at(capture, mid, total):
            raise ValueError(
                f"Could not decode frames from video near index {mid}: {path}"
            )
        near_end_candidates = [
            max(0, total - 1),
            max(0, total - 2),
            max(0, total - 30),
            max(0, total - 300),
            max(0, int(total * 0.95)),
        ]
        if any(_read_frame_at(capture, index, total) for index in near_end_candidates):
            return
        raise ValueError(
            f"Could not decode frames near the end of video: {path}"
        )
    finally:
        capture.release()


def _read_frame_at(capture: object, index: int, total: int) -> bool:
    import cv2

    if total > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))  # type: ignore[attr-defined]
    ok, frame = capture.read()  # type: ignore[attr-defined]
    return bool(ok) and frame is not None


def _cleanup_failed_ingest(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink(missing_ok=True)
    except OSError:
        pass


# Re-export helpers callers often need alongside ingest.
__all__ = [
    "ALLOWED_CONTAINERS",
    "ALLOWED_VIDEO_CODECS",
    "IngestedVideo",
    "VideoIngestLimits",
    "assert_regular_owned_file",
    "assert_safe_destination",
    "assert_safe_storage_dir",
    "assert_stored_video_matches_record",
    "ingest_uploaded_video",
    "require_playable_video",
    "resolve_stored_video_path",
    "sha256_file",
    "validate_video_extension",
]
