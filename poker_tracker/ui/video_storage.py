from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO

# NOTE: there is deliberately no BACKUPS_DIR here. It used to be defined in this
# module as well as in poker_tracker.persistence.backup, and the two copies drifted
# silently: redirecting this one -- the historical location, and the obvious place
# to monkeypatch -- redirected no backup at all, because backup_database resolves
# poker_tracker.persistence.backup.BACKUPS_DIR at call time. That module owns the
# value; import it from there, and patch it there.

# Anchored to the project root so the app behaves the same regardless of the
# working directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
EXPORTS_DIR = DATA_DIR / "exports"
ROI_PREVIEWS_DIR = DATA_DIR / "roi_previews"
CV_TIMELINES_DIR = DATA_DIR / "cv_timelines"
JOB_LOGS_DIR = DATA_DIR / "job_logs"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def ensure_data_directories(base_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Create local post-session storage directories if missing."""
    videos = base_dir / "videos"
    frames = base_dir / "frames"
    exports = base_dir / "exports"
    roi_previews = base_dir / "roi_previews"
    cv_timelines = base_dir / "cv_timelines"
    backups = base_dir / "backups"
    job_logs = base_dir / "job_logs"
    for path in (base_dir, videos, frames, exports, roi_previews, cv_timelines, backups, job_logs):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "data": base_dir,
        "videos": videos,
        "frames": frames,
        "exports": exports,
        "roi_previews": roi_previews,
        "cv_timelines": cv_timelines,
        "backups": backups,
        "job_logs": job_logs,
    }


def validate_video_extension(filename: str) -> str:
    """Return a validated lowercase extension for supported video files."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {extension or 'none'}")
    return extension


# Every common filesystem (ext4, APFS, HFS+, XFS, NTFS) caps a single path
# COMPONENT at 255 bytes, and unique_stored_video_path prepends a uuid4 hex plus
# an underscore to whatever this function returns. Sanitization leaves only
# [a-z0-9._-], so byte length equals character length and one budget covers both.
_MAX_STORED_NAME_BYTES = 255
_STORED_NAME_PREFIX_BYTES = 33  # len(uuid4().hex) + the "_" separator


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe filename preserving the extension.

    The stem is bounded so the name this returns still fits inside a path
    component once ``unique_stored_video_path`` has prefixed it. Without the
    bound an operator whose upload carried a long name got ENAMETOOLONG out of
    ``os.replace`` at the END of ``save_video_file`` -- after the whole video had
    been streamed to a temp file -- as a bare OSError with the 10k-character path
    in its message, rather than a stored video. Truncation cannot collide,
    because uniqueness comes from the uuid prefix and not from the stem.
    """
    extension = validate_video_extension(filename)
    stem = Path(filename).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9._-]+", "_", stem).strip("._-")
    budget = _MAX_STORED_NAME_BYTES - _STORED_NAME_PREFIX_BYTES - len(extension)
    # Re-strip after the cut: a truncation landing mid-run can expose trailing
    # "._-" that the first strip had buried, and ".mp4" must not follow a dot.
    stem = stem[:budget].strip("._-")
    stem = stem or "video"
    return f"{stem}{extension}"


def unique_stored_video_path(original_filename: str, videos_dir: Path = VIDEOS_DIR) -> Path:
    """Build a collision-resistant stored video path."""
    ensure_data_directories(videos_dir.parent)
    safe_name = safe_filename(original_filename)
    return videos_dir / f"{uuid.uuid4().hex}_{safe_name}"


def save_video_file(
    source: BinaryIO,
    original_filename: str,
    videos_dir: Path = VIDEOS_DIR,
    *,
    max_bytes: int | None = None,
) -> Path:
    """Atomically save an uploaded completed-session video to local disk.

    Prefer ``poker_tracker.ui.video_ingest.ingest_uploaded_video`` when the
    caller also needs playable-media validation before scheduling CV.
    """
    destination = unique_stored_video_path(original_filename, videos_dir)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination already exists: {destination}")

    fd, temp_name = tempfile.mkstemp(
        prefix=".upload_",
        suffix=destination.suffix.lower() or ".part",
        dir=str(destination.parent),
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
                if max_bytes is not None and written > max_bytes:
                    raise ValueError(
                        f"Video exceeds maximum size of {max_bytes} bytes."
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written < 1:
            raise ValueError("Video file is empty.")
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return destination


def video_frame_dir(video_id: int, frames_dir: Path = FRAMES_DIR) -> Path:
    """Return the frame output directory for one video."""
    path = frames_dir / f"video_{video_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path
