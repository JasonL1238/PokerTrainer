from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO


DATA_DIR = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
EXPORTS_DIR = DATA_DIR / "exports"
ROI_PREVIEWS_DIR = DATA_DIR / "roi_previews"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def ensure_data_directories(base_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Create local post-session storage directories if missing."""
    videos = base_dir / "videos"
    frames = base_dir / "frames"
    exports = base_dir / "exports"
    roi_previews = base_dir / "roi_previews"
    for path in (base_dir, videos, frames, exports, roi_previews):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "data": base_dir,
        "videos": videos,
        "frames": frames,
        "exports": exports,
        "roi_previews": roi_previews,
    }


def validate_video_extension(filename: str) -> str:
    """Return a validated lowercase extension for supported video files."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {extension or 'none'}")
    return extension


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe filename preserving the extension."""
    extension = validate_video_extension(filename)
    stem = Path(filename).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9._-]+", "_", stem).strip("._-")
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
) -> Path:
    """Save an uploaded completed-session video to local disk."""
    destination = unique_stored_video_path(original_filename, videos_dir)
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)
    return destination


def video_frame_dir(video_id: int, frames_dir: Path = FRAMES_DIR) -> Path:
    """Return the frame output directory for one video."""
    path = frames_dir / f"video_{video_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path
