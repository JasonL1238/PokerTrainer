"""Tests for hardened completed-session video ingest."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_tracker.ui.video_ingest import (
    VideoIngestLimits,
    assert_regular_owned_file,
    assert_stored_video_matches_record,
    ingest_uploaded_video,
    require_playable_video,
    sha256_file,
)


def create_synthetic_video(path: Path, *, frames: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 10 % 255, index * 5 % 255, index * 20 % 255)
        writer.write(frame)
    writer.release()
    return path


def test_ingest_uploaded_video_stores_playable_file(tmp_path: Path) -> None:
    source = create_synthetic_video(tmp_path / "source.avi")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with source.open("rb") as handle:
        ingested = ingest_uploaded_video(handle, "My Session.avi", videos_dir)

    assert ingested.path.exists()
    assert ingested.path.parent == videos_dir
    assert ingested.path.name.endswith("my_session.avi")
    assert ingested.file_size_bytes == ingested.path.stat().st_size
    assert ingested.content_sha256 == sha256_file(ingested.path)
    assert ingested.metadata.error == ""
    assert ingested.metadata.width == 64
    assert ingested.metadata.height == 48
    assert ingested.metadata.frame_count and ingested.metadata.frame_count >= 15


def test_ingest_rejects_empty_upload(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with empty.open("rb") as handle:
        with pytest.raises(ValueError, match="empty"):
            ingest_uploaded_video(handle, "empty.mp4", videos_dir)

    assert list(videos_dir.iterdir()) == []


def test_ingest_rejects_corrupt_upload_and_cleans_up(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not-a-video")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with corrupt.open("rb") as handle:
        with pytest.raises(ValueError):
            ingest_uploaded_video(handle, "corrupt.mp4", videos_dir)

    leftover = [path for path in videos_dir.iterdir()]
    assert leftover == []


def test_ingest_rejects_oversize_upload(tmp_path: Path) -> None:
    source = tmp_path / "big.mp4"
    source.write_bytes(b"0123456789")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    limits = VideoIngestLimits(max_bytes=4)

    with source.open("rb") as handle:
        with pytest.raises(ValueError, match="maximum size"):
            ingest_uploaded_video(handle, "big.mp4", videos_dir, limits=limits)

    assert list(videos_dir.iterdir()) == []


def test_require_playable_video_rejects_symlink(tmp_path: Path) -> None:
    real = create_synthetic_video(tmp_path / "real.avi")
    link = tmp_path / "linked.avi"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="Symlinked"):
        require_playable_video(link)


def test_assert_regular_owned_file_rejects_hardlink(tmp_path: Path) -> None:
    real = create_synthetic_video(tmp_path / "real.avi")
    linked = tmp_path / "hard.avi"
    try:
        os.link(real, linked)
    except OSError:
        pytest.skip("hardlinks unavailable on this filesystem")

    with pytest.raises(ValueError, match="Hard-linked"):
        assert_regular_owned_file(linked)


def test_assert_stored_video_matches_record_detects_truncation(tmp_path: Path) -> None:
    path = create_synthetic_video(tmp_path / "session.avi")
    expected = path.stat().st_size
    digest = sha256_file(path)

    assert_stored_video_matches_record(
        path,
        expected_size_bytes=expected,
        expected_sha256=digest,
    )

    path.write_bytes(path.read_bytes()[: max(1, expected // 2)])
    with pytest.raises(ValueError, match="size mismatch"):
        assert_stored_video_matches_record(path, expected_size_bytes=expected)


def test_assert_stored_video_matches_record_detects_hash_mismatch(
    tmp_path: Path,
) -> None:
    path = create_synthetic_video(tmp_path / "session.avi")
    with pytest.raises(ValueError, match="hash mismatch"):
        assert_stored_video_matches_record(
            path,
            expected_size_bytes=path.stat().st_size,
            expected_sha256="0" * 64,
        )


def test_ingest_rejects_symlink_storage_dir(tmp_path: Path) -> None:
    real_dir = tmp_path / "real_videos"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked_videos"
    linked_dir.symlink_to(real_dir)
    source = create_synthetic_video(tmp_path / "source.avi")

    with source.open("rb") as handle:
        with pytest.raises(ValueError, match="symlink"):
            ingest_uploaded_video(handle, "session.avi", linked_dir)
