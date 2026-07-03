from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_tracker.db import PokerDatabase
from poker_tracker.frame_extraction import (
    delete_extracted_frames,
    extract_frames_for_video,
    select_representative_frames,
)
from poker_tracker.jobs import create_job, mark_completed, mark_failed, mark_running, update_progress
from poker_tracker.models import VideoRecord
from poker_tracker.video_metadata import extract_video_metadata
from poker_tracker.video_storage import (
    ensure_data_directories,
    safe_filename,
    save_video_file,
    validate_video_extension,
)


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_video_storage_path_creation(tmp_path: Path) -> None:
    paths = ensure_data_directories(tmp_path / "data")

    assert paths["videos"].exists()
    assert paths["frames"].exists()
    assert paths["exports"].exists()


def test_safe_filename_generation_and_extension_validation() -> None:
    assert safe_filename("Session 01!.MP4") == "session_01.mp4"
    assert validate_video_extension("hand_review.mov") == ".mov"

    with pytest.raises(ValueError):
        validate_video_extension("live_capture.exe")


def test_save_video_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video-bytes")
    with source.open("rb") as input_file:
        saved = save_video_file(input_file, "My Session.mp4", tmp_path / "videos")

    assert saved.exists()
    assert saved.read_bytes() == b"video-bytes"
    assert saved.name.endswith("my_session.mp4")


def test_insert_video_metadata_and_jobs(tmp_path: Path) -> None:
    db = make_db()
    path = tmp_path / "video.avi"
    path.write_bytes(b"placeholder")
    video = db.create_video(
        VideoRecord(
            original_filename="video.avi",
            stored_path=str(path),
            file_size_bytes=path.stat().st_size,
            duration_seconds=2.0,
            fps=10.0,
            width=64,
            height=48,
            frame_count=20,
        )
    )

    job = create_job(db, video.id)
    mark_running(db, job.id)
    update_progress(db, job.id, 55, "Halfway")
    mark_completed(db, job.id)

    saved_video = db.fetch_video(video.id)
    saved_job = db.fetch_jobs_by_video(video.id)[0]
    assert saved_video.original_filename == "video.avi"
    assert saved_video.frame_count == 20
    assert saved_job.status == "completed"
    assert saved_job.progress_percent == 100

    db.close()


def test_metadata_extraction_on_synthetic_video(tmp_path: Path) -> None:
    video_path = create_synthetic_video(tmp_path / "synthetic.avi")

    metadata = extract_video_metadata(video_path)

    assert metadata.error == ""
    assert metadata.fps == pytest.approx(10, rel=0.2)
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.frame_count and metadata.frame_count >= 15
    assert metadata.duration_seconds and metadata.duration_seconds > 1


def test_metadata_extraction_fails_gracefully_for_missing_file(tmp_path: Path) -> None:
    metadata = extract_video_metadata(tmp_path / "missing.avi")

    assert metadata.error.startswith("Video file not found")
    assert metadata.duration_seconds is None


def test_job_progress_clamps_and_missing_job_errors(tmp_path: Path) -> None:
    db = make_db()
    path = tmp_path / "video.avi"
    path.write_bytes(b"placeholder")
    video = db.create_video(
        VideoRecord(
            original_filename="video.avi",
            stored_path=str(path),
            file_size_bytes=path.stat().st_size,
        )
    )
    job = create_job(db, video.id)

    update_progress(db, job.id, 500, "Too high")
    assert db.fetch_processing_job(job.id).progress_percent == 100

    update_progress(db, job.id, -50, "Too low")
    assert db.fetch_processing_job(job.id).progress_percent == 0

    mark_failed(db, job.id, "bad video")
    failed = db.fetch_processing_job(job.id)
    assert failed.status == "failed"
    assert failed.error_message == "bad video"

    with pytest.raises(ValueError):
        update_progress(db, 9999, 10, "missing")

    db.close()


def test_frame_extraction_stores_records_and_respects_max_frames(tmp_path: Path) -> None:
    db = make_db()
    video_path = create_synthetic_video(tmp_path / "synthetic.avi")
    metadata = extract_video_metadata(video_path)
    video = db.create_video(
        VideoRecord(
            original_filename="synthetic.avi",
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
            duration_seconds=metadata.duration_seconds,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            frame_count=metadata.frame_count,
        )
    )

    summary = extract_frames_for_video(
        db,
        video.id,
        frames_per_second=2,
        max_frames=3,
        frames_dir=tmp_path / "frames",
    )
    frames = db.fetch_frames_by_video(video.id)
    jobs = db.fetch_jobs_by_video(video.id)

    assert summary.frames_extracted == 3
    assert len(frames) == 3
    assert all(Path(frame.image_path).exists() for frame in frames)
    assert jobs[0].status == "completed"
    assert jobs[0].progress_percent == 100

    db.close()


def test_frame_extraction_validates_inputs_and_marks_corrupt_video_failed(tmp_path: Path) -> None:
    db = make_db()
    corrupt = tmp_path / "corrupt.avi"
    corrupt.write_bytes(b"not actually a video")
    video = db.create_video(
        VideoRecord(
            original_filename="corrupt.avi",
            stored_path=str(corrupt),
            file_size_bytes=corrupt.stat().st_size,
        )
    )

    with pytest.raises(ValueError, match="Video not found"):
        extract_frames_for_video(db, 9999, frames_dir=tmp_path / "frames")
    with pytest.raises(ValueError, match="frames_per_second"):
        extract_frames_for_video(db, video.id, frames_per_second=0, frames_dir=tmp_path / "frames")
    with pytest.raises(ValueError, match="interval_seconds"):
        extract_frames_for_video(db, video.id, interval_seconds=0, frames_dir=tmp_path / "frames")
    with pytest.raises(ValueError, match="max_frames"):
        extract_frames_for_video(db, video.id, max_frames=0, frames_dir=tmp_path / "frames")

    with pytest.raises(ValueError, match="Could not open video"):
        extract_frames_for_video(db, video.id, max_frames=1, frames_dir=tmp_path / "frames")

    assert db.fetch_jobs_by_video(video.id)[0].status == "failed"
    db.close()


def test_frame_extraction_rejects_end_before_start(tmp_path: Path) -> None:
    db = make_db()
    video_path = create_synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename="synthetic.avi",
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )

    with pytest.raises(ValueError, match="end_time_seconds"):
        extract_frames_for_video(
            db,
            video.id,
            start_time_seconds=2,
            end_time_seconds=1,
            frames_dir=tmp_path / "frames",
        )

    assert db.fetch_jobs_by_video(video.id)[0].status == "failed"
    db.close()


def test_deleting_extracted_frames(tmp_path: Path) -> None:
    db = make_db()
    video_path = create_synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename="synthetic.avi",
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )
    extract_frames_for_video(db, video.id, max_frames=2, frames_dir=tmp_path / "frames")

    deleted = delete_extracted_frames(db, video.id, frames_dir=tmp_path / "frames")

    assert deleted == 2
    assert db.fetch_frames_by_video(video.id) == []

    db.close()


def test_deleting_extracted_frames_handles_missing_files(tmp_path: Path) -> None:
    db = make_db()
    video_path = create_synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename="synthetic.avi",
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )
    extract_frames_for_video(db, video.id, max_frames=1, frames_dir=tmp_path / "frames")
    frame = db.fetch_frames_by_video(video.id)[0]
    Path(frame.image_path).unlink()

    assert delete_extracted_frames(db, video.id, frames_dir=tmp_path / "frames") == 1
    assert db.fetch_frames_by_video(video.id) == []
    db.close()


def test_select_representative_frames_limit(tmp_path: Path) -> None:
    db = make_db()
    video_path = create_synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename="synthetic.avi",
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )
    extract_frames_for_video(db, video.id, max_frames=5, frames_dir=tmp_path / "frames")

    selected = select_representative_frames(db.fetch_frames_by_video(video.id), limit=3)

    assert len(selected) == 3

    db.close()


def create_synthetic_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(20):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 10 % 255, index * 5 % 255, index * 20 % 255)
        writer.write(frame)
    writer.release()
    return path
