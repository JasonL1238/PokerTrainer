from pathlib import Path
import sqlite3

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from poker_tracker.db import PokerDatabase
from poker_tracker.image_utils import crop_roi_from_image, save_roi_crop_preview
from poker_tracker.models import ExtractedFrame, ProcessingJob, ROIProfile, ROIRegion, VideoRecord
from poker_tracker.roi import clip_roi_to_bounds, validate_roi_bounds
from poker_tracker.roi_profiles import (
    create_starter_clubwpt_profile,
    duplicate_roi_profile,
    export_roi_profile,
    generate_roi_crop_previews,
    import_roi_profile,
)


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_create_roi_profile_region_fetch_update_delete() -> None:
    db = make_db()
    profile = db.create_roi_profile(ROIProfile(name="ClubWPT test", video_width=100, video_height=80))
    region = db.create_roi_region(
        ROIRegion(
            profile_id=profile.id,
            roi_key="hero_card_1",
            roi_type="hero_card",
            label="Hero card 1",
            x=10,
            y=12,
            width=20,
            height=24,
        )
    )

    saved = db.fetch_roi_regions_by_profile(profile.id)[0]
    updated = saved.model_copy(update={"x": 15, "label": "Hero card left"})
    db.update_roi_region(updated)

    assert profile.id is not None
    assert region.id is not None
    assert db.fetch_roi_regions_by_profile(profile.id)[0].x == 15
    assert db.fetch_roi_regions_by_profile(profile.id)[0].label == "Hero card left"

    db.delete_roi_region(region.id)
    assert db.fetch_roi_regions_by_profile(profile.id) == []
    db.close()


def test_mark_profile_active_and_duplicate() -> None:
    db = make_db()
    first = db.create_roi_profile(ROIProfile(name="First"))
    second = db.create_roi_profile(ROIProfile(name="Second"))
    db.create_roi_region(
        ROIRegion(
            profile_id=first.id,
            roi_key="pot_total",
            roi_type="pot",
            label="Pot",
            x=1,
            y=2,
            width=10,
            height=12,
        )
    )

    db.mark_roi_profile_active(second.id)
    duplicate = duplicate_roi_profile(db, first.id, new_name="First copy")

    assert db.fetch_roi_profile(first.id).is_active is False
    assert db.fetch_roi_profile(second.id).is_active is True
    assert duplicate.name == "First copy"
    assert db.fetch_roi_regions_by_profile(duplicate.id)[0].roi_key == "pot_total"
    db.close()


def test_mark_missing_profile_active_does_not_deactivate_existing_profile() -> None:
    db = make_db()
    active = db.create_roi_profile(ROIProfile(name="Active", is_active=True))

    with pytest.raises(ValueError):
        db.mark_roi_profile_active(9999)

    assert db.fetch_roi_profile(active.id).is_active is True
    db.close()


def test_db_rejects_roi_region_outside_profile_dimensions_and_missing_profile() -> None:
    db = make_db()
    profile = db.create_roi_profile(ROIProfile(name="Bounded", video_width=50, video_height=50))

    with pytest.raises(ValueError, match="image width"):
        db.create_roi_region(
            ROIRegion(
                profile_id=profile.id,
                roi_key="too_wide",
                x=40,
                y=1,
                width=20,
                height=5,
            )
        )

    with pytest.raises(ValueError, match="ROI profile not found"):
        db.create_roi_region(
            ROIRegion(
                profile_id=9999,
                roi_key="missing_profile",
                x=1,
                y=1,
                width=5,
                height=5,
            )
        )
    db.close()


def test_duplicate_roi_key_rejected_by_database() -> None:
    db = make_db()
    profile = db.create_roi_profile(ROIProfile(name="Unique keys"))
    region = ROIRegion(profile_id=profile.id, roi_key="pot_total", x=1, y=1, width=5, height=5)
    db.create_roi_region(region)

    with pytest.raises(sqlite3.IntegrityError):
        db.create_roi_region(region.model_copy(update={"id": None}))

    db.close()


def test_roi_coordinate_and_bounds_validation() -> None:
    region = ROIRegion(
        profile_id=1,
        roi_key="board",
        roi_type="board_card",
        x=10,
        y=10,
        width=20,
        height=20,
    )
    validate_roi_bounds(region, image_width=40, image_height=40)

    with pytest.raises(ValueError):
        validate_roi_bounds(region, image_width=25, image_height=40)

    clipped = clip_roi_to_bounds(region, image_width=25, image_height=25)
    assert clipped.width == 15
    assert clipped.height == 15

    with pytest.raises(ValidationError):
        ROIRegion(profile_id=1, roi_key="bad", x=0, y=0, width=0, height=1)


def test_crop_roi_from_synthetic_image_and_save_preview(tmp_path: Path) -> None:
    image_path = create_synthetic_image(tmp_path / "frame.jpg")
    region = ROIRegion(
        profile_id=1,
        roi_key="hero_card_1",
        roi_type="hero_card",
        x=5,
        y=7,
        width=15,
        height=12,
    )
    frame = ExtractedFrame(
        id=42,
        video_id=1,
        job_id=1,
        timestamp_seconds=1.5,
        frame_index=15,
        image_path=str(image_path),
    )

    crop = crop_roi_from_image(image_path, region)
    result = save_roi_crop_preview(frame, region, previews_dir=tmp_path / "roi_previews")

    assert crop.shape[:2] == (12, 15)
    assert Path(result.crop_path).exists()
    assert result.crop_width == 15
    assert result.crop_height == 12
    assert result.source_frame_id == 42


def test_crop_roi_rejects_missing_and_out_of_bounds_images(tmp_path: Path) -> None:
    image_path = create_synthetic_image(tmp_path / "frame.jpg")
    outside = ROIRegion(profile_id=1, roi_key="outside", x=60, y=40, width=20, height=20)

    with pytest.raises(ValueError, match="Could not read image"):
        crop_roi_from_image(tmp_path / "missing.jpg", outside)

    with pytest.raises(ValueError, match="image width"):
        crop_roi_from_image(image_path, outside)

    clipped = crop_roi_from_image(image_path, outside, clip_to_bounds=True)
    assert clipped.shape[0] > 0
    assert clipped.shape[1] > 0


def test_generate_roi_crop_previews_from_db(tmp_path: Path) -> None:
    db = make_db()
    image_path = create_synthetic_image(tmp_path / "frame.jpg")
    frame = create_saved_frame(db, image_path)
    profile = db.create_roi_profile(ROIProfile(name="Preview profile", video_width=64, video_height=48))
    db.create_roi_region(
        ROIRegion(
            profile_id=profile.id,
            roi_key="pot_total",
            roi_type="pot",
            label="Pot",
            x=1,
            y=2,
            width=10,
            height=10,
        )
    )

    results = generate_roi_crop_previews(db, profile.id, frame.id, previews_dir=tmp_path / "previews")

    assert len(results) == 1
    assert Path(results[0].crop_path).exists()
    db.close()


def test_generate_roi_crop_previews_rejects_missing_ids(tmp_path: Path) -> None:
    db = make_db()
    image_path = create_synthetic_image(tmp_path / "frame.jpg")
    frame = create_saved_frame(db, image_path)
    profile = db.create_roi_profile(ROIProfile(name="Preview profile", video_width=64, video_height=48))

    with pytest.raises(ValueError, match="ROI profile not found"):
        generate_roi_crop_previews(db, 9999, frame.id, previews_dir=tmp_path / "previews")

    with pytest.raises(ValueError, match="Extracted frame not found"):
        generate_roi_crop_previews(db, profile.id, 9999, previews_dir=tmp_path / "previews")
    db.close()


def test_roi_profile_json_export_import_round_trip() -> None:
    db = make_db()
    profile = db.create_roi_profile(ROIProfile(name="Export profile", video_width=100, video_height=100))
    db.create_roi_region(
        ROIRegion(
            profile_id=profile.id,
            roi_key="seat_1_stack",
            roi_type="player_stack",
            label="Seat 1 stack",
            x=5,
            y=6,
            width=20,
            height=10,
            seat_index=1,
        )
    )

    payload = export_roi_profile(db, profile.id)
    imported = import_roi_profile(db, payload)

    imported_regions = db.fetch_roi_regions_by_profile(imported.id)
    assert imported.id != profile.id
    assert imported.name == "Export profile"
    assert imported_regions[0].roi_key == "seat_1_stack"
    assert imported_regions[0].seat_index == 1
    db.close()


def test_roi_profile_import_rejects_bad_region_without_creating_profile() -> None:
    db = make_db()
    payload = {
        "export_version": 1,
        "profile": ROIProfile(name="Bad import", video_width=50, video_height=50).model_dump(mode="json"),
        "regions": [
            ROIRegion(
                profile_id=1,
                roi_key="bad",
                x=45,
                y=1,
                width=20,
                height=10,
            ).model_dump(mode="json")
        ],
    }

    with pytest.raises(ValueError, match="image width"):
        import_roi_profile(db, payload)

    assert db.fetch_roi_profiles() == []
    db.close()


def test_roi_profile_delete_cascades_regions() -> None:
    db = make_db()
    profile = db.create_roi_profile(ROIProfile(name="Delete cascade"))
    db.create_roi_region(ROIRegion(profile_id=profile.id, roi_key="pot_total", x=1, y=1, width=5, height=5))

    db.delete_roi_profile(profile.id)

    assert db.fetch_roi_profile(profile.id) is None
    assert db.fetch_roi_regions_by_profile(profile.id) == []
    db.close()


def test_starter_preset_creation() -> None:
    db = make_db()
    profile = create_starter_clubwpt_profile(db, max_seats=6, video_width=1920, video_height=1080)
    regions = db.fetch_roi_regions_by_profile(profile.id)
    keys = {region.roi_key for region in regions}

    assert "hero_card_1" in keys
    assert "river" in keys
    assert "seat_6_stack" in keys
    assert "seat_6_bet" in keys
    assert "seat_6_name" in keys
    assert "table_area" in keys
    db.close()


def create_synthetic_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 60)
    image[7:19, 5:20] = (200, 100, 50)
    assert cv2.imwrite(str(path), image)
    return path


def create_saved_frame(db: PokerDatabase, image_path: Path) -> ExtractedFrame:
    video = db.create_video(
        VideoRecord(
            original_filename="synthetic.avi",
            stored_path=str(image_path.with_suffix(".avi")),
            file_size_bytes=0,
        )
    )
    job = db.create_processing_job(ProcessingJob(video_id=video.id, job_type="frame_extraction"))
    return db.create_extracted_frame(
        ExtractedFrame(
            video_id=video.id,
            job_id=job.id,
            timestamp_seconds=0.5,
            frame_index=5,
            image_path=str(image_path),
        )
    )
