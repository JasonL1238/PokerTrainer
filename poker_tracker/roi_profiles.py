from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from poker_tracker.db import PokerDatabase
from poker_tracker.image_utils import save_roi_crop_preview
from poker_tracker.models import ROICropResult, ROIProfile, ROIRegion
from poker_tracker.roi import common_roi_keys, validate_roi_bounds
from poker_tracker.video_storage import ROI_PREVIEWS_DIR


ROI_EXPORT_VERSION = 1


def create_starter_clubwpt_profile(
    db: PokerDatabase,
    *,
    name: str = "ClubWPT Gold starter profile",
    video_width: int | None = None,
    video_height: int | None = None,
    max_seats: int = 9,
) -> ROIProfile:
    """Create a starter profile with placeholder coordinates for manual editing."""
    profile = db.create_roi_profile(
        ROIProfile(
            name=name,
            description="Starter placeholder profile. Edit coordinates from extracted frames.",
            platform="ClubWPT Gold",
            table_layout=f"{max_seats}-max placeholder",
            video_width=video_width,
            video_height=video_height,
        )
    )
    for definition in common_roi_keys(max_seats):
        db.create_roi_region(
            ROIRegion(
                profile_id=profile.id,
                roi_key=definition["roi_key"],
                roi_type=definition["roi_type"],
                label=definition["label"],
                x=0,
                y=0,
                width=1,
                height=1,
                seat_index=definition.get("seat_index"),
                card_index=definition.get("card_index"),
                notes="Placeholder coordinates. Calibrate from a completed-session frame.",
            )
        )
    return profile


def duplicate_roi_profile(
    db: PokerDatabase,
    profile_id: int,
    *,
    new_name: str | None = None,
) -> ROIProfile:
    """Duplicate an ROI profile and all regions without reusing IDs."""
    source = db.fetch_roi_profile(profile_id)
    if source is None:
        raise ValueError(f"ROI profile not found: {profile_id}")
    duplicate = db.create_roi_profile(
        ROIProfile(
            name=new_name or f"{source.name} copy",
            description=source.description,
            platform=source.platform,
            table_layout=source.table_layout,
            video_width=source.video_width,
            video_height=source.video_height,
        )
    )
    for region in db.fetch_roi_regions_by_profile(profile_id):
        db.create_roi_region(
            region.model_copy(
                update={
                    "id": None,
                    "profile_id": duplicate.id,
                    "created_at": datetime.now().astimezone(),
                    "updated_at": datetime.now().astimezone(),
                }
            )
        )
    return duplicate


def export_roi_profile(db: PokerDatabase, profile_id: int) -> dict[str, Any]:
    """Export an ROI profile and all regions as JSON-compatible data."""
    profile = db.fetch_roi_profile(profile_id)
    if profile is None:
        raise ValueError(f"ROI profile not found: {profile_id}")
    return {
        "export_version": ROI_EXPORT_VERSION,
        "profile": _dump_model(profile),
        "regions": [_dump_model(region) for region in db.fetch_roi_regions_by_profile(profile_id)],
    }


def import_roi_profile(db: PokerDatabase, payload: dict[str, Any]) -> ROIProfile:
    """Import an ROI profile from JSON, creating new profile and region IDs."""
    version = payload.get("export_version", ROI_EXPORT_VERSION)
    if version != ROI_EXPORT_VERSION:
        raise ValueError(
            f"Unsupported export_version {version}; this app understands version {ROI_EXPORT_VERSION}."
        )
    profile_data = dict(payload["profile"])
    profile_data.pop("id", None)
    profile_data["is_active"] = False
    profile_model = ROIProfile(**profile_data)

    imported_regions: list[ROIRegion] = []
    for region_data in payload.get("regions", []):
        imported = dict(region_data)
        imported.pop("id", None)
        imported["profile_id"] = 0
        region = ROIRegion(**imported)
        validate_roi_bounds(
            region,
            image_width=profile_model.video_width,
            image_height=profile_model.video_height,
        )
        imported_regions.append(region)

    profile = db.create_roi_profile(profile_model)
    for region in imported_regions:
        db.create_roi_region(region.model_copy(update={"profile_id": profile.id}))
    return profile


def generate_roi_crop_previews(
    db: PokerDatabase,
    profile_id: int,
    frame_id: int,
    *,
    previews_dir: Path = ROI_PREVIEWS_DIR,
) -> list[ROICropResult]:
    """Generate preview crops for all regions in a profile from one extracted frame."""
    profile = db.fetch_roi_profile(profile_id)
    frame = db.fetch_extracted_frame(frame_id)
    if profile is None:
        raise ValueError(f"ROI profile not found: {profile_id}")
    if frame is None:
        raise ValueError(f"Extracted frame not found: {frame_id}")
    results: list[ROICropResult] = []
    for region in db.fetch_roi_regions_by_profile(profile_id):
        results.append(save_roi_crop_preview(frame, region, previews_dir=previews_dir))
    return results


def _dump_model(model: Any) -> dict[str, Any]:
    data = model.model_dump()
    for key, value in list(data.items()):
        if isinstance(value, (date, datetime)):
            data[key] = value.isoformat()
    return data


# TODO: Future interactive rectangle drawing should update ROIRegion rows through these helpers.
