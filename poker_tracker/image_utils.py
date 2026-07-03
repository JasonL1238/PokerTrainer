from __future__ import annotations

from pathlib import Path

import cv2

from poker_tracker.models import ExtractedFrame, ROICropResult, ROIRegion
from poker_tracker.roi import clip_roi_to_bounds, safe_roi_key, validate_roi_bounds
from poker_tracker.video_storage import ROI_PREVIEWS_DIR, ensure_data_directories


def image_dimensions(image_path: str | Path) -> tuple[int, int]:
    """Return image width and height for a stored extracted frame."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def crop_roi_from_image(
    image_path: str | Path,
    region: ROIRegion,
    *,
    clip_to_bounds: bool = False,
):
    """Crop an ROI from an image file without running detection or OCR."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image_height, image_width = image.shape[:2]
    effective_region = region
    if clip_to_bounds:
        effective_region = clip_roi_to_bounds(region, image_width, image_height)
    else:
        validate_roi_bounds(region, image_width=image_width, image_height=image_height)
    y1 = effective_region.y
    y2 = effective_region.y + effective_region.height
    x1 = effective_region.x
    x2 = effective_region.x + effective_region.width
    return image[y1:y2, x1:x2]


def save_roi_crop_preview(
    frame: ExtractedFrame,
    region: ROIRegion,
    *,
    previews_dir: Path = ROI_PREVIEWS_DIR,
    clip_to_bounds: bool = True,
) -> ROICropResult:
    """Save one ROI crop preview image for manual calibration QA."""
    ensure_data_directories(previews_dir.parent)
    crop = crop_roi_from_image(frame.image_path, region, clip_to_bounds=clip_to_bounds)
    if crop.size == 0:
        raise ValueError(f"ROI produced an empty crop: {region.roi_key}")
    frame_label = f"frame_{frame.id}" if frame.id is not None else Path(frame.image_path).stem
    output_dir = previews_dir / frame_label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_roi_key(region.roi_key)}.jpg"
    if not cv2.imwrite(str(output_path), crop):
        raise ValueError(f"Could not write ROI crop preview: {output_path}")
    crop_height, crop_width = crop.shape[:2]
    return ROICropResult(
        roi_key=region.roi_key,
        roi_type=region.roi_type,
        source_frame_id=frame.id,
        source_timestamp_seconds=frame.timestamp_seconds,
        source_image_path=frame.image_path,
        crop_path=str(output_path),
        crop_width=crop_width,
        crop_height=crop_height,
    )


# TODO: Future CV/OCR modules should consume these crops, not raw Streamlit UI state.
