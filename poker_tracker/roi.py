from __future__ import annotations

import re
from typing import Any

from poker_tracker.models import ROIRegion, ROIType


ROI_TYPES: tuple[ROIType, ...] = (
    "hero_card",
    "board_card",
    "pot",
    "player_stack",
    "player_bet",
    "player_name",
    "dealer_button",
    "active_indicator",
    "action_button",
    "table_area",
    "unknown",
)


def common_roi_keys(max_seats: int = 9) -> list[dict[str, Any]]:
    """Return starter ROI definitions for a fixed post-session table layout."""
    keys: list[dict[str, Any]] = [
        {"roi_key": "hero_card_1", "roi_type": "hero_card", "label": "Hero card 1", "card_index": 1},
        {"roi_key": "hero_card_2", "roi_type": "hero_card", "label": "Hero card 2", "card_index": 2},
        {"roi_key": "flop_1", "roi_type": "board_card", "label": "Flop card 1", "card_index": 1},
        {"roi_key": "flop_2", "roi_type": "board_card", "label": "Flop card 2", "card_index": 2},
        {"roi_key": "flop_3", "roi_type": "board_card", "label": "Flop card 3", "card_index": 3},
        {"roi_key": "turn", "roi_type": "board_card", "label": "Turn", "card_index": 4},
        {"roi_key": "river", "roi_type": "board_card", "label": "River", "card_index": 5},
        {"roi_key": "pot_total", "roi_type": "pot", "label": "Total pot"},
        {"roi_key": "dealer_button_area", "roi_type": "dealer_button", "label": "Dealer button area"},
        {"roi_key": "action_button_area", "roi_type": "action_button", "label": "Action button area"},
        {"roi_key": "table_area", "roi_type": "table_area", "label": "Main table area"},
    ]
    for seat in range(1, max_seats + 1):
        keys.extend(
            [
                {
                    "roi_key": f"seat_{seat}_stack",
                    "roi_type": "player_stack",
                    "label": f"Seat {seat} stack",
                    "seat_index": seat,
                },
                {
                    "roi_key": f"seat_{seat}_bet",
                    "roi_type": "player_bet",
                    "label": f"Seat {seat} bet",
                    "seat_index": seat,
                },
                {
                    "roi_key": f"seat_{seat}_name",
                    "roi_type": "player_name",
                    "label": f"Seat {seat} name",
                    "seat_index": seat,
                },
                {
                    "roi_key": f"seat_{seat}_active_indicator",
                    "roi_type": "active_indicator",
                    "label": f"Seat {seat} active indicator",
                    "seat_index": seat,
                },
                {
                    "roi_key": f"seat_{seat}_dealer_button_area",
                    "roi_type": "dealer_button",
                    "label": f"Seat {seat} dealer button area",
                    "seat_index": seat,
                },
            ]
        )
    return keys


def validate_roi_bounds(
    region: ROIRegion,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Validate that ROI coordinates are positive and optionally inside image bounds."""
    if region.x < 0 or region.y < 0:
        raise ValueError("ROI x/y coordinates must be non-negative.")
    if region.width <= 0 or region.height <= 0:
        raise ValueError("ROI width and height must be positive.")
    if image_width is not None and region.x + region.width > image_width:
        raise ValueError("ROI extends beyond the image width.")
    if image_height is not None and region.y + region.height > image_height:
        raise ValueError("ROI extends beyond the image height.")


def clip_roi_to_bounds(region: ROIRegion, image_width: int, image_height: int) -> ROIRegion:
    """Return a copy clipped to image bounds for preview cropping."""
    x = min(max(region.x, 0), max(image_width - 1, 0))
    y = min(max(region.y, 0), max(image_height - 1, 0))
    right = min(region.x + region.width, image_width)
    bottom = min(region.y + region.height, image_height)
    width = max(1, right - x)
    height = max(1, bottom - y)
    return region.model_copy(update={"x": x, "y": y, "width": width, "height": height})


def roi_region_to_dict(region: ROIRegion) -> dict[str, Any]:
    """Convert an ROI region to JSON-compatible data."""
    data = region.model_dump()
    for key in ("created_at", "updated_at"):
        data[key] = data[key].isoformat()
    return data


def roi_region_from_dict(payload: dict[str, Any]) -> ROIRegion:
    """Create an ROI region from exported JSON-compatible data."""
    return ROIRegion(**payload)


def safe_roi_key(value: str) -> str:
    """Normalize a user-facing ROI key for filenames and future CV/OCR code."""
    key = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-").lower()
    return key or "roi"
