"""Detector-output contract + region assignment for the 7-class YOLO pipeline.

This is the input layer of the reconstruction spine (see build_yolo_hand_timeline).
It is deliberately decoupled from *how* detections were produced so the spine can be
exercised today with synthetic fixtures and labeled ground-truth boxes, and switched
to the trained 7-class detector unchanged later. It reads saved / completed-session
data only; it never captures live tables.

Pipeline position:
    raw detections (per frame) -> assign_regions (seats/zones) + attribute readers
    -> per-frame table state -> [build_yolo_hand_timeline]

The 7 classes and their reconstruction jobs (labeling_poker/config.py, README):
    face_card, card_back, dealer_button, pot_text, stack_text, action_pill,
    active_turn_indicator.

Attribute reads (rank/suit, OCR amounts, pill colour) and the real anchored seat
model are SEPARATE sub-parts; here they are pluggable interfaces with stub
implementations that read straight from each detection's ``attr`` field, so
ground-truth boxes and synthetic fixtures work with no OCR/model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Canonical rank+suit normaliser shared with the labeler ("KD"/"10C" -> "Kd"/"Tc").
from labeling_poker.config import CLASSES, normalize_card_label

# Reuse the proven card zone split (hero vs board vs other) from the card-only builder.
from cv_lab.scripts.build_yolo_card_timeline import _zone_for_box

CLASS_SET = set(CLASSES)

# Coarse per-seat centroids for the current 8-max ClubWPT view, in normalized
# (cx, cy) coords. Seat 0 is the hero (bottom-center). This is the STUB seat model;
# the real anchored per-seat model (green-coin landmark_anchor) plugs into
# assign_regions() later without changing the spine.
SEAT_CENTROIDS: dict[int, tuple[float, float]] = {
    0: (0.50, 0.86),
    1: (0.16, 0.80),
    2: (0.05, 0.50),
    3: (0.18, 0.20),
    4: (0.50, 0.14),
    5: (0.82, 0.20),
    6: (0.95, 0.50),
    7: (0.84, 0.80),
}

# Clockwise seat ring starting at the hero seat. Action moves along this ring
# (dealer -> SB -> BB -> ...), used to assign positions in the spine.
SEAT_RING: list[int] = [0, 7, 6, 5, 4, 3, 2, 1]

# Detections whose zone is a table seat (as opposed to the shared board/pot).
_SEATED_CLASSES = {
    "card_back",
    "stack_text",
    "action_pill",
    "dealer_button",
    "active_turn_indicator",
}


@dataclass
class Detection:
    """One YOLO region box. ``attr`` is the attribute-read payload:

    - face_card: rank+suit label (e.g. "As"), or the detector's own "AS"/"10C"
    - pot_text / stack_text: the numeric value (float or numeric string)
    - action_pill: the action or colour ("raise"/"call"/"bet"/"check"/"gray"/None)
    - card_back / dealer_button / active_turn_indicator: unused (None)
    """

    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    attr: Any = None


@dataclass
class Frame:
    """One sampled frame's detections plus the geometry needed to normalize them."""

    image: str
    time_s: float
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    video_frame: int = 0


# --------------------------------------------------------------------------- #
# Attribute readers (STUBS). Real OCR / rank-suit / pill-colour classifiers
# replace these behind the same signatures.
# --------------------------------------------------------------------------- #
def read_card_label(det: Detection) -> str | None:
    """face_card attr -> canonical rank+suit ("As"), or None if unreadable."""
    try:
        return normalize_card_label("face_card", det.attr)
    except ValueError:
        return None


def read_amount(det: Detection) -> float | None:
    """pot_text / stack_text attr -> float value, or None."""
    if det.attr is None:
        return None
    try:
        return float(str(det.attr).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


_PILL_ALIASES = {
    "raise": "raise",
    "bet": "bet",
    "call": "call",
    "check": "check",
    "fold": "fold",
    "all-in": "all-in",
    "all_in": "all-in",
    "allin": "all-in",
    "post_blind": "post_blind",
    "bb": "post_blind",
    "sb": "post_blind",
}


def read_pill_action(det: Detection, *, dealt_in: bool) -> str | None:
    """action_pill attr -> action type. A gray pill is check when the seat still
    has cards, fold otherwise (README: "gray check-vs-fold is resolved by card
    backs disappearing"). Returns None when the pill carries no readable action."""
    if det.attr is None:
        return None
    text = str(det.attr).strip().lower()
    if text in _PILL_ALIASES:
        return _PILL_ALIASES[text]
    if text in {"gray", "grey", "neutral"}:
        return "check" if dealt_in else "fold"
    return None


# --------------------------------------------------------------------------- #
# Seat / zone assignment (coarse STUB geometry).
# --------------------------------------------------------------------------- #
def _center(det: Detection, frame: Frame) -> tuple[float, float]:
    x0, y0, x1, y1 = det.xyxy
    return ((x0 + x1) / 2.0 / frame.width, (y0 + y1) / 2.0 / frame.height)


def _nearest_seat(cx: float, cy: float) -> int:
    return min(
        SEAT_CENTROIDS,
        key=lambda s: (cx - SEAT_CENTROIDS[s][0]) ** 2 + (cy - SEAT_CENTROIDS[s][1]) ** 2,
    )


def assign_regions(frame: Frame) -> dict[str, Any]:
    """Assign a frame's detections to hero/board zones and table seats.

    Returns a normalized table view:
        {
          "hero": [card,...], "board": [card,...],
          "pot": float | None,
          "seats": { seat_index: {"card_back","stack","pill_action","dealer","turn"} },
          "dealer_seat": int | None, "active_seat": int | None,
        }
    """
    hero_dets: list[tuple[float, Detection]] = []
    board_dets: list[tuple[float, Detection]] = []
    pot_candidates: list[Detection] = []
    seats: dict[int, dict[str, Any]] = {}

    def seat(i: int) -> dict[str, Any]:
        return seats.setdefault(
            i, {"card_back": False, "stack": None, "pill_action": None, "dealer": False, "turn": False}
        )

    for det in frame.detections:
        if det.cls not in CLASS_SET:
            continue
        cx, cy = _center(det, frame)

        if det.cls == "face_card":
            zone = _zone_for_box(cx, cy)
            if zone == "board":
                board_dets.append((cx, det))
            else:  # hero (or a stray "other" -> treat as hero-side card)
                hero_dets.append((cx, det))
        elif det.cls == "pot_text":
            pot_candidates.append(det)
        elif det.cls in _SEATED_CLASSES:
            i = _nearest_seat(cx, cy)
            if det.cls == "card_back":
                seat(i)["card_back"] = True
            elif det.cls == "stack_text":
                seat(i)["stack"] = read_amount(det)
            elif det.cls == "action_pill":
                seat(i)["_pill_det"] = det  # resolved after dealt-in is known
            elif det.cls == "dealer_button":
                seat(i)["dealer"] = True
            elif det.cls == "active_turn_indicator":
                seat(i)["turn"] = True

    hero_cards = [c for c in (read_card_label(d) for _, d in sorted(hero_dets)) if c][:2]
    board_cards = [c for c in (read_card_label(d) for _, d in sorted(board_dets)) if c][:5]

    # Hero seat is dealt in when hero hole cards are visible.
    if hero_cards:
        seat(0)["card_back"] = True

    # Resolve pills now that dealt-in status per seat is known.
    for i, info in seats.items():
        pill_det = info.pop("_pill_det", None)
        if pill_det is not None:
            info["pill_action"] = read_pill_action(pill_det, dealt_in=info["card_back"])

    pot = None
    if pot_candidates:
        best = max(pot_candidates, key=lambda d: d.conf)
        pot = read_amount(best)

    dealer_seat = next((i for i, info in seats.items() if info["dealer"]), None)
    active_seat = next((i for i, info in seats.items() if info["turn"]), None)

    return {
        "hero": hero_cards,
        "board": board_cards,
        "pot": pot,
        "seats": seats,
        "dealer_seat": dealer_seat,
        "active_seat": active_seat,
    }


# --------------------------------------------------------------------------- #
# Adapters: produce Frame objects from each source.
# --------------------------------------------------------------------------- #
def frames_from_fixture(data: Iterable[dict[str, Any]]) -> list[Frame]:
    """Parse a list of plain dicts (JSON fixtures / tests) into Frames."""
    frames: list[Frame] = []
    for row in data:
        dets = [
            Detection(
                cls=str(d["cls"]),
                conf=float(d.get("conf", 1.0)),
                xyxy=tuple(float(v) for v in d["xyxy"]),  # type: ignore[arg-type]
                attr=d.get("attr"),
            )
            for d in row.get("detections", [])
        ]
        frames.append(
            Frame(
                image=str(row["image"]),
                time_s=float(row["time_s"]),
                width=int(row["width"]),
                height=int(row["height"]),
                detections=dets,
                video_frame=int(row.get("video_frame", 0)),
            )
        )
    return frames


def load_frames(path: str | Path) -> list[Frame]:
    """Load a frames fixture JSON file (a list of frame dicts)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("frames", [])
    return frames_from_fixture(raw)


def frame_from_yolo_rows(
    image: str,
    time_s: float,
    width: int,
    height: int,
    rows: Iterable[dict[str, Any]],
    *,
    video_frame: int = 0,
) -> Frame:
    """Adapt one image's YOLO detection rows (labeling_poker.inference.predict_cards
    shape: class/label/confidence/x1..y2) into a Frame. Downstream wiring for the
    trained 7-class detector; unlike the old card model, ``class`` is trusted here."""
    dets: list[Detection] = []
    for row in rows:
        cls = str(row.get("class", "")).strip()
        if cls not in CLASS_SET:
            continue
        attr = row.get("label")
        if cls in {"pot_text", "stack_text", "action_pill"}:
            attr = row.get("attr", row.get("value"))
        dets.append(
            Detection(
                cls=cls,
                conf=float(row.get("confidence", row.get("conf", 0.0)) or 0.0),
                xyxy=(
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ),
                attr=attr,
            )
        )
    return Frame(image=image, time_s=time_s, width=width, height=height, detections=dets, video_frame=video_frame)
