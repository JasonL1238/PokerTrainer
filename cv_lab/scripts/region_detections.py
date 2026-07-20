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

# Coarse per-seat avatar centroids for the current 8-max ClubWPT view, in
# normalized (cx, cy) coords. Seat 0 is the hero (bottom-center). Used as the
# fallback for classes without a learned anchor table below.
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

# Learned per-CLASS seat anchors (k-means over the human-labeled boxes in
# labels.sqlite3, v00 frames, normalized by true image dims). Each HUD element
# renders at its own per-seat position -- card backs sit above/inside of the
# avatar, bet texts toward the table center -- so nearest-avatar assignment
# flaps between adjacent seats; nearest-class-anchor does not. Regenerate by
# re-running the k-means when the table layout changes.
SEAT_ANCHORS_BY_CLASS: dict[str, dict[int, tuple[float, float]]] = {
    "card_back": {0: (0.500, 0.860), 1: (0.194, 0.623), 2: (0.117, 0.368), 3: (0.174, 0.164),
                  4: (0.480, 0.125), 5: (0.810, 0.164), 6: (0.868, 0.369), 7: (0.791, 0.618)},
    "stack_text": {0: (0.528, 0.809), 1: (0.180, 0.717), 2: (0.106, 0.458), 3: (0.162, 0.255),
                   4: (0.470, 0.195), 5: (0.833, 0.257), 6: (0.889, 0.461), 7: (0.815, 0.714)},
    "action_pill": {0: (0.521, 0.842), 1: (0.182, 0.754), 2: (0.109, 0.499), 3: (0.165, 0.291),
                    4: (0.468, 0.234), 5: (0.826, 0.296), 6: (0.881, 0.501), 7: (0.807, 0.750)},
    "dealer_button": {0: (0.424, 0.640), 1: (0.308, 0.680), 2: (0.154, 0.539), 3: (0.289, 0.245),
                      4: (0.584, 0.263), 5: (0.725, 0.227), 6: (0.787, 0.511), 7: (0.676, 0.671)},
    "active_turn_indicator": {0: (0.447, 0.788), 1: (0.259, 0.689), 2: (0.183, 0.437), 3: (0.239, 0.233),
                              4: (0.546, 0.177), 5: (0.752, 0.239), 6: (0.804, 0.445), 7: (0.733, 0.691)},
    "bet_text": {0: (0.467, 0.579), 1: (0.311, 0.591), 2: (0.256, 0.441), 3: (0.313, 0.313),
                 4: (0.491, 0.312), 5: (0.659, 0.314), 6: (0.714, 0.438), 7: (0.661, 0.581)},
}

# Seat ring in action order starting at the hero seat: on the ClubWPT layout the
# action moves hero (bottom-center) -> bottom-left -> up the left side -> across
# the top -> down the right side, i.e. ascending seat index (dealer -> SB=dealer+1
# -> BB=dealer+2 ... verified against live blind posts). Used for positions.
SEAT_RING: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]

# Detections whose zone is a table seat (as opposed to the shared board/pot).
_SEATED_CLASSES = {
    "card_back",
    "stack_text",
    "bet_text",
    "action_pill",
    "dealer_button",
    "active_turn_indicator",
}

# Classes whose attr is a numeric amount readable by the template OCR.
_AMOUNT_CLASSES = {"pot_text", "stack_text", "bet_text"}


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
    # Colour fallback emitted by the OCR reader when the pill word is unreadable:
    # gray = check/fold (resolved by whether the seat still holds cards), green =
    # call/bet (call is the safe default), orange = raise.
    if text in {"gray", "grey", "neutral"}:
        return "check" if dealt_in else "fold"
    if text == "green":
        return "call"
    if text == "orange":
        return "raise"
    return None


# --------------------------------------------------------------------------- #
# Seat / zone assignment (coarse STUB geometry).
# --------------------------------------------------------------------------- #
def _center(det: Detection, frame: Frame) -> tuple[float, float]:
    x0, y0, x1, y1 = det.xyxy
    return ((x0 + x1) / 2.0 / frame.width, (y0 + y1) / 2.0 / frame.height)


def _nearest_seat(cx: float, cy: float, cls: str = "") -> int:
    anchors = SEAT_ANCHORS_BY_CLASS.get(cls, SEAT_CENTROIDS)
    return min(
        anchors,
        key=lambda s: (cx - anchors[s][0]) ** 2 + (cy - anchors[s][1]) ** 2,
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
    # Nearest card-anchor seat for each card in the STRICT hero zone ("other"-zone
    # strays like villain showdown reveals don't vote). The spine's hero identity
    # is the convention hero zone == seat 0; these votes cross-check it.
    hero_zone_seat_votes: list[int] = []

    def seat(i: int) -> dict[str, Any]:
        return seats.setdefault(
            i, {"card_back": False, "stack": None, "bet": None, "pill_action": None,
                "dealer": False, "turn": False}
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
                if zone == "hero":
                    hero_zone_seat_votes.append(_nearest_seat(cx, cy, "card_back"))
        elif det.cls == "pot_text":
            pot_candidates.append(det)
        elif det.cls in _SEATED_CLASSES:
            i = _nearest_seat(cx, cy, det.cls)
            if det.cls == "card_back":
                seat(i)["card_back"] = True
            elif det.cls == "stack_text":
                seat(i)["stack"] = read_amount(det)
            elif det.cls == "bet_text":
                seat(i)["bet"] = read_amount(det)
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

    # Hero-seat cross-check: if most hero-zone cards sit nearer another seat's
    # card anchor than seat 0's, the layout/anchors have drifted and every
    # downstream "hero = seat 0" attribution (is_hero, hero_position, hero net)
    # is suspect. Majority vote so one flapped assignment doesn't warn.
    off_seat = sum(1 for v in hero_zone_seat_votes if v != 0)
    hero_seat_mismatch = off_seat * 2 > len(hero_zone_seat_votes)

    return {
        "hero": hero_cards,
        "board": board_cards,
        "pot": pot,
        "seats": seats,
        "dealer_seat": dealer_seat,
        "active_seat": active_seat,
        "hero_seat_mismatch": hero_seat_mismatch,
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


def _pad_crop_xyxy(image, x0: float, y0: float, x1: float, y1: float, pad: float):
    """Crop [x0,y0,x1,y1] (pixels) expanded by ``pad`` each side, clamped to image.

    Returns the sub-array, or None if degenerate/out of bounds. Uses plain array
    slicing so region_detections stays importable without cv2/numpy on the
    fixture path.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    xi0, yi0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
    xi1, yi1 = min(int(round(x1)), w), min(int(round(y1)), h)
    if xi1 <= xi0 or yi1 <= yi0:
        return None
    return image[yi0:yi1, xi0:xi1]


def frame_from_models(
    image,
    time_s: float,
    rows: Iterable[dict[str, Any]],
    *,
    classifier,
    image_name: str = "",
    pad: float = 0.12,
    video_frame: int = 0,
    ocr: bool = True,
) -> Frame:
    """Build a Frame from Model 1's region detections + Model 2's card classifier.

    This is the Design-A wiring: the region detector (Model 1) only *localizes*
    ``face_card`` boxes -- it does not name them. For each ``face_card`` box we
    crop the region out of ``image`` and hand it to ``classifier`` (Model 2,
    duck-typed: any object with ``classify(bgr_crop) -> (label, conf)``) to get
    the rank+suit, which becomes the detection's ``attr``. Amount classes
    (pot_text / stack_text / bet_text) and action_pill are read by the
    deterministic template OCR (ocr_readers); pass ``ocr=False`` to skip it
    (attrs then stay whatever the row provides, as on the fixture path).

    ``rows`` are Model 1 detection dicts with keys: class, confidence|conf,
    x1, y1, x2, y2. ``image`` is an HxWx3 BGR array (Model 1's input frame);
    width/height are taken from it.
    """
    h, w = (int(image.shape[0]), int(image.shape[1])) if image is not None else (0, 0)
    ocr_readers = None
    if ocr and image is not None:
        # Lazy import keeps the fixture path importable without cv2/numpy.
        from cv_lab.scripts import ocr_readers as _ocr

        ocr_readers = _ocr
    dets: list[Detection] = []
    for row in rows:
        cls = str(row.get("class", "")).strip()
        if cls not in CLASS_SET:
            continue
        xyxy = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        attr = row.get("attr")
        if cls == "face_card" and classifier is not None:
            crop = _pad_crop_xyxy(image, *xyxy, pad)
            if crop is not None and getattr(crop, "size", 1) > 0:
                label, _conf = classifier.classify(crop)
                attr = label
        elif ocr_readers is not None and attr is None and cls in _AMOUNT_CLASSES:
            attr = ocr_readers.read_amount_from_image(image, xyxy)
        elif ocr_readers is not None and attr is None and cls == "action_pill":
            attr = ocr_readers.read_pill_attr(image, xyxy)
        dets.append(
            Detection(
                cls=cls,
                conf=float(row.get("confidence", row.get("conf", 0.0)) or 0.0),
                xyxy=xyxy,
                attr=attr,
            )
        )
    return Frame(image=image_name, time_s=time_s, width=w, height=h,
                 detections=dets, video_frame=video_frame)
