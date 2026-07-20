"""Bridge: labeling_poker SQLite boxes -> region_detections Frame fixture JSON.

This lets the reconstruction spine (build_yolo_hand_timeline.py) run on your
hand-labeled ground-truth boxes with NO trained detector and NO OCR:

    python cv_lab/scripts/export_labeler_fixture.py --video-prefix v00 \
        --out cv_lab/results/labeled_frames_v00.json
    python cv_lab/scripts/build_yolo_hand_timeline.py \
        --frames cv_lab/results/labeled_frames_v00.json \
        --out cv_lab/results/hand_timeline_v00.json

Each face_card box's rank/suit label is passed through as the detection ``attr``
(the spine's read_card_label normalizes it). pot/stack/bet amounts and pill
actions are NOT labeled in the box tool, so their ``attr`` is left None -- the
spine reconstructs structure (positions, streets, dealt-in/folds, card
identities) but not amounts/actions, which need OCR/colour readers.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from labeling_poker.config import (  # noqa: E402
    CARD_LABEL_CLASS,
    DEFAULT_DB_PATH,
    EXISTING_DATASET_IMAGES_DIR,
)

STEM_RE = re.compile(r"v(\d+)_f(\d+)_t([\d.]+)")
_AMOUNT_CLASSES = {"pot_text", "stack_text", "bet_text"}


def _read_attr(cls: str, label, crop, bank) -> object:
    """Attribute payload for a box: card label for face_card, else deterministic OCR
    of the crop (numeric amount, or pill action word / background colour)."""
    if cls == CARD_LABEL_CLASS:
        return label
    if bank is None or crop is None or crop.size == 0:
        return None
    from cv_lab.scripts.ocr_readers import pill_color

    if cls in _AMOUNT_CLASSES:
        val, _ = bank.read_number(crop)
        return val
    if cls == "action_pill":
        word, _ = bank.read_word(crop)
        return word if word else pill_color(crop)
    return None


def build_fixture(db_path: Path, images_dir: Path, video_prefix: str, ocr: bool = True) -> list[dict]:
    import cv2

    from cv_lab.scripts.ocr_readers import TemplateOCR

    bank = TemplateOCR.load() if ocr else None
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT f.id AS id, f.path AS path
        FROM files f JOIN status s ON s.file_id = f.id
        WHERE s.status = 'labeled' AND f.path LIKE ?
        """,
        (f"%{video_prefix}_%",),
    ).fetchall()

    frames: list[dict] = []
    for r in rows:
        stem = Path(r["path"]).stem
        m = STEM_RE.search(stem)
        if not m:
            continue
        video_frame, time_s = int(m.group(2)), float(m.group(3))
        image_path = (images_dir / r["path"]).resolve()
        if not image_path.is_file():
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]

        dets = []
        for a in con.execute(
            "SELECT class, label, x1, y1, x2, y2 FROM annotations WHERE file_id = ?",
            (r["id"],),
        ):
            x1, y1, x2, y2 = (int(round(a[k])) for k in ("x1", "y1", "x2", "y2"))
            crop = img[max(0, y1):y2, max(0, x1):x2] if ocr else None
            attr = _read_attr(a["class"], a["label"], crop, bank)
            dets.append(
                {
                    "cls": a["class"],
                    "conf": 1.0,
                    "xyxy": [a["x1"], a["y1"], a["x2"], a["y2"]],
                    "attr": attr,
                }
            )
        frames.append(
            {
                "image": r["path"],
                "time_s": time_s,
                "width": width,
                "height": height,
                "video_frame": video_frame,
                "detections": dets,
            }
        )
    con.close()
    frames.sort(key=lambda fr: fr["video_frame"])
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--images", default=str(EXISTING_DATASET_IMAGES_DIR))
    ap.add_argument("--video-prefix", default="v00", help="only frames whose filename starts with this (e.g. v00)")
    ap.add_argument("--out", default="cv_lab/results/labeled_frames.json")
    ap.add_argument("--no-ocr", action="store_true", help="leave amount/pill attr=None (structure only)")
    args = ap.parse_args()

    frames = build_fixture(Path(args.db), Path(args.images), args.video_prefix, ocr=not args.no_ocr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frames, indent=2), encoding="utf-8")

    n_dets = sum(len(fr["detections"]) for fr in frames)
    n_cards = sum(1 for fr in frames for d in fr["detections"] if d["cls"] == CARD_LABEL_CLASS)
    n_amt = sum(1 for fr in frames for d in fr["detections"]
                if d["cls"] in _AMOUNT_CLASSES and isinstance(d["attr"], (int, float)))
    n_pill = sum(1 for fr in frames for d in fr["detections"]
                 if d["cls"] == "action_pill" and d["attr"])
    print(f"video_prefix={args.video_prefix}  ocr={not args.no_ocr}")
    print(f"frames={len(frames)}  detections={n_dets}  face_card_boxes={n_cards}")
    print(f"amounts_read={n_amt}  pills_read={n_pill}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
