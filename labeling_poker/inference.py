from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock

from .config import DEFAULT_MODEL_PATH, DEFAULT_REGION_MODEL_PATH, YOLOV12_VENDOR_CANDIDATES


_MODEL = None
_MODEL_LOCK = Lock()
_REGION_MODEL = None
_REGION_MODEL_LOCK = Lock()
_CARD_CLASSIFIER = None
_CARD_CLASSIFIER_LOCK = Lock()


def _vendor_path() -> Path:
    configured = os.environ.get("POKER_LABELER_YOLOV12_VENDOR")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"YOLOv12 vendor directory does not exist: {path}")
        return path
    for candidate in YOLOV12_VENDOR_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not find the local YOLOv12 vendor checkout")


def _model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                vendor = _vendor_path()
                if str(vendor) not in sys.path:
                    sys.path.insert(0, str(vendor))
                os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_MODEL_PATH.parent.parent / ".yolo_config"))
                os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MODEL_PATH.parent.parent / ".mpl_config"))
                from ultralytics import YOLO

                _MODEL = YOLO(str(Path(os.environ.get("POKER_LABELER_MODEL", DEFAULT_MODEL_PATH))))
    return _MODEL


def _region_model():
    global _REGION_MODEL
    if _REGION_MODEL is None:
        with _REGION_MODEL_LOCK:
            if _REGION_MODEL is None:
                vendor = _vendor_path()
                if str(vendor) not in sys.path:
                    sys.path.insert(0, str(vendor))
                os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_MODEL_PATH.parent.parent / ".yolo_config"))
                os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MODEL_PATH.parent.parent / ".mpl_config"))
                from ultralytics import YOLO

                _REGION_MODEL = YOLO(str(Path(os.environ.get("POKER_LABELER_REGION_MODEL", DEFAULT_REGION_MODEL_PATH))))
    return _REGION_MODEL


def predict_cards(image_path: Path | str, *, conf: float = 0.25, imgsz: int = 640) -> list[dict]:
    model = _model()
    result = model.predict(str(image_path), conf=conf, iou=0.5, imgsz=imgsz, verbose=False)[0]
    width, height = result.orig_shape[1], result.orig_shape[0]
    rows = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        model_names = model.names
        model_label = str(model_names[class_id])
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append({
            "class": "face_card",
            "label": model_label,
            "confidence": round(float(box.conf[0]), 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def predict_regions(image_path: Path | str, *, conf: float = 0.35, imgsz: int = 640) -> list[dict]:
    """Model 1 (region detector) predictions for all 8 spine classes.

    Unlike predict_cards, this does not name face_card boxes -- Model 1 only
    localizes them. label is always None; a human approves/edits the box+class
    only, which is all Model 1 training needs (export.py ignores label).
    """
    model = _region_model()
    result = model.predict(str(image_path), conf=conf, iou=0.3, imgsz=imgsz, verbose=False)[0]
    width, height = result.orig_shape[1], result.orig_shape[0]
    rows = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = str(model.names[class_id])
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append({
            "class": class_name,
            "label": None,
            "confidence": round(float(box.conf[0]), 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def _card_classifier():
    """Lazily load Model 2 without loading it for ordinary region review."""
    global _CARD_CLASSIFIER
    if _CARD_CLASSIFIER is None:
        with _CARD_CLASSIFIER_LOCK:
            if _CARD_CLASSIFIER is None:
                scripts_dir = Path(__file__).resolve().parent.parent / "cv_lab" / "scripts"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from card_classifier import CardClassifier

                weights = os.environ.get("POKER_LABELER_CARD_MODEL")
                _CARD_CLASSIFIER = CardClassifier(weights=weights or None)
    return _CARD_CLASSIFIER


def _padded_crop(image, row: dict, pad: float):
    height, width = image.shape[:2]
    box_width, box_height = row["x2"] - row["x1"], row["y2"] - row["y1"]
    x1 = max(0, int(round(row["x1"] - box_width * pad)))
    y1 = max(0, int(round(row["y1"] - box_height * pad)))
    x2 = min(width, int(round(row["x2"] + box_width * pad)))
    y2 = min(height, int(round(row["y2"] + box_height * pad)))
    return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def predict_two_model(image_path: Path | str, *, conf: float = 0.35, imgsz: int = 640, pad: float = 0.12) -> list[dict]:
    """Run Model 1 regions, then use Model 2 to name each face-card crop.

    This is used only for human-review bootstrap. It does not write a prediction
    to SQLite; a label enters either training set only after the reviewer saves it.
    """
    import cv2

    rows = predict_regions(image_path, conf=conf, imgsz=imgsz)
    card_rows = [row for row in rows if row["class"] == "face_card"]
    if not card_rows:
        return rows
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image {image_path}")
    predictions = _card_classifier().classify_batch([_padded_crop(image, row, pad) for row in card_rows])
    for row, (label, card_confidence) in zip(card_rows, predictions):
        row["label"] = label
        row["card_confidence"] = round(card_confidence, 4)
    return rows
