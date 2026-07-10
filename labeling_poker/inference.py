from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock

from .config import DEFAULT_MODEL_PATH, YOLOV12_VENDOR_CANDIDATES


_MODEL = None
_MODEL_LOCK = Lock()


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
