"""Runtime card rank/suit CLASSIFIER (Model 2) helper.

Loads the trained YOLO ``-cls`` checkpoint once and classifies a single-card crop
into one of the 52 canonical rank+suit labels ("As", "Kd", "Tc", ...). This is the
attribute reader the two-model pipeline calls after Model 1 (the region detector)
localizes a ``face_card`` box.

The model is loaded lazily and cached (mirrors labeling_poker.inference), so import
is cheap and the weights load on first use.

Usage:
    from card_classifier import CardClassifier
    clf = CardClassifier()                    # default weights under cv_lab/runs
    label, conf = clf.classify(bgr_crop)      # bgr_crop is an HxWx3 numpy array
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# Promoted stable weights; falls back to the raw run dir if not yet promoted.
DEFAULT_CLS_WEIGHTS = REPO_ROOT / "cv_lab" / "models" / "card_cls_v1.pt"
if not DEFAULT_CLS_WEIGHTS.exists():
    DEFAULT_CLS_WEIGHTS = REPO_ROOT / "cv_lab" / "runs" / "card_cls" / "cards_cls_v1" / "weights" / "best.pt"


class CardClassifier:
    """Lazy-loaded wrapper around the YOLO card rank/suit classifier."""

    def __init__(self, weights: str | Path | None = None,
                 vendor: str | Path | None = None, imgsz: int = 128,
                 device: str = "") -> None:
        self.weights = Path(weights) if weights else DEFAULT_CLS_WEIGHTS
        self.vendor = str(vendor) if vendor else str(DEFAULT_YOLOV12_VENDOR)
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.weights.exists():
                        raise FileNotFoundError(
                            f"card classifier weights not found: {self.weights}\n"
                            "train Model 2 first: cv_lab/scripts/train_card_classifier.py"
                        )
                    vendor = _resolve_vendor_path(self.vendor)
                    YOLO = _load_yolo_class(vendor)
                    self._model = YOLO(str(self.weights))
        return self._model

    def classify(self, bgr_crop: np.ndarray) -> tuple[str | None, float]:
        """Classify one BGR crop -> (canonical label, confidence). (None, 0.0) if empty."""
        if bgr_crop is None or bgr_crop.size == 0:
            return None, 0.0
        model = self._load()
        kwargs = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = model.predict(bgr_crop, **kwargs)[0]
        probs = result.probs
        if probs is None:
            return None, 0.0
        idx = int(probs.top1)
        return str(model.names[idx]), float(probs.top1conf)

    def classify_batch(self, crops: list[np.ndarray]) -> list[tuple[str | None, float]]:
        """Classify many crops in one predict call (faster than looping classify)."""
        valid = [(i, c) for i, c in enumerate(crops) if c is not None and c.size > 0]
        out: list[tuple[str | None, float]] = [(None, 0.0)] * len(crops)
        if not valid:
            return out
        model = self._load()
        kwargs = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = model.predict([c for _, c in valid], **kwargs)
        for (i, _), res in zip(valid, results):
            probs = res.probs
            if probs is not None:
                idx = int(probs.top1)
                out[i] = (str(model.names[idx]), float(probs.top1conf))
        return out
