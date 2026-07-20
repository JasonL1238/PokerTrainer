"""Two-model runtime pipeline: region detector (Model 1) + card classifier (Model 2).

This is the live glue the reconstruction spine was designed for. It keeps
region_detections.py detector-agnostic: this module runs the models and hands the
spine a ready ``Frame`` via ``region_detections.frame_from_yolo_rows``.

Flow per frame:
    1. Model 1 (8-class region detector) boxes all reconstruction classes,
       including ``face_card`` -- for face_card it only LOCALIZES.
    2. Each ``face_card`` box is cropped (with the same padding used to build the
       classifier's training crops) and passed to Model 2, which names the card's
       rank+suit. That label rides along as the detection's ``attr``.
    3. Non-card classes pass straight through with their box; OCR/pill attribute
       reads remain the spine's separate pluggable layer (unchanged here).

    python cv_lab/scripts/two_model_infer.py --image path/to/frame.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from threading import Lock

import cv2

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402
from card_classifier import CardClassifier  # noqa: E402
from cv_lab.scripts import region_detections as rd  # noqa: E402
from labeling_poker.config import CLASSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGION_WEIGHTS = REPO_ROOT / "cv_lab" / "models" / "region_spine_v1.pt"
if not DEFAULT_REGION_WEIGHTS.exists():
    DEFAULT_REGION_WEIGHTS = REPO_ROOT / "cv_lab" / "runs" / "yolo_cards" / "region_spine_v6_20260718" / "weights" / "best.pt"
# Match the crop padding build_card_cls_dataset.py used, so runtime crops match training.
CARD_CROP_PAD = 0.12


def _pad_box(x1, y1, x2, y2, w, h, pad):
    bw, bh = x2 - x1, y2 - y1
    xi0 = max(int(round(x1 - bw * pad)), 0)
    yi0 = max(int(round(y1 - bh * pad)), 0)
    xi1 = min(int(round(x2 + bw * pad)), w)
    yi1 = min(int(round(y2 + bh * pad)), h)
    return xi0, yi0, xi1, yi1


class RegionDetector:
    """Lazy-loaded 8-class region detector (Model 1)."""

    def __init__(self, weights: str | Path | None = None,
                 vendor: str | Path | None = None, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.5, device: str = "") -> None:
        self.weights = Path(weights) if weights else DEFAULT_REGION_WEIGHTS
        self.vendor = str(vendor) if vendor else str(DEFAULT_YOLOV12_VENDOR)
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self._model = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.weights.exists():
                        raise FileNotFoundError(
                            f"region detector weights not found: {self.weights}\n"
                            "train Model 1 first (train_yolov12_cards.py on yolo_regions_v1)."
                        )
                    vendor = _resolve_vendor_path(self.vendor)
                    YOLO = _load_yolo_class(vendor)
                    self._model = YOLO(str(self.weights))
        return self._model

    def predict(self, image_path: str | Path) -> tuple[list[dict], int, int]:
        """Return (rows, width, height). rows have class/confidence/x1..y2 (pixels)."""
        model = self._load()
        kwargs = {"conf": self.conf, "iou": self.iou, "imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = model.predict(str(image_path), **kwargs)[0]
        h, w = result.orig_shape[0], result.orig_shape[1]
        rows: list[dict] = []
        for box in result.boxes:
            cid = int(box.cls[0])
            name = str(model.names[cid])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append({"class": name, "confidence": round(float(box.conf[0]), 4),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return rows, w, h


class TwoModelPipeline:
    """Model 1 (regions) + Model 2 (card rank/suit) -> a spine-ready Frame."""

    def __init__(self, region_weights: str | Path | None = None,
                 card_weights: str | Path | None = None,
                 device: str = "", card_pad: float = CARD_CROP_PAD) -> None:
        self.detector = RegionDetector(region_weights, device=device)
        self.classifier = CardClassifier(card_weights, device=device)
        self.card_pad = card_pad

    def frame(self, image_path: str | Path, *, time_s: float = 0.0,
              video_frame: int = 0) -> rd.Frame:
        rows, w, h = self.detector.predict(image_path)
        img = cv2.imread(str(image_path))
        # frame_from_models classifies each face_card crop via Model 2 and runs the
        # template OCR over pot/stack/bet/pill boxes, same as the video path.
        return rd.frame_from_models(img, time_s, rows, classifier=self.classifier,
                                    image_name=str(image_path), pad=self.card_pad,
                                    video_frame=video_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--region-weights", default=str(DEFAULT_REGION_WEIGHTS))
    parser.add_argument("--card-weights", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    pipe = TwoModelPipeline(args.region_weights, args.card_weights or None, device=args.device)
    frame = pipe.frame(args.image)
    print(f"image={frame.image}  {frame.width}x{frame.height}  detections={len(frame.detections)}")
    by_cls: dict[str, int] = {}
    for d in frame.detections:
        by_cls[d.cls] = by_cls.get(d.cls, 0) + 1
    print("by class:", {c: by_cls.get(c, 0) for c in CLASSES})
    print("\nface cards read:")
    for d in frame.detections:
        if d.cls == "face_card":
            print(f"  {rd.read_card_label(d)!s:>4}  conf_box={d.conf:.2f}  xyxy={tuple(round(v) for v in d.xyxy)}")


if __name__ == "__main__":
    main()
