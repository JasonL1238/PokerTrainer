"""Train the card rank/suit CLASSIFIER (Model 2) on cropped single-card images.

This is the classification counterpart to train_yolov12_cards.py. Where the card
DETECTOR localizes+names cards on full frames, this model takes a tight crop of a
single card (handed to it by Model 1's face_card box at runtime) and outputs one
of the 52 rank+suit classes.

Ultralytics classification expects a folder-per-class dataset ROOT (not a
data.yaml):  <root>/{train,val}/<label>/*.png  -- exactly what
build_card_cls_dataset.py produces.

  python cv_lab/scripts/train_card_classifier.py \
      --data cv_lab/datasets/cards_cls_v5_20260717 \
      --epochs 60 --imgsz 128 --name cards_cls_v5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="cv_lab/datasets/cards_cls_v5_20260717",
                        help="classification dataset ROOT (contains train/ and val/)")
    parser.add_argument("--base", default="yolo11s-cls.pt",
                        help="base classification checkpoint (downloads on first run)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=128,
                        help="crops are small; 128 is plenty and keeps training fast")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="", help="YOLO device string, e.g. cpu, 0, mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="cv_lab/runs/card_cls")
    parser.add_argument("--name", default="cards_cls_v5")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="mixed precision; default off (MPS classify val has hit corrupt pred indices)")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False,
                        help="write confusion-matrix plots; default off for the same MPS val bug")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR))
    args = parser.parse_args()

    args.yolov12_vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(args.yolov12_vendor)
    model = YOLO(args.base)

    train_args = {
        "data": str(Path(args.data).resolve()),
        "task": "classify",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
        "amp": args.amp,
        "plots": args.plots,
    }
    if args.device:
        train_args["device"] = args.device

    print(f"base={args.base}")
    print(f"data={train_args['data']}")
    print(f"yolov12_vendor={args.yolov12_vendor or '<installed ultralytics>'}")
    result = model.train(**train_args)
    print(result)


if __name__ == "__main__":
    main()
