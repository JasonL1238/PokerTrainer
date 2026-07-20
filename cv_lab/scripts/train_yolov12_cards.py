"""Continue training the YOLOv12 card detector on a corrected YOLO dataset.

This wrapper intentionally loads the repo/vendor YOLOv12 fork instead of
assuming the globally installed Ultralytics package is compatible.
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
    parser.add_argument("--data", default="cv_lab/datasets/yolo_cards_autolabel_v3/data.yaml")
    parser.add_argument("--weights", default="cv_lab/models/best (4).pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="optional YOLO device string, e.g. cpu, 0, mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="cv_lab/runs/yolo_cards")
    parser.add_argument("--name", default="continue_v1")
    parser.add_argument("--patience", type=int, default=12)
    # --- anti-drift (catastrophic-forgetting) knobs, defaulted to the fine-tune recipe ---
    parser.add_argument("--freeze", type=int, default=10,
                        help="freeze the first N layers (backbone) so generic card features can't drift; 0 disables")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="initial LR; keep ~10x below from-scratch so the checkpoint is nudged, not overhauled")
    parser.add_argument("--lrf", type=float, default=0.01, help="final LR fraction (lr0*lrf)")
    parser.add_argument("--cos-lr", action="store_true", default=True,
                        help="cosine LR schedule for a gentle decay tail")
    parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR),
                        help="path to the sunsmarterjie/yolov12 checkout; use empty string for installed ultralytics")
    args = parser.parse_args()

    args.yolov12_vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(args.yolov12_vendor)
    model = YOLO(args.weights)

    train_args = {
        "data": str(Path(args.data)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": args.cos_lr,
    }
    if args.freeze:
        train_args["freeze"] = args.freeze
    if args.device:
        train_args["device"] = args.device

    print(f"weights={args.weights}")
    print(f"data={args.data}")
    print(f"yolov12_vendor={args.yolov12_vendor or '<installed ultralytics>'}")
    result = model.train(**train_args)
    print(result)


if __name__ == "__main__":
    main()
