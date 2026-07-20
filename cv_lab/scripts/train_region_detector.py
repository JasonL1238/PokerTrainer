"""Train the 8-class region detector (Model 1) for the reconstruction spine.

Model 1 boxes all reconstruction classes (face_card, card_back, dealer_button,
pot_text, stack_text, action_pill, active_turn_indicator, bet_text). For face_card
it only LOCALIZES -- Model 2 (the card classifier) names the rank/suit downstream.

This is a FRESH detector (8 classes), so unlike train_yolov12_cards.py it does NOT
use the anti-drift freeze/low-lr fine-tuning recipe. It warm-starts from a stock
yolo11s checkpoint -- NOT best (4).pt (the old yolov12 card detector): the vendored
yolov12 architecture crashes on backward under torch 2.13 (`.view` stride error),
so region detector training must stay on a yolo11 backbone. Ultralytics adapts the
detection head to the 8 region classes.

Prefer installed ultralytics (pass --yolov12-vendor "") over the yolov12 vendor:
vendor TAL hits shape-mismatch crashes on dense table frames.

  python cv_lab/scripts/train_region_detector.py \
      --data cv_lab/datasets/region_spine_v6_20260718/data.yaml \
      --epochs 100 --name region_spine_v6_20260718 --device mps \
      --yolov12-vendor "" --mosaic 0 --no-amp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = "yolo11s.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="cv_lab/datasets/region_spine_v6_20260718/data.yaml")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="warm-start checkpoint; head is re-fit to the 8 region classes. "
                             "Must be a yolo11 checkpoint -- yolov12 (e.g. best (4).pt) crashes on backward.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="table text (pot/stack/bet) is small; consider larger if it under-detects")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="cv_lab/runs/yolo_cards")
    parser.add_argument("--name", default="region_spine_v6_20260718")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--copy-paste", type=float, default=0.0,
                        help="copy-paste aug; 0 avoids TAL shape-mismatch crashes on dense table frames")
    parser.add_argument("--mosaic", type=float, default=0.0,
                        help="mosaic aug; 0 is safest on dense ClubWPT frames / MPS")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run epoch val during training; default off because MPS detect val corrupts class ids",
    )
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument(
        "--yolov12-vendor",
        default="",
        help="path to yolov12 vendor, or empty for installed ultralytics (recommended for Model 1)",
    )
    args = parser.parse_args()

    args.yolov12_vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(args.yolov12_vendor)
    model = YOLO(args.base)

    train_args = {
        "data": str(Path(args.data)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "lr0": args.lr0,
        "exist_ok": args.exist_ok,
        "copy_paste": args.copy_paste,
        "mosaic": args.mosaic,
        "mixup": 0.0,
        "cutmix": 0.0,
        "amp": args.amp,
        "plots": args.plots,
        "val": args.val,
        "save_period": args.save_period,
    }
    if args.device:
        train_args["device"] = args.device

    print(f"base={args.base}")
    print(f"data={args.data}")
    print(f"yolov12_vendor={args.yolov12_vendor or '<installed ultralytics>'}")
    print(f"val={args.val} (MPS detect val is flaky; default off, promote last.pt)")
    result = model.train(**train_args)
    print(result)

    # Optional post-train CPU val for a real mAP read without crashing the run.
    weights = Path(args.project) / args.name / "weights" / "last.pt"
    if weights.exists() and not args.val:
        print(f"post-train CPU val on {weights}")
        try:
            YOLO(str(weights)).val(
                data=str(Path(args.data)),
                imgsz=args.imgsz,
                batch=max(1, args.batch // 2),
                device="cpu",
                plots=False,
                workers=0,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort metrics only
            print(f"post-train CPU val failed (weights still saved): {exc}")


if __name__ == "__main__":
    main()
