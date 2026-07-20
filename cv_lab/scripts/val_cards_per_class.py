"""Per-class validation for the card detector -- the drift meter.

Runs YOLO val on a dataset and prints mAP50-95 for every one of the 53 classes,
sorted worst-first. Optionally diffs against a baseline JSON (the old checkpoint's
per-class scores) so you can see exactly which cards a fine-tune helped or hurt.

Usage:
  # baseline the OLD model before fine-tuning
  python cv_lab/scripts/val_cards_per_class.py \
      --weights "cv_lab/models/best (4).pt" \
      --data cv_lab/datasets/yolo_cards_clubwpt_mixed/data.yaml \
      --save-json cv_lab/results/card_baseline.json

  # after fine-tuning, compare
  python cv_lab/scripts/val_cards_per_class.py \
      --weights cv_lab/runs/yolo_cards/clubwpt_mixed/weights/best.pt \
      --data cv_lab/datasets/yolo_cards_clubwpt_mixed/data.yaml \
      --baseline cv_lab/results/card_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402


def _per_class_map(model, data: str, imgsz: int, device: str) -> dict[str, float]:
    kwargs = {"data": data, "imgsz": imgsz, "split": "val", "verbose": False}
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    # metrics.maps is a per-class array aligned to model.names indices.
    maps = list(getattr(metrics, "maps", []))
    return {model.names[i]: float(maps[i]) for i in range(len(maps)) if i in model.names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--save-json", default="", help="write per-class scores here")
    parser.add_argument("--baseline", default="", help="compare against a previously saved --save-json")
    parser.add_argument("--regress-threshold", type=float, default=0.02,
                        help="flag classes that dropped more than this vs baseline")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR))
    args = parser.parse_args()

    args.yolov12_vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(args.yolov12_vendor)
    model = YOLO(args.weights)

    scores = _per_class_map(model, args.data, args.imgsz, args.device)
    print(f"weights={args.weights}")
    print(f"data={args.data}")
    print(f"mean_mAP50-95={sum(scores.values())/max(len(scores),1):.4f}")

    baseline = {}
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    print("\nper-class mAP50-95 (worst first):")
    for name, val in sorted(scores.items(), key=lambda kv: kv[1]):
        if baseline:
            delta = val - baseline.get(name, val)
            flag = "  <-- REGRESSED" if delta < -args.regress_threshold else ""
            print(f"  {name:>4}: {val:.3f}  (Δ {delta:+.3f}){flag}")
        else:
            print(f"  {name:>4}: {val:.3f}")

    if baseline:
        regressed = [n for n, v in scores.items()
                     if v - baseline.get(n, v) < -args.regress_threshold]
        if regressed:
            print(f"\nDRIFT: {len(regressed)} class(es) regressed > {args.regress_threshold}: "
                  + ", ".join(sorted(regressed)))
        else:
            print(f"\nno class regressed more than {args.regress_threshold} -- drift contained")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"\nsaved={out}")


if __name__ == "__main__":
    main()
