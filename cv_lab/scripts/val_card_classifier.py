"""Per-class validation for the card rank/suit classifier (Model 2).

Overall top1/top5 hides the thin classes (some ranks have very few crops). This
runs YOLO classification val and derives per-class recall from the confusion
matrix, printed worst-first, so you can see exactly which cards the model is weak
on and target more data there.

Usage:
  python cv_lab/scripts/val_card_classifier.py \
      --weights cv_lab/models/card_cls_v1.pt \
      --data cv_lab/datasets/cards_cls_v5_20260717
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
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="classification dataset ROOT (train/ val/)")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR))
    args = parser.parse_args()

    vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(vendor)
    model = YOLO(args.weights)

    kwargs = {"data": str(Path(args.data).resolve()), "imgsz": args.imgsz,
              "split": args.split, "verbose": False}
    if args.device:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)

    print(f"weights={args.weights}")
    print(f"data={args.data}  split={args.split}")
    print(f"top1={metrics.top1:.4f}  top5={metrics.top5:.4f}")

    # Per-class recall from the confusion matrix. Ultralytics stores it as
    # matrix[pred, true], so recall for class i = matrix[i, i] / sum(matrix[:, i]).
    cm = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None) if cm is not None else None
    names = model.names
    if matrix is None:
        print("(no confusion matrix available; overall metrics only)")
        return

    n = len(names)
    per_class: list[tuple[str, float, int]] = []
    for i in range(n):
        col_sum = float(matrix[:, i].sum()) if i < matrix.shape[1] else 0.0
        correct = float(matrix[i, i]) if i < matrix.shape[0] and i < matrix.shape[1] else 0.0
        recall = correct / col_sum if col_sum > 0 else float("nan")
        per_class.append((str(names[i]), recall, int(col_sum)))

    print("\nper-class recall (worst first)  [label: recall  (n_val)]:")
    def sort_key(t: tuple[str, float, int]) -> float:
        return t[1] if t[1] == t[1] else -1.0  # NaN (no samples) sorts first
    weak = 0
    for label, recall, n_val in sorted(per_class, key=sort_key):
        if n_val == 0:
            print(f"  {label:>3}:   n/a   (0)  <-- no val samples")
            continue
        flag = "  <-- WEAK" if recall < 0.85 else ""
        if recall < 0.85:
            weak += 1
        print(f"  {label:>3}: {recall:.3f}  ({n_val}){flag}")
    print(f"\n{weak} class(es) below 0.85 recall")


if __name__ == "__main__":
    main()
