#!/usr/bin/env bash
# Anti-drift ClubWPT card-detector fine-tune, end to end.
#
# Once your ClubWPT card dataset is acquired and labeled in the 53-class scheme
# (images/{train,val} + labels/{train,val}, same class order as the existing
# cv_lab card datasets), run this from the repo root:
#
#   bash cv_lab/scripts/finetune_clubwpt.sh path/to/yolo_cards_clubwpt_v1
#
# It will: (1) baseline the current model, (2) build the rehearsal mix,
# (3) fine-tune with the anti-drift recipe, (4) report per-class drift.
set -euo pipefail

CLUBWPT_DIR="${1:?usage: finetune_clubwpt.sh <clubwpt_dataset_dir> [run_name]}"
RUN_NAME="${2:-clubwpt_mixed}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OLD_WEIGHTS="cv_lab/models/best (4).pt"
MIXED_DIR="cv_lab/datasets/yolo_cards_clubwpt_mixed"
MIXED_YAML="$MIXED_DIR/data.yaml"
RUN_DIR="cv_lab/runs/yolo_cards/$RUN_NAME"
BASELINE_JSON="cv_lab/results/card_baseline_${RUN_NAME}.json"

echo "==> [1/4] building rehearsal mix from ClubWPT + original datasets"
python cv_lab/scripts/build_mixed_card_dataset.py \
  --clubwpt "$CLUBWPT_DIR" \
  --out "$MIXED_DIR" \
  --rehearsal-frac 0.25 \
  --min-per-class 8

echo "==> [2/4] baselining the CURRENT model on the mixed val set (before fine-tune)"
python cv_lab/scripts/val_cards_per_class.py \
  --weights "$OLD_WEIGHTS" \
  --data "$MIXED_YAML" \
  --save-json "$BASELINE_JSON"

echo "==> [3/4] fine-tuning with anti-drift recipe (freeze=10, lr0=1e-3, cos_lr, patience=12)"
python cv_lab/scripts/train_yolov12_cards.py \
  --weights "$OLD_WEIGHTS" \
  --data "$MIXED_YAML" \
  --epochs 60 \
  --name "$RUN_NAME" \
  --exist-ok

echo "==> [4/4] measuring drift: per-class mAP of the fine-tuned model vs baseline"
python cv_lab/scripts/val_cards_per_class.py \
  --weights "$RUN_DIR/weights/best.pt" \
  --data "$MIXED_YAML" \
  --baseline "$BASELINE_JSON"

echo
echo "done. new weights: $RUN_DIR/weights/best.pt"
echo "if no class regressed, promote with:  cp \"$RUN_DIR/weights/best.pt\" \"cv_lab/models/best (5).pt\""
