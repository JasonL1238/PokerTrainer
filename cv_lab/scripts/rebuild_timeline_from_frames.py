"""Re-run the reconstruction spine on cached fixture frames (no GPU needed).

run_two_model_pipeline.py --dump-frames writes every sampled frame's detections
(with OCR/classifier attrs already filled) in region_detections.load_frames
format. This script replays that cache through build_hand_timeline, so spine
changes can be iterated in seconds instead of re-running video inference.

    python cv_lab/scripts/rebuild_timeline_from_frames.py \
        --frames cv_lab/results/frames_v05.json \
        --out cv_lab/results/two_model_timeline_v05.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(__file__))

from cv_lab.scripts import region_detections as rd  # noqa: E402
from build_yolo_hand_timeline import build_hand_timeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, help="fixture frames JSON (--dump-frames output)")
    parser.add_argument("--out", required=True, help="timeline JSON destination")
    args = parser.parse_args()

    frames = rd.load_frames(args.frames)
    print(f"loaded {len(frames)} cached frames from {args.frames}")
    timeline = build_hand_timeline(frames)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"timeline -> {out}")
    print(f"summary: {json.dumps(timeline.get('summary', {}))}")


if __name__ == "__main__":
    main()
