"""Backfill ``brightness`` onto a cached fixture's face_card detections.

The cached ``frames_v0X.json`` fixtures (from run_two_model_timeline.py's
--dump-frames path, replayed by rebuild_timeline_from_frames.py) only ever
stored ``cls``/``conf``/``xyxy``/``attr`` per detection -- no pixel data, so
the hero-fold "greyed out cards" signal added to region_detections.py can't be
read from them as-is. This script re-opens the SAME source video, decodes it
once sequentially, and for every fixture frame whose detections include a
hero-zone face_card box, re-crops that box from the matching video frame and
writes back a mean-grayscale ``brightness`` value -- so existing fixtures can
be reused (via rebuild_timeline_from_frames.py) without re-running the region
detector or card classifier.

    python cv_lab/scripts/add_brightness_to_fixture.py \
        --frames cv_lab/results/frames_v05.json \
        --video "data/videos/Screen Recording 2026-07-11 at 12.45.27 PM.mov" \
        --out cv_lab/results/frames_v05.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_yolo_card_timeline import _zone_for_box  # noqa: E402

_TOL_S = 0.5  # fixture frame time vs. decoded video frame time, seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="fixture frames JSON to augment")
    ap.add_argument("--video", required=True, help="source video the fixture was sampled from")
    ap.add_argument("--out", required=True, help="output fixture JSON path")
    args = ap.parse_args()

    import av
    import cv2

    fixture = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    targets = sorted({round(f["time_s"], 2) for f in fixture})
    by_time: dict[float, list[dict]] = {}
    for f in fixture:
        by_time.setdefault(round(f["time_s"], 2), []).append(f)

    container = av.open(args.video)
    stream = container.streams.video[0]
    ti = 0
    n_updated = 0
    n_dets = 0
    for frame in container.decode(stream):
        if frame.time is None or ti >= len(targets):
            continue
        while ti < len(targets) and frame.time > targets[ti] + _TOL_S:
            ti += 1  # video has no frame near this target; skip it
        if ti >= len(targets):
            break
        if abs(frame.time - targets[ti]) <= _TOL_S:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            for fx in by_time[targets[ti]]:
                for det in fx.get("detections", []):
                    if det.get("cls") != "face_card":
                        continue
                    x0, y0, x1, y1 = det["xyxy"]
                    cx, cy = (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h
                    if _zone_for_box(cx, cy) != "hero":
                        continue
                    xi0, yi0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
                    xi1, yi1 = min(int(round(x1)), w), min(int(round(y1)), h)
                    if xi1 <= xi0 or yi1 <= yi0:
                        continue
                    crop = img[yi0:yi1, xi0:xi1]
                    det["brightness"] = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
                    n_dets += 1
            n_updated += 1
            ti += 1
    container.close()

    Path(args.out).write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"matched {n_updated}/{len(targets)} fixture frames, "
          f"backfilled brightness on {n_dets} hero face_card detections")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
