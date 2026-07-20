"""End-to-end two-model runtime: video -> Model 1 -> crop -> Model 2 -> spine.

This is the live wiring of the Design-A architecture:

  Model 1 (region detector, 8 classes)  boxes every region incl. face_card
      |  for each face_card box: crop (+pad)
      v
  Model 2 (card classifier, 52 classes) names the rank+suit
      |  region_detections.frame_from_models fills each face_card's attr
      v
  reconstruction spine (build_yolo_hand_timeline.build_hand_timeline)
      |
      v  hand timeline JSON

Neither model is invoked by the spine directly -- we build region_detections.Frame
objects and hand them to build_hand_timeline(), exactly as the fixture path does.

  python cv_lab/scripts/run_two_model_pipeline.py \
      --video data/videos/clubwpt_session_01.mov \
      --start 0 --end 120 --interval 2 --device mps \
      --out cv_lab/results/two_model_timeline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import av

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402
from card_classifier import CardClassifier, DEFAULT_CLS_WEIGHTS  # noqa: E402
import region_detections as rd  # noqa: E402
from build_yolo_hand_timeline import build_hand_timeline  # noqa: E402

DEFAULT_DETECTOR = REPO_ROOT / "cv_lab" / "models" / "region_spine_v1.pt"
if not DEFAULT_DETECTOR.exists():
    DEFAULT_DETECTOR = REPO_ROOT / "cv_lab" / "runs" / "yolo_cards" / "region_spine_v1" / "weights" / "best.pt"
DEFAULT_VIDEO = REPO_ROOT / "data" / "videos" / "clubwpt_session_01.mov"


def _iou(a: dict, b: dict) -> float:
    ix0, iy0 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix1, iy1 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_inside(inner: dict, outer: dict) -> bool:
    cx = (inner["x1"] + inner["x2"]) / 2
    cy = (inner["y1"] + inner["y2"]) / 2
    return outer["x1"] <= cx <= outer["x2"] and outer["y1"] <= cy <= outer["y2"]


def _dedupe_face_cards(rows: list[dict], iou_thresh: float) -> list[dict]:
    """Collapse nested/overlapping face_card boxes that plain NMS leaves behind.

    Model 1 sometimes emits a small card box AND a larger one enclosing it (IoU
    below the NMS threshold, so both survive). Greedily keep the highest-conf
    face_card and drop any later one that overlaps it (by IoU) or whose center
    falls inside a kept box. Non-card rows pass through untouched.
    """
    others = [r for r in rows if r["class"] != "face_card"]
    cards = sorted((r for r in rows if r["class"] == "face_card"),
                   key=lambda r: r["confidence"], reverse=True)
    kept: list[dict] = []
    for c in cards:
        if any(_iou(c, k) >= iou_thresh or _center_inside(c, k) or _center_inside(k, c)
               for k in kept):
            continue
        kept.append(c)
    return others + kept


def _detect_regions(model, img, *, conf: float, imgsz: int, iou: float, device: str,
                    dedupe_iou: float) -> list[dict]:
    """Run Model 1 on a BGR frame -> deduped region rows for frame_from_models."""
    kwargs = {"conf": conf, "iou": iou, "imgsz": imgsz, "verbose": False}
    if device:
        kwargs["device"] = device
    result = model.predict(img, **kwargs)[0]
    rows: list[dict] = []
    for box in result.boxes:
        x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
        rows.append({
            "class": str(model.names[int(box.cls[0])]),  # true region class
            "confidence": float(box.conf[0]),
            "x1": x0, "y1": y0, "x2": x1, "y2": y1,
        })
    return _dedupe_face_cards(rows, dedupe_iou)


def _sample_times(container, stream, start: float, end: float, interval: float):
    """Yield (t_seconds, bgr_image) sampled every `interval`s via seek."""
    t = start
    while t <= end:
        container.seek(int(t / stream.time_base), stream=stream)
        frame = None
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= t:
                break
        if frame is not None:
            yield t, frame.to_ndarray(format="bgr24")
        t += interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--model1", default=str(DEFAULT_DETECTOR), help="region detector weights")
    parser.add_argument("--model2", default=str(DEFAULT_CLS_WEIGHTS), help="card classifier weights")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between sampled frames")
    parser.add_argument("--conf", type=float, default=0.35, help="Model 1 detection confidence")
    parser.add_argument("--iou", type=float, default=0.30, help="Model 1 NMS IoU threshold")
    parser.add_argument("--dedupe-iou", type=float, default=0.35,
                        help="collapse face_card boxes overlapping more than this (nested-box cleanup)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cls-imgsz", type=int, default=128)
    parser.add_argument("--pad", type=float, default=0.12, help="face_card crop expansion for Model 2")
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default=str(REPO_ROOT / "cv_lab" / "results" / "two_model_timeline.json"))
    parser.add_argument("--dump-detections", default="", help="optional: write raw per-frame detections here")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR))
    args = parser.parse_args()

    vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(vendor)
    print(f"loading Model 1 (region detector): {args.model1}")
    detector = YOLO(args.model1)
    print(f"loading Model 2 (card classifier): {args.model2}")
    classifier = CardClassifier(weights=args.model2, vendor=vendor,
                                imgsz=args.cls_imgsz, device=args.device)

    args.video = str(Path(args.video).expanduser().resolve())
    print(f"sampling {args.video}  [{args.start}s..{args.end}s every {args.interval}s]")
    container = av.open(args.video)
    stream = container.streams.video[0]
    frames: list[rd.Frame] = []
    raw_dump: list[dict] = []
    n_cards = 0
    for i, (t, img) in enumerate(_sample_times(container, stream, args.start, args.end, args.interval)):
        rows = _detect_regions(detector, img, conf=args.conf, imgsz=args.imgsz, iou=args.iou,
                               device=args.device, dedupe_iou=args.dedupe_iou)
        frame = rd.frame_from_models(img, t, rows, classifier=classifier,
                                     image_name=f"t{t:07.2f}", pad=args.pad, video_frame=i)
        frames.append(frame)
        cards = [d for d in frame.detections if d.cls == "face_card" and d.attr]
        n_cards += len(cards)
        if args.dump_detections:
            raw_dump.append({
                "t": t,
                "detections": [{"cls": d.cls, "conf": round(d.conf, 3),
                                "xyxy": [round(v, 1) for v in d.xyxy], "attr": d.attr}
                               for d in frame.detections],
            })
        if i % 10 == 0:
            print(f"  frame {i:>4} t={t:7.2f}s  regions={len(rows):>2}  named_cards={len(cards)}")
    container.close()

    print(f"\nsampled {len(frames)} frames, {n_cards} named cards total")
    print("building hand timeline via reconstruction spine...")
    timeline = build_hand_timeline(frames)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    summary = timeline.get("summary", {})
    print(f"\ntimeline -> {out}")
    print(f"summary: {json.dumps(summary)}")
    if args.dump_detections:
        Path(args.dump_detections).write_text(json.dumps(raw_dump, indent=2), encoding="utf-8")
        print(f"raw detections -> {args.dump_detections}")


if __name__ == "__main__":
    main()
