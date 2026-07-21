"""End-to-end: run the two-model pipeline over frames -> spine timeline.

Ties everything together for live validation:
    frames (dir of images or a video)
      -> TwoModelPipeline (Model 1 regions + Model 2 card rank/suit)
      -> region_detections Frames
      -> build_yolo_hand_timeline (the reconstruction spine)
      -> timeline JSON + a fixture JSON (so results are inspectable/re-runnable)

    # over a directory of extracted frames
    python cv_lab/scripts/run_two_model_timeline.py --frames-dir cv_lab/frames --device mps

    # over sampled frames straight from a video
    python cv_lab/scripts/run_two_model_timeline.py \
        --video data/videos/clubwpt_session_01.mov --every 5.0 --max-frames 20 --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from two_model_infer import TwoModelPipeline  # noqa: E402
from cv_lab.scripts import region_detections as rd  # noqa: E402
from cv_lab.scripts import build_yolo_hand_timeline as bt  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_T_RE = re.compile(r"t(\d+(?:\.\d+)?)s?", re.IGNORECASE)


def _time_from_name(name: str, fallback: float) -> float:
    m = _T_RE.search(name)
    return float(m.group(1)) if m else fallback


def _frame_to_dict(frame: rd.Frame) -> dict:
    d = asdict(frame)
    d["detections"] = [
        {"cls": det.cls, "conf": det.conf, "xyxy": list(det.xyxy), "attr": det.attr,
         "brightness": det.brightness}
        for det in frame.detections
    ]
    return d


def _iter_video_frames(video: Path, every_s: float, max_frames: int):
    """Yield (bgr_image, time_s, frame_index) sampled every `every_s` seconds."""
    import av  # PyAV, already a runtime dep
    container = av.open(str(video))
    stream = container.streams.video[0]
    next_t = 0.0
    yielded = 0
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        if frame.time + 1e-6 < next_t:
            continue
        yield frame.to_ndarray(format="bgr24"), float(frame.time), frame.index
        yielded += 1
        next_t = frame.time + every_s
        if max_frames and yielded >= max_frames:
            break
    container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", help="directory of extracted full-table frames")
    src.add_argument("--video", help="sample frames straight from a video file")
    parser.add_argument("--every", type=float, default=5.0, help="video sampling period (s)")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--region-weights", default="")
    parser.add_argument("--card-weights", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default="cv_lab/results/two_model_timeline.json")
    parser.add_argument("--fixture-out", default="cv_lab/results/two_model_frames.json")
    args = parser.parse_args()

    pipe = TwoModelPipeline(args.region_weights or None, args.card_weights or None, device=args.device)

    frames: list[rd.Frame] = []
    if args.frames_dir:
        paths = sorted(p for p in Path(args.frames_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for i, p in enumerate(paths):
            frame = pipe.frame(p, time_s=_time_from_name(p.name, float(i)), video_frame=i)
            frames.append(frame)
            cards = [rd.read_card_label(d) for d in frame.detections if d.cls == "face_card"]
            print(f"{p.name}: {len(frame.detections)} dets, cards={[c for c in cards if c]}")
    else:
        # Video path: run each sampled frame through the pipeline by writing a temp image.
        import cv2
        tmp = Path("cv_lab/results/_tmp_frame.png")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        for bgr, t, idx in _iter_video_frames(Path(args.video), args.every, args.max_frames):
            cv2.imwrite(str(tmp), bgr)
            frame = pipe.frame(tmp, time_s=t, video_frame=idx)
            frame.image = f"{Path(args.video).name}@{t:.2f}s"
            frames.append(frame)
            cards = [rd.read_card_label(d) for d in frame.detections if d.cls == "face_card"]
            print(f"t={t:6.2f}s: {len(frame.detections)} dets, cards={[c for c in cards if c]}")

    fixture = [_frame_to_dict(f) for f in frames]
    Path(args.fixture_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fixture_out).write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    timeline = bt.build_hand_timeline(frames)
    Path(args.out).write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    print(f"\nframes={timeline['summary']['frames']}  states={timeline['summary']['states']}  "
          f"hands={timeline['summary']['hands']}  complete_hands={timeline['summary']['complete_hands']}")
    print(f"fixture={args.fixture_out}")
    print(f"timeline={args.out}")


if __name__ == "__main__":
    main()
