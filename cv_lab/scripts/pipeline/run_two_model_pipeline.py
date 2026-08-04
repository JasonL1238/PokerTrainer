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

  python cv_lab/scripts/pipeline/run_two_model_pipeline.py \
      --video data/videos/clubwpt_session_01.mov \
      --start 0 --end 120 --interval 2 --device mps \
      --out cv_lab/results/two_model_timeline.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import av
import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import cv_lab.scripts.pipeline.region_detections as rd  # noqa: E402
from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline  # noqa: E402
from cv_lab.scripts.pipeline.card_classifier import (  # noqa: E402
    DEFAULT_CLS_WEIGHTS,
    CardClassifier,
)
from cv_lab.scripts.pipeline.classify_screen import classify as classify_screen  # noqa: E402
from cv_lab.scripts.pipeline.evaluate_yolo_cards import (  # noqa: E402
    DEFAULT_YOLOV12_VENDOR,
    _load_yolo_class,
    _resolve_vendor_path,
)

DEFAULT_DETECTOR = REPO_ROOT / "cv_lab" / "models" / "region_spine_v1.pt"
if not DEFAULT_DETECTOR.exists():
    DEFAULT_DETECTOR = REPO_ROOT / "cv_lab" / "runs" / "yolo_cards" / "region_spine_v1" / "weights" / "best.pt"
DEFAULT_VIDEO = REPO_ROOT / "data" / "videos" / "clubwpt_session_01.mov"
REVIEW_FRAME_MAX_WIDTH = 1280


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


def _sample_times(container, stream, start: float, end: float, interval: float,
                  stats: dict | None = None):
    """Yield (t_seconds, prev_static_until_s, bgr_image) for the frame IN EFFECT
    at each sample time, every ``interval`` seconds via seek.

    "In effect" means the last frame at or before the sample time -- the picture
    the screen was actually showing at t -- not the first frame after it. The
    distinction matters on variable-rate screen recordings, which encode a frame
    only when the display changes: seeking forward from t skips over the state
    that was on screen at t and can skip it at every sample time, so a table
    state that persisted for many seconds was never observed at all. Sampling
    the in-effect frame guarantees any state that persists at least one interval
    is emitted.

    ``t_seconds`` is the emitted frame's own presentation timestamp, never the
    requested time. Stamping a picture with the requested time moves it in time
    and misplaces coverage holes; the true timestamp keeps them where they are.

    A sample time answered by the frame already emitted is not re-emitted -- one
    decoded frame is one observation, and emitting it twice would let it
    debounce itself into a confirmed reading. But the answer itself is
    evidence: on a change-driven recording, "the frame at t is still the
    previous one" is the decoder proving no frame exists in between, i.e. the
    screen did not change through t. That proof is returned as
    ``prev_static_until_s`` on the NEXT emission (and left in
    ``stats["last_static_until_s"]`` for the final frame): the latest sample
    time the previous emitted frame was proven still on screen, or None when no
    such request landed. Callers hang it on the previous frame so the spine can
    tell a provably-static stretch from an unobserved one.

    Sampling stops once the decoder proves no undecoded frame remains, not at
    ``end``: past the last frame there is nothing left that could ever answer a
    later request. ``stats`` collects what the sampler did: times requested,
    frames emitted, requests answered by an already-emitted frame, the pending
    static-until proof for the last emitted frame, and the time sampling
    stopped (``None`` when it reached ``end`` with frames still available).
    """
    counts = {"requested": 0, "emitted": 0, "duplicate_times": 0, "ended_at": None,
              "last_static_until_s": None}
    if stats is not None:
        stats.clear()
        stats.update(counts)
        counts = stats
    t = start
    last_pts = None
    static_until: float | None = None
    stream_exhausted = False
    while t <= end:
        if stream_exhausted:
            # A previous request decoded to end of stream: no frame beyond the
            # one already emitted exists, so every later request is the same
            # answer. Stop rather than report one observation many times.
            counts["ended_at"] = t
            return
        counts["requested"] += 1
        container.seek(int(t / stream.time_base), stream=stream)
        best = None
        eof = True
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) <= t:
                best = frame
                continue
            if best is None:
                # The window starts before the stream's first frame; that first
                # frame answers the request, as a request is a normal ask, not
                # an exhausted stream.
                best = frame
            eof = False
            break
        if best is None:
            # Nothing decodable at all at or around t.
            counts["ended_at"] = t
            return
        if eof:
            stream_exhausted = True
        if best.pts == last_pts:
            counts["duplicate_times"] += 1
            # The static proof holds only for a genuine in-effect answer. A
            # bootstrap answer (the stream's first frame, still in the future
            # of t) was not on screen at t and proves nothing about it.
            if float(best.pts * stream.time_base) <= t:
                static_until = t
                counts["last_static_until_s"] = t
            t += interval
            continue
        prev_static = static_until
        static_until = None
        counts["last_static_until_s"] = None
        last_pts = best.pts
        counts["emitted"] += 1
        yield float(best.pts * stream.time_base), prev_static, best.to_ndarray(format="bgr24")
        t += interval


def _save_review_frame(image, destination: Path) -> None:
    """Save a compact visual-audit copy without changing model input pixels."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    if width > REVIEW_FRAME_MAX_WIDTH:
        scale = REVIEW_FRAME_MAX_WIDTH / width
        image = cv2.resize(
            image,
            (REVIEW_FRAME_MAX_WIDTH, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(
        str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 80]
    ):
        raise RuntimeError(f"Could not save review frame: {destination}")


def _write_progress(
    destination: Path | None,
    *,
    current: int,
    total: int,
    stage: str,
) -> None:
    """Publish compact machine-readable progress for the detached UI worker."""
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps({"stage": stage, "current": current, "total": total}),
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--model1", default=str(DEFAULT_DETECTOR), help="region detector weights")
    parser.add_argument("--model2", default=str(DEFAULT_CLS_WEIGHTS), help="card classifier weights")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between sampled frames (1.0 = dense sampling; the "
                             "reconstruction spine self-calibrates its debounces to this rate)")
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
    parser.add_argument("--dump-frames", default="",
                        help="optional: write fixture-compatible frames (region_detections.load_frames) "
                             "so the spine can be re-run on cached detections without GPU inference")
    parser.add_argument(
        "--frame-dir",
        default="",
        help="optional: save compact source frames used by the evidence-review UI",
    )
    parser.add_argument(
        "--progress-file",
        default="",
        help="optional: publish live frame-processing progress as JSON",
    )
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
    progress_path = Path(args.progress_file).resolve() if args.progress_file else None
    total_samples = max(1, math.floor((args.end - args.start) / args.interval) + 1)
    _write_progress(
        progress_path,
        current=0,
        total=total_samples,
        stage="frames",
    )
    frames: list[rd.Frame] = []
    raw_dump: list[dict] = []
    sampling: dict = {}
    nontable_reasons: dict = {}
    n_cards = 0
    n_nontable = 0
    for i, (t, prev_static, img) in enumerate(_sample_times(container, stream, args.start,
                                                            args.end, args.interval,
                                                            stats=sampling)):
        if prev_static is not None and frames:
            # The decoder proved the previous emitted frame was still on screen
            # through prev_static (see _sample_times); the spine reads this to
            # tell a provably-static stretch from an unobserved one.
            frames[-1].observed_static_until_s = prev_static
        screen_label, _anchor = classify_screen(img, reasons=nontable_reasons)
        image_name = f"t{t:09.2f}"
        h, w = int(img.shape[0]), int(img.shape[1])
        if screen_label != "table":
            # Tab-in-front / lobby / modal: keep the timestamp for coverage-gap
            # detection, but do not invent detections from a non-table screen.
            n_nontable += 1
            frames.append(
                rd.Frame(
                    image=image_name,
                    time_s=t,
                    width=w,
                    height=h,
                    detections=[],
                    video_frame=i,
                    screen="nontable",
                )
            )
            _write_progress(
                progress_path,
                current=i + 1,
                total=total_samples,
                stage="frames",
            )
            if args.dump_detections:
                raw_dump.append({"t": t, "screen": "nontable", "detections": []})
            if i % 10 == 0:
                print(f"  frame {i:>4} t={t:7.2f}s  screen=nontable (skipped inference)")
            continue

        rows = _detect_regions(detector, img, conf=args.conf, imgsz=args.imgsz, iou=args.iou,
                               device=args.device, dedupe_iou=args.dedupe_iou)
        if args.frame_dir:
            image_path = Path(args.frame_dir).resolve() / f"{image_name}.jpg"
            _save_review_frame(img, image_path)
            image_name = str(image_path)
        frame = rd.frame_from_models(img, t, rows, classifier=classifier,
                                     image_name=image_name, pad=args.pad, video_frame=i)
        frames.append(frame)
        _write_progress(
            progress_path,
            current=i + 1,
            total=total_samples,
            stage="frames",
        )
        cards = [d for d in frame.detections if d.cls == "face_card" and d.attr]
        n_cards += len(cards)
        if args.dump_detections:
            raw_dump.append({
                "t": t,
                "screen": "table",
                "detections": [{"cls": d.cls, "conf": round(d.conf, 3),
                                "xyxy": [round(v, 1) for v in d.xyxy], "attr": d.attr}
                               for d in frame.detections],
            })
        if i % 10 == 0:
            print(f"  frame {i:>4} t={t:7.2f}s  regions={len(rows):>2}  named_cards={len(cards)}")
    container.close()
    tail_static = sampling.get("last_static_until_s")
    if tail_static is not None and frames:
        frames[-1].observed_static_until_s = tail_static

    print(
        f"\nsampled {len(frames)} frames "
        f"({n_nontable} nontable skipped), {n_cards} named cards total"
    )
    # State plainly what the requested window did and did not cover, so a run
    # that asked for more video than exists reads as a shorter run rather than
    # as a full one.
    if sampling.get("ended_at") is not None:
        print(
            f"sampling stopped at t={sampling['ended_at']:.2f}s: the stream ends "
            f"before that time (requested up to {args.end}s)"
        )
    if sampling.get("duplicate_times"):
        print(
            f"{sampling['duplicate_times']} requested sample time(s) resolved to an "
            "already-sampled frame and were not emitted; the recording has no "
            "distinct frame there"
        )
    print("building hand timeline via reconstruction spine...")
    _write_progress(
        progress_path,
        current=total_samples,
        total=total_samples,
        stage="timeline",
    )
    timeline = build_hand_timeline(frames, sampling_interval_s=args.interval)
    if nontable_reasons:
        # WHY frames were called nontable, so a whole-recording blackout reads
        # as "capture scale outside the classifier band" instead of "recording
        # of a lobby". Additive metadata key.
        timeline["metadata"]["nontable_reasons"] = dict(sorted(nontable_reasons.items()))
        print(f"nontable reasons: {json.dumps(timeline['metadata']['nontable_reasons'])}")
    if args.frame_dir:
        used_images = {
            str(Path(state["image"]).resolve())
            for state in timeline.get("states", [])
            if state.get("image")
        }
        for saved_frame in Path(args.frame_dir).resolve().glob("*.jpg"):
            if str(saved_frame.resolve()) not in used_images:
                saved_frame.unlink()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    summary = timeline.get("summary", {})
    print(f"\ntimeline -> {out}")
    print(f"summary: {json.dumps(summary)}")
    if args.dump_detections:
        Path(args.dump_detections).write_text(json.dumps(raw_dump, indent=2), encoding="utf-8")
        print(f"raw detections -> {args.dump_detections}")
    if args.dump_frames:
        fixture = [{
            "image": f.image, "time_s": f.time_s, "width": f.width, "height": f.height,
            "video_frame": f.video_frame,
            "screen": getattr(f, "screen", "table"),
            # Written only when the decoder proved a static span, so fixtures
            # from dense recordings stay byte-identical to the pre-field era.
            **({"observed_static_until_s": f.observed_static_until_s}
               if f.observed_static_until_s is not None else {}),
            "detections": [{"cls": d.cls, "conf": round(d.conf, 4),
                            "xyxy": [round(v, 2) for v in d.xyxy], "attr": d.attr}
                           for d in f.detections],
        } for f in frames]
        Path(args.dump_frames).write_text(json.dumps(fixture), encoding="utf-8")
        print(f"fixture frames -> {args.dump_frames}")


if __name__ == "__main__":
    main()
