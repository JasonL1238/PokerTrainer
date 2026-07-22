"""Isolated frame extractor for CV experimentation.

Deliberately separate from poker_tracker/. Uses PyAV (bundles its own FFmpeg)
because macOS screen-recording .mov files often fail to open in OpenCV's build.

Usage:
    python cv_lab/scripts/extract_frames.py --video data/videos/clubwpt_session_01.mov \
        --count 12 --out cv_lab/frames --preview-width 900
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2


def extract(video: str, out_dir: Path, count: int, preview_width: int,
            start_frac: float = 0.0, end_frac: float = 1.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(video)
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base)
    span_start = duration * start_frac
    span_end = duration * end_frac
    targets = [span_start + (span_end - span_start) * i / max(1, count - 1)
               for i in range(count)]

    saved: list[Path] = []
    for idx, t in enumerate(targets):
        container.seek(int(t / stream.time_base), stream=stream)
        frame = None
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= t:
                break
        if frame is None:
            continue
        img = frame.to_ndarray(format="bgr24")
        name = f"frame_{idx:02d}_t{t:07.2f}s.png"
        full = out_dir / name
        cv2.imwrite(str(full), img)
        saved.append(full)

        # downscaled preview for cheap visual inspection
        h, w = img.shape[:2]
        scale = preview_width / w
        prev = cv2.resize(img, (preview_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(preview_dir / name.replace(".png", ".jpg")), prev,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"[{idx:02d}] t={t:7.2f}s  {w}x{h} -> {full.name}")

    container.close()
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="cv_lab/frames")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--preview-width", type=int, default=900)
    ap.add_argument("--start-frac", type=float, default=0.0)
    ap.add_argument("--end-frac", type=float, default=1.0)
    args = ap.parse_args()
    extract(args.video, Path(args.out), args.count, args.preview_width,
            args.start_frac, args.end_frac)
