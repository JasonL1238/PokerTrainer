"""Extract frames across a time WINDOW at a fixed interval.

Separate from poker_tracker/. Uses PyAV (macOS screen-recording .mov files
often fail to open in OpenCV's bundled FFmpeg).

Unlike extract_frames.py (evenly spaces N frames across the whole clip), this
samples every `interval` seconds between `start` and `end` — for densely
walking a single hand to find street transitions / decision points.

Usage:
    python cv_lab/scripts/extract_window.py --video data/videos/clubwpt_session_01.mov \
        --start 180 --end 270 --interval 2 --out cv_lab/hand01/frames --preview-width 900
    # add --full to also write full-res PNGs (default: preview JPGs only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2


def extract_window(video: str, out_dir: Path, start: float, end: float,
                   interval: float, preview_width: int, full: bool,
                   times: list[float] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(video)
    stream = container.streams.video[0]

    if times:
        targets = list(times)
    else:
        n = int((end - start) / interval) + 1
        targets = [start + i * interval for i in range(n)]

    saved: list[Path] = []
    for t in targets:
        container.seek(int(t / stream.time_base), stream=stream)
        frame = None
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= t:
                break
        if frame is None:
            continue
        img = frame.to_ndarray(format="bgr24")
        base = f"t{t:07.2f}s"
        if full:
            full_path = out_dir / f"{base}.png"
            cv2.imwrite(str(full_path), img)
            saved.append(full_path)

        h, w = img.shape[:2]
        scale = preview_width / w
        prev = cv2.resize(img, (preview_width, int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(preview_dir / f"{base}.jpg"), prev,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"t={t:7.2f}s  {w}x{h} -> {base}")

    container.close()
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="cv_lab/hand01/frames")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--preview-width", type=int, default=900)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--times", type=str, default="",
                    help="comma-separated explicit timestamps (seconds); overrides start/end/interval")
    args = ap.parse_args()
    times = [float(x) for x in args.times.split(",") if x.strip()] or None
    extract_window(args.video, Path(args.out), args.start, args.end,
                   args.interval, args.preview_width, args.full, times)
