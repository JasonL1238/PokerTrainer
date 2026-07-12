"""Automatic key-frame selector for poker screen recordings (no ground truth). v2.

Separate from poker_tracker/.

WHY v2 is region-aware (see cv_lab/notes/04): whole-frame motion-diff FAILS for
poker, because state changes vary enormously in pixel footprint. A 3-card flop is
a huge diff (~70) but a single turn card, a call chip, an action pill, or a pot
digit change is tiny (~1-2) and sinks below the per-second timer-tick noise. A
single global threshold cannot separate "turn card dealt" from "timer ticked".

So we diff SEMANTIC REGIONS instead:
  - POT number region  -> changes on every bet/call/raise (the pot updates)
  - BOARD region       -> changes on every street (flop/turn/river)
These regions are dead-quiet when idle (p90 ~0.1) and spike 5-70 on real events,
giving clean separation. A keyframe fires when ANY watched region changes; the
representative is the settled frame right AFTER the change burst.

Whole-frame diff is still computed, but only as a HAND-BOUNDARY / transition
signal (a big global burst = deal animation / pot push / screen change).

ROIs are normalized fractions calibrated to the ClubWPT Gold layout in THIS
recording (table fills the frame). Because edges vary between recordings, these
should later be re-anchored to on-screen landmarks (POT/BLINDS text) rather than
absolute fractions -- noted as the generalization step.

Usage:
  python cv_lab/scripts/select_keyframes.py --video data/videos/clubwpt_session_01.mov \
      --start 300 --end 396 --fps 3 --out cv_lab/hand01/auto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

import landmark_anchor as la

# normalized (x0, x1, y0, y1) ROIs for ClubWPT Gold, this recording's geometry
ROIS = {
    "pot": (0.47, 0.57, 0.345, 0.405),
    "board": (0.33, 0.63, 0.39, 0.55),
}
# per-region change thresholds (mean abs gray diff, 0..255) -- above idle noise (~0.1)
ROI_THRESH = {"pot": 1.5, "board": 3.0}   # pot 1.5 still ~14x the idle noise floor (~0.11)
WHOLE_FRAME_BOUNDARY = 4.0   # global diff above this = big animation / likely hand boundary


def _roi_norm(img, box, size=(160, 60)):
    """Crop a normalized (fraction-of-frame) ROI box."""
    H, W = img.shape[:2]
    x0, x1, y0, y1 = box
    r = img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)].astype(np.float32)
    return cv2.resize(r, size)


def _roi_px(img, pbox, size=(160, 60)):
    """Crop a pixel ROI box (from landmark anchoring)."""
    H, W = img.shape[:2]
    x0, x1, y0, y1 = pbox
    x0, x1 = max(0, min(x0, x1)), min(W, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(H, max(y0, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((size[1], size[0]), np.float32)
    return cv2.resize(img[y0:y1, x0:x1].astype(np.float32), size)


def analyze(video, start, end, fps, thumb_w, anchored=False):
    c = av.open(video)
    s = c.streams.video[0]
    step = 1.0 / fps
    targets = [start + i * step for i in range(int((end - start) / step) + 1)]

    times, whole, roi_series = [], [], {k: [] for k in ROIS}
    roi_snaps = []                   # per-sample dict of ROI crops, for region-based dedup
    prev_whole, prev_roi = None, {}
    ti = 0
    c.seek(int(start / s.time_base), stream=s)
    for fr in c.decode(s):
        t = float(fr.pts * s.time_base)
        if t < start:
            continue
        if ti >= len(targets):
            break
        if t >= targets[ti]:
            bgr = fr.to_ndarray(format="bgr24")
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            H, W = g.shape
            th = cv2.resize(g, (thumb_w, int(H * thumb_w / W))).astype(np.float32)
            times.append(round(targets[ti], 2))
            whole.append(float(np.mean(np.abs(th - prev_whole))) if prev_whole is not None else 0.0)
            prev_whole = th
            # per-frame landmark anchoring (edge/scale tolerant); fall back to
            # canonical fractions if the table can't be located this frame.
            amap = None
            if anchored:
                a = la.anchor(bgr)
                if a is not None:
                    amap = a["map_roi"]
            snap = {}
            for k in ROIS:
                if amap is not None:
                    r = _roi_px(g, amap(la.CANON_ROIS[k]))
                else:
                    r = _roi_norm(g, ROIS[k])
                roi_series[k].append(float(np.mean(np.abs(r - prev_roi[k]))) if k in prev_roi else 0.0)
                prev_roi[k] = r
                snap[k] = r
            roi_snaps.append(snap)
            ti += 1
        if t > end:
            break
    c.close()
    return times, whole, roi_series, roi_snaps


def select(times, whole, roi_series, roi_snaps, merge_diff=2.0):
    n = len(times)
    changed = []
    for i in range(n):
        c = any(roi_series[k][i] > ROI_THRESH[k] for k in ROIS)
        changed.append(c)

    # group consecutive changed samples into bursts
    bursts, i = [], 0
    while i < n:
        if changed[i]:
            j = i
            while j + 1 < n and changed[j + 1]:
                j += 1
            bursts.append((i, j))
            i = j + 1
        else:
            i += 1

    # representative = first settled (non-changed) sample AFTER each burst; else burst end
    reps = []
    for (a, b) in bursts:
        r = b + 1 if (b + 1 < n and not changed[b + 1]) else b
        reps.append(r)
    # always include the first settled state at window open
    if reps and reps[0] > 1:
        reps = [0] + reps
    elif not reps:
        reps = [0]

    # dedup near-identical reps by REGION signature (pot+board), NOT whole frame:
    # consecutive poker states differ only in small regions, so a whole-frame
    # compare would wrongly merge distinct states (the v1 failure mode).
    def region_diff(i, j):
        return sum(float(np.mean(np.abs(roi_snaps[i][k] - roi_snaps[j][k]))) for k in ROIS)

    merged = []
    for r in reps:
        if merged and region_diff(r, merged[-1]) < merge_diff:
            merged[-1] = r
        else:
            merged.append(r)

    boundaries = [times[i] for i in range(n) if whole[i] > WHOLE_FRAME_BOUNDARY]
    return {
        "keyframe_times": [times[r] for r in merged],
        "n_bursts": len(bursts),
        "boundary_hints": boundaries,
    }


def extract_full(video, out_dir: Path, kf_times, preview_w):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdir = out_dir / "preview"
    pdir.mkdir(parents=True, exist_ok=True)
    c = av.open(video)
    s = c.streams.video[0]
    saved = []
    for t in kf_times:
        c.seek(int(t / s.time_base), stream=s)
        frame = None
        for frame in c.decode(s):
            if float(frame.pts * s.time_base) >= t:
                break
        if frame is None:
            continue
        img = frame.to_ndarray(format="bgr24")
        base = f"kf_t{t:07.2f}s"
        cv2.imwrite(str(out_dir / f"{base}.png"), img)
        H, W = img.shape[:2]
        prev = cv2.resize(img, (preview_w, int(H * preview_w / W)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(pdir / f"{base}.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 82])
        saved.append(str(out_dir / f"{base}.png"))
    c.close()
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--out", default="cv_lab/hand01/auto")
    ap.add_argument("--thumb-width", type=int, default=240)
    ap.add_argument("--preview-width", type=int, default=1100)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--anchor", action="store_true",
                    help="use landmark (green-coin) anchoring for ROIs -> edge/scale tolerant")
    args = ap.parse_args()

    times, whole, roi_series, roi_snaps = analyze(args.video, args.start, args.end,
                                                  args.fps, args.thumb_width, args.anchor)
    res = select(times, whole, roi_series, roi_snaps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timeline.json").write_text(json.dumps({
        "roi_thresh": ROI_THRESH,
        "n_samples": len(times),
        "n_keyframes": len(res["keyframe_times"]),
        "keyframe_times": res["keyframe_times"],
        "boundary_hints": res["boundary_hints"],
        "timeline": [{"t": times[i], "whole": round(whole[i], 2),
                      **{k: round(roi_series[k][i], 2) for k in ROIS}} for i in range(len(times))],
    }, indent=2))
    print(f"samples={len(times)}  bursts={res['n_bursts']}  keyframes={len(res['keyframe_times'])}")
    print("keyframe_times:", res["keyframe_times"])
    print("boundary_hints:", res["boundary_hints"])
    if not args.no_extract:
        saved = extract_full(args.video, out_dir, res["keyframe_times"], args.preview_width)
        print(f"extracted {len(saved)} keyframes -> {out_dir}")
