"""Extract a handful of representative frames per selected hand for the
manual validation gallery (hand-history text vs. real video frames).

Not a permanent pipeline script -- a one-off tool for building the artifact
viewer. Requires the poker-cv conda env (cv2 + av).

    python cv_lab/scripts/pipeline/extract_gallery_frames.py --out-dir <scratch dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]

# DEVELOPMENT recordings ONLY. This table used to map v01-v04 and v06-v07 as
# well -- the validation and locked-test recordings -- and this script decoded
# them with av/cv2 to build a gallery, in direct violation of the corpus split
# (they are NEVER-OPEN until their gates run). Those entries are deleted, not
# commented out: any tool that needs a gallery frame from a held-out recording
# is a tool that must not run before the locked evaluation does. Do not add
# them back.
VIDEO_GLOB = {
    "v00": "*clubwpt_session_01*",
    "v05": "*2026-07-11*12.45.27*",
}

SELECTION = {
    "v00": [7, 8, 9],
    "v05": [1, 29, 32, 44, 45],
}

MAX_FRAMES_PER_HAND = 6


def hand_targets(hand: dict) -> list[tuple[float, str]]:
    """(time_s, label) pairs to grab for one hand, capped at MAX_FRAMES_PER_HAND."""
    targets = [(hand["t_start"], "start")]
    for s in hand["streets"][1:]:
        targets.append((s["time_s"], s["street"]))
    targets.append((hand["t_end"], "end"))
    targets.append((min(hand["t_end"] + 2, hand["t_end"] + 2), "result"))
    # dedupe by time, keep order, cap
    seen = set()
    out = []
    for t, label in targets:
        if t in seen:
            continue
        seen.add(t)
        out.append((t, label))
    if len(out) > MAX_FRAMES_PER_HAND:
        # keep start, end, result always; thin the middle
        head, tail = out[:1], out[-2:]
        mid = out[1:-2]
        keep = MAX_FRAMES_PER_HAND - 3
        step = max(1, len(mid) // max(keep, 1))
        mid = mid[::step][:keep]
        out = head + mid + tail
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for v, hand_numbers in SELECTION.items():
        tl = json.load(open(REPO_ROOT / f"cv_lab/results/two_model_timeline_{v}.json"))
        hands_by_num = {h["hand_number"]: h for h in tl["hands"]}
        video_path = glob.glob(str(REPO_ROOT / "data/videos" / VIDEO_GLOB[v]))[0]

        # gather all targets across this video's selected hands
        per_hand_targets = {}
        all_targets = []
        for n in hand_numbers:
            h = hands_by_num[n]
            tgts = hand_targets(h)
            per_hand_targets[n] = tgts
            all_targets.extend(t for t, _ in tgts)
        all_targets = sorted(set(all_targets))

        print(f"== {v}: {len(all_targets)} frames from {Path(video_path).name} ==")
        grabbed: dict[float, str] = {}
        container = av.open(video_path)
        stream = container.streams.video[0]
        for t in all_targets:
            offset = int(max(t - 3, 0) / stream.time_base)
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            found = None
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                if frame.time >= t:
                    found = frame
                    break
            if found is None:
                print("  MISS", t)
                continue
            img = found.to_ndarray(format="bgr24")
            h_img, w_img = img.shape[:2]
            scale = 900.0 / w_img
            img = cv2.resize(img, (900, int(h_img * scale)))
            fname = f"{v}_t{t:.0f}.jpg"
            cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 78])
            grabbed[t] = fname
        container.close()

        for n in hand_numbers:
            frames = [{"time_s": t, "label": label, "file": grabbed.get(t)}
                      for t, label in per_hand_targets[n] if t in grabbed]
            manifest.append({"video": v, "hand_number": n, "frames": frames})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote manifest with {len(manifest)} hands to {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
