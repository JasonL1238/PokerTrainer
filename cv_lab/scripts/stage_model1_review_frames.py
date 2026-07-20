"""Stage fresh frames into a Model-1-only review queue.

Separate from the cv_weak_cards (Model 2 / card rank) queue: this samples raw
frames from the videos, adds them to the labeling app's image pool as plain
undecided images (no annotations), and lists them in a dedicated priority
queue. In the labeling app, opening this queue makes the "bootstrap" prelabel
come from Model 1 (the 8-class region detector) instead of the card model --
every region box (face_card unlabeled, card_back, dealer_button, pot_text,
stack_text, action_pill, active_turn_indicator, bet_text) gets a prediction to
review. Press Save to approve (-> becomes Model 1 training data via export.py)
or the duplicate/exclude shortcut to reject a frame's predictions outright.

Samples on an odd-second grid (1s, 3s, 5s, ...) so it doesn't just re-surface
the same frames already staged by sweep_and_flag_weak_cards.py's even-second grid.

  python cv_lab/scripts/stage_model1_review_frames.py \
      --videos "data/videos/*.mov" --per-video 40 --queue-name model1_review
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import av
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_autolabel_v1" / "images"
DEFAULT_PRIORITY_DIR = REPO_ROOT / "labeling_poker" / "priority"


def _slug(idx: int) -> str:
    return f"v{idx:02d}"


def _grab_frame(container, stream, t: float):
    container.seek(int(t / stream.time_base), stream=stream)
    frame = None
    for frame in container.decode(stream):
        if float(frame.pts * stream.time_base) >= t:
            break
    return None if frame is None else frame.to_ndarray(format="bgr24")


def stage_video(video_path: Path, idx: int, per_video: int, images_dir: Path, seed: int) -> list[str]:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    rng = random.Random(seed + idx)
    candidate_times = [t for t in [x * 2 + 1 for x in range(int(duration // 2))]]
    rng.shuffle(candidate_times)
    chosen = sorted(candidate_times[:per_video])

    file_ids = []
    slug = _slug(idx)
    for t in chosen:
        img = _grab_frame(container, stream, t)
        if img is None:
            continue
        file_id = f"m1review_{slug}_t{t:08.2f}"
        split = "val" if (hash(file_id) % 100) < 20 else "train"
        out_path = images_dir / split / f"{file_id}.jpg"
        if out_path.is_file():
            file_ids.append(file_id)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        file_ids.append(file_id)
    container.close()
    print(f"[{video_path.name}] staged {len(file_ids)} frames as {_slug(idx)}")
    return file_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default=str(REPO_ROOT / "data" / "videos" / "*.mov"))
    parser.add_argument("--per-video", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--start-index", type=int, default=0,
                        help="first video slug index (vNN); use when staging a subset after earlier videos")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--queue-name", default="model1_review")
    parser.add_argument("--priority-dir", default=str(DEFAULT_PRIORITY_DIR))
    parser.add_argument("--append", action="store_true", help="append to an existing queue file instead of overwriting")
    args = parser.parse_args()

    videos = sorted(Path(p) for p in glob.glob(args.videos))
    if not videos:
        raise SystemExit(f"no videos matched {args.videos!r}")

    images_dir = Path(args.images_dir)
    all_ids: list[str] = []
    for i, video in enumerate(videos):
        all_ids.extend(stage_video(video, args.start_index + i, args.per_video, images_dir, args.seed))

    priority_dir = Path(args.priority_dir)
    priority_dir.mkdir(parents=True, exist_ok=True)
    queue_path = priority_dir / f"{args.queue_name}.txt"
    existing = []
    if args.append and queue_path.is_file():
        existing = [line.strip() for line in queue_path.read_text().splitlines() if line.strip()]
    merged = existing + [i for i in all_ids if i not in existing]
    queue_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

    print(f"\nstaged {len(all_ids)} new frames ({len(merged)} total in queue)")
    print(f"images written under: {images_dir}")
    print(f"queue: {queue_path}")
    print(f"\nreview: python -m labeling_poker.app   then open /?queue={args.queue_name}")


if __name__ == "__main__":
    main()
