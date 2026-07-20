"""Assemble an anti-drift ("rehearsal") card dataset for ClubWPT fine-tuning.

Fine-tuning the card detector on ClubWPT frames alone risks catastrophic
forgetting: cards that never appear in the ClubWPT captures (e.g. 3S) lose their
gradient signal and degrade. The fix is rehearsal -- mix a slice of the original
card datasets back in so every one of the 53 classes keeps getting reinforced.

This script builds a merged YOLO dataset directory by symlinking (or copying):
  * ALL ClubWPT train images/labels               (the new domain)
  * a stratified sample of the rehearsal datasets  (guards the old cards)
  * ALL val images/labels from both sides          (so drift is measured, not hidden)

The output is a self-contained dataset dir with its own data.yaml, ready to hand
to train_yolov12_cards.py.

Class lists MUST be identical across every input dataset (same names, same order)
-- YOLO label files store integer class IDs, so a mismatch silently mislabels
every box. The script refuses to run if the names disagree.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REHEARSAL = [
    "cv_lab/datasets/yolo_cards_autolabel_v1",
    "cv_lab/datasets/yolo_cards_autolabel_v2",
    "cv_lab/datasets/yolo_cards_autolabel_v3",
    "cv_lab/datasets/yolo_cards_card_changes_v1",
]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _parse_names(data_yaml: Path) -> list[str]:
    """Read the ordered class-name list from a YOLO data.yaml (names: 0: x block)."""
    names: dict[int, str] = {}
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not raw[:1].isspace() and stripped:  # dedented -> names block ended
                break
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                if key.isdigit():
                    names[int(key)] = val.strip().strip("'\"")
    if not names:
        raise ValueError(f"no names: block found in {data_yaml}")
    return [names[i] for i in range(len(names))]


def _label_classes(label_path: Path) -> set[int]:
    """Return the set of class IDs referenced by a YOLO label .txt (empty if none)."""
    classes: set[int] = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        cand = images_dir / f"{stem}{suf}"
        if cand.exists():
            return cand
    return None


def _collect_split(root: Path, split: str) -> list[dict]:
    """List (image, label, classes) triples for a split, keyed off label files."""
    labels_dir = root / "labels" / split
    images_dir = root / "images" / split
    rows: list[dict] = []
    if not labels_dir.exists():
        return rows
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        rows.append({
            "image": image_path,
            "label": label_path,
            "classes": _label_classes(label_path),
        })
    return rows


def _select_rehearsal(pool: list[dict], *, n_classes: int, rehearsal_target: int,
                      min_per_class: int, rng: random.Random) -> list[dict]:
    """Pick rehearsal images: first guarantee per-class coverage, then top up to target.

    Coverage wins over the ratio -- if hitting min_per_class needs more images than
    rehearsal_target, we take more. A rare card kept in the model beats a tidy ratio.
    """
    selected: list[dict] = []
    selected_ids: set[int] = set()  # id() of chosen rows, to dedupe
    covered: dict[int, int] = defaultdict(int)

    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        for cls in row["classes"]:
            by_class[cls].append(row)

    # Coverage pass: rarest classes first so scarce cards claim their images before
    # common ones exhaust the pool.
    for cls in sorted(range(n_classes), key=lambda c: len(by_class.get(c, []))):
        candidates = by_class.get(cls, [])
        rng.shuffle(candidates)
        for row in candidates:
            if covered[cls] >= min_per_class:
                break
            if id(row) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(id(row))
            for c in row["classes"]:
                covered[c] += 1

    # Ratio top-up: add random remaining images until we reach the target size.
    remaining = [row for row in pool if id(row) not in selected_ids]
    rng.shuffle(remaining)
    while len(selected) < rehearsal_target and remaining:
        row = remaining.pop()
        selected.append(row)
        selected_ids.add(id(row))
        for c in row["classes"]:
            covered[c] += 1

    return selected


def _link(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _place(rows: list[dict], out: Path, split: str, tag: str, *, copy: bool) -> None:
    """Symlink/copy rows into out/{images,labels}/{split} with collision-proof names."""
    for row in rows:
        stem = f"{tag}__{row['label'].stem}"
        _link(row["image"], out / "images" / split / f"{stem}{row['image'].suffix}", copy=copy)
        _link(row["label"], out / "labels" / split / f"{stem}.txt", copy=copy)


def _write_data_yaml(out: Path, names: list[str]) -> None:
    lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clubwpt", required=True,
                        help="ClubWPT card dataset dir (images/labels in the 53-class scheme)")
    parser.add_argument("--rehearsal", nargs="*", default=DEFAULT_REHEARSAL,
                        help="original card dataset dirs to sample for rehearsal")
    parser.add_argument("--out", default="cv_lab/datasets/yolo_cards_clubwpt_mixed",
                        help="output mixed-dataset dir to create")
    parser.add_argument("--rehearsal-frac", type=float, default=0.25,
                        help="rehearsal images as a fraction of the final train set (0-1)")
    parser.add_argument("--min-per-class", type=int, default=8,
                        help="guaranteed rehearsal images containing each class, pool permitting")
    parser.add_argument("--seed", type=int, default=1238, help="deterministic sampling seed")
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of symlinking (use if training host can't follow symlinks)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clubwpt = Path(args.clubwpt)
    rehearsal_dirs = [Path(p) for p in args.rehearsal]
    out = Path(args.out)

    # --- validate class lists are identical everywhere ---
    ref_names = _parse_names(clubwpt / "data.yaml")
    n_classes = len(ref_names)
    for d in rehearsal_dirs:
        names = _parse_names(d / "data.yaml")
        if names != ref_names:
            raise SystemExit(
                f"class mismatch: {d}/data.yaml differs from {clubwpt}/data.yaml.\n"
                "Every dataset must share the identical names list in the same order, "
                "or the integer class IDs in the label files mean different cards."
            )

    # --- collect splits ---
    club_train = _collect_split(clubwpt, "train")
    club_val = _collect_split(clubwpt, "val")
    reh_train_pool: list[dict] = []
    reh_val: list[dict] = []
    for d in rehearsal_dirs:
        reh_train_pool += _collect_split(d, "train")
        reh_val += _collect_split(d, "val")

    if not club_train:
        raise SystemExit(f"no ClubWPT train images found under {clubwpt}/labels/train")

    # rehearsal_target from R/(R+C) = frac  ->  R = C*frac/(1-frac)
    frac = min(max(args.rehearsal_frac, 0.0), 0.95)
    rehearsal_target = round(len(club_train) * frac / (1.0 - frac)) if frac > 0 else 0
    selected_reh = _select_rehearsal(
        reh_train_pool, n_classes=n_classes, rehearsal_target=rehearsal_target,
        min_per_class=args.min_per_class, rng=rng,
    )

    # --- build output tree fresh ---
    if out.exists():
        shutil.rmtree(out)
    _place(club_train, out, "train", "clubwpt", copy=args.copy)
    # tag rehearsal rows by their source dataset so filenames stay traceable
    for row in selected_reh:
        # row["label"] path is .../<dataset>/labels/train/<stem>.txt
        tag = row["label"].parents[2].name
        stem = f"reh_{tag}__{row['label'].stem}"
        _link(row["image"], out / "images" / "train" / f"{stem}{row['image'].suffix}", copy=args.copy)
        _link(row["label"], out / "labels" / "train" / f"{stem}.txt", copy=args.copy)
    _place(club_val, out, "val", "clubwpt", copy=args.copy)
    for row in reh_val:
        tag = row["label"].parents[2].name
        stem = f"reh_{tag}__{row['label'].stem}"
        _link(row["image"], out / "images" / "val" / f"{stem}{row['image'].suffix}", copy=args.copy)
        _link(row["label"], out / "labels" / "val" / f"{stem}.txt", copy=args.copy)

    _write_data_yaml(out, ref_names)

    # --- per-class coverage report so you can see which cards are thin ---
    train_coverage: dict[int, int] = defaultdict(int)
    for row in club_train + selected_reh:
        for c in row["classes"]:
            train_coverage[c] += 1
    thin = [(ref_names[c], train_coverage.get(c, 0))
            for c in range(n_classes) if train_coverage.get(c, 0) < args.min_per_class]

    total_train = len(club_train) + len(selected_reh)
    print(f"out={out}")
    print(f"clubwpt_train={len(club_train)} rehearsal_train={len(selected_reh)} "
          f"total_train={total_train} rehearsal_share={len(selected_reh)/max(total_train,1):.0%}")
    print(f"clubwpt_val={len(club_val)} rehearsal_val={len(reh_val)} "
          f"total_val={len(club_val) + len(reh_val)}")
    print(f"rehearsal_target={rehearsal_target} min_per_class={args.min_per_class} seed={args.seed}")
    if thin:
        print(f"WARNING: {len(thin)} class(es) below min_per_class in train "
              "(pool too small -- acquire or rehearse more of these):")
        print("  " + ", ".join(f"{name}={n}" for name, n in thin))
    else:
        print(f"all {n_classes} classes have >= {args.min_per_class} train images")
    print(f"data_yaml={out / 'data.yaml'}")


if __name__ == "__main__":
    main()
