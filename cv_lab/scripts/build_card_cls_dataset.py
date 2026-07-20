"""Build a cropped card rank/suit CLASSIFICATION dataset (Model 2).

The card classifier is trained on tight crops of single cards, not full frames.
Every crop source we have already carries a rank/suit label + a box, so we can
assemble the whole dataset with zero re-labeling:

  * labeling SQLite   -- hand-verified face_card boxes (absolute pixel coords)
  * local YOLO datasets -- 53-class detection labels (normalized coords)
  * GGPoker (Roboflow)  -- optional; excluded from defaults (domain mismatch)

Output is the Ultralytics classification layout (folder-per-class):

  <out>/train/<label>/*.png
  <out>/val/<label>/*.png

All labels are canonicalized through labeling_poker.config.normalize_card_label,
so the three different naming schemes ("As" / "AS"|"10C" / "10c") converge to the
same 52 canonical classes ("As", "Kd", "Tc", ...). "joker" is dropped.

Splits are decided at the SOURCE-IMAGE level (every crop from one frame lands in
the same split) so train/val never leak crops of the same physical card.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(__file__))

from labeling_poker.config import CARD_RANKS, CARD_SUITS, normalize_card_label  # noqa: E402
from labeling_poker.db import connect, get_annotations, get_status  # noqa: E402
from build_mixed_card_dataset import _find_image, _parse_names as _parse_names_block  # noqa: E402

# 52 canonical classes, matching normalize_card_label's output form.
CANONICAL_CLASSES = [f"{r}{s}" for r in CARD_RANKS for s in CARD_SUITS]


def _parse_names(data_yaml: Path) -> list[str]:
    """Read a YOLO data.yaml names list, tolerating both formats in play here.

    Local datasets use the block form (``names:\\n  0: 10C``); Roboflow/GGPoker
    exports use an inline list (``names: ['10c', ...]``). Try the block parser
    first, then fall back to parsing an inline ``names: [...]`` list.
    """
    try:
        return _parse_names_block(data_yaml)
    except ValueError:
        pass
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("names:") and "[" in stripped:
            inner = stripped[stripped.index("[") + 1: stripped.rindex("]")]
            return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    raise ValueError(f"no names: block or inline list found in {data_yaml}")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_SQLITE_DB = REPO_ROOT / "labeling_poker" / "data" / "labels.sqlite3"
DEFAULT_SQLITE_IMAGES = REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_autolabel_v1" / "images"
DEFAULT_DETECTION_DATASETS = [
    REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_autolabel_v1",
    REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_autolabel_v2",
    REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_autolabel_v3",
    REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_card_changes_v1",
]
DEFAULT_GGPOKER_ZIP = REPO_ROOT / "cv_lab" / "datasets" / "GGPoker Playing Cards.v5i.yolov12.zip"
DEFAULT_GGPOKER_DIR = REPO_ROOT / "cv_lab" / "datasets" / "GGPoker_cards_v5"


def _canon(raw: object) -> str | None:
    """Any source label -> canonical rank+suit ("As"), or None (joker/invalid)."""
    try:
        return normalize_card_label("face_card", raw)
    except ValueError:
        return None


def _pad_crop(img, x0: float, y0: float, x1: float, y1: float, pad: float):
    """Crop [x0,y0,x1,y1] (pixels) expanded by `pad` fraction each side, clamped."""
    h, w = img.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    xi0, yi0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
    xi1, yi1 = min(int(round(x1)), w), min(int(round(y1)), h)
    if xi1 <= xi0 or yi1 <= yi0:
        return None
    return img[yi0:yi1, xi0:xi1]


class Writer:
    """Writes crops into <out>/<split>/<label>/, tracking per-class counts."""

    def __init__(self, out: Path):
        self.out = out
        self.counts: dict[tuple[str, str], int] = defaultdict(int)  # (split,label)->n

    def write(self, crop, split: str, label: str, stem: str) -> bool:
        if crop is None or crop.size == 0:
            return False
        dst_dir = self.out / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        idx = self.counts[(split, label)]
        dst = dst_dir / f"{stem}.png"
        if dst.exists():  # keep unique
            dst = dst_dir / f"{stem}_{idx}.png"
        ok = cv2.imwrite(str(dst), crop)
        if ok:
            self.counts[(split, label)] += 1
        return ok


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def add_sqlite(writer: Writer, db_path: Path, images_dir: Path, pad: float) -> int:
    if not db_path.exists():
        print(f"  [sqlite] SKIP: db not found at {db_path}")
        return 0
    n = 0
    with connect(db_path) as conn:
        rows = conn.execute("SELECT id, path FROM files ORDER BY id").fetchall()
        for row in rows:
            if get_status(conn, row["id"]) not in {"labeled", "clean"}:
                continue
            rel = row["path"]
            split = "train" if rel.startswith("train/") else ("val" if rel.startswith("val/") else "train")
            img_path = images_dir / rel
            if not img_path.exists():
                continue
            img = None
            for anno in get_annotations(conn, row["id"]):
                if anno["class"] != "face_card":
                    continue
                label = _canon(anno["label"])
                if label is None:
                    continue
                if img is None:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        break
                crop = _pad_crop(img, anno["x1"], anno["y1"], anno["x2"], anno["y2"], pad)
                if writer.write(crop, split, label, f"sqlite__{row['id']}"):
                    n += 1
    print(f"  [sqlite] {n} crops")
    return n


def _add_yolo_detection_root(writer: Writer, root: Path, names: list[str], tag: str,
                             split_map: dict[str, str], pad: float) -> int:
    """Crop every box in a YOLO detection dataset (images/<split> + labels/<split>)."""
    n = 0
    for src_split, dst_split in split_map.items():
        labels_dir = root / "labels" / src_split
        images_dir = root / "images" / src_split
        if not labels_dir.exists():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            img_path = _find_image(images_dir, label_file.stem)
            if img_path is None:
                continue
            lines = [ln.split() for ln in label_file.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for bi, parts in enumerate(lines):
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(parts[1]) * w, float(parts[2]) * h,
                                      float(parts[3]) * w, float(parts[4]) * h)
                except (ValueError, IndexError):
                    continue
                if cid < 0 or cid >= len(names):
                    continue
                label = _canon(names[cid])
                if label is None:
                    continue
                crop = _pad_crop(img, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, pad)
                if writer.write(crop, dst_split, label, f"{tag}__{label_file.stem}_{bi}"):
                    n += 1
    return n


def add_detection_datasets(writer: Writer, roots: list[Path], pad: float) -> int:
    total = 0
    for root in roots:
        yaml = root / "data.yaml"
        if not yaml.exists():
            print(f"  [det:{root.name}] SKIP: no data.yaml")
            continue
        names = _parse_names(yaml)
        n = _add_yolo_detection_root(writer, root, names, root.name,
                                     {"train": "train", "val": "val"}, pad)
        print(f"  [det:{root.name}] {n} crops")
        total += n
    return total


def add_ggpoker(writer: Writer, zip_path: Path, extract_dir: Path, pad: float) -> int:
    if not extract_dir.exists():
        if not zip_path.exists():
            print(f"  [ggpoker] SKIP: neither {extract_dir.name} nor zip present")
            return 0
        print(f"  [ggpoker] extracting {zip_path.name} -> {extract_dir.name}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    yaml = extract_dir / "data.yaml"
    if not yaml.exists():
        print(f"  [ggpoker] SKIP: no data.yaml under {extract_dir}")
        return 0
    names = _parse_names(yaml)
    # Roboflow layout: <split>/images + <split>/labels (splits train/valid/test).
    n = 0
    for src_split, dst_split in {"train": "train", "valid": "val", "test": "val"}.items():
        labels_dir = extract_dir / src_split / "labels"
        images_dir = extract_dir / src_split / "images"
        if not labels_dir.exists():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            img_path = _find_image(images_dir, label_file.stem)
            if img_path is None:
                continue
            lines = [ln.split() for ln in label_file.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for bi, parts in enumerate(lines):
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(parts[1]) * w, float(parts[2]) * h,
                                      float(parts[3]) * w, float(parts[4]) * h)
                except (ValueError, IndexError):
                    continue
                if cid < 0 or cid >= len(names):
                    continue
                label = _canon(names[cid])
                if label is None:
                    continue
                crop = _pad_crop(img, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, pad)
                if writer.write(crop, dst_split, label, f"gg__{label_file.stem}_{bi}"):
                    n += 1
    print(f"  [ggpoker] {n} crops")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "cv_lab" / "datasets" / "cards_cls_v5_20260717"))
    parser.add_argument("--sources", nargs="*", default=["sqlite", "detection"],
                        choices=["sqlite", "detection", "ggpoker"])
    parser.add_argument("--pad", type=float, default=0.12, help="bbox expansion each side before crop")
    parser.add_argument("--sqlite-db", default=str(DEFAULT_SQLITE_DB))
    parser.add_argument("--sqlite-images", default=str(DEFAULT_SQLITE_IMAGES))
    parser.add_argument("--ggpoker-zip", default=str(DEFAULT_GGPOKER_ZIP))
    parser.add_argument("--ggpoker-dir", default=str(DEFAULT_GGPOKER_DIR))
    parser.add_argument("--clean", action="store_true", help="wipe the output dir first")
    args = parser.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    writer = Writer(out)
    print(f"out={out}  pad={args.pad}  sources={args.sources}")
    total = 0
    if "sqlite" in args.sources:
        total += add_sqlite(writer, Path(args.sqlite_db), Path(args.sqlite_images), args.pad)
    if "detection" in args.sources:
        total += add_detection_datasets(writer, DEFAULT_DETECTION_DATASETS, args.pad)
    if "ggpoker" in args.sources:
        total += add_ggpoker(writer, Path(args.ggpoker_zip), Path(args.ggpoker_dir), args.pad)

    # --- per-class coverage report (worst first) ---
    per_class: dict[str, dict[str, int]] = {c: {"train": 0, "val": 0} for c in CANONICAL_CLASSES}
    for (split, label), n in writer.counts.items():
        per_class.setdefault(label, {"train": 0, "val": 0})[split] = n
    print(f"\ntotal crops={total}  classes with data={sum(1 for c in per_class.values() if c['train'] + c['val'] > 0)}/52")
    print("per-class (train/val), worst first:")
    for label in sorted(per_class, key=lambda c: per_class[c]["train"] + per_class[c]["val"]):
        tr, va = per_class[label]["train"], per_class[label]["val"]
        flag = "  <-- EMPTY" if tr + va == 0 else ("  <-- no val" if va == 0 else "")
        print(f"  {label:>3}: {tr:>4} / {va:<4}{flag}")
    missing = [c for c in CANONICAL_CLASSES if per_class[c]["train"] + per_class[c]["val"] == 0]
    if missing:
        print(f"\nWARNING: {len(missing)} class(es) have ZERO crops: {', '.join(missing)}")
    no_val = [c for c in CANONICAL_CLASSES if per_class[c]["val"] == 0 and per_class[c]["train"] > 0]
    if no_val:
        print(f"NOTE: {len(no_val)} class(es) have no val crops (val accuracy blind there): {', '.join(no_val)}")


if __name__ == "__main__":
    main()
