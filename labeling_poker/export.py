from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

from .config import CLASSES, DEFAULT_DB_PATH, DEFAULT_IMAGES_DIR
from .db import connect, get_annotations, get_status, sync_files


def image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and data[12:16] == b"VP8X":
        return (1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little"))
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"unsupported image format: {path}")
    offset = 2
    sof_markers = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        length = int.from_bytes(data[offset:offset + 2], "big")
        if marker in sof_markers:
            return (int.from_bytes(data[offset + 5:offset + 7], "big"), int.from_bytes(data[offset + 3:offset + 5], "big"))
        offset += length
    raise ValueError(f"could not determine image size: {path}")


def split_ids(ids: list[str]) -> dict[str, list[str]]:
    total = len(ids)
    if total == 0:
        return {"train": [], "val": [], "test": []}
    train_end = max(1, int(total * 0.8)) if total > 1 else 1
    val_end = max(train_end, int(total * 0.9))
    return {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}


def yolo_line(box: dict, width: int, height: int) -> str:
    x1, y1 = max(0.0, min(width, float(box["x1"]))), max(0.0, min(height, float(box["y1"])))
    x2, y2 = max(0.0, min(width, float(box["x2"]))), max(0.0, min(height, float(box["y2"])))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    class_id = CLASSES.index(box["class"])
    return f"{class_id} {(x1 + x2) / 2 / width:.8f} {(y1 + y2) / 2 / height:.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"


def write_dataset(db_path: Path | str, images_dir: Path | str, output_dir: Path | str) -> dict[str, int]:
    images_dir, output_dir = Path(images_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        sync_files(connection, images_dir)
        rows = connection.execute("SELECT id, path FROM files ORDER BY id").fetchall()
        included = [row["id"] for row in rows if get_status(connection, row["id"]) in {"labeled", "clean"}]
        splits = split_ids(included)
        row_by_id = {row["id"]: row for row in rows}
        for split, split_ids_value in splits.items():
            image_out = output_dir / "images" / split
            label_out = output_dir / "labels" / split
            image_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)
            for file_id in split_ids_value:
                source = images_dir / row_by_id[file_id]["path"]
                shutil.copy2(source, image_out / source.name)
                destination_label = label_out / f"{source.stem}.txt"
                if get_status(connection, file_id) == "clean":
                    destination_label.write_text("")
                    continue
                width, height = image_size(source)
                lines = [yolo_line(box, width, height) for box in get_annotations(connection, file_id)]
                destination_label.write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml_lines = [f"path: {output_dir.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    return {split: len(ids) for split, ids in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled poker boxes as a YOLO detection dataset")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(write_dataset(args.db, args.images, args.out))


if __name__ == "__main__":
    main()

