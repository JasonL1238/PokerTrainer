"""Apply simple class-label corrections to an autolabeled YOLO card dataset.

This is for the common case where YOLO found the card box correctly but chose
the wrong card class. Edit corrections.csv, filling correct_label for mistakes
or action=delete for false positives, then run this script.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def _load_classes(dataset: Path) -> dict[str, int]:
    names = (dataset / "classes.txt").read_text(encoding="utf-8").splitlines()
    return {name.upper(): i for i, name in enumerate(names)}


def _normalize_label(label: str) -> str:
    label = label.strip().upper()
    if not label:
        return ""
    if len(label) == 2 and label[0] == "T":
        return "10" + label[1]
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cv_lab/datasets/yolo_cards_autolabel_v3")
    parser.add_argument("--corrections", default="", help="defaults to DATASET/corrections.csv")
    parser.add_argument("--backup-dir", default="", help="defaults to DATASET/labels_autolabel_backup")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    corrections = Path(args.corrections) if args.corrections else dataset / "corrections.csv"
    backup_dir = Path(args.backup_dir) if args.backup_dir else dataset / "labels_autolabel_backup"
    class_ids = _load_classes(dataset)

    by_label: dict[Path, dict[int, dict]] = {}
    with corrections.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_rel = row["label"]
            det_i = int(row["detection_index"])
            by_label.setdefault(dataset / label_rel, {})[det_i] = row

    changed_files = 0
    changed_detections = 0
    deleted_detections = 0
    backup_dir.mkdir(parents=True, exist_ok=True)

    for label_path, rows_by_index in sorted(by_label.items()):
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        lines = label_path.read_text(encoding="utf-8").splitlines()
        out_lines: list[str] = []
        file_changed = False

        backup_path = backup_dir / label_path.relative_to(dataset)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)

        for det_i, line in enumerate(lines):
            row = rows_by_index.get(det_i)
            if row is None:
                out_lines.append(line)
                continue
            action = row.get("action", "").strip().lower()
            correct_label = _normalize_label(row.get("correct_label", ""))
            if action in {"delete", "drop", "remove"}:
                deleted_detections += 1
                file_changed = True
                continue
            if not correct_label or correct_label == _normalize_label(row.get("pred_label", "")):
                out_lines.append(line)
                continue
            if correct_label not in class_ids:
                raise ValueError(
                    f"Unknown correct_label={correct_label!r} in {corrections}. "
                    "Use labels from classes.txt, e.g. AS, 10D, 7H."
                )
            parts = line.split()
            parts[0] = str(class_ids[correct_label])
            out_lines.append(" ".join(parts))
            changed_detections += 1
            file_changed = True

        if file_changed:
            label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            changed_files += 1

    print(f"dataset={dataset}")
    print(f"corrections={corrections}")
    print(f"backup_dir={backup_dir}")
    print(f"changed_files={changed_files}")
    print(f"changed_detections={changed_detections}")
    print(f"deleted_detections={deleted_detections}")


if __name__ == "__main__":
    main()
