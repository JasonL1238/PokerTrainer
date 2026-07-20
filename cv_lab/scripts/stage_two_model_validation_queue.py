"""Stage undecided frames for human review with the complete two-model bootstrap.

The labeler runs Model 1 (regions) and Model 2 (card names) only when a queued
frame is opened. Predictions remain in browser memory until the reviewer saves;
therefore this script never turns model output into training labels on its own.

After starting the labeler with the new model paths, open:
  http://127.0.0.1:5055/?queue=two_model_validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from labeling_poker.config import DEFAULT_DB_PATH, EXISTING_DATASET_IMAGES_DIR
from labeling_poker.db import connect, get_status, sync_files

def stage_queue(db_path: Path, images_dir: Path, queue_path: Path) -> int:
    images_dir = images_dir.resolve()
    with connect(db_path) as connection:
        sync_files(connection, images_dir)
        ids = [
            row["id"]
            for row in connection.execute("SELECT id, path FROM files ORDER BY id")
            if get_status(connection, row["id"]) == "undecided"
            and (images_dir / row["path"]).is_file()
        ]
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--images", type=Path, default=EXISTING_DATASET_IMAGES_DIR)
    parser.add_argument("--queue", type=Path, default=REPO_ROOT / "labeling_poker" / "priority" / "two_model_validation.txt")
    args = parser.parse_args()
    print(f"staged {stage_queue(args.db, args.images, args.queue)} undecided frames -> {args.queue}")


if __name__ == "__main__":
    main()
