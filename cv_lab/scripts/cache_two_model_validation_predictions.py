"""Apply the two retrained models to every queued frame for human approval.

Predictions are written to a JSON sidecar, never to SQLite. The labeling app
serves this cache for ``/?queue=two_model_validation``; a reviewer must still
save an image before its boxes or card names become training data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from labeling_poker.config import DEFAULT_DB_PATH, EXISTING_DATASET_IMAGES_DIR
from labeling_poker.db import connect, get_file
from labeling_poker.inference import predict_two_model


def _queue_ids(path: Path) -> list[str]:
    return list(dict.fromkeys(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--images", type=Path, default=EXISTING_DATASET_IMAGES_DIR)
    parser.add_argument("--queue", type=Path, default=REPO_ROOT / "labeling_poker" / "priority" / "two_model_validation.txt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "labeling_poker" / "data" / "predictions" / "two_model_validation.json")
    parser.add_argument("--resume", action="store_true", help="reuse successful cached items from --out")
    parser.add_argument("--write-every", type=int, default=25)
    args = parser.parse_args()

    items: dict[str, list[dict]] = {}
    if args.resume and args.out.is_file():
        try:
            items = json.loads(args.out.read_text(encoding="utf-8")).get("items", {})
        except (OSError, json.JSONDecodeError):
            pass
    payload = {
        "source": "two_model_review",
        "region_model": os.environ.get("POKER_LABELER_REGION_MODEL", ""),
        "card_model": os.environ.get("POKER_LABELER_CARD_MODEL", ""),
        "items": items,
        "errors": {},
    }
    ids = _queue_ids(args.queue)
    images_dir = args.images.resolve()
    with connect(args.db) as connection:
        for index, file_id in enumerate(ids, start=1):
            if file_id in items:
                continue
            row = get_file(connection, file_id)
            if row is None:
                payload["errors"][file_id] = "file not found in label database"
                continue
            image_path = (images_dir / row["path"]).resolve()
            if images_dir not in image_path.parents or not image_path.is_file():
                payload["errors"][file_id] = "image file not found"
                continue
            try:
                items[file_id] = predict_two_model(image_path)
            except Exception as exc:
                payload["errors"][file_id] = str(exc)
            if index % args.write_every == 0:
                _write(args.out, payload)
                print(f"{index}/{len(ids)} frames cached ({len(items)} successful)")
    _write(args.out, payload)
    print(f"cached {len(items)}/{len(ids)} queued frames -> {args.out}")


if __name__ == "__main__":
    main()
