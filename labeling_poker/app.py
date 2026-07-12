from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from .config import (
    CARD_LABEL_CLASS,
    CARD_RANKS,
    CARD_SUIT_SYMBOLS,
    CARD_SUITS,
    CLASSES,
    CLASS_COLORS,
    DEFAULT_DB_PATH,
    DEFAULT_IMAGES_DIR,
    DEFAULT_PRIORITY_DIR,
    EXISTING_DATASET_IMAGES_DIR,
    normalize_card_label,
)
from .db import (
    connect,
    file_ids,
    get_annotations,
    get_file,
    get_status,
    next_undecided,
    progress,
    save_annotations,
    seek,
    sync_files,
)
from .inference import predict_cards


def create_app(db_path: Path | str = DEFAULT_DB_PATH, images_dir: Path | str = DEFAULT_IMAGES_DIR) -> Flask:
    app = Flask(__name__)
    db_path = Path(db_path)
    images_dir = Path(images_dir).resolve()
    if images_dir == DEFAULT_IMAGES_DIR.resolve() and not any(images_dir.rglob("*")) and EXISTING_DATASET_IMAGES_DIR.is_dir():
        images_dir = EXISTING_DATASET_IMAGES_DIR.resolve()
    priority_dir = DEFAULT_PRIORITY_DIR

    def refresh() -> None:
        with connect(db_path) as connection:
            sync_files(connection, images_dir)

    def item_payload(file_id: str) -> dict | None:
        with connect(db_path) as connection:
            row = get_file(connection, file_id)
            if row is None:
                return None
            return {
                "id": row["id"],
                "path": row["path"],
                "image_url": f"/image/{row['id']}",
                "status": get_status(connection, row["id"]),
                "boxes": get_annotations(connection, row["id"]),
            }

    @app.get("/")
    def index():
        refresh()
        return render_template(
            "index.html",
            classes=CLASSES,
            colors=CLASS_COLORS,
            card_label_class=CARD_LABEL_CLASS,
            card_ranks=CARD_RANKS,
            card_suits=CARD_SUITS,
            card_suit_symbols=CARD_SUIT_SYMBOLS,
        )

    @app.get("/image/<file_id>")
    def image(file_id: str):
        refresh()
        with connect(db_path) as connection:
            row = get_file(connection, file_id)
        if row is None:
            return jsonify({"error": "image not found"}), 404
        path = (images_dir / row["path"]).resolve()
        if images_dir not in path.parents or not path.is_file():
            return jsonify({"error": "image file not found"}), 404
        return send_file(path)

    @app.get("/api/next")
    def api_next():
        refresh()
        queue_name = request.args.get("queue", "")
        priority_ids: list[str] = []
        if queue_name:
            queue_path = priority_dir / f"{queue_name}.txt"
            if queue_path.is_file():
                priority_ids = [line.strip() for line in queue_path.read_text().splitlines() if line.strip()]
        with connect(db_path) as connection:
            file_id = next_undecided(connection, priority_ids)
        return jsonify({"item": item_payload(file_id) if file_id else None})

    @app.get("/api/item/<file_id>")
    def api_item(file_id: str):
        refresh()
        payload = item_payload(file_id)
        return jsonify(payload or {"error": "image not found"}), 200 if payload else 404

    @app.get("/api/seek")
    def api_seek():
        refresh()
        direction = request.args.get("dir")
        if direction not in {"prev", "next"}:
            return jsonify({"error": "dir must be prev or next"}), 400
        with connect(db_path) as connection:
            target = seek(connection, request.args.get("id"), direction)
        return jsonify({"item": item_payload(target) if target else None})

    @app.post("/api/annotate")
    def api_annotate():
        payload = request.get_json(silent=True) or {}
        file_id = str(payload.get("id", ""))
        status_value = payload.get("status")
        boxes = payload.get("boxes", [])
        if status_value not in {"labeled", "clean", "duplicate"} or not isinstance(boxes, list):
            return jsonify({"error": "id, status=labeled|clean|duplicate, and boxes are required"}), 400
        with connect(db_path) as connection:
            if get_file(connection, file_id) is None:
                return jsonify({"error": "image not found"}), 404
            clean_boxes = []
            for box in boxes:
                try:
                    class_name = str(box["class"])
                    values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
                except (KeyError, TypeError, ValueError):
                    return jsonify({"error": "invalid box"}), 400
                if class_name not in CLASSES or values[2] <= values[0] or values[3] <= values[1]:
                    return jsonify({"error": "invalid class or box geometry"}), 400
                try:
                    label = normalize_card_label(class_name, box.get("label"))
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
                clean_boxes.append({"class": class_name, "label": label, **dict(zip(("x1", "y1", "x2", "y2"), values))})
            if status_value in {"clean", "duplicate"} and clean_boxes:
                return jsonify({"error": "clean or duplicate images cannot have boxes"}), 400
            save_annotations(connection, file_id, status_value, clean_boxes)
        return jsonify(item_payload(file_id))

    @app.get("/api/progress")
    def api_progress():
        refresh()
        with connect(db_path) as connection:
            return jsonify(progress(connection))

    @app.get("/api/bootstrap/<file_id>")
    def api_bootstrap(file_id: str):
        with connect(db_path) as connection:
            row = get_file(connection, file_id)
        if row is None:
            return jsonify({"error": "image not found"}), 404
        image_path = (images_dir / row["path"]).resolve()
        try:
            boxes = predict_cards(image_path)
        except Exception as exc:
            app.logger.exception("YOLO bootstrap failed for %s", file_id)
            return jsonify({"error": f"YOLO bootstrap failed: {exc}"}), 503
        return jsonify({"id": file_id, "boxes": boxes, "source": "yolov12", "model": "cv_lab/models/best (4).pt"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Poker table YOLO bounding-box labeling tool")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    create_app(args.db, args.images).run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
