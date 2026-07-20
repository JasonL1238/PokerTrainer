from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

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
    next_matching,
    progress,
    queue_progress,
    save_annotations,
    seek,
    sync_files,
)
from .inference import predict_cards, predict_regions, predict_two_model
from .label_audit import load_suspect_report, write_suspect_queue


def create_app(db_path: Path | str = DEFAULT_DB_PATH, images_dir: Path | str = DEFAULT_IMAGES_DIR) -> Flask:
    app = Flask(__name__)
    db_path = Path(db_path)
    images_dir = Path(images_dir).resolve()
    if images_dir == DEFAULT_IMAGES_DIR.resolve() and not any(images_dir.rglob("*")) and EXISTING_DATASET_IMAGES_DIR.is_dir():
        images_dir = EXISTING_DATASET_IMAGES_DIR.resolve()
    priority_dir = Path(os.environ.get("POKER_LABELER_PRIORITY_DIR", DEFAULT_PRIORITY_DIR))
    prediction_cache = {"mtime_ns": None, "items": {}}
    suspect_report_cache: dict[str, dict] = {}

    def refresh() -> None:
        with connect(db_path) as connection:
            sync_files(connection, images_dir)

    def suspect_items(queue_name: str = "labeled_sus") -> dict:
        report_path = priority_dir / f"{queue_name}_report.json"
        try:
            mtime_ns = report_path.stat().st_mtime_ns
        except FileNotFoundError:
            return {}
        cached = suspect_report_cache.get(queue_name)
        if cached is None or cached.get("mtime_ns") != mtime_ns:
            suspect_report_cache[queue_name] = {
                "mtime_ns": mtime_ns,
                "items": load_suspect_report(priority_dir, queue_name),
            }
        return suspect_report_cache[queue_name]["items"]

    def item_payload(file_id: str) -> dict | None:
        with connect(db_path) as connection:
            row = get_file(connection, file_id)
            if row is None:
                return None
            payload = {
                "id": row["id"],
                "path": row["path"],
                "image_url": f"/image/{row['id']}",
                "status": get_status(connection, row["id"]),
                "boxes": get_annotations(connection, row["id"]),
            }
        audit = suspect_items("labeled_sus").get(file_id) or suspect_items("unlabeled_sus").get(file_id)
        if isinstance(audit, dict) and audit.get("reasons"):
            payload["sus_reasons"] = audit["reasons"]
            payload["sus_severity"] = audit.get("severity")
        return payload

    @app.get("/")
    def index():
        refresh()
        # One primary labeling setup: always stay on the two-model review queue.
        if "queue" not in request.args:
            params = request.args.to_dict(flat=True)
            params["queue"] = "two_model_validation"
            return redirect(url_for("index", **params))
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

    def load_priority_ids(queue_name: str) -> list[str]:
        # Queue names map directly to filenames; restrict to a safe charset so the
        # parameter cannot traverse outside the priority directory.
        if not queue_name or not re.fullmatch(r"[A-Za-z0-9_-]+", queue_name):
            return []
        queue_path = priority_dir / f"{queue_name}.txt"
        if not queue_path.is_file():
            return []
        return [line.strip() for line in queue_path.read_text().splitlines() if line.strip()]

    def status_filter(default: str) -> str | None:
        value = request.args.get("status", default)
        if value not in {
            "all", "undecided", "labeled", "clean", "duplicate", "labeled_sus", "unlabeled_sus",
        }:
            return None
        return value

    def browse_selection(default_status: str) -> tuple[str | None, list[str], dict[str, bool]]:
        """Resolve Browse status + optional sus-queue priority scoping."""
        selected_status = status_filter(default_status)
        if selected_status is None:
            return None, [], {}
        options = navigation_options()
        priority_ids = load_priority_ids(request.args.get("queue", ""))
        if selected_status == "labeled_sus":
            # Virtual browse view: only currently labeled frames from the audit queue.
            priority_ids = load_priority_ids("labeled_sus")
            selected_status = "labeled"
            options["priority_only"] = True
        elif selected_status == "unlabeled_sus":
            # Virtual browse view: undecided frames whose auto-labels look shaky.
            priority_ids = load_priority_ids("unlabeled_sus")
            selected_status = "undecided"
            options["priority_only"] = True
        return selected_status, priority_ids, options

    def cached_two_model_boxes(file_id: str) -> list[dict] | None:
        """Return precomputed review predictions without ever writing labels."""
        cache_path = Path(os.environ.get(
            "POKER_LABELER_TWO_MODEL_CACHE",
            db_path.parent / "predictions" / "two_model_validation.json",
        ))
        try:
            mtime_ns = cache_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if prediction_cache["mtime_ns"] != mtime_ns:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                prediction_cache["items"] = payload.get("items", {}) if isinstance(payload, dict) else {}
                prediction_cache["mtime_ns"] = mtime_ns
            except (OSError, json.JSONDecodeError):
                return None
        boxes = prediction_cache["items"].get(file_id)
        return boxes if isinstance(boxes, list) else None

    def cached_two_model_items() -> dict[str, list[dict]]:
        """Read the whole prediction sidecar through the mtime-aware cache."""
        cached_two_model_boxes("")
        return {
            file_id: boxes
            for file_id, boxes in prediction_cache["items"].items()
            if isinstance(file_id, str) and isinstance(boxes, list)
        }

    def navigation_options() -> dict[str, bool]:
        """Read the optional chronological queue-review controls from the URL."""
        chronological = request.args.get("order") == "labeled_at"
        return {
            "priority_only": request.args.get("scope") == "queue",
            "order_by_updated_at": chronological,
            "start_with_latest": chronological and request.args.get("start") == "latest",
            "wrap_next": chronological and request.args.get("wrap") == "next",
        }

    @app.get("/manual-review")
    def manual_review():
        """Paginated visual index of pending two-model output and saved work."""
        refresh()
        selected_view = request.args.get("view", "pending")
        if selected_view not in {
            "pending", "labeled", "labeled_sus", "unlabeled_sus", "clean", "duplicate", "all",
        }:
            selected_view = "pending"
        try:
            page = max(1, int(request.args.get("page", "1")))
        except ValueError:
            page = 1
        per_page = 24
        queued_ids = load_priority_ids("two_model_validation")
        queued_set = set(queued_ids)
        sus_ids = load_priority_ids("labeled_sus")
        sus_set = set(sus_ids)
        unlabeled_sus_ids = load_priority_ids("unlabeled_sus")
        unlabeled_sus_set = set(unlabeled_sus_ids)
        sus_report = suspect_items("labeled_sus")
        unlabeled_sus_report = suspect_items("unlabeled_sus")
        cached_items = cached_two_model_items()
        with connect(db_path) as connection:
            rows = connection.execute(
                "SELECT f.id, f.path, s.status, s.updated_at "
                "FROM files f LEFT JOIN status s ON s.file_id = f.id"
            ).fetchall()
            annotations_by_id = {
                row["id"]: get_annotations(connection, row["id"])
                for row in rows
                if row["status"] is not None
            }
        row_by_id = {row["id"]: row for row in rows}
        pending = []
        for file_id in queued_ids:
            row = row_by_id.get(file_id)
            boxes = cached_items.get(file_id)
            if row is None or row["status"] is not None or boxes is None:
                continue
            pending.append({
                "id": file_id, "path": row["path"], "boxes": boxes,
                "kind": "Two-model prediction", "updated_at": None,
                "queue": "two_model_validation",
                "sus_reasons": list((unlabeled_sus_report.get(file_id) or {}).get("reasons") or []),
            })
        unlabeled_sus_items = []
        for file_id in unlabeled_sus_ids:
            row = row_by_id.get(file_id)
            boxes = cached_items.get(file_id)
            if row is None or row["status"] is not None or boxes is None:
                continue
            unlabeled_sus_items.append({
                "id": file_id, "path": row["path"], "boxes": boxes,
                "kind": "Unlabeled sus", "updated_at": None,
                "queue": "two_model_validation",
                "sus_reasons": list((unlabeled_sus_report.get(file_id) or {}).get("reasons") or []),
            })
        unlabeled_sus_items.sort(
            key=lambda item: (
                -int((unlabeled_sus_report.get(item["id"]) or {}).get("severity") or 0),
                item["id"],
            )
        )
        reviewed = [
            {
                "id": row["id"], "path": row["path"],
                "boxes": annotations_by_id[row["id"]], "kind": row["status"],
                "updated_at": row["updated_at"],
                "queue": "two_model_validation" if row["id"] in queued_set else (
                    "labeled_sus" if row["id"] in sus_set else (
                        "unlabeled_sus" if row["id"] in unlabeled_sus_set else ""
                    )
                ),
                "sus_reasons": list((sus_report.get(row["id"]) or {}).get("reasons") or []),
            }
            for row in rows if row["status"] is not None
        ]
        reviewed.sort(key=lambda item: (item["updated_at"] or "", item["id"]), reverse=True)
        labeled_sus_items = [
            item for item in reviewed
            if item["kind"] == "labeled" and item["id"] in sus_set
        ]
        labeled_sus_items.sort(
            key=lambda item: (
                -int((sus_report.get(item["id"]) or {}).get("severity") or 0),
                item["id"],
            )
        )
        if selected_view == "pending":
            items = pending
        elif selected_view == "labeled_sus":
            items = labeled_sus_items
        elif selected_view == "unlabeled_sus":
            items = unlabeled_sus_items
        elif selected_view == "all":
            items = reviewed + pending
        else:
            items = [item for item in reviewed if item["kind"] == selected_view]
        total = len(items)
        start = (page - 1) * per_page
        status_counts = {status: 0 for status in ("labeled", "clean", "duplicate")}
        for item in reviewed:
            status_counts[item["kind"]] += 1
        return render_template(
            "manual_review.html",
            items=items[start:start + per_page], selected_view=selected_view,
            page=page, total=total, has_prev=page > 1,
            has_next=start + per_page < total, colors=CLASS_COLORS,
            counts={
                "pending": len(pending),
                "labeled_sus": len(labeled_sus_items),
                "unlabeled_sus": len(unlabeled_sus_items),
                **status_counts,
            },
        )

    @app.get("/api/next")
    def api_next():
        refresh()
        selected_status, priority_ids, options = browse_selection("undecided")
        if selected_status is None:
            return jsonify({"error": "unknown status filter"}), 400
        with connect(db_path) as connection:
            file_id = next_matching(
                connection,
                selected_status,
                priority_ids,
                request.args.get("id") or None,
                **options,
            )
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
        selected_status, priority_ids, options = browse_selection("all")
        if selected_status is None:
            return jsonify({"error": "unknown status filter"}), 400
        # seek() does not need the initial-load-only option.
        options.pop("start_with_latest", None)
        with connect(db_path) as connection:
            target = seek(connection, request.args.get("id"), direction, selected_status, priority_ids, **options)
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
        queue_name = request.args.get("queue", "")
        status_arg = request.args.get("status", "")
        if status_arg == "labeled_sus" and not queue_name:
            queue_name = "labeled_sus"
        elif status_arg == "unlabeled_sus" and not queue_name:
            queue_name = "unlabeled_sus"
        priority_ids = load_priority_ids(queue_name)
        with connect(db_path) as connection:
            payload = progress(connection)
            if priority_ids:
                payload["queue"] = {"name": queue_name, **queue_progress(connection, priority_ids)}
            if queue_name == "labeled_sus" or status_arg == "labeled_sus":
                sus_ids = set(load_priority_ids("labeled_sus"))
                still_labeled = 0
                for file_id in sus_ids:
                    if get_status(connection, file_id) == "labeled":
                        still_labeled += 1
                payload["labeled_sus"] = still_labeled
            if queue_name == "unlabeled_sus" or status_arg == "unlabeled_sus":
                sus_ids = set(load_priority_ids("unlabeled_sus"))
                still_undecided = 0
                for file_id in sus_ids:
                    if get_status(connection, file_id) == "undecided":
                        still_undecided += 1
                payload["unlabeled_sus"] = still_undecided
        return jsonify(payload)

    @app.post("/api/labeled-sus/done")
    def api_labeled_sus_done():
        """Mark the current labeled_sus queue as reviewed: clear it, keep labels."""
        cleared = load_priority_ids("labeled_sus")
        write_suspect_queue([], priority_dir=priority_dir, queue_name="labeled_sus")
        suspect_report_cache.pop("labeled_sus", None)
        return jsonify({"cleared": len(cleared), "ids": cleared})

    @app.post("/api/unlabeled-sus/done")
    def api_unlabeled_sus_done():
        """Clear the unlabeled_sus queue without writing any labels."""
        cleared = load_priority_ids("unlabeled_sus")
        write_suspect_queue([], priority_dir=priority_dir, queue_name="unlabeled_sus")
        suspect_report_cache.pop("unlabeled_sus", None)
        return jsonify({"cleared": len(cleared), "ids": cleared})

    @app.get("/api/bootstrap/<file_id>")
    def api_bootstrap(file_id: str):
        with connect(db_path) as connection:
            row = get_file(connection, file_id)
        if row is None:
            return jsonify({"error": "image not found"}), 404
        image_path = (images_dir / row["path"]).resolve()
        if images_dir not in image_path.parents or not image_path.is_file():
            return jsonify({"error": "image file not found"}), 404
        source = request.args.get("source")
        try:
            if source == "region":
                boxes = predict_regions(image_path)
            elif source == "two_model":
                boxes = cached_two_model_boxes(file_id)
                if boxes is None:
                    boxes = predict_two_model(image_path)
            else:
                boxes = predict_cards(image_path)
        except Exception as exc:
            app.logger.exception("YOLO bootstrap failed for %s", file_id)
            return jsonify({"error": f"YOLO bootstrap failed: {exc}"}), 503
        if source == "region":
            return jsonify({"id": file_id, "boxes": boxes, "source": "region_spine_v1", "model": "cv_lab/models/region_spine_v1.pt"})
        if source == "two_model":
            return jsonify({
                "id": file_id,
                "boxes": boxes,
                "source": "two_model_review",
                "cached": cached_two_model_boxes(file_id) is not None,
                "region_model": os.environ.get("POKER_LABELER_REGION_MODEL", "cv_lab/models/region_spine_v1.pt"),
                "card_model": os.environ.get("POKER_LABELER_CARD_MODEL", "cv_lab/models/card_cls_v1.pt"),
            })
        return jsonify({"id": file_id, "boxes": boxes, "source": "yolov12", "model": "cv_lab/models/best (4).pt"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Poker table YOLO bounding-box labeling tool")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    # Debug mode exposes the Werkzeug remote-code-execution console; keep it
    # opt-in and never allow it on a non-loopback interface.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    if args.host not in {"127.0.0.1", "localhost"}:
        if debug:
            raise SystemExit("Refusing to run debug mode on a non-loopback host.")
        print(
            f"WARNING: binding to {args.host} exposes this unauthenticated labeling "
            "tool to the network. Use 127.0.0.1 unless you know what you are doing."
        )
    create_app(args.db, args.images).run(host=args.host, port=args.port, debug=debug)


if __name__ == "__main__":
    main()
