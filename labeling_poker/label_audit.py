"""Heuristic audit of poker labels / predictions for likely mistakes.

Produces priority queues for re-review in the labeler Browse menu:

- `labeled_sus` — saved labels that look wrong or incomplete
- `unlabeled_sus` — undecided frames whose cached two-model predictions look wrong
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .config import CARD_LABEL_RE, DEFAULT_DB_PATH, DEFAULT_PRIORITY_DIR
from .db import connect, get_annotations


SEVERITY = {
    "duplicate_card_labels": 100,
    "invalid_card_label": 95,
    "overlap_face_card": 90,
    "near_overlap_face_card": 85,
    "overlap_same_class": 80,
    "empty_prediction": 75,
    "multi_dealer_button": 70,
    "model_card_box_disagreement": 68,
    "model_card_disagreement": 65,
    "mixed_face_labels": 55,
    "face_cards_missing_rank": 50,
    "too_many_face_cards": 45,
    "multi_pot_text": 40,
    "multi_turn": 35,
    "weird_face_aspect": 25,
    "tiny_box": 20,
    "huge_box": 20,
}

# High-recall labeled_sus: prefer false positives over missing bad rank/suit labels.
# Includes face geometry that often means the wrong crop/identity was saved.
CARD_LABEL_REASON_CODES = frozenset({
    "duplicate_card_labels",
    "invalid_card_label",
    "model_card_disagreement",
    "model_card_box_disagreement",
    "mixed_face_labels",
    "face_cards_missing_rank",
    "overlap_face_card",
    "near_overlap_face_card",
    "weird_face_aspect",
    "too_many_face_cards",
})

# Unlabeled sus reviews model predictions before save; empty/missing cards matter too.
UNLABELED_SUS_REASON_CODES = CARD_LABEL_REASON_CODES | frozenset({
    "empty_prediction",
})

FACE_OVERLAP_HARD = 0.5
FACE_OVERLAP_SOFT = 0.25
MODEL_BOX_IOU = 0.3


def _iou(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def audit_boxes(boxes: list[dict]) -> list[str]:
    """Return human-readable reason codes for one frame's saved boxes."""
    if not boxes:
        return ["labeled_empty"]

    reasons: list[str] = []
    by_class = Counter(box["class"] for box in boxes)
    face = [box for box in boxes if box["class"] == "face_card"]
    labeled_face = [box for box in face if box.get("label")]
    unlabeled_face = [box for box in face if not box.get("label")]
    card_labels = [box["label"] for box in labeled_face]

    dup_labels = sorted(label for label, count in Counter(card_labels).items() if count > 1)
    if dup_labels:
        reasons.append(f"duplicate_card_labels:{','.join(dup_labels)}")

    bad_labels = sorted({
        str(label)
        for label in card_labels
        if not CARD_LABEL_RE.match(str(label))
    })
    if bad_labels:
        reasons.append(f"invalid_card_label:{','.join(bad_labels)}")

    if unlabeled_face and labeled_face:
        reasons.append(
            f"mixed_face_labels:unlabeled={len(unlabeled_face)}/labeled={len(labeled_face)}"
        )
    elif unlabeled_face:
        reasons.append(f"face_cards_missing_rank:{len(unlabeled_face)}")

    if by_class.get("dealer_button", 0) > 1:
        reasons.append(f"multi_dealer_button:{by_class['dealer_button']}")
    if by_class.get("pot_text", 0) > 2:
        reasons.append(f"multi_pot_text:{by_class['pot_text']}")
    if by_class.get("active_turn_indicator", 0) > 2:
        reasons.append(f"multi_turn:{by_class['active_turn_indicator']}")
    if by_class.get("face_card", 0) > 12:
        reasons.append(f"too_many_face_cards:{by_class['face_card']}")

    hard_face_overlap = 0.0
    soft_face_overlap = 0.0
    same_class_overlap = None
    for index, box_a in enumerate(boxes):
        for box_b in boxes[index + 1 :]:
            if box_a["class"] != box_b["class"]:
                continue
            overlap = _iou(box_a, box_b)
            if box_a["class"] == "face_card":
                if overlap >= FACE_OVERLAP_HARD:
                    hard_face_overlap = max(hard_face_overlap, overlap)
                elif overlap >= FACE_OVERLAP_SOFT:
                    soft_face_overlap = max(soft_face_overlap, overlap)
            elif overlap >= FACE_OVERLAP_HARD and same_class_overlap is None:
                same_class_overlap = (box_a["class"], overlap)
    if hard_face_overlap > 0:
        reasons.append(f"overlap_face_card:{hard_face_overlap:.2f}")
    elif soft_face_overlap > 0:
        reasons.append(f"near_overlap_face_card:{soft_face_overlap:.2f}")
    if same_class_overlap is not None:
        class_name, overlap = same_class_overlap
        reasons.append(f"overlap_same_class:{class_name}:{overlap:.2f}")

    max_x = max(box["x2"] for box in boxes)
    max_y = max(box["y2"] for box in boxes)
    frame_area = max(max_x, 1.0) * max(max_y, 1.0)
    for box in boxes:
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        if width < 8 or height < 8:
            reasons.append(f"tiny_box:{box['class']}:{width:.1f}x{height:.1f}")
            break
        if width * height > 0.35 * frame_area and box["class"] in {
            "face_card",
            "card_back",
            "dealer_button",
            "action_pill",
        }:
            reasons.append(f"huge_box:{box['class']}:{width:.0f}x{height:.0f}")
            break

    for box in face:
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        if height <= 0:
            continue
        aspect = width / height
        if aspect < 0.28 or aspect > 1.8:
            reasons.append(f"weird_face_aspect:{aspect:.2f}")
            break

    return reasons


def reason_code(reason: str) -> str:
    return reason.split(":", 1)[0]


def severity_for(reasons: list[str]) -> int:
    return sum(SEVERITY.get(reason_code(reason), 10) for reason in reasons)


def load_two_model_cache(cache_path: Path | None) -> dict[str, list[dict]]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    return {
        file_id: boxes
        for file_id, boxes in items.items()
        if isinstance(file_id, str) and isinstance(boxes, list)
    }


def model_card_disagreement(human_boxes: list[dict], model_boxes: list[dict]) -> str | None:
    human = sorted(box["label"] for box in human_boxes if box.get("class") == "face_card" and box.get("label"))
    model = sorted(
        box.get("label")
        for box in model_boxes
        if box.get("class") == "face_card" and box.get("label")
    )
    if not human or not model:
        return None
    if Counter(human) == Counter(model):
        return None
    return f"model_card_disagreement:human={','.join(human)}|model={','.join(model)}"


def model_card_box_disagreement(human_boxes: list[dict], model_boxes: list[dict]) -> str | None:
    """Flag IoU-matched face boxes whose rank/suit disagree (high recall)."""
    human_faces = [
        box for box in human_boxes
        if box.get("class") == "face_card" and box.get("label")
    ]
    model_faces = [
        box for box in model_boxes
        if box.get("class") == "face_card" and box.get("label")
    ]
    if not human_faces or not model_faces:
        return None

    mismatches: list[str] = []
    used_model: set[int] = set()
    for human in human_faces:
        best_idx = -1
        best_iou = 0.0
        for index, model in enumerate(model_faces):
            if index in used_model:
                continue
            overlap = _iou(human, model)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = index
        if best_idx < 0 or best_iou < MODEL_BOX_IOU:
            continue
        used_model.add(best_idx)
        model_label = model_faces[best_idx]["label"]
        if model_label != human["label"]:
            mismatches.append(f"{human['label']}->{model_label}@{best_iou:.2f}")
    if not mismatches:
        return None
    return "model_card_box_disagreement:" + ",".join(mismatches[:6])


def audit_labeled_frames(
    connection: sqlite3.Connection,
    *,
    two_model_cache: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Return sorted suspect records for every labeled frame that looks wrong."""
    cache = two_model_cache or {}
    rows = connection.execute(
        "SELECT file_id, updated_at FROM status WHERE status = 'labeled' ORDER BY file_id"
    ).fetchall()
    suspects: list[dict] = []
    for row in rows:
        file_id = row["file_id"]
        boxes = get_annotations(connection, file_id)
        reasons = audit_boxes(boxes)
        model_boxes = cache.get(file_id, [])
        disagreement = model_card_disagreement(boxes, model_boxes)
        if disagreement:
            reasons.append(disagreement)
        box_disagreement = model_card_box_disagreement(boxes, model_boxes)
        if box_disagreement:
            reasons.append(box_disagreement)
        if not reasons:
            continue
        suspects.append(
            {
                "id": file_id,
                "updated_at": row["updated_at"],
                "reasons": reasons,
                "severity": severity_for(reasons),
                "box_count": len(boxes),
            }
        )
    suspects.sort(key=lambda item: (-item["severity"], item["id"]))
    return suspects


def filter_suspects_by_reasons(suspects: list[dict], allowed_codes: frozenset[str]) -> list[dict]:
    """Keep frames whose reasons intersect the allowed reason-code set."""
    filtered: list[dict] = []
    for item in suspects:
        kept = [reason for reason in item["reasons"] if reason_code(reason) in allowed_codes]
        if not kept:
            continue
        filtered.append({
            **item,
            "reasons": kept,
            "severity": severity_for(kept),
        })
    filtered.sort(key=lambda item: (-item["severity"], item["id"]))
    return filtered


def filter_card_label_suspects(suspects: list[dict]) -> list[dict]:
    """Keep frames with possible face_card rank/suit problems (high recall)."""
    return filter_suspects_by_reasons(suspects, CARD_LABEL_REASON_CODES)


def audit_unlabeled_frames(
    connection: sqlite3.Connection,
    *,
    two_model_cache: dict[str, list[dict]],
) -> list[dict]:
    """Return sorted suspect records for undecided frames with shaky predictions."""
    rows = connection.execute(
        "SELECT f.id AS file_id FROM files f "
        "LEFT JOIN status s ON s.file_id = f.id "
        "WHERE s.status IS NULL "
        "ORDER BY f.id"
    ).fetchall()
    suspects: list[dict] = []
    for row in rows:
        file_id = row["file_id"]
        if file_id not in two_model_cache:
            # Only flag frames we actually tried to auto-label; otherwise the queue
            # would swallow every undecided image that has not been cached yet.
            continue
        boxes = two_model_cache.get(file_id) or []
        reasons = ["empty_prediction"] if not boxes else [
            reason for reason in audit_boxes(boxes) if reason_code(reason) != "labeled_empty"
        ]
        if not reasons:
            continue
        suspects.append(
            {
                "id": file_id,
                "updated_at": None,
                "reasons": reasons,
                "severity": severity_for(reasons),
                "box_count": len(boxes),
            }
        )
    suspects.sort(key=lambda item: (-item["severity"], item["id"]))
    return suspects


def write_suspect_queue(
    suspects: list[dict],
    *,
    priority_dir: Path,
    queue_name: str = "labeled_sus",
) -> tuple[Path, Path]:
    priority_dir.mkdir(parents=True, exist_ok=True)
    queue_path = priority_dir / f"{queue_name}.txt"
    report_path = priority_dir / f"{queue_name}_report.json"
    queue_path.write_text("\n".join(item["id"] for item in suspects) + ("\n" if suspects else ""), encoding="utf-8")
    report = {
        "queue": queue_name,
        "count": len(suspects),
        "reason_counts": dict(
            Counter(reason_code(reason) for item in suspects for reason in item["reasons"])
        ),
        "items": {
            item["id"]: {
                "reasons": item["reasons"],
                "severity": item["severity"],
                "box_count": item["box_count"],
                "updated_at": item["updated_at"],
            }
            for item in suspects
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return queue_path, report_path


def load_suspect_report(priority_dir: Path, queue_name: str = "labeled_sus") -> dict:
    report_path = priority_dir / f"{queue_name}_report.json"
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    return items if isinstance(items, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flag suspicious poker labels or auto-label predictions")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument(
        "--target",
        choices=("labeled", "unlabeled"),
        default="labeled",
        help="labeled = audit saved SQLite labels; unlabeled = audit cached two-model predictions on undecided frames",
    )
    parser.add_argument(
        "--queue-name",
        default=None,
        help="priority queue basename (default: labeled_sus or unlabeled_sus from --target)",
    )
    parser.add_argument(
        "--two-model-cache",
        type=Path,
        default=DEFAULT_DB_PATH.parent / "predictions" / "two_model_validation.json",
    )
    parser.add_argument("--no-model-compare", action="store_true")
    parser.add_argument(
        "--all-reasons",
        action="store_true",
        help="Include non-card geometry/layout issues too (default: high-recall card rank/suit + face geometry)",
    )
    args = parser.parse_args(argv)
    queue_name = args.queue_name or ("unlabeled_sus" if args.target == "unlabeled" else "labeled_sus")

    cache = {} if args.no_model_compare else load_two_model_cache(args.two_model_cache)
    with connect(args.db) as connection:
        if args.target == "unlabeled":
            if args.no_model_compare or not cache:
                raise SystemExit("unlabeled audit requires a two-model prediction cache")
            suspects = audit_unlabeled_frames(connection, two_model_cache=cache)
            if not args.all_reasons:
                suspects = filter_suspects_by_reasons(suspects, UNLABELED_SUS_REASON_CODES)
        else:
            suspects = audit_labeled_frames(connection, two_model_cache=cache)
            if not args.all_reasons:
                suspects = filter_card_label_suspects(suspects)
    queue_path, report_path = write_suspect_queue(
        suspects, priority_dir=args.priority_dir, queue_name=queue_name
    )
    reason_counts: dict[str, int] = defaultdict(int)
    for item in suspects:
        for reason in item["reasons"]:
            reason_counts[reason_code(reason)] += 1
    print(f"suspect frames: {len(suspects)}")
    print(f"queue: {queue_path}")
    print(f"report: {report_path}")
    for code, count in sorted(reason_counts.items(), key=lambda pair: (-pair[1], pair[0])):
        print(f"  {code}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
