"""Mine hard examples from an offline YOLO card-labeling dataset.

This reads existing autolabel review artifacts and emits a prioritized list of
frames that likely need manual attention. It is intentionally read-only with
respect to detections.csv, corrections.csv, and missing_labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


DEFAULT_DATASET = "cv_lab/datasets/yolo_cards_autolabel_v3"
DEFAULT_OUT_DIRNAME = "hard_examples"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def _int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except (TypeError, ValueError):
        return default


def _norm_label(label: str) -> str:
    label = label.strip().upper()
    if len(label) == 2 and label[0] == "T":
        return "10" + label[1]
    return label


def _effective_label(row: dict) -> str:
    return _norm_label(row.get("correct_label", "") or row.get("pred_label", ""))


def _is_active(row: dict) -> bool:
    return row.get("action", "").strip().lower() not in {"delete", "drop", "remove"}


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _zone_for_box(cx: float, cy: float) -> str:
    """Same broad normalized card zones used by build_yolo_card_timeline.py."""
    if 0.40 <= cx <= 0.58 and cy >= 0.64:
        return "hero"
    if 0.32 <= cx <= 0.64 and 0.36 <= cy <= 0.55:
        return "board"
    return "other"


def _frame_sort_key(row: dict) -> tuple[str, float, int, str]:
    return (
        row.get("video", ""),
        _float(row, "time_s", 0.0),
        _int(row, "frame", 0),
        row.get("image", ""),
    )


def _image_stem(rel_image: str) -> str:
    return Path(rel_image).stem


def _review_image_for(dataset: Path, rel_image: str) -> str:
    review = dataset / "review" / f"{_image_stem(rel_image)}.jpg"
    if review.exists():
        return str(review.relative_to(dataset))
    return ""


def _infer_image_size(dataset: Path, frames: list[dict]) -> tuple[int, int] | None:
    for frame in frames:
        image = frame.get("image")
        if not image:
            continue
        img = cv2.imread(str(dataset / image))
        if img is None:
            continue
        height, width = img.shape[:2]
        return width, height
    return None


def _load_frame_rows(dataset: Path, rows: list[dict], missing_rows: list[dict]) -> list[dict]:
    manifest = _read_csv(dataset / "manifest.csv")
    frames: dict[str, dict] = {row["image"]: dict(row) for row in manifest if row.get("image")}

    for row in rows:
        image = row.get("image", "")
        if not image:
            continue
        frames.setdefault(image, {
            "split": row.get("split", ""),
            "video": row.get("video", ""),
            "frame": row.get("frame", ""),
            "time_s": row.get("time_s", ""),
            "image": image,
            "label": row.get("label", ""),
        })

    for row in missing_rows:
        image = row.get("image", "")
        if image:
            frames.setdefault(image, {"image": image})

    return sorted(frames.values(), key=_frame_sort_key)


def _active_rows_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _is_active(row) and row.get("image"):
            by_image[row["image"]].append(row)
    for image in by_image:
        by_image[image].sort(key=lambda row: _int(row, "detection_index", 0))
    return by_image


def _frame_state(
    frame: dict,
    rows: list[dict],
    *,
    image_width: int,
    image_height: int,
    low_conf: float,
    overlap_iou: float,
) -> dict:
    issues: list[str] = []
    details: list[str] = []
    zones = {"hero": [], "board": [], "other": []}
    labels: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    min_conf: float | None = None

    for row in rows:
        label = _effective_label(row)
        if not label:
            continue
        conf = _float(row, "conf", 0.0)
        min_conf = conf if min_conf is None else min(min_conf, conf)
        labels.append(label)
        if conf < low_conf:
            issues.append("low_confidence")
            details.append(f"{label} conf={conf:.3f}")

        x0, y0, x1, y1 = (_float(row, k, 0.0) for k in ("x0", "y0", "x1", "y1"))
        box = (x0, y0, x1, y1)
        boxes.append(box)
        cx = ((x0 + x1) / 2.0) / image_width if image_width else 0.0
        cy = ((y0 + y1) / 2.0) / image_height if image_height else 0.0
        zones[_zone_for_box(cx, cy)].append(label)

    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        issues.append("duplicate_corrected_label")
        details.append("duplicates=" + " ".join(duplicate_labels))

    max_iou = 0.0
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1:]:
            max_iou = max(max_iou, _iou(box_a, box_b))
    if max_iou >= overlap_iou:
        issues.append("overlapping_boxes")
        details.append(f"max_iou={max_iou:.3f}")

    hero_count = len(zones["hero"])
    board_count = len(zones["board"])
    if hero_count not in {0, 2}:
        issues.append("partial_hero_count")
        details.append(f"hero_count={hero_count}")
    if board_count not in {0, 3, 4, 5}:
        issues.append("partial_board_count")
        details.append(f"board_count={board_count}")

    return {
        "image": frame.get("image", ""),
        "split": frame.get("split", ""),
        "video": frame.get("video", ""),
        "frame": frame.get("frame", ""),
        "time_s": frame.get("time_s", ""),
        "detections": len(rows),
        "min_conf": "" if min_conf is None else f"{min_conf:.3f}",
        "hero_count": hero_count,
        "board_count": board_count,
        "other_count": len(zones["other"]),
        "labels": " ".join(labels),
        "signature": (
            tuple(sorted(zones["hero"])),
            tuple(sorted(zones["board"])),
            tuple(sorted(zones["other"])),
        ),
        "issues": sorted(set(issues)),
        "details": details,
    }


def _add_missing_issues(states: list[dict], missing_rows: list[dict]) -> None:
    missing_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in missing_rows:
        if row.get("image"):
            missing_by_image[row["image"]].append(row)

    for state in states:
        rows = missing_by_image.get(state["image"], [])
        if not rows:
            continue
        state["issues"].append("missing_labels_entry")
        notes = []
        for row in rows:
            note = row.get("note") or row.get("reason") or row.get("labels") or "missing label"
            notes.append(note)
        state["details"].append("missing=" + " | ".join(notes))


def _add_state_churn_issues(
    states: list[dict],
    *,
    churn_window_seconds: float,
    churn_change_threshold: int,
) -> None:
    change_indexes: list[int] = []
    for i in range(1, len(states)):
        if states[i]["signature"] != states[i - 1]["signature"]:
            change_indexes.append(i)

    for i in range(1, len(states) - 1):
        if (
            states[i - 1]["signature"] == states[i + 1]["signature"]
            and states[i]["signature"] != states[i - 1]["signature"]
        ):
            states[i]["issues"].append("state_flicker")
            states[i]["details"].append("one-frame A-B-A label state")

    if churn_window_seconds <= 0 or churn_change_threshold <= 0:
        return

    times = [_float(state, "time_s", float(i)) for i, state in enumerate(states)]
    for i, state in enumerate(states):
        start_t = times[i] - churn_window_seconds / 2.0
        end_t = times[i] + churn_window_seconds / 2.0
        changes = sum(1 for idx in change_indexes if start_t <= times[idx] <= end_t)
        if changes >= churn_change_threshold:
            state["issues"].append("state_churn")
            state["details"].append(f"{changes} state changes in {churn_window_seconds:.1f}s")


def _priority(issues: list[str]) -> int:
    weights = {
        "missing_labels_entry": 90,
        "duplicate_corrected_label": 80,
        "overlapping_boxes": 70,
        "state_flicker": 60,
        "state_churn": 50,
        "partial_hero_count": 45,
        "partial_board_count": 40,
        "low_confidence": 30,
    }
    return sum(weights.get(issue, 10) for issue in set(issues))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "priority",
        "image",
        "review_image",
        "split",
        "video",
        "frame",
        "time_s",
        "issue_types",
        "details",
        "detections",
        "min_conf",
        "hero_count",
        "board_count",
        "other_count",
        "labels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, rows: list[dict], *, dataset: Path) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>YOLO card hard examples</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#111;color:#eee}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
        "figure{margin:0;background:#1b1b1b;padding:10px;border:1px solid #333}",
        "img{width:100%;height:auto;display:block}",
        "figcaption{font-size:13px;line-height:1.35;margin-top:8px;color:#ddd}",
        "code{color:#9ee}",
        "</style>",
        "<h1>YOLO card hard examples</h1>",
        f"<p>Dataset: <code>{html.escape(str(dataset.resolve()))}</code></p>",
        "<div class='grid'>",
    ]
    for row in rows:
        review_image = row.get("review_image", "")
        image = review_image or row["image"]
        href = html.escape(str(Path("..") / image))
        caption = html.escape(
            f"priority={row['priority']} t={row['time_s']} issues={row['issue_types']} "
            f"labels={row['labels']} details={row['details']}"
        )
        if review_image:
            parts.append(f"<figure><a href='{href}'><img src='{href}'></a><figcaption>{caption}</figcaption></figure>")
        else:
            parts.append(f"<figure><figcaption><a href='{href}'>{html.escape(row['image'])}</a><br>{caption}</figcaption></figure>")
    parts.append("</div>")
    path.write_text("\n".join(parts), encoding="utf-8")


def mine_hard_examples(
    dataset: Path,
    out_dir: Path,
    *,
    image_width: int = 0,
    image_height: int = 0,
    low_conf: float = 0.35,
    overlap_iou: float = 0.50,
    churn_window_seconds: float = 4.0,
    churn_change_threshold: int = 3,
    write_html: bool = False,
) -> dict:
    detections = _read_csv(dataset / "detections.csv")
    corrections = _read_csv(dataset / "corrections.csv")
    missing_rows = _read_csv(dataset / "missing_labels.csv")
    effective_rows = corrections or detections

    frames = _load_frame_rows(dataset, effective_rows, missing_rows)
    inferred_size = _infer_image_size(dataset, frames)
    if inferred_size is not None and (image_width <= 0 or image_height <= 0):
        image_width, image_height = inferred_size
    by_image = _active_rows_by_image(effective_rows)
    states = [
        _frame_state(
            frame,
            by_image.get(frame.get("image", ""), []),
            image_width=image_width,
            image_height=image_height,
            low_conf=low_conf,
            overlap_iou=overlap_iou,
        )
        for frame in frames
    ]
    _add_missing_issues(states, missing_rows)
    _add_state_churn_issues(
        states,
        churn_window_seconds=churn_window_seconds,
        churn_change_threshold=churn_change_threshold,
    )

    hard_rows: list[dict] = []
    for state in states:
        issues = sorted(set(state["issues"]))
        if not issues:
            continue
        hard_rows.append({
            "priority": _priority(issues),
            "image": state["image"],
            "review_image": _review_image_for(dataset, state["image"]),
            "split": state["split"],
            "video": state["video"],
            "frame": state["frame"],
            "time_s": state["time_s"],
            "issue_types": ";".join(issues),
            "details": " | ".join(state["details"]),
            "detections": state["detections"],
            "min_conf": state["min_conf"],
            "hero_count": state["hero_count"],
            "board_count": state["board_count"],
            "other_count": state["other_count"],
            "labels": state["labels"],
        })

    hard_rows.sort(key=lambda row: (-int(row["priority"]), _float(row, "time_s", 0.0), row["image"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hard_examples.csv"
    json_path = out_dir / "summary.json"
    html_path = out_dir / "index.html"
    _write_csv(csv_path, hard_rows)

    issue_counts = Counter()
    for row in hard_rows:
        issue_counts.update(row["issue_types"].split(";"))
    summary = {
        "dataset": str(dataset),
        "source_rows": {
            "detections": len(detections),
            "corrections": len(corrections),
            "missing_labels": len(missing_rows),
            "effective": "corrections.csv" if corrections else "detections.csv",
        },
        "thresholds": {
            "low_conf": low_conf,
            "overlap_iou": overlap_iou,
            "image_width": image_width,
            "image_height": image_height,
            "churn_window_seconds": churn_window_seconds,
            "churn_change_threshold": churn_change_threshold,
        },
        "frames_scanned": len(states),
        "hard_frames": len(hard_rows),
        "issue_counts": dict(sorted(issue_counts.items())),
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "html": str(html_path) if write_html else "",
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if write_html:
        _write_html(html_path, hard_rows, dataset=dataset)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", default="", help="defaults to DATASET/hard_examples")
    parser.add_argument("--image-width", type=int, default=0, help="0 infers from the first dataset image")
    parser.add_argument("--image-height", type=int, default=0, help="0 infers from the first dataset image")
    parser.add_argument("--low-conf", type=float, default=0.35)
    parser.add_argument("--overlap-iou", type=float, default=0.50)
    parser.add_argument("--churn-window-seconds", type=float, default=4.0)
    parser.add_argument("--churn-change-threshold", type=int, default=3)
    parser.add_argument("--write-html", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else dataset / DEFAULT_OUT_DIRNAME
    summary = mine_hard_examples(
        dataset,
        out_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        low_conf=args.low_conf,
        overlap_iou=args.overlap_iou,
        churn_window_seconds=args.churn_window_seconds,
        churn_change_threshold=args.churn_change_threshold,
        write_html=args.write_html,
    )
    print(f"dataset={dataset}")
    print(f"out_dir={out_dir}")
    print(f"frames_scanned={summary['frames_scanned']}")
    print(f"hard_frames={summary['hard_frames']}")
    print(f"csv={summary['outputs']['csv']}")
    print(f"json={summary['outputs']['json']}")
    if args.write_html:
        print(f"html={summary['outputs']['html']}")


if __name__ == "__main__":
    main()
