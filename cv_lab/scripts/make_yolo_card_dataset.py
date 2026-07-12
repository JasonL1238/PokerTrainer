"""Create a YOLO-only card dataset from completed-session videos.

This is an offline/post-session data-prep tool. It samples saved video files,
pre-labels full frames with the current YOLOv12 card detector, and writes a
standard YOLO dataset plus an HTML gallery for manual correction/review.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from pathlib import Path

import av
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_yolo_cards import (  # noqa: E402
    DEFAULT_YOLOV12_VENDOR,
    REPO_ROOT,
    _draw_text,
    _draw_xyxy,
    _load_yolo_class,
    _resolve_vendor_path,
)


def _model_names(model) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[i]) for i in range(max(names) + 1)]
    return [str(name) for name in names]


def _iou(a, b) -> float:
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


def _dedupe_rows(rows: list[dict], *, iou_threshold: float) -> list[dict]:
    if iou_threshold <= 0:
        return rows
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: item["conf"], reverse=True):
        if any(_iou(row["xyxy"], kept_row["xyxy"]) >= iou_threshold for kept_row in kept):
            continue
        kept.append(row)
    return kept


def _predict_rows(
    model,
    image,
    *,
    conf: float,
    imgsz: int,
    dedupe_iou: float,
    classes: set[int] | None = None,
) -> list[dict]:
    result = model.predict(image, conf=conf, iou=0.5, imgsz=imgsz, verbose=False)[0]
    rows: list[dict] = []
    h, w = image.shape[:2]
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if classes is not None and cls_id not in classes:
            continue
        label = str(model.names[cls_id])
        if label.lower() == "joker":
            continue
        x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
        x0 = min(max(x0, 0.0), float(w - 1))
        x1 = min(max(x1, 0.0), float(w - 1))
        y0 = min(max(y0, 0.0), float(h - 1))
        y1 = min(max(y1, 0.0), float(h - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append({
            "class_id": cls_id,
            "label": label,
            "conf": float(box.conf[0]),
            "xyxy": (x0, y0, x1, y1),
        })
    rows = sorted(rows, key=lambda row: row["conf"], reverse=True)
    return _dedupe_rows(rows, iou_threshold=dedupe_iou)


def _yolo_line(row: dict, *, width: int, height: int) -> str:
    x0, y0, x1, y1 = row["xyxy"]
    xc = ((x0 + x1) / 2.0) / width
    yc = ((y0 + y1) / 2.0) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return f"{row['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _annotate(image, rows: list[dict], *, t: float):
    out = image.copy()
    for row in rows:
        label = f"{row['det_index']}:{row['label']} {row['conf']:.2f}"
        _draw_xyxy(out, row["xyxy"], (255, 255, 0), label, 2)
    summary = " ".join(f"{r['det_index']}:{r['label']}:{r['conf']:.2f}" for r in rows[:12]) or "no detections"
    _draw_text(out, f"t={t:.2f}s yolo={summary}", 22, 36, (255, 255, 255), 0.72)
    return out


def _state_signature(rows: list[dict], *, mode: str) -> tuple:
    if mode == "count":
        return (len(rows),)
    if mode == "labels":
        return tuple(sorted(row["label"] for row in rows))
    if mode == "labels_xy":
        parts = []
        for row in rows:
            x0, y0, x1, y1 = row["xyxy"]
            cx = round((x0 + x1) / 80.0)
            cy = round((y0 + y1) / 80.0)
            parts.append((row["label"], cx, cy))
        return tuple(sorted(parts))
    raise ValueError(f"Unsupported state mode: {mode}")


def _write_data_yaml(out_dir: Path, names: list[str]) -> None:
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name!r}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def _write_review_index(review_dir: Path, rows: list[dict], *, dataset_dir: Path) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>YOLO card autolabel review</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#111;color:#eee}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}",
        "figure{margin:0;background:#1b1b1b;padding:10px;border:1px solid #333}",
        "img{width:100%;height:auto;display:block}",
        "figcaption{font-size:13px;line-height:1.35;margin-top:8px;color:#ddd}",
        "code{color:#9ee}",
        "</style>",
        "<h1>YOLO card autolabel review</h1>",
        "<p>These are current-model pre-labels. Correct the matching YOLO txt files before training.</p>",
        f"<p>Dataset: <code>{html.escape(str(dataset_dir.resolve()))}</code></p>",
        "<div class='grid'>",
    ]
    for row in rows:
        src = html.escape(row["review_image"])
        caption = html.escape(
            f"{row['split']} {row['image']} t={row['time_s']} "
            f"detections={row['detections']} min_conf={row['min_conf']} labels={row['labels']}"
        )
        parts.append(f"<figure><a href='{src}'><img src='{src}'></a><figcaption>{caption}</figcaption></figure>")
    parts.append("</div>")
    (review_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _write_readme(out_dir: Path, *, weights: str, videos: list[str]) -> None:
    text = f"""# YOLO Card Dataset

This dataset was pre-labeled by the current YOLOv12 model. It is intended for
manual correction first, then continued YOLO training.

Source weights:

```text
{weights}
```

Source videos:

```text
{chr(10).join(videos)}
```

Manual workload:

1. Open `review/index.html` to spot bad labels quickly.
2. Correct boxes/classes in `images/train`, `images/val`, `labels/train`, and `labels/val` with a YOLO annotation tool.
3. Keep labels in YOLO txt format.
4. Train with `data.yaml`.

The pre-labels are useful for speed, but training on uncorrected model guesses
will mostly reinforce the same mistakes.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", dest="videos", required=True,
                        help="completed-session video; pass multiple times for multiple videos")
    parser.add_argument("--weights", default="cv_lab/models/best (4).pt")
    parser.add_argument("--out", default="cv_lab/datasets/yolo_cards_autolabel_v1")
    parser.add_argument("--stride", type=int, default=120, help="sample every N decoded frames")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dedupe-iou", type=float, default=0.65,
                        help="suppress lower-confidence overlapping boxes across classes")
    parser.add_argument("--review-max", type=int, default=400)
    parser.add_argument("--review-width", type=int, default=1400)
    parser.add_argument("--val-every", type=int, default=5, help="every Nth saved frame goes to val")
    parser.add_argument("--save-empty", action="store_true", help="also save sampled frames with zero detections")
    parser.add_argument("--only-card-changes", action="store_true",
                        help="save only when the YOLO-visible card state changes")
    parser.add_argument("--state-mode", choices=["labels", "labels_xy", "count"], default="labels",
                        help="how --only-card-changes decides whether a frame is new")
    parser.add_argument("--min-change-seconds", type=float, default=0.0,
                        help="minimum time between saved card-state changes")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR),
                        help="path to the sunsmarterjie/yolov12 checkout; use empty string for installed ultralytics")
    args = parser.parse_args()

    out_dir = Path(args.out)
    review_dir = out_dir / "review"
    for rel in ("images/train", "images/val", "labels/train", "labels/val", "review"):
        (out_dir / rel).mkdir(parents=True, exist_ok=True)

    args.yolov12_vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(args.yolov12_vendor)
    model = YOLO(args.weights)
    names = _model_names(model)
    _write_data_yaml(out_dir, names)
    _write_readme(out_dir, weights=args.weights, videos=args.videos)

    manifest_rows: list[dict] = []
    review_rows: list[dict] = []
    detection_rows: list[dict] = []
    saved = 0
    last_signature: tuple | None = None
    last_saved_t = -1e9

    for video_i, video in enumerate(args.videos):
        last_signature = None
        last_saved_t = -1e9
        container = av.open(video)
        stream = container.streams.video[0]
        tb = stream.time_base
        frame_i = -1
        for frame in container.decode(stream):
            frame_i += 1
            if frame_i % args.stride:
                continue
            image = frame.to_ndarray(format="bgr24")
            t = float(frame.pts * tb) if frame.pts is not None else 0.0
            rows = _predict_rows(
                model,
                image,
                conf=args.conf,
                imgsz=args.imgsz,
                dedupe_iou=args.dedupe_iou,
            )
            for det_i, row in enumerate(rows):
                row["det_index"] = det_i
            if not rows and not args.save_empty:
                continue
            if args.only_card_changes:
                signature = _state_signature(rows, mode=args.state_mode)
                if signature == last_signature:
                    continue
                if t - last_saved_t < args.min_change_seconds:
                    continue
                last_signature = signature
                last_saved_t = t

            split = "val" if args.val_every and saved % args.val_every == 0 else "train"
            stem = f"v{video_i:02d}_f{frame_i:07d}_t{t:08.2f}"
            image_name = f"{stem}.jpg"
            label_name = f"{stem}.txt"
            h, w = image.shape[:2]
            cv2.imwrite(str(out_dir / "images" / split / image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_text = "\n".join(_yolo_line(row, width=w, height=h) for row in rows)
            (out_dir / "labels" / split / label_name).write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")

            min_conf = min((row["conf"] for row in rows), default=0.0)
            labels = " ".join(row["label"] for row in rows)
            manifest_row = {
                "split": split,
                "video": video,
                "frame": frame_i,
                "time_s": f"{t:.2f}",
                "image": f"images/{split}/{image_name}",
                "label": f"labels/{split}/{label_name}",
                "detections": len(rows),
                "min_conf": f"{min_conf:.3f}",
                "labels": labels,
            }
            manifest_rows.append(manifest_row)
            for row in rows:
                x0, y0, x1, y1 = row["xyxy"]
                detection_rows.append({
                    **manifest_row,
                    "detection_index": row["det_index"],
                    "pred_class_id": row["class_id"],
                    "pred_label": row["label"],
                    "conf": f"{row['conf']:.3f}",
                    "x0": f"{x0:.1f}",
                    "y0": f"{y0:.1f}",
                    "x1": f"{x1:.1f}",
                    "y1": f"{y1:.1f}",
                    "correct_label": "",
                    "action": "",
                })

            if len(review_rows) < args.review_max:
                annotated = _annotate(image, rows, t=t)
                if args.review_width and annotated.shape[1] > args.review_width:
                    scale = args.review_width / annotated.shape[1]
                    annotated = cv2.resize(
                        annotated,
                        (args.review_width, int(round(annotated.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                review_name = f"{stem}.jpg"
                cv2.imwrite(str(review_dir / review_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
                review_rows.append({**manifest_row, "review_image": review_name})

            saved += 1
            if saved >= args.max_frames:
                break
        container.close()
        if saved >= args.max_frames:
            break

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "video", "frame", "time_s", "image", "label", "detections", "min_conf", "labels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    detection_fieldnames = [
        "split", "video", "frame", "time_s", "image", "label", "detection_index",
        "pred_class_id", "pred_label", "conf", "x0", "y0", "x1", "y1", "correct_label", "action",
    ]
    detections_path = out_dir / "detections.csv"
    corrections_path = out_dir / "corrections.csv"
    for csv_path in (detections_path, corrections_path):
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detection_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(detection_rows)
    _write_review_index(review_dir, review_rows, dataset_dir=out_dir)

    print(f"weights={args.weights}")
    print(f"yolov12_vendor={args.yolov12_vendor or '<installed ultralytics>'}")
    print(f"dataset={out_dir}")
    print(f"data_yaml={out_dir / 'data.yaml'}")
    print(f"manifest={manifest_path}")
    print(f"detections={detections_path}")
    print(f"corrections={corrections_path}")
    print(f"review_index={review_dir / 'index.html'}")
    print(f"saved_frames={saved}")
    print(f"review_frames={len(review_rows)}")


if __name__ == "__main__":
    main()
