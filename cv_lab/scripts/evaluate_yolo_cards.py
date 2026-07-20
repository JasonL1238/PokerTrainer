"""Evaluate a YOLO card detector against completed-session ClubWPT frames.

This is an offline/post-session diagnostic. It does not capture live tables and
does not provide current-hand advice.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from collections import Counter
from pathlib import Path

import av
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import read_table
import read_cards
import read_hero


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLOV12_VENDOR = REPO_ROOT.parent / "cv-backend" / "vendor" / "yolov12"


def _default_vendor_candidates() -> list[Path]:
    return [
        DEFAULT_YOLOV12_VENDOR,
        REPO_ROOT.parent / "YoloCardDetectTest" / "cv-backend" / "vendor" / "yolov12",
        REPO_ROOT.parent / "BlackJack_Trainer" / "cv-backend" / "vendor" / "yolov12",
    ]


def _resolve_vendor_path(vendor_path: str) -> str:
    if vendor_path:
        vendor = Path(vendor_path).expanduser().resolve()
        if vendor.exists():
            return str(vendor)
        if vendor_path == str(DEFAULT_YOLOV12_VENDOR):
            for candidate in _default_vendor_candidates():
                if candidate.exists():
                    return str(candidate.resolve())
        raise FileNotFoundError(
            f"YOLOv12 vendor path does not exist: {vendor}. "
            "Pass --yolov12-vendor /path/to/sunsmarterjie/yolov12."
        )
    return ""


def _load_yolo_class(vendor_path: str):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".yolo_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_config"))
    # numpy>=2 removed several legacy aliases the vendored ultralytics fork still
    # uses (e.g. np.trapz -> np.trapezoid). Shim them so the fork runs under
    # numpy 2 without editing the vendored tree or pinning the whole env down.
    import numpy as _np
    if not hasattr(_np, "trapz"):
        _np.trapz = _np.trapezoid  # type: ignore[attr-defined]
    if vendor_path:
        sys.path.insert(0, vendor_path)
    from ultralytics import YOLO
    return YOLO


def _norm_label(label: str) -> str | None:
    if label.lower() == "joker":
        return None
    rank = label[:-1].replace("10", "T")
    suit = label[-1:].lower()
    return f"{rank}{suit}"


def _predict_boxes(model, image, *, conf: float, imgsz: int) -> list[dict]:
    result = model.predict(image, conf=conf, iou=0.5, imgsz=imgsz, verbose=False)[0]
    out: list[dict] = []
    for box in result.boxes:
        label = _norm_label(model.names[int(box.cls[0])])
        if label is None:
            continue
        x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
        out.append({
            "label": label,
            "conf": float(box.conf[0]),
            "xyxy": (x0, y0, x1, y1),
        })
    return sorted(out, key=lambda row: row["conf"], reverse=True)


def _predict_cards(model, image, *, conf: float, imgsz: int) -> list[tuple[str, float]]:
    return [(row["label"], row["conf"]) for row in _predict_boxes(model, image, conf=conf, imgsz=imgsz)]


def _crop(img, box):
    x0, x1, y0, y1 = box
    return img[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]


def _xyxy_from_roi(map_roi, roi):
    x0, x1, y0, y1 = map_roi(roi)
    return max(x0, 0), max(y0, 0), max(x1, 0), max(y1, 0)


def _draw_text(img, text, x, y, color=(255, 255, 255), scale=0.58, thick=2):
    cv2.putText(img, str(text), (int(x) + 1, int(y) + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(text), (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_xyxy(img, xyxy, color, label="", thickness=2):
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if label:
        _draw_text(img, label, x0, max(y0 - 7, 18), color)


def _offset_boxes(rows, dx, dy):
    shifted = []
    for row in rows:
        x0, y0, x1, y1 = row["xyxy"]
        shifted.append({**row, "xyxy": (x0 + dx, y0 + dy, x1 + dx, y1 + dy)})
    return shifted


def _top_label(rows, limit=3):
    if not rows:
        return "-"
    return " ".join(f"{r['label']}:{r['conf']:.2f}" for r in rows[:limit])


def _write_gallery_index(out_dir: Path, rows: list[dict], *, yolo_only: bool = False) -> None:
    note = (
        "Only full-frame YOLO detections are drawn."
        if yolo_only
        else "Green boxes are board slots, orange is hero ROI, cyan boxes are YOLO detections."
    )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>YOLO card detector review</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#111;color:#eee}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}",
        "figure{margin:0;background:#1b1b1b;padding:10px;border:1px solid #333}",
        "img{width:100%;height:auto;display:block}",
        "figcaption{font-size:13px;line-height:1.35;margin-top:8px;color:#ddd}",
        "code{color:#9ee}",
        "</style>",
        "<h1>YOLO card detector review</h1>",
        f"<p>{html.escape(note)}</p>",
        "<div class='grid'>",
    ]
    for row in rows:
        src = html.escape(row["src"])
        caption = html.escape(row["caption"])
        parts.append(f"<figure><a href='{src}'><img src='{src}'></a><figcaption>{caption}</figcaption></figure>")
    parts.append("</div>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _annotate_gallery_frame(img, *, t, model, conf, imgsz, map_roi, board_expected, hero_expected):
    out = img.copy()
    full_rows = _predict_boxes(model, img, conf=conf, imgsz=imgsz)
    for row in full_rows:
        _draw_xyxy(out, row["xyxy"], (255, 255, 0), f"{row['label']} {row['conf']:.2f}", 2)

    for slot_i, slot in enumerate(read_cards.BOARD_SLOTS):
        xyxy = _xyxy_from_roi(map_roi, slot)
        x0, y0, x1, y1 = xyxy
        det = _predict_boxes(model, img[y0:y1, x0:x1], conf=conf, imgsz=imgsz)
        expected = board_expected[slot_i] if slot_i < len(board_expected) else None
        _draw_xyxy(out, xyxy, (70, 220, 70), f"board{slot_i} tmpl={expected or '-'} yolo={_top_label(det, 1)}", 2)
        for row in _offset_boxes(det[:3], x0, y0):
            _draw_xyxy(out, row["xyxy"], (255, 255, 0), f"{row['label']} {row['conf']:.2f}", 1)

    hx0, hy0, hx1, hy1 = _xyxy_from_roi(map_roi, read_hero.hero_roi_box())
    hero_roi = img[hy0:hy1, hx0:hx1]
    hero_det = _predict_boxes(model, hero_roi, conf=conf, imgsz=imgsz)
    _draw_xyxy(out, (hx0, hy0, hx1, hy1), (255, 140, 0),
               f"hero tmpl={' '.join(hero_expected) if hero_expected else '-'} yolo={_top_label(hero_det, 2)}", 2)
    for row in _offset_boxes(hero_det[:6], hx0, hy0):
        _draw_xyxy(out, row["xyxy"], (255, 255, 0), f"{row['label']} {row['conf']:.2f}", 1)

    _draw_text(out, f"t={t:.2f}s full_yolo={_top_label(full_rows, 8)}", 22, 36, (255, 255, 255), 0.72)
    return out, full_rows, hero_det


def _annotate_yolo_only_frame(img, *, t, model, conf, imgsz):
    out = img.copy()
    full_rows = _predict_boxes(model, img, conf=conf, imgsz=imgsz)
    for row in full_rows:
        _draw_xyxy(out, row["xyxy"], (255, 255, 0), f"{row['label']} {row['conf']:.2f}", 2)
    _draw_text(out, f"t={t:.2f}s yolo={_top_label(full_rows, 12)}", 22, 36, (255, 255, 255), 0.72)
    return out, full_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/videos/clubwpt_session_01.mov")
    parser.add_argument("--weights", default="cv_lab/models/best.pt")
    parser.add_argument("--models-dir", default="cv_lab/models")
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--max-table-frames", type=int, default=80)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR),
                        help="path to the sunsmarterjie/yolov12 checkout; use empty string for installed ultralytics")
    parser.add_argument("--gallery-out", default="", help="optional directory for annotated manual-review gallery")
    parser.add_argument("--gallery-max", type=int, default=120)
    parser.add_argument("--gallery-width", type=int, default=1400)
    parser.add_argument("--yolo-only-gallery", action="store_true",
                        help="gallery draws only full-frame YOLO detections; no template labels, ROIs, or crop detections")
    args = parser.parse_args()

    read_table.load_models(args.models_dir)
    args.yolov12_vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(args.yolov12_vendor)
    model = YOLO(args.weights)

    counts = Counter()
    examples: list[str] = []
    gallery_rows: list[dict] = []
    gallery_dir = Path(args.gallery_out) if args.gallery_out else None
    if gallery_dir:
        gallery_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(args.video)
    stream = container.streams.video[0]
    tb = stream.time_base
    frame_i = -1
    table_seen = 0

    for frame in container.decode(stream):
        frame_i += 1
        if frame_i % args.stride:
            continue
        img = frame.to_ndarray(format="bgr24")
        snap = read_table.read_table(img)
        if snap.get("screen") != "table":
            continue
        table_seen += 1
        t = float(frame.pts * tb)

        full_rows = _predict_boxes(model, img, conf=args.conf, imgsz=args.imgsz)
        full = [(row["label"], row["conf"]) for row in full_rows]
        counts["full_frames"] += 1
        counts["full_card_detections"] += len(full)

        if args.yolo_only_gallery:
            if gallery_dir and len(gallery_rows) < args.gallery_max:
                annotated, gallery_full_rows = _annotate_yolo_only_frame(
                    img, t=t, model=model, conf=args.conf, imgsz=args.imgsz)
                if args.gallery_width and annotated.shape[1] > args.gallery_width:
                    scale = args.gallery_width / annotated.shape[1]
                    annotated = cv2.resize(
                        annotated,
                        (args.gallery_width, int(round(annotated.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                name = f"yolo_only_{len(gallery_rows):04d}_t{t:07.2f}.jpg"
                cv2.imwrite(str(gallery_dir / name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
                gallery_rows.append({
                    "src": name,
                    "caption": f"t={t:.2f}s yolo={_top_label(gallery_full_rows, 12)}",
                })
            if table_seen >= args.max_table_frames:
                break
            continue

        label, anchor = read_table.classify(img)
        map_roi = anchor["map_roi"]
        board_expected = read_cards.read_board(img, map_roi, read_table._CARDS)
        hero_expected = read_hero.read_hero(img, map_roi, read_table._CARDS)
        if board_expected:
            counts["frames_with_template_board"] += 1
        if hero_expected:
            counts["frames_with_template_hero"] += 1

        for slot_i, slot in enumerate(read_cards.BOARD_SLOTS):
            crop = _crop(img, map_roi(slot))
            det = _predict_cards(model, crop, conf=args.conf, imgsz=args.imgsz)
            counts["board_slot_crops"] += 1
            counts["board_slot_detections"] += len(det)
            expected = board_expected[slot_i] if slot_i < len(board_expected) else None
            if expected and det and det[0][0] == expected:
                counts["board_slot_top1_matches_template"] += 1
            if det and len(examples) < 20:
                examples.append(f"t={t:.2f} board{slot_i} expected={expected} det={det[:3]}")

        hero_box = map_roi(read_hero.hero_roi_box())
        hero_roi = _crop(img, hero_box)
        hero_roi_det = _predict_cards(model, hero_roi, conf=args.conf, imgsz=args.imgsz)
        counts["hero_roi_crops"] += 1
        counts["hero_roi_detections"] += len(hero_roi_det)
        if hero_roi_det and len(examples) < 20:
            examples.append(f"t={t:.2f} hero_roi expected={hero_expected} det={hero_roi_det[:3]}")

        for face_i, face_spec in enumerate(read_hero.FACES):
            face = read_hero._deskew(hero_roi, face_spec)
            det = _predict_cards(model, face, conf=args.conf, imgsz=args.imgsz)
            counts["hero_face_crops"] += 1
            counts["hero_face_detections"] += len(det)
            expected = hero_expected[face_i] if face_i < len(hero_expected) else None
            if expected and det and det[0][0] == expected:
                counts["hero_face_top1_matches_template"] += 1
            if det and len(examples) < 20:
                examples.append(f"t={t:.2f} hero_face{face_i} expected={expected} det={det[:3]}")

        if gallery_dir and len(gallery_rows) < args.gallery_max:
            annotated, full_rows, hero_rows = _annotate_gallery_frame(
                img, t=t, model=model, conf=args.conf, imgsz=args.imgsz,
                map_roi=map_roi, board_expected=board_expected, hero_expected=hero_expected)
            if args.gallery_width and annotated.shape[1] > args.gallery_width:
                scale = args.gallery_width / annotated.shape[1]
                annotated = cv2.resize(
                    annotated,
                    (args.gallery_width, int(round(annotated.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            name = f"yolo_review_{len(gallery_rows):04d}_t{t:07.2f}.jpg"
            cv2.imwrite(str(gallery_dir / name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            gallery_rows.append({
                "src": name,
                "caption": (
                    f"t={t:.2f}s template hero={hero_expected} board={board_expected} "
                    f"full_yolo={_top_label(full_rows, 8)} hero_yolo={_top_label(hero_rows, 4)}"
                ),
            })

        if table_seen >= args.max_table_frames:
            break

    container.close()
    if gallery_dir:
        _write_gallery_index(gallery_dir, gallery_rows, yolo_only=args.yolo_only_gallery)

    print(f"weights={args.weights}")
    print(f"yolov12_vendor={args.yolov12_vendor or '<installed ultralytics>'}")
    print(f"classes={len(model.names)} conf={args.conf} imgsz={args.imgsz}")
    print(f"sampled_table_frames={table_seen} stride={args.stride}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    if examples:
        print("examples:")
        for line in examples:
            print(f"  {line}")
    if gallery_dir:
        print(f"gallery_frames={len(gallery_rows)}")
        print(f"gallery_index={gallery_dir / 'index.html'}")


if __name__ == "__main__":
    main()
