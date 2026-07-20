"""Sweep videos with the two-model CV pipeline and flag CV-struggling frames.

Runs Model 1 (region detector) + Model 2 (card classifier) across sampled
frames from one or more videos. Any frame containing a face_card whose
predicted rank falls in a known-weak set (confirmed confusions like 3<->5,
9<->T) or whose classifier confidence is low gets its full frame saved into
the labeling images dir and its file_id written to a labeling_poker priority
queue, so a human labeler sees it first.

  python cv_lab/scripts/sweep_and_flag_weak_cards.py \
      --videos "data/videos/*.mov" --interval 2.0 --device mps \
      --queue-name cv_weak_cards
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import av
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_yolo_cards import DEFAULT_YOLOV12_VENDOR, _load_yolo_class, _resolve_vendor_path  # noqa: E402
from card_classifier import CardClassifier, DEFAULT_CLS_WEIGHTS  # noqa: E402
from run_two_model_pipeline import DEFAULT_DETECTOR, _detect_regions  # noqa: E402

DEFAULT_IMAGES_DIR = REPO_ROOT / "cv_lab" / "datasets" / "yolo_cards_autolabel_v1" / "images"
DEFAULT_PRIORITY_DIR = REPO_ROOT / "labeling_poker" / "priority"
DEFAULT_REPORT = REPO_ROOT / "cv_lab" / "results" / "cv_weak_card_flags.json"
DEFAULT_RENDER_DIR = REPO_ROOT / "cv_lab" / "results" / "annotated_frames"

BOX_COLORS = {
    "face_card": (0, 255, 0), "card_back": (180, 180, 180),
    "pot_text": (0, 200, 255), "stack_text": (255, 180, 0),
    "bet_text": (255, 0, 255), "action_pill": (0, 140, 255),
    "active_turn_indicator": (0, 0, 255), "dealer_button": (255, 255, 0),
}


def _draw_annotations(img, rows: list[dict], card_reads_by_row_id: dict[int, dict]):
    vis = img.copy()
    for i, r in enumerate(rows):
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        color = BOX_COLORS.get(r["class"], (255, 255, 255))
        card = card_reads_by_row_id.get(i)
        label = f"{card['label']} {card['conf']:.2f}" if card else r["class"]
        if card and card["reasons"]:
            color = (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return vis


def _slug(video: Path, idx: int) -> str:
    return f"v{idx:02d}"


def _val_split(file_id: str, val_fraction: float = 0.2) -> str:
    return "val" if (hash(file_id) % 100) < int(val_fraction * 100) else "train"


def _sample_times(container, stream, start: float, end: float, interval: float):
    t = start
    while t <= end:
        container.seek(int(t / stream.time_base), stream=stream)
        frame = None
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= t:
                break
        if frame is not None:
            yield t, frame.to_ndarray(format="bgr24")
        t += interval


def _rank_of(label: str) -> str:
    return label[0]


def sweep_video(
    video_path: Path,
    idx: int,
    detector,
    classifier: CardClassifier,
    *,
    interval: float,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    pad: float,
    weak_ranks: set[str],
    low_conf: float,
    images_dir: Path,
    end: float | None,
    max_per_video: int,
    render_dir: Path | None,
) -> tuple[list[dict], list[dict]]:
    flagged: list[dict] = []
    manifest: list[dict] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    duration = end if end is not None else float(stream.duration * stream.time_base) if stream.duration else 0.0
    n_sampled = 0
    last_sig = None
    slug = _slug(video_path, idx)
    for t, img in _sample_times(container, stream, 0.0, duration, interval):
        n_sampled += 1
        rows = _detect_regions(detector, img, conf=conf, imgsz=imgsz, iou=iou,
                               device=device, dedupe_iou=0.35)
        card_reads = []
        card_reads_by_row_id = {}
        for row_i, r in enumerate(rows):
            if r["class"] != "face_card":
                continue
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            w, h = x2 - x1, y2 - y1
            px1 = max(0, int(x1 - w * pad))
            py1 = max(0, int(y1 - h * pad))
            px2 = min(img.shape[1], int(x2 + w * pad))
            py2 = min(img.shape[0], int(y2 + h * pad))
            crop = img[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            label, cls_conf = classifier.classify(crop)
            if label is None:
                continue
            reasons = []
            if _rank_of(label) in weak_ranks:
                reasons.append(f"weak_rank:{_rank_of(label)}")
            if cls_conf < low_conf:
                reasons.append("low_conf")
            card = {"label": label, "conf": round(cls_conf, 3), "reasons": reasons}
            card_reads.append(card)
            card_reads_by_row_id[row_i] = card
        frame_reasons = [c for c in card_reads if c["reasons"]]

        if render_dir is not None:
            vis = _draw_annotations(img, rows, card_reads_by_row_id)
            render_path = render_dir / slug / f"t{t:08.2f}.jpg"
            render_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(render_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            manifest.append({
                "video": str(video_path), "slug": slug, "t": round(t, 2),
                "image": str(render_path.relative_to(render_dir)),
                "cards": card_reads, "flagged": bool(frame_reasons),
            })
        signature = tuple(sorted(c["label"] for c in card_reads))
        if frame_reasons and signature != last_sig:
            last_sig = signature
            file_id = f"cvflag_{_slug(video_path, idx)}_t{t:08.2f}"
            split = _val_split(file_id)
            out_path = images_dir / split / f"{file_id}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
            min_conf = min(c["conf"] for c in card_reads) if card_reads else 1.0
            flagged.append({
                "file_id": file_id,
                "out_path": out_path,
                "split": split,
                "video": str(video_path),
                "t": round(t, 2),
                "cards": card_reads,
                "min_conf": round(min_conf, 3),
                "n_weak_rank": sum(1 for c in frame_reasons if any(r.startswith("weak_rank") for r in c["reasons"])),
            })
        elif signature != last_sig:
            # board/hero state changed but nothing flagged in it -- stop matching against
            # a stale signature so the next real state change is still detected as new.
            last_sig = signature
        if n_sampled % 50 == 0:
            print(f"  [{video_path.name}] sampled={n_sampled} t={t:.1f}s flagged_so_far={len(flagged)}")
    container.close()

    if max_per_video and len(flagged) > max_per_video:
        flagged.sort(key=lambda f: (-f["n_weak_rank"], f["min_conf"]))
        dropped, flagged = flagged[max_per_video:], flagged[:max_per_video]
        for f in dropped:
            Path(f["out_path"]).unlink(missing_ok=True)

    print(f"[{video_path.name}] done: sampled={n_sampled} frames, deduped+capped flagged={len(flagged)}")
    return flagged, manifest


def _write_gallery(render_dir: Path, manifest: list[dict], videos: list[Path]) -> None:
    by_slug: dict[str, list[dict]] = {}
    for entry in manifest:
        by_slug.setdefault(entry["slug"], []).append(entry)
    for frames in by_slug.values():
        frames.sort(key=lambda e: e["t"])
    video_names = {_slug(v, i): v.name for i, v in enumerate(videos)}
    data = {slug: [{"t": e["t"], "image": e["image"], "cards": e["cards"], "flagged": e["flagged"]}
                   for e in frames] for slug, frames in by_slug.items()}
    html = """<!doctype html><meta charset="utf-8"><title>CV annotated frames</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;background:#111;color:#eee;display:flex;height:100vh}
#side{width:220px;overflow-y:auto;background:#1b1b1b;padding:12px;box-sizing:border-box}
#side select{width:100%;margin-bottom:12px}
#side button{width:100%;padding:8px;margin:2px 0;background:#222;color:#eee;border:1px solid #444;cursor:pointer}
#main{flex:1;display:flex;flex-direction:column;align-items:center;padding:16px;overflow-y:auto}
#frame{max-width:100%;max-height:75vh;border:1px solid #333}
#caption{margin-top:10px;font-size:14px;line-height:1.5}
#nav{margin-top:10px}
#nav button{font-size:16px;padding:6px 16px;margin:0 6px;background:#2a2a2a;color:#eee;border:1px solid #444;cursor:pointer}
.flagged{color:#ff6b6b;font-weight:bold}
.card-chip{display:inline-block;margin:2px 4px;padding:2px 6px;border-radius:4px;background:#222;border:1px solid #444}
.card-chip.bad{border-color:#ff6b6b;color:#ff6b6b}
</style>
<div id="side">
  <select id="videoSelect"></select>
  <div id="frameList"></div>
</div>
<div id="main">
  <img id="frame">
  <div id="nav">
    <button id="prev">&larr; Prev</button>
    <span id="counter"></span>
    <button id="next">Next &rarr;</button>
  </div>
  <div id="caption"></div>
</div>
<script>
const DATA = __DATA__;
const NAMES = __NAMES__;
let slug = Object.keys(DATA)[0];
let idx = 0;
const sel = document.getElementById("videoSelect");
for (const s of Object.keys(DATA)) {
  const opt = document.createElement("option");
  opt.value = s; opt.textContent = `${NAMES[s] || s} (${DATA[s].length})`;
  sel.appendChild(opt);
}
sel.addEventListener("change", () => { slug = sel.value; idx = 0; render(); });
function render() {
  const frames = DATA[slug];
  const f = frames[idx];
  document.getElementById("frame").src = `${slug}/${f.image}`;
  document.getElementById("counter").textContent = `${idx + 1} / ${frames.length}`;
  const cards = f.cards.map(c => `<span class="card-chip${c.reasons.length ? ' bad' : ''}">${c.label} ${c.conf.toFixed(2)}${c.reasons.length ? ' ('+c.reasons.join(',')+')' : ''}</span>`).join(" ");
  document.getElementById("caption").innerHTML = `t=${f.t}s ${f.flagged ? '<span class="flagged">FLAGGED</span>' : ''}<br>${cards || '(no cards read)'}`;
}
document.getElementById("prev").addEventListener("click", () => { idx = Math.max(0, idx - 1); render(); });
document.getElementById("next").addEventListener("click", () => { idx = Math.min(DATA[slug].length - 1, idx + 1); render(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") { idx = Math.max(0, idx - 1); render(); }
  if (e.key === "ArrowRight") { idx = Math.min(DATA[slug].length - 1, idx + 1); render(); }
});
render();
</script>
"""
    html = html.replace("__DATA__", json.dumps(data)).replace("__NAMES__", json.dumps(video_names))
    (render_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default=str(REPO_ROOT / "data" / "videos" / "*.mov"))
    parser.add_argument("--model1", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--model2", default=str(DEFAULT_CLS_WEIGHTS))
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--end", type=float, default=0.0, help="cap seconds per video; 0 = full video")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pad", type=float, default=0.12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weak-ranks", default="3,5,9,T")
    parser.add_argument("--low-conf", type=float, default=0.35)
    parser.add_argument("--max-per-video", type=int, default=60,
                        help="after dedup, cap flagged frames per video (worst-first); 0 = no cap")
    parser.add_argument("--start-index", type=int, default=0,
                        help="first video slug index (vNN); use when sweeping a subset after earlier videos")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--queue-name", default="cv_weak_cards")
    parser.add_argument("--priority-dir", default=str(DEFAULT_PRIORITY_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--render-dir", default=str(DEFAULT_RENDER_DIR),
                        help="save an annotated jpg for every sampled frame here + build a gallery; empty to skip")
    parser.add_argument("--append", action="store_true",
                        help="merge flagged ids into an existing queue/report instead of overwriting")
    parser.add_argument("--yolov12-vendor", default=str(DEFAULT_YOLOV12_VENDOR))
    args = parser.parse_args()

    weak_ranks = {r.strip().upper() for r in args.weak_ranks.split(",") if r.strip()}
    videos = sorted(Path(p) for p in glob.glob(args.videos))
    if not videos:
        raise SystemExit(f"no videos matched {args.videos!r}")

    vendor = _resolve_vendor_path(args.yolov12_vendor)
    YOLO = _load_yolo_class(vendor)
    print(f"loading Model 1: {args.model1}")
    detector = YOLO(args.model1)
    print(f"loading Model 2: {args.model2}")
    classifier = CardClassifier(weights=args.model2, vendor=vendor, imgsz=128, device=args.device)

    images_dir = Path(args.images_dir)
    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
    all_flagged: list[dict] = []
    all_manifest: list[dict] = []
    for i, video in enumerate(videos):
        idx = args.start_index + i
        print(f"\n=== video {idx}: {video} ===")
        flagged, manifest = sweep_video(
            video, idx, detector, classifier,
            interval=args.interval, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=args.device, pad=args.pad, weak_ranks=weak_ranks, low_conf=args.low_conf,
            images_dir=images_dir, end=(args.end or None), max_per_video=args.max_per_video,
            render_dir=render_dir,
        )
        all_flagged.extend(flagged)
        all_manifest.extend(manifest)

    if render_dir is not None:
        (render_dir / "manifest.json").write_text(json.dumps(all_manifest, indent=2), encoding="utf-8")
        _write_gallery(render_dir, all_manifest, videos)
        print(f"\nannotated frames + gallery: {render_dir / 'index.html'}")

    all_flagged.sort(key=lambda f: (-f["n_weak_rank"], f["min_conf"]))
    for f in all_flagged:
        f.pop("out_path", None)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing_flagged: list[dict] = []
    if args.append and report_path.is_file():
        try:
            existing_flagged = json.loads(report_path.read_text(encoding="utf-8")).get("flagged", [])
        except (OSError, json.JSONDecodeError):
            existing_flagged = []
    seen_ids = {f["file_id"] for f in existing_flagged}
    merged_flagged = existing_flagged + [f for f in all_flagged if f["file_id"] not in seen_ids]
    merged_flagged.sort(key=lambda f: (-f.get("n_weak_rank", 0), f.get("min_conf", 1.0)))

    report = {
        "videos": [str(v) for v in videos],
        "interval": args.interval,
        "weak_ranks": sorted(weak_ranks),
        "low_conf_threshold": args.low_conf,
        "max_per_video": args.max_per_video,
        "n_flagged_frames": len(merged_flagged),
        "flagged": merged_flagged,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    priority_dir = Path(args.priority_dir)
    priority_dir.mkdir(parents=True, exist_ok=True)
    queue_path = priority_dir / f"{args.queue_name}.txt"
    existing_ids: list[str] = []
    if args.append and queue_path.is_file():
        existing_ids = [line.strip() for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    new_ids = [f["file_id"] for f in all_flagged]
    merged_ids = existing_ids + [i for i in new_ids if i not in existing_ids]
    # Keep worst-first order from the merged report when rewriting the queue.
    order = {f["file_id"]: i for i, f in enumerate(merged_flagged)}
    merged_ids.sort(key=lambda i: order.get(i, 10**9))
    queue_path.write_text("\n".join(merged_ids) + ("\n" if merged_ids else ""), encoding="utf-8")

    n_weak = sum(1 for f in all_flagged if f["n_weak_rank"] > 0)
    print(f"\n=== SUMMARY ===")
    print(f"videos swept: {len(videos)}")
    print(f"frames flagged this run: {len(all_flagged)} ({n_weak} contain a weak-rank card)")
    print(f"queue total: {len(merged_ids)}")
    print(f"images written under: {images_dir}")
    print(f"report: {report_path}")
    print(f"priority queue ({len(merged_ids)} ids): {queue_path}")
    print(f"\nlabel these first: python -m labeling_poker.app   then open /?queue={args.queue_name}"
          f" or fetch /api/next?queue={args.queue_name}")


if __name__ == "__main__":
    main()
