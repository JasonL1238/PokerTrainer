"""Render annotated review frames for manual validation.

Offline/post-session only. This script reads a saved deterministic CV timeline,
seeks through the original screen recording, and writes frames with the current
per-frame outputs drawn on top. It is meant for human QA of board/hero cards,
pot, stacks, bets, pills, active seat, dealer seat, and hand segmentation.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import av
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from landmark_anchor import REF_SEAT_COINS, REF_W, REF_H, anchor, CANON_ROIS
from read_cards import BOARD_SLOTS
from read_hero import hero_roi_box
from read_pills import _pill_box
from read_seats import _seat_bet_box, _seat_stack_box
from run_session import reconstruct, segment


VIDEO = "/Users/jasonli/Documents/GitHub/PokerTrainer/data/videos/clubwpt_session_01.mov"


COLORS = {
    "pot": (0, 220, 255),
    "board": (60, 220, 60),
    "hero": (255, 120, 0),
    "stack": (255, 255, 255),
    "bet": (0, 180, 255),
    "pill": (255, 0, 255),
    "active": (255, 80, 40),
    "dealer": (235, 235, 235),
    "bad": (30, 30, 230),
}


def _box(anchor_map, roi):
    x0, x1, y0, y1 = anchor_map(roi)
    return max(x0, 0), max(y0, 0), max(x1, 0), max(y1, 0)


def _draw_box(img, box, color, label="", thickness=2):
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if label:
        _text(img, label, x0, max(y0 - 7, 18), color)


def _text(img, text, x, y, color=(255, 255, 255), scale=0.62, thick=2):
    cv2.putText(img, str(text), (int(x) + 1, int(y) + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2,
                cv2.LINE_AA)
    cv2.putText(img, str(text), (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _seat_points(anchor_map):
    pts = []
    for u, v in REF_SEAT_COINS:
        x0, x1, y0, y1 = anchor_map((u, u, v, v))
        pts.append((x0, y0))
    return pts


def _annotate(img_bgr, snap, hand_idx=None, hand_summary=None):
    out = img_bgr.copy()
    if snap.get("screen") != "table":
        _text(out, f"t={snap.get('t')} screen={snap.get('screen')}", 24, 44, COLORS["bad"], 0.9)
        return out

    a = anchor(img_bgr)
    if a is None:
        _text(out, f"t={snap.get('t')} anchor failed", 24, 44, COLORS["bad"], 0.9)
        return out

    m = a["map_roi"]
    pot = snap.get("pot")
    pot_raw = snap.get("pot_raw")
    _draw_box(out, _box(m, CANON_ROIS["pot"]), COLORS["pot"], f"pot {pot} raw {pot_raw}")

    board = snap.get("board") or []
    for i, slot in enumerate(BOARD_SLOTS):
        label = board[i] if i < len(board) else ""
        _draw_box(out, _box(m, slot), COLORS["board"], label)

    hero = snap.get("hero") or []
    _draw_box(out, _box(m, hero_roi_box()), COLORS["hero"], "hero " + " ".join(hero))

    seat_pts = _seat_points(m)
    seats = {s["seat"]: s for s in snap.get("seats") or []}
    pills = {p["seat"]: p for p in snap.get("pills") or []}
    active = snap.get("active_seat")
    dealer = snap.get("dealer_seat")

    for idx, (cx, cy) in enumerate(seat_pts):
        color = COLORS["active"] if idx == active else (110, 190, 255)
        cv2.circle(out, (int(cx), int(cy)), 12, color, 3)
        _text(out, f"s{idx}", cx - 15, cy - 18, color, 0.55)

        seat = seats.get(idx, {})
        stack = seat.get("stack")
        bet = seat.get("bet")
        _draw_box(out, _box(m, _seat_stack_box(idx)), COLORS["stack"],
                  "" if stack is None else f"stk {stack}", 1)
        _draw_box(out, _box(m, _seat_bet_box(idx)), COLORS["bet"],
                  "" if bet is None else f"bet {bet}", 1)

        pill = pills.get(idx, {})
        if pill.get("present"):
            label = pill.get("word") or pill.get("color") or "pill"
            _draw_box(out, _box(m, _pill_box(idx)), COLORS["pill"], label, 1)

        if idx == dealer:
            _text(out, "D", cx + 16, cy + 20, COLORS["dealer"], 0.75, 3)

    header = [
        f"t={snap.get('t')}s",
        f"hand={hand_idx + 1 if hand_idx is not None else '?'}",
        f"anchor resid={a['resid']:.4f}",
        f"timeline resid={snap.get('resid')}",
        f"active=s{active}" if active is not None else "active=?",
        f"dealer=s{dealer}" if dealer is not None else "dealer=?",
    ]
    _text(out, " | ".join(header), 22, 36, (255, 255, 255), 0.72)

    if hand_summary:
        summary = (
            f"recon complete={hand_summary.get('complete')} "
            f"hero={hand_summary.get('hero')} board={hand_summary.get('board')} "
            f"final={hand_summary.get('final_pot')} winner=s{hand_summary.get('winner_seat')}"
        )
        _text(out, summary, 22, 68, (255, 255, 255), 0.62)

    return out


def _read_frame_at(video, target_t):
    container = av.open(video)
    stream = container.streams.video[0]
    tb = stream.time_base
    # Seek a little before the target. PyAV uses stream time-base units.
    seek_ts = max(int((target_t - 1.0) / float(tb)), 0)
    container.seek(seek_ts, any_frame=False, backward=True, stream=stream)
    best = None
    best_dt = 1e9
    for frame in container.decode(stream):
        t = float(frame.pts * tb)
        dt = abs(t - target_t)
        if dt < best_dt:
            best = frame.to_ndarray(format="bgr24")
            best_dt = dt
        if t > target_t + 0.35:
            break
    container.close()
    return best


def _hand_lookup(timeline):
    hands = segment(timeline)
    summaries = [reconstruct(hand) for hand in hands]
    lookup = {}
    for i, hand in enumerate(hands):
        for snap in hand:
            lookup[float(snap["t"])] = (i, summaries[i])
    return lookup, summaries


def _selected_indices(timeline, every, max_frames, only_table):
    candidates = [
        i for i, snap in enumerate(timeline)
        if not only_table or snap.get("screen") == "table"
    ]
    picked = candidates[::max(every, 1)]
    if max_frames and len(picked) > max_frames:
        picked = picked[:max_frames]
    return picked


def _write_index(out_dir, rows):
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>CV manual review</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#111;color:#eee}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}",
        "figure{margin:0;background:#1b1b1b;padding:10px;border:1px solid #333}",
        "img{width:100%;height:auto;display:block}",
        "figcaption{font-size:13px;line-height:1.35;margin-top:8px;color:#ddd}",
        "code{color:#9ee}",
        "</style>",
        "<h1>CV manual review</h1>",
        "<div class='grid'>",
    ]
    for row in rows:
        caption = html.escape(row["caption"])
        src = html.escape(row["src"])
        parts.append(f"<figure><a href='{src}'><img src='{src}'></a><figcaption>{caption}</figcaption></figure>")
    parts.append("</div>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--timeline", default="cv_lab/results/session_yolo_eval_after_hero_fix_timeline.json")
    ap.add_argument("--out", default="cv_lab/results/manual_review")
    ap.add_argument("--every", type=int, default=10, help="render every Nth timeline table sample")
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--include-nontable", action="store_true")
    args = ap.parse_args()

    timeline = json.load(open(args.timeline))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup, summaries = _hand_lookup(timeline)
    rows = []

    for n, idx in enumerate(_selected_indices(
            timeline, args.every, args.max_frames, only_table=not args.include_nontable)):
        snap = timeline[idx]
        t = float(snap["t"])
        frame = _read_frame_at(args.video, t)
        if frame is None:
            continue
        hand_idx, hand_summary = lookup.get(t, (None, None))
        annotated = _annotate(frame, snap, hand_idx, hand_summary)
        name = f"review_{n:04d}_t{t:07.2f}.jpg"
        cv2.imwrite(str(out_dir / name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 86])
        rows.append({
            "src": name,
            "caption": (
                f"t={t:.2f}s hand={hand_idx + 1 if hand_idx is not None else '?'} "
                f"screen={snap.get('screen')} pot={snap.get('pot')} "
                f"hero={snap.get('hero')} board={snap.get('board')}"
            ),
        })

    _write_index(out_dir, rows)
    complete = sum(1 for s in summaries if s.get("complete"))
    print(f"wrote {len(rows)} annotated frames to {out_dir}")
    print(f"open {out_dir / 'index.html'}")
    print(f"hands={len(summaries)} complete={complete}")


if __name__ == "__main__":
    main()
