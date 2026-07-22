"""Build an offline card-state timeline from YOLO card detections.

This is the first reconstruction layer after card detection: assign detected
cards to coarse table zones, collapse repeated states, and emit a structured
timeline for later OCR/action reconstruction. It uses saved completed-session
datasets only; it does not capture live tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


DEFAULT_DATASET = "cv_lab/datasets/yolo_cards_card_changes_v1"
DEFAULT_OUT = "cv_lab/results/yolo_card_timeline.json"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_label(label: str) -> str:
    label = label.strip().upper()
    if len(label) == 2 and label[0] == "T":
        return "10" + label[1]
    return label


def _effective_label(row: dict) -> str:
    return _norm_label(row.get("correct_label", "") or row.get("pred_label", ""))


def _image_size(dataset: Path, rel_image: str) -> tuple[int, int]:
    img = cv2.imread(str(dataset / rel_image))
    if img is None:
        raise FileNotFoundError(dataset / rel_image)
    h, w = img.shape[:2]
    return w, h


def _zone_for_box(cx: float, cy: float) -> str:
    """Coarse zones for the current ClubWPT table view, in normalized coords.

    These are intentionally broad and should be replaced by a learned/anchored
    seat+card-zone model when more views are added.
    """
    if 0.40 <= cx <= 0.58 and cy >= 0.64:
        return "hero"
    if 0.32 <= cx <= 0.64 and 0.36 <= cy <= 0.55:
        return "board"
    return "other"


def _stage(board_count: int) -> str:
    if board_count <= 0:
        return "preflop"
    if board_count == 3:
        return "flop"
    if board_count == 4:
        return "turn"
    if board_count >= 5:
        return "river"
    return "partial_board"


def _dedupe_zone(cards: list[dict], *, zone: str) -> list[dict]:
    if zone in {"board", "hero"}:
        cards = sorted(cards, key=lambda c: c["cx"])
    else:
        cards = sorted(cards, key=lambda c: (c["cy"], c["cx"]))

    out: list[dict] = []
    for card in cards:
        if any(abs(card["cx"] - kept["cx"]) < 0.025 and abs(card["cy"] - kept["cy"]) < 0.05 for kept in out):
            continue
        out.append(card)
    return out


def _state_signature(zones: dict[str, list[dict]]) -> tuple:
    return (
        tuple(card["label"] for card in zones["hero"]),
        tuple(card["label"] for card in zones["board"]),
        tuple(card["label"] for card in zones["other"]),
    )


def _cards_only(cards: list[dict]) -> list[str]:
    return [card["label"] for card in cards]


def _unique_cards(cards: list[str]) -> bool:
    return len(cards) == len(set(cards))


def _best_board(states: list[dict]) -> list[str]:
    best: list[str] = []
    best_t = -1.0
    for state in states:
        board = state["board_cards"]
        if len(board) not in {3, 4, 5}:
            continue
        if len(board) > len(best) or (len(board) == len(best) and state["time_s"] >= best_t):
            best = board
            best_t = state["time_s"]
    return best


def _street_events(states: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen_counts: set[int] = set()
    names = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
    for state in states:
        count = len(state["board_cards"])
        if count not in names or count in seen_counts:
            continue
        seen_counts.add(count)
        out.append({
            "street": names[count],
            "time_s": state["time_s"],
            "board": state["board_cards"],
        })
    return out


def _segment_card_hands(states: list[dict]) -> list[list[dict]]:
    hands: list[list[dict]] = []
    current: list[dict] = []
    current_hero: list[str] = []

    for state in states:
        hero = state["hero_cards"]
        board = state["board_cards"]
        has_cards = bool(hero or board)
        if not has_cards:
            continue

        boundary = False
        if current:
            prev = current[-1]
            prev_board = prev["board_cards"]
            hero_changed = len(hero) == 2 and hero != current_hero
            board_reset = bool(prev_board) and not board
            large_gap = state["time_s"] - prev["time_s"] > 30
            boundary = hero_changed and (board_reset or bool(prev_board) or large_gap)

        if boundary:
            hands.append(current)
            current = []
            current_hero = []

        current.append(state)
        if len(hero) == 2 and not current_hero:
            current_hero = hero

    if current:
        hands.append(current)
    return hands


def _summarize_hands(states: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for hand_i, hand_states in enumerate(_segment_card_hands(states), start=1):
        hero_candidates = [s["hero_cards"] for s in hand_states if len(s["hero_cards"]) == 2]
        hero = hero_candidates[0] if hero_candidates else []
        board = _best_board(hand_states)
        cards = hero + board
        complete_cards = len(hero) == 2 and len(board) in {0, 3, 4, 5} and _unique_cards(cards)
        warnings = []
        if len(hero) != 2:
            warnings.append("hero_cards_not_two")
        if board and len(board) not in {3, 4, 5}:
            warnings.append("invalid_board_count")
        if not _unique_cards(cards):
            warnings.append("duplicate_visible_cards")
        summaries.append({
            "hand_number": hand_i,
            "t_start": hand_states[0]["time_s"],
            "t_end": hand_states[-1]["time_s"],
            "n_states": len(hand_states),
            "hero": hero or None,
            "board": board,
            "streets": _street_events(hand_states),
            "complete_cards": complete_cards,
            "warnings": warnings,
            "source_images": [s["image"] for s in hand_states],
        })
    return summaries


def _load_missing(dataset: Path) -> dict[str, dict]:
    rows = _read_csv(dataset / "missing_labels.csv")
    return {row["image"]: row for row in rows}


def build_timeline(dataset: Path) -> dict:
    manifest = _read_csv(dataset / "manifest.csv")
    detections = _read_csv(dataset / "corrections.csv")
    if not detections:
        detections = _read_csv(dataset / "detections.csv")
    missing_by_image = _load_missing(dataset)

    detections_by_image: dict[str, list[dict]] = {}
    for row in detections:
        if row.get("action", "").strip().lower() in {"delete", "drop", "remove"}:
            continue
        detections_by_image.setdefault(row["image"], []).append(row)

    frames: list[dict] = []
    states: list[dict] = []
    events: list[dict] = []
    last_sig: tuple | None = None
    last_hero: list[str] = []
    last_board: list[str] = []

    for frame_i, row in enumerate(manifest):
        image = row["image"]
        width, height = _image_size(dataset, image)
        zones = {"hero": [], "board": [], "other": []}

        for det in detections_by_image.get(image, []):
            x0, y0, x1, y1 = [float(det[k]) for k in ("x0", "y0", "x1", "y1")]
            cx = ((x0 + x1) / 2.0) / width
            cy = ((y0 + y1) / 2.0) / height
            zone = _zone_for_box(cx, cy)
            zones[zone].append({
                "label": _effective_label(det),
                "conf": float(det.get("conf", 0) or 0),
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            })

        for zone in zones:
            zones[zone] = _dedupe_zone(zones[zone], zone=zone)

        hero = _cards_only(zones["hero"])[:2]
        board = _cards_only(zones["board"])[:5]
        other = _cards_only(zones["other"])
        missing = missing_by_image.get(image)
        frame_state = {
            "frame_index": frame_i,
            "video_frame": int(row["frame"]),
            "time_s": float(row["time_s"]),
            "image": image,
            "stage": _stage(len(board)),
            "hero_cards": hero,
            "board_cards": board,
            "other_cards": other,
            "zones": zones,
            "missing": missing or None,
        }
        frames.append(frame_state)

        sig = _state_signature(zones)
        if sig == last_sig:
            continue

        state = {
            "state_index": len(states),
            "time_s": frame_state["time_s"],
            "image": image,
            "stage": frame_state["stage"],
            "hero_cards": hero,
            "board_cards": board,
            "other_cards": other,
            "missing": missing or None,
        }
        states.append(state)

        if hero != last_hero:
            events.append({
                "type": "hero_cards_changed",
                "time_s": frame_state["time_s"],
                "from": last_hero,
                "to": hero,
                "image": image,
            })
        if board != last_board:
            events.append({
                "type": "board_changed",
                "time_s": frame_state["time_s"],
                "from": last_board,
                "to": board,
                "stage": frame_state["stage"],
                "image": image,
            })

        last_sig = sig
        last_hero = hero
        last_board = board

    hands = _summarize_hands(states)
    return {
        "metadata": {
            "dataset": str(dataset),
            "source": "yolo_card_detections",
            "notes": [
                "Offline completed-session reconstruction artifact.",
                "Zones are coarse geometry-based assignments for the current ClubWPT view.",
                "Cards marked missing in the review UI are carried through but need boxes before training.",
            ],
        },
        "summary": {
            "frames": len(frames),
            "states": len(states),
            "events": len(events),
            "hands": len(hands),
            "card_complete_hands": sum(1 for hand in hands if hand["complete_cards"]),
        },
        "hands": hands,
        "states": states,
        "events": events,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    timeline = build_timeline(dataset)
    out.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"dataset={dataset}")
    print(f"out={out}")
    print(f"frames={timeline['summary']['frames']}")
    print(f"states={timeline['summary']['states']}")
    print(f"events={timeline['summary']['events']}")
    print(f"hands={timeline['summary']['hands']}")
    print(f"card_complete_hands={timeline['summary']['card_complete_hands']}")


if __name__ == "__main__":
    main()
