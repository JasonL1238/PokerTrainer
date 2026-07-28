"""Frame-to-history evidence helpers for completed-session reconstruction review."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_tracker.ui.video_storage import CV_TIMELINES_DIR

ISSUE_GUIDANCE: dict[str, tuple[str, str]] = {
    "Cards / board": (
        "Card classifier",
        "Add this crop to card hard examples and check rank/suit confidence.",
    ),
    "Action / player": (
        "Action reconstruction",
        "Inspect pill attribution, active-seat order, and fold persistence.",
    ),
    "Amount / stack": (
        "Stack & bet OCR",
        "Compare stack and bet-text reads across the neighboring frames.",
    ),
    "Pot": (
        "Pot OCR & reconciliation",
        "Check pot-region OCR and the contribution/winner consensus.",
    ),
    "Street / hand boundary": (
        "Timeline segmentation",
        "Review card debounce, board reset, dealer movement, and sampling gaps.",
    ),
    "Winner / result": (
        "Settlement inference",
        "Inspect terminal stack recovery, pot sweep, and fold-only handling.",
    ),
    "Frame not useful": (
        "Keyframe selection",
        "Remove redundant states or prefer a clearer neighboring source frame.",
    ),
}

_STREET_BY_BOARD_COUNT = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}


def timeline_path_for_job(
    job_id: int, timeline_dir: Path = CV_TIMELINES_DIR
) -> Path:
    return timeline_dir / f"job_{job_id}_timeline.json"


def load_timeline_for_job(
    job_id: int, timeline_dir: Path = CV_TIMELINES_DIR
) -> dict[str, Any] | None:
    path = timeline_path_for_job(job_id, timeline_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("hands"), list) or not isinstance(
        payload.get("states"), list
    ):
        raise ValueError(f"Timeline has no hands/states lists: {path}")
    return payload


def states_for_hand(
    timeline: dict[str, Any], hand: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the distinct source states the reconstruction retained for one hand."""
    source_images = set(hand.get("source_images") or [])
    start = float(hand.get("t_start", 0))
    end = float(hand.get("t_end", start))
    states = [
        state
        for state in timeline.get("states", [])
        if state.get("image") in source_images
        and start <= float(state.get("time_s", start)) <= end
    ]
    return sorted(states, key=lambda state: (state.get("time_s", 0), state.get("state_index", 0)))


def observed_facts(state: dict[str, Any]) -> list[tuple[str, str]]:
    """Compact, human-readable model observations for one retained frame."""
    board = state.get("board_cards") or []
    street = _STREET_BY_BOARD_COUNT.get(len(board), str(state.get("stage") or "Unknown").title())
    stacks = state.get("stacks") or {}
    bets = state.get("bets") or {}
    pills = state.get("pills") or {}
    return [
        ("Street", street),
        ("Hero cards", _cards(state.get("hero_cards"))),
        ("Board", _cards(board)),
        ("Pot", _amount(state.get("pot"))),
        ("Active seat", _seat(state.get("active_seat"))),
        ("Dealt in", ", ".join(f"Seat {seat}" for seat in state.get("dealt_in", [])) or "None read"),
        ("Stacks", _seat_values(stacks)),
        ("Bets", _seat_values(bets)),
        (
            "Action pills",
            ", ".join(f"Seat {seat}: {action}" for seat, action in _sorted_items(pills))
            or "None read",
        ),
    ]


def history_impacts(
    hand: dict[str, Any],
    states: list[dict[str, Any]],
    frame_index: int,
) -> list[dict[str, str]]:
    """Explain exactly which reconstructed history facts came from one frame."""
    state = states[frame_index]
    previous = states[frame_index - 1] if frame_index > 0 else None
    impacts: list[dict[str, str]] = []

    if previous is None:
        hero = _cards(hand.get("hero"))
        players = hand.get("players") or []
        impacts.append(
            {
                "kind": "Hand boundary",
                "text": f"Started Hand #{hand.get('hand_number')} with hero {hero}.",
                "source": "first retained state",
            }
        )
        if players:
            player_labels = []
            for player in players:
                name = player.get("player_name") or f"Seat {player.get('seat')}"
                position = player.get("position") or "position unknown"
                player_labels.append(f"{name} ({position})")
            positions = ", ".join(player_labels)
            impacts.append(
                {
                    "kind": "Players",
                    "text": positions,
                    "source": "dealt-in cards + dealer button",
                }
            )

    prior_board = [] if previous is None else (previous.get("board_cards") or [])
    board = state.get("board_cards") or []
    if board != prior_board and board:
        impacts.append(
            {
                "kind": _STREET_BY_BOARD_COUNT.get(len(board), "Board"),
                "text": f"Board became {_cards(board)}.",
                "source": "face-card detections",
            }
        )

    for action in hand.get("actions") or []:
        if action.get("source_image") != state.get("image"):
            continue
        actor = " ".join(
            part
            for part in (action.get("position"), action.get("player_name"))
            if part
        )
        amount = "" if action.get("amount") is None else f" {action['amount']:g} BB"
        actor = actor or f"Seat {action.get('seat')}"
        impacts.append(
            {
                "kind": str(action.get("street", "")).title(),
                "text": f"{actor} {str(action.get('action_type', '')).replace('_', ' ')}"
                f"{amount}.",
                "source": str(action.get("derivation") or "reconstruction").replace("_", " "),
            }
        )

    if previous is not None and state.get("pot") != previous.get("pot"):
        impacts.append(
            {
                "kind": "Pot",
                "text": f"Observed pot changed from {_amount(previous.get('pot'))} "
                f"to {_amount(state.get('pot'))}.",
                "source": "pot OCR",
            }
        )

    if frame_index == len(states) - 1:
        result = hand.get("result") or "Result unresolved"
        pot = _amount(hand.get("pot"))
        impacts.append(
            {
                "kind": "Settlement",
                "text": f"{result}; final pot {pot}.",
                "source": (
                    "stack recovery + pot reconciliation"
                    if hand.get("reconciled")
                    else "terminal observations (not reconciled)"
                ),
            }
        )

    if not impacts:
        impacts.append(
            {
                "kind": "Confirmation",
                "text": "No new history line was added; this frame confirmed the stable table state.",
                "source": "temporal debounce",
            }
        )
    return impacts


def _cards(cards: Any) -> str:
    return " ".join(str(card) for card in (cards or [])) or "Not read"


def _amount(value: Any) -> str:
    return "Not read" if value is None else f"{float(value):g} BB"


def _seat(value: Any) -> str:
    return "Not read" if value is None else f"Seat {value}"


def _seat_values(values: dict[Any, Any]) -> str:
    return (
        ", ".join(f"Seat {seat}: {float(value):g}" for seat, value in _sorted_items(values))
        or "None read"
    )


def _sorted_items(values: dict[Any, Any]) -> list[tuple[Any, Any]]:
    return sorted(values.items(), key=lambda item: int(item[0]))
