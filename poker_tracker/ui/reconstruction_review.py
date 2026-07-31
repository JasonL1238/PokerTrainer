"""Frame-to-history evidence helpers for completed-session reconstruction review."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poker_tracker.persistence.completion import CompletionEvidence
from poker_tracker.ui.video_storage import CV_TIMELINES_DIR


@dataclass(frozen=True)
class KeyFrame:
    """One representative source frame for Study Approve side-by-side review."""

    label: str
    image_path: str
    timestamp_s: float | None = None

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


def empty_hands_review_message(timeline: dict[str, Any]) -> str:
    """Explain a completed job whose timeline has nothing to validate."""
    summary = timeline.get("summary") or {}
    metadata = timeline.get("metadata") or {}
    frames = int(summary.get("frames") or 0)
    table_frames = summary.get("table_frames")
    nontable_frames = summary.get("nontable_frames")
    layout = str(metadata.get("layout_profile") or "").strip()
    unsupported = layout.endswith("-unsupported")

    if (
        isinstance(table_frames, int)
        and isinstance(nontable_frames, int)
        and frames > 0
        and table_frames == 0
    ):
        detail = (
            f"All {frames} sampled frames were classified as non-table "
            "(lobby, modal, transition, or unrecognized layout), so detection never ran."
        )
        if unsupported:
            detail += (
                f" Layout {layout} is below the calibrated ClubWPT window size — "
                "record the full client closer to 1272×896 or larger."
            )
        return detail

    if unsupported:
        return (
            f"Reconstruction finished with no hands. Layout {layout} is outside the "
            "calibrated ClubWPT geometries, so the table was never reconstructed."
        )

    if frames > 0:
        return (
            f"Reconstruction finished over {frames} sampled frames but produced no "
            "hands to validate."
        )
    return "The reconstruction did not produce any hands to validate."


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


def hand_frame_progress(
    hand: dict[str, Any],
    reviews_by_image: dict[str, Any],
    *,
    countable_images: list[str] | None = None,
) -> dict[str, int]:
    """Count navigable frames vs permanently saved verdicts for one timeline hand.

    ``countable_images`` should be the same image list the UI can open (typically
    from ``states_for_hand``). Falling back to ``source_images`` alone can disagree
    with the frame carousel when the timeline is inconsistent.
    """
    images = (
        list(countable_images)
        if countable_images is not None
        else [str(image) for image in (hand.get("source_images") or [])]
    )
    total = len(images)
    reviewed = 0
    flagged = 0
    for image in images:
        review = reviews_by_image.get(image)
        if review is None:
            continue
        status = _review_status(review)
        if status in {"correct", "incorrect"}:
            reviewed += 1
        if status == "incorrect":
            flagged += 1
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": max(0, total - reviewed),
        "flagged": flagged,
    }


def hand_validation_label(
    hand: dict[str, Any],
    reviews_by_image: dict[str, Any],
    *,
    countable_images: list[str] | None = None,
) -> str:
    """Dropdown label that shows permanent partial-progress for a timeline hand."""
    number = int(hand.get("hand_number", 0))
    hero = " ".join(hand.get("hero") or []) or "cards unknown"
    progress = hand_frame_progress(
        hand, reviews_by_image, countable_images=countable_images
    )
    total = progress["total"]
    reviewed = progress["reviewed"]
    flagged = progress["flagged"]
    if total == 0:
        return f"Hand #{number} · {hero} · no retained frames"
    if reviewed == 0:
        status = "not started"
    elif reviewed < total:
        status = "in progress"
    else:
        status = "done"
    flag_bit = f" · {flagged} flagged" if flagged else ""
    return (
        f"Hand #{number} · {hero} · {reviewed}/{total} validated "
        f"({status}){flag_bit}"
    )


def first_unreviewed_frame_index(
    states: list[dict[str, Any]],
    reviews_by_image: dict[str, Any],
) -> int:
    """Resume point: first frame without a correct/incorrect verdict, else last."""
    if not states:
        return 0
    for index, state in enumerate(states):
        image = str(state.get("image") or "")
        status = _review_status(reviews_by_image.get(image))
        if status not in {"correct", "incorrect"}:
            return index
    return len(states) - 1


def _review_status(review: Any) -> str | None:
    if review is None:
        return None
    status = getattr(review, "status", None)
    if status is None and isinstance(review, dict):
        status = review.get("status")
    return str(status) if status is not None else None


def job_id_from_hand_notes(notes: str | None) -> int | None:
    """Parse ``job_<id>_timeline`` out of CV draft notes when present."""
    if not notes:
        return None
    match = re.search(r"job_(\d+)_timeline", notes)
    if match is None:
        return None
    return int(match.group(1))


def select_key_frames_for_review(
    states: list[dict[str, Any]],
    *,
    max_frames: int = 6,
) -> list[KeyFrame]:
    """Pick representative frames: hero cards, each board street, and terminal."""

    if not states or max_frames <= 0:
        return []

    selected: list[KeyFrame] = []
    seen_images: set[str] = set()

    def add(label: str, state: dict[str, Any]) -> None:
        image = str(state.get("image") or "").strip()
        if not image or image in seen_images or len(selected) >= max_frames:
            return
        seen_images.add(image)
        timestamp = state.get("time_s")
        selected.append(
            KeyFrame(
                label=label,
                image_path=image,
                timestamp_s=None if timestamp is None else float(timestamp),
            )
        )

    hero_state = next(
        (state for state in states if state.get("hero_cards")),
        states[0],
    )
    add(
        "Hero cards" if hero_state.get("hero_cards") else "Hand start",
        hero_state,
    )

    seen_board_lens: set[int] = set()
    for state in states:
        board_len = len(state.get("board_cards") or [])
        # Skip preflop (0); keep Flop/Turn/River from the shared street map.
        if board_len < 3 or board_len in seen_board_lens:
            continue
        label = _STREET_BY_BOARD_COUNT.get(board_len)
        if label is None:
            continue
        add(label, state)
        seen_board_lens.add(board_len)

    add("Terminal", states[-1])
    return selected


def key_frames_from_completion_evidence(
    evidence: CompletionEvidence,
    *,
    max_frames: int = 6,
) -> list[KeyFrame]:
    """Fall back to stored completion evidence frame refs when no timeline is open."""

    if max_frames <= 0:
        return []

    selected: list[KeyFrame] = []
    seen: set[str] = set()

    def add(label: str, path: str, timestamp_s: float | None = None) -> None:
        image = path.strip()
        if not image or image in seen or len(selected) >= max_frames:
            return
        seen.add(image)
        selected.append(
            KeyFrame(label=label, image_path=image, timestamp_s=timestamp_s)
        )

    preceding = evidence.preceding_boundary
    if preceding.frame_ref:
        add("Hand start", preceding.frame_ref, preceding.timestamp_s)

    following = evidence.following_boundary
    middle = [
        path
        for path in evidence.source_frames
        if path
        and path != preceding.frame_ref
        and path != following.frame_ref
    ]
    if middle:
        # Keep a small spread when many source frames exist.
        if len(middle) == 1:
            picks = middle
        else:
            picks = [middle[0], middle[len(middle) // 2], middle[-1]]
            # Preserve order while dropping duplicates from the sample.
            picks = list(dict.fromkeys(picks))
        for index, path in enumerate(picks):
            add(f"Source {index + 1}", path)
    elif evidence.source_frames and not preceding.frame_ref and not following.frame_ref:
        for index, path in enumerate(evidence.source_frames):
            add(f"Source {index + 1}", path)

    if following.frame_ref:
        add("Terminal", following.frame_ref, following.timestamp_s)
    return selected


def resolve_study_approve_key_frames(
    *,
    job_id: int | None,
    hand_number: int,
    evidence: CompletionEvidence,
    timeline_dir: Path = CV_TIMELINES_DIR,
    max_frames: int = 6,
) -> list[KeyFrame]:
    """Prefer timeline states; fall back to completion-evidence frame refs."""

    if job_id is not None:
        try:
            timeline = load_timeline_for_job(job_id, timeline_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            timeline = None
        if timeline is not None:
            hand_payload = next(
                (
                    hand
                    for hand in timeline.get("hands", [])
                    if int(hand.get("hand_number", -1)) == hand_number
                ),
                None,
            )
            if hand_payload is not None:
                states = states_for_hand(timeline, hand_payload)
                if states:
                    return select_key_frames_for_review(
                        states, max_frames=max_frames
                    )
    return key_frames_from_completion_evidence(evidence, max_frames=max_frames)


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
