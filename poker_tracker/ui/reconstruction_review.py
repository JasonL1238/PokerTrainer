"""Frame-to-history evidence helpers for completed-session reconstruction review."""
from __future__ import annotations

import json
import math
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


@dataclass(frozen=True)
class TimelineActionRef:
    """One reconstructed action line tied to a retained source frame."""

    street: str
    action_type: str
    player_name: str
    position: str
    amount: float | None
    source_image: str
    seat: int | None = None

    def label(self) -> str:
        actor = " ".join(part for part in (self.position, self.player_name) if part)
        if not actor and self.seat is not None:
            actor = f"Seat {self.seat}"
        actor = actor or "Unknown"
        amount = "" if self.amount is None else f" {self.amount:g} BB"
        return (
            f"{self.street.title()} · {actor} · "
            f"{self.action_type.replace('_', ' ').replace('-', ' ').title()}"
            f"{amount}"
        )


@dataclass(frozen=True)
class FrameIssueTarget:
    """A frame that blocks clean validation, plus the actions it produced."""

    frame_index: int
    source_image: str
    timestamp_seconds: float
    status: str  # "incorrect" | "unreviewed"
    issue_types: tuple[str, ...]
    notes: str
    actions: tuple[TimelineActionRef, ...]

    def summary(self) -> str:
        when = f"{self.timestamp_seconds:.2f}s"
        if self.status == "incorrect":
            issues = ", ".join(self.issue_types) or "flagged"
            return f"Frame {self.frame_index + 1} @ {when} · {issues}"
        return f"Frame {self.frame_index + 1} @ {when} · unreviewed"

    def action_labels(self) -> tuple[str, ...]:
        return tuple(action.label() for action in self.actions)


@dataclass(frozen=True)
class ActionCvIssue:
    """One CV read failure that affects a specific reconstructed action line."""

    kind: str
    detail: str
    frame_index: int | None = None
    # True only when the message hands the operator a figure to type in.
    # Matching this out of the prose is how a caption came to tell the
    # operator to re-enter the very number the warning had condemned.
    offers_a_value: bool = False

    def label(self) -> str:
        return f"{self.kind} — {self.detail}"


# Human explanations for OCR refusal codes surfaced on action lines.
UNKNOWN_AMOUNT_CODE_TEXT: dict[str, str] = {
    "below_calibrated_render_size": (
        "the on-screen text rendered below the size the reader is calibrated "
        "to trust, so it refused to guess"
    ),
    "no_digit_run": "no digits were found in the text region",
    "bet_boxes_disagree": (
        "two reads of the same bet box returned different numbers, so neither "
        "was trusted"
    ),
    "stack_boxes_disagree": (
        "two reads of the same stack box returned different numbers, so "
        "neither was trusted"
    ),
    "ambiguous_longest_run": (
        "more than one digit run was equally plausible, so none was chosen"
    ),
}

MONEY_ACTION_TYPES = {"bet", "raise", "call", "all_in", "all-in", "post_blind", "ante"}

# Distinct from a missing-value issue: this one questions whether the action
# happened at all, so it must never drive a "fill this field in" affordance.
ACTION_MAY_NOT_BELONG = "Action may not belong to this hand"

# Kinds whose remedy is the Stack-before field, so the editor must reveal it.
STACK_VALUE_KINDS = {"Stack before unknown", "Stack before looks post-action"}

_INFERRED_REASONS: dict[str, str] = {
    "inferred_round_complete": (
        "This line was inferred from the betting round completing rather than "
        "observed on a frame"
    ),
    "inferred_still_in": (
        "This line was inferred from the seat still being in the hand when the "
        "round completed, rather than observed on a frame"
    ),
}


def _inferred_reason(derivation: str) -> str:
    return _INFERRED_REASONS.get(
        derivation,
        "This line was inferred rather than observed on a frame",
    )


@dataclass(frozen=True)
class ValidationFrameContext:
    """Frame-review context needed beside the Import edit/approve panel."""

    job_id: int
    hand_number: int
    timeline_hand: dict[str, Any]
    states: list[dict[str, Any]]
    reviews_by_image: dict[str, Any]
    cursor_key: str
    pending_hand_key: str
    # First sampled second of the WHOLE recording (not this hand), so
    # "recording starts mid-hand" claims stay honest for recordings that
    # open on a lobby. None falls back to 0.0.
    recording_start_s: float | None = None

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

STREET_BY_BOARD_COUNT = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}


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
    street = STREET_BY_BOARD_COUNT.get(len(board), str(state.get("stage") or "Unknown").title())
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
                "kind": STREET_BY_BOARD_COUNT.get(len(board), "Board"),
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


def timeline_actions_for_image(
    hand: dict[str, Any], source_image: str
) -> tuple[TimelineActionRef, ...]:
    """Return reconstructed action lines attributed to one source frame."""

    refs: list[TimelineActionRef] = []
    for action in hand.get("actions") or []:
        if str(action.get("source_image") or "") != source_image:
            continue
        amount = action.get("amount")
        seat = action.get("seat")
        refs.append(
            TimelineActionRef(
                street=str(action.get("street") or "unknown"),
                action_type=str(action.get("action_type") or "unknown"),
                player_name=str(action.get("player_name") or ""),
                position=str(action.get("position") or ""),
                amount=None if amount is None else float(amount),
                source_image=source_image,
                seat=None if seat is None else int(seat),
            )
        )
    return tuple(refs)


def frame_issue_targets(
    hand: dict[str, Any],
    states: list[dict[str, Any]],
    reviews_by_image: dict[str, Any],
) -> list[FrameIssueTarget]:
    """Frames that still need attention (flagged or unreviewed), with linked actions."""

    targets: list[FrameIssueTarget] = []
    for index, state in enumerate(states):
        image = str(state.get("image") or "")
        if not image:
            continue
        review = reviews_by_image.get(image)
        status = _review_status(review)
        if status == "correct":
            continue
        if status == "incorrect":
            issue_types = tuple(_review_issue_types(review))
            notes = _review_notes(review)
            frame_status = "incorrect"
        else:
            issue_types = ()
            notes = ""
            frame_status = "unreviewed"
        targets.append(
            FrameIssueTarget(
                frame_index=index,
                source_image=image,
                timestamp_seconds=float(state.get("time_s", 0)),
                status=frame_status,
                issue_types=issue_types,
                notes=notes,
                actions=timeline_actions_for_image(hand, image),
            )
        )
    return targets


def match_db_action_to_frame_target(
    *,
    street: str,
    action_type: str,
    player_name: str,
    position: str,
    amount: float | None,
    targets: list[FrameIssueTarget],
    identity_only: bool = False,
) -> FrameIssueTarget | None:
    """Match a saved DB action to a flagged/unreviewed frame via timeline actions.

    Street and action type are required. Identity must also agree via position or
    player name so two same-street folds by different seats cannot collide. When
    both sides declare an amount, the amounts must match.

    ``identity_only`` relaxes the type and amount checks: frame flags belong to
    the frame, not to the action's identity, so an operator's type correction
    should not detach the row from its flagged source frame.
    """

    normalized_type = action_type.replace("-", "_")
    street_key = street.lower()
    candidates: list[tuple[int, FrameIssueTarget]] = []
    for target in targets:
        for action in target.actions:
            if action.street.lower() != street_key:
                continue
            if (
                not identity_only
                and action.action_type.replace("-", "_") != normalized_type
            ):
                continue
            position_match = bool(position and action.position and action.position == position)
            player_match = bool(
                player_name and action.player_name and action.player_name == player_name
            )
            if not position_match and not player_match:
                continue
            if (
                not identity_only
                and amount is not None
                and action.amount is not None
                and abs(float(amount) - float(action.amount)) >= 1e-6
            ):
                continue
            score = 8
            if position_match:
                score += 3
            if player_match:
                score += 2
            if identity_only:
                pass
            elif amount is None and action.amount is None:
                score += 1
            elif (
                amount is not None
                and action.amount is not None
                and abs(float(amount) - float(action.amount)) < 1e-6
            ):
                score += 2
            candidates.append((score, target))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].frame_index))
    # Ambiguous equal-score matches are unsafe for jump/badge wiring. Compare
    # EVERY top-scoring candidate, not just the runner-up: two candidates on
    # one frame plus a third elsewhere would otherwise pass the guard and
    # silently wire the row to the wrong frame.
    best_score = candidates[0][0]
    best_images = {
        target.source_image for score, target in candidates if score == best_score
    }
    if len(best_images) > 1:
        return None
    return candidates[0][1]


def match_db_action_to_timeline_action(
    hand: dict[str, Any],
    *,
    street: str,
    action_index: int | None,
    action_type: str,
    position: str,
    player_name: str,
) -> dict[str, Any] | None:
    """Match a saved DB action back to the timeline action that produced it.

    Street plus per-street action index is the primary key (imports preserve
    both). The action type and the actor identity (position or player name)
    must also agree, so an edited, reordered, or re-added line cannot silently
    borrow another slot's read issues. Returns None rather than guessing.
    """

    if action_index is None:
        return None
    street_key = street.lower()
    normalized_type = action_type.replace("-", "_")
    for action in hand.get("actions") or []:
        if str(action.get("street", "")).lower() != street_key:
            continue
        if action.get("action_index") != action_index:
            continue
        if str(action.get("action_type") or "").replace("-", "_") != normalized_type:
            return None
        t_position = str(action.get("position") or "")
        t_player = str(action.get("player_name") or "")
        position_declared = bool(position and t_position)
        player_declared = bool(player_name and t_player)
        if position_declared and t_position != position:
            return None
        if player_declared and t_player != player_name:
            return None
        if not position_declared and not player_declared:
            return None
        return action
    return None


def timeline_action_by_frame_and_seat(
    hand: dict[str, Any],
    source_image: str | None,
    seat: int | None,
) -> dict[str, Any] | None:
    """The reconstructed line a row came from, keyed by frame and seat.

    This key survives every correction the operator can make — street, order,
    type, actor, amount — because it names where the line was observed rather
    than what it currently claims. Used to keep explaining a row after an edit
    detaches it from its slot. Returns None when the pair is ambiguous, since
    guessing here would attribute another line's reads.
    """

    if not source_image or seat is None:
        return None
    matches = [
        action
        for action in hand.get("actions") or []
        if str(action.get("source_image") or "") == source_image
        and action.get("seat") == seat
    ]
    return matches[0] if len(matches) == 1 else None


def timeline_source_image_for_slot(
    hand: dict[str, Any],
    *,
    street: str,
    action_index: int | None,
    position: str,
    player_name: str,
) -> str | None:
    """Source frame of the timeline line occupying this street/index slot.

    Deliberately ignores the action type, unlike
    :func:`match_db_action_to_timeline_action`: this answers "which frame did
    this row come from" for frame-level facts that survive a type correction,
    not "may this row inherit that line's read issues". ``(street,
    action_index)`` is unique in a timeline hand, so no ambiguity arises.
    """

    if action_index is None:
        return None
    street_key = street.lower()
    for action in hand.get("actions") or []:
        if str(action.get("street", "")).lower() != street_key:
            continue
        if action.get("action_index") != action_index:
            continue
        t_position = str(action.get("position") or "")
        t_player = str(action.get("player_name") or "")
        if position and t_position and t_position != position:
            return None
        if player_name and t_player and t_player != player_name:
            return None
        return str(action.get("source_image") or "") or None
    return None


def seat_refusal_code(mapping: Any, seat: int | None) -> str | None:
    """Fetch a per-seat refusal code from a JSON dict keyed by str or int seat."""

    if seat is None or not isinstance(mapping, dict):
        return None
    value = mapping.get(str(seat), mapping.get(seat))
    return str(value) if value else None


def unknown_read_text(code: str | None) -> str | None:
    if code is None:
        return None
    return UNKNOWN_AMOUNT_CODE_TEXT.get(code, f"read refused ({code.replace('_', ' ')})")


def cv_issues_for_timeline_action(
    timeline_action: dict[str, Any],
    hand: dict[str, Any],
    states: list[dict[str, Any]],
    *,
    db_amount: float | None,
    db_stack_before: float | None,
    recording_start_s: float | None = None,
    db_street: str | None = None,
    db_action_type: str | None = None,
) -> list[ActionCvIssue]:
    """Explain which CV read failures affect one action line, tied to its frame.

    ``db_amount`` / ``db_stack_before`` are the current saved values: once the
    operator fills a field in, the matching issue stops being reported.
    ``recording_start_s`` is the whole recording's first sampled second (not
    this hand's) so mid-hand claims survive recordings that open on a lobby.

    ``db_street`` / ``db_action_type`` are the SAVED row's values. Every gate
    that asks "does this row need an amount" or "is this row on the frame's
    street" must read these, not the reconstructed line's frozen fields —
    otherwise a retyped fold is still asked for an amount, and a fold retyped
    to a bet goes silent.
    """

    source_image = str(timeline_action.get("source_image") or "")
    frame_index = next(
        (
            index
            for index, state in enumerate(states)
            if str(state.get("image") or "") == source_image
        ),
        None,
    )
    state = states[frame_index] if frame_index is not None else None
    seat_raw = timeline_action.get("seat")
    try:
        seat = None if seat_raw is None else int(seat_raw)
    except (TypeError, ValueError, OverflowError):
        seat = None
    action_type = str(
        db_action_type
        if db_action_type is not None
        else timeline_action.get("action_type") or ""
    ).replace("-", "_")
    frame_ref = (
        f"frame {frame_index + 1}" if frame_index is not None else "an unretained frame"
    )
    issues: list[ActionCvIssue] = []
    origin_street = str(timeline_action.get("street") or "").lower()
    row_street = str(db_street or origin_street).lower()
    if row_street and origin_street and row_street != origin_street:
        # The row was moved off the street it was reconstructed on, so the
        # frame below belongs to the line's ORIGINAL street.
        issues.append(
            ActionCvIssue(
                kind="Moved off its source street",
                detail=(
                    f"This line was reconstructed on the {origin_street} from "
                    f"{frame_ref}, but it is now saved on the {row_street}. "
                    "The frame-based notes below still describe the "
                    f"{origin_street} line — re-check them against the frames "
                    "for its new street."
                ),
                frame_index=frame_index,
            )
        )

    derivation = str(timeline_action.get("derivation") or "")
    if db_amount is None and action_type in MONEY_ACTION_TYPES:
        timeline_amount = timeline_action.get("amount")
        code = seat_refusal_code(state.get("bets_unknown") if state else None, seat)
        code_text = unknown_read_text(code)
        readable_bet = seat_value(state.get("bets") if state else None, seat)
        parsed_amount = _optional_float(timeline_amount)
        if parsed_amount is not None:
            if readable_bet is not None:
                box_note = (
                    f" ({frame_ref.capitalize()}'s bet box shows the seat's "
                    "total for the street, not this increment.)"
                )
            elif code_text is not None:
                # The reader saw a box and declined it; saying nothing was
                # there contradicts its own record for this frame.
                box_note = f" (On {frame_ref}, {code_text}.)"
            else:
                box_note = (
                    f" (The reader recorded no bet box for this seat on "
                    f"{frame_ref}.)"
                )
            detail = (
                f"The reconstruction read {parsed_amount:g} BB as the chips "
                "this seat added here, but the saved amount is empty — "
                "confirm it and re-enter it in the Amount field below."
                f"{box_note}"
            )
        elif _line_may_not_belong(states, state, seat, frame_index):
            # The accusation itself belongs to ACTION_MAY_NOT_BELONG below;
            # this branch only explains why the amount is missing, and must
            # never instruct a delete under an "Amount unknown" heading.
            box_read = seat_value(state.get("bets") if state else None, seat)
            box_refused = seat_refusal_code(state.get("bets_unknown") if state else None, seat)
            if box_read is None and box_refused is not None:
                detail = (
                    f"{frame_ref.capitalize()} shows no cards for this seat, "
                    "though the reader did see a bet box there and declined "
                    f"to read it ({unknown_read_text(box_refused)}). Confirm "
                    "this action happened before entering an amount for it."
                )
            elif box_read is not None:
                detail = (
                    f"{frame_ref.capitalize()} shows no cards for this seat, "
                    f"though the reader did read {box_read:g} BB in its bet "
                    "box there — that total belongs to an earlier action on "
                    "this street. Confirm this action happened before "
                    "entering an amount for it."
                )
            else:
                detail = (
                    f"{frame_ref.capitalize()} shows no cards for this seat, "
                    "so there was no bet box to read. Confirm this action "
                    "happened before entering an amount for it."
                )
        elif derivation.startswith("inferred"):
            detail = (
                f"{_inferred_reason(derivation)}, so no amount reading is "
                "tied to it. Work out the chips this seat added here from the "
                "neighboring frames and enter that in the Amount field below."
            )
        elif code_text is not None:
            detail = (
                f"On {frame_ref}, {code_text}. Read that frame and enter the "
                "chips this seat added on this action — the bet box shows "
                "the seat's running total for the street, not the increment."
            )
            if code == "below_calibrated_render_size":
                detail += (
                    " (For future recordings, capture the client at 1272×896 "
                    "or larger.)"
                )
        elif readable_bet is not None:
            detail = (
                f"{frame_ref.capitalize()} shows {readable_bet:g} BB total in "
                "this seat's bet box, but the share belonging to this action "
                "could not be isolated from it — work out the chips this seat "
                "added here and enter that below."
            )
        else:
            # Naming a frame that carries the number always beats a vaguer
            # claim, so the all-frames scan runs before the mid-hand branch —
            # and only that scan licenses an all-frames statement.
            elsewhere = _nearest_readable(states, frame_index, seat, "bets")
            if elsewhere is not None:
                value, at_index = elsewhere
                detail = (
                    f"This seat's bet box was not read on {frame_ref}. The "
                    f"reconstruction read {value:g} BB there on frame "
                    f"{at_index + 1} — that is the seat's running total for "
                    "the street, so work out the chips added on this action "
                    "and enter that below."
                )
            elif frame_index == 0 and _recording_starts_mid_hand(
                hand, recording_start_s
            ):
                detail = (
                    "The recording starts mid-hand and no frame shows this "
                    "seat's bet box, so this amount was never on screen. "
                    "Enter it below only if you can establish it another way."
                )
            else:
                detail = (
                    "No frame in this hand shows this seat's bet box, so the "
                    "amount was never on screen. Enter it below only if you "
                    "can establish it another way."
                )
        issues.append(
            ActionCvIssue(
                kind="Amount unknown", detail=detail, frame_index=frame_index
            )
        )

    if (
        db_stack_before is not None
        and action_type in MONEY_ACTION_TYPES
        and state is not None
        and seat is not None
    ):
        own_read = seat_value(state.get("stacks"), seat)
        # The premise is that the client's stack figure excludes chips already
        # in the bet box, so it only holds when the frame actually shows this
        # seat committing something. Without that, nothing moved and the
        # saved figure is simply the seat's stack.
        committed_here = (
            seat_value(state.get("bets"), seat) is not None
            or seat_refusal_code(state.get("bets_unknown"), seat) is not None
        ) and not _earlier_action_owns_the_box(
            hand, timeline_action, seat, timeline_action.get("street")
        )
        if (
            own_read is not None
            and committed_here
            and abs(own_read - db_stack_before) < 1e-6
        ):
            # The client's stack figure excludes chips already in the bet box,
            # so the action's own frame reads the stack AFTER this action.
            # Every branch that CITES a frame already refuses to use this one;
            # a saved value taken from it was never checked.
            issues.append(
                ActionCvIssue(
                    kind="Stack before looks post-action",
                    detail=(
                        f"The saved stack ({db_stack_before:g} BB) is what "
                        f"{frame_ref} reads after this seat's chips moved, so "
                        "it is the stack AFTER this action, not before it. "
                        "Re-read that frame and add back what this action "
                        "put in."
                    ),
                    frame_index=frame_index,
                )
            )

    if db_stack_before is None:
        stack_issue = _stack_before_issue(
            timeline_action,
            hand,
            states,
            state,
            seat=seat,
            frame_ref=frame_ref,
            frame_index=frame_index,
            db_amount=db_amount,
            db_action_type=action_type,
            recording_start_s=recording_start_s,
        )
        if stack_issue is not None:
            issues.append(stack_issue)

    if state is not None:
        if bool(state.get("coverage_gap")):
            gap_text = _format_seconds(state.get("prior_gap_s"))
            issues.append(
                ActionCvIssue(
                    kind="Coverage gap",
                    detail=(
                        f"No new readable table state was retained for "
                        f"{gap_text} before {frame_ref}, so an action in "
                        "between may be missing. Compare the neighboring "
                        "frames and add anything missed under 'Add a missing "
                        "action'."
                    ),
                    frame_index=frame_index,
                )
            )
        transitions = state.get("unmeasured_transitions") or []
        if seat is not None and seat in _int_set(transitions):
            issues.append(
                ActionCvIssue(
                    kind="Unmeasured transition",
                    detail=(
                        f"Seat {seat}'s committed chips could not be tracked "
                        f"continuously around {frame_ref}, so part of this "
                        "transition was not measured. Whatever amount this "
                        "line ends up with, check it against the frames on "
                        f"either side of {frame_ref}."
                    ),
                    frame_index=frame_index,
                )
            )
    return issues


def _stack_before_issue(
    timeline_action: dict[str, Any],
    hand: dict[str, Any],
    states: list[dict[str, Any]],
    state: dict[str, Any] | None,
    *,
    seat: int | None,
    frame_ref: str,
    frame_index: int | None,
    db_amount: float | None,
    db_action_type: str | None,
    recording_start_s: float | None,
) -> ActionCvIssue | None:
    """Explain a missing stack-before without asserting a mechanism we can't prove.

    Returns None when nothing in the timeline explains the gap — an unexplained
    hole stays visible through the empty field itself rather than earning a
    fabricated story. Every branch that asks the operator for a value points at
    a frame that actually carries one, or says plainly that none does.
    """

    kind = "Stack before unknown"

    def _hinted(detail: str, index: int | None = None) -> ActionCvIssue:
        """A stack issue whose value-offer flag always matches its hint."""
        return ActionCvIssue(
            kind=kind,
            detail=detail,
            frame_index=(
                hint_index
                if hint_index is not None
                else (index if index is not None else frame_index)
            ),
            offers_a_value=hint_offers_value,
        )

    row_type = str(
        db_action_type
        if db_action_type is not None
        else timeline_action.get("action_type") or ""
    ).replace("-", "_")
    nearest = _nearest_readable(states, frame_index, seat, "stacks", before_only=True)
    hint_value, hint_index = nearest if nearest is not None else (None, None)
    hint_stale = _seat_committed_between(hand, states, seat, hint_index, frame_index)
    timeline_stack = _optional_float(timeline_action.get("stack_before"))
    if timeline_stack is not None:
        # Point at the frame this figure was read from, and say plainly that
        # it cannot confirm itself: matching the number against the same OCR
        # read it came from has no diagnostic power, and would launder a
        # misread (this corpus contains a 10x decimal error) as verified.
        carrier = _frame_carrying_stack(states, seat, timeline_stack, frame_index)
        if carrier is not None:
            where = (
                f" That figure is the reader's own value for frame "
                f"{carrier + 1}, so it cannot confirm itself — read the stack "
                "off that frame yourself before entering it."
            )
        elif (
            state is not None
            and row_type in MONEY_ACTION_TYPES
            and (
                seat_value(state.get("bets"), seat) is not None
                or seat_refusal_code(state.get("bets_unknown"), seat) is not None
            )
            and seat_value(state.get("stacks"), seat) is not None
            and abs(
                (seat_value(state.get("stacks"), seat) or 0.0) - timeline_stack
            )
            < 1e-6
        ):
            # Re-offering this figure is what made clearing the field and
            # re-entering it loop forever.
            amount_note = (
                f" add this action's {_optional_float(timeline_action.get('amount')):g} BB back to it"
                if _optional_float(timeline_action.get("amount")) is not None
                else " add back whatever this action put in"
            )
            where = (
                f" That figure is what {frame_ref} reads AFTER this seat's "
                "chips moved, so it is not the stack before this action —"
                f"{amount_note}."
            )
        else:
            where = (
                " No earlier frame reads that value, so check it against the "
                "frames before entering it."
            )
        if "stack_ledger_incoherent" in (hand.get("warnings") or []):
            where += (
                " This hand's chip ledger does not balance, so treat the "
                "figure with extra suspicion."
            )
        lead = (
            f"This seat's stack before the action was not saved. The "
            f"reconstruction computed {timeline_stack:g} BB."
        )
        return ActionCvIssue(
            kind=kind,
            detail=(
                f"{lead}{where} Enter the result under **More fields → Stack "
                "before (BB)**."
            ),
            frame_index=carrier if carrier is not None else frame_index,
        )
    hint = _stack_field_hint(hint_index, hint_value, stale=hint_stale)
    hint_offers_value = hint_value is not None and hint_index is not None and not hint_stale
    derivation = str(timeline_action.get("derivation") or "")
    if derivation.startswith("inferred"):
        if _line_may_not_belong(states, state, seat, frame_index):
            # The frame that produced this line shows no sign of the seat: the
            # honest reading is that the line may not belong here, so do not
            # push the operator to legitimize it with a number. Deliberately
            # phrased as a check to run, not an instruction to delete — a
            # wrong accusation here destroys real data.
            last_held = _last_frame_holding_cards(states, seat, frame_index)
            if last_held is not None:
                # Say what the frames actually establish: the seat was dealt
                # in and later had no cards, which is a fold. Claiming it "may
                # never have been in the hand" is disproved by frame 1 and
                # invites the operator to dismiss a correct warning.
                evidence = (
                    f"This seat held cards through frame {last_held + 1} and "
                    f"none by {frame_ref}, so it had already folded and "
                    "cannot have acted here."
                )
            else:
                evidence = (
                    f"No frame up to {frame_ref} shows this seat holding "
                    "cards, so it may not have been in the hand at all."
                )
            return ActionCvIssue(
                kind=ACTION_MAY_NOT_BELONG,
                detail=(
                    "This line was not observed — it was inferred because the "
                    f"betting round completed. {evidence} Check {frame_ref}, "
                    "then delete this line; if the seat's fold is not "
                    "recorded either, add it under 'Add a missing action'."
                ),
                frame_index=frame_index,
            )
        return _hinted(
            f"{_inferred_reason(derivation)}, so the import did not attach "
            f"a stack to it. {hint}"
        )
    # A missing stack-before is usually caused by reads on EARLIER frames
    # (that is where the value would have come from), so scan backward for
    # the seat's most recent refusal before giving up.
    refusal = _latest_stack_refusal(states, frame_index, seat)
    if refusal is not None:
        code_text, at_index = refusal
        return _hinted(f"On frame {at_index + 1}, {code_text}. {hint}", at_index)
    starting_code = _starting_stack_unknown_code(hand, seat)
    if starting_code is not None:
        committed = starting_code.startswith("committed_at_start")
        unrecorded = starting_code == "unknown"
        if (committed or unrecorded) and _recording_starts_mid_hand(
            hand, recording_start_s
        ):
            detail = (
                "The recording starts mid-hand: this seat had already "
                "committed chips before the first frame, and those could not "
                f"be sized. {hint}"
            )
        elif committed:
            # The stack itself read fine; the chips already on the felt when
            # the hand opened could not be sized. Saying "never established
            # cleanly" here is disprovable on sight.
            detail = (
                "This seat's stack read fine, but the chips already in front "
                "of it when the hand opened could not be sized, so its "
                f"starting stack is unknown. {hint}"
            )
        elif unrecorded:
            detail = (
                "This seat's starting stack was never established cleanly "
                f"during the hand. {hint}"
            )
        else:
            code_text = unknown_read_text(starting_code) or "the read was refused"
            detail = (
                f"This seat's starting stack could not be read ({code_text}), "
                f"so the stack before this action could not be back-computed. "
                f"{hint}"
            )
        return _hinted(detail)
    action_type = str(
        db_action_type
        if db_action_type is not None
        else timeline_action.get("action_type") or ""
    ).replace("-", "_")
    if db_amount is None and action_type in MONEY_ACTION_TYPES:
        return _hinted(
            "The import could not derive this seat's stack before the "
            f"action, because this line's own amount is unknown. {hint}"
        )
    return None


def _latest_stack_refusal(
    states: list[dict[str, Any]],
    frame_index: int | None,
    seat: int | None,
) -> tuple[str, int] | None:
    """Walk backward from the source frame to the seat's most recent stack read.

    Returns the refusal explanation and its frame index, or None once a
    readable stack is found first (then the hole is not a per-frame failure).
    """

    if frame_index is None or seat is None:
        return None
    # Strictly before the action: the action's own frame shows state after
    # the chips moved, so it is never where stack-before would have come from.
    for index in range(min(frame_index, len(states)) - 1, -1, -1):
        state = states[index]
        code_text = unknown_read_text(
            seat_refusal_code(state.get("stacks_unknown"), seat)
        )
        if code_text is not None:
            return code_text, index
        if seat_value(state.get("stacks"), seat) is not None:
            return None
    return None


def _recording_starts_mid_hand(
    hand: dict[str, Any], recording_start_s: float | None = None
) -> bool:
    """True only when this hand was already underway at recording start.

    ``starting_stack_unknown`` alone is NOT such a signal — it also fires when
    any seat's starting-stack read is refused, which happens on ordinary hands.
    Only the hand that begins at the recording's first sampled second can
    predate the recording; ``recording_start_s`` defaults to 0.0 (sampling
    always starts there today) but callers with the full timeline should pass
    its first state's time so a lobby-opening recording stays honest.
    """

    try:
        t_start = float(hand.get("t_start", -1.0))
    except (TypeError, ValueError):
        return False
    start = 0.0 if recording_start_s is None else float(recording_start_s)
    warnings = hand.get("warnings") or []
    return t_start == start and "starting_stack_unknown" in warnings


def seat_value(mapping: Any, seat: int | None) -> float | None:
    """Fetch a per-seat numeric read from a JSON dict keyed by str or int seat."""

    if seat is None or not isinstance(mapping, dict):
        return None
    return _optional_float(mapping.get(str(seat), mapping.get(seat)))


def seat_holds_cards(state: dict[str, Any] | None, seat: int | None) -> bool:
    """Whether this seat still holds cards on the frame.

    ``dealt_in`` counts face-DOWN card backs only. At showdown the client
    flips villain cards face up and they move to ``villain_cards``, and a
    hand's terminal frame clears the table entirely — so absence from
    ``dealt_in`` alone is not evidence that a seat was out of the hand.
    Fails open: an unreadable frame must never be read as proof of absence.
    """

    if state is None or seat is None:
        return True
    dealt_in = state.get("dealt_in")
    if not isinstance(dealt_in, list):
        return True
    if seat in _int_set(dealt_in):
        return True
    villain_cards = state.get("villain_cards")
    if isinstance(villain_cards, dict):
        shown = villain_cards.get(str(seat), villain_cards.get(seat))
        # A whole hole-card pair is evidence of a live seat; a single card is
        # not — deal animations put in-flight board cards in this field and
        # attribute them to whichever seat they pass.
        if isinstance(shown, list) and len(shown) >= 2:
            return True
    return False


def _format_seconds(value: Any) -> str:
    """Render a gap length, degrading rather than raising on bad JSON."""

    try:
        return f"{float(value):g}s"
    except (TypeError, ValueError):
        return "several seconds"


def _optional_float(value: Any) -> float | None:
    """Coerce a JSON number, returning None rather than raising."""

    try:
        parsed = None if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is not None and not math.isfinite(parsed):
        return None
    return parsed


def _int_set(values: Any) -> set[int]:
    """Coerce a JSON list of seat numbers, skipping anything non-numeric."""

    result: set[int] = set()
    if not isinstance(values, (list, tuple, set)):
        return result
    for item in values:
        try:
            result.add(int(item))
        except (TypeError, ValueError, OverflowError):
            continue
    return result


def _frame_carrying_stack(
    states: list[dict[str, Any]],
    seat: int | None,
    value: float,
    frame_index: int | None,
) -> int | None:
    """Latest frame STRICTLY BEFORE the action whose stack read equals ``value``.

    Bounded deliberately: a stack often returns to the same figure later in a
    hand, so an unbounded scan offers the terminal settlement frame as
    evidence for a preflop stack. Nothing is lost by bounding — every value
    reachable later is also readable at or before the action.
    """

    if seat is None:
        return None
    # Strictly before: the action's own frame reads the stack AFTER the chips
    # moved, so it is never evidence for a stack-before.
    last = len(states) if frame_index is None else min(frame_index, len(states))
    for index in range(last - 1, -1, -1):
        read = seat_value(states[index].get("stacks"), seat)
        if read is not None and abs(read - value) < 1e-6:
            return index
    return None


def _last_frame_holding_cards(
    states: list[dict[str, Any]],
    seat: int | None,
    frame_index: int | None,
) -> int | None:
    """Latest frame before ``frame_index`` where this seat still held cards."""

    if seat is None:
        return None
    last = len(states) if frame_index is None else min(frame_index, len(states))
    for index in range(last - 1, -1, -1):
        if seat_holds_cards(states[index], seat):
            return index
    return None


def _line_may_not_belong(
    states: list[dict[str, Any]],
    state: dict[str, Any] | None,
    seat: int | None,
    frame_index: int | None,
) -> bool:
    """Whether an inferred line has no evidence its seat was in the hand.

    Deliberately conservative: this drives an offer to delete an action, and a
    false positive destroys real data. A hand's terminal frame has already
    cleared the table, so absence there proves nothing.
    """

    if seat_holds_cards(state, seat):
        return False
    if frame_index is not None and frame_index >= len(states) - 1:
        return False
    return True


def _nearest_readable(
    states: list[dict[str, Any]],
    frame_index: int | None,
    seat: int | None,
    field: str,
    *,
    before_only: bool = False,
) -> tuple[float, int] | None:
    """Nearest frame to ``frame_index`` whose ``field`` read this seat, if any.

    Searches outward so the operator is sent to the closest legible evidence
    rather than to the frame that by definition could not be read. Returns the
    value and its frame index, or None when no frame in the hand shows it.

    ``before_only`` restricts the search to frames strictly before the action.
    A stack read on the action's own frame is the stack AFTER the chips moved,
    so offering it as "the stack before this action" states a wrong number.
    """

    if seat is None or not states:
        return None
    origin = 0 if frame_index is None else frame_index
    indexes = range(len(states))
    if before_only:
        indexes = range(min(origin, len(states)))
    order = sorted(indexes, key=lambda index: (abs(index - origin), index))
    for index in order:
        value = seat_value(states[index].get(field), seat)
        if value is not None:
            return value, index
    return None


def _earlier_action_owns_the_box(
    hand: dict[str, Any],
    timeline_action: dict[str, Any],
    seat: int | None,
    street: Any,
) -> bool:
    """Whether chips already in this seat's box came from an earlier action.

    The post-action claim rests on the client's stack figure excluding chips
    in the bet box. That only identifies THIS action when nothing earlier on
    the street put them there — otherwise the box holds a blind post or a
    previous bet, and the frame's stack read is simply the seat's stack.
    """

    if seat is None:
        return False
    index = timeline_action.get("action_index")
    street_key = str(street or "").lower()
    for line in hand.get("actions") or []:
        if line.get("seat") != seat or line is timeline_action:
            continue
        if str(line.get("street") or "").lower() != street_key:
            continue
        other = line.get("action_index")
        if isinstance(index, int) and isinstance(other, int) and other < index:
            return True
    return False


def _seat_committed_between(
    hand: dict[str, Any],
    states: list[dict[str, Any]],
    seat: int | None,
    from_index: int | None,
    to_index: int | None,
) -> bool:
    """Whether this seat put chips in between two frames.

    A stack read from before an intervening bet is not the stack before THIS
    action, so a hint drawn from it must say so instead of naming a number
    the operator would enter verbatim.
    """

    if seat is None or from_index is None or to_index is None:
        return False
    # Strictly between: the action's own frame is where THIS action happens,
    # so money attributed to it must not count as an intervening commitment.
    images = {
        str(states[index].get("image") or "")
        for index in range(from_index + 1, min(to_index, len(states)))
    }
    for action in hand.get("actions") or []:
        if action.get("seat") != seat:
            continue
        if str(action.get("action_type") or "").replace("-", "_") not in (
            MONEY_ACTION_TYPES
        ):
            continue
        if str(action.get("source_image") or "") in images:
            return True
    return False


def _stack_field_hint(
    frame_index: int | None,
    value: float | None,
    *,
    stale: bool = False,
) -> str:
    """Point at the control that actually holds this field, by its real path.

    Attributes the number to the reader rather than to the image: these values
    come from OCR, which does misread (a 10x decimal error exists in this
    corpus), so claiming "the frame shows X" overstates what is known.
    ``stale`` marks a reading taken before this seat committed more chips, so
    the operator is not invited to enter it verbatim.
    """

    where = "under **More fields → Stack before (BB)**"
    if value is not None and frame_index is not None:
        if stale:
            return (
                f"The nearest earlier reading is {value:g} BB on frame "
                f"{frame_index + 1}, but this seat has put chips in since "
                "then, so that is not the stack before this action — work it "
                f"out from the frames and enter it {where}."
            )
        return (
            f"The reconstruction read {value:g} BB for this seat on frame "
            f"{frame_index + 1} — check that frame and enter the value {where}."
        )
    return (
        "No frame before this action shows this seat's stack, so leave the "
        f"field empty unless you can establish it; it lives {where}."
    )


def _starting_stack_unknown_code(hand: dict[str, Any], seat: int | None) -> str | None:
    """Why this seat's starting stack is unknown, per the timeline's own code.

    The reconstruction distinguishes "the stack read was refused" from
    "the stack read fine but the chips already committed could not be sized"
    (``committed_at_start_*``). Collapsing the two produces a message the
    operator can disprove by looking at the frame.
    """

    if seat is None:
        return None
    for player in hand.get("players") or []:
        if player.get("seat") != seat:
            continue
        if player.get("starting_stack") is not None:
            return None
        code = player.get("starting_stack_unknown")
        return str(code) if code else "unknown"
    return None


def _review_status(review: Any) -> str | None:
    if review is None:
        return None
    status = getattr(review, "status", None)
    if status is None and isinstance(review, dict):
        status = review.get("status")
    return str(status) if status is not None else None


def _review_issue_types(review: Any) -> list[str]:
    if review is None:
        return []
    raw = getattr(review, "issue_types", None)
    if raw is None and isinstance(review, dict):
        raw = review.get("issue_types")
    if not raw:
        return []
    return [str(item) for item in raw]


def _review_notes(review: Any) -> str:
    if review is None:
        return ""
    raw = getattr(review, "notes", None)
    if raw is None and isinstance(review, dict):
        raw = review.get("notes")
    return str(raw or "").strip()


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
        label = STREET_BY_BOARD_COUNT.get(board_len)
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
