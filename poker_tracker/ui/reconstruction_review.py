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


def _seat_code(mapping: Any, seat: int | None) -> str | None:
    """Fetch a per-seat refusal code from a JSON dict keyed by str or int seat."""

    if seat is None or not isinstance(mapping, dict):
        return None
    value = mapping.get(str(seat), mapping.get(seat))
    return str(value) if value else None


def _unknown_code_text(code: str | None) -> str | None:
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
) -> list[ActionCvIssue]:
    """Explain which CV read failures affect one action line, tied to its frame.

    ``db_amount`` / ``db_stack_before`` are the current saved values: once the
    operator fills a field in, the matching issue stops being reported.
    ``recording_start_s`` is the whole recording's first sampled second (not
    this hand's) so mid-hand claims survive recordings that open on a lobby.
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
    seat = None if seat_raw is None else int(seat_raw)
    action_type = str(timeline_action.get("action_type") or "").replace("-", "_")
    frame_ref = (
        f"frame {frame_index + 1}" if frame_index is not None else "an unretained frame"
    )
    issues: list[ActionCvIssue] = []

    derivation = str(timeline_action.get("derivation") or "")
    if db_amount is None and action_type in MONEY_ACTION_TYPES:
        timeline_amount = timeline_action.get("amount")
        code = _seat_code(state.get("bets_unknown") if state else None, seat)
        code_text = _unknown_code_text(code)
        readable_bet = _seat_value(state.get("bets") if state else None, seat)
        if timeline_amount is not None:
            detail = (
                f"The reconstruction read {float(timeline_amount):g} BB as the "
                "chips this seat added here, but the saved amount is empty — "
                f"confirm it against {frame_ref} (whose bet box shows the "
                "seat's total for the street) and re-enter it in the Amount "
                "field below."
            )
        elif not _seat_holds_cards(state, seat):
            detail = (
                f"{frame_ref.capitalize()} does not show this seat holding "
                "cards, so this line may not belong to the hand at all. Open "
                "that frame and delete this action if the seat was already "
                "out; otherwise enter the amount below."
            )
        elif derivation.startswith("inferred"):
            detail = (
                "This line was inferred from how the betting round completed "
                "rather than observed on a frame, so no amount reading is "
                "tied to it. Work the amount out from the neighboring frames "
                "and enter it below."
            )
        elif code_text is not None:
            detail = (
                f"On {frame_ref}, {code_text}. Read the amount off that frame "
                "and enter the chips this seat added in the Amount field "
                "below."
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
                    f"This seat's bet box was not read on {frame_ref}. Frame "
                    f"{at_index + 1} shows {value:g} BB there — work out the "
                    "chips added on this action and enter that below."
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
            recording_start_s=recording_start_s,
        )
        if stack_issue is not None:
            issues.append(stack_issue)

    if state is not None:
        if bool(state.get("coverage_gap")):
            gap = state.get("prior_gap_s")
            gap_text = f"{float(gap):g}s" if gap is not None else "several seconds"
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
        if seat is not None and seat in {int(item) for item in transitions}:
            issues.append(
                ActionCvIssue(
                    kind="Unmeasured transition",
                    detail=(
                        f"Seat {seat}'s committed chips could not be tracked "
                        f"continuously around {frame_ref}, so part of this "
                        "transition was not measured. Check this line's "
                        "amount against the frames on either side of "
                        f"{frame_ref} and correct it in the Amount field "
                        "below if it is wrong."
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
    recording_start_s: float | None,
) -> ActionCvIssue | None:
    """Explain a missing stack-before without asserting a mechanism we can't prove.

    Returns None when nothing in the timeline explains the gap — an unexplained
    hole stays visible through the empty field itself rather than earning a
    fabricated story. Every branch that asks the operator for a value points at
    a frame that actually carries one, or says plainly that none does.
    """

    kind = "Stack before unknown"
    timeline_stack = timeline_action.get("stack_before")
    if timeline_stack is not None:
        return ActionCvIssue(
            kind=kind,
            detail=(
                f"The reconstruction computed {float(timeline_stack):g} BB "
                "for this seat's stack before the action, but the saved field "
                "is empty — confirm it and re-enter it under **More fields → "
                "Stack before (BB)**."
            ),
            frame_index=frame_index,
        )
    nearest = _nearest_readable(states, frame_index, seat, "stacks")
    hint_value, hint_index = nearest if nearest is not None else (None, None)
    hint = _stack_field_hint(hint_index, hint_value)
    derivation = str(timeline_action.get("derivation") or "")
    if derivation.startswith("inferred"):
        if not _seat_holds_cards(state, seat):
            # The frame that produced this line shows the seat holding no
            # cards: the honest reading is that the line may not belong here,
            # so do not push the operator to legitimize it with a number.
            return ActionCvIssue(
                kind=kind,
                detail=(
                    f"This line was inferred from the betting round "
                    f"completing, but {frame_ref} does not show this seat "
                    "holding cards — it may not belong to the hand. Open that "
                    "frame and delete this action if the seat was already "
                    "out."
                ),
                frame_index=frame_index,
            )
        return ActionCvIssue(
            kind=kind,
            detail=(
                "This line was inferred from the betting round completing, so "
                f"the import did not attach a stack to it. {hint}"
            ),
            frame_index=hint_index if hint_index is not None else frame_index,
        )
    # A missing stack-before is usually caused by reads on EARLIER frames
    # (that is where the value would have come from), so scan backward for
    # the seat's most recent refusal before giving up.
    refusal = _latest_stack_refusal(states, frame_index, seat)
    if refusal is not None:
        code_text, at_index = refusal
        return ActionCvIssue(
            kind=kind,
            detail=f"On frame {at_index + 1}, {code_text}. {hint}",
            frame_index=hint_index if hint_index is not None else at_index,
        )
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
            code_text = _unknown_code_text(starting_code) or "the read was refused"
            detail = (
                f"This seat's starting stack could not be read ({code_text}), "
                f"so the stack before this action could not be back-computed. "
                f"{hint}"
            )
        return ActionCvIssue(
            kind=kind,
            detail=detail,
            frame_index=hint_index if hint_index is not None else frame_index,
        )
    action_type = str(timeline_action.get("action_type") or "").replace("-", "_")
    if db_amount is None and action_type in MONEY_ACTION_TYPES:
        return ActionCvIssue(
            kind=kind,
            detail=(
                "The import could not derive this seat's stack before the "
                f"action, because this line's own amount is unknown. {hint}"
            ),
            frame_index=hint_index if hint_index is not None else frame_index,
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
    for index in range(min(frame_index, len(states) - 1), -1, -1):
        state = states[index]
        code_text = _unknown_code_text(
            _seat_code(state.get("stacks_unknown"), seat)
        )
        if code_text is not None:
            return code_text, index
        if _seat_value(state.get("stacks"), seat) is not None:
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


def _seat_value(mapping: Any, seat: int | None) -> float | None:
    """Fetch a per-seat numeric read from a JSON dict keyed by str or int seat."""

    if seat is None or not isinstance(mapping, dict):
        return None
    value = mapping.get(str(seat), mapping.get(seat))
    return None if value is None else float(value)


def _seat_holds_cards(state: dict[str, Any] | None, seat: int | None) -> bool:
    """Whether this seat was dealt in on the frame — i.e. still in the hand."""

    if state is None or seat is None:
        return True
    dealt_in = state.get("dealt_in")
    if not isinstance(dealt_in, list):
        return True
    return int(seat) in {int(item) for item in dealt_in}


def _nearest_readable(
    states: list[dict[str, Any]],
    frame_index: int | None,
    seat: int | None,
    field: str,
) -> tuple[float, int] | None:
    """Nearest frame to ``frame_index`` whose ``field`` read this seat, if any.

    Searches outward so the operator is sent to the closest legible evidence
    rather than to the frame that by definition could not be read. Returns the
    value and its frame index, or None when no frame in the hand shows it.
    """

    if seat is None or not states:
        return None
    origin = 0 if frame_index is None else frame_index
    order = sorted(range(len(states)), key=lambda index: (abs(index - origin), index))
    for index in order:
        value = _seat_value(states[index].get(field), seat)
        if value is not None:
            return value, index
    return None


def _stack_field_hint(frame_index: int | None, value: float | None) -> str:
    """Point at the control that actually holds this field, by its real path."""

    where = "under **More fields → Stack before (BB)**"
    if value is not None and frame_index is not None:
        return (
            f"Frame {frame_index + 1} shows this seat at {value:g} BB — "
            f"confirm it and enter it {where}."
        )
    return (
        "No frame in this hand shows this seat's stack, so leave the field "
        f"empty unless you know the value; it lives {where}."
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
