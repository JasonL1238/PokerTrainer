"""Export YOLO card timelines as PokerTrainer draft session imports.

This is an offline, post-session bridge. It does not write to the app database
unless --import-db is explicitly provided, and by default it only exports hands
whose card fields satisfy the app's Hand validation rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv_lab.scripts.eval.validate_yolo_card_timeline import (  # noqa: E402
    WARNING_SEVERITY,
    validate_timeline,
)
from poker_tracker.math.cards import CardParseError, parse_visible_cards  # noqa: E402
from poker_tracker.persistence.completion import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    BoundaryEvidence,
    CompletionEvidence,
    derive_completion_status,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase  # noqa: E402
from poker_tracker.persistence.import_export import EXPORT_VERSION, import_session  # noqa: E402
from poker_tracker.persistence.models import Action, Hand, HandPlayer, Session  # noqa: E402

DEFAULT_TIMELINE = "cv_lab/results/yolo_card_timeline_card_changes_v1.json"
DEFAULT_OUT = "cv_lab/results/yolo_card_hands_draft_session.json"
VALID_BOARD_COUNTS = {0, 3, 4, 5}
# Validator codes that mean "cards are incomplete/invalid" rather than "the
# reconstruction is wrong". When include_incomplete is set, these alone must not
# block draft import — soft blanking + operator fill-in is the recovery path.
DRAFT_CARD_WARNING_CODES = frozenset(
    {
        "invalid_hero_count",
        "invalid_board_count",
        "hero_cards_not_two",
        "duplicate_visible_cards",
    }
)
# Operator-fillable reconstruction gaps. include_incomplete drafts keep these in
# completion_evidence, but must not refuse import — otherwise small-window OCR
# refusals and mid-hand coverage holes leave the session with 0 hands while the
# timeline still has studyable hero-preflop segments.
DRAFT_OPERATOR_FILLABLE_CODES = frozenset(
    {
        "amounts_unknown_in_ledger",
        "starting_stack_unknown",
        "mid_hand_coverage_gap",
        "hero_card_identity_split",
    }
)
HAND_CORRECTION_FIELDS = ["hand_number", "hero_cards", "board_cards", "action", "notes"]


def _card_to_app(token: str) -> str:
    """Convert detector labels like AS/10H/td to app cards like As/Th/Td."""
    text = token.strip()
    if not text:
        raise ValueError("empty card token")
    rank_text = text[:-1].upper()
    suit = text[-1].lower()
    rank = "T" if rank_text in {"10", "T"} else rank_text
    return f"{rank}{suit}"


def _cards_to_app(cards: list[str] | None) -> str:
    return " ".join(_card_to_app(card) for card in (cards or []))


def _split_card_text(value: str) -> list[str]:
    text = value.strip().replace(",", " ").replace("/", " ").replace("-", " ")
    if not text:
        return []
    tokens = text.split()
    if len(tokens) == 1 and len(tokens[0]) > 2:
        compact = tokens[0]
        if len(compact) % 2 != 0:
            raise ValueError(f"invalid compact card text: {value}")
        return [compact[index:index + 2] for index in range(0, len(compact), 2)]
    return tokens


def load_hand_corrections(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    corrections: dict[int, dict[str, str]] = {}
    for row in rows:
        hand_number = row.get("hand_number", "").strip()
        if not hand_number:
            continue
        corrections[int(hand_number)] = row
    return corrections


def apply_hand_corrections(timeline: dict[str, Any], corrections: dict[int, dict[str, str]]) -> dict[str, Any]:
    if not corrections:
        return timeline
    updated = json.loads(json.dumps(timeline))
    for hand in updated.get("hands", []):
        hand_number = hand.get("hand_number")
        correction = corrections.get(hand_number)
        if not correction:
            continue
        action = correction.get("action", "").strip().lower()
        hand["manual_correction"] = {
            "action": action or "keep",
            "notes": correction.get("notes", ""),
        }
        if action in {"drop", "delete", "remove", "skip"}:
            hand["drop_from_export"] = True
            continue
        if correction.get("hero_cards", "").strip():
            hand["hero"] = _split_card_text(correction["hero_cards"])
        if correction.get("board_cards", "").strip():
            hand["board"] = _split_card_text(correction["board_cards"])
        cards = (hand.get("hero") or []) + (hand.get("board") or [])
        hand["complete_cards"] = (
            len(hand.get("hero") or []) == 2
            and len(hand.get("board") or []) in VALID_BOARD_COUNTS
            and len(cards) == len(set(cards))
        )
        # Appended, never replaced. A CSV row is an operator note, not a new
        # reconstruction: replacing the list destroyed every rejection code the
        # row never touched, and a rejection clears only by producing new
        # evidence without it. A card edit the row *did* make is re-derived into
        # complete_cards above, and its card-count code stays as an
        # acknowledgeable warning the operator can accept in the app.
        warnings = list(hand.get("warnings") or [])
        if "manual_hand_correction" not in warnings:
            warnings.append("manual_hand_correction")
        hand["warnings"] = warnings
    return updated


# Severity per reconstruction fact: the fraction of a hand's trustworthiness the
# fact destroys, not a warning count. A fact that makes the reconstruction WRONG
# (rather than incomplete) is >= 0.5, so any one of them alone puts the hand under
# the 0.8 LOW_CONFIDENCE line and far under any review-free threshold.
#
# The validator's own table is the base, not a parallel copy. This export unions
# the validator's findings into the completion evidence, and an unrecognised code
# is classified as a rejection -- so nine of the validator's thirteen codes, its
# two mildest included, used to become permanent unclearable rejections purely
# because two independently maintained tables had drifted apart.
_EXPORT_ONLY_SEVERITY: dict[str, float] = {
    "pot_not_reconciled": 0.35,
    "hero_cards_not_two": 0.30,
    "pot_text_dropped": 0.15,
    "manual_hand_correction": 0.00,   # a human looked at it
}
SEVERITY: dict[str, float] = {**WARNING_SEVERITY, **_EXPORT_ONLY_SEVERITY}
_UNKNOWN_SEVERITY = 0.30
# Unread HUD amounts per observed state, above which a hand's numbers rest on
# materially less evidence than an ordinary one (see _confidence_for_hand).
_AMOUNTS_UNKNOWN_RATE = 0.5

# The confidence at or below which a hand is tagged LOW_CONFIDENCE for review.
#
# One constant, two consumers, because they were silently inconsistent: the
# evidence-gap caps in _confidence_for_hand clamp to exactly 0.80 and the tag test
# was a STRICT `confidence < 0.8`, so a hand capped for the very reason the cap
# exists -- an unobserved start or end, or a high rate of unreadable HUD amounts --
# landed on precisely the one value the tag excluded. Every unflagged case in the
# development corpus was one of these: g0723b hand 1 (partial_start) and g0723a
# hand 4 (partial_end, terminal_event "unobserved") both exported at
# confidence_score 0.8 with tags []. The comparison is inclusive now, so a capped
# hand is always tagged.
LOW_CONFIDENCE_AT = 0.80


def _boundary_flags(
    hand: dict[str, Any], *, preceded_by_hand: bool, followed_by_hand: bool
) -> tuple[bool, bool, str]:
    """(partial_start, partial_end, terminal_event) -- ONE definition, two callers.

    The start is proven only when the segmenter cut this hand at a detected fresh
    deal; the first hand of a recording began before the first frame. The end is
    proven either by a following hand's detected deal or by having observed this
    hand's own terminal event.
    """
    terminal_event = str(hand.get("terminal_event") or "unobserved")
    terminal_observed = terminal_event in {"showdown", "fold_win", "hero_fold"}
    return (not preceded_by_hand), (not (followed_by_hand or terminal_observed)), terminal_event


def _field_codes(hand: dict[str, Any]) -> set[str]:
    """Warning codes derived from spine FIELDS rather than the warnings list.

    `pot_text_dropped` is a boolean the spine sets when the pot consensus
    OVERRIDES the pot text -- two derived estimators agreeing against the one
    direct read. That is a designed outcome (the text freezes stale
    mid-betting), not a fatal one, which is why it is a field and not a spine
    warning: a spine warning blocks the export gate outright, and blocking
    every legitimate stale-text override would disable the consensus mechanism
    it reports on. But it was also consumed by NOTHING: the severity table
    priced it at 0.15 and the validator's vocabulary listed it while no
    producer ever attached the code to a hand, so 'text confirmed the pot' and
    'text was outvoted' exported identically. It now enters the same two
    channels every other code uses -- the confidence deduction and the exported
    warning_codes -- at the severity the table already declared.
    """
    return {"pot_text_dropped"} if hand.get("pot_text_dropped") else set()


def _confidence_for_hand(
    hand: dict[str, Any],
    validation_codes: list[str] | None = None,
    *,
    partial_start: bool = False,
    partial_end: bool = False,
    terminal_observed: bool = True,
) -> float:
    """Confidence in the RECONSTRUCTION, not in the detections.

    A hand starts at 1.0 only if every reconstruction invariant holds, and can
    never exceed 0.60 while the record is incomplete, however clean the boxes
    were. This replaces a 0.95-minus-0.15-per-warning formula that read only the
    warning list, the card counts and the hero -- so a session whose every board
    had been destroyed exported at 0.95 per hand with no tag, because losing a
    board raised no warning and an empty board is a legal card count.

    ``validation_codes`` are the validator's findings for this hand. They are
    unioned with the spine's own warnings so that a hand exported under
    --allow-validation-warnings still carries an honest score: the operator
    overrode the GATE, not the facts.
    """
    codes = (set(hand.get("warnings", [])) | set(validation_codes or [])
             | _field_codes(hand))
    deductions = sum(SEVERITY.get(w, _UNKNOWN_SEVERITY) for w in codes)
    confidence = 1.0 - deductions
    if not hand.get("complete_cards"):
        confidence = min(confidence, 0.60)
    if not hand.get("hero"):
        confidence = min(confidence, 0.40)
    if len(hand.get("board") or []) not in VALID_BOARD_COUNTS:
        confidence = min(confidence, 0.40)
    # Positive evidence required, not merely an absence of complaints: a hand with
    # no pot, no actions, no reconciliation, or an unresolved outcome is capped
    # regardless of how clean its cards look.
    if hand.get("pot") is None or not hand.get("actions"):
        confidence = min(confidence, 0.50)
    if not hand.get("reconciled"):
        confidence = min(confidence, 0.55)
    if hand.get("winner_seat") is None and not hand.get("hero_folded"):
        confidence = min(confidence, 0.55)
    states = hand.get("n_states") or 0
    missing = hand.get("anchor_missing_states") or 0
    if states and missing / states > 0.25:
        confidence = min(confidence, 0.45)
    # A boundary the segmenter never observed is missing evidence, exactly like a
    # missing pot or an unresolved winner above. The 06-21 recording opens with the
    # flop already dealt and 18.1 BB in the pot -- so that hand has no preflop
    # street at all and its players rows publish mid-flop stacks under the field
    # name `starting_stack` -- and it scored 1.0 with tags []. completion_status
    # caught it; the number a reviewer actually reads on the hand did not.
    # An unread terminal event is the same kind of gap on the other boundary: the
    # hand may be bracketed by neighbouring deals and still have no observed
    # resolution of its own.
    if partial_start or partial_end or not terminal_observed:
        confidence = min(confidence, LOW_CONFIDENCE_AT)
    # Unreadable HUD amounts are the same kind of gap. The counter was recorded,
    # serialized as cv_amounts_unknown, and read by nothing: a hand with 18 failed
    # numeric reads scored the same 1.0 as one with none. Rate, not count, so the
    # score does not depend on how many frames the hand happened to span.
    # Measured per hand over the development corpus: 18 of 21 hands sit at
    # 0.06-0.29 unread reads per state, the other three at 0.57, 0.86 and 1.23.
    if states and (hand.get("amounts_unknown") or 0) / states >= _AMOUNTS_UNKNOWN_RATE:
        confidence = min(confidence, LOW_CONFIDENCE_AT)
    # Two more gaps of exactly the same kind, recorded by the spine and read by
    # nothing until now -- each reached the export only as a cv_* field on the
    # completion evidence, so a hand carrying one scored the same 1.0 as a hand
    # carrying none:
    #   stack_conflicts             a seat's two stack_text boxes CONTRADICTED each
    #                               other, so its stack was discarded to unknown;
    #   bet_conflicts               the same fact on the BET channel, which until
    #                               the round-3 repair had no conflict rule at all
    #                               and resolved two boxes by detector list order
    #                               (65 development frames, and in all 65 the
    #                               winner was not the highest-confidence box);
    #   stack_outlier_checks_skipped  a frame had too few stack reads for the
    #                               sibling-median net (a LEGACY field: the spine
    #                               no longer produces it, but a timeline built by
    #                               an earlier build carries it and still earns
    #                               the cap);
    #   pot_text_off_column         a pot box was thrown out by the centre-column
    #                               guard -- not the same fact as "no pot detected".
    # The comments that introduced all three say they exist so the state can be
    # told apart downstream. This is downstream.
    #   settle_scan_skipped         the settlement scan could not examine one or
    #                               more of the hand's own transitions because the
    #                               pot read there was refused -- the winner and
    #                               t_end were chosen from a scan with known blind
    #                               spots (g0723a hand 5 exported with SIX skipped
    #                               transitions at confidence 1.0, tags []).
    if any(hand.get(key) for key in ("stack_conflicts", "bet_conflicts",
                                     "stack_outlier_checks_skipped",
                                     "pot_text_off_column", "settle_scan_skipped")):
        confidence = min(confidence, LOW_CONFIDENCE_AT)
    # A transition nothing could measure, or a money action of unknown size. The
    # hand is rejected outright by amounts_unknown_in_ledger, but a hand exported
    # past the gate (--allow-validation-warnings, or a manual correction) must
    # still carry an honest score.
    if hand.get("unmeasured_transitions") or hand.get("unknown_money_actions"):
        confidence = min(confidence, LOW_CONFIDENCE_AT)
    return max(0.0, min(1.0, round(confidence, 3)))


def _hand_notes(hand: dict[str, Any], *, timeline_path: Path) -> str:
    warnings = ", ".join(hand.get("warnings", [])) or "none"
    sources = ", ".join(hand.get("source_images", [])[:6])
    if len(hand.get("source_images", [])) > 6:
        sources += ", ..."
    notes = (
        "CV draft from YOLO card timeline. "
        f"timeline={timeline_path}; timeline_hand_number={hand.get('hand_number')}; "
        f"t={hand.get('t_start')}..{hand.get('t_end')}; "
        f"warnings={warnings}; source_images={sources}"
    )
    manual = hand.get("manual_correction")
    if manual:
        notes += f"; manual_correction={manual.get('action', 'keep')}; manual_notes={manual.get('notes', '')}"
    return notes


def timeline_hand_number_from_notes(notes: str | None) -> int | None:
    """Parse ``timeline_hand_number=<n>`` from CV hand notes when present."""
    if not notes:
        return None
    match = re.search(r"timeline_hand_number=(\d+)", notes)
    if match is None:
        return None
    return int(match.group(1))


# A code whose severity says the reconstruction is WRONG (>= 0.5) is the pipeline
# refusing the hand, not a note the operator may accept. A code this build does
# not recognise cannot be assessed at all, so it fails closed the same way. Only
# recognised, non-destructive codes are acknowledgeable warnings.
_REJECTION_SEVERITY = 0.5


def _split_source_codes(codes: set[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    rejections: list[str] = []
    for code in sorted(codes):
        if code not in SEVERITY or SEVERITY[code] >= _REJECTION_SEVERITY:
            rejections.append(code)
        else:
            warnings.append(code)
    return tuple(warnings), tuple(rejections)


def _inferred_blinds(hand: dict[str, Any]) -> dict[str, Any] | None:
    """Shape the timeline's forced_posts_bb for evidence, or None on timelines
    built before the field existed (pass-through exporter posture: serialize
    what the timeline carries, assert nothing about what it lacks)."""
    posts = hand.get("forced_posts_bb")
    if not isinstance(posts, (list, tuple)) or len(posts) < 2:
        return None
    return {
        "small_blind": float(posts[0]),
        "big_blind": float(posts[1]),
        "straddles": [float(p) for p in posts[2:]],
    }


def _completion_evidence_for_hand(
    hand: dict[str, Any],
    *,
    preceded_by_hand: bool,
    followed_by_hand: bool,
    validation_codes: list[str] | None,
    table_size: int | None,
    metadata: dict[str, Any],
) -> CompletionEvidence:
    """Build versioned boundary evidence from what the spine actually observed.

    Every claim here is conservative. A boundary the segmenter did not positively
    observe is reported as truncated, and any terminal event the spine could not
    read stays "unobserved" -- both of which block study readiness downstream
    rather than being talked past. Nothing is inferred from an absence of
    complaints.
    """

    source_images = [str(image) for image in hand.get("source_images", []) or []]
    partial_start, partial_end, terminal_event = _boundary_flags(
        hand, preceded_by_hand=preceded_by_hand, followed_by_hand=followed_by_hand,
    )
    terminal_observed = terminal_event in {"showdown", "fold_win", "hero_fold"}
    warning_codes, rejection_codes = _split_source_codes(
        set(hand.get("warnings", []) or []) | set(validation_codes or [])
        | _field_codes(hand)
    )
    return CompletionEvidence(
        evidence_version=EVIDENCE_SCHEMA_VERSION,
        partial_start=partial_start,
        partial_end=partial_end,
        terminal_event=terminal_event,
        first_source_timestamp_s=_as_optional_float(hand.get("t_start")),
        last_source_timestamp_s=_as_optional_float(hand.get("t_end")),
        preceding_boundary=BoundaryEvidence(
            kind="hand_start" if preceded_by_hand else "recording_start",
            timestamp_s=_as_optional_float(hand.get("t_start")),
            frame_ref=source_images[0] if source_images else "",
        ),
        following_boundary=BoundaryEvidence(
            kind=(
                "hand_end"
                if followed_by_hand or terminal_observed
                else "recording_end"
            ),
            timestamp_s=_as_optional_float(hand.get("t_end")),
            frame_ref=source_images[-1] if source_images else "",
        ),
        boundary_confidence=_confidence_for_hand(
            hand, validation_codes,
            partial_start=partial_start, partial_end=partial_end,
            terminal_observed=terminal_observed,
        ),
        source_frames=tuple(source_images),
        warning_codes=warning_codes,
        rejection_codes=rejection_codes,
        layout_profile=str(metadata.get("layout_profile", "") or ""),
        # Seat layout is only confirmed when every seat was resolved into a player
        # row, at least one state POSITIVELY placed the hero zone on seat 0, and
        # no state disagreed about it.
        #
        # The positive half is not redundant. hero_seat_mismatch is a majority
        # vote over the hero zone's own cards, so with an empty hero zone it
        # evaluates `0 > 0` and reports "confirmed" -- and an empty hero zone is
        # exactly what a layout drift produces (measured: a 1.24x vertical stretch
        # re-attributes hero's cards to two different villain seats as showdown
        # reveals while the board still zones 5/5 and the anchor residual stays in
        # tolerance). Asserting layout_supported on zero evidence is the one claim
        # this field must not make.
        layout_supported=(
            table_size is not None
            and bool(hand.get("hero_seat_confirmed"))
            and "hero_seat_mismatch" not in (hand.get("warnings") or [])
        ),
        table_size=table_size,
        pipeline_version=str(metadata.get("source", "") or ""),
        model_versions={
            str(key): str(value)
            for key, value in (metadata.get("model_versions") or {}).items()
        },
        extra={
            # Read-quality facts kept from the pre-evidence export so no existing
            # consumer of these keys loses data.
            "cv_amounts_unknown": hand.get("amounts_unknown", 0),
            # PASS-THROUGH, like cv_stack_outlier_checks_skipped below: the spine
            # no longer produces `amounts_rejected`, but a timeline built by an
            # earlier build does, and the exporter serializes the timeline it is
            # handed.
            "cv_amounts_rejected": hand.get("amounts_rejected", 0),
            # WHY they are unknown, per reason. A bare count cannot distinguish a
            # systematic reader failure on one HUD region from scattered
            # occlusions, and the count alone is what reached the app before.
            # `extra` is explicitly open (completion.CompletionEvidence), so this
            # is additive with no schema change and no migration.
            "cv_amounts_unknown_by_code": dict(hand.get("amounts_unknown_by_code") or {}),
            # Money the ledger could not account for: transitions whose stack
            # delta was unmeasurable, and actions proved but not sized.
            "cv_unmeasured_transitions": hand.get("unmeasured_transitions", 0),
            "cv_coverage_gaps": hand.get("coverage_gaps", 0),
            "cv_unknown_money_actions": hand.get("unknown_money_actions", 0),
            # Seats whose STARTING STACK the reader refused, as opposed to seats
            # whose box was simply absent. HandPlayer.starting_stack already
            # distinguishes None from 0.0; this says which None is which.
            "cv_starting_stack_unknown": {
                str(row.get("seat")): row.get("starting_stack_unknown")
                for row in (hand.get("players") or [])
                if row.get("starting_stack_unknown")
            },
            # A pot_text box thrown out by the centre-column guard is not the same
            # fact as "no pot was detected"; a seat whose two stack_text boxes
            # disagreed is not the same fact as "never read"; and a frame with too
            # few stack reads for the sibling-median net is not the same fact as
            # "checked and clean". All three used to be indistinguishable.
            "cv_pot_text_off_column": hand.get("pot_text_off_column", 0),
            "cv_stack_conflicts": hand.get("stack_conflicts", 0),
            "cv_bet_conflicts": hand.get("bet_conflicts", 0),
            # PASS-THROUGH of whatever the timeline carries. The spine no longer
            # produces `stack_outlier_checks_skipped` (its net is deleted), but a
            # timeline built by an earlier build still does, and this exporter's
            # job is to serialize the timeline it is handed rather than to assert
            # the field is absent.
            "cv_stack_outlier_checks_skipped": hand.get("stack_outlier_checks_skipped", 0),
            "cv_pot_collapses_repaired": hand.get("pot_collapses_repaired", 0),
            # Settlement-scan transitions skipped over a refused pot read: the
            # winner/t_end were chosen from a scan with that many blind spots.
            # The counter existed "so the state can be told apart downstream" --
            # the same argument the three cv_* fields above make -- and was
            # consumed by nothing.
            "cv_settle_scan_skipped": hand.get("settle_scan_skipped", 0),
            "cv_anchor_missing_states": hand.get("anchor_missing_states", 0),
            "cv_side_pot_detected": bool(hand.get("side_pot")),
            # The session-voted forced-post structure this hand was
            # reconstructed under (BB display units: sb, bb, then straddles).
            # Advisory: the operator's blind attestation remains the authority;
            # this says what the CV evidence showed so the attestation form can
            # be pre-checked against it (a straddled table used to arrive with
            # no trace of the straddle at all).
            "cv_inferred_blinds": _inferred_blinds(hand),
        },
    )


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_players(hand: dict[str, Any]) -> list[HandPlayer]:
    """Build HandPlayer rows from a reconstruction-spine hand (empty for card-only)."""
    players: list[HandPlayer] = []
    for row in hand.get("players", []) or []:
        seat = row.get("seat")
        name = row.get("player_name") or (
            "Hero" if row.get("is_hero") else (row.get("position") or f"Seat{seat}")
        )
        players.append(HandPlayer(
            hand_id=0,
            player_key=f"seat:{seat}" if seat is not None else f"name:{name}",
            seat_index=seat,
            player_name=name,
            position=row.get("position", "") or "",
            starting_stack=row.get("starting_stack"),
            is_hero=bool(row.get("is_hero")),
        ))
    return players


def _build_actions(hand: dict[str, Any]) -> list[Action]:
    """Build Action rows from a reconstruction-spine hand (empty for card-only)."""
    actions: list[Action] = []
    for row in hand.get("actions", []) or []:
        seat = row.get("seat")
        forced = row.get("forced_bet_type")
        live_post = row.get("is_live_post")
        if row.get("action_type") == "post_blind" and live_post is None:
            live_post = True
        actions.append(Action(
            hand_id=0,
            player_key=f"seat:{seat}" if seat is not None else None,
            street=row["street"],
            action_index=row.get("action_index"),
            player_name=row.get("player_name") or f"Seat{row.get('seat')}",
            position=row.get("position", "") or "",
            action_type=row["action_type"],
            amount=row.get("amount"),
            amount_semantics="incremental",
            forced_bet_type=forced,  # type: ignore[arg-type]
            is_live_post=live_post if live_post is None else bool(live_post),
            pot_before=row.get("pot_before"),
            stack_before=row.get("stack_before"),
            # Provenance: which frame produced this line. Kept so the
            # validation UI can still explain the row after the operator
            # corrects its street or order, which moves it off its slot.
            source_image=row.get("source_image") or None,
        ))
    return actions


def _assert_actions_reference_players(
    players: list[HandPlayer], actions: list[Action]
) -> None:
    """Every action must name a player row of the SAME hand.

    ``_build_players`` reads the spine's ``players`` and ``_build_actions`` reads
    its ``actions``; nothing used to compare them. The 07-11 development recording
    shipped a hand listing seats {0,1,2,3,4,7} with a preflop ``seat:5`` fold in
    its ledger, and persistence enforces the reference: ``import_session`` raises
    and rolls the whole transaction back, so ONE inconsistent hand destroyed the
    three good hands exported beside it -- 0 sessions, 0 hands imported.

    Raising here refuses the inconsistent hand alone. ``timeline_to_session_payload``
    catches ValueError per hand and records it in ``skipped``, so the rest of the
    session still lands and the operator is told which seat is unaccounted for.
    """
    known = {player.player_key for player in players}
    orphans = sorted({action.player_key for action in actions
                      if action.player_key is not None and action.player_key not in known})
    if orphans:
        raise ValueError(
            "Hand action ledger references seats that are not players of the hand: "
            f"{', '.join(orphans)} (players: {', '.join(sorted(known)) or 'none'})."
        )


def hero_participated_preflop(hand: dict[str, Any]) -> bool:
    """True when hero was dealt in or acted preflop (fold counts as playing).

    Sit-outs and segments with no hero cards and no hero preflop action are
    excluded: there is nothing to study about hero's preflop decision. A preflop
    fold after observing cards is included — coaching can still target that fold.
    Incomplete OCR (1 or 3+ detected hole cards) still counts as dealt-in so
    the draft can land for the operator to fill blanks.
    """
    if len(hand.get("hero") or []) >= 1:
        return True
    if hand.get("hero_folded"):
        return True
    hero_seats = {
        int(player["seat"])
        for player in (hand.get("players") or [])
        if player.get("is_hero") and player.get("seat") is not None
    }
    if not hero_seats:
        hero_seats = {0}
    for action in hand.get("actions") or []:
        seat = action.get("seat")
        if seat is None or int(seat) not in hero_seats:
            continue
        street = action.get("street")
        if street is None:
            continue
        if street != "preflop":
            continue
        if action.get("action_type") in {
            "fold",
            "check",
            "call",
            "bet",
            "raise",
            "all-in",
            "post_blind",
        }:
            return True
    return False


def _soft_card_fields_for_draft(
    hand: dict[str, Any],
) -> tuple[str, str]:
    """Return app-format hero/board cards, blanking fields the Hand model refuses.

    Incomplete drafts must still land so the operator can fill blanks. Invalid
    counts or duplicate visible cards would raise ValidationError and skip the
    whole hand — blanking the offending field keeps the draft importable.
    """
    hero_raw = list(hand.get("hero") or [])
    board_raw = list(hand.get("board") or [])
    hero_cards = _cards_to_app(hero_raw) if len(hero_raw) in {0, 2} else ""
    board_cards = (
        _cards_to_app(board_raw) if len(board_raw) in VALID_BOARD_COUNTS else ""
    )
    if hero_cards:
        hero_tokens = hero_cards.split()
        if len(set(hero_tokens)) != len(hero_tokens):
            hero_cards = ""
    if board_cards:
        board_tokens = board_cards.split()
        if len(set(board_tokens)) != len(board_tokens):
            board_cards = ""
    if hero_cards and board_cards:
        try:
            parse_visible_cards(hero_cards, board_cards)
        except CardParseError:
            board_cards = ""
    return hero_cards, board_cards


def hand_to_import_payload(
    hand: dict[str, Any],
    *,
    output_hand_number: int,
    timeline_path: Path,
    include_incomplete: bool = False,
    validation_codes: list[str] | None = None,
    preceded_by_hand: bool = False,
    followed_by_hand: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    board_count = len(hand.get("board") or [])

    if hand.get("drop_from_export"):
        return None

    if not hero_participated_preflop(hand):
        return None

    if not include_incomplete:
        if not hand.get("complete_cards"):
            return None
        if len(hand.get("hero") or []) != 2 or board_count not in VALID_BOARD_COUNTS:
            return None
        hero_cards = _cards_to_app(hand.get("hero"))
        board_cards = _cards_to_app(hand.get("board"))
    else:
        hero_cards, board_cards = _soft_card_fields_for_draft(hand)
    partial_start, partial_end, terminal_event = _boundary_flags(
        hand, preceded_by_hand=preceded_by_hand, followed_by_hand=followed_by_hand,
    )
    confidence = _confidence_for_hand(
        hand, validation_codes, partial_start=partial_start, partial_end=partial_end,
        terminal_observed=terminal_event in {"showdown", "fold_win", "hero_fold"},
    )

    # Extended fields present only on reconstruction-spine hands; card-only timelines
    # (the current 53-class YOLO path) leave these empty and unchanged.
    players = _build_players(hand)
    actions = _build_actions(hand)
    _assert_actions_reference_players(players, actions)
    hero_player = next((p for p in players if p.is_hero), None)
    table_size = len(players) if 2 <= len(players) <= 10 else None

    # Versioned completion evidence, not just read-quality counters: without an
    # evidence_version this build recognises, every reconstructed hand would stay
    # permanently blocked on COMPLETION_EVIDENCE_MISSING with no clearing action,
    # because re-running the reconstruction would produce the same unreadable blob.
    # cv_source is dropped on import; completion_evidence survives the round trip.
    evidence = _completion_evidence_for_hand(
        hand,
        preceded_by_hand=preceded_by_hand,
        followed_by_hand=followed_by_hand,
        validation_codes=validation_codes,
        table_size=table_size,
        metadata=metadata or {},
    )
    hand_model = Hand(
        session_id=0,
        hand_number=output_hand_number,
        game_type="",
        table_size=table_size,
        hero_position=hero_player.position if hero_player else "",
        hero_cards=hero_cards,
        board_cards=board_cards,
        pot_size=hand.get("pot"),
        result=hand.get("result", "") or "",
        hero_bb_won=hand.get("hero_bb_won"),
        review_status="needs_correction",
        confidence_score=confidence,
        source_type="cv_import",
        tags=["LOW_CONFIDENCE"] if confidence <= LOW_CONFIDENCE_AT else [],
        notes=_hand_notes(hand, timeline_path=timeline_path),
        completion_status=derive_completion_status(evidence, source_type="cv_import"),
        completion_evidence=dump_completion_evidence(evidence),
    )
    return {
        "hand": hand_model.model_dump(mode="json"),
        "players": [player.model_dump(mode="json") for player in players],
        "actions": [action.model_dump(mode="json") for action in actions],
        "reviews": [],
        "cv_source": {
            "timeline_hand_number": hand.get("hand_number"),
            "t_start": hand.get("t_start"),
            "t_end": hand.get("t_end"),
            "source_images": hand.get("source_images", []),
            "warnings": hand.get("warnings", []),
            "winner_seat": hand.get("winner_seat"),
            "reconciled": hand.get("reconciled"),
            "side_pot": hand.get("side_pot"),
        },
    }


def timeline_to_session_payload(
    timeline: dict[str, Any],
    *,
    timeline_path: Path,
    session_name: str,
    include_incomplete: bool = False,
    allow_validation_warnings: bool = False,
) -> dict[str, Any]:
    hands = []
    skipped = []
    # The gate NEVER skips validation. It used to: a timeline whose "states"
    # was missing or malformed -- exactly the shape a hand-edited, truncated,
    # or third-party file produces, and the shape MalformedTimeline exists for
    # -- bypassed the validator entirely and exported MORE hands than the same
    # timeline with its states intact (measured on g0711: 1 hand with states,
    # 6 without, two of them carrying fatal codes and two landing completion
    # "complete" with their state-level findings silently lost). With no
    # states list the state-level checks have nothing to examine, but every
    # hand-level check -- fatal codes, action-sequence legality, summary
    # consistency -- still runs against an empty states list, so a hand the
    # validator can reject from its own fields is still rejected.
    if isinstance(timeline.get("states"), list):
        validation_report = validate_timeline(timeline)
    else:
        validation_report = validate_timeline({**timeline, "states": []})
    validation_by_hand = {
        report.get("hand_number"): report
        for report in (validation_report or {}).get("hands", [])
    }
    timeline_hands = timeline.get("hands", [])
    # Boundary provenance: the segmenter cuts a new hand only at a positively
    # detected fresh deal, so every hand after the first has an observed start,
    # and every hand before the last has an observed end. The recording's own
    # first and last hands are truncated unless the spine read their boundary
    # event, which _completion_evidence_for_hand decides.
    metadata = timeline.get("metadata") or {}
    for position, hand in enumerate(timeline_hands):
        preceded_by_hand = position > 0
        followed_by_hand = position < len(timeline_hands) - 1
        manual_corrected = bool(hand.get("manual_correction"))
        validation = validation_by_hand.get(hand.get("hand_number"))
        codes = sorted({w.get("code") for w in (validation or {}).get("warnings", [])
                        if w.get("code")})
        blocking_codes = set(codes)
        if include_incomplete:
            # Card-shape warnings and operator-fillable reconstruction gaps stay
            # in validation_codes / completion_evidence, but do not gate import.
            blocking_codes -= DRAFT_CARD_WARNING_CODES | DRAFT_OPERATOR_FILLABLE_CODES
        if (
            validation
            and blocking_codes
            and not allow_validation_warnings
            and not manual_corrected
        ):
            skipped.append({
                "timeline_hand_number": hand.get("hand_number"),
                "reason": "validation_warnings",
                # Name the codes: "3 validation warnings" alone gave an operator
                # no way to tell a rejected side pot from a board regression.
                "codes": sorted(blocking_codes),
                "detail": (
                    f"{len(blocking_codes)} validation warnings "
                    f"({', '.join(sorted(blocking_codes)) or 'unspecified'}); "
                    "use --allow-validation-warnings to export anyway."
                ),
            })
            continue
        try:
            payload = hand_to_import_payload(
                hand,
                output_hand_number=len(hands) + 1,
                timeline_path=timeline_path,
                include_incomplete=include_incomplete,
                # A hand exported past the gate (--allow-validation-warnings, or
                # a manual correction) still carries the validator's findings in
                # its score: the operator overrode the gate, not the facts.
                validation_codes=codes,
                preceded_by_hand=preceded_by_hand,
                followed_by_hand=followed_by_hand,
                metadata=metadata,
            )
        except (ValueError, ValidationError) as exc:
            skipped.append({
                "timeline_hand_number": hand.get("hand_number"),
                "reason": type(exc).__name__,
                "detail": str(exc),
            })
            continue
        if payload is None:
            if hand.get("drop_from_export"):
                reason = "dropped"
                detail = "Hand was marked drop_from_export."
            elif not hero_participated_preflop(hand):
                reason = "hero_did_not_play_preflop"
                detail = "Hero was not dealt in and took no preflop action."
            else:
                reason = "incomplete_or_invalid_cards"
                detail = (
                    "Use --include-incomplete to export needs-correction drafts "
                    "when model validation allows it."
                )
            skipped.append({
                "timeline_hand_number": hand.get("hand_number"),
                "reason": reason,
                "detail": detail,
            })
            continue
        hands.append(payload)

    session = Session(
        name=session_name,
        date_played=date.today(),
        platform="ClubWPT Gold",
        notes=(
            "Imported from offline YOLO card timeline. "
            "Actions, stacks, pot, and winners still require OCR/state reconstruction."
        ),
    )
    return {
        "export_version": EXPORT_VERSION,
        "session": session.model_dump(mode="json"),
        "hands": hands,
        "cv_import_summary": {
            "timeline": str(timeline_path),
            "timeline_hands": len(timeline.get("hands", [])),
            "exported_hands": len(hands),
            "skipped_hands": len(skipped),
            "skipped": skipped,
            "validation_summary": (validation_report or {}).get("summary"),
        },
    }


def export_timeline(
    timeline_path: Path,
    out_path: Path,
    *,
    session_name: str,
    include_incomplete: bool = False,
    allow_validation_warnings: bool = False,
    hand_corrections_path: Path | None = None,
) -> dict[str, Any]:
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if hand_corrections_path is not None:
        timeline = apply_hand_corrections(timeline, load_hand_corrections(hand_corrections_path))
    payload = timeline_to_session_payload(
        timeline,
        timeline_path=timeline_path,
        session_name=session_name,
        include_incomplete=include_incomplete,
        allow_validation_warnings=allow_validation_warnings,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", default=DEFAULT_TIMELINE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--session-name", default="YOLO card draft session")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--allow-validation-warnings", action="store_true")
    parser.add_argument("--hand-corrections", help="Optional hand_corrections.csv from the hand review UI")
    parser.add_argument("--import-db", help="Optional SQLite DB path to import into after writing JSON.")
    args = parser.parse_args()

    timeline_path = Path(args.timeline)
    out_path = Path(args.out)
    payload = export_timeline(
        timeline_path,
        out_path,
        session_name=args.session_name,
        include_incomplete=args.include_incomplete,
        allow_validation_warnings=args.allow_validation_warnings,
        hand_corrections_path=Path(args.hand_corrections) if args.hand_corrections else None,
    )

    print(f"timeline={timeline_path}")
    print(f"out={out_path}")
    print(f"exported_hands={payload['cv_import_summary']['exported_hands']}")
    print(f"skipped_hands={payload['cv_import_summary']['skipped_hands']}")

    if args.import_db:
        db = PokerDatabase(args.import_db)
        db.init_db()
        session = import_session(db, payload)
        db.close()
        print(f"imported_session_id={session.id}")


if __name__ == "__main__":
    main()
