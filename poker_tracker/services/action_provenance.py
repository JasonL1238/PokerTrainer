"""Backfill which source frame produced each reconstructed action.

Schema 16 added ``actions.source_image`` but could not backfill it: the value
lives in the job timeline on disk, not in the database. Hands imported before
the migration therefore carry NULL, and every frame-derived warning on them
falls back to matching by street and order — which is exactly what goes wrong
once the operator corrects a row.

This repairs those rows lazily, when Import validation opens the hand and the
timeline is already loaded. It is deliberately conservative: a row is only
filled when it still matches its reconstructed line on street, order, type and
actor, so a row the operator has already edited is left alone rather than
given a guessed origin.
"""
from __future__ import annotations

from typing import Any

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Action
from poker_tracker.ui.reconstruction_review import match_db_action_to_timeline_action


def backfill_action_provenance(
    db: PokerDatabase,
    hand_id: int,
    timeline_hand: dict[str, Any],
) -> int:
    """Fill missing source frames for one hand. Returns how many were filled."""

    filled = 0
    for action in db.fetch_actions_by_hand(hand_id):
        if action.id is None or action.source_image:
            continue
        origin = match_db_action_to_timeline_action(
            timeline_hand,
            street=action.street,
            action_index=action.action_index,
            action_type=action.action_type,
            position=action.position,
            player_name=action.player_name,
        )
        if origin is None or not _fill_is_safe(action, origin, timeline_hand):
            continue
        image = str(origin.get("source_image") or "")
        if not image:
            continue
        db.set_action_source_image(action.id, image)
        filled += 1
    return filled


def _fill_is_safe(
    action: Action,
    origin: dict[str, Any],
    timeline_hand: dict[str, Any],
) -> bool:
    """Whether this match identifies the row's origin rather than merely fitting.

    The rows this repairs are the ones that predate provenance, so some were
    already moved — and the slot match keys on the row's CURRENT street and
    order, which a moved row shares with whatever line now occupies it. A
    wrong fill is permanent (the writer only touches NULLs) and silently
    disarms the render-time guard that rejects a disagreeing slot match, so
    the bar is: either nothing else in the hand could have produced this row,
    or its own figures pick out this line.
    """

    rivals = [
        line
        for line in timeline_hand.get("actions") or []
        if line is not origin
        and str(line.get("action_type") or "").replace("-", "_")
        == action.action_type.replace("-", "_")
        and (
            str(line.get("player_name") or "") == action.player_name
            or str(line.get("position") or "") == action.position
        )
    ]
    if not rivals:
        return True
    return _figures_agree(action, origin) and not any(
        _figures_agree(action, rival) for rival in rivals
    )


def _figures_agree(action: Action, line: dict[str, Any]) -> bool:
    """Whether the row's own amount and stack single this line out."""

    for saved, recorded in (
        (action.amount, line.get("amount")),
        (action.stack_before, line.get("stack_before")),
    ):
        if saved is None or recorded is None:
            continue
        try:
            if abs(float(saved) - float(recorded)) >= 1e-6:
                return False
        except (TypeError, ValueError):
            return False
    return action.amount is not None or action.stack_before is not None
