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
        if origin is None:
            continue
        image = str(origin.get("source_image") or "")
        if not image:
            continue
        db.set_action_source_image(action.id, image)
        filled += 1
    return filled
