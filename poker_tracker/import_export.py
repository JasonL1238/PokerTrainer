from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from poker_tracker.db import PokerDatabase
from poker_tracker.models import Action, Hand, HandPlayer, HandReview, Session


EXPORT_VERSION = 1


def export_hand(db: PokerDatabase, hand_id: int) -> dict[str, Any]:
    """Export one hand and its related rows as JSON-compatible data."""
    hand = db.fetch_hand(hand_id)
    if hand is None:
        raise ValueError(f"Hand not found: {hand_id}")
    return {
        "export_version": EXPORT_VERSION,
        "hand": _dump_model(hand),
        "players": [_dump_model(player) for player in db.fetch_players_by_hand(hand_id)],
        "actions": [_dump_model(action) for action in db.fetch_actions_by_hand(hand_id)],
        "reviews": [_dump_model(review) for review in db.fetch_reviews_by_hand(hand_id)],
    }


def export_session(db: PokerDatabase, session_id: int) -> dict[str, Any]:
    """Export one full session with hands, players, actions, and reviews."""
    session = db.fetch_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")
    return {
        "export_version": EXPORT_VERSION,
        "session": _dump_model(session),
        "hands": [
            export_hand(db, hand.id)
            for hand in db.fetch_hands_by_session(session_id)
            if hand.id is not None
        ],
    }


def export_session_json(db: PokerDatabase, session_id: int, path: str | Path) -> None:
    Path(path).write_text(json.dumps(export_session(db, session_id), indent=2), encoding="utf-8")


def import_session(db: PokerDatabase, payload: dict[str, Any]) -> Session:
    """Import a previously exported session into the current database."""
    session_data = dict(payload["session"])
    session_data.pop("id", None)
    session = db.create_session(Session(**session_data))

    for hand_payload in payload.get("hands", []):
        hand_data = dict(hand_payload["hand"])
        hand_data.pop("id", None)
        hand_data["session_id"] = session.id
        saved_hand = db.create_hand(Hand(**hand_data))

        for player_data in hand_payload.get("players", []):
            imported = dict(player_data)
            imported.pop("id", None)
            imported["hand_id"] = saved_hand.id
            db.create_hand_player(HandPlayer(**imported))

        for action_data in hand_payload.get("actions", []):
            imported = dict(action_data)
            imported.pop("id", None)
            imported["hand_id"] = saved_hand.id
            db.create_action(Action(**imported))

        for review_data in hand_payload.get("reviews", []):
            imported = dict(review_data)
            imported.pop("id", None)
            imported["hand_id"] = saved_hand.id
            db.create_hand_review(HandReview(**imported))

    return session


def import_session_json(db: PokerDatabase, path: str | Path) -> Session:
    return import_session(db, json.loads(Path(path).read_text(encoding="utf-8")))


def _dump_model(model: Any) -> dict[str, Any]:
    data = model.model_dump()
    for key, value in list(data.items()):
        if isinstance(value, (date, datetime)):
            data[key] = value.isoformat()
    return data
