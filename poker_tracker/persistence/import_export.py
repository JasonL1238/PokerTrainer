from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandCorrection,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SettlementEntry,
)

EXPORT_VERSION = 4
SUPPORTED_IMPORT_VERSIONS = {1, 2, 3, EXPORT_VERSION}


def export_hand(db: PokerDatabase, hand_id: int) -> dict[str, Any]:
    """Export one hand and its related rows as JSON-compatible data."""
    hand = db.fetch_hand(hand_id)
    if hand is None:
        raise ValueError(f"Hand not found: {hand_id}")
    settlement = db.fetch_hand_settlement(hand_id)
    return {
        "export_version": EXPORT_VERSION,
        "hand": _dump_model(hand),
        "players": [_dump_model(player) for player in db.fetch_players_by_hand(hand_id)],
        "actions": [_dump_model(action) for action in db.fetch_actions_by_hand(hand_id)],
        "settlement": None if settlement is None else _dump_model(settlement),
        "settlement_entries": [
            _dump_model(entry) for entry in db.fetch_settlement_entries(hand_id)
        ],
        "reviews": [_dump_model(review) for review in db.fetch_reviews_by_hand(hand_id)],
        "coaching_reviews": [
            _dump_model(review) for review in db.fetch_coaching_reviews_by_hand(hand_id)
        ],
        "corrections": [
            _dump_model(correction) for correction in db.fetch_hand_corrections(hand_id)
        ],
        "issues": [
            _dump_model(issue) for issue in db.fetch_hand_issues(hand_id=hand_id)
        ],
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
        "coaching_reviews": [
            _dump_model(review)
            for review in db.fetch_coaching_reviews_by_session(session_id)
        ],
    }


def export_session_json(db: PokerDatabase, session_id: int, path: str | Path) -> None:
    Path(path).write_text(json.dumps(export_session(db, session_id), indent=2), encoding="utf-8")


def import_session(db: PokerDatabase, payload: dict[str, Any]) -> Session:
    """Import a previously exported session into the current database."""
    version = payload.get("export_version", 1)
    if version not in SUPPORTED_IMPORT_VERSIONS:
        raise ValueError(
            f"Unsupported export_version {version}; this app understands "
            f"{sorted(SUPPORTED_IMPORT_VERSIONS)}."
        )
    session_data = dict(payload["session"])
    session_data.pop("id", None)
    session_model = Session(**session_data)
    validated_hands: list[
        tuple[
            Hand,
            list[HandPlayer],
            list[Action],
            HandSettlement | None,
            list[SettlementEntry],
            list[HandReview],
            list[CoachingResponse],
            list[HandCorrection],
            list[HandIssue],
        ]
    ] = []

    for hand_payload in payload.get("hands", []):
        hand_data = dict(hand_payload["hand"])
        hand_data.pop("id", None)
        hand_data["session_id"] = 0
        hand = Hand(**hand_data)

        players: list[HandPlayer] = []
        for player_data in hand_payload.get("players", []):
            imported = dict(player_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            players.append(HandPlayer(**imported))

        actions: list[Action] = []
        for action_data in hand_payload.get("actions", []):
            imported = dict(action_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            if "amount_semantics" not in imported:
                imported["amount_semantics"] = "unknown"
            actions.append(Action(**imported))
        _link_actions_to_players(actions, players)
        _normalize_duplicate_action_indexes(actions)

        settlement_data = hand_payload.get("settlement")
        settlement: HandSettlement | None = None
        if settlement_data is not None:
            imported_settlement = dict(settlement_data)
            imported_settlement["hand_id"] = 0
            settlement = HandSettlement(**imported_settlement)

        settlement_entries: list[SettlementEntry] = []
        for entry_data in hand_payload.get("settlement_entries", []):
            imported = dict(entry_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            settlement_entries.append(SettlementEntry(**imported))

        reviews: list[HandReview] = []
        for review_data in hand_payload.get("reviews", []):
            imported = dict(review_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            reviews.append(HandReview(**imported))

        coaching_reviews: list[CoachingResponse] = []
        for review_data in hand_payload.get("coaching_reviews", []):
            imported = dict(review_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            imported["session_id"] = 0
            coaching_reviews.append(CoachingResponse(**imported))

        corrections: list[HandCorrection] = []
        for correction_data in hand_payload.get("corrections", []):
            imported = dict(correction_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            corrections.append(HandCorrection(**imported))

        issues: list[HandIssue] = []
        for issue_data in hand_payload.get("issues", []):
            imported = dict(issue_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            issues.append(HandIssue(**imported))

        validated_hands.append(
            (
                hand,
                players,
                actions,
                settlement,
                settlement_entries,
                reviews,
                coaching_reviews,
                corrections,
                issues,
            )
        )

    session_coaching_reviews: list[CoachingResponse] = []
    for review_data in payload.get("coaching_reviews", []):
        imported = dict(review_data)
        imported.pop("id", None)
        imported["hand_id"] = None
        imported["session_id"] = 0
        session_coaching_reviews.append(CoachingResponse(**imported))

    with db.transaction():
        session = db.create_session(session_model)
        if session.id is None:
            raise RuntimeError("Imported session did not receive an id.")
        for (
            hand,
            players,
            actions,
            settlement,
            settlement_entries,
            reviews,
            coaching_reviews,
            corrections,
            issues,
        ) in validated_hands:
            saved_hand = db.create_hand(hand.model_copy(update={"session_id": session.id}))
            if saved_hand.id is None:
                raise RuntimeError("Imported hand did not receive an id.")
            for player in players:
                db.create_hand_player(player.model_copy(update={"hand_id": saved_hand.id}))
            for action in actions:
                db.create_action(action.model_copy(update={"hand_id": saved_hand.id}))
            if settlement is not None:
                db.upsert_hand_settlement(settlement.model_copy(update={"hand_id": saved_hand.id}))
                db.replace_settlement_entries(
                    saved_hand.id,
                    [
                        entry.model_copy(update={"hand_id": saved_hand.id})
                        for entry in settlement_entries
                    ],
                )
            for review in reviews:
                db.create_hand_review(review.model_copy(update={"hand_id": saved_hand.id}))
            for review in coaching_reviews:
                db.create_coaching_response(
                    review.model_copy(
                        update={"hand_id": saved_hand.id, "session_id": session.id}
                    )
                )
            for correction in corrections:
                db.create_hand_correction(
                    correction.model_copy(update={"hand_id": saved_hand.id})
                )
            for issue in issues:
                db.create_hand_issue(
                    issue.model_copy(update={"hand_id": saved_hand.id}),
                    apply_workflow=False,
                )
        for review in session_coaching_reviews:
            db.create_coaching_response(
                review.model_copy(update={"session_id": session.id})
            )

    return session


def import_hands_into_session(
    db: PokerDatabase,
    payload: dict[str, Any],
    session_id: int,
) -> Session:
    """Append an imported payload's hands to an existing session.

    Imported hand numbers are preserved when available. Collisions are assigned
    the next number in the target session, keeping every hand addressable.
    """

    target = db.fetch_session(session_id)
    if target is None:
        raise ValueError(f"Session not found: {session_id}")

    with db.transaction():
        temporary = import_session(db, payload)
        if temporary.id is None:
            raise RuntimeError("Imported session was not persisted.")
        for hand in db.fetch_hands_by_session(temporary.id):
            if hand.id is not None:
                db.move_hand_to_session(hand.id, session_id)
        db.delete_session(temporary.id)

    refreshed = db.fetch_session(session_id)
    if refreshed is None:
        raise RuntimeError("Target session could not be reloaded after import.")
    return refreshed


def import_session_json(db: PokerDatabase, path: str | Path) -> Session:
    return import_session(db, json.loads(Path(path).read_text(encoding="utf-8")))


def _dump_model(model: Any) -> dict[str, Any]:
    data = model.model_dump()
    for key, value in list(data.items()):
        if isinstance(value, (date, datetime)):
            data[key] = value.isoformat()
    return data


def _link_actions_to_players(actions: list[Action], players: list[HandPlayer]) -> None:
    """Attach imported legacy actions only when identity resolution is unambiguous."""

    for index, action in enumerate(actions):
        if action.player_key is not None:
            continue
        candidates = [
            player
            for player in players
            if player.player_name == action.player_name
            and (not action.position or player.position == action.position)
        ]
        if len(candidates) != 1:
            candidates = [player for player in players if player.player_name == action.player_name]
        if len(candidates) == 1:
            actions[index] = action.model_copy(update={"player_key": candidates[0].player_key})


def _normalize_duplicate_action_indexes(actions: list[Action]) -> None:
    """Resolve ambiguous legacy/import order using stable payload order."""
    counts: dict[tuple[str, int], int] = {}
    duplicate_streets: set[str] = set()
    for action in actions:
        if action.action_index is None:
            continue
        key = (action.street, action.action_index)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            duplicate_streets.add(action.street)
    if not duplicate_streets:
        return

    next_index = {street: 1 for street in duplicate_streets}
    for index, action in enumerate(actions):
        if action.street not in duplicate_streets:
            continue
        actions[index] = action.model_copy(update={"action_index": next_index[action.street]})
        next_index[action.street] += 1
