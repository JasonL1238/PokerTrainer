"""Action provenance survives operator corrections (schema 16)."""
from __future__ import annotations

import json

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import (
    EXPORT_VERSION,
    export_session,
    import_session,
)
from poker_tracker.persistence.models import Action, Hand, Session


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _seed(db: PokerDatabase) -> Action:
    session = db.create_session(Session(name="Provenance"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    return db.create_action(
        Action(
            hand_id=hand.id,
            street="flop",
            action_index=1,
            player_name="Seat2",
            position="UTG+1",
            action_type="bet",
            amount=6.5,
            source_image="/frames/t000069.00.jpg",
        )
    )


def test_schema_is_sixteen() -> None:
    assert SCHEMA_VERSION == 16


def test_source_image_round_trips_through_the_database() -> None:
    db = make_db()
    saved = _seed(db)
    assert saved.source_image == "/frames/t000069.00.jpg"
    fetched = db.fetch_actions_by_hand(saved.hand_id)[0]
    assert fetched.source_image == "/frames/t000069.00.jpg"
    db.close()


def test_correcting_street_and_order_does_not_move_provenance() -> None:
    """The whole point: action indexes are per-street, so a street correction
    lands the row on another slot. Provenance must not follow it."""
    db = make_db()
    saved = _seed(db)
    db.update_action(
        saved.model_copy(update={"street": "turn", "action_index": 1}),
        correction_notes="video shows this on the turn",
    )
    moved = db.fetch_actions_by_hand(saved.hand_id)[0]
    assert moved.street == "turn"
    assert moved.source_image == "/frames/t000069.00.jpg"
    db.close()


def test_correcting_actor_type_and_amount_does_not_move_provenance() -> None:
    db = make_db()
    saved = _seed(db)
    db.update_action(
        saved.model_copy(
            update={
                "player_name": "Seat4",
                "position": "LJ",
                "action_type": "raise",
                "amount": 18.0,
            }
        ),
        correction_notes="misattributed",
    )
    edited = db.fetch_actions_by_hand(saved.hand_id)[0]
    assert edited.source_image == "/frames/t000069.00.jpg"
    db.close()


def test_manual_actions_have_no_provenance() -> None:
    db = make_db()
    session = db.create_session(Session(name="Manual"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    saved = db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            player_name="Hero",
            action_type="fold",
        )
    )
    assert saved.source_image is None
    db.close()


def test_provenance_survives_export_and_import() -> None:
    db = make_db()
    saved = _seed(db)
    payload = export_session(db, db.fetch_sessions()[0].id)
    assert payload["export_version"] == EXPORT_VERSION
    exported = payload["hands"][0]["actions"][0]
    assert exported["source_image"] == "/frames/t000069.00.jpg"

    target = make_db()
    import_session(target, json.loads(json.dumps(payload)))
    restored = target.fetch_actions_by_hand(
        target.fetch_hands_by_session(target.fetch_sessions()[0].id)[0].id
    )[0]
    assert restored.source_image == "/frames/t000069.00.jpg"
    assert restored.amount == saved.amount
    db.close()
    target.close()


def test_a_v5_payload_without_provenance_still_imports() -> None:
    """Older exports carry no source_image; they must import unchanged."""
    db = make_db()
    _seed(db)
    payload = export_session(db, db.fetch_sessions()[0].id)
    payload["export_version"] = 5
    for hand_payload in payload["hands"]:
        for action in hand_payload["actions"]:
            action.pop("source_image", None)

    target = make_db()
    import_session(target, payload)
    restored = target.fetch_actions_by_hand(
        target.fetch_hands_by_session(target.fetch_sessions()[0].id)[0].id
    )[0]
    assert restored.source_image is None
    assert restored.amount == 6.5
    db.close()
    target.close()


def test_update_action_does_not_carry_provenance_from_the_caller() -> None:
    """A7 round 7 M1: the earlier tests used model_copy, which carries
    source_image forward, so the UPDATE's omission was never exercised. The UI
    builds a FRESH Action with no source_image — mutating the UPDATE to write
    the column left the whole suite green."""
    db = make_db()
    saved = _seed(db)
    # Exactly how app.py's edit form rebuilds the row: a new Action carrying
    # only the form fields.
    db.update_action(
        Action(
            id=saved.id,
            hand_id=saved.hand_id,
            street=saved.street,
            action_index=saved.action_index,
            player_name=saved.player_name,
            position=saved.position,
            action_type="raise",
            amount=18.0,
        ),
        correction_notes="retyped from the frame",
    )
    edited = db.fetch_actions_by_hand(saved.hand_id)[0]
    assert edited.action_type == "raise"
    assert edited.source_image == "/frames/t000069.00.jpg"
    db.close()
