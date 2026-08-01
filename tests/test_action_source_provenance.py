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
from poker_tracker.services.action_provenance import backfill_action_provenance


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


def _timeline_hand() -> dict:
    return {
        "hand_number": 1,
        "actions": [
            {
                "street": "flop",
                "action_index": 1,
                "seat": 2,
                "player_name": "Seat2",
                "position": "UTG+1",
                "action_type": "bet",
                "amount": 6.5,
                "source_image": "/frames/t000069.00.jpg",
            }
        ],
    }


def test_backfill_fills_legacy_rows() -> None:
    """B8/A8 round 8: migration 16 cannot backfill (the value lives in the
    timeline on disk), so every existing row stayed NULL and none of the
    provenance repairs reached the operator's actual data."""
    db = make_db()
    session = db.create_session(Session(name="Legacy"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    saved = db.create_action(
        Action(
            hand_id=hand.id,
            street="flop",
            action_index=1,
            player_name="Seat2",
            position="UTG+1",
            action_type="bet",
            amount=6.5,
        )
    )
    assert saved.source_image is None
    assert backfill_action_provenance(db, hand.id, _timeline_hand()) == 1
    assert (
        db.fetch_actions_by_hand(hand.id)[0].source_image
        == "/frames/t000069.00.jpg"
    )
    # Idempotent.
    assert backfill_action_provenance(db, hand.id, _timeline_hand()) == 0
    db.close()


def test_backfill_leaves_already_edited_rows_alone() -> None:
    """A row the operator moved no longer matches its line, so filling it
    would be a guess at where it came from."""
    db = make_db()
    session = db.create_session(Session(name="Edited"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_action(
        Action(
            hand_id=hand.id,
            street="turn",          # operator already moved it
            action_index=1,
            player_name="Seat2",
            position="UTG+1",
            action_type="bet",
            amount=6.5,
        )
    )
    assert backfill_action_provenance(db, hand.id, _timeline_hand()) == 0
    assert db.fetch_actions_by_hand(hand.id)[0].source_image is None
    db.close()


def test_backfill_never_overwrites_a_recorded_frame() -> None:
    db = make_db()
    saved = _seed(db)
    other = dict(_timeline_hand())
    other["actions"][0]["source_image"] = "/frames/t999999.00.jpg"
    assert backfill_action_provenance(db, saved.hand_id, other) == 0
    assert (
        db.fetch_actions_by_hand(saved.hand_id)[0].source_image
        == "/frames/t000069.00.jpg"
    )
    db.close()


def _two_line_timeline() -> dict:
    """A hand where one seat makes the same kind of action twice — the shape
    that lets a pre-upgrade move land on the wrong line."""
    return {
        "hand_number": 1,
        "actions": [
            {
                "street": "preflop", "action_index": 4, "seat": 6,
                "player_name": "Seat6", "position": "SB", "action_type": "call",
                "amount": 3.0, "stack_before": 191.0,
                "source_image": "/frames/A.jpg",
            },
            {
                "street": "preflop", "action_index": 7, "seat": 6,
                "player_name": "Seat6", "position": "SB", "action_type": "call",
                "amount": 10.0, "stack_before": 191.0,
                "source_image": "/frames/B.jpg",
            },
        ],
    }


def _legacy_row(db: PokerDatabase, **overrides) -> Action:
    session = db.create_session(Session(name="Legacy"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    payload = dict(
        hand_id=hand.id,
        street="preflop",
        action_index=4,
        player_name="Seat6",
        position="SB",
        action_type="call",
        amount=10.0,          # truly the frame-B line, renumbered into slot 4
        stack_before=191.0,
    )
    payload.update(overrides)
    return db.create_action(Action(**payload))


def test_backfill_refuses_a_row_that_could_have_come_from_two_lines() -> None:
    """B9 round 9: a row moved BEFORE the upgrade — exactly the population this
    repairs — can occupy a slot held by a different line with the same actor
    and type. The fill is permanent and disarms the render-time guard, so it
    must not be a guess."""
    db = make_db()
    row = _legacy_row(db)
    assert backfill_action_provenance(db, row.hand_id, _two_line_timeline()) == 0
    assert db.fetch_actions_by_hand(row.hand_id)[0].source_image is None
    db.close()


def test_backfill_accepts_when_the_rows_own_figures_single_a_line_out() -> None:
    db = make_db()
    # Amount 3.0 matches only the frame-A line.
    row = _legacy_row(db, amount=3.0)
    assert backfill_action_provenance(db, row.hand_id, _two_line_timeline()) == 1
    assert (
        db.fetch_actions_by_hand(row.hand_id)[0].source_image == "/frames/A.jpg"
    )
    db.close()


def test_backfill_refuses_when_the_row_has_no_figures_to_disambiguate() -> None:
    db = make_db()
    row = _legacy_row(db, amount=None, stack_before=None)
    assert backfill_action_provenance(db, row.hand_id, _two_line_timeline()) == 0
    assert db.fetch_actions_by_hand(row.hand_id)[0].source_image is None
    db.close()


def test_recorded_provenance_is_write_once_at_the_sql_layer() -> None:
    """A9 round 9: the caller already skips filled rows, so only a direct
    write exercises the IS NULL guard — and that guard is what makes a wrong
    fill impossible to compound."""
    db = make_db()
    saved = _seed(db)
    assert saved.id is not None
    db.set_action_source_image(saved.id, "/frames/OTHER.jpg")
    assert (
        db.fetch_actions_by_hand(saved.hand_id)[0].source_image
        == "/frames/t000069.00.jpg"
    )
    db.close()
