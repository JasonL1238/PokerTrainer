"""Re-importing a session's own export must not add a copy that looks original.

Import is an APPEND and stays one -- no payload may address, replace or edit an
existing session -- but the append used to be silent. Exporting a session and
re-importing it (after a scare, or because the operator forgot) minted a second
session with the same name and the same date holding a full copy of every hand,
with nothing anywhere distinguishing it from the first: the session list showed
identical buttons and Overview reported doubled sessions, hands and BB as if
that were the operator's poker.

These tests pin the two halves of the repair: a re-import is recognised from the
rows it wrote and says so, and a session that merely resembles another is left
alone.
"""

from __future__ import annotations

from datetime import date

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    export_session,
    import_hands_into_session,
    import_session,
)
from poker_tracker.persistence.models import Hand, Session


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _seed(db: PokerDatabase, *, name: str = "Fri night", hands: int = 3) -> Session:
    session = db.create_session(
        Session(name=name, date_played=date(2026, 7, 30), stakes="0.05/0.10")
    )
    assert session.id is not None
    for number in range(1, hands + 1):
        db.create_hand(
            Hand(
                session_id=session.id,
                hand_number=number,
                hero_cards="Ah Qs",
                hero_bb_won=10.0,
                notes=f"hand {number}",
            )
        )
    return session


def test_a_reimported_session_is_named_and_annotated_as_a_copy() -> None:
    db = make_db()
    original = _seed(db)
    payload = export_session(db, original.id)

    second = import_session(db, payload)
    third = import_session(db, payload)

    names = [session.name for session in db.fetch_sessions()]
    # Still appended -- three sessions, nine hands -- but no longer three
    # indistinguishable rows in the session list.
    assert len(names) == 3
    assert len(names) == len(set(names))
    assert original.name in names
    for copy in (second, third):
        assert copy.name.startswith("Fri night (re-imported copy #")
        assert str(copy.id) in copy.name
        assert f"session {original.id}" in copy.notes
        assert "BOTH copies are counted" in copy.notes
    db.close()


def test_the_copy_marker_names_the_session_it_duplicates_not_the_previous_copy() -> None:
    """The third import duplicates the ORIGINAL, which is the one to delete against."""
    db = make_db()
    original = _seed(db)
    payload = export_session(db, original.id)

    import_session(db, payload)
    third = import_session(db, payload)

    assert f"session {original.id}" in third.notes
    db.close()


def test_a_first_import_into_a_fresh_database_is_not_labelled() -> None:
    db = make_db()
    source = make_db()
    original = _seed(source)
    payload = export_session(source, original.id)

    imported = import_session(db, payload)

    assert imported.name == "Fri night"
    assert imported.notes == ""
    source.close()
    db.close()


def test_a_session_that_only_shares_a_name_and_date_is_not_a_copy() -> None:
    """The comparison is over content, not over the label two sessions share."""
    db = make_db()
    original = _seed(db)
    payload = export_session(db, original.id)
    # Same name, same date, one hand fewer.
    other = _seed(db, hands=2)

    imported = import_session(db, payload)

    assert other.name == "Fri night"
    assert imported.name.startswith("Fri night (re-imported copy #")
    assert f"session {original.id}" in imported.notes
    db.close()


def test_a_payload_whose_hands_differ_is_not_labelled_a_copy() -> None:
    db = make_db()
    original = _seed(db)
    payload = export_session(db, original.id)
    payload["hands"][0]["hand"]["hero_cards"] = "Kh Kd"

    imported = import_session(db, payload)

    assert imported.name == "Fri night"
    assert len(db.fetch_sessions()) == 2
    db.close()


def test_appending_hands_into_a_chosen_session_is_unaffected() -> None:
    """``import_hands_into_session`` is the operator's explicit append."""
    db = make_db()
    original = _seed(db)
    payload = export_session(db, original.id)
    target = db.create_session(Session(name="Target", date_played=date(2026, 7, 31)))

    refreshed = import_hands_into_session(db, payload, target.id)

    assert refreshed.name == "Target"
    assert refreshed.notes == ""
    assert len(db.fetch_sessions()) == 2
    assert len(db.fetch_hands_by_session(target.id)) == 3
    db.close()
