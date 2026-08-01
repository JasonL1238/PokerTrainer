"""Import validation: what is refused, with what message, and before what.

Three properties are pinned here.

1. Validation COMPLETES before any application-data write. Not "the transaction
   rolls back" -- that was already true and is a weaker guarantee -- but that a
   refused payload causes no INSERT to be attempted at all. The discriminator is
   ``sqlite3.Connection.total_changes``, which counts rows changed since the
   connection opened and is NOT decremented by a rollback.

2. Every malformed shape the phase names is refused with a message naming what
   was wrong and where. An operator may have spent an hour producing the file;
   "invalid import" does not tell them which of five hundred records to fix.

3. Duplicate import appends. It cannot overwrite, because the module has no
   update path at all.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    MAX_IMPORT_PAYLOAD_DEPTH,
    MAX_IMPORT_PAYLOAD_TEXT_CHARS,
    MAX_IMPORT_PAYLOAD_VALUES,
    ImportValidationError,
    export_session,
    import_hands_into_session,
    import_session,
    validate_import_payload,
)
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    Session,
    SettlementEntry,
)


def _make_db(path: Path | str = ":memory:") -> PokerDatabase:
    db = PokerDatabase(str(path))
    db.init_db()
    return db


def _hand(number: int = 1, **overrides: Any) -> dict[str, Any]:
    data = Hand(session_id=1, hand_number=number, source_type="manual").model_dump(
        mode="json"
    )
    data.update(overrides)
    return data


def _player(**overrides: Any) -> dict[str, Any]:
    data = dict(hand_id=1, player_key="hero", player_name="Hero")
    data.update(overrides)
    return HandPlayer(**data).model_dump(mode="json")


def _action(**overrides: Any) -> dict[str, Any]:
    data = dict(hand_id=1, street="flop", player_name="Hero", action_type="bet")
    data.update(overrides)
    return Action(**data).model_dump(mode="json")


def _entry(**overrides: Any) -> dict[str, Any]:
    data = dict(
        hand_id=1,
        entry_type="award",
        pot_index=0,
        player_name="Hero",
        amount=10.0,
    )
    data.update(overrides)
    return SettlementEntry(**data).model_dump(mode="json")


def _payload(*hand_entries: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "export_version": 6,
        "session": Session(name="Validation").model_dump(mode="json"),
        "hands": list(hand_entries) or [{"hand": _hand(), "players": [], "actions": []}],
    }
    payload.update(overrides)
    return payload


def _hand_entry(**parts: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"hand": _hand(), "players": [], "actions": []}
    entry.update(parts)
    return entry


def _assert_refused(db: PokerDatabase, payload: Any) -> None:
    """The payload must not import. Which exception it raises is not this claim."""
    try:
        import_session(db, payload)
    except Exception:  # noqa: BLE001 - the type is asserted by the message tests
        return
    pytest.fail("the payload was accepted")


def _full_dump(db: PokerDatabase) -> list[tuple[str, tuple[Any, ...]]]:
    """Every row of every table, ordered, as comparable plain data."""
    rows: list[tuple[str, tuple[Any, ...]]] = []
    tables = sorted(
        row["name"]
        for row in db._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    )
    for table in tables:
        for row in db._execute(f"SELECT * FROM {table}").fetchall():
            rows.append((table, tuple(row)))
    rows.sort(key=repr)
    return rows


# ---------------------------------------------------------------------------
# 1. Validation completes before any application-data write.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "broken"),
    [
        (
            "duplicate player key",
            _hand_entry(
                hand=_hand(4),
                players=[
                    _player(player_key="hero", player_name="Hero"),
                    _player(player_key="hero", player_name="Villain"),
                ],
            ),
        ),
        (
            "second hero",
            _hand_entry(
                hand=_hand(4),
                players=[
                    _player(player_key="a", player_name="Hero", is_hero=True),
                    _player(player_key="b", player_name="Villain", is_hero=True),
                ],
            ),
        ),
        (
            "action attributed to nobody",
            _hand_entry(
                hand=_hand(4),
                players=[_player(player_key="a", player_name="Hero")],
                actions=[_action(player_name="Ghost")],
            ),
        ),
        (
            "field of the wrong type",
            _hand_entry(hand=_hand(4, table_size="six")),
        ),
    ],
)
def test_no_row_is_written_when_a_later_record_is_invalid(
    tmp_path: Path, label: str, broken: dict[str, Any]
) -> None:
    """Three valid hands then a broken fourth: not one INSERT is attempted.

    Before this, the shape and type checks were pre-write but the uniqueness and
    relational ones were not: the session row and three hands landed, and the
    failure arrived from the middle of the write pass as a
    ``sqlite3.IntegrityError`` -- an exception the Streamlit import surface does
    not catch. Atomicity still held, so nothing survived, but "validation
    completes before application-data writes" was false as stated, and the
    operator got a traceback naming a database index.
    """
    path = tmp_path / "before-write.sqlite3"
    db = _make_db(path)
    existing = db.create_session(Session(name="Already here"))
    db.create_hand(Hand(session_id=existing.id, hand_number=1))

    before_dump = _full_dump(db)
    # Rollback does not decrement total_changes, so this separates "no write was
    # attempted" from "a partial write was undone".
    before_changes = db._connection.total_changes

    payload = _payload(
        _hand_entry(hand=_hand(1)),
        _hand_entry(hand=_hand(2)),
        _hand_entry(hand=_hand(3)),
        broken,
    )
    # Any exception, deliberately: which type a refusal carries is pinned by the
    # message tests below. What this one asserts is that nothing was written, and
    # catching only ImportValidationError would let a regression fail here for
    # the wrong reason -- on the exception type rather than on the writes.
    _assert_refused(db, payload)

    assert db._connection.total_changes == before_changes, (
        f"{label}: the import wrote rows before finishing validation"
    )
    assert _full_dump(db) == before_dump
    assert len(db.fetch_sessions()) == 1
    db.close()


def test_a_refused_import_leaves_the_database_file_byte_identical(
    tmp_path: Path,
) -> None:
    """The file on disk, not merely the rows a query can see."""
    path = tmp_path / "untouched.sqlite3"
    db = _make_db(path)
    session = db.create_session(Session(name="Keep me"))
    db.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="Ah Qs"))
    db.close()

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    reopened = _make_db(path)
    _assert_refused(
        reopened,
        _payload(
            _hand_entry(
                hand=_hand(9),
                players=[
                    _player(player_key="a", seat_index=2, player_name="A"),
                    _player(player_key="b", seat_index=2, player_name="B"),
                ],
            )
        ),
    )
    reopened.close()

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not path.with_name(path.name + "-wal").exists()


def test_the_validator_needs_no_database_at_all() -> None:
    """The structural proof behind the ordering claim.

    ``validate_import_payload`` takes no database, so every check it performs
    necessarily precedes every write. ``import_session``'s write block reads
    only what it returns.
    """
    validated = validate_import_payload(
        _payload(
            _hand_entry(
                hand=_hand(1),
                players=[_player(player_key="hero", player_name="Hero")],
                actions=[_action(player_name="Hero")],
            )
        )
    )

    assert validated.session.name == "Validation"
    assert len(validated.hands) == 1
    assert validated.hands[0].players[0].player_key == "hero"
    # Linked during validation, not discovered by the writer.
    assert validated.hands[0].actions[0].player_key == "hero"
    assert validated.hands[0].actions[0].action_index == 1


# ---------------------------------------------------------------------------
# 2. Malformed input, one shape per test, each with its own actionable error.
# ---------------------------------------------------------------------------


def test_wrong_field_type_names_the_record_and_the_field() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(db, _payload(_hand_entry(hand=_hand(1, pot_size="a lot"))))

    message = str(caught.value)
    assert "hands[0].hand" in message
    assert "pot_size" in message
    assert db.fetch_sessions() == []
    db.close()


def test_a_non_object_where_a_record_belongs_names_what_was_found() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError, match=r"hands\[0\].players\[1\]"):
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[_player(player_key="a", player_name="A"), "not a player"],
                )
            ),
        )
    assert db.fetch_sessions() == []
    db.close()


def test_a_list_field_holding_a_scalar_is_refused_by_name() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError, match=r"hands\[0\].actions: expected a list"):
        import_session(db, _payload(_hand_entry(hand=_hand(1), actions="fold")))
    db.close()


def test_duplicate_player_key_in_one_hand_names_both_records() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[
                        _player(player_key="hero", player_name="Hero"),
                        _player(player_key="hero", player_name="Villain"),
                    ],
                )
            ),
        )
    message = str(caught.value)
    assert "hands[0].players[1]" in message
    assert "players[0]" in message
    assert "player_key 'hero'" in message
    assert db.fetch_sessions() == []
    db.close()


def test_duplicate_seat_index_in_one_hand_names_both_records() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[
                        _player(player_key="a", seat_index=4, player_name="A"),
                        _player(player_key="b", seat_index=4, player_name="B"),
                    ],
                )
            ),
        )
    assert "seat_index 4" in str(caught.value)
    assert "players[0]" in str(caught.value)
    db.close()


def test_two_heroes_in_one_hand_are_refused_before_the_write() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError, match="only one Hero"):
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[
                        _player(player_key="a", player_name="A", is_hero=True),
                        _player(player_key="b", player_name="B", is_hero=True),
                    ],
                )
            ),
        )
    db.close()


def test_duplicate_hand_number_in_one_payload_names_both_hands() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(db, _payload(_hand_entry(hand=_hand(7)), _hand_entry(hand=_hand(7))))

    message = str(caught.value)
    assert "hands[1]" in message
    assert "hands[0]" in message
    assert "hand_number 7" in message
    db.close()


def test_an_action_referencing_an_absent_player_key_is_refused() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[_player(player_key="hero", player_name="Hero")],
                    actions=[_action(player_key="seat:9", player_name="Hero")],
                )
            ),
        )
    message = str(caught.value)
    assert "hands[0].actions[0]" in message
    assert "'seat:9'" in message
    db.close()


def test_an_action_naming_a_player_the_roster_omits_is_refused() -> None:
    """The relationship a payload can break without breaking any constraint."""
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[_player(player_key="hero", player_name="Hero")],
                    actions=[_action(player_name="Villain")],
                )
            ),
        )
    message = str(caught.value)
    assert "hands[0].actions[0]" in message
    assert "'Villain'" in message
    assert "1 player" in message
    db.close()


def test_an_action_naming_one_of_two_identically_named_players_is_refused() -> None:
    """Ambiguous is refused, not silently left unattributed with a NULL key."""
    db = _make_db()
    with pytest.raises(ImportValidationError, match="shared by 2 players"):
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[
                        _player(player_key="a", player_name="Player"),
                        _player(player_key="b", player_name="Player"),
                    ],
                    actions=[_action(player_name="Player")],
                )
            ),
        )
    db.close()


def test_a_settlement_award_to_an_absent_player_is_refused() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(
            db,
            _payload(
                _hand_entry(
                    hand=_hand(1),
                    players=[_player(player_key="hero", player_name="Hero")],
                    settlement_entries=[_entry(player_name="Ghost")],
                )
            ),
        )
    assert "hands[0].settlement_entries[0]" in str(caught.value)
    assert "'Ghost'" in str(caught.value)
    db.close()


def test_a_hand_with_no_roster_at_all_still_imports_and_arrives_blocked() -> None:
    """The declared limit of the relationship rule, so it is reviewable.

    A hand that declares NO players is an older shape, not a contradiction: there
    is no roster for the actions to disagree with. Refusing it would make legacy
    payloads unimportable. The ledger refuses to derive accounting for such a
    hand, so it arrives visibly blocked rather than silently wrong.
    """
    db = _make_db()
    imported = import_session(
        db,
        _payload(
            _hand_entry(
                hand=_hand(1, source_type="cv_import", completion_status="uncertain"),
                players=[],
                actions=[_action(player_name="Nobody")],
            )
        ),
    )
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert db.fetch_actions_by_hand(hand.id)[0].player_key is None
    assert hand.completion_status != "complete"
    assert hand.review_status != "reviewed"
    db.close()


def test_an_unsupported_schema_version_names_what_is_understood() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError) as caught:
        import_session(db, _payload(export_version=99))

    message = str(caught.value)
    assert "99" in message
    assert "[1, 2, 3, 4, 5, 6]" in message
    assert db.fetch_sessions() == []
    db.close()


def test_a_schema_version_of_the_wrong_type_is_distinguishable_in_the_message() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError, match="export_version '6'"):
        import_session(db, _payload(export_version="6"))
    db.close()


def test_a_truncated_payload_names_the_part_that_is_missing() -> None:
    db = _make_db()
    with pytest.raises(ImportValidationError, match="no 'session' object"):
        import_session(db, {"export_version": 6, "hands": []})

    with pytest.raises(ImportValidationError, match=r"hands\[0\]: no 'hand' object"):
        import_session(db, _payload({"players": [], "actions": []}))

    assert db.fetch_sessions() == []
    db.close()


def test_a_payload_that_is_not_an_object_is_refused_without_an_attribute_error() -> None:
    db = _make_db()
    for junk in ("a string", 7, ["a", "list"], None):
        with pytest.raises(ImportValidationError, match="payload: expected an object"):
            import_session(db, junk)
    db.close()


def test_oversized_text_is_refused_by_import_session_itself() -> None:
    """The ceiling used to live only in the Streamlit upload widgets.

    Every programmatic caller -- the CV pipeline exporter, any future upload
    surface -- was bounded by nothing at all.
    """
    db = _make_db()
    payload = _payload(
        _hand_entry(hand=_hand(1, notes="x" * (MAX_IMPORT_PAYLOAD_TEXT_CHARS + 1)))
    )
    with pytest.raises(ImportValidationError, match="MB of text"):
        import_session(db, payload)
    assert db.fetch_sessions() == []
    db.close()


def test_a_deeply_nested_payload_is_refused_rather_than_recursed_into() -> None:
    db = _make_db()
    nested: Any = "leaf"
    for _ in range(MAX_IMPORT_PAYLOAD_DEPTH + 2):
        nested = {"nested": nested}
    payload = _payload(_hand_entry(hand=_hand(1, completion_evidence=nested)))

    with pytest.raises(ImportValidationError, match="nests deeper"):
        import_session(db, payload)
    db.close()


def test_a_payload_with_too_many_values_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling itself, exercised at a size a test can afford to build."""
    monkeypatch.setattr(
        "poker_tracker.persistence.import_export.MAX_IMPORT_PAYLOAD_VALUES", 50
    )
    db = _make_db()
    with pytest.raises(ImportValidationError, match="more than 50 values"):
        import_session(
            db, _payload(*[_hand_entry(hand=_hand(n + 1)) for n in range(20)])
        )
    db.close()


def test_the_value_ceiling_is_large_enough_for_a_real_session() -> None:
    """A guard on the constant: the ceiling must not refuse ordinary work."""
    assert MAX_IMPORT_PAYLOAD_VALUES >= 1_000_000
    assert MAX_IMPORT_PAYLOAD_TEXT_CHARS == 10 * 1024 * 1024
    assert MAX_IMPORT_PAYLOAD_DEPTH >= 32


def test_an_unhashable_export_version_is_refused_rather_than_raising_type_error() -> None:
    """Found by the fuzz suite: raw payload values reaching a hashed lookup.

    ``[] in SUPPORTED_IMPORT_VERSIONS`` raised ``TypeError: unhashable type:
    'list'`` from inside the import, which no import surface catches.
    """
    db = _make_db()
    with pytest.raises(ImportValidationError, match="Unsupported export_version"):
        import_session(db, _payload(export_version=[]))
    assert db.fetch_sessions() == []
    db.close()


def test_an_unhashable_completion_status_re_derives_instead_of_raising() -> None:
    """The sibling of the above, and it lands on the conservative default.

    ``[] in _IMPORT_COMPLETION_RESTRICTION`` raised the same TypeError. A
    declared completion status is never trusted at any export version, so a
    value of the wrong shape is simply another untrusted claim: it is replaced
    by what the hand's own evidence derives, which for an evidence-free manual
    hand is ``not_applicable``.
    """
    db = _make_db()
    imported = import_session(
        db, _payload(_hand_entry(hand=_hand(1, completion_status=[])))
    )
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "not_applicable"
    assert hand.review_status != "reviewed"
    db.close()


@pytest.mark.parametrize(
    "field", ["derived_result_substituted", "unreadable_columns"]
)
def test_a_payload_cannot_carry_a_read_time_derivation(field: str) -> None:
    """Found by the fuzz suite: a field the export drops but the import accepted.

    Both fields are marked ``exclude=True`` and both document that no export,
    payload or database row can carry them. That was true of the export half
    only: ``Hand(**payload)`` accepted ``derived_result_substituted`` and
    ``create_hand`` then refused the object from the middle of the write pass.
    """
    db = _make_db()
    value: Any = True if field == "derived_result_substituted" else ["hero_cards"]
    baseline = db._connection.total_changes

    imported = import_session(db, _payload(_hand_entry(hand=_hand(1, **{field: value}))))
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert getattr(hand, field) in (False, ())
    assert db._connection.total_changes > baseline, "the import must have written"
    db.close()


def test_every_refusal_is_a_value_error_so_existing_callers_keep_catching_it() -> None:
    """``ImportValidationError`` is new; the surfaces that catch refusals are not."""
    assert issubclass(ImportValidationError, ValueError)
    db = _make_db()
    with pytest.raises(ValueError):
        import_session(db, {"export_version": 6, "hands": []})
    db.close()


# ---------------------------------------------------------------------------
# 3. Duplicate import.
# ---------------------------------------------------------------------------


def test_reimporting_the_same_payload_appends_a_second_independent_session() -> None:
    """The stated contract: import creates, it never addresses an existing session."""
    source = _make_db()
    session = source.create_session(Session(name="Twice"))
    source.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="Ah Qs"))
    source.create_hand(Hand(session_id=session.id, hand_number=2, hero_cards="Kd Kc"))
    payload = export_session(source, session.id)

    target = _make_db()
    first = import_session(target, copy.deepcopy(payload))
    second = import_session(target, copy.deepcopy(payload))

    assert first.id != second.id
    assert len(target.fetch_sessions()) == 2
    assert len(target.fetch_hands_by_session(first.id)) == 2
    assert len(target.fetch_hands_by_session(second.id)) == 2
    source.close()
    target.close()


def test_import_cannot_touch_the_session_whose_id_the_payload_carries() -> None:
    """The regression test for the no-overwrite guarantee.

    The payload declares session id 1 and the target database already has a
    session 1 holding its own hands. If anyone ever added an update-by-id path,
    this is the test that catches it: the existing session's every row must be
    untouched and the import must land somewhere else entirely.
    """
    source = _make_db()
    session = source.create_session(Session(name="Payload session"))
    source.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="Ah Qs"))
    payload = export_session(source, session.id)
    assert payload["session"]["id"] == 1

    target = _make_db()
    resident = target.create_session(Session(name="Resident", notes="Do not touch."))
    assert resident.id == 1
    for number in (1, 2, 3):
        target.create_hand(Hand(session_id=resident.id, hand_number=number))
    resident_hands = [
        hand.model_dump(mode="json") for hand in target.fetch_hands_by_session(resident.id)
    ]

    imported = import_session(target, payload)

    assert imported.id != resident.id
    reloaded = target.fetch_session(resident.id)
    assert reloaded is not None
    assert reloaded.model_dump(mode="json") == resident.model_dump(mode="json")
    assert [
        hand.model_dump(mode="json") for hand in target.fetch_hands_by_session(resident.id)
    ] == resident_hands
    assert len(target.fetch_hands_by_session(imported.id)) == 1
    source.close()
    target.close()


def test_appending_into_an_existing_session_requires_naming_it() -> None:
    """The only path that adds to an existing session takes an explicit target."""
    source = _make_db()
    session = source.create_session(Session(name="Appendable"))
    source.create_hand(Hand(session_id=session.id, hand_number=1))
    payload = export_session(source, session.id)

    target = _make_db()
    destination = target.create_session(Session(name="Destination"))
    target.create_hand(Hand(session_id=destination.id, hand_number=1))

    refreshed = import_hands_into_session(target, payload, destination.id)

    assert refreshed.id == destination.id
    assert len(target.fetch_sessions()) == 1
    numbers = sorted(
        hand.hand_number for hand in target.fetch_hands_by_session(destination.id)
    )
    assert numbers == [1, 2], "a colliding hand number is renumbered, not overwritten"

    with pytest.raises(ValueError, match="Session not found"):
        import_hands_into_session(target, payload, 4242)
    source.close()
    target.close()


def test_an_append_refused_mid_payload_leaves_the_target_session_untouched() -> None:
    """The append path validates before writing too, because it delegates."""
    target = _make_db()
    destination = target.create_session(Session(name="Destination"))
    target.create_hand(Hand(session_id=destination.id, hand_number=1))
    before = _full_dump(target)
    before_changes = target._connection.total_changes

    try:
        import_hands_into_session(
            target,
            _payload(
                _hand_entry(hand=_hand(5)),
                _hand_entry(
                    hand=_hand(6),
                    players=[
                        _player(player_key="x", player_name="A"),
                        _player(player_key="x", player_name="B"),
                    ],
                ),
            ),
            destination.id,
        )
    except Exception:  # noqa: BLE001 - see _assert_refused
        pass
    else:
        pytest.fail("the payload was accepted")

    assert target._connection.total_changes == before_changes
    assert _full_dump(target) == before
    target.close()
