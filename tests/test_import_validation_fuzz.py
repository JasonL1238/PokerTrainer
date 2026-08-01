"""Property and fuzz tests for import-payload validation (PLAN Phase 14).

An import payload is user-supplied JSON. The example-based tests cover the
malformed shapes somebody thought of; these generate the ones nobody did, by
taking a valid export and corrupting it the way a hand edit, a partial write, or
a producer written against a different version would.

Three invariants hold for every payload, well-formed or not:

    the validator returns a ValidatedImport, or raises ImportValidationError
    -- never TypeError, AttributeError, KeyError, IntegrityError, or RecursionError

    a refused payload leaves the database with zero rows changed
    -- not "rolled back": no INSERT is attempted at all

    a payload the validator accepts, the writer can write
    -- a validator that accepts what the writer refuses is a half-write waiting
       for the right input

The third is the one that matters most. Both of the shipped defects this suite
exists to prevent were exactly that gap: uniqueness and relational rules that
lived only in SQLite's indexes, so validation passed and the write pass raised.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    EXPORT_VERSION,
    ImportValidationError,
    ValidatedImport,
    export_session,
    import_session,
    validate_import_payload,
)
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

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Anything a hand-edited field might hold, including the values the models are
# built to refuse.
JUNK = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=True, allow_infinity=True, width=32),
    st.text(max_size=12),
    st.lists(st.integers(max_value=10), max_size=3),
    st.dictionaries(st.text(max_size=4), st.integers(max_value=10), max_size=3),
)


def _make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _valid_payload() -> dict[str, Any]:
    """A real export of a session carrying one row of every exported entity."""
    db = _make_db()
    session = db.create_session(Session(name="Fuzz source", stakes="1/2"))
    for number in (1, 2):
        hand = db.create_hand(
            Hand(
                session_id=session.id,
                hand_number=number,
                hero_cards="Ah Qs",
                board_cards="Qd 7s 2c",
                table_size=6,
                tags=["BIG_POT"],
            )
        )
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="hero",
                seat_index=0,
                player_name="Hero",
                starting_stack=200,
                is_hero=True,
            )
        )
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="villain",
                seat_index=1,
                player_name="Villain",
                starting_stack=180,
            )
        )
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key="hero",
                street="flop",
                player_name="Hero",
                action_type="bet",
                amount=10,
                source_image="frames/a.png",
            )
        )
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key="villain",
                street="flop",
                player_name="Villain",
                action_type="call",
                amount=10,
            )
        )
        db.upsert_hand_settlement(
            HandSettlement(hand_id=hand.id, status="settled", gross_pot=20, net_pot=20)
        )
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key="hero",
                    player_name="Hero",
                    amount=20,
                )
            ],
        )
        db.create_hand_review(
            HandReview(
                hand_id=hand.id,
                hand_summary="s",
                theory_coach="t",
                exploit_coach="e",
                study_lesson="l",
            )
        )
        db.create_coaching_response(
            CoachingResponse(
                hand_id=hand.id,
                session_id=session.id,
                provider_name="anthropic",
                model_name="claude",
                raw_prompt="p",
                raw_response="r",
                review_type="hand",
            )
        )
        db.create_hand_correction(
            HandCorrection(hand_id=hand.id, correction_type="hand_facts")
        )
        db.create_hand_issue(
            HandIssue(
                hand_id=hand.id,
                issue_types=["cards"],
                description="d",
                evidence_snapshot={"k": "v"},
            )
        )
    payload = export_session(db, session.id)
    db.close()
    return payload


_TEMPLATE = _valid_payload()
_HAND_KEYS = (
    "players",
    "actions",
    "settlement_entries",
    "reviews",
    "coaching_reviews",
    "corrections",
    "issues",
)


@st.composite
def corrupted_payload(draw: Any) -> Any:
    """A real export with one or more plausible corruptions applied."""
    payload = copy.deepcopy(_TEMPLATE)
    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        payload = draw(st.sampled_from(_MUTATIONS))(draw, payload)
    return payload


def _pick_hand_entry(draw: Any, payload: Any) -> dict[str, Any] | None:
    """One hand entry, or None once an earlier mutation removed the structure.

    Every mutation composes with every other, so each has to survive a payload
    a previous one already made nonsense of. A strategy that raises while
    GENERATING an input tells you nothing about the code under test.
    """
    if not isinstance(payload, dict):
        return None
    hands = payload.get("hands")
    if not isinstance(hands, list) or not hands:
        return None
    entry = hands[draw(st.integers(0, len(hands) - 1))]
    return entry if isinstance(entry, dict) else None


def _pick_records(draw: Any, entry: dict[str, Any], key: str) -> list[Any] | None:
    records = entry.get(key)
    if not isinstance(records, list) or not records:
        return None
    return records


def _corrupt_hand_field(draw: Any, payload: Any) -> Any:
    entry = _pick_hand_entry(draw, payload)
    if entry is None or not isinstance(entry.get("hand"), dict):
        return payload
    entry["hand"][draw(st.sampled_from(sorted(Hand.model_fields)))] = draw(JUNK)
    return payload


def _corrupt_child_field(draw: Any, payload: Any) -> Any:
    entry = _pick_hand_entry(draw, payload)
    if entry is None:
        return payload
    records = _pick_records(draw, entry, draw(st.sampled_from(_HAND_KEYS)))
    if records is None:
        return payload
    record = records[draw(st.integers(0, len(records) - 1))]
    if not isinstance(record, dict) or not record:
        return payload
    record[draw(st.sampled_from(sorted(record)))] = draw(JUNK)
    return payload


def _drop_a_key(draw: Any, payload: Any) -> Any:
    """A truncated write, or a producer that never emitted the field."""
    entry = _pick_hand_entry(draw, payload)
    if entry is None:
        return payload
    hand = entry.get("hand")
    if draw(st.booleans()) and isinstance(hand, dict) and hand:
        hand.pop(draw(st.sampled_from(sorted(hand))), None)
    else:
        entry.pop(draw(st.sampled_from(("hand",) + _HAND_KEYS)), None)
    return payload


def _replace_a_list_with_a_scalar(draw: Any, payload: Any) -> Any:
    entry = _pick_hand_entry(draw, payload)
    if entry is None:
        return payload
    entry[draw(st.sampled_from(_HAND_KEYS))] = draw(JUNK)
    return payload


def _duplicate_a_record(draw: Any, payload: Any) -> Any:
    """The identifier collisions: two players on one key, two hands on one number."""
    entry = _pick_hand_entry(draw, payload)
    if entry is None:
        return payload
    if draw(st.booleans()):
        payload["hands"].append(copy.deepcopy(entry))
        return payload
    records = _pick_records(draw, entry, draw(st.sampled_from(_HAND_KEYS)))
    if records is not None:
        records.append(copy.deepcopy(records[0]))
    return payload


def _break_a_reference(draw: Any, payload: Any) -> Any:
    """Rename or re-key a player without updating what points at them."""
    entry = _pick_hand_entry(draw, payload)
    if entry is None:
        return payload
    players = _pick_records(draw, entry, "players")
    if players is None:
        return payload
    player = players[draw(st.integers(0, len(players) - 1))]
    if not isinstance(player, dict):
        return payload
    player[draw(st.sampled_from(("player_key", "player_name")))] = draw(
        st.text(min_size=1, max_size=6)
    )
    return payload


def _corrupt_the_root(draw: Any, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    key = draw(st.sampled_from(("export_version", "session", "hands", "coaching_reviews")))
    if draw(st.booleans()):
        payload.pop(key, None)
    else:
        payload[key] = draw(JUNK)
    return payload


def _corrupt_the_session(draw: Any, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    session = payload.get("session")
    if not isinstance(session, dict) or not session:
        return payload
    session[draw(st.sampled_from(sorted(Session.model_fields)))] = draw(JUNK)
    return payload


_MUTATIONS = (
    _corrupt_hand_field,
    _corrupt_child_field,
    _drop_a_key,
    _replace_a_list_with_a_scalar,
    _duplicate_a_record,
    _break_a_reference,
    _corrupt_the_root,
    _corrupt_the_session,
)


def _assert_refusal_is_actionable(exc: ImportValidationError) -> None:
    message = str(exc)
    assert message.strip(), "a refusal with no message is not actionable"
    assert message != exc.location, "a refusal must say what was wrong, not only where"
    if exc.location:
        assert message.startswith(f"{exc.location}: ")


@given(corrupted_payload())
@SETTINGS
def test_a_corrupted_payload_is_accepted_cleanly_or_refused_actionably(
    payload: Any,
) -> None:
    try:
        validated = validate_import_payload(payload)
    except ImportValidationError as exc:
        _assert_refusal_is_actionable(exc)
        return
    assert isinstance(validated, ValidatedImport)
    assert validated.export_version in {1, 2, 3, 4, 5, EXPORT_VERSION}


@given(corrupted_payload())
@SETTINGS
def test_a_refused_payload_writes_no_row_and_an_accepted_one_writes_cleanly(
    payload: Any,
) -> None:
    """The gap that shipped twice: a validator that accepts what the writer refuses."""
    db = _make_db()
    baseline = db._connection.total_changes
    try:
        session = import_session(db, payload)
    except ImportValidationError as exc:
        _assert_refusal_is_actionable(exc)
        # Rollback does not decrement total_changes, so this asserts that no
        # INSERT was attempted, not merely that none survived.
        assert db._connection.total_changes == baseline
        assert db.fetch_sessions() == []
        assert db.fetch_all_hands() == []
    else:
        assert session.id is not None
        assert len(db.fetch_sessions()) == 1
        for hand in db.fetch_hands_by_session(session.id):
            assert hand.id is not None
            db.fetch_players_by_hand(hand.id)
            db.fetch_actions_by_hand(hand.id)
    finally:
        db.close()


_JSON = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-100, max_value=100),
        st.text(max_size=8),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=6), children, max_size=4),
    ),
    max_leaves=25,
)


@given(_JSON)
@SETTINGS
def test_arbitrary_json_never_escapes_as_a_non_validation_error(payload: Any) -> None:
    """Anything that parses as JSON may be uploaded; nothing may crash the import."""
    db = _make_db()
    baseline = db._connection.total_changes
    try:
        import_session(db, payload)
    except ImportValidationError as exc:
        _assert_refusal_is_actionable(exc)
        assert db._connection.total_changes == baseline
    else:
        assert len(db.fetch_sessions()) == 1
    finally:
        db.close()


@given(
    st.dictionaries(
        st.sampled_from(sorted(Session.model_fields)), JUNK, min_size=1, max_size=4
    )
)
@SETTINGS
def test_a_corrupt_session_record_never_leaves_a_session_behind(
    overrides: dict[str, Any],
) -> None:
    payload = copy.deepcopy(_TEMPLATE)
    payload["session"].update(overrides)
    db = _make_db()
    try:
        import_session(db, payload)
    except ImportValidationError:
        assert db.fetch_sessions() == []
    else:
        assert len(db.fetch_sessions()) == 1
    finally:
        db.close()


def test_the_template_this_suite_mutates_is_itself_importable() -> None:
    """A fuzz suite whose starting point is already refused proves nothing."""
    db = _make_db()
    session = import_session(db, copy.deepcopy(_TEMPLATE))
    hands = db.fetch_hands_by_session(session.id)

    assert len(hands) == 2
    for hand in hands:
        assert len(db.fetch_players_by_hand(hand.id)) == 2
        assert len(db.fetch_actions_by_hand(hand.id)) == 2
        assert db.fetch_hand_settlement(hand.id) is not None
        assert db.fetch_settlement_entries(hand.id)
        assert db.fetch_reviews_by_hand(hand.id)
        assert db.fetch_coaching_reviews_by_hand(hand.id)
        assert db.fetch_hand_corrections(hand.id)
        assert db.fetch_hand_issues(hand_id=hand.id)
    db.close()


@pytest.mark.parametrize("depth", [200, 2000])
def test_a_pathologically_nested_payload_raises_no_recursion_error(depth: int) -> None:
    """The traversal keeps its own stack, so nesting cannot exhaust Python's."""
    nested: Any = "leaf"
    for _ in range(depth):
        nested = [nested]

    db = _make_db()
    with pytest.raises(ImportValidationError, match="nests deeper"):
        import_session(db, {"export_version": 6, "session": nested, "hands": []})
    db.close()
