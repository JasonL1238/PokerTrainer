"""Tests for compact manual solver-spot entry."""

from __future__ import annotations

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Session
from poker_tracker.services.manual_spot_entry import (
    ManualSpotDefaults,
    ManualSpotInput,
    PostflopActionInput,
    build_manual_spot,
    format_postflop_line,
    parse_manual_spot_lines,
    parse_postflop_line,
    save_manual_spot,
    save_manual_spots,
    validate_manual_spot,
)
from poker_tracker.services.study_readiness import accounting_is_established
from poker_tracker.solver.eligibility import prepare_solver_spot


def _srp_bb_spot(**overrides) -> ManualSpotInput:
    payload = dict(
        hand_number=1,
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
        hero_position="BB",
        villain_position="BTN",
        table_size=6,
        starting_stack=100.0,
        pot_type="single_raised",
        opener="villain",
        open_to=2.5,
        postflop_actions=(
            PostflopActionInput("flop", "hero", "check"),
            PostflopActionInput("flop", "villain", "bet", 3.75),
            PostflopActionInput("flop", "hero", "call", 3.75),
        ),
        winner="hero",
    )
    payload.update(overrides)
    return ManualSpotInput(**payload)


def test_validate_manual_spot_requires_solver_basics() -> None:
    assert validate_manual_spot(_srp_bb_spot(hero_cards="Ah")) == [
        "Enter both Hero hole cards, like Ah Qs."
    ]
    assert any(
        "flop board" in error
        for error in validate_manual_spot(_srp_bb_spot(board_cards="Qd 7s"))
    )


def test_build_manual_spot_creates_blinds_open_and_call() -> None:
    built = build_manual_spot(_srp_bb_spot())
    assert built.hand.game_type == "NLHE cash"
    assert built.hand.source_type == "manual"
    assert [player.player_key for player in built.players] == [
        "hero",
        "villain",
        "sb_fold",
    ]
    kinds = [(action.street, action.action_type, action.amount) for action in built.actions]
    assert ("preflop", "post_blind", 0.5) in kinds
    assert ("preflop", "post_blind", 1.0) in kinds
    assert ("preflop", "fold", None) in kinds
    assert ("preflop", "raise", 2.5) in kinds
    assert ("preflop", "call", 1.5) in kinds
    assert ("flop", "check", None) in kinds
    assert ("flop", "bet", 3.75) in kinds


def test_save_manual_spot_is_solver_eligible(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "manual_spot.db")
    db.init_db()
    session = db.create_session(Session(name="Spot entry"))
    assert session.id is not None

    hand, accounting, warnings = save_manual_spot(db, session.id, _srp_bb_spot())
    assert hand.id is not None
    assert warnings == []
    assert accounting.is_authoritative
    assert accounting_is_established(hand, accounting)

    players = db.fetch_players_by_hand(hand.id)
    actions = db.fetch_actions_by_hand(hand.id)
    prepared = prepare_solver_spot(hand, players, actions, accounting)
    assert prepared.eligibility.eligible
    assert prepared.spot is not None
    assert prepared.spot.pot_type == "single_raised"
    assert prepared.spot.board == "Qd 7s 2c"
    assert prepared.spot.hero_cards == "Ah Qs"
    db.close()


def test_save_three_bet_spot_is_solver_eligible(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "manual_3bet.db")
    db.init_db()
    session = db.create_session(Session(name="3bet spot"))
    assert session.id is not None

    spot = _srp_bb_spot(
        pot_type="three_bet",
        opener="villain",
        three_bettor="hero",
        open_to=2.5,
        three_bet_to=9.0,
        postflop_actions=(
            PostflopActionInput("flop", "hero", "bet", 6.0),
            PostflopActionInput("flop", "villain", "call", 6.0),
        ),
    )
    hand, accounting, _ = save_manual_spot(db, session.id, spot)
    prepared = prepare_solver_spot(
        hand,
        db.fetch_players_by_hand(hand.id),
        db.fetch_actions_by_hand(hand.id),
        accounting,
    )
    assert prepared.eligibility.eligible
    assert prepared.spot is not None
    assert prepared.spot.pot_type == "three_bet"
    db.close()


def test_parse_postflop_line_infers_call_and_streets() -> None:
    actions, errors = parse_postflop_line(
        "x/b3.5/c | x/b8/f",
        hero_position="BB",
        villain_position="BTN",
    )
    assert errors == []
    assert [
        (row.street, row.actor, row.action_type, row.amount) for row in actions
    ] == [
        ("flop", "hero", "check", None),
        ("flop", "villain", "bet", 3.5),
        ("flop", "hero", "call", 3.5),
        ("turn", "hero", "check", None),
        ("turn", "villain", "bet", 8.0),
        ("turn", "hero", "fold", None),
    ]
    assert format_postflop_line(actions) == "hx/vb3.5/hc3.5 | hx/vb8/hf"


def test_parse_postflop_line_respects_actor_prefixes() -> None:
    actions, errors = parse_postflop_line(
        "vb3/hc/hx",
        hero_position="BTN",
        villain_position="BB",
    )
    assert errors == []
    assert [(row.actor, row.action_type, row.amount) for row in actions] == [
        ("villain", "bet", 3.0),
        ("hero", "call", 3.0),
        ("hero", "check", None),
    ]


def test_parse_manual_spot_lines_batch(tmp_path) -> None:
    defaults = ManualSpotDefaults()
    parsed = parse_manual_spot_lines(
        "\n".join(
            [
                "AhQs | Qd7s2c | x/b3.5/c | hero",
                "KdKh | Ah9c2s | BB vs BTN | 3bet | open2.5 | 3b9 | x/b6/c | villain",
                "# comment ignored",
            ]
        ),
        defaults,
        starting_hand_number=4,
    )
    assert parsed.errors == ()
    assert len(parsed.spots) == 2
    assert parsed.spots[0].hand_number == 4
    assert parsed.spots[0].winner == "hero"
    assert parsed.spots[0].pot_type == "single_raised"
    assert parsed.spots[1].hand_number == 5
    assert parsed.spots[1].pot_type == "three_bet"
    assert parsed.spots[1].three_bet_to == 9.0
    assert parsed.spots[1].winner == "villain"

    db = PokerDatabase(tmp_path / "manual_batch.db")
    db.init_db()
    session = db.create_session(Session(name="Batch"))
    assert session.id is not None
    results = save_manual_spots(db, session.id, parsed.spots)
    assert len(results) == 2
    assert all(accounting.is_authoritative for _, accounting, _ in results)
    hands = db.fetch_hands_by_session(session.id)
    assert [hand.hand_number for hand in hands] == [4, 5]
    db.close()


def test_parse_manual_spot_lines_reports_bad_line() -> None:
    parsed = parse_manual_spot_lines(
        "AhQs | Qd7s2c",
        ManualSpotDefaults(),
        starting_hand_number=1,
    )
    assert parsed.spots == ()
    assert any("need hero cards" in error for error in parsed.errors)


# ---------------------------------------------------------------------------
# An unfinished betting line must not be saveable, and must not be settleable
# ---------------------------------------------------------------------------


def _unclosed(line: str, winner: str = "hero") -> ManualSpotInput:
    """The compact spot an operator gets by typing the decision point and stopping."""
    actions, errors = parse_postflop_line(
        line, hero_position="BB", villain_position="BTN"
    )
    # The parser is deliberately not the gate: every one of these lines is legal
    # token by token, which is why nothing downstream noticed they never ended.
    assert errors == [], errors
    return _srp_bb_spot(postflop_actions=actions, winner=winner)


def test_a_line_that_never_answers_a_wager_is_refused() -> None:
    """`x/b3.5` is Hero checking, Villain betting, and Hero never acting.

    It used to validate clean, save clean, and reconcile clean: the ledger
    refunded Villain's uncalled 3.5 -- the arithmetic of a hand Hero folded --
    while the whole pot was pushed to the declared winner, publishing a hero
    result that no completion of the hand produces at the product's highest
    evidence class.
    """
    errors = validate_manual_spot(_unclosed("x/b3.5"))
    assert errors, "an unanswered 3.5 BB bet was accepted as a completed hand"
    assert any("never answers" in error and "flop" in error for error in errors), errors


def test_the_refusal_is_about_closure_not_about_who_was_declared_the_winner() -> None:
    """Declaring the bettor instead does not make the missing action exist.

    Those numbers happen to coincide with "the opponent folded", but the record
    does not say anyone folded, so the agreement is a coincidence rather than an
    observation -- and one keystroke turns it back into the fabrication above.
    """
    assert validate_manual_spot(_unclosed("x/b3.5", winner="villain"))


def test_every_shape_of_unfinished_line_is_refused_not_just_the_reported_one() -> None:
    """The class is "the recorded action sequence is not closed", not one line."""
    for line in ("x/b3.5", "b3/r9", "x/b3.5/c | x/b8", "b3/r9/r27"):
        assert validate_manual_spot(_unclosed(line)), line


def test_a_closed_line_is_still_saveable_even_when_it_is_short() -> None:
    """Closure is the rule, not completeness: a study spot may stop early.

    `x/b3.5/c` stops before the turn and `x/x` stops with nobody facing
    anything. Both are ordinary things to record, and refusing them would turn a
    fabrication guard into a coverage limitation.
    """
    assert validate_manual_spot(_unclosed("x/b3.5/c")) == []
    assert validate_manual_spot(_unclosed("x/x")) == []
    assert validate_manual_spot(_unclosed("x/b3.5/f", winner="villain")) == []


def test_a_shove_that_is_folded_to_is_closed_and_still_accepted() -> None:
    """An uncalled wager IS how most hands end -- when the fold is recorded.

    The manual form gives both seats one starting stack, so "the opponent was
    all-in for less" is not a shape it can express; the ledger-level version of
    that control lives in tests/test_hand_accounting_service.py. What this pins
    is that shoving the last chip and being folded to stays saveable.
    """
    actions, errors = parse_postflop_line(
        "b17.5/f", hero_position="BB", villain_position="BTN"
    )
    assert errors == [], errors
    spot = _srp_bb_spot(
        postflop_actions=actions, starting_stack=20.0, winner="hero"
    )
    assert validate_manual_spot(spot) == []


def test_a_declared_winner_who_folded_is_refused_before_anything_is_written(
    tmp_path,
) -> None:
    """It used to raise LedgerError AFTER the hand row was committed.

    The operator got a traceback and a stranded hand carrying a settlement that
    could never reconcile, from a form that had reported no validation errors.
    """
    spot = _unclosed("x/b3.5/f", winner="hero")
    errors = validate_manual_spot(spot)
    assert any("folded" in error for error in errors), errors

    db = PokerDatabase(tmp_path / "folded_winner.db")
    db.init_db()
    session = db.create_session(Session(name="Folded winner"))
    assert session.id is not None
    try:
        save_manual_spot(db, session.id, spot)
    except ValueError:
        pass
    else:  # pragma: no cover - the save must not succeed
        raise AssertionError("a folded player was accepted as the declared winner")
    assert db.fetch_hands_by_session(session.id) == []
    db.close()


def test_the_batch_paste_path_refuses_the_same_line(tmp_path) -> None:
    """All four entry points funnel through validate_manual_spot; prove two."""
    parsed = parse_manual_spot_lines(
        "AhQs | Qd7s2c | x/b3.5 | hero",
        ManualSpotDefaults(),
        starting_hand_number=1,
    )
    assert parsed.spots == ()
    assert any("never answers" in error for error in parsed.errors), parsed.errors


def test_validation_refuses_what_the_ledger_would_refuse_after_the_write(
    tmp_path,
) -> None:
    """"No validation errors" has to mean "saving this will work".

    ``save_manual_spot`` commits the hand, its players and its actions and only
    then builds a ledger, so a rule enforced solely by the ledger reaches the
    operator as a traceback rather than as a field error. Both of these lines are
    legal token by token and were accepted by ``validate_manual_spot``.
    """
    acts_after_folding = _unclosed("x/b3.5/f | x/x", winner="villain")
    assert any(
        "cannot act again" in error
        for error in validate_manual_spot(acts_after_folding)
    ), validate_manual_spot(acts_after_folding)

    over_stack = _unclosed("x/b3.5/c | ai97.5", winner="hero")
    assert any(
        "stack" in error for error in validate_manual_spot(over_stack)
    ), validate_manual_spot(over_stack)

    db = PokerDatabase(tmp_path / "impossible.db")
    db.init_db()
    session = db.create_session(Session(name="Impossible"))
    assert session.id is not None
    for spot in (acts_after_folding, over_stack):
        try:
            save_manual_spot(db, session.id, spot)
        except ValueError:
            pass
        else:  # pragma: no cover - the save must not succeed
            raise AssertionError("an impossible spot was saved")
    assert db.fetch_hands_by_session(session.id) == []
    db.close()


def test_an_all_in_for_exactly_the_stack_is_still_accepted() -> None:
    """The over-commit rule is the ledger's, not a stricter one."""
    actions, errors = parse_postflop_line(
        "x/ai97.5/c", hero_position="BB", villain_position="BTN"
    )
    assert errors == [], errors
    assert validate_manual_spot(_srp_bb_spot(postflop_actions=actions)) == []
