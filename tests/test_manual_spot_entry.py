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
