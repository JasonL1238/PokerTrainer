"""Tests for compact manual solver-spot entry."""

from __future__ import annotations

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Session
from poker_tracker.services.manual_spot_entry import (
    ManualSpotInput,
    PostflopActionInput,
    build_manual_spot,
    save_manual_spot,
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
