from __future__ import annotations

import pytest

from poker_tracker.math.accounting import (
    LedgerAction,
    LedgerError,
    LedgerPlayer,
    RakePolicy,
    build_hand_ledger,
)


def _player(name: str, stack: float = 100, seat: int | None = None) -> LedgerPlayer:
    return LedgerPlayer(name=name, starting_stack=stack, seat=seat)


def _action(
    player: str,
    kind: str,
    amount: float = 0,
    street: str = "preflop",
) -> LedgerAction:
    """Create an accounting action.

    Amounts in this module are always *incremental chips committed*, including
    raises. A persistence/UI adapter may accept raise-to amounts, but it must
    normalize them before invoking the accounting reducer.
    """
    return LedgerAction(player=player, street=street, kind=kind, amount=amount)


def test_heads_up_blinds_call_and_showdown_are_conserved() -> None:
    ledger = build_hand_ledger(
        players=[_player("SB", seat=1), _player("BB", seat=2)],
        actions=[
            _action("SB", "post_blind", 0.5),
            _action("BB", "post_blind", 1),
            _action("SB", "call", 0.5),
            _action("BB", "check", street="flop"),
            _action("SB", "check", street="flop"),
        ],
        winners={0: ("BB",)},
    )

    assert ledger.contributions == pytest.approx({"SB": 1, "BB": 1})
    assert ledger.refunds == pytest.approx({"SB": 0, "BB": 0})
    assert ledger.gross_pot == pytest.approx(2)
    assert ledger.rake == pytest.approx(0)
    assert ledger.net_pot == pytest.approx(2)
    assert ledger.payouts == pytest.approx({"SB": 0, "BB": 2})
    assert ledger.net_results == pytest.approx({"SB": -1, "BB": 1})
    assert ledger.is_balanced is True
    assert sum(ledger.net_results.values()) + ledger.rake == pytest.approx(0)


def test_action_snapshots_expose_review_math_from_one_source_of_truth() -> None:
    ledger = build_hand_ledger(
        players=[_player("SB", seat=1), _player("BB", seat=2)],
        actions=[
            _action("SB", "post_blind", 0.5),
            _action("BB", "post_blind", 1),
            _action("SB", "call", 0.5),
            _action("BB", "check", street="flop"),
        ],
        winners={0: ("SB",)},
    )

    call = ledger.snapshots[2]
    assert call.player == "SB"
    assert call.pot_before == pytest.approx(1.5)
    assert call.pot_after == pytest.approx(2)
    assert call.stack_before == pytest.approx(99.5)
    assert call.stack_after == pytest.approx(99)
    assert call.to_call_before == pytest.approx(0.5)
    assert call.call_increment == pytest.approx(0.5)
    assert call.street_contribution_after == pytest.approx(1)
    assert call.hand_contribution_after == pytest.approx(1)
    assert call.effective_stack_before == pytest.approx(99)

    flop = ledger.snapshots[3]
    assert flop.pot_before == pytest.approx(2)
    assert flop.effective_stack_before == pytest.approx(99)
    assert flop.spr_before == pytest.approx(49.5)


def test_folded_dead_money_remains_in_a_multiway_pot() -> None:
    ledger = build_hand_ledger(
        players=[_player("A"), _player("B"), _player("C")],
        actions=[
            _action("A", "bet", 5, "flop"),
            _action("B", "call", 5, "flop"),
            _action("C", "call", 5, "flop"),
            _action("A", "fold", street="turn"),
            _action("B", "bet", 15, "turn"),
            _action("C", "call", 15, "turn"),
        ],
        winners={0: ("B",), 1: ("B",)},
    )

    assert ledger.contributions == pytest.approx({"A": 5, "B": 20, "C": 20})
    assert ledger.gross_pot == pytest.approx(45)
    assert ledger.payouts == pytest.approx({"A": 0, "B": 45, "C": 0})
    assert ledger.net_results == pytest.approx({"A": -5, "B": 25, "C": -20})
    assert all("A" not in pot.eligible_players for pot in ledger.pots)
    assert ledger.is_balanced is True


def test_all_ins_build_main_and_side_pots_by_contribution_layer() -> None:
    ledger = build_hand_ledger(
        players=[
            _player("A", stack=100, seat=1),
            _player("B", stack=60, seat=2),
            _player("C", stack=20, seat=3),
            _player("D", stack=100, seat=4),
        ],
        actions=[
            _action("A", "all-in", 100, "turn"),
            _action("B", "all-in", 60, "turn"),
            _action("C", "all-in", 20, "turn"),
            _action("D", "all-in", 100, "turn"),
        ],
        winners={0: ("C",), 1: ("B",), 2: ("A",)},
    )

    assert [pot.amount for pot in ledger.pots] == pytest.approx([80, 120, 80])
    assert [pot.eligible_players for pot in ledger.pots] == [
        ("A", "B", "C", "D"),
        ("A", "B", "D"),
        ("A", "D"),
    ]
    assert ledger.refunds == pytest.approx({"A": 0, "B": 0, "C": 0, "D": 0})
    assert ledger.gross_pot == pytest.approx(280)
    assert ledger.payouts == pytest.approx({"A": 80, "B": 120, "C": 80, "D": 0})
    assert ledger.net_results == pytest.approx({"A": -20, "B": 60, "C": 60, "D": -100})
    assert ledger.is_balanced is True


def test_unmatched_all_in_overbet_is_returned_before_settlement() -> None:
    ledger = build_hand_ledger(
        players=[_player("A", stack=100), _player("B", stack=60)],
        actions=[
            _action("A", "all-in", 100, "river"),
            _action("B", "all-in", 60, "river"),
        ],
        winners={0: ("B",)},
    )

    assert ledger.contributions == pytest.approx({"A": 100, "B": 60})
    assert ledger.refunds == pytest.approx({"A": 40, "B": 0})
    assert len(ledger.pots) == 1
    assert ledger.pots[0].amount == pytest.approx(120)
    assert ledger.gross_pot == pytest.approx(120)
    assert ledger.payouts == pytest.approx({"A": 0, "B": 120})
    # The refund offsets part of A's raw contribution.
    assert ledger.net_results == pytest.approx({"A": -60, "B": 60})
    assert ledger.is_balanced is True


def test_folded_contribution_still_matches_an_overbet_before_refund() -> None:
    ledger = build_hand_ledger(
        players=[
            _player("A", stack=100),
            _player("B", stack=60),
            _player("C", stack=80),
        ],
        actions=[
            _action("C", "bet", 80, "river"),
            _action("B", "all-in", 60, "river"),
            _action("A", "all-in", 100, "river"),
            # C declines the last 20. Its prior commitment remains dead money
            # and therefore matches A through the 80-chip contribution level.
            _action("C", "fold", street="river"),
        ],
        winners={0: ("B",), 1: ("A",)},
    )

    assert ledger.refunds == pytest.approx({"A": 20, "B": 0, "C": 0})
    assert [pot.amount for pot in ledger.pots] == pytest.approx([180, 40])
    assert [pot.eligible_players for pot in ledger.pots] == [
        ("A", "B"),
        ("A",),
    ]
    assert ledger.payouts == pytest.approx({"A": 40, "B": 180, "C": 0})
    assert ledger.net_results == pytest.approx({"A": -40, "B": 120, "C": -80})
    assert ledger.is_balanced is True


def test_antes_are_dead_money_and_do_not_reduce_the_preflop_call_increment() -> None:
    ledger = build_hand_ledger(
        players=[_player("BTN"), _player("SB"), _player("BB")],
        actions=[
            _action("BTN", "ante", 1),
            _action("SB", "ante", 1),
            _action("BB", "ante", 1),
            _action("SB", "post_blind", 1),
            _action("BB", "post_blind", 2),
            _action("BTN", "fold"),
            _action("SB", "call", 1),
            _action("BB", "check"),
        ],
        winners={0: ("SB",), 1: ("SB",)},
    )

    assert ledger.contributions == pytest.approx({"BTN": 1, "SB": 3, "BB": 3})
    assert ledger.gross_pot == pytest.approx(7)
    assert ledger.payouts == pytest.approx({"BTN": 0, "SB": 7, "BB": 0})
    assert ledger.net_results == pytest.approx({"BTN": -1, "SB": 4, "BB": -3})
    assert ledger.snapshots[6].to_call_before == pytest.approx(1)
    assert ledger.snapshots[6].call_increment == pytest.approx(1)
    assert ledger.is_balanced is True


def test_rake_is_capped_and_included_in_conservation() -> None:
    ledger = build_hand_ledger(
        players=[_player("A"), _player("B")],
        actions=[
            _action("A", "bet", 50, "river"),
            _action("B", "call", 50, "river"),
        ],
        winners={0: ("A",)},
        rake=RakePolicy(rate=0.05, cap=3),
    )

    assert ledger.gross_pot == pytest.approx(100)
    assert ledger.rake == pytest.approx(3)
    assert ledger.net_pot == pytest.approx(97)
    assert ledger.payouts == pytest.approx({"A": 97, "B": 0})
    assert ledger.net_results == pytest.approx({"A": 47, "B": -50})
    assert sum(ledger.net_results.values()) + ledger.rake == pytest.approx(0)
    assert ledger.is_balanced is True


def test_no_flop_no_drop_waives_rake_until_a_flop_action_exists() -> None:
    players = [_player("A"), _player("B")]
    preflop_actions = [
        _action("A", "bet", 10),
        _action("B", "call", 10),
    ]
    policy = RakePolicy(rate=0.05, cap=3, no_flop_no_drop=True)

    preflop_ledger = build_hand_ledger(
        players,
        preflop_actions,
        winners={0: ("A",)},
        rake=policy,
    )
    flop_ledger = build_hand_ledger(
        players,
        [*preflop_actions, _action("A", "check", street="flop")],
        winners={0: ("A",)},
        rake=policy,
    )
    board_runout_ledger = build_hand_ledger(
        players,
        preflop_actions,
        winners={0: ("A",)},
        rake=policy,
        flop_seen=True,
    )

    assert preflop_ledger.rake == pytest.approx(0)
    assert preflop_ledger.payouts["A"] == pytest.approx(20)
    assert flop_ledger.rake == pytest.approx(1)
    assert flop_ledger.payouts["A"] == pytest.approx(19)
    assert board_runout_ledger.rake == pytest.approx(1)
    assert board_runout_ledger.payouts["A"] == pytest.approx(19)


def test_effective_stack_and_spr_exclude_players_already_all_in() -> None:
    ledger = build_hand_ledger(
        players=[
            _player("A", stack=100),
            _player("B", stack=10),
            _player("C", stack=100),
        ],
        actions=[
            _action("B", "all-in", 10),
            _action("A", "call", 10),
            _action("C", "call", 10),
            _action("A", "bet", 10, "flop"),
            _action("C", "call", 10, "flop"),
        ],
        winners={0: ("B",), 1: ("A",)},
    )

    first_deep_action = ledger.snapshots[1]
    flop_bet = ledger.snapshots[3]
    assert first_deep_action.effective_stack_range_before == pytest.approx((100, 100))
    assert first_deep_action.spr_range_before == pytest.approx((10, 10))
    assert flop_bet.effective_stack_range_before == pytest.approx((90, 90))
    assert flop_bet.spr_range_before == pytest.approx((3, 3))


def test_split_pot_uses_declared_order_for_an_indivisible_odd_chip() -> None:
    ledger = build_hand_ledger(
        players=[_player("A", seat=1), _player("B", seat=2), _player("C", seat=3)],
        actions=[
            _action("A", "bet", 5, "river"),
            _action("B", "call", 5, "river"),
            _action("C", "call", 5, "river"),
        ],
        winners={0: ("B", "C")},
        # A one-chip accounting quantum makes the fifteenth chip indivisible.
        rake=RakePolicy(rounding_unit=1),
        odd_chip_order=("C", "B"),
    )

    assert ledger.gross_pot == pytest.approx(15)
    assert ledger.payouts == pytest.approx({"A": 0, "B": 7, "C": 8})
    assert ledger.net_results == pytest.approx({"A": -5, "B": 2, "C": 3})
    assert ledger.is_balanced is True


def test_dead_money_is_settled_without_being_charged_to_a_player() -> None:
    ledger = build_hand_ledger(
        players=[_player("A"), _player("B")],
        actions=[
            _action("A", "bet", 5, "river"),
            _action("B", "call", 5, "river"),
        ],
        winners={0: ("B",)},
        dead_money=2,
    )

    assert ledger.gross_pot == pytest.approx(12)
    assert ledger.payouts == pytest.approx({"A": 0, "B": 12})
    assert ledger.net_results == pytest.approx({"A": -5, "B": 7})
    assert sum(ledger.net_results.values()) == pytest.approx(2)
    assert ledger.is_balanced is True


def test_missing_winners_produces_an_unsettled_but_inspectable_ledger() -> None:
    ledger = build_hand_ledger(
        players=[_player("A"), _player("B")],
        actions=[
            _action("A", "bet", 10, "river"),
            _action("B", "call", 10, "river"),
        ],
    )

    assert ledger.gross_pot == pytest.approx(20)
    assert ledger.payouts == pytest.approx({"A": 0, "B": 0})
    assert ledger.is_balanced is False
    assert ledger.warnings
    assert any("winner" in warning.lower() or "unsettled" in warning.lower() for warning in ledger.warnings)


@pytest.mark.parametrize(
    ("player_specs", "action_specs", "winners", "dead_money"),
    [
        (
            [("A", 100), ("A", 100)],
            [],
            None,
            0,
        ),
        (
            [("A", 100)],
            [("missing", "bet", 1)],
            None,
            0,
        ),
        (
            [("A", 100)],
            [("A", "bet", -1)],
            None,
            0,
        ),
        (
            [("A", 10), ("B", 10)],
            [("A", "bet", 11), ("B", "call", 10)],
            {0: ("B",)},
            0,
        ),
        (
            [("A", 100), ("B", 100)],
            [("A", "fold", 0), ("A", "bet", 1)],
            None,
            0,
        ),
        (
            [("A", 100), ("B", 100)],
            [("A", "bet", 1), ("B", "call", 1), ("A", "fold", 0)],
            {0: ("A",)},
            0,
        ),
        (
            [("A", 100), ("B", 100)],
            [("A", "bet", 1), ("B", "call", 1)],
            {0: ("missing",)},
            0,
        ),
        (
            [("A", 100), ("B", 100)],
            [("A", "bet", 1), ("B", "call", 1)],
            {1: ("A",)},
            0,
        ),
        (
            [("A", 100), ("B", 100)],
            [("A", "bet", 1), ("B", "call", 1)],
            None,
            -1,
        ),
    ],
    ids=[
        "duplicate-player",
        "unknown-action-player",
        "negative-contribution",
        "overcommit",
        "action-after-fold",
        "folded-winner",
        "unknown-winner",
        "winner-for-missing-pot",
        "negative-dead-money",
    ],
)
def test_structurally_invalid_ledgers_raise(
    player_specs: list[tuple[str, float]],
    action_specs: list[tuple[str, str, float]],
    winners: dict[int, tuple[str, ...]] | None,
    dead_money: float,
) -> None:
    with pytest.raises(LedgerError):
        players = [_player(name, stack) for name, stack in player_specs]
        actions = [_action(player, kind, amount) for player, kind, amount in action_specs]
        build_hand_ledger(
            players=players,
            actions=actions,
            winners=winners,
            dead_money=dead_money,
        )


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {"rate": -0.01},
        {"rate": 1.01},
        {"cap": -1},
        {"rounding_unit": 0},
    ],
)
def test_invalid_rake_policy_raises(policy_kwargs: dict[str, float]) -> None:
    with pytest.raises(LedgerError):
        policy = RakePolicy(**policy_kwargs)
        build_hand_ledger(
            players=[_player("A"), _player("B")],
            actions=[_action("A", "bet", 1), _action("B", "call", 1)],
            winners={0: ("A",)},
            rake=policy,
        )
