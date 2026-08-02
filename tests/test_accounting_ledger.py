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
        winners={0: ("B",)},
    )

    # A leaving for less caps nobody -- B and C contest every chip in the hand --
    # so this is one pot, not a main pot plus a layer holding A's abandoned 5.
    assert len(ledger.pots) == 1
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
        winners={0: ("SB",)},
    )

    # Antes are dead money, so they pool into the main pot instead of forming a
    # layer per poster. The folded button's ante is contested by whoever wins the
    # hand; it used to become a side pot only the button was a contributor to.
    assert len(ledger.pots) == 1
    assert ledger.pots[0].eligible_players == ("SB", "BB")
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


# --- Forced posts PLAN Phase 7 names but nothing pinned ---------------------


def test_a_live_straddle_raises_the_amount_to_call() -> None:
    """A straddle is a live post: it sets the price, unlike an ante.

    Blinds 1/2, UTG straddles to 4. The next player faces 4, not 2, and the
    straddler's own 4 counts toward what they have already put in.
    """
    players = [
        LedgerPlayer(name="sb", starting_stack=100, seat=0),
        LedgerPlayer(name="bb", starting_stack=100, seat=1),
        LedgerPlayer(name="straddler", starting_stack=100, seat=2),
        LedgerPlayer(name="utg1", starting_stack=100, seat=3),
    ]
    actions = [
        LedgerAction(player="sb", street="preflop", kind="post_blind", amount=1),
        LedgerAction(player="bb", street="preflop", kind="post_blind", amount=2),
        LedgerAction(player="straddler", street="preflop", kind="post_blind", amount=4),
        LedgerAction(player="utg1", street="preflop", kind="call", amount=4),
        LedgerAction(player="sb", street="preflop", kind="fold"),
        LedgerAction(player="bb", street="preflop", kind="fold"),
        LedgerAction(player="straddler", street="preflop", kind="check"),
    ]
    # Four distinct commitment levels (1, 2, 4) make three layers; the
    # straddler is eligible for all of them.
    layers = build_hand_ledger(players, actions).pots
    ledger = build_hand_ledger(
        players, actions, {pot.index: ("straddler",) for pot in layers}
    )

    # The caller faced the straddle, not the big blind.
    call_snapshot = next(s for s in ledger.snapshots if s.player == "utg1")
    assert call_snapshot.to_call_before == 4
    # Checking behind a straddle you already paid for is legal.
    assert ledger.is_legal, ledger.legality_issues
    assert ledger.gross_pot == 11
    assert ledger.is_balanced


def test_a_dead_blind_reaches_the_pot_without_buying_any_of_the_call() -> None:
    """A dead post is money owed to the table, not money toward this bet.

    The returning player posts 2 dead and still owes the full big blind to see
    a flop. Charging the dead chip against the call would let them in cheap and
    would under-count the pot by the same amount.
    """
    players = [
        LedgerPlayer(name="bb", starting_stack=100, seat=0),
        LedgerPlayer(name="returning", starting_stack=100, seat=1),
    ]
    actions = [
        LedgerAction(player="bb", street="preflop", kind="post_blind", amount=2),
        LedgerAction(
            player="returning",
            street="preflop",
            kind="post_blind",
            amount=2,
            is_live_post=False,
        ),
        LedgerAction(player="returning", street="preflop", kind="call", amount=2),
        LedgerAction(player="bb", street="preflop", kind="check"),
    ]
    ledger = build_hand_ledger(players, actions, {0: ("bb",)})

    # The dead post did not reduce what was owed.
    call_snapshot = next(
        s for s in ledger.snapshots if s.player == "returning" and s.kind == "call"
    )
    assert call_snapshot.to_call_before == 2
    assert ledger.is_legal, ledger.legality_issues
    # 2 blind + 2 dead + 2 call: every chip posted is in the pot.
    assert ledger.contributions["returning"] == 4
    assert ledger.gross_pot == 6
    assert ledger.is_balanced


def test_three_way_all_in_builds_two_side_pots_and_conserves() -> None:
    """The layering case: three distinct stacks make a main and two sides."""
    players = [
        LedgerPlayer(name="short", starting_stack=10, seat=0),
        LedgerPlayer(name="mid", starting_stack=40, seat=1),
        LedgerPlayer(name="deep", starting_stack=100, seat=2),
    ]
    actions = [
        LedgerAction(player="short", street="preflop", kind="all-in", amount=10),
        LedgerAction(player="mid", street="preflop", kind="all-in", amount=40),
        LedgerAction(player="deep", street="preflop", kind="all-in", amount=100),
    ]
    unsettled = build_hand_ledger(players, actions)
    # 30 main (10 x 3), 60 side (30 x 2), and 60 uncalled returned to deep.
    assert [pot.amount for pot in unsettled.pots] == [30, 60]
    assert unsettled.refunds["deep"] == 60

    ledger = build_hand_ledger(
        players, actions, {0: ("short",), 1: ("mid",)}
    )
    assert ledger.payouts["short"] == 30
    assert ledger.payouts["mid"] == 60
    assert ledger.net_results["short"] == 20
    assert ledger.net_results["mid"] == 20
    assert ledger.net_results["deep"] == -40
    assert sum(ledger.net_results.values()) == 0
    assert ledger.is_balanced


def test_a_lone_ante_is_not_refunded_as_an_uncalled_bet() -> None:
    """Dead money nobody matched stays in the pot.

    Refunds used to be measured against total contributions, so an ante or dead
    blind that no opponent matched looked exactly like an unmatched overbet and
    was handed back. A single button ante of 5 was returned in full and the pot
    was short by the same 5 chips — money that left the table entirely.
    """
    players = [_player("BTN"), _player("BB")]
    actions = [
        _action("BTN", "ante", 5),
        _action("BB", "post_blind", 2),
        _action("BTN", "call", 2),
        _action("BB", "check"),
    ]
    ledger = build_hand_ledger(players, actions, {0: ("BB",)})

    assert ledger.refunds == pytest.approx({"BTN": 0, "BB": 0})
    assert ledger.contributions == pytest.approx({"BTN": 7, "BB": 2})
    assert ledger.gross_pot == pytest.approx(9)
    assert ledger.payouts == pytest.approx({"BTN": 0, "BB": 9})
    assert ledger.net_results == pytest.approx({"BTN": -7, "BB": 7})
    assert sum(ledger.net_results.values()) == pytest.approx(0)
    assert ledger.is_balanced is True


def test_an_unmatched_live_bet_is_still_refunded() -> None:
    """The repair must not stop real uncalled bets from coming back."""
    players = [_player("A"), _player("B")]
    actions = [
        _action("A", "bet", 50, "river"),
        _action("B", "fold", street="river"),
    ]
    # Every chip comes back, so there is no contestable pot to declare.
    ledger = build_hand_ledger(players, actions)
    assert ledger.refunds["A"] == pytest.approx(50)
    assert ledger.gross_pot == pytest.approx(0)
    assert ledger.net_results == pytest.approx({"A": 0, "B": 0})


def test_dead_money_and_a_live_overbet_are_settled_independently() -> None:
    """One player holding both an unmatched ante and an unmatched bet."""
    players = [_player("A", stack=200), _player("B", stack=200)]
    actions = [
        _action("A", "ante", 3),
        _action("A", "bet", 50, "river"),
        _action("B", "call", 20, "river"),
        _action("B", "fold", street="river"),
    ]
    ledger = build_hand_ledger(players, actions, {0: ("A",)})

    # 30 of the bet was never called and comes back; the ante does not.
    assert ledger.refunds == pytest.approx({"A": 30, "B": 0})
    assert ledger.gross_pot == pytest.approx(43)
    assert sum(ledger.net_results.values()) == pytest.approx(0)


# --- Adversarial round 2 findings -------------------------------------------


def test_a_player_all_in_for_their_ante_can_still_win_the_pot() -> None:
    """Eligibility must follow the chips, including dead ones — and stop there.

    Deriving pot eligibility from live contributions alone left a player whose
    entire stack went in as an ante eligible for no pot at all — while their
    chips sat in one they could not be declared the winner of. A hand the short
    stack won became unrecordable, which is routine once a stack is at or below
    the ante.

    Widening the first layer's eligible set to fix that overshot: the layer was
    still sized by the LIVE wagering, so the short stack contested 35 chips
    having covered 5. It must be eligible for a layer capped at what it covered
    — 5 from each of the three seats — and for nothing above it.
    """
    players = [
        _player("short", stack=5),
        _player("B", stack=100),
        _player("C", stack=100),
    ]
    actions = [
        _action("short", "ante", 5),
        _action("B", "bet", 15),
        _action("C", "call", 15),
    ]
    unsettled = build_hand_ledger(players, actions)
    assert "short" in unsettled.pots[0].eligible_players
    assert [pot.amount for pot in unsettled.pots] == pytest.approx([15, 20])
    assert "short" not in unsettled.pots[1].eligible_players

    ledger = build_hand_ledger(players, actions, {0: ("short",), 1: ("B",)})
    assert ledger.payouts["short"] == pytest.approx(15)
    assert ledger.net_results["short"] == pytest.approx(10)
    assert sum(ledger.net_results.values()) == pytest.approx(0)
    assert ledger.is_balanced is True


def test_a_folded_ante_poster_is_not_eligible() -> None:
    """Dead money reaching the pot does not buy a folded player a claim on it."""
    players = [_player("A"), _player("B"), _player("C")]
    actions = [
        _action("C", "ante", 1),
        _action("C", "fold"),
        _action("A", "bet", 10),
        _action("B", "call", 10),
    ]
    ledger = build_hand_ledger(players, actions)
    assert "C" not in ledger.pots[0].eligible_players


def test_a_layer_split_off_by_a_folded_blind_is_not_a_layer_at_all() -> None:
    """A boundary that caps nobody is not a pot boundary.

    Blinds 1/2 with a straddle to 4: the small and big blinds fold for less.
    Nobody was all-in, nobody's eligibility is capped -- every chip is contested
    by exactly the same two players, which is the definition of ONE pot. Calling
    the extra layers "side pots" because their index is not zero was a false
    statement about the hand; calling them a "Dead-money layer" was a second
    false statement about chips that are ordinary live wagering. There is no
    name for them because they should not be emitted.
    """
    players = [
        LedgerPlayer(name="sb", starting_stack=100, seat=0),
        LedgerPlayer(name="bb", starting_stack=100, seat=1),
        LedgerPlayer(name="straddler", starting_stack=100, seat=2),
        LedgerPlayer(name="utg1", starting_stack=100, seat=3),
    ]
    actions = [
        LedgerAction(player="sb", street="preflop", kind="post_blind", amount=1),
        LedgerAction(player="bb", street="preflop", kind="post_blind", amount=2),
        LedgerAction(player="straddler", street="preflop", kind="post_blind", amount=4),
        LedgerAction(player="utg1", street="preflop", kind="call", amount=4),
        LedgerAction(player="sb", street="preflop", kind="fold"),
        LedgerAction(player="bb", street="preflop", kind="fold"),
        LedgerAction(player="straddler", street="preflop", kind="check"),
    ]
    layers = build_hand_ledger(players, actions).pots

    assert [pot.amount for pot in layers] == pytest.approx([11])
    assert [pot.cause for pot in layers] == ["main"]
    assert [pot.label for pot in layers] == ["Main pot"]
    # Same contest over every chip, which is why there is only one pot.
    assert {pot.eligible_players for pot in layers} == {("straddler", "utg1")}


def test_only_a_layer_that_caps_eligibility_is_called_a_side_pot() -> None:
    """A short stack all-in for less is the one thing that makes a side pot."""
    players = [
        _player("short", stack=10, seat=0),
        _player("mid", stack=40, seat=1),
        _player("deep", stack=100, seat=2),
    ]
    actions = [
        _action("short", "all-in", 10),
        _action("mid", "all-in", 40),
        _action("deep", "call", 40),
    ]
    layers = build_hand_ledger(players, actions).pots

    assert [pot.cause for pot in layers] == ["main", "side"]
    assert [pot.label for pot in layers] == ["Main pot", "Side pot"]
    # The side pot is exactly the layer the short stack cannot win.
    assert layers[0].eligible_players == ("short", "mid", "deep")
    assert layers[1].eligible_players == ("mid", "deep")


def test_a_seat_all_in_for_only_an_ante_caps_the_layer_above_it() -> None:
    """A forced post can be the whole stack, and that still makes a side pot.

    The short stack's chips are dead, but they are still what it covered, so the
    level set has to be drawn from total commitments. Pooling them into the main
    pot and widening that pot's eligible set instead left the layer sized by the
    live wagering the short stack never matched -- and this test asserted that as
    correct while its own name states the opposite.
    """
    players = [_player("short", stack=5), _player("B"), _player("C")]
    actions = [
        _action("short", "ante", 5),
        _action("B", "bet", 15),
        _action("C", "call", 15),
    ]
    layers = build_hand_ledger(players, actions).pots

    assert [pot.cause for pot in layers] == ["main", "side"]
    assert [pot.amount for pot in layers] == pytest.approx([15, 20])
    assert layers[0].eligible_players == ("short", "B", "C")
    assert layers[1].eligible_players == ("B", "C")


# --- Adversarial round 16: forced-post-only all-ins ---------------------------
#
# The family is `live_contribution == 0 and dead_contribution > 0` with three or
# more seats and at least one live wager. The pot layers used to be derived from
# LIVE contributions alone, so a seat with no live level of its own never had a
# layer capped at what it covered: it was added to the first live layer whole and
# could be declared the winner of every chip in it. Nothing was loud about it --
# chip conservation still held, so `is_balanced`, `is_settled` and `is_legal`
# were all True with no warning and no legality issue, and the hand reconciled as
# authoritative with an 11x hero result.


def test_a_seat_all_in_for_its_ante_cannot_win_the_live_betting_it_never_covered() -> None:
    """The reported hand, exactly: 3-handed, ante 1, C all-in from the ante."""
    players = [_player("A", stack=100, seat=0), _player("B", stack=100, seat=1), _player("C", stack=1, seat=2)]
    actions = [
        _action("A", "ante", 1),
        _action("B", "ante", 1),
        _action("C", "ante", 1),
        _action("A", "bet", 10),
        _action("B", "call", 10),
        _action("A", "check", street="flop"),
        _action("B", "check", street="flop"),
    ]
    layers = build_hand_ledger(players, actions).pots

    # Three chips C contested, twenty it did not.
    assert [pot.amount for pot in layers] == pytest.approx([3, 20])
    assert [pot.cause for pot in layers] == ["main", "side"]
    assert layers[0].eligible_players == ("A", "B", "C")
    assert layers[1].eligible_players == ("A", "B")

    ledger = build_hand_ledger(players, actions, {0: ("C",), 1: ("A",)})
    assert ledger.payouts["C"] == pytest.approx(3)
    assert ledger.net_results == pytest.approx({"A": 9, "B": -11, "C": 2})
    assert ledger.is_balanced is True
    assert ledger.warnings == ()
    assert ledger.legality_issues == ()

    # The award that used to reconcile -- C taking the whole 23 -- is now refused
    # rather than being accepted as the derived truth.
    with pytest.raises(LedgerError):
        build_hand_ledger(players, actions, {0: ("C",), 1: ("C",)})


def test_a_seat_all_in_for_a_dead_blind_is_capped_the_same_way() -> None:
    """The trigger is a forced post with no live level, not the ante specifically."""
    players = [_player("A", stack=100), _player("B", stack=100), _player("C", stack=1)]
    actions = [
        LedgerAction(
            player="C", street="preflop", kind="post_blind", amount=1, is_live_post=False
        ),
        _action("A", "bet", 10),
        _action("B", "call", 10),
    ]
    layers = build_hand_ledger(players, actions).pots

    assert [pot.amount for pot in layers] == pytest.approx([3, 18])
    assert [pot.cause for pot in layers] == ["main", "side"]
    assert "C" not in layers[1].eligible_players

    ledger = build_hand_ledger(players, actions, {0: ("C",), 1: ("A",)})
    assert ledger.net_results == pytest.approx({"A": 8, "B": -10, "C": 2})


def test_a_forced_post_all_in_caps_a_layer_next_to_a_genuine_side_pot() -> None:
    """A real side pot in the same hand does not absorb the forced-post cap."""
    players = [
        _player("A", stack=100, seat=0),
        _player("B", stack=100, seat=1),
        _player("C", stack=1, seat=2),
        _player("D", stack=40, seat=3),
    ]
    actions = [
        _action(name, "ante", 1) for name in ("A", "B", "C", "D")
    ] + [
        _action("D", "all-in", 39),
        _action("A", "call", 39),
        _action("B", "call", 39),
    ]
    layers = build_hand_ledger(players, actions).pots

    assert [pot.amount for pot in layers] == pytest.approx([4, 117])
    assert layers[0].eligible_players == ("A", "B", "C", "D")
    assert layers[1].eligible_players == ("A", "B", "D")

    ledger = build_hand_ledger(players, actions, {0: ("C",), 1: ("D",)})
    assert ledger.payouts["C"] == pytest.approx(4)
    assert ledger.net_results["C"] == pytest.approx(3)
    assert ledger.is_balanced is True


def test_unmatched_dead_money_is_still_won_by_whoever_takes_the_pot() -> None:
    """Capping by total commitment must not hand a lone ante back to its poster.

    A big-blind ante is one seat posting for the table, so its poster's total
    commitment is a chip higher than anyone else's on every hand it is dealt.
    That level caps nobody -- there is no opponent it stopped short -- so it is
    not a layer, and the seat that wins the hand wins the ante with it.
    """
    players = [_player("BB", seat=0), _player("BTN", seat=1)]
    actions = [
        _action("BB", "ante", 1),
        _action("BB", "post_blind", 1),
        _action("BTN", "raise", 9),
        _action("BB", "call", 8),
    ]
    ledger = build_hand_ledger(players, actions, {0: ("BTN",)})

    assert [pot.amount for pot in ledger.pots] == pytest.approx([19])
    assert ledger.pots[0].eligible_players == ("BB", "BTN")
    assert ledger.payouts["BTN"] == pytest.approx(19)
    assert ledger.net_results == pytest.approx({"BB": -10, "BTN": 10})


def test_a_seat_whose_only_post_came_back_still_contests_the_antes() -> None:
    """An uncalled live post is returned; being in the hand is not.

    Heads-up, one seat is all-in for its ante and the other posts a blind no
    remaining chip can call. The blind comes back — an uncalled bet is measured
    against live money — but the ante is still a pot, and the seat that played
    for it can still be declared the winner of it.
    """
    players = [_player("alice", stack=2, seat=0), _player("bob", stack=1, seat=1)]
    actions = [_action("alice", "ante", 2), _action("bob", "post_blind", 1)]

    ledger = build_hand_ledger(players, actions, {0: ("bob",)})
    assert ledger.refunds["bob"] == pytest.approx(1)
    assert [pot.amount for pot in ledger.pots] == pytest.approx([2])
    assert ledger.pots[0].eligible_players == ("alice", "bob")
    assert ledger.net_results == pytest.approx({"alice": -2, "bob": 2})
    assert ledger.is_balanced is True


def test_a_pot_made_only_of_dead_money_is_the_main_pot() -> None:
    """No live chip was wagered, so there is one pot and everyone contests it."""
    players = [_player("A"), _player("B")]
    actions = [_action("A", "ante", 1), _action("B", "ante", 1)]
    layers = build_hand_ledger(players, actions, dead_money=2).pots

    assert [pot.cause for pot in layers] == ["main"]
    assert layers[0].label == "Main pot"
    assert layers[0].amount == pytest.approx(4)


def test_antes_do_not_coarsen_an_evenly_chopped_pot() -> None:
    """The split quantum is the finest denomination the hand demonstrates.

    Summing four 0.25 antes into a single 1.00 destroyed the hundredths those
    antes prove the table deals in, so an exactly-even chop derived an odd
    split — and which seat gained the extra chip was decided by award order.
    """
    players = [_player(name) for name in ("A", "B", "C", "D")]
    actions = [_action(name, "ante", 0.25) for name in ("A", "B", "C", "D")]
    actions += [
        _action("A", "bet", 10),
        _action("B", "call", 10),
        _action("C", "fold"),
        _action("D", "fold"),
    ]
    layers = build_hand_ledger(players, actions).pots
    ledger = build_hand_ledger(
        players,
        actions,
        {pot.index: ("A", "B") for pot in layers},
        odd_chip_order=("A", "B"),
    )
    assert ledger.gross_pot == pytest.approx(21.0)
    # Exactly even, and the same whichever way the odd-chip order runs.
    assert ledger.payouts["A"] == pytest.approx(10.5)
    assert ledger.payouts["B"] == pytest.approx(10.5)

    reversed_order = build_hand_ledger(
        players,
        actions,
        {pot.index: ("A", "B") for pot in layers},
        odd_chip_order=("B", "A"),
    )
    assert reversed_order.payouts == pytest.approx(ledger.payouts)


# --- Adversarial round 17: what an uncalled refund may and may not take away ---
#
# Two contribution figures decide two different questions, and collapsing them
# into one breaks whichever question loses. What a seat has left IN the pot --
# live money that stuck, plus every dead chip -- is what SIZES a layer and what
# caps the eligibility of the seats an all-in stopped short. What a seat PUT UP
# before any uncalled money came back is what says it played the hand at all.
# The round-16 repair reported measuring eligibility against the settled figure
# as a defect in its own fix; these pin both halves of the corrected rule so the
# next repair to this module cannot quietly collapse them again.


def test_a_refunded_seat_still_contests_a_layer_above_what_stayed_in() -> None:
    """An uncalled bet coming back does not un-play the hand it was bet into.

    Four seats. ``S`` is all-in for nothing but its ante and is capped at the
    layer holding the antes -- that much is the round-16 rule. ``R`` shoves 100,
    two seats are all-in for 30 behind dead blinds of 5, and 70 of R's shove is
    returned as uncalled, leaving 30 of R's chips in a pot whose top layer is cut
    at 35.

    R matched every live chip its opponents wagered; the 5 they are ahead by is
    dead money owed to the table, not a wager R declined. Measuring eligibility
    against what STAYED IN would cap R below that cut and refuse the only
    truthful declaration on a hand R won, which is the unrecordable-hand failure
    that started this whole chain. Measuring against what R put up keeps R in.
    """
    players = [
        _player("S", stack=1, seat=0),
        _player("R", stack=200, seat=1),
        _player("Z", stack=35, seat=2),
        _player("W", stack=35, seat=3),
    ]
    actions = [
        _action("S", "ante", 1),
        LedgerAction(
            player="Z", street="preflop", kind="post_blind", amount=5, is_live_post=False
        ),
        LedgerAction(
            player="W", street="preflop", kind="post_blind", amount=5, is_live_post=False
        ),
        _action("R", "bet", 100),
        _action("Z", "all-in", 30),
        _action("W", "all-in", 30),
    ]
    layers = build_hand_ledger(players, actions).pots

    assert [pot.amount for pot in layers] == pytest.approx([4, 97])
    assert layers[0].eligible_players == ("S", "R", "Z", "W")
    # R has 30 in the pot and the layer is cut at 35, and R is still eligible.
    assert layers[1].eligible_players == ("R", "Z", "W")

    ledger = build_hand_ledger(players, actions, {0: ("S",), 1: ("R",)})
    assert ledger.refunds["R"] == pytest.approx(70)
    assert ledger.payouts["R"] == pytest.approx(97)
    assert ledger.net_results == pytest.approx({"S": 3, "R": 67, "Z": -35, "W": -35})
    assert ledger.is_balanced is True
    assert ledger.legality_issues == ()


def test_a_seat_capped_by_an_all_in_is_still_refused_the_layer_above() -> None:
    """The same hand from the other side: gross commitment is not a free pass.

    ``S`` put up one chip and is out of the layer holding ninety-seven, which is
    the guarantee the round-16 repair exists to provide. Stated next to the test
    above so a future change cannot satisfy one by discarding the other.
    """
    players = [
        _player("S", stack=1, seat=0),
        _player("R", stack=200, seat=1),
        _player("Z", stack=35, seat=2),
        _player("W", stack=35, seat=3),
    ]
    actions = [
        _action("S", "ante", 1),
        LedgerAction(
            player="Z", street="preflop", kind="post_blind", amount=5, is_live_post=False
        ),
        LedgerAction(
            player="W", street="preflop", kind="post_blind", amount=5, is_live_post=False
        ),
        _action("R", "bet", 100),
        _action("Z", "all-in", 30),
        _action("W", "all-in", 30),
    ]

    with pytest.raises(LedgerError):
        build_hand_ledger(players, actions, {0: ("S",), 1: ("S",)})
