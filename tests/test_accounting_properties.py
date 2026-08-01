"""Property and fuzz tests for chip conservation (PLAN Phase 7 and Phase 14).

Chips do not appear or vanish. Every hand, whatever its shape, must satisfy:

    sum(payouts) + rake == gross_pot
    sum(net_results) + rake == 0
    every player's net result >= -their starting stack

The example-based ledger tests cover structures somebody thought of. These
generate structures nobody did — random stacks, random bet sizes, random all-in
layering, random rake policies — and assert the invariants that must hold across
all of them. A conservation bug that only shows up at a three-way side pot with
an odd chip and a capped rake is exactly what this is for.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from poker_tracker.math.accounting import (
    LedgerAction,
    LedgerError,
    LedgerPlayer,
    RakePolicy,
    build_hand_ledger,
)

# Whole chips keep the generated hands legal without fighting float noise; the
# ledger works in Decimal internally, and fractional stacks are covered by the
# example-based suite.
STACKS = st.integers(min_value=2, max_value=500)
NAMES = ("alice", "bob", "carol", "dana")

SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)


@st.composite
def all_in_hand(draw):
    """A preflop all-in shove-fest: everyone commits their whole stack.

    This is the structure that produces layered side pots, which is where
    conservation is hardest and where a rounding slip hides best.
    """
    count = draw(st.integers(min_value=2, max_value=4))
    names = list(NAMES[:count])
    stacks = [draw(STACKS) for _ in names]
    players = [
        LedgerPlayer(name=name, starting_stack=float(stack), seat=index)
        for index, (name, stack) in enumerate(zip(names, stacks, strict=True))
    ]
    actions = [
        LedgerAction(player=name, street="preflop", kind="all-in", amount=float(stack))
        for name, stack in zip(names, stacks, strict=True)
    ]
    return players, actions, names, stacks


@st.composite
def rake_policy(draw):
    rate = draw(st.sampled_from([0.0, 0.02, 0.05, 0.10]))
    cap = draw(st.one_of(st.none(), st.sampled_from([1.0, 5.0, 20.0])))
    return RakePolicy(rate=rate, cap=cap, rounding_unit=0.01)


def _winner_map(pots, preferred=()):
    """Award each pot to an eligible player, preferring ``preferred``.

    A short stack is genuinely not eligible for the side pots it never covered,
    so a winner map must be built per layer. Naming one player across every pot
    is not a harder test, it is an invalid hand.
    """
    awards = {}
    for pot in pots:
        eligible = tuple(pot.eligible_players)
        if not eligible:
            continue
        chosen = tuple(name for name in preferred if name in eligible)
        awards[pot.index] = chosen or (eligible[0],)
    return awards


def _sum(values) -> Decimal:
    return sum((Decimal(str(v)) for v in values), Decimal("0"))


# --- Conservation -----------------------------------------------------------


@given(hand=all_in_hand(), policy=rake_policy())
@SETTINGS
def test_payouts_plus_rake_equal_the_gross_pot(hand, policy):
    players, actions, names, _stacks = hand
    unsettled = build_hand_ledger(players, actions, rake=policy)
    ledger = build_hand_ledger(
        players,
        actions,
        _winner_map(unsettled.pots, (names[0],)),
        rake=policy,
    )
    assume(ledger.is_settled)
    assert _sum(ledger.payouts.values()) + Decimal(str(ledger.rake)) == Decimal(
        str(ledger.gross_pot)
    )
    assert ledger.is_balanced


@given(hand=all_in_hand(), policy=rake_policy())
@SETTINGS
def test_net_results_and_rake_sum_to_zero(hand, policy):
    """Chips leaving the players equal chips reaching the pot and the house."""
    players, actions, names, _stacks = hand
    unsettled = build_hand_ledger(players, actions, rake=policy)
    ledger = build_hand_ledger(
        players, actions, _winner_map(unsettled.pots, (names[0],)), rake=policy
    )
    assume(ledger.is_settled)
    assert _sum(ledger.net_results.values()) + Decimal(str(ledger.rake)) == Decimal("0")


@given(hand=all_in_hand())
@SETTINGS
def test_nobody_loses_more_than_they_brought(hand):
    players, actions, names, stacks = hand
    unsettled = build_hand_ledger(players, actions)
    ledger = build_hand_ledger(players, actions, _winner_map(unsettled.pots, (names[0],)))
    assume(ledger.is_settled)
    for player, stack in zip(names, stacks, strict=True):
        assert ledger.net_results[player] >= -float(stack) - 1e-9


@given(hand=all_in_hand())
@SETTINGS
def test_contributions_never_exceed_starting_stacks(hand):
    players, actions, names, stacks = hand
    ledger = build_hand_ledger(players, actions)
    for player, stack in zip(names, stacks, strict=True):
        assert ledger.contributions[player] <= float(stack) + 1e-9


@given(hand=all_in_hand())
@SETTINGS
def test_the_short_stack_can_only_win_what_it_covered(hand):
    """A side-pot invariant: winning from the bottom layer is capped."""
    players, actions, names, stacks = hand
    shortest = min(range(len(stacks)), key=lambda i: stacks[i])
    short_name = names[shortest]
    unsettled = build_hand_ledger(players, actions)
    ledger = build_hand_ledger(
        players, actions, _winner_map(unsettled.pots, (short_name,))
    )
    assume(ledger.is_settled)
    # The short stack can never take more than every player matching its own
    # commitment, less whatever rake came off.
    covered = sum(min(stack, stacks[shortest]) for stack in stacks)
    assert ledger.payouts[short_name] <= float(covered) + 1e-9


# --- Splits -----------------------------------------------------------------


@given(hand=all_in_hand())
@SETTINGS
def test_a_chopped_pot_still_conserves_chips(hand):
    players, actions, names, _stacks = hand
    unsettled = build_hand_ledger(players, actions)
    assume(len(unsettled.pots) >= 1)
    everyone = tuple(names)
    ledger = build_hand_ledger(
        players,
        actions,
        _winner_map(unsettled.pots, everyone),
        odd_chip_order=everyone,
    )
    assume(ledger.is_settled)
    assert _sum(ledger.payouts.values()) + Decimal(str(ledger.rake)) == Decimal(
        str(ledger.gross_pot)
    )


# --- Rake bounds ------------------------------------------------------------


@given(hand=all_in_hand(), policy=rake_policy())
@SETTINGS
def test_rake_never_exceeds_its_cap_or_the_pot(hand, policy):
    players, actions, names, _stacks = hand
    unsettled = build_hand_ledger(players, actions, rake=policy)
    ledger = build_hand_ledger(
        players, actions, _winner_map(unsettled.pots, (names[0],)), rake=policy
    )
    assert ledger.rake >= 0
    assert ledger.rake <= ledger.gross_pot + 1e-9
    if policy.cap is not None:
        assert ledger.rake <= policy.cap + 1e-9


@given(hand=all_in_hand())
@SETTINGS
def test_zero_rake_leaves_the_pot_whole(hand):
    players, actions, names, _stacks = hand
    unsettled = build_hand_ledger(players, actions, rake=RakePolicy(rate=0.0))
    ledger = build_hand_ledger(
        players,
        actions,
        _winner_map(unsettled.pots, (names[0],)),
        rake=RakePolicy(rate=0.0),
    )
    assert ledger.rake == 0
    assert ledger.net_pot == ledger.gross_pot


# --- Fuzz: malformed input must raise, never miscompute ----------------------


@given(
    amount=st.one_of(
        st.floats(min_value=-1e6, max_value=-0.01),
        st.just(float("nan")),
        st.just(float("inf")),
    )
)
@SETTINGS
def test_impossible_amounts_raise_rather_than_settle(amount):
    """A hand that cannot be true must fail loudly, not produce a number."""
    players = [
        LedgerPlayer(name="alice", starting_stack=100.0, seat=0),
        LedgerPlayer(name="bob", starting_stack=100.0, seat=1),
    ]
    actions = [LedgerAction(player="alice", street="preflop", kind="bet", amount=amount)]
    with pytest.raises((LedgerError, ValueError, ArithmeticError)):
        build_hand_ledger(players, actions)


@given(stack=st.floats(min_value=-1e6, max_value=-0.01))
@SETTINGS
def test_negative_starting_stacks_raise(stack):
    players = [LedgerPlayer(name="alice", starting_stack=stack, seat=0)]
    with pytest.raises(LedgerError):
        build_hand_ledger(players, [])


@given(
    names=st.lists(
        st.sampled_from(NAMES), min_size=2, max_size=4, unique=False
    )
)
@SETTINGS
def test_duplicate_player_names_are_rejected(names):
    assume(len(set(names)) < len(names))
    players = [
        LedgerPlayer(name=name, starting_stack=100.0, seat=index)
        for index, name in enumerate(names)
    ]
    with pytest.raises(LedgerError):
        build_hand_ledger(players, [])


@given(hand=all_in_hand())
@SETTINGS
def test_an_unknown_winner_is_rejected_rather_than_paid(hand):
    players, actions, _names, _stacks = hand
    unsettled = build_hand_ledger(players, actions)
    assume(unsettled.pots)
    with pytest.raises(LedgerError):
        build_hand_ledger(
            players, actions, {index: ("nobody-at-this-table",) for index in range(len(unsettled.pots))}
        )


@given(hand=all_in_hand())
@SETTINGS
def test_an_unsettled_ledger_is_never_balanced(hand):
    """Balance must mean "checked and correct", never "not yet checked"."""
    players, actions, _names, _stacks = hand
    ledger = build_hand_ledger(players, actions)
    if not ledger.is_settled:
        assert ledger.is_balanced is False
        assert ledger.warnings
