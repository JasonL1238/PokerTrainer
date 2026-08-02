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

The suite also generates forced posts, because that is where the accounting bugs
this project actually shipped lived: unmatched antes and dead blinds were being
refunded as uncalled bets and leaving the pot, and a player all-in for nothing
but an ante was eligible for no pot at all while their chips sat in one. A
property suite that cannot generate the input family a known bug lived in proves
nothing about that family, so ``forced_post_hand`` builds antes (every seat and
button-only), small and big blinds, dead blinds, straddles, and seats whose whole
stack is smaller than the post they owe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# Whole chips keep the shove-fest hands legal without fighting float noise; the
# ledger works in Decimal internally. ``forced_post_hand`` below draws its own
# denomination and does generate fractional amounts.
STACKS = st.integers(min_value=2, max_value=500)
NAMES = ("alice", "bob", "carol", "dana")
_ZERO = Decimal("0")

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


@dataclass(frozen=True)
class ForcedPostHand:
    """A generated hand that starts with forced posts, plus the facts to check it.

    ``dead`` and ``live`` split each seat's commitment the way a room does: antes
    and dead blinds are owed to the table, everything else buys a place in a pot
    layer. The ledger's public result does not expose that split, and it is
    exactly what an uncalled-bet refund is allowed to touch, so the generator
    records it while it builds the action line.

    ``posted`` and ``wagered`` split the same chips by choice instead: a seat
    whose entire commitment is ``posted`` never voluntarily put a chip in, and if
    that commitment used its whole stack it is the seat that must still be able
    to win the layer holding its chips.
    """

    players: list[LedgerPlayer]
    actions: list[LedgerAction]
    names: list[str]
    stacks: list[Decimal]
    dead: dict[str, Decimal]
    live: dict[str, Decimal]
    posted: dict[str, Decimal]
    wagered: dict[str, Decimal]
    folded: set[str] = field(default_factory=set)

    @property
    def all_in_on_a_forced_post(self) -> tuple[str, ...]:
        """Seats still in the hand whose whole stack went in as a forced post."""

        return tuple(
            name
            for index, name in enumerate(self.names)
            if name not in self.folded
            and self.wagered[name] == 0
            and self.posted[name] == self.stacks[index]
        )


# Blind pairs a real table posts, in units of the chip below, including the
# equal-blind structure some tournaments deal so the generator does not only ever
# produce sb < bb.
BLINDS = st.sampled_from([(1, 2), (1, 3), (2, 2), (2, 4), (5, 10)])
ANTES = st.sampled_from([0, 1, 2, 5])
# "button" is the button-ante structure, where one seat posts for the table.
ANTE_STYLES = st.sampled_from(["none", "all", "button"])
# The denomination every amount in a generated hand is a whole multiple of. A
# 0.5/1 game with 0.25 antes is ordinary, and the split granularity a chopped pot
# is divided at is read off the hand's own amounts including the dead ones, so a
# suite that only ever generates whole chips never exercises that reading.
CHIPS = st.sampled_from([Decimal("1"), Decimal("0.5"), Decimal("0.25"), Decimal("0.05")])


@st.composite
def forced_post_hand(draw):
    """A preflop hand whose money starts with antes, blinds, dead posts and straddles.

    Stacks are drawn small enough to sit below the posts they owe, because a seat
    all-in for part of its ante or blind is ordinary once antes grow and is the
    shape both historic accounting bugs lived in.

    A seat only folds while facing a bet. Folding for free is legal at a real
    client but it is the one line that can strand a layer with no eligible
    player, and that is a coverage limitation of the reducer rather than a
    conservation question, so the generator stays out of it.

    Every amount is a whole multiple of one drawn chip, built in ``Decimal`` and
    converted to float only at the ledger boundary, so the invariants compare
    exact numbers rather than float noise.
    """

    count = draw(st.integers(min_value=2, max_value=4))
    names = list(NAMES[:count])
    chip = draw(CHIPS)
    stacks = [chip * draw(st.integers(min_value=1, max_value=120)) for _ in names]
    small_blind, big_blind = (chip * unit for unit in draw(BLINDS))
    ante = chip * draw(ANTES)
    ante_style = draw(ANTE_STYLES)
    wants_straddle = draw(st.booleans())
    wants_dead_blind = draw(st.booleans())
    choices = draw(
        st.lists(
            st.sampled_from(["call", "fold", "shove"]),
            min_size=count,
            max_size=count,
        )
    )

    players = [
        LedgerPlayer(name=name, starting_stack=float(stack), seat=index)
        for index, (name, stack) in enumerate(zip(names, stacks, strict=True))
    ]
    actions: list[LedgerAction] = []
    remaining = dict(zip(names, stacks, strict=True))
    dead = {name: _ZERO for name in names}
    live = {name: _ZERO for name in names}
    posted = {name: _ZERO for name in names}
    wagered = {name: _ZERO for name in names}
    street_live = {name: _ZERO for name in names}
    folded: set[str] = set()

    def commit(name, amount, kind, *, is_live_post=True, forced):
        """Put in as much of ``amount`` as the seat still has, or nothing."""

        capped = min(amount, remaining[name])
        if capped <= 0:
            return
        actions.append(
            LedgerAction(
                player=name,
                street="preflop",
                kind=kind,
                amount=float(capped),
                is_live_post=is_live_post,
            )
        )
        remaining[name] -= capped
        if kind != "ante" and is_live_post:
            live[name] += capped
            street_live[name] += capped
        else:
            dead[name] += capped
        if forced:
            posted[name] += capped
        else:
            wagered[name] += capped

    if ante_style == "all":
        for name in names:
            commit(name, ante, "ante", forced=True)
    elif ante_style == "button":
        commit(names[0], ante, "ante", forced=True)
    if wants_dead_blind and count >= 3:
        # A returning player owes a dead blind before the live one is posted.
        commit(names[1], big_blind, "post_blind", is_live_post=False, forced=True)
    commit(names[-2], small_blind, "post_blind", forced=True)
    commit(names[-1], big_blind, "post_blind", forced=True)
    if wants_straddle:
        commit(names[0], big_blind * 2, "post_blind", forced=True)

    for name, choice in zip(names, choices, strict=True):
        if name in folded or remaining[name] <= 0:
            continue
        to_call = max(street_live.values()) - street_live[name]
        still_in = [other for other in names if other not in folded]
        if choice == "fold" and to_call > 0 and len(still_in) > 1:
            actions.append(LedgerAction(player=name, street="preflop", kind="fold"))
            folded.add(name)
            continue
        if choice == "shove":
            commit(name, remaining[name], "all-in", forced=False)
            continue
        if to_call <= 0:
            actions.append(LedgerAction(player=name, street="preflop", kind="check"))
            continue
        if to_call >= remaining[name]:
            commit(name, remaining[name], "all-in", forced=False)
        else:
            commit(name, to_call, "call", forced=False)

    return ForcedPostHand(
        players=players,
        actions=actions,
        names=names,
        stacks=stacks,
        dead=dead,
        live=live,
        posted=posted,
        wagered=wagered,
        folded=folded,
    )


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


# --- Forced posts -----------------------------------------------------------


@given(hand=forced_post_hand())
@SETTINGS
def test_every_forced_post_chip_is_in_a_pot_or_refunded(hand):
    """The central invariant, stated over hands that begin with forced posts.

    Every chip a seat committed is either in a pot layer or handed back. Never
    both, never neither.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    contributed = _sum(ledger.contributions.values())
    refunded = _sum(ledger.refunds.values())
    assert contributed - refunded == Decimal(str(ledger.gross_pot))
    assert _sum(pot.amount for pot in ledger.pots) == Decimal(str(ledger.gross_pot))
    for name, stack in zip(hand.names, hand.stacks, strict=True):
        assert Decimal(str(ledger.refunds[name])) <= Decimal(str(ledger.contributions[name]))
        assert Decimal(str(ledger.contributions[name])) <= stack


@given(hand=forced_post_hand())
@SETTINGS
def test_a_forced_post_is_never_returned_as_an_uncalled_bet(hand):
    """Dead money nobody matched stays in the pot.

    An unmatched ante or dead blind looks exactly like an unmatched overbet if
    refunds are measured against a seat's total commitment, and it used to be
    handed back — money that left the table entirely. A refund can therefore
    never exceed the live chips the seat wagered, and the pot must always hold
    every dead chip posted into it.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    for name in hand.names:
        assert Decimal(str(ledger.refunds[name])) <= hand.live[name]
    assert Decimal(str(ledger.gross_pot)) >= _sum(hand.dead.values())


@given(hand=forced_post_hand())
@SETTINGS
def test_a_seat_all_in_for_only_a_forced_post_can_still_be_declared_the_winner(hand):
    """Eligibility follows the chips, including the dead ones.

    A seat whose whole stack went in as an ante has chips in the main pot, so it
    must be eligible for that pot and the hand it won must be recordable. Deriving
    eligibility from live contributions alone made such a hand unrecordable, which
    is routine once a stack is at or below the ante.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    for name in hand.all_in_on_a_forced_post:
        if Decimal(str(ledger.contributions[name])) <= 0:
            continue
        eligible_layers = [pot for pot in ledger.pots if name in pot.eligible_players]
        assert eligible_layers, f"{name} has chips in the pot but can win none of it"
        settled = build_hand_ledger(
            hand.players, hand.actions, _winner_map(ledger.pots, (name,))
        )
        assert settled.payouts[name] > 0
        assert _sum(settled.net_results.values()) == Decimal("0")


@given(hand=forced_post_hand())
@SETTINGS
def test_every_seat_that_put_a_chip_up_contests_the_main_pot(hand):
    """Playing the hand is measured by what a seat PUT UP, not by what stayed in.

    An uncalled wager is returned to the seat that made it, and a seat whose only
    live post came back -- because the money facing it was a forced post nobody
    had a chip left to call -- ends the hand with nothing of its own in the pot.
    It still played the hand, and the pot it played for is still there, so it must
    still be able to be declared the winner of it. Capping main-pot eligibility at
    the SETTLED figure instead makes that hand unrecordable, which is the failure
    every widening in this module has been a reaction to.

    The main pot only. Layers above it are capped at what the table matched of a
    seat's own commitment, and
    ``test_no_seat_is_paid_more_than_the_table_matched_of_its_own_commitment``
    is the guarantee on that side.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    assume(ledger.pots)
    for name in hand.names:
        if name in hand.folded or Decimal(str(ledger.contributions[name])) <= 0:
            continue
        assert name in ledger.pots[0].eligible_players, (
            f"{name} put chips up and never folded, but cannot win the main pot"
        )


@given(hand=forced_post_hand(), policy=rake_policy())
@SETTINGS
def test_a_settled_forced_post_hand_conserves_chips(hand, policy):
    unsettled = build_hand_ledger(hand.players, hand.actions, rake=policy)
    ledger = build_hand_ledger(
        hand.players,
        hand.actions,
        _winner_map(unsettled.pots, (hand.names[0],)),
        rake=policy,
    )
    assume(ledger.is_settled)
    assert _sum(ledger.payouts.values()) + Decimal(str(ledger.rake)) == Decimal(
        str(ledger.gross_pot)
    )
    assert _sum(ledger.net_results.values()) + Decimal(str(ledger.rake)) == Decimal("0")
    assert ledger.is_balanced
    for name, stack in zip(hand.names, hand.stacks, strict=True):
        assert ledger.net_results[name] >= -float(stack) - 1e-9
    for pot in ledger.pots:
        # A layer can never be raked past its own size.
        assert pot.net_amount >= -1e-9
        assert pot.rake <= pot.amount + 1e-9


@given(hand=forced_post_hand(), policy=rake_policy())
@SETTINGS
def test_a_chopped_forced_post_pot_conserves_every_chip(hand, policy):
    """Chop every layer, then check the chips.

    The split granularity is read off the hand's own amounts, dead ones included,
    so a chopped pot built from fractional antes is where a coarsened quantum
    shows up: it cannot move the gross, the rake or conservation, only who takes
    an odd chip, and that is what this pins.
    """
    everyone = tuple(hand.names)
    unsettled = build_hand_ledger(hand.players, hand.actions, rake=policy)
    ledger = build_hand_ledger(
        hand.players,
        hand.actions,
        _winner_map(unsettled.pots, everyone),
        rake=policy,
        odd_chip_order=everyone,
    )
    assume(ledger.is_settled)
    assert _sum(ledger.payouts.values()) + Decimal(str(ledger.rake)) == Decimal(
        str(ledger.gross_pot)
    )
    assert _sum(ledger.net_results.values()) + Decimal(str(ledger.rake)) == Decimal("0")
    assert ledger.is_balanced
    # Reversing the audited odd-chip order can move at most the odd chips, never
    # the pot: the same hand must still pay out the same total.
    reversed_order = build_hand_ledger(
        hand.players,
        hand.actions,
        _winner_map(unsettled.pots, everyone),
        rake=policy,
        odd_chip_order=tuple(reversed(everyone)),
    )
    assert _sum(reversed_order.payouts.values()) == _sum(ledger.payouts.values())


@given(hand=forced_post_hand())
@SETTINGS
def test_a_forced_post_layer_is_only_emitted_when_it_caps_somebody_out(hand):
    """A side pot is a capped eligibility, not an index above zero.

    Forced posts are the cheapest way to make extra layers — a blind that folds
    for less used to split the pot without capping anybody — so this is the
    family where labelling every layer after the first a "side pot" states
    something false. Naming those layers "dead money" instead was the same
    mistake pointing the other way, since their chips are ordinary live
    wagering. A boundary that caps nobody is not a pot boundary at all, so the
    stronger property is that no such layer is emitted: every layer after the
    main pot must name at least one seat still in the hand that cannot win it.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    previous: tuple[str, ...] | None = None
    for pot in ledger.pots:
        if previous is None:
            assert pot.cause == "main"
            assert pot.label == "Main pot"
        else:
            capped_out = set(previous) - set(pot.eligible_players)
            assert capped_out, f"layer {pot.index} was split off without capping anybody"
            assert pot.cause == "side"
            assert pot.label == "Side pot"
        previous = pot.eligible_players


@given(hand=forced_post_hand())
@SETTINGS
def test_no_seat_is_paid_more_than_the_table_matched_of_its_own_commitment(hand):
    """The cap a side pot exists to enforce, stated as an invariant.

    A seat can win, from each opponent still in the hand, only as much as that
    opponent matched of what the seat itself put up — plus whatever folded seats
    abandoned and whatever dead money the others owed the table. Deriving the pot
    layers from LIVE contributions alone broke this for the whole
    ``live == 0 and dead > 0`` family: a seat all-in for nothing but a forced post
    had no live level, so no layer was ever capped at its commitment and it
    contested every chip of the first live layer. Three ante chips took twenty-three.

    Nothing about that was loud — chip conservation held, so the ledger reported
    balanced, settled and legal with no warning — which is why the guarantee has
    to be asserted directly rather than inferred from the balance verdicts.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    put_up = {name: Decimal(str(ledger.contributions[name])) for name in hand.names}
    in_pot = {
        name: put_up[name] - Decimal(str(ledger.refunds[name])) for name in hand.names
    }
    others = {
        name: [other for other in hand.names if other != name] for name in hand.names
    }
    for name in hand.names:
        settled = build_hand_ledger(
            hand.players, hand.actions, _winner_map(ledger.pots, (name,))
        )
        if not settled.is_settled:
            continue
        matched = in_pot[name] + _sum(
            min(in_pot[other], put_up[name])
            for other in others[name]
            if other not in hand.folded
        )
        # A seat that folded left everything it had in the pot behind, and dead
        # money is owed to the table rather than wagered at anybody, so neither
        # is bounded by what this seat covered.
        abandoned = _sum(in_pot[other] for other in others[name] if other in hand.folded)
        owed_to_the_table = _sum(
            hand.dead[other] for other in others[name] if other not in hand.folded
        )
        assert Decimal(str(settled.payouts[name])) <= (
            matched + abandoned + owed_to_the_table
        ), f"{name} was paid more than the table matched of its own commitment"


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
