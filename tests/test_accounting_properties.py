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
    BlindStructure,
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


# The forced-bet names that set what the rest of the table owes. Antes,
# big-blind antes and dead blinds are owed to the table and name no wager level.
# Written here from the specification sentence "forced posts are not live", so
# this suite identifies a forced post without importing the reducer's opinion.
_LIVE_STRUCTURAL_TYPES = frozenset({"small_blind", "big_blind", "straddle", "bring_in"})


def _is_forced_row(action: LedgerAction) -> bool:
    return action.kind in {"ante", "post_blind"} or action.forced_bet_type is not None


def _is_live_structural_row(action: LedgerAction) -> bool:
    if action.kind == "ante":
        return False
    if action.forced_bet_type is not None:
        return action.forced_bet_type in _LIVE_STRUCTURAL_TYPES and bool(
            action.is_live_post
        )
    return action.kind == "post_blind" and bool(action.is_live_post)


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
    # Dead money declared for the table with no contributing seat: a rake-back
    # promotion chip, a bad-beat drop returned to the pot. It is owed to the
    # table exactly as an ante is, so the model puts it in the lowest layer and
    # lets it confer eligibility on nobody. Carried on the generated hand so the
    # payout-cap properties sweep it rather than only ever seeing zero.
    external_dead: Decimal = _ZERO
    # The structural forced-bet sizes the generator DEALT, as opposed to the ones
    # the resulting action line happens to show. They differ exactly when a seat
    # was too short to post in full, which is the family the blind structure
    # exists for, so the generator has to record what it meant.
    blinds: BlindStructure | None = None

    @property
    def short_live_post(self) -> bool:
        """Did forced posting take a live poster's last chip?

        The condition under which the action line cannot demonstrate the
        structure it was dealt, and therefore the condition under which the
        reducer must refuse rather than infer.

        Asked of the seat's FINAL forced commitment, not of its running total at
        the instant its blind row is reduced. This oracle used to ask the second
        question, which is the same mistake the reducer was making: a seat whose
        ante is recorded BELOW its blind is not yet out of chips at the blind,
        so both the reducer and its oracle agreed the post was full, and the
        property could not see the defect it exists to catch. An oracle that
        reimplements the code under test proves only that the code is
        self-consistent.

        Asked of what the row IS, not of the kind that carries it. A forced post
        which took its poster's last chip is routinely booked as ``all-in`` with
        the forced-bet type recorded alongside; keying on ``post_blind`` alone
        made this oracle blind to exactly the row the refusal exists for.
        """
        stacks = dict(zip(self.names, self.stacks, strict=True))
        committed = {name: Decimal("0") for name in self.names}
        voluntary: set[str] = set()
        candidates: list[tuple[int, str]] = []
        exhausted_at_post: set[int] = set()
        for index, action in enumerate(self.actions):
            if action.kind not in {
                "ante",
                "post_blind",
                "bet",
                "call",
                "raise",
                "all-in",
            }:
                continue
            committed[action.player] += Decimal(str(action.amount))
            if not _is_forced_row(action):
                voluntary.add(action.player)
            if _is_live_structural_row(action):
                candidates.append((index, action.player))
                if committed[action.player] == stacks[action.player]:
                    exhausted_at_post.add(index)
        # A seat that later CHOSE to put chips in was never short of the blind it
        # posted, so only a stack spent entirely on forced posts counts.
        exhausted = {
            name
            for name, total in committed.items()
            if name not in voluntary and total > 0 and total == stacks[name]
        }
        return any(
            index in exhausted_at_post or player in exhausted
            for index, player in candidates
        )

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
    antes_after_blinds = draw(st.booleans())
    wants_straddle = draw(st.booleans())
    wants_dead_blind = draw(st.booleans())
    external_dead = chip * draw(st.sampled_from([0, 0, 0, 1, 4]))
    # Whether this recording books a forced post that took its poster's last
    # chip as ``all-in`` carrying its forced-bet type, or as the plain kind.
    # Both spell the same event and must derive byte-identical chips.
    relabel_exhausting_posts = draw(st.booleans())
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

    def commit(name, amount, kind, *, is_live_post=True, forced, forced_type=None):
        """Put in as much of ``amount`` as the seat still has, or nothing.

        ``forced_type`` is what the room CALLS this post. When it is given and
        the post takes the seat's last chip, the row may be booked as ``all-in``
        carrying that name -- the shape a real recording produces when a forced
        post exhausts its poster, and the shape the CV spine and the hand editor
        both write into ``actions.forced_bet_type`` / ``actions.is_live_post``.

        The generator classified its OWN live/dead bookkeeping from ``kind``, and
        emitted no ``forced_bet_type`` at all, so it was structurally incapable of
        producing that row -- which is how a money classifier that also keyed on
        ``kind`` alone survived every property below while counting a dead ante as
        chosen live money. Liveness is decided here from the spec sentence
        instead: a forced post is live only when it is one of the STRUCTURAL bets
        that set the wager level, whatever kind carries it.
        """

        capped = min(amount, remaining[name])
        if capped <= 0:
            return
        exhausts = capped == remaining[name]
        booked = kind
        carried = forced_type
        if forced_type is not None and exhausts and relabel_exhausting_posts:
            booked = "all-in"
        elif forced_type is not None and not relabel_exhausting_posts:
            carried = None
        actions.append(
            LedgerAction(
                player=name,
                street="preflop",
                kind=booked,
                amount=float(capped),
                is_live_post=is_live_post,
                forced_bet_type=carried,
            )
        )
        remaining[name] -= capped
        is_live_money = not forced or (
            forced_type in {"small_blind", "big_blind", "straddle"} and is_live_post
        )
        if is_live_money:
            live[name] += capped
            street_live[name] += capped
        else:
            dead[name] += capped
        if forced:
            posted[name] += capped
        else:
            wagered[name] += capped

    def post_antes():
        if ante_style == "all":
            for name in names:
                commit(name, ante, "ante", forced=True, forced_type="ante")
        elif ante_style == "button":
            commit(names[0], ante, "ante", forced=True, forced_type="ante")

    # The order two FORCED rows are listed in is not a fact about the hand: a
    # room takes both in one motion, a reconstruction resolves them from the same
    # chip-movement burst, and the hand editor lets an operator renumber them.
    # The generator used to emit antes first, always, which made it structurally
    # incapable of producing a seat whose ante follows its blind -- the shape in
    # which the reducer's short-post refusal silently stopped firing.
    if not antes_after_blinds:
        post_antes()
    if wants_dead_blind and count >= 3:
        # A returning player owes a dead blind before the live one is posted.
        commit(
            names[1],
            big_blind,
            "post_blind",
            is_live_post=False,
            forced=True,
            forced_type="dead_blind",
        )
    commit(names[-2], small_blind, "post_blind", forced=True, forced_type="small_blind")
    commit(names[-1], big_blind, "post_blind", forced=True, forced_type="big_blind")
    if wants_straddle:
        commit(
            names[0], big_blind * 2, "post_blind", forced=True, forced_type="straddle"
        )
    if antes_after_blinds:
        post_antes()

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
        external_dead=external_dead,
        blinds=BlindStructure(
            small_blind=float(small_blind),
            big_blind=float(big_blind),
            straddles=(float(big_blind * 2),) if wants_straddle else (),
        ),
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


# --- The model, restated here and nowhere else ------------------------------
#
# Three helpers derived from the four rules of the pot-layering specification,
# NOT from ``accounting._build_pots``. Every payout property below is stated
# through them, so there is exactly one place in this suite where the model is
# written down and it can be read against the specification by eye:
#
#   1. Boundaries are cut at distinct LIVE contribution levels, after refunds.
#   2. ALL dead money goes into the lowest layer and opens no boundary.
#   3. A seat is eligible for a layer if its own LIVE contribution reaches that
#      layer's level; every unfolded seat that put ANY chip up contests the main
#      pot.
#   4. A folded seat's chips stay where they landed and it is eligible for none.


def _model_figures(hand, ledger) -> tuple[dict[str, Decimal], Decimal]:
    """Each seat's LIVE money after refunds, and the whole dead pool.

    The two quantities rule 1 and rule 2 are stated over. ``hand.live`` is what
    the generator wagered before any uncalled money came back; the refund is read
    off the ledger because it is not in dispute -- it is measured against live
    money only, and ``test_a_forced_post_is_never_returned_as_an_uncalled_bet``
    is the guarantee on that.
    """

    live = {
        name: hand.live[name] - Decimal(str(ledger.refunds[name]))
        for name in hand.names
    }
    pool = _sum(hand.dead.values()) + hand.external_dead
    return live, pool


def _model_main_pot_eligibility(hand) -> set[str]:
    """Who contests the main pot, from rule 3's second sentence and rule 4.

    "Every unfolded seat that put ANY chip up -- live or dead -- is eligible for
    the main pot", read PRE-refund: a seat whose only live post came back still
    played the hand, and the return is what the table did about it afterwards.

    This exists because every payout property below used to take the main pot's
    eligible set straight off ``ledger.pots[0].eligible_players`` and then assert
    the cap FOR THAT SET. Whatever set the reducer named, the property agreed with
    it -- so widening the main pot to a seat that put nothing up, or to the
    declarer of external dead money, passed all thirty properties while paying
    that seat the whole dead pool. The cap arithmetic was independent of the
    reducer; the eligibility it was applied to was not, and eligibility is half
    the model.
    """

    return {
        name
        for name in hand.names
        if name not in hand.folded and hand.live[name] + hand.dead[name] > _ZERO
    }


def _model_payout_cap(
    seat: str, live: dict[str, Decimal], pool: Decimal, contests_main: bool
) -> Decimal:
    """The most chips ``seat`` can collect, read straight off the spec sentence.

    You get your own live money back, you win from each opponent only the LIVE
    chips that opponent matched OF YOURS, and you win the dead money only if you
    win the layer it sits in -- which rule 2 makes always the main pot.

    Three things this deliberately does NOT do, each of which is how the property
    it replaces got it wrong:

    * the seat's OWN dead posts never appear inside the ``min()``. An opponent
      does not match your ante. Your ante is owed to the table and is already
      counted once, in ``pool``.
    * the OPPONENT's dead posts never appear inside the ``min()`` either, for the
      same reason.
    * a folded opponent gets no free pass to contribute its whole stack. It is
      capped at ``min(live[other], live[seat])`` exactly like anybody else,
      because rule 4 leaves a folded seat's chips in the layers its LIVE money
      reached and no higher, and a seat short of that cannot reach them.
    """

    own = live[seat]
    matched = _sum(min(live[other], own) for other in live if other != seat)
    return own + matched + (pool if contests_main else _ZERO)


def _eligibility_threshold(eligible, live, contenders) -> Decimal:
    """The live level a layer's eligible set behaves as a threshold on."""

    return min((live[name] for name in eligible), default=_ZERO)


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
def test_a_seat_that_wins_every_layer_it_can_is_paid_exactly_the_model_cap(hand):
    """THE payout property. It replaces two that agreed with the reducer's own error.

    Award every layer to one seat wherever that seat is eligible. The chips it
    collects must be, to the chip:

        its own live money back
        + the LIVE chips each opponent matched of it
        + the whole dead pool, if it contests the main pot

    and a seat that contests no layer must be paid nothing. Stated as EQUALITY
    rather than as an upper bound on purpose: an inequality catches only
    overpayment, and half the disagreements this model resolved were the pot
    layers holding dead money ABOVE the main pot, which UNDERPAYS the short seat
    the antes it is owed. One assertion catches both directions and, swept over
    every seat of a hand, pins every layer amount in the ladder.

    WHAT THIS REPLACES, AND WHY THE OLD ONE COULD NOT CATCH WHAT IT WAS FOR.
    The property here used to cap a seat at

        in_pot[name] + sum(min(in_pot[other], put_up[name]) for unfolded others)
        + sum(in_pot[other] for folded others)
        + sum(dead[other] for unfolded others)

    with ``put_up`` = the seat's contribution INCLUDING ITS OWN DEAD POSTS and
    ``in_pot`` = the opponent's contribution including THAT OPPONENT'S dead posts.
    That is not an independent check of the reducer; it is the reducer's own
    modelling error written down a second time. ``_build_pots`` cut its first
    boundary at a short seat's live-plus-dead TOTAL, and this expression measured
    against exactly the same total, so the two agreed by construction and their
    agreement carried no information. On the reported hand -- big-blind ante of
    10, stacks 26/500/500 -- it evaluates to min(26,26) + min(20,26) + min(20,26)
    = 66, which is to the chip the number the reducer produced and 8 chips more
    than the 58 the model owes: the big blind was paid 4 live chips by each of two
    opponents that neither had wagered against it, and the hand reported settled,
    balanced and legal with no warning. 3110 passing tests did not see it because
    the invariant encoded the same error as the code.

    The replacement takes its cap from the specification instead
    (``_model_payout_cap``), so the two sides can disagree. On the same hand it
    gives 16 + min(20,16) + min(20,16) + 10 = 58.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, pool = _model_figures(hand, ledger)
    contests_main = set(ledger.pots[0].eligible_players)
    # Rule 3 is checked, not borrowed. Taking the eligible set off the ledger and
    # then asserting the cap FOR THAT SET is agreement by construction: widen the
    # main pot to a seat that put nothing up and the cap widens with it.
    assert contests_main == _model_main_pot_eligibility(hand), (
        "the main pot's eligible set is not the unfolded seats that put a chip up"
    )

    for name in hand.names:
        settled = build_hand_ledger(
            hand.players,
            hand.actions,
            _winner_map(ledger.pots, (name,)),
            dead_money=dead_money,
        )
        if not settled.is_settled:
            continue
        paid = Decimal(str(settled.payouts[name]))
        if name in contests_main:
            assert paid == _model_payout_cap(name, live, pool, True), (
                f"{name} wins every layer it is eligible for and is not paid the "
                "chips the model says it is owed"
            )
        else:
            # Folded, or in the hand having put nothing up. Eligibility is nested,
            # so a seat out of the main pot is out of every layer.
            assert paid == _ZERO, f"{name} contests no layer and was paid {paid}"


@given(hand=forced_post_hand(), policy=rake_policy())
@SETTINGS
def test_no_declared_settlement_can_pay_a_seat_past_the_model_cap(hand, policy):
    """The same cap as an upper bound, under rake and after a chop.

    The equality above is stated over one specific award map. This one sweeps
    arbitrary ones -- every layer chopped between every seat eligible for it,
    with a raked policy on top -- because the bound has to hold of a seat's actual
    share AFTER the split and after the drop, not of the layer it came out of.
    A rake allocator that moved a chip from one layer to another, or a split that
    handed an odd chip across a boundary, would break this while leaving the
    ladder itself correct.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(
        hand.players, hand.actions, dead_money=dead_money, rake=policy
    )
    assume(ledger.pots)
    live, pool = _model_figures(hand, ledger)
    contests_main = set(ledger.pots[0].eligible_players)
    awards = {
        pot.index: tuple(pot.eligible_players)
        for pot in ledger.pots
        if pot.eligible_players
    }
    assume(len(awards) == len(ledger.pots))
    settled = build_hand_ledger(
        hand.players,
        hand.actions,
        awards,
        dead_money=dead_money,
        rake=policy,
        odd_chip_order=tuple(hand.names),
    )
    for name in hand.names:
        paid = Decimal(str(settled.payouts[name]))
        assert paid >= _ZERO
        assert paid <= _model_payout_cap(name, live, pool, name in contests_main), (
            f"{name} was paid past the live chips its opponents matched of it"
        )


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


# --- Adversarial round 18: dead money is not a layer boundary ----------------


@given(hand=forced_post_hand())
@SETTINGS
def test_unequal_dead_money_alone_never_splits_the_pot(hand):
    """Nobody stopped short of the live wagering, so there is one pot.

    Antes and dead blinds are owed to the table, not wagered at anybody, so a
    seat that owes more of them is not a seat anyone declined to match. Cutting
    the pot at every distinct TOTAL commitment made a side pot out of exactly
    that difference -- one seat's ante against another's dead blind, everybody
    matching the same live bet, nobody all-in -- and then refused the seat that
    won the hand the chips it had won, because it was "not eligible for pot 1".

    The generator reaches this on every hand with a button ante or a dead blind
    where the action closes, which is why it is stated over the whole family
    rather than as one example.

    Counted over CONTESTABLE layers rather than over all of them. A line in which
    every seat that wagered above the line folded leaves chips in a band no
    remaining seat is eligible for; the model strands that band rather than
    merging it down into a pot a short seat could win. Such a band is not a pot
    boundary in the sense this property is about -- nobody is being held apart
    from anybody -- and counting it would make the property assert the merge-down
    that the payout cap exists to forbid.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    settled_live, _pool = _model_figures(hand, ledger)
    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)
    line = max(settled_live[name] for name in contenders)
    if all(settled_live[name] == line for name in contenders):
        contestable = [pot for pot in ledger.pots if pot.eligible_players]
        assert len(contestable) == 1, (
            "every seat still in the hand covered the same live wager, so there "
            "is nothing for a second layer to hold apart"
        )


@given(hand=forced_post_hand())
@SETTINGS
def test_every_layer_boundary_is_a_clean_cut_at_a_live_contribution_level(hand):
    """The ladder half of the model: WHERE a boundary may be drawn, and on WHAT.

    This is the property that forbids the phantom side pot, and its currency is
    the whole point. It used to assert the clean cut over each seat's TOTAL
    commitment --

        max(put_up[dropped]) <= min(put_up[kept])

    -- which is the reducer's modelling error stated as an invariant: it is
    satisfied by a boundary cut at a seat's live-plus-ante total, and satisfied by
    nothing else. On the big-blind ante hand the big blind's total of 26 exceeds
    the 20 of the two seats kept in the layer above it, so the CORRECT ladder
    fails that assertion outright. A folded seat's ante can put it on the wrong
    side of a cut it has nothing to do with.

    The cut is on LIVE money, and on live money the statement is not merely true
    but strict:

    * WHERE. A boundary exists only where a seat still in the hand stopped short
      of the live wagering. Unequal dead money cannot open one, because no
      opponent can decline a forced post, and a seat's own dead posts cannot
      raise the level its opponents are charged into the layer below at.
    * HOW. Every seat dropped by a boundary wagered strictly less live money than
      every seat kept, so the cut is a threshold on live contribution and nothing
      else. Eligibility is therefore recoverable from the live figures alone,
      which is asserted directly below.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    live, _pool = _model_figures(hand, ledger)
    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)
    line = max(live[name] for name in contenders)
    short = {name for name in contenders if live[name] < line}
    for lower, upper in zip(ledger.pots, ledger.pots[1:], strict=False):
        assert set(upper.eligible_players) < set(lower.eligible_players), (
            f"layer {upper.index} does not strictly narrow the layer below it"
        )
        dropped = set(lower.eligible_players) - set(upper.eligible_players)
        assert dropped & short, (
            f"layer {upper.index} was opened without stopping anybody short of the "
            f"live wager: it dropped {sorted(dropped)}, none of whom declined a chip"
        )
        if not upper.eligible_players:
            continue
        assert max(live[name] for name in dropped) < min(
            live[name] for name in upper.eligible_players
        ), (
            f"layer {upper.index} is not a clean cut on live money: it dropped "
            f"{sorted(dropped)} while keeping a seat that wagered no more"
        )
        # And the threshold is exact in both directions: every unfolded seat that
        # reached the level is IN, every seat that did not is OUT. A boundary at a
        # total-contribution level fails this -- it separates seats holding equal
        # live money, which is the phantom in its purest form.
        threshold = _eligibility_threshold(upper.eligible_players, live, contenders)
        assert set(upper.eligible_players) == {
            name for name in contenders if live[name] >= threshold
        }, f"layer {upper.index} is not a threshold on live contribution"


@given(hand=forced_post_hand())
@SETTINGS
def test_every_dead_chip_is_in_the_main_pot_and_no_layer_above_it_holds_one(hand):
    """Rule 2, stated so that a leak in either direction is caught.

    All dead money -- antes, big-blind antes, dead blinds, and dead money declared
    for the table -- goes into the LOWEST layer whole. Two consequences, and the
    reason both are asserted rather than one:

    * the main pot is never smaller than the dead pool. When the reducer carried
      part of the antes UP into a side pot the short seat could not win, the short
      seat was underpaid the antes it was owed, and that was 58.6% of the
      disagreements the model resolved -- more than the overpayment shape, and
      completely silent, because chip conservation still held.
    * everything above the main pot is purely live, so the layers above it total
      exactly the live money wagered above the main pot's ceiling.

    The second is stated through the main pot's own live ceiling, recovered from
    the ladder rather than from the reducer: it is the highest live level whose
    threshold set is still the main pot's eligible set. A hand whose main pot is
    nothing but dead money has a ceiling of zero, and every live chip is above it.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, pool = _model_figures(hand, ledger)
    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)

    assert Decimal(str(ledger.pots[0].amount)) >= pool, (
        "the main pot holds less than the dead money owed to the table, so part "
        "of it leaked into a layer the seats that owed it may not be able to win"
    )
    if len(ledger.pots) == 1:
        return
    contests_main = set(ledger.pots[0].eligible_players)
    ceiling = max(
        [_ZERO]
        + [
            level
            for level in {live[name] for name in hand.names if live[name] > 0}
            if {name for name in contenders if live[name] >= level} == contests_main
        ]
    )
    above = _sum(pot.amount for pot in ledger.pots[1:])
    assert above == _sum(
        max(_ZERO, live[name] - ceiling) for name in hand.names
    ), "the layers above the main pot do not hold exactly the live money above it"


@given(hand=forced_post_hand())
@SETTINGS
def test_a_seat_short_of_the_live_wager_is_paid_only_the_live_chips_it_was_matched(hand):
    """The short seat's cap, with no free pass for a folded opponent.

    This replaces a property whose cap was

        sum(in_pot[other] if other in folded else min(in_pot[other], put_up[name]))

    with ``put_up`` and ``in_pot`` both including their own seat's DEAD posts. Two
    separate errors sat in that one line, and both of them pointed the same way as
    the reducer, which is why 3110 tests agreed with a layering that pays a
    live-short seat chips no opponent matched:

    * ``put_up[name]`` conflated the seat's live wager with its own antes. A seat
      live-short by 5 with a 5 ante was allowed to be paid as though it had
      wagered 5 more than it did, from every opponent. On the reported big-blind
      ante hand the expression came out at 66 -- exactly the number the reducer
      produced, and 8 more than the 58 the model owes.
    * the folded-seat free pass let a folded opponent contribute its ENTIRE
      commitment to the cap. That was a concession to the reducer's old
      "merge an unwinnable layer DOWN rather than strand it" branch, which is
      itself the sixth critical waiting to happen: merging folded chips down past
      a short seat's live level pays that seat money nobody wagered at it. The
      model strands such a layer instead, so a folded opponent is capped at
      ``min(live[other], live[name])`` like everybody else and the free pass is
      not needed.

    THE SEED-1 WEAKENING, REVISITED. That docstring recorded the free pass being
    added after ``--hypothesis-seed=1`` at 2500 examples reached "a hand where one
    seat's dead blind and live small blind total 6 before it folds, against two
    seats holding 2 each", and nobody acted on the signal. Under the correct model
    that hand is fully explicable and needs no weakening: the folded seat's live
    money above 2 sits in a band whose eligible set is empty, and an empty band is
    stranded, not merged down, so no seat holding 2 can be paid out of it. The
    tight cap below is asserted with no exception for folded seats, and the suite
    is run at that seed and example count to prove the hand no longer escapes it.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, pool = _model_figures(hand, ledger)
    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)
    line = max(live[name] for name in contenders)
    contests_main = set(ledger.pots[0].eligible_players)

    for name in contenders:
        if live[name] >= line:
            continue
        settled = build_hand_ledger(
            hand.players,
            hand.actions,
            _winner_map(ledger.pots, (name,)),
            dead_money=dead_money,
        )
        if not settled.is_settled:
            continue
        assert Decimal(str(settled.payouts[name])) <= _model_payout_cap(
            name, live, pool, name in contests_main
        ), f"{name} stopped short of the live wager and was paid past what it was matched"


# --- The blind structure ----------------------------------------------------
#
# The reported defect was that ``to_call`` was derived from the largest OBSERVED
# contribution, so a big blind all-in for 4 in a 5/10 game told the rest of the
# table it owed 5. The repair makes the structural sizes an INPUT. These
# properties pin what that input may and may not do, over hands nobody wrote by
# hand -- because this module has produced a critical in each of the last four
# adversarial rounds, three of them introduced by the repair to the previous one.


BLIND_STRUCTURES = st.one_of(
    st.none(),
    st.builds(
        BlindStructure,
        small_blind=st.sampled_from([None, 0.0, 0.5, 1.0, 2.0, 5.0]),
        big_blind=st.sampled_from([0.5, 1.0, 2.0, 5.0, 10.0, 1000.0]),
        straddles=st.sampled_from([(), (4000.0,), (2000.0, 4000.0)]),
    ),
)


@given(hand=forced_post_hand(), structure=BLIND_STRUCTURES, policy=rake_policy())
@SETTINGS
def test_no_blind_structure_moves_a_single_chip(hand, structure, policy):
    """The safety property the whole repair rests on: the floor is not money.

    A declared structure changes what the reducer will CALL LEGAL. It must never
    change what the reducer COUNTS -- not the contributions, not a refund, not a
    pot layer, not a payout, not the rake, not the balance. Any structure at all
    is swept here, including absurd and self-contradictory ones, because the
    guarantee has to hold for a mis-declaration too: an operator who types the
    wrong big blind must get a loud legality complaint, never a quietly
    different pot.

    This is also the guarantee that makes the schema 19 migration safe. Existing
    rows declare nothing, and adding a declaration later cannot restate a single
    figure the hand already reported.
    """
    try:
        baseline = build_hand_ledger(hand.players, hand.actions, rake=policy)
    except LedgerError:
        assume(False)
        return
    try:
        declared = build_hand_ledger(
            hand.players, hand.actions, rake=policy, blinds=structure
        )
    except LedgerError:
        # A structure no room could have is refused outright, which is a refusal
        # to derive rather than a different derivation.
        assume(False)
        return

    assert declared.contributions == baseline.contributions
    assert declared.refunds == baseline.refunds
    assert declared.payouts == baseline.payouts
    assert declared.net_results == baseline.net_results
    assert declared.gross_pot == baseline.gross_pot
    assert declared.rake == baseline.rake
    assert declared.net_pot == baseline.net_pot
    assert declared.folded_players == baseline.folded_players
    assert declared.is_settled == baseline.is_settled
    assert declared.is_balanced == baseline.is_balanced
    assert [
        (pot.amount, pot.contributors, pot.eligible_players, pot.cause)
        for pot in declared.pots
    ] == [
        (pot.amount, pot.contributors, pot.eligible_players, pot.cause)
        for pot in baseline.pots
    ]


@given(hand=forced_post_hand())
@SETTINGS
def test_an_unreadable_forced_post_is_never_silently_accepted(hand):
    """Loudness, stated as a property rather than as the one reported example.

    Whenever a live forced post took its poster's last chip, the action line
    cannot show the size of the bet it was paying, and an undeclared hand must
    say so. The complaint names the seat and the clearing action, and
    ``is_legal`` goes False so no surface can present the hand as reconciled.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    complaints = [
        issue
        for issue in ledger.legality_issues
        if "Declare the blind structure" in issue
    ]
    if hand.short_live_post:
        assert complaints, "a short live forced post was accepted in silence"
        assert ledger.is_legal is False
    else:
        assert not complaints


@given(hand=forced_post_hand())
@SETTINGS
def test_declaring_the_structure_that_was_dealt_answers_the_complaint(hand):
    """And the complaint has a reachable clearing action, on every generated hand.

    The generator knows the sizes it dealt. Declaring exactly those must remove
    every unreadable-post complaint -- otherwise the blocker would be one an
    honest operator could not clear, which is a worse failure than the one it
    replaced.
    """
    ledger = build_hand_ledger(hand.players, hand.actions, blinds=hand.blinds)
    assert not [
        issue
        for issue in ledger.legality_issues
        if "Declare the blind structure" in issue
    ]


@given(hand=forced_post_hand())
@SETTINGS
def test_a_declared_structure_never_lowers_the_amount_to_call(hand):
    """It is a floor. A declaration may raise what a seat owes and never reduce it.

    That direction is what stops the input becoming the next free parameter: no
    value an operator types can excuse a call the recording proves was short.
    """
    baseline = build_hand_ledger(hand.players, hand.actions)
    declared = build_hand_ledger(hand.players, hand.actions, blinds=hand.blinds)
    for before, after in zip(baseline.snapshots, declared.snapshots, strict=True):
        assert after.to_call_before >= before.to_call_before - 1e-9
