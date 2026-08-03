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


CAPPED_NAMES = ("alice", "bob", "carol", "dana", "erin", "frank")


@st.composite
def capped_dead_money_hand(draw):
    """Hands built to reach the family the operator's amendment governs.

    WHY A SECOND GENERATOR EXISTS. ``forced_post_hand`` draws a room -- blinds,
    antes, straddles, folds -- and the amended rule 2 only bites when some
    contributor's forced post EXCEEDS the smallest total commitment among the
    seats eligible for the layer it started in. That needs a large post beside a
    small surviving stack, and the extra structure the amendment's two failure
    modes need on top of it:

    * a seat still in the hand with dead money and ZERO live money, sitting under
      a live band ABOVE the dead cap -- the shape in which "spill the risen dead
      into the live band above" strands that seat's own chips where it cannot win
      them. It needs at least three distinct live levels with two seats sharing
      the top one, or the uncalled-bet refund flattens the ladder before the
      layering ever sees it.
    * three or more distinct total commitments carrying dead money past two
      successive caps -- the shape in which "let the excess rise once and then
      stop" overpays the middle seats.

    Both were REACHABLE in principle from the room generator and were not being
    reached: a mutation run scored 15 of 18 with the payout-cap properties alone,
    and all three survivors were survivors of the INPUT FAMILY rather than of the
    assertion. Broadening the family is the fix; weakening the assertion or
    dropping the mutant would not have been.

    The line is deliberately schematic -- every commitment is a forced post or a
    single all-in -- because what is under test here is the layering, not the
    betting grammar. ``forced_post_hand`` covers the grammar.
    """

    chip = draw(CHIPS)
    count = draw(st.integers(min_value=3, max_value=6))
    names = list(CAPPED_NAMES[:count])
    top = chip * draw(st.integers(min_value=1, max_value=40))
    mid = chip * draw(st.integers(min_value=0, max_value=40))
    if mid > top:
        top, mid = mid, top
    ante = chip * draw(st.sampled_from([0, 1, 2, 5, 20, 60, 100]))
    external_dead = chip * draw(st.sampled_from([0, 0, 0, 1, 4]))

    # Seat 0 posts and wagers nothing live. Seats 1 and 2 share the top live
    # level so the uncalled-bet refund cannot flatten the ladder.
    live_plan = [_ZERO, top, top]
    dead_plan = [
        ante,
        chip * draw(st.sampled_from([0, 1, 5])),
        chip * draw(st.sampled_from([0, 1, 5])),
    ]
    for _ in names[3:]:
        live_plan.append(draw(st.sampled_from([_ZERO, mid, top])))
        dead_plan.append(chip * draw(st.sampled_from([0, 1, 2, 5, 20, 60])))
    # A seat holding the top live level never folds. Folding every seat that
    # wagered at a level strands the band those chips landed in -- the model
    # refuses to merge it down, which is correct and is a different property's
    # subject (``test_a_layer_no_remaining_seat_can_win_is_stranded``). Letting
    # this generator produce it would make the payout properties assert
    # settleability they never promised.
    peak = max(live_plan)
    folds = [
        draw(st.booleans()) and live_plan[index] < peak for index in range(count)
    ]

    players = [
        LedgerPlayer(
            name=name,
            starting_stack=float(dead_plan[index] + live_plan[index]),
            seat=index,
        )
        for index, name in enumerate(names)
    ]
    actions: list[LedgerAction] = []
    dead = {name: _ZERO for name in names}
    live = {name: _ZERO for name in names}
    posted = {name: _ZERO for name in names}
    wagered = {name: _ZERO for name in names}
    folded: set[str] = set()

    for index, name in enumerate(names):
        if dead_plan[index] > 0:
            actions.append(
                LedgerAction(
                    player=name,
                    street="preflop",
                    kind="ante",
                    amount=float(dead_plan[index]),
                    is_live_post=False,
                    forced_bet_type="ante",
                )
            )
            dead[name] += dead_plan[index]
            posted[name] += dead_plan[index]
    for index, name in enumerate(names):
        if live_plan[index] > 0:
            actions.append(
                LedgerAction(
                    player=name,
                    street="preflop",
                    kind="all-in",
                    amount=float(live_plan[index]),
                )
            )
            live[name] += live_plan[index]
            wagered[name] += live_plan[index]
    for index, name in enumerate(names):
        # A seat that never put a chip up cannot be the one seat left in, and a
        # hand where every contributor folded has no eligible seat at all -- the
        # reducer refuses it by design, which is a different property's subject.
        if folds[index] and dead_plan[index] + live_plan[index] > 0:
            actions.append(LedgerAction(player=name, street="preflop", kind="fold"))
            folded.add(name)
    if not [
        name
        for name in names
        if name not in folded and dead[name] + live[name] > 0
    ]:
        keep = next(
            name for name in names if dead[name] + live[name] > 0
        ) if any(dead[n] + live[n] > 0 for n in names) else names[0]
        actions = [
            action
            for action in actions
            if not (action.player == keep and action.kind == "fold")
        ]
        folded.discard(keep)

    return ForcedPostHand(
        players=players,
        actions=actions,
        names=names,
        stacks=[dead_plan[i] + live_plan[i] for i in range(count)],
        dead=dead,
        live=live,
        posted=posted,
        wagered=wagered,
        folded=folded,
        external_dead=external_dead,
        blinds=None,
    )


@st.composite
def refunded_shove_hand(draw):
    """A shove that came PARTLY back, beside opponents carrying dead money.

    THE FAMILY, AND THE MUTANT THAT PROVED IT WAS MISSING. Rule 2's cap operand
    is a seat's TOTAL COMMITMENT, and ``_build_pots`` reads that as live money
    AFTER the uncalled-bet refund, falling back on what the seat put up only when
    the post-refund total is zero. Widen that fallback to ``max(total, put_up)``
    -- one plausible slip, since the zero case is written as a fallback rather
    than as a boundary -- and a seat refunded 90 of a 100-chip shove is capped as
    if it still had 100 at risk, so opponents' antes stop being capped out of the
    main pot and it collects them. On the smallest witness (two 50-chip antes, a
    100-chip shove called for 10) the shipped model derives 50 and 80 and pays
    the shover 50; the widened fallback derives one pot of 130 and pays it 130 --
    80 chips over the closed-form cap, settled, balanced, legal and WARNING-FREE.

    The payout-cap properties are stated over exactly the right quantity to catch
    that: they model the total as post-refund live plus own dead, and they
    ``assume`` away only the seat refunded to NOTHING. But neither existing
    generator can build the shape. ``forced_post_hand`` refunds a shover only
    when the room leaves it unmatched, and never against opponents whose antes
    then exceed what it kept; ``capped_dead_money_hand`` gives its top live level
    to two seats on purpose, "or the uncalled-bet refund flattens the ladder",
    which is precisely the case ruled out here. So the mutant survived the whole
    accounting suite -- 164 tests -- and it is a survivor of the INPUT FAMILY,
    not of the assertion. This generator is the fix; weakening the mutant or
    relaxing the ``assume`` would not have been.

    THE SHAPE, and every part of it is load-bearing. One seat shoves strictly
    more than anybody can call, so it is the unique highest live contributor and
    is refunded down to the call level but NOT to zero. Its opponents call that
    level and carry antes large enough to push their own totals past the
    shover's, so the shover holds the smallest commitment among the seats
    contesting the main pot and the amended cap decides the hand.
    """

    chip = draw(CHIPS)
    count = draw(st.integers(min_value=2, max_value=5))
    names = list(CAPPED_NAMES[:count])
    call_level = chip * draw(st.integers(min_value=1, max_value=20))
    # Strictly more than the call level, so the refund is partial and the shover
    # is the unique top live contributor.
    shove = call_level + chip * draw(st.integers(min_value=1, max_value=40))
    shover_ante = chip * draw(st.sampled_from([0, 0, 1, 2]))
    external_dead = chip * draw(st.sampled_from([0, 0, 0, 1, 4]))
    antes = [shover_ante] + [
        chip * draw(st.sampled_from([0, 1, 2, 5, 20, 50, 100])) for _ in names[1:]
    ]
    # A caller may be short of the call level, which is what puts a live boundary
    # under the dead one. Nobody folds: a folded caller cannot flatten the
    # refund, but it can strand the shover as the only seat left, and the shapes
    # this generator exists for all need at least two contenders.
    calls = [call_level] + [
        draw(st.sampled_from([call_level, call_level, chip * draw(
            st.integers(min_value=1, max_value=20))]))
        for _ in names[1:]
    ]
    calls = [min(amount, call_level) for amount in calls]
    calls[0] = shove

    players = [
        LedgerPlayer(
            name=name,
            starting_stack=float(antes[index] + calls[index]),
            seat=index,
        )
        for index, name in enumerate(names)
    ]
    actions: list[LedgerAction] = []
    dead = {name: _ZERO for name in names}
    live = {name: _ZERO for name in names}
    posted = {name: _ZERO for name in names}
    wagered = {name: _ZERO for name in names}

    for index, name in enumerate(names):
        if antes[index] > 0:
            actions.append(
                LedgerAction(
                    player=name,
                    street="preflop",
                    kind="ante",
                    amount=float(antes[index]),
                    is_live_post=False,
                    forced_bet_type="ante",
                )
            )
            dead[name] += antes[index]
            posted[name] += antes[index]
    for index, name in enumerate(names):
        if calls[index] > 0:
            actions.append(
                LedgerAction(
                    player=name,
                    street="preflop",
                    kind="all-in",
                    amount=float(calls[index]),
                )
            )
            live[name] += calls[index]
            wagered[name] += calls[index]

    return ForcedPostHand(
        players=players,
        actions=actions,
        names=names,
        stacks=[antes[i] + calls[i] for i in range(count)],
        dead=dead,
        live=live,
        posted=posted,
        wagered=wagered,
        folded=set(),
        external_dead=external_dead,
        blinds=None,
    )


# The input family every LAYERING property is stated over: an ordinary room, a
# room built to reach the amended rule's own failure modes, and a room in which
# rule 2's cap operand is measured across an uncalled-bet refund. Properties
# about the blind structure or the betting grammar stay on ``forced_post_hand``
# alone, because the schematic generators have no grammar to test.
LAYERING_HANDS = st.one_of(
    forced_post_hand(), capped_dead_money_hand(), refunded_shove_hand()
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
# Helpers derived from the four rules of the pot-layering specification AS THE
# OPERATOR AMENDED THEM, and NOT from ``accounting._build_pots``. Every payout
# property below is stated through them, so there is exactly one place in this
# suite where the model is written down and it can be read against the
# specification by eye:
#
#   1. Boundaries are cut at distinct LIVE contribution levels, after refunds.
#   2. Dead money goes into the LOWEST layer, but EACH CONTRIBUTOR'S dead chips
#      count into a layer only up to the smallest TOTAL commitment among that
#      layer's eligible seats. The excess rises into the layer above, eligible to
#      the seats whose own total reached above that cap.
#   3. A seat is eligible for a layer if its own LIVE contribution reaches that
#      layer's level; every unfolded seat that put ANY chip up contests the main
#      pot.
#   4. A folded seat's chips stay where they landed and it is eligible for none.
#
# WHY RULE 2 IS NOT THE ONE THIS FILE USED TO STATE. It used to read "ALL dead
# money goes into the lowest layer", unconditionally, and every payout property
# here was written through a cap that said so. That cap measured a short seat's
# ceiling using that seat's own dead posts -- the same modelling error the
# reducer made -- so the suite and the code AGREED BY SHARING A PREMISE and the
# agreement carried no information. The operator has since ruled: a seat whose
# whole commitment is 60 against three 100-chip antes is owed 240, not the 360 in
# the pot and not the 540 the old cap allowed. The cap below is a closed form
# that NEVER CONSULTS THE LAYERING -- it is written from the four rules and the
# five worked examples alone -- so the two sides can disagree, and the mutation
# suite in ``tests/test_accounting_pot_layering_mutants.py`` proves they do.


def _model_figures(hand, ledger) -> tuple[dict[str, Decimal], Decimal]:
    """Each seat's LIVE money after refunds, and the whole dead pool.

    The two quantities rules 1 and 2 are stated over. ``hand.live`` is what the
    generator wagered before any uncalled money came back; the refund is read off
    the ledger because it is not in dispute -- it is measured against live money
    only, and ``test_a_forced_post_is_never_returned_as_an_uncalled_bet`` is the
    guarantee on that.
    """

    live = {
        name: hand.live[name] - Decimal(str(ledger.refunds[name]))
        for name in hand.names
    }
    pool = _sum(hand.dead.values()) + hand.external_dead
    return live, pool


def _model_totals(hand, live: dict[str, Decimal]) -> dict[str, Decimal]:
    """Each seat's TOTAL COMMITMENT: live money that stuck plus its own posts.

    Rule 2's cap operand. Its own dead posts are IN it -- that is what "total"
    means -- and it is the one place a seat's own antes are allowed to raise a
    number, precisely because the number they raise is the seat's own ceiling and
    never what an opponent is charged.
    """

    return {name: live[name] + hand.dead[name] for name in hand.names}


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


def _refunded_to_nothing(hand, totals: dict[str, Decimal]) -> set[str]:
    """Seats still in the hand whose whole commitment came back as an uncalled bet.

    The ONE shape the amended rule 2 does not decide, and the reason it is
    isolated here rather than absorbed into the cap. Rule 3's "put ANY chip up" is
    read before the refund, so such a seat contests the main pot; rule 2's cap is
    written against a TOTAL COMMITMENT which, read after the refund, is zero. The
    specification never has to choose a measurement point because none of the five
    worked examples contains a refund.

    The reducer takes a reading (it caps against what the seat put up) AND
    DISCLOSES IT -- ``test_a_seat_whose_whole_commitment_came_back_is_disclosed``
    is the guarantee on that. The cap properties below decline to score those
    hands rather than encode the reducer's choice, which is what "do not derive
    your expectation from the implementation" means when the specification is
    genuinely silent.
    """

    return {
        name
        for name in hand.names
        if name not in hand.folded
        and hand.live[name] + hand.dead[name] > _ZERO
        and totals[name] <= _ZERO
    }


def _model_payout_cap(
    seat: str,
    live: dict[str, Decimal],
    dead: dict[str, Decimal],
    folded: set[str],
    external_dead: Decimal,
) -> Decimal:
    """The most chips ``seat`` can collect, as a closed form over the four rules.

    It never looks at a layer. That is the whole point: the previous cap was
    satisfied by the layering it was meant to referee.

        cap(w) = live[w]
               + SUM over o != w of min(live[o], live[w])
               + SUM over ALL contributors x, FOLDED INCLUDED, of
                     min(dead[x], total[w])
               + (the dead money above every surviving total, if w holds the
                  largest surviving total)
               + external dead money

    TERM BY TERM, AND THE ERROR EACH ONE FORBIDS.

    * ``live[w]``: your own live money back.
    * ``min(live[o], live[w])``: each opponent's live money matched ONLY up to
      YOUR live level. Your own DEAD posts must never appear inside this min().
      An opponent does not match your ante. Folding them in is exactly what turns
      240 into 540 on the operator's worked example (e), and it is what the
      shipped reducer's first boundary used to do.
    * ``min(dead[x], total[w])``: rule 2 as amended. ``w`` is eligible for dead
      layer k exactly when total[w] exceeds layer k-1's cap; the highest such
      layer has a cap of exactly total[w], because w is itself the smallest total
      still eligible there. So each contributor's dead money reaches w capped at
      w's OWN total -- which is where a seat's own antes legitimately appear, as
      its own ceiling. Worked example (a) pins that the comparison is against the
      seat's DEAD alone and not dead-plus-live: the big blind's 16 live in the
      layer plus its 10 ante is 26, above the 20 cap, and the whole ante still
      stays down.
    * the terminal term: rule 2 says the excess "rises into the layer above,
      eligible to the seats whose own total reached above that cap", and says
      nothing when no seat's total reaches above it. Such money can only ever be
      a FOLDED seat's, since an unfolded seat's dead is at most its own total and
      the last cap is the largest surviving total. It stays in the top layer,
      which only a seat holding that largest total can win. This is the single
      place a seat's ceiling is not simply its own total, and it is written out
      rather than hidden -- ``_unruled_dead_money_warnings`` discloses the same
      hand.
    * ``external_dead``: money declared for the table has no contributing seat,
      so rule 2's cap -- written per contributor against that contributor's own
      total -- has no operand for it. It stays in the lowest layer whole.

    A folded seat, or one that put nothing in, collects nothing.
    """

    names = list(live)
    total = {name: live[name] + dead[name] for name in names}
    if seat in folded or total[seat] <= _ZERO:
        return _ZERO

    own_live = live[seat]
    cap = own_live + _sum(min(live[other], own_live) for other in names if other != seat)
    cap += _sum(min(dead[name], total[seat]) for name in names)

    surviving = [total[name] for name in names if name not in folded and total[name] > 0]
    if surviving and total[seat] == max(surviving):
        ceiling = max(surviving)
        cap += _sum(max(_ZERO, dead[name] - ceiling) for name in names)
    return cap + external_dead


def _amendment_bites(hand, live: dict[str, Decimal], totals: dict[str, Decimal]) -> bool:
    """True when the amended rule 2 can differ from the unconditional one.

    Some contributor's dead money exceeds the smallest TOTAL commitment among the
    seats eligible for the main pot, so at least one dead chip is capped out of
    the layer it started in. Derived from the specification sentence, not from
    either implementation, so it can be used to say "on this family the two rules
    coincide and the old assertions must still hold to the chip".
    """

    contests = _model_main_pot_eligibility(hand)
    if not contests:
        return False
    floor = min(totals[name] for name in contests)
    return any(hand.dead[name] > floor for name in hand.names)


def _live_threshold_sets(hand, live: dict[str, Decimal]) -> set[frozenset[str]]:
    """Every eligible set rule 1 + rule 3 can legally produce for a live band."""

    contenders = [name for name in hand.names if name not in hand.folded]
    return {
        frozenset(name for name in contenders if live[name] >= level)
        for level in {live[name] for name in hand.names if live[name] > 0}
    }


def _total_cut_sets(hand, totals: dict[str, Decimal]) -> set[frozenset[str]]:
    """Every eligible set rule 2's second sentence can legally produce.

    ``{unfolded : total > cap}`` where the cap is zero (which is rule 3's second
    sentence, the main pot) or an unfolded seat's own total commitment. Nothing
    else may cut a dead layer, which is the amended form of "no phantom side
    pot": rule 1 still governs live money, and the amendment adds exactly one new
    legal kind of boundary.
    """

    contenders = [name for name in hand.names if name not in hand.folded]
    caps = {_ZERO} | {totals[name] for name in contenders}
    return {
        frozenset(name for name in contenders if totals[name] > cap) for cap in caps
    }


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


@given(hand=LAYERING_HANDS)
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


@given(hand=LAYERING_HANDS)
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


@given(hand=LAYERING_HANDS)
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


@given(hand=LAYERING_HANDS, policy=rake_policy())
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


@given(hand=LAYERING_HANDS)
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

    WHAT THIS TEST USED TO BE WORTH, WHICH WAS NOTHING. It compared each layer
    against its PREDECESSOR and asserted ``cause == "side"`` for every index
    above zero -- and ``_build_pots`` computed ``cause`` as ``"main" if index ==
    0 else "side"`` and sorted the ladder by decreasing eligible-set size, so
    both halves were true by construction. The set difference against the
    predecessor cannot be empty when the sort makes the predecessor no smaller
    and the merge makes no two sets equal, and the cause assertion was reading
    back the arithmetic that produced it. A mutant that mislabelled a layer could
    not be written against it. That is the same shared-premise failure the payout
    cap had, one field over.

    WHAT IT ASSERTS NOW. The comparison is against the MAIN POT rather than the
    predecessor, and the main pot is identified by the property the word names --
    the layer every other layer's eligible seats can also win -- rather than by
    being first. So the assertions are:

    * exactly one layer is the main pot, and it is the one at index 0;
    * every other layer's eligible set is a PROPER SUBSET of the main pot's, so
      it names a seat that is still in the hand, is contesting the main pot, and
      cannot win this one;
    * no two layers share an eligible set, which is what stops a boundary that
      caps nobody being emitted as two layers instead of one.

    None of those follows from the index, and the last two do not follow from the
    ordering either.
    """
    ledger = build_hand_ledger(hand.players, hand.actions)
    if not ledger.pots:
        return
    main = ledger.pots[0]
    assert main.cause == "main"
    assert main.label == "Main pot"
    assert [pot.cause for pot in ledger.pots].count("main") == 1, (
        "a ladder has exactly one main pot"
    )
    seen: list[frozenset[str]] = []
    for pot in ledger.pots[1:]:
        eligible = set(pot.eligible_players)
        capped_out = set(main.eligible_players) - eligible
        assert capped_out, f"layer {pot.index} was split off without capping anybody"
        assert eligible < set(main.eligible_players), (
            f"layer {pot.index} is contested by a seat the main pot is not"
        )
        assert pot.cause == "side"
        assert pot.label == "Side pot"
        seen.append(frozenset(eligible))
    assert len(seen) == len(set(seen)), "two layers share an eligible set"


@given(hand=LAYERING_HANDS)
@SETTINGS
def test_a_seat_that_wins_every_layer_it_can_is_paid_exactly_the_model_cap(hand):
    """THE payout property, restated for the operator's amended rule 2.

    Award every layer to one seat wherever that seat is eligible. The chips it
    collects must be, to the chip, ``_model_payout_cap``:

        its own live money back
        + the LIVE chips each opponent matched of it, capped at ITS OWN live level
        + each contributor's dead money capped at ITS OWN total commitment
        + dead money that had nowhere left to rise, if it holds the largest
          surviving total
        + external dead money, if it contests the main pot

    and a seat that contests no layer must be paid nothing. Stated as EQUALITY
    rather than as an upper bound on purpose: an inequality catches only
    overpayment, and half the disagreements the model resolved were pot layers
    holding dead money ABOVE the main pot, which UNDERPAYS the short seat the
    antes it is owed. One assertion catches both directions and, swept over every
    seat of a hand, pins every layer amount in the ladder.

    WHY THE EQUALITY IS EXACT AND NOT AN ACCIDENT. A seat is eligible for exactly
    the live bands at or below its own live level, and for exactly the dead layers
    whose cap window it exceeded. Summing the first set gives the live terms;
    summing the second gives ``sum_x min(dead[x], total[seat])``, because the
    highest dead layer the seat can reach has a cap of precisely its own total.
    Where the cascade places every dead chip before reaching that far, each
    contributor's dead is already below the last cap and the ``min`` is inert.

    WHAT THIS REPLACES, TWICE OVER.

    Round 19 replaced a cap of

        in_pot[name] + sum(min(in_pot[other], put_up[name]) for unfolded others)
        + sum(in_pot[other] for folded others)
        + sum(dead[other] for unfolded others)

    in which ``put_up`` and ``in_pot`` both included their own seat's DEAD posts.
    That was not an independent check of the reducer; it was the reducer's own
    modelling error written down a second time, and on the reported big-blind ante
    hand both came out at 66 where the model owes 58.

    Round 20 -- this one -- replaced its successor, which added the WHOLE dead
    pool for any seat contesting the main pot. That encoded unconditional rule 2,
    and the operator has amended rule 2: a seat whose entire commitment is 60
    against three 100-chip antes is owed 240, not the 360 in the pot. The old
    expression returns 360 there and would have actively REJECTED the corrected
    layering. It measured a short seat's ceiling using other seats' dead posts
    uncapped, which is the same premise the reducer held, which is why a suite of
    thirty properties agreed with five consecutive criticals.

    THE ONE FAMILY THIS DECLINES TO SCORE. A seat whose whole commitment came
    back as an uncalled bet contests the main pot (rule 3, read pre-refund) while
    its total commitment (rule 2's operand, read post-refund) is zero. The
    specification does not say which measurement point rule 2 uses, so scoring it
    either way would be encoding a guess. The reducer takes a reading and
    DISCLOSES it; ``test_a_seat_whose_whole_commitment_came_back_is_disclosed``
    is the assertion on that, and it is the reason this ``assume`` is not a hole.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, _pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, live)
    assume(not _refunded_to_nothing(hand, totals))
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
        expected = _model_payout_cap(
            name, live, hand.dead, hand.folded, hand.external_dead
        )
        if name in contests_main:
            assert paid == expected, (
                f"{name} wins every layer it is eligible for and is not paid the "
                "chips the model says it is owed"
            )
        else:
            # Folded, or in the hand having put nothing up. Every layer's eligible
            # set is contained in the main pot's, so a seat out of the main pot is
            # out of every layer.
            assert paid == _ZERO, f"{name} contests no layer and was paid {paid}"


@given(hand=LAYERING_HANDS)
@SETTINGS
def test_no_layering_can_offer_a_seat_more_than_the_table_matched_of_it(hand):
    """The cap applied to the LADDER, with no settlement in the way.

    ``test_a_seat_that_wins_every_layer_it_can_is_paid_exactly_the_model_cap``
    needs the ledger to settle: a hand whose ladder contains a layer no remaining
    seat can win never settles, and a seat whose award is rejected is skipped. So
    a layering could be wrong on exactly the hands no winner assignment happens to
    expose, and the equality would never see it.

    This one reads the ladder directly. For every unfolded seat, the layers it is
    eligible for total at most what the table matched of it. It needs no winner,
    no rake and no chop, so it covers the unsettleable hands too -- and it is the
    assertion a release gate can run over recorded hands, which is why it is
    stated separately rather than folded into the equality above.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, _pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, live)
    assume(not _refunded_to_nothing(hand, totals))
    for name in hand.names:
        if name in hand.folded:
            continue
        reachable = _sum(
            pot.amount for pot in ledger.pots if name in pot.eligible_players
        )
        assert reachable <= _model_payout_cap(
            name, live, hand.dead, hand.folded, hand.external_dead
        ), f"{name} may be declared the winner of more than the table matched of it"


@given(hand=LAYERING_HANDS)
@SETTINGS
def test_winning_every_layer_you_are_eligible_for_never_loses_chips(hand):
    """The amendment must not strand a seat's own chips above its reach.

    Worked example (b) turns on this: ``ao`` posts a 7 ante, wagers nothing live,
    wins the main pot and comes out at exactly zero. Capping a contributor's dead
    money at the smallest total in the layer means some of that contributor's own
    chips can RISE -- and if they rose into a layer their own poster cannot win,
    the poster would be guaranteed to lose money whatever it holds. That is the
    cheap implementation of the amendment (spill the risen dead into whatever live
    band sits above the cap) and it is wrong; this is the property that says so.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    for name in hand.names:
        if name in hand.folded:
            continue
        committed = hand.live[name] + hand.dead[name]
        if committed <= _ZERO:
            continue
        reachable = _sum(
            pot.amount for pot in ledger.pots if name in pot.eligible_players
        )
        refunded = Decimal(str(ledger.refunds[name]))
        assert reachable + refunded >= committed, (
            f"{name} wins every layer it is eligible for and still loses chips: "
            "some of its own money is in a layer it cannot reach"
        )


@given(hand=LAYERING_HANDS, policy=rake_policy())
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
    live, _pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, live)
    assume(not _refunded_to_nothing(hand, totals))
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
        assert paid <= _model_payout_cap(
            name, live, hand.dead, hand.folded, hand.external_dead
        ), f"{name} was paid past what the table matched of it"


@pytest.mark.parametrize("ante", [1, 2, 5])
def test_a_seat_whose_whole_commitment_came_back_is_disclosed(ante: int):
    """The one shape the cap properties decline to score must never be silent.

    Four of the properties above ``assume`` this family away, because the
    specification does not say whether rule 2's TOTAL COMMITMENT is measured
    before or after the uncalled-bet refund and encoding either reading would make
    the suite agree with the reducer by construction. An ``assume`` that removes a
    family from every check is a hole unless something else covers it. This is
    that something else: the reducer must NAME the hand rather than publish a
    number nobody has ruled on.

    Stated by construction rather than through ``forced_post_hand``. The shape is
    reachable only when the whole pot is dead money and exactly one seat wagered
    live, uncalled -- 459 of 459 generated hands missed it -- and a property that
    can never satisfy its own precondition guards nothing. Parameterised over an
    ante below, equal to and above the refunded post so the disclosure does not
    depend on which side of the cap the poster lands.
    """

    players = [LedgerPlayer(name="alice", starting_stack=ante), LedgerPlayer(name="bob", starting_stack=2)]
    actions = [
        LedgerAction(player="alice", street="preflop", kind="ante", amount=ante, is_live_post=False),
        LedgerAction(player="bob", street="preflop", kind="post_blind", amount=2),
    ]
    ledger = build_hand_ledger(players, actions)

    assert ledger.refunds["bob"] == pytest.approx(2)
    note = "\n".join(ledger.warnings)
    assert "'bob'" in note and "uncalled bet" in note, (
        "bob contests a pot it contributed nothing to and the ledger said nothing "
        "about it"
    )
    # And the reading it took is the conservative one: bob may reach only what
    # alice's post covered of what bob put up, never the whole ante.
    reachable = _sum(
        pot.amount for pot in ledger.pots if "bob" in pot.eligible_players
    )
    assert reachable == min(Decimal(ante), Decimal(2))


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


# --- The ladder itself: where a boundary may be cut, and on what --------------
#
# Round 18 established that unequal dead money is not a layer boundary and that
# every cut is a clean threshold on LIVE contribution. Round 20's amendment to
# rule 2 adds exactly ONE new legal kind of boundary -- a cap on a contributor's
# dead money at the smallest TOTAL commitment among the layer's eligible seats --
# and nothing else. These properties are the round-18 statements re-derived for
# the two-ladder model, and the first of them is written so that the family where
# the old rule and the new rule COINCIDE still gets the old assertion to the chip.


@given(hand=LAYERING_HANDS)
@SETTINGS
def test_unequal_dead_money_alone_never_splits_the_pot(hand):
    """The phantom side pot, still forbidden where the amendment does not bite.

    Antes and dead blinds are owed to the table, not wagered at anybody, so a seat
    that owes more of them is not a seat anyone declined to match. Cutting the pot
    at every distinct TOTAL commitment made a side pot out of exactly that
    difference -- one seat's 5 ante against another's 3 dead blind, everybody
    matching the same 20 live, nobody all-in -- and then refused the seat that won
    the hand the chips it had won, because it was "not eligible for pot 1". That
    is the operator's worked example (d) and it is a single pot of 88.

    WHAT THE AMENDMENT CHANGED ABOUT THIS PROPERTY, AND WHAT IT DID NOT. The
    amendment lifts a forced post's excess over the smallest total commitment into
    a layer of its own, so unequal dead money CAN now open a boundary -- but only
    when some post exceeds that floor. In (d) the posts are 5 and 3 against a
    floor of 20, so nothing rises and the answer is unchanged. The guard is
    therefore conditioned on ``_amendment_bites``, which is read off the
    specification sentence and not off the ladder: where the two rules coincide,
    the round-18 statement must still hold exactly, and weakening it to "the
    amendment might have done something" would retire the property that forbids
    the phantom.

    Counted over CONTESTABLE layers rather than over all of them. A line in which
    every seat that wagered above the line folded leaves chips in a band no
    remaining seat is eligible for; the model strands that band rather than
    merging it down into a pot a short seat could win.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    settled_live, _pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, settled_live)
    assume(not _amendment_bites(hand, settled_live, totals))
    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)
    line = max(settled_live[name] for name in contenders)
    if all(settled_live[name] == line for name in contenders):
        contestable = [pot for pot in ledger.pots if pot.eligible_players]
        assert len(contestable) == 1, (
            "every seat still in the hand covered the same live wager and no "
            "forced post exceeded the smallest total commitment, so there is "
            "nothing for a second layer to hold apart"
        )


@given(hand=LAYERING_HANDS)
@SETTINGS
def test_every_layer_boundary_is_a_cut_the_two_rules_allow(hand):
    """WHERE a boundary may be drawn, and on WHAT. The amended form.

    This is the property that forbids the phantom side pot, and its currency is
    the whole point. It used to assert the clean cut over each seat's TOTAL
    commitment --

        max(put_up[dropped]) <= min(put_up[kept])

    -- which is the reducer's round-18 modelling error stated as an invariant: it
    is satisfied by a boundary cut at a seat's live-plus-ante total and by nothing
    else. Round 19 replaced it with "every boundary is a threshold on LIVE
    contribution", which is what rule 1 says and was right while rule 2 was
    unconditional. Rule 2 is no longer unconditional, so there are now exactly TWO
    legal kinds of cut and this states both:

    * a LIVE band's eligible set is ``{unfolded : live >= level}`` for a live
      contribution level of some seat (rule 1 with rule 3's first sentence);
    * a DEAD layer's eligible set is ``{unfolded : total > cap}`` for a cap that is
      zero or an unfolded seat's own total commitment (rule 2's second sentence,
      which at cap zero reduces to rule 3's second sentence).

    Nothing else may cut a layer. A cut at a folded seat's total, at a
    live-plus-dead total used as a LIVE level, or at any figure no seat holds,
    fails this.

    WHAT IS DELIBERATELY NO LONGER ASSERTED. The old property required each layer
    to strictly narrow the one below it. Under the amendment the ladder does not
    nest: a dead layer's eligible set is a cut on TOTAL and a live band's is a cut
    on LIVE, and neither need contain the other. ``A`` with a 100 ante and no live
    money, ``B`` live 100, ``C`` live 40 lays out as 40 {A,B,C} / 80 {B,C} /
    60 {A,B} / 60 {B}, in which {B,C} and {A,B} do not nest. Retaining the chain
    would REJECT the corrected model, which is precisely how the previous property
    suite came to agree with five consecutive criticals. What survives is what is
    still true and still load-bearing: every layer is contained in the main pot,
    no two layers share an eligible set, and each one after the first excludes a
    seat that could win the one below.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, _pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, live)
    # Rule 3 reads "put ANY chip up" before the refund and rule 2's cap reads a
    # total commitment after it, so a seat refunded to nothing is legally in the
    # main pot while belonging to no total cut. That shape is disclosed rather
    # than ruled on; see ``_refunded_to_nothing``.
    assume(not _refunded_to_nothing(hand, totals))
    legal = _live_threshold_sets(hand, live) | _total_cut_sets(hand, totals)

    seen: set[frozenset[str]] = set()
    main = frozenset(ledger.pots[0].eligible_players)
    for pot in ledger.pots:
        eligible = frozenset(pot.eligible_players)
        if not eligible:
            # A band every seat that reached it folded out of. Stranded, never
            # merged down; ``test_a_layer_no_remaining_seat_can_win_is_stranded``
            # in the model suite is the guarantee on that.
            continue
        assert eligible in legal, (
            f"layer {pot.index} is eligible to {sorted(eligible)}, which is "
            "neither a threshold on live contribution nor a cut on total "
            "commitment"
        )
        assert eligible <= main, (
            f"layer {pot.index} is contestable by a seat the main pot is not"
        )
        assert eligible not in seen, (
            f"layer {pot.index} repeats an eligible set: two layers no settlement "
            "can tell apart were emitted separately"
        )
        seen.add(eligible)
    for lower, upper in zip(ledger.pots, ledger.pots[1:], strict=False):
        if not upper.eligible_players:
            continue
        assert set(lower.eligible_players) - set(upper.eligible_players), (
            f"layer {upper.index} was opened without capping anybody out of the "
            "layer below it"
        )


@given(hand=LAYERING_HANDS)
@SETTINGS
def test_dead_money_starts_in_the_lowest_layer_and_rises_only_when_capped(hand):
    """Rule 2 as amended, stated so a leak in either direction is caught.

    The previous form of this property asserted that the main pot is never smaller
    than the WHOLE dead pool, because rule 2 was unconditional. That is now false
    by design -- worked example (e) has a dead pool of 360 and a main pot of 240 --
    and asserting it would reject the operator's ruling. What replaces it is the
    amended sentence, in both directions:

    * NOTHING RISES THAT WAS NOT CAPPED. Each contributor's dead money reaches the
      main pot up to the smallest total commitment among the seats eligible for
      it, so the main pot holds at least ``sum_x min(dead[x], floor)`` plus all the
      external dead money. When the amendment does not bite this is the whole dead
      pool, which is exactly the round-19 statement, so that guarantee is kept
      rather than dropped.
    * NOTHING RISES FURTHER THAN IT WAS CAPPED. Everything above the main pot is
      live money above the main pot's live ceiling, plus dead money that genuinely
      exceeded the floor -- never more.

    The second bound is what catches "carry the antes up into a side pot the short
    seat cannot win", which underpaid the short seat on 58.6% of the round-19
    disagreements and was completely silent, because chip conservation still held.
    """
    dead_money = float(hand.external_dead)
    ledger = build_hand_ledger(hand.players, hand.actions, dead_money=dead_money)
    assume(ledger.pots)
    live, pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, live)
    contests_main = _model_main_pot_eligibility(hand)
    assume(contests_main)
    assume(not _refunded_to_nothing(hand, totals))
    floor = min(totals[name] for name in contests_main)

    stays_down = (
        _sum(min(hand.dead[name], floor) for name in hand.names) + hand.external_dead
    )
    assert Decimal(str(ledger.pots[0].amount)) >= stays_down, (
        "the main pot holds less dead money than the cap leaves in it, so a "
        "forced post no seat was capped out of leaked into a layer above"
    )
    if not _amendment_bites(hand, live, totals):
        assert Decimal(str(ledger.pots[0].amount)) >= pool, (
            "no forced post exceeded the smallest total commitment, so the whole "
            "dead pool is owed to the lowest layer"
        )

    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)
    ceiling = max(
        [_ZERO]
        + [
            level
            for level in {live[name] for name in hand.names if live[name] > 0}
            if {name for name in contenders if live[name] >= level} == contests_main
        ]
    )
    rises = _sum(max(_ZERO, hand.dead[name] - floor) for name in hand.names)
    above = _sum(pot.amount for pot in ledger.pots[1:])
    assert above <= _sum(
        max(_ZERO, live[name] - ceiling) for name in hand.names
    ) + rises, (
        "the layers above the main pot hold more than the live money above the "
        "main pot's ceiling plus the dead money the cap lifted out of it"
    )


@given(hand=LAYERING_HANDS)
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
    live, _pool = _model_figures(hand, ledger)
    totals = _model_totals(hand, live)
    assume(not _refunded_to_nothing(hand, totals))
    contenders = [name for name in hand.names if name not in hand.folded]
    assume(contenders)
    line = max(live[name] for name in contenders)

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
            name, live, hand.dead, hand.folded, hand.external_dead
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
