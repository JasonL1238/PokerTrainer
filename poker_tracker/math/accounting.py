"""Authoritative chip accounting for completed poker hands.

The reducer in this module deliberately works with normalized, incremental
commitments.  A raise of 8 BB means eight additional chips entered the pot;
importers that receive "raise to" amounts must normalize them first.

The ledger does not evaluate cards or choose winners.  Settlement is explicit:
callers provide the ordered winner(s) for each generated pot layer.  That keeps
chip conservation independently testable and prevents incomplete histories
from silently inventing outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Literal, Protocol

LedgerStreet = Literal["preflop", "flop", "turn", "river", "showdown"]
LedgerActionKind = Literal[
    "ante",
    "post_blind",
    "bet",
    "call",
    "raise",
    "all-in",
    "fold",
    "check",
    "show",
    "win",
]

PotLayerCause = Literal["main", "side"]

_COMMITMENT_KINDS = {"ante", "post_blind", "bet", "call", "raise", "all-in"}
_BETTING_COMMITMENT_KINDS = _COMMITMENT_KINDS - {"ante"}
_NON_COMMITMENT_KINDS = {"fold", "check", "show", "win"}
# Forced-bet names that identify a STRUCTURAL LIVE forced bet: one whose size is
# set by the room and which therefore sets what every other seat owes. Antes,
# big-blind antes and dead blinds are excluded on purpose -- they are owed to the
# table and establish no wager level, so a short one says nothing about
# ``to_call`` and must not raise the blind-structure blocker.
#
# This set exists because the action KIND is not a reliable name for "this was a
# forced post". A recording that books a blind which took its poster's last chip
# as ``all-in`` describes the same event, and keying only on ``post_blind`` let
# exactly that recording escape the refusal below -- the reported defect, one
# relabel away. Where a recording states the forced-bet type, that statement is
# what is read.
_LIVE_STRUCTURAL_FORCED_BETS = frozenset(
    {"small_blind", "big_blind", "straddle", "bring_in"}
)
_FORCED_POST_KINDS = {"ante", "post_blind"}
_FLOP_STREETS = {"flop", "turn", "river", "showdown"}
_ZERO = Decimal("0")
# The coarsest denomination a chopped pot may be divided at. A whole chip is the
# indivisible unit a real table deals in; anything above it is a claim about the
# room that the hand's own action line cannot demonstrate, and it was exactly
# that claim -- taken verbatim from a declared field -- that let "Chip unit"
# redirect a chop. See _split_granularity.
_MAX_SPLIT_QUANTUM = Decimal("1")
# The one place a layer's operator-facing name is written. Every consumer reads
# it through ``PotLayer.label`` so a layer cannot be renamed in one surface and
# not another, and so no surface can invent a name from the layer's index.
_POT_LAYER_LABELS: dict[PotLayerCause, str] = {
    "main": "Main pot",
    "side": "Side pot",
}


class LedgerError(ValueError):
    """Raised when a completed-hand ledger is structurally impossible."""


class PlayerRecord(Protocol):
    player_key: str
    player_name: str
    starting_stack: float | None
    seat_index: int | None


class ActionRecord(Protocol):
    player_key: str | None
    player_name: str
    street: str
    action_type: str
    amount: float | None
    amount_semantics: str
    is_live_post: bool | None
    forced_bet_type: str | None
    pot_before: float | None


@dataclass(frozen=True)
class LedgerPlayer:
    name: str
    starting_stack: float
    seat: int | None = None
    key: str | None = None


@dataclass(frozen=True)
class LedgerAction:
    player: str
    street: LedgerStreet
    kind: LedgerActionKind
    amount: float = 0
    is_live_post: bool = True
    # What the recording says this commitment WAS, when it says anything: one of
    # ``models.ForcedBetType`` or None. It is not a second copy of ``kind`` --
    # a recording is free to book a forced post that took its poster's last chip
    # under ``all-in``, and the CV spine and the hand editor both can. The blind
    # structure refusal reads this so a forced post stays identifiable when the
    # kind stops naming it. Defaults to None, which means "the recording said
    # nothing", so every existing construction keeps its exact behaviour.
    forced_bet_type: str | None = None


@dataclass(frozen=True)
class BlindStructure:
    """The room's structural forced-bet sizes, in the same chip unit as the actions.

    A fact about the TABLE, not about the line -- the same category as
    ``RakePolicy`` and ``dead_money``, and supplied the same way, because the
    action line cannot demonstrate it.

    Why it has to be an input at all.  ``to_call`` used to be derived purely
    from the largest observed contribution on the street.  That is exact
    whenever every forced post was made in full, and silently wrong the moment
    one was not: blinds 5/10 with a big blind all-in for 4 leaves 5 (the small
    blind) as the largest observed post, so the reducer told the button that
    calling 10 was illegal and that the amount to call was 5.  An operator who
    obeyed the product's own error message entered 5, and the hand reconciled --
    balanced, legal, authoritative, warning-free -- around a 14-chip pot whose
    truth is 24.  A short all-in blind does not lower what anybody else owes.

    Why it cannot be inferred.  A small blind of 5 is equally consistent with
    5/10 and with 5/5; real structures include 1/3, 2/2, 2/5 and 5/10, and a
    straddle is whatever the room permits.  Guessing produces a different wrong
    answer rather than no answer, so nothing here guesses.

    ``small_blind`` is deliberately carried even though only the largest forced
    bet moves ``to_call``: it is half of what an operator means by "5/10", it is
    what makes the stored declaration checkable by a human, and validating it
    against the big blind catches a transposed entry that would otherwise pass
    silently. ``None`` means it was not stated, which is honest and costs
    nothing -- it is not written as 0, because 0 is a claim.

    ``straddles`` is ordered from the first straddle outward, each strictly
    larger than the forced bet before it, so a re-straddle is expressible and
    the largest live forced bet is simply the last of them.

    Field order is ``small_blind`` then ``big_blind`` so that a positional
    ``BlindStructure(5, 10)`` reads as the "5/10" an operator would say. The
    validator refuses a transposition, so the ordering trap cannot fail
    silently either way.
    """

    small_blind: float | None
    big_blind: float
    straddles: tuple[float, ...] = ()


@dataclass(frozen=True)
class RakePolicy:
    """Rake assumptions in the same chip unit as the actions.

    Rake is rounded down to ``rounding_unit``.  Rooms differ in their exact
    rounding and eligibility rules, so callers must store the policy used with
    any derived result instead of treating this default as universal.
    """

    rate: float = 0
    cap: float | None = None
    rounding_unit: float = 0.01
    no_flop_no_drop: bool = False


@dataclass(frozen=True)
class ActionSnapshot:
    index: int
    player: str
    street: LedgerStreet
    kind: LedgerActionKind
    amount: float
    pot_before: float
    pot_after: float
    stack_before: float
    stack_after: float
    to_call_before: float
    call_increment: float
    street_contribution_after: float
    hand_contribution_after: float
    effective_stack_before: float
    effective_stack_range_before: tuple[float, float]
    spr_before: float | None
    spr_range_before: tuple[float, float] | None
    active_players: tuple[str, ...]


@dataclass(frozen=True)
class PotLayer:
    index: int
    amount: float
    net_amount: float
    rake: float
    contributors: tuple[str, ...]
    eligible_players: tuple[str, ...]
    winners: tuple[str, ...] = ()
    cause: PotLayerCause = "main"

    @property
    def label(self) -> str:
        """What this layer is, derived from what created it and not from its index.

        A side pot is one specific thing: a layer that exists because a player
        still in the hand did not cover the live wagering, so that player cannot
        win it.  Almost always that means all-in for less; on a recorded line
        that stops mid-wager it can also mean a seat whose answer was never
        recorded, which ``hand_accounting._unanswered_wager_issues`` refuses
        separately.  Either way the layer names a seat that cannot win it, which
        is what the word means.  Every layer above the first used to be a side pot,
        which is a false statement about the ordinary hands that produce a
        boundary for other reasons -- a blind that folds for less, an unmatched
        ante -- and reading it teaches an operator studying their own hand the
        wrong definition of the term.  Naming those layers "dead money" instead
        was the same mistake pointing the other way: a layer split off by a
        folded small blind holds nothing but live wagering between the players
        who stayed.

        ``_build_pots`` emits a boundary in exactly two places, and both of them
        name a seat still in the hand that cannot win the layer above: a LIVE
        contribution level a seat failed to reach, and -- under the operator's
        amended rule 2 -- a dead-money cap a seat's TOTAL commitment failed to
        reach.  So every layer after the first IS a side pot and this mapping has
        no third case to name.

        AND IT IS NOW ACTUALLY DERIVED.  ``cause`` used to be computed as
        ``"main" if index == 0 else "side"`` -- a fact about the list, under a
        docstring claiming to state a fact about the layer.  The two agree only
        while ``_build_pots`` emits the widest eligible set first, which nothing
        checked, so any reordering of the ladder relabelled the pots and every
        consumer believed the new labels.  The main pot is now identified as the
        layer whose eligible set contains every other layer's, which is what the
        word means and which a reordering cannot fake.

        WHAT THE AMENDMENT CHANGED ABOUT THIS SENTENCE.  It used to read "no
        layer above the main pot ever holds a forced post".  That is no longer
        true and must not be relied on: a forced post larger than the smallest
        total commitment in the layer it started in has its excess lifted into a
        layer above, and that layer can hold nothing but dead money.  Worked
        example (e) is exactly it -- a 60-chip seat against three 100-chip antes
        lays out as 240 everyone may win and 120 only the three anteing seats
        may.  The main pot itself may still be nothing but forced posts -- that is
        what a table with a stack all-in for its ante produces -- and it is still
        the main pot.
        """

        return _POT_LAYER_LABELS[self.cause]


@dataclass(frozen=True)
class HandLedger:
    contributions: dict[str, float]
    refunds: dict[str, float]
    payouts: dict[str, float]
    net_results: dict[str, float]
    gross_pot: float
    rake: float
    net_pot: float
    pots: tuple[PotLayer, ...]
    snapshots: tuple[ActionSnapshot, ...]
    folded_players: tuple[str, ...]
    warnings: tuple[str, ...]
    legality_issues: tuple[str, ...]
    is_settled: bool
    is_balanced: bool
    is_legal: bool


def build_hand_ledger(
    players: Sequence[LedgerPlayer],
    actions: Sequence[LedgerAction],
    winners: Mapping[int, Sequence[str]] | None = None,
    *,
    rake: RakePolicy | None = None,
    odd_chip_order: Sequence[str] = (),
    dead_money: float = 0,
    flop_seen: bool | None = None,
    blinds: BlindStructure | None = None,
) -> HandLedger:
    """Reduce normalized completed-hand actions into pots and player results.

    ``winners`` maps each generated pot index to one or more ordered winners.
    Pot 0 is the main pot; a later layer is generated only where a player still
    in the hand cannot win it, which is what makes it a side pot, and
    ``PotLayer.cause`` records that.  LIVE boundaries are cut at live
    contribution levels only: unequal dead money -- one seat's ante against
    another's dead blind, a button ante -- never opens a live band, because no
    opponent can decline a forced post, and a short seat's own dead posts never
    raise the level its opponents are charged into the main pot at.  Dead money
    starts in the lowest layer and, under the operator's amended rule 2, the part
    of a forced post above the smallest TOTAL commitment among that layer's
    eligible seats rises into a layer of its own, eligible to the seats whose own
    total reached past the cap.  Those two ladders do not nest in general, so the
    eligible sets are ordered widest-first rather than chained.  See
    ``_build_pots`` for the model in full.  Omitting winners returns a useful but
    explicitly unsettled ledger. ``flop_seen`` is an
    optional completed-hand fact for histories where a board ran out without
    any postflop action (for example, a preflop all-in). When omitted, the
    ledger preserves the historic behavior of inferring it from action streets.

    ``blinds`` is the room's structural forced-bet sizes (see ``BlindStructure``).
    Preflop, the live wagering starts at the largest structural forced bet
    whether or not anybody posted it in full, so a blind or straddle that is
    all-in for less never lowers what the rest of the table owes.  It is a
    FLOOR and never a ceiling: it is combined with the observed street maximum
    by ``max``, so a declared structure can only ever raise the amount to call
    and can never excuse an under-call the recording demonstrates.

    Omitting ``blinds`` is not a silent fall back to the observed maximum.  A
    live forced post that leaves its poster all-in above the declared floor --
    which is every such post when nothing is declared, since the floor is then
    zero -- does not demonstrate the size of the forced bet it was paying, so it
    is reported as a legality issue naming the seat and the clearing action.
    The hand stays fully inspectable and every chip figure is still derived; it
    simply may not present as legal, reconciled, or study-ready until the
    structure is declared.  A hand whose forced posts were all made in full says
    nothing about a structure it does not need, so it is untouched.

    THE SCOPE OF THAT REFUSAL, STATED HONESTLY.  It reaches a forced post the
    RECORDING IDENTIFIES as one -- by ``kind == "post_blind"``, or by a
    ``forced_bet_type`` naming a live structural bet on a row booked under
    another kind (see ``_is_live_structural_post``).  A recording that books a
    short blind as a plain ``all-in`` and states no forced-bet type has said
    nothing that distinguishes it from an ordinary short shove, and nothing here
    can tell them apart; such a hand still derives ``to_call`` from the observed
    maximum and is NOT refused.  The CV reconstruction spine emits exactly that
    shape for a seat whose stack reads zero, so this refusal does not cover
    every reconstructed hand.  Declaring the structure covers those hands
    correctly -- the floor is applied whatever the kinds are -- but the operator
    is not prompted to.
    """

    player_order, starting = _validate_players(players)
    dead = _decimal(dead_money, "dead_money")
    if dead < 0:
        raise LedgerError("dead_money must not be negative.")
    policy = rake or RakePolicy()
    rate, cap, unit = _validate_rake(policy)
    # The largest structural live forced bet, or zero when nothing is declared.
    # Zero is not "the observed maximum": it is what makes every all-in forced
    # post uninterpretable below, which is the loud half of the contract.
    preflop_floor = _validate_blinds(blinds)
    winner_map = {index: tuple(names) for index, names in (winners or {}).items()}
    odd_order = _validate_odd_chip_order(odd_chip_order, starting)

    contributions = {name: _ZERO for name in player_order}
    # Split the same chips two ways. ``live_contributions`` buys a place in a pot
    # layer and is what an uncalled bet is measured against; ``dead_contributions``
    # (antes, dead blinds) is owed to the table and joins the main pot whole.
    # Measuring refunds against the total instead hands back an unmatched ante as
    # though it were an overbet, and removes it from the pot entirely.
    live_contributions = {name: _ZERO for name in player_order}
    dead_contributions = {name: _ZERO for name in player_order}
    street_contributions = {name: _ZERO for name in player_order}
    current_street: LedgerStreet | None = None
    folded: set[str] = set()
    all_in: set[str] = set()
    snapshots: list[ActionSnapshot] = []
    warnings: list[str] = []
    legality_issues: list[str] = []
    street_order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
    last_street_rank = -1
    last_full_raise = _ZERO
    last_wager_faced: dict[str, Decimal] = {}
    # Decided over the WHOLE recording before a single row is reduced, so the
    # verdict cannot depend on the order two forced rows happen to be listed in,
    # and so every preflop complaint that names a wager level is silenced --
    # including any that precede the post. See ``_unreadable_forced_posts``.
    unreadable_posts = _unreadable_forced_posts(actions, starting, preflop_floor)
    # True once this ledger has stated it cannot determine the preflop wager
    # level, from which point it must stop making claims that depend on one.
    structure_unreadable = bool(unreadable_posts)

    for index, action in enumerate(actions):
        if action.player not in starting:
            raise LedgerError(f"Action {index + 1} references unknown player {action.player!r}.")
        if action.player in folded:
            raise LedgerError(f"Player {action.player!r} acts after folding.")
        if action.player in all_in and action.kind not in {"show", "win"}:
            legality_issues.append(
                f"Action {index + 1}: player {action.player!r} acts after being all-in."
            )
        if action.kind not in _COMMITMENT_KINDS | _NON_COMMITMENT_KINDS:
            raise LedgerError(f"Unsupported action kind {action.kind!r}.")
        if action.street not in street_order:
            raise LedgerError(f"Action {index + 1} has unsupported street {action.street!r}.")
        amount = _decimal(action.amount, f"action {index + 1} amount")
        if amount < 0:
            raise LedgerError("Action amounts must not be negative.")
        if action.kind in _COMMITMENT_KINDS and amount <= 0:
            raise LedgerError(f"{action.kind} requires a positive incremental amount.")
        if action.kind in _NON_COMMITMENT_KINDS and amount != 0:
            raise LedgerError(f"{action.kind} cannot commit chips.")

        if action.street != current_street:
            rank = street_order[action.street]
            if rank < last_street_rank:
                legality_issues.append(
                    f"Action {index + 1}: street order moves backward to {action.street}."
                )
            last_street_rank = max(last_street_rank, rank)
            current_street = action.street
            street_contributions = {name: _ZERO for name in player_order}
            # Preflop the minimum full raise starts at the structural forced bet,
            # for the same reason the amount to call does: a big blind all-in for
            # 4 in a 5/10 game does not make 5 a legal raise size.
            last_full_raise = (
                preflop_floor if action.street == "preflop" else _ZERO
            )
            last_wager_faced = {}

        committed_before = contributions[action.player]
        stack_before = starting[action.player] - committed_before
        if amount > stack_before:
            raise LedgerError(
                f"Player {action.player!r} commits {amount} with only {stack_before} remaining."
            )

        active_before = tuple(name for name in player_order if name not in folded)
        actionable_before = tuple(
            name for name in active_before if name not in all_in
        )
        opponent_effective = [
            min(stack_before, starting[name] - contributions[name])
            for name in actionable_before
            if name != action.player
        ]
        if opponent_effective:
            effective_low = min(opponent_effective)
            effective_high = max(opponent_effective)
        else:
            effective_low = effective_high = stack_before
        effective_before = effective_low
        # The floor answers "what does this seat owe to keep playing", which is a
        # question only a VOLUNTARY action faces. A forced post is not answering
        # a wager -- its size is set by the room and no legality check here reads
        # ``to_call`` for one -- so applying the floor to it would move nothing
        # except the ``to_call_before`` a post's own snapshot reports. Leaving
        # posts out is what makes a hand whose blinds were all made in full
        # derive BYTE-IDENTICALLY with and without a declared structure, right
        # down to every snapshot, which is the property the migration rests on.
        wager_floor = (
            preflop_floor
            if action.street == "preflop" and not _is_forced_post(action)
            else _ZERO
        )
        # The live wagering in force: what the street has demonstrably seen, or
        # the structural forced bet, whichever is larger. Taking the observed
        # maximum ALONE is the defect this floor exists for; taking the declared
        # floor alone would let a declaration excuse an under-call the recording
        # proves. Only ``max`` has neither failure.
        betting_max = max(
            max(street_contributions.values(), default=_ZERO), wager_floor
        )
        player_street_before = street_contributions[action.player]
        to_call = max(_ZERO, betting_max - player_street_before)
        pot_before = dead + sum(contributions.values(), _ZERO)
        new_total = player_street_before + amount
        aggressive_increment = (
            max(_ZERO, new_total - betting_max)
            if action.kind in {"raise", "all-in"}
            else _ZERO
        )

        action_label = f"Action {index + 1}"
        # Every complaint in this block is a claim about the wager level in
        # force. Once an unreadable forced post has been seen, the preflop wager
        # level is exactly what this ledger has just said it cannot determine, so
        # stating one would contradict the blocker it raised -- and that
        # contradiction is not academic: the reported defect is an operator who
        # obeyed "the amount to call is 5" and entered a 14-chip pot whose truth
        # was 24. Nothing is lost by staying quiet. ``is_legal`` is already False
        # from the blocker, so the hand is blocked either way, and declaring the
        # structure brings every one of these checks back with a level it can
        # defend. The stack-based ``all-in`` check below is deliberately outside
        # the guard: it reads no wager level.
        #
        # ``structure_unreadable`` is set later in this same loop, when the post
        # is reached. Forced posts precede voluntary action on every recording a
        # room can produce, so the flag is always set before the actions it
        # silences; a malformed line that posts a blind after a call would report
        # that call against the pre-post level, which is no worse than today.
        if not (structure_unreadable and action.street == "preflop"):
            if action.kind == "check" and to_call > 0:
                legality_issues.append(f"{action_label}: check while facing {to_call}.")
            if action.kind == "call":
                if to_call <= 0:
                    legality_issues.append(f"{action_label}: call with nothing to call.")
                elif amount != to_call and not (
                    amount == stack_before and amount < to_call
                ):
                    legality_issues.append(
                        f"{action_label}: call commits {amount}, but the amount to "
                        f"call is {to_call}."
                    )
            if action.kind == "bet" and betting_max > 0:
                legality_issues.append(
                    f"{action_label}: bet used while facing an existing wager."
                )
            if action.kind == "raise":
                raise_size = new_total - betting_max
                if to_call <= 0 or raise_size <= 0:
                    legality_issues.append(
                        f"{action_label}: raise does not increase the wager."
                    )
                elif last_full_raise > 0 and raise_size < last_full_raise:
                    legality_issues.append(
                        f"{action_label}: raise size {raise_size} is below the minimum "
                        f"full raise {last_full_raise}."
                    )
            if (
                aggressive_increment > 0
                and action.player in last_wager_faced
                and last_full_raise > 0
                and betting_max - last_wager_faced[action.player] < last_full_raise
            ):
                legality_issues.append(
                    f"{action_label}: betting was not reopened after a short all-in raise."
                )
        if action.kind == "all-in" and amount != stack_before:
            legality_issues.append(
                f"{action_label}: all-in commits {amount}, but {stack_before} remains."
            )

        if action.kind in _COMMITMENT_KINDS:
            contributions[action.player] += amount
            is_live_bet = _is_live_money(action)
            # Live money buys a place in a pot layer and can come back as an
            # uncalled bet. Dead money -- antes and dead blinds -- is owed to the
            # table: it joins the main pot whole and is never returnable.
            if is_live_bet:
                live_contributions[action.player] += amount
            else:
                dead_contributions[action.player] += amount
            if is_live_bet:
                street_contributions[action.player] += amount
                new_max = max(street_contributions.values(), default=_ZERO)
                if action.kind == "post_blind" and new_max > betting_max:
                    last_full_raise = max(last_full_raise, new_max)
                elif action.kind == "bet" and new_max > betting_max:
                    last_full_raise = new_max
                elif action.kind == "raise" and new_max > betting_max:
                    raise_size = new_max - betting_max
                    if raise_size >= last_full_raise:
                        last_full_raise = raise_size
                elif action.kind == "all-in" and new_max > betting_max:
                    raise_size = new_max - betting_max
                    if last_full_raise == 0 or raise_size >= last_full_raise:
                        last_full_raise = raise_size
        if action.kind not in {"ante", "post_blind", "show", "win"}:
            last_wager_faced[action.player] = max(
                max(street_contributions.values(), default=_ZERO), wager_floor
            )
        if action.kind == "fold":
            folded.add(action.player)
        if contributions[action.player] == starting[action.player]:
            all_in.add(action.player)
        if index in unreadable_posts:
            # A forced post that took the poster's last chip proves only that the
            # poster ran out; it does not say what the room required. Above the
            # declared floor there is nothing left to read it against -- with no
            # declaration the floor is zero, so EVERY such post lands here, which
            # is the intended loud default. The ledger refuses to name an amount
            # to call rather than inventing one from the largest post it happens
            # to see. See ``BlindStructure`` and ``_unreadable_forced_posts``.
            legality_issues.append(
                f"{action_label}: {action.player!r} is all-in posting a live "
                f"forced bet of {amount}, which the declared blind structure "
                f"does not cover ({_blind_floor_text(preflop_floor)}). A short "
                "forced post does not demonstrate the structural size it was "
                "paying, so the amount every other seat owes cannot be read off "
                "this hand. Declare the blind structure (small blind, big blind, "
                "any straddle) for this hand."
            )

        pot_after = dead + sum(contributions.values(), _ZERO)
        call_increment = min(amount, to_call) if action.kind in {"call", "all-in"} else _ZERO
        snapshots.append(
            ActionSnapshot(
                index=index,
                player=action.player,
                street=action.street,
                kind=action.kind,
                amount=_float(amount),
                pot_before=_float(pot_before),
                pot_after=_float(pot_after),
                stack_before=_float(stack_before),
                stack_after=_float(starting[action.player] - contributions[action.player]),
                to_call_before=_float(to_call),
                call_increment=_float(call_increment),
                street_contribution_after=_float(street_contributions[action.player]),
                hand_contribution_after=_float(contributions[action.player]),
                effective_stack_before=_float(effective_before),
                effective_stack_range_before=(
                    _float(effective_low),
                    _float(effective_high),
                ),
                spr_before=(
                    _float(effective_before / pot_before) if pot_before > 0 else None
                ),
                spr_range_before=(
                    (
                        _float(effective_low / pot_before),
                        _float(effective_high / pot_before),
                    )
                    if pot_before > 0
                    else None
                ),
                active_players=active_before,
            )
        )

    # Only live money can be uncalled. An ante nobody matched is not an overbet.
    refunds = _uncalled_refunds(player_order, live_contributions)
    settled_contributions = {
        name: live_contributions[name] - refunds[name] for name in player_order
    }
    # The two figures decide two different questions and must stay apart.
    # ``settled_contributions`` is LIVE money that stuck: it is what a seat chose
    # to wager, so it is the only thing that cuts a LIVE boundary and the only
    # thing that decides who may contest a live band. ``dead_contributions`` is
    # owed to the table: it starts in the LOWEST layer, and under the amended
    # rule 2 the part of it above the smallest total commitment among that
    # layer's eligible seats rises into a layer of its own, eligible by total.
    # ``dead`` -- the UNATTRIBUTED dead money -- has no contributor to cap it
    # against, so it stays in the lowest layer whole and buys its declarer
    # nothing, because no seat wagered it.
    raw_pots = _build_pots(
        player_order,
        settled_contributions,
        dead_contributions,
        contributions,
        folded,
        dead,
    )
    for pot in raw_pots:
        if not pot["eligible_players"]:
            # Reachable only from a line where every seat that reached the top
            # live level folded, which no legal hand produces but a truncated
            # recording can. The chips stay in the layer they reached: merging
            # them DOWN would hand a seat live money no opponent it faced ever
            # matched, which is the same overpayment every repair in this module
            # has been chasing. Naming it costs the hand ``is_legal`` and leaves
            # it permanently unsettled, which is the loud half of the contract.
            legality_issues.append(
                f"Pot {pot['index']} holds {_float(pot['amount'])} chips that no "
                "player still in the hand can win: every seat that wagered at "
                "that level folded. The recording is incomplete or the fold rows "
                "are misordered; this hand cannot be settled as recorded."
            )
    warnings.extend(
        _unruled_dead_money_warnings(
            player_order,
            settled_contributions,
            dead_contributions,
            contributions,
            folded,
            dead,
        )
    )
    _validate_winners(winner_map, raw_pots, starting, folded)

    gross = sum((pot["amount"] for pot in raw_pots), _ZERO)
    observed_flop = (
        any(action.street in _FLOP_STREETS for action in actions)
        if flop_seen is None
        else flop_seen
    )
    rake_total = _compute_rake(
        gross,
        rate=rate,
        cap=cap,
        unit=unit,
        waived=policy.no_flop_no_drop and not observed_flop,
    )
    rake_by_pot = _allocate_rake(raw_pots, rake_total, unit)
    payouts = {name: _ZERO for name in player_order}
    rendered_pots: list[PotLayer] = []
    # Derived from the observed action line only. `unit` -- the declared "Chip
    # unit" -- is deliberately NOT passed and neither is anything computed from
    # it: it rounds the rake and nothing else, so the granularity a chopped pot
    # is divided at is the same at every declared value, at every rake rate.
    # Each dead contribution individually: summing four 0.25 antes into 1.00
    # destroys the hundredths the hand demonstrably deals in, and coarsening
    # the quantum turns an exactly-even chop into an odd one.
    split_unit = _split_granularity(
        [*settled_contributions.values(), *dead_contributions.values()], dead
    )

    for index, pot in enumerate(raw_pots):
        pot_winners = winner_map.get(index, ())
        pot_rake = rake_by_pot[index]
        net_amount = pot["amount"] - pot_rake
        if pot_winners:
            _split_pot(
                net_amount,
                pot_winners,
                payouts,
                unit=split_unit,
                odd_order=odd_order,
            )
        rendered_pots.append(
            PotLayer(
                index=index,
                amount=_float(pot["amount"]),
                net_amount=_float(net_amount),
                rake=_float(pot_rake),
                contributors=pot["contributors"],
                eligible_players=pot["eligible_players"],
                winners=pot_winners,
                cause=pot["cause"],
            )
        )

    is_settled = bool(raw_pots) and len(winner_map) == len(raw_pots)
    if raw_pots and not is_settled:
        warnings.append("Ledger is unsettled because one or more pots have no declared winner.")
    if not raw_pots:
        warnings.append("Ledger has no contestable pot.")

    net_results = {
        name: payouts[name] + refunds[name] - contributions[name] for name in player_order
    }
    paid = sum(payouts.values(), _ZERO)
    is_balanced = is_settled and paid + rake_total == gross

    return HandLedger(
        contributions=_float_map(contributions),
        refunds=_float_map(refunds),
        payouts=_float_map(payouts),
        net_results=_float_map(net_results),
        gross_pot=_float(gross),
        rake=_float(rake_total),
        net_pot=_float(gross - rake_total),
        pots=tuple(rendered_pots),
        snapshots=tuple(snapshots),
        folded_players=tuple(name for name in player_order if name in folded),
        warnings=tuple(warnings),
        legality_issues=tuple(dict.fromkeys(legality_issues)),
        is_settled=is_settled,
        is_balanced=is_balanced,
        is_legal=not legality_issues,
    )


def build_ledger_from_records(
    players: Sequence[PlayerRecord],
    actions: Sequence[ActionRecord],
    *,
    dead_money: float = 0,
    winners: Mapping[int, Sequence[str]] | None = None,
    rake: RakePolicy | None = None,
    odd_chip_order: Sequence[str] = (),
    flop_seen: bool | None = None,
    blinds: BlindStructure | None = None,
) -> HandLedger:
    """Adapt persisted application records to the normalized ledger contract.

    New records explicitly declare whether an amount is incremental or a
    betting-round total ("raise to"). Historic records marked ``unknown`` are
    rejected for monetary actions rather than being silently reinterpreted.
    """

    normalized_players: list[LedgerPlayer] = []
    identities_by_name: dict[str, list[str]] = {}
    for player in players:
        if player.starting_stack is None:
            raise LedgerError(
                f"Player {player.player_name!r} needs a starting stack for derived accounting."
            )
        identity = getattr(player, "player_key", None) or player.player_name
        identities_by_name.setdefault(player.player_name, []).append(identity)
        normalized_players.append(
            LedgerPlayer(
                name=player.player_name,
                starting_stack=player.starting_stack,
                seat=getattr(player, "seat_index", None),
                key=identity,
            )
        )

    normalized_actions: list[LedgerAction] = []
    street: str | None = None
    street_contributions: dict[str, Decimal] = {
        player.key or player.name: _ZERO for player in normalized_players
    }
    for index, action in enumerate(actions):
        kind = action.action_type
        if kind == "all_in":
            kind = "all-in"
        if kind not in _COMMITMENT_KINDS | _NON_COMMITMENT_KINDS:
            raise LedgerError(f"Action {index + 1} has unsupported kind {kind!r}.")
        identity = getattr(action, "player_key", None)
        if identity is None:
            candidates = identities_by_name.get(action.player_name, [])
            if len(candidates) != 1:
                raise LedgerError(
                    f"Action {index + 1} cannot resolve player {action.player_name!r} "
                    "to one stable identity."
                )
            identity = candidates[0]
        if identity not in street_contributions:
            raise LedgerError(
                f"Action {index + 1} references unknown player identity {identity!r}."
            )
        if action.street != street:
            street = action.street
            street_contributions = {
                player.key or player.name: _ZERO for player in normalized_players
            }
        if kind in _COMMITMENT_KINDS:
            if action.amount is None:
                raise LedgerError(f"Action {index + 1} needs an incremental amount.")
            semantics = getattr(action, "amount_semantics", "unknown")
            if semantics == "unknown":
                raise LedgerError(
                    f"Action {index + 1} has unknown amount semantics and needs correction."
                )
            raw_amount = _decimal(action.amount, f"action {index + 1} amount")
            if semantics == "incremental":
                amount = raw_amount
            elif semantics == "raise_to":
                if kind == "ante":
                    raise LedgerError("Ante amounts cannot use raise-to semantics.")
                amount = raw_amount - street_contributions[identity]
                if amount <= 0:
                    raise LedgerError(
                        f"Action {index + 1} raise-to amount does not exceed the "
                        "player's current street contribution."
                    )
            else:
                raise LedgerError(
                    f"Action {index + 1} has unsupported amount semantics {semantics!r}."
                )
        else:
            # Historical win rows sometimes store the reported result in
            # ``amount``.  It is evidence, not a chip commitment or pot award.
            amount = 0
        normalized_actions.append(
            LedgerAction(
                player=identity,
                street=action.street,
                kind=kind,
                amount=_float(amount),
                is_live_post=(
                    True
                    if getattr(action, "is_live_post", None) is None
                    else bool(action.is_live_post)
                ),
                # Carried verbatim, including from a row whose ``action_type``
                # is not ``post_blind``. That row is the whole reason the field
                # exists here: a blind which took its poster's last chip is
                # routinely booked as an all-in, and reading only the kind let
                # it past the blind-structure refusal.
                forced_bet_type=getattr(action, "forced_bet_type", None),
            )
        )
        # The raise-to baseline is the seat's LIVE street contribution, decided
        # by the one predicate the reducer uses. Deciding it here from the kind
        # alone made "raise to 40" mean two different chip amounts depending on
        # whether the seat's dead post was spelled ``ante`` or ``all-in`` +
        # ``forced_bet_type='ante'``.
        if kind in _COMMITMENT_KINDS and _is_live_money(normalized_actions[-1]):
            street_contributions[identity] += amount

    return build_hand_ledger(
        normalized_players,
        normalized_actions,
        winners=winners,
        rake=rake,
        odd_chip_order=odd_chip_order,
        dead_money=dead_money,
        flop_seen=flop_seen,
        blinds=blinds,
    )


def blind_structure(
    small_blind: float | None,
    big_blind: float | None,
    straddles: Sequence[float] = (),
) -> BlindStructure | None:
    """The one definition of "a blind structure was declared", for every caller.

    The big blind is what the reducer's floor is built from, so its absence IS
    the absence of a declaration -- and every surface that stores, reads,
    neutralises, or displays the structure has to agree about that or they will
    disagree about whether a hand is study-ready. Persisted callers pass the
    three ``hand_settlements`` columns straight in.
    """

    if big_blind is None:
        return None
    return BlindStructure(
        small_blind=small_blind,
        big_blind=big_blind,
        straddles=tuple(straddles),
    )


def _is_forced_post(action: LedgerAction) -> bool:
    """Whether this commitment was posted under duress rather than chosen.

    Forced posts are the rows a seat has no say in, so they carry no information
    about what the seat would have done -- and, for the ordering question below,
    they are the rows a recording may legitimately list in any order relative to
    one another.
    """

    return action.kind in _FORCED_POST_KINDS or action.forced_bet_type is not None


def _is_live_structural_post(action: LedgerAction) -> bool:
    """Whether this commitment is a live forced bet that sets the wager level.

    Two ways a recording can say so, because only one of them is always
    available: the action KIND (``post_blind``, the shape the hand editor and the
    manual writer produce), or the recorded FORCED-BET TYPE (the shape that
    survives when a post which took its poster's last chip was booked as
    ``all-in``). Requiring the kind alone is what let the second shape past.

    WHERE THE RECORDING NAMES THE FORCED BET, THAT NAME DECIDES.  A row spelled
    ``post_blind`` and typed ``dead_blind`` is a dead post whether or not anybody
    filled in the separate post-status field, and the status field defaults to
    live (see ``build_ledger_from_records``).  Reading the kind first meant the
    two operator-facing selectboxes in the hand editor could disagree and the
    silent default won: a dead blind the operator had named as such was counted
    as chosen live money.  ``ante`` is never structural whatever type is carried,
    because an ante sets no wager level -- that is what
    ``_LIVE_STRUCTURAL_FORCED_BETS`` exists to say.
    """

    if action.kind not in _COMMITMENT_KINDS or action.kind == "ante":
        return False
    if action.forced_bet_type is not None:
        return (
            action.forced_bet_type in _LIVE_STRUCTURAL_FORCED_BETS
            and bool(action.is_live_post)
        )
    return action.kind == "post_blind" and bool(action.is_live_post)


def _is_live_money(action: LedgerAction) -> bool:
    """Whether this commitment is money the seat CHOSE to wager.

    Spec rule 1: forced posts are not live.  This is the ONE place that decides
    it for the pot layering, and it decides it the same way the blind-structure
    refusal already did -- through ``_is_forced_post`` and
    ``_is_live_structural_post`` -- rather than from ``action.kind`` alone.

    Keying on the kind was the same defect that ``_is_live_structural_post``'s
    docstring records having already been fixed once, left standing in the money
    classifier: a forced post which took its poster's last chip is routinely
    booked as ``all-in`` carrying ``forced_bet_type='ante'``, and the layering
    read that row as a chosen live wager.  Under the live-level model that is not
    a rounding difference -- live money is the only thing that opens a boundary
    and the only thing that decides eligibility above the main pot -- so the ante
    became a live level, the layer that should hold nothing but forced posts
    swelled by the post times the number of opponents covering it, and the seat
    was paid live chips no opponent had wagered against it.  Worked example (c),
    spelled that way, paid its ante-only seat +4 instead of +2, reported settled,
    balanced, legal and warning-free.
    """

    return not _is_forced_post(action) or _is_live_structural_post(action)


def _unreadable_forced_posts(
    actions: Sequence[LedgerAction],
    starting: Mapping[str, Decimal],
    preflop_floor: Decimal,
) -> frozenset[int]:
    """Indexes of live forced posts the declared structure cannot account for.

    A live forced post whose poster ended all-in proves only that the poster ran
    out; above the declared floor there is nothing left to read the structural
    size against, so the wager level cannot be named. That verdict is decided
    HERE, in one pass over the whole recording, for two reasons.

    ORDER. Asking "is this seat all-in yet" at the instant the post's own row is
    reduced is not the same question as "did forced posting exhaust this seat",
    and the two disagree whenever the seat has another forced row after its live
    blind -- its ante, or its dead blind. Those rows have no canonical order: the
    same hand, same seats, same stacks, same chips, is a legal recording either
    way, and both orders derive byte-identical contributions, pots, refunds,
    payouts and results. Deciding at the instant made the REFUSAL depend on that
    order, so moving an ante row below a blind row silently turned a blocked hand
    into a reconciled one around a pot that was 10 chips short. A verdict that a
    row reordering can flip is not a verdict. The seat's FINAL commitment answers
    it, and only forced rows may make up that total -- a seat that later chose to
    put chips in was never short of the blind it posted.

    TIMING. The flag this returns silences preflop complaints that name a wager
    level, and a flag raised part-way through the pass silences nothing before
    it. A recording whose live post lands after a voluntary action -- malformed,
    but reachable from a reconstruction with a misordered street -- would
    otherwise still print the sentence that misled the operator, once. Knowing
    every unreadable post before the loop starts closes that.

    Returns an empty set for a recording the main pass will reject outright, so
    a malformed hand raises its structural error rather than a verdict about it.
    """

    committed = {name: _ZERO for name in starting}
    voluntary: set[str] = set()
    candidates: list[tuple[int, str]] = []
    exhausted_at_post: set[int] = set()
    for index, action in enumerate(actions):
        if action.player not in committed or action.kind not in _COMMITMENT_KINDS:
            continue
        try:
            amount = _decimal(action.amount, f"action {index + 1} amount")
        except LedgerError:
            return frozenset()
        if amount <= 0 or amount > starting[action.player] - committed[action.player]:
            return frozenset()
        committed[action.player] += amount
        if not _is_forced_post(action):
            voluntary.add(action.player)
        if (
            action.street == "preflop"
            and _is_live_structural_post(action)
            # ``preflop_floor``, deliberately, and not the ``wager_floor`` the
            # main pass withholds from forced posts: this asks whether the
            # DECLARATION covers the post, not what the poster owed.
            and amount > preflop_floor
        ):
            candidates.append((index, action.player))
            if committed[action.player] == starting[action.player]:
                exhausted_at_post.add(index)
    exhausted = {
        name
        for name, total in committed.items()
        if name not in voluntary and total > 0 and total == starting[name]
    }
    return frozenset(
        index
        for index, player in candidates
        if index in exhausted_at_post or player in exhausted
    )


def _validate_blinds(blinds: BlindStructure | None) -> Decimal:
    """The largest structural live forced bet a declared structure establishes.

    Returns zero for an absent declaration, which is deliberately NOT the same
    thing as "use the observed maximum": zero makes every all-in forced post
    uninterpretable in ``build_hand_ledger``, so the missing declaration is
    reported instead of guessed at.

    The refusals are the ones that catch a declaration nobody could have meant.
    A transposed "5/10" entered as big blind 5 and small blind 10 would
    otherwise be accepted and would silently lower the floor by half; a
    non-positive big blind declares no structure at all while still clearing the
    unobservable-post check; and a straddle that does not exceed the forced bet
    before it is not a straddle. Refusing here rather than warning is right
    because these are impossible rooms, not unusual ones.
    """

    if blinds is None:
        return _ZERO
    small = (
        None
        if blinds.small_blind is None
        else _decimal(blinds.small_blind, "small blind")
    )
    big = _decimal(blinds.big_blind, "big blind")
    if small is not None and small < 0:
        raise LedgerError("Blind sizes must not be negative.")
    if big <= 0:
        raise LedgerError("A declared blind structure needs a positive big blind.")
    if small is not None and small > big:
        raise LedgerError("The small blind must not exceed the big blind.")
    floor = big
    for index, value in enumerate(blinds.straddles, start=1):
        straddle = _decimal(value, f"straddle {index}")
        if straddle <= floor:
            raise LedgerError(
                f"Straddle {index} must exceed the forced bet before it."
            )
        floor = straddle
    return floor


def _blind_floor_text(floor: Decimal) -> str:
    if floor <= 0:
        return "no blind structure is declared"
    return f"its largest forced bet is {floor}"


def _validate_players(
    players: Sequence[LedgerPlayer],
) -> tuple[tuple[str, ...], dict[str, Decimal]]:
    if not players:
        raise LedgerError("At least one player is required.")
    order: list[str] = []
    starting: dict[str, Decimal] = {}
    seats: set[int] = set()
    for player in players:
        name = player.name.strip()
        if not name:
            raise LedgerError("Player names must not be blank.")
        identity = (player.key or name).strip()
        if not identity:
            raise LedgerError("Player keys must not be blank.")
        if identity in starting:
            raise LedgerError(f"Duplicate player identity {identity!r}.")
        stack = _decimal(player.starting_stack, f"starting stack for {name}")
        if stack < 0:
            raise LedgerError("Starting stacks must not be negative.")
        if player.seat is not None:
            if player.seat in seats:
                raise LedgerError(f"Duplicate seat {player.seat}.")
            seats.add(player.seat)
        order.append(identity)
        starting[identity] = stack
    return tuple(order), starting


def _validate_rake(policy: RakePolicy) -> tuple[Decimal, Decimal | None, Decimal]:
    rate = _decimal(policy.rate, "rake rate")
    cap = None if policy.cap is None else _decimal(policy.cap, "rake cap")
    unit = _decimal(policy.rounding_unit, "rake rounding unit")
    if not _ZERO <= rate <= Decimal("1"):
        raise LedgerError("Rake rate must be between zero and one.")
    if cap is not None and cap < 0:
        raise LedgerError("Rake cap must not be negative.")
    if unit <= 0:
        raise LedgerError("Rake rounding unit must be positive.")
    return rate, cap, unit


def _validate_odd_chip_order(
    order: Sequence[str], starting: Mapping[str, Decimal]
) -> tuple[str, ...]:
    if len(set(order)) != len(order):
        raise LedgerError("odd_chip_order contains duplicate players.")
    unknown = [name for name in order if name not in starting]
    if unknown:
        raise LedgerError(f"odd_chip_order references unknown players: {', '.join(unknown)}.")
    return tuple(order)


def _uncalled_refunds(
    order: Sequence[str], contributions: Mapping[str, Decimal]
) -> dict[str, Decimal]:
    refunds = {name: _ZERO for name in order}
    ranked = sorted(((amount, name) for name, amount in contributions.items()), reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return refunds
    highest, player = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else _ZERO
    if highest > second and sum(1 for amount, _ in ranked if amount == highest) == 1:
        refunds[player] = highest - second
    return refunds


def _build_pots(
    order: Sequence[str],
    live_settled: Mapping[str, Decimal],
    dead_posted: Mapping[str, Decimal],
    put_up: Mapping[str, Decimal],
    folded: set[str],
    external_dead: Decimal,
) -> list[dict]:
    """Lay the pot out under the operator's model, rule 2 AS AMENDED.

    THE MODEL, in four sentences.  Rule 2 is the only one the operator changed;
    the other three are unchanged, and worked examples (a)-(d) must come out to
    the chip exactly as they did before:

      1. Layer boundaries are cut at the distinct LIVE contribution levels,
         measured after uncalled-bet refunds have been returned.  Live money is
         what a player CHOSE to wager.  Forced posts are never live.
      2. Dead money -- antes, big-blind antes, dead blinds and externally
         declared dead money -- goes into the LOWEST layer, but EACH
         CONTRIBUTOR'S dead chips count into a layer only up to the smallest
         TOTAL commitment among that layer's eligible seats.  The excess rises
         into the layer above, eligible to the seats whose own total reached
         above that cap.
      3. A seat is eligible for a layer if its own LIVE contribution reaches that
         layer's level.  Every unfolded seat that put ANY chip up -- live or dead
         -- is eligible for the main pot.
      4. A folded seat's chips stay in the layers they reached and it is eligible
         for none.

    WHAT THE AMENDMENT FIXED.  Rule 2 used to be unconditional: every dead chip
    went whole into the lowest layer.  That paid a seat whose ENTIRE commitment
    was smaller than an opponent's forced post more than the table had matched of
    it -- a 60-chip seat against three 100-chip antes collected 360 where the
    table had matched 240 of it, and a 40-chip seat against five 100-chip antes
    collected 540.  The previous round could not resolve that from the four
    worked examples, so it published the number and refused to call the hand
    study-ready.  The operator has now ruled, and the cap below is that ruling.

    TWO LADDERS, BUILT INDEPENDENTLY AND THEN INTERLEAVED.

      THE LIVE LADDER (rule 1).  Distinct positive live levels l_1 < ... < l_m
      cut bands (l_{j-1}, l_j].  Band j holds each seat's live money clamped into
      it and is eligible to every unfolded seat with live >= l_j (rule 3, first
      sentence).

      THE DEAD LADDER (rule 2, amended).  A cascade of caps:

          E_0    = every unfolded seat that put ANY chip up   (rule 3, 2nd sent.)
          c_k    = min{ commitment[m] : m in E_k }            (the layer's cap)
          E_k+1  = { m unfolded : commitment[m] > c_k }       (rule 2, 2nd sent.)

      Seat n's CUMULATIVE dead money placed through dead layer k is
      min(dead[n], c_k), so what it places IN layer k is
      min(dead[n], c_k) - min(dead[n], c_{k-1}).  The caps strictly increase --
      every member of E_{k+1} has a commitment above c_k, so the next minimum is
      above it too -- which is what makes the cascade terminate.

    WHY THE DEAD LADDER IS NOT FOLDED INTO THE LIVE BANDS.  The cheaper
    implementation -- let risen dead money spill into whatever live band sits
    above the cap -- is wrong, and worked example (b)'s invariant is why.  Take A
    with a 100 ante and no live money, B live 100, C live 40, none folded.  The
    cap is 40, so 60 of A's ante rises; the live band above the cap is (40, 100],
    eligible to B alone.  Spilling A's 60 in there leaves A -- still in the hand,
    having committed 100 -- able to win at most 40 and guaranteed to lose 60
    whatever it holds, which breaks "a seat that wins every layer it is eligible
    for cannot lose chips" (that is exactly what "ao wins the main pot and comes
    out at zero" asserts).  Making A eligible for that live band instead is
    worse: A would collect B's live money A never matched.  The risen dead money
    keeps its own layer, eligible by TOTAL, which is what rule 2's second
    sentence says in so many words.

    A CONSEQUENCE, STATED SO NOTHING DOWNSTREAM ASSUMES OTHERWISE.  The eligible
    sets no longer form a chain.  A dead layer's eligible set is a cut on TOTAL
    commitment and a live band's is a cut on LIVE contribution, and neither need
    contain the other: the A/B/C hand above lays out as 40 {A,B,C} / 80 {B,C} /
    60 {A,B} / 60 {B}, in which {B,C} and {A,B} do not nest.  Layers are emitted
    widest-eligible-set first, then by lower bound, then dead before live, which
    reproduces the conventional main/side ordering on every hand where the model
    does nest.  Every layer's eligible set is still contained in the main pot's,
    and no two layers share one.

    WHAT EACH ARGUMENT IS.  ``live_settled`` is live money that stuck, after
    refunds: it cuts the live boundaries, sizes the live bands, and decides
    eligibility for every live band.  ``dead_posted`` is each seat's forced posts
    and ``external_dead`` is dead money declared for the table with no
    contributing seat.  ``put_up`` is what a seat committed BEFORE any uncalled
    money came back, and it answers one question: did this seat play the hand at
    all.  A seat whose sole live post was returned -- because the only money
    facing it was a forced post nobody had a chip left to call -- still played,
    and still contests the pot it played for.

    WHERE ``put_up`` REACHES THE CAP, AND WHY IT MUST.  Rule 2's operand is a
    seat's TOTAL COMMITMENT, which is its live money after refunds plus its own
    dead posts.  For the one seat shape where that is zero and the seat is still
    in the hand -- its only live money was returned uncalled, which can only
    happen when no other seat has live money at all -- the commitment is read at
    the same point rule 3's "put ANY chip up" is read: what it put up.  Reading
    it as zero instead would cap every layer that seat is eligible for at zero,
    empty the main pot, and make a hand that seat won unrecordable.  Reading
    ``put_up`` for a seat whose post-refund commitment is POSITIVE would be the
    old overpayment in a new place -- a seat refunded 90 of a 100-chip bet has 10
    at risk, not 100 -- so it is deliberately not done.

    THE FIVE FAILURES THIS REPLACES, in the order they happened:

    * Refunding unmatched forced posts as uncalled bets took an ante out of the
      pot entirely.  Refunds are measured against LIVE money, upstream of here, so
      dead money always reaches a layer.
    * Deriving the levels from live money alone but pooling dead chips into the
      main pot left a seat whose ENTIRE commitment is a forced post contesting the
      whole first live layer: three ante chips took twenty-three.  Rule 2's dead
      ladder is what answers that, and it answers it by sizing the layer at the
      dead money rather than by cutting the live betting somewhere new.
    * Cutting at every distinct TOTAL commitment manufactured a side pot out of
      nothing but unequal dead money -- one seat's 5 ante against another's 3 dead
      blind with everybody matching the same 20 live, one pot of 88 reported as
      80 plus a phantom 8.  Rule 1 forbids it: 8 is not a live level, and the
      amended rule 2 does not cut there either, because 5 and 3 are both under
      the 20 cap and nothing rises.
    * Gating that on "some seat is live-short" removed the phantom only from
      hands where nobody was capped, and split two seats holding identical
      commitments across one boundary.
    * Cutting the first boundary at a short seat's TOTAL -- its live chips plus
      its OWN dead posts -- charged every opponent into the main pot up to that
      inflated level, so a seat live-short by 5 with a 5 ante was paid 5 live
      chips by each opponent that none of them had wagered against it.  An
      opponent does not match your ante.  That is why the LIVE ladder's levels
      are live and only live, and why the DEAD ladder caps dead money against
      the total without ever charging an opponent's live money into it.
    """

    dead_total = sum(dead_posted.values(), _ZERO) + external_dead
    contenders = [name for name in order if name not in folded]
    # Rule 3, second sentence: playing the hand is measured by what a seat PUT UP.
    played = tuple(name for name in contenders if put_up.get(name, _ZERO) > 0)

    # Rule 2's cap operand. See "WHERE ``put_up`` REACHES THE CAP" above.
    commitment: dict[str, Decimal] = {}
    for name in order:
        total = live_settled.get(name, _ZERO) + dead_posted.get(name, _ZERO)
        commitment[name] = total if total > 0 else put_up.get(name, _ZERO)

    bands: list[dict] = []

    # --- rule 1: the LIVE ladder ------------------------------------------
    levels = sorted({amount for amount in live_settled.values() if amount > 0})
    previous = _ZERO
    for level in levels:
        contributors = tuple(
            name for name in order if live_settled.get(name, _ZERO) > previous
        )
        amount = sum(
            (min(live_settled[name], level) - previous for name in contributors),
            _ZERO,
        )
        bands.append(
            {
                "kind": "live",
                "lo": previous,
                "hi": level,
                "amount": amount,
                "contributors": contributors,
                # Rule 3, first sentence, and rule 4.
                "eligible": tuple(
                    name
                    for name in contenders
                    if live_settled.get(name, _ZERO) >= level
                ),
            }
        )
        previous = level

    # --- rule 2 (amended) + rules 3 and 4: the DEAD ladder ------------------
    if dead_total > 0:
        if not played:
            # Nobody still in the hand put a chip up, so no seat can be declared
            # the winner of the money that is there. Unchanged behaviour: refuse.
            raise LedgerError("A pot has no eligible player.")
        eligible_now = played
        placed = {name: _ZERO for name in order}
        previous_cap = _ZERO
        # The caps strictly increase and each step strictly shrinks the eligible
        # set, so the model needs at most one pass per seat. The bound exists so
        # that a cap rule which does NOT increase fails loudly instead of
        # hanging -- a mutation that breaks the cascade must not become a
        # timeout, because a timeout is indistinguishable from a slow machine.
        for _ in range(len(order) + 1):
            cap = min(commitment[name] for name in eligible_now)
            layer_dead = _ZERO
            # Named apart from the live ladder's ``contributors`` because they are
            # different types -- this one is accumulated -- and sharing the name
            # made the two ladders look like one loop to a reader and to mypy.
            dead_contributors: list[str] = []
            for name in order:
                own = dead_posted.get(name, _ZERO)
                share = min(own, cap) - min(own, previous_cap)
                if share > 0:
                    layer_dead += share
                    placed[name] += share
                    dead_contributors.append(name)
            if previous_cap == _ZERO:
                # Externally declared dead money has no contributing seat, so
                # rule 2's cap -- written per contributor, against that
                # contributor's own total -- has no operand for it. It stays in
                # the lowest layer whole and never rises.
                layer_dead += external_dead
            remaining = sum(
                (dead_posted.get(name, _ZERO) - placed[name] for name in order),
                _ZERO,
            )
            next_eligible = tuple(
                name for name in contenders if commitment[name] > cap
            )
            if remaining > 0 and not next_eligible:
                # Nothing above to rise into. The excess can only ever be a
                # FOLDED seat's -- an unfolded seat's own dead money is at most
                # its own commitment, and the cascade's last cap is the largest
                # surviving commitment -- so this never strands a seat's own
                # chips above it. It does let the largest surviving stack collect
                # a folded seat's oversized ante, which is the correct poker
                # answer and is what the disclosure below names, because the
                # operator's rule 2 does not say where money with nowhere to
                # rise belongs.
                for name in order:
                    over = dead_posted.get(name, _ZERO) - placed[name]
                    if over > 0:
                        layer_dead += over
                        placed[name] = dead_posted.get(name, _ZERO)
                        if name not in dead_contributors:
                            dead_contributors.append(name)
                remaining = _ZERO
            if layer_dead > 0:
                bands.append(
                    {
                        "kind": "dead",
                        "lo": previous_cap,
                        "hi": cap,
                        "amount": layer_dead,
                        "contributors": tuple(dead_contributors),
                        # Rule 2, second sentence. At cap zero this is rule 3's
                        # second sentence, so the ladder is uniform.
                        "eligible": eligible_now,
                    }
                )
            if remaining <= 0:
                break
            previous_cap = cap
            eligible_now = next_eligible
        else:  # pragma: no cover - unreachable while the caps increase
            raise LedgerError(
                "The dead-money cascade did not terminate: the layer caps are "
                "not strictly increasing."
            )

    if not bands:
        return []

    # Layers are ordered widest eligible set first, then by lower bound, then
    # dead before live so the dead layer carrying rule 3's wider main-pot
    # eligibility is never displaced by a live band of equal width. Under the
    # amendment the two ladders do not nest, so this ordering is what reproduces
    # conventional main/side numbering wherever they do.
    bands.sort(
        key=lambda band: (
            -len(band["eligible"]),
            band["lo"],
            0 if band["kind"] == "dead" else 1,
            band["hi"],
        )
    )

    # Bands with the SAME eligible set are one layer. No settlement can tell them
    # apart, and emitting them separately would put a "Side pot" label on a layer
    # no seat is short of. This is what collapses the ordinary hand -- where the
    # dead layer and the first live band are contested by the same seats -- back
    # to a single pot. Grouped rather than merged pairwise: the two ladders
    # interleave, so two bands sharing an eligible set need not be neighbours.
    merged: list[dict] = []
    by_eligible: dict[tuple[str, ...], dict] = {}
    for band in bands:
        key = band["eligible"]
        top = by_eligible.get(key)
        if top is None:
            copy = dict(band)
            by_eligible[key] = copy
            merged.append(copy)
            continue
        top["lo"] = min(top["lo"], band["lo"])
        top["hi"] = max(top["hi"], band["hi"])
        top["amount"] += band["amount"]
        top["contributors"] = tuple(
            dict.fromkeys(top["contributors"] + band["contributors"])
        )

    if not merged[0]["eligible"]:
        raise LedgerError("A pot has no eligible player.")

    # The MAIN pot is the layer every seat still contesting anything may win --
    # the one whose eligible set contains every other layer's. That is a fact
    # about the ladder, so it is read off the ladder.
    #
    # It used to be written ``"main" if index == 0 else "side"``, which is a fact
    # about the LIST, and ``PotLayer.label`` claims in so many words to be
    # derived "from what created it and not from its index". The two agree only
    # while the sort above puts the widest eligible set first, and nothing
    # checked that: a layering that emitted the ladder in any other order still
    # labelled its first entry "Main pot" and every consumer, the settlement
    # editor's pot numbers included, believed it. Deriving it means such a
    # layering produces NO main pot and fails loudly instead.
    #
    # Every eligible set really is contained in the widest one -- a live band's
    # is {contender : live >= level} and a dead layer's is {contender : total >
    # cap}, and both are subsets of the seats that put a chip up -- but that is
    # a property of the two rules rather than of the list index, and this is
    # where it is asserted rather than assumed.
    reach = {name for band in merged for name in band["eligible"]}
    return [
        {
            "index": index,
            "amount": band["amount"],
            "contributors": band["contributors"],
            "eligible_players": band["eligible"],
            "cause": "main" if reach <= set(band["eligible"]) else "side",
        }
        for index, band in enumerate(merged)
    ]


def _unruled_dead_money_warnings(
    order: Sequence[str],
    live_settled: Mapping[str, Decimal],
    dead_posted: Mapping[str, Decimal],
    put_up: Mapping[str, Decimal],
    folded: set[str],
    external_dead: Decimal,
) -> list[str]:
    """Name the dead-money shapes the operator's rule 2 still does not decide.

    WHAT THIS USED TO BE, AND WHY IT IS NARROWER NOW.  The previous round refused
    every hand in which a main-pot seat could be paid a forced post larger than
    its own total commitment, because rule 2 was unconditional -- all dead money
    into the lowest layer -- and nothing in the four worked examples said whether
    a 40-chip seat should collect five opponents' 100-chip antes.  It published
    540 and declined to call the hand study-ready.

    THE OPERATOR HAS RULED.  Amended rule 2 caps each contributor's dead chips at
    the smallest TOTAL commitment among the layer's eligible seats and lifts the
    excess into a layer eligible to the seats whose own total reached above that
    cap.  That hand is now 240 to the short seat, which is exactly what the table
    matched of it, and it is answered rather than refused.  So the refusal is
    withdrawn FOR THAT FAMILY -- and only for it.  Removing the whole guard
    because most of what it covered became decidable is how a defect returns, so
    what stays is the residue: the two shapes where the amended rule still has
    nothing to compare a dead chip against.

    (1) DEAD MONEY WITH NOWHERE TO RISE.  Rule 2 lifts the excess "into the layer
        above, eligible to the seats whose own total reached above that cap".
        When no seat still in the hand has a total above the cap there is no such
        layer, and the rule stops mid-sentence.  The excess can only ever be a
        FOLDED seat's -- an unfolded seat's dead money is at most its own
        commitment, and the last cap the cascade reaches is the largest surviving
        commitment -- so nothing here strands a seat's own chips.  What it does
        do is hand the largest surviving stack a folded seat's oversized ante.
        That is the conventional poker answer and the layering takes it, but it
        is a chip figure the operator's sentence does not reach, and it is the
        one place a seat's ceiling is not simply its own total.

    (2) A SEAT CONTESTING A POT IT CONTRIBUTED NOTHING TO.  Rule 3's "put ANY
        chip up" is read BEFORE the uncalled bet came back, so a seat whose only
        live post was returned still contests the dead money; reading it after
        the refund would make a hand that seat won unrecordable.  Rule 2's cap is
        written against a seat's TOTAL COMMITMENT, and for that one seat the
        post-refund total is zero.  The layering reads the cap at the same point
        rule 3 is read -- what the seat put up -- because reading it as zero caps
        every layer that seat is eligible for at zero and empties the main pot.
        That is a reading, not a ruling: the operator's sentence never has to
        choose a measurement point because none of the five worked examples
        contains a refund.  It reaches only hands whose whole pot is dead money
        and exactly one seat wagered live, uncalled, so it is narrow -- and it is
        named rather than assumed.

    THE ONE ASSUMPTION DELIBERATELY NOT WARNED ABOUT.  Externally declared dead
    money has no contributing seat, so rule 2's cap -- written per CONTRIBUTOR,
    against that contributor's own total -- has no operand for it; it stays in
    the lowest layer whole and a seat with a commitment of 1 may win 1000 of it.
    That is what a real table does with an overlay, a penalty or a carried pot,
    the amendment's stated motivation ("more than the table had matched of it")
    does not bite when no seat put the money up, and the previous round reached
    the same conclusion.  It is recorded here as an assumption rather than
    disclosed per hand, because a refusal on every hand carrying declared dead
    money would train an operator to ignore the refusals that matter.

    THIS CHANGES NO CHIP FIGURE.  The verdict stays legal, balanced and settled --
    the arithmetic is not in doubt -- but the warning reaches ``_cross_check``,
    which folds ledger warnings into its issues, so ``is_authoritative`` is False
    and the hand lands in ``needs_correction`` with the seats and both numbers
    named.  A wrong prediction that is visibly rejected is a coverage limitation;
    the same prediction published as authoritative is a release blocker.
    """

    dead_total = sum(dead_posted.values(), _ZERO) + external_dead
    if dead_total <= 0:
        return []
    contenders = [name for name in order if name not in folded]
    played = [name for name in contenders if put_up.get(name, _ZERO) > 0]
    if not played:
        return []

    commitment: dict[str, Decimal] = {}
    for name in order:
        total = live_settled.get(name, _ZERO) + dead_posted.get(name, _ZERO)
        commitment[name] = total if total > 0 else put_up.get(name, _ZERO)

    notes: list[str] = []

    # (1) Dead money above every surviving seat's commitment.
    ceiling = max(commitment[name] for name in played)
    strays = sorted(
        (
            (dead_posted.get(name, _ZERO) - ceiling, name)
            for name in order
            if dead_posted.get(name, _ZERO) > ceiling
        ),
        reverse=True,
    )
    if strays:
        excess, poster = strays[0]
        collectors = sorted(name for name in played if commitment[name] == ceiling)
        notes.append(
            f"{_float(excess)} chips of {poster!r}'s forced post are above every "
            f"remaining seat's commitment, the largest of which is "
            f"{_float(ceiling)} ({', '.join(repr(name) for name in collectors)}). "
            "The pot model lifts capped dead money into the layer above, but "
            "names no layer when no seat's total reaches past the cap; this hand "
            "is not study-ready until it does. The chips above are the "
            "conventional answer and the only recordable one -- a folded seat's "
            "post cannot be returned and a layer no seat can win is refused -- "
            "so nothing in the recording is wrong and no correction to it "
            "clears this. It is the model that is silent here, and re-entering "
            "the post as external dead money would clear the note by deleting a "
            f"seat's {_float(dead_posted.get(poster, _ZERO))}-chip commitment "
            "from the results."
        )

    # (2) A seat whose whole commitment came back as an uncalled bet.
    for name in played:
        if live_settled.get(name, _ZERO) + dead_posted.get(name, _ZERO) > 0:
            continue
        notes.append(
            f"{name!r} contests the dead money having had its entire live post "
            f"of {_float(put_up.get(name, _ZERO))} returned as an uncalled bet, "
            "so its commitment after refunds is zero. The pot model caps dead "
            "money at a seat's total commitment but does not say whether that is "
            "measured before or after the refund; this hand is not study-ready "
            "until it does."
        )
    return notes


def _validate_winners(
    winners: Mapping[int, tuple[str, ...]],
    pots: Sequence[dict],
    starting: Mapping[str, Decimal],
    folded: set[str],
) -> None:
    for index, names in winners.items():
        if not isinstance(index, int) or index < 0 or index >= len(pots):
            raise LedgerError(f"Winner declaration references missing pot {index}.")
        if not names:
            raise LedgerError(f"Pot {index} must have at least one winner.")
        if len(set(names)) != len(names):
            raise LedgerError(f"Pot {index} repeats a winner.")
        for name in names:
            if name not in starting:
                raise LedgerError(f"Pot {index} references unknown winner {name!r}.")
            if name in folded:
                raise LedgerError(f"Folded player {name!r} cannot win pot {index}.")
            if name not in pots[index]["eligible_players"]:
                raise LedgerError(f"Player {name!r} is not eligible for pot {index}.")


def _compute_rake(
    gross: Decimal,
    *,
    rate: Decimal,
    cap: Decimal | None,
    unit: Decimal,
    waived: bool,
) -> Decimal:
    if waived or rate == 0 or gross == 0:
        return _ZERO
    raw = gross * rate
    if cap is not None:
        raw = min(raw, cap)
    return _round_down(raw, unit)


def _allocate_rake(pots: Sequence[dict], total: Decimal, unit: Decimal) -> list[Decimal]:
    """Spread one rake total across the pot layers, never past a layer's own size.

    Every non-final share used to be rounded DOWN to the declared unit and the
    whole leftover charged to the LAST pot with no cap at that pot's amount, so
    an ordinary hand under an ordinary policy could rake a pot beyond what it
    contained: a 149.25 main pot and a 0.50 side pot at 5% capped at 5 with a
    whole-chip drop took 4 from the main pot and charged the leftover 1 to a pot
    of 0.50, giving that layer ``net_amount = -0.50`` and paying its winner minus
    half a chip. ``is_balanced`` stayed True because the negative preserved
    ``paid + rake == gross``, so a hand whose side-pot winner also won the main
    pot certified as authoritative and study-ready, and a hand whose pots had
    different winners became permanently unreconcilable: the derived payout was
    negative, ``SettlementEntry.amount`` is ``ge=0`` so the operator could not
    declare it, and ACCOUNTING_NOT_AUTHORITATIVE named a save that could never
    clear it.

    Each share is therefore capped at its own pot, and the rounding leftover is
    offered to the layers in order, each taking only what it still has room for.
    ``_validate_rake`` bounds the rate at one and ``_compute_rake`` caps the
    total at the gross, so the leftover always finds room and the loop always
    settles at zero.
    """
    if not pots:
        return []
    if total <= 0:
        return [_ZERO for _ in pots]
    gross = sum((pot["amount"] for pot in pots), _ZERO)
    allocated: list[Decimal] = []
    remaining = total
    for pot in pots:
        share = _round_down(total * pot["amount"] / gross, unit)
        share = min(share, remaining, pot["amount"])
        allocated.append(share)
        remaining -= share
    for index, pot in enumerate(pots):
        if remaining <= 0:
            break
        take = min(pot["amount"] - allocated[index], remaining)
        if take > 0:
            allocated[index] += take
            remaining -= take
    return allocated


def _split_granularity(
    contributions: Iterable[Decimal], dead_money: Decimal
) -> Decimal:
    """The finest denomination the hand's own numbers are written in.

    ``rake_rounding_unit`` is a DECLARED operator field ("Chip unit" in the
    settlement editor, and a verbatim value in an import payload) with no upper
    bound. Its documented job is rounding the rake, which is a real room rule --
    a house that drops whole dollars against a 0.50 blind is ordinary. Its
    undocumented second job was to be the granularity a CHOPPED pot was divided
    at: ``_split_pot`` rounds each winner's share DOWN to the unit and gives
    every leftover chip to the front of ``odd_chip_order``, so with the rake rate
    at zero -- where neither declared-chips disclosure fires and no correction is
    recorded -- one number in the settlement editor moved the derived hero result
    and made a fabricated ``hands.hero_bb_won`` reconcile exactly.

    Round 8 bounded the declared unit by the greatest common divisor of the
    observed contributions and claimed that removed the dial. It did not. Every
    DIVISOR of that gcd was still reachable, and divisors do not agree once the
    pot is anything but two equal halves: on three seats contributing 8 each and
    a two-way chop of 24, units 0.01/1/2/4 pay 12/12 while 3, 5 and 8 pay 16/8.
    The bound was also the wrong direction. The gcd is an UPPER bound on the
    denomination in play -- every amount is a whole multiple of the real chip, so
    the real chip DIVIDES the gcd and may be far finer -- and splitting at an
    upper bound maximises the redistribution, because rounding the base share
    down sheds up to a whole unit from every winner before the round-robin hands
    those chips back from the front of the order.

    So the split granularity is now derived from evidence alone -- the settled
    contributions and the declared dead money, never the declared unit and never
    anything computed from it -- and it is the FINEST signal those amounts carry,
    not the coarsest. Concretely it is the smallest decimal place any observed
    amount is written in, capped at one whole chip. That is deliberately the
    conservative direction. A denomination cannot be established from above: three
    seats each committing 8 share a factor of 8 and demonstrate nothing about
    8-chips, whereas an amount of 49.75 does demonstrate that hundredths exist.
    Choosing the finest evidence keeps every derived chop as close to an even one
    as the hand allows, so the most any granularity can move a seat is under one
    chip -- against half the pot, which is what the declared unit could move.

    The odd chip survives, because it is real: chips are indivisible, so a
    21-chip pot chopped two ways is genuinely pushed 11/10, and deriving
    10.5/10.5 would raise a false blocker against an honest declared award. What
    it can no longer be is a dial -- the deviation from an even chop is now under
    one chip on every hand, and pinned by the hand's own amounts rather than
    chosen by an operator. ``_split_pot`` always distributes the full amount, so
    no granularity can change the gross, the rake, or chip conservation; only who
    receives an odd chip, and that is decided by the audited ``odd_chip_order``.
    """
    quantum = _MAX_SPLIT_QUANTUM
    for amount in (*contributions, dead_money):
        if amount <= 0:
            continue
        exponent = amount.normalize().as_tuple().exponent
        if not isinstance(exponent, int):
            # 'n', 'N' or 'F' -- a non-finite Decimal. `_decimal` already refuses
            # those on the way in, so this is unreachable defence that keeps the
            # helper total rather than raising from inside the split.
            continue
        # normalize() first, so an amount that arrived as "5.0" and one that
        # arrived as "5" describe the same denomination. Without it the split of
        # one hand depended on whether its amounts reached the ledger as ints or
        # as floats.
        candidate = Decimal(1).scaleb(exponent)
        if candidate < quantum:
            quantum = candidate
    return quantum


def _split_pot(
    amount: Decimal,
    winners: Sequence[str],
    payouts: dict[str, Decimal],
    *,
    unit: Decimal,
    odd_order: Sequence[str],
) -> None:
    base = _round_down(amount / len(winners), unit)
    for winner in winners:
        payouts[winner] += base
    remainder = amount - base * len(winners)
    priority = [name for name in odd_order if name in winners]
    priority.extend(name for name in winners if name not in priority)
    cursor = 0
    while remainder >= unit:
        payouts[priority[cursor % len(priority)]] += unit
        remainder -= unit
        cursor += 1
    if remainder:
        payouts[priority[cursor % len(priority)]] += remainder


def _round_down(value: Decimal, unit: Decimal) -> Decimal:
    return (value / unit).to_integral_value(rounding=ROUND_DOWN) * unit


def _decimal(value: float | int | Decimal, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise LedgerError(f"{label} must be numeric.") from exc
    if not result.is_finite():
        raise LedgerError(f"{label} must be finite.")
    return result


def _float(value: Decimal) -> float:
    return float(value)


def _float_map(values: Mapping[str, Decimal]) -> dict[str, float]:
    return {name: _float(value) for name, value in values.items()}
