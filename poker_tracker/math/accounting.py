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
# How this hand's antes were taken, DECLARED and never inferred. See
# ``AnteMode`` and ``_resolve_ante_mode``.
AnteModeName = Literal["NONE", "PER_PLAYER", "SINGLE_PAYER_TABLE_ANTE"]

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
# relabel away. Where a recording states the forced-bet type and the rest of the
# row does not deny it, that statement is what is read; where the row denies it,
# see ``_forced_bet_row_conflict``.
_LIVE_STRUCTURAL_FORCED_BETS = frozenset(
    {"small_blind", "big_blind", "straddle", "bring_in"}
)
_FORCED_POST_KINDS = {"ante", "post_blind"}
# The kinds a forced post can legitimately be BOOKED UNDER. What makes a
# commitment a forced post is its kind; ``forced_bet_type`` only refines WHICH
# forced post it is. ``all-in`` is here because a post that took its poster's
# last chip is routinely booked that way and the type is then the only thing
# naming it -- that is the whole reason the field reaches the reducer.
#
# ``bet``, ``call`` and ``raise`` are answers to a wager level, which is the one
# thing a forced post never is, so no room can produce one of them as a post.
# Letting a type promote them was a defect with teeth: ``call`` is the kind whose
# legality is an equality against ``to_call``, so a big blind all-in for 4 in a
# 5/10 game with the button calling 5 read as ILLEGAL unlabelled and LEGAL the
# moment the button's row carried any type at all, and a ``dead_blind`` label on
# that same call moved a chip out of the pot as well. The non-commitment kinds
# are excluded for the same reason and cost nothing: they commit no chips.
_FORCED_POST_CAPABLE_KINDS = _FORCED_POST_KINDS | {"all-in"}
# Forced-bet names that identify an ANTE: money owed to the table by the seat
# rather than wagered against an opponent, and the ONLY pool the declared ante
# mode governs. ``dead_blind``, missed blinds and penalty posts are deliberately
# absent -- see ``_is_ante_post`` and rule 3's last clause, which leaves every
# non-ante forced post on the treatment it already had.
#
# ``button_ante`` and ``table_ante`` are not currently spellable in
# ``models.ForcedBetType``; they are listed so that adding either to the
# recording vocabulary classifies it as an ante on the day it is added, rather
# than silently dropping it into the non-ante pool where the mode cannot reach
# it.
_ANTE_FORCED_BET_TYPES = frozenset(
    {"ante", "big_blind_ante", "button_ante", "table_ante"}
)
# The forced-bet names that are species of a BLIND POST: the live structural
# ones plus the dead blind, which is a blind that sets no wager level. ``ante``
# is not among them, which is the whole point of keeping the two sets apart.
_BLIND_FORCED_BET_TYPES = _LIVE_STRUCTURAL_FORCED_BETS | {"dead_blind"}
# What each forced-post KIND may legitimately be typed as.
#
# A kind that names a forced post has already named its species, and the label
# only says WHICH post inside that species: ``ante`` typed ``big_blind_ante`` is
# still an ante, ``post_blind`` typed ``dead_blind`` is still a blind. A label
# from the OTHER species is not a refinement, it is a second and incompatible
# claim about the same row, and it was able to move the row out of the pool its
# kind put it in. That is how tagging the only ante row ``straddle`` silenced the
# undeclared-ante-mode refusal and turned a hand the product must refuse into an
# accepted one: a guard a selectbox can switch off is not a guard.
#
# ``all-in`` is deliberately absent rather than mapped to everything. That kind
# names no species at all -- it says the poster ran out of chips -- so there the
# label is the only signal the KIND leaves standing and no name contradicts it.
# The row's post status can still contradict one; this table answers only the
# kind-versus-label question.
_KIND_FORCED_BET_TYPES: dict[str, frozenset[str]] = {
    "ante": _ANTE_FORCED_BET_TYPES,
    "post_blind": _BLIND_FORCED_BET_TYPES,
}
# The forced-bet names that are DEAD money: owed to the table, answering no
# wager level. Every name is either live-structural or dead, which is what makes
# the liveness check below total rather than a list of special cases.
_DEAD_FORCED_BET_TYPES = _ANTE_FORCED_BET_TYPES | {"dead_blind"}
# Whether each forced-bet name IS a live post, by definition of the name. A
# ``big_blind`` is live and a ``dead_blind`` is dead however the separate
# post-status field is filled in, so the two fields are not independent: a row
# carrying both is stating the same fact twice and they can disagree.
#
# That disagreement was the second knob on the reported defect. A big blind
# all-in for 4 with blinds undeclared is refused as an unreadable forced post
# and lays out 12/2; setting the row's ``Post status`` to dead -- one selectbox
# on the same row of the same panel, with the panel auto-opened by the very
# warning that names the row -- silenced the refusal, moved 8 chips of a
# 14-chip pot, and presented the hand as study-ready with zero blockers. Naming
# the row ``dead_blind`` instead reaches the same place by the other selectbox.
# Neither is answerable and both are reportable, which is the rule commit
# 1f8f01d established for the kind-versus-label axis and this table extends to
# the liveness axis.
_LABEL_IS_LIVE_POST: dict[str, bool] = {
    **{name: True for name in _LIVE_STRUCTURAL_FORCED_BETS},
    **{name: False for name in _DEAD_FORCED_BET_TYPES},
}
# The two ways one row can contradict itself, which read differently to an
# operator and therefore get different sentences.
_FORCED_BET_KIND_CONFLICT = "kind"
_FORCED_BET_STATUS_CONFLICT = "status"
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
    # What the recording says this post's STATUS was, when it says anything:
    # True for a live post, False for a dead one, None for "the recording said
    # nothing". The three states are kept apart all the way to the reducer
    # because collapsing None into True here made an unspecified status
    # indistinguishable from a stated one, and the liveness contradiction below
    # can only be raised against a status the operator actually stated. An
    # unstated status reads as live everywhere, which is what the collapsed
    # field already did, so no existing construction moves a chip.
    is_live_post: bool | None = None
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


class AnteMode:
    """How this hand's antes were taken -- a DECLARATION, never an inference.

    A fact about the TABLE, not about the line: the same category as
    ``BlindStructure``, ``RakePolicy`` and ``dead_money``, supplied the same way,
    persisted in the same row, and refused the same way when it is missing.

    THE THREE MODES.

    ``NONE``
        No antes in this hand. Declaring it over a hand that contains ante posts
        is a contradiction and is refused, because one of the two is wrong and
        the model must not choose which.

    ``PER_PLAYER``
        Every seat -- or several seats -- antes individually. Each contributor's
        personal ante is capped, for placement in a layer, at the smallest TOTAL
        commitment among that layer's eligible seats, and the excess rises into a
        layer eligible to the seats whose own totals reached above the cap. This
        is the rule shipped last round, retained unchanged.

    ``SINGLE_PAYER_TABLE_ANTE``
        One seat posts a consolidated ante FOR THE TABLE -- a big-blind ante or a
        button ante. That ante is TABLE MONEY: it goes whole into the main pot
        and is never capped against a shorter blind.

    WHY IT CANNOT BE INFERRED, and this is the operator's explicit ruling rather
    than a preference. The two modes give DIFFERENT pots on the same recording.
    Blinds 1/2, the big blind posts a 2 ante and the small blind is all-in for
    its 1-chip blind: declared ``SINGLE_PAYER_TABLE_ANTE`` that is a 5-chip main
    pot and the small blind nets +4; declared ``PER_PLAYER`` it is a 4-chip main
    pot and +3. Nothing in the action line distinguishes them -- both spell "the
    big blind put 2 chips in that nobody had to match" -- so a rule that guessed
    from the shape of the posts would be choosing between two correct answers by
    coin flip and publishing the result as authoritative. Inferring this is
    precisely the class of move that produced five consecutive criticals in this
    module.

    WHAT THE MODE DOES NOT REACH, stated here because getting it wrong is the
    dangerous implementation error:

    * LIVE money -- blinds, straddles, bring-ins, bets, calls, raises -- is
      untouched by every mode. A straddle beside a consolidated ante cuts an
      ordinary live boundary above it and nothing interacts.
    * DEAD BLINDS, missed blinds and penalty posts are capped in EVERY mode,
      including ``SINGLE_PAYER_TABLE_ANTE``. Rule 3's last clause leaves non-ante
      forced posts on their existing treatment, and this is an ANTE mode: it
      names antes. So a ``SINGLE_PAYER_TABLE_ANTE`` hand carrying a dead blind
      runs BOTH dead-money rules at once, on two disjoint pools. A reducer that
      branched once on the mode and treated all dead money alike would pass every
      worked example and then silently lose the cap on exactly those hands --
      live 5/5/5 with a 100 consolidated ante and a 50 dead blind is main 120
      with 45 risen, not main 165.
    * EXTERNALLY DECLARED dead money is capped under every mode. It has no seat,
      so it is not "one seat posts a consolidated ante", and capping is the
      strict direction. See ``build_hand_ledger``'s ``dead_money`` note.
    * A refunded uncalled bet is not in the pot at all, in every mode.
    """

    NONE: AnteModeName = "NONE"
    PER_PLAYER: AnteModeName = "PER_PLAYER"
    SINGLE_PAYER_TABLE_ANTE: AnteModeName = "SINGLE_PAYER_TABLE_ANTE"

    #: Every declarable mode, in the order an operator-facing control lists them.
    ALL: tuple[AnteModeName, ...] = (
        "NONE",
        "PER_PLAYER",
        "SINGLE_PAYER_TABLE_ANTE",
    )


# The refusal every surface shows when a hand's antes have no declared mode.
# One string prefix so a consumer can recognise the class without matching prose.
UNDECLARED_ANTE_MODE_PREFIX = "This hand contains ante posts"


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
    ante_mode: str | None = None,
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
    of a CAPPED forced post above the smallest TOTAL commitment among that
    layer's eligible seats rises into a layer of its own, eligible to the seats
    whose own total reached past the cap.  Those two ladders do not nest in
    general, so the eligible sets are ordered widest-first rather than chained.
    See ``_build_pots`` for the model in full.  Omitting winners returns a useful
    but explicitly unsettled ledger. ``flop_seen`` is an
    optional completed-hand fact for histories where a board ran out without
    any postflop action (for example, a preflop all-in). When omitted, the
    ledger preserves the historic behavior of inferring it from action streets.

    ``ante_mode`` is how this hand's antes were taken (see ``AnteMode``), and it
    decides which dead chips the cap above governs.  Under ``PER_PLAYER`` and
    ``NONE`` every dead chip is capped, which is the rule this reducer already
    shipped.  Under ``SINGLE_PAYER_TABLE_ANTE`` the consolidated ante is TABLE
    MONEY: it goes whole into the main pot and is never capped against a shorter
    blind, while every non-ante forced post in the SAME hand still runs the
    cascade.  It is a declaration in the same sense as ``blinds`` and is refused
    the same way when a hand contains antes and does not carry one -- see
    ``_resolve_ante_mode``, which also states exactly what a refused hand
    derives.  A hand with no antes needs no declaration and is untouched.

    ``dead_money`` is chips no seat put up -- an overlay, a carried pot, a
    penalty returned to the table.  It is capped exactly as a recorded dead post
    is, under whichever rule the mode selects for the capped pool, which is the
    operator's fifth ruling.  It used to join the main pot whole and unwarned,
    so a seat that had committed 2 chips could be paid 312 of it.

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
    RECORDING IDENTIFIES as one and does not mark dead -- by
    ``kind == "post_blind"``, or by a ``forced_bet_type`` naming a live
    structural bet on a row booked as ``all-in``, which is the only other kind a
    post can be written under (see ``_is_live_structural_post`` and
    ``_FORCED_POST_CAPABLE_KINDS``).  A ``bet``, ``call`` or ``raise`` carrying
    such a type is a contradictory recording rather than a post: it is derived
    from its kind and refused, and the refusal is what
    ``_forced_bet_row_conflict`` writes.  A recording that books a short blind as
    a plain ``all-in`` and states no forced-bet type has said nothing that
    distinguishes it from an ordinary short shove, and nothing here can tell them
    apart; such a hand still derives ``to_call`` from the observed maximum and is
    NOT refused.  The CV reconstruction spine emits exactly that shape for a seat
    whose stack reads zero, so this refusal does not cover every reconstructed
    hand.  Declaring the structure covers those hands correctly -- the floor is
    applied whatever the kinds are -- but the operator is not prompted to.

    WHAT A RECORDING CAN STILL SAY TO PUT A POST OUT OF SCOPE, stated because it
    is the sharp edge here.  A blind that is NAMED dead -- ``forced_bet_type``
    ``dead_blind``, or a post status of dead -- answers no wager level, so this
    refusal genuinely does not reach it and the chips are owed to the table.
    Where the row states BOTH of those fields and they disagree, the recording is
    refused for the contradiction and the post stays in scope; that is
    ``_forced_bet_row_conflict``.  Where the row states only ONE of them, there
    is nothing on the row to check it against, and a live blind an operator has
    marked dead is believed.  That is a single unfalsifiable claim rather than a
    contradiction, and no rule in this module can see it.
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
    # The dead pool is kept SPLIT rather than summed, because the declared ante
    # mode governs exactly one half of it: ``ante_contributions`` is what
    # ``SINGLE_PAYER_TABLE_ANTE`` exempts from the cap, and
    # ``other_dead_contributions`` (dead blinds, missed blinds, penalty posts)
    # keeps its existing capped treatment under EVERY mode. Summing them and
    # branching once on the mode is the dangerous shortcut here: it passes every
    # worked example and then silently loses the cap on any consolidated-ante
    # hand that also carries a dead blind, because the exemption swallows the
    # dead blind too.
    ante_contributions = {name: _ZERO for name in player_order}
    other_dead_contributions = {name: _ZERO for name in player_order}
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
        # Outside the ``structure_unreadable`` guard with the stack check above,
        # and for the same reason: it reads no wager level, only the row's own
        # three fields.
        row_conflict = _forced_bet_row_conflict(action)
        if row_conflict == _FORCED_BET_STATUS_CONFLICT:
            stated = "live" if action.is_live_post else "dead"
            # Whichever field named the opposing liveness is the one to quote
            # back, so the sentence points at both halves of the contradiction.
            if action.forced_bet_type in _LABEL_IS_LIVE_POST:
                named_by = f"typed as a {action.forced_bet_type!r} forced post"
                named = "live" if _LABEL_IS_LIVE_POST[action.forced_bet_type] else "dead"
                remedy = (
                    "It is derived as though neither field were set. Either "
                    "correct the forced post field or correct Post status in "
                    "Edit actions."
                )
            else:
                named_by = f"booked as {action.kind!r}"
                named = "dead"
                remedy = (
                    f"It is derived from its kind, as an ordinary {action.kind}. "
                    "Either correct the action kind or correct Post status in "
                    "Edit actions."
                )
            legality_issues.append(
                f"{action_label}: this row is {named_by}, which is {named} "
                f"money, but its post status says {stated}. A live forced post "
                "sets what every other seat owes and a dead one is owed to the "
                "table, so the two fields state opposite things about the same "
                "chips and they lay out into different pots. The pot model will "
                "not choose which one the recording meant. " + remedy
            )
        elif row_conflict == _FORCED_BET_KIND_CONFLICT:
            preamble = (
                f"{action_label}: this row is booked as {action.kind!r} but "
                f"typed as a {action.forced_bet_type!r} forced post."
            )
            if action.kind in _FORCED_POST_CAPABLE_KINDS:
                # Both facts describe a forced post; they disagree about WHICH,
                # and the two answers live in different pools.
                legality_issues.append(
                    f"{preamble} A row booked {action.kind!r} is never a "
                    f"{action.forced_bet_type}, so the kind and the forced-post "
                    "type name two different forced posts, which are laid out "
                    "into different pots and answer the ante mode differently. "
                    "The pot model will not choose which one the recording "
                    f"meant. It is derived from its kind, as an ordinary "
                    f"{action.kind}. Either correct the action kind or correct "
                    "the forced post field in Edit actions."
                )
            else:
                legality_issues.append(
                    f"{preamble} A {action.kind} is never a forced post, so the "
                    "kind and the forced-post type describe two different "
                    "events -- chips the seat chose to put in, or chips the "
                    "room required -- and the pot model will not choose which "
                    "one the recording meant. It is derived as an ordinary "
                    f"{action.kind}. Either correct the action kind or clear "
                    "the forced post field in Edit actions."
                )

        if action.kind in _COMMITMENT_KINDS:
            contributions[action.player] += amount
            is_live_bet = _is_live_money(action)
            # Live money buys a place in a pot layer and can come back as an
            # uncalled bet. Dead money -- antes and dead blinds -- is owed to the
            # table: it joins the main pot whole and is never returnable.
            if is_live_bet:
                live_contributions[action.player] += amount
            elif _is_ante_post(action):
                ante_contributions[action.player] += amount
            else:
                other_dead_contributions[action.player] += amount
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
                "any straddle) for this hand. Marking the post dead -- with "
                "Forced post or with Post status -- silences this sentence "
                "without answering it and moves chips out of the pot every "
                "other seat was answering, so do that only if the post really "
                "was a dead one."
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
    # THE DECLARATION GATE. Decided over the whole recording, from the ante rows
    # alone, so the verdict cannot depend on the order two forced rows happen to
    # be listed in -- the same reasoning ``_unreadable_forced_posts`` records for
    # the blind-structure refusal. A refused hand still derives every figure; it
    # simply cannot present as legal. See ``_resolve_ante_mode``.
    resolved_mode, ante_mode_refusals = _resolve_ante_mode(
        ante_mode,
        [name for name in player_order if ante_contributions[name] > 0],
    )
    legality_issues.extend(ante_mode_refusals)

    # RULING 3, AS ONE BRANCH OF TWO LINES. The mode selects which dead chips the
    # cap governs, and nothing else about the layering changes: the live ladder,
    # the cascade, the eligibility rules and the abandoned-excess rule are all
    # mode-independent. Writing the branch here rather than inside ``_build_pots``
    # keeps the layering itself free of the mode, so there is exactly one place a
    # future edit could leak the exemption into the wrong pool.
    #
    # Note what does NOT move into the uncapped pool under SINGLE_PAYER:
    # ``other_dead_contributions`` and the external dead money. Rule 3's last
    # clause leaves non-ante forced posts on their existing treatment, and this is
    # an ANTE mode.
    if resolved_mode == AnteMode.SINGLE_PAYER_TABLE_ANTE:
        capped_dead = dict(other_dead_contributions)
        uncapped_dead = dict(ante_contributions)
    else:
        capped_dead = {
            name: ante_contributions[name] + other_dead_contributions[name]
            for name in player_order
        }
        uncapped_dead = {name: _ZERO for name in player_order}
    dead_contributions = {
        name: ante_contributions[name] + other_dead_contributions[name]
        for name in player_order
    }

    # The figures decide different questions and must stay apart.
    # ``settled_contributions`` is LIVE money that stuck: it is what a seat chose
    # to wager, so it is the only thing that cuts a LIVE boundary and the only
    # thing that decides who may contest a live band. ``capped_dead`` is dead
    # money the cap governs: it starts in the LOWEST layer, and under the amended
    # rule 2 the part of it above the smallest total commitment among that
    # layer's eligible seats rises into a layer of its own, eligible by total.
    # ``uncapped_dead`` is the consolidated table ante, which sits whole in the
    # main pot. ``dead`` -- the UNATTRIBUTED, operator-typed dead money -- is in
    # the CAPPED pool under every mode, which is the operator's fifth ruling: it
    # used to join the main pot whole and buy the shortest seat at the table the
    # lot of it.
    raw_pots = _build_pots(
        player_order,
        settled_contributions,
        capped_dead,
        uncapped_dead,
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
            resolved_mode == AnteMode.SINGLE_PAYER_TABLE_ANTE,
        )
    )
    # A hand carrying antes with no declared mode is refused, not merely warned,
    # so the refusal above already blocks it. The pot figures published beside
    # that refusal are derived under the capped rule; the note names it so that
    # an operator reading the layers knows which reading they are looking at.
    if ante_mode_refusals:
        warnings.append(
            "The pot layers shown are derived under the capped (PER_PLAYER) "
            "reading of this hand's antes, which is the strict direction and "
            "what this product derived before the ante mode existed. They are "
            "not a decision about the mode and this hand is not study-ready "
            "until the mode is declared."
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
    ante_mode: str | None = None,
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
                # Carried verbatim, None included. ``None`` is what an
                # unspecified ``Post status`` and a NULL column both look like,
                # and it is not the same fact as a stated ``live`` -- only a
                # stated one can contradict the forced-bet name beside it.
                is_live_post=(
                    None
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
        ante_mode=ante_mode,
    )


def declared_ante_mode(value: str | None) -> AnteModeName | None:
    """The one definition of "an ante mode was declared", for every caller.

    The direct analogue of ``blind_structure``, and it exists for the same
    reason: every surface that stores, reads, neutralises or displays the
    declaration has to agree about whether one was made, or they will disagree
    about whether a hand is study-ready. Persisted callers pass the
    ``hand_settlements.ante_mode`` column straight in.

    An empty string is read as ABSENT rather than refused, because that is what a
    NULL column round-tripped through a text field looks like, and "undeclared"
    is a state the product already handles loudly. Anything else that is not a
    known mode is refused here rather than silently dropped: dropping it would
    turn a corrupt declaration into a missing one, and the missing one derives
    under a different rule.
    """

    if value is None or value == "":
        return None
    if value not in AnteMode.ALL:
        raise LedgerError(
            f"Unknown ante mode {value!r}; expected one of {', '.join(AnteMode.ALL)}."
        )
    return value


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


def _named_post_liveness(action: LedgerAction) -> bool | None:
    """Whether the row NAMES itself live or dead, ignoring its post status.

    A forced-bet name carries a liveness: ``big_blind`` is a live structural bet
    and ``dead_blind`` and every ante are money owed to the table. A KIND carries
    one only for ``ante``, which is dead by definition. ``post_blind`` names the
    species and not the liveness -- a blind is legitimately either -- and
    ``all-in`` names neither, so both return None and the post status stands
    alone and unchallenged there.

    Returns None where nothing on the row names a liveness, which is the only
    honest answer and the one that raises nothing. Read against the label
    DIRECTLY rather than through ``_readable_forced_bet_type``, because that
    function asks this one and the two would otherwise chase each other; the
    caller has already settled the kind-versus-label question before asking.
    """

    if action.forced_bet_type in _LABEL_IS_LIVE_POST:
        return _LABEL_IS_LIVE_POST[action.forced_bet_type]
    if action.kind == "ante":
        return False
    return None


def _forced_bet_row_conflict(action: LedgerAction) -> str | None:
    """Which of the row's own fields contradict each other, or None.

    THE ONE PLACE a row is judged self-contradictory. Three operator-supplied
    facts sit on one row of the hand editor -- the action ``kind``, the "Forced
    post" name, and the "Post status" -- and each pair of them can state
    incompatible things about the same chips. Commit 1f8f01d ruled that such a
    pair is REFUSED rather than silently resolved, and applied the ruling to the
    kind-versus-label pair only. This applies it to the liveness pair as well,
    which is the axis the reported defect walked through.

    ``_FORCED_BET_KIND_CONFLICT`` -- the kind and the label name different
    events. A kind that can never be a forced post (``bet``, ``call``, ``raise``,
    and the kinds that commit nothing) can carry no forced-bet name at all, and a
    kind that IS a forced post has already named its species, so a name from the
    other species is not a refinement: an ``ante`` typed ``straddle`` and a
    ``post_blind`` typed ``ante`` each state two incompatible things.

    ``_FORCED_BET_STATUS_CONFLICT`` -- the row's stated post status disagrees
    with the liveness the row already named. ``big_blind`` marked dead and
    ``dead_blind`` marked live are the same contradiction from opposite ends, and
    both reach the same place: a live structural post stops being one, the
    blind-structure refusal it raised goes quiet, and the chips move from the
    pool that answers the wager level to the pool owed to the table.

    ONLY A STATED STATUS CAN CONFLICT. ``is_live_post`` is None when the
    recording said nothing, and an unstated status contradicts nothing -- it
    reads as live because that is what it has always read as. This is why the
    field is carried as a tri-state rather than collapsed at the boundary.
    """

    if action.forced_bet_type is not None:
        if action.kind not in _FORCED_POST_CAPABLE_KINDS:
            return _FORCED_BET_KIND_CONFLICT
        allowed = _KIND_FORCED_BET_TYPES.get(action.kind)
        if allowed is not None and action.forced_bet_type not in allowed:
            return _FORCED_BET_KIND_CONFLICT
    named = _named_post_liveness(action)
    if (
        action.is_live_post is not None
        and named is not None
        and bool(action.is_live_post) is not named
    ):
        return _FORCED_BET_STATUS_CONFLICT
    return None


def _readable_forced_bet_type(action: LedgerAction) -> str | None:
    """The row's forced-bet label, or None where the row contradicts itself.

    THE ONE PLACE the label is turned into something the rest of the module may
    read, so that every pool -- forced, live-structural, ante -- reaches the same
    verdict about the same row and no single predicate can be talked out of its
    kind on its own.

    Unreadable means the row derives exactly as it would with the contradicting
    fields unstated, which is the strict direction and the one that cannot be
    steered. The label is not thrown away silently --
    ``_forced_bet_row_conflict`` is the same reading turned the other way round,
    and the reducer refuses the hand on it.
    """

    if action.forced_bet_type is None:
        return None
    if _forced_bet_row_conflict(action) is not None:
        return None
    return action.forced_bet_type


def _reads_as_live_post(action: LedgerAction) -> bool:
    """Whether the row's post status leaves it live, for the pools that ask.

    An unstated status reads as live, which is what the field's old ``True``
    default already did. A status that CONTRADICTS the liveness the row named is
    one of the two facts in the contradiction, so it is set aside with the label
    and the row falls back to the unstated reading -- otherwise the operator
    could still take a refusal off a row by adding a second field that disagrees
    with the first, which is the whole exposure being closed here.
    """

    if _forced_bet_row_conflict(action) == _FORCED_BET_STATUS_CONFLICT:
        return True
    return action.is_live_post is not False


def _mislabelled_forced_bet(action: LedgerAction) -> bool:
    """Whether this row carries a forced-bet name the rest of the row denies.

    Two operator-supplied facts that contradict each other. The hand editor
    offers the "Forced post" and "Post status" selectboxes on EVERY action row,
    so ``call`` typed ``big_blind``, ``fold`` typed ``ante``, ``ante`` typed
    ``dead_blind`` and ``big_blind`` marked dead are each one mis-click away, and
    an importer can write the same pairs.

    The reducer derives such a row from its KIND, which is the strict direction
    and is byte-identical to the row with the contradicting fields unstated, and
    then refuses the hand. It does not pick the other reading and it does not
    stay quiet: the two readings are different events -- one is chips the seat
    chose to wager and the other is chips the room required, or one is money owed
    to the table and the other is a wager the table must answer -- and they give
    different pots. Choosing between two operator-supplied facts is exactly what
    ``_resolve_ante_mode`` refuses to do for the ante mode, for the same reason.

    The cross-species half of this is what stops the kind rule from opening a new
    hole where it closes one. Deriving an ``ante`` typed ``dead_blind`` from its
    kind without saying so would discard a fact the operator entered, and a
    ``post_blind`` typed ``ante`` -- which used to raise the undeclared-ante-mode
    refusal by joining the ante pool -- would go quiet the moment its kind stopped
    letting it in. Neither row is answerable; both are reportable.

    A row whose only contradiction is between its KIND and its post status --
    an ``ante`` marked live, which carries no name at all -- is not covered here,
    because there is no name to call mislabelled. ``_forced_bet_row_conflict`` is
    what the reducer reads, and it reports that row too.
    """

    return (
        action.forced_bet_type is not None
        and _readable_forced_bet_type(action) is None
    )


def _is_forced_post(action: LedgerAction) -> bool:
    """Whether this commitment was posted under duress rather than chosen.

    Forced posts are the rows a seat has no say in, so they carry no information
    about what the seat would have done -- and, for the ordering question below,
    they are the rows a recording may legitimately list in any order relative to
    one another.

    THE KIND DECIDES, THE TYPE REFINES. ``forced_bet_type`` is read through
    ``_readable_forced_bet_type``, so it is read only on a kind that could be a
    post in the first place and only where it names something inside that kind's
    own species. A label is not allowed to turn a voluntary action into a forced
    one, and it was: any type at all on a ``call`` excused it from the preflop
    wager floor, and a dead type on one moved its chips out of the live pool as
    well. It is not allowed to move a post between species either -- that is the
    half that let a relabelled ante switch off the ante-mode declaration gate.
    See ``_mislabelled_forced_bet``, which refuses the contradiction rather than
    letting either reading win silently.
    """

    return (
        action.kind in _FORCED_POST_KINDS
        or _readable_forced_bet_type(action) is not None
    )


def _is_ante_post(action: LedgerAction) -> bool:
    """Whether this commitment is an ANTE -- the one pool the ante mode governs.

    THE KIND PUTS THE ROW IN THE POOL AND NO LABEL TAKES IT OUT. ``kind='ante'``
    is an ante whatever the forced-post selectbox says, because that is the
    ledger's own word for the thing the ante mode governs. Reading the label
    ahead of the kind meant tagging the only ante row ``straddle``,
    ``big_blind`` or ``dead_blind`` emptied the ante pool, which switched OFF the
    undeclared-ante-mode refusal and flipped a hand the product must refuse to
    accepted -- without moving a chip, so no cross-check could see it. The mode
    declaration is the thing standing between an operator and a silently wrong
    pot; it cannot be optional at the click of a selectbox.

    The label still decides where the kind says nothing. ``all-in`` names no
    species of post, so ``forced_bet_type='big_blind_ante'`` on a short ante is
    the only signal there is -- and it must stay readable, or worked example (f)
    silently reverts to the capped answer on every recording that spells its
    short ante that way. ``post_blind`` names the blind species, so a label
    cannot move it into the ante pool either; that direction used to be the loud
    one, and ``_mislabelled_forced_bet`` keeps it loud now that the kind decides.

    A row the recording says nothing about falls back to its kind, which is the
    only signal there is and is what every existing construction supplies.
    """

    if action.kind not in _FORCED_POST_CAPABLE_KINDS:
        return False
    readable = _readable_forced_bet_type(action)
    if readable is not None:
        return readable in _ANTE_FORCED_BET_TYPES
    return action.kind == "ante"


def _resolve_ante_mode(
    declared: str | None, ante_posters: Sequence[str]
) -> tuple[AnteModeName, list[str]]:
    """Validate the declared ante mode against the antes this hand contains.

    Returns the mode the layering will run under and the refusals the hand must
    carry. AMBIGUOUS HANDS ARE REFUSED, NEVER INFERRED -- the operator ruled
    against inference explicitly.

    THE THREE REFUSALS, each naming the missing declaration and the clearing
    action, in the same register as the undeclared blind structure:

    1. Antes present, no mode declared. The commonest one, and the whole of the
       migration impact: every hand already in the store that contains an ante
       lands here on the day the column ships.
    2. Mode declared ``NONE`` but antes were posted. A contradiction between two
       operator-supplied facts; the model refuses rather than deciding which of
       them to believe.
    3. Mode declared ``SINGLE_PAYER_TABLE_ANTE`` but two or more seats posted
       antes. The declaration says ONE seat posts for the table. Two anteing
       seats under that declaration is a hand mixing a consolidated ante with
       personal ones, and no defensible answer exists: capping all of them breaks
       worked example (f), capping none breaks (e), and splitting them requires
       knowing which post was the consolidated one -- which is exactly the
       inference the ruling forbids. Flagged as a coverage limitation, not
       answered.

    WHAT A REFUSED HAND DERIVES, and why it is not a fourth mode. A refusal is
    reported as a legality issue, exactly like the undeclared blind structure:
    the hand stays fully inspectable, every chip figure is still derived, and it
    simply may not present as legal, reconciled or study-ready. The layering runs
    under the CAPPED rule while the refusal stands, for two reasons. It is the
    strict direction -- capping can only ever reduce a short seat's take, never
    manufacture an overpayment -- and it is byte-for-byte what this reducer
    already did before the mode existed, so a stored hand's displayed figures do
    not move underneath the operator on the day it blocks. That is a derivation
    the hand is blocked on, not an inferred declaration: nothing writes a mode,
    nothing clears the refusal, and the only thing that clears it is the operator
    declaring the mode.

    A hand with no antes at all needs no declaration and is never refused.
    ``NONE`` is not a guess for such a hand, it is the only thing it can be.
    """

    posters = sorted(dict.fromkeys(ante_posters))
    named = ", ".join(repr(name) for name in posters)

    if declared is None:
        if not posters:
            return AnteMode.NONE, []
        return AnteMode.PER_PLAYER, [
            f"{UNDECLARED_ANTE_MODE_PREFIX} ({named}) but no ante mode is "
            "declared, so the pot model cannot say whether each ante is capped "
            "against the shortest seat's total commitment or is a consolidated "
            "table ante that is not. The two give different pots on the same "
            "recording and the action line cannot tell them apart. Declare the "
            "ante mode (NONE, PER_PLAYER or SINGLE_PAYER_TABLE_ANTE) for this "
            "hand, alongside the blind structure and the rake policy."
        ]

    if declared not in AnteMode.ALL:
        raise LedgerError(
            f"Unknown ante mode {declared!r}; expected one of "
            f"{', '.join(AnteMode.ALL)}."
        )

    if declared == AnteMode.NONE:
        if not posters:
            return AnteMode.NONE, []
        return AnteMode.PER_PLAYER, [
            f"The declared ante mode is NONE, but {named} posted an ante. One of "
            "the two is wrong and the pot model will not choose which. Either "
            "correct the ante rows in Edit actions, or declare the ante mode "
            "this hand was actually dealt under (PER_PLAYER, or "
            "SINGLE_PAYER_TABLE_ANTE for a big-blind or button ante)."
        ]

    if declared == AnteMode.SINGLE_PAYER_TABLE_ANTE and len(posters) > 1:
        return AnteMode.PER_PLAYER, [
            "The declared ante mode is SINGLE_PAYER_TABLE_ANTE, which means one "
            f"seat posts a consolidated ante for the table, but {len(posters)} "
            f"seats posted antes ({named}). A consolidated table ante and "
            "personal antes obey different capping rules and the model will not "
            "guess which post is which. Declare PER_PLAYER if every one of these "
            "is a personal ante, or correct the recording in Edit actions so "
            "that only the consolidated table ante is typed as an ante."
        ]

    return declared, []


def _is_live_structural_post(action: LedgerAction) -> bool:
    """Whether this commitment is a live forced bet that sets the wager level.

    Two ways a recording can say so, because only one of them is always
    available: the action KIND (``post_blind``, the shape the hand editor and the
    manual writer produce), or the recorded FORCED-BET TYPE (the shape that
    survives when a post which took its poster's last chip was booked as
    ``all-in``). Requiring the kind alone is what let the second shape past.

    WHERE THE RECORDING NAMES THE FORCED BET, THAT NAME DECIDES.  A row spelled
    ``post_blind`` and typed ``dead_blind`` is a dead post, and a
    ``post_blind`` typed ``big_blind`` is a live one, whatever order the
    predicate happens to ask its questions in.  ``ante`` is never structural
    whatever type is carried, because an ante sets no wager level -- that is what
    ``_LIVE_STRUCTURAL_FORCED_BETS`` exists to say.

    THE NAME AND THE POST STATUS ARE THE SAME FACT STATED TWICE, so they are not
    combined here, they are RECONCILED first.  The two operator-facing
    selectboxes in the hand editor can disagree, and this predicate used to
    resolve the disagreement silently by ANDing them: a live big blind marked
    dead simply stopped being structural, which took the blind-structure refusal
    off a hand that needed it and moved its chips into the pool owed to the
    table.  ``_forced_bet_row_conflict`` now refuses that row instead, and
    ``_reads_as_live_post`` is what this reads in place of the raw field.

    Like every other reader of the name, this one asks the KIND first: a
    ``bet``, ``call`` or ``raise`` answers a wager level and can never be the
    thing that sets one, whatever type it carries.  For the same reason it reads
    the name through ``_readable_forced_bet_type``, so an ``ante`` typed
    ``big_blind`` cannot promote itself into the pool that sets the level, and a
    ``post_blind`` typed ``ante`` -- which names nothing inside the blind species
    -- stays the structural post its kind says it is instead of demoting itself
    out of the unreadable-post refusal.
    """

    if action.kind not in _FORCED_POST_CAPABLE_KINDS or action.kind == "ante":
        return False
    if not _reads_as_live_post(action):
        return False
    readable = _readable_forced_bet_type(action)
    if readable is not None:
        return readable in _LIVE_STRUCTURAL_FORCED_BETS
    return action.kind == "post_blind"


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


def _layer_cap(
    commitment: Mapping[str, Decimal], eligible: Sequence[str]
) -> Decimal:
    """Rule 2's cap: the SMALLEST total commitment among a layer's eligible seats.

    One line, named, because it is the sentence the whole cascade turns on and
    because reading it as the LARGEST makes every cap vacuous while leaving chip
    conservation, eligibility and the boundary rules all intact. A rule that can
    be silently inverted deserves a name a reader can check against the
    specification.
    """

    return min(commitment[name] for name in eligible)


def _build_pots(
    order: Sequence[str],
    live_settled: Mapping[str, Decimal],
    capped_dead: Mapping[str, Decimal],
    uncapped_dead: Mapping[str, Decimal],
    put_up: Mapping[str, Decimal],
    folded: set[str],
    external_dead: Decimal,
) -> list[dict]:
    """Lay the pot out under the operator's model, rules 2 and 4 AS AMENDED.

    THE MODEL, in four sentences.  Rules 2 and 4 are the ones the operator has
    changed; the other two are unchanged, and worked examples (a)-(e) must come
    out to the chip exactly as they did before:

      1. Layer boundaries are cut at the distinct LIVE contribution levels,
         measured after uncalled-bet refunds have been returned.  Live money is
         what a player CHOSE to wager.  Forced posts are never live.
      2. Dead money goes into the LOWEST layer, but EACH CONTRIBUTOR'S CAPPED
         dead chips count into a layer only up to the smallest TOTAL commitment
         among that layer's eligible seats.  The excess rises into the layer
         above, eligible to the seats whose own total reached above that cap.
         Which chips are capped is decided by the declared ante mode, ABOVE this
         function: see ``AnteMode`` and ``build_hand_ledger``.  ``uncapped_dead``
         is the consolidated table ante, which is table money and sits whole in
         the main pot.
      3. A seat is eligible for a layer if its own LIVE contribution reaches that
         layer's level.  Every unfolded seat that put ANY chip up -- live or dead
         -- is eligible for the main pot.
      4. A folded seat's chips stay in the layers they reached and it is eligible
         for none.  Its forced post that no surviving seat could cover is
         ABANDONED to whoever wins the layer it stopped in.

    WHY THE POOLS ARRIVE ALREADY SPLIT.  This function takes ``capped_dead`` and
    ``uncapped_dead`` rather than a mode, deliberately.  The mode governs which
    chips go in which pool and NOTHING else -- not the cascade, not eligibility,
    not the abandoned excess -- so the branch lives in one place upstream and the
    layering below is mode-free.  A reducer that branched on the mode down here
    would have to decide, at every one of half a dozen sites, whether this
    particular line is about antes or about dead money, and getting one of them
    wrong loses the cap on a whole family silently: a SINGLE_PAYER hand carrying
    a dead blind runs BOTH rules at once, on disjoint pools.

    WHAT THE AMENDMENTS FIXED.

    Rule 2 used to be unconditional: every dead chip went whole into the lowest
    layer.  That paid a seat whose ENTIRE commitment was smaller than an
    opponent's forced post more than the table had matched of it -- a 60-chip
    seat against three 100-chip antes collected 360 where the table had matched
    240 of it, and a 40-chip seat against five 100-chip antes collected 540.

    Externally declared dead money was then left OUTSIDE that cap, on the
    argument that money with no contributing seat has no operand to be capped
    against.  It has one: the seat trying to collect it.  Uncapped, a seat that
    committed 2 chips could be paid 312 -- settled, balanced, legal and
    warning-free -- and it happened on 11,038 of 111,426 measured assignments.
    The operator's fifth ruling puts it in the capped pool under every mode, and
    ``min(external, cap)`` below is that ruling.  It is not an ante, so no mode
    exempts it.

    Rule 4 used to stop mid-sentence: the excess "rises into the layer above,
    eligible to the seats whose own total reached above that cap", and named no
    layer when no surviving total reached above it.  The chips went to the top
    layer and the hand was refused as not study-ready, on 7.79% of
    tournament-shaped hands, with -- in the refusal's own words -- no correction
    to the recording that could clear it.  The operator has ruled: such a post
    belongs to the pot.  A button that antes 50,000 and folds against two 20,000
    stacks is now one settleable 90,000 pot.

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
    eligibility for every live band.  ``capped_dead`` is each seat's dead posts
    that the cap governs and ``uncapped_dead`` is the consolidated table ante the
    declared mode exempts; between them they are every dead chip a seat posted,
    and the mode decided the split before this function was called.
    ``external_dead`` is dead money declared for the table with no contributing
    seat, and it is in the CAPPED pool under every mode.  ``put_up`` is what a
    seat committed BEFORE any uncalled money came back, and it answers one
    question: did this seat play the hand at all.  A seat whose sole live post
    was returned -- because the only money facing it was a forced post nobody had
    a chip left to call -- still played, and still contests the pot it played
    for.

    WHY THE UNCAPPED POOL IS SAFE, and why it cannot strand a chip.  The
    consolidated ante enters the FIRST dead layer -- the main pot -- and its
    poster is by construction eligible for that layer, because it put a chip up.
    So "a seat that wins every layer it is eligible for cannot lose chips" still
    holds for the poster: worked example (b) is exactly it, where the ante-only
    seat wins the main pot and comes out at zero.  It is also the one thing that
    must never be routed through the cascade: capping table money against the
    shortest blind is the defect worked example (f) names.

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

    dead_total = (
        sum(capped_dead.values(), _ZERO)
        + sum(uncapped_dead.values(), _ZERO)
        + external_dead
    )
    contenders = [name for name in order if name not in folded]
    # Rule 3, second sentence: playing the hand is measured by what a seat PUT UP.
    played = tuple(name for name in contenders if put_up.get(name, _ZERO) > 0)

    # Rule 2's cap operand: a seat's WHOLE commitment, both dead pools included.
    # This is mode-INDEPENDENT and must stay so. The mode says which of a seat's
    # chips the cap governs; it never says a seat committed fewer chips than it
    # did, and leaving the consolidated ante out of its poster's own total would
    # lower the cap every other contributor is measured against.
    # See "WHERE ``put_up`` REACHES THE CAP" above.
    commitment: dict[str, Decimal] = {}
    for name in order:
        total = (
            live_settled.get(name, _ZERO)
            + capped_dead.get(name, _ZERO)
            + uncapped_dead.get(name, _ZERO)
        )
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
        placed_external = _ZERO
        previous_cap = _ZERO
        first_layer = True
        # The caps strictly increase and each step strictly shrinks the eligible
        # set, so the model needs at most one pass per seat. The bound exists so
        # that a cap rule which does NOT increase fails loudly instead of
        # hanging -- a mutation that breaks the cascade must not become a
        # timeout, because a timeout is indistinguishable from a slow machine.
        for _ in range(len(order) + 1):
            cap = _layer_cap(commitment, eligible_now)
            layer_dead = _ZERO
            # Named apart from the live ladder's ``contributors`` because they are
            # different types -- this one is accumulated -- and sharing the name
            # made the two ladders look like one loop to a reader and to mypy.
            dead_contributors: list[str] = []
            for name in order:
                own = capped_dead.get(name, _ZERO)
                share = min(own, cap) - min(own, previous_cap)
                if share > 0:
                    layer_dead += share
                    placed[name] += share
                    dead_contributors.append(name)
            # RULING 5. Externally declared dead money runs the SAME cascade as a
            # recorded dead post. It has no contributing seat, so it has no
            # commitment of its own to be capped by -- but the operand rule 2
            # actually uses is the COLLECTING seat's total, and that exists for
            # every seat. Left uncapped it paid a 2-chip seat 312 chips.
            external_share = (
                min(external_dead, cap) - min(external_dead, previous_cap)
            )
            if external_share > 0:
                layer_dead += external_share
                placed_external += external_share
            if first_layer:
                # RULING 3, SINGLE_PAYER_TABLE_ANTE. The consolidated ante is
                # TABLE MONEY: it is placed whole in the first dead layer -- which
                # is the main pot, since every unfolded seat that put a chip up is
                # eligible for it -- and never enters the cascade at all. Under
                # every other mode this pool is empty and the line does nothing,
                # which is what makes rule 3's "retained" clause checkable rather
                # than merely asserted.
                for name in order:
                    own = uncapped_dead.get(name, _ZERO)
                    if own > 0:
                        layer_dead += own
                        if name not in dead_contributors:
                            dead_contributors.append(name)
            remaining = sum(
                (capped_dead.get(name, _ZERO) - placed[name] for name in order),
                _ZERO,
            ) + (external_dead - placed_external)
            next_eligible = tuple(
                name for name in contenders if commitment[name] > cap
            )
            if remaining > 0 and not next_eligible:
                # RULING 4. Nothing above to rise into, so the excess is ABANDONED
                # to whoever wins this layer. It can only ever be a FOLDED seat's
                # post or external money -- an unfolded seat's own capped dead is
                # at most its own commitment, and the cascade's last cap is the
                # largest surviving commitment -- so this never strands a seat's
                # own chips above it.
                #
                # This used to be the conventional poker answer published under a
                # refusal, because the operator's rule 2 stopped mid-sentence
                # here. The operator has now ruled that such a post belongs to the
                # pot, so the chips are unchanged and the refusal is withdrawn --
                # for THIS shape only. See ``_unruled_dead_money_warnings`` for
                # what the withdrawal deliberately leaves standing.
                for name in order:
                    over = capped_dead.get(name, _ZERO) - placed[name]
                    if over > 0:
                        layer_dead += over
                        placed[name] = capped_dead.get(name, _ZERO)
                        if name not in dead_contributors:
                            dead_contributors.append(name)
                over_external = external_dead - placed_external
                if over_external > 0:
                    layer_dead += over_external
                    placed_external = external_dead
                remaining = _ZERO
            first_layer = False
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
    single_payer_table_ante: bool,
) -> list[str]:
    """Name the dead-money shapes the operator's rule 2 still does not decide.

    WHAT THIS USED TO BE, AND WHY IT IS NARROWER NOW.  The previous round refused
    every hand in which a main-pot seat could be paid a forced post larger than
    its own total commitment, because rule 2 was unconditional -- all dead money
    into the lowest layer -- and nothing in the four worked examples said whether
    a 40-chip seat should collect five opponents' 100-chip antes.  It published
    540 and declined to call the hand study-ready.

    THE OPERATOR HAS RULED, TWICE.  Amended rule 2 caps each contributor's dead
    chips at the smallest TOTAL commitment among the layer's eligible seats and
    lifts the excess into a layer eligible to the seats whose own total reached
    above that cap.  That hand is now 240 to the short seat, which is exactly what
    the table matched of it.  Ruling 4 then answered the leftover: a forced post
    from a seat that FOLDED, which no surviving seat could cover, belongs to the
    pot.  So two families of refusal are withdrawn -- and only those two.
    Removing the whole guard because most of what it covered became decidable is
    how a defect returns wearing the fix's clothes.

    WHAT SURVIVES, PRECISELY, AND WHY.

    (1) EXTERNAL DEAD MONEY UNDER ``SINGLE_PAYER_TABLE_ANTE``, WHERE RULING 5
        SELECTS TWO RULES AND NAMES NEITHER.  Ruling 5 says operator-typed dead
        money is capped "under whichever rule the hand's ante mode selects".
        Under ``NONE`` and ``PER_PLAYER`` the mode selects exactly one rule and
        there is nothing to decide.  Under ``SINGLE_PAYER_TABLE_ANTE`` it selects
        TWO -- uncapped for the consolidated ante, capped for everything else --
        and the seven worked examples contain no external dead money at all, so
        nothing in the acceptance set constrains the choice.

        THE READING TAKEN IS CAPPED, for three reasons.  The consolidated ante is
        by definition a recorded post by a named seat ("one seat posts a
        consolidated ante for the table") and external money has no seat, so it
        is not that.  Capping is the strict direction: it can only reduce a short
        seat's take, never manufacture an overpayment.  And reading it the other
        way makes ruling 5 a no-op on every SINGLE_PAYER hand, which cannot be
        what a ruling written to fix 11,038 over-payments meant.

        It is DISCLOSED rather than assumed, and only where it moves a chip --
        that is, where the declared amount exceeds the smallest total commitment
        among the seats contesting the main pot, so the two readings genuinely
        differ.  Reversing it is a one-line change (move the external amount into
        the uncapped pool) under which all seven worked examples still pass,
        which is exactly why it must be ruled on rather than inherited.

    (1a) WHAT IS NOT WARNED, AND WHY IT IS A COMPOSITION RATHER THAN A GUESS.
        External dead money above EVERY surviving commitment has no layer to rise
        into, and takes the same abandoned route a folded seat's post now takes.
        That is not an extra assumption: ruling 5 says external money is treated
        "exactly like recorded dead money", and ruling 4 settles recorded dead
        money in precisely this shape.  Composing two rulings the operator wrote
        is not the same thing as inventing a third.  There is also no alternative
        -- the chips cannot be returned and a layer no seat can win is refused --
        so a warning here would name no clearing action, which is the failure mode
        the round-20 note itself recorded.

        A FOLDED SEAT'S OWN POST IN THIS SHAPE NO LONGER WARNS EITHER.  That is
        ruling 4, and worked example (g) is the case: a button that antes 50,000
        and folds against two 20,000 stacks settles as one 90,000 pot.  The
        layering was ALREADY producing that pot; what blocked the hand was this
        note, whose own text admitted no correction to the recording could clear
        it.  So the change here is a study-readiness change and not a layering
        change, which is worth saying plainly because a fix aimed at
        ``_build_pots`` would have been aimed at the wrong function.

    (2) A SEAT CONTESTING A POT IT CONTRIBUTED NOTHING TO.  Rule 3's "put ANY
        chip up" is read BEFORE the uncalled bet came back, so a seat whose only
        live post was returned still contests the dead money; reading it after
        the refund would make a hand that seat won unrecordable.  Rule 2's cap is
        written against a seat's TOTAL COMMITMENT, and for that one seat the
        post-refund total is zero.  The layering reads the cap at the same point
        rule 3 is read -- what the seat put up -- because reading it as zero caps
        every layer that seat is eligible for at zero and empties the main pot.
        That is a reading, not a ruling: the operator's sentences never have to
        choose a measurement point because none of the seven worked examples
        contains a refund.  It reaches only hands whose whole pot is dead money
        and exactly one seat wagered live, uncalled, so it is narrow -- and it is
        named rather than assumed.  It is also the single class in which an
        independent oracle written from the spec and this reducer disagree, and
        it is warned on 100% of its occurrences, which is what makes it a
        coverage limitation rather than a release blocker.

    (3) DEAD MONEY WITH NO UNFOLDED CONTRIBUTOR AT ALL is not warned here because
        it is REFUSED outright, in ``_build_pots``: rule 3 makes nobody eligible
        for the main pot, so there is no layer to award.  Ruling 4 does not reach
        it either -- worked example (g) has two surviving seats with 20,000 each
        -- so that refusal stands unchanged.

    WHAT IS NO LONGER AN ASSUMPTION.  Externally declared dead money used to sit
    in the lowest layer WHOLE and uncapped, so a seat with a commitment of 1 could
    win 1000 of it, and that was recorded here as a deliberate assumption rather
    than disclosed.  It was wrong: it produced 11,038 over-payments in 111,426
    measured assignments.  Ruling 5 puts it in the capped pool under every mode,
    and the note above covers only what is left -- the excess above every
    surviving commitment.

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

    # (1) External dead money on a SINGLE_PAYER hand, where the mode selects two
    # capping rules and ruling 5 does not say which one operator-typed money
    # falls under. Disclosed only where the two readings differ -- below the
    # floor they place the same chips in the same layer and there is nothing to
    # decide.
    #
    # A FOLDED SEAT'S POST WITH NOWHERE TO RISE, AND EXTERNAL MONEY IN THE SAME
    # SHAPE, ARE DELIBERATELY NOT TESTED HERE ANY MORE: ruling 4 settles the
    # first outright and composes with ruling 5's "exactly like recorded dead
    # money" for the second. Re-testing either would keep 7.79% of
    # tournament-shaped hands blocked on a refusal the operator withdrew.
    floor = min(commitment[name] for name in played)
    if single_payer_table_ante and external_dead > floor:
        notes.append(
            f"{_float(external_dead)} chips are declared as external dead money "
            "on a hand whose ante mode is SINGLE_PAYER_TABLE_ANTE, and that mode "
            "applies two different capping rules -- the consolidated table ante "
            "is not capped, every other dead chip is. The ruling does not say "
            "which one money with no seat behind it falls under, and none of the "
            "worked examples contains any. This hand is derived under the CAPPED "
            "reading, so the declared money reaches a seat only up to that seat's "
            f"own total commitment; the smallest among the seats contesting the "
            f"main pot is {_float(floor)}, so the reading moves chips here. If "
            "the declared money was meant to sit whole in the main pot like a "
            "table ante, the ante mode or the declared amount needs correcting in "
            "Edit settlement."
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
