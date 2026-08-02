"""Persistence-aware completed-hand accounting and reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import blake2s
from math import isclose

from poker_tracker.math.accounting import (
    HandLedger,
    LedgerError,
    RakePolicy,
    build_ledger_from_records,
)
from poker_tracker.persistence.completion import ASSUMPTION_DEPENDENCE_PREFIX
from poker_tracker.persistence.db import UNREADABLE_SETTLEMENT_PREFIX, PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    SettlementEntry,
    utc_now,
)

# Float-representation noise only. Every chip comparison below is derived from
# Decimal arithmetic, so this is a guard against round-tripping through float,
# never a licence to disagree about chips.
_FLOAT_TOLERANCE = 1e-9
#
# Why every comparison in this module uses _FLOAT_TOLERANCE and nothing else:
#
# There used to be a second, wider tolerance for the two RECORDED restatements
# of the rake policy -- ``settlement.rake_amount`` and ``settlement.net_pot`` --
# on the argument that they are the same policy read one rounding step apart
# (a recorded 3.3 where the ledger charged 3), and that
# ``persist_reconciliation`` rewrites both from the ledger anyway so the slack
# could never excuse anything real.
#
# The second half of that argument is only true on the settlement-editor path,
# which nulls both fields before ``persist_reconciliation`` rewrites them.
# ``import_session`` never calls ``persist_reconciliation`` at all, and every
# readiness surface reads through ``reconcile_persisted_hand``, so on an
# imported payload the recorded pair was never rewritten and was compared
# against a tolerance the SAME payload supplied: ``rake_rate``, ``rake_cap`` and
# ``rake_rounding_unit`` together reached ``gross_pot / 4``. One
# ``import_session`` call landed a hand recording a rake 24.5% of its own pot
# away from its own action line, presenting as reconciled, authoritative and
# study-ready with an empty blocker tuple -- and re-exported the forged pair.
#
# A tolerance whose width is set by the data it is judging is not a tolerance.
# There is no honest source for a recorded rake that disagrees with the policy
# stored beside it: this product writes both from the ledger, and a row that
# disagrees is either hand-edited or forged. The clearing action is the same
# single click the operator already has -- saving the settlement re-derives
# both figures -- so the strict comparison fails closed onto a reachable fix.
#
# The rake POLICY is a different problem, and comparing exactly does not touch
# it: ``rake_rate``, ``rake_cap``, ``rake_rounding_unit`` and ``no_flop_no_drop``
# move the DERIVED side of the hero-result and payout cross-checks rather than
# the tolerance, so any hero result in ``[-contribution, gross - contribution]``
# can be made to reconcile exactly. Declared dead money does the same thing in
# the opposite direction.
#
# That residue was mitigated for eight rounds by DISCLOSURE, raised from a list
# of per-field conditions ("rake_rate > 0", "dead_money > 0", ...). Every round
# found one more combination the list did not cover -- most recently a rounding
# unit at a zero rate -- because a field list can only ever describe the shapes
# already demonstrated. It is no longer used to decide anything.
#
# What decides now is DEPENDENCE, measured rather than enumerated. See
# ``_derive_assumption_dependence``: the same hand is cross-checked twice, once
# with the stored settlement assumptions and once with neutral ones, and the
# hand is assumption-dependent when removing the declaration changes the VERDICT
# (the neutral pass stops reconciling, or cannot be built) or the FIGURES (it
# reconciles too, but derives a different gross pot, rake, net pot, payout, or
# hero result). There is no field list in that rule, so there is no per-field
# hole left in it: a declared input that moves nothing is silent, and a declared
# input this hand's reported numbers rest on is named, measured, and blocking
# until the operator attests to that specific measurement of that specific
# declaration.

# Delta terms that are MAGNITUDES rather than signed movements, with the wording
# that says so. ``_ledger_deltas`` measures the payout term as the largest
# absolute per-seat change, because two seats' payouts can move in opposite
# directions under one declaration and there is no single direction to state; the
# other four terms are signed differences of a single figure. Rendering all five
# through one signed formatter made the same token mean opposite things.
_UNSIGNED_DELTA_TERMS: dict[str, Callable[[float], str]] = {
    "payout": lambda value: (
        f"the largest payout for any seat by {abs(value):g} chips"
    ),
}


@dataclass(frozen=True)
class AssumptionDependence:
    """One declared settlement input this hand's reconciliation rests on.

    ``code`` carries the measured chip movement, which is what makes an
    acknowledgement an attestation to a QUANTITY: the code recorded against
    0.01 chips of rake is a different string from the one covering 80.01 chips,
    so growing the pot under an unchanged policy cannot inherit the earlier
    attestation. Nothing compares policies field by field to notice that.
    """

    input_name: str  # "rake_policy" | "dead_money" | "settlement_assumptions"
    declared: str  # the stored value, for display
    neutral: str  # what it is compared against, for display
    deltas: tuple[tuple[str, float], ...]  # (figure, declared - neutral) in chips
    code: str  # stable, quantity-bearing acknowledgement key

    def describe(self) -> str:
        """The sentence the operator is asked to attest to, in the direction it states.

        ``deltas`` are DECLARED minus NEUTRAL, which is the right convention for
        the acknowledgement code -- that is what makes the code an attestation to a
        quantity -- but this sentence's subject is the REMOVAL of the declaration,
        so every signed term has to be negated to render it. It was not, and 5 of
        the 6 terms on a 50%-rake hand were printed backwards: the operator was told
        that withdrawing a rake which destroys 40 chips of their result would cost
        them 40 more, and the same string is the ACCOUNTING_ASSUMPTION_DEPENDENT
        blocker detail, the caption directly above 'Confirm this assumption', and a
        line handed to the coaching provider on an attested, study-ready hand. An
        operator who read it carefully and applied its own clearing action ("if the
        chips did not move that way, correct the declared winner, the rake policy or
        the dead money instead") would have gone and edited correct data.

        The repair is here and NOT in ``_ledger_deltas``: the delta values are
        embedded verbatim in the dependence code, so negating them there would lapse
        every stored attestation for a display defect.

        ``payout`` is deliberately not rendered as a signed movement.
        ``_ledger_deltas`` measures it as the largest ABSOLUTE change over all
        seats, because different seats can move in opposite directions and there is
        no single direction to state; printed through the signed formatter it always
        read "+", so a declared rake taking 40 off the hero's payout and 75 chips of
        declared dead money adding 75 to it printed the same token with opposite
        meanings. It is now worded as the magnitude it is.
        """
        movement = ", ".join(
            _UNSIGNED_DELTA_TERMS[name](value)
            if name in _UNSIGNED_DELTA_TERMS
            else f"{name} {_format_chips(-value)}"
            for name, value in self.deltas
        )
        head = (
            f"{self.input_name} is declared as {self.declared} (neutral: "
            f"{self.neutral}); removing that assumption "
        )
        if not movement:
            return head + "stops this hand reconciling without moving any reported figure"
        return head + f"moves {movement}"


@dataclass(frozen=True)
class AccountingReconciliation:
    ledger: HandLedger
    settlement: HandSettlement | None
    entries: tuple[SettlementEntry, ...]
    issues: tuple[str, ...]
    is_authoritative: bool
    # Additive. Existing consumers construct and read the five fields above
    # unchanged; nothing about the response shape they already depend on moves.
    assumption_dependence: tuple[AssumptionDependence, ...] = ()


@dataclass(frozen=True)
class _Declaration:
    """Every operator-declared settlement input one cross-check pass reads.

    This is the complete set of things ``_cross_check`` takes from what somebody
    DECLARED rather than from what the recording observed, and it is the unit the
    dependence rule neutralises. ``neutral()`` is the same hand with the whole
    declaration withdrawn.

    It exists because the previous neutralisation set was two values -- a rake
    policy and a dead-money amount -- named individually at the call site, which
    is a two-entry field list wearing a different hat. The declared POT AWARDS
    were outside it, and they are the largest declared input there is: on a
    reconstructed hand the CV exporter emits no settlement rows at all, so the
    winner of every pot is typed in by an operator, and one dropdown moved the
    reported hero result by the whole pot with nothing measuring it.

    Completeness is a property, not a promise:
    ``test_a_neutral_declaration_derives_a_ledger_from_the_recording_alone``
    mutates every settlement column and every settlement entry and asserts the
    fully neutral ledger does not move, so a future input added to the derived
    side without being added here fails a test rather than opening a hole.
    """

    rake: RakePolicy
    dead_money: float
    awards: tuple[tuple[SettlementEntry, str], ...]
    # The winners the RECORDING leaves no choice about, used in place of the
    # declared award rows when this declaration is the withdrawn one. See
    # ``_forced_winners``: withdrawing the awards to "nobody won anything" is not
    # a state any recording can produce, so it made every settled hand
    # award-dependent by construction.
    forced_winners: tuple[tuple[int, str], ...] | None = None

    @property
    def winners(self) -> dict[int, list[str]] | None:
        if self.forced_winners is not None:
            forced: dict[int, list[str]] = {}
            for index, identity in self.forced_winners:
                forced.setdefault(index, []).append(identity)
            return forced
        if not self.awards:
            return None
        winners: dict[int, list[str]] = {}
        for entry, identity in self.awards:
            winners.setdefault(entry.pot_index or 0, []).append(identity)
        return winners

    @property
    def odd_chip_order(self) -> tuple[str, ...]:
        if self.forced_winners is not None:
            # Every forced pot has exactly one eligible seat, so there is no odd
            # chip to order.
            return ()
        order: list[str] = []
        for _entry, identity in self.awards:
            if identity not in order:
                order.append(identity)
        return tuple(order)

    def without_awards(
        self, forced: tuple[tuple[int, str], ...] | None = None
    ) -> _Declaration:
        return _Declaration(
            rake=self.rake, dead_money=self.dead_money, awards=(), forced_winners=forced
        )

    def with_neutral_rake(self) -> _Declaration:
        return _Declaration(
            rake=_NEUTRAL_RAKE, dead_money=self.dead_money, awards=self.awards
        )

    def with_neutral_dead_money(self) -> _Declaration:
        return _Declaration(
            rake=self.rake, dead_money=_NEUTRAL_DEAD_MONEY, awards=self.awards
        )

    @classmethod
    def neutral(cls, forced: tuple[tuple[int, str], ...] | None = None) -> _Declaration:
        return cls(
            rake=_NEUTRAL_RAKE,
            dead_money=_NEUTRAL_DEAD_MONEY,
            awards=(),
            forced_winners=forced,
        )

    @property
    def is_neutral(self) -> bool:
        return (
            self.rake == _NEUTRAL_RAKE
            and self.dead_money == _NEUTRAL_DEAD_MONEY
            and not self.awards
        )


@dataclass(frozen=True)
class _HandRecords:
    """Every durable record one reconciliation reads, fetched exactly once.

    Both cross-check passes run against the same records, so the second pass
    costs one ledger build rather than a second round of five queries, and the
    two passes cannot disagree because they read the store at different moments.
    """

    hand: Hand
    players: list[HandPlayer]
    actions: list[Action]
    settlement: HandSettlement | None
    entries: tuple[SettlementEntry, ...]
    awards: tuple[tuple[SettlementEntry, str], ...]
    refunds: tuple[tuple[SettlementEntry, str], ...]
    flop_seen: bool
    hero_key: str | None

    @property
    def declared_rake(self) -> RakePolicy:
        if self.settlement is None:
            return RakePolicy()
        return RakePolicy(
            rate=self.settlement.rake_rate,
            cap=self.settlement.rake_cap,
            rounding_unit=self.settlement.rake_rounding_unit,
            no_flop_no_drop=self.settlement.no_flop_no_drop,
        )

    @property
    def declared_dead_money(self) -> float:
        return 0.0 if self.settlement is None else self.settlement.dead_money

    @property
    def declaration(self) -> _Declaration:
        return _Declaration(
            rake=self.declared_rake,
            dead_money=self.declared_dead_money,
            awards=self.awards,
        )


@dataclass(frozen=True)
class _CrossCheck:
    ledger: HandLedger
    issues: tuple[str, ...]

    @property
    def reconciles(self) -> bool:
        """The ledger verdict, without the stored ``status`` label.

        ``is_authoritative`` additionally requires ``settlement.status ==
        'reconciled'``, which is a persisted label rather than a property of the
        chips. The dependence rule compares chips against chips, so it uses this.
        """
        return (
            self.ledger.is_settled
            and self.ledger.is_balanced
            and self.ledger.is_legal
            and not self.issues
        )


def reconcile_persisted_hand(
    db: PokerDatabase,
    hand_id: int,
) -> AccountingReconciliation:
    """Build and cross-check one completed hand from durable records."""

    records = _load_hand_records(db, hand_id)
    checked = _cross_check(records, records.declaration)
    is_authoritative = (
        records.settlement is not None
        and records.settlement.status == "reconciled"
        and checked.reconciles
    )
    return AccountingReconciliation(
        ledger=checked.ledger,
        settlement=records.settlement,
        entries=records.entries,
        issues=checked.issues,
        is_authoritative=is_authoritative,
        assumption_dependence=_derive_assumption_dependence(records, checked),
    )


def attest_assumption(db: PokerDatabase, hand_id: int, code: str) -> bool:
    """Record an attestation to a dependence this hand CURRENTLY measures.

    The supported door to ``db.acknowledge_accounting_assumption``. The writer
    itself re-measures and refuses a code naming no current dependence (via a
    call-time import of ``reconcile_persisted_hand``, since this module imports
    ``db`` at module level), so a shape-valid fabrication supplied to the raw
    writer directly can no longer evict a genuine attestation or file a
    hand_corrections row for an attestation nobody made. This door is kept
    because measuring BEFORE calling the writer is what keeps the one caller's
    "the write was refused" branch reachable, which is where the honest error
    message lives -- and it fails closed: a refused attestation leaves the hand
    blocked.
    """
    measured = {
        item.code for item in reconcile_persisted_hand(db, hand_id).assumption_dependence
    }
    if code not in measured:
        return False
    return db.acknowledge_accounting_assumption(hand_id, code)


def persist_reconciliation(
    db: PokerDatabase,
    hand_id: int,
    *,
    status_when_valid: str = "reconciled",
) -> AccountingReconciliation:
    """Recompute summaries and persist a truthful reconciliation status."""

    result = reconcile_persisted_hand(db, hand_id)
    hand_players = db.fetch_players_by_hand(hand_id)
    # An uncalled-bet refund derived from an action line that never closed is not
    # a fact about the hand -- it is the arithmetic of a fold nobody made. Writing
    # it as a durable settlement row would file that fabrication in the store, and
    # would then have to be un-filed by hand once the missing action is recorded.
    # The hand is blocked either way; it just does not get a manufactured row.
    if not any(entry.entry_type == "refund" for entry in result.entries) and not (
        _unanswered_wager_issues(hand_players, result.ledger)
    ):
        players_by_key = {player.player_key: player for player in hand_players}
        derived_refunds = [
            SettlementEntry(
                hand_id=hand_id,
                entry_type="refund",
                player_key=identity,
                player_name=players_by_key[identity].player_name,
                amount=amount,
                entry_order=index,
            )
            for index, (identity, amount) in enumerate(
                (
                    (identity, amount)
                    for identity, amount in result.ledger.refunds.items()
                    if amount > 0
                ),
                start=1,
            )
            if identity in players_by_key
        ]
        if derived_refunds:
            db.replace_settlement_entries(
                hand_id, [*result.entries, *derived_refunds]
            )
            result = reconcile_persisted_hand(db, hand_id)
    existing = result.settlement or HandSettlement(hand_id=hand_id)
    valid = (
        result.ledger.is_settled
        and result.ledger.is_balanced
        and result.ledger.is_legal
        and not result.issues
    )
    status = status_when_valid if valid else "needs_correction"
    updated = existing.model_copy(
        update={
            "status": status,
            "gross_pot": result.ledger.gross_pot,
            "rake_amount": result.ledger.rake,
            "net_pot": result.ledger.net_pot,
            "is_balanced": result.ledger.is_balanced,
            "warnings": list(result.issues),
            "updated_at": utc_now(),
        }
    )
    db.upsert_hand_settlement(updated)
    return reconcile_persisted_hand(db, hand_id)


# ---------------------------------------------------------------------------
# Loading and cross-checking, split so both passes share one fetch
# ---------------------------------------------------------------------------


def _load_hand_records(db: PokerDatabase, hand_id: int) -> _HandRecords:
    hand = db.fetch_hand(hand_id)
    if hand is None:
        raise LedgerError(f"Hand {hand_id} does not exist.")
    players = db.fetch_players_by_hand(hand_id)
    actions = db.fetch_actions_by_hand(hand_id)
    settlement = db.fetch_hand_settlement(hand_id)
    entries = tuple(db.fetch_settlement_entries(hand_id))

    award_rows = [entry for entry in entries if entry.entry_type == "award"]
    refund_rows = [entry for entry in entries if entry.entry_type == "refund"]
    awards = tuple(
        (entry, _entry_identity(entry, players))
        for entry in sorted(
            award_rows, key=lambda row: (row.pot_index or 0, row.entry_order)
        )
    )
    refunds = tuple((entry, _entry_identity(entry, players)) for entry in refund_rows)

    hero_players = [player for player in players if player.is_hero]
    return _HandRecords(
        hand=hand,
        players=players,
        actions=actions,
        settlement=settlement,
        entries=entries,
        awards=awards,
        refunds=refunds,
        flop_seen=bool(hand.board_cards)
        or any(
            action.street in {"flop", "turn", "river", "showdown"} for action in actions
        ),
        hero_key=hero_players[0].player_key if len(hero_players) == 1 else None,
    )


def _unreadable_row_issues(records: _HandRecords) -> list[str]:
    """A record the persistence layer had to degrade to read blocks the verdict.

    ``db._salvaged_row`` marks every fetched model whose stored row held column
    values this build could not validate (``unreadable_columns``, a read-time,
    dump-excluded marker). The degraded values are model defaults — an
    unreadable action amount reads back as None, an unreadable stack as
    unknown — and a default is not an observation, so a ledger built over one
    may not present as reconciled, authoritative, or study-ready. The same
    argument, and the same reporting channel, as the degraded-settlement
    warning two branches below: the issue names the rows and columns to fix,
    and ACCOUNTING_NOT_AUTHORITATIVE carries it to the operator instead of a
    traceback.
    """
    labelled = [
        *(
            (f"player row for {player.player_name!r}", player.unreadable_columns)
            for player in records.players
        ),
        *(
            (
                f"action row ({action.street} {action.action_type} "
                f"by {action.player_name!r})",
                action.unreadable_columns,
            )
            for action in records.actions
        ),
        *(
            (
                f"settlement entry row for {entry.player_name!r}",
                entry.unreadable_columns,
            )
            for entry in records.entries
        ),
        ("hand row", records.hand.unreadable_columns),
    ]
    return [
        f"Stored {label} could not be fully read "
        f"(unreadable: {', '.join(columns)}); correct and re-save it before "
        "this hand's accounting can be trusted."
        for label, columns in labelled
        if columns
    ]


def _unanswered_wager_issues(
    players: list[HandPlayer], ledger: HandLedger
) -> list[str]:
    """An uncalled wager nobody was left able to answer means the line never closed.

    The ledger refunds an uncalled wager -- the excess one seat committed above
    every other seat -- and that refund is only ever produced by one of two real
    situations: everyone else FOLDED to it, or everyone else was ALL-IN and
    physically could not match it. In both, the excess chips were never live to
    anybody, so handing them back is right.

    There is a third way to reach a refund, and it is not a situation at all: the
    recorded action line simply stops while an opponent still has cards and chips.
    A manual spot typed as ``x/b3.5`` (Hero checks, Villain bets, Hero never
    acts), a mid-street truncation, a reconstructed hand whose closing action was
    not observed. The ledger cannot tell that apart from a fold-win by looking at
    contributions alone, so it refunded the bet AND paid the whole pot to the
    declared winner: a settlement asserting simultaneously that Villain's wager
    went unanswered and that Hero won it. Every figure derived from that hand --
    the hero result substituted into the Hands library and the win rate, the pot
    on the Study page, the math facts handed to the coaching provider -- is a
    number no completion of the hand produces, and the whole stack of surfaces
    called it "reconciled".

    The distinguishing fact is available here and was never consulted: whether
    any seat that is NOT the refunded one is still in the hand (did not fold) and
    still has chips behind (was not all-in). If one is, the wager was live when
    the record ended and this hand has no derivable result, whoever is declared
    the winner of it -- including the case where the declared winner happens to be
    the refunded seat, where the arithmetic coincides with "the opponent folded"
    but the record does not say so.

    Reported as a cross-check issue rather than raised, on the same reasoning as
    every other issue in this module: ``persist_reconciliation`` turns it into
    ``needs_correction``, ``is_authoritative`` goes False, and
    ACCOUNTING_NOT_AUTHORITATIVE carries the sentence to the operator with the
    action that clears it -- record the call, the fold, or the all-in.

    Derived from the ledger and the player rows only, never from the declaration,
    so both dependence passes see the same issue and the measurement of what the
    declaration moves is unchanged by it.
    """

    stacks = {player.player_key: player.starting_stack for player in players}
    names = {player.player_key: player.player_name for player in players}
    folded = set(ledger.folded_players)
    issues: list[str] = []
    for identity, refund in sorted(ledger.refunds.items()):
        if refund <= _FLOAT_TOLERANCE:
            continue
        live: list[str] = []
        for other, contributed in sorted(ledger.contributions.items()):
            if other == identity or other in folded:
                continue
            # A seat that put nothing in is not "still in the hand" for this
            # purpose: the ledger already leaves it out of every pot's eligible
            # set, and a table seat that was never dealt in or folded before the
            # recording began must not be read as an unanswered decision.
            if contributed <= _FLOAT_TOLERANCE:
                continue
            behind = stacks.get(other)
            if behind is None or behind - contributed > _FLOAT_TOLERANCE:
                live.append(names.get(other, other))
        if live:
            issues.append(
                f"{names.get(identity, identity)} is refunded {refund:g} as an "
                f"uncalled wager, but {', '.join(live)} neither folded nor was "
                "all-in, so the recorded action line never closed the betting. "
                "Record the call, fold, raise, or all-in that ended it; no result "
                "can be derived from a hand that stops mid-wager."
            )
    return issues


def _cross_check(records: _HandRecords, declaration: _Declaration) -> _CrossCheck:
    """Derive the ledger under one settlement declaration and cross-check it.

    Pure with respect to ``records``: it takes the declaration as an argument
    instead of reading it off the settlement, which is what makes the neutral
    second pass possible without a second fetch and without a second code path.

    Every read of the declaration below goes through ``declaration``, never
    through ``records``, so a pass that neutralises an input really is a pass
    that never saw it -- including the declared awards, which are both an input
    to the derived ledger (through the winners and the odd-chip order) and the
    other side of the payout comparison.
    """

    hand = records.hand
    players = records.players
    settlement = records.settlement
    ledger = build_ledger_from_records(
        players,
        records.actions,
        dead_money=declaration.dead_money,
        winners=declaration.winners,
        rake=declaration.rake,
        odd_chip_order=list(declaration.odd_chip_order),
        flop_seen=records.flop_seen,
    )
    issues = [
        *ledger.warnings,
        *ledger.legality_issues,
        *_unanswered_wager_issues(records.players, ledger),
        *_unreadable_row_issues(records),
    ]
    # One tolerance, and it is float-representation noise. Nothing here is
    # compared at anything wider, and no settlement field may widen it. Pinned by
    # round14::test_the_dependence_tolerance_is_the_float_noise_floor (the constant
    # itself), round4::test_an_imported_settlement_cannot_set_its_own_reconciliation_tolerance
    # and round5::test_rake_rate_and_chip_unit_together_cannot_widen_the_pot_check /
    # ::test_an_imported_settlement_cannot_set_its_own_tolerance_with_two_fields (no
    # payload can set it). The pointer here used to name
    # `_no_settlement_field_may_widen_a_gate`, a test that does not exist under that
    # or any name, so a maintainer following it could not tell whether the invariant
    # was covered at all.
    if settlement is None:
        issues.append("No persisted settlement assumptions or awards.")
    else:
        # A row the persistence layer had to degrade to read (a negative rake
        # rate, a zero rounding unit, a NaN) is reported here rather than in a
        # traceback, so ACCOUNTING_NOT_AUTHORITATIVE names the column to fix.
        issues.extend(
            note
            for note in settlement.warnings
            if note.startswith(UNREADABLE_SETTLEMENT_PREFIX)
        )
        _compare_optional(
            issues, "gross pot", settlement.gross_pot, ledger.gross_pot, _FLOAT_TOLERANCE
        )
        _compare_optional(
            issues, "rake", settlement.rake_amount, ledger.rake, _FLOAT_TOLERANCE
        )
        _compare_optional(
            issues, "net pot", settlement.net_pot, ledger.net_pot, _FLOAT_TOLERANCE
        )
        if settlement.status == "reconciled" and not ledger.is_settled:
            issues.append("Settlement is marked reconciled but not every pot has a winner.")
        if settlement.is_balanced and not ledger.is_balanced:
            issues.append("Settlement is marked balanced but chip conservation fails.")

    if hand.pot_size is not None and not isclose(
        hand.pot_size, ledger.gross_pot, abs_tol=_FLOAT_TOLERANCE
    ):
        issues.append("Observed final pot does not match the derived gross pot.")
    hero_players = [player for player in players if player.is_hero]
    if len(hero_players) > 1:
        issues.append("More than one player is marked as Hero.")
    elif len(hero_players) == 1 and hand.hero_bb_won is not None and ledger.is_settled:
        hero_result = ledger.net_results.get(hero_players[0].player_key)
        # Exact. `hands.hero_bb_won` is an observation of what the hero actually
        # won at the table, not a restatement of the rake policy, so no rake
        # policy may excuse a disagreement with the action line. Granting it the
        # rake slack let 'Chip unit' -- an unbounded operator field an import
        # payload also supplies -- buy up to a quarter of the pot of licence.
        if hero_result is None or not isclose(
            hand.hero_bb_won, hero_result, abs_tol=_FLOAT_TOLERANCE
        ):
            issues.append("Observed Hero result does not match the derived ledger result.")
    elif not hero_players and (hand.hero_bb_won is not None or hand.hero_cards):
        # Zero heroes used to skip the branch entirely, so unticking 'Hero' in the
        # player editor DELETED the hero-result cross-check: a hand recording
        # hero_cards and a fabricated hero_bb_won reconciled against nothing,
        # became authoritative, and accepted a promotion -- while every list view
        # kept rendering the fabricated result, which is only substituted with the
        # derived one when a hero row exists.
        issues.append(
            "This hand records a Hero result or Hero cards, but no player is "
            "marked as Hero; tick 'Hero' for the correct seat in Edit players."
        )

    declared_refunds: dict[str, float] = {}
    for entry, identity in records.refunds:
        declared_refunds[identity] = declared_refunds.get(identity, 0) + (entry.amount or 0)
    if records.refunds:
        for identity in set(declared_refunds) | set(ledger.refunds):
            # Uncalled bets are returned before any rake is taken, so a refund
            # carries no rounding ambiguity.
            if not isclose(
                declared_refunds.get(identity, 0),
                ledger.refunds.get(identity, 0),
                abs_tol=_FLOAT_TOLERANCE,
            ):
                issues.append(
                    f"Declared refund for {identity!r} does not match the derived refund."
                )

    # 'Observed payout' is an optional column, so an award row may declare a
    # winner without an amount. That used to disable the WHOLE comparison: one
    # blank cell anywhere in the award set skipped every other row, so a declared
    # payout of 9999 against a derived 250 reconciled and rendered study-ready.
    # Not knowing pot 1's payout is not evidence about pot 0's, so the check is
    # now per identity, and a partially-declared identity still has to satisfy
    # the half of the claim it did make: the amounts it DID declare cannot
    # already exceed the identity's whole derived payout, because the blank rows
    # can only add more.
    declared_awards: dict[str, float] = {}
    fully_declared: dict[str, bool] = {}
    for entry, identity in declaration.awards:
        declared_awards[identity] = declared_awards.get(identity, 0) + (entry.amount or 0)
        fully_declared[identity] = (
            fully_declared.get(identity, True) and entry.amount is not None
        )
    if declaration.awards:
        for identity in set(declared_awards) | set(ledger.payouts):
            declared = declared_awards.get(identity, 0)
            derived = ledger.payouts.get(identity, 0)
            # Exact, for the same reason the hero result is: a declared award
            # says how many chips a seat was pushed, which is an observation of
            # the hand and not a restatement of the rake policy.
            if fully_declared.get(identity, True):
                mismatch = not isclose(declared, derived, abs_tol=_FLOAT_TOLERANCE)
            else:
                mismatch = declared - derived > _FLOAT_TOLERANCE
            if mismatch:
                issues.append(
                    f"Declared awards for {identity!r} do not match derived payouts."
                )

    return _CrossCheck(ledger=ledger, issues=tuple(dict.fromkeys(issues)))


# ---------------------------------------------------------------------------
# The dependence rule
# ---------------------------------------------------------------------------

RAKE_POLICY_INPUT = "rake_policy"
DEAD_MONEY_INPUT = "dead_money"
POT_AWARD_INPUT = "declared_pot_awards"
# No single input alone breaks the reconciliation but the declaration as a whole
# does -- so no half can be attested to on its own, and the set is named as one.
JOINT_INPUT = "settlement_assumptions"

_NEUTRAL_RAKE = RakePolicy()
_NEUTRAL_DEAD_MONEY = 0.0


def _derive_assumption_dependence(
    records: _HandRecords, baseline: _CrossCheck
) -> tuple[AssumptionDependence, ...]:
    """Does what this hand reports rest on operator-declared assumptions?

    One rule, no field list:

    1. Cross-check the hand with the stored settlement declaration (``baseline``).
    2. Cross-check the SAME records with the declaration WITHDRAWN -- see
       ``_Declaration``, which is the complete set of inputs ``_cross_check``
       takes from what somebody declared rather than from what was recorded.
       Everything else -- the action line, the board, whether a flop was seen,
       the recorded pot and hero result -- is the hand, not the declaration, and
       is held constant.
    3. The hand is assumption-dependent when removing the declaration changes
       either VERDICT or FIGURES: the neutral pass stops reconciling (or cannot
       be built at all), or it reconciles too but derives a different gross pot,
       rake, net pot, payout, or hero result.
    4. Each declared input is then neutralised on its own to attribute the
       dependence, by the same two-part test. When no input is individually
       load-bearing but the declaration as a whole is, it is reported as one
       joint dependence, because attesting to any half would be attesting to
       something that on its own proves nothing.

    The declared POT AWARDS are inputs 3 and 4 of that set, and leaving them out
    was a two-entry field list wearing a different hat. On a reconstructed hand
    the CV exporter emits no ``settlement`` key at all, so every award row was
    typed into the Accounting reconciliation panel by an operator -- the same
    panel, the same save, as the rake and the dead money. Nothing observed it,
    and it is the single input the derived payouts and therefore the reported
    hero result are computed from: on a hand recording a null ``pot_size`` and a
    null ``hero_bb_won`` (the ordinary state of a freshly imported hand) one
    dropdown moved the recorded hero result by the whole 80-chip pot, in either
    direction, with no measurement, no disclosure, no correction record and an
    EMPTY blocker tuple. A declared rake of the same 40 chips was named,
    measured and blocked. Both are now the same rule.

    Step 3 used to ask about the verdict alone -- "does it still reconcile?" --
    and that is a hole the size of the product. A hand that records none of the
    figures the cross-check compares (no ``gross_pot``, ``rake_amount`` or
    ``net_pot`` on the settlement, a null ``hands.pot_size``, a null
    ``hands.hero_bb_won``, award rows with no amount) reconciles under EVERY
    policy, because there is nothing left for a policy to contradict -- and that
    is the ordinary state of a freshly imported hand, since ``import_session``
    never calls ``persist_reconciliation``. A declared 90% rake was therefore
    measured as assumption-INDEPENDENT while it moved the hero result the Study
    page and the win rate actually display by 72 of the pot's 80 chips, because
    ``_hands_with_accounting_results`` and ``math.analytics`` substitute the
    DERIVED result on every authoritative hand. Chips that move what the product
    reports are the thing being disclosed, so they are what is measured.

    A hand that reconciles identically under both is not assumption-dependent and
    is disclosed nothing -- which is as important as the blocking half. The
    previous field-list gate fired on ``rake_rate > 0`` even where the policy
    provably took no chips (a zero cap; no-flop-no-drop on a hand with no board),
    and an operator trained to click Acknowledge through disclosures that mean
    nothing is an operator who will click through the one that means something.

    One short-circuit, and it is an identity rather than a heuristic: a
    declaration that already IS the neutral declaration cannot be depended on,
    because both passes would be the same call. Nothing here asks which fields
    are suspicious.

    Cost, in extra ledger builds beyond the baseline pass, measured rather than
    reasoned about (``test_the_documented_dependence_cost_ceiling_is_the_measured
    _one``): zero on a hand that declared nothing -- no settlement row, no awards,
    which is every freshly reconstructed hand before anyone opens the Accounting
    reconciliation panel -- one on a SHOWDOWN hand whose only declaration is its
    awards, four on a showdown hand declaring all three, and one more in each case
    on a FOLD WIN, where ``_forced_winners`` fires and a second neutral pass is
    built against the forced winners: two, and five.

    The ceiling used to be stated as "up to four", which understated the commonest
    hand shape there is. It is now derived from a counter in the test rather than
    from this sentence, so it cannot drift again.
    """

    declared = records.declaration
    if declared.is_neutral:
        return ()
    if not baseline.reconciles:
        # Nothing rests on the assumptions yet: the hand does not reconcile with
        # them either, and ACCOUNTING_NOT_AUTHORITATIVE already blocks it.
        return ()

    award_less = _try_cross_check(records, _Declaration.neutral())
    forced = _forced_winners(award_less)
    neutral = (
        award_less
        if forced is None
        else _try_cross_check(records, _Declaration.neutral(forced))
    )
    if not _is_dependent(records, baseline, neutral):
        return ()

    settlement = records.settlement
    found: list[AssumptionDependence] = []
    for input_name, without, declared_text, neutral_text in (
        (
            RAKE_POLICY_INPUT,
            declared.with_neutral_rake(),
            _rake_text(settlement),
            _rake_text(None),
        ),
        (
            DEAD_MONEY_INPUT,
            declared.with_neutral_dead_money(),
            _chips_text(declared.dead_money),
            _chips_text(_NEUTRAL_DEAD_MONEY),
        ),
        (
            POT_AWARD_INPUT,
            declared.without_awards(forced),
            _awards_text(declared.awards),
            _forced_awards_text(forced),
        ),
    ):
        if without == declared:
            # This input was never declared, so neutralising it is the baseline
            # pass and can never differ from it. Skipped for cost, not for
            # judgement: running it would return the same answer.
            continue
        neutralised = (
            neutral if without.is_neutral else _try_cross_check(records, without)
        )
        if _is_dependent(records, baseline, neutralised):
            found.append(
                _build_dependence(
                    records,
                    baseline,
                    neutralised,
                    input_name=input_name,
                    declared=declared_text,
                    neutral=neutral_text,
                )
            )
    if not found:
        # Defence in depth, and deliberately not advertised as a working part of
        # the rule: no hand shape is known to reach it. Rake and dead money
        # compose additively through gross -> rake -> net -> payout -> hero, so a
        # pair that moves chips has a half that moves chips, and a search over
        # ~415,000 shapes (seats x contributions x winner sets x rates x rounding
        # units x dead-money values) found none where both halves are individually
        # harmless; withdrawing the awards on a settled hand always removes every
        # payout, so it is never individually harmless either. It is kept because
        # its absence would silently return ``()`` -- the one failure mode this
        # module must not have -- and it is pinned by
        # ``test_the_joint_fallback_still_names_a_dependence_no_half_can_explain``,
        # which injects the per-input passes rather than pretending a hand
        # reaches it.
        found.append(
            _build_dependence(
                records,
                baseline,
                neutral,
                input_name=JOINT_INPUT,
                declared=(
                    f"rake {_rake_text(settlement)}; dead money "
                    f"{_chips_text(declared.dead_money)}; awards "
                    f"{_awards_text(declared.awards)}"
                ),
                neutral=(
                    f"rake {_rake_text(None)}; dead money "
                    f"{_chips_text(_NEUTRAL_DEAD_MONEY)}; awards "
                    f"{_forced_awards_text(forced)}"
                ),
            )
        )
    return tuple(found)


def _forced_winners(
    award_less: _CrossCheck | None,
) -> tuple[tuple[int, str], ...] | None:
    """The winners the RECORDING leaves no choice about, or None when it leaves one.

    "Withdraw the declaration" has to name a state the hand could actually be in
    without it, or the measurement stops being about the declaration. For the
    rake and the dead money that state is obvious: no rake, no dead money. For
    the awards it is NOT "nobody won anything" -- no recording produces that, an
    award-less ledger is never ``is_settled``, and comparing against it therefore
    made every hand whose baseline reconciles award-dependent by construction. A
    tautological block is the anti-pattern this rule exists to end: an operator
    who must press "Confirm this assumption" on every hand, including hands
    declaring no rake and no dead money at all, is an operator being trained to
    press it without reading it.

    What the recording can determine is the winner of a pot exactly one seat is
    still eligible for -- everyone else folded. Those pots are answered by the
    action line, not by the operator, so declaring them moves nothing and is
    silent. A pot two or more seats are eligible for is a showdown, and the
    winner of it genuinely is a declaration nothing in the recording corroborates:
    that hand is still measured, still named, and still blocked.

    Returns None when any pot has more than one eligible seat, which is the case
    where withdrawing the awards outright remains the honest comparison.
    """
    if award_less is None:
        return None
    forced: list[tuple[int, str]] = []
    for pot in award_less.ledger.pots:
        if len(pot.eligible_players) != 1:
            return None
        forced.append((pot.index, pot.eligible_players[0]))
    return tuple(forced) or None


def _is_dependent(
    records: _HandRecords, baseline: _CrossCheck, neutralised: _CrossCheck | None
) -> bool:
    """Does removing this declaration change the verdict OR any reported figure?

    The single test every neutralisation goes through -- the whole declaration,
    the rake alone, the dead money alone, the declared awards alone -- so there
    is one place where "depends on" is defined and no pass can be scoped
    differently from another.

    ``None`` means the ledger refused to build without the declaration, which is
    the strongest form of dependence there is: removing it left an impossible
    hand. Fails closed on purpose.

    The VERDICT half covers what ``_ledger_deltas`` structurally cannot: the five
    headline figures are not the whole cross-check, and a pass can stop
    reconciling while every one of them stands still -- an unsettled ledger with
    no payouts to move (a pot whose entire value is taken as rake), a balance or
    legality verdict that flips, or a recorded restatement the neutral policy now
    contradicts. It is reachable, not decorative:
    ``test_withdrawing_the_awards_is_a_dependence_the_figures_cannot_show``
    exercises it on a real hand and fails if this branch is removed.
    """
    if neutralised is None:
        return True
    if not neutralised.reconciles:
        return True
    return bool(_ledger_deltas(baseline.ledger, neutralised.ledger, records.hero_key))


def _try_cross_check(
    records: _HandRecords, declaration: _Declaration
) -> _CrossCheck | None:
    """Cross-check under an alternative declaration; None when the ledger refuses.

    Fails closed on purpose. A neutral pass that cannot even be built is a pass
    that does not reconcile, so the hand is assumption-dependent -- removing the
    declaration left an impossible hand, which is the strongest possible form of
    "the reconciliation rested on it".
    """
    try:
        return _cross_check(records, declaration)
    except LedgerError:
        return None


def _build_dependence(
    records: _HandRecords,
    baseline: _CrossCheck,
    neutralised: _CrossCheck | None,
    *,
    input_name: str,
    declared: str,
    neutral: str,
) -> AssumptionDependence:
    deltas = (
        ()
        if neutralised is None
        else _ledger_deltas(baseline.ledger, neutralised.ledger, records.hero_key)
    )
    if neutralised is None:
        measure = "unbuildable"
    elif deltas:
        measure = "|".join(f"{name}{_format_chips(value)}" for name, value in deltas)
    else:
        # Measured, and the answer was zero: no reported figure moved, yet the
        # neutral pass stopped reconciling. Reachable -- withdraw the awards on a
        # pot whose entire value is taken as rake and every payout is already
        # zero, so the ledger goes unsettled with nothing to move -- and named
        # rather than silently dropped, because a dependence the operator is
        # asked to attest to must say what it measured.
        measure = "verdict-only"
    return AssumptionDependence(
        input_name=input_name,
        declared=declared,
        neutral=neutral,
        deltas=deltas,
        code=(
            f"{ASSUMPTION_DEPENDENCE_PREFIX}:{input_name}:"
            f"{_declaration_fingerprint(records, baseline, declared, neutral)}:{measure}"
        ),
    )


def _declaration_fingerprint(
    records: _HandRecords,
    baseline: _CrossCheck,
    declared: str,
    neutral: str,
) -> str:
    """A short, stable digest of WHAT was declared, over WHICH hand.

    The measured movement alone is not enough to identify an attestation. A 50%
    rake on an 80-chip pot and a 25% rake on a corrected 160-chip pot both remove
    40 chips and move every headline figure by the same amount, so the operator's
    confirmation of the first was inherited verbatim by the second: a policy they
    never saw, over an action line that had doubled, cleared its own blocker.

    Digesting the declared policy text, the neutral text it is compared against,
    the gross pot the measurement was taken on, the settled per-seat
    contribution vector, and which seat is the hero binds the attestation to the
    declaration and the hand as well as to the chips. It is recomputed from the
    stored records on every read, so it is byte-stable across an idempotent
    re-save exactly as the movement is.

    The contribution vector and the hero term are round 13's addition, closing
    the stated binding's gap to the ACTION LINE: the measured deltas are
    declared-minus-neutral, so the contributions cancel out of every one of
    them, and two seats committing 40 each produced a code byte-identical to
    four seats committing 20 each -- same gross pot, same declaration, same
    movement -- while the derived hero result differed by 20 chips. An
    attestation must not survive a rewrite that changes whose chips, or how
    many of each seat's, the sentence it attests to is about.
    """
    dead_money = 0.0 if records.settlement is None else records.settlement.dead_money
    contributions = ";".join(
        f"{identity}={amount:.6f}"
        for identity, amount in sorted(baseline.ledger.contributions.items())
    )
    context = (
        f"{declared} -> {neutral} @ gross {baseline.ledger.gross_pot:.6f} "
        f"@ dead {dead_money:.6f} @ contributions {contributions} "
        f"@ hero {records.hero_key or ''}"
    )
    return blake2s(context.encode("utf-8"), digest_size=5).hexdigest()


def _ledger_deltas(
    declared: HandLedger, neutral: HandLedger, hero_key: str | None
) -> tuple[tuple[str, float], ...]:
    """How many chips each headline figure moves when the assumption is removed."""
    candidates: list[tuple[str, float]] = [
        ("gross", declared.gross_pot - neutral.gross_pot),
        ("rake", declared.rake - neutral.rake),
        ("net", declared.net_pot - neutral.net_pot),
    ]
    identities = set(declared.payouts) | set(neutral.payouts)
    payout = max(
        (
            abs(declared.payouts.get(key, 0.0) - neutral.payouts.get(key, 0.0))
            for key in identities
        ),
        default=0.0,
    )
    candidates.append(("payout", payout))
    if hero_key is not None:
        candidates.append(
            (
                "hero",
                declared.net_results.get(hero_key, 0.0)
                - neutral.net_results.get(hero_key, 0.0),
            )
        )
    return tuple(
        (name, value) for name, value in candidates if abs(value) > _FLOAT_TOLERANCE
    )


def _format_chips(value: float) -> str:
    """A signed chip amount that round-trips to the float it was measured from.

    Both passes are recomputed from the same stored records on every read, so
    this is byte-stable across an idempotent re-save -- which is what lets an
    acknowledgement survive ``persist_reconciliation`` re-saving the settlement.

    It is also INJECTIVE, which the previous ``f"{value:+.6f}"`` was not: every
    movement in (1e-9, 5e-7) formatted as "+0", so distinct measured quantities
    above the dependence tolerance (1e-9) shared one acknowledgement string while
    the module's stated invariant was that the measurement is in the code.
    ``format(value, "+")`` is Python's shortest round-tripping repr with a forced
    sign, so ``float(_format_chips(x)) == x`` for every finite x and no two
    distinct floats can collide.
    """
    return format(value, "+")


def _awards_text(awards: tuple[tuple[SettlementEntry, str], ...]) -> str:
    """The declared pot awards, in the order that decides the odd chip.

    ``_load_hand_records`` sorts award rows by ``(pot_index, entry_order)``, and
    that order IS the declared odd-chip order, so re-ordering two winners of a
    chopped pot changes this text and therefore the attestation fingerprint --
    which is the round-8 finding that a declared-award audit snapshot excluded
    the column deciding who received the chips.
    """
    if not awards:
        return "no declared winner"
    return "; ".join(
        f"pot {entry.pot_index or 0} -> {identity}"
        + ("" if entry.amount is None else f" ({_chips_text(entry.amount)})")
        for entry, identity in awards
    )


def _forced_awards_text(forced: tuple[tuple[int, str], ...] | None) -> str:
    """What the awards are compared against, named so the operator can check it."""
    if forced is None:
        return "no declared winner"
    return "the only seat the action line leaves eligible: " + "; ".join(
        f"pot {index} -> {identity}" for index, identity in forced
    )


def _chips_text(value: float) -> str:
    return f"{value:g} chips"


def _rake_text(settlement: HandSettlement | None) -> str:
    policy = RakePolicy() if settlement is None else RakePolicy(
        rate=settlement.rake_rate,
        cap=settlement.rake_cap,
        rounding_unit=settlement.rake_rounding_unit,
        no_flop_no_drop=settlement.no_flop_no_drop,
    )
    return (
        f"rate={policy.rate:g}, cap={policy.cap}, unit={policy.rounding_unit:g}, "
        f"no_flop_no_drop={policy.no_flop_no_drop}"
    )


def _entry_identity(entry: SettlementEntry, players: list) -> str:
    if entry.player_key is not None:
        if not any(player.player_key == entry.player_key for player in players):
            raise LedgerError(
                f"Settlement entry references unknown player key {entry.player_key!r}."
            )
        return entry.player_key
    matches = [player.player_key for player in players if player.player_name == entry.player_name]
    if len(matches) != 1:
        raise LedgerError(
            f"Settlement entry cannot resolve player {entry.player_name!r} "
            "to one stable identity."
        )
    return matches[0]


def _compare_optional(
    issues: list[str],
    label: str,
    recorded: float | None,
    derived: float,
    tolerance: float,
) -> None:
    if recorded is not None and not isclose(recorded, derived, abs_tol=tolerance):
        issues.append(f"Recorded {label} does not match the derived ledger.")
