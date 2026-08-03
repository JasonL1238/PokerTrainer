"""The whole (action kind x forced-bet type) matrix, in both directions.

A forced-bet label says WHICH forced post a row is. It is not allowed to say
whether the row is one at all: the KIND puts a row in its pool and no label
takes it out. The half of that rule which was still open is the one with teeth
-- tagging the only ``kind='ante'`` row ``straddle``, ``big_blind`` or
``dead_blind`` emptied the ante pool, which switched OFF the
undeclared-ante-mode refusal and flipped a hand the product must refuse into an
accepted one, without moving a chip. The declaration is what stands between an
operator and a silently wrong pot, so a guard a selectbox can disable is the
release-blocking shape, not a coverage limitation.

The mirror is asserted alongside it: a label may not add a row to a pool its
kind keeps it out of either, which is what a ``post_blind`` typed ``ante`` was
doing.
"""

from __future__ import annotations

import pytest

from poker_tracker.math.accounting import (
    UNDECLARED_ANTE_MODE_PREFIX,
    AnteMode,
    BlindStructure,
    LedgerAction,
    LedgerPlayer,
    _is_ante_post,
    _is_forced_post,
    _is_live_structural_post,
    _mislabelled_forced_bet,
    _readable_forced_bet_type,
    build_hand_ledger,
)

# Every name ``models.ForcedBetType`` can hold. The two ante names that are not
# yet spellable there are covered by the predicate tests below rather than here,
# because no recording can produce them today.
ALL_FORCED_BET_TYPES = (
    "small_blind",
    "big_blind",
    "ante",
    "big_blind_ante",
    "straddle",
    "dead_blind",
    "bring_in",
)
ALL_KINDS = (
    "ante",
    "post_blind",
    "bet",
    "call",
    "raise",
    "all-in",
    "fold",
    "check",
)
# Which labels each kind can carry without contradicting itself. A kind that
# names a forced post has already named its species and the label only refines
# it; ``all-in`` names no species, so there every label is readable and the
# label is the only signal there is. Every other kind can carry none.
COMPATIBLE_LABELS: dict[str, frozenset[str]] = {
    "ante": frozenset({"ante", "big_blind_ante"}),
    "post_blind": frozenset(
        {"small_blind", "big_blind", "straddle", "bring_in", "dead_blind"}
    ),
    "all-in": frozenset(ALL_FORCED_BET_TYPES),
}
# The one label on the one kind whose reading legitimately changes the money: a
# blind the recording names as dead is dead, and nothing but the label can say
# so. Named here so that a future change anywhere else in the compatible half of
# the matrix has to come past this list.
MONEY_MOVING_COMPATIBLE_CELLS = {("post_blind", "dead_blind")}


def _label_of(kind: str) -> frozenset[str]:
    return COMPATIBLE_LABELS.get(kind, frozenset())


def _row(kind: str, forced: str | None, amount: float = 1) -> LedgerAction:
    return LedgerAction("C", "preflop", kind, amount, forced_bet_type=forced)


# --------------------------------------------------------------------------
# The predicates, which are where the pools are actually decided.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_no_label_takes_an_ante_row_out_of_the_ante_pool(forced: str) -> None:
    """The reported defect at the predicate. ``kind='ante'`` IS an ante, always.

    ``_is_ante_post`` is what feeds the ante-mode declaration gate, so a label
    that can answer it False is a label that can switch the gate off.
    """

    labelled = _row("ante", forced)
    assert _is_ante_post(labelled) is True
    assert _is_forced_post(labelled) is True
    # An ante answers no wager level whatever it is typed as, so it can never
    # promote itself into the pool that sets one.
    assert _is_live_structural_post(labelled) is False


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_no_label_puts_a_blind_post_into_the_ante_pool(forced: str) -> None:
    """The mirror. ``kind='post_blind'`` names the blind species, not the ante one."""

    labelled = _row("post_blind", forced)
    assert _is_ante_post(labelled) is False
    assert _is_forced_post(labelled) is True


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_a_post_that_took_the_last_chip_is_still_named_by_its_label(
    forced: str,
) -> None:
    """The exception the kind rule must not close.

    ``all-in`` says the poster ran out, not what the poster was paying, so the
    label is the only signal there is. Closing it here would silently revert the
    consolidated-ante reading on every recording that spells its short ante that
    way.
    """

    labelled = _row("all-in", forced)
    assert _readable_forced_bet_type(labelled) == forced
    assert _mislabelled_forced_bet(labelled) is False
    assert _is_forced_post(labelled) is True
    assert _is_ante_post(labelled) is (forced in {"ante", "big_blind_ante"})


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_a_contradicting_label_is_reported_and_derives_as_the_cleared_row(
    kind: str, forced: str
) -> None:
    """The whole matrix at the predicate level, both directions at once.

    Compatible cells read the label. Contradicting cells read none of it: every
    pool answers exactly what it answers for the same row with the field
    cleared, which is the strict direction and the one an operator cannot steer.
    """

    labelled = _row(kind, forced)
    cleared = _row(kind, None)
    compatible = forced in _label_of(kind)

    assert _mislabelled_forced_bet(labelled) is not compatible
    if compatible:
        assert _readable_forced_bet_type(labelled) == forced
        return

    assert _readable_forced_bet_type(labelled) is None
    assert _is_forced_post(labelled) is _is_forced_post(cleared)
    assert _is_ante_post(labelled) is _is_ante_post(cleared)
    assert _is_live_structural_post(labelled) is _is_live_structural_post(cleared)


def test_the_two_species_do_not_overlap() -> None:
    """No name may be both an ante and a blind, or the kind rule decides nothing."""

    assert _label_of("ante") & _label_of("post_blind") == frozenset()
    assert set(ALL_FORCED_BET_TYPES) == _label_of("ante") | _label_of("post_blind")


def test_every_recordable_forced_bet_type_is_classified() -> None:
    """A name added to the recording vocabulary must be filed, not defaulted.

    An unclassified name would be a contradiction on both forced-post kinds and
    a free-floating signal on ``all-in``, which is a silent third behaviour
    nobody chose. Tying the sweep to ``models.ForcedBetType`` makes adding one
    fail here rather than in a hand six months later.
    """

    from typing import get_args

    from poker_tracker.persistence.models import ForcedBetType

    assert set(get_args(ForcedBetType)) == set(ALL_FORCED_BET_TYPES)


# --------------------------------------------------------------------------
# The guard itself, end to end. A predicate is not the product; the refusal is.
# --------------------------------------------------------------------------


def _ante_hand(forced: str | None, *, ante_mode: str | None = None):
    """A hand whose only ante is A's row, sized so the cap actually bites.

    30 against a 4-chip seat is worked example (f): capped it lays out 16 and 26,
    exempt it lays out one pot of 42. The two readings are visibly different
    money, so a label that could move the row between them is not asserting
    about nothing.
    """

    players = [
        LedgerPlayer("A", 40, 0),
        LedgerPlayer("short", 4, 1),
        LedgerPlayer("B", 40, 2),
    ]
    actions = [
        LedgerAction(
            "A", "preflop", "ante", 30, is_live_post=False, forced_bet_type=forced
        ),
        LedgerAction("A", "preflop", "bet", 4),
        LedgerAction("short", "preflop", "all-in", 4),
        LedgerAction("B", "preflop", "call", 4),
    ]
    return build_hand_ledger(players, actions, ante_mode=ante_mode)


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_no_label_switches_off_the_undeclared_ante_mode_refusal(forced: str) -> None:
    """The reported defect end to end: is_legal must not flip on a selectbox."""

    undeclared = _ante_hand(forced)
    assert any(
        issue.startswith(UNDECLARED_ANTE_MODE_PREFIX)
        for issue in undeclared.legality_issues
    )
    assert undeclared.is_legal is False


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_no_label_switches_off_the_declared_none_contradiction(forced: str) -> None:
    """The same gate's other refusal, which the same relabel used to silence."""

    declared_none = _ante_hand(forced, ante_mode=AnteMode.NONE)
    assert any(
        "The declared ante mode is NONE" in issue
        for issue in declared_none.legality_issues
    )
    assert declared_none.is_legal is False


@pytest.mark.parametrize("mode", (AnteMode.PER_PLAYER, AnteMode.SINGLE_PAYER_TABLE_ANTE))
@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_no_label_changes_which_capping_rule_an_ante_row_runs_under(
    forced: str, mode: str
) -> None:
    """Silencing the gate is half of it; the other half is the layering it guards.

    ``SINGLE_PAYER_TABLE_ANTE`` exempts the ante pool from the cap. A label that
    could move a ``kind='ante'`` row out of that pool would quietly re-cap a
    consolidated table ante -- a chip difference with no refusal beside it.
    """

    cleared = _ante_hand(None, ante_mode=mode)
    labelled = _ante_hand(forced, ante_mode=mode)
    assert _signature(labelled) == _signature(cleared)
    # The two modes must not agree, or the assertion above is about nothing.
    assert _signature(_ante_hand(None, ante_mode=AnteMode.PER_PLAYER)) != _signature(
        _ante_hand(None, ante_mode=AnteMode.SINGLE_PAYER_TABLE_ANTE)
    )


def _signature(ledger) -> tuple:
    """Everything the ledger derives, so "derives identically" is not a guess."""

    return (
        ledger.gross_pot,
        ledger.rake,
        ledger.net_pot,
        tuple(sorted(ledger.contributions.items())),
        tuple(sorted(ledger.refunds.items())),
        tuple(sorted(ledger.payouts.items())),
        tuple(sorted(ledger.net_results.items())),
        tuple(
            (pot.amount, pot.net_amount, pot.cause, pot.eligible_players, pot.winners)
            for pot in ledger.pots
        ),
        ledger.is_settled,
        ledger.is_balanced,
        tuple(ledger.warnings),
    )


def _matrix_hand(kind: str, forced: str | None):
    """One hand shape every kind can legally occupy, so the cells are comparable."""

    players = [
        LedgerPlayer("A", 100, 0),
        LedgerPlayer("B", 100, 1),
        LedgerPlayer("C", 40, 2),
    ]
    amount = 0 if kind in {"fold", "check"} else (40 if kind == "all-in" else 10)
    actions = [
        LedgerAction("A", "preflop", "post_blind", 5),
        LedgerAction("B", "preflop", "post_blind", 10),
        LedgerAction("C", "preflop", kind, amount, forced_bet_type=forced),
        LedgerAction("A", "preflop", "call", 5),
    ]
    return build_hand_ledger(
        players, actions, blinds=BlindStructure(5, 10), ante_mode=AnteMode.PER_PLAYER
    )


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_matrix_moves_no_chip_except_where_the_label_is_the_only_signal(
    kind: str, forced: str
) -> None:
    """What may and may not change, cell by cell, against the cleared row.

    A contradicting cell adds exactly one legality issue -- the contradiction --
    and changes nothing else at all. It never REMOVES one: an operator cannot
    clear a refusal by relabelling the row it was raised against, which is the
    property the whole family is about.
    """

    cleared = _matrix_hand(kind, None)
    labelled = _matrix_hand(kind, forced)
    compatible = forced in _label_of(kind)

    contradictions = [
        issue for issue in labelled.legality_issues if "but typed as a" in issue
    ]
    remainder = [
        issue for issue in labelled.legality_issues if issue not in contradictions
    ]

    if compatible:
        assert contradictions == []
        if (kind, forced) not in MONEY_MOVING_COMPATIBLE_CELLS and kind != "all-in":
            assert _signature(labelled) == _signature(cleared)
        return

    assert len(contradictions) == 1
    assert f"booked as {kind!r}" in contradictions[0]
    assert repr(forced) in contradictions[0]
    assert remainder == list(cleared.legality_issues)
    assert _signature(labelled) == _signature(cleared)


@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
@pytest.mark.parametrize("kind", ("ante", "post_blind"))
def test_a_label_never_clears_a_refusal_the_unlabelled_row_raises(
    kind: str, forced: str
) -> None:
    """Stated as the one-way property, over a hand carrying both open gates.

    Neither the blind structure nor the ante mode is declared here, so the
    unlabelled row raises whatever it raises and the labelled row must raise at
    least the same set. ``dead_blind`` on a ``post_blind`` is the single
    exception and is exempted by name: a blind the recording calls dead answers
    no wager level, so the unreadable-post refusal genuinely does not reach it,
    and no kind can say that for the label.
    """

    players = [
        LedgerPlayer("SB", 100, 0),
        LedgerPlayer("BB", 4, 1),
        LedgerPlayer("BTN", 100, 2),
    ]

    def hand(label: str | None):
        return build_hand_ledger(
            players,
            [
                LedgerAction("SB", "preflop", "post_blind", 5),
                LedgerAction("BB", "preflop", kind, 4, forced_bet_type=label),
                LedgerAction("BTN", "preflop", "call", 5),
            ],
        )

    cleared = set(hand(None).legality_issues)
    labelled = set(hand(forced).legality_issues)
    if (kind, forced) in MONEY_MOVING_COMPATIBLE_CELLS:
        return
    assert cleared <= labelled
