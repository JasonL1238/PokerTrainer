"""The whole (action kind x forced-bet type x post status) matrix.

Three operator-supplied facts sit on one row of the hand editor -- the action
kind, the "Forced post" name and the "Post status" -- and each pair of them can
state incompatible things about the same chips. The ruling is that such a pair
is REFUSED, never silently resolved, because the two readings give different
pots and the model must not choose between two things the operator said.

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

THE LIVENESS AXIS is the same exposure reached by the OTHER selectbox on the
same row, and it was left open when the kind-versus-label axis was closed. A big
blind all-in for 4 with the structure undeclared is refused and lays out 12/2;
setting Post status to dead silenced the refusal, laid out 4/10, and presented
the hand as study-ready with no blockers at all. Naming the row ``dead_blind``
instead gets to the same place by the other box. Neither knob is worth closing
alone, so the sweep below is over all three fields at once -- that is the
artifact that stops a fourth axis appearing.

WHERE THE RULE STOPS, asserted here as explicitly as the rule itself. A
contradiction needs TWO stated facts. A row that states only one -- a live blind
whose Post status alone says dead, or one whose label alone says ``dead_blind``
-- is a single unfalsifiable claim, and it is believed. That is a coverage
limitation of the recording format, not a defect this module can close.
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
# What each forced-bet name SAYS about liveness, written from the poker meaning
# of the name rather than imported from the reducer, so this file states the
# specification instead of echoing the code under test. The four structural bets
# set what every other seat owes; the antes and the dead blind are owed to the
# table and answer no wager level.
LABEL_IS_LIVE = {
    "small_blind": True,
    "big_blind": True,
    "straddle": True,
    "bring_in": True,
    "ante": False,
    "big_blind_ante": False,
    "dead_blind": False,
}
# The three states the "Post status" selectbox produces. ``None`` is the
# unspecified one and it is the interesting member: it reads as live, exactly as
# the field always has, and it contradicts nothing -- only a STATED status can
# disagree with the name beside it.
ALL_POST_STATUSES = (None, True, False)


def _label_of(kind: str) -> frozenset[str]:
    return COMPATIBLE_LABELS.get(kind, frozenset())


def _named_liveness(kind: str, forced: str | None) -> bool | None:
    """What the row says about its own liveness before the status is read.

    A compatible forced-bet name says it outright. Otherwise the kind says it
    only for ``ante``, which is dead by definition; ``post_blind`` names a
    species that is legitimately either, and ``all-in`` names nothing.
    """

    if forced is not None and forced in _label_of(kind):
        return LABEL_IS_LIVE[forced]
    if kind == "ante":
        return False
    return None


def _expected_conflict(kind: str, forced: str | None, status: bool | None) -> str | None:
    """Which contradiction this cell must raise: 'kind', 'status', or none.

    The kind-versus-label question is answered first because a name the kind
    rejects has not been read at all, so it cannot go on to contradict the
    status as well -- one row, one refusal, naming the pair that actually
    disagrees.
    """

    if forced is not None and forced not in _label_of(kind):
        return "kind"
    named = _named_liveness(kind, forced)
    if status is not None and named is not None and status is not named:
        return "status"
    return None


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

    A name must also be filed as LIVE or DEAD, or the post-status check below
    has nothing to compare against and the new name silently exempts itself from
    the liveness contradiction.
    """

    from typing import get_args

    from poker_tracker.persistence.models import ForcedBetType

    assert set(get_args(ForcedBetType)) == set(ALL_FORCED_BET_TYPES)
    assert set(LABEL_IS_LIVE) == set(ALL_FORCED_BET_TYPES)


@pytest.mark.parametrize("status", ALL_POST_STATUSES)
@pytest.mark.parametrize("forced", ALL_FORCED_BET_TYPES)
def test_a_stated_post_status_cannot_contradict_the_name_and_be_read(
    forced: str, status: bool | None
) -> None:
    """The liveness axis at the predicate, on the kind where it decides money.

    A ``post_blind`` typed ``big_blind`` and marked dead states two opposite
    things about the same chips. The predicate that decides whether the row sets
    the wager level used to AND the two, so the dead half simply won and the row
    stopped being structural. It must instead read as though neither field were
    set, which for a ``post_blind`` is the live structural post its kind says it
    is.
    """

    row = LedgerAction(
        "C", "preflop", "post_blind", 1, is_live_post=status, forced_bet_type=forced
    )
    conflict = _expected_conflict("post_blind", forced, status)

    if conflict is None:
        assert _readable_forced_bet_type(row) == forced
        assert _is_live_structural_post(row) is LABEL_IS_LIVE[forced]
        return

    assert _readable_forced_bet_type(row) is None
    assert _mislabelled_forced_bet(row) is True
    # The row falls back to exactly the facts that were NOT in the
    # contradiction: a rejected name leaves the status standing, and a rejected
    # pair leaves the bare kind. Either way the operator cannot steer it --
    # adding a second, disagreeing field must never take a refusal off a row.
    fallback_status = status if conflict == "kind" else None
    bare = LedgerAction("C", "preflop", "post_blind", 1, is_live_post=fallback_status)
    assert _is_live_structural_post(row) is _is_live_structural_post(bare)
    if conflict == "status":
        assert _is_live_structural_post(row) is True


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


def _matrix_hand(kind: str, forced: str | None, status: bool | None = None):
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
        LedgerAction(
            "C",
            "preflop",
            kind,
            amount,
            is_live_post=status,
            forced_bet_type=forced,
        ),
        LedgerAction("A", "preflop", "call", 5),
    ]
    return build_hand_ledger(
        players, actions, blinds=BlindStructure(5, 10), ante_mode=AnteMode.PER_PLAYER
    )


def _contradictions(ledger) -> tuple[list[str], list[str]]:
    """Split a hand's issues into self-contradiction refusals and everything else."""

    flagged = [
        issue
        for issue in ledger.legality_issues
        if "but typed as a" in issue or "post status says" in issue
    ]
    return flagged, [
        issue for issue in ledger.legality_issues if issue not in flagged
    ]


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


# --------------------------------------------------------------------------
# The third field. Everything above sweeps two of the row's three facts.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ALL_POST_STATUSES)
@pytest.mark.parametrize("forced", (None, *ALL_FORCED_BET_TYPES))
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_full_kind_label_status_matrix_refuses_exactly_the_contradictions(
    kind: str, forced: str | None, status: bool | None
) -> None:
    """All three fields at once: what is refused, and what the refused row derives.

    Every cell that states two facts which cannot both be true raises EXACTLY one
    refusal, naming the pair that disagrees, and then derives from the facts that
    were not in the contradiction -- a rejected name leaves the status standing, a
    rejected pair leaves the bare kind. Nothing else about the hand moves: the
    remaining issues are the fallback row's, chip for chip and issue for issue.

    Every cell that states nothing contradictory raises no refusal at all, which
    is the half that stops this rule from turning ordinary recordings into
    blockers.
    """

    cell = _matrix_hand(kind, forced, status)
    conflict = _expected_conflict(kind, forced, status)
    flagged, remainder = _contradictions(cell)

    if conflict is None:
        assert flagged == []
        return

    fallback = _matrix_hand(kind, None, status if conflict == "kind" else None)
    assert len(flagged) == 1
    if conflict == "kind":
        assert "but typed as a" in flagged[0]
        assert f"booked as {kind!r}" in flagged[0]
        assert repr(forced) in flagged[0]
    else:
        assert "post status says " + ("live" if status else "dead") in flagged[0]
    # One row, one refusal, naming the pair that has to be resolved first. The
    # fallback row can carry a contradiction of its own -- an ``ante`` typed
    # ``big_blind`` and marked live contradicts itself twice, and fixing the name
    # leaves the status still disagreeing with the kind -- so what is compared
    # here is everything the two hands say APART from their own contradictions.
    assert remainder == _contradictions(fallback)[1]
    assert _signature(cell) == _signature(fallback)


@pytest.mark.parametrize("status", ALL_POST_STATUSES)
@pytest.mark.parametrize("forced", (None, *ALL_FORCED_BET_TYPES))
def test_no_post_status_clears_the_refusal_the_unstated_row_raises(
    forced: str | None, status: bool | None
) -> None:
    """The reported defect as a one-way property, over the hand that raised it.

    Blinds undeclared and a big blind all-in for 4: the bare row is refused as an
    unreadable forced post. A cell may add refusals and may not remove them. The
    two cells that legitimately may -- the ones where a SINGLE field says the
    post was dead, with nothing on the row to check it against -- are named
    rather than assumed, because naming them is the only honest way to say where
    this rule stops.
    """

    players = [
        LedgerPlayer("SB", 100, 0),
        LedgerPlayer("BB", 4, 1),
        LedgerPlayer("BTN", 100, 2),
    ]

    def hand(label: str | None, post_status: bool | None):
        return build_hand_ledger(
            players,
            [
                LedgerAction("SB", "preflop", "post_blind", 5),
                LedgerAction(
                    "BB",
                    "preflop",
                    "post_blind",
                    4,
                    is_live_post=post_status,
                    forced_bet_type=label,
                ),
                LedgerAction("BTN", "preflop", "call", 5),
            ],
        )

    # What the row still says once its contradictions have been set aside. A
    # single uncontradicted "this post was dead" is unfalsifiable and believed;
    # two facts that agree it was dead are that same single claim written twice.
    # A name the KIND rejected never reaches the liveness question at all, so the
    # status it left standing is a single claim too.
    conflict = _expected_conflict("post_blind", forced, status)
    reads_dead = (conflict == "kind" and status is False) or (
        conflict is None and (status is False or forced == "dead_blind")
    )
    bare = set(hand(None, None).legality_issues)
    cell = set(hand(forced, status).legality_issues)
    if reads_dead:
        assert bare - cell != set()
        return
    assert bare <= cell


@pytest.mark.parametrize("status", ALL_POST_STATUSES)
def test_the_reported_hand_derives_the_same_pot_whatever_the_post_status_says(
    status: bool | None,
) -> None:
    """The product boundary, reproduced. Only the Post status selectbox varies.

    Blinds undeclared, big blind all-in for 4 typed ``big_blind``. Live and
    unspecified refused it and laid out 12/2 with the big blind netting +8;
    dead laid out 4/10, netted the big blind 0, raised no issue whatsoever and
    presented the hand as study-ready. Eight chips of a fourteen-chip pot moved
    on one selectbox, and the refusal that was standing over the row named it an
    "all-in posting a LIVE forced bet" -- so the product's own message pointed at
    the box that made the message disappear.

    All three now derive the same chips and none of them is study-ready.
    """

    players = [
        LedgerPlayer("SB", 100, 0),
        LedgerPlayer("BB", 4, 1),
        LedgerPlayer("BTN", 100, 2),
    ]
    ledger = build_hand_ledger(
        players,
        [
            LedgerAction("SB", "preflop", "post_blind", 5),
            LedgerAction(
                "BB",
                "preflop",
                "post_blind",
                4,
                is_live_post=status,
                forced_bet_type="big_blind",
            ),
            LedgerAction("BTN", "preflop", "call", 5),
        ],
        winners={0: ("BB",), 1: ("SB",)},
    )

    assert [pot.amount for pot in ledger.pots] == [12, 2]
    assert ledger.net_results["BB"] == 8
    assert ledger.is_legal is False
    assert any(
        "all-in posting a live forced bet of 4" in issue
        for issue in ledger.legality_issues
    )
    if status is False:
        assert any("post status says dead" in issue for issue in ledger.legality_issues)


@pytest.mark.parametrize("status", ALL_POST_STATUSES)
def test_the_twin_knob_on_the_same_row_is_closed_with_it(
    status: bool | None,
) -> None:
    """The other selectbox, on the same row, reaching the same place.

    ``dead_blind`` on a live blind silences the same refusal and moves the same
    chips. Closing either knob alone buys nothing, so this asserts the pair: with
    the status stated live the row contradicts itself and is refused, and with
    the status unstated the single claim stands and the hand derives the dead
    reading -- which is the coverage limitation, stated rather than papered over.
    """

    players = [
        LedgerPlayer("SB", 100, 0),
        LedgerPlayer("BB", 4, 1),
        LedgerPlayer("BTN", 100, 2),
    ]
    ledger = build_hand_ledger(
        players,
        [
            LedgerAction("SB", "preflop", "post_blind", 5),
            LedgerAction(
                "BB",
                "preflop",
                "post_blind",
                4,
                is_live_post=status,
                forced_bet_type="dead_blind",
            ),
            LedgerAction("BTN", "preflop", "call", 5),
        ],
        winners={0: ("BB",), 1: ("SB",)},
    )

    if status is True:
        assert [pot.amount for pot in ledger.pots] == [12, 2]
        assert ledger.is_legal is False
        assert any("post status says live" in issue for issue in ledger.legality_issues)
        assert any(
            "all-in posting a live forced bet of 4" in issue
            for issue in ledger.legality_issues
        )
    else:
        assert [pot.amount for pot in ledger.pots] == [4, 10]
        assert tuple(ledger.legality_issues) == ()
