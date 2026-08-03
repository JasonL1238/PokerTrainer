"""The ante-mode declaration: the gate, the migration, and the mode boundary.

Adversarial round 21. The operator ruled three things, and this file is the
acceptance record for the two of them that are not chip arithmetic:

  RULING 2  The ante mode is an EXPLICIT DECLARED INPUT -- NONE, PER_PLAYER, or
            SINGLE_PAYER_TABLE_ANTE. A hand containing antes with no declared
            mode is AMBIGUOUS and is REFUSED, naming the missing declaration and
            the clearing action. It is never inferred from the shape of the
            posts.
  RULING 3  The dead-money cap is mode-dependent, and the mode governs EXACTLY
            ONE pool -- antes. Non-ante forced posts keep their existing capped
            treatment under every mode.

``tests/test_accounting_pot_layering_model.py`` holds the seven worked examples
and the chip figures. This file holds what the declaration DOES: when it refuses,
what a refused hand still derives, what the mode reaches and what it does not,
and the boundary cases (B1)-(B3) that the seven examples do not constrain.

WHY THE BOUNDARY CASES ARE HERE RATHER THAN INFERRED FROM THE EXAMPLES. Not one
of the seven mixes a consolidated ante with a dead blind, puts a straddle beside
a table ante, or contains external dead money at all. An implementation can pass
all seven and still be wrong about every one of those, and the mode-wide branch
in (B2) is the specific wrong implementation that does.
"""

from __future__ import annotations

import pytest

from poker_tracker.math.accounting import (
    UNDECLARED_ANTE_MODE_PREFIX,
    AnteMode,
    BlindStructure,
    LedgerAction,
    LedgerError,
    LedgerPlayer,
    build_hand_ledger,
    declared_ante_mode,
)

PER_PLAYER = AnteMode.PER_PLAYER
SINGLE_PAYER = AnteMode.SINGLE_PAYER_TABLE_ANTE
NONE = AnteMode.NONE


def _player(name: str, stack: float) -> LedgerPlayer:
    return LedgerPlayer(name=name, starting_stack=stack)


def _live(player: str, kind: str, amount: float, forced: str | None = None):
    return LedgerAction(
        player=player,
        street="preflop",
        kind=kind,
        amount=amount,
        forced_bet_type=forced,
    )


def _ante(player: str, amount: float, forced: str = "ante"):
    return LedgerAction(
        player=player,
        street="preflop",
        kind="ante",
        amount=amount,
        is_live_post=False,
        forced_bet_type=forced,
    )


def _dead_blind(player: str, amount: float):
    return LedgerAction(
        player=player,
        street="preflop",
        kind="post_blind",
        amount=amount,
        is_live_post=False,
        forced_bet_type="dead_blind",
    )


def _refusals(ledger) -> list[str]:
    return [
        issue
        for issue in ledger.legality_issues
        if "ante mode" in issue or "ante" in issue.lower() and "declare" in issue.lower()
    ]


# ---------------------------------------------------------------------------
# RULING 2: the declaration gate
# ---------------------------------------------------------------------------


def test_a_hand_with_antes_and_no_declared_mode_is_refused_not_defaulted():
    """The migration, and the whole of ruling 2, in one hand.

    The refusal has to do four things, and each of them is a separate way this
    could have shipped as a defect:

      * BLOCK. ``is_legal`` False, so ``persist_reconciliation`` writes
        ``needs_correction`` and study readiness stops. A hand laid out under one
        of two readings without recording which is not study-ready.
      * NAME THE SEATS. An operator cannot act on "this hand has antes"; they can
        act on "BB anted".
      * NAME THE DECLARATION AND WHERE IT LIVES -- beside the blind structure and
        the rake policy, which is the control the operator already knows.
      * STILL DERIVE EVERYTHING. A blocked hand must stay inspectable, exactly as
        an undeclared blind structure leaves it. The pot figures are still there;
        they are just not certified.
    """

    players = [_player("A", 100), _player("B", 100), _player("C", 1)]
    actions = [
        _ante("A", 1),
        _ante("B", 1),
        _ante("C", 1),
        _live("A", "bet", 10),
        _live("B", "call", 10),
    ]

    ledger = build_hand_ledger(players, actions, {0: ("C",), 1: ("A",)})

    assert ledger.is_legal is False
    issue = next(
        note
        for note in ledger.legality_issues
        if note.startswith(UNDECLARED_ANTE_MODE_PREFIX)
    )
    assert "'A'" in issue and "'B'" in issue and "'C'" in issue
    assert "NONE" in issue and "PER_PLAYER" in issue
    assert "SINGLE_PAYER_TABLE_ANTE" in issue
    assert "Declare the ante mode" in issue
    assert "blind structure" in issue and "rake policy" in issue

    # Fully inspectable: every figure is still derived.
    assert [pot.amount for pot in ledger.pots] == pytest.approx([3, 20])
    assert ledger.contributions == pytest.approx({"A": 11, "B": 11, "C": 1})
    assert ledger.net_results == pytest.approx({"A": 9, "B": -11, "C": 2})
    assert ledger.is_balanced is True


def test_a_hand_with_no_antes_needs_no_declaration_and_is_untouched():
    """The other half of the migration, and the half that must stay silent.

    ``NONE`` is not a guess for a hand with no antes -- it is the only thing such
    a hand can be -- so an absent declaration is resolved without a word. This is
    what keeps every ordinary cash-game hand in the store deriving byte-identically
    on the day the column ships, and it is asserted against the DECLARED readings
    rather than merely against an empty warning list, so a future default that
    quietly changed the layering would fail here.
    """

    players = [_player("SB", 100), _player("BB", 100), _player("BTN", 100)]
    actions = [
        _live("SB", "post_blind", 5, "small_blind"),
        _live("BB", "post_blind", 10, "big_blind"),
        _live("BTN", "call", 10),
        _live("SB", "call", 5),
        _live("BB", "check", 0),
    ]

    undeclared = build_hand_ledger(
        players, actions, {0: ("BB",)}, blinds=BlindStructure(5, 10)
    )
    assert undeclared.warnings == ()
    assert undeclared.legality_issues == ()
    assert undeclared.is_legal is True

    for mode in (NONE, PER_PLAYER, SINGLE_PAYER):
        declared = build_hand_ledger(
            players, actions, {0: ("BB",)}, blinds=BlindStructure(5, 10), ante_mode=mode
        )
        assert declared.warnings == undeclared.warnings
        assert declared.legality_issues == undeclared.legality_issues
        assert [pot.amount for pot in declared.pots] == pytest.approx(
            [pot.amount for pot in undeclared.pots]
        )
        assert declared.net_results == pytest.approx(undeclared.net_results)


def test_a_dead_blind_alone_is_not_an_ante_and_needs_no_declaration():
    """The mode is an ANTE mode: it names antes and nothing else.

    A returning player's dead blind is a forced post that is not live and is
    capped under every mode. Making it trigger the ante declaration would block a
    family the ruling never mentions, and an operator asked to declare an ante
    structure for a hand with no antes learns to answer the prompt at random.
    """

    players = [_player(name, 100) for name in ("a", "b", "c")]
    actions = [
        _dead_blind("b", 3),
        _live("a", "bet", 20),
        _live("b", "call", 20),
        _live("c", "call", 20),
    ]

    ledger = build_hand_ledger(players, actions, {0: ("a",)})
    assert ledger.legality_issues == ()
    assert ledger.warnings == ()
    assert ledger.is_legal is True
    assert [pot.amount for pot in ledger.pots] == pytest.approx([63])


def test_declaring_none_over_a_hand_that_antes_is_refused_as_a_contradiction():
    """Two operator-supplied facts disagree; the model refuses to pick one."""

    players = [_player("A", 100), _player("B", 100)]
    actions = [_ante("A", 1), _ante("B", 1), _live("A", "bet", 10), _live("B", "call", 10)]

    ledger = build_hand_ledger(players, actions, {0: ("A",)}, ante_mode=NONE)
    assert ledger.is_legal is False
    issue = next(note for note in ledger.legality_issues if "declared ante mode is NONE" in note)
    assert "'A'" in issue and "'B'" in issue
    assert "correct the ante rows" in issue
    assert "SINGLE_PAYER_TABLE_ANTE" in issue


def test_a_table_ante_declaration_over_two_anteing_seats_is_refused():
    """The mixed hand, flagged as a coverage limitation rather than answered.

    ``SINGLE_PAYER_TABLE_ANTE`` says ONE seat posts for the table. Two anteing
    seats under that declaration is a hand mixing a consolidated ante with a
    personal one, and there is no defensible guess: capping all of them breaks
    worked example (f), capping none breaks (e), and splitting them requires
    knowing which post was the consolidated one -- which is precisely the
    inference ruling 2 forbids. The plausible real shape is a big-blind ante plus
    a late-entry seat's own ante; if it turns up in real recordings the operator
    must rule, and a third mode (exactly one named post is table money) would be
    the natural answer. Inventing it here is not this module's call.
    """

    players = [_player("A", 100), _player("B", 100)]
    actions = [_ante("A", 1), _ante("B", 1), _live("A", "bet", 10), _live("B", "call", 10)]

    ledger = build_hand_ledger(players, actions, {0: ("A",)}, ante_mode=SINGLE_PAYER)
    assert ledger.is_legal is False
    issue = next(note for note in ledger.legality_issues if "SINGLE_PAYER_TABLE_ANTE" in note)
    assert "2 seats posted antes" in issue
    assert "'A'" in issue and "'B'" in issue
    assert "Declare PER_PLAYER" in issue

    # One anteing seat is fine under either reading -- that is the point of the
    # declaration -- so the refusal is about the AMBIGUITY and not about the mode.
    one_seat = build_hand_ledger(
        players,
        [_ante("A", 1), _live("A", "bet", 10), _live("B", "call", 10)],
        {0: ("A",)},
        ante_mode=SINGLE_PAYER,
    )
    assert one_seat.is_legal is True


def test_a_refused_hand_derives_under_the_capped_reading_and_says_so():
    """What a blocked hand shows, and why it is not a fourth mode.

    A refusal has to publish SOMETHING -- a hand nobody can inspect is not
    blocked, it is broken. The layers shown are the capped (PER_PLAYER) reading,
    for two reasons that both have to hold: it is the strict direction, so it can
    only understate a short seat's take; and it is byte-for-byte what this reducer
    derived before the mode existed, so a stored hand's displayed figures do not
    move underneath the operator on the day it starts blocking.

    That is a derivation the hand is BLOCKED on, not an inferred declaration, and
    the ledger says so in its own words rather than leaving the operator to guess
    which of the two readings they are looking at.
    """

    players = [_player("SB", 1), _player("BB", 4), _player("BTN", 2)]
    actions = [
        _live("SB", "post_blind", 1, "small_blind"),
        _live("BB", "post_blind", 2, "big_blind"),
        _ante("BB", 2, "big_blind_ante"),
        _live("BTN", "call", 2),
    ]

    refused = build_hand_ledger(players, actions, blinds=BlindStructure(1, 2))
    capped = build_hand_ledger(
        players, actions, blinds=BlindStructure(1, 2), ante_mode=PER_PLAYER
    )

    assert refused.is_legal is False
    assert [pot.amount for pot in refused.pots] == pytest.approx(
        [pot.amount for pot in capped.pots]
    )
    assert [pot.amount for pot in refused.pots] == pytest.approx([4, 3])
    note = "\n".join(refused.warnings)
    assert "capped (PER_PLAYER) reading" in note
    assert "not a decision about the mode" in note

    # And declaring the OTHER reading really does move the chips, which is what
    # makes the refusal worth having rather than bureaucracy.
    table_ante = build_hand_ledger(
        players, actions, blinds=BlindStructure(1, 2), ante_mode=SINGLE_PAYER
    )
    assert [pot.amount for pot in table_ante.pots] == pytest.approx([5, 2])


def test_an_unknown_ante_mode_is_refused_rather_than_ignored():
    """A corrupt declaration must not degrade into a missing one silently.

    ``declared_ante_mode`` is the single definition of "a mode was declared", and
    it refuses a value it does not know instead of dropping it -- dropping it
    would turn a corrupt declaration into an absent one, and an absent one derives
    under a different rule.

    An EMPTY STRING is read as absent, because that is what a NULL column
    round-tripped through a text field looks like, and "undeclared" is a state the
    product already handles loudly.
    """

    assert declared_ante_mode(None) is None
    assert declared_ante_mode("") is None
    assert declared_ante_mode(SINGLE_PAYER) == SINGLE_PAYER
    with pytest.raises(LedgerError):
        declared_ante_mode("BIG_BLIND_ANTE")

    players = [_player("A", 100), _player("B", 100)]
    actions = [_ante("A", 1), _live("A", "bet", 10), _live("B", "call", 10)]
    with pytest.raises(LedgerError):
        build_hand_ledger(players, actions, ante_mode="TABLE")


def test_the_mode_is_never_inferred_from_one_seat_having_anted():
    """The inference that is most tempting, and that the operator ruled against.

    Exactly one seat anted, which is what a big-blind ante looks like -- and also
    what a late-entry seat posting its own ante looks like. The two give different
    pots on this very hand. A reducer that "helpfully" read one anteing seat as a
    table ante would silently pay the small blind 5 instead of 4 on hands nobody
    declared, so the refusal is asserted here against the specific shape a future
    convenience default would take.
    """

    players = [_player("SB", 1), _player("BB", 4), _player("BTN", 2)]
    actions = [
        _live("SB", "post_blind", 1, "small_blind"),
        _live("BB", "post_blind", 2, "big_blind"),
        _ante("BB", 2, "big_blind_ante"),
        _live("BTN", "call", 2),
    ]

    ledger = build_hand_ledger(players, actions, blinds=BlindStructure(1, 2))
    assert ledger.is_legal is False
    assert any(
        note.startswith(UNDECLARED_ANTE_MODE_PREFIX) for note in ledger.legality_issues
    )


# ---------------------------------------------------------------------------
# THE MODE BOUNDARY: what the declaration reaches, and what it does not
# ---------------------------------------------------------------------------


def test_b1_a_straddle_beside_a_table_ante_is_live_and_nothing_interacts():
    """(B1) STRADDLES ARE LIVE MONEY AND NO MODE TOUCHES THEM.

    A straddle is a live structural forced bet -- it sets a wager level other
    seats must match -- and it sits in the same vocabulary as the small blind, the
    big blind and the bring-in. So under SINGLE_PAYER_TABLE_ANTE a hand carrying
    both a straddle and a consolidated ante lays the ante whole into the main pot
    with the straddle cutting an ordinary live boundary above it, and the two do
    not interact at all.

    1/2 blinds straddled to 4, a 10-chip table ante posted by the big blind,
    everyone in for the straddle: three live bands over an uncapped 10-chip main
    pot. Nothing about the layering changes if the ante is removed except the
    10 chips.
    """

    players = [
        _player("SB", 100),
        _player("BB", 100),
        _player("STR", 100),
        _player("BTN", 4),
    ]
    actions = [
        _live("SB", "post_blind", 1, "small_blind"),
        _live("BB", "post_blind", 2, "big_blind"),
        _live("STR", "post_blind", 4, "straddle"),
        _ante("BB", 10, "big_blind_ante"),
        _live("BTN", "all-in", 4),
        _live("SB", "call", 3),
        _live("BB", "call", 2),
    ]

    ledger = build_hand_ledger(
        players,
        actions,
        {0: ("BTN",)},
        blinds=BlindStructure(1, 2, (4.0,)),
        ante_mode=SINGLE_PAYER,
    )

    # One pot: every seat reached the same live level of 4, and the 10-chip table
    # ante sits in it whole.
    assert [pot.amount for pot in ledger.pots] == pytest.approx([26])
    assert ledger.pots[0].eligible_players == ("SB", "BB", "STR", "BTN")
    assert ledger.is_legal is True
    assert ledger.warnings == ()
    # The 4-chip seat wins the table ante whole, which is the exemption working:
    # capped against its own 4-chip total the ante would have been cut to 4.
    assert ledger.payouts["BTN"] == pytest.approx(26)


def test_b2_a_table_ante_beside_a_dead_blind_runs_both_rules_at_once():
    """(B2) THE ONE A MODE-WIDE BRANCH DIES ON, and no worked example covers it.

    Ruling 3's last clause leaves non-ante forced posts on their existing
    treatment, and the mode is an ANTE mode. So a SINGLE_PAYER_TABLE_ANTE hand
    carrying a dead blind runs BOTH dead-money rules at once, on disjoint pools:
    the consolidated ante bypasses the cascade into the main pot while the dead
    blind runs the cascade and may rise.

    live 5/5/5, a 100-chip consolidated ante from ``BBA`` and a 50-chip dead blind
    from ``DB``. The short seat's total is 5.

        main  120 = 15 live + the whole 100 ante + 5 of the dead blind
        above  45 = the rest of the dead blind, to the two seats above 5

    A REDUCER THAT BRANCHES ONCE ON THE MODE gives one pot of 165 and overpays the
    short seat by 45 -- and it passes all seven worked examples, because not one
    of them mixes a consolidated ante with a dead blind. This test is the whole
    reason the two pools are carried separately rather than summed.
    """

    players = [_player("BBA", 105), _player("DB", 55), _player("short", 5)]
    actions = [
        _ante("BBA", 100, "big_blind_ante"),
        _dead_blind("DB", 50),
        _live("BBA", "bet", 5),
        _live("DB", "call", 5),
        _live("short", "all-in", 5),
    ]

    ledger = build_hand_ledger(
        players, actions, {0: ("short",), 1: ("BBA",)}, ante_mode=SINGLE_PAYER
    )

    assert [pot.amount for pot in ledger.pots] == pytest.approx([120, 45])
    assert ledger.pots[0].eligible_players == ("BBA", "DB", "short")
    assert ledger.pots[1].eligible_players == ("BBA", "DB")
    assert ledger.payouts["short"] == pytest.approx(120)
    assert ledger.net_results["short"] == pytest.approx(115)
    assert ledger.is_balanced is True
    assert ledger.is_legal is True

    # Under PER_PLAYER both pools are capped, so the short seat reaches 5 of each
    # and the cascade runs a rung further: the ante's excess over the dead blind's
    # own total opens a third layer only its poster can win.
    per_player = build_hand_ledger(
        players,
        actions,
        {0: ("short",), 1: ("BBA",), 2: ("BBA",)},
        ante_mode=PER_PLAYER,
    )
    assert [pot.amount for pot in per_player.pots] == pytest.approx([25, 95, 45])
    assert per_player.pots[2].eligible_players == ("BBA",)
    assert per_player.payouts["short"] == pytest.approx(25)
    assert per_player.is_balanced is True


def test_b3_external_dead_money_is_capped_under_every_mode():
    """(B3) RULING 5, asserted where the mode could plausibly have reached it.

    External dead money is in the CAPPED pool under every mode, including
    SINGLE_PAYER_TABLE_ANTE. It has no seat, so it is not "one seat posts a
    consolidated ante for the table", and capping is the strict direction.

    The same hand as (B2) plus 60 chips typed in as external dead money. The short
    seat's total is 5, so it reaches 5 of the declared amount and the other 55
    rises -- exactly as a dead blind of the same size would.
    """

    players = [_player("BBA", 105), _player("DB", 55), _player("short", 5)]
    actions = [
        _ante("BBA", 100, "big_blind_ante"),
        _dead_blind("DB", 50),
        _live("BBA", "bet", 5),
        _live("DB", "call", 5),
        _live("short", "all-in", 5),
    ]

    ledger = build_hand_ledger(
        players,
        actions,
        {0: ("short",), 1: ("BBA",), 2: ("BBA",)},
        dead_money=60,
        ante_mode=SINGLE_PAYER,
    )

    # The main pot gains 5 of the declared money -- the short seat's own total --
    # and the remaining 55 runs the cascade above it exactly as a dead post of the
    # same size would, opening the same rungs.
    assert [pot.amount for pot in ledger.pots] == pytest.approx([125, 95, 5])
    assert ledger.pots[0].eligible_players == ("BBA", "DB", "short")
    assert ledger.payouts["short"] == pytest.approx(125)
    assert ledger.is_balanced is True
    # Uncapped -- the pre-ruling behaviour -- the main pot was 180 and this seat
    # took all 60 of the declared money on a 5-chip commitment. The cap is what
    # stops that, and this is the direction ruling 5 moves: strictly less.
    assert ledger.pots[0].amount < 180


def test_the_mode_reaches_antes_and_nothing_else():
    """The mode boundary as a table, asserted rather than documented.

    Switching the declaration between PER_PLAYER and SINGLE_PAYER_TABLE_ANTE may
    move ANTE chips and must move nothing else. Held to it by building the same
    hand five ways -- blinds only, a straddle, a dead blind, external dead money,
    and an ante -- and checking that only the ante build differs between the two
    modes.
    """

    base_players = [_player("A", 40), _player("B", 40), _player("short", 4)]
    base = [
        _live("A", "bet", 4),
        _live("B", "call", 4),
        _live("short", "all-in", 4),
    ]

    def pots(extra, **kwargs):
        return [
            pytest.approx(pot.amount)
            for pot in build_hand_ledger(
                base_players, [*extra, *base], **kwargs
            ).pots
        ]

    unmoved = [
        ([], {}),
        ([_live("A", "post_blind", 8, "straddle")], {"blinds": BlindStructure(1, 2, (8.0,))}),
        ([_dead_blind("A", 30)], {}),
        ([], {"dead_money": 30}),
    ]
    for extra, kwargs in unmoved:
        assert pots(extra, ante_mode=PER_PLAYER, **kwargs) == pots(
            extra, ante_mode=SINGLE_PAYER, **kwargs
        )

    # The ante, and only the ante, moves.
    with_ante = [_ante("A", 30, "big_blind_ante")]
    assert pots(with_ante, ante_mode=PER_PLAYER) != pots(
        with_ante, ante_mode=SINGLE_PAYER
    )


def test_a_short_ante_booked_as_an_all_in_is_still_an_ante():
    """The relabelling that the pool split has to survive.

    A big-blind ante that took its poster's last chip is routinely booked as
    ``all-in`` carrying ``forced_bet_type='big_blind_ante'`` -- the CV spine and
    the hand editor both write that row. Classifying the ante pool from the action
    KIND alone would put it in the non-ante pool, outside the consolidated-ante
    exemption, so worked example (f) would silently revert to the capped answer on
    every recording that spells its short ante that way. It would also silence the
    declaration gate on such a hand, which is worse: no refusal at all.
    """

    players = [_player("SB", 1), _player("BB", 4), _player("BTN", 2)]

    def hand(ante_row):
        return [
            _live("SB", "post_blind", 1, "small_blind"),
            _live("BB", "post_blind", 2, "big_blind"),
            ante_row,
            _live("BTN", "call", 2),
        ]

    plain = hand(_ante("BB", 2, "big_blind_ante"))
    relabelled = hand(
        LedgerAction(
            player="BB",
            street="preflop",
            kind="all-in",
            amount=2,
            is_live_post=False,
            forced_bet_type="big_blind_ante",
        )
    )

    for actions in (plain, relabelled):
        # The gate sees it either way.
        undeclared = build_hand_ledger(players, actions, blinds=BlindStructure(1, 2))
        assert any(
            note.startswith(UNDECLARED_ANTE_MODE_PREFIX)
            for note in undeclared.legality_issues
        )
        # And the exemption reaches it either way.
        declared = build_hand_ledger(
            players,
            actions,
            {0: ("SB",), 1: ("BB",)},
            blinds=BlindStructure(1, 2),
            ante_mode=SINGLE_PAYER,
        )
        assert [pot.amount for pot in declared.pots] == pytest.approx([5, 2])
        assert declared.net_results["SB"] == pytest.approx(4)


def test_an_ante_row_the_recording_typed_as_a_dead_blind_is_refused():
    """Two operator facts contradict, so the hand is refused rather than resolved.

    This test previously asserted that the label decides: a row spelled ``ante``
    but typed ``dead_blind`` was a dead blind, and so did not trigger the ante
    declaration. That reading is what let the Forced-post selectbox switch the
    declaration off. Tagging the only ante row anything else emptied the ante
    pool, and a hand the product exists to refuse became one it accepted, with
    no chip moving to make it visible.

    The rule the operator set for the ante mode itself settles this: refuse an
    ambiguous hand rather than infer the format. A kind and a label naming two
    different forced posts is that ambiguity, and neither one gets to win
    quietly. The chips are unchanged -- the labelled row lays out exactly as the
    same row with the field cleared -- so what the refusal costs is a correction,
    not a number.
    """

    players = [_player("A", 40), _player("short", 4), _player("B", 40)]
    actions = [
        LedgerAction(
            player="A",
            street="preflop",
            kind="ante",
            amount=30,
            is_live_post=False,
            forced_bet_type="dead_blind",
        ),
        _live("A", "bet", 4),
        _live("short", "all-in", 4),
        _live("B", "call", 4),
    ]

    cleared = [
        action
        if action.forced_bet_type is None
        else LedgerAction(
            player=action.player,
            street=action.street,
            kind=action.kind,
            amount=action.amount,
            is_live_post=action.is_live_post,
        )
        for action in actions
    ]

    # Undeclared, the contradiction is named AND the ante declaration still
    # fires -- which is the whole point: the label cannot switch it off.
    ledger = build_hand_ledger(players, actions, {0: ("short",), 1: ("A",)})
    assert ledger.is_legal is False
    assert any("two different forced posts" in issue for issue in ledger.legality_issues)
    assert any("no ante mode is declared" in issue for issue in ledger.legality_issues)

    # The refusal costs a correction, not a number: the labelled row derives
    # exactly as the same row with the field cleared.
    declared = build_hand_ledger(
        players, actions, {0: ("short",), 1: ("A",)}, ante_mode=PER_PLAYER
    )
    without_label = build_hand_ledger(
        players, cleared, {0: ("short",), 1: ("A",)}, ante_mode=PER_PLAYER
    )
    assert [pot.amount for pot in declared.pots] == pytest.approx([16, 26])
    assert declared.payouts["short"] == pytest.approx(16)
    assert [pot.amount for pot in declared.pots] == pytest.approx(
        [pot.amount for pot in without_label.pots]
    )
    assert declared.payouts == without_label.payouts
    # Only the verdict differs, and only in the safe direction.
    assert without_label.is_legal is True
    assert declared.is_legal is False
