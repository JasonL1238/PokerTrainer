"""The pot-layering model, stated as the seven hands that define it.

Adversarial round 21. ``poker_tracker.math.accounting`` produced a critical in
five consecutive rounds, four of them introduced by the repair to the previous
one, and every repair was argued from a hand rather than from a rule. The
operator fixed the rule in round 19, AMENDED RULE 2 in round 20, and in round 21
made the ANTE MODE an explicit declaration, made the dead-money cap
mode-dependent, and ruled on the folded post. These are the acceptance criteria:

  1. Pot layer boundaries are cut at distinct LIVE contribution levels, measured
     after uncalled-bet refunds have been returned. Live money is what a player
     CHOSE to wager. Forced posts are not live. UNCHANGED by every ruling.
  2. THE ANTE MODE IS AN EXPLICIT DECLARED INPUT -- NONE, PER_PLAYER, or
     SINGLE_PAYER_TABLE_ANTE. A hand containing antes with no declared mode is
     AMBIGUOUS and is refused, naming the declaration and the clearing action.
     It is never inferred from the shape of the posts.
  3. Dead money goes into the LOWEST layer, but each contributor's CAPPED dead
     chips count into a layer only up to the smallest TOTAL commitment among that
     layer's eligible seats; the excess rises into the layer above, eligible to
     the seats whose own total reached past the cap. Under PER_PLAYER every ante
     is capped (the rule shipped in round 20, RETAINED). Under
     SINGLE_PAYER_TABLE_ANTE the consolidated ante is TABLE MONEY: it goes whole
     into the main pot and is never capped. Dead blinds and other non-ante forced
     posts are capped under EVERY mode -- the mode names antes only.
  4. A seat is eligible for a layer if its own LIVE contribution reaches that
     layer's level. Every unfolded seat that put ANY chip up -- live or dead --
     is eligible for the main pot.
  5. A folded seat's chips stay in the layers they reached and it is eligible for
     none. Its forced post that no surviving seat could cover is ABANDONED to
     whoever wins, rather than blocking the hand.
  6. EXTERNAL dead money -- an amount the operator types -- is capped exactly
     like a recorded dead post, under whichever rule the mode selects.

The seven worked examples below are the operator's, verbatim, and each must come
out to the chip. Two of them ((a) and (b)) were WRONG on the round-18 reducer and
reported settled, balanced, legal and warning-free while being wrong; two of them
((c) and (d)) were right and are here so the repair to the first two cannot be
paid for out of them; the fifth ((e)) is the one round 20's amendment exists for;
(f) and (g) are what round 21's rulings move. (a)-(e) MUST BE UNCHANGED by round
21 -- if any of them moves, the implementation is wrong and not the
specification. The remaining tests pin the cases the specification leaves to a
decision, and say which decision was taken.
"""

from __future__ import annotations

import pytest

from poker_tracker.math.accounting import (
    AnteMode,
    BlindStructure,
    LedgerAction,
    LedgerError,
    LedgerPlayer,
    RakePolicy,
    build_hand_ledger,
)

# Rulings 2 and 3. Every ante-bearing hand below now DECLARES how its antes
# were taken, because an undeclared mode on a hand containing antes is a
# refusal rather than a default. Which mode each hand declares is chosen to
# match what the hand describes -- a big-blind ante is SINGLE_PAYER, seats
# anteing individually are PER_PLAYER -- and the chip figures asserted here
# are UNCHANGED from before the modes existed. That is the acceptance
# criterion: worked examples (a)-(e) must not move.
PER_PLAYER = AnteMode.PER_PLAYER
SINGLE_PAYER = AnteMode.SINGLE_PAYER_TABLE_ANTE


def _player(name: str, stack: float) -> LedgerPlayer:
    return LedgerPlayer(name=name, starting_stack=stack)


def _live(player: str, kind: str, amount: float, street: str = "preflop") -> LedgerAction:
    return LedgerAction(player=player, street=street, kind=kind, amount=amount)


def _dead(player: str, amount: float, kind: str = "ante") -> LedgerAction:
    return LedgerAction(
        player=player, street="preflop", kind=kind, amount=amount, is_live_post=False
    )


# --- (a) the big-blind ante ---------------------------------------------------


def _big_blind_ante_hand():
    """Blinds 10/20, a 10 big-blind ante, and a big blind with 26 behind.

    live: BB 16, SB 20, BTN 20.  dead: BB 10.

    The shipped reducer derived ONE pot of 66 and paid the big blind 40, because
    it cut its first boundary at the big blind's TOTAL of 26 -- 16 live plus its
    own 10 ante -- and charged each opponent 20 into it. Neither opponent wagered
    more than 16 against the big blind's live money. The four extra chips from
    each of them were reported as settled, balanced and legal with no warning.
    """

    players = [_player("BB", 26), _player("SB", 500), _player("BTN", 500)]
    actions = [
        _dead("BB", 10),
        _live("SB", "post_blind", 10),
        _live("BB", "post_blind", 16),
        _live("BTN", "call", 20),
        _live("SB", "call", 10),
    ]
    return players, actions


def test_a_big_blind_ante_does_not_charge_opponents_into_the_short_blinds_layer():
    players, actions = _big_blind_ante_hand()
    structure = BlindStructure(10, 20)
    layers = build_hand_ledger(
        players, actions, blinds=structure, ante_mode=SINGLE_PAYER
    ).pots

    assert [pot.amount for pot in layers] == pytest.approx([58, 8])
    assert [pot.cause for pot in layers] == ["main", "side"]
    # 16 of live money from each of three seats, plus the whole 10-chip ante.
    assert layers[0].eligible_players == ("BB", "SB", "BTN")
    # The 4 live chips each of the two deep seats put in above the big blind.
    assert layers[1].eligible_players == ("SB", "BTN")

    ledger = build_hand_ledger(
        players,
        actions,
        {0: ("BB",), 1: ("SB",)},
        blinds=structure,
        ante_mode=SINGLE_PAYER,
    )
    assert ledger.payouts["BB"] == pytest.approx(58)
    assert ledger.net_results == pytest.approx({"BB": 32, "SB": -12, "BTN": -20})
    assert ledger.is_balanced is True
    assert ledger.is_legal is True
    assert ledger.warnings == ()
    assert ledger.legality_issues == ()


def test_the_big_blind_ante_hand_refuses_the_award_it_used_to_pay():
    """The declaration that produced +40 is now outside the big blind's reach."""

    players, actions = _big_blind_ante_hand()
    with pytest.raises(LedgerError):
        build_hand_ledger(
            players,
            actions,
            {0: ("BB",), 1: ("BB",)},
            blinds=BlindStructure(10, 20),
            ante_mode=SINGLE_PAYER,
        )


# --- (b) the ante-only seat whose totals coincide -----------------------------


def test_an_ante_only_seat_wins_its_own_ante_back_and_nothing_else():
    """The case a total-commitment layering is STRUCTURALLY unable to express.

    ``ao`` posts a 7 ante and wagers nothing live; two opponents wager 7 live
    each. Every seat's TOTAL commitment is 7, so a layering cut at total levels
    has exactly one level to cut at, finds one pot of 21, and lets ``ao`` be
    declared the winner of all of it -- +14 on a seat no opponent matched a chip
    of. There is no version of the old rule that produces the right answer here,
    which is why the rule had to change rather than be patched again.

    Cut at LIVE levels the hand is two pots: the ante alone, which ``ao`` is owed
    because rule 2 puts every dead chip in the lowest layer, and the 14 of live
    betting ``ao`` never covered. ``ao`` wins the main pot and comes out at
    exactly zero.
    """

    players = [_player("ao", 7), _player("X", 100), _player("Y", 100)]
    actions = [_dead("ao", 7), _live("X", "bet", 7), _live("Y", "call", 7)]

    layers = build_hand_ledger(players, actions, ante_mode=SINGLE_PAYER).pots
    assert [pot.amount for pot in layers] == pytest.approx([7, 14])
    assert layers[0].eligible_players == ("ao", "X", "Y")
    assert layers[1].eligible_players == ("X", "Y")

    ledger = build_hand_ledger(
        players, actions, {0: ("ao",), 1: ("X",)}, ante_mode=SINGLE_PAYER
    )
    assert ledger.payouts["ao"] == pytest.approx(7)
    assert ledger.net_results == pytest.approx({"ao": 0, "X": 7, "Y": -7})
    assert ledger.is_balanced is True

    with pytest.raises(LedgerError):
        build_hand_ledger(
            players, actions, {0: ("ao",), 1: ("ao",)}, ante_mode=SINGLE_PAYER
        )


# --- (c) the round-3 guarantee, which must survive ----------------------------


def test_a_seat_all_in_from_its_ante_still_wins_the_antes_it_is_owed():
    """Three-handed, ante 1 each, C all-in from its ante alone.

    The dead pool is 3 -- one chip from each seat -- so C is owed all three, and
    the 20 of live betting is a layer above it. This is the round-3 guarantee and
    it is unchanged by the model; it is here so a future repair cannot satisfy
    example (b) by taking it away. Compare
    ``test_an_ante_only_seat_wins_its_own_ante_back_and_nothing_else``: the
    difference between +2 and 0 is entirely whether the opponents owed antes too.
    """

    players = [_player("A", 100), _player("B", 100), _player("C", 1)]
    actions = [
        _dead("A", 1),
        _dead("B", 1),
        _dead("C", 1),
        _live("A", "bet", 10),
        _live("B", "call", 10),
    ]

    layers = build_hand_ledger(players, actions, ante_mode=PER_PLAYER).pots
    assert [pot.amount for pot in layers] == pytest.approx([3, 20])
    assert layers[0].eligible_players == ("A", "B", "C")
    assert layers[1].eligible_players == ("A", "B")

    ledger = build_hand_ledger(
        players, actions, {0: ("C",), 1: ("A",)}, ante_mode=PER_PLAYER
    )
    assert ledger.net_results == pytest.approx({"A": 9, "B": -11, "C": 2})
    assert ledger.is_balanced is True


# --- (d) the phantom side pot, which must stay gone ---------------------------


@pytest.mark.parametrize("winner", ["a", "b", "c", "d"])
def test_unequal_dead_money_is_one_pot_any_of_the_four_may_win(winner: str):
    """A posts a 5 ante, B a 3 dead blind, all four wager 20 live.

    Nobody is all-in and nobody declined a chip, so there is nothing for a second
    layer to hold apart: one pot of 88, and any of the four may be declared its
    winner. Parameterised over all four because the phantom's whole signature was
    that the two seats owing dead money could win chips the other two could not.
    """

    players = [_player(name, 100) for name in "abcd"]
    actions = [
        _dead("a", 5),
        _dead("b", 3, kind="post_blind"),
        _live("a", "bet", 20),
        _live("b", "call", 20),
        _live("c", "call", 20),
        _live("d", "call", 20),
    ]

    # Asserted under BOTH modes, because worked example (d) does not
    # discriminate: PER_PLAYER caps a's 5 against the 20-chip floor (no
    # effect) and SINGLE_PAYER exempts it (no effect). A mode that moved
    # this hand would be reaching money the ruling does not name.
    for mode in (PER_PLAYER, SINGLE_PAYER):
        layers = build_hand_ledger(players, actions, ante_mode=mode).pots
        assert [pot.amount for pot in layers] == pytest.approx([88])
        assert layers[0].eligible_players == ("a", "b", "c", "d")

        ledger = build_hand_ledger(
            players, actions, {0: (winner,)}, ante_mode=mode
        )
        assert ledger.payouts[winner] == pytest.approx(88)
        assert ledger.is_balanced is True
        assert ledger.warnings == ()
        assert ledger.legality_issues == ()


# --- What the specification leaves open, and what was decided -----------------


def test_a_layer_no_remaining_seat_can_win_is_stranded_and_never_merged_down():
    """The sixth critical, refused in advance.

    A truncated recording can leave every seat that wagered at the top live level
    folded. The reducer used to merge such a layer DOWN -- "folded money is
    abandoned to whoever wins: merge it down rather than strand it" -- and that
    merge was not cap-checked. Here it would put 14 chips wagered between two
    seats that both left into the pot a seat all-in for 3 can win, paying it 26
    where the table matched 12.

    The model has no rule that reaches those chips, so they stay in the band they
    landed in and that band has no eligible seat. The hand is therefore
    permanently unsettleable and reports so: a named legality issue, ``is_legal``
    False, and the standing unsettled warning. Every chip figure is still derived,
    because a blocked hand still has to be inspectable -- but nothing about it can
    present as reconciled, and no award can move those chips to anybody.
    """

    players = [_player("A", 3), _player("B", 3), _player("F1", 100), _player("F2", 100)]
    actions = [
        _live("A", "all-in", 3),
        _live("B", "all-in", 3),
        _live("F1", "raise", 10),
        _live("F2", "call", 10),
        _live("F1", "fold", 0, street="flop"),
        _live("F2", "fold", 0, street="flop"),
    ]

    ledger = build_hand_ledger(players, actions)
    assert [pot.amount for pot in ledger.pots] == pytest.approx([12, 14])
    assert ledger.pots[0].eligible_players == ("A", "B")
    assert ledger.pots[1].eligible_players == ()

    assert ledger.is_legal is False
    assert any(
        "no player still in the hand can win" in issue
        for issue in ledger.legality_issues
    )
    assert ledger.is_settled is False
    assert ledger.is_balanced is False
    assert ledger.warnings

    # And there is no award that reaches the stranded chips.
    with pytest.raises(LedgerError):
        build_hand_ledger(players, actions, {0: ("A",), 1: ("A",)})


def test_dead_money_whose_every_contributor_folded_is_refused_not_awarded():
    """Nobody still in the hand put a chip up, so nobody can be declared to win.

    Rule 3 makes eligibility for the main pot follow a seat's OWN chips. When
    every seat that contributed has folded there is no seat the rule reaches, and
    inventing one would be a guess about a hand the recording does not describe.
    """

    players = [_player("A", 100), _player("B", 100)]
    actions = [_dead("A", 1), _live("A", "fold", 0), _live("B", "fold", 0)]
    # Refused under EVERY mode: ruling 4 abandons a folded seat's post to
    # whoever wins the layer, and here there is no layer and no winner --
    # rule 3 leaves nobody eligible for the main pot. Ruling 4 does not
    # reach this shape and the refusal stands, which is deliberate.
    for mode in (PER_PLAYER, SINGLE_PAYER, None):
        with pytest.raises(LedgerError):
            build_hand_ledger(players, actions, ante_mode=mode)


def test_declared_dead_money_confers_eligibility_on_nobody_by_itself():
    """External dead money joins the main pot; it does not buy anybody into it.

    A seat becomes eligible for the main pot through its OWN chips. Declared dead
    money has no contributing seat, so on a hand where no seat contributed at all
    it leaves nobody eligible and is refused rather than being split between seats
    that put nothing up. Where seats DID contribute it simply joins their pot, and
    a folded seat is no more eligible for it than for anything else.
    """

    with pytest.raises(LedgerError):
        build_hand_ledger([_player("A", 100), _player("B", 100)], [], dead_money=5)

    ledger = build_hand_ledger(
        [_player("A", 100), _player("B", 100), _player("C", 100)],
        [_live("A", "bet", 10), _live("B", "call", 10), _live("C", "fold", 0)],
        dead_money=5,
    )
    assert [pot.amount for pot in ledger.pots] == pytest.approx([25])
    assert ledger.pots[0].eligible_players == ("A", "B")


def test_a_seat_whose_only_live_post_was_returned_still_contests_the_dead_money():
    """"Put any chip up" is measured before the uncalled bet comes back.

    Rule 3 says every unfolded seat that put ANY chip in contests the main pot. A
    seat whose only live post was returned because nobody had a chip left to call
    it did put a chip in; the return is what the table did about it afterwards,
    and it does not un-play the hand. Reading the rule against the post-refund
    figure instead would make a hand this seat won unrecordable, which is the
    failure every widening in this module has been a reaction to.

    This is the one place the model's own reference oracle cannot express the
    rule: it is handed post-refund live money, so it reads such a seat as having
    put nothing in. 658 of 60,000 generated hands hit it and it is the ONLY
    systematic difference between the two.

    ROUND 20: THE AMENDMENT MEETS THAT READING, AND THE COLLISION IS DISCLOSED.
    Amended rule 2 caps each contributor's dead money at the smallest TOTAL
    COMMITMENT among the layer's eligible seats. ``bob``'s total commitment is 1
    before the refund and 0 after it, and the specification never has to choose
    between them because none of the five worked examples contains a refund.
    Reading it after the refund caps the main pot at zero, empties it, and makes
    exactly the hand this test exists for unrecordable -- so the layering reads it
    at the same point rule 3 is read, WHAT THE SEAT PUT UP, and says so out loud
    in a warning rather than publishing the figure as settled fact.

    The chips therefore moved: this used to be one pot of 2 with ``bob`` winning
    all of it, which is unconditional rule 2. It is now 1 that ``bob`` may win --
    the chip of alice's ante that bob's own chip reached -- and 1 that only alice
    does. Bob's net is +1 either way, and +1 is also what a real table produces
    by returning alice's uncalled excess instead of bob's post, so the amendment
    moved the layer without moving the outcome.
    """

    players = [_player("alice", 2), _player("bob", 1)]
    actions = [_dead("alice", 2), _live("bob", "post_blind", 1)]

    ledger = build_hand_ledger(
        players, actions, {0: ("bob",), 1: ("alice",)}, ante_mode=PER_PLAYER
    )
    assert ledger.refunds["bob"] == pytest.approx(1)
    assert [pot.amount for pot in ledger.pots] == pytest.approx([1, 1])
    assert ledger.pots[0].eligible_players == ("alice", "bob")
    assert ledger.pots[1].eligible_players == ("alice",)
    assert ledger.net_results == pytest.approx({"alice": -1, "bob": 1})
    assert ledger.is_balanced is True

    note = "\n".join(ledger.warnings)
    assert "'bob'" in note and "uncalled bet" in note


def test_rake_and_a_chop_cannot_move_a_chip_across_a_layer_boundary():
    """Rake comes out of the layer it was charged to, and a split stays inside one.

    Stated on the big-blind ante hand, whose two layers have different eligible
    sets and very different sizes, because that is where an allocator that took
    from the wrong pot or a split that rounded across a boundary would show.
    """

    players, actions = _big_blind_ante_hand()
    structure = BlindStructure(10, 20)
    policy = RakePolicy(rate=0.05, cap=5.0, rounding_unit=1.0)

    ledger = build_hand_ledger(
        players,
        actions,
        {0: ("BB", "SB"), 1: ("SB", "BTN")},
        rake=policy,
        blinds=structure,
        odd_chip_order=("BB", "SB", "BTN"),
    )
    for pot in ledger.pots:
        assert pot.rake >= 0
        assert pot.rake <= pot.amount
        assert pot.net_amount == pytest.approx(pot.amount - pot.rake)
    assert sum(pot.net_amount for pot in ledger.pots) == pytest.approx(
        sum(ledger.payouts.values())
    )
    assert sum(ledger.payouts.values()) + ledger.rake == pytest.approx(
        ledger.gross_pot
    )
    assert ledger.is_balanced is True
    # BTN is out of the main pot, so nothing it is paid can have come from there.
    assert ledger.payouts["BTN"] <= ledger.pots[1].net_amount + 1e-9


# --- the money classifier: what counts as LIVE -------------------------------


_DEAD_FORCED_BET_TYPES = ("ante", "big_blind_ante", "dead_blind")


def _round3_hand(c_row: LedgerAction):
    """Worked example (c), with the short seat's forced post spelled by caller.

    Three-handed, ante 1 each, C is all-in from its ante alone, A and B wager 10.
    Truth: main 3 [A, B, C], side 20 [A, B], C net +2.
    """

    players = [_player("A", 100), _player("B", 100), _player("C", 1)]
    actions = [
        _dead("A", 1),
        _dead("B", 1),
        c_row,
        _live("A", "bet", 10),
        _live("B", "call", 10),
    ]
    return players, actions


@pytest.mark.parametrize("forced_bet_type", _DEAD_FORCED_BET_TYPES)
def test_a_forced_post_is_dead_money_whatever_action_kind_carries_it(forced_bet_type):
    """Rule 1 is about the forced post, not about how a recording spelled it.

    A forced post that took its poster's last chip is routinely booked as
    ``all-in`` carrying a ``forced_bet_type``; the CV spine and the hand editor
    both produce that row, and ``actions.forced_bet_type`` /
    ``actions.is_live_post`` are durable columns an operator sets from two
    separate selectboxes. The money classifier decided liveness from
    ``action.kind`` alone, so every one of those rows was counted as chosen live
    money. Under the live-level model live money is the ONLY thing that opens a
    boundary, so the forced post became a live level: worked example (c) paid its
    ante-only seat +4 instead of +2 and reported settled, balanced, legal and
    warning-free.

    The same identification the blind-structure refusal already used
    (``_is_forced_post`` / ``_is_live_structural_post``) now decides the money,
    so every spelling of one event derives byte-identical chips.
    """

    truth = build_hand_ledger(
        *_round3_hand(_dead("C", 1)),
        winners={0: ("C",), 1: ("A",)},
        ante_mode=PER_PLAYER,
    )
    relabelled = build_hand_ledger(
        *_round3_hand(
            LedgerAction(
                player="C",
                street="preflop",
                kind="all-in",
                amount=1,
                is_live_post=False,
                forced_bet_type=forced_bet_type,
            )
        ),
        winners={0: ("C",), 1: ("A",)},
        ante_mode=PER_PLAYER,
    )

    assert [pot.amount for pot in truth.pots] == pytest.approx([3, 20])
    assert truth.net_results["C"] == pytest.approx(2)
    assert [pot.amount for pot in relabelled.pots] == pytest.approx([3, 20])
    assert relabelled.pots[0].eligible_players == ("A", "B", "C")
    assert relabelled.pots[1].eligible_players == ("A", "B")
    assert relabelled.net_results == pytest.approx(truth.net_results)
    assert relabelled.contributions == pytest.approx(truth.contributions)
    assert relabelled.payouts == pytest.approx(truth.payouts)
    assert relabelled.is_legal is True
    assert relabelled.warnings == ()
    assert relabelled.legality_issues == ()


def test_a_dead_blind_the_recording_named_is_dead_even_with_no_post_status():
    """The two operator-facing fields cannot disagree with the silent default.

    ``is_live_post`` defaults to True when a recording says nothing, and the hand
    editor exposes "Forced post" and "Post status" as separate selectboxes. A row
    spelled ``post_blind`` and typed ``dead_blind`` with the status left
    unspecified was therefore read as a live wager, which under the live-level
    model manufactured a live level out of a dead blind. Where the recording
    names the forced bet, that name decides.
    """

    players = [_player(name, 100) for name in ("a", "b", "c", "d")]

    def hand(b_row: LedgerAction):
        return [
            _dead("a", 5),
            b_row,
            _live("a", "bet", 20),
            _live("b", "call", 20),
            _live("c", "call", 20),
            _live("d", "call", 20),
        ]

    explicit = build_hand_ledger(
        players,
        hand(_dead("b", 3, kind="post_blind")),
        winners={0: ("a",)},
        ante_mode=PER_PLAYER,
    )
    named_only = build_hand_ledger(
        players,
        hand(
            LedgerAction(
                player="b",
                street="preflop",
                kind="post_blind",
                amount=3,
                forced_bet_type="dead_blind",
            )
        ),
        winners={0: ("a",)},
        ante_mode=PER_PLAYER,
    )

    # Worked example (d): one pot of 88, no phantom side pot, no legality noise.
    assert [pot.amount for pot in explicit.pots] == pytest.approx([88])
    assert [pot.amount for pot in named_only.pots] == pytest.approx([88])
    assert named_only.pots[0].eligible_players == ("a", "b", "c", "d")
    assert named_only.net_results == pytest.approx(explicit.net_results)
    assert named_only.is_legal is True
    assert named_only.legality_issues == ()


def test_a_live_blind_booked_as_an_all_in_is_still_live_money():
    """The repair must not sweep live forced bets into the dead pool.

    ``small_blind``/``big_blind``/``straddle``/``bring_in`` are structural LIVE
    forced bets: they set what the table owes, and a short one is still a wager.
    The dead vocabulary is exactly the three names in
    ``_DEAD_FORCED_BET_TYPES``.
    """

    players = [_player("SB", 4), _player("BB", 100), _player("BTN", 100)]

    def hand(sb_row: LedgerAction):
        return [
            sb_row,
            _live("BB", "post_blind", 10),
            _live("BTN", "call", 10),
            _live("BB", "check", 0),
        ]

    plain = build_hand_ledger(
        players,
        hand(_live("SB", "post_blind", 4)),
        winners={0: ("SB",), 1: ("BB",)},
        blinds=BlindStructure(5, 10),
    )
    booked_all_in = build_hand_ledger(
        players,
        hand(
            LedgerAction(
                player="SB",
                street="preflop",
                kind="all-in",
                amount=4,
                forced_bet_type="small_blind",
            )
        ),
        winners={0: ("SB",), 1: ("BB",)},
        blinds=BlindStructure(5, 10),
    )

    assert [pot.amount for pot in plain.pots] == pytest.approx([12, 12])
    assert [pot.amount for pot in booked_all_in.pots] == pytest.approx([12, 12])
    assert booked_all_in.net_results == pytest.approx(plain.net_results)


# --- (e) the amendment, and what rule 2 still does not decide ----------------


def test_the_amendment_pays_a_short_seat_what_the_table_matched_and_no_more():
    """Worked example (e). The hand the amendment exists for.

    ``short``'s whole commitment is a 60-chip forced post; three opponents post
    100-chip antes and wager nothing live. Under unconditional rule 2 the whole
    360 sat in one main pot and ``short`` could be declared the winner of all of
    it -- 300 chips more than any opponent had covered of a 60-chip seat -- and
    the previous round could not tell from worked examples (a)-(d) whether that
    was right, so it published the number and refused to call the hand
    study-ready. The operator has ruled.

    Each opponent's ante counts into the lowest layer only up to 60, the smallest
    total commitment among the seats eligible for it, so the main pot is
    60 + 3 x 60 = 240 and the 40 x 3 excess rises into a layer eligible to the
    three seats whose own total reached past 60. ``short`` wins 240 and nets +180.

    THE LAYER ABOVE IS DEAD MONEY, WHICH THE PREVIOUS MODEL SAID COULD NEVER
    HAPPEN. There is no live money anywhere in this hand, so rule 1 cuts no
    boundary at all; the boundary is rule 2's cap. Anything downstream that
    assumes a layer above the main pot holds only live wagering is now wrong.
    """

    players = [
        _player("short", 60),
        _player("o1", 500),
        _player("o2", 500),
        _player("o3", 500),
    ]
    actions = [
        _dead("short", 60),
        _dead("o1", 100),
        _dead("o2", 100),
        _dead("o3", 100),
    ]

    layers = build_hand_ledger(players, actions, ante_mode=PER_PLAYER).pots
    assert [pot.amount for pot in layers] == pytest.approx([240, 120])
    assert layers[0].eligible_players == ("short", "o1", "o2", "o3")
    assert layers[1].eligible_players == ("o1", "o2", "o3")
    assert [pot.cause for pot in layers] == ["main", "side"]

    ledger = build_hand_ledger(
        players, actions, {0: ("short",), 1: ("o1",)}, ante_mode=PER_PLAYER
    )
    assert ledger.payouts["short"] == pytest.approx(240)
    assert ledger.net_results == pytest.approx(
        {"short": 180, "o1": 20, "o2": -100, "o3": -100}
    )
    assert ledger.is_balanced is True
    assert ledger.is_legal is True
    assert ledger.warnings == ()

    # And the award unconditional rule 2 allowed is now outside short's reach.
    with pytest.raises(LedgerError):
        build_hand_ledger(
            players, actions, {0: ("short",), 1: ("short",)}, ante_mode=PER_PLAYER
        )


def test_the_forced_post_no_main_pot_seat_could_cover_is_now_answered():
    """The hand round 19 refused: 120 to a 40-chip seat, and study-ready.

    Antes of 100 with a 40-chip stack short of its own ante -- the family round 19
    named in a warning and declined to call authoritative, because unconditional
    rule 2 put every ante whole into the lowest layer and paid ``btn`` all 240 of
    them on a 40-chip commitment. (The same shape six-handed is the 540 the
    warning's text quoted.)

    The operator's amendment caps each CONTRIBUTOR's ante at 40 -- the smallest
    total commitment among the seats eligible for the lowest layer -- so the main
    pot is 3 x 40: ``btn``'s own 40-chip post plus 40 of each 100-chip ante, the
    folded small blind's included, because rule 4 leaves a folded seat's chips in
    the layers they reached. Everything above rises to ``bb``, the one seat whose
    total reached past 40, and joins its live band there.

    ``btn`` collects 120: its own 40, plus 40 of ``sb``'s ante and 40 of ``bb``'s.
    That is exactly what the table matched of a 40-chip commitment. The warning is
    withdrawn because the question it named has been answered, and the hand is
    legal, balanced, settled and warning-free.
    """

    players = [
        _player("btn", 40),
        _player("sb", 5000),
        _player("bb", 5000),
    ]
    actions = [
        _dead("btn", 40),
        _dead("sb", 100),
        _dead("bb", 100),
        _live("sb", "post_blind", 100),
        _live("bb", "post_blind", 200),
        _live("sb", "fold", 0),
    ]

    ledger = build_hand_ledger(
        players,
        actions,
        winners={0: ("btn",), 1: ("bb",)},
        blinds=BlindStructure(100, 200),
        ante_mode=PER_PLAYER,
    )

    assert [pot.amount for pot in ledger.pots] == pytest.approx([120, 320])
    assert ledger.pots[0].eligible_players == ("btn", "bb")
    assert ledger.pots[1].eligible_players == ("bb",)
    assert ledger.payouts["btn"] == pytest.approx(120)
    assert ledger.net_results["btn"] == pytest.approx(80)
    assert ledger.is_legal is True
    assert ledger.is_balanced is True
    assert ledger.is_settled is True
    assert ledger.warnings == ()


def test_the_seven_worked_examples_are_answered_without_a_warning():
    """No refusal may fire where the operator has ruled, over all SEVEN examples.

    (a) and (b) the unmatched post is the SHORT seat's own; (c) each opponent's
    ante equals the short seat's whole commitment; (d) nobody is short; (e) is the
    capped cascade itself; (f) is the consolidated-ante exemption; (g) is the
    folded post ruling 4 abandons to the pot. A check that fired on any of them
    would be overriding the specification instead of naming the hand it does not
    cover -- which is exactly what the round-19 refusal would do to (e), and what
    the round-20 refusal would do to (g).
    """

    hands = [
        (
            _big_blind_ante_hand(),
            {0: ("BB",), 1: ("SB",)},
            BlindStructure(10, 20),
            SINGLE_PAYER,
        ),
        (
            (
                [_player("ao", 7), _player("X", 100), _player("Y", 100)],
                [_dead("ao", 7), _live("X", "bet", 7), _live("Y", "call", 7)],
            ),
            {0: ("ao",), 1: ("X",)},
            None,
            SINGLE_PAYER,
        ),
        (_round3_hand(_dead("C", 1)), {0: ("C",), 1: ("A",)}, None, PER_PLAYER),
        (
            (
                [_player(name, 100) for name in ("a", "b", "c", "d")],
                [
                    _dead("a", 5),
                    _dead("b", 3, kind="post_blind"),
                    _live("a", "bet", 20),
                    _live("b", "call", 20),
                    _live("c", "call", 20),
                    _live("d", "call", 20),
                ],
            ),
            {0: ("c",)},
            None,
            PER_PLAYER,
        ),
        (
            (
                [
                    _player("short", 60),
                    _player("o1", 500),
                    _player("o2", 500),
                    _player("o3", 500),
                ],
                [
                    _dead("short", 60),
                    _dead("o1", 100),
                    _dead("o2", 100),
                    _dead("o3", 100),
                ],
            ),
            {0: ("short",), 1: ("o1",)},
            None,
            PER_PLAYER,
        ),
        (
            _worked_example_f_hand(),
            {0: ("SB",), 1: ("BB",)},
            BlindStructure(1, 2),
            SINGLE_PAYER,
        ),
        (
            _worked_example_g_hand(),
            {0: ("SB",)},
            None,
            SINGLE_PAYER,
        ),
    ]
    for (players, actions), winners, structure, mode in hands:
        ledger = build_hand_ledger(
            players, actions, winners=winners, blinds=structure, ante_mode=mode
        )
        assert ledger.warnings == ()
        assert ledger.legality_issues == ()
        assert ledger.is_legal is True
        assert ledger.is_balanced is True


def _worked_example_f_hand():
    """Blinds 1/2, a 2-chip big-blind ante, the small blind all-in for its blind.

    A third seat is REQUIRED for the operator's stated figures: main 5 + side 2 is
    7, while the ante plus the small blind plus the big blind is only 5. The
    button calling the 2 makes it 7 exactly, and the pre-ruling model returns main
    4 / side 3 with the small blind at +3, which is verbatim what the ruling
    quotes as "today". So this is the operator's hand.

    live: SB 1, BB 2, BTN 2.  ante: BB 2.
    """

    players = [_player("SB", 1), _player("BB", 4), _player("BTN", 2)]
    actions = [
        _live("SB", "post_blind", 1),
        _live("BB", "post_blind", 2),
        LedgerAction(
            player="BB",
            street="preflop",
            kind="ante",
            amount=2,
            is_live_post=False,
            forced_bet_type="big_blind_ante",
        ),
        _live("BTN", "call", 2),
    ]
    return players, actions


def _worked_example_g_hand():
    """The button antes 50,000 and folds; small and big blinds have 20,000 each."""

    players = [_player("BTN", 50001), _player("SB", 20000), _player("BB", 20000)]
    actions = [
        LedgerAction(
            player="BTN",
            street="preflop",
            kind="ante",
            amount=50000,
            is_live_post=False,
            forced_bet_type="big_blind_ante",
        ),
        _live("SB", "all-in", 20000),
        _live("BB", "all-in", 20000),
        _live("BTN", "fold", 0),
    ]
    return players, actions


def test_worked_example_f_a_consolidated_ante_is_never_capped_by_a_short_blind():
    """RULING 3, and the ONE acceptance example where the two modes disagree.

    Blinds 1/2, the big blind posts a 2-chip ante, the small blind is all-in for
    its 1-chip blind, the button calls.

    SINGLE_PAYER_TABLE_ANTE: the consolidated ante is TABLE MONEY. It goes whole
    into the main pot and is not capped against the shortest seat's 1-chip total.
        main 5 = 1 x 3 live + the whole 2-chip ante   [SB, BB, BTN]
        side 2 = (2 - 1) x 2 live                     [BB, BTN]
        SB wins main -> +4.

    PER_PLAYER: the same ante IS capped at the smallest total among the main
    pot's eligible seats, which is the small blind's 1, so 1 chip of it rises.
        main 4 / side 3, SB -> +3.

    Both readings are asserted, because the mode is load-bearing rather than
    cosmetic: nothing in the action line distinguishes them, which is exactly why
    the operator ruled that it must be declared and never inferred.
    """

    players, actions = _worked_example_f_hand()
    structure = BlindStructure(1, 2)

    table_ante = build_hand_ledger(
        players,
        actions,
        {0: ("SB",), 1: ("BB",)},
        blinds=structure,
        ante_mode=SINGLE_PAYER,
    )
    assert [pot.amount for pot in table_ante.pots] == pytest.approx([5, 2])
    assert table_ante.pots[0].eligible_players == ("SB", "BB", "BTN")
    assert table_ante.pots[1].eligible_players == ("BB", "BTN")
    assert table_ante.payouts["SB"] == pytest.approx(5)
    assert table_ante.net_results["SB"] == pytest.approx(4)
    assert table_ante.is_legal is True
    assert table_ante.is_balanced is True
    assert table_ante.warnings == ()

    per_player = build_hand_ledger(
        players,
        actions,
        {0: ("SB",), 1: ("BB",)},
        blinds=structure,
        ante_mode=PER_PLAYER,
    )
    assert [pot.amount for pot in per_player.pots] == pytest.approx([4, 3])
    assert per_player.net_results["SB"] == pytest.approx(3)
    assert per_player.is_legal is True
    assert per_player.is_balanced is True

    # Chip conservation is not a function of the declaration: the same 7 chips
    # went in and the same 7 come out, whichever reading lays them out.
    assert table_ante.gross_pot == pytest.approx(per_player.gross_pot) == 7


def test_a_folded_seats_uncoverable_post_is_now_the_pots_and_settles_clean():
    """RULING 4. The refusal that fired on 7.79% of tournament-shaped hands.

    Rule 2 lifts the capped excess "into the layer above, eligible to the seats
    whose own total reached above that cap". When NO seat still in the hand has a
    total above the cap there is no such layer and the sentence stops. That can
    only happen to a FOLDED seat's post or to external money -- an unfolded
    seat's capped dead money is at most its own commitment, and the cascade's
    last cap is the largest surviving commitment -- so it never strands a seat's
    own chips.

    The previous round derived the conventional answer and then DECLINED TO CALL
    THE HAND STUDY-READY, in a warning whose own text admitted that no correction
    to the recording could clear it. The operator has ruled: such a post is
    abandoned to whoever wins. So this hand is now legal, balanced, settled and
    WARNING-FREE, and the chips are exactly the chips it always derived -- which
    is worth stating plainly, because it means ruling 4 is a study-readiness
    change and not a layering change, and a fix aimed at ``_build_pots`` would
    have been aimed at the wrong function.

    ``big`` posts a 50-chip ante and folds; two 3-chip stacks are all-in for their
    own antes. All 56 chips are in one pot the two survivors contest.
    """

    players = [_player("big", 500), _player("s1", 3), _player("s2", 3)]
    actions = [
        _dead("big", 50),
        _dead("s1", 3),
        _dead("s2", 3),
        _live("big", "fold", 0),
    ]

    ledger = build_hand_ledger(players, actions, {0: ("s1",)}, ante_mode=PER_PLAYER)
    assert [pot.amount for pot in ledger.pots] == pytest.approx([56])
    assert ledger.pots[0].eligible_players == ("s1", "s2")
    assert ledger.is_balanced is True
    assert ledger.is_legal is True
    assert ledger.is_settled is True
    assert ledger.warnings == ()


def test_worked_example_g_the_button_ante_that_folded_settles_under_both_modes():
    """RULING 4 IS MODE-INDEPENDENT, and the two modes reach it by different routes.

    Worked example (g): the button antes 50,000 and folds; the small and big
    blinds have 20,000 each. One pot of 90,000, settleable, under both readings.

    Under SINGLE_PAYER_TABLE_ANTE the consolidated ante is table money and goes
    whole into the main pot; the abandoned-excess branch never fires. Under
    PER_PLAYER the ante is capped at 20,000, the 30,000 remainder has no layer
    above it to rise into -- no unfolded total exceeds 20,000 -- and ruling 4
    abandons it into the top dead layer. Different routes, same 90,000, which is
    what "ruling 4 does not mention modes" has to mean if it means anything.
    """

    players = [_player("BTN", 50001), _player("SB", 20000), _player("BB", 20000)]
    actions = [
        LedgerAction(
            player="BTN",
            street="preflop",
            kind="ante",
            amount=50000,
            is_live_post=False,
            forced_bet_type="big_blind_ante",
        ),
        _live("SB", "all-in", 20000),
        _live("BB", "all-in", 20000),
        _live("BTN", "fold", 0),
    ]

    for mode in (SINGLE_PAYER, PER_PLAYER):
        ledger = build_hand_ledger(players, actions, {0: ("SB",)}, ante_mode=mode)
        assert [pot.amount for pot in ledger.pots] == pytest.approx([90000])
        assert ledger.pots[0].eligible_players == ("SB", "BB")
        assert ledger.payouts["SB"] == pytest.approx(90000)
        assert ledger.net_results["SB"] == pytest.approx(70000)
        assert ledger.is_settled is True
        assert ledger.is_balanced is True
        assert ledger.is_legal is True
        assert ledger.warnings == ()


def test_external_dead_money_is_capped_like_a_recorded_post_under_every_mode():
    """RULING 5, and the direction it moves the answer.

    Operator-typed dead money used to join the main pot WHOLE and unwarned, so a
    seat that had committed 2 chips could be paid 312. It is now capped exactly as
    a recorded forced post is: a seat collects it only up to its own total
    commitment, and the rest rises to the seats that committed more.

    ``short`` commits 2 chips, two opponents commit 20 each, and 30 chips are
    declared. The old model gave one pot of 72 that ``short`` could take entirely.
    The capped model gives 6 (2 x 3 live) + 2 of the declared money in the main
    pot, with the other 28 in a layer only the two deep seats reached.
    """

    players = [_player("short", 2), _player("d1", 20), _player("d2", 20)]
    actions = [
        _live("short", "all-in", 2),
        _live("d1", "all-in", 20),
        _live("d2", "all-in", 20),
    ]

    for mode in (PER_PLAYER, SINGLE_PAYER, None):
        ledger = build_hand_ledger(
            players, actions, {0: ("short",), 1: ("d1",)},
            dead_money=30,
            ante_mode=mode,
        )
        assert [pot.amount for pot in ledger.pots] == pytest.approx([8, 64])
        assert ledger.pots[0].eligible_players == ("short", "d1", "d2")
        assert ledger.pots[1].eligible_players == ("d1", "d2")
        assert ledger.payouts["short"] == pytest.approx(8)
        assert ledger.net_results["short"] == pytest.approx(6)
        assert ledger.is_balanced is True
        assert ledger.is_legal is True


def test_external_dead_money_under_a_table_ante_discloses_the_reading_it_took():
    """THE ONE THING RULING 5 DOES NOT DECIDE, and it is disclosed rather than assumed.

    Ruling 5 caps external dead money "under whichever rule the hand's ante mode
    selects". Under SINGLE_PAYER_TABLE_ANTE the mode selects TWO rules -- the
    consolidated ante is uncapped, everything else is capped -- and no worked
    example contains external dead money, so nothing in the acceptance set
    constrains the choice. The capped reading is taken (it is the strict
    direction, and external money has no seat so it is not "one seat posts a
    consolidated ante"), and the hand says so instead of publishing the figure as
    settled fact.

    Disclosed only where it MOVES A CHIP. Below the smallest total commitment
    contesting the main pot the two readings place the same chips in the same
    layer, so there is nothing to decide and nothing to say -- an operator trained
    to click past disclosures that mean nothing will click past the one that
    means something.
    """

    players = [_player("BB", 12), _player("short", 2), _player("d", 20)]
    actions = [
        LedgerAction(
            player="BB",
            street="preflop",
            kind="ante",
            amount=10,
            is_live_post=False,
            forced_bet_type="big_blind_ante",
        ),
        _live("BB", "all-in", 2),
        _live("short", "all-in", 2),
        _live("d", "all-in", 20),
    ]

    def disclosures(**kwargs) -> list[str]:
        # The standing "no declared winner" note is about settlement, not about
        # the reading, so it is filtered out rather than asserted around.
        return [
            note
            for note in build_hand_ledger(players, actions, **kwargs).warnings
            if "external dead money" in note
        ]

    moves = disclosures(dead_money=30, ante_mode=SINGLE_PAYER)
    assert len(moves) == 1
    assert "SINGLE_PAYER_TABLE_ANTE" in moves[0]
    assert "CAPPED reading" in moves[0]

    # The same hand under PER_PLAYER selects one rule, so there is nothing to
    # disclose -- and neither does a declared amount small enough that the two
    # readings coincide.
    assert disclosures(dead_money=30, ante_mode=PER_PLAYER) == []
    assert disclosures(dead_money=1, ante_mode=SINGLE_PAYER) == []

    # THE THRESHOLD IS THE FLOOR, AND IT IS PINNED AGAINST BEING RAISED.
    # Above, every seat but the short one is refunded down to 2, so the hand's
    # whole commitment is small and a threshold read as "the sum of every
    # commitment" instead of "the smallest" would still fire on it -- a mutation
    # that silences this disclosure on real hands survived that reading. This is
    # the same disclosure over a REFUND-FREE table where the two thresholds are
    # far apart: commitments 16 / 2 / 10 / 10, floor 2, sum 38, declared 30. The
    # reading moves chips, so the hand must say so.
    wide = [_player("BB", 16), _player("short", 2), _player("c", 10), _player("d", 10)]
    wide_actions = [
        LedgerAction(
            player="BB",
            street="preflop",
            kind="ante",
            amount=6,
            is_live_post=False,
            forced_bet_type="big_blind_ante",
        ),
        _live("BB", "all-in", 10),
        _live("short", "all-in", 2),
        _live("c", "call", 10),
        _live("d", "call", 10),
    ]
    wide_ledger = build_hand_ledger(
        wide, wide_actions, dead_money=30, ante_mode=SINGLE_PAYER
    )
    assert all(value == 0 for value in wide_ledger.refunds.values())
    assert len([n for n in wide_ledger.warnings if "external dead money" in n]) == 1


def test_the_amendment_does_not_disclose_a_hand_it_answers():
    """The retired refusal must not survive as an over-broad one.

    The round-19 warning fired on any main-pot seat whose total commitment was
    below another seat's forced post. That is precisely the amended family, so
    leaving it in place would refuse every hand the operator just ruled on and the
    ruling would have bought nothing. Two hands hold it to that: the amendment's
    own worked example, and a table where two live seats cover a short one's ante
    difference. Neither may produce a note.
    """

    short_seat = build_hand_ledger(
        [_player("short", 60), _player("o1", 500), _player("o2", 500)],
        [_dead("short", 60), _dead("o1", 100), _dead("o2", 100)],
        {0: ("short",), 1: ("o1",)},
        ante_mode=PER_PLAYER,
    )
    assert short_seat.warnings == ()

    covered = build_hand_ledger(
        [_player("A", 500), _player("B", 500), _player("C", 26)],
        [
            _dead("C", 10),
            _live("A", "bet", 20),
            _live("B", "call", 20),
            _live("C", "all-in", 16),
        ],
        {0: ("C",), 1: ("A",)},
        ante_mode=PER_PLAYER,
    )
    assert covered.warnings == ()


# --- rule 2's cap operand, measured across an uncalled-bet refund -------------


def test_a_partly_refunded_shove_is_capped_at_what_it_kept_not_at_what_it_bet():
    """Rule 2's TOTAL COMMITMENT is what stuck, not what was pushed forward.

    ``alice`` shoves 100 into two seats that between them can only call 10, so 90
    comes straight back. Its commitment is the 10 that stayed. ``bob`` and
    ``carol`` each posted a 50-chip ante on top of their call, so their totals
    are 60 and alice holds the smallest commitment among the seats contesting the
    main pot. Amended rule 2 therefore caps each ante at 10 in the main pot and
    lifts the other 40 apiece into a layer only bob and carol reached:

        main  50 = 10 x 3 live + 10 + 10 of the two antes   [alice, bob, carol]
        side  80 = the 40 x 2 excess                        [bob, carol]

    alice wins the main pot and is paid 50 -- its own 10 back, the 10 each
    opponent matched of it, and 10 of each opponent's ante, which is every chip
    the table matched of a 10-chip commitment.

    WHY THIS HAND IS HERE. ``_build_pots`` reads the cap operand after the refund
    and falls back on what the seat PUT UP only when the post-refund total is
    zero, and that fallback is a narrow boundary written as a default. Widening
    it to ``max(total, put_up)`` -- so a refunded seat is capped as if it still
    had the whole shove at risk -- makes this hand one pot of 130 and pays alice
    all of it, 80 chips past what the table matched, settled, balanced, legal and
    WARNING-FREE. That mutant survived all 164 accounting tests, because neither
    layering generator could build a hand whose refund is PARTIAL and whose
    opponents carry dead money above what the shover kept. It is killed now by
    the payout-cap properties alone, over ``refunded_shove_hand``; this test is
    the hand written down so the family is legible without running a mutant.
    """

    ledger = build_hand_ledger(
        [_player("alice", 100), _player("bob", 60), _player("carol", 60)],
        [
            _dead("bob", 50),
            _dead("carol", 50),
            _live("alice", "bet", 100),
            _live("bob", "all-in", 10),
            _live("carol", "all-in", 10),
        ],
        {0: ("alice",), 1: ("bob",)},
        ante_mode=PER_PLAYER,
    )

    assert [pot.amount for pot in ledger.pots] == pytest.approx([50, 80])
    assert ledger.pots[0].eligible_players == ("alice", "bob", "carol")
    assert ledger.pots[1].eligible_players == ("bob", "carol")
    assert ledger.refunds["alice"] == pytest.approx(90)
    assert ledger.payouts["alice"] == pytest.approx(50)
    assert ledger.net_results["alice"] == pytest.approx(40)
    assert ledger.is_balanced is True
    assert ledger.is_legal is True


# --- what names a layer ------------------------------------------------------


def test_the_main_pot_is_the_layer_everyone_can_win_and_not_the_first_one():
    """``cause`` is a fact about the layer, which is why it is read off the layer.

    ``PotLayer.label`` says in so many words that it is derived "from what
    created it and not from its index", and ``_build_pots`` computed it as
    ``"main" if index == 0 else "side"`` -- a fact about the LIST. The two agree
    only while the ladder is emitted widest-eligible-set first, and nothing
    checked that, so any reordering relabelled the pots and every consumer,
    including the settlement editor's pot numbers, believed the new labels.

    The hand below is the one where the distinction is visible: ``A`` posts a
    100-chip ante and wagers nothing, ``B`` bets 100 and ``C`` is all-in for 40.
    The ladder does NOT nest -- layer 1 is contested by (B, C) and layer 2 by (A)
    alone, which are disjoint -- so "everything after the first is narrower than
    the one before it" is false here, and only containment in the MAIN pot's
    eligible set survives as the thing the word means.
    """

    ledger = build_hand_ledger(
        [_player("A", 100), _player("B", 500), _player("C", 40)],
        [_dead("A", 100), _live("B", "bet", 100), _live("C", "all-in", 40)],
        {0: ("A",), 1: ("B",), 2: ("A",)},
        ante_mode=PER_PLAYER,
    )

    assert [pot.amount for pot in ledger.pots] == pytest.approx([40, 80, 60])
    assert ledger.pots[0].eligible_players == ("A", "B", "C")
    assert ledger.pots[1].eligible_players == ("B", "C")
    assert ledger.pots[2].eligible_players == ("A",)
    # The ladder does not nest: layers 1 and 2 share nobody.
    assert not set(ledger.pots[1].eligible_players) & set(ledger.pots[2].eligible_players)
    # Exactly one layer is the main pot, and it is the one every other layer's
    # seats can also win -- not merely the one printed first.
    reach = {name for pot in ledger.pots for name in pot.eligible_players}
    main = [pot for pot in ledger.pots if pot.cause == "main"]
    assert len(main) == 1
    assert reach == set(main[0].eligible_players)
    assert main[0].index == 0
    assert [pot.label for pot in ledger.pots] == ["Main pot", "Side pot", "Side pot"]
