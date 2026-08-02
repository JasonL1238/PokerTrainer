"""The pot-layering model, stated as the four hands that define it.

Adversarial round 19. ``poker_tracker.math.accounting`` produced a critical in
five consecutive rounds, four of them introduced by the repair to the previous
one, and every repair was argued from a hand rather than from a rule. The
operator has now fixed the rule, and these are its acceptance criteria:

  1. Pot layer boundaries are cut at distinct LIVE contribution levels, measured
     after uncalled-bet refunds have been returned. Live money is what a player
     CHOSE to wager. Forced posts are not live.
  2. ALL dead money -- antes, dead blinds, and externally declared dead money --
     goes entirely into the LOWEST layer. It is owed to the table, nobody can
     decline it, and it therefore never opens a boundary.
  3. A seat is eligible for a layer if its own LIVE contribution reaches that
     layer's level. Every unfolded seat that put ANY chip up -- live or dead --
     is eligible for the main pot.
  4. A folded seat's chips stay in the layers they reached and it is eligible for
     none.

The four worked examples below are the operator's, verbatim, and each must come
out to the chip. Two of them ((a) and (b)) were WRONG on the shipped reducer and
reported settled, balanced, legal and warning-free while being wrong; two of them
((c) and (d)) were right and are here so the repair to the first two cannot be
paid for out of them. The remaining tests pin the cases the specification leaves
to a decision, and say which decision was taken.
"""

from __future__ import annotations

import pytest

from poker_tracker.math.accounting import (
    BlindStructure,
    LedgerAction,
    LedgerError,
    LedgerPlayer,
    RakePolicy,
    build_hand_ledger,
)


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
    layers = build_hand_ledger(players, actions, blinds=structure).pots

    assert [pot.amount for pot in layers] == pytest.approx([58, 8])
    assert [pot.cause for pot in layers] == ["main", "side"]
    # 16 of live money from each of three seats, plus the whole 10-chip ante.
    assert layers[0].eligible_players == ("BB", "SB", "BTN")
    # The 4 live chips each of the two deep seats put in above the big blind.
    assert layers[1].eligible_players == ("SB", "BTN")

    ledger = build_hand_ledger(
        players, actions, {0: ("BB",), 1: ("SB",)}, blinds=structure
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
            players, actions, {0: ("BB",), 1: ("BB",)}, blinds=BlindStructure(10, 20)
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

    layers = build_hand_ledger(players, actions).pots
    assert [pot.amount for pot in layers] == pytest.approx([7, 14])
    assert layers[0].eligible_players == ("ao", "X", "Y")
    assert layers[1].eligible_players == ("X", "Y")

    ledger = build_hand_ledger(players, actions, {0: ("ao",), 1: ("X",)})
    assert ledger.payouts["ao"] == pytest.approx(7)
    assert ledger.net_results == pytest.approx({"ao": 0, "X": 7, "Y": -7})
    assert ledger.is_balanced is True

    with pytest.raises(LedgerError):
        build_hand_ledger(players, actions, {0: ("ao",), 1: ("ao",)})


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

    layers = build_hand_ledger(players, actions).pots
    assert [pot.amount for pot in layers] == pytest.approx([3, 20])
    assert layers[0].eligible_players == ("A", "B", "C")
    assert layers[1].eligible_players == ("A", "B")

    ledger = build_hand_ledger(players, actions, {0: ("C",), 1: ("A",)})
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

    layers = build_hand_ledger(players, actions).pots
    assert [pot.amount for pot in layers] == pytest.approx([88])
    assert layers[0].eligible_players == ("a", "b", "c", "d")

    ledger = build_hand_ledger(players, actions, {0: (winner,)})
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
    with pytest.raises(LedgerError):
        build_hand_ledger(players, actions)


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
    """

    players = [_player("alice", 2), _player("bob", 1)]
    actions = [_dead("alice", 2), _live("bob", "post_blind", 1)]

    ledger = build_hand_ledger(players, actions, {0: ("bob",)})
    assert ledger.refunds["bob"] == pytest.approx(1)
    assert [pot.amount for pot in ledger.pots] == pytest.approx([2])
    assert ledger.pots[0].eligible_players == ("alice", "bob")
    assert ledger.net_results == pytest.approx({"alice": -2, "bob": 2})
    assert ledger.is_balanced is True


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
        *_round3_hand(_dead("C", 1)), winners={0: ("C",), 1: ("A",)}
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
        players, hand(_dead("b", 3, kind="post_blind")), winners={0: ("a",)}
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


# --- the case rule 2 does not decide -----------------------------------------


def test_a_forced_post_no_main_pot_seat_could_cover_is_not_study_ready():
    """Antes 100 each with a stack short of its own ante: 540, not 240, and loudly.

    Rule 2 is unconditional -- all dead money goes whole into the lowest layer --
    and worked examples (a) and (d) both REQUIRE it: in (a) the big blind's
    unmatched 10 ante sits in a main pot the two deep seats may win, and in (d) c
    may win a's 5 ante and b's 3 dead blind having posted neither. So the chips
    below are what the model says and this test does not change them.

    But in all four worked examples every forced post is within reach of every
    seat that may win it, so none of them decides this hand: a 40-chip stack
    short of its own 100 ante collects five opponents' full 100 antes, 300 chips
    more than any of them covered of it. Whether an unmatched forced post is
    capped at what the winner covered is a question about the MODEL. Until the
    operator answers it, the hand is refused as study-ready rather than published:
    the warning reaches ``_cross_check``, which folds ledger warnings into its
    issues, so ``is_authoritative`` is False.
    """

    players = [
        _player("utg", 5000),
        _player("mp", 5000),
        _player("co", 5000),
        _player("btn", 40),
        _player("sb", 5000),
        _player("bb", 5000),
    ]
    actions = [
        _dead("utg", 100),
        _dead("mp", 100),
        _dead("co", 100),
        _dead("btn", 40),
        _dead("sb", 100),
        _dead("bb", 100),
        _live("sb", "post_blind", 100),
        _live("bb", "post_blind", 200),
        _live("utg", "fold", 0),
        _live("mp", "fold", 0),
        _live("co", "fold", 0),
        _live("sb", "fold", 0),
        _live("bb", "check", 0),
    ]

    ledger = build_hand_ledger(
        players, actions, winners={0: ("btn",), 1: ("bb",)}, blinds=BlindStructure(100, 200)
    )

    assert [pot.amount for pot in ledger.pots] == pytest.approx([540, 200])
    assert ledger.is_legal is True
    assert ledger.is_balanced is True
    assert ledger.is_settled is True
    # The chips are not in doubt; their attribution is.
    note = "\n".join(ledger.warnings)
    assert "forced post" in note
    assert "'btn'" in note and "40" in note and "100" in note


def test_the_four_worked_examples_are_not_touched_by_that_refusal():
    """The refusal must not fire where the operator has already ruled.

    (a) and (b) the unmatched post is the SHORT seat's own; (c) each opponent's
    ante equals the short seat's whole commitment; (d) nobody is short. A check
    that fired on any of them would be overriding the specification instead of
    naming the hand it does not cover.
    """

    hands = [
        (_big_blind_ante_hand(), {0: ("BB",), 1: ("SB",)}, BlindStructure(10, 20)),
        (
            (
                [_player("ao", 7), _player("X", 100), _player("Y", 100)],
                [_dead("ao", 7), _live("X", "bet", 7), _live("Y", "call", 7)],
            ),
            {0: ("ao",), 1: ("X",)},
            None,
        ),
        (_round3_hand(_dead("C", 1)), {0: ("C",), 1: ("A",)}, None),
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
        ),
    ]
    for (players, actions), winners, structure in hands:
        ledger = build_hand_ledger(players, actions, winners=winners, blinds=structure)
        assert ledger.warnings == ()
        assert ledger.legality_issues == ()
        assert ledger.is_legal is True
        assert ledger.is_balanced is True
