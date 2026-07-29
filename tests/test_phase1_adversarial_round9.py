"""Regressions for the round-9 adversarial findings against Phase 1.

Every test here failed before its fix. Round 9's themes are *a round-8 fix that
closed only the exact shape it was demonstrated on*, *a policy allocation that can
exceed the thing it is allocated from*, and *a blocker whose named clearing action
has no writer behind it*.

The chip-unit tests below are deliberately built on hands where the CONTRIBUTORS
OUTNUMBER THE WINNERS. Round 8 bounded the declared ``rake_rounding_unit`` by the
greatest common divisor of the observed contributions and pinned that with two
equal contributions, where every divisor halves the pot evenly and the bound
therefore looks like a fix. As soon as a third seat contributes, the divisors
disagree and the declared field is a dial again.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from poker_tracker.math.accounting import (
    LedgerAction,
    LedgerPlayer,
    RakePolicy,
    build_hand_ledger,
)
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import export_session, import_session
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SettlementEntry,
    utc_now,
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness

# Every value a settlement editor or an import payload can put in "Chip unit",
# including ones that divide the observed contributions, ones that do not, and
# ones above the pot. None of them may change a derived payout at a zero rate.
CHIP_UNITS = [0.001, 0.01, 0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 20.0, 100.0, 1e6]


def _clean_evidence(**overrides: object) -> dict[str, object]:
    payload = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.92,
            layout_supported=True,
            table_size=6,
        )
    )
    payload.update(overrides)
    return payload


def _open_db(tmp_path: Path, name: str = "round9.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed_three_way_chop(
    db: PokerDatabase,
    *,
    bet: float = 8.0,
    hero_bb_won: float | None,
) -> Hand:
    """Three seats commit ``bet`` each; hero and villain are declared co-winners.

    The honest chop of the 3x pot is half each, so the hero's true net is
    ``bet / 2``. Anything else recorded in ``hands.hero_bb_won`` is a fabrication
    the ledger has to contradict at every declared chip unit.
    """
    session = db.create_session(Session(name="Round 9", date_played=date(2026, 1, 1)))
    assert session.id is not None
    seats = (
        ("hero", "Hero", True),
        ("villain", "Villain", False),
        ("third", "Third", False),
    )
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=bet * len(seats),
            hero_bb_won=hero_bb_won,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None
    for key, name, is_hero in seats:
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                is_hero=is_hero,
                starting_stack=1000,
            )
        )
    for index, (key, name, _) in enumerate(seats, start=1):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=name,
                action_type="bet" if index == 1 else "call",
                amount=bet,
            )
        )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=key,
                player_name=key.capitalize(),
                amount=None,
                entry_order=order,
            )
            for key, order in (("hero", 1), ("villain", 2))
        ],
    )
    return hand


# ---------------------------------------------------------------------------
# Findings 1, 3 and 6 -- 'Chip unit' still redistributed a chopped pot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chip_unit", CHIP_UNITS)
def test_no_chip_unit_moves_the_split_when_contributors_outnumber_winners(
    chip_unit: float,
) -> None:
    """Three seats contribute 8, two chop the 24: the honest split is 12/12.

    Round 8's rule -- honour the declared unit when it divides the greatest common
    divisor of the observed contributions, otherwise fall back to that divisor --
    admits 1, 2, 4 and 8 as "coherent" here, and they do not agree: 1, 2 and 4 pay
    12/12 while 8 pays 16/8. Worse, the fallback for a NON-dividing unit was the
    gcd itself, the most distorting choice of all, so 3, 5 and 100 also paid 16/8.
    Raising one unbounded display field therefore still doubled the derived hero
    result.
    """
    keys = ("hero", "villain", "third")
    ledger = build_hand_ledger(
        [LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key) for key in keys],
        [
            LedgerAction(
                player=key,
                street="river",
                kind="bet" if index == 0 else "call",
                amount=8,
            )
            for index, key in enumerate(keys)
        ],
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, rounding_unit=chip_unit),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.payouts == {"hero": 12.0, "villain": 12.0, "third": 0.0}
    assert ledger.net_results == {"hero": 4.0, "villain": 4.0, "third": -8.0}
    assert ledger.is_balanced is True


@pytest.mark.parametrize("chip_unit", CHIP_UNITS)
def test_no_chip_unit_moves_a_split_sized_by_per_player_totals(
    chip_unit: float,
) -> None:
    """The old bound was read off per-player TOTALS, not off the action line.

    Six 5-chip bets across three seats total 10 a seat, so a declared unit of 10
    passed the divisor rule even though no amount on the hand is a 10 -- and paid
    20/10 instead of the honest 15/15. A denomination the hand never showed is not
    evidence of anything.
    """
    keys = ("hero", "villain", "third")
    actions = []
    for street in ("flop", "turn"):
        for index, key in enumerate(keys):
            actions.append(
                LedgerAction(
                    player=key,
                    street=street,
                    kind="bet" if index == 0 else "call",
                    amount=5,
                )
            )
    ledger = build_hand_ledger(
        [LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key) for key in keys],
        actions,
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, rounding_unit=chip_unit),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.payouts == {"hero": 15.0, "villain": 15.0, "third": 0.0}
    assert ledger.is_balanced is True


@pytest.mark.parametrize("chip_unit", CHIP_UNITS)
def test_the_settlement_editor_chip_unit_cannot_land_a_fabricated_hero_result(
    tmp_path: Path, chip_unit: float
) -> None:
    """End to end, through the writer the settlement editor actually calls.

    The honest hero net on this hand is +4. A recorded +8 reconciled exactly at
    every unit that did not divide 8, with an empty blocker tuple, no evidence
    warning code and no ``hand_corrections`` row -- so the operator was told the
    forged figure was authoritative and could promote it to ``reviewed``.
    """
    db = _open_db(tmp_path)
    hand = _seed_three_way_chop(db, hero_bb_won=8.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=chip_unit)
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.payouts == {"hero": 12.0, "villain": 12.0, "third": 0.0}
    assert result.ledger.net_results["hero"] == pytest.approx(4.0)
    assert "Observed Hero result does not match the derived ledger result." in result.issues
    assert result.is_authoritative is False

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")
    db.close()


def test_an_imported_chip_unit_cannot_be_certified_by_the_named_clearing_action(
    tmp_path: Path,
) -> None:
    """The payload sets the unit; the blocker then tells the operator to save.

    An import landed the forged hand un-authoritative only because the imported
    settlement arrives ``unsettled`` -- the hero cross-check was ALREADY silenced.
    ACCOUNTING_NOT_AUTHORITATIVE then said to "save the settlement until its
    status reads reconciled", and performing exactly that one action certified the
    forgery. A blocker must never name the button that launders the defect.
    """
    source = _open_db(tmp_path, "round9-source.db")
    hand = _seed_three_way_chop(source, hero_bb_won=8.0)
    assert hand.id is not None and hand.session_id is not None
    source.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=0.01)
    )
    persist_reconciliation(source, hand.id)
    payload = export_session(source, hand.session_id)
    source.close()

    entry = payload["hands"][0]
    entry["hand"]["hero_bb_won"] = 8.0
    entry["settlement"]["status"] = "reconciled"
    entry["settlement"]["rake_rate"] = 0.0
    entry["settlement"]["rake_cap"] = None
    entry["settlement"]["dead_money"] = 0.0
    entry["settlement"]["rake_rounding_unit"] = 3.0

    target = _open_db(tmp_path, "round9-target.db")
    import_session(target, payload)
    session_id = target.fetch_sessions()[0].id
    assert session_id is not None
    landed = target.fetch_hands_by_session(session_id)[0]
    assert landed.id is not None

    imported = reconcile_persisted_hand(target, landed.id)
    assert imported.ledger.payouts == {"hero": 12.0, "villain": 12.0, "third": 0.0}
    assert (
        "Observed Hero result does not match the derived ledger result." in imported.issues
    )

    # The clearing action the blocker names, performed verbatim.
    saved = persist_reconciliation(target, landed.id)
    assert saved.ledger.payouts == {"hero": 12.0, "villain": 12.0, "third": 0.0}
    assert saved.is_authoritative is False
    stored = target.fetch_hand(landed.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=saved, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")
    target.close()


@pytest.mark.parametrize("chip_unit", CHIP_UNITS)
def test_a_zero_rate_rake_policy_really_does_take_nothing(chip_unit: float) -> None:
    """``upsert_hand_settlement`` skips DECLARED_RAKE_CODE at a zero rate.

    The justification written above that gate -- "a cap or a rounding unit on a
    zero rate takes nothing" -- was false while the unit also sized the split, and
    that made the one audit channel silent exactly where the attack was cheapest.
    This pins the claim the gate rests on: at rate zero, no unit, cap or
    no-flop-no-drop setting moves a single derived chip.
    """
    keys = ("hero", "villain", "third")
    ledger = build_hand_ledger(
        [LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key) for key in keys],
        [
            LedgerAction(
                player=key,
                street="river",
                kind="bet" if index == 0 else "call",
                amount=7,
            )
            for index, key in enumerate(keys)
        ],
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, cap=chip_unit, rounding_unit=chip_unit, no_flop_no_drop=True),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.rake == 0.0
    # 21 chips two ways is a genuine odd chip and is kept -- as a constant.
    assert ledger.payouts == {"hero": 11.0, "villain": 10.0, "third": 0.0}


def test_declared_dead_money_sets_the_split_granularity() -> None:
    """Dead money is an observed amount for the purpose of sizing a chip.

    ``_split_granularity`` reads the declared dead money alongside the settled
    contributions, and deleting that term left the whole suite green while
    changing who got paid -- no test anywhere observed whether it participated.
    Two 2-chip contributions and 0.3 of dead money make a 4.3 pot: with the dead
    money the observed denomination is 0.1 and the chop is 2.2/2.1, without it the
    contributions alone read as whole chips and the chop is 2.3/2.0.
    """
    ledger = build_hand_ledger(
        [
            LedgerPlayer(name="Hero", starting_stack=1000, key="hero"),
            LedgerPlayer(name="Villain", starting_stack=1000, key="villain"),
        ],
        [
            LedgerAction(player="hero", street="river", kind="bet", amount=2),
            LedgerAction(player="villain", street="river", kind="call", amount=2),
        ],
        dead_money=0.3,
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, rounding_unit=0.01),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.gross_pot == pytest.approx(4.3)
    assert ledger.payouts == {"hero": 2.2, "villain": 2.1}
    assert ledger.is_balanced is True


def test_a_finer_observed_amount_pins_a_finer_split(tmp_path: Path) -> None:
    """The granularity is the finest signal the amounts carry, not the coarsest.

    A denomination cannot be established from above -- three seats each committing
    8 share a factor of 8 and demonstrate nothing about 8-chips -- so reading the
    greatest common divisor as the denomination was the maximally distorting
    choice and the one the declared unit was allowed to select from. Reading the
    finest decimal place instead keeps every chop as close to even as the hand
    allows: 2.25 a seat is written in hundredths, so the 6.75 pot is chopped
    3.38/3.37 and not 3.75/3.00, whatever "Chip unit" says.
    """
    keys = ("hero", "villain", "third")
    ledger = build_hand_ledger(
        [LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key) for key in keys],
        [
            LedgerAction(
                player=key,
                street="river",
                kind="bet" if index == 0 else "call",
                amount=2.25,
            )
            for index, key in enumerate(keys)
        ],
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, rounding_unit=1000.0),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.gross_pot == pytest.approx(6.75)
    assert ledger.payouts == {"hero": 3.38, "villain": 3.37, "third": 0.0}
    assert ledger.is_balanced is True


def test_the_split_does_not_depend_on_int_versus_float_amounts() -> None:
    """``Decimal('5.0')`` and ``Decimal('5')`` are the same denomination.

    The granularity is read off each amount's own scale, and amounts reach the
    ledger as ints from a hand-built test and as floats from the store. Without
    ``normalize()`` those two routes would disagree about the same hand.
    """
    keys = ("hero", "villain", "third")

    def _payouts(amount: float | int) -> dict[str, float]:
        return build_hand_ledger(
            [
                LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key)
                for key in keys
            ],
            [
                LedgerAction(
                    player=key,
                    street="river",
                    kind="bet" if index == 0 else "call",
                    amount=amount,
                )
                for index, key in enumerate(keys)
            ],
            winners={0: ("hero", "villain")},
            rake=RakePolicy(rate=0, rounding_unit=0.01),
            odd_chip_order=["hero", "villain"],
        ).payouts

    assert _payouts(7) == _payouts(7.0) == {"hero": 11.0, "villain": 10.0, "third": 0.0}


# ---------------------------------------------------------------------------
# Finding 4 -- a pot could be raked past its own size
# ---------------------------------------------------------------------------


def test_a_side_pot_is_never_raked_beyond_its_own_size() -> None:
    """An ordinary hand under an ordinary policy produced a negative payout.

    Every non-final pot's proportional share was rounded DOWN to the declared unit
    and the whole leftover charged to the LAST pot with no cap at that pot's
    amount. A 149.25 main pot and a 0.50 side pot at 5% capped at 5 with a
    whole-chip drop took 4 from the main pot and charged 1 to a pot of 0.50, so
    that layer showed ``net_amount = -0.50`` and paid its winner minus half a
    chip. ``is_balanced`` stayed True because the negative preserved
    ``paid + rake == gross``, so the hand certified as authoritative.
    """
    ledger = build_hand_ledger(
        [
            LedgerPlayer(name="Hero", starting_stack=1000, key="hero"),
            LedgerPlayer(name="Villain", starting_stack=1000, key="villain"),
            LedgerPlayer(name="Short", starting_stack=49.75, key="short"),
        ],
        [
            LedgerAction(player="short", street="preflop", kind="bet", amount=49.75),
            LedgerAction(player="hero", street="preflop", kind="call", amount=49.75),
            LedgerAction(player="villain", street="preflop", kind="call", amount=49.75),
            LedgerAction(player="hero", street="flop", kind="bet", amount=0.25),
            LedgerAction(player="villain", street="flop", kind="call", amount=0.25),
        ],
        winners={0: ("short",), 1: ("hero",)},
        rake=RakePolicy(rate=0.05, cap=5, rounding_unit=1),
        odd_chip_order=["hero"],
    )

    assert ledger.rake == pytest.approx(5.0)
    for pot in ledger.pots:
        assert pot.rake <= pot.amount + 1e-9
        assert pot.net_amount >= 0
    assert all(value >= 0 for value in ledger.payouts.values())
    assert ledger.payouts["hero"] == pytest.approx(0.5)
    assert ledger.is_balanced is True


@pytest.mark.parametrize("rounding_unit", [1.0, 5.0, 45.0])
def test_a_coarse_drop_cannot_charge_a_side_pot_more_than_it_holds(
    rounding_unit: float,
) -> None:
    """The coarser the declared drop, the larger the leftover that was dumped.

    At rate 0.5 with a 45-chip drop the 4-chip side pot was charged 41 and its
    winner paid -41. A negative derived payout is unreconcilable by construction:
    ``SettlementEntry.amount`` is ``ge=0``, so the operator cannot declare it and
    ACCOUNTING_NOT_AUTHORITATIVE names a save that can never clear.
    """
    ledger = build_hand_ledger(
        [
            LedgerPlayer(name="Hero", starting_stack=1000, key="hero"),
            LedgerPlayer(name="Villain", starting_stack=1000, key="villain"),
            LedgerPlayer(name="Short", starting_stack=48, key="short"),
        ],
        [
            LedgerAction(player="short", street="preflop", kind="bet", amount=48),
            LedgerAction(player="hero", street="preflop", kind="call", amount=48),
            LedgerAction(player="villain", street="preflop", kind="call", amount=48),
            LedgerAction(player="hero", street="flop", kind="bet", amount=2),
            LedgerAction(player="villain", street="flop", kind="call", amount=2),
        ],
        winners={0: ("short",), 1: ("hero",)},
        rake=RakePolicy(rate=0.5, rounding_unit=rounding_unit),
        odd_chip_order=["hero"],
    )

    for pot in ledger.pots:
        assert pot.rake <= pot.amount + 1e-9
        assert pot.net_amount >= 0
    assert all(value >= 0 for value in ledger.payouts.values())
    assert ledger.is_balanced is True


# ---------------------------------------------------------------------------
# Finding 5 -- STALE_COACHING_EVIDENCE named an action with no writer
# ---------------------------------------------------------------------------


def _seed_coached_hand(db: PokerDatabase) -> Hand:
    session = db.create_session(Session(name="Coached", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            hero_bb_won=0.0,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None
    db.create_coaching_response(
        CoachingResponse(
            provider_name="anthropic",
            model_name="test-model",
            raw_prompt="prompt",
            raw_response="response",
            review_type="hand",
            hand_id=hand.id,
            session_id=session.id,
            parsed_sections={"summary": "ok"},
        )
    )
    return hand


def test_a_stale_imported_coaching_review_can_be_discarded_without_a_provider(
    tmp_path: Path,
) -> None:
    """Every imported coaching row is staled, and re-running needs a provider.

    ``_mark_imported_analysis_stale`` forces ``is_stale`` on import, so an imported
    hand lands blocked by STALE_COACHING_EVIDENCE. Its only clearing action was
    "Re-run coaching in Study → Coach", whose only writer is
    ``create_coaching_response`` -- and the Coach button is disabled when no LLM
    provider is configured, which is exactly the state an operator importing a
    colleague's session is in. The solver twin has had ``delete_solver_run`` for
    this since round 8; coaching had no discard writer at all.
    """
    source = _open_db(tmp_path, "coach-source.db")
    hand = _seed_coached_hand(source)
    assert hand.session_id is not None
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "coach-target.db")
    import_session(target, payload)
    session_id = target.fetch_sessions()[0].id
    assert session_id is not None
    landed = target.fetch_hands_by_session(session_id)[0]
    assert landed.id is not None

    def _readiness():
        return evaluate_study_readiness(
            landed,
            accounting=None,
            coaching_reviews=target.fetch_coaching_reviews_by_hand(landed.id),
            hand_reviews=target.fetch_reviews_by_hand(landed.id),
            user_confirmed=True,
        )

    blocked = _readiness()
    assert blocked.has("STALE_COACHING_EVIDENCE")
    blocker = next(
        item for item in blocked.blockers if item.code == "STALE_COACHING_EVIDENCE"
    )
    assert "Discard stale coaching" in blocker.clearing_action

    assert target.discard_stale_coaching(landed.id) == 1
    assert not _readiness().has("STALE_COACHING_EVIDENCE")
    target.close()


def test_the_discard_covers_the_legacy_retained_review_table(tmp_path: Path) -> None:
    """``_coaching_blockers`` considers ``hand_reviews`` too, so the writer must.

    A legacy row staled by the same correction path blocks study exactly like a
    ``coaching_reviews`` row does, and it is not rendered in the Coach tab, so a
    discard that covered only the new table would leave the blocker standing with
    the operator looking at an empty list.
    """
    db = _open_db(tmp_path, "coach-legacy.db")
    hand = _seed_coached_hand(db)
    assert hand.id is not None
    db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="summary",
            theory_coach="theory",
            exploit_coach="exploit",
            study_lesson="lesson",
            is_stale=True,
            stale_reason="Hand was corrected.",
        )
    )

    def _readiness():
        return evaluate_study_readiness(
            hand,
            accounting=None,
            coaching_reviews=db.fetch_coaching_reviews_by_hand(hand.id),
            hand_reviews=db.fetch_reviews_by_hand(hand.id),
            user_confirmed=True,
        )

    assert _readiness().has("STALE_COACHING_EVIDENCE")
    assert db.discard_stale_coaching(hand.id) == 1
    assert db.fetch_reviews_by_hand(hand.id) == []
    assert not _readiness().has("STALE_COACHING_EVIDENCE")
    db.close()


def test_discarding_stale_coaching_keeps_a_current_review(tmp_path: Path) -> None:
    """The discard is aimed at stale evidence, never at the hand's live coaching."""
    db = _open_db(tmp_path, "coach-keep.db")
    hand = _seed_coached_hand(db)
    assert hand.id is not None and hand.session_id is not None
    db.create_coaching_response(
        CoachingResponse(
            provider_name="anthropic",
            model_name="test-model",
            raw_prompt="stale prompt",
            raw_response="stale response",
            review_type="hand",
            hand_id=hand.id,
            session_id=hand.session_id,
            is_stale=True,
            stale_reason="Hand was corrected.",
            created_at=utc_now() - timedelta(hours=1),
        )
    )

    assert db.discard_stale_coaching(hand.id) == 1
    remaining = db.fetch_coaching_reviews_by_hand(hand.id)
    assert [review.is_stale for review in remaining] == [False]
    db.close()
