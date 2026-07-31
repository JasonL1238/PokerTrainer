"""Regressions for the round-8 adversarial findings against Phase 1.

Every test here failed before its fix. The round-8 themes are *a settlement field
whose second, undocumented job is to redistribute chips*, *an attestation whose
quantity can be moved by a field outside the policy it attested to*, *an audit
snapshot that omits the very column deciding the outcome*, *two public writers of
one table that disclose differently*, *a blocker naming as impossible the actions
that actually clear it*, and *a confirmation gate whose label promises evidence
the product never rendered*.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from poker_tracker.maintenance.data_health import audit_data_health
from poker_tracker.math.accounting import (
    LedgerAction,
    LedgerPlayer,
    RakePolicy,
    build_hand_ledger,
)
from poker_tracker.persistence.backup import backup_database
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    BoundaryEvidence,
    CompletionEvidence,
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import (
    DECLARED_RAKE_CODE,
    SCHEMA_VERSION,
    SOURCE_CORRECTION_CODE,
    PokerDatabase,
)
from poker_tracker.persistence.import_export import export_session, import_session
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness
from poker_tracker.ui.view_models import completion_evidence_rows
from tests.conftest import attest_declared_assumptions


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


def _open_db(tmp_path: Path, name: str = "round8.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed_chopped_hand(
    db: PokerDatabase,
    *,
    bet: float = 10.0,
    hero_bb_won: float | None = 0.0,
    winners: tuple[str, ...] = ("hero", "villain"),
    amounts: tuple[float | None, ...] = (None, None),
    orders: tuple[int, ...] = (1, 2),
    extra_callers: tuple[str, ...] = (),
    table_size: int | None = 6,
) -> Hand:
    """A chopped pot: bet/call at ``bet``, and both seats declared winners of pot 0.

    An even chop is the shape the chip unit used to redirect: the honest derived
    payouts are half each and the hero nets zero, so any nonzero
    ``hands.hero_bb_won`` is a fabrication the ledger must contradict.
    """
    session = db.create_session(Session(name="Round 8", date_played=date(2026, 1, 1)))
    assert session.id is not None
    seats = (("hero", "Hero", True), ("villain", "Villain", False)) + tuple(
        (key, key.capitalize(), False) for key in extra_callers
    )
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=table_size,
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
                amount=amount,
                entry_order=order,
            )
            for key, amount, order in zip(winners, amounts, orders, strict=True)
        ],
    )
    return hand


def _acknowledge_all(db: PokerDatabase, hand_id: int) -> None:
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    updated = acknowledge_codes(evidence, list(evidence.unresolved_codes))
    db.update_hand_completion(
        hand_id,
        completion_evidence=dump_completion_evidence(updated),
        notes="Acknowledged in test.",
    )


# ---------------------------------------------------------------------------
# Findings 1, 3 and 4 -- 'Chip unit' was also the pot-splitting granularity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chip_unit", [0.01, 1.0, 3.0, 7.0, 15.0, 20.0, 1000.0, 100000.0])
def test_the_chip_unit_cannot_redirect_a_chopped_pot(
    tmp_path: Path, chip_unit: float
) -> None:
    """A chip denomination must not decide which seat was pushed the pot.

    ``_split_pot`` rounded each winner's share DOWN to ``rake_rounding_unit`` and
    handed every leftover chip to the first name in ``odd_chip_order``, so raising
    one unbounded field in the settlement editor -- with the rake rate at zero, so
    neither declared-chips disclosure fired -- paid an even chop entirely to the
    hero. It was a continuous dial: on this 20-chip chop, 3 paid the hero 11, 7
    paid 13 and 20 paid the lot, and each value made its own fabricated
    ``hands.hero_bb_won`` reconcile exactly, authoritative and study-ready with an
    empty blocker tuple.
    """
    db = _open_db(tmp_path)
    # Forged: the honest chop nets the hero zero, whatever denomination is claimed.
    hand = _seed_chopped_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=chip_unit)
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.payouts == {"hero": 10.0, "villain": 10.0}
    assert result.ledger.net_results == {"hero": 0.0, "villain": 0.0}
    assert "Observed Hero result does not match the derived ledger result." in result.issues
    assert result.is_authoritative is False

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")
    db.close()


@pytest.mark.parametrize(
    "chip_unit",
    [0.001, 0.05, 0.25, 2.0, 5.0, 9.0, 11.0, 25.0, 49.0, 50.0, 51.0, 99.0, 250.0],
)
def test_no_chip_unit_moves_the_derived_split_of_an_even_chop(chip_unit: float) -> None:
    """The dial is gone at the ledger, which is what both attack paths ran through.

    Every contribution in a hand is a whole multiple of the smallest chip in play,
    so a denomination the action line contradicts describes chips that were never
    on the table. On two equal contributions every coherent denomination divides
    the pot into two equal halves, so the derived split is pinned no matter what
    the settlement or an import payload declares.
    """
    ledger = build_hand_ledger(
        [
            LedgerPlayer(name="Hero", starting_stack=1000, key="hero"),
            LedgerPlayer(name="Villain", starting_stack=1000, key="villain"),
        ],
        [
            LedgerAction(player="hero", street="river", kind="bet", amount=50),
            LedgerAction(player="villain", street="river", kind="call", amount=50),
        ],
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, rounding_unit=chip_unit),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.payouts == {"hero": 50.0, "villain": 50.0}
    assert ledger.is_balanced is True


@pytest.mark.parametrize("chip_unit", [0.001, 0.01, 1.0, 3.0, 7.0, 21.0, 1000.0])
def test_a_genuine_odd_chip_survives_and_ignores_the_declared_denomination(
    chip_unit: float,
) -> None:
    """The odd chip is real and must survive the fix -- as a constant, not a dial.

    Chips are indivisible: a 21-chip pot chopped two ways at whole-chip
    denomination really is pushed 11/10, and deriving 10.5/10.5 there would raise
    a false blocker against an honest declared award. Round 8 kept the odd chip by
    letting the DECLARED unit size it whenever it divided the observed gcd, and
    round 9 showed that is still a dial: this very hand paid 10.5/10.5 at 0.01,
    11/10 at 1 and 14/7 at 7, all three "coherent" by that rule. The chip is now
    sized by the finest denomination the hand's own numbers are written in, so the
    genuine 11/10 stands at every declared unit and none of them can move it.
    """
    players = [
        LedgerPlayer(name="Hero", starting_stack=1000, key="hero"),
        LedgerPlayer(name="Villain", starting_stack=1000, key="villain"),
        LedgerPlayer(name="Third", starting_stack=1000, key="third"),
    ]
    ledger = build_hand_ledger(
        players,
        [
            LedgerAction(player="hero", street="river", kind="bet", amount=7),
            LedgerAction(player="villain", street="river", kind="call", amount=7),
            LedgerAction(player="third", street="river", kind="call", amount=7),
        ],
        winners={0: ("hero", "villain")},
        rake=RakePolicy(rate=0, rounding_unit=chip_unit),
        odd_chip_order=["hero", "villain"],
    )

    assert ledger.payouts == {"hero": 11.0, "villain": 10.0, "third": 0.0}
    assert ledger.is_balanced is True


def test_an_imported_chip_unit_cannot_land_a_reconciled_chopped_pot(
    tmp_path: Path,
) -> None:
    """``import_session`` never calls ``persist_reconciliation``, so the payload's
    own ``rake_rounding_unit`` was what every readiness surface split the pot with.
    One hand-written v5 payload landed a fabricated hero result as reconciled,
    authoritative and study-ready with an empty blocker tuple.
    """
    source = _open_db(tmp_path, "export-source.db")
    hand = _seed_chopped_hand(source, hero_bb_won=10.0)
    assert hand.id is not None and hand.session_id is not None
    source.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=0.01)
    )
    persist_reconciliation(source, hand.id)
    payload = export_session(source, hand.session_id)
    source.close()

    entry = payload["hands"][0]
    entry["hand"]["hero_bb_won"] = 10.0
    entry["settlement"]["status"] = "reconciled"
    entry["settlement"]["rake_rounding_unit"] = 20.0
    entry["settlement"]["rake_rate"] = 0.0
    entry["settlement"]["dead_money"] = 0.0

    target = _open_db(tmp_path, "import-target.db")
    import_session(target, payload)
    landed = target.fetch_hands_by_session(target.fetch_sessions()[0].id)[0]
    assert landed.id is not None
    result = reconcile_persisted_hand(target, landed.id)

    assert result.ledger.payouts == {"hero": 10.0, "villain": 10.0}
    assert result.is_authoritative is False
    readiness = evaluate_study_readiness(landed, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")
    target.close()


# ---------------------------------------------------------------------------
# Findings 2 and 6 -- the rake attestation did not cover the rounding unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chip_unit", "hero_bb_won", "expected_rake", "expects_dependence"),
    [
        # 80 chips genuinely leave the pot, and the recorded -40 is only true
        # because of that. The declaration is load-bearing.
        (0.01, -40.0, 80.0, True),
        # The SAME 100% rate, rounded down to a unit larger than the whole rake,
        # takes nothing at all. The recorded +40 is what the action line derives
        # on its own, so nothing rests on the declaration.
        (81.0, 40.0, 0.0, False),
    ],
    ids=["unit-takes-the-whole-pot", "unit-takes-nothing"],
)
def test_the_chip_unit_decides_whether_a_declared_rake_is_a_dependence(
    tmp_path: Path,
    chip_unit: float,
    hero_bb_won: float,
    expected_rake: float,
    expects_dependence: bool,
) -> None:
    """An attestation covers the quantity attested to, and the unit sets it.

    ``_compute_rake`` rounds the raw rake DOWN to ``rake_rounding_unit``, so on an
    80-chip pot at a declared 100% rate, moving the unit alone from 0.01 to 81
    moves the rake from 80 to 0 and the derived hero result by the whole pot.
    ``_rake_policy`` omitted the field, so that write compared equal and an
    acknowledgement earned against an 80-chip rake carried over to a rake of
    nothing.

    AMENDED, and the amendment is the point. This test used to assert that at
    unit 81 the hand stays BLOCKED on ``UNRESOLVED_SOURCE_WARNING``. That was a
    field-shaped claim standing in for a chip-shaped one, and it was wrong in the
    lenient direction and in the strict direction at once:

    * at unit 81 the rake is zero, the hero won an 80-chip pot he put 40 into,
      and the recorded +40 is exactly what the action line derives with no
      declaration at all. Blocking there disclosed an assumption nothing rested
      on -- and an operator taught that Acknowledge is the price of using the
      settlement editor is an operator who will acknowledge the one that matters;
    * at unit 0.01 the rate is IDENTICAL and the hand is entirely different: it
      reconciles only because 80 declared chips were destroyed.

    ``rake_rate`` cannot tell those apart, and neither can any other tuple of
    fields, which is why nothing enumerates fields any more. The measurement
    does tell them apart, and this pins both directions of it.
    """
    db = _open_db(tmp_path)
    hand = _seed_chopped_hand(
        db,
        bet=40.0,
        hero_bb_won=hero_bb_won,
        winners=("hero",),
        amounts=(None,),
        orders=(1,),
    )
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=1.0, rake_rounding_unit=chip_unit)
    )
    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)

    assert result.ledger.rake == pytest.approx(expected_rake)
    assert result.is_authoritative is True
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)

    # The claim is about the RAKE. The declared pot award is a second, unrelated
    # declaration measured on its own, so it is named here and answered where the
    # hand is expected to become ready, rather than being folded into the rake's
    # verdict.
    rake_dependence = [
        item for item in result.assumption_dependence if item.input_name == "rake_policy"
    ]
    assert [item.input_name for item in result.assumption_dependence if item.input_name
            != "rake_policy"] == ["declared_pot_awards"]

    if not expects_dependence:
        assert rake_dependence == []
        assert DECLARED_RAKE_CODE not in evidence.warning_codes
        assert stored.completion_status == "complete"
        assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT") is True
        attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
        awarded = db.fetch_hand(hand.id)
        assert awarded is not None
        assert (
            evaluate_study_readiness(
                awarded, accounting=result, user_confirmed=True
            ).is_ready
            is True
        )
        db.close()
        return

    (dependence,) = rake_dependence
    assert dict(dependence.deltas)["rake"] == pytest.approx(80.0)
    assert dict(dependence.deltas)["hero"] == pytest.approx(-80.0)
    # AMENDED in round 12: an operator's rake declaration is recorded in the
    # operator's own evidence channel, so it no longer demotes the
    # RECONSTRUCTION's completion status or presents as a pipeline finding. The
    # blocker that holds the hand is the measured dependence, which is what this
    # test was always about.
    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert DECLARED_RAKE_CODE not in evidence.unresolved_codes
    assert stored.completion_status == "complete"
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    assert readiness.has("UNRESOLVED_SOURCE_WARNING") is False

    # Acknowledging every pipeline warning is not an attestation to the chips --
    # and, since round 12, there is no pipeline warning here to acknowledge.
    _acknowledge_all(db, hand.id)
    partly = db.fetch_hand(hand.id)
    assert partly is not None
    assert DECLARED_RAKE_CODE not in parse_completion_evidence(
        partly.completion_evidence
    ).acknowledged_codes
    assert (
        evaluate_study_readiness(partly, accounting=result, user_confirmed=True).has(
            "ACCOUNTING_ASSUMPTION_DEPENDENT"
        )
        is True
    )

    assert dependence.code in attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    assert (
        evaluate_study_readiness(
            attested, accounting=result, user_confirmed=True
        ).is_ready
        is True
    )
    db.close()


def test_re_saving_the_same_chip_unit_keeps_the_rake_acknowledgement(
    tmp_path: Path,
) -> None:
    """The unit joining the policy must not make the disclosure unclearable.

    ``persist_reconciliation`` re-saves the settlement on every reconcile, so a
    field that re-raised on an unchanged value would leave the operator with a
    block no action clears.

    AMENDED in round 12: the disclosure lives in the operator's own evidence
    channel and is idempotent by construction, so an unchanged re-save leaves it
    exactly where it was and leaves the reconstruction's completion status alone.
    """
    db = _open_db(tmp_path)
    hand = _seed_chopped_hand(
        db, bet=40.0, hero_bb_won=0.0, winners=("hero",), amounts=(None,), orders=(1,)
    )
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=1.0, rake_rounding_unit=5.0)
    )
    persist_reconciliation(db, hand.id)
    first = db.fetch_hand(hand.id)
    assert first is not None

    persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)

    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert DECLARED_RAKE_CODE not in evidence.acknowledged_codes
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    assert evidence.unresolved_codes == ()
    assert stored.completion_status == "complete"
    assert (
        parse_completion_evidence(first.completion_evidence).declared_settlement_codes
        == evidence.declared_settlement_codes
    )
    db.close()


# ---------------------------------------------------------------------------
# Finding 5 -- the odd-chip order is a declared fact and belongs in the audit
# ---------------------------------------------------------------------------


def test_swapping_the_declared_odd_chip_order_is_recorded_as_a_correction(
    tmp_path: Path,
) -> None:
    """The 'Order' column of the awards editor decides who receives the odd chip.

    ``_declared_award_state`` sorted its claims alphabetically and documented row
    order as deliberately excluded, so swapping two award rows moved the whole odd
    chip between seats, flipped the derived hero result, cleared
    ACCOUNTING_NOT_AUTHORITATIVE, and left no ``hand_corrections`` row and no
    completion-evidence disclosure at all.
    """
    db = _open_db(tmp_path)
    # Three contributions of 7 make an odd chip physically unavoidable: the 21-chip
    # pot cannot be chopped evenly two ways at whole-chip granularity, which is the
    # coarsest denomination the hand's own numbers demonstrate. The declared "Chip
    # unit" is deliberately left at its honest default -- since round 9 it cannot
    # manufacture an odd chip, so the chip this test moves has to be a real one.
    hand = _seed_chopped_hand(db, bet=7.0, hero_bb_won=4.0, extra_callers=("third",))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=0.01)
    )
    first = persist_reconciliation(db, hand.id)
    assert first.ledger.payouts["hero"] == pytest.approx(11.0)
    _acknowledge_all(db, hand.id)
    # Newest first, so the new rows are the ones whose ids are not already here.
    before_ids = {item.id for item in db.fetch_hand_corrections(hand.id)}

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
            for key, order in (("hero", 2), ("villain", 1))
        ],
    )
    swapped = persist_reconciliation(db, hand.id)
    recorded = [
        item for item in db.fetch_hand_corrections(hand.id) if item.id not in before_ids
    ]
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)

    assert swapped.ledger.payouts["hero"] == pytest.approx(10.0)
    assert [item.correction_type for item in recorded] == ["settlement_award_update"]
    assert recorded[0].before_state != recorded[0].after_state
    assert SOURCE_CORRECTION_CODE in evidence.unresolved_codes
    assert stored.completion_status == "uncertain"
    db.close()


def test_re_saving_an_unchanged_award_order_records_no_correction(
    tmp_path: Path,
) -> None:
    """Order entered the snapshot, so an idempotent save must still compare equal."""
    db = _open_db(tmp_path)
    hand = _seed_chopped_hand(db, hero_bb_won=0.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    before = len(db.fetch_hand_corrections(hand.id))

    db.replace_settlement_entries(hand.id, db.fetch_settlement_entries(hand.id))

    assert len(db.fetch_hand_corrections(hand.id)) == before
    db.close()


# ---------------------------------------------------------------------------
# Findings 7 and 14 -- create_settlement_entry is a public writer
# ---------------------------------------------------------------------------


def test_create_settlement_entry_discloses_a_re_declared_award(
    tmp_path: Path,
) -> None:
    """Both public writers of ``settlement_entries`` must disclose a new winner.

    Adding a second award row for a pot turns a single winner into a chop and
    moves every derived payout. Through ``replace_settlement_entries`` that costs
    a ``settlement_award_update`` correction and an acknowledgement of
    ``source_facts_corrected``; through its sibling it used to cost nothing at
    all, leaving ``completion_status`` at ``complete`` with no audit trail.
    """
    db = _open_db(tmp_path)
    hand = _seed_chopped_hand(
        db, hero_bb_won=10.0, winners=("hero",), amounts=(None,), orders=(1,)
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    before_ids = {item.id for item in db.fetch_hand_corrections(hand.id)}

    db.create_settlement_entry(
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=0,
            player_key="villain",
            player_name="Villain",
            amount=None,
            entry_order=2,
        )
    )
    recorded = [
        item for item in db.fetch_hand_corrections(hand.id) if item.id not in before_ids
    ]
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)

    assert [item.correction_type for item in recorded] == ["settlement_award_update"]
    assert SOURCE_CORRECTION_CODE in evidence.unresolved_codes
    assert stored.completion_status == "uncertain"
    db.close()


def test_create_settlement_entry_demotes_a_reviewed_hand(tmp_path: Path) -> None:
    """The demotion the writer's own comment justifies had no direct coverage.

    It was only ever exercised through ``replace_settlement_entries``, which
    repeats both calls itself afterwards, so deleting them from the sibling
    writer left the whole suite green.
    """
    db = _open_db(tmp_path)
    hand = _seed_chopped_hand(
        db, hero_bb_won=0.0, winners=("hero",), amounts=(None,), orders=(1,)
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="Hero bet the river.",
            theory_coach="Bet more.",
            exploit_coach="Bet less.",
            study_lesson="Size up.",
        )
    )
    db._execute(
        "UPDATE hands SET review_status = 'reviewed' WHERE id = ?", (hand.id,)
    )
    db._commit()

    db.create_settlement_entry(
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=0,
            player_key="villain",
            player_name="Villain",
            amount=None,
            entry_order=2,
        )
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    assert stored.review_status == "needs_correction"
    assert all(review.is_stale for review in db.fetch_reviews_by_hand(hand.id))
    db.close()


# ---------------------------------------------------------------------------
# Findings 8 and 9 -- the layout blocker named as impossible the actions that work
# ---------------------------------------------------------------------------


def _layout_blocker(hand: Hand):
    readiness = evaluate_study_readiness(hand, accounting=None, user_confirmed=True)
    return next(
        (item for item in readiness.blockers if item.code == "UNSUPPORTED_TABLE_LAYOUT"),
        None,
    )


def test_the_layout_blocker_names_correct_hand_facts_for_a_missing_table_size() -> None:
    """``hand.table_size`` is an ordinary editable column, not pipeline evidence.

    Its detail rendered under a clearing action ending "Correcting the table size
    by hand does not clear it", and typing the table size was the only action that
    did clear it.
    """
    evidence = _clean_evidence()
    blocked = Hand(
        session_id=1,
        hand_number=1,
        table_size=None,
        source_type="cv_import",
        completion_status="complete",
        completion_evidence=evidence,
    )
    blocker = _layout_blocker(blocked)

    assert blocker is not None
    assert blocker.detail == ("hand.table_size is not recorded",)
    assert "Hand facts" in blocker.clearing_action
    assert "does not clear it" not in blocker.clearing_action

    corrected = blocked.model_copy(update={"table_size": 6})
    assert _layout_blocker(corrected) is None


def test_the_layout_blocker_names_acknowledgement_for_a_hero_seat_mismatch() -> None:
    """``hero_seat_mismatch`` is an acknowledgeable warning, and one press cleared it.

    The blocker told the operator that only a new reconstruction could clear a
    line that ``acknowledge_codes`` removes, which reads as "this hand is beyond
    repair" to someone holding the fix.
    """
    evidence = _clean_evidence(warning_codes=["hero_seat_mismatch"])
    hand = Hand(
        session_id=1,
        hand_number=1,
        table_size=6,
        source_type="cv_import",
        completion_status="complete",
        completion_evidence=evidence,
    )
    blocker = _layout_blocker(hand)

    assert blocker is not None
    assert blocker.detail == ("hero_seat_mismatch",)
    assert "Acknowledge" in blocker.clearing_action
    assert "does not clear it" not in blocker.clearing_action

    acknowledged = hand.model_copy(
        update={
            "completion_evidence": dump_completion_evidence(
                acknowledge_codes(
                    parse_completion_evidence(evidence), ["hero_seat_mismatch"]
                )
            )
        }
    )
    assert _layout_blocker(acknowledged) is None


def test_the_layout_blocker_still_says_a_reconstruction_is_the_only_fix() -> None:
    """The sentence is exactly right about the evidence-borne causes, and stays."""
    hand = Hand(
        session_id=1,
        hand_number=1,
        table_size=6,
        source_type="cv_import",
        completion_status="complete",
        completion_evidence=_clean_evidence(layout_supported=False),
    )
    blocker = _layout_blocker(hand)

    assert blocker is not None
    assert "Correcting the table size by hand does not clear it" in blocker.clearing_action
    assert _layout_blocker(hand.model_copy(update={"table_size": 9})) is not None


@pytest.mark.parametrize("recorded", [2, 9])
def test_a_recorded_table_size_that_contradicts_the_evidence_blocks(
    recorded: int,
) -> None:
    """Any typed value satisfied the gate, because the two were never compared."""
    hand = Hand(
        session_id=1,
        hand_number=1,
        table_size=recorded,
        source_type="cv_import",
        completion_status="complete",
        completion_evidence=_clean_evidence(table_size=6),
    )
    blocker = _layout_blocker(hand)

    assert blocker is not None
    assert blocker.detail == (
        f"hand.table_size={recorded} disagrees with evidence.table_size=6",
    )


# ---------------------------------------------------------------------------
# Finding 10 -- the confirmation gate promised evidence nothing rendered
# ---------------------------------------------------------------------------


def test_the_confirmation_gate_has_evidence_to_render() -> None:
    """"I have read the evidence above" had no display consumer for any field.

    The pipeline wrote the boundaries, the terminal event, the confidence, the
    source timestamps and frames, the layout profile and the pipeline/model
    versions; the store persisted them, the exporter round-tripped them and the
    confirmation key digested them. Grepping the whole application for those field
    names returned nothing, so the operator attested to evidence they were never
    shown.
    """
    evidence = parse_completion_evidence(
        _clean_evidence(
            terminal_event="showdown",
            first_source_timestamp_s=10.0,
            last_source_timestamp_s=60.0,
            preceding_boundary={
                "kind": "hand_start",
                "timestamp_s": 9.0,
                "frame_ref": "frames/1.png",
                "confidence": 0.9,
            },
            following_boundary={"kind": "hand_end", "timestamp_s": 61.0},
            source_frames=["frames/1.png", "frames/2.png"],
            layout_profile="clubwpt_6max",
            pipeline_version="two-model-v7",
            model_versions={"detector": "v7"},
        )
    )
    rendered = " | ".join(f"{label}: {value}" for label, value in completion_evidence_rows(evidence))

    for expected in (
        "showdown",
        "0.92",
        "10s → 60s",
        "hand_start",
        "hand_end",
        "frames/1.png",
        "frames/2.png",
        "clubwpt_6max",
        "two-model-v7",
        "detector=v7",
        "6 seats",
    ):
        assert expected in rendered, expected


def test_unreadable_evidence_still_renders_rather_than_breaking_the_gate() -> None:
    """The parser never raises, and neither may the panel drawn above the gate."""
    rows = completion_evidence_rows(parse_completion_evidence({}))

    assert rows
    assert any("not readable by this build" in value for _, value in rows)


def test_boundary_rows_are_omitted_when_the_pipeline_recorded_none() -> None:
    """An empty boundary is not evidence and must not be drawn as a filled row."""
    evidence = CompletionEvidence(
        evidence_version=EVIDENCE_SCHEMA_VERSION,
        preceding_boundary=BoundaryEvidence(),
        following_boundary=BoundaryEvidence(kind="hand_end", timestamp_s=61.0),
    )
    labels = [label for label, _ in completion_evidence_rows(evidence)]

    assert "Preceding boundary" not in labels
    assert "Following boundary" in labels


# ---------------------------------------------------------------------------
# Finding 11 -- the import card-restore guard had no test at all
# ---------------------------------------------------------------------------


def test_an_import_payload_cannot_overwrite_readable_card_columns(
    tmp_path: Path,
) -> None:
    """``restore_unreadable_card_columns`` may only add a blocker, never remove one.

    It writes the recorded text of a column the reader could not parse back into
    the row, bypassing ``Hand``'s validation on purpose, and is guarded by
    ``AND {column} = ''``. Removing that guard let a payload's marker text replace
    the hand's real hero and board cards -- the two source facts every card gate
    is derived from -- and the whole suite stayed green.
    """
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Round 8", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Kd",
            board_cards="2c 7d 9s",
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None

    db.restore_unreadable_card_columns(
        hand.id, {"hero_cards": "2s 2h", "board_cards": "3c 3d 3h"}
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    assert stored.hero_cards == "Ah Kd"
    assert stored.board_cards == "2c 7d 9s"
    db.close()


def test_the_card_restore_still_fills_a_column_the_payload_left_empty(
    tmp_path: Path,
) -> None:
    """The guard must not disable the producer it guards."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Round 8", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="",
            board_cards="",
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None

    db.restore_unreadable_card_columns(hand.id, {"board_cards": "2c 7d ??"})
    row = db._execute(
        "SELECT board_cards FROM hands WHERE id = ?", (hand.id,)
    ).fetchone()

    assert row["board_cards"] == "2c 7d ??"
    db.close()


# ---------------------------------------------------------------------------
# Finding 13 -- the pinned-snapshot schema exemption had no test
# ---------------------------------------------------------------------------


def test_a_pinned_snapshot_behind_the_live_schema_still_passes_the_audit(
    tmp_path: Path,
) -> None:
    """A pinned pre-migration snapshot is stamped at the PRE-upgrade version.

    Comparing it against the current one downgrades an intact rollback point to a
    warning immediately after any successful upgrade, which is the state every
    operator is in on their first post-upgrade health check.
    """
    live = tmp_path / "live.sqlite3"
    db = PokerDatabase(str(live))
    db.init_db()
    db.close()
    backups = tmp_path / "backups"
    backups.mkdir()
    snapshot = backup_database(live, backups, pinned=True)
    behind = SCHEMA_VERSION - 1
    connection = sqlite3.connect(snapshot)
    connection.execute(
        "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
        (str(behind),),
    )
    connection.commit()
    connection.close()

    report = audit_data_health(
        str(live),
        backup_dir=str(backups),
        expected_schema_version=SCHEMA_VERSION,
        restore_backups=True,
    )
    backups_check = next(item for item in report.checks if item.name == "backups")

    assert backups_check.status == "pass", backups_check.detail


# ---------------------------------------------------------------------------
# Findings 10, 12, 15 and 16 -- the UI surfaces that carry the gate
# ---------------------------------------------------------------------------


def _seed_ui_hand(path: Path) -> int:
    """One reconciled, study-ready reconstructed hand with full pipeline evidence."""
    from tests.conftest import attest_declared_assumptions

    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Round 8 UI", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20,
            hero_bb_won=10,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(
                terminal_event="showdown",
                first_source_timestamp_s=10.0,
                last_source_timestamp_s=60.0,
                source_frames=["frames/1.png"],
                layout_profile="clubwpt_6max",
                pipeline_version="two-model-v7",
                model_versions={"detector": "v7"},
            ),
        )
    )
    assert hand.id is not None
    for key, name, hero, position in (
        ("hero", "Hero", True, "BTN"),
        ("villain", "Villain", False, "BB"),
    ):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                position=position,
                is_hero=hero,
                starting_stack=100,
            )
        )
    for key, name, kind in (("hero", "Hero", "bet"), ("villain", "Villain", "call")):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                street="river",
                action_type=kind,
                amount=10,
                amount_semantics="incremental",
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=20,
                entry_order=1,
            )
        ],
    )
    persist_reconciliation(db, hand.id)
    attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    hand_id = hand.id
    db.close()
    return hand_id


def test_reconstruction_evidence_surface_renders_the_fields_confirmation_names(
    tmp_path: Path, monkeypatch
) -> None:
    """Confirmation attests to concrete evidence fields — they must be drawn.

    On this hand confirmation is the only remaining gate, so before the fix the
    entire content above the control was the one sentence saying it had not been
    confirmed.
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import poker_tracker.persistence.db as db_module

    path = tmp_path / "round8-ui.sqlite3"
    hand_id = _seed_ui_hand(path)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    script = path.parent / "_round8_evidence.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "from poker_tracker.persistence.completion import parse_completion_evidence",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"hand = db.fetch_hand({hand_id})",
                "app_module.show_reconstruction_evidence(",
                "    hand, parse_completion_evidence(hand.completion_evidence)",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not list(app.exception)

    rendered = "\n".join(item.value for item in app.markdown)
    for expected in ("showdown", "two-model-v7", "clubwpt_6max", "detector=v7"):
        assert expected in rendered, expected


def test_hand_study_readiness_blocks_on_a_legacy_stale_hand_review(
    tmp_path: Path,
) -> None:
    """``app.hand_study_readiness`` feeds three surfaces, two of which promote.

    Deleting its ``hand_reviews`` fetch left the whole suite green, so the
    regression the Study page's own copy carries a comment about -- a blocked hand
    reporting "Study-ready · 0 blockers" -- could reappear on the Insights KPI, the
    saved-hands writer and the Settings → Coach promotion path with nothing going
    red. The issue and solver inputs of the same helper are covered; only this one
    was not.
    """
    import app as app_module

    hand_id = _seed_ui_hand(tmp_path / "round8-legacy.sqlite3")
    db = PokerDatabase(str(tmp_path / "round8-legacy.sqlite3"))
    db.init_db()
    db.create_hand_review(
        HandReview(
            hand_id=hand_id,
            hand_summary="Hero bet the river.",
            theory_coach="Bet more.",
            exploit_coach="Bet less.",
            study_lesson="Size up.",
            is_stale=True,
            stale_reason="Hand evidence changed; rerun coaching.",
        )
    )
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    accounting = reconcile_persisted_hand(db, hand_id)

    readiness = app_module.hand_study_readiness(
        db, stored, accounting, None, user_confirmed=True
    )

    assert readiness.has("STALE_COACHING_EVIDENCE")
    assert readiness.is_ready is False
    db.close()


def test_the_landing_hero_says_marked_reviewed(tmp_path: Path, monkeypatch) -> None:
    """"N% reviewed" is the ambiguous form Phase 1 exists to eliminate.

    PLAN.md states the wording verbatim as a requirement, and nothing pinned it:
    the first number a new user sees could silently regress to a bare percentage
    of a workflow label.
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import poker_tracker.persistence.db as db_module

    path = tmp_path / "round8-hero.sqlite3"
    _seed_ui_hand(path)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    app = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30
    ).run()
    assert not list(app.exception)

    rendered = "\n".join(item.value for item in app.markdown)

    assert "marked reviewed" in rendered


def test_a_hand_typed_in_as_reconstructed_is_stored_unproven(
    tmp_path: Path, monkeypatch
) -> None:
    """A reconstructed hand without completion evidence stays unproven on disk.

    ``_hand_from_row`` repairs the ``cv_import``/``not_applicable`` pair on every
    read, so a model-only assertion cannot see the writer regress; the stored
    row still has to be written correctly, because the repair is defence in depth
    and not the contract. Sessions no longer offers a Source picker — assert the
    persistence writer directly.
    """
    path = tmp_path / "round8-add-hand.sqlite3"
    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Round 8 writer", date_played=date(2026, 1, 1)))
    assert session.id is not None
    created = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            source_type="cv_import",
        )
    )
    assert created.id is not None
    db.close()

    connection = sqlite3.connect(str(path))
    stored = connection.execute(
        "SELECT source_type, completion_status FROM hands WHERE hand_number = 2"
    ).fetchone()
    connection.close()

    assert stored == ("cv_import", "uncertain")
