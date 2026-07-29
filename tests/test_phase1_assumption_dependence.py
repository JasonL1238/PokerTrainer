"""Regressions for the dependence-derived settlement-assumption rule.

Rounds 4 through 9 each closed one shape of the same defect: a settlement input
the operator declares -- dead money, a rake rate, a rake cap, a chip unit, a
no-flop-no-drop waiver -- moves the DERIVED side of a cross-check, so comparing
the recorded figures exactly detects nothing and any hero result in
``[-contribution, gross - contribution]`` can be made to reconcile. The
mitigation each round was a DISCLOSURE raised by a list of per-field conditions,
and each following round found one more field combination the list did not
cover.

The rule under test replaces the list with a measurement. Reconcile the hand
with the stored settlement assumptions; reconcile the SAME records again with a
neutral rake policy and no dead money; if it survives the first and not the
second, its reconciliation is assumption-dependent. There is no field in that
rule, so there is no per-field hole in it.

Every test below fails on the pre-change tree, and the file states for each what
it did there.
"""

from __future__ import annotations

import sqlite3
from datetime import date
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
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import DECLARED_RAKE_CODE, PokerDatabase
from poker_tracker.persistence.import_export import export_session, import_session
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness
from tests.conftest import attest_declared_assumptions

CHIP_UNITS = [0.001, 0.01, 0.25, 1.0, 5.0, 7.0, 81.0, 1e6]


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


def _open_db(tmp_path: Path, name: str = "dependence.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed(
    db: PokerDatabase,
    *,
    seats: int = 2,
    bet: float = 40.0,
    winners: tuple[str, ...] = ("hero",),
    hero_bb_won: float | None,
    pot_size: float | None,
    board_cards: str = "Qd 7s 2c",
    street: str = "river",
    award_amounts: tuple[float | None, ...] | None = None,
) -> Hand:
    """``seats`` seats commit ``bet`` each; ``winners`` are declared to share pot 0."""
    session = db.create_session(Session(name="Dependence", date_played=date(2026, 1, 1)))
    assert session.id is not None
    keys = ["hero", "villain", "third", "fourth", "fifth"][:seats]
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards=board_cards,
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None
    for key in keys:
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000,
            )
        )
    for index, key in enumerate(keys, start=1):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street=street,  # type: ignore[arg-type]
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 1 else "call",
                amount=bet,
            )
        )
    amounts = award_amounts or tuple(None for _ in winners)
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
            for order, (key, amount) in enumerate(
                zip(winners, amounts, strict=True), start=1
            )
        ],
    )
    return hand


def _named(result: object, input_name: str):
    """The one measured dependence naming ``input_name``.

    Every hand below declares its pot award as well as its rake or dead money,
    because a reconstructed hand's award rows are typed into the same panel by
    the same operator and are measured as their own declared input. Selecting by
    name keeps each test pinning the declaration it is about.
    """
    (found,) = [
        item
        for item in result.assumption_dependence  # type: ignore[attr-defined]
        if item.input_name == input_name
    ]
    return found


def _inputs(result: object) -> list[str]:
    return [item.input_name for item in result.assumption_dependence]  # type: ignore[attr-defined]


def _acknowledge_every_warning(db: PokerDatabase, hand_id: int) -> Hand:
    """Everything the pre-change product offered the operator, pressed."""
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    db.update_hand_completion(
        hand_id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(evidence, list(evidence.unresolved_codes))
        ),
        notes="Acknowledged in test.",
    )
    refreshed = db.fetch_hand(hand_id)
    assert refreshed is not None
    return refreshed


# ---------------------------------------------------------------------------
# 1. A writer that is not upsert_hand_settlement
# ---------------------------------------------------------------------------


def test_a_hand_edited_settlement_row_cannot_launder_a_fabricated_hero_result(
    tmp_path: Path,
) -> None:
    """The verdict must live with the READER, or one UPDATE statement defeats it.

    Every declared-chips disclosure through round 9 was written by
    ``upsert_hand_settlement``. ``hand_settlements`` carries no CHECK constraint,
    so a plain ``UPDATE`` -- a hand-edited database, a future writer, a migration
    -- reached ``reconcile_persisted_hand`` having raised no code, recorded no
    correction and demoted nothing, and the hand presented as reconciled,
    authoritative and study-ready with an EMPTY blocker tuple while its recorded
    hero result disagreed with its own action line by half the pot.

    BEFORE this change: readiness.is_ready is True with zero blockers.
    """
    db = _open_db(tmp_path)
    # 80 chips in, hero wins the lot: the honest hero result is +40.
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0)
    assert hand.id is not None
    path = db.db_path
    db.close()

    connection = sqlite3.connect(path)
    connection.execute(
        """
        INSERT INTO hand_settlements (
            hand_id, status, dead_money, rake_rate, rake_cap, rake_rounding_unit,
            no_flop_no_drop, gross_pot, rake_amount, net_pot, is_balanced,
            warnings, created_at, updated_at
        )
        VALUES (?, 'reconciled', 0, 0.5, NULL, 0.01, 0, 80.0, 40.0, 40.0, 1,
                '[]', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """,
        (hand.id,),
    )
    connection.commit()
    connection.close()

    db = PokerDatabase(path)
    result = reconcile_persisted_hand(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)

    # The ledger models the declaration faithfully and reconciles: that is the
    # whole problem, and it is not the defect.
    assert result.is_authoritative is True
    assert result.ledger.rake == pytest.approx(40.0)
    # No writer ran, so no disclosure was ever recorded.
    assert evidence.warning_codes == ()
    assert stored.completion_status == "complete"

    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    dependence = _named(result, "rake_policy")
    assert dict(dependence.deltas)["hero"] == pytest.approx(-40.0)

    # The clearing action the blocker names, performed verbatim, actually clears it.
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


# ---------------------------------------------------------------------------
# 2. The attestation is bound to chips, not to a policy tuple
# ---------------------------------------------------------------------------


def test_an_attestation_does_not_cover_the_same_policy_taking_more_chips(
    tmp_path: Path,
) -> None:
    """A quantity attested to is a quantity, and the pot is half of it.

    ``_record_declared_chip_adjustment`` re-raised its disclosure when the POLICY
    changed. The chips a policy takes are ``gross x rate``, and only ``rate`` was
    watched, so an attestation earned while the policy destroyed 0.1 chips
    carried over unchanged once a corrected action line grew the pot 8000-fold
    and the identical policy destroyed 800.1 chips. No comparison of settlement
    fields can notice that, because no settlement field moved.

    BEFORE this change: the grown hand reports the same acknowledged disclosure
    and readiness raises no accounting blocker at all.
    """
    db = _open_db(tmp_path)
    hand = _seed(db, bet=0.05, hero_bb_won=None, pot_size=None, street="flop")
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=1.0, rake_rounding_unit=0.01)
    )
    small = persist_reconciliation(db, hand.id)
    assert small.ledger.rake == pytest.approx(0.1)
    tiny = _named(small, "rake_policy")
    assert tiny.code in attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    assert (
        evaluate_study_readiness(
            attested, accounting=small, user_confirmed=True
        ).has("ACCOUNTING_ASSUMPTION_DEPENDENT")
        is False
    )

    for index, key in enumerate(("hero", "villain"), start=3):
        db.create_corrected_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 3 else "call",
                amount=400.0,
            ),
            correction_notes="Recovered the river.",
        )
    grown = persist_reconciliation(db, hand.id)
    assert grown.ledger.rake == pytest.approx(800.1)
    large = _named(grown, "rake_policy")

    assert large.code != tiny.code
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    # The attestation lives in its own channel and never in the pipeline one:
    # `acknowledged_codes` is cleared by the generic Source-warnings Acknowledge.
    assert tiny.code in evidence.confirmed_assumption_codes
    assert large.code not in evidence.confirmed_assumption_codes
    assert not any(
        code.startswith("declared_settlement_dependence")
        for code in (*evidence.warning_codes, *evidence.acknowledged_codes)
    )
    assert (
        evaluate_study_readiness(stored, accounting=grown, user_confirmed=True).has(
            "ACCOUNTING_ASSUMPTION_DEPENDENT"
        )
        is True
    )

    assert large.code in attest_declared_assumptions(db, hand.id)
    reattested = db.fetch_hand(hand.id)
    assert reattested is not None
    assert (
        evaluate_study_readiness(
            reattested, accounting=grown, user_confirmed=True
        ).has("ACCOUNTING_ASSUMPTION_DEPENDENT")
        is False
    )
    db.close()


# ---------------------------------------------------------------------------
# 3. A forged import payload
# ---------------------------------------------------------------------------


def test_a_forged_import_payload_declaring_a_rake_policy_stays_blocked(
    tmp_path: Path,
) -> None:
    """One ``import_session`` call, then every offered acknowledgement, pressed.

    ``import_session`` never calls ``persist_reconciliation``, so a payload's
    declared policy and its fabricated ``hero_bb_won`` arrive together and are
    cross-checked against each other. The pre-change product disclosed the
    policy and let a single Acknowledge in the Source warnings panel clear it,
    which is an attestation to the FACT of a declaration by an operator who has
    never seen this hand.

    BEFORE this change: after acknowledging every code the payload raised, the
    imported hand is study-ready with zero blockers.
    """
    source = _open_db(tmp_path, "source.db")
    hand = _seed(source, hero_bb_won=40.0, pot_size=80.0)
    assert hand.id is not None
    persist_reconciliation(source, hand.id)
    payload = export_session(source, hand.session_id)
    source.close()

    entry = payload["hands"][0]
    entry["hand"]["hero_bb_won"] = 0.0
    entry["settlement"]["rake_rate"] = 0.5
    entry["settlement"]["rake_amount"] = 40.0
    entry["settlement"]["net_pot"] = 40.0
    entry["settlement"]["status"] = "reconciled"

    target = _open_db(tmp_path, "target.db")
    session = import_session(target, payload)
    assert session.id is not None
    landed = target.fetch_hands_by_session(session.id)[0]
    assert landed.id is not None
    result = reconcile_persisted_hand(target, landed.id)

    assert result.is_authoritative is True  # the forgery reconciles exactly
    stored = _acknowledge_every_warning(target, landed.id)
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")

    dependence = _named(result, "rake_policy")
    assert dict(dependence.deltas)["hero"] == pytest.approx(-40.0)
    assert dependence.code in attest_declared_assumptions(target, landed.id)
    attested = target.fetch_hand(landed.id)
    assert attested is not None
    assert (
        evaluate_study_readiness(
            attested, accounting=result, user_confirmed=True
        ).is_ready
        is True
    )
    target.close()


# ---------------------------------------------------------------------------
# 4. The round-8 and round-9 chopped-pot shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seats", "bet", "winners", "rate", "declared_hero", "honest_hero"),
    [
        # Round 8's shape: two contributors, one winner, the whole pot raked.
        (2, 40.0, ("hero",), 1.0, -40.0, 40.0),
        # Round 9's shape: three contributors, TWO winners -- the case round 8's
        # repair could not see, because with equal contributions every divisor
        # halves the pot evenly and the bound looked like a fix.
        (3, 8.0, ("hero", "villain"), 0.25, 1.0, 4.0),
        # More contributors still, and a rake that takes exactly half.
        (4, 10.0, ("hero", "villain"), 0.5, 0.0, 10.0),
    ],
    ids=["two-seats-one-winner", "three-seats-two-winners", "four-seats-two-winners"],
)
def test_a_chopped_pot_shape_cannot_be_reconciled_by_a_declared_rake_unannounced(
    tmp_path: Path,
    seats: int,
    bet: float,
    winners: tuple[str, ...],
    rate: float,
    declared_hero: float,
    honest_hero: float,
) -> None:
    """The same defect at three shapes, repaired once rather than three times.

    Each row records a ``hands.hero_bb_won`` that the action line alone
    contradicts and that only the declared rake makes true. Rounds 8 and 9 fixed
    the two-seat shape and then the three-seat shape; a rule with no field list
    and no shape in it closes all three and the ones nobody has drawn yet.

    BEFORE this change: after acknowledging every offered code, each of these is
    study-ready with zero blockers.
    """
    db = _open_db(tmp_path)
    gross = seats * bet
    hand = _seed(
        db,
        seats=seats,
        bet=bet,
        winners=winners,
        hero_bb_won=declared_hero,
        pot_size=gross,
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=rate))
    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is True
    assert result.ledger.net_results["hero"] == pytest.approx(declared_hero)
    # What the same records derive with nothing declared.
    neutral = build_hand_ledger(
        [
            LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key)
            for key in ["hero", "villain", "third", "fourth", "fifth"][:seats]
        ],
        [
            LedgerAction(
                player=key,
                street="river",
                kind="bet" if index == 0 else "call",
                amount=bet,
            )
            for index, key in enumerate(
                ["hero", "villain", "third", "fourth", "fifth"][:seats]
            )
        ],
        winners={0: winners},
        rake=RakePolicy(),
        odd_chip_order=list(winners),
    )
    assert neutral.net_results["hero"] == pytest.approx(honest_hero)

    stored = _acknowledge_every_warning(db, hand.id)
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    dependence = _named(result, "rake_policy")
    assert dict(dependence.deltas)["hero"] == pytest.approx(declared_hero - honest_hero)

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


# ---------------------------------------------------------------------------
# 5. New field combinations: a declaration that moves nothing is never disclosed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settlement_kwargs", "board_cards", "street"),
    [
        # A rate with a ZERO cap. Nothing is ever taken, and the pre-change gate
        # announced it on every hand because it only looked at the rate.
        ({"rake_rate": 0.5, "rake_cap": 0.0}, "Qd 7s 2c", "river"),
        # No flop, no drop, on a hand that saw no flop: the waiver is real and
        # the rate is irrelevant. Disclosed before; silent now.
        (
            {"rake_rate": 0.4, "no_flop_no_drop": True},
            "",
            "preflop",
        ),
        # A chip unit coarser than the whole rake: 2% of an 80-chip pot is 1.6,
        # rounded down to a 5-chip denomination is nothing.
        ({"rake_rate": 0.02, "rake_rounding_unit": 5.0}, "Qd 7s 2c", "river"),
    ],
    ids=["zero-cap", "no-flop-no-drop-preflop", "unit-coarser-than-the-rake"],
)
def test_a_declared_rake_that_takes_no_chips_is_never_disclosed(
    tmp_path: Path,
    settlement_kwargs: dict[str, object],
    board_cards: str,
    street: str,
) -> None:
    """Over-disclosure is the other half of the defect, and it is not cosmetic.

    Three field combinations the ``rake_rate > 0`` gate got WRONG in the lenient
    direction: each declares a rate, each takes zero chips, each derives exactly
    the ledger the action line derives on its own, and each was announced to the
    operator as an unresolved source warning that demoted the hand to
    ``uncertain``. An operator taught that Acknowledge is the price of using the
    settlement editor is an operator who will acknowledge the one that matters --
    which is precisely how rounds 7, 8 and 9 got their fabricated hero results
    past the disclosure that had already been shown.

    BEFORE this change: each of these raises DECLARED_RAKE_CODE, sets
    completion_status to 'uncertain', and is not study-ready.
    """
    db = _open_db(tmp_path)
    hand = _seed(
        db,
        hero_bb_won=40.0,  # the honest, unraked result
        pot_size=80.0,
        board_cards=board_cards,
        street=street,
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, **settlement_kwargs))
    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)

    assert result.ledger.rake == pytest.approx(0.0)
    assert result.is_authoritative is True
    # The rake is never named. The declared pot award is a different declaration
    # and is measured on its own, so it is answered rather than assumed away.
    assert _inputs(result) == ["declared_pot_awards"]
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    assert stored.completion_status == "complete"
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


# ---------------------------------------------------------------------------
# 6. New field combination: a zero rate, a declared chip unit, and dead money
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chip_unit", CHIP_UNITS)
def test_a_zero_rate_leaves_the_neutral_pass_identical_at_every_chip_unit(
    chip_unit: float,
) -> None:
    """The claim the short-circuit rests on, pinned as a claim.

    ``_derive_assumption_dependence`` skips the second pass entirely when the
    stored policy declares a zero rate and no dead money, on the grounds that the
    neutral ledger is then bit-identical. That was FALSE until round 9 -- the
    declared ``rake_rounding_unit`` also sized a chopped pot, so at rate zero it
    could still move chips between seats -- and it is the one place a saved fetch
    could hide a real difference. Every derived figure is compared, not just the
    rake.
    """
    keys = ("hero", "villain", "third")
    players = [
        LedgerPlayer(name=key.capitalize(), starting_stack=1000, key=key)
        for key in keys
    ]
    actions = [
        LedgerAction(
            player=key, street="river", kind="bet" if index == 0 else "call", amount=7
        )
        for index, key in enumerate(keys)
    ]
    declared = build_hand_ledger(
        players,
        actions,
        winners={0: ("hero", "villain")},
        rake=RakePolicy(
            rate=0,
            cap=chip_unit,
            rounding_unit=chip_unit,
            no_flop_no_drop=True,
        ),
        odd_chip_order=["hero", "villain"],
    )
    neutral = build_hand_ledger(
        players,
        actions,
        winners={0: ("hero", "villain")},
        rake=RakePolicy(),
        odd_chip_order=["hero", "villain"],
    )

    assert declared.gross_pot == neutral.gross_pot
    assert declared.rake == neutral.rake == 0.0
    assert declared.net_pot == neutral.net_pot
    assert declared.payouts == neutral.payouts
    assert declared.net_results == neutral.net_results
    assert declared.refunds == neutral.refunds


@pytest.mark.parametrize("chip_unit", [0.01, 1.0, 81.0])
def test_declared_dead_money_at_a_zero_rate_is_a_dependence_at_every_chip_unit(
    tmp_path: Path, chip_unit: float
) -> None:
    """Zero rate, a declared chip unit, and dead money doing all the work.

    Round 9 closed the chip unit's second job of sizing a chopped pot, which made
    ``rake_rate == 0`` an honest reason to raise nothing about the RAKE. It said
    nothing about the other declared input in the same row: 75 unobserved chips
    move the gross pot from 80 to 155 and the hero's result with it, at every
    chip unit, and the pre-change product cleared that with one Acknowledge of a
    boolean disclosure.

    BEFORE this change: after acknowledging the offered code, study-ready.
    """
    db = _open_db(tmp_path)
    hand = _seed(db, hero_bb_won=115.0, pot_size=155.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            dead_money=75.0,
            rake_rate=0.0,
            rake_rounding_unit=chip_unit,
        )
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.gross_pot == pytest.approx(155.0)
    assert result.ledger.rake == pytest.approx(0.0)
    assert result.is_authoritative is True
    dependence = _named(result, "dead_money")
    assert dict(dependence.deltas)["gross"] == pytest.approx(75.0)

    stored = _acknowledge_every_warning(db, hand.id)
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")

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


# ---------------------------------------------------------------------------
# 7. New field combination: the rake changes with no settlement write at all
# ---------------------------------------------------------------------------


def test_correcting_the_board_turns_a_silent_rake_policy_into_a_dependence(
    tmp_path: Path,
) -> None:
    """``no_flop_no_drop`` is decided by ``board_cards``, which is not a settlement field.

    The whole pre-change disclosure lived in ``upsert_hand_settlement`` and
    compared one settlement row against the previous one. Whether the waiver
    applies is decided by ``hands.board_cards`` and the action streets, and
    ``update_hand_facts`` writes those without touching the settlement at all --
    so correcting a hand from "no flop" to "flop, turn, river" moved the rake
    from 0 to 40 chips, moved the derived hero result by the same 40, and
    compared equal at every settlement field there is.

    BEFORE this change: after acknowledging every offered code the corrected hand
    is study-ready, and its recorded hero result is off by half the pot.
    """
    db = _open_db(tmp_path)
    hand = _seed(
        db, hero_bb_won=40.0, pot_size=80.0, board_cards="", street="preflop"
    )
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.5, no_flop_no_drop=True)
    )
    waived = persist_reconciliation(db, hand.id)
    assert waived.ledger.rake == pytest.approx(0.0)
    assert _inputs(waived) == ["declared_pot_awards"]
    _acknowledge_every_warning(db, hand.id)
    attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    settled = db.fetch_hand(hand.id)
    assert settled is not None
    assert (
        evaluate_study_readiness(
            settled, accounting=waived, user_confirmed=True
        ).is_ready
        is True
    )

    # One correction to a source fact, no settlement write of any kind.
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    db.update_hand_facts(
        stored.model_copy(update={"board_cards": "Qd 7s 2c", "hero_bb_won": 0.0}),
        correction_notes="Recovered the board from the recording.",
    )
    # Twice: persist_reconciliation judges validity from the reconcile it runs
    # BEFORE rewriting the recorded gross/rake/net, so the first call after a
    # source-fact correction is still comparing against the pre-correction
    # restatements. The settlement editor nulls those three columns before
    # saving, which is the same thing by another route.
    persist_reconciliation(db, hand.id)
    dropped = persist_reconciliation(db, hand.id)

    assert dropped.ledger.rake == pytest.approx(40.0)
    assert dropped.is_authoritative is True
    dependence = _named(dropped, "rake_policy")
    assert dict(dependence.deltas)["hero"] == pytest.approx(-40.0)

    corrected = _acknowledge_every_warning(db, hand.id)
    readiness = evaluate_study_readiness(
        corrected, accounting=dropped, user_confirmed=True
    )
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    db.close()


# ---------------------------------------------------------------------------
# 8. Manual hands, and the confirmation that must NOT clear this
# ---------------------------------------------------------------------------


def test_a_manual_hand_is_disclosed_but_never_blocked_by_its_own_declaration(
    tmp_path: Path,
) -> None:
    """The operator IS the source on a manual hand, so there is no claim to outrank."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Manual", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            pot_size=80.0,
            hero_bb_won=0.0,
            source_type="manual",
        )
    )
    assert hand.id is not None
    for key in ("hero", "villain"):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000,
            )
        )
    for index, key in enumerate(("hero", "villain"), start=1):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 1 else "call",
                amount=40.0,
            )
        )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                entry_order=1,
            )
        ],
    )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    # Measured and reported...
    assert result.assumption_dependence != ()
    # ...and never blocking, exactly like every other reconstruction-only blocker.
    readiness = evaluate_study_readiness(stored, accounting=result)
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT") is False
    assert readiness.is_ready is True
    assert stored.completion_evidence == {}
    db.close()


def test_confirming_the_hand_as_a_whole_does_not_clear_the_assumption(
    tmp_path: Path,
) -> None:
    """Two different questions, and only one of them is about the declared chips.

    'I have read the evidence above and confirm this hand is correct' is a
    judgement about the reconstruction. Whether 40 chips were really dropped is a
    claim about the room. Letting the first answer the second is exactly how the
    round-4 through round-9 disclosures were cleared without being read.
    """
    db = _open_db(tmp_path)
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    result = persist_reconciliation(db, hand.id)
    stored = _acknowledge_every_warning(db, hand.id)

    for confirmed in (False, True):
        readiness = evaluate_study_readiness(
            stored, accounting=result, user_confirmed=confirmed
        )
        assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT") is True

    dependence = _named(result, "rake_policy")
    with pytest.raises(ValueError):
        db.acknowledge_accounting_assumption(hand.id, "source_facts_corrected")
    assert dependence.code in attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    # The narrow attestation clears the narrow blocker and nothing else: the
    # whole-hand confirmation is still required on its own.
    assert (
        evaluate_study_readiness(attested, accounting=result, user_confirmed=False).has(
            "USER_CONFIRMATION_MISSING"
        )
        is True
    )
    assert (
        evaluate_study_readiness(
            attested, accounting=result, user_confirmed=True
        ).is_ready
        is True
    )
    db.close()
