"""Regressions for the round-7 adversarial findings against Phase 1.

Every test here failed before its fix. The round-7 themes are *a settlement input
that moves the DERIVED side of a cross-check instead of widening its tolerance*,
*retained analysis that declares its own freshness in an import payload*, *an
attestation that covers the fact of a declaration but never its magnitude*, *a
documented busy timeout that one pragma does not honour*, and *a read-only audit
that writes beside the files it is only inspecting*.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from poker_tracker.maintenance.data_health import audit_data_health
from poker_tracker.persistence.backup import backup_database
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import (
    DECLARED_DEAD_MONEY_CODE,
    DECLARED_RAKE_CODE,
    SCHEMA_VERSION,
    PokerDatabase,
)
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
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness
from tests.conftest import attest_declared_assumptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _open_db(tmp_path: Path, name: str = "round7.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed_hand(
    db: PokerDatabase,
    *,
    pot_size: float | None = 20.0,
    hero_bb_won: float | None = 0.0,
    award: float | None = 20.0,
    declare_award: bool = True,
    source_type: str = "cv_import",
) -> Hand:
    """Two players, bet 10 / call 10, whole pot awarded to the hero.

    Derived gross pot 20 before rake. The hero contributed 10 and is pushed the
    net pot, so the derived hero result is +10 with no rake and 0 with a rake of
    10 -- which is what makes the rake policy a dial over the hero cross-check.
    """
    session = db.create_session(Session(name="Round 7", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type=source_type,  # type: ignore[arg-type]
            completion_status=(
                "not_applicable" if source_type == "manual" else "complete"
            ),
            completion_evidence={} if source_type == "manual" else _clean_evidence(),
        )
    )
    assert hand.id is not None
    for key, name, hero in (("hero", "Hero", True), ("villain", "Villain", False)):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                is_hero=hero,
                starting_stack=1000,
            )
        )
    for index, (key, name, kind) in enumerate(
        (("hero", "Hero", "bet"), ("villain", "Villain", "call")), start=1
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=name,
                action_type=kind,
                amount=10.0,
            )
        )
    if declare_award:
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key="hero",
                    player_name="Hero",
                    amount=award,
                    entry_order=1,
                )
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
# Finding 1 -- the rake policy is a payload-controllable dial that moves the
# DERIVED side of the hero cross-check, disclosed nowhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hero_bb_won", "rake_rate", "rake_cap"),
    [
        (0.0, 0.5, None),  # the hero "broke even" on a pot he won
        (-10.0, 1.0, None),  # the hero "lost" 10 on a pot he won
        (5.0, 0.25, None),
        (2.0, 1.0, 8.0),  # the same forgery through the cap
    ],
)
def test_a_declared_rake_cannot_silently_carry_a_fabricated_hero_result(
    tmp_path: Path, hero_bb_won: float, rake_rate: float, rake_cap: float | None
) -> None:
    """Choosing the rake makes any hero result reconcile exactly.

    ``rake_rate``, ``rake_cap`` and ``no_flop_no_drop`` move the derived ledger
    rather than widen a tolerance, so the hero cross-check being exact buys
    nothing: the derived hero result is ``payout - contribution``, and the rake
    is subtracted from the payout. Dead money -- the mirror-image settlement
    input, which CREATES chips the action line never observed -- is disclosed as
    ``declared_unobserved_chips``. A rake that DESTROYS them must be disclosed
    the same way, or the reconciled verdict rests silently on an operator input
    nothing observed.
    """
    db = _open_db(tmp_path)
    # The pot is awarded to the hero with no declared payout, so the only
    # cross-check left is the hero result the rake is being tuned against.
    hand = _seed_hand(db, hero_bb_won=hero_bb_won, award=None)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=rake_rate, rake_cap=rake_cap)
    )
    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    evidence = parse_completion_evidence(stored.completion_evidence)
    # AMENDED in round 12. The disclosure is recorded in the operator's own
    # channel, and the readiness gate is the measured dependence rather than a
    # pipeline warning: writing an operator's rake into `warning_codes` demoted
    # the RECONSTRUCTION's completion status and reported the operator's own
    # declaration as something "the pipeline flagged". The claim this test is
    # named for is unchanged and strictly stronger -- the blocker that now fires
    # cannot be cleared by the generic one-click Acknowledge.
    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    assert DECLARED_RAKE_CODE not in evidence.unresolved_codes
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    assert stored.completion_status == "complete"


def test_a_zero_rake_policy_discloses_nothing(tmp_path: Path) -> None:
    """The disclosure is about declared chips, so a policy that takes none is silent.

    The named claim is about the RAKE, and it is unchanged: a zero-rate policy is
    never named as a dependence at all. What the hand still owes is an
    attestation to its declared pot award, which is a different declaration and
    is measured on its own -- so this asserts the rake is silent both before and
    after that unrelated attestation, rather than asserting the hand happens to
    have no blockers.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.0))
    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    evidence = parse_completion_evidence(stored.completion_evidence)
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    assert DECLARED_RAKE_CODE not in evidence.declared_settlement_codes
    assert [item.input_name for item in result.assumption_dependence] == [
        "declared_pot_awards"
    ]

    attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    readiness = evaluate_study_readiness(
        attested, accounting=result, user_confirmed=True
    )
    assert readiness.is_ready is True


def test_an_acknowledged_rake_declaration_survives_reconciliation(
    tmp_path: Path,
) -> None:
    """Acknowledging is the clearing action, and re-reconciling must not undo it.

    ``persist_reconciliation`` re-saves the settlement on every reconcile, so a
    disclosure that re-raised on every write could never be cleared at all.

    AMENDED after the dependence rule landed, and the amendment is the finding.
    This hand's reconciliation rests ENTIRELY on the declared rake: the recorded
    ``hero_bb_won`` of 0.0 matches the derived result only because 10 chips were
    declared away, and under a neutral policy the same records derive +10. The
    test used to assert that one press of Acknowledge in the Source warnings
    panel -- an acknowledgement of the *fact* that a rake was declared -- left
    the hand study-ready. That is exactly the click-through the round-7 through
    round-9 findings kept escaping through, so the assertion was enshrining the
    hole rather than pinning a fix.

    What is preserved is the claim that was always correct: an attestation, once
    given, must SURVIVE re-reconciliation, or the operator is left with a
    permanently unclearable blocker. What changed is which attestation is
    required -- the measured chip movement, not the existence of a policy.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=0.0, award=None)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    persist_reconciliation(db, hand.id)
    _acknowledge_all(db, hand.id)

    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    # AMENDED again in round 12: there is no pipeline warning to acknowledge, and
    # _acknowledge_all above therefore acknowledges nothing. The declaration is
    # recorded in the operator's own channel, which no Acknowledge control reads.
    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert DECLARED_RAKE_CODE not in evidence.acknowledged_codes
    assert evidence.unresolved_codes == ()

    # Acknowledging pipeline warnings is NOT an attestation to the chips.
    blocked = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert blocked.is_ready is False
    assert blocked.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    (dependence,) = [
        item for item in result.assumption_dependence if item.input_name == "rake_policy"
    ]
    assert dict(dependence.deltas)["hero"] == pytest.approx(-10.0)

    # The clearing action the blocker names, performed verbatim -- for this
    # declaration and for the declared pot award beside it, which is a separate
    # declaration answered by a separate press of the same control.
    codes = attest_declared_assumptions(db, hand.id)
    assert dependence.code in codes
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    ready = evaluate_study_readiness(attested, accounting=result, user_confirmed=True)
    assert ready.is_ready is True

    # ...and it survives another reconcile, which is the original claim: the
    # measured code is byte-stable across an idempotent re-save.
    again = persist_reconciliation(db, hand.id)
    assert [item.code for item in again.assumption_dependence] == list(codes)
    reread = db.fetch_hand(hand.id)
    assert reread is not None
    assert (
        evaluate_study_readiness(
            reread, accounting=again, user_confirmed=True
        ).is_ready
        is True
    )


def test_changing_an_acknowledged_rake_policy_re_raises_the_disclosure(
    tmp_path: Path,
) -> None:
    """An attestation covers the quantity attested to, not the fact of attesting.

    AMENDED in round 12: the quantity lives in the measured dependence code, and
    that is the thing an attestation is bound to, so this asserts on the
    attestation rather than on a pipeline warning that no longer carries a
    quantity at all.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=0.0, award=None)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    persist_reconciliation(db, hand.id)
    attested = attest_declared_assumptions(db, hand.id, only="rake_policy")
    assert attested

    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=1.0))
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    # The declaration is still recorded, and it is still not a pipeline finding.
    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert evidence.unresolved_codes == ()
    # ...and the earlier attestation covers nothing this hand now measures, so
    # the hand is blocked again without anyone re-declaring anything.
    result = reconcile_persisted_hand(db, hand.id)
    measured = {item.code for item in result.assumption_dependence}
    assert not measured & set(attested)
    assert (
        evaluate_study_readiness(
            stored, accounting=result, user_confirmed=True
        ).is_ready
        is False
    )


def test_an_imported_rake_declaration_is_disclosed_in_the_importing_database(
    tmp_path: Path,
) -> None:
    """The forgery must not survive an export/import round trip unannounced."""
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, hero_bb_won=-10.0, award=None)
    assert hand.id is not None
    source.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=1.0))
    persist_reconciliation(source, hand.id)
    _acknowledge_all(source, hand.id)
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    result = reconcile_persisted_hand(target, imported.id)
    stored = target.fetch_hand(imported.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    # AMENDED in round 12: the importing database re-measures the dependence from
    # the chips, which is a stronger disclosure than the pipeline warning this
    # used to assert -- and, unlike that warning, one press of the generic
    # Acknowledge does not clear it.
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
    assert DECLARED_RAKE_CODE not in parse_completion_evidence(
        stored.completion_evidence
    ).warning_codes


# ---------------------------------------------------------------------------
# Finding 2 -- an import payload declared its own coaching staleness
# ---------------------------------------------------------------------------


def test_an_imported_coaching_review_cannot_declare_itself_current(
    tmp_path: Path,
) -> None:
    """``is_stale`` is a blocker input, and no importing database can verify it.

    Retained coaching describes facts in the database that produced it. Once it
    crosses into another database nothing here can check that it still describes
    this hand, so it arrives stale and is re-run -- exactly as an acknowledgement
    and a ``reviewed`` promotion do not travel either.
    """
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, hero_bb_won=10.0)
    assert hand.id is not None
    source.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="superseded summary",
            theory_coach="t",
            exploit_coach="e",
            study_lesson="l",
        )
    )
    source.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            session_id=hand.session_id,
            review_type="hand",
            provider_name="test",
            model_name="test",
            raw_prompt="p",
            raw_response="superseded summary",
        )
    )
    payload = export_session(source, hand.session_id)
    source.close()

    for review in payload["hands"][0]["reviews"]:
        review["is_stale"] = False
        review["stale_reason"] = ""
    for review in payload["hands"][0]["coaching_reviews"]:
        review["is_stale"] = False
        review["stale_reason"] = ""

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    assert [review.is_stale for review in target.fetch_reviews_by_hand(imported.id)] == [
        True
    ]
    assert [
        review.is_stale
        for review in target.fetch_coaching_reviews_by_hand(imported.id)
    ] == [True]

    result = reconcile_persisted_hand(target, imported.id)
    stored = target.fetch_hand(imported.id)
    assert stored is not None
    readiness = evaluate_study_readiness(
        stored,
        accounting=result,
        coaching_reviews=target.fetch_coaching_reviews_by_hand(imported.id),
        hand_reviews=target.fetch_reviews_by_hand(imported.id),
        user_confirmed=True,
    )
    assert readiness.has("STALE_COACHING_EVIDENCE")
    assert readiness.is_ready is False


def test_an_imported_session_coaching_review_cannot_declare_itself_current(
    tmp_path: Path,
) -> None:
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, hero_bb_won=10.0)
    assert hand.id is not None
    source.create_coaching_response(
        CoachingResponse(
            hand_id=None,
            session_id=hand.session_id,
            review_type="session",
            provider_name="test",
            model_name="test",
            raw_prompt="p",
            raw_response="superseded session summary",
        )
    )
    payload = export_session(source, hand.session_id)
    source.close()
    for review in payload["coaching_reviews"]:
        review["is_stale"] = False
        review["stale_reason"] = ""

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    reviews = target.fetch_coaching_reviews_by_session(session.id)
    assert [review.is_stale for review in reviews] == [True]


# ---------------------------------------------------------------------------
# Finding 3 -- the reconciliation slack was still set by import-payload fields
# ---------------------------------------------------------------------------


def test_an_imported_settlement_cannot_restate_the_rake_it_recorded(
    tmp_path: Path,
) -> None:
    """One ``import_session`` call landed a recorded rake 4.9 chips off the ledger.

    ``persist_reconciliation`` rewrites ``rake_amount`` and ``net_pot`` from the
    ledger, so the tolerance those two comparisons used could only ever be
    exercised by a row the product did not write -- an import payload, which
    supplied all three fields the tolerance was derived from.
    """
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, hero_bb_won=0.0, award=10.0)
    assert hand.id is not None
    payload = export_session(source, hand.session_id)
    source.close()
    payload["hands"][0]["settlement"] = {
        "hand_id": 0,
        "rake_rate": 1.0,
        "rake_cap": 10.0,
        "rake_rounding_unit": 10.0,
        "status": "reconciled",
        "is_balanced": True,
        "gross_pot": 20.0,
        "rake_amount": 14.9,
        "net_pot": 5.1,
    }

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    result = reconcile_persisted_hand(target, imported.id)

    assert result.ledger.rake == pytest.approx(10.0)
    assert "Recorded rake does not match the derived ledger." in result.issues
    assert "Recorded net pot does not match the derived ledger." in result.issues
    assert result.is_authoritative is False
    stored = target.fetch_hand(imported.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")
    assert readiness.is_ready is False


@pytest.mark.parametrize("rounding_unit", [0.01, 1.0, 10.0, 100000.0])
def test_no_chip_denomination_excuses_a_recorded_restatement(
    tmp_path: Path, rounding_unit: float
) -> None:
    """The 'Chip unit' field cannot move any comparison at all, at any size."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=0.0, award=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            status="reconciled",
            is_balanced=True,
            rake_rate=0.5,
            rake_rounding_unit=rounding_unit,
            gross_pot=20.0,
            rake_amount=10.0 + rounding_unit / 2,
            net_pot=max(10.0 - rounding_unit / 2, 0.0),
        )
    )
    result = reconcile_persisted_hand(db, hand.id)
    assert "Recorded rake does not match the derived ledger." in result.issues
    assert result.is_authoritative is False


# ---------------------------------------------------------------------------
# Finding 4 -- an attestation covered the fact of a declaration, never its size
# ---------------------------------------------------------------------------


def test_raising_an_acknowledged_dead_money_amount_re_raises_the_disclosure(
    tmp_path: Path,
) -> None:
    """0.5 chips of unobserved money attested to is not 4980 attested to.

    AMENDED in round 12 for the same reason as the rake case above: the quantity
    is carried by the measured dependence code, which is what an attestation is
    bound to, and the declaration itself is no longer a pipeline warning.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=0.5))
    persist_reconciliation(db, hand.id)
    attested = attest_declared_assumptions(db, hand.id, only="dead_money")
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert parse_completion_evidence(stored.completion_evidence).unresolved_codes == ()

    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=4980.0))
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    assert DECLARED_DEAD_MONEY_CODE in evidence.declared_settlement_codes
    result = reconcile_persisted_hand(db, hand.id)
    measured = {item.code for item in result.assumption_dependence}
    assert not measured & set(attested)
    assert (
        evaluate_study_readiness(
            stored, accounting=result, user_confirmed=True
        ).is_ready
        is False
    )


def test_re_saving_the_same_dead_money_keeps_its_acknowledgement(
    tmp_path: Path,
) -> None:
    """Re-raising on every write would make the disclosure impossible to clear.

    AMENDED in round 12: the answer to a declared input is the attestation to its
    measured chip movement, so that is what must survive an idempotent re-save.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=0.5))
    persist_reconciliation(db, hand.id)
    attested = attest_declared_assumptions(db, hand.id)
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=0.5))

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    assert DECLARED_DEAD_MONEY_CODE in evidence.declared_settlement_codes
    assert evidence.unresolved_codes == ()
    assert set(attested) <= set(evidence.confirmed_assumption_codes)
    result = reconcile_persisted_hand(db, hand.id)
    assert (
        evaluate_study_readiness(
            stored, accounting=result, user_confirmed=True
        ).has("ACCOUNTING_ASSUMPTION_DEPENDENT")
        is False
    )


# ---------------------------------------------------------------------------
# Finding 5 -- the documented busy timeout was not honoured by one pragma
# ---------------------------------------------------------------------------


def test_a_concurrent_writer_does_not_kill_a_second_opener(tmp_path: Path) -> None:
    """``PRAGMA journal_mode = WAL`` never runs SQLite's busy handler.

    A rollback-journal file is exactly what a brand-new install and a restored
    pinned snapshot are, and the product opens the same database from the app,
    the CV worker and the solver worker. The second opener must wait out the
    writer rather than die with a raw "database is locked".
    """
    path = tmp_path / "busy.db"
    first = PokerDatabase(str(path))
    first.init_db()
    first.close()
    # Back to the journal mode a restored pinned snapshot arrives in.
    plain = sqlite3.connect(path)
    plain.execute("PRAGMA journal_mode = DELETE")
    plain.commit()
    plain.close()

    holder = sqlite3.connect(path)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        "INSERT INTO sessions (name, date_played, created_at) VALUES ('x', '2026-01-01', '2026-01-01T00:00:00')"
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            PokerDatabase(str(path), busy_timeout_ms=250)
    finally:
        holder.rollback()
        holder.close()
    assert "another process" in str(caught.value).lower()

    # And once the writer is gone the same open succeeds.
    second = PokerDatabase(str(path), busy_timeout_ms=250)
    second.init_db()
    second.close()


def test_concurrent_openers_all_start_against_one_database(tmp_path: Path) -> None:
    """Three processes opening a fresh file together must all come up."""
    path = tmp_path / "race.db"
    child = tmp_path / "child.py"
    child.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from poker_tracker.persistence.db import PokerDatabase\n"
        "db = PokerDatabase(sys.argv[1])\n"
        "db.init_db()\n"
        "print(db.schema_version())\n"
        "db.close()\n",
        encoding="utf-8",
    )
    processes = [
        subprocess.Popen(
            [sys.executable, str(child), str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    outcomes = [process.communicate() for process in processes]
    for (out, err), process in zip(outcomes, processes, strict=True):
        assert process.returncode == 0, err
        assert out.strip() == str(SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Finding 6 -- the read-only backup audit wrote sidecars beside the backups
# ---------------------------------------------------------------------------


def _wal_snapshot(backup_dir: Path, live: Path, name: str) -> Path:
    """A snapshot in the journal mode every pre-Phase-1 build wrote."""
    destination = backup_dir / name
    shutil.copyfile(live, destination)
    connection = sqlite3.connect(destination)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.commit()
    connection.close()
    for suffix in ("-shm", "-wal"):
        Path(f"{destination}{suffix}").unlink(missing_ok=True)
    return destination


def test_auditing_a_wal_mode_backup_writes_nothing_beside_it(tmp_path: Path) -> None:
    """SQLite must create a -shm sidecar to read a WAL database, even read-only.

    The audit is documented as read-only, and orphaned sidecars in the operator's
    backup directory are a data-health finding in their own right.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    live = data_dir / "live.db"
    db = PokerDatabase(str(live))
    db.init_db()
    db.close()
    backup_dir = data_dir / "backups"
    backup_dir.mkdir()
    _wal_snapshot(backup_dir, live, "poker_tracker_20260101T000000000000Z.sqlite3")
    before = sorted(path.name for path in backup_dir.iterdir())

    audit_data_health(database_path=live, data_dir=data_dir, backup_dir=backup_dir)

    assert sorted(path.name for path in backup_dir.iterdir()) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_intact_wal_backup_on_a_read_only_mount_still_passes(
    tmp_path: Path,
) -> None:
    """An archival mount must not turn an intact backup into a failed one."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    live = data_dir / "live.db"
    db = PokerDatabase(str(live))
    db.init_db()
    db.close()
    backup_dir = data_dir / "backups"
    backup_dir.mkdir()
    _wal_snapshot(backup_dir, live, "poker_tracker_20260101T000000000000Z.sqlite3")
    backup_database(live, backup_dir)

    os.chmod(backup_dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        report = audit_data_health(
            database_path=live, data_dir=data_dir, backup_dir=backup_dir
        )
    finally:
        os.chmod(backup_dir, stat.S_IRWXU)
    backups = next(check for check in report.checks if check.name == "backups")
    assert backups.status == "pass", backups.details


# ---------------------------------------------------------------------------
# Finding 7 -- removing a hand left session coaching claiming to describe it
# ---------------------------------------------------------------------------


def test_deleting_a_hand_stales_the_session_coaching_that_summarised_it(
    tmp_path: Path,
) -> None:
    """Moving a hand out of a session stales it; deleting one must too."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    session_id = hand.session_id
    db.create_coaching_response(
        CoachingResponse(
            hand_id=None,
            session_id=session_id,
            review_type="session",
            provider_name="test",
            model_name="test",
            raw_prompt="p",
            raw_response="You lost 5bb net",
        )
    )
    assert [
        review.is_stale for review in db.fetch_coaching_reviews_by_session(session_id)
    ] == [False]

    db.delete_hand(hand.id)

    reviews = db.fetch_coaching_reviews_by_session(session_id)
    assert [review.is_stale for review in reviews] == [True]
    assert reviews[0].stale_reason


# ---------------------------------------------------------------------------
# Finding 8 -- one unreadable JSON column still hid a whole session
# ---------------------------------------------------------------------------


def test_an_unreadable_tags_column_does_not_hide_the_rest_of_the_session(
    tmp_path: Path,
) -> None:
    """``completion_evidence`` degrades; ``tags`` raised out of the same fetch."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db._execute("UPDATE hands SET tags = 'not json' WHERE id = ?", (hand.id,))
    db._commit()

    hands = db.fetch_hands_by_session(hand.session_id)
    assert [item.id for item in hands] == [hand.id]
    assert hands[0].tags == []


def test_an_unreadable_settlement_warnings_column_does_not_raise_into_the_study_page(
    tmp_path: Path,
) -> None:
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    db._execute(
        "UPDATE hand_settlements SET warnings = 'oops' WHERE hand_id = ?", (hand.id,)
    )
    db._commit()

    settlement = db.fetch_hand_settlement(hand.id)
    assert settlement is not None
    assert settlement.warnings == []
    reconcile_persisted_hand(db, hand.id)


# ---------------------------------------------------------------------------
# Finding 9 -- adding a seat invalidates the settlement, and nothing pinned it
# ---------------------------------------------------------------------------


def test_adding_a_seat_invalidates_the_settlement_and_stales_retained_analysis(
    tmp_path: Path,
) -> None:
    """``create_hand_player`` does both; only the review_status demotion was pinned."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", is_balanced=True)
    )
    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            session_id=hand.session_id,
            review_type="hand",
            provider_name="test",
            model_name="test",
            raw_prompt="p",
            raw_response="two-handed pot",
        )
    )
    db.create_hand_review(HandReview(
            hand_id=hand.id,
            hand_summary="two-handed pot",
            theory_coach="t",
            exploit_coach="e",
            study_lesson="l",
        ))
    for review in db.fetch_coaching_reviews_by_hand(hand.id):
        db._execute(
            "UPDATE coaching_reviews SET is_stale = 0, stale_reason = '' WHERE id = ?",
            (review.id,),
        )
    for review in db.fetch_reviews_by_hand(hand.id):
        db._execute(
            "UPDATE hand_reviews SET is_stale = 0, stale_reason = '' WHERE id = ?",
            (review.id,),
        )
    db._commit()

    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="third",
            player_name="Third",
            is_hero=False,
            starting_stack=1000,
        )
    )

    settlement = db.fetch_hand_settlement(hand.id)
    assert settlement is not None
    assert settlement.status != "reconciled"
    assert settlement.is_balanced is False
    assert all(
        review.is_stale for review in db.fetch_coaching_reviews_by_hand(hand.id)
    )
    assert all(review.is_stale for review in db.fetch_reviews_by_hand(hand.id))


# ---------------------------------------------------------------------------
# Finding 10 -- the version discriminator refused a schema ahead of its stamp
# but not a stamp ahead of its schema
# ---------------------------------------------------------------------------


_PRE_V13_SCHEMA = """
CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT);
INSERT INTO schema_metadata VALUES ('schema_version','13');
CREATE TABLE sessions(
    id INTEGER PRIMARY KEY, name TEXT, date_played TEXT, created_at TEXT);
CREATE TABLE hands(
    id INTEGER PRIMARY KEY, session_id INTEGER, hand_number INTEGER,
    game_type TEXT DEFAULT 'cash', blinds_antes TEXT DEFAULT '',
    table_size INTEGER, effective_stack REAL, hero_position TEXT DEFAULT '',
    hero_cards TEXT DEFAULT '', board_cards TEXT DEFAULT '', pot_size REAL,
    result TEXT DEFAULT '', hero_bb_won REAL,
    review_status TEXT DEFAULT 'reviewed', confidence_score REAL,
    source_type TEXT DEFAULT 'cv_import', tags TEXT DEFAULT '[]',
    notes TEXT DEFAULT '', created_at TEXT DEFAULT '2026-01-01T00:00:00');
INSERT INTO sessions (id,name,date_played,created_at)
    VALUES (1,'s','2026-01-01','2026-01-01T00:00:00');
INSERT INTO hands (id,session_id,hand_number) VALUES (1,1,1);
"""


def test_a_stamp_ahead_of_its_own_schema_is_refused(tmp_path: Path) -> None:
    """The mirror of the schema-ahead-of-stamp refusal, which was already made.

    A v13 stamp over a pre-v13 ``hands`` table used to open: ``init_db`` saw the
    current version, skipped the migration chain AND the pre-migration snapshot,
    and every write then died with a bare ``sqlite3.OperationalError`` about a
    missing column -- on a database that was never backed up and whose operator
    was never warned.
    """
    path = tmp_path / "ahead.db"
    seeded = sqlite3.connect(path)
    seeded.executescript(_PRE_V13_SCHEMA)
    seeded.commit()
    seeded.close()

    db = PokerDatabase(str(path))
    with pytest.raises(RuntimeError) as caught:
        db.init_db()
    assert "restore the database from a backup" in str(caught.value).lower()
    db.close()

    # And nothing was written to it: the refused file keeps its journal mode.
    plain = sqlite3.connect(path)
    assert plain.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    plain.close()


def test_a_genuine_pre_v13_database_still_migrates(tmp_path: Path) -> None:
    """The refusal must key on the stamp, not on the missing column alone."""
    path = tmp_path / "v12.db"
    seeded = sqlite3.connect(path)
    seeded.executescript(_PRE_V13_SCHEMA.replace("'13'", "'12'"))
    seeded.commit()
    seeded.close()

    db = PokerDatabase(str(path), busy_timeout_ms=2000)
    db.init_db()
    assert db.schema_version() == SCHEMA_VERSION
    migrated = db.fetch_hand(1)
    assert migrated is not None
    assert migrated.completion_status == "uncertain"
    db.close()


def test_a_read_only_mount_is_named_instead_of_reported_as_a_lock(
    tmp_path: Path,
) -> None:
    """The container scenario: SQLite's own wording points at the wrong thing.

    ``attempt to write a readonly database`` reads as "your database is broken".
    The database is fine; the mount is not writable, and SQLite needs the whole
    directory because it writes journal files beside the file.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    path = tmp_path / "ro.db"
    db = PokerDatabase(str(path))
    db.init_db()
    db.close()
    plain = sqlite3.connect(path)
    plain.execute("PRAGMA journal_mode = DELETE")
    plain.commit()
    plain.close()

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(RuntimeError) as caught:
            PokerDatabase(str(path), busy_timeout_ms=100)
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)
    assert "read-only filesystem" in str(caught.value)
    assert "intact and unchanged" in str(caught.value)
