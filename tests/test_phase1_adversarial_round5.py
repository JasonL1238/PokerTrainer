"""Regressions for the round-5 adversarial findings against Phase 1.

Every test here failed before its fix. The round-5 theme is *derived truth that
was still taken from an input*: a reconciliation tolerance computed from two
operator- and payload-supplied settlement fields and then applied to pre-rake
quantities, an acknowledgement that travelled inside an import payload, a
read-time card annotation that got persisted and became permanent, a migration
that replayed itself against a live database because the stamp row was gone, and
an evidence write that weakened a hand without returning it to needs_correction.
"""

from __future__ import annotations

import copy
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    UNREADABLE_CARDS_KEY,
    CompletionEvidence,
    acknowledge_codes,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
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
from poker_tracker.services.study_readiness import (
    evaluate_study_readiness,
    is_reconstructed_hand,
)


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


def _open_db(tmp_path: Path, name: str = "round5.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed_hand(
    db: PokerDatabase,
    *,
    pot_size: float | None = 30.0,
    hero_bb_won: float | None = 20.0,
    award: float = 20.0,
    street: str = "river",
    board: str = "Qd 7s 2c",
    bet: float = 10.0,
) -> Hand:
    """A two-player hand whose only action line is bet/call, so the pot is exact.

    With ``bet=10`` the derived gross pot is 20 and the hero's derived net result
    is +10, whatever rake policy is stored on top.
    """
    session = db.create_session(Session(name="Round 5", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards=board,
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
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
                street=street,
                action_index=index,
                player_name=name,
                action_type=kind,
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
                player_key="hero",
                player_name="Hero",
                amount=award,
                entry_order=1,
            )
        ],
    )
    return hand


# ---------------------------------------------------------------------------
# Finding 1 -- the tolerance was the product of two settlement fields, and it
# was applied to quantities computed before any rake was taken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rake_rate", "chip_unit"),
    [
        (1.0, 100000.0),  # 'Rake %' = 100, 'Chip unit' unbounded
        (1.0, 1000.0),
        (0.5, 1000.0),
        (1.0, 20.0),
        (0.1, 20.0),  # an entirely plausible room policy
    ],
)
def test_rake_rate_and_chip_unit_together_cannot_widen_the_pot_check(
    tmp_path: Path, rake_rate: float, chip_unit: float
) -> None:
    """Round-4 finding 1 was bounded, not fixed: `min(unit, gross * rate) / 2`.

    Both inputs are operator-typed settlement fields that an import payload also
    supplies verbatim, so their product reached ``gross_pot / 2`` -- and it was
    applied to the gross-pot and observed-pot comparisons, which are computed
    BEFORE any rake and therefore cannot carry a rake-rounding error at all.
    A recorded pot 50% larger than the derived one reconciled and rendered
    study-ready with zero blockers.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=30.0, hero_bb_won=20.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id, rake_rate=rake_rate, rake_rounding_unit=chip_unit
        )
    )
    result = persist_reconciliation(db, hand.id)

    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"
    assert result.is_authoritative is False
    assert "Observed final pot does not match the derived gross pot." in result.issues

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")


def test_a_waived_rake_buys_no_reconciliation_slack(tmp_path: Path) -> None:
    """`no_flop_no_drop` charges zero rake, so it must grant zero slack.

    `_reconciliation_tolerance` documented exactly this -- "a zero-rake policy
    therefore gets no slack at all" -- while computing the slack from a
    hypothetical ``gross_pot * rake_rate`` that never consulted the waiver. A
    stock 5% / whole-chip / no-flop-no-drop policy granted 0.5 of slack on a hand
    the policy definitionally rakes at 0.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.5, hero_bb_won=10.5, street="preflop", board="")
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            rake_rate=0.05,
            rake_rounding_unit=1.0,
            no_flop_no_drop=True,
        )
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.rake == pytest.approx(0.0)
    assert result.is_authoritative is False
    assert "Observed final pot does not match the derived gross pot." in result.issues
    assert (
        "Observed Hero result does not match the derived ledger result." in result.issues
    )


def test_an_imported_settlement_cannot_set_its_own_tolerance_with_two_fields(
    tmp_path: Path,
) -> None:
    """Round-4 finding 2's shape, through the product of two payload fields.

    One `import_session` call landed a hand whose recorded pot was 50% larger
    than its own action line, presenting as reconciled, authoritative and
    study-ready with an empty blocker tuple.
    """
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, pot_size=30.0, hero_bb_won=20.0)
    assert hand.id is not None
    payload = export_session(source, hand.session_id)
    source.close()
    payload["hands"][0]["settlement"] = {
        "hand_id": 0,
        "rake_rate": 1.0,
        "rake_rounding_unit": 1000000.0,
        "status": "reconciled",
        "is_balanced": True,
    }

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    result = persist_reconciliation(target, imported.id)
    stored = target.fetch_hand(imported.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)

    assert result.is_authoritative is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")
    assert readiness.is_ready is False


def test_an_honest_whole_chip_rake_still_absorbs_its_own_rounding(
    tmp_path: Path,
) -> None:
    """An honest rake policy still reconciles, and no recorded figure is excused.

    Rounds 4-6 granted the recorded ``rake_amount`` and ``net_pot`` a tolerance
    derived from the settlement's own fields. Round 7 removed it: every input it
    was derived from also arrives in an import payload, and ``import_session``
    never rewrites the recorded pair, so the payload set both sides of its own
    comparison. What this test still protects is the original intent -- rake
    rounding itself must not become a mismatch -- verified through the path the
    product actually writes: the settlement editor nulls the recorded figures and
    ``persist_reconciliation`` re-derives them.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=9.0, award=19.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            rake_rate=0.05,
            rake_rounding_unit=1.0,
            rake_amount=1.3,
        )
    )
    # The read-only path every readiness surface uses.
    result = reconcile_persisted_hand(db, hand.id)
    assert result.ledger.rake == pytest.approx(1.0)
    assert "Recorded rake does not match the derived ledger." in result.issues
    assert result.is_authoritative is False

    # And the product refuses to call it reconciled...
    result = persist_reconciliation(db, hand.id)
    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"

    # ...while the honest policy itself still reconciles once the recorded
    # figure is the product's own, re-derived from the ledger.
    result = persist_reconciliation(db, hand.id)
    assert result.ledger.rake == pytest.approx(1.0)
    assert result.settlement is not None
    assert result.settlement.rake_amount == pytest.approx(1.0)
    assert result.settlement.status == "reconciled"
    assert result.is_authoritative is True


# ---------------------------------------------------------------------------
# Finding 2 -- an acknowledgement is an operator act, and cannot travel in JSON
# ---------------------------------------------------------------------------


def _forged_payload(base: dict, **evidence_overrides: object) -> dict:
    payload = copy.deepcopy(base)
    payload["hands"][0]["hand"]["completion_evidence"] = _clean_evidence(
        **evidence_overrides
    )
    return payload


def test_an_import_payload_cannot_pre_acknowledge_its_own_warnings(
    tmp_path: Path,
) -> None:
    """A payload that supplies `acknowledged_codes` used to be believed.

    `derive_completion_status` consumes `unresolved_codes`, so a forger who wrote
    a CONSISTENT blob -- the same warning in `warning_codes` and in
    `acknowledged_codes` -- promoted the hand to 'complete' with an empty blocker
    tuple, having attested to nothing in the importing operator's database. It
    also silenced a genuine pipeline warning: `hero_seat_mismatch` arriving
    pre-acknowledged dropped out of `unresolved_codes`, so UNSUPPORTED_TABLE_LAYOUT
    never fired and app.py drew no Acknowledge control for it.
    """
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    base = export_session(source, hand.session_id)
    source.close()

    payload = _forged_payload(
        base,
        warning_codes=["hero_seat_mismatch", "declared_unobserved_chips"],
        acknowledged_codes=["hero_seat_mismatch", "declared_unobserved_chips"],
    )
    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    evidence = parse_completion_evidence(imported.completion_evidence)

    assert evidence.acknowledged_codes == ()
    # AMENDED in round 12: `declared_unobserved_chips` is not a pipeline warning
    # at all any more -- it is the operator's own settlement declaration, and it
    # has its own channel, so a payload listing it in `warning_codes` or
    # `acknowledged_codes` gets it relocated rather than honoured. The claim under
    # test is unchanged for the code this test is really about: a payload cannot
    # pre-acknowledge a genuine pipeline warning.
    assert set(evidence.unresolved_codes) == {"hero_seat_mismatch"}
    assert evidence.declared_settlement_codes == ("declared_unobserved_chips",)
    assert "declared_unobserved_chips" not in evidence.warning_codes
    assert imported.completion_status == "uncertain"

    result = reconcile_persisted_hand(target, imported.id)
    readiness = evaluate_study_readiness(
        imported, accounting=result, user_confirmed=True
    )
    assert readiness.has("UNRESOLVED_SOURCE_WARNING")
    assert readiness.has("UNSUPPORTED_TABLE_LAYOUT")
    assert readiness.is_ready is False


# ---------------------------------------------------------------------------
# Finding 3 -- a read-time card annotation must never become stored evidence
# ---------------------------------------------------------------------------


def test_correcting_an_unreadable_card_column_clears_the_blocker_after_import(
    tmp_path: Path,
) -> None:
    """`unreadable_card_columns` is derived by `_hand_from_row`, never evidence.

    It used to be persisted by every writer that round-tripped a fetched hand's
    evidence -- the Acknowledge button, and `import_session` -- so after one
    export/import the hand carried a permanent INVALID_HERO_OR_BOARD_CARDS whose
    stated clearing action ("fix the hero and board cards") did nothing, because
    `_unreadable_card_columns` reads the STORED evidence first and no writer ever
    removed the key.
    """
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    source._execute("UPDATE hands SET board_cards = 'Qd 7s' WHERE id = ?", (hand.id,))
    source._commit()
    degraded = source.fetch_hand(hand.id)
    assert degraded is not None
    assert UNREADABLE_CARDS_KEY in degraded.completion_evidence
    assert evaluate_study_readiness(
        degraded, accounting=None, user_confirmed=True
    ).has("INVALID_HERO_OR_BOARD_CARDS")
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    row = target._execute(
        "SELECT completion_evidence FROM hands WHERE id = ?", (imported.id,)
    ).fetchone()
    assert UNREADABLE_CARDS_KEY not in row["completion_evidence"]

    # And the marker still cannot be laundered in by an acknowledgement write.
    target._execute(
        "UPDATE hands SET board_cards = 'Qd 7s' WHERE id = ?", (imported.id,)
    )
    target._commit()
    reread = target.fetch_hand(imported.id)
    assert reread is not None
    assert UNREADABLE_CARDS_KEY in reread.completion_evidence
    target.update_hand_completion(
        imported.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(
                parse_completion_evidence(reread.completion_evidence), []
            )
        ),
    )
    stored_row = target._execute(
        "SELECT completion_evidence FROM hands WHERE id = ?", (imported.id,)
    ).fetchone()
    assert UNREADABLE_CARDS_KEY not in stored_row["completion_evidence"]

    # Now do exactly what the blocker's clearing action names.
    broken = target.fetch_hand(imported.id)
    assert broken is not None
    target.update_hand_facts(broken.model_copy(update={"board_cards": "Qd 7s 2c"}))
    fixed = target.fetch_hand(imported.id)
    assert fixed is not None
    assert fixed.board_cards == "Qd 7s 2c"
    assert UNREADABLE_CARDS_KEY not in fixed.completion_evidence
    assert not evaluate_study_readiness(
        fixed, accounting=None, user_confirmed=True
    ).has("INVALID_HERO_OR_BOARD_CARDS")


# ---------------------------------------------------------------------------
# Finding 4 -- a live schema-13 database with no stamp row must be refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM schema_metadata WHERE key = 'schema_version'",
        "DROP TABLE schema_metadata",
    ],
)
def test_a_live_v13_database_with_no_version_stamp_is_refused(
    tmp_path: Path, damage: str
) -> None:
    """`_readable_schema_version` conflated "fresh" with "stamp lost".

    Both returned 0, so the whole chain replayed and `_migrate_to_v13` -- whose
    own docstring says it must never run against a live database -- reset every
    reconstructed hand to uncertain/needs_correction, destroying every operator
    confirmation, silently. A garbage stamp was already refused; a missing one on
    a database full of user tables was not.
    """
    path = tmp_path / "live.db"
    db = _open_db(tmp_path, "live.db")
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    db._execute("UPDATE hands SET source_type = 'corrected_cv' WHERE id = ?", (hand.id,))
    db._commit()
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    db.update_hand_status(hand.id, "reviewed")
    db.close()

    raw = sqlite3.connect(str(path))
    raw.execute(damage)
    raw.commit()
    raw.close()

    reopened = PokerDatabase(str(path))
    with pytest.raises(RuntimeError, match="Restore the database from a backup"):
        reopened.init_db()
    reopened.close()

    survivor = sqlite3.connect(str(path))
    survivor.row_factory = sqlite3.Row
    row = survivor.execute(
        "SELECT review_status, completion_status FROM hands WHERE id = ?", (hand.id,)
    ).fetchone()
    survivor.close()
    assert row["review_status"] == "reviewed"
    assert row["completion_status"] == "complete"


def test_a_genuine_pre_versioning_database_still_migrates(tmp_path: Path) -> None:
    """The refusal above must not catch a legacy database that never had a stamp.

    A pre-versioning file has user tables and no ``schema_metadata`` row, which is
    exactly the shape the new guard refuses -- unless it discriminates on what the
    physical schema actually contains rather than on the stamp's absence.
    """
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(path))
    raw.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date_played TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE hands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            hand_number INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    raw.commit()
    raw.close()

    db = PokerDatabase(str(path))
    db.init_db()
    assert db.schema_version() == 13
    db.close()


# ---------------------------------------------------------------------------
# Finding 5 -- 'reviewed' must not outlive the evidence it was granted on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weakened",
    [
        {"warning_codes": ["hero_seat_mismatch"], "rejection_codes": ["ocr_refused_board"]},
        {"rejection_codes": ["board_unreadable"]},
    ],
)
def test_an_evidence_write_that_weakens_completion_demotes_reviewed(
    tmp_path: Path, weakened: dict
) -> None:
    """`update_hand_completion` re-derived the column and left review_status alone.

    Every other writer that invalidates a hand routes through
    `_demote_reviewed_hand`. This one wrote a hand to completion_status
    'uncertain' -- with a pipeline REJECTION in its own evidence -- while it was
    still labelled 'reviewed' and still counted toward the landing hero's
    "N% marked reviewed". `update_hand_status` refuses to create that pair.

    Since round 14 the only weakening a caller blob can land is a code
    ADDITION -- the writer records no observations -- so the demotion is
    exercised through added warning/rejection codes, which is also the only
    weakening the product itself can produce through this door.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    db.update_hand_status(hand.id, "reviewed")
    promoted = db.fetch_hand(hand.id)
    assert promoted is not None
    assert (promoted.completion_status, promoted.review_status) == ("complete", "reviewed")

    db.update_hand_completion(hand.id, completion_evidence=_clean_evidence(**weakened))
    after = db.fetch_hand(hand.id)
    assert after is not None
    assert after.completion_status != "complete"
    assert after.review_status == "needs_correction"


@pytest.mark.parametrize(
    "blob",
    [
        {},
        {"terminal_event": "unobserved"},
    ],
)
def test_an_observation_weakening_blob_lands_nothing_at_all(
    tmp_path: Path, blob: dict
) -> None:
    """Round 14 strengthened the round-5 repair: a blob claiming weaker
    OBSERVATIONS (or none) is not honoured-and-demoted, it is ignored.

    'Reviewed never outlives the evidence it was granted on' still holds,
    because the evidence it was granted on is untouched: the stored blob, the
    completion status, and the review status are all byte-identical after the
    write. The pre-round-14 behaviour -- honouring the weaker blob and
    demoting -- let a caller blob rewrite the pipeline's observations in the
    OTHER direction too, which round 14 demonstrated as a promotion of an
    unprovable hand.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    db.update_hand_status(hand.id, "reviewed")
    before = db.fetch_hand(hand.id)
    assert before is not None
    assert (before.completion_status, before.review_status) == ("complete", "reviewed")

    payload = {} if not blob else _clean_evidence(**blob)
    db.update_hand_completion(hand.id, completion_evidence=payload)
    after = db.fetch_hand(hand.id)
    assert after is not None
    assert after.completion_status == "complete"
    assert after.review_status == "reviewed"
    assert after.completion_evidence == before.completion_evidence


def test_acknowledging_a_warning_does_not_demote_a_reviewed_hand(
    tmp_path: Path,
) -> None:
    """The negative direction: the app's only `update_hand_completion` call site.

    Acknowledging a warning on a hand that stays 'complete' is not an
    invalidation and must not cost the operator their confirmation.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    db.update_hand_completion(
        hand.id, completion_evidence=_clean_evidence(warning_codes=["board_zone_yield_zero"])
    )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(evidence, list(evidence.unresolved_codes))
        ),
    )
    promoted = db.fetch_hand(hand.id)
    assert promoted is not None
    assert promoted.completion_status == "complete"
    db.update_hand_status(hand.id, "reviewed")

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(
                parse_completion_evidence(stored.completion_evidence), []
            )
        ),
    )
    after = db.fetch_hand(hand.id)
    assert after is not None
    assert after.completion_status == "complete"
    assert after.review_status == "reviewed"


# ---------------------------------------------------------------------------
# Finding 6 -- clearing the Hero flag deleted the hero-result cross-check
# ---------------------------------------------------------------------------


def test_a_recorded_hero_result_with_no_hero_seat_still_blocks(
    tmp_path: Path,
) -> None:
    """`elif len(hero_players) == 1` skipped the check entirely at zero heroes.

    Unticking 'Hero' in the player editor is an ordinary correction, and it made
    a hand recording `hero_bb_won = 999` against a derived +10 reconcile with no
    issues and accept a promotion to 'reviewed' -- while every list view kept
    rendering and ranking on the fabricated 999.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=999.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    blocked = persist_reconciliation(db, hand.id)
    assert blocked.is_authoritative is False

    hero = next(p for p in db.fetch_players_by_hand(hand.id) if p.is_hero)
    db.update_hand_player(hero.model_copy(update={"is_hero": False}))
    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is False
    assert any("Hero" in issue for issue in result.issues)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")


# ---------------------------------------------------------------------------
# Finding 7 -- settlement entries were dropped when the settlement row was null
# ---------------------------------------------------------------------------


def test_awards_survive_a_round_trip_without_a_settlement_row(
    tmp_path: Path,
) -> None:
    """`db.create_settlement_entry` needs no settlement row, and the exporter
    emits the entries regardless -- but `import_session` nested the entry write
    inside ``if settlement is not None``, so a settled, balanced hand silently
    arrived with no declared winner and no reported count.
    """
    source = _open_db(tmp_path, "src.db")
    hand = _seed_hand(source, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    assert source.fetch_hand_settlement(hand.id) is None
    before = reconcile_persisted_hand(source, hand.id)
    assert before.ledger.is_settled is True
    payload = export_session(source, hand.session_id)
    source.close()
    assert payload["hands"][0]["settlement"] is None
    assert len(payload["hands"][0]["settlement_entries"]) == 1

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    entries = target.fetch_settlement_entries(imported.id)
    assert [(e.entry_type, e.player_key, e.amount) for e in entries] == [
        ("award", "hero", 20.0)
    ]
    after = reconcile_persisted_hand(target, imported.id)
    assert after.ledger.is_settled is True
    assert after.ledger.payouts == before.ledger.payouts


# ---------------------------------------------------------------------------
# Finding 8 -- a blocker must not name a panel the product never draws
# ---------------------------------------------------------------------------


def test_a_migrated_hand_is_not_told_to_use_a_panel_that_never_renders(
    tmp_path: Path,
) -> None:
    """The v13 migration leaves `completion_evidence` at '{}' for every row.

    That evidence carries no codes, so app.py returns before drawing the Source
    warnings panel -- yet COMPLETION_NOT_COMPLETE told the operator to
    "acknowledge each remaining source warning in the Source warnings panel".
    Following it left the hand exactly where it was, plus one new blocker.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=10.0)
    assert hand.id is not None
    db._execute(
        "UPDATE hands SET completion_status = 'uncertain', "
        "review_status = 'needs_correction', completion_evidence = '{}' WHERE id = ?",
        (hand.id,),
    )
    db._commit()
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    result = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    assert evidence.is_known is False
    assert evidence.warning_codes == ()

    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    blocker = next(b for b in readiness.blockers if b.code == "COMPLETION_NOT_COMPLETE")
    assert "Source warnings panel" not in blocker.clearing_action
    assert "Correct hand facts" not in blocker.clearing_action
    assert blocker.clearing_action.startswith("Only a new reconstruction clears this.")

    # And following the two actions it used to name still leaves the hand blocked,
    # which is why naming them was the defect.
    db.update_hand_facts(stored.model_copy(update={"table_size": 9}))
    corrected = db.fetch_hand(hand.id)
    assert corrected is not None
    corrected_evidence = parse_completion_evidence(corrected.completion_evidence)
    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(
                corrected_evidence, list(corrected_evidence.unresolved_codes)
            )
        ),
    )
    after = db.fetch_hand(hand.id)
    assert after is not None
    assert after.completion_status == "uncertain"
    assert evaluate_study_readiness(
        after, accounting=None, user_confirmed=True
    ).has("COMPLETION_EVIDENCE_MISSING")


# ---------------------------------------------------------------------------
# Surviving mutants -- live mechanisms no test pinned
# ---------------------------------------------------------------------------


def test_derive_completion_status_refuses_an_unreadable_evidence_version() -> None:
    """The `is_known` gate inside `derive_completion_status` was unprotected.

    A blob declaring `evidence_version` 0, 2 or 99 with otherwise complete-looking
    fields must derive 'uncertain'. Deleting the gate passed the whole suite, so
    the entire import ceiling for future-version payloads rested on nothing.
    """
    for version in (0, 2, 99):
        evidence = parse_completion_evidence(
            _clean_evidence(evidence_version=version)
        )
        assert (
            derive_completion_status(evidence, source_type="cv_import") == "uncertain"
        ), version
    readable = parse_completion_evidence(_clean_evidence())
    assert derive_completion_status(readable, source_type="cv_import") == "complete"


def test_a_reconstructed_source_is_reconstructed_even_at_not_applicable() -> None:
    """Both halves of `is_reconstructed_hand`, in the direction no test covered.

    The round-3 test named for this varies only `completion_status`, so it proves
    that half twice. Dropping the `source_type` half passed the whole suite while
    silently exempting a cv_import hand from every completion, layout,
    source-warning and confirmation blocker.
    """
    reconstructed = Hand(
        session_id=1,
        hand_number=1,
        source_type="cv_import",
        completion_status="not_applicable",
    )
    assert is_reconstructed_hand(reconstructed) is True
    manual = Hand(
        session_id=1,
        hand_number=2,
        source_type="manual",
        completion_status="not_applicable",
    )
    assert is_reconstructed_hand(manual) is False


def test_an_unresolved_hero_seat_mismatch_blocks_the_layout(tmp_path: Path) -> None:
    """The `hero_seat_mismatch` branch of `_layout_blockers` was untested.

    With `layout_supported=True`, a valid `evidence.table_size` and a recorded
    `hand.table_size`, it is the only thing that raises UNSUPPORTED_TABLE_LAYOUT
    for a hand whose hero seat could not be attributed -- and deleting it passed
    the whole suite.
    """
    hand = Hand(
        session_id=1,
        hand_number=1,
        table_size=6,
        source_type="cv_import",
        completion_status="uncertain",
        completion_evidence=_clean_evidence(warning_codes=["hero_seat_mismatch"]),
    )
    readiness = evaluate_study_readiness(hand, accounting=None, user_confirmed=True)
    blocker = next(b for b in readiness.blockers if b.code == "UNSUPPORTED_TABLE_LAYOUT")
    assert blocker.detail == ("hero_seat_mismatch",)
