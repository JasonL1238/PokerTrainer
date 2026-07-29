"""Regressions for the round-2 adversarial findings against Phase 1.

Every test here failed before its fix. They are grouped by the contract they
defend, not by the module they touch, because most of the findings were seams
between two modules that each looked correct on its own.
"""

from __future__ import annotations

import ast
import hashlib
import sqlite3
from pathlib import Path

import pytest

import poker_tracker.persistence.db as db_module
from cv_lab.scripts.eval.validate_yolo_card_timeline import WARNING_SEVERITY
from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    SEVERITY,
    apply_hand_corrections,
    timeline_to_session_payload,
)
from poker_tracker.persistence.completion import (
    CompletionEvidence,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import (
    EXPORT_VERSION,
    export_session,
    import_hands_into_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
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

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _clean_evidence(**overrides: object) -> dict[str, object]:
    evidence = CompletionEvidence(
        evidence_version=1,
        partial_start=False,
        partial_end=False,
        terminal_event="showdown",
        boundary_confidence=0.95,
        layout_supported=True,
        table_size=2,
    )
    payload = dump_completion_evidence(evidence)
    payload.update(overrides)
    return payload


def _make_db(path: Path | str = ":memory:") -> PokerDatabase:
    db = PokerDatabase(str(path))
    db.init_db()
    return db


def _seed_reconciled_cv_hand(db: PokerDatabase, **hand_overrides: object) -> int:
    """One card-complete, fully reconciled cv_import hand whose only gap is confirmation.

    The declared pot award is attested to here, because on a reconstructed hand
    that award is an operator declaration like the rake: the CV exporter emits no
    settlement rows, so nothing observed who won, and the measured dependence
    blocks until it is confirmed. Attesting is what the operator does in the
    Accounting reconciliation panel, so the fixture does it too.
    """

    session = db.create_session(Session(name="Readiness"))
    fields: dict[str, object] = {
        "session_id": session.id,
        "hand_number": 1,
        "table_size": 2,
        "hero_cards": "Ah Qs",
        "board_cards": "Qd 7s 2c",
        "pot_size": 20,
        "hero_bb_won": 10,
        "source_type": "cv_import",
        "completion_status": "complete",
        "completion_evidence": _clean_evidence(),
    }
    fields.update(hand_overrides)
    hand = db.create_hand(Hand(**fields))
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            seat_index=0,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    villain = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="villain",
            seat_index=1,
            player_name="Villain",
            position="BB",
            starting_stack=100,
        )
    )
    for actor, action_type in ((hero, "bet"), (villain, "call")):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_type=action_type,
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
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=20,
                entry_order=1,
            )
        ],
    )
    persist_reconciliation(db, hand.id)
    attest_declared_assumptions(db, hand.id)
    return hand.id


def _readiness(db: PokerDatabase, hand_id: int, *, user_confirmed: bool = True):
    try:
        accounting = reconcile_persisted_hand(db, hand_id)
        error = None
    except Exception as exc:  # noqa: BLE001 - mirrors the app's own catch-all render path
        accounting = None
        error = str(exc)
    return evaluate_study_readiness(
        db.fetch_hand(hand_id),
        accounting=accounting,
        accounting_error=error,
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand_id),
        hand_reviews=db.fetch_reviews_by_hand(hand_id),
        solver_runs=db.fetch_solver_runs_by_hand(hand_id),
        user_confirmed=user_confirmed,
    )


# ---------------------------------------------------------------------------
# The rejection-code contract: a rejection clears only by new evidence
# ---------------------------------------------------------------------------


def _timeline_with_rejected_hand() -> dict[str, object]:
    base = {
        "hand_number": 2,
        "hero": ["Ah", "Kd"],
        "board": ["2c", "7d", "9s", "Jh", "3c"],
        "complete_cards": True,
        "terminal_event": "showdown",
        "warnings": ["side_pot_unsupported", "actions_collapsed_to_one_street"],
        "t_start": 10.0,
        "t_end": 60.0,
        "pot": 42.0,
        "reconciled": True,
        "winner_seat": 3,
        "n_states": 40,
        "anchor_missing_states": 0,
        "source_images": ["frames/a.jpg"],
        "actions": [
            {"street": "preflop", "seat": 1, "action_type": "raise", "amount": 3.0}
        ],
        "players": [
            {"seat": seat, "position": "BTN", "is_hero": seat == 2}
            for seat in range(1, 7)
        ],
    }
    return {
        "hands": [
            dict(base, hand_number=1),
            dict(base),
            dict(base, hand_number=3),
        ]
    }


def _exported_evidence(timeline: dict[str, object]) -> CompletionEvidence:
    payload = timeline_to_session_payload(
        timeline,
        timeline_path=Path("timeline.json"),
        session_name="Draft",
        allow_validation_warnings=True,
    )
    return parse_completion_evidence(payload["hands"][1]["hand"]["completion_evidence"])


def test_a_hand_corrections_row_cannot_erase_a_pipeline_rejection_code() -> None:
    """A CSV row is an operator note, not a new reconstruction."""

    timeline = _timeline_with_rejected_hand()
    before = _exported_evidence(timeline)
    assert set(before.rejection_codes) == {
        "side_pot_unsupported",
        "actions_collapsed_to_one_street",
    }

    corrected = apply_hand_corrections(
        timeline,
        {2: {"action": "keep", "hero_cards": "Ah Kd", "notes": "looked at the frames"}},
    )
    after = _exported_evidence(corrected)

    assert set(after.rejection_codes) == set(before.rejection_codes)
    assert "manual_hand_correction" in after.warning_codes
    # Acknowledging every acknowledgeable code still cannot promote a refused hand.
    from poker_tracker.persistence.completion import acknowledge_codes

    acknowledged = acknowledge_codes(after, list(after.warning_codes))
    assert derive_completion_status(acknowledged, source_type="cv_import") == "uncertain"


def test_every_validator_warning_code_is_known_to_the_exporter() -> None:
    """Two private severity tables disagreeing turned mild findings into rejections."""

    source = Path("cv_lab/scripts/eval/validate_yolo_card_timeline.py").read_text(
        encoding="utf-8"
    )
    emitted = {
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_warn"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert emitted, "the validator emits warning codes; the scan found none"
    assert not (emitted - set(SEVERITY))
    assert not (set(WARNING_SEVERITY) - set(SEVERITY))
    assert {
        code: WARNING_SEVERITY[code]
        for code in WARNING_SEVERITY
        if SEVERITY.get(code) != WARNING_SEVERITY[code]
    } == {}


def test_a_mild_validator_finding_stays_an_acknowledgeable_warning() -> None:
    """`missing_label` is the validator's mildest finding; it must not refuse a hand."""

    from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
        hand_to_import_payload,
    )

    hand = _timeline_with_rejected_hand()["hands"][1]
    clean = dict(hand, warnings=[])
    payload = hand_to_import_payload(
        clean,
        output_hand_number=1,
        timeline_path=Path("timeline.json"),
        validation_codes=["missing_label"],
        preceded_by_hand=True,
        followed_by_hand=True,
    )
    evidence = parse_completion_evidence(payload["hand"]["completion_evidence"])

    assert evidence.rejection_codes == ()
    assert "missing_label" in evidence.warning_codes


# ---------------------------------------------------------------------------
# Migration safety
# ---------------------------------------------------------------------------


def _seed_v12(path: Path, rows: list[tuple[str | None, str]]) -> None:
    original = db_module.SCHEMA_VERSION
    db_module.SCHEMA_VERSION = 12
    try:
        db = PokerDatabase(str(path))
        db.init_db()
        session = db.create_session(Session(name="Legacy"))
        db._execute("ALTER TABLE hands RENAME TO hands_old")
        db._execute(
            """
            CREATE TABLE hands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                hand_number INTEGER NOT NULL,
                game_type TEXT NOT NULL DEFAULT '',
                blinds_antes TEXT NOT NULL DEFAULT '',
                table_size INTEGER,
                effective_stack REAL,
                hero_position TEXT NOT NULL DEFAULT '',
                hero_cards TEXT NOT NULL DEFAULT '',
                board_cards TEXT NOT NULL DEFAULT '',
                pot_size REAL,
                result TEXT NOT NULL DEFAULT '',
                hero_bb_won REAL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                confidence_score REAL,
                source_type TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        db._execute("DROP TABLE hands_old")
        for number, (source_type, review_status) in enumerate(rows, start=1):
            db._execute(
                """
                INSERT INTO hands (
                    session_id, hand_number, review_status, source_type, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session.id, number, review_status, source_type, "2026-01-01T00:00:00"),
            )
        db._commit()
        db.close()
    finally:
        db_module.SCHEMA_VERSION = original


def test_the_v13_migration_classifies_a_row_whose_source_type_is_null(
    tmp_path: Path,
) -> None:
    """SQL three-valued logic made `source_type <> 'manual'` skip every NULL row."""

    path = tmp_path / "null_source.sqlite3"
    _seed_v12(path, [("manual", "reviewed"), (None, "reviewed"), ("cv_import", "reviewed")])

    db = _make_db(path)
    rows = {
        row["hand_number"]: (row["completion_status"], row["review_status"])
        for row in db._execute(
            "SELECT hand_number, completion_status, review_status FROM hands"
        ).fetchall()
    }
    db.close()

    assert rows[1] == ("not_applicable", "reviewed")  # manual is untouched
    assert rows[2] == ("uncertain", "needs_correction")  # NULL is not provably manual
    assert rows[3] == ("uncertain", "needs_correction")


def test_a_stale_pre_transaction_version_read_does_not_rerun_the_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes can both read the pre-migration version before either commits."""

    path = tmp_path / "race.sqlite3"
    _seed_v12(path, [("cv_import", "reviewed")])

    first = _make_db(path)
    hand_id = first.fetch_hands_by_session(1)[0].id
    first._execute("UPDATE hands SET review_status = 'reviewed' WHERE id = ?", (hand_id,))
    first._commit()
    first.close()

    second = PokerDatabase(str(path))
    calls = {"count": 0}
    real_reader = db_module._readable_schema_version

    def stale_unreserved_read(connection: object) -> int | None:
        calls["count"] += 1
        if calls["count"] == 1:
            # init_db's read before it holds the write reservation: what the
            # losing process saw while the winner was still migrating.
            return SCHEMA_VERSION - 1
        return real_reader(connection)

    monkeypatch.setattr(db_module, "_readable_schema_version", stale_unreserved_read)
    second.init_db()
    status = second.fetch_hand(hand_id).review_status
    second.close()

    assert calls["count"] >= 2, "the version must be re-read under the reservation"
    assert status == "reviewed"


def test_a_refused_newer_schema_file_is_not_written_to(tmp_path: Path) -> None:
    """A restored snapshot is journal_mode=delete; a refused open must not rewrite it."""

    path = tmp_path / "future.sqlite3"
    seed = sqlite3.connect(str(path))
    seed.execute("PRAGMA journal_mode=DELETE")
    seed.execute("CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    seed.execute("INSERT INTO schema_metadata VALUES ('schema_version', ?)", (str(SCHEMA_VERSION + 1),))
    seed.commit()
    seed.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    db = PokerDatabase(str(path))
    with pytest.raises(RuntimeError, match="newer than this app understands"):
        db.init_db()
    db.close()

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob("future.sqlite3-*"))


@pytest.mark.parametrize("stamp", ["13.0", "", "thirteen", "-1"])
def test_an_unreadable_schema_version_stamp_is_refused_instead_of_replayed(
    tmp_path: Path, stamp: str
) -> None:
    """A negative stamp silently replayed the v13 migration against live data."""

    path = tmp_path / f"stamp_{abs(hash(stamp))}.sqlite3"
    db = _make_db(path)
    session = db.create_session(Session(name="Live"))
    # Evidence is attached at creation, as the CV producer does:
    # update_hand_completion no longer records observations (round 14).
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    db.update_hand_status(hand.id, "reviewed")
    db._execute(
        "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'", (stamp,)
    )
    db._commit()
    db.close()

    reopened = PokerDatabase(str(path))
    with pytest.raises(RuntimeError, match="schema version"):
        reopened.init_db()
    reopened.close()

    verifier = sqlite3.connect(str(path))
    assert verifier.execute("SELECT review_status FROM hands").fetchone() == ("reviewed",)
    verifier.close()


# ---------------------------------------------------------------------------
# Read-path robustness: one corrupt row must not take down a session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corrupt", ["COMPLETE", "complete ", "done", 5])
def test_a_corrupt_completion_status_degrades_instead_of_raising(corrupt: object) -> None:
    """completion_evidence is fully defended; the status column was not."""

    db = _make_db()
    session = db.create_session(Session(name="Corrupt"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="cv_import")
    )
    db._execute(
        "UPDATE hands SET completion_status = ? WHERE id = ?", (corrupt, hand.id)
    )
    db._commit()

    assert db.fetch_hand(hand.id).completion_status == "uncertain"
    assert len(db.fetch_hands_by_session(session.id)) == 1
    db.close()


def test_a_row_with_no_readable_source_type_is_treated_as_reconstructed() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Corrupt source"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db._execute(
        "UPDATE hands SET source_type = 'legacy_ocr' WHERE id = ?", (hand.id,)
    )
    db._commit()

    loaded = db.fetch_hand(hand.id)
    assert loaded.source_type == "cv_import"
    assert loaded.completion_status == "uncertain"
    db.close()


# ---------------------------------------------------------------------------
# 'reviewed' must never outlive the evidence it was granted on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer", ["settlement_entries", "settlement", "player", "action"])
def test_breaking_a_promoted_hands_evidence_returns_it_to_needs_correction(
    writer: str,
) -> None:
    db = _make_db()
    hand_id = _seed_reconciled_cv_hand(db)
    assert _readiness(db, hand_id).is_ready is True
    db.update_hand_status(hand_id, "reviewed")

    if writer == "settlement_entries":
        db.replace_settlement_entries(hand_id, [])
    elif writer == "settlement":
        settlement = db.fetch_hand_settlement(hand_id)
        db.upsert_hand_settlement(settlement.model_copy(update={"dead_money": 500.0}))
    elif writer == "player":
        db.create_hand_player(
            HandPlayer(
                hand_id=hand_id,
                player_key="late",
                seat_index=2,
                player_name="Late",
                position="CO",
                starting_stack=100,
            )
        )
    else:
        db.create_action(
            Action(
                hand_id=hand_id,
                player_key="hero",
                player_name="Hero",
                position="BTN",
                street="river",
                action_type="bet",
                amount=5,
                amount_semantics="incremental",
            )
        )

    assert db.fetch_hand(hand_id).review_status == "needs_correction"
    db.close()


# ---------------------------------------------------------------------------
# Import is not a promotion path
# ---------------------------------------------------------------------------


def test_an_untampered_round_trip_does_not_land_reviewed_on_a_reconstructed_hand() -> None:
    """User confirmation is per-render and cannot travel in a payload."""

    source = _make_db()
    hand_id = _seed_reconciled_cv_hand(source)
    source.update_hand_status(hand_id, "reviewed")
    payload = export_session(source, source.fetch_hand(hand_id).session_id)
    source.close()

    target = _make_db()
    imported = import_session(target, payload)
    hand = target.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "complete"
    assert hand.review_status == "needs_correction"
    target.close()


def test_a_payload_cannot_relabel_a_reconstructed_hand_as_manual() -> None:
    """source_type was the one field the whole completion invariant trusted verbatim."""

    source = _make_db()
    session = source.create_session(Session(name="Relabel"))
    source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            source_type="cv_import",
            completion_evidence=_clean_evidence(),
        )
    )
    payload = export_session(source, session.id)
    source.close()

    payload["hands"][0]["hand"]["source_type"] = "manual"
    payload["hands"][0]["hand"]["review_status"] = "reviewed"
    payload["hands"][0]["hand"].pop("completion_status", None)

    target = _make_db()
    with pytest.raises(ValueError, match="source_type 'manual'"):
        import_session(target, payload)
    assert target.fetch_sessions() == []
    target.close()


def test_import_never_upgrades_a_declared_completion_status() -> None:
    """A self-consistent forged evidence blob upgraded a declared `partial` to `complete`."""

    payload = {
        "export_version": EXPORT_VERSION,
        "session": {
            "name": "Forged",
            "date_played": "2026-01-01",
            "platform": "x",
            "stakes": "1/2",
            "notes": "",
        },
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "hero_cards": "Ah Kd",
                    "board_cards": "2c 7d 9s",
                    "source_type": "cv_import",
                    "review_status": "reviewed",
                    "table_size": 6,
                    "completion_status": "partial",
                    "completion_evidence": _clean_evidence(),
                },
                "players": [],
                "actions": [],
            }
        ],
    }

    db = _make_db()
    session = import_session(db, payload)
    hand = db.fetch_hands_by_session(session.id)[0]

    assert hand.completion_status == "partial"
    assert hand.review_status == "needs_correction"
    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        db.update_hand_status(hand.id, "reviewed")
    db.close()


def test_import_hands_into_session_keeps_session_level_coaching() -> None:
    """The temporary session's cascade deleted every session-level coaching row."""

    source = _make_db()
    session = source.create_session(Session(name="Coached"))
    hand = source.create_hand(Hand(session_id=session.id, hand_number=1))
    for review_type, hand_id in (("hand", hand.id), ("session", None)):
        source.create_coaching_response(
            CoachingResponse(
                session_id=session.id,
                hand_id=hand_id,
                review_type=review_type,
                provider_name="p",
                model_name="m",
                raw_prompt="prompt",
                raw_response="response",
            )
        )
    payload = export_session(source, session.id)
    source.close()

    target = _make_db()
    destination = target.create_session(Session(name="Destination"))
    import_hands_into_session(target, payload, destination.id)
    stored = target.fetch_coaching_reviews_by_session(destination.id)

    assert [row.review_type for row in stored] == ["session"]
    target.close()


# ---------------------------------------------------------------------------
# Every blocker's stated clearing action must actually clear it
# ---------------------------------------------------------------------------


def test_correcting_a_clean_complete_hand_leaves_a_clearable_source_warning() -> None:
    """A correction demoted the column but left evidence that still derived `complete`."""

    db = _make_db()
    hand_id = _seed_reconciled_cv_hand(db)
    corrected = db.update_hand_facts(
        db.fetch_hand(hand_id).model_copy(update={"board_cards": "Qd 7s 3c"}),
        correction_notes="turn misread",
    )
    evidence = parse_completion_evidence(corrected.completion_evidence)

    assert corrected.completion_status == "uncertain"
    # The column and its own evidence must agree, and the operator must have a
    # code to acknowledge -- otherwise the Source warnings panel never renders.
    assert derive_completion_status(evidence, source_type=corrected.source_type) == "uncertain"
    assert evidence.unresolved_codes

    from poker_tracker.persistence.completion import acknowledge_codes

    acknowledged = acknowledge_codes(evidence, list(evidence.warning_codes))
    restored = db.update_hand_completion(
        hand_id,
        completion_evidence=dump_completion_evidence(acknowledged),
        notes="reviewed the corrected facts",
    )
    assert restored.completion_status == "complete"
    db.close()


def test_replaying_unchanged_evidence_cannot_restore_complete_after_a_correction() -> None:
    db = _make_db()
    hand_id = _seed_reconciled_cv_hand(db)
    corrected = db.update_hand_facts(
        db.fetch_hand(hand_id).model_copy(update={"board_cards": "Qd 7s 3c"}),
        correction_notes="turn misread",
    )

    replayed = db.update_hand_completion(
        hand_id, completion_evidence=corrected.completion_evidence, notes="replay"
    )

    assert replayed.completion_status == "uncertain"
    db.close()


def test_the_layout_blocker_states_the_only_action_that_actually_clears_it() -> None:
    """Its text used to promise Correct hand facts, which cannot write the evidence."""

    db = _make_db()
    hand_id = _seed_reconciled_cv_hand(
        db,
        table_size=None,
        completion_evidence=_clean_evidence(layout_supported=False, table_size=None),
    )
    blocker = next(
        item
        for item in _readiness(db, hand_id).blockers
        if item.code == "UNSUPPORTED_TABLE_LAYOUT"
    )

    db.update_hand_facts(
        db.fetch_hand(hand_id).model_copy(update={"table_size": 2}),
        correction_notes="confirmed heads-up layout",
    )
    # The action the old text prescribed provably does not clear it...
    assert _readiness(db, hand_id).has("UNSUPPORTED_TABLE_LAYOUT")
    assert "Correcting the table size by hand does not clear it" in blocker.clearing_action

    # ...an evidence write cannot stand in for a reconstruction either --
    # round 14 inverted update_hand_completion's merge, so a caller blob
    # records no observations and the blocker stays...
    db.update_hand_completion(
        hand_id,
        completion_evidence=_clean_evidence(layout_supported=True, table_size=2),
        notes="attempted in-place evidence rewrite",
    )
    assert _readiness(db, hand_id).has("UNSUPPORTED_TABLE_LAYOUT")

    # ...and the action the text does name -- a new reconstruction, which
    # lands as a NEW hand carrying its own evidence -- produces a hand with no
    # layout blocker beside the still-blocked original.
    rebuilt_id = _seed_reconciled_cv_hand(
        db,
        table_size=2,
        completion_evidence=_clean_evidence(layout_supported=True, table_size=2),
    )
    assert not _readiness(db, rebuilt_id).has("UNSUPPORTED_TABLE_LAYOUT")
    assert _readiness(db, hand_id).has("UNSUPPORTED_TABLE_LAYOUT")
    db.close()


# ---------------------------------------------------------------------------
# UI surfaces must not present a blocked hand as study-ready
# ---------------------------------------------------------------------------


def _seed_promoted_then_broken(path: Path) -> int:
    """A hand promoted to reviewed whose ledger is then invalidated by hand."""

    db = PokerDatabase(str(path))
    db.init_db()
    hand_id = _seed_reconciled_cv_hand(db)
    db.update_hand_status(hand_id, "reviewed")
    # Break the ledger the way a hand-edited database or an older build could,
    # without going through a writer that demotes the promotion.
    db._execute("DELETE FROM settlement_entries WHERE hand_id = ?", (hand_id,))
    db._execute(
        "UPDATE hands SET review_status = 'reviewed' WHERE id = ?", (hand_id,)
    )
    db._commit()
    db.close()
    return hand_id


def _run_page(path: Path, page: str, monkeypatch: pytest.MonkeyPatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    app = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30
    ).run()
    app.radio[0].set_value(page)
    app.run()
    assert not list(app.exception)
    return app


def test_study_never_offers_reviewed_for_a_stored_reviewed_but_blocked_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filter stripped 'reviewed', then a fallback re-appended the stored value."""

    import streamlit as st

    from poker_tracker.ui.navigation import Page

    path = tmp_path / "sticky.sqlite3"
    _seed_promoted_then_broken(path)

    app = _run_page(path, Page.STUDY, monkeypatch)
    widget = next(item for item in app.selectbox if item.label == "Review status")
    rendered = "\n".join(item.value for item in app.markdown)

    assert "Not study-ready" in rendered
    assert "reviewed" not in widget.options
    assert widget.value != "reviewed"
    st.cache_resource.clear()


def test_insights_not_study_ready_counts_a_hand_blocked_outside_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The KPI counted only completion, so an unreconciled hand rendered green 0."""

    import streamlit as st

    from poker_tracker.ui.navigation import Page

    path = tmp_path / "aggregate.sqlite3"
    _seed_promoted_then_broken(path)

    app = _run_page(path, Page.INSIGHTS, monkeypatch)
    rendered = "\n".join(item.value for item in app.markdown)

    assert 'pt-kpi-label">Not study-ready</div><div class="pt-kpi-value">1' in rendered
    assert "pt-kpi pt-kpi-negative" in rendered
    st.cache_resource.clear()
