"""Regressions for the round-3 adversarial findings against Phase 1.

Every test here failed before its fix. They are grouped by the contract they
defend rather than by module, because each finding sat in a seam: a schema step
that ran before the column it indexed existed, an import ceiling that only held
in one direction, a readiness composition that dropped one evidence source, and
three blockers whose stated clearing action the product could not perform.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from cv_lab.scripts.pipeline import region_detections as rd
from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline
from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    timeline_to_session_payload,
)
from poker_tracker.persistence.backup import (
    BACKUP_KEEP_COUNT,
    PINNED_PREFIX,
    backup_database,
    backups_dir_for,
)
from poker_tracker.persistence.completion import (
    CompletionEvidence,
    acknowledge_codes,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import export_session, import_session
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
    SolverRun,
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.regression_promotion import (
    promote_issue_to_regression,
    record_regression_observation,
)
from poker_tracker.services.study_readiness import (
    evaluate_study_readiness,
    is_reconstructed_hand,
)

CV_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cv"



def _prove_regression(db, issue_id: int) -> None:
    """Satisfy the release-blocking closure gate for a test that is not about it.

    Closing a release-blocking issue requires a regression observed failing for
    the defect and passing after the fix. These tests predate that rule and are
    about something else, so they record the evidence rather than route around it.
    """
    case = promote_issue_to_regression(
        db, issue_id, kind="cached_state", fixture_path="tests/fixture.py::test_case"
    )
    record_regression_observation(
        db, case.id, failing_before=True, passing_after=True, fixing_commit="deadbee"
    )


def _clean_evidence(**overrides: object) -> CompletionEvidence:
    payload = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=1,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.92,
            layout_supported=True,
            table_size=6,
        )
    )
    payload.update(overrides)
    return parse_completion_evidence(payload)


def _seed_reconstructed_hand(
    db: PokerDatabase, evidence: CompletionEvidence
) -> tuple[Session, Hand]:
    session = db.create_session(Session(name="Round 3", date_played=date(2026, 1, 1)))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            source_type="cv_import",
            completion_status=derive_completion_status(
                evidence, source_type="cv_import"
            ),
            completion_evidence=dump_completion_evidence(evidence),
        )
    )
    return session, hand


# ---------------------------------------------------------------------------
# Finding 1 -- a pre-v5 roi_profiles table bricked every open
# ---------------------------------------------------------------------------

_PRE_V5_ROI_SCHEMA = """
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date_played TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT '',
    stakes TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE roi_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT '',
    video_width INTEGER NOT NULL,
    video_height INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _write_pre_v5_roi_database(path: Path, *, stored_version: int | None) -> None:
    """A pre-versioning database whose roi_profiles predates is_active/table_layout."""
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_PRE_V5_ROI_SCHEMA)
        connection.execute(
            "CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        if stored_version is not None:
            connection.execute(
                "INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
                (str(stored_version),),
            )
        connection.execute(
            "INSERT INTO sessions (name, date_played, created_at) "
            "VALUES ('Legacy', '2024-01-01', '2024-01-01T00:00:00')"
        )
        connection.execute(
            "INSERT INTO roi_profiles (name, platform, video_width, video_height, "
            "created_at, updated_at) VALUES ('legacy', 'ClubWPT Gold', 1920, 1080, "
            "'2024-01-01T00:00:00', '2024-01-01T00:00:00')"
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize("stored_version", [None, 1, 4])
def test_a_pre_v5_roi_profiles_table_still_migrates_to_the_current_schema(
    tmp_path: Path, stored_version: int | None
) -> None:
    """_create_base_schema indexed a column only the legacy backfill adds.

    The index ran before the backfill and through executescript(), so the failure
    was committed, permanent, and left no pre-migration snapshot.
    """
    path = tmp_path / f"legacy_roi_{stored_version}.sqlite3"
    _write_pre_v5_roi_database(path, stored_version=stored_version)

    db = PokerDatabase(path)
    db.init_db()
    try:
        assert db.schema_version() == SCHEMA_VERSION
        columns = {
            row["name"]
            for row in db._execute("PRAGMA table_info(roi_profiles)").fetchall()
        }
        assert {"is_active", "table_layout"} <= columns
        assert db._execute("SELECT COUNT(*) AS n FROM roi_profiles").fetchone()["n"] == 1
        assert db._execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"] == 1
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Finding 2 -- import must only ever weaken a declared completion status
# ---------------------------------------------------------------------------


def test_import_never_relaxes_a_declared_partial_into_a_clearable_uncertain() -> None:
    """`partial` is permanent; `uncertain` clears with one acknowledgement.

    The ceiling only fired when the re-derived status was exactly `complete`, so a
    payload declaring the stronger `partial` was silently downgraded whenever its
    evidence derived `uncertain` -- which every stripped, corrupt, pre-v5 or
    future-version payload does.
    """
    db = PokerDatabase(":memory:")
    db.init_db()
    try:
        _, hand = _seed_reconstructed_hand(
            db, _clean_evidence(warning_codes=["missing_label"])
        )
        assert hand.completion_status == "uncertain"

        payload = export_session(db, hand.session_id)
        payload["hands"][0]["hand"]["completion_status"] = "partial"
        imported = import_session(db, payload)
        restored = db.fetch_hands_by_session(imported.id)[0]

        assert restored.completion_status == "partial"

        stored = parse_completion_evidence(restored.completion_evidence)
        db.update_hand_completion(
            restored.id,
            completion_evidence=dump_completion_evidence(
                acknowledge_codes(stored, ["missing_label"])
            ),
        )
        assert db.fetch_hand(restored.id).completion_status == "partial"
        with pytest.raises(ValueError):
            db.update_hand_status(restored.id, "reviewed")
        assert not evaluate_study_readiness(
            db.fetch_hand(restored.id), accounting=None, user_confirmed=True
        ).is_ready
    finally:
        db.close()


@pytest.mark.parametrize(
    "evidence_override",
    [None, "not json at all", {"evidence_version": 2}],
    ids=["stripped", "corrupt", "future_version"],
)
def test_an_unreadable_payload_keeps_the_declared_partial(
    evidence_override: object,
) -> None:
    db = PokerDatabase(":memory:")
    db.init_db()
    try:
        _, hand = _seed_reconstructed_hand(db, _clean_evidence())
        payload = export_session(db, hand.session_id)
        payload["hands"][0]["hand"]["completion_status"] = "partial"
        payload["hands"][0]["hand"]["completion_evidence"] = evidence_override
        imported = import_session(db, payload)
        assert db.fetch_hands_by_session(imported.id)[0].completion_status == "partial"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Finding 3 -- a corrupt blob must never break a whole session's read
# ---------------------------------------------------------------------------


def test_a_deeply_nested_evidence_blob_degrades_instead_of_raising(
    tmp_path: Path,
) -> None:
    """RecursionError is a RuntimeError, so neither guard's except clause caught it."""
    blob = "[" * 60000 + "]" * 60000
    assert parse_completion_evidence(blob) == CompletionEvidence()

    path = tmp_path / "nested.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    try:
        _, hand = _seed_reconstructed_hand(db, _clean_evidence())
        db._execute(
            "UPDATE hands SET completion_evidence = ? WHERE id = ?", (blob, hand.id)
        )
        db._commit()
        hands = db.fetch_hands_by_session(hand.session_id)
        assert [item.id for item in hands] == [hand.id]
        # The unreadable blob degrades to all-unknown evidence, which blocks --
        # and the loss is RECORDED rather than read back as "no evidence": the
        # hand carries the same unreadable-column marker a damaged scalar gets,
        # bounded so a 120 KB adversarial blob is not copied into a blocker's
        # detail whole.
        assert parse_completion_evidence(hands[0].completion_evidence).is_known is False
        assert hands[0].unreadable_columns == ("completion_evidence",)
        assert hands[0].review_status == "needs_correction"
        readiness = evaluate_study_readiness(
            hands[0], accounting=None, user_confirmed=True
        )
        assert readiness.has("COMPLETION_EVIDENCE_MISSING")
        assert readiness.has("COMPLETION_NOT_COMPLETE")
        assert readiness.has("UNREADABLE_HAND_COLUMNS")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Finding 5 -- a settlement correction must stale retained coaching and solver
# ---------------------------------------------------------------------------


def _seed_settled_manual_hand(db: PokerDatabase) -> tuple[Session, Hand]:
    session = db.create_session(Session(name="Awards", date_played=date(2026, 1, 1)))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            table_size=6,
            pot_size=30,
            hero_bb_won=-10,
        )
    )
    seats = (
        ("hero", "Hero", "BTN", True),
        ("villain_a", "VillainA", "SB", False),
        ("villain_b", "VillainB", "BB", False),
    )
    for index, (key, name, position, is_hero) in enumerate(seats):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=name,
                position=position,
                starting_stack=100,
                is_hero=is_hero,
            )
        )
    for key, name, position, is_hero in seats:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                position=position,
                street="river",
                action_type="bet" if is_hero else "call",
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
                player_key="villain_a",
                player_name="VillainA",
                amount=30,
                entry_order=1,
            )
        ],
    )
    persist_reconciliation(db, hand.id)
    return session, hand


def test_reassigning_the_pot_stales_the_coaching_and_solver_it_invalidates(
    tmp_path: Path,
) -> None:
    """The demotion to needs_correction was cosmetic: one click undid it.

    Coaching is grounded in the ledger, the winners and the hero result, so a
    corrected winner must not leave a retained review presented as current.
    """
    db = PokerDatabase(tmp_path / "awards.sqlite3")
    db.init_db()
    try:
        session, hand = _seed_settled_manual_hand(db)
        db.create_coaching_response(
            CoachingResponse(
                hand_id=hand.id,
                session_id=session.id,
                review_type="hand",
                provider_name="claude",
                model_name="test",
                raw_prompt="VillainA won the 30bb pot at showdown.",
                raw_response="VillainA showed down the winner from the SB.",
                created_at=datetime.now(UTC),
            )
        )
        db.create_solver_run(
            SolverRun(
                hand_id=hand.id,
                status="completed",
                input_hash="hash-villain-a",
                evidence={"winner": "villain_a"},
            )
        )
        db.update_hand_status(hand.id, "reviewed")
        assert db.fetch_hand(hand.id).review_status == "reviewed"

        with db.transaction():
            db.replace_settlement_entries(
                hand.id,
                [
                    SettlementEntry(
                        hand_id=hand.id,
                        entry_type="award",
                        pot_index=0,
                        player_key="villain_b",
                        player_name="VillainB",
                        amount=30,
                        entry_order=1,
                    )
                ],
            )
            persist_reconciliation(db, hand.id)

        assert all(
            review.is_stale for review in db.fetch_coaching_reviews_by_hand(hand.id)
        )
        assert [run.status for run in db.fetch_solver_runs_by_hand(hand.id)] == ["stale"]
        readiness = evaluate_study_readiness(
            db.fetch_hand(hand.id),
            accounting=reconcile_persisted_hand(db, hand.id),
            coaching_reviews=db.fetch_coaching_reviews_by_hand(hand.id),
            hand_reviews=db.fetch_reviews_by_hand(hand.id),
            solver_runs=db.fetch_solver_runs_by_hand(hand.id),
        )
        # update_hand_status is documented as a single-table floor and cannot see
        # coaching or solver rows, so readiness is what has to refuse this.
        assert not readiness.is_ready
        assert readiness.has("STALE_COACHING_EVIDENCE")
        assert readiness.has("STALE_SOLVER_EVIDENCE")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Findings 8 and 12 -- a blocker may not promise an action the product lacks
# ---------------------------------------------------------------------------


def test_a_rejected_hand_is_not_promised_an_acknowledgement_that_cannot_clear_it() -> (
    None
):
    """acknowledge_codes refuses rejections, so the acknowledge wording was false."""
    evidence = _clean_evidence(rejection_codes=["action_sequence_illegal"])
    hand = Hand(
        session_id=1,
        hand_number=1,
        source_type="cv_import",
        hero_cards="Ah Kd",
        completion_status=derive_completion_status(evidence, source_type="cv_import"),
        completion_evidence=dump_completion_evidence(evidence),
    )
    readiness = evaluate_study_readiness(hand, accounting=None, user_confirmed=True)
    blocker = next(
        item for item in readiness.blockers if item.code == "COMPLETION_NOT_COMPLETE"
    )
    action = blocker.clearing_action.lower()
    assert "only a new reconstruction clears this" in action
    assert "cannot be acknowledged" in action
    assert "action_sequence_illegal" in " ".join(blocker.detail)


def test_the_stale_solver_blocker_only_names_actions_the_product_offers() -> None:
    hand = Hand(
        session_id=1,
        hand_number=1,
        source_type="manual",
        completion_status="not_applicable",
    )
    readiness = evaluate_study_readiness(
        hand,
        accounting=None,
        solver_runs=[SolverRun(hand_id=1, status="stale", input_hash="h")],
    )
    blocker = next(
        item for item in readiness.blockers if item.code == "STALE_SOLVER_EVIDENCE"
    )
    assert hasattr(PokerDatabase, "delete_solver_run"), (
        "STALE_SOLVER_EVIDENCE names a delete control; the store must provide one."
    )
    assert "delete" in blocker.clearing_action.lower()


def test_deleting_the_stale_solver_run_clears_the_solver_blocker(
    tmp_path: Path,
) -> None:
    db = PokerDatabase(tmp_path / "solver.sqlite3")
    db.init_db()
    try:
        session, hand = _seed_settled_manual_hand(db)
        run = db.create_solver_run(
            SolverRun(hand_id=hand.id, status="stale", input_hash="hash-stale")
        )
        assert evaluate_study_readiness(
            db.fetch_hand(hand.id),
            accounting=reconcile_persisted_hand(db, hand.id),
            solver_runs=db.fetch_solver_runs_by_hand(hand.id),
        ).has("STALE_SOLVER_EVIDENCE")

        db.delete_solver_run(run.id)

        assert db.fetch_solver_runs_by_hand(hand.id) == []
        assert not evaluate_study_readiness(
            db.fetch_hand(hand.id),
            accounting=reconcile_persisted_hand(db, hand.id),
            solver_runs=db.fetch_solver_runs_by_hand(hand.id),
        ).has("STALE_SOLVER_EVIDENCE")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Finding 9 -- flagging for debugging must record its own demotion
# ---------------------------------------------------------------------------


def test_flagging_a_hand_for_debugging_records_the_demotion_in_its_evidence(
    tmp_path: Path,
) -> None:
    """The column said uncertain while the evidence still derived complete.

    Nothing rendered the Source warnings panel, so the blocker named an action
    that did not exist on screen and the hand was stranded for good.
    """
    db = PokerDatabase(tmp_path / "flagged.sqlite3")
    db.init_db()
    try:
        _, hand = _seed_reconstructed_hand(db, _clean_evidence())
        assert hand.completion_status == "complete"

        issue = db.create_hand_issue(
            HandIssue(hand_id=hand.id, issue_types=["cards"], description="misread")
        )
        _prove_regression(db, issue.id)
        db.resolve_hand_issue(issue.id, resolution_notes="fixed in pipeline")

        flagged = db.fetch_hand(hand.id)
        evidence = parse_completion_evidence(flagged.completion_evidence)
        assert flagged.completion_status == "uncertain"
        assert derive_completion_status(
            evidence, source_type=flagged.source_type
        ) == flagged.completion_status
        assert evidence.unresolved_codes, (
            "no code means show_source_warning_controls draws nothing"
        )

        # And the action the blocker names actually clears it.
        db.update_hand_completion(
            hand.id,
            completion_evidence=dump_completion_evidence(
                acknowledge_codes(evidence, evidence.unresolved_codes)
            ),
        )
        assert db.fetch_hand(hand.id).completion_status == "complete"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Finding 10 -- the pre-migration snapshot must survive rotation
# ---------------------------------------------------------------------------


def test_the_pre_migration_snapshot_is_pinned_against_rotation(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    """v13 irreversibly forces needs_correction, so its rollback point is the only one."""
    path = tmp_path / "rotating.sqlite3"
    _write_pre_v5_roi_database(path, stored_version=None)

    db = PokerDatabase(path)
    db.init_db()
    db.close()

    # The snapshot lives with the database it can roll back, which for anything
    # but the live database is beside that database.
    snapshot_dir = backups_dir_for(path)
    pinned = list(snapshot_dir.glob(f"{PINNED_PREFIX}*.sqlite3"))
    assert len(pinned) == 1, "the pre-migration snapshot must be pinned"
    assert not list(snapshot_dir.glob("poker_tracker_*.sqlite3"))
    assert not list(isolated_backup_dir.glob("*.sqlite3"))

    for _ in range(BACKUP_KEEP_COUNT + 2):
        backup_database(path, snapshot_dir)

    assert pinned[0].exists(), "rotation deleted the only migration rollback point"


# ---------------------------------------------------------------------------
# Finding 11 -- both halves of is_reconstructed_hand carry weight
# ---------------------------------------------------------------------------


def test_a_manual_hand_that_is_not_not_applicable_is_still_reconstructed() -> None:
    """source_type alone is not the rule; the pair is.

    A source-only predicate would drop the completion, layout, source-warning and
    confirmation blockers from a hand stored as manual + uncertain in one step.
    """
    hand = Hand(
        session_id=1,
        hand_number=1,
        source_type="manual",
        completion_status="uncertain",
        hero_cards="Ah Kd",
    )
    assert hand.completion_status == "uncertain"
    assert is_reconstructed_hand(hand) is True

    readiness = evaluate_study_readiness(hand, accounting=None)
    assert readiness.has("COMPLETION_NOT_COMPLETE")
    assert readiness.has("COMPLETION_EVIDENCE_MISSING")
    assert readiness.has("USER_CONFIRMATION_MISSING")

    exempt = hand.model_copy(update={"completion_status": "not_applicable"})
    assert is_reconstructed_hand(exempt) is False


# ---------------------------------------------------------------------------
# Finding 7 -- the exporter's layout evidence must match what PLAN describes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    ["g0621_ar1750_frames.json.gz", "g0723a_baseline_ar1397_frames.json.gz"],
)
def test_the_exported_layout_evidence_matches_the_documented_rule(
    fixture_name: str,
) -> None:
    """PLAN claimed table_size was always None and every hand carried the blocker.

    It is not: the spine does resolve player rows, so `layout_supported` is True
    and UNSUPPORTED_TABLE_LAYOUT does not fire. This pins the real behaviour so
    the documented gap cannot drift away from it again.
    """
    fixture = CV_FIXTURES / fixture_name
    frames = rd.frames_from_fixture(
        json.loads(gzip.open(fixture, "rb").read())
    )
    payload = timeline_to_session_payload(
        build_hand_timeline(frames),
        timeline_path=Path("timeline.json"),
        session_name="Layout",
        allow_validation_warnings=True,
    )
    assert payload["hands"]
    for exported in payload["hands"]:
        evidence = parse_completion_evidence(exported["hand"]["completion_evidence"])
        assert evidence.table_size == len(exported["players"])
        assert evidence.layout_supported is True
        hand = Hand.model_validate(
            {**exported["hand"], "session_id": 1, "hand_number": 1}
        )
        readiness = evaluate_study_readiness(hand, accounting=None)
        assert not readiness.has("UNSUPPORTED_TABLE_LAYOUT")


# ---------------------------------------------------------------------------
# Findings 4 and 6 -- the Study page's own readiness object
# ---------------------------------------------------------------------------

CONFIRM_LABEL = "I have read the evidence above and confirm this hand is correct"


def test_the_study_page_composes_legacy_hand_reviews_into_readiness(
    tmp_path, monkeypatch
) -> None:
    """Three of the five review-status writers trust the Study page's readiness.

    It fetched issues, coaching and solver runs but not hand_reviews, so a hand
    whose legacy retained review had been staled by a correction rendered
    "Study-ready · 0 blockers" while every other surface blocked it.
    """
    from poker_tracker.persistence.models import HandReview
    from tests.test_study_readiness_ui import (
        _run_validation_editors,
        _saved_review_status,
        _seed_hand,
    )

    path = tmp_path / "legacy_review.sqlite3"
    hand_id = _seed_hand(
        path,
        source_type="manual",
        completion_status="not_applicable",
        completion_evidence={},
        review_status="unreviewed",
    )
    db = PokerDatabase(path)
    db.init_db()
    db.create_hand_review(
        HandReview(
            hand_id=hand_id,
            hand_summary="Hero bet river.",
            theory_coach="Bet 75%.",
            exploit_coach="Villain overfolds.",
            ev_math_notes="EV +3bb",
            study_lesson="Thin value.",
            next_review_question="Bet turn?",
            created_at=datetime.now(UTC),
        )
    )
    hero = next(item for item in db.fetch_players_by_hand(hand_id) if item.is_hero)
    db.update_hand_player(
        hero.model_copy(update={"player_name": "Hero (fixed)"}),
        correction_notes="name was misread",
    )
    persist_reconciliation(db, hand_id)
    db.close()

    app = _run_validation_editors(
        path, monkeypatch, hand_id, frames_validated=False
    )
    assert not list(app.exception)
    rendered = " ".join(str(item.value) for item in app.markdown)
    assert "What's blocking Study" in rendered or "Not study-ready" in rendered
    assert "Study-ready · 0 blockers" not in rendered
    assert "Stale coaching" in rendered or "Coaching evidence" in rendered
    next(
        button
        for button in app.button
        if button.label == "Finish validation — send to Study"
    ).click()
    app.run()
    assert _saved_review_status(path, hand_id) != "reviewed"


def test_the_study_confirmation_does_not_survive_an_evidence_change(
    tmp_path, monkeypatch
) -> None:
    """Confirmation digests the evidence; a correction replaces the digest key.

    Keyed on the hand alone, a confirmation survived the very correction that
    invalidated it, so USER_CONFIRMATION_MISSING never came back and the hand was
    re-promotable without anyone re-reading anything.
    """
    import app as app_module
    from poker_tracker.services.hand_accounting import reconcile_persisted_hand
    from poker_tracker.services.study_readiness import evaluate_study_readiness
    from tests.test_study_readiness_ui import _seed_hand

    path = tmp_path / "confirm_reset.sqlite3"
    hand_id = _seed_hand(path, completion_status="complete", review_status="unreviewed")

    db = PokerDatabase(path)
    db.init_db()
    hand = db.fetch_hand(hand_id)
    accounting = reconcile_persisted_hand(db, hand_id)
    key_before = app_module.study_confirmation_key(hand, accounting)

    hero = next(item for item in db.fetch_players_by_hand(hand_id) if item.is_hero)
    db.update_hand_player(
        hero.model_copy(update={"starting_stack": 42}),
        correction_notes="hero stack was misread",
    )
    persist_reconciliation(db, hand_id)
    hand_after = db.fetch_hand(hand_id)
    accounting_after = reconcile_persisted_hand(db, hand_id)
    key_after = app_module.study_confirmation_key(hand_after, accounting_after)
    assert key_before != key_after

    readiness = evaluate_study_readiness(
        hand_after,
        accounting=accounting_after,
        user_confirmed=False,
    )
    assert readiness.has("USER_CONFIRMATION_MISSING")
    db.close()
