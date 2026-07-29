from __future__ import annotations

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import (
    EXPORT_VERSION,
    export_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandIssue,
    HandPlayer,
    Session,
)


def _db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_flagging_hand_saves_snapshot_and_marks_derived_reviews_stale() -> None:
    db = _db()
    session = db.create_session(Session(name="Issue inbox"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=3,
            source_type="cv_import",
            review_status="reviewed",
            hero_cards="Ah Qs",
        )
    )
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            player_name="Hero",
            is_hero=True,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key=hero.player_key,
            street="preflop",
            player_name="Hero",
            action_type="call",
            amount=1,
        )
    )
    db.create_coaching_response(
        CoachingResponse(
            provider_name="fixture",
            model_name="deterministic",
            raw_prompt="prompt",
            raw_response="response",
            review_type="hand",
            hand_id=hand.id,
            session_id=session.id,
        )
    )

    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards", "actions"],
            description="Board and flop action look wrong.",
        )
    )

    assert issue.id is not None
    assert issue.status == "open"
    assert issue.evidence_snapshot["hand"]["hero_cards"] == "Ah Qs"
    assert issue.evidence_snapshot["players"][0]["player_key"] == "hero"
    assert issue.evidence_snapshot["actions"][0]["action_type"] == "call"
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    coaching = db.fetch_coaching_reviews_by_hand(hand.id)[0]
    assert coaching.is_stale is True
    assert "flagged" in coaching.stale_reason
    db.close()


def test_v11_to_v12_migration_preserves_existing_solver_tables(tmp_path) -> None:
    path = tmp_path / "v11.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    db._execute("DROP INDEX idx_hand_issues_status")
    db._execute("DROP INDEX idx_hand_issues_hand")
    db._execute("DROP TABLE hand_issues")
    db._execute(
        """
        INSERT INTO solver_range_profiles (
            name, notation, source, created_at, updated_at
        )
        VALUES ('Legacy range', 'AA', 'user',
                '2026-07-28T00:00:00+00:00',
                '2026-07-28T00:00:00+00:00')
        """
    )
    # A real schema-11 file predates the v13 columns. Leaving them in place while
    # stamping the file back to 11 fabricates a database whose schema is ahead of
    # its own stamp, which init_db now refuses outright -- because that is exactly
    # what a live v13 database with a lost stamp looks like, and replaying
    # _migrate_to_v13 against one discards every operator confirmation.
    db._execute("ALTER TABLE hands DROP COLUMN completion_status")
    db._execute("ALTER TABLE hands DROP COLUMN completion_evidence")
    db._execute(
        "UPDATE schema_metadata SET value = '11' WHERE key = 'schema_version'"
    )
    db._commit()
    db.close()

    migrated = PokerDatabase(path)
    migrated.init_db()

    assert migrated.schema_version() == SCHEMA_VERSION
    assert migrated._execute(
        "SELECT name FROM solver_range_profiles"
    ).fetchone()["name"] == "Legacy range"
    assert migrated.fetch_hand_issues() == []
    migrated.close()


def test_issue_can_be_resolved_without_deleting_debug_evidence() -> None:
    db = _db()
    session = db.create_session(Session(name="Resolution"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["hand_boundary"],
            description="This may belong to the previous hand.",
        )
    )

    resolved = db.resolve_hand_issue(
        issue.id,
        resolution_notes="Confirmed boundary and corrected exporter.",
    )

    assert resolved.status == "resolved"
    assert resolved.resolved_at is not None
    assert resolved.resolution_notes == "Confirmed boundary and corrected exporter."
    assert resolved.evidence_snapshot["hand"]["hand_number"] == 1
    assert db.fetch_hand_issues(status="open") == []
    assert db.fetch_hand_issues(status="resolved")[0].id == issue.id
    db.close()


def test_issue_queue_survives_export_import() -> None:
    source = _db()
    session = source.create_session(Session(name="Portable issue"))
    hand = source.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="cv_import")
    )
    source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["pot_or_result"],
            description="Final result does not match the recording.",
        )
    )

    payload = export_session(source, session.id)
    target = _db()
    imported_session = import_session(target, payload)
    imported_hand = target.fetch_hands_by_session(imported_session.id)[0]
    imported_issue = target.fetch_hand_issues(hand_id=imported_hand.id)[0]

    assert EXPORT_VERSION == 5
    assert payload["export_version"] == EXPORT_VERSION
    assert imported_issue.description == "Final result does not match the recording."
    assert imported_issue.issue_types == ["pot_or_result"]
    assert imported_issue.evidence_snapshot["hand"]["source_type"] == "cv_import"
    source.close()
    target.close()


def test_issue_snapshot_includes_completion_fields_for_new_issues() -> None:
    db = _db()
    session = db.create_session(Session(name="Snapshot completion"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="cv_import")
    )

    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["hand_boundary"],
            description="Recording appears to start mid-hand.",
        )
    )

    snapshot = issue.evidence_snapshot["hand"]
    assert snapshot["completion_status"] == "uncertain"
    assert snapshot.get("completion_evidence") == {}
    db.close()
