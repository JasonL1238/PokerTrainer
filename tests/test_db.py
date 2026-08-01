import pytest
from pydantic import ValidationError

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandReview,
    ProcessingJob,
    ReconstructionFrameReview,
    Session,
    VideoRecord,
)


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_database_initialization_is_idempotent() -> None:
    db = make_db()
    db.init_db()

    assert db.fetch_sessions() == []
    assert db.schema_version() == SCHEMA_VERSION

    db.close()


def test_schema_supports_new_hand_fields() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))

    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_position="BTN",
            hero_cards="AhQs",
            board_cards="Qd 7s 2c",
            hero_bb_won=4.5,
            review_status="needs_correction",
            confidence_score=0.72,
            source_type="corrected_cv",
            tags=["MISSED_VALUE", "BIG_POT"],
        )
    )

    saved = db.fetch_hands_by_session(session.id)[0]
    assert hand.id is not None
    assert saved.hero_cards == "Ah Qs"
    assert saved.review_status == "needs_correction"
    assert saved.confidence_score == 0.72
    assert saved.source_type == "corrected_cv"
    assert saved.tags == ["MISSED_VALUE", "BIG_POT"]

    db.close()


def test_create_session_hand_action_review_still_works() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session", stakes="1/2 NL"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="As Qh"))
    action = db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            player_name="Hero",
            position="BTN",
            action_type="raise",
            amount=2.5,
        )
    )
    review = db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="summary",
            theory_coach="theory",
            exploit_coach="exploit",
            ev_math_notes="math",
            study_lesson="lesson",
            next_review_question="question",
            notes="review note",
        )
    )

    assert session.id is not None
    assert hand.id is not None
    assert action.id is not None
    assert action.action_index == 1
    assert review.id is not None
    assert db.fetch_reviews_by_hand(hand.id)[0].ev_math_notes == "math"

    db.close()


def test_action_order_increments_by_hand_and_street() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    first = db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="A", action_type="raise")
    )
    second = db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="B", action_type="call")
    )
    flop_first = db.create_action(
        Action(hand_id=hand.id, street="flop", player_name="A", action_type="bet")
    )

    assert first.action_index == 1
    assert second.action_index == 2
    assert flop_first.action_index == 1

    db.close()


def test_create_hand_players_and_status_update() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_name="Hero",
            position="BTN",
            starting_stack=200,
            is_hero=True,
        )
    )
    db.update_hand_status(hand.id, "reviewed")

    players = db.fetch_players_by_hand(hand.id)
    saved = db.fetch_hand(hand.id)
    assert players[0].player_name == "Hero"
    assert players[0].is_hero is True
    assert saved.review_status == "reviewed"

    db.close()


def test_reconstruction_frame_review_upserts_and_filters_by_hand() -> None:
    db = make_db()
    video = db.create_video(
        VideoRecord(
            original_filename="session.mp4",
            stored_path="/tmp/session.mp4",
            file_size_bytes=10,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="completed",
        )
    )
    first = ReconstructionFrameReview(
        job_id=job.id,
        hand_number=1,
        source_image="/tmp/frame.jpg",
        timestamp_seconds=4.0,
        status="incorrect",
        issue_types=["Action / player"],
        notes="Seat 4 called.",
    )
    db.upsert_reconstruction_frame_review(first)
    db.upsert_reconstruction_frame_review(
        first.model_copy(
            update={"status": "correct", "issue_types": [], "notes": ""}
        )
    )
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job.id,
            hand_number=2,
            source_image="/tmp/other.jpg",
            timestamp_seconds=9.0,
            status="incorrect",
            issue_types=["Pot"],
        )
    )

    hand_one = db.fetch_reconstruction_frame_reviews(job.id, hand_number=1)
    assert len(hand_one) == 1
    assert hand_one[0].status == "correct"
    assert hand_one[0].issue_types == []
    assert len(db.fetch_reconstruction_frame_reviews(job.id)) == 2
    db.close()


def test_update_and_delete_action() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    action = db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="call")
    )

    db.update_action(
        Action(
            id=action.id,
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            player_name="Hero",
            action_type="raise",
            amount=8,
        )
    )
    assert db.fetch_actions_by_hand(hand.id)[0].action_type == "raise"

    db.delete_action(action.id)
    assert db.fetch_actions_by_hand(hand.id) == []

    db.close()


def test_delete_hand_cascades_related_rows() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_action(Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="win"))

    db.delete_hand(hand.id)

    assert db.fetch_hands_by_session(session.id) == []
    assert db.fetch_actions_by_hand(hand.id) == []

    db.close()


def test_validation_catches_bad_cards_and_action_types() -> None:
    with pytest.raises(ValidationError):
        Hand(session_id=1, hand_number=1, hero_cards="Ax Qs")

    with pytest.raises(ValidationError):
        Hand(session_id=1, hand_number=1, board_cards="Qd Qd 2c")

    with pytest.raises(ValidationError):
        Hand(session_id=1, hand_number=1, hero_cards="Ah Qs", board_cards="Ah 7d 2c")

    with pytest.raises(ValidationError):
        Action(hand_id=1, street="preflop", player_name="Hero", action_type="punt")


def test_transaction_commits_grouped_writes() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))

    with db.transaction():
        hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
        db.create_hand_player(HandPlayer(hand_id=hand.id, player_name="Hero", is_hero=True))
        db.create_action(
            Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="call")
        )

    assert len(db.fetch_hands_by_session(session.id)) == 1
    assert len(db.fetch_players_by_hand(hand.id)) == 1
    assert len(db.fetch_actions_by_hand(hand.id)) == 1

    db.close()


def test_transaction_rolls_back_all_writes_on_error() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))

    with pytest.raises(RuntimeError):
        with db.transaction():
            hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
            db.create_action(
                Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="call")
            )
            raise RuntimeError("boom")

    # The hand and action from the failed transaction must not be persisted.
    assert db.fetch_hands_by_session(session.id) == []

    db.close()


def test_nested_transactions_defer_to_outermost() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))

    with pytest.raises(RuntimeError):
        with db.transaction():
            with db.transaction():
                db.create_hand(Hand(session_id=session.id, hand_number=1))
            raise RuntimeError("outer failure after inner success")

    assert db.fetch_hands_by_session(session.id) == []

    db.close()


def test_delete_session_cascades_hands_and_actions() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="win")
    )

    db.delete_session(session.id)

    assert db.fetch_session(session.id) is None
    assert db.fetch_hands_by_session(session.id) == []
    assert db.fetch_actions_by_hand(hand.id) == []

    db.close()


def test_init_db_refuses_newer_schema_version() -> None:
    db = make_db()
    db._execute(
        "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    db._commit()

    with pytest.raises(RuntimeError, match="newer"):
        db.init_db()

    db.close()


@pytest.mark.parametrize("ahead_by", [1, 5, 100])
def test_the_refusal_names_both_versions_and_repeats(tmp_path, ahead_by: int) -> None:
    """The message has to say what to do, and saying it once is not enough.

    An operator who runs an older build against a newer database sees this at
    every start until they upgrade; a refusal that degraded into an open on the
    second attempt would be worse than one that never happened.
    """
    path = tmp_path / "newer.sqlite3"
    seeded = PokerDatabase(path)
    seeded.init_db()
    future_version = SCHEMA_VERSION + ahead_by
    seeded._execute(
        "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
        (str(future_version),),
    )
    seeded._commit()
    seeded.close()

    for _ in range(2):
        reopened = PokerDatabase(path)
        with pytest.raises(RuntimeError) as caught:
            reopened.init_db()
        message = str(caught.value)
        assert str(future_version) in message
        assert str(SCHEMA_VERSION) in message
        assert "Update the app" in message
        assert reopened.schema_version() == future_version
        reopened.close()


def test_a_newer_database_cannot_be_written_even_without_init_db(tmp_path) -> None:
    """The constructor cannot raise, so the refusal has to survive into the writes.

    It returns before switching the file to WAL on purpose: converting a newer
    database's journal mode is itself a write to a file this build does not
    understand. That left the object it returns holding a live connection on a
    database whose tables are all present -- a newer build's schema is a superset
    of this one's -- so a caller that skipped init_db wrote to it and nothing
    complained.
    """
    path = tmp_path / "newer-write.sqlite3"
    seeded = PokerDatabase(path)
    seeded.init_db()
    seeded.create_session(Session(name="Written by the newer build"))
    seeded._execute(
        "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    seeded._commit()
    seeded.close()
    before = path.read_bytes()

    refused = PokerDatabase(path)
    with pytest.raises(RuntimeError, match="newer than this app"):
        refused.create_session(Session(name="Written by the older build"))
    # Reads still work: the version and the backup path both need them.
    assert refused.schema_version() == SCHEMA_VERSION + 1
    assert len(refused.fetch_sessions()) == 1
    refused.close()

    assert path.read_bytes() == before


def _v12_hands_table_columns() -> list[str]:
    """The exact hands columns a schema-12 database holds, in order."""
    return [
        "id",
        "session_id",
        "hand_number",
        "game_type",
        "blinds_antes",
        "table_size",
        "effective_stack",
        "hero_position",
        "hero_cards",
        "board_cards",
        "pot_size",
        "result",
        "hero_bb_won",
        "review_status",
        "confidence_score",
        "source_type",
        "tags",
        "notes",
        "created_at",
    ]


def test_fresh_schema_has_completion_columns() -> None:
    db = make_db()

    columns = {row["name"] for row in db._execute("PRAGMA table_info(hands)").fetchall()}

    assert {"completion_status", "completion_evidence"} <= columns
    db.close()


def test_fresh_and_migrated_hands_schema_are_identical(tmp_path) -> None:
    def signature(database: PokerDatabase) -> list[tuple]:
        return [
            (row["name"], row["type"], row["notnull"], row["dflt_value"])
            for row in database._execute("PRAGMA table_info(hands)").fetchall()
        ]

    fresh = make_db()
    expected = signature(fresh)
    fresh.close()

    path = tmp_path / "legacy.sqlite3"
    legacy = PokerDatabase(path)
    legacy.init_db()
    legacy._execute("ALTER TABLE hands DROP COLUMN completion_status")
    legacy._execute("ALTER TABLE hands DROP COLUMN completion_evidence")
    legacy._execute("ALTER TABLE hands DROP COLUMN study_inclusion")
    legacy._execute("UPDATE schema_metadata SET value = '12' WHERE key = 'schema_version'")
    legacy._commit()
    assert [row[0] for row in signature(legacy)] == _v12_hands_table_columns()
    legacy.close()

    migrated = PokerDatabase(path)
    migrated.init_db()

    assert signature(migrated) == expected
    migrated.close()


def test_update_hand_status_rejects_unknown_review_status() -> None:
    db = make_db()
    session = db.create_session(Session(name="Status"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    with pytest.raises(ValueError, match="Unknown review status"):
        db.update_hand_status(hand.id, "approved")

    assert db.fetch_hand(hand.id).review_status == "unreviewed"
    db.close()


@pytest.mark.parametrize("completion_status", ["uncertain", "partial"])
def test_update_hand_status_refuses_reviewed_for_unproven_hand(completion_status: str) -> None:
    db = make_db()
    session = db.create_session(Session(name="Unproven"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status=completion_status,
        )
    )

    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        db.update_hand_status(hand.id, "reviewed")

    assert db.fetch_hand(hand.id).review_status != "reviewed"
    db.update_hand_status(hand.id, "needs_correction")
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    db.close()


def test_update_hand_status_refuses_reviewed_with_open_issue() -> None:
    from poker_tracker.persistence.models import HandIssue

    db = make_db()
    session = db.create_session(Session(name="Open issue"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["pot_or_result"],
            description="Pot does not match the recording.",
        )
    )

    with pytest.raises(ValueError, match="open debugging issue"):
        db.update_hand_status(hand.id, "reviewed")

    db.close()


@pytest.mark.parametrize("source_type", ["cv_import", "corrected_cv"])
def test_update_hand_status_refuses_reviewed_for_a_reconstructed_not_applicable_hand(
    source_type: str,
) -> None:
    """'not_applicable' exempts a hand from every completion blocker.

    import_session already rejects that pair as a laundering vector; create_hand
    accepts it, so the store floor must refuse the promotion too.
    """
    db = make_db()
    session = db.create_session(Session(name="Laundering"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type=source_type,
            completion_status="not_applicable",
        )
    )

    with pytest.raises(ValueError, match="not_applicable"):
        db.update_hand_status(hand.id, "reviewed")

    assert db.fetch_hand(hand.id).review_status != "reviewed"
    # A non-promoting status change is still allowed on the same row.
    db.update_hand_status(hand.id, "needs_correction")
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    db.close()


def test_update_hand_status_allows_reviewed_for_manual_hand() -> None:
    db = make_db()
    session = db.create_session(Session(name="Manual"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    db.update_hand_status(hand.id, "reviewed")

    saved = db.fetch_hand(hand.id)
    assert saved.review_status == "reviewed"
    assert saved.completion_status == "not_applicable"
    db.close()


def test_create_hand_persists_completion_status_and_evidence() -> None:
    db = make_db()
    session = db.create_session(Session(name="Completion"))
    evidence = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.9,
            source_frames=("frames/a.jpg",),
        )
    )
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=evidence,
        )
    )

    saved = db.fetch_hand(hand.id)
    assert saved.completion_status == "complete"
    assert saved.completion_evidence["source_frames"] == ["frames/a.jpg"]
    assert parse_completion_evidence(saved.completion_evidence).is_known is True
    db.close()


def test_cv_hand_without_declared_completion_defaults_to_uncertain() -> None:
    db = make_db()
    session = db.create_session(Session(name="Default"))

    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="cv_import")
    )

    assert hand.completion_status == "uncertain"
    assert db.fetch_hand(hand.id).completion_status == "uncertain"
    db.close()


def test_fetch_hand_parses_corrupt_completion_evidence_as_empty() -> None:
    db = make_db()
    session = db.create_session(Session(name="Corrupt"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db._execute(
        "UPDATE hands SET completion_evidence = ? WHERE id = ?",
        ("{not valid json", hand.id),
    )
    db._commit()

    saved = db.fetch_hand(hand.id)

    assert saved.completion_evidence == {}
    assert parse_completion_evidence(saved.completion_evidence).is_known is False
    db.close()


def _uncertain_cv_hand(db: PokerDatabase, session_id: int) -> Hand:
    evidence = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.9,
            warning_codes=("pot_not_reconciled",),
        )
    )
    return db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=1,
            source_type="cv_import",
            completion_status="uncertain",
            completion_evidence=evidence,
        )
    )


def test_update_hand_completion_promotes_uncertain_to_complete_on_acknowledgement() -> None:
    db = make_db()
    session = db.create_session(Session(name="Acknowledge"))
    hand = _uncertain_cv_hand(db, session.id)
    evidence = acknowledge_codes(
        parse_completion_evidence(hand.completion_evidence), ["pot_not_reconciled"]
    )

    updated = db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(evidence),
        notes="Reviewed the recording; the pot is right.",
    )

    assert updated.completion_status == "complete"
    assert db.fetch_hand(hand.id).completion_status == "complete"
    db.close()


def test_update_hand_completion_never_promotes_a_partial_hand() -> None:
    db = make_db()
    session = db.create_session(Session(name="Partial"))
    truncated = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=True,
            terminal_event="showdown",
            boundary_confidence=0.9,
            warning_codes=("truncated_recording",),
        )
    )
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="partial",
            completion_evidence=truncated,
        )
    )
    accepted = acknowledge_codes(
        parse_completion_evidence(truncated), ["truncated_recording"]
    )

    updated = db.update_hand_completion(
        hand.id, completion_evidence=dump_completion_evidence(accepted)
    )

    assert updated.completion_status == "partial"
    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        db.update_hand_status(hand.id, "reviewed")
    db.close()


def test_update_hand_completion_records_a_flat_correction() -> None:
    db = make_db()
    session = db.create_session(Session(name="Audit"))
    hand = _uncertain_cv_hand(db, session.id)
    evidence = acknowledge_codes(
        parse_completion_evidence(hand.completion_evidence), ["pot_not_reconciled"]
    )

    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(evidence),
        notes="Accepted after re-watching.",
    )

    correction = db.fetch_hand_corrections(hand.id)[0]
    assert correction.correction_type == "hand_facts"
    assert correction.before_state == {
        "completion_status": "uncertain",
        "acknowledged_codes": "",
    }
    assert correction.after_state == {
        "completion_status": "complete",
        "acknowledged_codes": "pot_not_reconciled",
    }
    assert correction.notes == "Accepted after re-watching."
    db.close()


def test_update_hand_completion_does_not_restale_coaching_or_solver_evidence() -> None:
    from poker_tracker.persistence.models import CoachingResponse

    db = make_db()
    session = db.create_session(Session(name="No restale"))
    hand = _uncertain_cv_hand(db, session.id)
    db.create_coaching_response(
        CoachingResponse(
            provider_name="test",
            model_name="fixture",
            raw_prompt="prompt",
            raw_response="response",
            review_type="hand",
            hand_id=hand.id,
            session_id=session.id,
        )
    )
    evidence = acknowledge_codes(
        parse_completion_evidence(hand.completion_evidence), ["pot_not_reconciled"]
    )

    db.update_hand_completion(
        hand.id, completion_evidence=dump_completion_evidence(evidence)
    )

    assert not any(
        review.is_stale for review in db.fetch_coaching_reviews_by_hand(hand.id)
    )
    db.close()


def test_correcting_facts_demotes_a_complete_hand_to_uncertain() -> None:
    db = make_db()
    session = db.create_session(Session(name="Demote"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            hero_cards="Ah Kd",
        )
    )

    db.update_hand_facts(hand.model_copy(update={"hero_cards": "Ah Qd"}))

    assert db.fetch_hand(hand.id).completion_status == "uncertain"
    db.close()


def test_correcting_facts_leaves_a_partial_hand_partial() -> None:
    db = make_db()
    session = db.create_session(Session(name="Sticky"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="partial",
            hero_cards="Ah Kd",
        )
    )

    db.update_hand_facts(hand.model_copy(update={"hero_cards": "Ah Qd"}))

    assert db.fetch_hand(hand.id).completion_status == "partial"
    db.close()


def test_correcting_facts_leaves_a_manual_hand_not_applicable() -> None:
    db = make_db()
    session = db.create_session(Session(name="Manual facts"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, hero_cards="Ah Kd")
    )

    db.update_hand_facts(hand.model_copy(update={"hero_cards": "Ah Qd"}))

    assert db.fetch_hand(hand.id).completion_status == "not_applicable"
    db.close()


def test_flagging_for_debugging_demotes_completion() -> None:
    from poker_tracker.persistence.models import HandIssue

    db = make_db()
    session = db.create_session(Session(name="Flag"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
        )
    )
    partial = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            source_type="cv_import",
            completion_status="partial",
        )
    )

    for target in (hand, partial):
        db.create_hand_issue(
            HandIssue(
                hand_id=target.id,
                issue_types=["hand_boundary"],
                description="Boundary looks wrong.",
            )
        )

    assert db.fetch_hand(hand.id).completion_status == "uncertain"
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    assert db.fetch_hand(partial.id).completion_status == "partial"
    db.close()


def test_accounting_evidence_update_demotes_a_complete_hand() -> None:
    db = make_db()
    session = db.create_session(Session(name="Accounting demote"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
        )
    )

    db.update_hand_accounting_evidence(hand.id, pot_size=42, hero_bb_won=7)

    assert db.fetch_hand(hand.id).completion_status == "uncertain"
    db.close()
