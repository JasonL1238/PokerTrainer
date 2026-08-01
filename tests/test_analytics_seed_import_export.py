import json

import pytest

from poker_tracker.math.analytics import compute_session_stats
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    IMPORTED_HAND_KEY,
    CompletionEvidence,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    EXPORT_VERSION,
    SUPPORTED_IMPORT_VERSIONS,
    ImportValidationError,
    export_hand,
    export_session,
    import_hands_into_session,
    import_session,
)
from poker_tracker.persistence.models import Action, Hand, Session
from poker_tracker.services.study_readiness import evaluate_study_readiness
from tests.fixtures.seed_data import create_sample_data


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _evidence_beyond_import_provenance(hand: Hand) -> dict[str, object]:
    """Everything the import stored EXCEPT the provenance stamp it always adds.

    ``import_session`` marks every hand it lands with ``IMPORTED_HAND_KEY``,
    because the manual-hand exemption is the argument "you entered this hand
    yourself" and that is false for a hand which arrived as user-supplied JSON.
    The marker carries no ``evidence_version``, so it is not reconstruction
    evidence; the assertions below still require that import fabricates none.
    """
    assert hand.completion_evidence.get(IMPORTED_HAND_KEY) is True
    assert not parse_completion_evidence(hand.completion_evidence).is_known
    return {
        key: value
        for key, value in hand.completion_evidence.items()
        if key != IMPORTED_HAND_KEY
    }


def test_analytics_calculations() -> None:
    db = make_db()
    session = db.create_session(Session(name="Stats"))
    win = db.create_hand(
        Hand(session_id=session.id, hand_number=1, hero_bb_won=10, tags=["BIG_POT"])
    )
    loss = db.create_hand(
        Hand(session_id=session.id, hand_number=2, hero_bb_won=-4, tags=["MULTIWAY"])
    )
    db.create_action(Action(hand_id=win.id, street="flop", player_name="Hero", action_type="bet"))
    db.create_action(Action(hand_id=loss.id, street="turn", player_name="Hero", action_type="call"))
    # Promoted after its actions exist: adding an action to a promoted hand now
    # returns it to needs_correction, so 'reviewed' cannot outlive its evidence.
    db.update_hand_status(win.id, "reviewed")

    stats = compute_session_stats(db, session.id)

    assert stats.hand_count == 2
    assert stats.total_hero_bb == 6
    assert stats.average_hero_bb == 3
    assert stats.biggest_winning_hands[0].hand_number == 1
    assert stats.biggest_losing_hands[0].hand_number == 2
    assert stats.hands_by_tag == {"BIG_POT": 1, "MULTIWAY": 1}
    assert stats.hands_by_review_status["reviewed"] == 1
    assert stats.hands_by_review_status["unreviewed"] == 1
    assert stats.action_counts_by_type == {"bet": 1, "call": 1}
    assert stats.aggression_count == 1
    assert stats.passive_count == 1

    db.close()


def test_seed_data_creation() -> None:
    db = make_db()
    session = create_sample_data(db)
    hands = db.fetch_hands_by_session(session.id)
    stats = compute_session_stats(db, session.id)

    assert len(hands) == 5
    assert {"MISSED_VALUE", "MULTIWAY", "PREFLOP_3BET_SPOT"}.issubset(stats.hands_by_tag)
    assert stats.biggest_losing_hands[0].hero_bb_won == -65
    for hand in hands:
        monetary_actions = [
            action for action in db.fetch_actions_by_hand(hand.id) if action.amount is not None
        ]
        assert monetary_actions
        assert all(action.amount_semantics == "unknown" for action in monetary_actions)

    db.close()


def test_json_export_for_hand_and_session() -> None:
    db = make_db()
    session = create_sample_data(db)
    hand = db.fetch_hands_by_session(session.id)[0]

    hand_payload = export_hand(db, hand.id)
    session_payload = export_session(db, session.id)

    assert hand_payload["hand"]["hand_number"] == 1
    assert hand_payload["actions"]
    assert session_payload["session"]["name"] == "Sample post-session review"
    assert len(session_payload["hands"]) == 5
    json.dumps(session_payload)

    db.close()


def test_json_export_rejects_missing_ids() -> None:
    db = make_db()

    with pytest.raises(ValueError, match="Hand not found"):
        export_hand(db, 9999)

    with pytest.raises(ValueError, match="Session not found"):
        export_session(db, 9999)

    db.close()


def test_json_import_round_trip() -> None:
    source = make_db()
    session = create_sample_data(source)
    payload = export_session(source, session.id)

    target = make_db()
    imported = import_session(target, payload)

    assert imported.id is not None
    assert len(target.fetch_hands_by_session(imported.id)) == 5
    imported_first = target.fetch_hands_by_session(imported.id)[0]
    assert target.fetch_actions_by_hand(imported_first.id)
    assert target.fetch_players_by_hand(imported_first.id)
    assert imported_first.tags

    source.close()
    target.close()


def test_json_import_preserves_review_status_tags_and_action_order() -> None:
    source = make_db()
    session = source.create_session(Session(name="Import details"))
    hand = source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            review_status="needs_correction",
            tags=["MISSED_VALUE", "RIVER_DECISION"],
        )
    )
    source.create_action(Action(hand_id=hand.id, street="river", player_name="Hero", action_type="check"))
    source.create_action(Action(hand_id=hand.id, street="river", player_name="Villain", action_type="bet", amount=12))
    payload = export_session(source, session.id)

    target = make_db()
    imported = import_session(target, payload)
    imported_hand = target.fetch_hands_by_session(imported.id)[0]
    imported_actions = target.fetch_actions_by_hand(imported_hand.id)

    assert imported_hand.review_status == "needs_correction"
    assert imported_hand.tags == ["MISSED_VALUE", "RIVER_DECISION"]
    assert [action.action_index for action in imported_actions] == [1, 2]
    assert [action.player_name for action in imported_actions] == ["Hero", "Villain"]

    source.close()
    target.close()


def test_json_import_rejects_bad_payload_without_creating_session() -> None:
    """A truncated payload names the missing part, rather than raising KeyError.

    ``payload["session"]`` used to raise a bare ``KeyError('session')``. It was
    caught, so nothing broke, but the operator was shown a one-word key name for
    a file they may have spent an hour producing.
    """
    db = make_db()

    with pytest.raises(ImportValidationError, match="no 'session' object"):
        import_session(db, {"hands": []})

    assert db.fetch_sessions() == []
    db.close()


def test_json_import_rejects_invalid_hand_without_partial_session() -> None:
    db = make_db()
    payload = {
        "session": Session(name="Bad import").model_dump(mode="json"),
        "hands": [
            {
                "hand": Hand(session_id=1, hand_number=1, hero_cards="Ah Qs").model_dump(mode="json")
                | {"board_cards": "Ah 7d 2c"},
                "players": [],
                "actions": [],
                "reviews": [],
            }
        ],
    }

    # The pydantic detail is preserved verbatim; what is added is which record of
    # the file carried it.
    with pytest.raises(ImportValidationError, match=r"hands\[0\].hand: .*validation error"):
        import_session(db, payload)

    assert db.fetch_sessions() == []
    db.close()


def _complete_evidence() -> dict:
    """Evidence that derive_completion_status classifies as complete."""
    return dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            first_source_timestamp_s=10.0,
            last_source_timestamp_s=48.5,
            boundary_confidence=0.92,
            source_frames=("frames/a.png", "frames/b.png"),
            layout_profile="clubwpt-6max",
            layout_supported=True,
            table_size=6,
            pipeline_version="two-model-v7",
            model_versions={"detector": "v7"},
        )
    )


def _payload_with_hand(hand_data: dict, version: int) -> dict:
    return {
        "export_version": version,
        "session": Session(name=f"v{version} import").model_dump(mode="json"),
        "hands": [
            {
                "hand": hand_data,
                "players": [],
                "actions": [],
                "reviews": [],
            }
        ],
    }


def _legacy_hand_payload(source_type: str, review_status: str = "reviewed") -> dict:
    """A pre-v5 hand payload: no completion keys at all."""
    data = Hand(
        session_id=1,
        hand_number=1,
        hero_cards="Ah Qs",
        source_type=source_type,
        review_status=review_status,
    ).model_dump(mode="json")
    data.pop("completion_status", None)
    data.pop("completion_evidence", None)
    return data


def test_export_version_is_six_and_every_older_version_still_imports() -> None:
    assert EXPORT_VERSION == 6
    # Every previously released version must stay importable: bumping the
    # export version alone would silently orphan payloads already on disk.
    assert SUPPORTED_IMPORT_VERSIONS == {1, 2, 3, 4, 5, 6}


def test_export_includes_completion_status_and_evidence() -> None:
    db = make_db()
    session = db.create_session(Session(name="Completion export"))
    evidence = _complete_evidence()
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=evidence,
        )
    )

    hand_payload = export_hand(db, hand.id)
    session_payload = export_session(db, session.id)

    assert hand_payload["export_version"] == EXPORT_VERSION
    assert hand_payload["hand"]["completion_status"] == "complete"
    assert hand_payload["hand"]["completion_evidence"] == evidence
    assert session_payload["hands"][0]["hand"]["completion_evidence"] == evidence
    db.close()


def test_export_payload_is_json_serializable() -> None:
    db = make_db()
    session = db.create_session(Session(name="Serializable"))
    db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_complete_evidence(),
        )
    )

    json.dumps(export_session(db, session.id))
    db.close()


def test_import_v5_round_trips_completion_status_and_evidence() -> None:
    source = make_db()
    session = source.create_session(Session(name="Round trip"))
    evidence = _complete_evidence()
    source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            review_status="reviewed",
            completion_status="complete",
            completion_evidence=evidence,
        )
    )
    payload = export_session(source, session.id)

    target = make_db()
    imported = import_session(target, payload)
    hand = target.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "complete"
    # Every field round-trips; the only addition is the import provenance stamp.
    assert hand.completion_evidence == {**evidence, IMPORTED_HAND_KEY: True}
    # Completion round-trips; the promotion does not. Readiness additionally
    # requires explicit user confirmation, which is per-render and never
    # persisted, so it cannot travel in a payload.
    assert hand.review_status == "needs_correction"
    assert parse_completion_evidence(hand.completion_evidence).is_known
    assert (
        derive_completion_status(
            parse_completion_evidence(hand.completion_evidence),
            source_type="cv_import",
        )
        == "complete"
    )
    source.close()
    target.close()


def test_import_v1_manual_hand_defaults_to_not_applicable() -> None:
    db = make_db()
    payload = _payload_with_hand(_legacy_hand_payload("manual"), 1)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "not_applicable"
    assert _evidence_beyond_import_provenance(hand) == {}
    # Round 13: `reviewed` is this database operator's attestation and cannot
    # travel in a payload for ANY declared source_type -- a payload relabelled
    # `manual` was re-promoted straight to `reviewed` by the floor this used to
    # pin. The label is one tick and one save away for the operator who now
    # vouches for the hand.
    assert hand.review_status == "needs_correction"
    db.close()


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("source_type", ["cv_import", "corrected_cv"])
def test_import_legacy_cv_hand_defaults_to_uncertain_and_needs_correction(
    version: int, source_type: str
) -> None:
    db = make_db()
    payload = _payload_with_hand(_legacy_hand_payload(source_type), version)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "uncertain"
    assert hand.review_status == "needs_correction"
    assert _evidence_beyond_import_provenance(hand) == {}
    db.close()


def test_import_v4_payload_is_still_accepted() -> None:
    db = make_db()
    payload = _payload_with_hand(_legacy_hand_payload("manual"), 4)
    payload["export_version"] = 4

    imported = import_session(db, payload)

    assert payload["export_version"] == 4
    assert len(db.fetch_hands_by_session(imported.id)) == 1
    db.close()


def test_import_v5_rejects_reconstructed_hand_declared_not_applicable() -> None:
    db = make_db()
    hand_data = Hand(
        session_id=1, hand_number=1, source_type="cv_import"
    ).model_dump(mode="json")
    hand_data["completion_status"] = "not_applicable"
    payload = _payload_with_hand(hand_data, 5)

    with pytest.raises(ValueError, match="not_applicable"):
        import_session(db, payload)

    assert db.fetch_sessions() == []
    db.close()


def test_import_v5_normalizes_manual_hand_to_not_applicable() -> None:
    db = make_db()
    hand_data = Hand(session_id=1, hand_number=1).model_dump(mode="json")
    hand_data["completion_status"] = "complete"
    payload = _payload_with_hand(hand_data, 5)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.source_type == "manual"
    assert hand.completion_status == "not_applicable"
    db.close()


def test_import_v5_refuses_a_manual_hand_carrying_reconstruction_evidence() -> None:
    """'manual' + 'not_applicable' is the blanket exemption; it must be earned.

    Relabelling a reconstructed hand's source_type as 'manual' was enough to skip
    every completion, layout, source-warning and confirmation blocker, because
    source_type was the one field the invariant trusted verbatim.
    """
    db = make_db()
    hand_data = Hand(session_id=1, hand_number=1).model_dump(mode="json")
    hand_data["completion_evidence"] = _complete_evidence()
    payload = _payload_with_hand(hand_data, 5)

    with pytest.raises(ValueError, match="source_type 'manual'"):
        import_session(db, payload)

    assert db.fetch_sessions() == []
    db.close()


def test_import_v5_downgrades_reviewed_status_on_an_uncertain_hand() -> None:
    db = make_db()
    hand_data = Hand(
        session_id=1,
        hand_number=1,
        source_type="cv_import",
        review_status="needs_correction",
    ).model_dump(mode="json")
    hand_data["completion_status"] = "uncertain"
    hand_data["review_status"] = "reviewed"
    payload = _payload_with_hand(hand_data, 5)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "uncertain"
    assert hand.review_status == "needs_correction"
    db.close()


def test_import_v5_downgrades_reviewed_status_on_a_partial_hand() -> None:
    """A declared 'partial' with no evidence is kept, and its 'reviewed' refused.

    The declared status is re-derived, and import may only ever WEAKEN what the
    payload claimed. An evidence-free claim re-derives to 'uncertain', which is
    weaker than the declared, permanent 'partial', so the declared status stands;
    a claim backed by truncation evidence stays 'partial' the same way (see
    tests/test_phase1_readiness_bypass.py). This used to land as the clearable
    'uncertain', which one Acknowledge click walked to 'complete'
    (tests/test_phase1_adversarial_round3.py). Either way the hand is blocked and
    its declared 'reviewed' is refused.
    """
    db = make_db()
    hand_data = Hand(
        session_id=1,
        hand_number=1,
        source_type="cv_import",
        review_status="needs_correction",
    ).model_dump(mode="json")
    hand_data["completion_status"] = "partial"
    hand_data["review_status"] = "reviewed"
    payload = _payload_with_hand(hand_data, 5)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "partial"
    assert hand.review_status == "needs_correction"
    db.close()


def test_import_v5_keeps_partial_when_the_evidence_proves_truncation() -> None:
    db = make_db()
    hand_data = Hand(
        session_id=1,
        hand_number=1,
        source_type="cv_import",
        review_status="needs_correction",
    ).model_dump(mode="json")
    hand_data["completion_status"] = "complete"
    hand_data["review_status"] = "reviewed"
    hand_data["completion_evidence"] = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=True,
            terminal_event="showdown",
            boundary_confidence=0.9,
        )
    )
    payload = _payload_with_hand(hand_data, 5)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "partial"
    assert hand.review_status == "needs_correction"
    db.close()


@pytest.mark.parametrize("evidence", ["not-json", 7, ["a"], None])
def test_import_v5_with_corrupt_evidence_stores_empty_evidence(evidence: object) -> None:
    db = make_db()
    hand_data = Hand(
        session_id=1, hand_number=1, source_type="cv_import"
    ).model_dump(mode="json")
    hand_data["completion_status"] = "uncertain"
    hand_data["completion_evidence"] = evidence
    payload = _payload_with_hand(hand_data, 5)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert _evidence_beyond_import_provenance(hand) == {}
    db.close()


def test_import_rejects_an_unreleased_future_export_version() -> None:
    db = make_db()
    payload = _payload_with_hand(_legacy_hand_payload("manual"), 7)

    with pytest.raises(ValueError, match="Unsupported export_version 7"):
        import_session(db, payload)

    assert db.fetch_sessions() == []
    db.close()


def test_import_rejection_creates_no_partial_session() -> None:
    db = make_db()
    good = Hand(
        session_id=1, hand_number=1, hero_cards="Ah Qs"
    ).model_dump(mode="json")
    bad = Hand(session_id=1, hand_number=2, source_type="cv_import").model_dump(
        mode="json"
    )
    bad["completion_status"] = "not_applicable"
    payload = {
        "export_version": 5,
        "session": Session(name="Half bad").model_dump(mode="json"),
        "hands": [
            {"hand": good, "players": [], "actions": [], "reviews": []},
            {"hand": bad, "players": [], "actions": [], "reviews": []},
        ],
    }

    with pytest.raises(ValueError):
        import_session(db, payload)

    assert db.fetch_sessions() == []
    assert db.fetch_all_hands() == []
    db.close()


def test_import_still_creates_a_duplicate_session_on_reimport() -> None:
    source = make_db()
    session = create_sample_data(source)
    payload = export_session(source, session.id)

    target = make_db()
    first = import_session(target, payload)
    second = import_session(target, payload)

    assert first.id != second.id
    assert len(target.fetch_sessions()) == 2
    assert len(target.fetch_hands_by_session(first.id)) == 5
    assert len(target.fetch_hands_by_session(second.id)) == 5
    source.close()
    target.close()


def test_import_hands_into_session_inherits_completion_defaults() -> None:
    db = make_db()
    target = db.create_session(Session(name="Existing session"))
    payload = _payload_with_hand(_legacy_hand_payload("cv_import"), 3)

    refreshed = import_hands_into_session(db, payload, target.id)
    hands = db.fetch_hands_by_session(refreshed.id)

    assert len(hands) == 1
    assert hands[0].completion_status == "uncertain"
    assert hands[0].review_status == "needs_correction"
    assert len(db.fetch_sessions()) == 1
    db.close()


# ---------------------------------------------------------------------------
# Per-version conservative defaults, end to end into study readiness.
#
# The tests above check the persisted fields. These check the product
# consequence: an old payload must arrive blocked, not merely annotated.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_import_manual_hand_defaults_to_not_applicable_for_every_legacy_version(
    version: int,
) -> None:
    db = make_db()
    payload = _payload_with_hand(_legacy_hand_payload("manual"), version)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.completion_status == "not_applicable"
    assert _evidence_beyond_import_provenance(hand) == {}
    # Round 13: a manual hand's declared review status IS rewritten by the
    # import, exactly as a reconstructed hand's is. The claim `manual` arrives
    # in the same user-supplied JSON as the claim `reviewed`, so the first
    # cannot be the reason to trust the second; the importing operator re-earns
    # the label with one tick and one save.
    assert hand.review_status == "needs_correction"
    db.close()


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_a_legacy_cv_import_arrives_blocked_for_study(version: int) -> None:
    db = make_db()
    payload = _payload_with_hand(_legacy_hand_payload("cv_import"), version)

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]
    readiness = evaluate_study_readiness(hand, accounting=None, user_confirmed=True)

    assert readiness.is_ready is False
    assert set(readiness.codes()) == {
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
        "UNSUPPORTED_TABLE_LAYOUT",
        "ACCOUNTING_NOT_AUTHORITATIVE",
    }
    db.close()


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_a_legacy_payload_cannot_smuggle_in_a_reviewed_cv_hand(version: int) -> None:
    """The import writes through create_hand, which bypasses update_hand_status."""
    db = make_db()
    payload = _payload_with_hand(
        _legacy_hand_payload("corrected_cv", review_status="reviewed"), version
    )

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]

    assert hand.review_status == "needs_correction"
    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        db.update_hand_status(hand.id, "reviewed")
    db.close()


def test_a_v5_round_trip_survives_a_second_export() -> None:
    """Export -> import -> export adds the import stamp once and never drifts again."""
    source = make_db()
    session = source.create_session(Session(name="Double round trip"))
    evidence = _complete_evidence()
    source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=evidence,
        )
    )
    first = export_session(source, session.id)

    target = make_db()
    imported = import_session(target, first)
    second = export_session(target, imported.id)

    third_db = make_db()
    third = export_session(third_db, import_session(third_db, second).id)

    assert second["export_version"] == first["export_version"] == EXPORT_VERSION
    assert first["hands"][0]["hand"]["completion_evidence"] == evidence
    # The stamp is applied once and is idempotent: a payload that already
    # carries it re-exports byte-identically, so nothing accumulates across an
    # arbitrary number of round trips.
    assert second["hands"][0]["hand"]["completion_evidence"] == {
        **evidence,
        IMPORTED_HAND_KEY: True,
    }
    assert (
        third["hands"][0]["hand"]["completion_evidence"]
        == second["hands"][0]["hand"]["completion_evidence"]
    )
    assert (
        third["hands"][0]["hand"]["completion_status"]
        == second["hands"][0]["hand"]["completion_status"]
        == first["hands"][0]["hand"]["completion_status"]
        == "complete"
    )
    json.dumps(second)
    json.dumps(third)
    source.close()
    target.close()
    third_db.close()
