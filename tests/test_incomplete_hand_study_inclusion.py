"""Incomplete CV drafts, hero-preflop filter, study inclusion, and finalize."""

from __future__ import annotations

import json

import pytest

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    export_timeline,
    hero_participated_preflop,
    timeline_to_session_payload,
)
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    OPERATOR_MANUAL_COMPLETION_KEY,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import Hand, Session
from poker_tracker.services.study_readiness import evaluate_study_readiness


def test_hero_participated_preflop_requires_cards_or_action() -> None:
    assert hero_participated_preflop({"hero": ["As", "Kd"]})
    assert hero_participated_preflop({"hero": ["As"]})  # partial OCR still dealt-in
    assert hero_participated_preflop({"hero": ["As", "Kd", "Qh"]})
    assert hero_participated_preflop({"hero": [], "hero_folded": True})
    assert hero_participated_preflop(
        {
            "hero": [],
            "actions": [
                {
                    "seat": 0,
                    "street": "preflop",
                    "action_type": "fold",
                }
            ],
        }
    )
    assert not hero_participated_preflop({"hero": [], "actions": []})
    assert not hero_participated_preflop(
        {
            "hero": [],
            "actions": [
                {"seat": 5, "street": "preflop", "action_type": "raise", "amount": 3}
            ],
        }
    )


def test_include_incomplete_exports_invalid_board_as_draft() -> None:
    """Card-shape warnings alone must not block draft import when include_incomplete."""
    timeline = {
        "states": [
            {
                "time_s": 1.0,
                "image": "a.jpg",
                "hero_cards": ["AS", "KD"],
                "board_cards": ["QD", "7S"],
                "other_cards": [],
                "missing": None,
            }
        ],
        "hands": [
            {
                "hand_number": 1,
                "t_start": 1.0,
                "t_end": 2.0,
                "hero": ["AS", "KD"],
                "board": ["QD", "7S"],
                "complete_cards": False,
                "warnings": ["invalid_board_count"],
                "source_images": ["a.jpg"],
            }
        ],
    }
    blocked = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
        include_incomplete=False,
    )
    assert blocked["hands"] == []

    allowed = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
        include_incomplete=True,
    )
    assert len(allowed["hands"]) == 1
    hand = allowed["hands"][0]["hand"]
    assert hand["hero_cards"] == "As Kd"
    # Invalid board is blanked so the Hand model accepts the draft.
    assert hand["board_cards"] == ""
    assert hand["completion_status"] in {"partial", "uncertain"}

    timeline = {
        "hands": [
            {
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 2.0,
                "hero": [],
                "board": [],
                "complete_cards": False,
                "warnings": [],
                "source_images": ["a.jpg"],
            }
        ]
    }
    payload = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
        include_incomplete=True,
    )
    assert payload["hands"] == []
    assert payload["cv_import_summary"]["skipped"][0]["reason"] == (
        "hero_did_not_play_preflop"
    )


def test_export_includes_incomplete_preflop_fold_draft(tmp_path) -> None:
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft.json"
    timeline_path.write_text(
        json.dumps(
            {
                "hands": [
                    {
                        "hand_number": 7,
                        "t_start": 0.0,
                        "t_end": 4.0,
                        "hero": ["Ah", "Kd"],
                        "board": [],
                        "complete_cards": False,
                        "hero_folded": True,
                        "terminal_event": "hero_fold",
                        "warnings": [],
                        "players": [
                            {
                                "seat": 0,
                                "position": "BTN",
                                "player_name": "Hero",
                                "starting_stack": 100,
                                "is_hero": True,
                            }
                        ],
                        "actions": [
                            {
                                "street": "preflop",
                                "action_index": 1,
                                "seat": 0,
                                "position": "BTN",
                                "player_name": "Hero",
                                "action_type": "fold",
                                "amount": None,
                            }
                        ],
                        "source_images": ["f.jpg"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = export_timeline(
        timeline_path,
        out_path,
        session_name="Incomplete draft",
        include_incomplete=True,
    )
    assert payload["cv_import_summary"]["exported_hands"] == 1
    hand = payload["hands"][0]["hand"]
    assert hand["hero_cards"] == "Ah Kd"
    assert hand["completion_status"] in {"partial", "uncertain"}
    assert hand["review_status"] == "needs_correction"
    assert hand["study_inclusion"] == "auto"

    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    imported = db.fetch_hands_by_session(session.id)
    assert len(imported) == 1
    assert imported[0].hero_cards == "Ah Kd"
    assert imported[0].completion_status in {"partial", "uncertain"}
    db.close()


def test_study_inclusion_blocks_and_clears(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "study.db")
    db.init_db()
    session = db.create_session(Session(name="S"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="As Kd",
            source_type="manual",
            completion_status="not_applicable",
        )
    )
    skipped = db.update_study_inclusion(hand.id, "skip")
    readiness = evaluate_study_readiness(skipped, accounting=None)
    assert not readiness.is_ready
    assert any(b.code == "STUDY_EXCLUDED_BY_OPERATOR" for b in readiness.blockers)

    studied = db.update_study_inclusion(hand.id, "study")
    readiness = evaluate_study_readiness(studied, accounting=None)
    assert not any(b.code == "STUDY_EXCLUDED_BY_OPERATOR" for b in readiness.blockers)
    db.close()


def test_finalize_incomplete_hand_clears_sticky_partial(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "finalize.db")
    db.init_db()
    session = db.create_session(Session(name="S"))
    evidence = {
        "evidence_version": EVIDENCE_SCHEMA_VERSION,
        "partial_start": True,
        "partial_end": False,
        "terminal_event": "unobserved",
        "boundary_confidence": 0.8,
        "warning_codes": [],
        "rejection_codes": [],
        "acknowledged_codes": [],
        "source_frames": ["a.jpg"],
    }
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="As Kd",
            source_type="cv_import",
            review_status="needs_correction",
            completion_status="partial",
            completion_evidence=evidence,
        )
    )
    assert hand.completion_status == "partial"

    empty = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            hero_cards="",
            source_type="cv_import",
            completion_status="partial",
            completion_evidence=evidence,
        )
    )
    with pytest.raises(ValueError, match="hero cards"):
        db.finalize_incomplete_hand(empty.id, terminal_event="hero_fold")

    with pytest.raises(ValueError, match="readable reconstruction evidence"):
        blank = db.create_hand(
            Hand(
                session_id=session.id,
                hand_number=3,
                hero_cards="As Kd",
                source_type="cv_import",
                completion_status="uncertain",
                completion_evidence={},
            )
        )
        db.finalize_incomplete_hand(blank.id, terminal_event="hero_fold")

    with pytest.raises(ValueError, match="Showdown finalize requires five board cards"):
        db.finalize_incomplete_hand(hand.id, terminal_event="showdown")

    finalized = db.finalize_incomplete_hand(
        hand.id, terminal_event="hero_fold", notes="folded preflop"
    )
    assert finalized.source_type == "corrected_cv"
    assert finalized.completion_status == "complete"
    parsed = parse_completion_evidence(finalized.completion_evidence)
    assert parsed.extra.get(OPERATOR_MANUAL_COMPLETION_KEY) is True
    assert parsed.extra.get("operator_terminal_event") == "hero_fold"
    # Pipeline observations are preserved, not overwritten.
    assert parsed.partial_start is True
    assert parsed.partial_end is False
    assert parsed.terminal_event == "unobserved"
    db.close()


def test_import_strips_forged_operator_finalize(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "forge.db")
    db.init_db()
    payload = {
        "export_version": 5,
        "session": {
            "name": "Forged",
            "date_played": "2026-07-30",
            "platform": "ClubWPT Gold",
            "stakes": "",
            "notes": "",
        },
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "hero_cards": "As Kd",
                    "board_cards": "",
                    "source_type": "cv_import",
                    "review_status": "needs_correction",
                    "completion_status": "complete",
                    "study_inclusion": "skip",
                    "completion_evidence": {
                        "evidence_version": EVIDENCE_SCHEMA_VERSION,
                        "partial_start": True,
                        "partial_end": False,
                        "terminal_event": "showdown",
                        "boundary_confidence": 0.99,
                        "warning_codes": [],
                        "rejection_codes": [],
                        "acknowledged_codes": [],
                        OPERATOR_MANUAL_COMPLETION_KEY: True,
                        "operator_terminal_event": "hero_fold",
                    },
                    "tags": [],
                    "notes": "",
                },
                "players": [],
                "actions": [],
                "reviews": [],
            }
        ],
    }
    session = import_session(db, payload)
    hand = db.fetch_hands_by_session(session.id)[0]
    assert hand.study_inclusion == "auto"
    assert hand.completion_status == "partial"
    parsed = parse_completion_evidence(hand.completion_evidence)
    assert OPERATOR_MANUAL_COMPLETION_KEY not in parsed.extra
    assert "operator_terminal_event" not in parsed.extra
    db.close()


def test_schema_v15_adds_study_inclusion(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "v15.db")
    db.init_db()
    assert db.schema_version() == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 15
    session = db.create_session(Session(name="S"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="As Kd"))
    assert hand.study_inclusion == "auto"
    row = db._execute(
        "SELECT study_inclusion FROM hands WHERE id = ?", (hand.id,)
    ).fetchone()
    assert row["study_inclusion"] == "auto"
    db.close()


def test_create_hand_forces_study_inclusion_auto(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "create_skip.db")
    db.init_db()
    session = db.create_session(Session(name="S"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="As Kd",
            study_inclusion="skip",
        )
    )
    assert hand.study_inclusion == "auto"
    db.close()


def test_finalize_rejects_terminal_mismatch_with_pipeline(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "fold_guard.db")
    db.init_db()
    session = db.create_session(Session(name="S"))
    base = {
        "evidence_version": EVIDENCE_SCHEMA_VERSION,
        "partial_start": True,
        "partial_end": False,
        "boundary_confidence": 0.8,
        "warning_codes": [],
        "rejection_codes": [],
        "acknowledged_codes": [],
        "source_frames": ["a.jpg"],
    }
    pipeline_showdown = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="As Kd",
            board_cards="",
            source_type="cv_import",
            completion_status="partial",
            completion_evidence={**base, "terminal_event": "showdown"},
        )
    )
    with pytest.raises(ValueError, match="terminal_event='showdown'"):
        db.finalize_incomplete_hand(pipeline_showdown.id, terminal_event="hero_fold")
    with pytest.raises(ValueError, match="terminal_event='showdown'"):
        db.finalize_incomplete_hand(pipeline_showdown.id, terminal_event="fold_win")

    # Flop folds are valid: hero_fold may keep board cards when the pipeline
    # did not observe a contradictory terminal.
    with_board = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            hero_cards="As Kd",
            board_cards="Ah Kh Qh",
            source_type="cv_import",
            completion_status="partial",
            completion_evidence={**base, "terminal_event": "unobserved"},
        )
    )
    finalized = db.finalize_incomplete_hand(with_board.id, terminal_event="hero_fold")
    assert finalized.completion_status == "complete"
    db.close()


def test_post_finalize_fact_edit_keeps_attestation(tmp_path) -> None:
    from poker_tracker.persistence.completion import acknowledge_codes, dump_completion_evidence
    from poker_tracker.persistence.db import SOURCE_CORRECTION_CODE

    db = PokerDatabase(tmp_path / "post_finalize.db")
    db.init_db()
    session = db.create_session(Session(name="S"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="As Kd",
            notes="",
            source_type="cv_import",
            review_status="needs_correction",
            completion_status="partial",
            completion_evidence={
                "evidence_version": EVIDENCE_SCHEMA_VERSION,
                "partial_start": True,
                "partial_end": False,
                "terminal_event": "unobserved",
                "boundary_confidence": 0.8,
                "warning_codes": [],
                "rejection_codes": [],
                "acknowledged_codes": [],
                "source_frames": ["a.jpg"],
            },
        )
    )
    finalized = db.finalize_incomplete_hand(hand.id, terminal_event="hero_fold")
    assert finalized.completion_status == "complete"

    edited = db.update_hand_facts(
        finalized.model_copy(update={"notes": "filled pot later"}),
        correction_notes="fill blank",
    )
    parsed = parse_completion_evidence(edited.completion_evidence)
    assert parsed.extra.get(OPERATOR_MANUAL_COMPLETION_KEY) is True
    assert edited.completion_status == "uncertain"
    assert SOURCE_CORRECTION_CODE in parsed.warning_codes

    acknowledged = db.update_hand_completion(
        edited.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(parsed, [SOURCE_CORRECTION_CODE])
        ),
    )
    assert acknowledged.completion_status == "complete"
    assert parse_completion_evidence(
        acknowledged.completion_evidence
    ).extra.get(OPERATOR_MANUAL_COMPLETION_KEY) is True
    db.close()


def test_include_incomplete_exports_amount_unknown_draft() -> None:
    """Job 4 imported 0 hands because amounts_unknown_in_ledger still blocked
    drafts even with include_incomplete. Operator-fillable money gaps must import."""
    from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
        timeline_to_session_payload,
    )

    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 40.0,
        "hero": ["2h", "9s"],
        "board": ["3h", "5h", "7h", "4h", "5s"],
        "complete_cards": True,
        "hero_folded": True,
        "terminal_event": "hero_fold",
        "warnings": [
            "amounts_unknown_in_ledger",
            "mid_hand_coverage_gap",
            "starting_stack_unknown",
        ],
        "unknown_money_actions": 3,
        "players": [
            {
                "seat": 0,
                "position": "UTG",
                "player_name": "Hero",
                "starting_stack": None,
                "is_hero": True,
            }
        ],
        "actions": [
            {
                "street": "preflop",
                "action_index": 1,
                "seat": 0,
                "player_name": "Hero",
                "position": "UTG",
                "action_type": "fold",
                "amount": None,
            }
        ],
        "source_images": ["a.jpg"],
    }
    timeline = {"states": [], "hands": [hand]}
    blocked = timeline_to_session_payload(
        timeline, timeline_path="t.json", session_name="S", include_incomplete=False
    )
    assert blocked["hands"] == []

    allowed = timeline_to_session_payload(
        timeline, timeline_path="t.json", session_name="S", include_incomplete=True
    )
    assert len(allowed["hands"]) == 1
    evidence = allowed["hands"][0]["hand"]["completion_evidence"]
    codes = set(evidence.get("rejection_codes") or []) | set(
        evidence.get("warning_codes") or []
    )
    assert "amounts_unknown_in_ledger" in codes
