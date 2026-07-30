import json

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    LOW_CONFIDENCE_AT,
    _card_to_app,
    _confidence_for_hand,
    apply_hand_corrections,
    export_timeline,
    load_hand_corrections,
    timeline_to_session_payload,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_session


def test_card_to_app_normalizes_detector_labels() -> None:
    assert _card_to_app("AS") == "As"
    assert _card_to_app("10H") == "Th"
    assert _card_to_app("td") == "Td"


def test_timeline_to_session_payload_exports_only_valid_complete_hands() -> None:
    timeline = {
        "hands": [
            {
                "hand_number": 7,
                "t_start": 1.0,
                "t_end": 9.0,
                "hero": ["AS", "10H"],
                "board": ["QD", "7S", "2C"],
                "complete_cards": True,
                "warnings": [],
                "source_images": ["images/val/frame_000001.jpg"],
            },
            {
                "hand_number": 8,
                "t_start": 10.0,
                "t_end": 11.0,
                "hero": ["AS"],
                "board": ["QD", "7S"],
                "complete_cards": False,
                "warnings": ["hero_cards_not_two", "invalid_board_count"],
                "source_images": [],
            },
            {
                "hand_number": 9,
                "t_start": 12.0,
                "t_end": 13.0,
                "hero": ["AS", "10H"],
                "board": ["AS", "7S", "2C"],
                "complete_cards": False,
                "warnings": ["duplicate_visible_cards"],
                "source_images": [],
            },
        ]
    }

    payload = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
    )

    assert payload["session"]["name"] == "Draft"
    assert len(payload["hands"]) == 1
    assert payload["hands"][0]["hand"]["hero_cards"] == "As Th"
    assert payload["hands"][0]["hand"]["board_cards"] == "Qd 7s 2c"
    assert payload["hands"][0]["hand"]["source_type"] == "cv_import"
    assert payload["hands"][0]["hand"]["review_status"] == "needs_correction"
    assert payload["cv_import_summary"]["skipped_hands"] == 2


def test_timeline_to_session_payload_skips_validation_warning_hands() -> None:
    # A mid-hand board regression (board vanishes then comes back) is a real
    # sequence problem and must keep blocking export. (A transient state-level
    # duplicate that the hand's final voted cards resolve no longer warns.)
    timeline = {
        "states": [
            {
                "time_s": 1.0,
                "image": "a.jpg",
                "hero_cards": ["AS", "10H"],
                "board_cards": ["QD", "7S", "2C"],
                "other_cards": [],
                "missing": None,
            },
            {
                "time_s": 2.0,
                "image": "b.jpg",
                "hero_cards": ["AS", "10H"],
                "board_cards": [],
                "other_cards": [],
                "missing": None,
            },
            {
                "time_s": 3.0,
                "image": "c.jpg",
                "hero_cards": ["AS", "10H"],
                "board_cards": ["QD", "7S", "2C"],
                "other_cards": [],
                "missing": None,
            },
        ],
        "hands": [
            {
                "hand_number": 1,
                "t_start": 1.0,
                "t_end": 3.0,
                "hero": ["AS", "10H"],
                "board": ["QD", "7S", "2C"],
                "complete_cards": True,
                "warnings": [],
                "source_images": ["a.jpg", "b.jpg", "c.jpg"],
            }
        ],
    }

    payload = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
    )
    allowed = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
        allow_validation_warnings=True,
    )

    assert payload["hands"] == []
    assert payload["cv_import_summary"]["skipped"][0]["reason"] == "validation_warnings"
    assert allowed["hands"][0]["hand"]["hero_cards"] == "As Th"


def test_hand_corrections_override_warning_hand_for_export(tmp_path) -> None:
    timeline = {
        "states": [
            {
                "time_s": 1.0,
                "image": "a.jpg",
                "hero_cards": ["AS"],
                "board_cards": ["QD", "7S"],
                "other_cards": [],
                "missing": None,
            }
        ],
        "hands": [
            {
                "hand_number": 3,
                "t_start": 1.0,
                "t_end": 1.0,
                "hero": ["AS"],
                "board": ["QD", "7S"],
                "complete_cards": False,
                "warnings": ["invalid_board_count"],
                "source_images": ["a.jpg"],
            }
        ],
    }
    corrections_path = tmp_path / "hand_corrections.csv"
    corrections_path.write_text(
        "hand_number,hero_cards,board_cards,action,notes\n"
        "3,Ah Qs,Qd 7s 2c,,manual fix\n",
        encoding="utf-8",
    )

    corrected = apply_hand_corrections(timeline, load_hand_corrections(corrections_path))
    payload = timeline_to_session_payload(
        corrected,
        timeline_path="timeline.json",
        session_name="Draft",
    )

    assert payload["hands"][0]["hand"]["hero_cards"] == "Ah Qs"
    assert payload["hands"][0]["hand"]["board_cards"] == "Qd 7s 2c"
    assert "manual_correction=keep" in payload["hands"][0]["hand"]["notes"]


def test_export_payload_imports_into_app_database(tmp_path) -> None:
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft_session.json"
    timeline_path.write_text(
        json.dumps(
            {
                "hands": [
                    {
                        "hand_number": 1,
                        "t_start": 1.0,
                        "t_end": 6.0,
                        "hero": ["AH", "QS"],
                        "board": ["QD", "7S", "2C", "9H", "KC"],
                        "complete_cards": True,
                        "warnings": [],
                        "source_images": ["images/train/frame_000001.jpg"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = export_timeline(timeline_path, out_path, session_name="CV Import")
    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    hands = db.fetch_hands_by_session(session.id)

    assert out_path.exists()
    assert len(hands) == 1
    assert hands[0].hero_cards == "Ah Qs"
    assert hands[0].board_cards == "Qd 7s 2c 9h Kc"
    assert hands[0].source_type == "cv_import"
    assert hands[0].review_status == "needs_correction"
    db.close()


def test_export_imports_full_reconstructed_hand(tmp_path) -> None:
    # A reconstruction-spine hand carries players/actions/pot/winner; the exporter
    # must surface all of them into the app DB, not just the cards.
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft_session.json"
    timeline_path.write_text(
        json.dumps({
            "hands": [{
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 8.0,
                "hero": ["As", "Kd"],
                "board": ["2c", "7d", "9h", "Ts", "Jc"],
                "complete_cards": True,
                "warnings": [],
                "players": [
                    {"seat": 0, "position": "SB", "player_name": "Hero", "starting_stack": 100, "is_hero": True},
                    {"seat": 4, "position": "BTN", "player_name": "Seat4", "starting_stack": 100, "is_hero": False},
                ],
                "actions": [
                    {"street": "preflop", "action_index": 1, "seat": 4, "position": "BTN",
                     "player_name": "Seat4", "action_type": "raise", "amount": 3.0,
                     "pot_before": 0.0, "stack_before": 100.0},
                    {"street": "flop", "action_index": 1, "seat": 0, "position": "SB",
                     "player_name": "Hero", "action_type": "bet", "amount": 7.0,
                     "pot_before": 6.0, "stack_before": 97.0},
                ],
                "pot": 20.0,
                "winner_seat": 0,
                "result": "Hero wins",
                "hero_bb_won": 10.0,
                "reconciled": True,
                "source_images": ["f.jpg"],
            }]
        }),
        encoding="utf-8",
    )

    payload = export_timeline(timeline_path, out_path, session_name="CV Import")
    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    hands = db.fetch_hands_by_session(session.id)

    assert len(hands) == 1
    hand = hands[0]
    assert hand.pot_size == 20.0
    assert hand.result == "Hero wins"
    assert hand.hero_bb_won == 10.0
    assert hand.hero_position == "SB"
    assert hand.source_type == "cv_import"

    players = db.fetch_players_by_hand(hand.id)
    actions = db.fetch_actions_by_hand(hand.id)
    assert len(players) == 2
    assert any(p.is_hero for p in players)
    assert {a.action_type for a in actions} == {"raise", "bet"}
    assert {a.amount for a in actions} == {3.0, 7.0}
    db.close()


# --------------------------------------------------------------------------- #
# Confidence + rejection: a hand whose reconstruction is WRONG must not export
# clean. Before this, confidence read only the warning list and the card counts,
# so a hand that had lost its whole community row scored 0.95 with no tag.
# --------------------------------------------------------------------------- #
def _spine_hand(**overrides):
    """A fully reconstructed, internally consistent hand."""
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 8.0,
        "n_states": 20,
        "hero": ["As", "Kd"],
        "board": ["2c", "7d", "9h", "Ts", "Jc"],
        "complete_cards": True,
        "warnings": [],
        "players": [
            {"seat": 0, "position": "SB", "player_name": "Hero",
             "starting_stack": 100.0, "is_hero": True},
            {"seat": 4, "position": "BTN", "player_name": "Seat4",
             "starting_stack": 100.0, "is_hero": False},
        ],
        "actions": [
            {"street": "preflop", "action_index": 1, "seat": 4, "position": "BTN",
             "player_name": "Seat4", "action_type": "raise", "amount": 3.0,
             "pot_before": 0.0, "stack_before": 100.0},
            {"street": "flop", "action_index": 1, "seat": 0, "position": "SB",
             "player_name": "Hero", "action_type": "bet", "amount": 7.0,
             "pot_before": 6.0, "stack_before": 97.0},
        ],
        "streets": [{"street": s} for s in ("preflop", "flop", "turn", "river")],
        "pot": 20.0,
        "side_pot": None,
        "winner_seat": 0,
        "result": "Hero wins",
        "hero_bb_won": 10.0,
        "hero_folded": False,
        "reconciled": True,
        "amounts_unknown": 0,
        "amounts_rejected": 0,
        "anchor_missing_states": 0,
        # The spine emits this on every hand: at least one state POSITIVELY
        # placed the hero zone on seat 0. layout_supported requires it, because
        # the mismatch check it used to rely on alone is vacuously False when the
        # hero zone is empty -- which is exactly what a layout drift produces.
        "hero_seat_confirmed": True,
        "source_images": ["f.jpg"],
    }
    hand.update(overrides)
    return hand


def _export(hand, **kwargs):
    return timeline_to_session_payload(
        {"states": [], "hands": [hand]},
        timeline_path="timeline.json",
        session_name="Draft",
        **kwargs,
    )


def test_clean_hand_still_scores_high():
    """An INTERIOR hand: both its boundaries were observed, so nothing is missing.
    A lone hand is deliberately not used here -- it has no preceding deal, so it is
    truncated at the start and is capped below 1.0 on that evidence alone."""
    payload = timeline_to_session_payload(
        {"states": [], "hands": [
            _spine_hand(hand_number=n, terminal_event="showdown", source_images=[f"f{n}.jpg"])
            for n in (1, 2, 3)
        ]},
        timeline_path="timeline.json",
        session_name="Draft",
    )
    assert payload["cv_import_summary"]["exported_hands"] == 3
    assert payload["hands"][1]["hand"]["confidence_score"] >= 0.9
    assert payload["hands"][1]["hand"]["tags"] == []


def test_board_zone_yield_zero_hand_is_rejected():
    """The exact 06-21 export: clean cards, no board, and a spine warning that
    the old gate never saw. It exported at 0.95 with tags=[]."""
    hand = _spine_hand(board=[], warnings=["board_zone_yield_zero"])
    payload = _export(hand)

    assert payload["cv_import_summary"]["exported_hands"] == 0
    skip = payload["cv_import_summary"]["skipped"][0]
    assert skip["reason"] == "validation_warnings"
    assert "board_zone_yield_zero" in skip["codes"]


def test_side_pot_hand_is_rejected_explicitly():
    """Detect and reject, never silently discard: the side pot is real, the spine
    cannot represent it, and the skip reason says so by name."""
    hand = _spine_hand(side_pot=0.2, warnings=["side_pot_unsupported"])
    payload = _export(hand)

    assert payload["cv_import_summary"]["exported_hands"] == 0
    skip = payload["cv_import_summary"]["skipped"][0]
    assert "side_pot_unsupported" in skip["codes"]
    assert "side_pot_unsupported" in skip["detail"]


def test_confidence_reflects_reconstruction_not_warning_count():
    """No warnings at all, clean cards, a reconciled pot -- and an empty board on
    a hand whose own street summary lists four streets. The old formula scored
    this 0.95 because an empty board is a legal card count."""
    hand = _spine_hand(board=[], warnings=[])
    payload = _export(hand, allow_validation_warnings=True)
    assert payload["hands"][0]["hand"]["confidence_score"] <= 0.4
    assert payload["hands"][0]["hand"]["tags"] == ["LOW_CONFIDENCE"]


def test_confidence_capped_when_the_anchor_was_missing_on_many_states():
    hand = _spine_hand(n_states=20, anchor_missing_states=9)
    payload = _export(hand)
    assert payload["hands"][0]["hand"]["confidence_score"] <= 0.45


def test_confidence_capped_when_the_pot_never_reconciled():
    hand = _spine_hand(reconciled=False)
    payload = _export(hand, allow_validation_warnings=True)
    assert payload["hands"][0]["hand"]["confidence_score"] <= 0.55


def test_completion_evidence_carries_cv_read_quality(tmp_path) -> None:
    """Read-quality facts must survive the import. cv_source is dropped by the
    importer; completion_evidence is an already-typed open mapping that is not."""
    hand = _spine_hand(amounts_unknown=4, amounts_rejected=2, anchor_missing_states=1)
    payload = _export(hand, allow_validation_warnings=True)
    evidence = payload["hands"][0]["hand"]["completion_evidence"]
    assert evidence["cv_amounts_rejected"] == 2
    assert evidence["cv_amounts_unknown"] == 4
    assert evidence["cv_anchor_missing_states"] == 1
    assert evidence["cv_side_pot_detected"] is False

    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    stored = db.fetch_hands_by_session(session.id)[0]
    assert stored.completion_evidence["cv_amounts_rejected"] == 2
    db.close()


def test_unknown_stack_is_none_not_zero_after_import() -> None:
    """A seat that was never read must import as None; a seat that genuinely read
    zero (an all-in showing "0 BB") must import as 0.0. Conflating them would
    erase every all-in seat from the stack ledger."""
    hand = _spine_hand(players=[
        {"seat": 0, "position": "SB", "player_name": "Hero",
         "starting_stack": None, "is_hero": True},
        {"seat": 4, "position": "BTN", "player_name": "Seat4",
         "starting_stack": 0.0, "is_hero": False},
    ])
    payload = _export(hand)

    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    stored = db.fetch_hands_by_session(session.id)[0]
    by_seat = {p.seat_index: p for p in db.fetch_players_by_hand(stored.id)}
    assert by_seat[0].starting_stack is None
    assert by_seat[4].starting_stack == 0.0
    db.close()


# --------------------------------------------------------------------------- #
# Versioned completion evidence: without a producer, every reconstructed hand is
# permanently blocked on COMPLETION_EVIDENCE_MISSING and the blocker's stated
# clearing action ("re-run the reconstruction") provably cannot clear it.
# --------------------------------------------------------------------------- #
def test_export_attaches_versioned_completion_evidence() -> None:
    from poker_tracker.persistence.completion import (
        EVIDENCE_SCHEMA_VERSION,
        parse_completion_evidence,
    )

    payload = _export(_spine_hand(terminal_event="showdown"))
    evidence = parse_completion_evidence(
        payload["hands"][0]["hand"]["completion_evidence"]
    )

    assert evidence.evidence_version == EVIDENCE_SCHEMA_VERSION
    assert evidence.is_known is True
    assert evidence.terminal_event == "showdown"
    assert evidence.first_source_timestamp_s == 0.0
    assert evidence.last_source_timestamp_s == 8.0
    assert evidence.source_frames == ("f.jpg",)
    assert evidence.boundary_confidence is not None
    assert evidence.table_size == 2
    assert evidence.layout_supported is True
    # Read-quality facts from the pre-evidence export must survive unchanged.
    assert evidence.extra["cv_amounts_unknown"] == 0


def _multi_hand_payload(**overrides):
    hands = [
        _spine_hand(hand_number=n, terminal_event="showdown", source_images=[f"f{n}.jpg"], **overrides)
        for n in (1, 2, 3)
    ]
    return timeline_to_session_payload(
        {"states": [], "hands": hands},
        timeline_path="timeline.json",
        session_name="Draft",
    )


def test_an_interior_reconstructed_hand_can_reach_complete() -> None:
    """The whole point of the evidence producer: a proven hand is no longer stuck."""
    payload = _multi_hand_payload()
    statuses = [item["hand"]["completion_status"] for item in payload["hands"]]

    assert statuses[1] == "complete"


def test_the_first_hand_of_a_recording_is_reported_truncated() -> None:
    """Its deal was never observed, so the start boundary cannot be claimed."""
    from poker_tracker.persistence.completion import parse_completion_evidence

    payload = _multi_hand_payload()
    first = parse_completion_evidence(payload["hands"][0]["hand"]["completion_evidence"])

    assert first.partial_start is True
    assert payload["hands"][0]["hand"]["completion_status"] == "partial"


def test_a_hand_whose_terminal_event_was_never_read_is_uncertain() -> None:
    """Both boundaries are bracketed by neighbours, so only the unread terminal
    event is left to block: the hand is unproven, not truncated."""
    payload = timeline_to_session_payload(
        {
            "states": [],
            "hands": [
                _spine_hand(hand_number=1, terminal_event="showdown"),
                _spine_hand(hand_number=2, terminal_event="unobserved"),
                _spine_hand(hand_number=3, terminal_event="showdown"),
            ],
        },
        timeline_path="timeline.json",
        session_name="Draft",
    )

    assert payload["hands"][1]["hand"]["completion_status"] == "uncertain"


def _interior_evidence(**overrides):
    from poker_tracker.persistence.completion import parse_completion_evidence

    payload = timeline_to_session_payload(
        {
            "states": [],
            "hands": [
                _spine_hand(hand_number=1, terminal_event="showdown"),
                _spine_hand(hand_number=2, terminal_event="showdown", **overrides),
                _spine_hand(hand_number=3, terminal_event="showdown"),
            ],
        },
        timeline_path="timeline.json",
        session_name="Draft",
        allow_validation_warnings=True,
    )
    return parse_completion_evidence(payload["hands"][1]["hand"]["completion_evidence"])


def test_a_destructive_spine_warning_becomes_an_unacknowledgeable_rejection_code() -> None:
    from poker_tracker.persistence.completion import (
        acknowledge_codes,
        derive_completion_status,
    )

    evidence = _interior_evidence(warnings=["side_pot_unsupported"])

    assert evidence.rejection_codes == ("side_pot_unsupported",)
    accepted = acknowledge_codes(evidence, ["side_pot_unsupported"])
    assert accepted.acknowledged_codes == ()
    assert derive_completion_status(accepted, source_type="cv_import") == "uncertain"


def test_a_recoverable_spine_warning_stays_an_acknowledgeable_warning_code() -> None:
    evidence = _interior_evidence(warnings=["pot_not_reconciled"])

    assert evidence.warning_codes == ("pot_not_reconciled",)
    assert evidence.rejection_codes == ()


def test_the_reconstructed_hand_reaches_the_app_with_readable_evidence(tmp_path) -> None:
    """End to end: exporter -> import_session -> readiness. The evidence blocker
    must actually clear, which is what makes its clearing action truthful."""
    from poker_tracker.services.study_readiness import evaluate_study_readiness

    payload = _multi_hand_payload()
    db = PokerDatabase(str(tmp_path / "cv.sqlite3"))
    db.init_db()
    session = import_session(db, payload)
    interior = db.fetch_hands_by_session(session.id)[1]

    readiness = evaluate_study_readiness(interior, accounting=None, user_confirmed=True)

    assert interior.completion_status == "complete"
    assert readiness.has("COMPLETION_EVIDENCE_MISSING") is False
    assert readiness.has("COMPLETION_NOT_COMPLETE") is False
    assert readiness.has("UNSUPPORTED_TABLE_LAYOUT") is False
    assert readiness.codes() == ("ACCOUNTING_NOT_AUTHORITATIVE",)
    db.close()


# --------------------------------------------------------------------------- #
# Adversarial round 1: nets that must reach the EXPORT GATE, not just the notes.
# --------------------------------------------------------------------------- #
def _seq_hand(actions, **overrides):
    """A clean hand whose only defect is its action sequence.

    Every seat the ledger names gets a player row: an action referencing a seat
    the hand does not list is itself a rejection (the app's ingest rolls the whole
    payload back on it), and these cases are about action ORDER, not bookkeeping.
    """
    hand = _spine_hand(actions=actions, **overrides)
    listed = {player["seat"] for player in hand["players"]}
    for seat in sorted({action["seat"] for action in actions} - listed):
        hand["players"].append({
            "seat": seat, "position": f"P{seat}", "player_name": f"Seat{seat}",
            "starting_stack": 100.0, "is_hero": False,
        })
    return hand


def test_a_hand_that_acts_after_going_all_in_is_rejected():
    """Measured on the 07-11 recording: seat 3 goes all-in for 71.2, then calls 2.0
    from a 0.0 stack, then folds -- and the hand exported at confidence 1.0 with
    completion_status 'complete' and no warning anywhere. A seat with no chips left
    cannot act again."""
    payload = _export(_seq_hand([
        {"street": "flop", "action_index": 1, "seat": 3, "action_type": "all-in",
         "amount": 71.2, "stack_before": 71.2},
        {"street": "flop", "action_index": 2, "seat": 4, "action_type": "raise",
         "amount": 9.0, "stack_before": 90.0},
        {"street": "flop", "action_index": 3, "seat": 3, "action_type": "call",
         "amount": 2.0, "stack_before": 0.0},
    ]))
    assert payload["cv_import_summary"]["exported_hands"] == 0
    codes = payload["cv_import_summary"]["skipped"][0]["codes"]
    assert "action_sequence_illegal" in codes


def test_a_hand_that_folds_right_after_its_own_call_is_rejected():
    """06-21 river: seat 5 checks, seat 0 bets 75.0, seat 5 CALLS 75.0, seat 5
    FOLDS. A call closes the action to you; it reopens only if someone raises."""
    payload = _export(_seq_hand([
        {"street": "river", "action_index": 1, "seat": 5, "action_type": "check"},
        {"street": "river", "action_index": 2, "seat": 0, "action_type": "bet",
         "amount": 75.0},
        {"street": "river", "action_index": 3, "seat": 5, "action_type": "call",
         "amount": 75.0},
        {"street": "river", "action_index": 4, "seat": 5, "action_type": "fold"},
    ]))
    assert payload["cv_import_summary"]["exported_hands"] == 0
    assert "action_sequence_illegal" in payload["cv_import_summary"]["skipped"][0]["codes"]


def test_a_hand_that_checks_facing_a_raise_is_rejected():
    """06-21 preflop: seat 1 raises to 6.0, then seat 3 -- who has not put a chip
    in on the street -- checks."""
    payload = _export(_seq_hand([
        {"street": "preflop", "action_index": 1, "seat": 5, "action_type": "call",
         "amount": 1.0},
        {"street": "preflop", "action_index": 2, "seat": 1, "action_type": "raise",
         "amount": 6.0},
        {"street": "preflop", "action_index": 3, "seat": 3, "action_type": "check"},
    ]))
    assert payload["cv_import_summary"]["exported_hands"] == 0
    assert "action_sequence_illegal" in payload["cv_import_summary"]["skipped"][0]["codes"]


def test_a_legal_limped_pot_is_not_flagged():
    """Negative control. Blinds are not emitted as actions, so the big blind's
    option-check follows a street of calls with no aggression -- legal, and the
    commonest preflop shape there is."""
    payload = _export(_seq_hand([
        {"street": "preflop", "action_index": 1, "seat": 4, "action_type": "call",
         "amount": 2.0},
        {"street": "preflop", "action_index": 2, "seat": 5, "action_type": "call",
         "amount": 2.0},
        {"street": "preflop", "action_index": 3, "seat": 0, "action_type": "check"},
        {"street": "flop", "action_index": 1, "seat": 0, "action_type": "check"},
        {"street": "flop", "action_index": 2, "seat": 4, "action_type": "bet",
         "amount": 5.0},
        {"street": "flop", "action_index": 3, "seat": 0, "action_type": "call",
         "amount": 5.0},
    ]))
    assert payload["cv_import_summary"]["exported_hands"] == 1


def test_a_legal_reraise_after_calling_is_not_flagged():
    """Negative control: a call is reopened by a raise, and the caller may then
    raise or fold. Only a fold with NO intervening aggression is impossible."""
    payload = _export(_seq_hand([
        {"street": "preflop", "action_index": 1, "seat": 4, "action_type": "raise", "amount": 3.0},
        {"street": "preflop", "action_index": 2, "seat": 0, "action_type": "call", "amount": 3.0},
        {"street": "flop", "action_index": 1, "seat": 4, "action_type": "bet", "amount": 5.0},
        {"street": "flop", "action_index": 2, "seat": 0, "action_type": "call", "amount": 5.0},
        {"street": "flop", "action_index": 3, "seat": 5, "action_type": "raise", "amount": 20.0},
        {"street": "flop", "action_index": 4, "seat": 0, "action_type": "fold"},
    ]))
    assert payload["cv_import_summary"]["exported_hands"] == 1


def test_every_spine_fatal_code_actually_blocks_the_export():
    """Each code in SPINE_FATAL_CODES exists to keep a hand out of the app. Two of
    them -- amount_scale_implausible (the stack ledger is systematically wrong) and
    anchor_unavailable (cards were detected then dropped for want of a transform)
    -- could be deleted from the frozenset with the whole suite green."""
    # Named literally, NOT iterated over the frozenset: a loop over
    # SPINE_FATAL_CODES stops testing a code the moment someone deletes it, which
    # is exactly the regression this pins.
    for code in (
        "board_zone_yield_zero",
        "board_empty_but_streets_advanced",
        "actions_collapsed_to_one_street",
        "amount_scale_implausible",
        "side_pot_unsupported",
        "hero_seat_mismatch",
        "anchor_unavailable",
        "stack_ledger_incoherent",
        "contributions_exceed_pot",
        "result_contradicts_hero_net",
        # Phase 6: the hand's MONEY LEDGER carries an unknown -- a money action
        # nothing could size, or a transition nothing could measure.
        "amounts_unknown_in_ledger",
        # Round-1 repair: a player row whose starting stack the reader refused.
        # Accounting fails permanently on a None starting stack, so the hand can
        # never become authoritative; it used to export as "complete" and
        # dead-end downstream with no clearing action.
        "starting_stack_unknown",
        # Tab/lobby covering the table across a critical mid-hand change.
        "mid_hand_coverage_gap",
    ):
        payload = _export(_spine_hand(warnings=[code]))
        assert payload["cv_import_summary"]["exported_hands"] == 0, code
        assert code in payload["cv_import_summary"]["skipped"][0]["codes"], code


def test_an_unrecognised_spine_warning_blocks_the_export():
    """The stated contract: 'a code this build does not recognise cannot be
    assessed at all, so it fails closed the same way'. It did not -- the hand
    exported at 0.7 while its own completion_evidence listed the code under
    rejection_codes."""
    payload = _export(_spine_hand(warnings=["some_future_fatal_code"]))

    assert payload["cv_import_summary"]["exported_hands"] == 0
    assert "some_future_fatal_code" in payload["cv_import_summary"]["skipped"][0]["codes"]


def test_a_recognised_recoverable_spine_warning_still_reaches_the_gate_as_itself():
    """Negative control for the rule above: a code this build DOES recognise is
    assessed on its own severity, not swept up as unknown."""
    from cv_lab.scripts.eval.validate_yolo_card_timeline import RECOGNISED_SPINE_CODES

    assert "pot_text_dropped" in RECOGNISED_SPINE_CODES
    payload = _export(_spine_hand(warnings=["pot_text_dropped"]))
    assert payload["cv_import_summary"]["exported_hands"] == 1


def test_a_truncated_hand_cannot_score_full_confidence():
    """06-21 hand 1: the recording opens with the flop already dealt and 18.1 BB in
    the pot, so the hand has no preflop street and its 'starting_stack' values are
    mid-flop stacks. completion_status was correctly 'partial' -- and the number a
    reviewer reads on the hand was still 1.0 with tags []."""
    payload = _multi_hand_payload()
    first = payload["hands"][0]["hand"]
    interior = payload["hands"][1]["hand"]

    assert interior["confidence_score"] == 1.0
    assert first["confidence_score"] < 1.0


def test_a_hand_whose_terminal_event_was_never_read_cannot_score_full_confidence():
    """Same rule on the other boundary: an unobserved ending is missing evidence,
    not an absence of complaints."""
    payload = timeline_to_session_payload(
        {
            "states": [],
            "hands": [
                _spine_hand(hand_number=1, terminal_event="showdown"),
                _spine_hand(hand_number=2, terminal_event="unobserved"),
                _spine_hand(hand_number=3, terminal_event="showdown"),
            ],
        },
        timeline_path="timeline.json",
        session_name="Draft",
    )

    assert payload["hands"][1]["hand"]["confidence_score"] < 1.0


def test_a_hand_with_mostly_unreadable_amounts_cannot_score_full_confidence():
    """amounts_unknown was recorded, serialized into completion_evidence as
    cv_amounts_unknown, and then consulted by nothing: a hand in which 18 numeric
    HUD reads failed scored the same 1.0 as one in which none did. Confidence is
    documented as requiring positive evidence, and a hand whose stacks, bets and
    pot were unreadable in most of its states has materially less of it.

    Threshold from the development corpus: per-hand amounts_unknown / n_states
    runs 0.06-0.29 on 18 of 21 hands and 0.57 / 0.86 / 1.23 on the other three."""
    payload = timeline_to_session_payload(
        {"states": [], "hands": [
            _spine_hand(hand_number=1, terminal_event="showdown"),
            _spine_hand(hand_number=2, terminal_event="showdown",
                        n_states=21, amounts_unknown=18),
            _spine_hand(hand_number=3, terminal_event="showdown"),
        ]},
        timeline_path="timeline.json",
        session_name="Draft",
    )
    assert payload["hands"][1]["hand"]["confidence_score"] < 1.0


def test_an_ordinary_rate_of_unreadable_amounts_does_not_lower_confidence():
    """Negative control at the corpus's ordinary rate (0.29 unread per state)."""
    payload = timeline_to_session_payload(
        {"states": [], "hands": [
            _spine_hand(hand_number=1, terminal_event="showdown"),
            _spine_hand(hand_number=2, terminal_event="showdown",
                        n_states=21, amounts_unknown=6),
            _spine_hand(hand_number=3, terminal_event="showdown"),
        ]},
        timeline_path="timeline.json",
        session_name="Draft",
    )
    assert payload["hands"][1]["hand"]["confidence_score"] == 1.0


def test_partial_end_is_claimed_from_evidence_not_optimism():
    """partial_start was pinned; partial_end and the terminal-event
    discrimination were not, so all three could be made unconditionally
    optimistic with the full suite green. These fields gate study readiness."""
    from poker_tracker.persistence.completion import parse_completion_evidence

    payload = timeline_to_session_payload(
        {"states": [], "hands": [_spine_hand(hand_number=1, terminal_event="unobserved")]},
        timeline_path="timeline.json",
        session_name="Draft",
    )
    evidence = parse_completion_evidence(payload["hands"][0]["hand"]["completion_evidence"])

    # Last hand of the recording AND no terminal event read: both boundaries open.
    assert evidence.partial_end is True
    assert evidence.terminal_event == "unobserved"
    assert evidence.following_boundary.kind == "recording_end"


def test_a_read_terminal_event_closes_the_end_boundary():
    """Negative control, and the showdown-vs-fold_win discrimination itself: a
    terminal event the spine DID read proves the end even with no following
    hand."""
    from poker_tracker.persistence.completion import parse_completion_evidence

    for event in ("showdown", "fold_win", "hero_fold"):
        payload = timeline_to_session_payload(
            {"states": [], "hands": [_spine_hand(hand_number=1, terminal_event=event)]},
            timeline_path="timeline.json",
            session_name="Draft",
        )
        evidence = parse_completion_evidence(payload["hands"][0]["hand"]["completion_evidence"])
        assert evidence.partial_end is False, event
        assert evidence.terminal_event == event
        assert evidence.following_boundary.kind == "hand_end", event


def test_an_evidence_capped_hand_is_actually_tagged_low_confidence():
    """The 0.80 cap and the LOW_CONFIDENCE tag disagreed by one strict inequality.

    `_confidence_for_hand` clamps to exactly 0.80 when a hand's start or end was
    never observed, or when its unreadable-amount rate is high; the tag test was
    `confidence < 0.8`. So the cap that exists to flag an incomplete boundary
    produced precisely the value the tag excluded, and every unflagged case in the
    development corpus was one of these -- g0723b hand 1 (partial_start) and
    g0723a hand 4 (partial_end, terminal_event 'unobserved') both shipped at
    confidence_score 0.8 with tags []."""
    payload = _multi_hand_payload()
    first = payload["hands"][0]["hand"]
    interior = payload["hands"][1]["hand"]

    assert first["confidence_score"] == LOW_CONFIDENCE_AT
    assert first["tags"] == ["LOW_CONFIDENCE"], (
        "a hand capped for missing boundary evidence must be tagged for review"
    )
    assert interior["confidence_score"] == 1.0
    assert interior["tags"] == []


def test_low_confidence_tag_and_the_evidence_caps_share_one_constant():
    """Structural guard: the cap must never again sit on the wrong side of the
    tag's comparison. Anything at or below the constant is tagged."""
    assert _confidence_for_hand(
        _spine_hand(terminal_event="unobserved"), validation_codes=[],
        partial_start=False, partial_end=False, terminal_observed=False,
    ) == LOW_CONFIDENCE_AT


def test_layout_is_not_supported_without_positive_hero_evidence():
    """`layout_supported` was gated solely on the ABSENCE of hero_seat_mismatch,
    and that check is a majority vote over the hero zone's own cards -- so with an
    empty hero zone it evaluates `0 > 0` and reports "hero seat confirmed" on the
    strength of nothing.

    An empty hero zone is exactly what a layout drift produces. Measured: a 1.24x
    vertical stretch pushes hero's cards 0.0021 of reference-y past the hero band,
    they are re-attributed to two different villain seats as showdown reveals, the
    board still zones 5/5, unanchored_cards stays 0 and the anchor residual stays
    inside tolerance -- and hero_seat_mismatch was False at every step."""
    from poker_tracker.persistence.completion import parse_completion_evidence

    payload = _export(_spine_hand(hero_seat_confirmed=False))
    evidence = parse_completion_evidence(
        payload["hands"][0]["hand"]["completion_evidence"])
    assert evidence.layout_supported is False

    confirmed = _export(_spine_hand())
    assert parse_completion_evidence(
        confirmed["hands"][0]["hand"]["completion_evidence"]).layout_supported is True


def test_action_for_a_seat_that_is_not_a_player_is_refused_not_shipped():
    """Round 4, adversary C: the CV export emitted an action whose player_key had
    no matching HandPlayer row, and the app's real ingest path rolls the WHOLE
    payload back on it.

    ``_build_players`` reads the spine's ``players`` and ``_build_actions`` reads
    its ``actions``, with no cross-check. On the 07-11 development recording a
    hand shipped players {seat:0,1,2,3,4,7} alongside a preflop
    {'player_key': 'seat:5', 'action_type': 'fold'}; import_session then raised
    "Action player key does not belong to this hand." and rolled back the
    transaction -- 0 sessions, 0 hands, with three good hands in the same payload
    destroyed by the one malformed one.

    A hand whose own action ledger references a seat it does not list is
    internally inconsistent. It must be refused ALONE, so the rest of the session
    still lands."""
    orphan = _spine_hand(actions=[
        {"street": "preflop", "action_index": 1, "seat": 4, "position": "BTN",
         "player_name": "Seat4", "action_type": "raise", "amount": 3.0},
        {"street": "preflop", "action_index": 2, "seat": 5, "position": "",
         "player_name": "Seat5", "action_type": "fold", "amount": None},
        {"street": "flop", "action_index": 1, "seat": 0, "position": "SB",
         "player_name": "Hero", "action_type": "bet", "amount": 7.0},
    ])
    payload = timeline_to_session_payload(
        {"states": [], "hands": [
            _spine_hand(hand_number=1, terminal_event="showdown"),
            dict(orphan, hand_number=2),
            _spine_hand(hand_number=3, terminal_event="showdown"),
        ]},
        timeline_path="timeline.json",
        session_name="Draft",
    )
    # The two healthy hands survive; only the inconsistent one is skipped.
    assert payload["cv_import_summary"]["exported_hands"] == 2
    skipped = payload["cv_import_summary"]["skipped"]
    assert [s["timeline_hand_number"] for s in skipped] == [2]
    assert "seat:5" in skipped[0]["detail"]

    # And the invariant itself holds for every exported hand.
    for exported in payload["hands"]:
        keys = {player["player_key"] for player in exported["players"]}
        assert {action["player_key"] for action in exported["actions"]} <= keys


def test_discarded_read_evidence_moves_the_score_it_is_recorded_beside():
    """Round 4, adversary B: three counters were recorded, serialized, and read by
    nothing.

    ``stack_conflicts`` counts seats whose two stack_text boxes CONTRADICTED each
    other, so the seat's stack was discarded to unknown. ``stack_outlier_checks_
    skipped`` counts frames with too few stack reads for the sibling-median net, so
    the dropped-decimal net did not run at all. ``pot_text_off_column`` counts pot
    boxes thrown out by the centre-column guard. Each is an evidence gap of exactly
    the kind ``amounts_unknown`` already caps the score for -- and each reached the
    export only as a cv_* field on completion_evidence, while _confidence_for_hand
    read none of them. Measured: a hand with a contradicted stack exported at
    confidence 1.0 with tags [].

    The comments that introduced them say they exist so downstream can "tell the
    state apart". Nothing downstream did."""
    for field in ("stack_conflicts", "stack_outlier_checks_skipped",
                  "pot_text_off_column"):
        clean = _confidence_for_hand(_spine_hand(**{field: 0}))
        assert clean == 1.0, field
        assert _confidence_for_hand(_spine_hand(**{field: 1})) <= LOW_CONFIDENCE_AT, field

    # ... and it reaches the exported record as the review tag, not just a number.
    payload = timeline_to_session_payload(
        {"states": [], "hands": [
            _spine_hand(hand_number=1, terminal_event="showdown"),
            _spine_hand(hand_number=2, terminal_event="showdown", stack_conflicts=1),
            _spine_hand(hand_number=3, terminal_event="showdown"),
        ]},
        timeline_path="timeline.json", session_name="Draft",
    )
    flagged = payload["hands"][1]["hand"]
    assert flagged["confidence_score"] <= LOW_CONFIDENCE_AT
    assert "LOW_CONFIDENCE" in flagged["tags"]


# --------------------------------------------------------------------------- #
# Phase 6: UNKNOWN reaches the app as UNKNOWN.
# --------------------------------------------------------------------------- #
def test_an_unknown_amount_blocks_promotion_to_complete() -> None:
    """`derive_completion_status` promotes a hand whose boundaries are observed,
    whose terminal event is read, and whose codes are clear. A hand carrying money
    of unknown size satisfies all three and is still not a complete record of what
    happened -- PLAN.md: "Mark an incomplete sequence non-authoritative".

    The mechanism is the existing one, with no new machinery: the code's severity
    is at or above _REJECTION_SEVERITY, so _split_source_codes files it under
    rejection_codes, and a rejection code can never be acknowledged away."""
    def _interior(**overrides):
        """The MIDDLE hand of three: both boundaries observed, so nothing else
        holds it back and the completion status turns on this code alone."""
        return timeline_to_session_payload(
            {"states": [], "hands": [
                _spine_hand(hand_number=1, terminal_event="showdown"),
                _spine_hand(hand_number=2, terminal_event="showdown", **overrides),
                _spine_hand(hand_number=3, terminal_event="showdown"),
            ]},
            timeline_path="timeline.json", session_name="Draft",
            allow_validation_warnings=True,
        )

    clean = _interior()
    assert clean["hands"][1]["hand"]["completion_status"] == "complete"

    payload = _interior(warnings=["amounts_unknown_in_ledger"])
    hand = payload["hands"][1]["hand"]
    assert hand["completion_status"] == "uncertain"
    assert "amounts_unknown_in_ledger" in hand["completion_evidence"]["rejection_codes"]
    assert "amounts_unknown_in_ledger" not in hand["completion_evidence"]["warning_codes"]
    assert hand["confidence_score"] <= LOW_CONFIDENCE_AT
    assert "LOW_CONFIDENCE" in hand["tags"]


def test_starting_stack_none_is_distinguishable_from_zero_in_the_payload() -> None:
    """`HandPlayer.starting_stack` has always allowed None, and nothing asserted
    that the distinction survives the export. It has to: 0.0 is an all-in seat
    showing "0 BB", None is a stack nobody could read, and the two drive different
    effective-stack and SPR conclusions."""
    hand = _spine_hand(terminal_event="showdown")
    hand["players"] = [
        {"seat": 0, "position": "SB", "player_name": "Hero",
         "starting_stack": 0.0, "is_hero": True, "starting_stack_unknown": None},
        {"seat": 4, "position": "BTN", "player_name": "Seat4",
         "starting_stack": None, "is_hero": False,
         "starting_stack_unknown": "suffix_not_bb"},
    ]
    players = _export(hand)["hands"][0]["players"]
    by_seat = {p["seat_index"]: p for p in players}
    assert by_seat[0]["starting_stack"] == 0.0
    assert by_seat[4]["starting_stack"] is None
    evidence = _export(hand)["hands"][0]["hand"]["completion_evidence"]
    assert evidence["cv_starting_stack_unknown"] == {"4": "suffix_not_bb"}


def test_per_code_unknown_counts_reach_the_completion_evidence() -> None:
    """Only the bare `cv_amounts_unknown` count was carried, and a count cannot
    tell a systematic reader failure on one HUD region from scattered occlusions.
    `extra` is an explicitly open mapping, so this is additive -- no schema change,
    no migration, and it survives the export/import round trip."""
    hand = _spine_hand(
        terminal_event="showdown",
        amounts_unknown=7,
        amounts_unknown_by_code={"suffix_not_bb": 5, "run_clipped": 2},
        unmeasured_transitions=2,
        unknown_money_actions=1,
    )
    evidence = _export(hand, allow_validation_warnings=True)["hands"][0]["hand"][
        "completion_evidence"]
    assert evidence["cv_amounts_unknown"] == 7
    assert evidence["cv_amounts_unknown_by_code"] == {"suffix_not_bb": 5,
                                                      "run_clipped": 2}
    assert evidence["cv_unmeasured_transitions"] == 2
    assert evidence["cv_unknown_money_actions"] == 1
    # ... and an unmeasured transition on a hand exported past the gate still
    # costs the score, so the operator overrode the GATE, not the facts.
    assert _confidence_for_hand(hand) <= LOW_CONFIDENCE_AT


def test_a_refused_starting_stack_is_a_rejection_not_a_silent_complete() -> None:
    """Round-1 repair (adversary B): `starting_stack_unknown` was written into
    CompletionEvidence.extra as cv_starting_stack_unknown and read by NOTHING --
    not by _confidence_for_hand, not by _split_source_codes, not by the
    validator. The hand exported with warning_codes=[] / rejection_codes=[],
    derive_completion_status returned "complete", and accounting then failed
    PERMANENTLY (LedgerError: a None starting stack) with no clearing action
    attached -- a false 'complete' claim and a silent dead end (g0711 timeline
    hand 5, seat 5, code no_digit_run). The spine now raises the fatal code."""
    from poker_tracker.persistence.completion import derive_completion_status

    evidence = _interior_evidence(warnings=["starting_stack_unknown"])
    assert "starting_stack_unknown" in evidence.rejection_codes
    assert derive_completion_status(evidence, source_type="cv_import") == "uncertain"

    hand = _spine_hand(
        warnings=["starting_stack_unknown"],
        players=[
            {"seat": 0, "position": "SB", "player_name": "Hero",
             "starting_stack": 100.0, "is_hero": True},
            {"seat": 4, "position": "BTN", "player_name": "Seat4",
             "starting_stack": None, "starting_stack_unknown": "no_digit_run",
             "is_hero": False},
        ],
    )
    payload = _export(hand, allow_validation_warnings=True)
    exported = payload["hands"][0]["hand"]
    # A lone exported hand also has unobserved recording boundaries, so its
    # status is "partial" here; the load-bearing claim is that it is never
    # "complete" (the interior-evidence assertion above pins "uncertain").
    assert exported["completion_status"] != "complete"
    assert exported["confidence_score"] <= 0.5
    evidence_blob = exported["completion_evidence"]
    assert "starting_stack_unknown" in evidence_blob["rejection_codes"]
    assert evidence_blob["cv_starting_stack_unknown"] == {"4": "no_digit_run"}


def test_settle_scan_blind_spots_reach_evidence_and_cap_confidence() -> None:
    """Round-1 repair (adversary B): `settle_scan_skipped` -- the spine's own
    record that the settlement scan could not examine one or more of the hand's
    transitions because the pot read there was refused -- was computed and
    consumed by nothing. g0723a hand 5 exported with SIX skipped settlement
    transitions at confidence 1.0, tags [], and an operator saw nothing. The
    same argument that carried cv_pot_text_off_column and cv_stack_conflicts
    into the evidence applies: the counter exists so the state can be told
    apart downstream, and this is downstream."""
    from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
        _confidence_for_hand,
    )

    assert _confidence_for_hand(_spine_hand()) == 1.0
    assert _confidence_for_hand(_spine_hand(settle_scan_skipped=6)) <= 0.80

    payload = _export(_spine_hand(settle_scan_skipped=6), allow_validation_warnings=True)
    exported = payload["hands"][0]["hand"]
    assert exported["completion_evidence"]["cv_settle_scan_skipped"] == 6
    assert "LOW_CONFIDENCE" in exported["tags"]
    clean = _export(_spine_hand(), allow_validation_warnings=True)
    assert clean["hands"][0]["hand"]["completion_evidence"]["cv_settle_scan_skipped"] == 0


def test_pot_text_dropped_reaches_the_exported_record() -> None:
    """THE ROUND-2 B3 REGRESSION. The spine sets the `pot_text_dropped` FIELD
    when the pot consensus overrides the pot text (two derived estimators
    agreeing against the one direct read), the severity table priced the code
    at 0.15 and the validator's vocabulary listed it -- but no producer ever
    attached the code to a hand, so 'text confirmed the pot' and 'text was
    outvoted' exported identically: the field was set True with zero effect
    anywhere. The field now enters the confidence deduction and the exported
    warning_codes at exactly the severity the table declares. It stays an
    acknowledgeable warning, not a gate: blocking every legitimate stale-text
    override would disable the consensus mechanism it reports on."""
    import pytest

    from poker_tracker.persistence.completion import parse_completion_evidence

    clean = _spine_hand()
    dropped = _spine_hand(pot_text_dropped=True)
    conf_clean = _confidence_for_hand(clean)
    conf_dropped = _confidence_for_hand(dropped)
    assert conf_clean - conf_dropped == pytest.approx(0.15), (
        "the severity table's 0.15 deduction must actually be charged")
    payload = timeline_to_session_payload(
        {"states": [], "hands": [
            _spine_hand(hand_number=1, terminal_event="showdown", source_images=["f1.jpg"]),
            _spine_hand(hand_number=2, terminal_event="showdown",
                        source_images=["f2.jpg"], pot_text_dropped=True),
            _spine_hand(hand_number=3, terminal_event="showdown", source_images=["f3.jpg"]),
        ]},
        timeline_path="timeline.json",
        session_name="Draft",
    )
    assert payload["cv_import_summary"]["exported_hands"] == 3, (
        "an outvoted pot text is a note, not a gate")
    evidence = parse_completion_evidence(payload["hands"][1]["hand"]["completion_evidence"])
    assert "pot_text_dropped" in evidence.warning_codes
    assert "pot_text_dropped" not in evidence.rejection_codes
    clean_evidence = parse_completion_evidence(payload["hands"][0]["hand"]["completion_evidence"])
    assert "pot_text_dropped" not in clean_evidence.warning_codes


def test_a_states_less_timeline_still_runs_every_hand_level_check() -> None:
    """THE ROUND-2 B4 REGRESSION. `validate_timeline(timeline) if
    isinstance(timeline.get("states"), list) else None` treated "the validator
    cannot see this timeline" as "no gate needed": stripping the states list
    from a real g0711 timeline exported 6 hands where the validated form
    exported 1, two of them carrying FATAL codes (amounts_unknown_in_ledger,
    starting_stack_unknown) and two landing completion "complete" -- reachable
    from any hand-edited, truncated, or third-party timeline, the exact
    producers MalformedTimeline exists for. With no states the state-level
    checks have nothing to examine, but every hand-level check still runs, so
    a hand the validator can reject from its own fields is still rejected."""
    fatal = _spine_hand(hand_number=1, warnings=["amounts_unknown_in_ledger"],
                        unknown_money_actions=1)
    clean = _spine_hand(hand_number=2, terminal_event="showdown")
    for timeline in ({"hands": [fatal, clean]},               # states missing
                     {"hands": [fatal, clean], "states": "x"}):  # states malformed
        payload = timeline_to_session_payload(
            timeline, timeline_path="t.json", session_name="S")
        skipped = {s["timeline_hand_number"]: s for s in payload["cv_import_summary"]["skipped"]}
        assert 1 in skipped, "a fatal-code hand must not export unvalidated"
        assert "amounts_unknown_in_ledger" in skipped[1]["codes"]
        assert payload["cv_import_summary"]["exported_hands"] == 1
