"""Round-trip completeness for the session import/export contract.

Phase 11 requires sessions, hands, players, actions, settlements, reviews,
coaching, corrections, issues, completion evidence and the relevant provenance
to survive an export followed by an import into another database. The tests here
prove that as a property of every field of every record rather than by spot
checking a handful of columns: one fully populated source database is exported,
imported into a second database, and the two are compared field by field.

Import deliberately does NOT preserve everything, and every such field is listed
in ``_EXEMPTIONS`` with the rule that replaces equality and the reason the value
may not travel. That makes each exemption a reviewable claim -- and a checked
one, because an exemption states what the imported value must be, not merely
that the comparison should be skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    IMPORTED_HAND_KEY,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    IMPORTED_ANALYSIS_STALE_REASON,
    IMPORTED_ISSUE_REOPEN_NOTE,
    export_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandCorrection,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SettlementEntry,
)


def _make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


# ---------------------------------------------------------------------------
# The exemption table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exemption:
    """One field that may legitimately differ across a round trip.

    ``reason`` is the justification a reviewer reads. ``expected`` receives the
    source value and the whole imported record and returns the value the
    imported field must hold, so an exemption never degrades into "ignore this
    column".
    """

    reason: str
    expected: Callable[[Any, dict[str, Any]], Any]


def _unchanged_but_stamped(source: Any, _imported: dict[str, Any]) -> Any:
    """Completion evidence survives verbatim apart from the import provenance stamp."""
    return {**source, IMPORTED_HAND_KEY: True}


_EXEMPTIONS: dict[tuple[str, str], Exemption] = {
    ("session", "id"): Exemption(
        "Autoincrement primary key. The importing database allocates its own, "
        "which is also what makes a second import a second session rather than "
        "an overwrite.",
        lambda _source, imported: imported["id"],
    ),
    ("hand", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("hand", "session_id"): Exemption(
        "Re-parented onto the session the import created.",
        lambda _source, imported: imported["session_id"],
    ),
    ("hand", "study_inclusion"): Exemption(
        "A local operator preference for the importing operator's Study queue, "
        "not a transferable fact about the hand. Reset to auto so a forged "
        "'skip' or 'study' is not attributed to the importer.",
        lambda _source, _imported: "auto",
    ),
    ("hand", "completion_evidence"): Exemption(
        "Stamped as imported. The manual-hand exemption is the argument 'you "
        "entered this hand yourself', which is false for a hand that arrived as "
        "user-supplied JSON.",
        _unchanged_but_stamped,
    ),
    ("player", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("player", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("action", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("action", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("settlement", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("settlement_entry", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("settlement_entry", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("review", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("review", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("review", "is_stale"): Exemption(
        "A retained review describes the rows of the database that produced it. "
        "Nothing here can verify it, so it lands stale and is re-run locally.",
        lambda _source, _imported: True,
    ),
    ("review", "stale_reason"): Exemption(
        "Set alongside is_stale when the payload declared none.",
        lambda source, _imported: source or IMPORTED_ANALYSIS_STALE_REASON,
    ),
    ("coaching", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("coaching", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("coaching", "session_id"): Exemption(
        "Re-parented onto the session the import created.",
        lambda _source, imported: imported["session_id"],
    ),
    ("coaching", "is_stale"): Exemption(
        "Same rule as a hand review: an assertion about evidence cannot travel "
        "in the payload that carries the evidence.",
        lambda _source, _imported: True,
    ),
    ("coaching", "stale_reason"): Exemption(
        "Set alongside is_stale when the payload declared none.",
        lambda source, _imported: source or IMPORTED_ANALYSIS_STALE_REASON,
    ),
    ("correction", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("correction", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("issue", "id"): Exemption(
        "Autoincrement primary key.",
        lambda _source, imported: imported["id"],
    ),
    ("issue", "hand_id"): Exemption(
        "Re-parented onto the hand the import created.",
        lambda _source, imported: imported["hand_id"],
    ),
    ("issue", "updated_at"): Exemption(
        "Stamped when the reopened issue is written here.",
        lambda _source, imported: imported["updated_at"],
    ),
}

# Fields whose exemption depends on what the source record held, so the rule is
# expressed per test rather than in the table above. Each is asserted explicitly
# by name in the tests that create the condition.
_CONDITIONAL_EXEMPTIONS = {
    ("hand", "review_status"),
    ("hand", "completion_status"),
    ("issue", "status"),
    ("issue", "resolved_at"),
    ("issue", "resolution_notes"),
    ("issue", "description"),
}


# Hand- and session-scoped tables that deliberately do NOT travel in a session
# payload, each with the reason. The guard test below derives the full set of
# such tables from the live schema and requires every one to be either exported
# or listed here, so a new table cannot silently fall outside the contract.
_NOT_EXPORTED: dict[str, str] = {
    "videos": (
        "A recording is a file on this machine's data mount, addressed by a "
        "path that means nothing in the importing database. The row would "
        "outlive the bytes it points at."
    ),
    "solver_runs": (
        "A run is metadata for three files on disk (command, result, log). "
        "Importing the metadata without the outputs would present a solve that "
        "cannot be opened."
    ),
}


def _hand_scoped_tables(db: PokerDatabase) -> set[str]:
    names = {
        row["name"]
        for row in db._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    scoped = set()
    for name in names:
        columns = {
            row["name"] for row in db._execute(f"PRAGMA table_info({name})").fetchall()
        }
        if columns & {"hand_id", "session_id"}:
            scoped.add(name)
    return scoped


# ---------------------------------------------------------------------------
# A source database with every exported entity populated.
# ---------------------------------------------------------------------------


def _complete_evidence() -> dict[str, object]:
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


def _seed_every_entity(db: PokerDatabase) -> Session:
    """One session carrying at least one row of every entity a payload exports."""

    session = db.create_session(
        Session(
            name="Every entity",
            platform="ClubWPT Gold",
            stakes="1/2 NL",
            notes="Round-trip source.",
        )
    )
    for number, source_type in ((1, "cv_import"), (2, "manual")):
        hand = db.create_hand(
            Hand(
                session_id=session.id,
                hand_number=number,
                game_type="No-limit Hold'em",
                blinds_antes="1/2 NL",
                table_size=6,
                effective_stack=100,
                hero_position="BTN",
                hero_cards="Ah Qs",
                board_cards="Qd 7s 2c",
                pot_size=30,
                result="Hero wins with top pair",
                hero_bb_won=12.5,
                confidence_score=0.8,
                source_type=source_type,
                tags=["BIG_POT", "RIVER_DECISION"],
                notes="Hand notes.",
                completion_evidence=(
                    _complete_evidence() if source_type == "cv_import" else {}
                ),
            )
        )
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="hero",
                seat_index=0,
                player_name="Hero",
                position="BTN",
                starting_stack=200,
                is_hero=True,
                notes="Player notes.",
            )
        )
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="villain",
                seat_index=3,
                player_name="Villain",
                position="BB",
                starting_stack=180,
            )
        )
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key="villain",
                street="preflop",
                action_index=1,
                player_name="Villain",
                position="BB",
                action_type="post_blind",
                amount=2,
                amount_semantics="raise_to",
                forced_bet_type="big_blind",
                is_live_post=True,
            )
        )
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key="hero",
                street="flop",
                action_index=1,
                player_name="Hero",
                position="BTN",
                action_type="bet",
                amount=10,
                amount_semantics="incremental",
                pot_before=5,
                stack_before=200,
                notes="Action notes.",
                # Per-action frame provenance, schema v16.
                source_image="data/frames/job_1/frame_0042.png",
            )
        )
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key="villain",
                street="flop",
                action_index=2,
                player_name="Villain",
                position="BB",
                action_type="call",
                amount=10,
                amount_semantics="incremental",
                forced_bet_type=None,
            )
        )
        db.upsert_hand_settlement(
            HandSettlement(
                hand_id=hand.id,
                status="settled",
                dead_money=1.5,
                rake_rate=0.05,
                rake_cap=3.0,
                rake_rounding_unit=0.01,
                no_flop_no_drop=True,
                gross_pot=25.0,
                rake_amount=1.0,
                net_pot=24.0,
                is_balanced=True,
                warnings=["assumed_uncalled_returned"],
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
                    amount=24.0,
                    entry_order=1,
                ),
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="refund",
                    player_key="villain",
                    player_name="Villain",
                    amount=2.0,
                    entry_order=1,
                ),
            ],
        )
        db.create_hand_review(
            HandReview(
                hand_id=hand.id,
                hand_summary="Top pair on a wet board.",
                theory_coach="Bet for value.",
                exploit_coach="Villain folds too much.",
                ev_math_notes="EV notes.",
                study_lesson="Size up.",
                next_review_question="Would a check-raise be better?",
                notes="Review notes.",
            )
        )
        db.create_coaching_response(
            CoachingResponse(
                hand_id=hand.id,
                session_id=session.id,
                provider_name="anthropic",
                model_name="claude",
                raw_prompt="prompt",
                raw_response="response",
                review_type="hand",
                parsed_sections={"theory": "text"},
            )
        )
        db.create_hand_correction(
            HandCorrection(
                hand_id=hand.id,
                correction_type="hand_facts",
                before_state={"board_cards": "Qd 7s 2d"},
                after_state={"board_cards": "Qd 7s 2c"},
                notes="Misread the club.",
            )
        )
        db.create_hand_issue(
            HandIssue(
                hand_id=hand.id,
                issue_types=["cards", "actions"],
                description="Board card looked wrong.",
                evidence_snapshot={"frozen": "at flag time"},
            )
        )
    db.create_coaching_response(
        CoachingResponse(
            session_id=session.id,
            provider_name="anthropic",
            model_name="claude",
            raw_prompt="session prompt",
            raw_response="session response",
            review_type="session",
            parsed_sections={"summary": "text"},
        )
    )
    return session


def _snapshot(db: PokerDatabase, session_id: int) -> dict[str, Any]:
    """Every exported entity of one session, as plain data, in a stable order."""

    session = db.fetch_session(session_id)
    assert session is not None
    hands = []
    for hand in db.fetch_hands_by_session(session_id):
        assert hand.id is not None
        hands.append(
            {
                "hand": hand.model_dump(mode="json"),
                "player": [
                    player.model_dump(mode="json")
                    for player in sorted(
                        db.fetch_players_by_hand(hand.id),
                        key=lambda item: item.player_key,
                    )
                ],
                "action": [
                    action.model_dump(mode="json")
                    for action in sorted(
                        db.fetch_actions_by_hand(hand.id),
                        key=lambda item: (item.street, item.action_index or 0),
                    )
                ],
                "settlement": (
                    None
                    if (found := db.fetch_hand_settlement(hand.id)) is None
                    else found.model_dump(mode="json")
                ),
                "settlement_entry": [
                    entry.model_dump(mode="json")
                    for entry in sorted(
                        db.fetch_settlement_entries(hand.id),
                        key=lambda item: (item.entry_type, item.entry_order),
                    )
                ],
                "review": [
                    review.model_dump(mode="json")
                    for review in db.fetch_reviews_by_hand(hand.id)
                ],
                "coaching": [
                    review.model_dump(mode="json")
                    for review in db.fetch_coaching_reviews_by_hand(hand.id)
                ],
                "correction": [
                    correction.model_dump(mode="json")
                    for correction in db.fetch_hand_corrections(hand.id)
                ],
                "issue": [
                    issue.model_dump(mode="json")
                    for issue in db.fetch_hand_issues(hand_id=hand.id)
                ],
            }
        )
    return {
        "session": session.model_dump(mode="json"),
        "hands": hands,
        "session_coaching": [
            review.model_dump(mode="json")
            for review in db.fetch_coaching_reviews_by_session(session_id)
            if review.hand_id is None
        ],
    }


def _compare_record(
    kind: str,
    source: dict[str, Any],
    imported: dict[str, Any],
    path: str,
    failures: list[str],
) -> None:
    assert set(source) == set(imported), f"{path}: field sets differ"
    for name, value in source.items():
        if (kind, name) in _CONDITIONAL_EXEMPTIONS:
            continue
        exemption = _EXEMPTIONS.get((kind, name))
        if exemption is None:
            if imported[name] != value:
                failures.append(
                    f"{path}.{name}: {value!r} did not survive the round trip "
                    f"(imported {imported[name]!r}) and is not a declared exemption"
                )
            continue
        expected = exemption.expected(value, imported)
        if imported[name] != expected:
            failures.append(
                f"{path}.{name}: exemption '{exemption.reason[:40]}...' expected "
                f"{expected!r}, found {imported[name]!r}"
            )


def _compare(source: dict[str, Any], imported: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    _compare_record("session", source["session"], imported["session"], "session", failures)
    assert len(source["hands"]) == len(imported["hands"]), "hand count changed"
    for index, (source_hand, imported_hand) in enumerate(
        zip(source["hands"], imported["hands"], strict=True)
    ):
        base = f"hands[{index}]"
        _compare_record(
            "hand", source_hand["hand"], imported_hand["hand"], f"{base}.hand", failures
        )
        for kind in (
            "player",
            "action",
            "settlement_entry",
            "review",
            "coaching",
            "correction",
            "issue",
        ):
            assert len(source_hand[kind]) == len(imported_hand[kind]), (
                f"{base}.{kind}: {len(source_hand[kind])} exported, "
                f"{len(imported_hand[kind])} imported"
            )
            for item_index, (source_item, imported_item) in enumerate(
                zip(source_hand[kind], imported_hand[kind], strict=True)
            ):
                _compare_record(
                    kind,
                    source_item,
                    imported_item,
                    f"{base}.{kind}[{item_index}]",
                    failures,
                )
        assert (source_hand["settlement"] is None) == (
            imported_hand["settlement"] is None
        ), f"{base}.settlement presence changed"
        if source_hand["settlement"] is not None:
            _compare_record(
                "settlement",
                source_hand["settlement"],
                imported_hand["settlement"],
                f"{base}.settlement",
                failures,
            )
    assert len(source["session_coaching"]) == len(imported["session_coaching"])
    for index, (source_item, imported_item) in enumerate(
        zip(source["session_coaching"], imported["session_coaching"], strict=True)
    ):
        _compare_record(
            "coaching",
            source_item,
            imported_item,
            f"session_coaching[{index}]",
            failures,
        )
    return failures


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_every_populated_entity_survives_a_round_trip_field_by_field() -> None:
    """Export a fully populated session, import it, compare every field.

    The importing database is NOT empty: it already holds an unrelated session
    and hand, so every autoincrement id differs between the two databases and an
    id exemption that had quietly become an identity comparison would fail.
    """
    source = _make_db()
    session = _seed_every_entity(source)
    payload = export_session(source, session.id)

    target = _make_db()
    decoy = target.create_session(Session(name="Already here"))
    target.create_hand(Hand(session_id=decoy.id, hand_number=1))

    imported = import_session(target, payload)
    assert imported.id != session.id, "the fixture must force differing ids"

    failures = _compare(_snapshot(source, session.id), _snapshot(target, imported.id))
    assert failures == []

    source.close()
    target.close()


def test_every_hand_scoped_table_is_either_exported_or_declared_not_exported() -> None:
    """A new hand- or session-scoped table cannot fall outside the contract silently.

    The set is derived from the live schema rather than written down, so adding
    a table with a hand_id or session_id forces a decision here about whether it
    travels.
    """
    db = _make_db()
    exported = {
        "sessions",
        "hands",
        "hand_players",
        "actions",
        "hand_settlements",
        "settlement_entries",
        "hand_reviews",
        "coaching_reviews",
        "hand_corrections",
        "hand_issues",
    }
    scoped = _hand_scoped_tables(db) | {"sessions"}

    undeclared = scoped - exported - set(_NOT_EXPORTED)
    assert undeclared == set(), (
        "these tables carry a hand_id or session_id but neither travel in a "
        f"session payload nor say why they do not: {sorted(undeclared)}"
    )
    assert all(reason.strip() for reason in _NOT_EXPORTED.values())
    db.close()


def test_the_exemption_table_names_only_real_fields() -> None:
    """A typo in an exemption key would silently disable a field comparison."""
    fields = {
        "session": set(Session.model_fields),
        "hand": set(Hand.model_fields),
        "player": set(HandPlayer.model_fields),
        "action": set(Action.model_fields),
        "settlement": set(HandSettlement.model_fields),
        "settlement_entry": set(SettlementEntry.model_fields),
        "review": set(HandReview.model_fields),
        "coaching": set(CoachingResponse.model_fields),
        "correction": set(HandCorrection.model_fields),
        "issue": set(HandIssue.model_fields),
    }
    for kind, name in list(_EXEMPTIONS) + list(_CONDITIONAL_EXEMPTIONS):
        assert kind in fields, f"unknown record kind {kind!r} in the exemption table"
        assert name in fields[kind], f"{kind} has no field {name!r}"


def test_provenance_columns_survive_the_round_trip() -> None:
    """The columns that say where a reconstructed fact came from are not decoration."""
    source = _make_db()
    session = _seed_every_entity(source)
    payload = export_session(source, session.id)

    target = _make_db()
    imported = import_session(target, payload)
    hand = target.fetch_hands_by_session(imported.id)[0]
    actions = target.fetch_actions_by_hand(hand.id)

    assert [action.source_image for action in actions] == [
        None,
        "data/frames/job_1/frame_0042.png",
        None,
    ]
    assert [action.forced_bet_type for action in actions] == ["big_blind", None, None]
    assert [action.is_live_post for action in actions] == [True, None, None]
    assert hand.source_type == "cv_import"
    evidence = hand.completion_evidence
    assert evidence["source_frames"] == ["frames/a.png", "frames/b.png"]
    assert evidence["pipeline_version"] == "two-model-v7"
    assert evidence["model_versions"] == {"detector": "v7"}
    source.close()
    target.close()


def test_a_resolved_issue_round_trips_reopened_and_keeps_its_evidence() -> None:
    """The conditional exemptions on hand_issues, asserted by name.

    A resolution says somebody looked at this hand and fixed the thing; the
    importing operator has looked at nothing. What travels is the evidence, the
    types and the previous resolution text, folded into the description.
    """
    source = _make_db()
    session = source.create_session(Session(name="Resolved issue"))
    hand = source.create_hand(Hand(session_id=session.id, hand_number=1))
    issue = source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            # Deliberately not a release-blocking type: closing one of those
            # needs a promoted regression case, which is a different fixture and
            # a different requirement.
            issue_types=["coaching"],
            description="Pot read 47, ledger says 42.",
            evidence_snapshot={"pot_size": 47},
        )
    )
    source.resolve_hand_issue(issue.id, resolution_notes="Corrected the pot to 42.")
    payload = export_session(source, session.id)

    target = _make_db()
    imported = import_session(target, payload)
    landed = target.fetch_hand_issues(
        hand_id=target.fetch_hands_by_session(imported.id)[0].id
    )[0]

    assert landed.status == "open"
    assert landed.resolved_at is None
    assert landed.resolution_notes == ""
    assert landed.evidence_snapshot == {"pot_size": 47}
    assert landed.issue_types == ["coaching"]
    assert "Pot read 47" in landed.description
    assert IMPORTED_ISSUE_REOPEN_NOTE in landed.description
    assert "Corrected the pot to 42." in landed.description
    source.close()
    target.close()


@pytest.mark.parametrize("declared", ["reviewed", "needs_correction", "unreviewed"])
def test_review_status_never_travels_upward(declared: str) -> None:
    """The conditional exemption on hands.review_status, asserted by name."""
    source = _make_db()
    session = source.create_session(Session(name="Review status"))
    hand = source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Qs",
            source_type="manual",
        )
    )
    if declared != "unreviewed":
        source.update_hand_status(hand.id, declared)
    payload = export_session(source, session.id)

    target = _make_db()
    landed = target.fetch_hands_by_session(import_session(target, payload).id)[0]

    assert landed.review_status == (
        "needs_correction" if declared == "reviewed" else declared
    )
    source.close()
    target.close()
