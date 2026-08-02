"""Phase 10 coverage for the Hands library and the Study workspace.

Two claims carry this file. The first is that the Hands list cannot show a
reconstruction-confidence grade, or a clean row, after a correction has
invalidated what those readings described -- proven by correcting a hand
*between* two renders of the same running app, so a cached badge fails. The
second is that Study's queue navigation survives a mutation that changes which
hands belong in the queue, without stranding the operator on a hand it can no
longer show and without silently swapping the hand under them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import PokerDatabase
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
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.ui.session_library import filter_hands, hand_search_text
from poker_tracker.ui.view_models import build_hand_rows, reconstruction_confidence
from tests.conftest import attest_declared_assumptions

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


# A reconstruction whose stored evidence derives 'complete', so the store's
# promotion floor accepts 'reviewed' on it. Seeding a hand as reviewed directly
# does not survive: adding a seat or an action demotes it, by design.
CLEAN_EVIDENCE: dict[str, object] = {
    "evidence_version": 1,
    "partial_start": False,
    "partial_end": False,
    "terminal_event": "showdown",
    "first_source_timestamp_s": 1.0,
    "last_source_timestamp_s": 9.0,
    "preceding_boundary": {
        "kind": "hand_start",
        "timestamp_s": 1.0,
        "frame_ref": "frames/start.png",
        "confidence": 0.94,
        "codes": [],
    },
    "following_boundary": {
        "kind": "hand_end",
        "timestamp_s": 9.0,
        "frame_ref": "frames/end.png",
        "confidence": 0.91,
        "codes": [],
    },
    "boundary_confidence": 0.91,
    "source_frames": ["frames/start.png", "frames/end.png"],
    "warning_codes": [],
    "rejection_codes": [],
    "acknowledged_codes": [],
    "layout_profile": "clubwpt-6max",
    "layout_supported": True,
    "table_size": 6,
    "pipeline_version": "two-model-v7",
    "model_versions": {"detector": "v7"},
}


def _open_db(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _seed_cv_hand(
    db: PokerDatabase,
    session: Session,
    *,
    hand_number: int = 1,
    confidence_score: float | None = 0.95,
    review_status: str = "reviewed",
    completion_status: str = "complete",
    source_type: str = "cv_import",
    hero_cards: str = "Ah Qs",
    hero_bb_won: float | None = 10,
    reconcile: bool = False,
) -> Hand:
    """One reconstructed hand with two seats, optionally reconciled."""

    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=hand_number,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="BTN",
            hero_cards=hero_cards,
            board_cards="Qd 7s 2c",
            pot_size=20,
            hero_bb_won=hero_bb_won,
            review_status="unreviewed",
            source_type=source_type,
            completion_status=completion_status,
            completion_evidence=CLEAN_EVIDENCE,
            confidence_score=confidence_score,
            study_inclusion="study",
        )
    )
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            player_name="Hero",
            seat_index=1,
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="villain",
            player_name="Villain",
            seat_index=2,
            position="BB",
            starting_stack=100,
        )
    )
    for player_key, player_name, action_type in (
        ("hero", "Hero", "bet"),
        ("villain", "Villain", "call"),
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                street="river",
                player_key=player_key,
                player_name=player_name,
                action_type=action_type,
                amount=10,
                amount_semantics="incremental",
            )
        )
    if reconcile:
        db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key="hero",
                    player_name="Hero",
                    amount=20,
                    entry_order=1,
                )
            ],
        )
        persist_reconciliation(db, hand.id)
        attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    if review_status != "unreviewed":
        # After the seats and actions, because each of those demotes a promoted
        # hand -- the store refuses to let 'reviewed' outlive the evidence it was
        # granted on, and seeding around that would test a state the app cannot
        # produce.
        db.update_hand_status(hand.id, review_status)
    return db.fetch_hand(hand.id)


def _seed_coaching(db: PokerDatabase, hand: Hand, session: Session) -> CoachingResponse:
    return db.create_coaching_response(
        CoachingResponse(
            provider_name="stub",
            model_name="stub-1",
            raw_prompt="post-session review",
            raw_response="Recorded line looks fine.",
            review_type="hand",
            hand_id=hand.id,
            session_id=session.id,
            parsed_sections={"hand_summary": "Bet flop, took it down."},
        )
    )


def _correct_hero_cards(path: Path, hand_id: int, cards: str = "Kh Kd") -> None:
    """Apply a real correction through the store, exactly as the fact editor does."""

    db = _open_db(path)
    stored = db.fetch_hand(hand_id)
    db.update_hand_facts(
        stored.model_copy(update={"hero_cards": cards}),
        correction_notes="Reconstruction read the wrong hero cards.",
    )
    db.close()


# ---------------------------------------------------------------------------
# App mounting
# ---------------------------------------------------------------------------


def _configure_app_env(path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _mount(path: Path, monkeypatch, name: str, body: list[str]) -> AppTest:
    """Run one app surface in isolation against ``path``.

    A driver script rather than the whole shell, because these assertions are
    about one surface and the shell would otherwise decide which page is showing.
    The script imports the already-imported ``app`` module, so ``monkeypatch``
    applied to ``app.<name>`` in the test process is in force inside the run.
    """

    _configure_app_env(path, monkeypatch)
    script = path.parent / f"_mount_{name}.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                *body,
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not list(app.exception), [str(item) for item in app.exception]
    return app


def _mount_hands(path: Path, monkeypatch) -> AppTest:
    return _mount(
        path,
        monkeypatch,
        "hands",
        ["app_module.show_hands_workspace(db)"],
    )


def _mount_study(path: Path, monkeypatch) -> AppTest:
    return _mount(
        path,
        monkeypatch,
        "study",
        [
            "sessions = db.fetch_sessions()",
            "app_module.show_study_workspace(db, sessions[0] if sessions else None)",
        ],
    )


def _segmented(app: AppTest, key: str):
    """``st.segmented_control`` reaches AppTest as a button_group element."""

    return next(item for item in app.get("button_group") if item.key == key)


def _page_text(app: AppTest) -> str:
    """Every rendered string on the page, badges and captions included."""

    parts: list[str] = []
    for collection in (
        app.markdown,
        app.caption,
        app.warning,
        app.info,
        app.error,
        app.success,
    ):
        parts.extend(str(item.value) for item in collection)
    parts.extend(str(item.label) for item in app.button)
    parts.extend(str(item.label) for item in app.expander)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The confidence reading itself
# ---------------------------------------------------------------------------


def test_reconstruction_confidence_retires_a_grade_a_correction_overruled() -> None:
    """The score describes a read; a correction replaces the read it described."""

    draft = Hand(session_id=1, hand_number=1, source_type="cv_import", confidence_score=0.95)
    corrected = draft.model_copy(update={"source_type": "corrected_cv"})
    manual = Hand(
        session_id=1,
        hand_number=2,
        source_type="manual",
        completion_status="not_applicable",
    )

    assert reconstruction_confidence(draft).label == "High"
    assert reconstruction_confidence(draft).describes_current_facts

    retired = reconstruction_confidence(corrected)
    assert retired.label == "Superseded by your corrections"
    assert not retired.describes_current_facts
    # The original grade is still legible, as history, inside the sentence that
    # says it is history -- never as the headline word.
    assert "High" in retired.detail
    assert "corrected" in retired.detail

    assert reconstruction_confidence(manual).label == "Not applicable"
    assert not reconstruction_confidence(manual).describes_current_facts


def test_hand_row_view_model_carries_the_retired_reading_too() -> None:
    """The unused view model cannot become a second, staler source of the badge."""

    session = Session(id=1, name="Night")
    corrected = Hand(
        id=7,
        session_id=1,
        hand_number=1,
        source_type="corrected_cv",
        confidence_score=0.95,
    )

    row = build_hand_rows([session], [corrected], {})[0]

    assert row.confidence_label == "Superseded by your corrections"
    assert row.confidence_is_current is False


# ---------------------------------------------------------------------------
# Hands: evidence state without entering Study
# ---------------------------------------------------------------------------


def test_hand_library_shows_issue_stale_and_source_state_on_the_row(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "states.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Tuesday", date_played=date(2026, 7, 28)))
    hand = _seed_cv_hand(db, session)
    _seed_coaching(db, hand, session)
    db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards"],
            description="Turn card unreadable in the recording.",
        )
    )
    db.close()
    # The issue write demotes the hand; the coaching row is staled by the same
    # workflow. Both are states the row has to report without opening Study.
    _correct_hero_cards(path, hand.id)

    app = _mount_hands(path, monkeypatch)
    text = _page_text(app)

    assert "1 open issue" in text
    assert "Stale analysis" in text
    assert "Corrected CV" in text
    # Every one of those states is carried by words, not only by a badge colour.
    assert "Reconstruction confidence · Superseded by your corrections" in text


def test_hands_list_confidence_cannot_survive_a_correction(tmp_path, monkeypatch) -> None:
    """The load-bearing one: a correction lands between two renders of one app.

    Any memoisation of the row -- ``st.cache_data`` over the hand list, a
    session-state copy of the badge, a view model built once at first render --
    leaves the first render's "High" on the page and fails here.
    """

    path = tmp_path / "correction.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Wednesday", date_played=date(2026, 7, 29)))
    hand = _seed_cv_hand(db, session)
    _seed_coaching(db, hand, session)
    db.close()

    app = _mount_hands(path, monkeypatch)
    before = _page_text(app)
    assert "Reconstruction confidence · High" in before
    assert "Stale analysis" not in before

    _correct_hero_cards(path, hand.id)

    # Same AppTest instance, same widget state, same cached database resource.
    app.run()
    assert not list(app.exception)
    after = _page_text(app)

    assert "Reconstruction confidence · High" not in after
    assert "Reconstruction confidence · Superseded by your corrections" in after
    assert "Stale analysis" in after

    # And again through a rerun the operator causes, so a badge keyed on widget
    # state rather than on the hand is caught as well.
    app.text_input(key="hand_library_search").set_value("kh").run()
    assert not list(app.exception)
    typed = _page_text(app)
    assert "Reconstruction confidence · High" not in typed
    assert "Reconstruction confidence · Superseded by your corrections" in typed
    st.cache_resource.clear()


def test_every_hands_list_path_renders_badges_from_the_same_builder() -> None:
    """One builder, so a second list view cannot grow its own staler badge.

    ``render_hand_results`` is mounted by the Hands library and by the session
    hand browser. A row rendered by either has to reach its evidence words
    through ``hand_evidence_badges``, which reads ``reconstruction_confidence``,
    or the guarantee above holds on one surface and not the other.
    """

    source = Path(APP_PATH).read_text()
    assert source.count("def hand_evidence_badges(") == 1
    assert source.count("hand_evidence_badges(") == 2  # definition plus one call
    builder_start = source.index("def hand_evidence_badges(")
    builder_end = source.index("\ndef ", builder_start + 1)
    assert "reconstruction_confidence(" in source[builder_start:builder_end]
    # Nothing outside the builder may format a raw confidence grade.
    assert "confidence_label(" not in source


# ---------------------------------------------------------------------------
# Hands: search, filters, and the debugging inbox
# ---------------------------------------------------------------------------


def test_hand_search_finds_a_hand_by_what_was_written_on_its_issue() -> None:
    session = Session(id=1, name="Night", date_played=date(2026, 8, 1))
    flagged = Hand(id=1, session_id=1, hand_number=1, hero_cards="Ah Qs")
    clean = Hand(id=2, session_id=1, hand_number=2, hero_cards="7c 2d")
    sessions_by_id = {1: session}
    issue_text = {1: "Cards Turn card unreadable in the recording"}

    found = filter_hands(
        [flagged, clean],
        sessions_by_id,
        query="unreadable",
        issue_text_by_hand=issue_text,
        issue_hand_ids={1},
    )

    assert found == [flagged]
    # And the word the badge shows is searchable too.
    assert "open issue" in hand_search_text(flagged, session, has_open_issue=True)
    assert "august" in hand_search_text(flagged, session)


def test_hand_flag_and_source_filters_narrow_to_the_named_population() -> None:
    session = Session(id=1, name="Night", date_played=date(2026, 8, 1))
    flagged = Hand(id=1, session_id=1, hand_number=1, source_type="cv_import")
    stale = Hand(id=2, session_id=1, hand_number=2, source_type="corrected_cv")
    clean = Hand(id=3, session_id=1, hand_number=3, source_type="manual")
    hands = [flagged, stale, clean]
    sessions_by_id = {1: session}

    def narrow(**kwargs) -> list[int]:
        return [
            hand.id
            for hand in filter_hands(
                hands, sessions_by_id, issue_hand_ids={1}, stale_hand_ids={2}, **kwargs
            )
        ]

    assert narrow(flag_filter="open_issue") == [1]
    assert narrow(flag_filter="stale") == [2]
    assert narrow(flag_filter="clean") == [3]
    assert narrow(source_filter="corrected_cv") == [2]
    assert narrow(source_filter="manual") == [3]
    assert narrow() == [1, 2, 3]


def test_issue_inbox_counts_only_the_issues_it_can_open(tmp_path, monkeypatch) -> None:
    """A header claiming more open work than it lists is an unsubstantiated count."""

    path = tmp_path / "inbox.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Thursday", date_played=date(2026, 7, 30)))
    listed = _seed_cv_hand(db, session, hand_number=1)
    orphan = _seed_cv_hand(db, session, hand_number=2)
    for hand in (listed, orphan):
        db.create_hand_issue(
            HandIssue(
                hand_id=hand.id,
                issue_types=["actions"],
                description=f"Missing raise on hand {hand.hand_number}.",
            )
        )
    db.close()

    app = _mount(
        path,
        monkeypatch,
        "inbox",
        [
            "sessions = db.fetch_sessions()",
            "hands = db.fetch_hands_by_session(sessions[0].id)",
            # One hand is deliberately withheld from the library map, which is
            # what a deleted hand with a surviving issue row looks like here.
            "hands_by_id = {hands[0].id: hands[0]}",
            "app_module.show_hand_issue_queue(",
            "    db,",
            "    db.fetch_hand_issues(status='open'),",
            "    hands_by_id,",
            "    {sessions[0].id: sessions[0]},",
            ")",
        ],
    )
    text = _page_text(app)

    assert "Saved debugging issue queue (1 open)" in text
    assert "1 further open issue(s) reference hands that are no longer" in text
    assert any(button.label == "Open" for button in app.button)
    st.cache_resource.clear()


def test_issue_inbox_open_button_is_idempotent_across_reruns(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "inbox_idempotent.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Friday", date_played=date(2026, 7, 31)))
    hand = _seed_cv_hand(db, session)
    db.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["stacks"], description="Stack misread.")
    )
    session_id = session.id
    db.close()

    app = _mount(
        path,
        monkeypatch,
        "inbox_idem",
        [
            "sessions = db.fetch_sessions()",
            "hands = db.fetch_hands_by_session(sessions[0].id)",
            "app_module.show_hand_issue_queue(",
            "    db, db.fetch_hand_issues(status='open'),",
            "    {item.id: item for item in hands},",
            "    {sessions[0].id: sessions[0]},",
            ")",
        ],
    )
    opener = next(button for button in app.button if button.label == "Open")
    opener.click().run()
    assert not list(app.exception)
    assert app.session_state["active_session_id"] == session_id
    # A second rerun without touching the button must land on the same target
    # rather than re-firing the opener or accumulating a second one.
    app.run()
    assert not list(app.exception)
    assert app.session_state["active_session_id"] == session_id
    assert not list(app.exception)
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Hands: large-list bounds
# ---------------------------------------------------------------------------


def test_hands_page_bounds_reconciliation_to_the_rendered_page(
    tmp_path, monkeypatch
) -> None:
    """Per-render cost follows the page, not the library.

    A count, not a clock: the assertion is on how many reconciliations one render
    performs, which is stable across machines. Before pagination was allowed to
    choose first, a 90-hand library cost 90 reconciliations on every keystroke in
    the search box.
    """

    path = tmp_path / "large.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Grind", date_played=date(2026, 7, 20)))
    for number in range(1, 91):
        _seed_cv_hand(db, session, hand_number=number, reconcile=True)
    db.close()

    import app as app_module

    calls: list[int] = []
    original = app_module.reconcile_persisted_hand

    def _counted(database, hand_id):
        calls.append(hand_id)
        return original(database, hand_id)

    monkeypatch.setattr(app_module, "reconcile_persisted_hand", _counted)

    app = _mount_hands(path, monkeypatch)
    first_render = len(calls)

    assert "90 of 90 saved hands match." in _page_text(app)
    # One page is twenty rows. The bound is the page plus nothing: no per-row
    # staleness query, no library-wide reconciliation pass.
    assert first_render <= 20, first_render

    calls.clear()
    app.text_input(key="hand_library_search").set_value("grind").run()
    assert not list(app.exception)
    assert len(calls) <= 20, len(calls)

    calls.clear()
    next(button for button in app.button if button.label == "Next →").click().run()
    assert not list(app.exception)
    assert len(calls) <= 20, len(calls)
    assert "showing 21–40" in _page_text(app)
    st.cache_resource.clear()


def test_unsettled_hands_are_never_reconciled_for_a_list(tmp_path, monkeypatch) -> None:
    """A hand with no reconciled settlement cannot produce a derived result.

    So the list must not pay two ledger builds to discover that. This is the
    bound that holds even when a filter forces the whole library to be resolved.
    """

    path = tmp_path / "unsettled.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Drafts", date_played=date(2026, 7, 21)))
    for number in range(1, 41):
        _seed_cv_hand(db, session, hand_number=number, reconcile=False)
    db.close()

    import app as app_module

    calls: list[int] = []
    original = app_module.reconcile_persisted_hand

    def _counted(database, hand_id):
        calls.append(hand_id)
        return original(database, hand_id)

    monkeypatch.setattr(app_module, "reconcile_persisted_hand", _counted)

    app = _mount_hands(path, monkeypatch)
    assert calls == []

    # Engaging the result filter is the one control that has to resolve results
    # before paginating; it still reconciles nothing here, because nothing here
    # could have been substituted.
    _segmented(app, "hand_library_result").set_value("wins").run()
    assert not list(app.exception)
    assert calls == []
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Study: navigation and the queue after mutations
# ---------------------------------------------------------------------------


def _seed_study_queue(path: Path, count: int = 3) -> tuple[int, list[int]]:
    db = _open_db(path)
    session = db.create_session(Session(name="Study night", date_played=date(2026, 7, 25)))
    ids = [
        _seed_cv_hand(db, session, hand_number=number, review_status="reviewed").id
        for number in range(1, count + 1)
    ]
    session_id = session.id
    db.close()
    return session_id, ids


def test_study_next_and_previous_walk_the_queue(tmp_path, monkeypatch) -> None:
    path = tmp_path / "walk.sqlite3"
    _, ids = _seed_study_queue(path)

    app = _mount_study(path, monkeypatch)
    assert "Hand 1 of 3" in _page_text(app)

    next(button for button in app.button if button.label == "Next hand →").click().run()
    assert not list(app.exception)
    assert "Hand 2 of 3" in _page_text(app)
    assert app.session_state["study_hand_id"] == ids[1]

    next(
        button for button in app.button if button.label == "← Previous hand"
    ).click().run()
    assert not list(app.exception)
    assert "Hand 1 of 3" in _page_text(app)
    assert app.session_state["study_hand_id"] == ids[0]
    st.cache_resource.clear()


def test_study_queue_shrinking_under_the_operator_does_not_skip_a_hand(
    tmp_path, monkeypatch
) -> None:
    """A correction to a hand EARLIER in the queue must not move the open one."""

    path = tmp_path / "shrink.sqlite3"
    _, ids = _seed_study_queue(path)

    app = _mount_study(path, monkeypatch)
    next(button for button in app.button if button.label == "Next hand →").click().run()
    assert app.session_state["study_hand_id"] == ids[1]

    # Hand 1 is corrected elsewhere, which demotes it out of the reviewed queue.
    _correct_hero_cards(path, ids[0])
    app.run()
    assert not list(app.exception)

    text = _page_text(app)
    # The operator is still on the hand they opened, now first of two, and the
    # denominator moved with the queue rather than being left at three.
    assert app.session_state["study_hand_id"] == ids[1]
    assert "Hand 1 of 2" in text
    # Next still lands on hand 3, which is what "does not skip" means here.
    next(button for button in app.button if button.label == "Next hand →").click().run()
    assert not list(app.exception)
    assert app.session_state["study_hand_id"] == ids[2]
    assert "Hand 2 of 2" in _page_text(app)
    st.cache_resource.clear()


def test_study_does_not_strand_the_operator_on_a_hand_it_cannot_show(
    tmp_path, monkeypatch
) -> None:
    """Correcting the OPEN hand pins Study to it; there must be a way off."""

    path = tmp_path / "strand.sqlite3"
    _, ids = _seed_study_queue(path)

    app = _mount_study(path, monkeypatch)
    next(button for button in app.button if button.label == "Next hand →").click().run()
    assert app.session_state["study_hand_id"] == ids[1]

    _correct_hero_cards(path, ids[1])
    app.run()
    assert not list(app.exception)

    text = _page_text(app)
    assert "is not approved for study yet" in text
    assert "Study is pinned to hand #2" in text

    release = next(
        button
        for button in app.button
        if button.label == "Leave this hand and show the study queue"
    )
    release.click().run()
    assert not list(app.exception)
    # The pin is gone: Study is back on the queue, not on the hand it could not
    # show, and the remaining approved hands are reachable again.
    assert app.session_state["study_hand_id"] in {ids[0], ids[2]}
    assert "Hand 1 of 2" in _page_text(app)
    assert "is not approved for study yet" not in _page_text(app)
    st.cache_resource.clear()


def test_study_says_so_when_the_open_hand_left_the_queue_entirely(
    tmp_path, monkeypatch
) -> None:
    """A deleted hand must not be replaced by a different one in silence."""

    path = tmp_path / "deleted.sqlite3"
    _, ids = _seed_study_queue(path)

    app = _mount_study(path, monkeypatch)
    next(button for button in app.button if button.label == "Next hand →").click().run()
    assert app.session_state["study_hand_id"] == ids[1]

    db = _open_db(path)
    db.delete_hand(ids[1])
    db.close()

    app.run()
    assert not list(app.exception)
    text = _page_text(app)
    assert "no longer in this study queue" in text
    assert app.session_state["study_hand_id"] == ids[0]
    st.cache_resource.clear()


def test_study_navigation_reports_a_superseded_confidence(tmp_path, monkeypatch) -> None:
    """The Study caption goes through the same retirement rule as the list.

    Two hands rather than one correction applied mid-run, because a correction
    demotes the hand out of Study by design and the caption under test only
    exists on a queued hand. The pipeline scored both at 0.95; only the one the
    operator has corrected is barred from showing that as a current grade.
    """

    path = tmp_path / "study_confidence.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Confidence", date_played=date(2026, 7, 26)))
    _seed_cv_hand(db, session, hand_number=1, review_status="reviewed")
    _seed_cv_hand(
        db,
        session,
        hand_number=2,
        review_status="reviewed",
        source_type="corrected_cv",
    )
    db.close()

    app = _mount_study(path, monkeypatch)
    first = _page_text(app)
    assert "Reconstruction confidence · High" in first
    assert "Superseded by your corrections" not in first

    next(button for button in app.button if button.label == "Next hand →").click().run()
    assert not list(app.exception)
    second = _page_text(app)
    assert "Reconstruction confidence · High" not in second
    assert "Reconstruction confidence · Superseded by your corrections" in second
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Study: historical evidence stays visibly historical
# ---------------------------------------------------------------------------


def test_stale_coaching_is_separated_by_words_not_only_colour(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "coaching.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Coached", date_played=date(2026, 7, 24)))
    hand = _seed_cv_hand(db, session, review_status="reviewed")
    _seed_coaching(db, hand, session)
    db.close()

    # A correction stales the saved review; a second review is then current.
    _correct_hero_cards(path, hand.id)
    db = _open_db(path)
    _seed_coaching(db, db.fetch_hand(hand.id), session)
    db.close()

    app = _mount(
        path,
        monkeypatch,
        "coach",
        [
            "sessions = db.fetch_sessions()",
            "hand = db.fetch_hands_by_session(sessions[0].id)[0]",
            "from poker_tracker.services.study_readiness import evaluate_study_readiness",
            "readiness = evaluate_study_readiness(hand, accounting=None)",
            "app_module.show_study_coach_review(",
            "    db, sessions[0], hand,",
            "    db.fetch_actions_by_hand(hand.id),",
            "    db.fetch_players_by_hand(hand.id),",
            "    None, None,",
            "    db.fetch_coaching_reviews_by_hand(hand.id),",
            "    readiness,",
            ")",
        ],
    )
    text = _page_text(app)

    assert "Retained stale coaching" in text
    assert "1 prior coaching review(s) are retained as stale evidence." in text
    assert any("STALE ·" in item.label for item in app.expander)
    assert not any("CURRENT ·" in item.label for item in app.expander)
    st.cache_resource.clear()


def test_stale_solver_run_is_named_as_stale(tmp_path, monkeypatch) -> None:
    path = tmp_path / "solver.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Solved", date_played=date(2026, 7, 23)))
    hand = _seed_cv_hand(db, session, review_status="reviewed")
    db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="completed",
            input_hash="fixture-hash",
            evidence={
                "backend": "texassolver",
                "backend_version": "v0.0.0",
                "action_frequencies": [{"action": "bet", "frequency": 1.0}],
            },
        )
    )
    db.close()
    _correct_hero_cards(path, hand.id)

    app = _mount(
        path,
        monkeypatch,
        "solver",
        [
            "sessions = db.fetch_sessions()",
            "hand = db.fetch_hands_by_session(sessions[0].id)[0]",
            "from poker_tracker.services.study_readiness import evaluate_study_readiness",
            "runs = db.fetch_solver_runs_by_hand(hand.id)",
            "readiness = evaluate_study_readiness(hand, accounting=None, solver_runs=runs)",
            "app_module._show_solver_runs(",
            "    db, sessions[0], hand,",
            "    db.fetch_players_by_hand(hand.id),",
            "    db.fetch_actions_by_hand(hand.id),",
            "    None, None, runs, readiness,",
            ")",
        ],
    )
    text = _page_text(app)

    assert "Status · Stale" in text
    assert "Hand evidence changed; rerun solver analysis." in text
    # And no strategy frequencies are drawn beside that word: a stale run's
    # numbers are withheld, not printed with a coloured heading over them.
    assert not app.get("progress")
    assert "Recorded Hero combo strategy" not in text
    assert any(button.label == "Delete stale run" for button in app.button)
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Study-adjacent edits: audit history and targeted invalidation
# ---------------------------------------------------------------------------


def test_player_edit_requires_a_reason_writes_an_audit_row_and_invalidates_narrowly(
    tmp_path, monkeypatch
) -> None:
    """Validation, confirmation, audit history, and invalidation that is targeted."""

    path = tmp_path / "player_edit.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Edits", date_played=date(2026, 7, 22)))
    edited = _seed_cv_hand(db, session, hand_number=1, review_status="reviewed")
    neighbour = _seed_cv_hand(db, session, hand_number=2, review_status="reviewed")
    _seed_coaching(db, edited, session)
    _seed_coaching(db, neighbour, session)
    edited_id, neighbour_id = edited.id, neighbour.id
    db.close()

    app = _mount(
        path,
        monkeypatch,
        "player_edit",
        [
            f"players = db.fetch_players_by_hand({edited_id})",
            "app_module.show_player_editor(db, players, force_open=True)",
        ],
    )

    def _name_field(current: AppTest):
        return next(item for item in current.text_input if item.label == "Player")

    def _reason_field(current: AppTest):
        return next(item for item in current.text_input if item.label == "Correction reason")

    def _submit(current: AppTest):
        return next(item for item in current.button if item.label == "Update player")

    _name_field(app).set_value("Renamed villain")
    _submit(app).click().run()
    assert not list(app.exception)
    # Validation: no correction reason, so nothing was written.
    assert "Add a correction reason" in _page_text(app)

    verifier = _open_db(path)
    assert verifier.fetch_hand_corrections(edited_id) == []
    verifier.close()

    _name_field(app).set_value("Renamed villain")
    _reason_field(app).set_value("Reconstruction read the wrong seat name.")
    _submit(app).click().run()
    assert not list(app.exception)

    verifier = _open_db(path)
    corrections = verifier.fetch_hand_corrections(edited_id)
    edited_reviews = verifier.fetch_coaching_reviews_by_hand(edited_id)
    neighbour_reviews = verifier.fetch_coaching_reviews_by_hand(neighbour_id)
    neighbour_status = verifier.fetch_hand(neighbour_id).review_status
    verifier.close()

    assert len(corrections) == 1
    assert "wrong seat name" in corrections[0].notes
    assert corrections[0].before_state != corrections[0].after_state
    # Targeted: the edited hand's retained coaching is stale, the untouched
    # hand's hand-level coaching is not, and the neighbour keeps its promotion.
    assert all(review.is_stale for review in edited_reviews)
    assert not any(review.is_stale for review in neighbour_reviews)
    assert neighbour_status == "reviewed"
    st.cache_resource.clear()


def test_action_edit_requires_a_reason_and_files_its_own_audit_row(
    tmp_path, monkeypatch
) -> None:
    """The action editor owes the same four things the player editor does."""

    path = tmp_path / "action_edit.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Actions", date_played=date(2026, 7, 18)))
    edited = _seed_cv_hand(db, session, hand_number=1, review_status="reviewed")
    neighbour = _seed_cv_hand(db, session, hand_number=2, review_status="reviewed")
    _seed_coaching(db, edited, session)
    _seed_coaching(db, neighbour, session)
    edited_id, neighbour_id = edited.id, neighbour.id
    db.close()

    app = _mount(
        path,
        monkeypatch,
        "action_edit",
        [
            f"actions = db.fetch_actions_by_hand({edited_id})",
            f"players = db.fetch_players_by_hand({edited_id})",
            "app_module.show_action_editor_contents(",
            f"    db, actions, players, hand_id={edited_id},",
            ")",
        ],
    )

    def _reason(current: AppTest):
        return next(
            item
            for item in current.text_input
            if item.label == "Why are you changing this?"
        )

    def _save(current: AppTest):
        return next(item for item in current.button if item.label == "Save")

    _save(app).click().run()
    assert not list(app.exception)
    assert "so the change is auditable" in _page_text(app)

    verifier = _open_db(path)
    assert verifier.fetch_hand_corrections(edited_id) == []
    verifier.close()

    _reason(app).set_value("The video shows a call, not a bet.")
    _save(app).click().run()
    assert not list(app.exception)

    verifier = _open_db(path)
    corrections = verifier.fetch_hand_corrections(edited_id)
    neighbour_reviews = verifier.fetch_coaching_reviews_by_hand(neighbour_id)
    edited_reviews = verifier.fetch_coaching_reviews_by_hand(edited_id)
    verifier.close()

    assert len(corrections) == 1
    assert "shows a call" in corrections[0].notes
    assert all(review.is_stale for review in edited_reviews)
    assert not any(review.is_stale for review in neighbour_reviews)
    st.cache_resource.clear()


def test_a_derived_result_is_labelled_as_derived_in_the_library(
    tmp_path, monkeypatch
) -> None:
    """A ledger reconstruction and a recorded figure are the same float on screen."""

    path = tmp_path / "derived.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Derived", date_played=date(2026, 7, 17)))
    # No observed hero result: everything shown for this hand comes from the
    # reconciled ledger, and the row has to say so.
    _seed_cv_hand(
        db,
        session,
        review_status="unreviewed",
        hero_bb_won=None,
        reconcile=True,
    )
    db.close()

    app = _mount_hands(path, monkeypatch)
    text = _page_text(app)

    assert "Result derived from the reconciled ledger" in text
    st.cache_resource.clear()


@pytest.mark.parametrize("width", [360, 1280])
def test_hands_library_renders_at_narrow_and_wide_widths(
    tmp_path, monkeypatch, width: int
) -> None:
    """Study views must survive a phone-width column stack.

    Width is not something AppTest can set, so this asserts the invariant that
    makes the narrow layout possible: every filter control lives inside the
    keyed container the theme's wrapping rules target, and none of them is drawn
    outside it.
    """

    path = tmp_path / f"width_{width}.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Narrow", date_played=date(2026, 7, 19)))
    _seed_cv_hand(db, session)
    db.close()

    app = _mount_hands(path, monkeypatch)

    keys = {item.key for item in app.get("button_group")} | {
        item.key for item in app.text_input
    }
    assert {
        "hand_library_search",
        "hand_library_status",
        "hand_library_result",
        "hand_library_source",
        "hand_library_flag",
    } <= keys
    theme = Path(
        Path(APP_PATH).parent / "poker_tracker" / "ui" / "ui_theme.py"
    ).read_text()
    assert ".st-key-hand_filters" in theme
    st.cache_resource.clear()


def test_the_session_hand_browser_states_the_same_evidence_as_the_library(
    tmp_path, monkeypatch
) -> None:
    """The shared builder is not enough; its inputs have to be shared too.

    ``hand_evidence_badges`` runs on both list surfaces, but two of its five
    badges are driven by lookups the caller supplies. The session hand browser
    passed neither, so one hand carrying an open issue and correction-invalidated
    coaching rendered as an unremarkable row there while the library named both.
    A row that states less on one surface than on another is the harder version
    of the defect, because neither screen looks wrong by itself.
    """

    path = tmp_path / "browser_states.sqlite3"
    db = _open_db(path)
    session = db.create_session(Session(name="Tuesday", date_played=date(2026, 7, 28)))
    hand = _seed_cv_hand(db, session)
    _seed_coaching(db, hand, session)
    db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards"],
            description="Turn card unreadable in the recording.",
        )
    )
    db.close()
    _correct_hero_cards(path, hand.id)

    library = _page_text(_mount_hands(path, monkeypatch))
    st.cache_resource.clear()
    browser = _page_text(
        _mount(
            path,
            monkeypatch,
            "session_browser",
            [
                "sessions = db.fetch_sessions()",
                "app_module.show_session_hand_browser(db, sessions[0])",
            ],
        )
    )

    for state in ("1 open issue", "Stale analysis"):
        assert state in library, state
        assert state in browser, f"the session hand browser is silent about: {state}"
    st.cache_resource.clear()
