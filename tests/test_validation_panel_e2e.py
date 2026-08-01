"""End-to-end coverage of the Import validation panel's frame-derived surface.

Round 13 found this whole surface unexecuted by the suite: the only test that
mounted the panel passed no frame context, so `_cv_issues_for_db_action` always
returned an empty list. Five separate call-site deletions — including one that
removes every CV warning from the screen, and one that stops 31 of 33 rows ever
getting provenance — left the suite green.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from poker_tracker.persistence import db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Action, Hand, HandPlayer, Session
from poker_tracker.services.validated_hand_import import CV_TIMELINE_IDENTITY_KEY

FRAME = "/frames/cv_job_1/t000000.00.jpg"


def _seed(
    path: Path,
    *,
    action_type: str = "call",
    amount: float | None = None,
    stack_before: float | None = 190.0,
) -> int:
    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Panel"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            review_status="needs_correction",
            notes="CV draft from YOLO card timeline. timeline=/x/job_1_timeline.json",
            completion_evidence={
                CV_TIMELINE_IDENTITY_KEY: {"job_id": 1, "timeline_hand_number": 1}
            },
        )
    )
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="seat:1",
            seat_index=1,
            player_name="Seat1",
            position="UTG",
            starting_stack=200.0,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            player_name="Seat1",
            position="UTG",
            action_type=action_type,
            amount=amount,        # empty money amount -> must be flagged
            # Equal to the seat's read on its OWN frame, so the post-action
            # check fires: rank 2, and it CONDEMNS the figure it names.
            stack_before=stack_before,
        )
    )
    hand_id = hand.id
    db.close()
    return hand_id


TIMELINE_HAND = {
    "hand_number": 1,
    "t_start": 0.0,
    "warnings": [],
    "players": [
        {"seat": 1, "player_name": "Seat1", "position": "UTG", "starting_stack": 200.0}
    ],
    "actions": [
        {
            "street": "preflop", "action_index": 1, "seat": 1,
            "player_name": "Seat1", "position": "UTG", "action_type": "call",
            "amount": 10.0, "source_image": FRAME, "derivation": "action_pill",
        }
    ],
}

STATES = [
    {
        "state_index": 0, "time_s": 0.0, "image": FRAME, "board_cards": [],
        "dealt_in": [1], "stacks": {"1": 190.0}, "bets": {"1": 10.0},
        "bets_unknown": {"1": "below_calibrated_render_size"},
        "stacks_unknown": {}, "unmeasured_transitions": [1],
        # A gap, so a low-rank kind is emitted and the ordering matters.
        "coverage_gap": True, "prior_gap_s": 6.0,
    }
]


def _run_panel(
    path: Path,
    hand_id: int,
    monkeypatch,
    *,
    job_id: int = 1,
    timeline_hand: dict | None = None,
    states: list | None = None,
) -> AppTest:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()

    script = path.parent / f"_panel_{hand_id}_{job_id}_{len(states or STATES)}.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "from poker_tracker.persistence.db import PokerDatabase",
                "from poker_tracker.ui.reconstruction_review import "
                "ValidationFrameContext",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"hand = db.fetch_hand({hand_id})",
                f"timeline_hand = json.loads(r'''{json.dumps(timeline_hand or TIMELINE_HAND)}''')",
                f"states = json.loads(r'''{json.dumps(states or STATES)}''')",
                "import streamlit as _st",
                "_st.session_state["
                f"app_module._validation_action_expander_key({hand_id}, 1)] = True",
                "app_module.render_validation_edit_and_approve(",
                "    db,",
                "    hand,",
                "    frames_validated=True,",
                "    frame_context=ValidationFrameContext(",
                f"        job_id={job_id},",
                "        hand_number=1,",
                "        timeline_hand=timeline_hand,",
                "        states=states,",
                "        reviews_by_image={},",
                "        cursor_key='c',",
                "        pending_hand_key='p',",
                "        recording_start_s=0.0,",
                "    ),",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not list(app.exception), list(app.exception)
    return app


def _panel_text(app: AppTest) -> str:
    parts = [item.value for item in app.warning]
    parts += [item.value for item in app.markdown]
    parts += [item.value for item in app.caption]
    parts += [item.label for item in app.expander]
    return "\n".join(str(part) for part in parts)


def _action_badge(app: AppTest) -> str:
    """The collapsed label of the action row, which carries its badge."""

    return next(
        item.label for item in app.expander if item.label.startswith("01 · ")
    )


def test_the_panel_renders_its_cv_warnings(tmp_path, monkeypatch) -> None:
    """Deleting the render call, or the issue computation feeding it, must
    fail here — not pass with a green suite."""
    hand_id = _seed(tmp_path / "panel.db")
    app = _run_panel(tmp_path / "panel.db", hand_id, monkeypatch)
    text = _panel_text(app)
    assert "Amount unknown" in text, "no CV warning reached the screen"
    assert "Stack before looks post-action" in text
    # The badge is the only thing an operator sees before expanding a row, and
    # it holds two kinds — so the most severe must come first. This row emits
    # the condemning check LAST in derivation order, so an unsorted badge
    # would hide it behind "+N more".
    badge = _action_badge(app)
    assert "⚑" in badge, f"the row carried no badge: {badge}"
    assert badge.index("stack before looks post-action") < badge.index(
        "amount unknown"
    ), f"badge is not severity-ordered: {badge}"
    assert "+" in badge, "the fixture no longer exercises badge truncation"


def test_the_panel_warns_against_copying_a_condemned_figure(
    tmp_path, monkeypatch
) -> None:
    """The caption branch that only fires when a warning rules out its own
    figure. Round 13 added it; round 14 found the e2e fixture could not reach
    it, so removing it left the suite green."""
    hand_id = _seed(tmp_path / "caption.db")
    app = _run_panel(tmp_path / "caption.db", hand_id, monkeypatch)
    captions = [str(item.value) for item in app.caption]
    assert any(
        "rules out the figure it names" in caption for caption in captions
    ), "the condemned-figure caption never rendered"


def test_the_panel_opens_the_stack_field_when_a_warning_needs_it(
    tmp_path, monkeypatch
) -> None:
    """Removing the field-opening flag must fail here."""
    hand_id = _seed(tmp_path / "field.db")
    app = _run_panel(tmp_path / "field.db", hand_id, monkeypatch)
    labels = [item.label for item in app.checkbox]
    assert any("More fields" in label for label in labels), labels
    opened = next(item for item in app.checkbox if "More fields" in item.label)
    assert opened.value is True, "the advanced block did not open"


def test_the_panel_backfills_provenance_for_its_own_job(tmp_path, monkeypatch) -> None:
    """Reverting the panel to call the guard without the backfill must fail."""
    path = tmp_path / "fill.db"
    hand_id = _seed(path)
    _run_panel(path, hand_id, monkeypatch)

    db = PokerDatabase(str(path))
    db.init_db()
    stored = db.fetch_actions_by_hand(hand_id)[0].source_image
    db.close()
    assert stored == FRAME, "the panel did not repair this hand's provenance"


def test_the_panel_refuses_a_foreign_jobs_frames(tmp_path, monkeypatch) -> None:
    """Removing the guard from the panel must fail: the next step writes to
    the database, and that write is one-shot."""
    path = tmp_path / "foreign.db"
    hand_id = _seed(path)
    app = _run_panel(path, hand_id, monkeypatch, job_id=3)

    text = _panel_text(app)
    assert "imported from job 1" in text
    assert "Amount unknown" not in text, "a foreign job's frames explained a row"

    db = PokerDatabase(str(path))
    db.init_db()
    stored = db.fetch_actions_by_hand(hand_id)[0].source_image
    db.close()
    assert stored is None, f"a foreign job's frame was written: {stored}"


# A second shape: the warning NAMES a usable figure rather than condemning
# one, so `needs_stack_before` is the only thing that can open the field.
OFFERING_TIMELINE_HAND = {
    "hand_number": 1,
    "t_start": 0.0,
    "warnings": [],
    "players": [
        {"seat": 1, "player_name": "Seat1", "position": "UTG", "starting_stack": 200.0}
    ],
    "actions": [
        {
            "street": "preflop", "action_index": 1, "seat": 1,
            "player_name": "Seat1", "position": "UTG", "action_type": "check",
            "amount": None, "source_image": "/frames/cv_job_1/t000009.00.jpg",
            "derivation": "inferred_round_complete",
        }
    ],
}

OFFERING_STATES = [
    {
        "state_index": 0, "time_s": 0.0, "image": FRAME, "board_cards": [],
        "dealt_in": [1], "stacks": {"1": 200.0}, "bets": {},
        "bets_unknown": {}, "stacks_unknown": {}, "unmeasured_transitions": [],
        "coverage_gap": False,
    },
    {
        "state_index": 1, "time_s": 9.0, "image": "/frames/cv_job_1/t000009.00.jpg",
        "board_cards": [], "dealt_in": [1], "stacks": {}, "bets": {},
        "bets_unknown": {}, "stacks_unknown": {}, "unmeasured_transitions": [],
        "coverage_gap": False,
    },
]


def test_the_panel_opens_the_stack_field_for_an_offered_value(
    tmp_path, monkeypatch
) -> None:
    """C06: with a condemned figure present the field opens anyway, so only a
    row whose warning NAMES a usable value pins `needs_stack_before`."""
    path = tmp_path / "offer.db"
    hand_id = _seed(path, action_type="check", amount=None, stack_before=None)
    app = _run_panel(
        path,
        hand_id,
        monkeypatch,
        timeline_hand=OFFERING_TIMELINE_HAND,
        states=OFFERING_STATES,
    )
    text = _panel_text(app)
    assert "The reconstruction read 200 BB" in text, text[:400]
    captions = [str(item.value) for item in app.caption]
    assert any(
        "Stack before is requested by a warning" in caption
        for caption in captions
    ), "the offered-value caption never rendered"
    opened = next(item for item in app.checkbox if "More fields" in item.label)
    assert opened.value is True, "the advanced block did not open for an offer"
