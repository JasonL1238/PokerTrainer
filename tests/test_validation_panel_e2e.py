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
from poker_tracker.persistence.models import Action, Hand, Session
from poker_tracker.services.validated_hand_import import CV_TIMELINE_IDENTITY_KEY

FRAME = "/frames/cv_job_1/t000000.00.jpg"


def _seed(path: Path) -> int:
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
    db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            player_name="Seat1",
            position="UTG",
            action_type="call",
            amount=None,          # empty money amount -> must be flagged
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
            "amount": None, "source_image": FRAME, "derivation": "action_pill",
        }
    ],
}

STATES = [
    {
        "state_index": 0, "time_s": 0.0, "image": FRAME, "board_cards": [],
        "dealt_in": [1], "stacks": {"1": 200.0}, "bets": {},
        "bets_unknown": {"1": "below_calibrated_render_size"},
        "stacks_unknown": {}, "unmeasured_transitions": [], "coverage_gap": False,
    }
]


def _run_panel(path: Path, hand_id: int, monkeypatch, *, job_id: int = 1) -> AppTest:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()

    script = path.parent / f"_panel_{hand_id}_{job_id}.py"
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
                f"timeline_hand = json.loads(r'''{json.dumps(TIMELINE_HAND)}''')",
                f"states = json.loads(r'''{json.dumps(STATES)}''')",
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
    assert "calibrated" in text, "the refusal explanation was not rendered"
    # The badge is the only thing an operator sees before expanding a row.
    badge = _action_badge(app)
    assert "⚑" in badge, f"the row carried no badge: {badge}"
    assert "amount unknown" in badge, badge


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
