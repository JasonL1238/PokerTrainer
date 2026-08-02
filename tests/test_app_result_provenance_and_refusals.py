"""Regressions for four app-surface defects, all of the same shape.

Each one is a screen that states something the rows underneath it do not
support, and each is quiet rather than loud:

* the Overview's "Reconciled results" KPI counted SUBSTITUTIONS while every
  other provenance surface counted what the ledger ESTABLISHED, so a library in
  which every hand was chip-proven printed the same figure as one that had never
  been reconciled -- and the opposite of what the session panel and Insights said
  about the same hand;
* the manual-spot form printed "Could not save hand" over a hand that was in the
  database;
* the flagship replay figure drew an unreviewed CV reconstruction's result and
  pot with no evidence class on it, on a page where every text figure carried
  one;
* a database stamped by a newer build arrived as a raw traceback rather than as
  the refusal it is.

The assertions are against seeded rows and rendered markup, never against copy
alone: a test that only checks a heading passes just as happily on a page whose
numbers are backwards.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.ui.navigation import Page
from tests.conftest import attest_declared_assumptions

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_cached_database() -> Iterator[None]:
    """``get_database`` is ``@st.cache_resource`` and outlives one AppTest."""
    yield
    st.cache_resource.clear()


def _open(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _configure(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _run(
    path: Path, monkeypatch: pytest.MonkeyPatch, page: Page | None = None
) -> AppTest:
    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=90).run()
    assert not list(app.exception), [str(item) for item in app.exception]
    if page is not None:
        app.radio[0].set_value(page)
        app.run()
        assert not list(app.exception), [str(item) for item in app.exception]
    return app


def _page_text(app: AppTest) -> str:
    parts: list[str] = []
    for group in (
        app.markdown,
        app.caption,
        app.info,
        app.warning,
        app.error,
        app.success,
        app.code,
        app.subheader,
    ):
        parts.extend(str(item.value) for item in group)
    parts.extend(str(item.label) for item in app.button)
    parts.extend(str(item.label) for item in app.expander)
    return "\n".join(parts)


def _kpi(text: str, label: str) -> str:
    """The value rendered in the KPI card with ``label``."""
    marker = f'"pt-kpi-label">{label}</div><div class="pt-kpi-value">'
    start = text.index(marker) + len(marker)
    return text[start : text.index("<", start)]


def _seed_hand(
    db: PokerDatabase,
    session_id: int,
    hand_number: int,
    *,
    hero_bb_won: float | None = 5.0,
    source_type: str = "manual",
    review_status: str = "unreviewed",
) -> Hand:
    return db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=hand_number,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20,
            hero_bb_won=hero_bb_won,
            review_status=review_status,
            source_type=source_type,
            completion_status=(
                "not_applicable" if source_type == "manual" else "uncertain"
            ),
        )
    )


def _reconcile_in_place(db: PokerDatabase, hand: Hand) -> Hand:
    """Give ``hand`` an action line and a settlement that derive Hero +10 BB.

    Hero bets 10 and Villain calls, so the gross pot is 20 and awarding it to
    Hero leaves Hero net +10. A hand seeded with ``hero_bb_won=10`` is therefore
    CHIP-PROVEN: the recorded figure and the derived one agree exactly, which is
    the state the accounting panel exists to reach and the state in which no
    substitution happens.
    """
    assert hand.id is not None
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            seat_index=0,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    villain = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="villain",
            seat_index=1,
            player_name="Villain",
            position="BB",
            starting_stack=100,
        )
    )
    for actor, action_type in ((hero, "bet"), (villain, "call")):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_type=action_type,
                amount=10,
                amount_semantics="incremental",
            )
        )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", rake_rate=0.0)
    )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=20,
                entry_order=1,
            )
        ],
    )
    persist_reconciliation(db, hand.id)
    attest_declared_assumptions(db, hand.id)
    reloaded = db.fetch_hand(hand.id)
    assert reloaded is not None
    return reloaded


def _seed_chip_proven_hand(
    db: PokerDatabase, session_id: int, hand_number: int
) -> Hand:
    hand = _seed_hand(db, session_id, hand_number, hero_bb_won=10.0)
    return _reconcile_in_place(db, hand)


def _seed_ledger_only_hand(db: PokerDatabase, session_id: int, hand_number: int) -> Hand:
    """A hand whose result exists ONLY because the ledger derived it."""
    hand = _seed_hand(db, session_id, hand_number, hero_bb_won=None)
    return _reconcile_in_place(db, hand)


# ---------------------------------------------------------------------------
# B3: one question, one resolver
# ---------------------------------------------------------------------------


def test_overview_counts_a_chip_proven_result_as_reconciled_not_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded figure and the derived one agree, which is the STRONGEST state.

    Pre-repair the KPI counted substitutions, and a ledger that confirms the
    recorded number substitutes nothing, so this hand -- reconciled, attested,
    chip-proven -- was reported on the front page as "recorded as observed".
    """
    path = tmp_path / "proven.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_chip_proven_hand(db, session.id, 1)
    db.close()

    text = _page_text(_run(path, monkeypatch))

    assert _kpi(text, "Reconciled results") == "1"
    assert "0 were recorded as observed" in text


def test_overview_agrees_with_the_session_panel_about_the_same_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two surfaces in one running app, one hand, one verdict.

    The session evidence panel has always read ``analytics.resolve_hero_result``.
    Overview read a flag with a near-identical label and printed the opposite,
    and neither screen cites the other, so nothing on either page could reveal
    the contradiction.
    """
    path = tmp_path / "agree.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_chip_proven_hand(db, session.id, 1)
    db.close()

    app = _run(path, monkeypatch)
    overview = _page_text(app)
    app.radio[0].set_value(Page.SESSIONS)
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]
    sessions = _page_text(app)

    assert _kpi(overview, "Reconciled results") == "1"
    assert _kpi(sessions, "Result provenance") == "1 reconciled"
    assert "0 recorded as observed" in sessions


def test_a_chip_proven_library_does_not_read_like_one_that_was_never_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The KPI has to distinguish the two states it exists to distinguish.

    Pre-repair both libraries printed ``Reconciled results 0``: the proven one
    because confirming a recorded number substitutes nothing, the unreconciled
    one because there was nothing to substitute.
    """
    proven_path = tmp_path / "all_proven.sqlite3"
    db = _open(proven_path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_chip_proven_hand(db, session.id, 1)
    _seed_chip_proven_hand(db, session.id, 2)
    db.close()
    proven = _page_text(_run(proven_path, monkeypatch))

    bare_path = tmp_path / "none_proven.sqlite3"
    db = _open(bare_path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_hand(db, session.id, 1, hero_bb_won=10.0)
    _seed_hand(db, session.id, 2, hero_bb_won=10.0)
    db.close()
    bare = _page_text(_run(bare_path, monkeypatch))

    assert _kpi(proven, "Reconciled results") == "2"
    assert "0 were recorded as observed" in proven
    assert _kpi(bare, "Reconciled results") == "0"
    assert "2 were recorded as observed" in bare


def test_a_ledger_only_result_is_still_counted_as_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case the old flag DID catch must keep being caught."""
    path = tmp_path / "ledger_only.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_ledger_only_hand(db, session.id, 1)
    _seed_hand(db, session.id, 2, hero_bb_won=4.0)
    db.close()

    text = _page_text(_run(path, monkeypatch))

    assert _kpi(text, "Reconciled results") == "1"
    assert "1 were recorded as observed" in text


def test_the_hand_row_badges_a_chip_proven_result_as_derived_from_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row badge is the second consumer of the same mislabel.

    ``hand_evidence_badges`` read the substitution flag, so the provenance badge
    was absent on exactly the hands whose provenance is strongest.
    """
    path = tmp_path / "badge.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_chip_proven_hand(db, session.id, 1)
    db.close()

    text = _page_text(_run(path, monkeypatch, page=Page.SESSIONS))

    assert "Result derived from the reconciled ledger" in text


def test_provenance_is_reported_without_widening_the_display_copy_write_guard(
    tmp_path: Path,
) -> None:
    """The two questions stay separate: "the ledger proved it" and "the value changed".

    ``derived_result_substituted`` is a WRITE guard -- ``db._refuse_display_copy``
    rejects a copy whose ``hero_bb_won`` is not the stored column. Redefining it
    to mean "reconciled" would have made 'Correct hand facts' refuse every
    chip-proven hand. So the basis is reported separately and the flag keeps its
    value-inequality meaning, which is what these four assertions pin.
    """
    import app as app_module

    db = _open(tmp_path / "guard.sqlite3")
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    proven = _seed_chip_proven_hand(db, session.id, 1)
    ledger_only = _seed_ledger_only_hand(db, session.id, 2)
    assert proven.id is not None and ledger_only.id is not None

    resolved = app_module._resolve_hands_for_display(db, [proven, ledger_only])
    by_id = {hand.id: hand for hand in resolved.hands}

    # Proven: the ledger established it, and nothing was substituted.
    assert resolved.bases[proven.id] == "reconciled"
    assert by_id[proven.id].derived_result_substituted is False
    assert by_id[proven.id].hero_bb_won == 10.0
    # ...so the corrections form can still save it.
    db.update_hand_facts(by_id[proven.id], correction_notes="Board typo.")

    # Ledger-only: also reconciled, and this copy IS a substitution.
    assert resolved.bases[ledger_only.id] == "reconciled"
    assert by_id[ledger_only.id].derived_result_substituted is True
    assert by_id[ledger_only.id].hero_bb_won == 10.0
    with pytest.raises(ValueError, match="substituted"):
        db.update_hand_facts(by_id[ledger_only.id], correction_notes="Board typo.")
    db.close()


# ---------------------------------------------------------------------------
# B6: a refusal that leaves nothing behind
# ---------------------------------------------------------------------------


def test_a_refused_manual_spot_leaves_no_hand_in_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Could not save hand" has to mean the hand was not saved.

    ``save_manual_spot`` commits the hand, its players, its actions and its
    settlement rows in its own transaction and reconciles AFTERWARDS, so any
    refusal the reconciliation raises arrives over a row that already exists. The
    reachable instance was ``x/b3.5/f`` with Winner = Hero -- field validation
    never checked that the declared winner was still in the hand -- and the form
    printed "Could not save hand: Folded player 'hero' cannot win pot 0." while
    the same render's danger zone said "removes all 0 hands" and retyping the
    hand was rejected as a duplicate number.

    The failure is INJECTED rather than typed, deliberately. The one input that
    used to reach it is now refused by ``validate_manual_spot`` before any write,
    which closes that instance and leaves the class untouched: the writer still
    commits before the last check runs. What this pins is the boundary -- whatever
    the post-write check refuses, the store is left as it was found.
    """
    from poker_tracker.math.accounting import LedgerError
    from poker_tracker.services import manual_spot_entry

    path = tmp_path / "refused.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    db.close()

    app = _run(path, monkeypatch, page=Page.SESSIONS)

    def _refuse(*args: object, **kwargs: object) -> None:
        raise LedgerError("Folded player 'hero' cannot win pot 0.")

    monkeypatch.setattr(manual_spot_entry, "persist_reconciliation", _refuse)

    app.text_input(key="manual_spot_hero_cards").set_value("Ah Qs")
    app.text_input(key="manual_spot_board_cards").set_value("Qd 7s 2c")
    app.text_input(key="manual_spot_line_draft").set_value("x/b3.5/c")
    save = next(item for item in app.button if item.label == "Save hand")
    save.click().run()
    assert not list(app.exception), [str(item) for item in app.exception]

    errors = [str(item.value) for item in app.error]
    assert any("Could not save hand" in message for message in errors), errors

    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
        actions = connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
        settlements = connection.execute(
            "SELECT COUNT(*) FROM hand_settlements"
        ).fetchone()[0]
    assert (rows, actions, settlements) == (0, 0, 0)


def test_a_refused_batch_of_manual_spots_saves_none_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch form reports total failure, so it must not keep the first lines.

    Same boundary, other entry point: ``save_manual_spots`` saved each spot in
    turn, so a refusal on the second line left the first one committed under a
    message that said no hands were saved.
    """
    from poker_tracker.math.accounting import LedgerError
    from poker_tracker.services import manual_spot_entry

    path = tmp_path / "refused_batch.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    db.close()

    app = _run(path, monkeypatch, page=Page.SESSIONS)

    real = manual_spot_entry.persist_reconciliation
    calls: list[int] = []

    def _refuse_the_second(db_arg: object, hand_id: int, *args: object, **kwargs: object):
        calls.append(hand_id)
        if len(calls) > 1:
            raise LedgerError("Folded player 'hero' cannot win pot 0.")
        return real(db_arg, hand_id, *args, **kwargs)

    monkeypatch.setattr(manual_spot_entry, "persist_reconciliation", _refuse_the_second)

    app.radio(key="manual_spot_entry_mode").set_value("Multi-hand paste").run()
    app.text_area(key="manual_spot_batch_text").set_value(
        "AhQs | Qd7s2c | x/b3.5/c | hero\nKdKh | Ah9c2s | x/b3.5/c | hero"
    )
    save = next(item for item in app.button if item.label == "Save hands")
    save.click().run()
    assert not list(app.exception), [str(item) for item in app.exception]

    errors = [str(item.value) for item in app.error]
    assert any("Could not save hands" in message for message in errors), errors
    assert len(calls) == 2

    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
    assert rows == 0


# ---------------------------------------------------------------------------
# B7: the largest figure on the page carries its evidence class
# ---------------------------------------------------------------------------


def test_the_overview_replay_figure_names_the_evidence_class_it_is_drawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreviewed CV reconstruction may not be drawn as a completed record.

    The figure prints a hero result in result colour and a pot on the felt. Every
    text figure on the same page is classed; this one carried only the hand
    number and read "Completed hand #1", which is about ``completion_status`` and
    scans as "finished and recorded".
    """
    path = tmp_path / "draft.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    assert session.id is not None
    _seed_hand(db, session.id, 1, source_type="cv_import", hero_bb_won=10.0)
    db.close()

    text = _page_text(_run(path, monkeypatch))

    figure = next(item for item in text.split("\n") if "<figcaption>" in item)
    assert "Hand #1" in figure
    assert "CV draft" in figure
    assert "not confirmed" in figure
    assert "Completed hand #1" not in figure


# ---------------------------------------------------------------------------
# B8: a refusal presented as a refusal
# ---------------------------------------------------------------------------


def test_a_database_from_a_newer_build_is_refused_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is correct and total; only its presentation was at fault.

    Every startup refusal in the persistence layer is a ``RuntimeError`` out of
    the constructor or ``init_db``, and none of them were caught, so the operator
    got Streamlit's red traceback panel instead of the product's own surface.
    """
    path = tmp_path / "newer.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Alpha"))
    db.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value = '99' WHERE key = 'schema_version'"
        )

    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=90).run()

    assert not list(app.exception), [str(item) for item in app.exception]
    errors = "\n".join(str(item.value) for item in app.error)
    # The message keeps naming the version and the remedy.
    assert "newer than this app understands" in errors
    assert "cannot be opened by this build" in errors
    captions = "\n".join(str(item.value) for item in app.caption)
    assert "Nothing was read, written or migrated" in captions
