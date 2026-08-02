"""Two carry-forward defects of the same shape: a number the product contradicts.

Round 3 repaired the Overview's "Reconciled results" KPI by reading
``analytics.resolve_hero_result``'s basis instead of the substitution flag. It
repaired the CONSUMER. ``view_models`` kept a second implementation -- a helper
named ``result_basis_label`` and a ``derived_result_count``/
``observed_result_count`` pair -- that answered the same question from the flag,
dead for Overview and live for whoever reached for it next, under the name of
the correct helper. The first half of this module pins that the wrong path is
gone rather than merely unused.

The second half is the same disagreement one layer up. The importer recognises a
re-imported session and labels it a copy, and every total then counted the copy's
hands, BB and themes a second time -- so the product knew the rows were duplicates
and the numbers said the operator had played twice the poker. Both silently
counting the copy and silently dropping it are the same class of error; what the
totals owe is the exclusion, stated.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.math.analytics import (
    build_hand_evidence,
    duplicate_import_sessions,
    duplicate_import_source,
    population_exclusion,
    select_population,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import export_session, import_session
from poker_tracker.persistence.models import Hand, Session
from poker_tracker.ui import view_models
from poker_tracker.ui.navigation import Page
from poker_tracker.ui.view_models import build_portfolio_summary

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
APP_SOURCE = Path(APP_PATH).read_text(encoding="utf-8")
VIEW_MODELS_SOURCE = Path(view_models.__file__).read_text(encoding="utf-8")


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


def _run(
    path: Path, monkeypatch: pytest.MonkeyPatch, page: Page | None = None
) -> AppTest:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
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
    marker = f'"pt-kpi-label">{label}</div><div class="pt-kpi-value">'
    start = text.index(marker) + len(marker)
    return text[start : text.index("<", start)]


def _seed_session(
    db: PokerDatabase, *, name: str = "Fri night", hands: int = 3, bb: float = 10.0
) -> Session:
    session = db.create_session(
        Session(name=name, date_played=date(2026, 7, 30), stakes="0.05/0.10")
    )
    assert session.id is not None
    for number in range(1, hands + 1):
        db.create_hand(
            Hand(
                session_id=session.id,
                hand_number=number,
                hero_cards="Ah Qs",
                hero_bb_won=bb,
                review_status="reviewed",
                completion_status="not_applicable",
                tags=["BIG_POT"],
            )
        )
    return session


# ---------------------------------------------------------------------------
# Carry-forward 2: the provenance label at its source
# ---------------------------------------------------------------------------


def test_view_models_cannot_answer_a_provenance_question_from_the_write_guard() -> None:
    """The wrong implementation is deleted, not left beside the right one.

    ``derived_result_substituted`` is a WRITE guard: ``db._refuse_display_copy``
    rejects a copy whose ``hero_bb_won`` is not the stored column, so the flag
    means "this value changed" and nothing more. A module that can still reach it
    can still produce a provenance label from it, and this codebase has now paid
    twice for a second weaker way to answer one question (``referenced_paths`` in
    retention, the two resource-limit implementations). So the check is that the
    name does not appear in the display layer at all.
    """
    assert "derived_result_substituted" not in VIEW_MODELS_SOURCE
    assert not hasattr(view_models, "result_basis_label")
    # And the field it fed is gone from the summary, so no consumer can read a
    # substitution count believing it read a provenance count.
    fields = build_portfolio_summary([], [], result_bases={}).__dataclass_fields__
    assert "derived_result_count" not in fields
    assert "observed_result_count" not in fields
    assert "result_basis_counts" in fields


def test_the_app_reads_the_write_guard_only_where_it_writes_it() -> None:
    """One producer, one consumer, and no third reader inventing a meaning.

    ``_resolve_hands_for_display`` sets the flag so a writer can refuse the copy;
    ``hand_evidence_badges`` used to read it to decide whether to print "derived
    from the reconciled ledger", which is a different question and gave the wrong
    answer on exactly the hands whose provenance is strongest.
    """
    start = APP_SOURCE.index("class ResolvedHands:")
    end = APP_SOURCE.index("def _hands_with_accounting_results(")
    outside = APP_SOURCE[:start] + APP_SOURCE[end:]
    assert "derived_result_substituted" in APP_SOURCE[start:end]
    assert "derived_result_substituted" not in outside
    # ``result_basis`` has no default, so a caller cannot omit provenance and
    # silently fall back to anything.
    signature = APP_SOURCE[
        APP_SOURCE.index("def hand_evidence_badges(") : APP_SOURCE.index(
            "def hand_evidence_badges("
        )
        + 400
    ]
    assert "result_basis: ResultBasis," in signature
    assert "result_basis: ResultBasis | None" not in signature


def test_the_summary_splits_results_by_where_the_number_came_from() -> None:
    """The chip-proven case, which the substitution count could never see.

    Hand 1's ledger ESTABLISHED the figure and agreed with the recorded one, so
    nothing was substituted and the flag is False. Hand 2's result exists only
    because the ledger derived it, so the flag is True. Counting the flag reports
    one reconciled result; counting the basis reports two, which is the truth.
    The flags below are set to the values the substitution actually produces, so
    this cannot pass by accident.
    """
    session = Session(id=1, name="Alpha")
    chip_proven = Hand(id=1, session_id=1, hand_number=1, hero_bb_won=10.0)
    ledger_only = Hand(
        id=2, session_id=1, hand_number=2, hero_bb_won=10.0
    ).model_copy(update={"derived_result_substituted": True})
    observed = Hand(id=3, session_id=1, hand_number=3, hero_bb_won=4.0)
    blank = Hand(id=4, session_id=1, hand_number=4)

    summary = build_portfolio_summary(
        [chip_proven, ledger_only, observed, blank],
        [session],
        result_bases={1: "reconciled", 2: "reconciled", 3: "observed", 4: "none"},
    )

    assert chip_proven.derived_result_substituted is False
    assert summary.result_basis_counts["reconciled"] == 2
    assert summary.result_basis_counts["observed"] == 1
    assert summary.result_basis_counts["none"] == 1
    assert summary.result_basis_counts["unattributed"] == 0
    # The four bases partition the hands, so a surface can print any one of them
    # beside the hand count without the parts exceeding the whole.
    assert sum(summary.result_basis_counts.values()) == summary.hand_count == 4


# ---------------------------------------------------------------------------
# Carry-forward 4: a labelled duplicate counted twice
# ---------------------------------------------------------------------------


def test_the_marker_the_importer_writes_is_the_marker_the_totals_read() -> None:
    """Writer and reader agree by construction, not by two copies of a sentence."""
    db = _open(Path(":memory:"))
    original = _seed_session(db)
    payload = export_session(db, original.id)

    copy = import_session(db, payload)

    assert duplicate_import_source(original) is None
    assert duplicate_import_source(copy) == original.id
    assert duplicate_import_sessions(db.fetch_sessions()) == {copy.id: original.id}
    db.close()


def test_a_third_copy_names_the_original_so_one_session_survives() -> None:
    """Two copies both point at the original, so exactly one session is counted."""
    db = _open(Path(":memory:"))
    original = _seed_session(db)
    payload = export_session(db, original.id)

    second = import_session(db, payload)
    third = import_session(db, payload)

    assert duplicate_import_sessions(db.fetch_sessions()) == {
        second.id: original.id,
        third.id: original.id,
    }
    db.close()


def test_no_population_admits_a_re_imported_copy_and_all_of_them_say_so() -> None:
    """Including ``all_saved``, whose rule now states the one thing it drops.

    Leaving a population that still counted the copy would be the second, weaker
    answer again: a caller wanting the doubled figure back could just widen the
    population and get it, with a rule sentence promising it was only mixing
    evidence classes.
    """
    db = _open(Path(":memory:"))
    original = _seed_session(db)
    payload = export_session(db, original.id)
    copy = import_session(db, payload)

    evidence = build_hand_evidence(db, db.fetch_all_hands(), load_coaching=False)
    by_session = {item.hand.session_id for item in evidence}
    assert by_session == {original.id, copy.id}

    for population in ("confirmed", "reconciled", "all_saved"):
        snapshot = select_population(evidence, population)
        assert all(
            member.hand.session_id != copy.id for member in snapshot.members
        ), population
        assert snapshot.excluded_by_reason.get("duplicate_session") == 3, population

    duplicates = [item for item in evidence if item.hand.session_id == copy.id]
    for population in ("confirmed", "reconciled", "all_saved"):
        assert all(
            population_exclusion(item, population) == "duplicate_session"
            for item in duplicates
        ), population
    db.close()


def test_a_session_that_is_not_a_labelled_copy_is_still_counted() -> None:
    """The exclusion follows the label, so nothing is dropped on resemblance.

    Two sessions with the same name and date but different hands are two
    different sessions, and the importer says so by not labelling either.
    """
    db = _open(Path(":memory:"))
    first = _seed_session(db, hands=3)
    second = _seed_session(db, hands=2)

    assert duplicate_import_sessions(db.fetch_sessions()) == {}
    summary = build_portfolio_summary(
        db.fetch_all_hands(), db.fetch_sessions(), result_bases={}
    )

    assert {first.id, second.id} == {
        item.id for item in db.fetch_sessions() if item.id is not None
    }
    assert summary.session_count == 2
    assert summary.hand_count == 5
    assert summary.excluded_sessions == ()
    assert summary.exclusion_statement == "No sessions were left out of these totals."
    db.close()


def test_a_reimported_session_is_not_added_to_the_operators_poker_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point, on the front page of the running app.

    Three hands at +10 BB, exported and re-imported. Pre-repair Overview printed
    2 sessions, 6 hands and +60 BB confirmed -- the operator's own export read
    back as a second night of poker -- with the copy's name the only thing
    anywhere saying otherwise.
    """
    path = tmp_path / "duplicate.sqlite3"
    db = _open(path)
    original = _seed_session(db, hands=3, bb=10.0)
    import_session(db, export_session(db, original.id))
    db.close()

    text = _page_text(_run(path, monkeypatch))

    assert _kpi(text, "Sessions") == "1"
    assert _kpi(text, "Hands") == "3"
    assert "+30 BB" in text
    assert "+60 BB" not in text


def test_the_overview_states_which_session_it_left_out_and_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently dropping the copy is the same error as silently counting it.

    A total computed over a set that quietly lost rows is indistinguishable on
    screen from a total that is right, so the copy is named, its hands and BB are
    stated, and the session list keeps a row for it saying it is not counted --
    otherwise the operator has no way to find the thing to delete.
    """
    path = tmp_path / "stated.sqlite3"
    db = _open(path)
    original = _seed_session(db, hands=3, bb=10.0)
    copy = import_session(db, export_session(db, original.id))
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "re-imported copies" in text
    assert copy.name in text
    assert f"duplicates session {original.id}" in text
    assert "3 hand(s)" in text
    # The session list keeps a row for the copy, and that row says it is out of
    # the totals -- a list and a headline that disagree with nothing explaining
    # the gap is the silent version of this same defect.
    # Two sessions listed, one counted, and the header says which is which
    # rather than leaving the Sessions KPI and the list to disagree in silence.
    assert "2 LISTED · 1 COUNTED" in text
    listed = app.dataframe[0].value
    notes = dict(zip(listed["Session"], listed["In totals"], strict=True))
    assert notes == {
        original.name: "Counted",
        copy.name: f"Not counted — re-imported copy of session {original.id}",
    }


def test_insights_reports_the_duplicate_as_an_exclusion_with_its_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The population layer already had the vocabulary; it just had no reason.

    Pre-repair every Insights denominator doubled and the theme index counted the
    same tag on six hands instead of three, so a pattern drawn from one session
    looked twice as well evidenced as it was.
    """
    path = tmp_path / "insights.sqlite3"
    db = _open(path)
    original = _seed_session(db, hands=3, bb=10.0)
    import_session(db, export_session(db, original.id))
    db.close()

    text = _page_text(_run(path, monkeypatch, page=Page.INSIGHTS))

    assert "In a session the importer labelled a re-imported copy" in text
    assert "3 of 6 saved hands" in text
    # The theme index counts three hands, not six.
    assert re.search(r"Big Pot[^\n]*", text) is not None
    assert "3 of 3 hands" in text
