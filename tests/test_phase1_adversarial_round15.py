"""Round-15 regressions (workflow round 6): every finding repaired as a family.

Nine rounds failed to converge because each repair fixed the hand shape the
adversary demonstrated. Round 15's findings are grouped here by the GENERAL defect
behind them, and each family test sweeps several independent instances rather than
the one reproduced:

* **A SQL predicate classified a row the reader reclassifies.** Five live sites --
  ``discard_stale_coaching`` (``is_stale = 1``), ``resolve_hand_issue`` and
  ``update_hand_status`` (``status = 'open'``), ``_validate_single_hero``
  (``is_hero = 1``), ``fetch_cached_solver_run`` (``status = 'completed'``), plus
  ``fetch_hand_issues``' own status filter -- answered in the COLUMN's space while
  every blocker, list and gate reads the MODEL. The consequences were symmetric: a
  clearing action that matched nothing and flashed success anyway, and a
  store-level floor blind to the row it is the floor for. Every site now classifies
  through the reader, and an AST scan fails on a new raw-column predicate.
* **Two datetimes read from the database could be incomparable.** One naive
  ``created_at`` beside one aware one raised ``TypeError`` out of readiness, taking
  the Study page down for that hand and Insights down for the whole database.
  Normalised once at the model boundary, so every comparison in the product is
  fixed rather than the two the reproduction named.
* **An attestation travelled in the payload carrying its evidence.** A one-field
  JSON edit landed a debugging issue ``resolved``, in a state
  ``resolve_hand_issue`` refuses to create, and cleared OPEN_DEBUGGING_ISSUE.
* **A read-time degradation marker was laundered by a round trip.** The card marker
  was restored from round 5 and the hand-column marker, added later, was not.
  Restoration is now keyed on the marker SET and on the table's own columns.
* **Two degradations on one row reached opposite ``review_status`` verdicts.** A
  hand whose BOARD could not be read counted as ``reviewed`` everywhere while
  Study refused it; a hand whose ``confidence_score`` could not be read did not.
* **A consumer prescribed an acknowledgement for a pipeline REJECTION.**
  ``_source_warning_blockers`` was repaired for this in an earlier round and
  ``_layout_blockers`` was not. The warning/rejection split now lives on
  ``CompletionEvidence`` and is enforced.
* **A derived figure was written into an observed-fact column** on a precondition
  (``ledger.is_settled``) strictly weaker than the one every consumer of a derived
  figure uses. Gated in a service, refused again at the writer, and no UI call site
  may reach the writer at all.
* **The measurement's own sentence stated every direction backwards.**

The rest are unpinned behaviours the mutation adversary demonstrated surviving the
whole suite, and documented claims nothing enforced.

Every test below fails on the pre-repair tree; the docstrings state what it did
there.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from dataclasses import fields as dataclass_fields
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app as app_module
import poker_tracker.persistence.db as db_module
from poker_tracker.math.accounting import RakePolicy
from poker_tracker.math.analytics import compute_session_stats
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    UNREADABLE_CARDS_KEY,
    UNREADABLE_HAND_COLUMNS_KEY,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    IMPORTED_ISSUE_REOPEN_NOTE,
    export_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SettlementEntry,
    SolverRun,
)
from poker_tracker.services import hand_accounting
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.settlement_sync import (
    SettlementSyncRefused,
    sync_recorded_figures_from_ledger,
)
from poker_tracker.services.study_readiness import (
    StudyReadiness,
    evaluate_study_readiness,
)
from poker_tracker.ui.navigation import Page
from tests.conftest import attest_declared_assumptions

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app.py")

# Stored values a boolean-ish or enum column can hold that the MODEL reads
# differently from a naive SQL literal. Not a list of "suspicious" values: it is
# every storage class SQLite has, applied to columns whose readers deliberately
# answer a different question from the raw cell.
TRUTHY_NOT_ONE: tuple[object, ...] = (2, -1, "yes", "1", 0.5, "true")
UNREADABLE_ISSUE_STATUSES: tuple[str, ...] = (
    "in_progress",
    "OPEN",
    "open ",
    "closed",
    "1",
    "",
)


def _open_db(tmp_path: Path, name: str = "round15.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _clean_evidence(**overrides: object) -> dict[str, object]:
    payload = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.92,
            layout_supported=True,
            table_size=6,
        )
    )
    payload.update(overrides)
    return payload


def _raw(db: PokerDatabase, sql: str, params: tuple[object, ...] = ()) -> None:
    """A hand-edited row: written outside every model, as a hostile UPDATE would."""
    connection = sqlite3.connect(db.db_path)
    connection.execute(sql, params)
    connection.commit()
    connection.close()


def _seed_hand(
    db: PokerDatabase,
    *,
    seats: int = 2,
    bet: float = 40.0,
    winners: tuple[str, ...] = ("hero",),
    source_type: str = "cv_import",
    pot_size: float | None = None,
    hero_bb_won: float | None = None,
    rake_rate: float = 0.0,
    rake_cap: float | None = None,
    rounding_unit: float = 0.01,
    no_flop_no_drop: bool = False,
    dead_money: float = 0.0,
    settlement: bool = True,
    awards: bool = True,
    fold_win: bool = False,
    hand_number: int = 1,
    session_id: int | None = None,
) -> Hand:
    """``seats`` seats commit ``bet`` each; ``winners`` are declared to share pot 0."""
    if session_id is None:
        created = db.create_session(Session(name="Round 15", date_played=date(2026, 1, 1)))
        assert created.id is not None
        session_id = created.id
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=hand_number,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type=source_type,  # type: ignore[arg-type]
            completion_status="not_applicable" if source_type == "manual" else "complete",
            completion_evidence={} if source_type == "manual" else _clean_evidence(),
        )
    )
    assert hand.id is not None
    keys = ["hero", "villain", "third", "fourth", "fifth"][:seats]
    for key in keys:
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000,
            )
        )
    for index, key in enumerate(keys, start=1):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="preflop",
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 1 else "call",
                amount=bet,
            )
        )
    if fold_win:
        for index, key in enumerate(keys[1:], start=1):
            db.create_action(
                Action(
                    hand_id=hand.id,
                    player_key=key,
                    street="flop",
                    action_index=index,
                    player_name=key.capitalize(),
                    action_type="fold",
                    amount=0.0,
                )
            )
    if settlement:
        db.upsert_hand_settlement(
            HandSettlement(
                hand_id=hand.id,
                status="settled",
                dead_money=dead_money,
                rake_rate=rake_rate,
                rake_cap=rake_cap,
                rake_rounding_unit=rounding_unit,
                no_flop_no_drop=no_flop_no_drop,
            )
        )
    if awards:
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key=key,
                    player_name=key.capitalize(),
                    amount=None,
                    entry_order=order,
                )
                for order, key in enumerate(winners, start=1)
            ],
        )
    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    return refreshed


def _readiness(
    db: PokerDatabase, hand_id: int, *, user_confirmed: bool = False
) -> StudyReadiness:
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    try:
        accounting = reconcile_persisted_hand(db, hand_id)
        error: str | None = None
    except Exception as exc:  # noqa: BLE001 - the readiness surface's own contract
        accounting, error = None, str(exc)
    return evaluate_study_readiness(
        hand,
        accounting=accounting,
        accounting_error=error,
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand_id),
        hand_reviews=db.fetch_reviews_by_hand(hand_id),
        solver_runs=db.fetch_solver_runs_by_hand(hand_id),
        user_confirmed=user_confirmed,
    )


def _codes(readiness: StudyReadiness) -> tuple[str, ...]:
    return tuple(blocker.code for blocker in readiness.blockers)


# ---------------------------------------------------------------------------
# Family 1 -- no SQL predicate classifies a row the reader reclassifies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stored", TRUTHY_NOT_ONE)
@pytest.mark.parametrize("table", ["coaching_reviews", "hand_reviews"])
def test_discard_stale_coaching_clears_exactly_what_the_reader_calls_stale(
    tmp_path: Path, table: str, stored: object
) -> None:
    """The named clearing action must match the rows the blocker counted.

    ``_coaching_response_from_row`` and ``_review_from_row`` decide staleness with
    ``bool(is_stale)``; ``discard_stale_coaching`` filtered ``WHERE is_stale = 1``.
    Every value here reads stale, raised STALE_COACHING_EVIDENCE, drew the "Discard
    stale coaching" control the blocker names -- and matched no row, so the product
    flashed "Discarded 0 stale coaching review(s)." as a SUCCESS and re-rendered the
    identical blocker with nothing else in the product able to clear it. Both
    retained tables were affected. Pre-repair: ``discarded == 0`` and the blocker
    stands for every parametrisation except the string ``"1"``.
    """
    db = _open_db(tmp_path, f"stale_{table}_{stored!r}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    if table == "coaching_reviews":
        db.create_coaching_response(
            CoachingResponse(
                hand_id=hand.id,
                review_type="hand",
                provider_name="p",
                model_name="m",
                raw_prompt="x",
                raw_response="y",
            )
        )
    else:
        db.create_hand_review(
            HandReview(
                hand_id=hand.id,
                hand_summary="s",
                theory_coach="t",
                exploit_coach="e",
                study_lesson="l",
            )
        )
    _raw(db, f"UPDATE {table} SET is_stale = ?", (stored,))  # noqa: S608 - fixed table

    assert "STALE_COACHING_EVIDENCE" in _codes(_readiness(db, hand.id))
    assert db.discard_stale_coaching(hand.id) == 1
    assert "STALE_COACHING_EVIDENCE" not in _codes(_readiness(db, hand.id))
    db.close()


@pytest.mark.parametrize("stored", UNREADABLE_ISSUE_STATUSES)
def test_the_resolve_control_resolves_the_issue_the_blocker_counted(
    tmp_path: Path, stored: str
) -> None:
    """``resolve_hand_issue`` must accept exactly the row ``_hand_issue_from_row`` opens.

    The reader forces ``status='open'`` on a row it cannot fully read, on the stated
    ground that an unverifiable resolution is not a resolution. The writer filtered
    ``WHERE id = ? AND status = 'open'``, so the Study page listed the issue as
    unresolved, drew the Resolution-notes form OPEN_DEBUGGING_ISSUE names, and the
    submit answered "Could not resolve issue: Open hand issue not found." --
    contradicting the page above it, with nothing in the product able to clear it.
    Pre-repair: every parametrisation raises ``ValueError`` here.
    """
    db = _open_db(tmp_path, f"issue_{stored.strip() or 'blank'}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    issue = db.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["other"], description="wrong seat paid")
    )
    assert issue.id is not None
    _raw(db, "UPDATE hand_issues SET status = ?", (stored,))

    assert db.fetch_hand_issues(hand_id=hand.id)[0].status == "open"
    assert "OPEN_DEBUGGING_ISSUE" in _codes(_readiness(db, hand.id))

    resolved = db.resolve_hand_issue(issue.id, resolution_notes="Fixed the award row.")

    assert resolved.status == "resolved"
    assert "OPEN_DEBUGGING_ISSUE" not in _codes(_readiness(db, hand.id))
    db.close()


@pytest.mark.parametrize("stored", UNREADABLE_ISSUE_STATUSES)
def test_the_store_floor_refuses_a_promotion_the_reader_still_blocks(
    tmp_path: Path, stored: str
) -> None:
    """The documented unbypassable floor must see what the blocker sees.

    ``update_hand_status``' ``EXISTS(... status='open')`` subquery was blind to
    exactly the rows ``_hand_issue_from_row`` opens, so the store ACCEPTED a
    ``reviewed`` promotion that readiness refuses. Pre-repair: the promotion is
    accepted and the stored status becomes ``reviewed``.
    """
    db = _open_db(tmp_path, f"floor_{stored.strip() or 'blank'}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    # `apply_workflow=False`, the same door import uses: recording the issue must
    # be the ONLY thing standing between this hand and `reviewed`, or the
    # completion floor answers first and the open-issue floor is never exercised.
    db.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["other"], description="wrong seat paid"),
        apply_workflow=False,
    )
    attest_declared_assumptions(db, hand.id)
    _raw(db, "UPDATE hand_issues SET status = ?", (stored,))

    assert "OPEN_DEBUGGING_ISSUE" in _codes(_readiness(db, hand.id))
    with pytest.raises(ValueError, match="open debugging issue"):
        db.update_hand_status(hand.id, "reviewed")
    stored_hand = db.fetch_hand(hand.id)
    assert stored_hand is not None
    assert stored_hand.review_status != "reviewed"
    db.close()


@pytest.mark.parametrize("stored", UNREADABLE_ISSUE_STATUSES)
def test_the_issue_status_filter_answers_in_the_model_space(
    tmp_path: Path, stored: str
) -> None:
    """``fetch_hand_issues(status='open')`` feeds the cross-session unresolved inbox.

    Filtering ``status = ?`` in the column's space meant the queue that lists open
    issues disagreed with the blocker that counts them: the hand was blocked and the
    issue appeared nowhere the operator could act on it. Pre-repair: the filtered
    list is empty.
    """
    db = _open_db(tmp_path, f"filter_{stored.strip() or 'blank'}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    db.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["other"], description="wrong seat paid")
    )
    _raw(db, "UPDATE hand_issues SET status = ?", (stored,))

    assert [issue.status for issue in db.fetch_hand_issues(status="open")] == ["open"]
    assert db.fetch_hand_issues(status="resolved") == []
    db.close()


@pytest.mark.parametrize("stored", TRUTHY_NOT_ONE)
def test_a_second_hero_is_refused_however_the_first_is_stored(
    tmp_path: Path, stored: object
) -> None:
    """"A hand can have only one Hero player." must use the reader's definition.

    ``_hand_player_from_row`` answers ``bool(is_hero)``; the guard's
    ``AND is_hero = 1`` did not, so a stored ``2`` read as the hero and a SECOND one
    was accepted. Pre-repair: ``create_hand_player`` succeeds and
    ``fetch_players_by_hand`` returns two players with ``is_hero`` True.
    """
    db = _open_db(tmp_path, f"hero_{stored!r}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    _raw(db, "UPDATE hand_players SET is_hero = ? WHERE player_key = 'hero'", (stored,))

    with pytest.raises(ValueError, match="only one Hero"):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="intruder",
                player_name="Intruder",
                is_hero=True,
                starting_stack=1000,
            )
        )
    heroes = [p for p in db.fetch_players_by_hand(hand.id) if p.is_hero]
    assert len(heroes) == 1
    db.close()


def test_the_solver_cache_never_returns_a_run_the_reader_calls_stale(
    tmp_path: Path,
) -> None:
    """``fetch_cached_solver_run`` is the sixth instance, in the other direction.

    ``_solver_run_from_row`` degrades a run whose columns cannot be read to
    ``stale``, precisely so a result nobody can inspect is not presented as study
    evidence. ``WHERE status = 'completed'`` handed exactly such a run back as a
    cache hit. Pre-repair: the corrupt run is returned with ``status == 'stale'``,
    i.e. the cache serves a run the store itself calls invalid.
    """
    db = _open_db(tmp_path, "cache.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    run = db.create_solver_run(
        SolverRun(hand_id=hand.id, input_hash="abc", status="completed")
    )
    assert run.id is not None
    _raw(db, "UPDATE solver_runs SET exploitability_pct = 'not a number' WHERE id = ?", (run.id,))

    assert db.fetch_solver_run(run.id) is not None
    assert db.fetch_solver_run(run.id).status == "stale"
    assert db.fetch_cached_solver_run("abc") is None
    db.close()


# Raw-column classification predicates that are CORRECT, each with the reason. A
# predicate on one of these columns is only legitimate when the column, not the
# reader's verdict, is the subject.
_RAW_COLUMN_PREDICATES = {
    # Bulk staling writes. They move the COLUMN so the reader sees the rows staled;
    # a row the reader already degrades to stale needs no write.
    ("poker_tracker/persistence/db.py", "_stale_retained_analysis"),
    ("poker_tracker/persistence/db.py", "_flag_hand_for_debugging"),
    # A demotion. A value the model cannot read already degrades to
    # needs_correction, so skipping the UPDATE changes no verdict.
    ("poker_tracker/persistence/db.py", "_demote_reviewed_hand"),
    # Live-run bookkeeping owned by the job worker: 'is a background process still
    # holding this row?' is a question about the column, not about study evidence.
    ("poker_tracker/persistence/db.py", "update_solver_run"),
    ("poker_tracker/persistence/db.py", "update_processing_job"),
    ("poker_tracker/persistence/db.py", "fetch_active_solver_runs"),
    ("poker_tracker/persistence/db.py", "fetch_running_jobs"),
    ("poker_tracker/persistence/db.py", "fetch_active_jobs"),
}

_CLASSIFICATION_COLUMNS = (
    "is_stale",
    "is_hero",
    "status",
    "review_status",
    "completion_status",
    "source_type",
)
_PREDICATE_RE = re.compile(
    r"\b(" + "|".join(_CLASSIFICATION_COLUMNS) + r")\s*(?:=|!=|<>|\bIN\b)",
    re.IGNORECASE,
)


def _sql_predicate_text(sql: str) -> str:
    """Only the WHERE-clause text of a statement, with SET bodies removed.

    A ``SET status = CASE WHEN status IN (...)`` body rewrites the column and is not
    a classification of a row; the predicate that selects WHICH rows is.
    """
    if not re.search(r"(?i)\b(WHERE|SET)\b", sql):
        # A dynamically assembled clause fragment (``"status = ?"``,
        # ``" AND status IN ("``) is itself a predicate.
        return sql
    kept: list[str] = []
    for chunk in re.split(r"(?i)\bSET\b", sql):
        segments = re.split(r"(?i)\bWHERE\b", chunk)
        kept.extend(segments[1:])
    text = " ".join(kept)
    return re.split(r"(?i)\bORDER\s+BY\b|\bLIMIT\b|\bGROUP\s+BY\b", text)[0]


class _SqlPredicateWalker(ast.NodeVisitor):
    """Every SQL string literal, with the function it is written in."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.docstrings: set[int] = set()
        self.found: set[tuple[str, str]] = set()

    def _note_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            self.docstrings.add(id(first.value))

    def visit_Module(self, node: ast.Module) -> None:
        self._note_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._note_docstring(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._note_docstring(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _check(self, node: ast.AST, text: str) -> None:
        if _PREDICATE_RE.search(_sql_predicate_text(text)):
            self.found.add((self.module, self.scope[-1] if self.scope else "<module>"))

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            self._check(node, node.value)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        literal = " ".join(
            value.value
            for value in node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
        self._check(node, literal)
        self.generic_visit(node)


def _raw_column_predicate_sites() -> set[tuple[str, str]]:
    relative = "poker_tracker/persistence/db.py"
    walker = _SqlPredicateWalker(relative)
    walker.visit(ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8")))
    return walker.found


def test_no_sql_predicate_classifies_a_row_the_reader_reclassifies() -> None:
    """The family regression: the next site cannot be added quietly.

    Five sites answered "is this row stale / open / the hero / completed?" in the
    COLUMN's space while every blocker and gate read the MODEL, and the readers
    exist precisely because the two disagree. Enumerating the sites is what failed
    for nine rounds, so the set is enforced here: a new raw-column predicate on a
    reclassified column fails this test until it is listed WITH the reason the
    column, rather than the reader's verdict, is the right subject there.
    """
    assert _raw_column_predicate_sites() == _RAW_COLUMN_PREDICATES


# ---------------------------------------------------------------------------
# Family 2 -- no two stored timestamps can be incomparable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "naive_sql", "aware_sql", "expected"),
    [
        (
            "coaching against legacy coaching",
            "UPDATE hand_reviews SET is_stale = 1, created_at = '2026-02-01T00:00:00'",
            "UPDATE coaching_reviews SET is_stale = 1, "
            "created_at = '2026-01-01T00:00:00+00:00'",
            "STALE_COACHING_EVIDENCE",
        ),
        (
            "coaching the other way round",
            "UPDATE coaching_reviews SET is_stale = 1, created_at = '2026-02-01T00:00:00'",
            "UPDATE hand_reviews SET is_stale = 1, created_at = '2026-01-01T00:00:00+00:00'",
            "STALE_COACHING_EVIDENCE",
        ),
    ],
)
def test_readiness_never_raises_on_a_mixed_timestamp_offset(
    tmp_path: Path, case: str, naive_sql: str, aware_sql: str, expected: str
) -> None:
    """A readiness surface degrades and BLOCKS; it never throws.

    ``datetime.fromisoformat`` faithfully returns a NAIVE datetime for an
    offset-less string -- a hand-edited column, an older build's row, an import
    payload -- and the models accepted it, so ONE such value made ``max(stale)`` and
    ``newest_current >= newest_stale`` raise ``TypeError: can't compare offset-naive
    and offset-aware datetimes`` out of ``_coaching_blockers``. Nothing caught it:
    the Study page died before rendering and Insights died for the ENTIRE database,
    because it loops readiness over every hand. Pre-repair: ``TypeError``.
    """
    db = _open_db(tmp_path, f"tz_{abs(hash(case))}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="s",
            theory_coach="t",
            exploit_coach="e",
            study_lesson="l",
        )
    )
    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            review_type="hand",
            provider_name="p",
            model_name="m",
            raw_prompt="x",
            raw_response="y",
        )
    )
    _raw(db, naive_sql)
    _raw(db, aware_sql)

    assert expected in _codes(_readiness(db, hand.id))
    db.close()


def test_solver_readiness_never_raises_on_a_mixed_timestamp_offset(
    tmp_path: Path,
) -> None:
    """``_solver_blockers`` has the identical ``max(stale)`` shape. Pre-repair: TypeError."""
    db = _open_db(tmp_path, "tz_solver.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    db.create_solver_run(SolverRun(hand_id=hand.id, input_hash="a", status="stale"))
    db.create_solver_run(SolverRun(hand_id=hand.id, input_hash="b", status="completed"))
    _raw(db, "UPDATE solver_runs SET created_at = '2026-02-01T00:00:00' WHERE status = 'stale'")
    _raw(
        db,
        "UPDATE solver_runs SET created_at = '2026-01-01T00:00:00+00:00' "
        "WHERE status = 'completed'",
    )

    assert "STALE_SOLVER_EVIDENCE" in _codes(_readiness(db, hand.id))
    db.close()


def test_every_persisted_timestamp_is_timezone_aware() -> None:
    """The rule is at the model boundary, so it holds for every field of every model.

    Fixing the two comparisons the reproduction named would have left every other
    ``max``, ``sorted(key=created_at)`` and ``>=`` in the product exposed. This is
    the property that makes the family closed.
    """
    naive = datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001 - the hostile value itself
    samples = (
        Session(name="S", created_at=naive),
        HandReview(
            hand_id=1,
            hand_summary="s",
            theory_coach="t",
            exploit_coach="e",
            study_lesson="l",
            created_at=naive,
        ),
        CoachingResponse(
            hand_id=1,
            review_type="hand",
            provider_name="p",
            model_name="m",
            raw_prompt="x",
            raw_response="y",
            created_at=naive,
        ),
        SolverRun(
            hand_id=1,
            input_hash="a",
            created_at=naive,
            started_at=naive,
            completed_at=naive,
            heartbeat_at=naive,
        ),
        HandIssue(
            hand_id=1,
            issue_types=["other"],
            description="d",
            created_at=naive,
            updated_at=naive,
            resolved_at=naive,
        ),
    )
    for model in samples:
        for name, value in model.__dict__.items():
            if isinstance(value, datetime):
                assert value.tzinfo is not None, f"{type(model).__name__}.{name} is naive"
                assert value == naive.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Family 3 -- an attestation cannot travel in the payload carrying its evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared_status", "declared_notes", "declared_resolved_at"),
    [
        ("resolved", "", None),
        ("resolved", "Fixed upstream.", "2026-01-02T00:00:00+00:00"),
    ],
)
def test_an_import_payload_cannot_declare_its_debugging_issue_resolved(
    tmp_path: Path,
    declared_status: str,
    declared_notes: str,
    declared_resolved_at: str | None,
) -> None:
    """One JSON field cleared OPEN_DEBUGGING_ISSUE and landed the hand study-ready.

    The importer wrote the declared status verbatim, so the documented Download
    session JSON / Import uploaded session path produced an issue that was
    ``resolved`` with empty notes and a null ``resolved_at`` -- a state
    ``db.resolve_hand_issue`` refuses to create -- and ``db.update_hand_status``
    then accepted the ``reviewed`` promotion, its own floor finding nothing open.
    The second parametrisation is the same forgery with the two fields filled in,
    which is why requiring them would only have moved it along: the resolution is an
    attestation made in the exporting database and is not verifiable here at any
    level of detail. Pre-repair: the first row imports ``resolved``, readiness
    returns ``is_ready`` True with an empty blocker tuple.
    """
    source = _open_db(tmp_path, f"issue_src_{declared_notes or 'blank'}.db")
    hand = _seed_hand(source, fold_win=True)
    assert hand.id is not None and hand.session_id is not None
    attest_declared_assumptions(source, hand.id)
    source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["pot_or_result"],
            description="the pot went to the wrong seat",
        )
    )
    assert "OPEN_DEBUGGING_ISSUE" in _codes(_readiness(source, hand.id))
    payload = export_session(source, hand.session_id)
    source.close()

    payload["hands"][0]["issues"][0]["status"] = declared_status
    payload["hands"][0]["issues"][0]["resolution_notes"] = declared_notes
    payload["hands"][0]["issues"][0]["resolved_at"] = declared_resolved_at

    target = _open_db(tmp_path, f"issue_dst_{declared_notes or 'blank'}.db")
    import_session(target, payload)
    imported = target.fetch_all_hands()[0]
    assert imported.id is not None

    (issue,) = target.fetch_hand_issues(hand_id=imported.id)
    assert issue.status == "open"
    assert issue.resolved_at is None
    assert "OPEN_DEBUGGING_ISSUE" in _codes(
        _readiness(target, imported.id, user_confirmed=True)
    )
    assert _readiness(target, imported.id, user_confirmed=True).is_ready is False
    with pytest.raises(ValueError):
        target.update_hand_status(imported.id, "reviewed")
    target.close()


def test_a_reopened_issue_keeps_the_exporting_databases_resolution_notes(
    tmp_path: Path,
) -> None:
    """Nothing is discarded by reopening: the prior resolution travels as history."""
    source = _open_db(tmp_path, "notes_src.db")
    hand = _seed_hand(source, fold_win=True)
    assert hand.id is not None and hand.session_id is not None
    issue = source.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["actions"], description="missing call")
    )
    assert issue.id is not None
    source.resolve_hand_issue(issue.id, resolution_notes="Re-read the river pill.")
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "notes_dst.db")
    import_session(target, payload)
    imported = target.fetch_all_hands()[0]
    assert imported.id is not None
    (reopened,) = target.fetch_hand_issues(hand_id=imported.id)

    assert reopened.status == "open"
    assert reopened.resolution_notes == ""
    assert "missing call" in reopened.description
    assert IMPORTED_ISSUE_REOPEN_NOTE in reopened.description
    assert "Re-read the river pill." in reopened.description
    target.close()


# ---------------------------------------------------------------------------
# Family 4 -- every read-time degradation marker survives a round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "column", "value", "marker", "blocker"),
    [
        ("board of two cards", "board_cards", "Qd 7s", UNREADABLE_CARDS_KEY,
         "INVALID_HERO_OR_BOARD_CARDS"),
        ("unparseable hero cards", "hero_cards", "zz zz", UNREADABLE_CARDS_KEY,
         "INVALID_HERO_OR_BOARD_CARDS"),
        ("confidence out of range", "confidence_score", "42.0",
         UNREADABLE_HAND_COLUMNS_KEY, "UNREADABLE_HAND_COLUMNS"),
        ("negative effective stack", "effective_stack", "-12.5",
         UNREADABLE_HAND_COLUMNS_KEY, "UNREADABLE_HAND_COLUMNS"),
        ("table size of 99", "table_size", "99",
         UNREADABLE_HAND_COLUMNS_KEY, "UNREADABLE_HAND_COLUMNS"),
        ("negative pot", "pot_size", "-5.0",
         UNREADABLE_HAND_COLUMNS_KEY, "UNREADABLE_HAND_COLUMNS"),
    ],
)
def test_every_degradation_marker_survives_an_export_import_round_trip(
    tmp_path: Path, case: str, column: str, value: str, marker: str, blocker: str
) -> None:
    """A round trip is not a third, undocumented clearing action that repairs by discarding.

    The exporter emits the degraded fallback and the importer strips the marker (it
    is a derivation about the current row). Round 5 put the PRODUCER back for the two
    CARD columns by name; UNREADABLE_HAND_COLUMNS arrived later with no equivalent,
    so its blocker vanished on import, ``fetch_hand_corrections`` was empty, and the
    text of what the columns held no longer existed in either database -- including
    for the columns whose own clearing action says they "cannot be repaired in the
    product". Pre-repair: the last four rows import with an empty blocker tuple.
    """
    source = _open_db(tmp_path, f"marker_src_{column}_{case[:6]}.db")
    hand = _seed_hand(source, fold_win=True)
    assert hand.id is not None and hand.session_id is not None
    attest_declared_assumptions(source, hand.id)
    _raw(source, f"UPDATE hands SET {column} = ? WHERE id = ?", (value, hand.id))  # noqa: S608

    degraded = source.fetch_hand(hand.id)
    assert degraded is not None
    assert degraded.completion_evidence[marker][column]
    assert blocker in _codes(_readiness(source, hand.id))
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, f"marker_dst_{column}_{case[:6]}.db")
    import_session(target, payload)
    imported = target.fetch_all_hands()[0]
    assert imported.id is not None

    assert imported.completion_evidence[marker][column], case
    assert blocker in _codes(_readiness(target, imported.id, user_confirmed=True)), case
    raw_value = sqlite3.connect(target.db_path).execute(
        f"SELECT {column} FROM hands WHERE id = ?", (imported.id,)  # noqa: S608
    ).fetchone()[0]
    assert str(raw_value) == value, case
    target.close()


def test_every_restorable_hand_column_degrades_into_a_restorable_fallback() -> None:
    """The restore's guard is keyed on the model's own defaults, so it cannot silently skip.

    ``restore_unreadable_columns`` writes a column back only when it currently holds
    the fallback the degradation leaves. If a ``hands`` column is added whose default
    serialises to something else, the restore would quietly stop covering it -- the
    exact decay that lost UNREADABLE_HAND_COLUMNS for two rounds -- so the default
    set is asserted here instead of trusted.
    """
    allowed = {None, *db_module._RESTORABLE_FALLBACKS}
    for name, field in Hand.model_fields.items():
        if name in db_module._EVIDENCE_OWNED_COLUMNS or field.exclude:
            continue
        if field.is_required():
            # session_id / hand_number / created_at / tags: identity and
            # constructor-required columns, never degraded to a fallback.
            continue
        default = field.get_default(call_default_factory=True)
        if isinstance(default, datetime):
            # A timestamp column is never at a fallback: `create_hand` stamps one.
            # It is listed in `_EVIDENCE_OWNED_COLUMNS` for that reason.
            raise AssertionError(f"{name} is a timestamp and must be import-owned")
        if isinstance(default, tuple | list | dict):
            assert not default, name
            continue
        assert default in allowed, f"{name} defaults to {default!r}"


def test_a_restore_cannot_overwrite_a_column_that_reads_as_a_fact(
    tmp_path: Path,
) -> None:
    """Round 8's guard, generalised from the two card columns to every column.

    The restore bypasses ``Hand``'s validation deliberately, so without the guard a
    payload's marker text could replace a hand's real hero cards, board, pot or
    result -- the source facts every gate derives from.
    """
    db = _open_db(tmp_path, "guard.db")
    hand = _seed_hand(db, pot_size=80.0, hero_bb_won=40.0, fold_win=True)
    assert hand.id is not None

    db.restore_unreadable_columns(
        hand.id,
        {
            "hero_cards": "2s 2h",
            "board_cards": "3c 3d 3h",
            "pot_size": "-99.0",
            "confidence_score": "42.0",
            "table_size": "99",
        },
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.hero_cards == "Ah Qs"
    assert stored.board_cards == "Qd 7s 2c"
    assert stored.pot_size == 80.0
    assert stored.table_size == 6
    # The one column that WAS at its fallback is the one that gets the producer back.
    assert stored.completion_evidence[UNREADABLE_HAND_COLUMNS_KEY] == {
        "confidence_score": "42.0"
    }
    db.close()


# ---------------------------------------------------------------------------
# Family 5 -- every degradation reaches the same review_status verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "column", "value", "blocker"),
    [
        ("board of two cards", "board_cards", "Qd 7s", "INVALID_HERO_OR_BOARD_CARDS"),
        ("board repeats a hero card", "board_cards", "Ah 7s 2c",
         "INVALID_HERO_OR_BOARD_CARDS"),
        ("unparseable hero cards", "hero_cards", "zz zz", "INVALID_HERO_OR_BOARD_CARDS"),
        ("confidence out of range", "confidence_score", "42.0", "UNREADABLE_HAND_COLUMNS"),
        ("table size of 99", "table_size", "99", "UNREADABLE_HAND_COLUMNS"),
    ],
)
def test_every_hand_degradation_demotes_the_review_status(
    tmp_path: Path, case: str, column: str, value: str, blocker: str
) -> None:
    """One contract, one place: a hand whose facts cannot be read was not reviewed.

    ``_degraded_hand`` forced ``needs_correction`` and ``_degrade_unreadable_cards``
    did not, so the two degradations on the same row reached opposite verdicts: a
    ``reviewed`` hand hand-edited to a two-card board -- the columns that ARE the
    study material -- counted as reviewed in ``compute_session_stats``, in the
    Insights "Unresolved" KPI and in every list row, while Study refused it. The
    demotion is now keyed on "does this hand carry ANY read-time degradation
    marker?" rather than on which degradation produced it. Pre-repair: the three
    card rows read back ``reviewed``.
    """
    db = _open_db(tmp_path, f"demote_{column}_{case[:6]}.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None and hand.session_id is not None
    attest_declared_assumptions(db, hand.id)
    _raw(db, "UPDATE hands SET review_status = 'reviewed' WHERE id = ?", (hand.id,))
    _raw(db, f"UPDATE hands SET {column} = ? WHERE id = ?", (value, hand.id))  # noqa: S608

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.review_status == "needs_correction", case
    assert blocker in _codes(_readiness(db, hand.id)), case
    stats = compute_session_stats(db, hand.session_id)
    assert stats.hands_by_review_status.get("reviewed", 0) == 0, case
    db.close()


def test_a_restored_card_column_demotes_the_hand_that_receives_it(
    tmp_path: Path,
) -> None:
    """The writer side of the same contract.

    ``restore_unreadable_columns`` writes a corrupt column back deliberately, so the
    importing database derives the blocker for itself. It neither demoted nor
    recorded a correction, so a hand promoted to ``reviewed`` and then given its
    corrupt board back stayed ``reviewed``. The reader-side rule covers this without
    a second guard, which is the point of putting it there.
    """
    db = _open_db(tmp_path, "restore_demote.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    attest_declared_assumptions(db, hand.id)
    _raw(db, "UPDATE hands SET review_status = 'reviewed', board_cards = '' WHERE id = ?",
         (hand.id,))
    assert db.fetch_hand(hand.id).review_status == "reviewed"

    db.restore_unreadable_columns(hand.id, {"board_cards": "Qd 7s"})

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.review_status == "needs_correction"
    assert "INVALID_HERO_OR_BOARD_CARDS" in _codes(_readiness(db, hand.id))
    db.close()


# ---------------------------------------------------------------------------
# Family 6 -- a rejection is never described as acknowledgeable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["hero_seat_mismatch", "board_unreadable"])
def test_no_blocker_offers_acknowledge_for_a_rejected_code(
    tmp_path: Path, code: str
) -> None:
    """Three blockers report the same code on the same page; all three must agree.

    COMPLETION_NOT_COMPLETE and UNRESOLVED_SOURCE_WARNING said "a rejection cannot
    be acknowledged or corrected away". UNSUPPORTED_TABLE_LAYOUT said "Accept
    hero_seat_mismatch with Acknowledge in the Source warnings panel" -- a control
    ``app.show_source_warning_controls`` does not draw for a rejection and
    ``completion.acknowledge_codes`` silently drops, so an operator holding a
    genuinely unrepairable hand was told one of the three sentences was a fix.
    Pre-repair: the layout blocker names Acknowledge for the first code.
    """
    db = _open_db(tmp_path, f"reject_{code}.db")
    session = db.create_session(Session(name="R15", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            source_type="cv_import",
            completion_status="uncertain",
            completion_evidence=_clean_evidence(rejection_codes=[code]),
        )
    )
    assert hand.id is not None

    readiness = _readiness(db, hand.id)
    assert readiness.blockers
    prescriptions = (
        "Acknowledge in the Source warnings",
        "acknowledge the remaining codes",
        "with Acknowledge",
    )
    named = [
        blocker
        for blocker in readiness.blockers
        if code in " ".join(blocker.detail) or code in blocker.clearing_action
    ]
    assert named, code
    for blocker in named:
        for prescription in prescriptions:
            assert prescription not in blocker.clearing_action, (
                f"{blocker.code} offers '{prescription}' for a rejection"
            )
        assert "cannot be acknowledged" in blocker.clearing_action or (
            "REJECTED" in blocker.clearing_action
        ), blocker.code
    db.close()


_UNRESOLVED_CODES_CONSUMERS = {
    # The two derived splits, which are defined in terms of the mixed property.
    ("poker_tracker/persistence/completion.py", "unresolved_rejection_codes"),
    ("poker_tracker/persistence/completion.py", "unresolved_warning_codes"),
    ("app.py", "show_source_warning_controls"),
    # Reaches it only to say there is nothing to acknowledge.
    ("poker_tracker/services/study_readiness.py", "_completion_clearing_action"),
}


class _AttributeReadWalker(ast.NodeVisitor):
    def __init__(self, module: str, attribute: str) -> None:
        self.module = module
        self.attribute = attribute
        self.scope: list[str] = []
        self.found: set[tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.attribute:
            self.found.add((self.module, self.scope[-1] if self.scope else "<module>"))
        self.generic_visit(node)


def _attribute_readers(attribute: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    targets = [Path("app.py"), *sorted(Path("poker_tracker").rglob("*.py"))]
    for relative in targets:
        walker = _AttributeReadWalker(relative.as_posix(), attribute)
        walker.visit(ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8")))
        found |= walker.found
    return found


def test_no_consumer_prescribes_an_action_from_unresolved_codes() -> None:
    """``unresolved_codes`` mixes warnings and rejections, so it cannot name an action.

    ``_source_warning_blockers`` was repaired for exactly this and ``_layout_blockers``
    was not: an enumerated repair applied to one consumer. The split now lives on
    ``CompletionEvidence`` (``unresolved_warning_codes`` /
    ``unresolved_rejection_codes``) and the mixed property's readers are enforced, so
    a consumer added later cannot quietly rejoin the ambiguous one.
    """
    assert _attribute_readers("unresolved_codes") == _UNRESOLVED_CODES_CONSUMERS


# ---------------------------------------------------------------------------
# Finding A1 -- a derived figure never becomes an observed fact unattested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("declared rake on a hand recording nothing", {"rake_rate": 0.25}),
        (
            "declared dead money over genuine observations",
            {"pot_size": 80.0, "hero_bb_won": 40.0, "dead_money": 75.0},
        ),
        ("a declared chop of a showdown pot", {"winners": ("hero", "villain")}),
        ("a rounding unit at a zero rake rate", {"rounding_unit": 81.0}),
        (
            "four seats, a rake and a cap",
            {"seats": 4, "rake_rate": 0.5, "rake_cap": 10.0},
        ),
    ],
)
def test_the_derived_ledger_never_replaces_an_unestablished_observation(
    tmp_path: Path, case: str, kwargs: dict[str, object]
) -> None:
    """"Replace observed final pot/result with the derived ledger values" is gated.

    Its precondition was ``reconciled.ledger.is_settled``, which is strictly weaker
    than the gate every other consumer of a derived figure takes. On the first
    parametrisation it wrote the declaration-derived +20 into ``hands.hero_bb_won``
    when the declaration-free result the action line supports is +40; Study
    correctly refused the hand, so ``compute_session_stats`` declined to substitute
    the derived result and fell back to ``hand.hero_bb_won`` -- which this path had
    just overwritten WITH it -- and published it as an OBSERVED result. On the second
    it replaced ``pot_size`` 80 and ``hero_bb_won`` +40, two genuine observations,
    with 155 and +115. Pre-repair: every row writes both columns and the session
    statistics move.
    """
    db = _open_db(tmp_path, f"sync_{abs(hash(case))}.db")
    hand = _seed_hand(db, **kwargs)  # type: ignore[arg-type]
    assert hand.id is not None and hand.session_id is not None
    before = db.fetch_hand(hand.id)
    assert before is not None
    stats_before = compute_session_stats(db, hand.session_id)

    with pytest.raises(SettlementSyncRefused):
        sync_recorded_figures_from_ledger(db, hand.id)

    after = db.fetch_hand(hand.id)
    assert after is not None
    assert after.pot_size == before.pot_size, case
    assert after.hero_bb_won == before.hero_bb_won, case
    stats_after = compute_session_stats(db, hand.session_id)
    assert stats_after.total_hero_bb == stats_before.total_hero_bb, case
    assert stats_after.observed_result_count == stats_before.observed_result_count, case
    db.close()


def test_the_sync_records_the_derived_summary_once_it_is_established(
    tmp_path: Path,
) -> None:
    """The control still performs its purpose on the hands it is legitimate for.

    A fold win declaring no rake and no dead money is the ordinary case: the awards
    are forced by the action line, nothing is measured as dependent, and the derived
    summary IS the record. A manual hand and an attested reconstructed hand are the
    other two.
    """
    db = _open_db(tmp_path, "sync_ok.db")
    fold_win = _seed_hand(db, fold_win=True)
    assert fold_win.id is not None
    sync_recorded_figures_from_ledger(db, fold_win.id)
    stored = db.fetch_hand(fold_win.id)
    assert stored is not None
    assert stored.pot_size == pytest.approx(80.0)
    assert stored.hero_bb_won == pytest.approx(40.0)

    manual = _seed_hand(db, source_type="manual", rake_rate=0.05, hand_number=2,
                        session_id=fold_win.session_id)
    assert manual.id is not None
    sync_recorded_figures_from_ledger(db, manual.id)
    stored_manual = db.fetch_hand(manual.id)
    assert stored_manual is not None
    assert stored_manual.pot_size == pytest.approx(80.0)

    attested = _seed_hand(db, rake_rate=0.25, hand_number=3, session_id=fold_win.session_id)
    assert attested.id is not None
    attest_declared_assumptions(db, attested.id)
    sync_recorded_figures_from_ledger(db, attested.id)
    stored_attested = db.fetch_hand(attested.id)
    assert stored_attested is not None
    assert stored_attested.hero_bb_won == pytest.approx(20.0)
    db.close()


def test_the_writer_itself_refuses_a_derived_figure_under_an_unattested_declaration(
    tmp_path: Path,
) -> None:
    """Defence in depth at the writer, because fixing a call site fixes one call site.

    ``db.py`` cannot ask ``accounting_is_established`` without inverting the
    layering, but it CAN take its own single-pass measurement
    (``_declared_chips_taken``) and refuse on that alone when nothing is attested.
    A second writer added later therefore cannot re-open the hole.
    """
    db = _open_db(tmp_path, "writer_guard.db")
    hand = _seed_hand(db, rake_rate=0.25)
    assert hand.id is not None
    reconciled = persist_reconciliation(db, hand.id)

    with pytest.raises(ValueError, match="settlement declaration moves chips"):
        db.update_hand_accounting_evidence(
            hand.id,
            pot_size=reconciled.ledger.gross_pot,
            hero_bb_won=reconciled.ledger.net_results["hero"],
        )
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.pot_size is None
    assert stored.hero_bb_won is None
    db.close()


def test_no_ui_call_site_writes_the_recorded_pot_or_hero_result() -> None:
    """The gate is taken once, in a service, and the UI has no arguments to get wrong.

    Nine rounds each repaired one UI call site and the next round found another. The
    only caller of ``update_hand_accounting_evidence`` outside ``db.py`` is
    ``services.settlement_sync``.
    """
    callers: set[str] = set()
    for relative in [Path("app.py"), *sorted(Path("poker_tracker").rglob("*.py"))]:
        if relative.as_posix() == "poker_tracker/persistence/db.py":
            continue
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "update_hand_accounting_evidence"
            ):
                callers.add(relative.as_posix())
    assert callers == {"poker_tracker/services/settlement_sync.py"}


# ---------------------------------------------------------------------------
# Finding A2 / C2 -- the sentence states the direction it claims to state
# ---------------------------------------------------------------------------


def _neutral_ledger_figures(
    db: PokerDatabase, hand_id: int, *, rake: RakePolicy | None, winners: dict | None
) -> dict[str, float]:
    """The same hand's ledger under an explicitly named alternative declaration."""
    from poker_tracker.math.accounting import build_ledger_from_records

    ledger = build_ledger_from_records(
        db.fetch_players_by_hand(hand_id),
        db.fetch_actions_by_hand(hand_id),
        dead_money=0.0,
        winners=winners,
        rake=rake,
        flop_seen=True,
    )
    return {
        "gross": ledger.gross_pot,
        "rake": ledger.rake,
        "net": ledger.net_pot,
        "hero": ledger.net_results.get("hero", 0.0),
    }


@pytest.mark.parametrize(
    ("case", "kwargs", "input_name"),
    [
        ("a 50% rake", {"rake_rate": 0.5, "hero_bb_won": 0.0}, "rake_policy"),
        ("a 25% rake", {"rake_rate": 0.25}, "rake_policy"),
        ("75 chips of dead money", {"dead_money": 75.0}, "dead_money"),
        ("a declared showdown winner", {}, "declared_pot_awards"),
    ],
)
def test_the_described_movement_is_the_movement_of_removing_the_declaration(
    tmp_path: Path, case: str, kwargs: dict[str, object], input_name: str
) -> None:
    """Every signed term equals NEUTRAL minus DECLARED, which is what the sentence says.

    ``deltas`` are declared-minus-neutral -- the right convention for the code, which
    is what makes it an attestation to a quantity -- and ``describe()`` rendered them
    verbatim into a sentence whose subject is the REMOVAL. On a 50%-rake hand 5 of 6
    printed terms were directionally wrong: the operator was told that withdrawing a
    rake destroying 40 chips of their result would cost them 40 more. The same string
    is the blocker detail, the caption above 'Confirm this assumption', and a line
    handed to the coaching provider. Pre-repair: every signed term has the wrong sign.
    """
    db = _open_db(tmp_path, f"describe_{abs(hash(case))}.db")
    hand = _seed_hand(db, **kwargs)  # type: ignore[arg-type]
    assert hand.id is not None
    reconciled = persist_reconciliation(db, hand.id)
    declared = {
        "gross": reconciled.ledger.gross_pot,
        "rake": reconciled.ledger.rake,
        "net": reconciled.ledger.net_pot,
        "hero": reconciled.ledger.net_results.get("hero", 0.0),
    }
    (dependence,) = [
        item
        for item in reconciled.assumption_dependence
        if item.input_name == input_name
    ]
    if input_name == "rake_policy":
        neutral = _neutral_ledger_figures(
            db, hand.id, rake=None, winners={0: ["hero"]}
        )
    elif input_name == "dead_money":
        neutral = _neutral_ledger_figures(
            db, hand.id, rake=None, winners={0: ["hero"]}
        )
    else:
        neutral = _neutral_ledger_figures(db, hand.id, rake=None, winners=None)

    sentence = dependence.describe()
    assert dependence.deltas, case
    for name, value in dependence.deltas:
        if name == "payout":
            # A magnitude, not a signed movement: two seats' payouts can move in
            # opposite directions under one declaration.
            assert f"the largest payout for any seat by {abs(value):g} chips" in sentence
            continue
        truth = neutral[name] - declared[name]
        assert f"{name} {truth:+}" in sentence, (
            f"{case}/{name}: printed {sentence!r}, truth {truth:+}"
        )
    db.close()


def test_the_described_movement_names_every_measured_figure(tmp_path: Path) -> None:
    """The measurement is in the sentence, not only in the code.

    Blanking the movement made ``describe()`` fall into its "without moving any
    reported figure" branch unconditionally, so a dependence whose deltas are
    rake+40 / net-40 / payout+40 / hero-40 told the operator they were attesting to
    zero chips -- and told the coaching provider the same -- while the suite stayed
    green, because its only assertion on ``describe()`` was that verdict-only
    wording. Pre-repair (with the movement blanked): 1317 passed.
    """
    db = _open_db(tmp_path, "movement.db")
    hand = _seed_hand(db, rake_rate=0.5, hero_bb_won=0.0)
    assert hand.id is not None
    reconciled = persist_reconciliation(db, hand.id)
    (dependence,) = [
        item
        for item in reconciled.assumption_dependence
        if item.input_name == "rake_policy"
    ]
    sentence = dependence.describe()

    assert "without moving any reported figure" not in sentence
    for name, _ in dependence.deltas:
        assert name in sentence, name
    assert "40" in sentence
    db.close()


# ---------------------------------------------------------------------------
# Finding C1 -- both coaching gates say why they are blocked
# ---------------------------------------------------------------------------


class _StubProvider:
    """A configured provider, so Coach Review reaches the branch under test."""

    provider_name = "fixture"
    model_name = "deterministic"

    def generate_hand_review(self, prompt: str) -> str:
        return "Hand Summary: unused."

    def generate_session_review(self, prompt: str) -> str:
        return self.generate_hand_review(prompt)


def _run_page(path: Path, page: str, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    monkeypatch.setattr(
        "poker_tracker.coaching.llm_providers.get_provider_from_env",
        lambda *args, **kwargs: _StubProvider(),
    )
    import streamlit as st

    st.cache_resource.clear()
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio[0].set_value(page)
    app.run()
    assert not list(app.exception)
    return app


def _seed_unattested_hand(path: Path) -> int:
    db = PokerDatabase(str(path))
    db.init_db()
    hand = _seed_hand(db, rake_rate=0.5, hero_bb_won=0.0)
    assert hand.id is not None
    persist_reconciliation(db, hand.id)
    hand_id = hand.id
    db.close()
    return hand_id


@pytest.mark.parametrize("page", [Page.STUDY, Page.SETTINGS])
def test_both_coaching_surfaces_name_the_unattested_assumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: str
) -> None:
    """The only explanation of why coaching is blocked was unprotected at both surfaces.

    Replacing ``if unattested_assumption_dependence(hand, accounting):`` with
    ``if False:`` at app.py:1552 or app.py:4589 passed the WHOLE suite. With the
    Study branch gone the control fell through to "Reconcile a legal, balanced ledger
    before generating coaching from this hand." -- false on this hand, whose ledger
    is legal, balanced, settled and authoritative -- and named no action the operator
    could take. With the Coach Review branch gone the operator got a disabled button
    and no message at all.
    """
    path = tmp_path / f"coach_{page}.sqlite3"
    _seed_unattested_hand(path)

    app = _run_page(path, page, monkeypatch)
    rendered = " ".join(
        str(item.value)
        for group in (app.markdown, app.info, app.warning, app.caption, app.error)
        for item in group
    )

    # The sentence both surfaces share, and the two false fallbacks each falls
    # through to when its branch is removed.
    assert "Coaching is disabled until you confirm the declared settlement" in rendered
    assert "Reconcile a legal, balanced ledger before generating coaching" not in rendered
    assert (
        "Coaching is disabled until the completed hand has a legal, balanced"
        not in rendered
    )
    import streamlit as st

    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Finding C3 -- the attestation's identity covers every declared field
# ---------------------------------------------------------------------------


def _codes_by_input(db: PokerDatabase, hand_id: int) -> dict[str, str]:
    """Every measured dependence's acknowledgement code, keyed by declared input."""
    reconciled = reconcile_persisted_hand(db, hand_id)
    return {item.input_name: item.code for item in reconciled.assumption_dependence}


def test_the_declaration_fingerprint_separates_every_declared_rake_field(
    tmp_path: Path,
) -> None:
    """A declared field droppable from the attestation's identity is a laundering seam.

    ``_declaration_fingerprint`` digests the strings ``_rake_text`` and
    ``_awards_text`` return, so the declaration half of "the attestation is bound to
    a quantity AND to the declaration" is only as complete as those two functions --
    and ``no_flop_no_drop``, ``rake_rounding_unit`` (the unbounded operator field two
    rounds landed criticals on) and the declared award AMOUNT could each be dropped
    from them with the whole suite green, because the only test on the digest pinned
    that it is sensitive to its ARGUMENTS and never that the arguments carry the whole
    declaration.

    The field set is asserted against ``RakePolicy``'s own dataclass fields, so a
    field added to the policy later fails this test until it is perturbed here.
    """
    policy_fields = {item.name for item in dataclass_fields(RakePolicy)}
    perturbed = {"rate", "cap", "rounding_unit", "no_flop_no_drop"}
    assert policy_fields == perturbed, (
        "RakePolicy gained a field; perturb it below so the fingerprint keeps covering it"
    )

    # Each variant differs from the baseline in exactly one declared rake field, and
    # every one of them is a declaration the same recording could carry.
    variants: dict[str, dict[str, object]] = {
        "rate": {"rake_rate": 0.5},
        "rate + cap": {"rake_rate": 0.5, "rake_cap": 10.0},
        "rate + unit": {"rake_rate": 0.5, "rounding_unit": 5.0},
        "rate + no_flop_no_drop": {"rake_rate": 0.5, "no_flop_no_drop": True},
        "rate + dead money": {"rake_rate": 0.5, "dead_money": 7.0},
    }
    codes: dict[str, str] = {}
    for index, (label, kwargs) in enumerate(variants.items()):
        db = _open_db(tmp_path, f"fp_{index}.db")
        hand = _seed_hand(db, **kwargs)  # type: ignore[arg-type]
        assert hand.id is not None
        persist_reconciliation(db, hand.id)
        by_input = _codes_by_input(db, hand.id)
        assert "rake_policy" in by_input, label
        codes[label] = by_input["rake_policy"]
        db.close()

    assert len(set(codes.values())) == len(codes), (
        f"two rake declarations share one attestation code: {codes}"
    )


def test_the_declaration_fingerprint_separates_every_declared_award_field(
    tmp_path: Path,
) -> None:
    """The declared award AMOUNT and the odd-chip ORDER are part of the declaration too.

    Dropping the amount from ``_awards_text`` survived the whole suite. The order is
    the column that decides who receives the odd chip of a chopped pot, which is the
    round-8 finding that the award snapshot excluded it.
    """
    variants: dict[str, tuple[float | None, float | None, tuple[int, int]]] = {
        "no amounts": (None, None, (1, 2)),
        "amounts declared": (41.0, 41.0, (1, 2)),
        "order reversed": (None, None, (2, 1)),
    }
    codes: dict[str, str] = {}
    for index, (label, (first, second, order)) in enumerate(variants.items()):
        db = _open_db(tmp_path, f"fp_award_{index}.db")
        hand = _seed_hand(db, bet=41.0, winners=("hero", "villain"), awards=False)
        assert hand.id is not None
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key=key,
                    player_name=key.capitalize(),
                    amount=amount,
                    entry_order=entry_order,
                )
                for key, amount, entry_order in (
                    ("hero", first, order[0]),
                    ("villain", second, order[1]),
                )
            ],
        )
        persist_reconciliation(db, hand.id)
        by_input = _codes_by_input(db, hand.id)
        assert "declared_pot_awards" in by_input, label
        codes[label] = by_input["declared_pot_awards"]
        db.close()

    assert len(set(codes.values())) == len(codes), (
        f"two award declarations share one attestation code: {codes}"
    )


# ---------------------------------------------------------------------------
# Findings C4, C7, C8, C10, C11 -- claims nothing enforced
# ---------------------------------------------------------------------------


def test_the_display_substitution_marks_the_copy_it_produces(tmp_path: Path) -> None:
    """``app.py``'s substitution is the only producer of ``derived_result_substituted``.

    The writer half of the round-10 repair is pinned (both writers refuse a copy
    carrying the flag), but nothing asserted the read-time substitution SETS it, so
    forcing ``substituted = False`` passed the whole suite and made
    ``db._refuse_display_copy`` permanently inert -- the defence-in-depth layer added
    precisely because "fixing the call site fixes one call site" was entirely
    decorative.
    """
    db = _open_db(tmp_path, "substituted.db")
    hand = _seed_hand(db, fold_win=True)
    assert hand.id is not None
    persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.hero_bb_won is None

    (listed,) = app_module._hands_with_accounting_results(db, [stored])

    assert listed.hero_bb_won == pytest.approx(40.0)
    assert listed.derived_result_substituted is True
    with pytest.raises(ValueError, match="substituted"):
        db.update_hand_facts(listed, correction_notes="Should be refused.")
    db.close()


def test_a_hand_with_no_settlement_row_never_presents_as_reconciled(
    tmp_path: Path,
) -> None:
    """"No persisted settlement assumptions or awards." is the only thing saying so.

    Removing that issue made a hand with award rows but no settlement row
    ``reconciles = True``, so ``persist_reconciliation`` auto-created a
    ``HandSettlement`` stamped ``status='reconciled'`` -- a declaration-free hand
    marked reconciled by the product -- and on a fold win the award declaration is
    forced-silent, so no dependence was measured either and the hand reached
    ``is_authoritative`` True. Pre-repair (with the issue removed): the whole suite
    passed.
    """
    db = _open_db(tmp_path, "no_settlement.db")
    hand = _seed_hand(db, fold_win=True, settlement=False)
    assert hand.id is not None

    reconciled = reconcile_persisted_hand(db, hand.id)
    assert reconciled.settlement is None
    assert "No persisted settlement assumptions or awards." in reconciled.issues
    assert reconciled.is_authoritative is False
    assert "ACCOUNTING_NOT_AUTHORITATIVE" in _codes(_readiness(db, hand.id))

    saved = persist_reconciliation(db, hand.id)
    assert saved.is_authoritative is False
    stored_settlement = db.fetch_hand_settlement(hand.id)
    assert stored_settlement is not None
    assert stored_settlement.status != "reconciled"
    db.close()


def test_an_unbuildable_neutral_pass_says_so_in_the_code(tmp_path: Path) -> None:
    """The strongest form of dependence must still say what it measured.

    The sibling ``verdict-only`` label is pinned; ``unbuildable`` could be renamed to
    the empty string with the whole suite green. Injected rather than pretending a
    hand shape reaches it, exactly as the JOINT_INPUT fallback is.
    """
    db = _open_db(tmp_path, "unbuildable.db")
    hand = _seed_hand(db, rake_rate=0.25)
    assert hand.id is not None
    records = hand_accounting._load_hand_records(db, hand.id)
    baseline = hand_accounting._cross_check(records, records.declaration)

    dependence = hand_accounting._build_dependence(
        records,
        baseline,
        None,
        input_name=hand_accounting.RAKE_POLICY_INPUT,
        declared="declared",
        neutral="neutral",
    )

    assert dependence.code.endswith(":unbuildable")
    assert dependence.deltas == ()
    db.close()


def test_the_writer_side_audit_takes_a_single_pass_measurement(tmp_path: Path) -> None:
    """``_declared_chips_taken`` derives ONE ledger, and its comment now says so.

    The comment above the disclosure claimed it "derives the hand's ledger under the
    stored policy and again under a neutral one and reports how many chips each
    declaration actually moves", crediting the writer-side audit with the dependence
    rule's dual reconciliation. It takes a strictly smaller measurement -- winners are
    deliberately not fetched, because ``upsert_hand_settlement`` calls it before the
    award rows may exist -- which is what PLAN.md says and what the code does.
    """
    db = _open_db(tmp_path, "single_pass.db")
    hand = _seed_hand(db, rake_rate=0.25, dead_money=5.0)
    assert hand.id is not None
    settlement = db.fetch_hand_settlement(hand.id)
    assert settlement is not None

    builds = {"n": 0}
    real = db_module.build_ledger_from_records

    def counting(*args: object, **kwargs: object):
        builds["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    db_module.build_ledger_from_records = counting  # type: ignore[assignment]
    try:
        taken = db._declared_chips_taken(settlement)
    finally:
        db_module.build_ledger_from_records = real  # type: ignore[assignment]

    assert builds["n"] == 1
    assert set(taken) == {"rake", "dead_money"}
    assert taken["dead_money"] == pytest.approx(5.0)
    db.close()


@pytest.mark.parametrize(
    ("case", "kwargs", "expected_builds"),
    [
        (
            "truly nothing declared",
            {"settlement": False, "awards": False, "fold_win": True},
            1,
        ),
        ("showdown, awards only", {}, 2),
        ("fold win, awards only", {"fold_win": True}, 3),
        ("showdown, rake + dead money + awards", {"rake_rate": 0.05, "dead_money": 5.0}, 5),
        (
            "fold win, rake + dead money + awards",
            {"rake_rate": 0.05, "dead_money": 5.0, "fold_win": True},
            6,
        ),
    ],
)
def test_the_documented_dependence_cost_ceiling_is_the_measured_one(
    tmp_path: Path, case: str, kwargs: dict[str, object], expected_builds: int
) -> None:
    """The documented ceiling was "up to four", and a fold win costs five extra passes.

    ``_forced_winners`` fires whenever every pot has exactly one eligible seat -- an
    ordinary fold win, the commonest hand shape there is -- and a second neutral pass
    is built against the forced winners. The ceiling is derived from this counter
    rather than from prose in ``_derive_assumption_dependence`` and PLAN.md, so it
    cannot drift from the code again.
    """
    db = _open_db(tmp_path, f"cost_{abs(hash(case))}.db")
    hand = _seed_hand(db, **kwargs)  # type: ignore[arg-type]
    assert hand.id is not None

    builds = {"n": 0}
    real = hand_accounting.build_ledger_from_records

    def counting(*args: object, **kwargs_inner: object):
        builds["n"] += 1
        return real(*args, **kwargs_inner)  # type: ignore[arg-type]

    hand_accounting.build_ledger_from_records = counting  # type: ignore[assignment]
    try:
        reconcile_persisted_hand(db, hand.id)
    finally:
        hand_accounting.build_ledger_from_records = real  # type: ignore[assignment]

    assert builds["n"] == expected_builds, case
    db.close()


def test_the_readme_no_longer_claims_the_chip_unit_moves_no_payout() -> None:
    """README told the operator no value typed in Chip unit changes a derived payout.

    Three values change it by 0, +1 and +40 chips on the same 80-chip pot, and the
    reassuring half of the paragraph contradicted the accurate half beside it. An
    operator who believed it had no reason to scrutinise the field before pressing
    'Confirm this assumption' -- the exact click-through the dependence rule exists to
    prevent.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "no value you type in\n`Chip unit` changes a derived payout" not in readme
    assert "no value you type in `Chip unit` changes a derived payout" not in readme
    assert "does** change derived payouts" in readme


def test_the_chip_unit_really_does_move_the_derived_payout(tmp_path: Path) -> None:
    """The behavioural half of the README claim, so the document cannot drift back."""
    from poker_tracker.math.accounting import build_ledger_from_records

    db = _open_db(tmp_path, "chip_unit.db")
    hand = _seed_hand(db, rake_rate=0.5)
    assert hand.id is not None
    players = db.fetch_players_by_hand(hand.id)
    actions = db.fetch_actions_by_hand(hand.id)

    payouts = {}
    for unit in (0.01, 1.0, 3.0, 81.0):
        ledger = build_ledger_from_records(
            players,
            actions,
            winners={0: ["hero"]},
            rake=RakePolicy(rate=0.5, cap=None, rounding_unit=unit),
            flop_seen=True,
        )
        payouts[unit] = ledger.payouts["hero"]

    assert payouts[0.01] == pytest.approx(40.0)
    assert payouts[3.0] == pytest.approx(41.0)
    assert payouts[81.0] == pytest.approx(80.0)
    db.close()
