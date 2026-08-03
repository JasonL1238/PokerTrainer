"""The v20 staling predicate must cover every hand the amended cap re-derives.

The amended dead-money cap ships in the same release as schema 20 and changes no
schema of its own, so ``_migrate_to_v20`` is the one place that runs once per
upgraded file. It used to select that population with
``SELECT hand_id FROM hand_settlements WHERE dead_money > 0`` -- the DECLARED
EXTERNAL column -- while the rule it was announcing applies to RECORDED dead
posts too.

Measured against ``b58b7e3``, the first schema-19 build, a hand whose dead money
is entirely recorded moves:

    b58b7e3  pots=[(0, 180, ('B','C','D'), 'main'), (1, 280, ('B','D'), 'side')]
    HEAD     pots=[(0, 300, ('B','C','D'), 'main'), (1, 160, ('B','D'), 'side')]

with an identical gross of 460, an identical pot count, identical eligible sets,
and ``legal``/``settled``/``balanced`` true and warning-free on both. Only the
distribution and the hero result move -- C nets 120 under the old rule and 240
under the new one -- so no existing cross-check can see the change, and the
coaching written about the old number survived beside it labelled current.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from tests.legacy_schema_fixtures import physical_schema

NOW = "2026-01-01T00:00:00+00:00"

# hand 1: dead money recorded in the action line and nowhere else.
# hand 2: dead money declared externally, live blinds only in the line.
# hand 3: an ante, which is dead money whatever the ante mode turns out to be.
# hand 4: live blinds only, no dead money anywhere -- the control.
_DEAD_BLIND_HAND = 1
_DECLARED_DEAD_MONEY_HAND = 2
_ANTE_HAND = 3
_LIVE_ONLY_HAND = 4

_ACTIONS: tuple[tuple[int, str, float, str | None, int | None], ...] = (
    (_DEAD_BLIND_HAND, "post_blind", 100.0, "dead_blind", None),
    (_DEAD_BLIND_HAND, "post_blind", 5.0, "small_blind", None),
    (_DEAD_BLIND_HAND, "all-in", 60.0, None, None),
    (_DECLARED_DEAD_MONEY_HAND, "post_blind", 5.0, "small_blind", None),
    (_DECLARED_DEAD_MONEY_HAND, "post_blind", 10.0, "big_blind", None),
    (_ANTE_HAND, "ante", 1.0, None, None),
    (_ANTE_HAND, "post_blind", 10.0, "big_blind", 1),
    (_LIVE_ONLY_HAND, "post_blind", 5.0, None, None),
    (_LIVE_ONLY_HAND, "post_blind", 10.0, "big_blind", 1),
    (_LIVE_ONLY_HAND, "call", 10.0, None, None),
)


def _write_v19_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(physical_schema(19))
        connection.execute(
            "CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_metadata (key, value) VALUES ('schema_version', '19')"
        )
        connection.execute(
            "INSERT INTO sessions (id, name, date_played, created_at) "
            "VALUES (1, 'Upgrade', '2026-01-01', ?)",
            (NOW,),
        )
        for hand_id in (
            _DEAD_BLIND_HAND,
            _DECLARED_DEAD_MONEY_HAND,
            _ANTE_HAND,
            _LIVE_ONLY_HAND,
        ):
            connection.execute(
                "INSERT INTO hands (id, session_id, hand_number, created_at, "
                "review_status) VALUES (?, 1, ?, ?, 'reviewed')",
                (hand_id, hand_id, NOW),
            )
            # Every hand carries retained analysis, so "not staled" below is a
            # statement about the predicate and never about an empty table.
            connection.execute(
                "INSERT INTO hand_reviews (hand_id, hand_summary, theory_coach, "
                "exploit_coach, study_lesson, created_at) "
                "VALUES (?, 's', 't', 'e', 'l', ?)",
                (hand_id, NOW),
            )
            connection.execute(
                "INSERT INTO coaching_reviews (provider_name, model_name, "
                "raw_prompt, raw_response, review_type, hand_id, created_at) "
                "VALUES ('p', 'm', 'q', 'a', 'hand', ?, ?)",
                (hand_id, NOW),
            )
            connection.execute(
                "INSERT INTO solver_runs (hand_id, status, input_hash, created_at) "
                "VALUES (?, 'completed', ?, ?)",
                (hand_id, f"hash-{hand_id}", NOW),
            )
            connection.execute(
                "INSERT INTO hand_settlements (hand_id, status, dead_money, "
                "created_at, updated_at) VALUES (?, 'reconciled', ?, ?, ?)",
                (
                    hand_id,
                    2.5 if hand_id == _DECLARED_DEAD_MONEY_HAND else 0.0,
                    NOW,
                    NOW,
                ),
            )
        connection.execute(
            "INSERT INTO coaching_reviews (provider_name, model_name, raw_prompt, "
            "raw_response, review_type, session_id, created_at) "
            "VALUES ('p', 'm', 'q', 'a', 'session', 1, ?)",
            (NOW,),
        )
        for index, (hand_id, kind, amount, forced, live) in enumerate(_ACTIONS, start=1):
            connection.execute(
                "INSERT INTO actions (hand_id, street, action_index, player_name, "
                "action_type, amount, amount_semantics, forced_bet_type, "
                "is_live_post) VALUES (?, 'preflop', ?, 'P', ?, ?, 'incremental', "
                "?, ?)",
                (hand_id, index, kind, amount, forced, live),
            )
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def migrated(tmp_path: Path, isolated_backup_dir: Path) -> PokerDatabase:
    path = tmp_path / "v19.sqlite3"
    _write_v19_database(path)
    db = PokerDatabase(path)
    db.init_db()
    assert db.schema_version() == SCHEMA_VERSION
    yield db
    db.close()


def _staleness(db: PokerDatabase) -> dict[int, tuple[int, int, str]]:
    return {
        int(row["hand_id"]): (
            int(
                db._execute(
                    "SELECT is_stale FROM hand_reviews WHERE hand_id = ?",
                    (row["hand_id"],),
                ).fetchone()["is_stale"]
            ),
            int(
                db._execute(
                    "SELECT is_stale FROM coaching_reviews WHERE hand_id = ? "
                    "AND review_type = 'hand'",
                    (row["hand_id"],),
                ).fetchone()["is_stale"]
            ),
            str(
                db._execute(
                    "SELECT status FROM solver_runs WHERE hand_id = ?",
                    (row["hand_id"],),
                ).fetchone()["status"]
            ),
        )
        for row in db._execute("SELECT id AS hand_id FROM hands").fetchall()
    }


def test_a_hand_whose_dead_money_is_recorded_rather_than_declared_is_staled(
    migrated: PokerDatabase,
) -> None:
    """The reported defect: ``dead_money`` is the declared EXTERNAL column only.

    A dead blind in the action line carries ``dead_money = 0``, so it fell
    outside the predicate while the amended cap moved its distribution and its
    hero result. The coaching written about the old figure then stayed labelled
    current on a hand that still reconciles cleanly, which is the silent-accept
    half of the contract this migration exists to keep on the loud side.
    """
    assert _staleness(migrated)[_DEAD_BLIND_HAND] == (1, 1, "stale")


def test_an_ante_hand_is_staled_by_the_same_predicate(
    migrated: PokerDatabase,
) -> None:
    """An ante is dead money, so the cap re-derives its hand too.

    Such a hand also gains the undeclared-ante-mode refusal, but that blocks
    study readiness rather than staling anything: declaring the mode clears the
    refusal and the retained coaching would come back current beside a figure
    that had moved underneath it.
    """
    assert _staleness(migrated)[_ANTE_HAND] == (1, 1, "stale")


def test_the_declared_external_amount_is_still_covered(
    migrated: PokerDatabase,
) -> None:
    """Widening the predicate must not drop the half it already had."""
    assert _staleness(migrated)[_DECLARED_DEAD_MONEY_HAND] == (1, 1, "stale")


def test_a_hand_with_only_live_blinds_is_left_alone(
    migrated: PokerDatabase,
) -> None:
    """Over-strict is acceptable; staling the whole store is not.

    Every hand ever dealt has blinds. A predicate that keyed on "has a forced
    post" would mark every hand in the file stale and destroy the signal, so the
    second branch has to exclude the small blind, big blind, straddle and
    bring-in that are LIVE money and that the cap never touched.
    """
    assert _staleness(migrated)[_LIVE_ONLY_HAND] == (0, 0, "completed")


def test_the_session_review_is_staled_once_any_hand_in_it_was(
    migrated: PokerDatabase,
) -> None:
    row = migrated._execute(
        "SELECT is_stale, stale_reason FROM coaching_reviews "
        "WHERE review_type = 'session' AND session_id = 1"
    ).fetchone()

    assert int(row["is_stale"]) == 1
    assert "re-derived under the amended dead-money rule" in str(row["stale_reason"])


def test_the_migration_does_not_touch_review_status(
    migrated: PokerDatabase,
) -> None:
    """The family guard the widened predicate must not trip.

    ``_stale_retained_analysis`` and ``_invalidate_hand_derivatives`` are
    separate methods because staling retained output and discarding the
    operator's own confirmation are different acts. This migration performs only
    the first, and widening WHICH hands it reaches must not turn it into the
    second -- every hand here was seeded ``reviewed``.
    """
    statuses = {
        int(row["id"]): str(row["review_status"])
        for row in migrated._execute("SELECT id, review_status FROM hands").fetchall()
    }

    assert set(statuses.values()) == {"reviewed"}


def test_the_reason_names_the_recorded_half_of_the_rule(
    migrated: PokerDatabase,
) -> None:
    """The operator has to be able to tell what moved from the message alone."""
    reason = str(
        migrated._execute(
            "SELECT stale_reason FROM hand_reviews WHERE hand_id = ?",
            (_DEAD_BLIND_HAND,),
        ).fetchone()["stale_reason"]
    )

    assert "recorded forced post" in reason
    assert "capped against each seat's own total commitment" in reason
