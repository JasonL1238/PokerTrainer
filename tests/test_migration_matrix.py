"""Every supported starting version, migrated from its own physical schema.

The suite already proved that migration 13 does what its docstring says, from
three hand-picked starting points. What it could not prove is that the OTHER
migrations do anything at all: the fixture they ran against was a current-shape
database with two columns dropped and its stamp rewritten, so migrations 14 and
later executed ``ALTER TABLE ... ADD COLUMN`` against tables that already had the
column, and every assertion about the result was satisfied by the fixture rather
than by the migration.

The databases here are built from ``tests.legacy_schema_fixtures``, which stores
the DDL of each version as it actually was. Every table that existed at a version
is seeded, the file is migrated by opening it, and the row-preservation invariant
asserts that nothing was lost -- for every supported starting version, not for a
selection of them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from tests.legacy_schema_fixtures import (
    HIGHEST_KNOWN_VERSION,
    STEP_VERSIONS,
    SUPPORTED_START_VERSIONS,
    TableSnapshot,
    as_version,
    assert_rows_survived,
    physical_schema,
    snapshot_tables,
    write_legacy_database,
)


def _label(stored_version: int | None) -> str:
    return "pre-versioning" if stored_version is None else f"v{stored_version}"


def _open_migrated(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _schema_objects(ddl: str) -> set[str]:
    """Tables, table.column pairs and indexes a schema declares."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(ddl)
        objects: set[str] = set()
        for row in connection.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            name, kind = str(row[0]), str(row[1])
            if kind == "index":
                objects.add(f"index:{name}")
                continue
            objects.add(name)
            objects.update(
                f"{name}.{column[1]}"
                for column in connection.execute(f"PRAGMA table_info({name})").fetchall()
            )
        return objects
    finally:
        connection.close()


def _rewritten_columns(stored_version: int | None) -> dict[str, frozenset[str]]:
    """The columns the migration chain is documented to rewrite, per start version.

    Anything not listed here must come out of the migration byte for byte, which
    is the whole point of keeping the list this short.
    """
    version = as_version(stored_version)
    # The version stamp itself is the one row every migration is meant to change.
    rewritten = {"schema_metadata": frozenset({"value"})}
    if version < 8:
        # _migrate_to_v8 renumbers each street's actions before it can enforce
        # uniqueness; the fixture below that version holds two rows at index 9.
        rewritten["actions"] = frozenset({"action_index"})
    if version < 13:
        # _migrate_to_v13's MIGRATION IMPACT: review_status is forced back to
        # needs_correction for every hand that is not provably manual.
        rewritten["hands"] = frozenset({"review_status"})
    if version < 20:
        # _migrate_to_v20's MIGRATION IMPACT: the same release caps dead money
        # against each seat's own total commitment, so every stored hand that
        # holds any -- recorded in the action line, or declared in
        # hand_settlements.dead_money -- is re-derived and the analysis retained
        # beside it was written against a hero result this build no longer
        # produces. Which hands that is, and which are left alone, is proved in
        # tests/test_migration_v20_dead_money_staling.py. Only the
        # FRESHNESS columns move -- the coaching text, the awards, the actions,
        # the settlement and the operator's own review_status are all untouched,
        # which is why `hands` is deliberately absent from this entry.
        rewritten["hand_reviews"] = frozenset({"is_stale", "stale_reason"})
        rewritten["coaching_reviews"] = frozenset({"is_stale", "stale_reason"})
        rewritten["solver_runs"] = frozenset({"status", "error_message"})
    return rewritten


# --------------------------------------------------------------------------
# 1. The fixture is genuinely old, and covers every migration that exists
# --------------------------------------------------------------------------


def test_every_migration_has_a_historical_shape() -> None:
    """A new migration cannot land without the schema it migrates from.

    Without this, the matrix below would keep passing while covering one version
    less than the product has -- which is exactly how versions 14 through 17 came
    to be untested.
    """
    declared = {as_version(version) for version in SUPPORTED_START_VERSIONS}

    assert set(db_module._MIGRATIONS) <= declared | {SCHEMA_VERSION}
    assert HIGHEST_KNOWN_VERSION == SCHEMA_VERSION, (
        f"Schema version {SCHEMA_VERSION} has no declared historical shape in "
        "tests/legacy_schema_fixtures.py; add its step before releasing it."
    )


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_a_version_fixture_lacks_everything_the_next_migration_adds(
    stored_version: int | None,
) -> None:
    """Proof the fixture is not a current-shape database with a rewritten stamp."""
    version = as_version(stored_version)
    if version >= HIGHEST_KNOWN_VERSION:
        pytest.skip("The newest version has no later migration to lack.")
    # Versions 1-4 introduced nothing of their own, so the next shape change is
    # the next version that declares a step, not always version + 1.
    next_step = min(step for step in STEP_VERSIONS if step > version)
    fixture_objects = _schema_objects(physical_schema(stored_version))
    current_objects = _schema_objects(physical_schema(HIGHEST_KNOWN_VERSION))

    absent = current_objects - fixture_objects
    introduced_next = _schema_objects(physical_schema(next_step)) - fixture_objects

    assert introduced_next, f"Version {next_step} introduces nothing."
    assert introduced_next <= absent


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_no_supported_version_carries_a_column_the_chain_would_drop(
    stored_version: int | None,
) -> None:
    """Forward-only means additive: a column that existed must still exist.

    A migration that does remove one has to prove, in its own test, that it
    carried the data somewhere or that the column was empty.
    """
    fixture_columns = {
        name for name in _schema_objects(physical_schema(stored_version)) if "." in name
    }
    current_columns = {
        name
        for name in _schema_objects(physical_schema(HIGHEST_KNOWN_VERSION))
        if "." in name
    }

    assert fixture_columns <= current_columns


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_every_table_a_version_had_holds_a_row(
    tmp_path: Path, stored_version: int | None
) -> None:
    """The invariant below proves nothing about a table nobody seeded.

    A table left empty would satisfy every count and every digest comparison
    while a migration quietly emptied it in production, so the fixture's own
    coverage has to be asserted rather than assumed.
    """
    path = tmp_path / f"seeded-{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)
    connection = sqlite3.connect(path)
    snapshot = snapshot_tables(connection)
    connection.close()

    empty = sorted(
        table
        for table, contents in snapshot.items()
        # A pre-versioning file has the metadata table and no stamp row in it,
        # which is exactly what makes it pre-versioning.
        if not contents.row_count
        and not (table == "schema_metadata" and stored_version is None)
    )

    assert empty == []


# --------------------------------------------------------------------------
# 2. The row-preservation invariant, over the whole supported range
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_every_row_survives_the_migration_from(
    tmp_path: Path, isolated_backup_dir: Path, stored_version: int | None
) -> None:
    path = tmp_path / f"{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)
    reader = sqlite3.connect(path)
    before = snapshot_tables(reader)
    reader.close()
    assert before["hands"].row_count == 4

    migrated = _open_migrated(path)

    after = snapshot_tables(migrated._connection)
    assert_rows_survived(before, after, rewritten=_rewritten_columns(stored_version))
    assert migrated.schema_version() == SCHEMA_VERSION
    migrated.close()


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_the_migrated_database_is_structurally_complete(
    tmp_path: Path, isolated_backup_dir: Path, stored_version: int | None
) -> None:
    """Every table, column and index this build requires, and no broken reference."""
    path = tmp_path / f"complete-{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)

    migrated = _open_migrated(path)

    report = migrated.schema_integrity()
    assert report.is_intact, report.describe()
    # The migration chain has to be complete on its own. init_db can add back a
    # column a damaged file is missing, and without this assertion that repair
    # would quietly cover for a migration that forgot one.
    assert migrated.restored_columns == ()
    migrated.close()


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_the_migrated_database_reads_back_through_the_model_layer(
    tmp_path: Path, isolated_backup_dir: Path, stored_version: int | None
) -> None:
    """Rows surviving as bytes is not enough; they have to load as hands again."""
    path = tmp_path / f"readback-{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)

    migrated = _open_migrated(path)

    sessions = migrated.fetch_sessions()
    assert [session.name for session in sessions] == ["Legacy session"]
    hands = migrated.fetch_hands_by_session(sessions[0].id)
    assert [hand.hand_number for hand in hands] == [1, 2, 3, 4]
    assert len(migrated.fetch_players_by_hand(2)) == 2
    assert len(migrated.fetch_actions_by_hand(2)) == 2
    assert len(migrated.fetch_reviews_by_hand(2)) == 1
    assert len(migrated.fetch_coaching_reviews_by_hand(2)) == 1
    assert len(migrated.fetch_coaching_reviews_by_session(sessions[0].id)) == 1
    assert len(migrated.fetch_videos()) == 1
    migrated.close()


# --------------------------------------------------------------------------
# 3. Each migration observed doing its own work
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_each_later_migration_supplies_its_documented_default(
    tmp_path: Path, isolated_backup_dir: Path, stored_version: int | None
) -> None:
    """The additive migrations, checked where they actually had work to do.

    Each assertion is guarded by the version that introduced the column, so it
    runs only on the databases that genuinely lacked it -- and the invariant test
    above covers the other direction, where a database that already had the
    column keeps its own value.
    """
    version = as_version(stored_version)
    path = tmp_path / f"defaults-{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)

    migrated = _open_migrated(path)

    if version < 7:
        # _migrate_to_v7 invents identity for rows that had none, and links each
        # action to the seat it named, which is the whole point of the column.
        keys = {
            row["player_name"]: row["player_key"]
            for row in migrated._execute(
                "SELECT player_name, player_key FROM hand_players"
            ).fetchall()
        }
        assert all(keys.values())
        assert {
            row["player_name"]: row["player_key"]
            for row in migrated._execute(
                "SELECT player_name, player_key FROM actions"
            ).fetchall()
        } == keys
    if version < 8:
        assert [
            row["action_index"]
            for row in migrated._execute(
                "SELECT action_index FROM actions ORDER BY id"
            ).fetchall()
        ] == [1, 2]
        with pytest.raises(sqlite3.IntegrityError):
            migrated._execute(
                "INSERT INTO actions (hand_id, street, action_index, player_name, "
                "action_type, amount_semantics) "
                "VALUES (2, 'river', 2, 'Dup', 'check', 'unknown')"
            )
    if version < 13:
        rows = {
            row["id"]: row
            for row in migrated._execute(
                "SELECT id, completion_status, completion_evidence FROM hands"
            ).fetchall()
        }
        # No evidence is fabricated for a hand nobody has looked at since.
        assert {row["completion_evidence"] for row in rows.values()} == {"{}"}
        assert rows[1]["completion_status"] == "not_applicable"
        if version >= 5:
            assert rows[2]["completion_status"] == "uncertain"
    if version < 14:
        assert (
            migrated._execute("SELECT content_sha256 FROM videos").fetchone()[0] == ""
        )
    if version < 15:
        assert {
            row["study_inclusion"]
            for row in migrated._execute("SELECT study_inclusion FROM hands").fetchall()
        } == {"auto"}
    if version < 16:
        assert [
            row["source_image"]
            for row in migrated._execute(
                "SELECT source_image FROM actions ORDER BY id"
            ).fetchall()
        ] == [None, None]
    if version < 17:
        assert (
            migrated._execute("SELECT COUNT(*) FROM regression_cases").fetchone()[0] == 0
        )
    if 11 <= version < 18:
        # Unbackfilled by design: a retained solver result must not claim a tree
        # nobody recorded for it.
        assert (
            migrated._execute("SELECT run_parameters FROM solver_runs").fetchone()[0]
            == "{}"
        )
    migrated.close()


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_schema_13_classifies_history_once_and_never_replays_it(
    tmp_path: Path, isolated_backup_dir: Path, stored_version: int | None
) -> None:
    """The one migration that rewrites rows must not touch a database past it.

    Hand 2 is seeded reviewed and complete. Replaying _migrate_to_v13 would knock
    it back to needs_correction and uncertain, discarding the operator's own
    confirmation -- which is why the fixture seeds a state the migration would
    visibly destroy.
    """
    version = as_version(stored_version)
    path = tmp_path / f"v13-{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)

    migrated = _open_migrated(path)

    rows = {
        row["id"]: (row["source_type"], row["review_status"], row["completion_status"])
        for row in migrated._execute(
            "SELECT id, source_type, review_status, completion_status FROM hands"
        ).fetchall()
    }
    if version >= 13:
        assert rows[2] == ("cv_import", "reviewed", "complete")
        assert rows[3] == ("corrected_cv", "needs_correction", "partial")
    elif version >= 5:
        assert rows[1] == ("manual", "reviewed", "not_applicable")
        assert rows[2] == ("cv_import", "needs_correction", "uncertain")
        assert rows[4] == ("third_party_tool", "needs_correction", "uncertain")
    else:
        # Before schema 5 the column did not exist, so every backfilled row takes
        # the conservative default and is treated as manual.
        assert {value[0] for value in rows.values()} == {"manual"}
        assert {value[2] for value in rows.values()} == {"not_applicable"}
    migrated.close()


# --------------------------------------------------------------------------
# 4. The invariant's own fail-before
# --------------------------------------------------------------------------


def _migrate_with(
    monkeypatch: pytest.MonkeyPatch, version: int, damage: str
) -> None:
    """Run the real migration for ``version``, then damage the database."""
    original = db_module._MIGRATIONS[version]

    def damaged(db: PokerDatabase) -> None:
        original(db)
        db._execute(damage)

    monkeypatch.setitem(db_module._MIGRATIONS, version, damaged)


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("DELETE FROM settlement_entries WHERE id = 2", "settlement_entries"),
        ("UPDATE hands SET notes = 'rewritten' WHERE id = 1", "hands rows changed"),
    ],
)
def test_the_invariant_catches_a_migration_that_loses_data(
    tmp_path: Path,
    isolated_backup_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    expected: str,
) -> None:
    """A guard that cannot fail proves nothing, so make it fail on purpose.

    Each damage below is what a careless future migration does by accident: a
    DELETE whose WHERE clause is wider than intended, and a backfill that
    overwrites a column it meant only to read. A migration that drops a column
    never reaches this assertion, because ``init_db`` itself refuses the result;
    that path is covered in ``test_persistence_integrity``.
    """
    path = tmp_path / "lossy.sqlite3"
    write_legacy_database(path, stored_version=SCHEMA_VERSION - 1)
    reader = sqlite3.connect(path)
    before = snapshot_tables(reader)
    reader.close()
    _migrate_with(monkeypatch, SCHEMA_VERSION, damage)

    migrated = _open_migrated(path)

    with pytest.raises(AssertionError, match=expected):
        assert_rows_survived(
            before,
            snapshot_tables(migrated._connection),
            rewritten=_rewritten_columns(SCHEMA_VERSION - 1),
        )
    migrated.close()


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        ({}, "existed before the migration and is gone"),
        (
            {"hand_corrections": TableSnapshot(columns=("id",), rows=((1,),))},
            "lost column",
        ),
        (
            {
                "hand_corrections": TableSnapshot(
                    columns=("id", "notes"), rows=()
                )
            },
            "row",
        ),
    ],
)
def test_the_invariant_reports_a_table_column_or_row_that_stopped_existing(
    after: dict[str, TableSnapshot], expected: str
) -> None:
    """The structural half of the guard, without needing a migration to write it."""
    before = {
        "hand_corrections": TableSnapshot(columns=("id", "notes"), rows=((1, "kept"),))
    }

    with pytest.raises(AssertionError, match=expected):
        assert_rows_survived(before, after)


@pytest.mark.parametrize("stored_version", SUPPORTED_START_VERSIONS, ids=_label)
def test_a_second_open_changes_nothing(
    tmp_path: Path, isolated_backup_dir: Path, stored_version: int | None
) -> None:
    """The chain is a one-way door: reopening a migrated file must be inert."""
    path = tmp_path / f"reopen-{_label(stored_version)}.sqlite3"
    write_legacy_database(path, stored_version=stored_version)
    first = _open_migrated(path)
    after_first = snapshot_tables(first._connection)
    first.close()

    second = _open_migrated(path)

    assert_rows_survived(after_first, snapshot_tables(second._connection))
    second.close()
