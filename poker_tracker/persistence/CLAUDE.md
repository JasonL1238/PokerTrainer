# persistence/ — agent instructions

Canonical rules: [../../docs/agent-guidelines.md](../../docs/agent-guidelines.md).
This file adds only what is different here.

`db.py` is **6,938 lines**. Never read it whole — `grep -n "def <name>" db.py`
and read a window.

## Schema changes

`SCHEMA_VERSION = 20` (`db.py:81`). The chain is `_migrate_to_v6` … `_migrate_to_v20`,
applied in order by `init_db()` (`db.py:812-832`) for any store below current.

- **State the migration impact before you change a schema.** Which existing rows
  change, whether the change is reversible, and what an older build does with the
  new file. `db.py:350` refuses to open a store newer than the code understands.
- A migration writes a **pre-migration snapshot** first. That snapshot is the only
  rollback point; if it lands somewhere temporary, the rollback is gone. This has
  happened: the v13 migration irreversibly rewrote `review_status` on every
  reconstructed hand while its snapshot went to a temp tree and was deleted.
- Add a new `_migrate_to_vN`; never edit an existing one. Old stores replay the
  chain, so editing history changes what already-migrated databases claim to be.
- Update `tests/test_migration_matrix.py` and the schema-signature tests
  (`tests/test_db.py`, `tests/test_schema_v13_migration_paths.py` use
  `PRAGMA table_info`).

## Writes

- Lifecycle status writes use compare-and-swap: pass `expected_statuses` so a
  lost race leaves the row alone. `update_solver_run` (`db.py:4128`) implements it
  as `AND status IN (...)` on the UPDATE's WHERE clause and returns the *current*
  row on a miss rather than raising. Omitting it is how a finished job gets
  clobbered.
- Destructive operations take a pinned snapshot first. Do not add a delete path
  without one.
- Do not remove the only writer of a column. Several `update_*` methods look
  unreferenced but are the sole post-insert writer for their table; deleting them
  makes the table write-once and the only repair a destructive delete.

## Never

- Read, migrate, or open `poker_tracker.db` at the repo root. That is the
  operator's live database. Tests are redirected away from it by
  `tests/conftest.py`; keep it that way.
- Put a video, frame, or timeline in a SQL column. Those are files.
