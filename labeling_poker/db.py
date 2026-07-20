from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import IMAGE_SUFFIXES


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    class TEXT NOT NULL,
    label TEXT,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS annotations_file_idx ON annotations(file_id);
CREATE TABLE IF NOT EXISTS status (
    file_id TEXT PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('labeled', 'clean', 'duplicate')),
    updated_at TEXT NOT NULL
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    status_sql = connection.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'status'").fetchone()[0]
    if "duplicate" not in status_sql:
        connection.execute("ALTER TABLE status RENAME TO status_legacy")
        connection.execute("CREATE TABLE status (file_id TEXT PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE, status TEXT NOT NULL CHECK(status IN ('labeled', 'clean', 'duplicate')), updated_at TEXT NOT NULL)")
        connection.execute("INSERT INTO status(file_id, status, updated_at) SELECT file_id, status, updated_at FROM status_legacy")
        connection.execute("DROP TABLE status_legacy")
        connection.commit()
    annotation_columns = {row["name"] for row in connection.execute("PRAGMA table_info(annotations)")}
    if "label" not in annotation_columns:
        connection.execute("ALTER TABLE annotations ADD COLUMN label TEXT")
        connection.commit()
    return connection


def sync_files(connection: sqlite3.Connection, images_dir: Path | str) -> list[str]:
    directory = Path(images_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    present_ids: set[str] = set()
    for path in paths:
        file_id = path.stem
        present_ids.add(file_id)
        relative_path = path.relative_to(directory).as_posix()
        connection.execute(
            "INSERT INTO files(id, path) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET path=excluded.path",
            (file_id, relative_path),
        )
    # Drop undecided rows whose image file disappeared so Browse doesn't land on 404s.
    for row in connection.execute("SELECT id FROM files").fetchall():
        if row["id"] in present_ids:
            continue
        if get_status(connection, row["id"]) == "undecided":
            connection.execute("DELETE FROM files WHERE id = ?", (row["id"],))
    connection.commit()
    return [path.stem for path in paths]


def file_ids(connection: sqlite3.Connection) -> list[str]:
    return [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id")]


def get_file(connection: sqlite3.Connection, file_id: str) -> sqlite3.Row | None:
    return connection.execute("SELECT id, path FROM files WHERE id = ?", (file_id,)).fetchone()


def get_annotations(connection: sqlite3.Connection, file_id: str) -> list[dict]:
    rows = connection.execute(
        "SELECT class, label, x1, y1, x2, y2 FROM annotations WHERE file_id = ? ORDER BY id",
        (file_id,),
    )
    return [dict(row) for row in rows]


def get_status(connection: sqlite3.Connection, file_id: str) -> str:
    row = connection.execute("SELECT status FROM status WHERE file_id = ?", (file_id,)).fetchone()
    return row["status"] if row else "undecided"


def save_annotations(connection: sqlite3.Connection, file_id: str, status_value: str, boxes: Iterable[dict]) -> None:
    if status_value not in {"labeled", "clean", "duplicate"}:
        raise ValueError("status must be labeled, clean, or duplicate")
    now = datetime.now(timezone.utc).isoformat()
    with connection:
        connection.execute("DELETE FROM annotations WHERE file_id = ?", (file_id,))
        if status_value == "labeled":
            connection.executemany(
                "INSERT INTO annotations(file_id, class, label, x1, y1, x2, y2) VALUES(?, ?, ?, ?, ?, ?, ?)",
                [(file_id, b["class"], b.get("label"), b["x1"], b["y1"], b["x2"], b["y2"]) for b in boxes],
            )
        connection.execute(
            "INSERT INTO status(file_id, status, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(file_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
            (file_id, status_value, now),
        )


STATUS_FILTERS = {"all", "undecided", "labeled", "clean", "duplicate"}


def _ordered_ids(connection: sqlite3.Connection, priority_ids: Iterable[str] = ()) -> list[str]:
    """Return file IDs with the optional priority list first, without duplicates."""
    ids = file_ids(connection)
    known = set(ids)
    priority = []
    seen = set()
    for file_id in priority_ids:
        if file_id in known and file_id not in seen:
            priority.append(file_id)
            seen.add(file_id)
    return priority + [file_id for file_id in ids if file_id not in seen]


def _matching_ids(
    connection: sqlite3.Connection,
    status_filter: str,
    priority_ids: Iterable[str] = (),
    *,
    priority_only: bool = False,
    order_by_updated_at: bool = False,
) -> list[str]:
    """Return IDs matching a status, optionally scoped and ordered by label time."""
    ids = _ordered_ids(connection, priority_ids)
    if priority_only:
        priority_set = set(priority_ids)
        ids = [file_id for file_id in ids if file_id in priority_set]
    if status_filter == "all":
        return ids

    status_by_id = {
        row["file_id"]: (row["status"], row["updated_at"])
        for row in connection.execute("SELECT file_id, status, updated_at FROM status")
    }
    matches = [file_id for file_id in ids if status_by_id.get(file_id, ("undecided", ""))[0] == status_filter]
    if order_by_updated_at:
        # ISO-8601 timestamps sort chronologically as text. Use the file ID as a
        # deterministic tie-breaker for labels saved in the same clock tick.
        matches.sort(key=lambda file_id: (status_by_id[file_id][1], file_id))
    return matches


def next_matching(
    connection: sqlite3.Connection,
    status_filter: str = "undecided",
    priority_ids: Iterable[str] = (),
    current_id: str | None = None,
    *,
    priority_only: bool = False,
    order_by_updated_at: bool = False,
    start_with_latest: bool = False,
    wrap_next: bool = False,
) -> str | None:
    """Find the first or next file that matches a saved-label status.

    Keeping the current ID in the ordering even after it is re-saved with a new
    status lets the UI advance naturally after, for example, changing a labeled
    image to clean.
    """
    if status_filter not in STATUS_FILTERS:
        raise ValueError(f"unknown status filter {status_filter!r}")
    ids = _matching_ids(
        connection,
        status_filter,
        priority_ids,
        priority_only=priority_only,
        order_by_updated_at=order_by_updated_at,
    )
    if current_id is None:
        return ids[-1] if start_with_latest and ids else (ids[0] if ids else None)
    return seek(
        connection,
        current_id,
        "next",
        status_filter,
        priority_ids,
        priority_only=priority_only,
        order_by_updated_at=order_by_updated_at,
        wrap_next=wrap_next,
    )


def next_undecided(connection: sqlite3.Connection, priority_ids: Iterable[str] = ()) -> str | None:
    return next_matching(connection, "undecided", priority_ids)


def seek(
    connection: sqlite3.Connection,
    current_id: str | None,
    direction: str,
    status_filter: str = "all",
    priority_ids: Iterable[str] = (),
    *,
    priority_only: bool = False,
    order_by_updated_at: bool = False,
    wrap_next: bool = False,
) -> str | None:
    """Move through files in either direction, limited to one status when asked."""
    if direction not in {"prev", "next"}:
        raise ValueError("direction must be prev or next")
    if status_filter not in STATUS_FILTERS:
        raise ValueError(f"unknown status filter {status_filter!r}")
    ids = _matching_ids(
        connection,
        status_filter,
        priority_ids,
        priority_only=priority_only,
        order_by_updated_at=order_by_updated_at,
    )
    if not ids:
        return None
    if current_id not in ids:
        return ids[0] if direction == "next" else ids[-1]
    index = ids.index(current_id) + (1 if direction == "next" else -1)
    if 0 <= index < len(ids):
        return ids[index]
    if direction == "next" and wrap_next:
        return ids[0]
    return None


def progress(connection: sqlite3.Connection) -> dict[str, int]:
    total = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    counts = {row["status"]: row["count"] for row in connection.execute("SELECT status, COUNT(*) AS count FROM status GROUP BY status")}
    labeled = counts.get("labeled", 0)
    clean = counts.get("clean", 0)
    duplicate = counts.get("duplicate", 0)
    return {"total": total, "labeled": labeled, "clean": clean, "duplicate": duplicate, "undecided": total - labeled - clean - duplicate}


def queue_progress(connection: sqlite3.Connection, priority_ids: list[str]) -> dict[str, int]:
    total = len(priority_ids)
    if total == 0:
        return {"total": 0, "labeled": 0, "clean": 0, "duplicate": 0, "undecided": 0}
    placeholders = ",".join("?" * total)
    rows = connection.execute(
        f"SELECT status, COUNT(*) AS count FROM status WHERE file_id IN ({placeholders}) GROUP BY status",
        priority_ids,
    )
    counts = {row["status"]: row["count"] for row in rows}
    labeled = counts.get("labeled", 0)
    clean = counts.get("clean", 0)
    duplicate = counts.get("duplicate", 0)
    return {"total": total, "labeled": labeled, "clean": clean, "duplicate": duplicate, "undecided": total - labeled - clean - duplicate}
