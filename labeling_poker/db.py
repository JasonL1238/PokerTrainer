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
    for path in paths:
        file_id = path.stem
        relative_path = path.relative_to(directory).as_posix()
        connection.execute(
            "INSERT INTO files(id, path) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET path=excluded.path",
            (file_id, relative_path),
        )
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


def next_undecided(connection: sqlite3.Connection, priority_ids: Iterable[str] = ()) -> str | None:
    ids = file_ids(connection)
    undecided = {row["id"] for row in connection.execute("SELECT id FROM files WHERE id NOT IN (SELECT file_id FROM status)")}
    for file_id in priority_ids:
        if file_id in undecided:
            return file_id
    return next((file_id for file_id in ids if file_id in undecided), None)


def seek(connection: sqlite3.Connection, current_id: str | None, direction: str) -> str | None:
    ids = file_ids(connection)
    if not ids:
        return None
    if current_id not in ids:
        return ids[0] if direction == "next" else ids[-1]
    index = ids.index(current_id) + (1 if direction == "next" else -1)
    return ids[index] if 0 <= index < len(ids) else None


def progress(connection: sqlite3.Connection) -> dict[str, int]:
    total = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    counts = {row["status"]: row["count"] for row in connection.execute("SELECT status, COUNT(*) AS count FROM status GROUP BY status")}
    labeled = counts.get("labeled", 0)
    clean = counts.get("clean", 0)
    duplicate = counts.get("duplicate", 0)
    return {"total": total, "labeled": labeled, "clean": clean, "duplicate": duplicate, "undecided": total - labeled - clean - duplicate}
