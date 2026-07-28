from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    ExtractedFrame,
    Hand,
    HandCorrection,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    ProcessingJob,
    ReconstructionFrameReview,
    ROIProfile,
    ROIRegion,
    Session,
    SettlementEntry,
    SolverRangeProfile,
    SolverRun,
    VideoRecord,
)
from poker_tracker.ui.roi import validate_roi_bounds

# Anchored to the project root so launching from another directory does not
# silently create a second, empty database.
DEFAULT_DB_PATH = os.environ.get(
    "POKER_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "poker_tracker.db"),
)
SCHEMA_VERSION = 12


class PokerDatabase:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        # One connection is shared across Streamlit's script-run threads, so every
        # statement goes through _execute() under a re-entrant lock, and grouped
        # writes use transaction() for atomicity.
        self._lock = threading.RLock()
        self._txn_depth = 0
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._execute("PRAGMA foreign_keys = ON")
        self._execute("PRAGMA journal_mode = WAL")
        self._execute("PRAGMA synchronous = NORMAL")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, params)

    def _commit(self) -> None:
        # Inside transaction() the outermost exit owns the commit/rollback.
        with self._lock:
            if self._txn_depth == 0:
                self._connection.commit()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[PokerDatabase]:
        """Group multiple writes into a single atomic commit.

        Re-entrant: nested transaction() blocks (and the per-method commits of
        the CRUD helpers called inside) defer to the outermost block, which
        commits on success and rolls back everything on the first exception.
        ``immediate=True`` acquires SQLite's write reservation before any reads,
        allowing check-then-insert policies to serialize across connections.
        """
        with self._lock:
            if self._txn_depth == 0 and immediate:
                self._connection.execute("BEGIN IMMEDIATE")
            self._txn_depth += 1
            try:
                yield self
            except BaseException:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._connection.rollback()
                raise
            else:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._connection.commit()

    def init_db(self) -> None:
        """Create the schema and apply any pending versioned migrations."""
        with self._lock:
            stored_version = self.schema_version()
            if stored_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {stored_version} is newer than this app "
                    f"understands ({SCHEMA_VERSION}). Update the app before opening it."
                )
            self._create_base_schema()
            if stored_version < 5:
                # Pre-versioning databases: idempotent column backfill.
                self._apply_legacy_backfill()
            for version in range(max(stored_version, 5) + 1, SCHEMA_VERSION + 1):
                migration = _MIGRATIONS.get(version)
                if migration is None:
                    raise RuntimeError(f"No migration registered for schema version {version}.")
                migration(self)
            self._execute(
                """
                INSERT INTO schema_metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            self._commit()

    def _create_base_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date_played TEXT NOT NULL,
                platform TEXT NOT NULL DEFAULT '',
                stakes TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                hand_number INTEGER NOT NULL,
                game_type TEXT NOT NULL DEFAULT '',
                blinds_antes TEXT NOT NULL DEFAULT '',
                table_size INTEGER,
                effective_stack REAL,
                hero_position TEXT NOT NULL DEFAULT '',
                hero_cards TEXT NOT NULL DEFAULT '',
                board_cards TEXT NOT NULL DEFAULT '',
                pot_size REAL,
                result TEXT NOT NULL DEFAULT '',
                hero_bb_won REAL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                confidence_score REAL,
                source_type TEXT NOT NULL DEFAULT 'manual',
                tags TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                player_key TEXT NOT NULL,
                seat_index INTEGER,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL DEFAULT '',
                starting_stack REAL,
                is_hero INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                player_key TEXT,
                street TEXT NOT NULL,
                action_index INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL DEFAULT '',
                action_type TEXT NOT NULL,
                amount REAL,
                amount_semantics TEXT NOT NULL DEFAULT 'incremental',
                forced_bet_type TEXT,
                is_live_post INTEGER,
                pot_before REAL,
                stack_before REAL,
                notes TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                hand_summary TEXT NOT NULL,
                theory_coach TEXT NOT NULL,
                exploit_coach TEXT NOT NULL,
                ev_math_notes TEXT NOT NULL DEFAULT '',
                study_lesson TEXT NOT NULL,
                next_review_question TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                is_stale INTEGER NOT NULL DEFAULT 0,
                stale_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS coaching_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                raw_prompt TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                review_type TEXT NOT NULL,
                safety_mode TEXT NOT NULL DEFAULT 'post_session_only',
                hand_id INTEGER,
                session_id INTEGER,
                parsed_sections TEXT NOT NULL DEFAULT '{}',
                is_stale INTEGER NOT NULL DEFAULT 0,
                stale_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                correction_type TEXT NOT NULL,
                before_state TEXT NOT NULL DEFAULT '{}',
                after_state TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                issue_types TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL,
                evidence_snapshot TEXT NOT NULL DEFAULT '{}',
                resolution_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS solver_range_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                notation TEXT NOT NULL,
                table_size INTEGER,
                position TEXT NOT NULL DEFAULT '',
                scenario TEXT NOT NULL DEFAULT '',
                pot_type TEXT NOT NULL DEFAULT '',
                stack_bb REAL,
                description TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS solver_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                backend_name TEXT NOT NULL DEFAULT 'texassolver',
                backend_version TEXT NOT NULL DEFAULT '',
                input_hash TEXT NOT NULL,
                spot TEXT NOT NULL DEFAULT '{}',
                range_ip TEXT NOT NULL DEFAULT '{}',
                range_oop TEXT NOT NULL DEFAULT '{}',
                assumptions TEXT NOT NULL DEFAULT '[]',
                evidence TEXT NOT NULL DEFAULT '{}',
                command_path TEXT NOT NULL DEFAULT '',
                result_path TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL DEFAULT '',
                exploitability_pct REAL,
                runtime_seconds REAL,
                error_message TEXT NOT NULL DEFAULT '',
                pid INTEGER,
                heartbeat_at TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                duration_seconds REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                frame_count INTEGER,
                uploaded_at TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS processing_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                video_id INTEGER NOT NULL,
                progress_percent REAL NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                pid INTEGER,
                heartbeat_at TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS extracted_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                timestamp_seconds REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,
                UNIQUE(video_id, frame_index, image_path)
            );

            CREATE TABLE IF NOT EXISTS reconstruction_frame_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                hand_number INTEGER NOT NULL,
                source_image TEXT NOT NULL,
                timestamp_seconds REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'unreviewed',
                issue_types TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,
                UNIQUE(job_id, hand_number, source_image)
            );

            CREATE TABLE IF NOT EXISTS roi_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                platform TEXT NOT NULL DEFAULT 'ClubWPT Gold',
                table_layout TEXT NOT NULL DEFAULT '',
                video_width INTEGER,
                video_height INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS roi_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                roi_key TEXT NOT NULL,
                roi_type TEXT NOT NULL DEFAULT 'unknown',
                label TEXT NOT NULL DEFAULT '',
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                seat_index INTEGER,
                card_index INTEGER,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES roi_profiles(id) ON DELETE CASCADE,
                UNIQUE(profile_id, roi_key)
            );

            CREATE INDEX IF NOT EXISTS idx_hands_session_id ON hands(session_id);
            CREATE INDEX IF NOT EXISTS idx_hand_players_hand_id ON hand_players(hand_id);
            CREATE INDEX IF NOT EXISTS idx_actions_hand_id ON actions(hand_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_hand_id ON hand_reviews(hand_id);
            CREATE INDEX IF NOT EXISTS idx_coaching_reviews_hand_id ON coaching_reviews(hand_id);
            CREATE INDEX IF NOT EXISTS idx_coaching_reviews_session_id ON coaching_reviews(session_id);
            CREATE INDEX IF NOT EXISTS idx_hand_corrections_hand_id
                ON hand_corrections(hand_id, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_hand_issues_status
                ON hand_issues(status, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_hand_issues_hand
                ON hand_issues(hand_id, status, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_solver_runs_hand
                ON solver_runs(hand_id, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_solver_runs_cache
                ON solver_runs(input_hash, status);
            CREATE INDEX IF NOT EXISTS idx_solver_runs_status
                ON solver_runs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_videos_session_id ON videos(session_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON processing_jobs(video_id);
            CREATE INDEX IF NOT EXISTS idx_frames_video_id ON extracted_frames(video_id);
            CREATE INDEX IF NOT EXISTS idx_reconstruction_reviews_job_hand
                ON reconstruction_frame_reviews(job_id, hand_number);
            CREATE INDEX IF NOT EXISTS idx_roi_profiles_active ON roi_profiles(is_active);
            CREATE INDEX IF NOT EXISTS idx_roi_regions_profile_id ON roi_regions(profile_id);
            """
        )

    def _apply_legacy_backfill(self) -> None:
        """Backfill columns added before schema versioning existed (idempotent)."""
        self._ensure_column("hands", "game_type", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hands", "blinds_antes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hands", "table_size", "INTEGER")
        self._ensure_column("hands", "effective_stack", "REAL")
        self._ensure_column("hands", "review_status", "TEXT NOT NULL DEFAULT 'unreviewed'")
        self._ensure_column("hands", "confidence_score", "REAL")
        self._ensure_column("hands", "source_type", "TEXT NOT NULL DEFAULT 'manual'")
        self._ensure_column("hands", "tags", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("hand_reviews", "ev_math_notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hand_reviews", "next_review_question", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hand_reviews", "notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(
            "coaching_reviews", "safety_mode", "TEXT NOT NULL DEFAULT 'post_session_only'"
        )
        self._ensure_column("coaching_reviews", "parsed_sections", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("videos", "frame_count", "INTEGER")
        self._ensure_column("roi_profiles", "table_layout", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("roi_profiles", "is_active", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("roi_regions", "notes", "TEXT NOT NULL DEFAULT ''")

    def _ensure_column(self, table_name: str, column_name: str, column_spec: str) -> None:
        columns = {
            row["name"] for row in self._execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            self._execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}")

    def create_session(self, session: Session) -> Session:
        payload = session.model_dump()
        cursor = self._execute(
            """
            INSERT INTO sessions (name, date_played, platform, stakes, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"],
                _serialize_date(payload["date_played"]),
                payload["platform"],
                payload["stakes"],
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return session.model_copy(update={"id": cursor.lastrowid})

    def create_hand(self, hand: Hand) -> Hand:
        payload = hand.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hands (
                session_id, hand_number, game_type, blinds_antes, table_size,
                effective_stack, hero_position, hero_cards, board_cards, pot_size,
                result, hero_bb_won, review_status, confidence_score, source_type,
                tags, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["hand_number"],
                payload["game_type"],
                payload["blinds_antes"],
                payload["table_size"],
                payload["effective_stack"],
                payload["hero_position"],
                payload["hero_cards"],
                payload["board_cards"],
                payload["pot_size"],
                payload["result"],
                payload["hero_bb_won"],
                payload["review_status"],
                payload["confidence_score"],
                payload["source_type"],
                _serialize_json(payload["tags"]),
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return hand.model_copy(update={"id": cursor.lastrowid})

    def update_hand_status(self, hand_id: int, review_status: str) -> None:
        self._execute(
            "UPDATE hands SET review_status = ? WHERE id = ?",
            (review_status, hand_id),
        )
        self._commit()

    def update_hand_facts(self, hand: Hand, *, correction_notes: str = "") -> Hand:
        """Persist corrected source facts and retain an auditable before/after event."""

        hand = Hand.model_validate(hand.model_dump())
        if hand.id is None:
            raise ValueError("Cannot update a hand without an id.")
        stored = self.fetch_hand(hand.id)
        if stored is None:
            raise ValueError("Hand not found.")
        if hand.session_id != stored.session_id or hand.hand_number != stored.hand_number:
            raise ValueError("Hand corrections cannot move or renumber the hand.")

        fields = (
            "game_type",
            "blinds_antes",
            "table_size",
            "effective_stack",
            "hero_position",
            "hero_cards",
            "board_cards",
            "pot_size",
            "result",
            "hero_bb_won",
            "tags",
            "notes",
        )
        before_state = {field: getattr(stored, field) for field in fields}
        source_type = (
            "corrected_cv"
            if stored.source_type in {"cv_import", "corrected_cv"}
            else stored.source_type
        )
        corrected = hand.model_copy(
            update={
                "source_type": source_type,
                "review_status": "needs_correction",
            }
        )
        after_state = {field: getattr(corrected, field) for field in fields}
        if before_state == after_state:
            return stored

        with self.transaction():
            payload = corrected.model_dump()
            cursor = self._execute(
                """
                UPDATE hands
                SET game_type = ?, blinds_antes = ?, table_size = ?,
                    effective_stack = ?, hero_position = ?, hero_cards = ?,
                    board_cards = ?, pot_size = ?, result = ?, hero_bb_won = ?,
                    review_status = 'needs_correction', source_type = ?,
                    tags = ?, notes = ?
                WHERE id = ?
                """,
                (
                    payload["game_type"],
                    payload["blinds_antes"],
                    payload["table_size"],
                    payload["effective_stack"],
                    payload["hero_position"],
                    payload["hero_cards"],
                    payload["board_cards"],
                    payload["pot_size"],
                    payload["result"],
                    payload["hero_bb_won"],
                    payload["source_type"],
                    _serialize_json(payload["tags"]),
                    payload["notes"],
                    payload["id"],
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Hand not found.")
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand.id,
                    correction_type="hand_facts",
                    before_state=before_state,
                    after_state=after_state,
                    notes=correction_notes.strip(),
                )
            )
            self._invalidate_hand_derivatives(hand.id, force_review_status=True)

        refreshed = self.fetch_hand(hand.id)
        if refreshed is None:
            raise RuntimeError("Corrected hand could not be reloaded.")
        return refreshed

    def create_hand_correction(self, correction: HandCorrection) -> HandCorrection:
        with self.transaction():
            return self._record_hand_correction(correction)

    def _record_hand_correction(self, correction: HandCorrection) -> HandCorrection:
        if self.fetch_hand(correction.hand_id) is None:
            raise ValueError("Hand not found.")
        payload = correction.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hand_corrections (
                hand_id, correction_type, before_state, after_state, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["correction_type"],
                _serialize_json(payload["before_state"]),
                _serialize_json(payload["after_state"]),
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return correction.model_copy(update={"id": cursor.lastrowid})

    def fetch_hand_corrections(self, hand_id: int) -> list[HandCorrection]:
        rows = self._execute(
            """
            SELECT * FROM hand_corrections
            WHERE hand_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_hand_correction_from_row(row) for row in rows]

    def create_hand_issue(
        self, issue: HandIssue, *, apply_workflow: bool = True
    ) -> HandIssue:
        """Save a debug-later report and freeze the evidence visible at flag time."""

        hand = self.fetch_hand(issue.hand_id)
        if hand is None:
            raise ValueError("Hand not found.")
        snapshot = issue.evidence_snapshot or {
            "hand": hand.model_dump(mode="json"),
            "session": (
                None
                if (session := self.fetch_session(hand.session_id)) is None
                else session.model_dump(mode="json")
            ),
            "players": [
                player.model_dump(mode="json")
                for player in self.fetch_players_by_hand(issue.hand_id)
            ],
            "actions": [
                action.model_dump(mode="json")
                for action in self.fetch_actions_by_hand(issue.hand_id)
            ],
            "corrections": [
                correction.model_dump(mode="json")
                for correction in self.fetch_hand_corrections(issue.hand_id)
            ],
        }
        saved_issue = issue.model_copy(
            update={
                "description": issue.description.strip(),
                "evidence_snapshot": snapshot,
            }
        )
        payload = saved_issue.model_dump()
        with self.transaction():
            cursor = self._execute(
                """
                INSERT INTO hand_issues (
                    hand_id, status, issue_types, description, evidence_snapshot,
                    resolution_notes, created_at, updated_at, resolved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["hand_id"],
                    payload["status"],
                    _serialize_json(payload["issue_types"]),
                    payload["description"],
                    _serialize_json(payload["evidence_snapshot"]),
                    payload["resolution_notes"],
                    _serialize_datetime(payload["created_at"]),
                    _serialize_datetime(payload["updated_at"]),
                    _serialize_optional_datetime(payload["resolved_at"]),
                ),
            )
            if apply_workflow and saved_issue.status == "open":
                self._flag_hand_for_debugging(saved_issue.hand_id)
        return saved_issue.model_copy(update={"id": cursor.lastrowid})

    def fetch_hand_issues(
        self,
        *,
        hand_id: int | None = None,
        status: str | None = None,
    ) -> list[HandIssue]:
        clauses: list[str] = []
        params: list[object] = []
        if hand_id is not None:
            clauses.append("hand_id = ?")
            params.append(hand_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        rows = self._execute(
            f"""
            SELECT * FROM hand_issues
            {where}
            ORDER BY created_at DESC, id DESC
            """,
            tuple(params),
        ).fetchall()
        return [_hand_issue_from_row(row) for row in rows]

    def resolve_hand_issue(
        self, issue_id: int, *, resolution_notes: str
    ) -> HandIssue:
        notes = resolution_notes.strip()
        if not notes:
            raise ValueError("Resolution notes are required.")
        now = datetime.now(UTC)
        with self.transaction():
            cursor = self._execute(
                """
                UPDATE hand_issues
                SET status = 'resolved', resolution_notes = ?,
                    updated_at = ?, resolved_at = ?
                WHERE id = ? AND status = 'open'
                """,
                (
                    notes,
                    _serialize_datetime(now),
                    _serialize_datetime(now),
                    issue_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Open hand issue not found.")
        row = self._execute(
            "SELECT * FROM hand_issues WHERE id = ?", (issue_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Resolved hand issue could not be reloaded.")
        return _hand_issue_from_row(row)

    def _flag_hand_for_debugging(self, hand_id: int) -> None:
        reason = "Hand was flagged for future debugging."
        self._execute(
            "UPDATE hands SET review_status = 'needs_correction' WHERE id = ?",
            (hand_id,),
        )
        self._execute(
            """
            UPDATE hand_reviews
            SET is_stale = 1, stale_reason = ?
            WHERE hand_id = ?
            """,
            (reason, hand_id),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1, stale_reason = ?
            WHERE hand_id = ? AND review_type = 'hand'
            """,
            (reason, hand_id),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1,
                stale_reason = 'A hand in this session was flagged for future debugging.'
            WHERE review_type = 'session'
              AND session_id = (SELECT session_id FROM hands WHERE id = ?)
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE solver_runs
            SET status = CASE
                    WHEN status IN ('queued', 'running') THEN 'cancelling'
                    ELSE 'stale'
                END,
                error_message = 'Hand was flagged for future debugging.'
            WHERE hand_id = ? AND status IN ('queued', 'running', 'completed')
            """,
            (hand_id,),
        )

    def move_hand_to_session(self, hand_id: int, session_id: int) -> Hand:
        """Move a hand between sessions and resolve hand-number collisions safely."""

        hand = self.fetch_hand(hand_id)
        if hand is None:
            raise ValueError("Hand not found.")
        if self.fetch_session(session_id) is None:
            raise ValueError("Target session not found.")
        if hand.session_id == session_id:
            return hand

        with self.transaction():
            existing_numbers = {
                item.hand_number for item in self.fetch_hands_by_session(session_id)
            }
            hand_number = hand.hand_number
            if hand_number in existing_numbers:
                hand_number = max(existing_numbers, default=0) + 1
            self._execute(
                "UPDATE hands SET session_id = ?, hand_number = ? WHERE id = ?",
                (session_id, hand_number, hand_id),
            )
            self._execute(
                """
                UPDATE coaching_reviews
                SET session_id = ?
                WHERE review_type = 'hand' AND hand_id = ?
                """,
                (session_id, hand_id),
            )
            self._execute(
                """
                UPDATE coaching_reviews
                SET is_stale = 1,
                    stale_reason = 'A hand moved into or out of this session; rerun coaching.'
                WHERE review_type = 'session' AND session_id IN (?, ?)
                """,
                (hand.session_id, session_id),
            )

        moved = self.fetch_hand(hand_id)
        if moved is None:
            raise RuntimeError("Moved hand could not be reloaded.")
        return moved

    def update_hand_accounting_evidence(
        self,
        hand_id: int,
        *,
        pot_size: float | None,
        hero_bb_won: float | None,
    ) -> None:
        stored = self.fetch_hand(hand_id)
        if stored is None:
            raise ValueError("Hand not found.")
        before_state = {
            "pot_size": stored.pot_size,
            "hero_bb_won": stored.hero_bb_won,
        }
        after_state = {"pot_size": pot_size, "hero_bb_won": hero_bb_won}
        if before_state == after_state:
            return
        with self.transaction():
            cursor = self._execute(
                """
                UPDATE hands
                SET pot_size = ?, hero_bb_won = ?,
                    review_status = 'needs_correction',
                    source_type = CASE
                        WHEN source_type IN ('cv_import', 'corrected_cv') THEN 'corrected_cv'
                        ELSE source_type
                    END
                WHERE id = ?
                """,
                (pot_size, hero_bb_won, hand_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Hand not found.")
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand_id,
                    correction_type="hand_facts",
                    before_state=before_state,
                    after_state=after_state,
                    notes="Updated accounting evidence.",
                )
            )
            self._invalidate_hand_derivatives(hand_id, force_review_status=True)

    def delete_hand(self, hand_id: int) -> None:
        self._execute("DELETE FROM hands WHERE id = ?", (hand_id,))
        self._commit()

    def create_hand_player(self, player: HandPlayer) -> HandPlayer:
        payload = player.model_dump()
        self._validate_single_hero(payload["hand_id"], payload["is_hero"], exclude_player_id=None)
        cursor = self._execute(
            """
            INSERT INTO hand_players (
                hand_id, player_key, seat_index, player_name, position,
                starting_stack, is_hero, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["player_key"],
                payload["seat_index"],
                payload["player_name"],
                payload["position"],
                payload["starting_stack"],
                int(payload["is_hero"]),
                payload["notes"],
            ),
        )
        self._invalidate_hand_derivatives(player.hand_id)
        self._commit()
        return player.model_copy(update={"id": cursor.lastrowid})

    def update_hand_player(
        self, player: HandPlayer, *, correction_notes: str = ""
    ) -> HandPlayer:
        if player.id is None:
            raise ValueError("Cannot update a player without an id.")
        payload = player.model_dump()
        with self.transaction():
            stored_row = self._execute(
                "SELECT * FROM hand_players WHERE id = ?", (payload["id"],)
            ).fetchone()
            if stored_row is None:
                raise ValueError("Cannot update a player that no longer exists.")
            stored = _hand_player_from_row(stored_row)
            if stored.hand_id != payload["hand_id"] or stored.player_key != payload["player_key"]:
                raise ValueError("Saved player identity no longer matches this hand.")
            self._validate_single_hero(
                payload["hand_id"],
                payload["is_hero"],
                exclude_player_id=payload["id"],
            )
            cursor = self._execute(
                """
                UPDATE hand_players
                SET seat_index = ?, player_name = ?, position = ?,
                    starting_stack = ?, is_hero = ?, notes = ?
                WHERE id = ? AND hand_id = ? AND player_key = ?
                """,
                (
                    payload["seat_index"],
                    payload["player_name"],
                    payload["position"],
                    payload["starting_stack"],
                    int(payload["is_hero"]),
                    payload["notes"],
                    payload["id"],
                    payload["hand_id"],
                    payload["player_key"],
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Saved player identity no longer matches this hand.")
            self._execute(
                """
                UPDATE actions
                SET player_name = ?, position = ?
                WHERE hand_id = ? AND player_key = ?
                """,
                (
                    payload["player_name"],
                    payload["position"],
                    payload["hand_id"],
                    payload["player_key"],
                ),
            )
            self._execute(
                """
                UPDATE settlement_entries
                SET player_name = ?
                WHERE hand_id = ? AND player_key = ?
                """,
                (
                    payload["player_name"],
                    payload["hand_id"],
                    payload["player_key"],
                ),
            )
            updated = player.model_copy()
            if stored != updated:
                self._record_hand_correction(
                    HandCorrection(
                        hand_id=player.hand_id,
                        correction_type="player_update",
                        before_state=stored.model_dump(mode="json"),
                        after_state=updated.model_dump(mode="json"),
                        notes=correction_notes.strip(),
                    )
                )
                self._invalidate_hand_derivatives(
                    payload["hand_id"], force_review_status=True
                )
        return updated

    def _validate_single_hero(
        self,
        hand_id: int,
        is_hero: bool,
        *,
        exclude_player_id: int | None,
    ) -> None:
        if not is_hero:
            return
        row = self._execute(
            """
            SELECT id FROM hand_players
            WHERE hand_id = ? AND is_hero = 1
              AND (? IS NULL OR id != ?)
            LIMIT 1
            """,
            (hand_id, exclude_player_id, exclude_player_id),
        ).fetchone()
        if row is not None:
            raise ValueError("A hand can have only one Hero player.")

    def create_action(self, action: Action) -> Action:
        payload = action.model_dump()
        with self.transaction():
            self._resolve_action_player(payload)
            action_index = payload["action_index"] or self.next_action_index(
                payload["hand_id"], payload["street"]
            )
            self._assert_action_index_available(payload["hand_id"], payload["street"], action_index)
            return self._insert_action(action, payload, action_index)

    def create_corrected_action(
        self, action: Action, *, correction_notes: str = ""
    ) -> Action:
        """Add an action during review and preserve it as correction evidence."""

        with self.transaction():
            saved = self.create_action(action)
            self._record_hand_correction(
                HandCorrection(
                    hand_id=saved.hand_id,
                    correction_type="action_create",
                    before_state={},
                    after_state=saved.model_dump(mode="json"),
                    notes=correction_notes.strip(),
                )
            )
            self._invalidate_hand_derivatives(
                saved.hand_id, force_review_status=True
            )
        return saved

    def _insert_action(self, action: Action, payload: dict, action_index: int) -> Action:
        cursor = self._execute(
            """
            INSERT INTO actions (
                hand_id, player_key, street, action_index, player_name, position,
                action_type, amount, amount_semantics, forced_bet_type,
                is_live_post, pot_before, stack_before, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["player_key"],
                payload["street"],
                action_index,
                payload["player_name"],
                payload["position"],
                payload["action_type"],
                payload["amount"],
                payload["amount_semantics"],
                payload["forced_bet_type"],
                (None if payload["is_live_post"] is None else int(payload["is_live_post"])),
                payload["pot_before"],
                payload["stack_before"],
                payload["notes"],
            ),
        )
        self._invalidate_hand_derivatives(payload["hand_id"])
        self._commit()
        return action.model_copy(
            update={
                "id": cursor.lastrowid,
                "action_index": action_index,
                "player_key": payload["player_key"],
            }
        )

    def next_action_index(self, hand_id: int, street: str) -> int:
        row = self._execute(
            """
            SELECT COALESCE(MAX(action_index), 0) + 1 AS next_index
            FROM actions
            WHERE hand_id = ? AND street = ?
            """,
            (hand_id, street),
        ).fetchone()
        return int(row["next_index"])

    def update_action(
        self, action: Action, *, correction_notes: str = ""
    ) -> Action:
        if action.id is None:
            raise ValueError("Cannot update an action without an id.")
        payload = action.model_dump()
        with self.transaction():
            stored_row = self._execute(
                "SELECT * FROM actions WHERE id = ?", (payload["id"],)
            ).fetchone()
            if stored_row is None:
                raise ValueError("Cannot update an action that no longer exists.")
            stored = _action_from_row(stored_row)
            stored_hand_id = stored.hand_id
            if stored_hand_id != payload["hand_id"]:
                raise ValueError("Action does not belong to the requested hand.")
            self._resolve_action_player(payload)
            action_index = payload["action_index"] or self.next_action_index(
                stored_hand_id, payload["street"]
            )
            self._assert_action_index_available(
                stored_hand_id,
                payload["street"],
                action_index,
                exclude_action_id=payload["id"],
            )
            payload["action_index"] = action_index
            self._update_action_row(payload, stored_hand_id)
            updated = action.model_copy(
                update={
                    "action_index": action_index,
                    "player_key": payload["player_key"],
                }
            )
            if stored != updated:
                self._record_hand_correction(
                    HandCorrection(
                        hand_id=stored_hand_id,
                        correction_type="action_update",
                        before_state=stored.model_dump(mode="json"),
                        after_state=updated.model_dump(mode="json"),
                        notes=correction_notes.strip(),
                    )
                )
                self._invalidate_hand_derivatives(
                    stored_hand_id, force_review_status=True
                )
        return updated

    def _update_action_row(self, payload: dict, stored_hand_id: int) -> None:
        cursor = self._execute(
            """
            UPDATE actions
            SET player_key = ?, street = ?, action_index = ?, player_name = ?, position = ?,
                action_type = ?, amount = ?, amount_semantics = ?,
                forced_bet_type = ?, is_live_post = ?,
                pot_before = ?, stack_before = ?, notes = ?
            WHERE id = ? AND hand_id = ?
            """,
            (
                payload["player_key"],
                payload["street"],
                payload["action_index"],
                payload["player_name"],
                payload["position"],
                payload["action_type"],
                payload["amount"],
                payload["amount_semantics"],
                payload["forced_bet_type"],
                (None if payload["is_live_post"] is None else int(payload["is_live_post"])),
                payload["pot_before"],
                payload["stack_before"],
                payload["notes"],
                payload["id"],
                stored_hand_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("Saved action identity no longer matches this hand.")

    def _assert_action_index_available(
        self,
        hand_id: int,
        street: str,
        action_index: int,
        *,
        exclude_action_id: int | None = None,
    ) -> None:
        row = self._execute(
            """
            SELECT id
            FROM actions
            WHERE hand_id = ? AND street = ? AND action_index = ?
              AND (? IS NULL OR id != ?)
            LIMIT 1
            """,
            (
                hand_id,
                street,
                action_index,
                exclude_action_id,
                exclude_action_id,
            ),
        ).fetchone()
        if row is not None:
            raise ValueError("Action index must be unique within each hand and street.")

    def delete_action(self, action_id: int, *, correction_notes: str = "") -> None:
        with self.transaction():
            row = self._execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
            self._execute("DELETE FROM actions WHERE id = ?", (action_id,))
            if row is not None:
                stored = _action_from_row(row)
                self._record_hand_correction(
                    HandCorrection(
                        hand_id=stored.hand_id,
                        correction_type="action_delete",
                        before_state=stored.model_dump(mode="json"),
                        after_state={},
                        notes=correction_notes.strip(),
                    )
                )
                self._invalidate_hand_derivatives(
                    stored.hand_id, force_review_status=True
                )

    def _invalidate_hand_settlement(self, hand_id: int) -> None:
        self._execute(
            """
            UPDATE hand_settlements
            SET status = 'needs_correction',
                is_balanced = 0,
                warnings = ?,
                updated_at = ?
            WHERE hand_id = ?
            """,
            (
                _serialize_json(["Players or actions changed; reconcile again."]),
                _serialize_datetime(datetime.now(UTC)),
                hand_id,
            ),
        )

    def _invalidate_hand_derivatives(
        self, hand_id: int, *, force_review_status: bool = False
    ) -> None:
        """Flag derived reviews and reconciliation state after source edits."""
        has_saved_hand_review = self._execute(
            """
            SELECT (
                EXISTS(SELECT 1 FROM hand_reviews WHERE hand_id = ?)
                OR EXISTS(
                    SELECT 1 FROM coaching_reviews
                    WHERE hand_id = ? AND review_type = 'hand'
                )
            ) AS has_review
            """,
            (hand_id, hand_id),
        ).fetchone()
        self._invalidate_hand_settlement(hand_id)
        self._execute(
            """
            UPDATE hand_reviews
            SET is_stale = 1,
                stale_reason = 'Hand evidence changed; rerun coaching.'
            WHERE hand_id = ?
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1,
                stale_reason = 'Hand evidence changed; rerun coaching.'
            WHERE hand_id = ? AND review_type = 'hand'
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE solver_runs
            SET status = CASE
                    WHEN status IN ('queued', 'running') THEN 'cancelling'
                    ELSE 'stale'
                END,
                error_message = 'Hand evidence changed; rerun solver analysis.'
            WHERE hand_id = ? AND status IN ('queued', 'running', 'completed')
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1,
                stale_reason = 'A hand in this session changed; rerun coaching.'
            WHERE review_type = 'session'
              AND session_id = (SELECT session_id FROM hands WHERE id = ?)
            """,
            (hand_id,),
        )
        if force_review_status or (
            has_saved_hand_review is not None and bool(has_saved_hand_review["has_review"])
        ):
            self._execute(
                "UPDATE hands SET review_status = 'needs_correction' WHERE id = ?",
                (hand_id,),
            )

    def _resolve_action_player(self, payload: dict) -> None:
        player_key = payload["player_key"]
        if player_key is not None:
            row = self._execute(
                """
                SELECT 1 FROM hand_players
                WHERE hand_id = ? AND player_key = ?
                """,
                (payload["hand_id"], player_key),
            ).fetchone()
            if row is None:
                raise ValueError("Action player key does not belong to this hand.")
            return
        rows = self._execute(
            """
            SELECT player_key, position
            FROM hand_players
            WHERE hand_id = ? AND player_name = ?
            ORDER BY id
            """,
            (payload["hand_id"], payload["player_name"]),
        ).fetchall()
        exact = [row for row in rows if row["position"] == payload["position"]]
        candidates = exact if len(exact) == 1 else rows
        if len(candidates) == 1:
            payload["player_key"] = candidates[0]["player_key"]
        elif rows and payload["amount_semantics"] != "unknown":
            raise ValueError("A monetary action must resolve to exactly one player in the hand.")

    def create_hand_review(self, review: HandReview) -> HandReview:
        payload = review.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hand_reviews (
                hand_id, hand_summary, theory_coach, exploit_coach, ev_math_notes,
                study_lesson, next_review_question, notes, is_stale,
                stale_reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["hand_summary"],
                payload["theory_coach"],
                payload["exploit_coach"],
                payload["ev_math_notes"],
                payload["study_lesson"],
                payload["next_review_question"],
                payload["notes"],
                int(payload["is_stale"]),
                payload["stale_reason"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return review.model_copy(update={"id": cursor.lastrowid})

    def fetch_sessions(self) -> list[Session]:
        rows = self._execute("SELECT * FROM sessions ORDER BY date_played DESC, id DESC").fetchall()
        return [_session_from_row(row) for row in rows]

    def fetch_hands_by_session(self, session_id: int) -> list[Hand]:
        rows = self._execute(
            "SELECT * FROM hands WHERE session_id = ? ORDER BY hand_number, id",
            (session_id,),
        ).fetchall()
        return [_hand_from_row(row) for row in rows]

    def fetch_all_hands(self) -> list[Hand]:
        """Return all hands for portfolio-level browsing and insights."""
        rows = self._execute("SELECT * FROM hands ORDER BY created_at DESC, id DESC").fetchall()
        return [_hand_from_row(row) for row in rows]

    def fetch_session(self, session_id: int) -> Session | None:
        row = self._execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return None if row is None else _session_from_row(row)

    def fetch_hand(self, hand_id: int) -> Hand | None:
        row = self._execute("SELECT * FROM hands WHERE id = ?", (hand_id,)).fetchone()
        return None if row is None else _hand_from_row(row)

    def fetch_actions_by_hand(self, hand_id: int) -> list[Action]:
        rows = self._execute(
            """
            SELECT * FROM actions
            WHERE hand_id = ?
            ORDER BY
                CASE street
                    WHEN 'preflop' THEN 1
                    WHEN 'flop' THEN 2
                    WHEN 'turn' THEN 3
                    WHEN 'river' THEN 4
                    WHEN 'showdown' THEN 5
                    ELSE 5
                END,
                action_index,
                id
            """,
            (hand_id,),
        ).fetchall()
        return [_action_from_row(row) for row in rows]

    def fetch_players_by_hand(self, hand_id: int) -> list[HandPlayer]:
        rows = self._execute(
            """
            SELECT * FROM hand_players
            WHERE hand_id = ?
            ORDER BY is_hero DESC, seat_index, position, id
            """,
            (hand_id,),
        ).fetchall()
        return [_hand_player_from_row(row) for row in rows]

    def upsert_hand_settlement(self, settlement: HandSettlement) -> HandSettlement:
        payload = settlement.model_dump()
        self._execute(
            """
            INSERT INTO hand_settlements (
                hand_id, status, dead_money, rake_rate, rake_cap,
                rake_rounding_unit, no_flop_no_drop, gross_pot, rake_amount,
                net_pot, is_balanced, warnings, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hand_id) DO UPDATE SET
                status = excluded.status,
                dead_money = excluded.dead_money,
                rake_rate = excluded.rake_rate,
                rake_cap = excluded.rake_cap,
                rake_rounding_unit = excluded.rake_rounding_unit,
                no_flop_no_drop = excluded.no_flop_no_drop,
                gross_pot = excluded.gross_pot,
                rake_amount = excluded.rake_amount,
                net_pot = excluded.net_pot,
                is_balanced = excluded.is_balanced,
                warnings = excluded.warnings,
                updated_at = excluded.updated_at
            """,
            (
                payload["hand_id"],
                payload["status"],
                payload["dead_money"],
                payload["rake_rate"],
                payload["rake_cap"],
                payload["rake_rounding_unit"],
                int(payload["no_flop_no_drop"]),
                payload["gross_pot"],
                payload["rake_amount"],
                payload["net_pot"],
                int(payload["is_balanced"]),
                _serialize_json(payload["warnings"]),
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        saved = self.fetch_hand_settlement(settlement.hand_id)
        if saved is None:
            raise RuntimeError("Settlement upsert did not persist a row.")
        return saved

    def fetch_hand_settlement(self, hand_id: int) -> HandSettlement | None:
        row = self._execute(
            "SELECT * FROM hand_settlements WHERE hand_id = ?", (hand_id,)
        ).fetchone()
        return None if row is None else _hand_settlement_from_row(row)

    def create_settlement_entry(self, entry: SettlementEntry) -> SettlementEntry:
        payload = entry.model_dump()
        self._resolve_settlement_entry_player(payload)
        cursor = self._execute(
            """
            INSERT INTO settlement_entries (
                hand_id, entry_type, pot_index, player_key, player_name,
                amount, entry_order
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["entry_type"],
                payload["pot_index"],
                payload["player_key"],
                payload["player_name"],
                payload["amount"],
                payload["entry_order"],
            ),
        )
        self._commit()
        return entry.model_copy(
            update={"id": cursor.lastrowid, "player_key": payload["player_key"]}
        )

    def replace_settlement_entries(
        self, hand_id: int, entries: list[SettlementEntry]
    ) -> list[SettlementEntry]:
        if any(entry.hand_id != hand_id for entry in entries):
            raise ValueError("All settlement entries must belong to the requested hand.")
        with self.transaction():
            self._execute("DELETE FROM settlement_entries WHERE hand_id = ?", (hand_id,))
            saved = [self.create_settlement_entry(entry) for entry in entries]
        return saved

    def fetch_settlement_entries(self, hand_id: int) -> list[SettlementEntry]:
        rows = self._execute(
            """
            SELECT * FROM settlement_entries
            WHERE hand_id = ?
            ORDER BY
                CASE entry_type WHEN 'award' THEN 1 ELSE 2 END,
                pot_index,
                entry_order,
                id
            """,
            (hand_id,),
        ).fetchall()
        return [_settlement_entry_from_row(row) for row in rows]

    def _resolve_settlement_entry_player(self, payload: dict) -> None:
        if payload["player_key"] is not None:
            row = self._execute(
                """
                SELECT player_name FROM hand_players
                WHERE hand_id = ? AND player_key = ?
                """,
                (payload["hand_id"], payload["player_key"]),
            ).fetchone()
            if row is None:
                raise ValueError("Settlement player key does not belong to this hand.")
            return
        rows = self._execute(
            """
            SELECT player_key
            FROM hand_players
            WHERE hand_id = ? AND player_name = ?
            """,
            (payload["hand_id"], payload["player_name"]),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("A settlement entry must resolve to exactly one player in the hand.")
        payload["player_key"] = rows[0]["player_key"]

    def fetch_reviews_by_hand(self, hand_id: int) -> list[HandReview]:
        rows = self._execute(
            "SELECT * FROM hand_reviews WHERE hand_id = ? ORDER BY created_at DESC, id DESC",
            (hand_id,),
        ).fetchall()
        return [_review_from_row(row) for row in rows]

    def create_coaching_response(self, response: CoachingResponse) -> CoachingResponse:
        payload = response.model_dump()
        cursor = self._execute(
            """
            INSERT INTO coaching_reviews (
                provider_name, model_name, raw_prompt, raw_response, review_type,
                safety_mode, hand_id, session_id, parsed_sections, is_stale,
                stale_reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["provider_name"],
                payload["model_name"],
                payload["raw_prompt"],
                payload["raw_response"],
                payload["review_type"],
                payload["safety_mode"],
                payload["hand_id"],
                payload["session_id"],
                _serialize_json(payload["parsed_sections"]),
                int(payload["is_stale"]),
                payload["stale_reason"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return response.model_copy(update={"id": cursor.lastrowid})

    def fetch_coaching_reviews_by_hand(self, hand_id: int) -> list[CoachingResponse]:
        rows = self._execute(
            """
            SELECT * FROM coaching_reviews
            WHERE hand_id = ? AND review_type = 'hand'
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_coaching_response_from_row(row) for row in rows]

    def fetch_coaching_reviews_by_session(self, session_id: int) -> list[CoachingResponse]:
        rows = self._execute(
            """
            SELECT * FROM coaching_reviews
            WHERE session_id = ? AND review_type = 'session'
            ORDER BY created_at DESC, id DESC
            """,
            (session_id,),
        ).fetchall()
        return [_coaching_response_from_row(row) for row in rows]

    def create_solver_range_profile(
        self, profile: SolverRangeProfile
    ) -> SolverRangeProfile:
        payload = profile.model_dump()
        cursor = self._execute(
            """
            INSERT INTO solver_range_profiles (
                name, notation, table_size, position, scenario, pot_type,
                stack_bb, description, source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"].strip(),
                payload["notation"].strip(),
                payload["table_size"],
                payload["position"].strip(),
                payload["scenario"].strip(),
                payload["pot_type"].strip(),
                payload["stack_bb"],
                payload["description"].strip(),
                payload["source"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        return profile.model_copy(update={"id": cursor.lastrowid})

    def update_solver_range_profile(
        self, profile: SolverRangeProfile
    ) -> SolverRangeProfile:
        if profile.id is None:
            raise ValueError("Cannot update a solver range profile without an id.")
        payload = profile.model_dump()
        cursor = self._execute(
            """
            UPDATE solver_range_profiles
            SET name = ?, notation = ?, table_size = ?, position = ?,
                scenario = ?, pot_type = ?, stack_bb = ?, description = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                payload["name"].strip(),
                payload["notation"].strip(),
                payload["table_size"],
                payload["position"].strip(),
                payload["scenario"].strip(),
                payload["pot_type"].strip(),
                payload["stack_bb"],
                payload["description"].strip(),
                _serialize_datetime(payload["updated_at"]),
                payload["id"],
            ),
        )
        self._commit()
        if cursor.rowcount != 1:
            raise ValueError("Solver range profile not found.")
        return profile

    def fetch_solver_range_profiles(self) -> list[SolverRangeProfile]:
        rows = self._execute(
            "SELECT * FROM solver_range_profiles ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
        return [_solver_range_profile_from_row(row) for row in rows]

    def delete_solver_range_profile(self, profile_id: int) -> None:
        self._execute("DELETE FROM solver_range_profiles WHERE id = ?", (profile_id,))
        self._commit()

    def create_solver_run(self, run: SolverRun) -> SolverRun:
        payload = run.model_dump()
        cursor = self._execute(
            """
            INSERT INTO solver_runs (
                hand_id, status, backend_name, backend_version, input_hash,
                spot, range_ip, range_oop, assumptions, evidence,
                command_path, result_path, log_path, exploitability_pct,
                runtime_seconds, error_message, pid, heartbeat_at, created_at,
                started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["status"],
                payload["backend_name"],
                payload["backend_version"],
                payload["input_hash"],
                _serialize_json(payload["spot"]),
                _serialize_json(payload["range_ip"]),
                _serialize_json(payload["range_oop"]),
                _serialize_json(payload["assumptions"]),
                _serialize_json(payload["evidence"]),
                payload["command_path"],
                payload["result_path"],
                payload["log_path"],
                payload["exploitability_pct"],
                payload["runtime_seconds"],
                payload["error_message"],
                payload["pid"],
                _serialize_optional_datetime(payload["heartbeat_at"]),
                _serialize_datetime(payload["created_at"]),
                _serialize_optional_datetime(payload["started_at"]),
                _serialize_optional_datetime(payload["completed_at"]),
            ),
        )
        self._commit()
        return run.model_copy(update={"id": cursor.lastrowid})

    def update_solver_run(
        self,
        run_id: int,
        *,
        expected_statuses: tuple[str, ...] | None = None,
        **changes: object,
    ) -> SolverRun:
        allowed = {
            "status",
            "backend_version",
            "evidence",
            "exploitability_pct",
            "runtime_seconds",
            "error_message",
            "pid",
            "command_path",
            "result_path",
            "log_path",
            "heartbeat_at",
            "started_at",
            "completed_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported solver-run fields: {', '.join(sorted(unknown))}")
        if not changes:
            saved = self.fetch_solver_run(run_id)
            if saved is None:
                raise ValueError("Solver run not found.")
            return saved
        assignments: list[str] = []
        values: list[object] = []
        for key, value in changes.items():
            assignments.append(f"{key} = ?")
            if key == "evidence":
                values.append(_serialize_json(value))
            elif key in {"heartbeat_at", "started_at", "completed_at"}:
                values.append(_serialize_optional_datetime(value))  # type: ignore[arg-type]
            else:
                values.append(value)
        values.append(run_id)
        status_clause = ""
        if expected_statuses:
            placeholders = ", ".join("?" for _ in expected_statuses)
            status_clause = f" AND status IN ({placeholders})"
            values.extend(expected_statuses)
        cursor = self._execute(
            f"UPDATE solver_runs SET {', '.join(assignments)} "
            f"WHERE id = ?{status_clause}",
            tuple(values),
        )
        self._commit()
        if cursor.rowcount != 1:
            if expected_statuses:
                saved = self.fetch_solver_run(run_id)
                if saved is not None:
                    return saved
            raise ValueError("Solver run not found.")
        saved = self.fetch_solver_run(run_id)
        if saved is None:
            raise RuntimeError("Updated solver run could not be reloaded.")
        return saved

    def fetch_solver_run(self, run_id: int) -> SolverRun | None:
        row = self._execute("SELECT * FROM solver_runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else _solver_run_from_row(row)

    def fetch_solver_runs_by_hand(self, hand_id: int) -> list[SolverRun]:
        rows = self._execute(
            """
            SELECT * FROM solver_runs
            WHERE hand_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_solver_run_from_row(row) for row in rows]

    def fetch_cached_solver_run(self, input_hash: str) -> SolverRun | None:
        row = self._execute(
            """
            SELECT * FROM solver_runs
            WHERE input_hash = ? AND status = 'completed'
            ORDER BY completed_at DESC, id DESC
            LIMIT 1
            """,
            (input_hash,),
        ).fetchone()
        return None if row is None else _solver_run_from_row(row)

    def fetch_active_solver_runs(self) -> list[SolverRun]:
        rows = self._execute(
            """
            SELECT * FROM solver_runs
            WHERE status IN ('queued', 'running', 'cancelling')
            ORDER BY created_at, id
            """
        ).fetchall()
        return [_solver_run_from_row(row) for row in rows]

    def create_video(self, video: VideoRecord) -> VideoRecord:
        payload = video.model_dump()
        cursor = self._execute(
            """
            INSERT INTO videos (
                session_id, original_filename, stored_path, file_size_bytes,
                duration_seconds, fps, width, height, frame_count, uploaded_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["original_filename"],
                payload["stored_path"],
                payload["file_size_bytes"],
                payload["duration_seconds"],
                payload["fps"],
                payload["width"],
                payload["height"],
                payload["frame_count"],
                _serialize_datetime(payload["uploaded_at"]),
                payload["notes"],
            ),
        )
        self._commit()
        return video.model_copy(update={"id": cursor.lastrowid})

    def update_video_metadata(
        self,
        video_id: int,
        *,
        duration_seconds: float | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        frame_count: int | None = None,
    ) -> None:
        self._execute(
            """
            UPDATE videos
            SET duration_seconds = ?, fps = ?, width = ?, height = ?, frame_count = ?
            WHERE id = ?
            """,
            (duration_seconds, fps, width, height, frame_count, video_id),
        )
        self._commit()

    def update_video_session(self, video_id: int, session_id: int | None) -> VideoRecord:
        """Attach a stored video to a session, move it, or leave it unassigned."""

        if self.fetch_video(video_id) is None:
            raise ValueError("Video not found.")
        if session_id is not None and self.fetch_session(session_id) is None:
            raise ValueError("Target session not found.")
        self._execute(
            "UPDATE videos SET session_id = ? WHERE id = ?",
            (session_id, video_id),
        )
        self._commit()
        updated = self.fetch_video(video_id)
        if updated is None:
            raise RuntimeError("Updated video could not be reloaded.")
        return updated

    def fetch_video(self, video_id: int) -> VideoRecord | None:
        row = self._execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return None if row is None else _video_from_row(row)

    def fetch_videos(self, session_id: int | None = None) -> list[VideoRecord]:
        if session_id is None:
            rows = self._execute(
                "SELECT * FROM videos ORDER BY uploaded_at DESC, id DESC"
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM videos WHERE session_id = ? ORDER BY uploaded_at DESC, id DESC",
                (session_id,),
            ).fetchall()
        return [_video_from_row(row) for row in rows]

    def create_processing_job(self, job: ProcessingJob) -> ProcessingJob:
        payload = job.model_dump()
        cursor = self._execute(
            """
            INSERT INTO processing_jobs (
                job_type, status, video_id, progress_percent, message, error_message,
                pid, heartbeat_at, created_at, started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["job_type"],
                payload["status"],
                payload["video_id"],
                payload["progress_percent"],
                payload["message"],
                payload["error_message"],
                payload["pid"],
                _serialize_optional_datetime(payload["heartbeat_at"]),
                _serialize_datetime(payload["created_at"]),
                _serialize_optional_datetime(payload["started_at"]),
                _serialize_optional_datetime(payload["completed_at"]),
            ),
        )
        self._commit()
        return job.model_copy(update={"id": cursor.lastrowid})

    def update_processing_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress_percent: float | None = None,
        message: str | None = None,
        error_message: str | None = None,
        pid: int | None = None,
        heartbeat_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        current = self.fetch_processing_job(job_id)
        if current is None:
            raise ValueError(f"Processing job not found: {job_id}")
        self._execute(
            """
            UPDATE processing_jobs
            SET status = ?, progress_percent = ?, message = ?, error_message = ?,
                pid = ?, heartbeat_at = ?, started_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                status or current.status,
                current.progress_percent if progress_percent is None else progress_percent,
                current.message if message is None else message,
                current.error_message if error_message is None else error_message,
                current.pid if pid is None else pid,
                _serialize_optional_datetime(
                    heartbeat_at if heartbeat_at is not None else current.heartbeat_at
                ),
                _serialize_optional_datetime(
                    started_at if started_at is not None else current.started_at
                ),
                _serialize_optional_datetime(
                    completed_at if completed_at is not None else current.completed_at
                ),
                job_id,
            ),
        )
        self._commit()

    def fetch_processing_job(self, job_id: int) -> ProcessingJob | None:
        row = self._execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else _processing_job_from_row(row)

    def fetch_jobs_by_video(self, video_id: int) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE video_id = ? ORDER BY created_at DESC, id DESC",
            (video_id,),
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_recent_jobs(self, limit: int = 20) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_running_jobs(self) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE status = 'running' ORDER BY created_at, id"
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_active_jobs(self) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE status IN ('queued', 'running') "
            "ORDER BY created_at, id"
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def create_extracted_frame(self, frame: ExtractedFrame) -> ExtractedFrame:
        payload = frame.model_dump()
        cursor = self._execute(
            """
            INSERT OR IGNORE INTO extracted_frames (
                video_id, job_id, timestamp_seconds, frame_index, image_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["video_id"],
                payload["job_id"],
                payload["timestamp_seconds"],
                payload["frame_index"],
                payload["image_path"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        frame_id = (
            cursor.lastrowid
            or self._execute(
                """
            SELECT id FROM extracted_frames
            WHERE video_id = ? AND frame_index = ? AND image_path = ?
            """,
                (payload["video_id"], payload["frame_index"], payload["image_path"]),
            ).fetchone()["id"]
        )
        return frame.model_copy(update={"id": frame_id})

    def fetch_frames_by_video(self, video_id: int) -> list[ExtractedFrame]:
        rows = self._execute(
            """
            SELECT * FROM extracted_frames
            WHERE video_id = ?
            ORDER BY timestamp_seconds, frame_index, id
            """,
            (video_id,),
        ).fetchall()
        return [_extracted_frame_from_row(row) for row in rows]

    def fetch_extracted_frame(self, frame_id: int) -> ExtractedFrame | None:
        row = self._execute("SELECT * FROM extracted_frames WHERE id = ?", (frame_id,)).fetchone()
        return None if row is None else _extracted_frame_from_row(row)

    def delete_frame_records_by_video(self, video_id: int) -> None:
        self._execute("DELETE FROM extracted_frames WHERE video_id = ?", (video_id,))
        self._commit()

    def upsert_reconstruction_frame_review(
        self, review: ReconstructionFrameReview
    ) -> ReconstructionFrameReview:
        payload = review.model_dump()
        self._execute(
            """
            INSERT INTO reconstruction_frame_reviews (
                job_id, hand_number, source_image, timestamp_seconds, status,
                issue_types, notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, hand_number, source_image) DO UPDATE SET
                timestamp_seconds = excluded.timestamp_seconds,
                status = excluded.status,
                issue_types = excluded.issue_types,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                payload["job_id"],
                payload["hand_number"],
                payload["source_image"],
                payload["timestamp_seconds"],
                payload["status"],
                _serialize_json(payload["issue_types"]),
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        saved = self._execute(
            """
            SELECT * FROM reconstruction_frame_reviews
            WHERE job_id = ? AND hand_number = ? AND source_image = ?
            """,
            (review.job_id, review.hand_number, review.source_image),
        ).fetchone()
        if saved is None:
            raise RuntimeError("The frame review could not be reloaded.")
        return _reconstruction_frame_review_from_row(saved)

    def fetch_reconstruction_frame_reviews(
        self, job_id: int, hand_number: int | None = None
    ) -> list[ReconstructionFrameReview]:
        if hand_number is None:
            rows = self._execute(
                """
                SELECT * FROM reconstruction_frame_reviews
                WHERE job_id = ?
                ORDER BY hand_number, timestamp_seconds, id
                """,
                (job_id,),
            ).fetchall()
        else:
            rows = self._execute(
                """
                SELECT * FROM reconstruction_frame_reviews
                WHERE job_id = ? AND hand_number = ?
                ORDER BY timestamp_seconds, id
                """,
                (job_id, hand_number),
            ).fetchall()
        return [_reconstruction_frame_review_from_row(row) for row in rows]

    def create_roi_profile(self, profile: ROIProfile) -> ROIProfile:
        payload = profile.model_dump()
        cursor = self._execute(
            """
            INSERT INTO roi_profiles (
                name, description, platform, table_layout, video_width, video_height,
                created_at, updated_at, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"],
                payload["description"],
                payload["platform"],
                payload["table_layout"],
                payload["video_width"],
                payload["video_height"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
                int(payload["is_active"]),
            ),
        )
        self._commit()
        saved = profile.model_copy(update={"id": cursor.lastrowid})
        if saved.is_active and saved.id is not None:
            self.mark_roi_profile_active(saved.id)
            saved = saved.model_copy(update={"is_active": True})
        return saved

    def update_roi_profile(self, profile: ROIProfile) -> ROIProfile:
        if profile.id is None:
            raise ValueError("Cannot update an ROI profile without an id.")
        payload = profile.model_dump()
        self._execute(
            """
            UPDATE roi_profiles
            SET name = ?, description = ?, platform = ?, table_layout = ?,
                video_width = ?, video_height = ?, updated_at = ?, is_active = ?
            WHERE id = ?
            """,
            (
                payload["name"],
                payload["description"],
                payload["platform"],
                payload["table_layout"],
                payload["video_width"],
                payload["video_height"],
                _serialize_datetime(payload["updated_at"]),
                int(payload["is_active"]),
                payload["id"],
            ),
        )
        self._commit()
        if profile.is_active:
            self.mark_roi_profile_active(profile.id)
        return profile

    def fetch_roi_profile(self, profile_id: int) -> ROIProfile | None:
        row = self._execute("SELECT * FROM roi_profiles WHERE id = ?", (profile_id,)).fetchone()
        return None if row is None else _roi_profile_from_row(row)

    def fetch_roi_profiles(self) -> list[ROIProfile]:
        rows = self._execute(
            "SELECT * FROM roi_profiles ORDER BY is_active DESC, updated_at DESC, id DESC"
        ).fetchall()
        return [_roi_profile_from_row(row) for row in rows]

    def mark_roi_profile_active(self, profile_id: int) -> None:
        if self.fetch_roi_profile(profile_id) is None:
            raise ValueError(f"ROI profile not found: {profile_id}")
        self._execute("UPDATE roi_profiles SET is_active = 0")
        self._execute(
            "UPDATE roi_profiles SET is_active = 1, updated_at = ? WHERE id = ?",
            (_serialize_datetime(datetime.now().astimezone()), profile_id),
        )
        self._commit()

    def delete_roi_profile(self, profile_id: int) -> None:
        self._execute("DELETE FROM roi_profiles WHERE id = ?", (profile_id,))
        self._commit()

    def create_roi_region(self, region: ROIRegion) -> ROIRegion:
        self._validate_roi_region_for_profile(region)
        payload = region.model_dump()
        cursor = self._execute(
            """
            INSERT INTO roi_regions (
                profile_id, roi_key, roi_type, label, x, y, width, height,
                seat_index, card_index, notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["profile_id"],
                payload["roi_key"],
                payload["roi_type"],
                payload["label"],
                payload["x"],
                payload["y"],
                payload["width"],
                payload["height"],
                payload["seat_index"],
                payload["card_index"],
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        return region.model_copy(update={"id": cursor.lastrowid})

    def update_roi_region(self, region: ROIRegion) -> ROIRegion:
        if region.id is None:
            raise ValueError("Cannot update an ROI region without an id.")
        self._validate_roi_region_for_profile(region)
        payload = region.model_dump()
        self._execute(
            """
            UPDATE roi_regions
            SET roi_key = ?, roi_type = ?, label = ?, x = ?, y = ?, width = ?,
                height = ?, seat_index = ?, card_index = ?, notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                payload["roi_key"],
                payload["roi_type"],
                payload["label"],
                payload["x"],
                payload["y"],
                payload["width"],
                payload["height"],
                payload["seat_index"],
                payload["card_index"],
                payload["notes"],
                _serialize_datetime(payload["updated_at"]),
                payload["id"],
            ),
        )
        self._commit()
        return region

    def delete_roi_region(self, region_id: int) -> None:
        self._execute("DELETE FROM roi_regions WHERE id = ?", (region_id,))
        self._commit()

    def fetch_roi_regions_by_profile(self, profile_id: int) -> list[ROIRegion]:
        rows = self._execute(
            """
            SELECT * FROM roi_regions
            WHERE profile_id = ?
            ORDER BY roi_type, seat_index, card_index, roi_key, id
            """,
            (profile_id,),
        ).fetchall()
        return [_roi_region_from_row(row) for row in rows]

    def _validate_roi_region_for_profile(self, region: ROIRegion) -> None:
        profile = self.fetch_roi_profile(region.profile_id)
        if profile is None:
            raise ValueError(f"ROI profile not found: {region.profile_id}")
        validate_roi_bounds(
            region,
            image_width=profile.video_width,
            image_height=profile.video_height,
        )

    def schema_version(self) -> int:
        try:
            row = self._execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0  # fresh database: schema_metadata does not exist yet
        return 0 if row is None else int(row["value"])

    def delete_session(self, session_id: int) -> None:
        """Delete a session; hands, actions, reviews cascade. Videos are kept (unlinked)."""
        self._execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._commit()

    def delete_video(self, video_id: int) -> None:
        """Delete a video row; jobs and extracted frames cascade. Files are the caller's job."""
        self._execute("DELETE FROM videos WHERE id = ?", (video_id,))
        self._commit()


def _migrate_to_v6(db: PokerDatabase) -> None:
    db._ensure_column("processing_jobs", "pid", "INTEGER")
    db._ensure_column("processing_jobs", "heartbeat_at", "TEXT")


def _migrate_to_v7(db: PokerDatabase) -> None:
    db._ensure_column("hand_players", "player_key", "TEXT")
    db._ensure_column("hand_players", "seat_index", "INTEGER")
    db._ensure_column("actions", "player_key", "TEXT")
    db._ensure_column("actions", "amount_semantics", "TEXT NOT NULL DEFAULT 'unknown'")
    db._ensure_column("actions", "forced_bet_type", "TEXT")
    db._ensure_column("actions", "is_live_post", "INTEGER")
    db._execute(
        """
        UPDATE hand_players
        SET player_key = 'hand:' || hand_id || ':player:' || id
        WHERE player_key IS NULL OR TRIM(player_key) = ''
        """
    )
    db._execute(
        """
        UPDATE actions
        SET player_key = (
            SELECT hp.player_key
            FROM hand_players AS hp
            WHERE hp.hand_id = actions.hand_id
              AND hp.player_name = actions.player_name
              AND hp.position = actions.position
              AND (
                  SELECT COUNT(*)
                  FROM hand_players AS exact_hp
                  WHERE exact_hp.hand_id = actions.hand_id
                    AND exact_hp.player_name = actions.player_name
                    AND exact_hp.position = actions.position
              ) = 1
            ORDER BY hp.id
            LIMIT 1
        )
        WHERE player_key IS NULL
        """
    )
    db._execute(
        """
        UPDATE actions
        SET player_key = (
            SELECT hp.player_key
            FROM hand_players AS hp
            WHERE hp.hand_id = actions.hand_id
              AND hp.player_name = actions.player_name
              AND (
                  SELECT COUNT(*)
                  FROM hand_players AS name_hp
                  WHERE name_hp.hand_id = actions.hand_id
                    AND name_hp.player_name = actions.player_name
              ) = 1
            ORDER BY hp.id
            LIMIT 1
        )
        WHERE player_key IS NULL
        """
    )
    db._connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS hand_settlements (
            hand_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'unsettled',
            dead_money REAL NOT NULL DEFAULT 0,
            rake_rate REAL NOT NULL DEFAULT 0,
            rake_cap REAL,
            rake_rounding_unit REAL NOT NULL DEFAULT 0.01,
            no_flop_no_drop INTEGER NOT NULL DEFAULT 0,
            gross_pot REAL,
            rake_amount REAL,
            net_pot REAL,
            is_balanced INTEGER NOT NULL DEFAULT 0,
            warnings TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS settlement_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            entry_type TEXT NOT NULL,
            pot_index INTEGER,
            player_key TEXT,
            player_name TEXT NOT NULL,
            amount REAL,
            entry_order INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_hand_players_hand_key
            ON hand_players(hand_id, player_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_hand_players_hand_seat
            ON hand_players(hand_id, seat_index)
            WHERE seat_index IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_actions_player_key
            ON actions(hand_id, player_key);
        CREATE INDEX IF NOT EXISTS idx_settlement_entries_hand
            ON settlement_entries(hand_id, entry_type, pot_index, entry_order);
        """
    )


def _migrate_to_v8(db: PokerDatabase) -> None:
    """Make persisted action order deterministic before enforcing uniqueness."""
    db._execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY hand_id, street
                    ORDER BY action_index, id
                ) AS normalized_index
            FROM actions
        )
        UPDATE actions
        SET action_index = (
            SELECT normalized_index
            FROM ranked
            WHERE ranked.id = actions.id
        )
        """
    )
    db._execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_hand_street_order
        ON actions(hand_id, street, action_index)
        """
    )


def _migrate_to_v9(db: PokerDatabase) -> None:
    db._connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS reconstruction_frame_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            hand_number INTEGER NOT NULL,
            source_image TEXT NOT NULL,
            timestamp_seconds REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'unreviewed',
            issue_types TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,
            UNIQUE(job_id, hand_number, source_image)
        );
        CREATE INDEX IF NOT EXISTS idx_reconstruction_reviews_job_hand
            ON reconstruction_frame_reviews(job_id, hand_number);
        """
    )


def _migrate_to_v10(db: PokerDatabase) -> None:
    """Retain correction evidence and make superseded coaching explicit."""

    db._ensure_column("hand_reviews", "is_stale", "INTEGER NOT NULL DEFAULT 0")
    db._ensure_column("hand_reviews", "stale_reason", "TEXT NOT NULL DEFAULT ''")
    db._ensure_column("coaching_reviews", "is_stale", "INTEGER NOT NULL DEFAULT 0")
    db._ensure_column("coaching_reviews", "stale_reason", "TEXT NOT NULL DEFAULT ''")
    db._connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS hand_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            correction_type TEXT NOT NULL,
            before_state TEXT NOT NULL DEFAULT '{}',
            after_state TEXT NOT NULL DEFAULT '{}',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hand_corrections_hand_id
            ON hand_corrections(hand_id, created_at, id);
        """
    )


def _migrate_to_v11(db: PokerDatabase) -> None:
    """Add auditable external-solver runs and reusable user range profiles."""

    db._connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS solver_range_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            notation TEXT NOT NULL,
            table_size INTEGER,
            position TEXT NOT NULL DEFAULT '',
            scenario TEXT NOT NULL DEFAULT '',
            pot_type TEXT NOT NULL DEFAULT '',
            stack_bb REAL,
            description TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS solver_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            backend_name TEXT NOT NULL DEFAULT 'texassolver',
            backend_version TEXT NOT NULL DEFAULT '',
            input_hash TEXT NOT NULL,
            spot TEXT NOT NULL DEFAULT '{}',
            range_ip TEXT NOT NULL DEFAULT '{}',
            range_oop TEXT NOT NULL DEFAULT '{}',
            assumptions TEXT NOT NULL DEFAULT '[]',
            evidence TEXT NOT NULL DEFAULT '{}',
            command_path TEXT NOT NULL DEFAULT '',
            result_path TEXT NOT NULL DEFAULT '',
            log_path TEXT NOT NULL DEFAULT '',
            exploitability_pct REAL,
            runtime_seconds REAL,
            error_message TEXT NOT NULL DEFAULT '',
            pid INTEGER,
            heartbeat_at TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_solver_runs_hand
            ON solver_runs(hand_id, created_at, id);
        CREATE INDEX IF NOT EXISTS idx_solver_runs_cache
            ON solver_runs(input_hash, status);
        CREATE INDEX IF NOT EXISTS idx_solver_runs_status
            ON solver_runs(status, created_at);
        """
    )


def _migrate_to_v12(db: PokerDatabase) -> None:
    """Add a persistent queue for hands that need future debugging."""

    db._connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS hand_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            issue_types TEXT NOT NULL DEFAULT '[]',
            description TEXT NOT NULL,
            evidence_snapshot TEXT NOT NULL DEFAULT '{}',
            resolution_notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hand_issues_status
            ON hand_issues(status, created_at, id);
        CREATE INDEX IF NOT EXISTS idx_hand_issues_hand
            ON hand_issues(hand_id, status, created_at, id);
        """
    )


# Versioned migrations run in order and refuse databases written by newer apps.
_MIGRATIONS: dict[int, Callable[[PokerDatabase], None]] = {
    6: _migrate_to_v6,
    7: _migrate_to_v7,
    8: _migrate_to_v8,
    9: _migrate_to_v9,
    10: _migrate_to_v10,
    11: _migrate_to_v11,
    12: _migrate_to_v12,
}


def _serialize_date(value: date) -> str:
    return value.isoformat()


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def _serialize_optional_datetime(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _serialize_json(value: Any) -> str:
    return json.dumps(value)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _session_from_row(row: sqlite3.Row) -> Session:
    data = _row_dict(row)
    data["date_played"] = _parse_date(data["date_played"])
    data["created_at"] = _parse_datetime(data["created_at"])
    return Session(**data)


def _hand_from_row(row: sqlite3.Row) -> Hand:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["tags"] = _parse_json_list(data.get("tags", "[]"))
    return Hand(**data)


def _hand_player_from_row(row: sqlite3.Row) -> HandPlayer:
    data = _row_dict(row)
    data["is_hero"] = bool(data["is_hero"])
    return HandPlayer(**data)


def _action_from_row(row: sqlite3.Row) -> Action:
    data = _row_dict(row)
    if data.get("is_live_post") is not None:
        data["is_live_post"] = bool(data["is_live_post"])
    return Action(**data)


def _hand_settlement_from_row(row: sqlite3.Row) -> HandSettlement:
    data = _row_dict(row)
    data["no_flop_no_drop"] = bool(data["no_flop_no_drop"])
    data["is_balanced"] = bool(data["is_balanced"])
    data["warnings"] = _parse_json_list(data.get("warnings", "[]"))
    data["created_at"] = _parse_datetime(data["created_at"])
    data["updated_at"] = _parse_datetime(data["updated_at"])
    return HandSettlement(**data)


def _settlement_entry_from_row(row: sqlite3.Row) -> SettlementEntry:
    return SettlementEntry(**_row_dict(row))


def _review_from_row(row: sqlite3.Row) -> HandReview:
    data = _row_dict(row)
    data["is_stale"] = bool(data.get("is_stale", 0))
    data["created_at"] = _parse_datetime(data["created_at"])
    return HandReview(**data)


def _hand_correction_from_row(row: sqlite3.Row) -> HandCorrection:
    data = _row_dict(row)
    data["before_state"] = _parse_json_dict(data.get("before_state", "{}"))
    data["after_state"] = _parse_json_dict(data.get("after_state", "{}"))
    data["created_at"] = _parse_datetime(data["created_at"])
    return HandCorrection(**data)


def _hand_issue_from_row(row: sqlite3.Row) -> HandIssue:
    data = _row_dict(row)
    data["issue_types"] = _parse_json_list(data.get("issue_types", "[]"))
    data["evidence_snapshot"] = _parse_json_object(
        data.get("evidence_snapshot", "{}"), {}
    )
    data["created_at"] = _parse_datetime(data["created_at"])
    data["updated_at"] = _parse_datetime(data["updated_at"])
    data["resolved_at"] = _parse_optional_datetime(data.get("resolved_at"))
    return HandIssue(**data)


def _coaching_response_from_row(row: sqlite3.Row) -> CoachingResponse:
    data = _row_dict(row)
    data["is_stale"] = bool(data.get("is_stale", 0))
    data["created_at"] = _parse_datetime(data["created_at"])
    data["parsed_sections"] = _parse_json_dict(data.get("parsed_sections", "{}"))
    return CoachingResponse(**data)


def _solver_range_profile_from_row(row: sqlite3.Row) -> SolverRangeProfile:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["updated_at"] = _parse_datetime(data["updated_at"])
    return SolverRangeProfile(**data)


def _solver_run_from_row(row: sqlite3.Row) -> SolverRun:
    data = _row_dict(row)
    for key in ("spot", "range_ip", "range_oop", "evidence"):
        data[key] = _parse_json_object(data.get(key, "{}"), {})
    data["assumptions"] = _parse_json_object(data.get("assumptions", "[]"), [])
    data["created_at"] = _parse_datetime(data["created_at"])
    data["started_at"] = _parse_optional_datetime(data.get("started_at"))
    data["completed_at"] = _parse_optional_datetime(data.get("completed_at"))
    data["heartbeat_at"] = _parse_optional_datetime(data.get("heartbeat_at"))
    return SolverRun(**data)


def _video_from_row(row: sqlite3.Row) -> VideoRecord:
    data = _row_dict(row)
    data["uploaded_at"] = _parse_datetime(data["uploaded_at"])
    return VideoRecord(**data)


def _processing_job_from_row(row: sqlite3.Row) -> ProcessingJob:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["started_at"] = _parse_optional_datetime(data["started_at"])
    data["completed_at"] = _parse_optional_datetime(data["completed_at"])
    data["heartbeat_at"] = _parse_optional_datetime(data.get("heartbeat_at"))
    return ProcessingJob(**data)


def _extracted_frame_from_row(row: sqlite3.Row) -> ExtractedFrame:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    return ExtractedFrame(**data)


def _reconstruction_frame_review_from_row(
    row: sqlite3.Row,
) -> ReconstructionFrameReview:
    data = _row_dict(row)
    data["issue_types"] = _parse_json_list(data.get("issue_types", "[]"))
    data["created_at"] = _parse_datetime(data["created_at"])
    data["updated_at"] = _parse_datetime(data["updated_at"])
    return ReconstructionFrameReview(**data)


def _roi_profile_from_row(row: sqlite3.Row) -> ROIProfile:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["updated_at"] = _parse_datetime(data["updated_at"])
    data["is_active"] = bool(data["is_active"])
    return ROIProfile(**data)


def _roi_region_from_row(row: sqlite3.Row) -> ROIRegion:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["updated_at"] = _parse_datetime(data["updated_at"])
    return ROIRegion(**data)


def _parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _parse_json_dict(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def _parse_json_object(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# TODO: Add separate repository modules for CV/OCR-derived hand imports later.
# TODO: Add migration management before this grows beyond the first local schema.
