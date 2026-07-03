from __future__ import annotations

import sqlite3
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from poker_tracker.models import (
    Action,
    CoachingResponse,
    ExtractedFrame,
    Hand,
    HandPlayer,
    HandReview,
    ProcessingJob,
    ROIProfile,
    ROIRegion,
    Session,
    VideoRecord,
)
from poker_tracker.roi import validate_roi_bounds


DEFAULT_DB_PATH = "poker_tracker.db"
SCHEMA_VERSION = 5


class PokerDatabase:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self._connection.close()

    def init_db(self) -> None:
        """Create or migrate the local SQLite schema."""
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
                street TEXT NOT NULL,
                action_index INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL DEFAULT '',
                action_type TEXT NOT NULL,
                amount REAL,
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
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
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
            CREATE INDEX IF NOT EXISTS idx_videos_session_id ON videos(session_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON processing_jobs(video_id);
            CREATE INDEX IF NOT EXISTS idx_frames_video_id ON extracted_frames(video_id);
            CREATE INDEX IF NOT EXISTS idx_roi_profiles_active ON roi_profiles(is_active);
            CREATE INDEX IF NOT EXISTS idx_roi_regions_profile_id ON roi_regions(profile_id);
            """
        )
        self._ensure_column("hands", "game_type", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hands", "blinds_antes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hands", "table_size", "INTEGER")
        self._ensure_column("hands", "effective_stack", "REAL")
        self._ensure_column("hands", "review_status", "TEXT NOT NULL DEFAULT 'unreviewed'")
        self._ensure_column("hands", "confidence_score", "REAL")
        self._ensure_column("hands", "source_type", "TEXT NOT NULL DEFAULT 'manual'")
        self._ensure_column("hands", "tags", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("hand_reviews", "ev_math_notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(
            "hand_reviews", "next_review_question", "TEXT NOT NULL DEFAULT ''"
        )
        self._ensure_column("hand_reviews", "notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("coaching_reviews", "safety_mode", "TEXT NOT NULL DEFAULT 'post_session_only'")
        self._ensure_column("coaching_reviews", "parsed_sections", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("videos", "frame_count", "INTEGER")
        self._ensure_column("roi_profiles", "table_layout", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("roi_profiles", "is_active", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("roi_regions", "notes", "TEXT NOT NULL DEFAULT ''")
        self._connection.execute(
            """
            INSERT INTO schema_metadata (key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(SCHEMA_VERSION),),
        )
        self._connection.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_spec: str) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            self._connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}"
            )

    def create_session(self, session: Session) -> Session:
        payload = session.model_dump()
        cursor = self._connection.execute(
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
        self._connection.commit()
        return session.model_copy(update={"id": cursor.lastrowid})

    def create_hand(self, hand: Hand) -> Hand:
        payload = hand.model_dump()
        cursor = self._connection.execute(
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
        self._connection.commit()
        return hand.model_copy(update={"id": cursor.lastrowid})

    def update_hand_status(self, hand_id: int, review_status: str) -> None:
        self._connection.execute(
            "UPDATE hands SET review_status = ? WHERE id = ?",
            (review_status, hand_id),
        )
        self._connection.commit()

    def delete_hand(self, hand_id: int) -> None:
        self._connection.execute("DELETE FROM hands WHERE id = ?", (hand_id,))
        self._connection.commit()

    def create_hand_player(self, player: HandPlayer) -> HandPlayer:
        payload = player.model_dump()
        cursor = self._connection.execute(
            """
            INSERT INTO hand_players (
                hand_id, player_name, position, starting_stack, is_hero, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["player_name"],
                payload["position"],
                payload["starting_stack"],
                int(payload["is_hero"]),
                payload["notes"],
            ),
        )
        self._connection.commit()
        return player.model_copy(update={"id": cursor.lastrowid})

    def create_action(self, action: Action) -> Action:
        payload = action.model_dump()
        action_index = payload["action_index"] or self.next_action_index(
            payload["hand_id"], payload["street"]
        )
        cursor = self._connection.execute(
            """
            INSERT INTO actions (
                hand_id, street, action_index, player_name, position, action_type,
                amount, pot_before, stack_before, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["street"],
                action_index,
                payload["player_name"],
                payload["position"],
                payload["action_type"],
                payload["amount"],
                payload["pot_before"],
                payload["stack_before"],
                payload["notes"],
            ),
        )
        self._connection.commit()
        return action.model_copy(update={"id": cursor.lastrowid, "action_index": action_index})

    def next_action_index(self, hand_id: int, street: str) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(action_index), 0) + 1 AS next_index
            FROM actions
            WHERE hand_id = ? AND street = ?
            """,
            (hand_id, street),
        ).fetchone()
        return int(row["next_index"])

    def update_action(self, action: Action) -> Action:
        if action.id is None:
            raise ValueError("Cannot update an action without an id.")
        payload = action.model_dump()
        self._connection.execute(
            """
            UPDATE actions
            SET street = ?, action_index = ?, player_name = ?, position = ?,
                action_type = ?, amount = ?, pot_before = ?, stack_before = ?, notes = ?
            WHERE id = ?
            """,
            (
                payload["street"],
                payload["action_index"] or self.next_action_index(
                    payload["hand_id"], payload["street"]
                ),
                payload["player_name"],
                payload["position"],
                payload["action_type"],
                payload["amount"],
                payload["pot_before"],
                payload["stack_before"],
                payload["notes"],
                payload["id"],
            ),
        )
        self._connection.commit()
        return action

    def delete_action(self, action_id: int) -> None:
        self._connection.execute("DELETE FROM actions WHERE id = ?", (action_id,))
        self._connection.commit()

    def create_hand_review(self, review: HandReview) -> HandReview:
        payload = review.model_dump()
        cursor = self._connection.execute(
            """
            INSERT INTO hand_reviews (
                hand_id, hand_summary, theory_coach, exploit_coach, ev_math_notes,
                study_lesson, next_review_question, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._connection.commit()
        return review.model_copy(update={"id": cursor.lastrowid})

    def fetch_sessions(self) -> list[Session]:
        rows = self._connection.execute(
            "SELECT * FROM sessions ORDER BY date_played DESC, id DESC"
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def fetch_hands_by_session(self, session_id: int) -> list[Hand]:
        rows = self._connection.execute(
            "SELECT * FROM hands WHERE session_id = ? ORDER BY hand_number, id",
            (session_id,),
        ).fetchall()
        return [_hand_from_row(row) for row in rows]

    def fetch_session(self, session_id: int) -> Session | None:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return None if row is None else _session_from_row(row)

    def fetch_hand(self, hand_id: int) -> Hand | None:
        row = self._connection.execute(
            "SELECT * FROM hands WHERE id = ?", (hand_id,)
        ).fetchone()
        return None if row is None else _hand_from_row(row)

    def fetch_actions_by_hand(self, hand_id: int) -> list[Action]:
        rows = self._connection.execute(
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
        rows = self._connection.execute(
            """
            SELECT * FROM hand_players
            WHERE hand_id = ?
            ORDER BY is_hero DESC, position, id
            """,
            (hand_id,),
        ).fetchall()
        return [_hand_player_from_row(row) for row in rows]

    def fetch_reviews_by_hand(self, hand_id: int) -> list[HandReview]:
        rows = self._connection.execute(
            "SELECT * FROM hand_reviews WHERE hand_id = ? ORDER BY created_at DESC, id DESC",
            (hand_id,),
        ).fetchall()
        return [_review_from_row(row) for row in rows]

    def create_coaching_response(self, response: CoachingResponse) -> CoachingResponse:
        payload = response.model_dump()
        cursor = self._connection.execute(
            """
            INSERT INTO coaching_reviews (
                provider_name, model_name, raw_prompt, raw_response, review_type,
                safety_mode, hand_id, session_id, parsed_sections, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._connection.commit()
        return response.model_copy(update={"id": cursor.lastrowid})

    def fetch_coaching_reviews_by_hand(self, hand_id: int) -> list[CoachingResponse]:
        rows = self._connection.execute(
            """
            SELECT * FROM coaching_reviews
            WHERE hand_id = ? AND review_type = 'hand'
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_coaching_response_from_row(row) for row in rows]

    def fetch_coaching_reviews_by_session(self, session_id: int) -> list[CoachingResponse]:
        rows = self._connection.execute(
            """
            SELECT * FROM coaching_reviews
            WHERE session_id = ? AND review_type = 'session'
            ORDER BY created_at DESC, id DESC
            """,
            (session_id,),
        ).fetchall()
        return [_coaching_response_from_row(row) for row in rows]

    def create_video(self, video: VideoRecord) -> VideoRecord:
        payload = video.model_dump()
        cursor = self._connection.execute(
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
        self._connection.commit()
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
        self._connection.execute(
            """
            UPDATE videos
            SET duration_seconds = ?, fps = ?, width = ?, height = ?, frame_count = ?
            WHERE id = ?
            """,
            (duration_seconds, fps, width, height, frame_count, video_id),
        )
        self._connection.commit()

    def fetch_video(self, video_id: int) -> VideoRecord | None:
        row = self._connection.execute(
            "SELECT * FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        return None if row is None else _video_from_row(row)

    def fetch_videos(self, session_id: int | None = None) -> list[VideoRecord]:
        if session_id is None:
            rows = self._connection.execute(
                "SELECT * FROM videos ORDER BY uploaded_at DESC, id DESC"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM videos WHERE session_id = ? ORDER BY uploaded_at DESC, id DESC",
                (session_id,),
            ).fetchall()
        return [_video_from_row(row) for row in rows]

    def create_processing_job(self, job: ProcessingJob) -> ProcessingJob:
        payload = job.model_dump()
        cursor = self._connection.execute(
            """
            INSERT INTO processing_jobs (
                job_type, status, video_id, progress_percent, message, error_message,
                created_at, started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["job_type"],
                payload["status"],
                payload["video_id"],
                payload["progress_percent"],
                payload["message"],
                payload["error_message"],
                _serialize_datetime(payload["created_at"]),
                _serialize_optional_datetime(payload["started_at"]),
                _serialize_optional_datetime(payload["completed_at"]),
            ),
        )
        self._connection.commit()
        return job.model_copy(update={"id": cursor.lastrowid})

    def update_processing_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress_percent: float | None = None,
        message: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        current = self.fetch_processing_job(job_id)
        if current is None:
            raise ValueError(f"Processing job not found: {job_id}")
        self._connection.execute(
            """
            UPDATE processing_jobs
            SET status = ?, progress_percent = ?, message = ?, error_message = ?,
                started_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                status or current.status,
                current.progress_percent if progress_percent is None else progress_percent,
                current.message if message is None else message,
                current.error_message if error_message is None else error_message,
                _serialize_optional_datetime(started_at if started_at is not None else current.started_at),
                _serialize_optional_datetime(
                    completed_at if completed_at is not None else current.completed_at
                ),
                job_id,
            ),
        )
        self._connection.commit()

    def fetch_processing_job(self, job_id: int) -> ProcessingJob | None:
        row = self._connection.execute(
            "SELECT * FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return None if row is None else _processing_job_from_row(row)

    def fetch_jobs_by_video(self, video_id: int) -> list[ProcessingJob]:
        rows = self._connection.execute(
            "SELECT * FROM processing_jobs WHERE video_id = ? ORDER BY created_at DESC, id DESC",
            (video_id,),
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_recent_jobs(self, limit: int = 20) -> list[ProcessingJob]:
        rows = self._connection.execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def create_extracted_frame(self, frame: ExtractedFrame) -> ExtractedFrame:
        payload = frame.model_dump()
        cursor = self._connection.execute(
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
        self._connection.commit()
        frame_id = cursor.lastrowid or self._connection.execute(
            """
            SELECT id FROM extracted_frames
            WHERE video_id = ? AND frame_index = ? AND image_path = ?
            """,
            (payload["video_id"], payload["frame_index"], payload["image_path"]),
        ).fetchone()["id"]
        return frame.model_copy(update={"id": frame_id})

    def fetch_frames_by_video(self, video_id: int) -> list[ExtractedFrame]:
        rows = self._connection.execute(
            """
            SELECT * FROM extracted_frames
            WHERE video_id = ?
            ORDER BY timestamp_seconds, frame_index, id
            """,
            (video_id,),
        ).fetchall()
        return [_extracted_frame_from_row(row) for row in rows]

    def fetch_extracted_frame(self, frame_id: int) -> ExtractedFrame | None:
        row = self._connection.execute(
            "SELECT * FROM extracted_frames WHERE id = ?", (frame_id,)
        ).fetchone()
        return None if row is None else _extracted_frame_from_row(row)

    def delete_frame_records_by_video(self, video_id: int) -> None:
        self._connection.execute("DELETE FROM extracted_frames WHERE video_id = ?", (video_id,))
        self._connection.commit()

    def create_roi_profile(self, profile: ROIProfile) -> ROIProfile:
        payload = profile.model_dump()
        cursor = self._connection.execute(
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
        self._connection.commit()
        saved = profile.model_copy(update={"id": cursor.lastrowid})
        if saved.is_active and saved.id is not None:
            self.mark_roi_profile_active(saved.id)
            saved = saved.model_copy(update={"is_active": True})
        return saved

    def update_roi_profile(self, profile: ROIProfile) -> ROIProfile:
        if profile.id is None:
            raise ValueError("Cannot update an ROI profile without an id.")
        payload = profile.model_dump()
        self._connection.execute(
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
        self._connection.commit()
        if profile.is_active:
            self.mark_roi_profile_active(profile.id)
        return profile

    def fetch_roi_profile(self, profile_id: int) -> ROIProfile | None:
        row = self._connection.execute(
            "SELECT * FROM roi_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return None if row is None else _roi_profile_from_row(row)

    def fetch_roi_profiles(self) -> list[ROIProfile]:
        rows = self._connection.execute(
            "SELECT * FROM roi_profiles ORDER BY is_active DESC, updated_at DESC, id DESC"
        ).fetchall()
        return [_roi_profile_from_row(row) for row in rows]

    def mark_roi_profile_active(self, profile_id: int) -> None:
        if self.fetch_roi_profile(profile_id) is None:
            raise ValueError(f"ROI profile not found: {profile_id}")
        self._connection.execute("UPDATE roi_profiles SET is_active = 0")
        self._connection.execute(
            "UPDATE roi_profiles SET is_active = 1, updated_at = ? WHERE id = ?",
            (_serialize_datetime(datetime.now().astimezone()), profile_id),
        )
        self._connection.commit()

    def delete_roi_profile(self, profile_id: int) -> None:
        self._connection.execute("DELETE FROM roi_profiles WHERE id = ?", (profile_id,))
        self._connection.commit()

    def create_roi_region(self, region: ROIRegion) -> ROIRegion:
        self._validate_roi_region_for_profile(region)
        payload = region.model_dump()
        cursor = self._connection.execute(
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
        self._connection.commit()
        return region.model_copy(update={"id": cursor.lastrowid})

    def update_roi_region(self, region: ROIRegion) -> ROIRegion:
        if region.id is None:
            raise ValueError("Cannot update an ROI region without an id.")
        self._validate_roi_region_for_profile(region)
        payload = region.model_dump()
        self._connection.execute(
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
        self._connection.commit()
        return region

    def delete_roi_region(self, region_id: int) -> None:
        self._connection.execute("DELETE FROM roi_regions WHERE id = ?", (region_id,))
        self._connection.commit()

    def fetch_roi_regions_by_profile(self, profile_id: int) -> list[ROIRegion]:
        rows = self._connection.execute(
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
        row = self._connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        return 0 if row is None else int(row["value"])


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
    return Action(**_row_dict(row))


def _review_from_row(row: sqlite3.Row) -> HandReview:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    return HandReview(**data)


def _coaching_response_from_row(row: sqlite3.Row) -> CoachingResponse:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["parsed_sections"] = _parse_json_dict(data.get("parsed_sections", "{}"))
    return CoachingResponse(**data)


def _video_from_row(row: sqlite3.Row) -> VideoRecord:
    data = _row_dict(row)
    data["uploaded_at"] = _parse_datetime(data["uploaded_at"])
    return VideoRecord(**data)


def _processing_job_from_row(row: sqlite3.Row) -> ProcessingJob:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    data["started_at"] = _parse_optional_datetime(data["started_at"])
    data["completed_at"] = _parse_optional_datetime(data["completed_at"])
    return ProcessingJob(**data)


def _extracted_frame_from_row(row: sqlite3.Row) -> ExtractedFrame:
    data = _row_dict(row)
    data["created_at"] = _parse_datetime(data["created_at"])
    return ExtractedFrame(**data)


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


# TODO: Add separate repository modules for CV/OCR-derived hand imports later.
# TODO: Add migration management before this grows beyond the first local schema.
