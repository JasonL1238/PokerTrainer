"""Physical schemas as this product actually wrote them, one version at a time.

Rewinding a current-shape database by dropping a couple of columns and
overwriting its version stamp proves that a migration is idempotent; it cannot
prove that the migration does anything, because every table already physically
contains what the migration is supposed to add. ``_seed_v12_database`` in
``test_persistence_integrity`` is exactly that shape, which is why migrations 14
and later had never been observed doing work.

The DDL below is cumulative and additive: ``_V0_SCHEMA`` is the pre-versioning
shape, and ``_SCHEMA_STEPS[n]`` is the DDL that version ``n`` introduced, so
``physical_schema(n)`` is a genuine version-``n`` file -- missing every table,
column and index that arrived later. ``test_migration_matrix`` asserts that
every version in ``db._MIGRATIONS`` has a step here, so the next migration
cannot land without its historical shape.

``snapshot_tables`` and ``assert_rows_survived`` are the row-preservation
invariant: they are what catches the migration that has not been written yet,
because they name no table and no column of their own.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

NOW = "2026-01-02T03:04:05+00:00"

# The pre-versioning shape: every column poker_tracker.persistence.db's
# _apply_legacy_backfill adds is deliberately absent, so opening one of these
# exercises the backfill instead of stepping over it.
_V0_SCHEMA = """
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date_played TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT '',
    stakes TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE hands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    hand_number INTEGER NOT NULL,
    hero_position TEXT NOT NULL DEFAULT '',
    hero_cards TEXT NOT NULL DEFAULT '',
    board_cards TEXT NOT NULL DEFAULT '',
    pot_size REAL,
    result TEXT NOT NULL DEFAULT '',
    hero_bb_won REAL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE hand_players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT NOT NULL DEFAULT '',
    starting_stack REAL,
    is_hero INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
);

CREATE TABLE actions (
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

CREATE TABLE hand_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL,
    hand_summary TEXT NOT NULL,
    theory_coach TEXT NOT NULL,
    exploit_coach TEXT NOT NULL,
    study_lesson TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
);

CREATE TABLE coaching_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    raw_prompt TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    review_type TEXT NOT NULL,
    hand_id INTEGER,
    session_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    duration_seconds REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    uploaded_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
);

CREATE TABLE processing_jobs (
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

CREATE TABLE extracted_frames (
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

CREATE TABLE roi_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT 'ClubWPT Gold',
    video_width INTEGER,
    video_height INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE roi_regions (
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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (profile_id) REFERENCES roi_profiles(id) ON DELETE CASCADE,
    UNIQUE(profile_id, roi_key)
);

CREATE INDEX idx_hands_session_id ON hands(session_id);
CREATE INDEX idx_hand_players_hand_id ON hand_players(hand_id);
CREATE INDEX idx_actions_hand_id ON actions(hand_id);
CREATE INDEX idx_reviews_hand_id ON hand_reviews(hand_id);
CREATE INDEX idx_coaching_reviews_hand_id ON coaching_reviews(hand_id);
CREATE INDEX idx_coaching_reviews_session_id ON coaching_reviews(session_id);
CREATE INDEX idx_videos_session_id ON videos(session_id);
CREATE INDEX idx_jobs_video_id ON processing_jobs(video_id);
CREATE INDEX idx_frames_video_id ON extracted_frames(video_id);
CREATE INDEX idx_roi_regions_profile_id ON roi_regions(profile_id);
"""

# Each entry is what that version added, and nothing else. Versions 1-4 predate
# both the stamp and every column below, so they share the v0 shape: db.init_db
# collapses them onto the same legacy backfill.
_SCHEMA_STEPS: dict[int, str] = {
    # Schema 5 is the first stamped version: the columns the legacy backfill
    # adds are already present, which is why a stamp of 5 must not need it.
    5: """
    ALTER TABLE hands ADD COLUMN game_type TEXT NOT NULL DEFAULT '';
    ALTER TABLE hands ADD COLUMN blinds_antes TEXT NOT NULL DEFAULT '';
    ALTER TABLE hands ADD COLUMN table_size INTEGER;
    ALTER TABLE hands ADD COLUMN effective_stack REAL;
    ALTER TABLE hands ADD COLUMN review_status TEXT NOT NULL DEFAULT 'unreviewed';
    ALTER TABLE hands ADD COLUMN confidence_score REAL;
    ALTER TABLE hands ADD COLUMN source_type TEXT NOT NULL DEFAULT 'manual';
    ALTER TABLE hands ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE hand_reviews ADD COLUMN ev_math_notes TEXT NOT NULL DEFAULT '';
    ALTER TABLE hand_reviews ADD COLUMN next_review_question TEXT NOT NULL DEFAULT '';
    ALTER TABLE hand_reviews ADD COLUMN notes TEXT NOT NULL DEFAULT '';
    ALTER TABLE coaching_reviews ADD COLUMN safety_mode TEXT NOT NULL
        DEFAULT 'post_session_only';
    ALTER TABLE coaching_reviews ADD COLUMN parsed_sections TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE videos ADD COLUMN frame_count INTEGER;
    ALTER TABLE roi_profiles ADD COLUMN table_layout TEXT NOT NULL DEFAULT '';
    ALTER TABLE roi_profiles ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE roi_regions ADD COLUMN notes TEXT NOT NULL DEFAULT '';
    CREATE INDEX idx_roi_profiles_active ON roi_profiles(is_active);
    """,
    6: """
    ALTER TABLE processing_jobs ADD COLUMN pid INTEGER;
    ALTER TABLE processing_jobs ADD COLUMN heartbeat_at TEXT;
    """,
    7: """
    ALTER TABLE hand_players ADD COLUMN player_key TEXT;
    ALTER TABLE hand_players ADD COLUMN seat_index INTEGER;
    ALTER TABLE actions ADD COLUMN player_key TEXT;
    ALTER TABLE actions ADD COLUMN amount_semantics TEXT NOT NULL DEFAULT 'unknown';
    ALTER TABLE actions ADD COLUMN forced_bet_type TEXT;
    ALTER TABLE actions ADD COLUMN is_live_post INTEGER;

    CREATE TABLE hand_settlements (
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

    CREATE TABLE settlement_entries (
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

    CREATE UNIQUE INDEX idx_hand_players_hand_key
        ON hand_players(hand_id, player_key);
    CREATE UNIQUE INDEX idx_hand_players_hand_seat
        ON hand_players(hand_id, seat_index)
        WHERE seat_index IS NOT NULL;
    CREATE INDEX idx_actions_player_key ON actions(hand_id, player_key);
    CREATE INDEX idx_settlement_entries_hand
        ON settlement_entries(hand_id, entry_type, pot_index, entry_order);
    """,
    8: """
    CREATE UNIQUE INDEX idx_actions_hand_street_order
        ON actions(hand_id, street, action_index);
    """,
    9: """
    CREATE TABLE reconstruction_frame_reviews (
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
    CREATE INDEX idx_reconstruction_reviews_job_hand
        ON reconstruction_frame_reviews(job_id, hand_number);
    """,
    10: """
    ALTER TABLE hand_reviews ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE hand_reviews ADD COLUMN stale_reason TEXT NOT NULL DEFAULT '';
    ALTER TABLE coaching_reviews ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE coaching_reviews ADD COLUMN stale_reason TEXT NOT NULL DEFAULT '';

    CREATE TABLE hand_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hand_id INTEGER NOT NULL,
        correction_type TEXT NOT NULL,
        before_state TEXT NOT NULL DEFAULT '{}',
        after_state TEXT NOT NULL DEFAULT '{}',
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_hand_corrections_hand_id
        ON hand_corrections(hand_id, created_at, id);
    """,
    11: """
    CREATE TABLE solver_range_profiles (
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

    CREATE TABLE solver_runs (
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

    CREATE INDEX idx_solver_runs_hand ON solver_runs(hand_id, created_at, id);
    CREATE INDEX idx_solver_runs_cache ON solver_runs(input_hash, status);
    CREATE INDEX idx_solver_runs_status ON solver_runs(status, created_at);
    """,
    12: """
    CREATE TABLE hand_issues (
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
    CREATE INDEX idx_hand_issues_status ON hand_issues(status, created_at, id);
    CREATE INDEX idx_hand_issues_hand ON hand_issues(hand_id, status, created_at, id);
    """,
    13: """
    ALTER TABLE hands ADD COLUMN completion_status TEXT NOT NULL
        DEFAULT 'not_applicable';
    ALTER TABLE hands ADD COLUMN completion_evidence TEXT NOT NULL DEFAULT '{}';
    """,
    14: """
    ALTER TABLE videos ADD COLUMN content_sha256 TEXT NOT NULL DEFAULT '';
    """,
    15: """
    ALTER TABLE hands ADD COLUMN study_inclusion TEXT NOT NULL DEFAULT 'auto';
    """,
    16: """
    ALTER TABLE actions ADD COLUMN source_image TEXT;
    """,
    17: """
    CREATE TABLE regression_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id INTEGER NOT NULL,
        correction_id INTEGER,
        kind TEXT NOT NULL,
        fixture_path TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'proposed',
        failing_before INTEGER NOT NULL DEFAULT 0,
        passing_after INTEGER NOT NULL DEFAULT 0,
        fixing_commit TEXT NOT NULL DEFAULT '',
        report_path TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (issue_id) REFERENCES hand_issues(id) ON DELETE CASCADE,
        FOREIGN KEY (correction_id) REFERENCES hand_corrections(id) ON DELETE SET NULL
    );
    CREATE INDEX idx_regression_cases_issue
        ON regression_cases(issue_id, status, id);
    """,
    18: """
    ALTER TABLE solver_runs ADD COLUMN run_parameters TEXT NOT NULL DEFAULT '{}';
    """,
}

# The versions that physically changed the file. Versions 1-4 are absent because
# they changed nothing a database can be distinguished by: the stamp did not
# exist yet, and init_db collapses all of them onto the same legacy backfill.
STEP_VERSIONS: tuple[int, ...] = tuple(sorted(_SCHEMA_STEPS))

# The highest shape this module can build. A migration added above this without a
# step here is caught by test_every_migration_has_a_historical_shape.
HIGHEST_KNOWN_VERSION = max(_SCHEMA_STEPS)

# Every supported starting point, in the form init_db sees it: None is a
# pre-versioning file with no stamp row at all.
SUPPORTED_START_VERSIONS: tuple[int | None, ...] = (None, *range(1, HIGHEST_KNOWN_VERSION + 1))

# One row per table, keyed by the table each row belongs in. Columns a given
# version does not have yet are dropped at insert time, so the same declaration
# seeds every shape.
_SEED_ROWS: dict[str, tuple[dict[str, object], ...]] = {
    "sessions": (
        {
            "id": 1,
            "name": "Legacy session",
            "date_played": "2026-01-01",
            "platform": "ClubWPT Gold",
            "stakes": "0.5/1",
            "notes": "seeded",
            "created_at": NOW,
        },
    ),
    # A reconstructed hand carrying completion_status='complete' and
    # review_status='reviewed' is the adversarial case for schema 13: replaying
    # that migration against a database that already passed it would knock this
    # row back to uncertain/needs_correction, which the invariant then reports.
    "hands": (
        {
            "id": 1,
            "session_id": 1,
            "hand_number": 1,
            "game_type": "NLHE",
            "blinds_antes": "0.5/1",
            "table_size": 6,
            "effective_stack": 100.0,
            "hero_position": "BTN",
            "hero_cards": "Ah Qs",
            "board_cards": "Qd 7s 2c",
            "pot_size": 20.0,
            "result": "won",
            "hero_bb_won": 10.0,
            "review_status": "reviewed",
            "confidence_score": 0.9,
            "source_type": "manual",
            "tags": '["BIG_POT"]',
            "notes": "manual hand",
            "created_at": NOW,
            "completion_status": "not_applicable",
            "completion_evidence": "{}",
            "study_inclusion": "study",
        },
        {
            "id": 2,
            "session_id": 1,
            "hand_number": 2,
            "game_type": "NLHE",
            "blinds_antes": "0.5/1",
            "table_size": 6,
            "effective_stack": 100.0,
            "hero_position": "BB",
            "hero_cards": "Kh Kd",
            "board_cards": "2s 5h 9c",
            "pot_size": 30.0,
            "result": "lost",
            "hero_bb_won": -12.0,
            "review_status": "reviewed",
            "confidence_score": 0.8,
            "source_type": "cv_import",
            "tags": "[]",
            "notes": "reconstructed hand",
            "created_at": NOW,
            "completion_status": "complete",
            "completion_evidence": "{}",
            "study_inclusion": "auto",
        },
        {
            "id": 3,
            "session_id": 1,
            "hand_number": 3,
            "game_type": "NLHE",
            "blinds_antes": "0.5/1",
            "table_size": 6,
            "effective_stack": 80.0,
            "hero_position": "CO",
            "hero_cards": "7h 7s",
            "board_cards": "",
            "pot_size": 4.0,
            "result": "folded",
            "hero_bb_won": -1.0,
            "review_status": "needs_correction",
            "confidence_score": 0.4,
            "source_type": "corrected_cv",
            "tags": "[]",
            "notes": "corrected hand",
            "created_at": NOW,
            "completion_status": "partial",
            "completion_evidence": "{}",
            "study_inclusion": "skip",
        },
        {
            "id": 4,
            "session_id": 1,
            "hand_number": 4,
            "game_type": "NLHE",
            "blinds_antes": "0.5/1",
            "table_size": 6,
            "effective_stack": 60.0,
            "hero_position": "SB",
            "hero_cards": "",
            "board_cards": "",
            "pot_size": None,
            "result": "",
            "hero_bb_won": None,
            "review_status": "unreviewed",
            "confidence_score": None,
            "source_type": "third_party_tool",
            "tags": "[]",
            "notes": "imported from elsewhere",
            "created_at": NOW,
            "completion_status": "uncertain",
            "completion_evidence": "{}",
            "study_inclusion": "auto",
        },
    ),
    # Distinct names, so the schema-7 player_key backfill has an unambiguous
    # match to make; an ambiguous one is left NULL by design.
    "hand_players": (
        {
            "id": 1,
            "hand_id": 2,
            "player_key": "hero",
            "seat_index": 0,
            "player_name": "Hero",
            "position": "BB",
            "starting_stack": 100.0,
            "is_hero": 1,
            "notes": "",
        },
        {
            "id": 2,
            "hand_id": 2,
            "player_key": "villain",
            "seat_index": 3,
            "player_name": "Villain",
            "position": "BTN",
            "starting_stack": 120.0,
            "is_hero": 0,
            "notes": "",
        },
    ),
    "hand_reviews": (
        {
            "id": 1,
            "hand_id": 2,
            "hand_summary": "summary",
            "theory_coach": "theory",
            "exploit_coach": "exploit",
            "ev_math_notes": "ev",
            "study_lesson": "lesson",
            "next_review_question": "question",
            "notes": "",
            "is_stale": 0,
            "stale_reason": "",
            "created_at": NOW,
        },
    ),
    "coaching_reviews": (
        {
            "id": 1,
            "provider_name": "legacy",
            "model_name": "fixture",
            "raw_prompt": "prompt",
            "raw_response": "response",
            "review_type": "hand",
            "safety_mode": "post_session_only",
            "hand_id": 2,
            "session_id": None,
            "parsed_sections": "{}",
            "is_stale": 0,
            "stale_reason": "",
            "created_at": NOW,
        },
        {
            "id": 2,
            "provider_name": "legacy",
            "model_name": "fixture",
            "raw_prompt": "prompt",
            "raw_response": "response",
            "review_type": "session",
            "safety_mode": "post_session_only",
            "hand_id": None,
            "session_id": 1,
            "parsed_sections": "{}",
            "is_stale": 0,
            "stale_reason": "",
            "created_at": NOW,
        },
    ),
    "videos": (
        {
            "id": 1,
            "session_id": 1,
            "original_filename": "session.mp4",
            "stored_path": "videos/session.mp4",
            "file_size_bytes": 1024,
            "content_sha256": "a" * 64,
            "duration_seconds": 600.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 18000,
            "uploaded_at": NOW,
            "notes": "",
        },
    ),
    "processing_jobs": (
        {
            "id": 1,
            "job_type": "cv_reconstruction",
            "status": "completed",
            "video_id": 1,
            "progress_percent": 100.0,
            "message": "done",
            "error_message": "",
            "pid": None,
            "heartbeat_at": None,
            "created_at": NOW,
            "started_at": NOW,
            "completed_at": NOW,
        },
    ),
    "extracted_frames": (
        {
            "id": 1,
            "video_id": 1,
            "job_id": 1,
            "timestamp_seconds": 12.5,
            "frame_index": 375,
            "image_path": "frames/job_1/frame_000375.png",
            "created_at": NOW,
        },
    ),
    "roi_profiles": (
        {
            "id": 1,
            "name": "ClubWPT 6-max",
            "description": "seeded",
            "platform": "ClubWPT Gold",
            "table_layout": "six_max",
            "video_width": 1920,
            "video_height": 1080,
            "created_at": NOW,
            "updated_at": NOW,
            "is_active": 1,
        },
    ),
    "roi_regions": (
        {
            "id": 1,
            "profile_id": 1,
            "roi_key": "seat_0_stack",
            "roi_type": "stack",
            "label": "Seat 0 stack",
            "x": 10,
            "y": 20,
            "width": 80,
            "height": 24,
            "seat_index": 0,
            "card_index": None,
            "notes": "",
            "created_at": NOW,
            "updated_at": NOW,
        },
    ),
    "hand_settlements": (
        {
            "hand_id": 2,
            "status": "reconciled",
            "dead_money": 0.5,
            "rake_rate": 0.05,
            "rake_cap": 3.0,
            "rake_rounding_unit": 0.01,
            "no_flop_no_drop": 1,
            "gross_pot": 30.0,
            "rake_amount": 1.5,
            "net_pot": 28.5,
            "is_balanced": 1,
            "warnings": "[]",
            "created_at": NOW,
            "updated_at": NOW,
        },
    ),
    "settlement_entries": (
        {
            "id": 1,
            "hand_id": 2,
            "entry_type": "award",
            "pot_index": 0,
            "player_key": "villain",
            "player_name": "Villain",
            "amount": 28.5,
            "entry_order": 1,
        },
        {
            "id": 2,
            "hand_id": 2,
            "entry_type": "rake",
            "pot_index": 0,
            "player_key": None,
            "player_name": "",
            "amount": 1.5,
            "entry_order": 2,
        },
    ),
    "reconstruction_frame_reviews": (
        {
            "id": 1,
            "job_id": 1,
            "hand_number": 2,
            "source_image": "frames/job_1/frame_000375.png",
            "timestamp_seconds": 12.5,
            "status": "confirmed",
            "issue_types": "[]",
            "notes": "",
            "created_at": NOW,
            "updated_at": NOW,
        },
    ),
    "hand_corrections": (
        {
            "id": 1,
            "hand_id": 2,
            "correction_type": "hand_facts",
            "before_state": '{"pot_size": 28.0}',
            "after_state": '{"pot_size": 30.0}',
            "notes": "Frame shows the river bet.",
            "created_at": NOW,
        },
    ),
    "solver_range_profiles": (
        {
            "id": 1,
            "name": "BTN open",
            "notation": "22+, A2s+",
            "table_size": 6,
            "position": "BTN",
            "scenario": "open",
            "pot_type": "srp",
            "stack_bb": 100.0,
            "description": "seeded",
            "source": "user",
            "created_at": NOW,
            "updated_at": NOW,
        },
    ),
    "solver_runs": (
        {
            "id": 1,
            "hand_id": 2,
            "status": "completed",
            "backend_name": "texassolver",
            "backend_version": "0.2.0",
            "input_hash": "legacy-hash",
            "spot": "{}",
            "range_ip": "{}",
            "range_oop": "{}",
            "assumptions": "[]",
            "evidence": "{}",
            "command_path": "solver/run_1/command.txt",
            "result_path": "solver/run_1/result.json",
            "log_path": "solver/run_1/run.log",
            "exploitability_pct": 0.4,
            "runtime_seconds": 12.0,
            "error_message": "",
            "pid": None,
            "heartbeat_at": None,
            "created_at": NOW,
            "started_at": NOW,
            "completed_at": NOW,
            "run_parameters": "{}",
        },
    ),
    "hand_issues": (
        {
            "id": 1,
            "hand_id": 3,
            "status": "open",
            "issue_types": '["hand_boundary"]',
            "description": "Boundary is unclear.",
            "evidence_snapshot": '{"hand_number": 3}',
            "resolution_notes": "",
            "created_at": NOW,
            "updated_at": NOW,
            "resolved_at": None,
        },
    ),
    "regression_cases": (
        {
            "id": 1,
            "issue_id": 1,
            "correction_id": 1,
            "kind": "fixture",
            "fixture_path": "tests/fixtures/boundary.json",
            "status": "proven",
            "failing_before": 1,
            "passing_after": 1,
            "fixing_commit": "abc1234",
            "report_path": "reports/boundary.md",
            "notes": "",
            "created_at": NOW,
            "updated_at": NOW,
        },
    ),
}

# Insert order, so a seeded row's parent always exists first.
_SEED_ORDER: tuple[str, ...] = (
    "sessions",
    "hands",
    "hand_players",
    "actions",
    "hand_reviews",
    "coaching_reviews",
    "videos",
    "processing_jobs",
    "extracted_frames",
    "roi_profiles",
    "roi_regions",
    "hand_settlements",
    "settlement_entries",
    "reconstruction_frame_reviews",
    "hand_corrections",
    "solver_range_profiles",
    "solver_runs",
    "hand_issues",
    "regression_cases",
)


def as_version(stored_version: int | None) -> int:
    """The version number a stamp of ``None`` (pre-versioning) stands for."""
    return 0 if stored_version is None else stored_version


def _action_rows(version: int) -> tuple[dict[str, object], ...]:
    """Two river actions, sharing an action_index only where that was possible.

    Before schema 8 there was no uniqueness index, and real files hold repeated
    indexes; from 8 on the index exists and the same file could not.
    """
    duplicate_order = version < 8
    return tuple(
        {
            "id": action_id,
            "hand_id": 2,
            "player_key": key,
            "street": "river",
            "action_index": 9 if duplicate_order else action_id,
            "player_name": name,
            "position": position,
            "action_type": action_type,
            "amount": amount,
            "amount_semantics": "incremental",
            "forced_bet_type": None,
            "is_live_post": None,
            "pot_before": pot_before,
            "stack_before": 100.0,
            "notes": "",
            "source_image": "frames/job_1/frame_000375.png",
        }
        for action_id, key, name, position, action_type, amount, pot_before in (
            (1, "villain", "Villain", "BTN", "bet", 10.0, 10.0),
            (2, "hero", "Hero", "BB", "call", 10.0, 20.0),
        )
    )


def physical_schema(stored_version: int | None) -> str:
    """The DDL a database of this version physically carried."""
    version = as_version(stored_version)
    if version > HIGHEST_KNOWN_VERSION:
        raise ValueError(
            f"No historical schema is declared for version {version}. Add its "
            "step to _SCHEMA_STEPS before relying on it."
        )
    parts = [_V0_SCHEMA]
    parts.extend(step for number, step in sorted(_SCHEMA_STEPS.items()) if number <= version)
    return "\n".join(parts)


def _existing_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def write_legacy_database(path: Path, *, stored_version: int | None) -> None:
    """Create a real database of that version and seed every table it had.

    ``stored_version`` of None means a pre-versioning file: the
    ``schema_metadata`` table exists but carries no ``schema_version`` row, which
    is how a database written before versioning arrives.
    """
    version = as_version(stored_version)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(physical_schema(stored_version))
        connection.execute(
            "CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        if stored_version is not None:
            connection.execute(
                "INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
                (str(stored_version),),
            )
        seeds = dict(_SEED_ROWS)
        seeds["actions"] = _action_rows(version)
        for table in _SEED_ORDER:
            if not _table_exists(connection, table):
                continue
            available = _existing_columns(connection, table)
            for row in seeds[table]:
                columns = [column for column in row if column in available]
                placeholders = ", ".join("?" for _ in columns)
                connection.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                    tuple(row[column] for column in columns),
                )
        connection.commit()
    finally:
        connection.close()


@dataclass(frozen=True)
class TableSnapshot:
    """One table as it stood: its columns, its rows, and a digest of both."""

    columns: tuple[str, ...]
    rows: tuple[tuple, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def digest(self, *, ignore: frozenset[str] = frozenset()) -> str:
        kept = [
            index for index, column in enumerate(self.columns) if column not in ignore
        ]
        payload = repr([tuple(row[index] for index in kept) for row in self.rows])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def project(self, columns: Sequence[str]) -> tuple[tuple, ...]:
        index = {column: position for position, column in enumerate(self.columns)}
        return tuple(tuple(row[index[column]] for column in columns) for row in self.rows)


def snapshot_tables(connection: sqlite3.Connection) -> dict[str, TableSnapshot]:
    """Every user table's full ordered contents.

    Ordered by rowid because that is the one ordering every table in this schema
    has, including ``hand_settlements``, whose primary key is its rowid alias.
    """
    snapshots: dict[str, TableSnapshot] = {}
    for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        table = str(row[0])
        columns = tuple(
            str(column[1])
            for column in connection.execute(f"PRAGMA table_info({table})").fetchall()
        )
        quoted = ", ".join(columns)
        rows = tuple(
            tuple(record)
            for record in connection.execute(
                f"SELECT {quoted} FROM {table} ORDER BY rowid"
            ).fetchall()
        )
        snapshots[table] = TableSnapshot(columns=columns, rows=rows)
    return snapshots


def assert_rows_survived(
    before: Mapping[str, TableSnapshot],
    after: Mapping[str, TableSnapshot],
    *,
    rewritten: Mapping[str, frozenset[str]] | None = None,
) -> None:
    """Assert no table lost a row, no column vanished, and no value changed.

    ``rewritten`` names the ``table -> {column}`` pairs a migration's MIGRATION
    IMPACT docstring says it rewrites; those columns are excluded from the value
    comparison and from nothing else, so a rewrite still cannot delete a row or
    drop a column.

    This is deliberately generic. It names no table and no column of its own, so
    it applies to the migration that has not been written yet: the failure it is
    built to catch is a future migration that rebuilds a table and silently loses
    the rows it did not copy.
    """
    allowed = dict(rewritten or {})
    for table, snapshot in before.items():
        assert table in after, (
            f"Table {table} existed before the migration and is gone after it. A "
            "migration that removes a table must be shown to have carried its "
            "rows somewhere."
        )
        current = after[table]
        vanished = sorted(set(snapshot.columns) - set(current.columns))
        assert not vanished, (
            f"{table} lost column(s) {', '.join(vanished)}. A migration that "
            "drops a column has to show that it carried the data somewhere or "
            "that the column was empty."
        )
        assert current.row_count >= snapshot.row_count, (
            f"{table} held {snapshot.row_count} row(s) before the migration and "
            f"{current.row_count} after it."
        )
        ignore = allowed.get(table, frozenset())
        kept = [column for column in snapshot.columns if column not in ignore]
        expected = snapshot.project(kept)
        observed = current.project(kept)[: snapshot.row_count]
        assert snapshot.digest(ignore=ignore) == TableSnapshot(
            columns=tuple(kept), rows=observed
        ).digest(), (
            f"{table} rows changed across the migration.\n"
            f"before: {expected}\nafter:  {observed}"
        )
