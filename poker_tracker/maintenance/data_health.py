"""Read-only health checks for PokerTrainer's database and stored artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

CheckStatus = Literal["pass", "warning", "fail"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = Path(
    os.environ.get("POKER_DB_PATH", PROJECT_ROOT / "poker_tracker.db")
)
DEFAULT_DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
BACKUP_GLOB = "poker_tracker_*.sqlite3"
DETAIL_LIMIT = 20

_ARTIFACT_REFERENCES: tuple[tuple[str, str, str], ...] = (
    ("videos", "id", "stored_path"),
    ("extracted_frames", "id", "image_path"),
    ("reconstruction_frame_reviews", "id", "source_image"),
)

# Stable minimum contract shared by all supported PokerTrainer databases. Newer
# version-specific tables may be additive, but these tables and columns are
# required for the core completed-session workflow to be usable.
_MINIMUM_SCHEMA: dict[str, set[str]] = {
    "schema_metadata": {"key", "value"},
    "sessions": {"id", "name", "date_played"},
    "hands": {"id", "session_id", "hand_number", "review_status", "source_type"},
    "hand_players": {"id", "hand_id", "player_key"},
    "actions": {"id", "hand_id", "street", "action_index", "action_type"},
    "hand_reviews": {"id", "hand_id"},
    "coaching_reviews": {"id", "review_type"},
    "videos": {"id", "stored_path", "file_size_bytes"},
    "processing_jobs": {"id", "status", "video_id"},
    "extracted_frames": {"id", "video_id", "job_id", "image_path"},
}


@dataclass(frozen=True)
class CheckResult:
    """One independently actionable health-check result."""

    name: str
    status: CheckStatus
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class HealthReport:
    """Complete read-only health report for one PokerTrainer data store."""

    database_path: str
    data_dir: str
    backup_dir: str
    checked_at: str
    checks: tuple[CheckResult, ...]

    @property
    def healthy(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warning" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "database_path": self.database_path,
            "data_dir": self.data_dir,
            "backup_dir": self.backup_dir,
            "checked_at": self.checked_at,
            "healthy": self.healthy,
            "has_warnings": self.has_warnings,
            "checks": [asdict(check) for check in self.checks],
        }


def audit_data_health(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    backup_dir: str | Path | None = None,
    expected_schema_version: int | None = None,
    restore_backups: bool = False,
) -> HealthReport:
    """Audit the live SQLite file, recorded artifacts, and retained backups.

    The live database and backups are opened in SQLite read-only/query-only
    mode. SQLite may still maintain its WAL shared-memory sidecars while reading.
    A restore drill copies each backup into an isolated temporary database; it
    never overwrites the live database or any backup file.
    """

    database = _absolute_path(database_path)
    data = _absolute_path(data_dir)
    backups = (
        _absolute_path(backup_dir)
        if backup_dir is not None
        else data / "backups"
    )

    checks = list(
        _audit_database(
            database,
            data,
            expected_schema_version=expected_schema_version,
        )
    )
    checks.append(
        _audit_backups(
            backups,
            live_database=database,
            expected_schema_version=expected_schema_version,
            restore_backups=restore_backups,
        )
    )
    return HealthReport(
        database_path=str(database),
        data_dir=str(data),
        backup_dir=str(backups),
        checked_at=datetime.now(UTC).isoformat(),
        checks=tuple(checks),
    )


def _absolute_path(value: str | Path) -> Path:
    path = Path(value)
    try:
        path = path.expanduser()
    except RuntimeError:
        # Preserve an unresolvable "~user" literally so the audit returns a
        # structured missing/inaccessible result instead of crashing.
        pass
    return path.absolute()


def _audit_database(
    database_path: Path,
    data_dir: Path,
    *,
    expected_schema_version: int | None,
) -> tuple[CheckResult, ...]:
    try:
        database_stat = database_path.stat()
    except FileNotFoundError:
        return (
            CheckResult(
                "database_file",
                "fail",
                "Database file does not exist.",
                (str(database_path),),
            ),
        )
    except OSError as exc:
        return (
            CheckResult(
                "database_file",
                "fail",
                "Database file cannot be inspected.",
                (f"{type(exc).__name__}: {exc}",),
            ),
        )
    if not stat.S_ISREG(database_stat.st_mode):
        return (
            CheckResult(
                "database_file",
                "fail",
                "Database path is not a regular file.",
                (str(database_path),),
            ),
        )

    results = [
        CheckResult(
            "database_file",
            "pass",
            f"Database file is present ({database_stat.st_size} bytes).",
        )
    ]
    try:
        with closing(_connect_read_only(database_path)) as connection:
            results.append(_quick_check(connection))
            results.append(_foreign_key_check(connection))
            results.append(_schema_check(connection, expected_schema_version))
            results.append(_schema_contract_check(connection))
            results.append(_artifact_check(connection, database_path, data_dir))
    except (OSError, sqlite3.Error) as exc:
        results.append(
            CheckResult(
                "database_open",
                "fail",
                "Database could not be audited in read-only mode.",
                (f"{type(exc).__name__}: {exc}",),
            )
        )
    return tuple(results)


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{database_path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=5,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _quick_check(connection: sqlite3.Connection) -> CheckResult:
    rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    if rows == ["ok"]:
        return CheckResult("sqlite_quick_check", "pass", "SQLite quick check passed.")
    return CheckResult(
        "sqlite_quick_check",
        "fail",
        "SQLite reported structural corruption.",
        _limited_details(rows),
    )


def _foreign_key_check(connection: sqlite3.Connection) -> CheckResult:
    count = 0
    details: list[str] = []
    for row in connection.execute("PRAGMA foreign_key_check"):
        count += 1
        if len(details) < DETAIL_LIMIT:
            details.append(
                f"{row[0]} row {row[1]} references missing {row[2]} row "
                f"(constraint {row[3]})."
            )
    if count == 0:
        return CheckResult(
            "foreign_key_check",
            "pass",
            "No broken database relationships were found.",
        )
    return CheckResult(
        "foreign_key_check",
        "fail",
        f"Found {count} broken database relationship(s).",
        _details_with_hidden_count(details, count),
    )


def _schema_check(
    connection: sqlite3.Connection,
    expected_schema_version: int | None,
) -> CheckResult:
    if not _table_exists(connection, "schema_metadata"):
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema metadata is missing.",
        )
    row = connection.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema version is missing.",
        )
    try:
        version = int(row[0])
    except (TypeError, ValueError):
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema version is not an integer.",
            (repr(row[0]),),
        )
    if version <= 0:
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema version must be positive.",
            (str(version),),
        )
    if expected_schema_version is not None and version > expected_schema_version:
        return CheckResult(
            "schema_version",
            "fail",
            "Database schema is newer than this PokerTrainer build supports.",
            (f"database={version}", f"supported={expected_schema_version}"),
        )
    if expected_schema_version is not None and version < expected_schema_version:
        return CheckResult(
            "schema_version",
            "warning",
            "Database schema is older than this build and needs migration.",
            (f"database={version}", f"current={expected_schema_version}"),
        )
    return CheckResult(
        "schema_version",
        "pass",
        f"PokerTrainer schema version {version} is readable.",
    )


def _schema_contract_check(connection: sqlite3.Connection) -> CheckResult:
    problems: list[str] = []
    for table, required_columns in _MINIMUM_SCHEMA.items():
        if not _table_exists(connection, table):
            problems.append(f"missing table: {table}")
            continue
        columns = _table_columns(connection, table)
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            problems.append(
                f"{table}: missing column(s) {', '.join(missing_columns)}"
            )
    if problems:
        return CheckResult(
            "schema_contract",
            "fail",
            f"Core schema is incomplete in {len(problems)} place(s).",
            _limited_details(problems),
        )
    return CheckResult(
        "schema_contract",
        "pass",
        "Core PokerTrainer tables and columns are present.",
    )


def _artifact_check(
    connection: sqlite3.Connection,
    database_path: Path,
    data_dir: Path,
) -> CheckResult:
    problem_details: list[str] = []
    missing_count = 0
    size_mismatch_count = 0
    reference_count = 0

    for table, id_column, path_column in _ARTIFACT_REFERENCES:
        required_columns = {id_column, path_column}
        if table == "videos":
            required_columns.add("file_size_bytes")
        if not _table_has_columns(connection, table, required_columns):
            continue
        extra = ", file_size_bytes" if table == "videos" else ""
        query = f"SELECT {id_column}, {path_column}{extra} FROM {table}"
        for row in connection.execute(query):
            reference_count += 1
            stored_path = str(row[path_column])
            resolved = _resolve_artifact_path(stored_path, database_path, data_dir)
            label = f"{table} row {row[id_column]}"
            try:
                is_file = resolved.is_file()
            except OSError as exc:
                missing_count += 1
                _append_limited(
                    problem_details,
                    f"{label}: cannot inspect {stored_path}: {exc}",
                )
                continue
            if not is_file:
                missing_count += 1
                _append_limited(problem_details, f"{label}: missing {stored_path}")
                continue
            if table == "videos":
                try:
                    expected_size = int(row["file_size_bytes"])
                    actual_size = resolved.stat().st_size
                except (OSError, TypeError, ValueError) as exc:
                    size_mismatch_count += 1
                    _append_limited(
                        problem_details,
                        f"{label}: cannot verify size: {exc}",
                    )
                    continue
                if actual_size != expected_size:
                    size_mismatch_count += 1
                    _append_limited(
                        problem_details,
                        f"{label}: expected {expected_size} bytes, found {actual_size}"
                    )

    problem_count = missing_count + size_mismatch_count
    if problem_count:
        return CheckResult(
            "artifact_files",
            "fail",
            (
                f"Found {missing_count} missing artifact(s) and "
                f"{size_mismatch_count} video size mismatch(es)."
            ),
            _details_with_hidden_count(problem_details, problem_count),
        )
    return CheckResult(
        "artifact_files",
        "pass",
        f"All {reference_count} recorded artifact reference(s) are present.",
    )


def _resolve_artifact_path(
    stored_path: str,
    database_path: Path,
    data_dir: Path,
) -> Path:
    path = Path(stored_path).expanduser()
    if path.is_absolute():
        return path
    candidates = (database_path.parent / path, data_dir / path)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return candidates[0]


def _audit_backups(
    backup_dir: Path,
    *,
    live_database: Path,
    expected_schema_version: int | None,
    restore_backups: bool,
) -> CheckResult:
    try:
        backup_dir_stat = backup_dir.stat()
    except FileNotFoundError:
        return CheckResult(
            "backups",
            "warning",
            "Backup directory does not exist yet.",
            (str(backup_dir),),
        )
    except OSError as exc:
        return CheckResult(
            "backups",
            "fail",
            "Backup directory cannot be inspected.",
            (f"{type(exc).__name__}: {exc}",),
        )
    if not stat.S_ISDIR(backup_dir_stat.st_mode):
        return CheckResult(
            "backups",
            "fail",
            "Backup path is not a directory.",
            (str(backup_dir),),
        )

    try:
        with os.scandir(backup_dir) as entries:
            backups = sorted(
                Path(entry.path)
                for entry in entries
                if fnmatch.fnmatchcase(entry.name, BACKUP_GLOB)
            )
    except OSError as exc:
        return CheckResult(
            "backups",
            "fail",
            "Backup directory cannot be read.",
            (f"{type(exc).__name__}: {exc}",),
        )
    if not backups:
        return CheckResult(
            "backups",
            "warning",
            "No retained PokerTrainer backups were found.",
            (str(backup_dir),),
        )

    failures: list[str] = []
    warnings: list[str] = []
    for backup in backups:
        backup_failures, backup_warnings = _backup_issues(
            backup,
            live_database=live_database,
            expected_schema_version=expected_schema_version,
            restore=restore_backups,
        )
        failures.extend(backup_failures)
        warnings.extend(backup_warnings)
    if failures:
        return CheckResult(
            "backups",
            "fail",
            f"{len(failures)} backup verification problem(s) were found.",
            _limited_details(failures),
        )
    if warnings:
        return CheckResult(
            "backups",
            "warning",
            f"Backups are readable but have {len(warnings)} compatibility warning(s).",
            _limited_details(warnings),
        )

    action = "passed a restore drill" if restore_backups else "are readable"
    return CheckResult(
        "backups",
        "pass",
        f"All {len(backups)} retained backup(s) {action}.",
    )


def _backup_issues(
    backup_path: Path,
    *,
    live_database: Path,
    expected_schema_version: int | None,
    restore: bool,
) -> tuple[list[str], list[str]]:
    if backup_path.is_symlink():
        return (
            [f"{backup_path.name}: symlinks are not independent backups"],
            [],
        )
    try:
        backup_stat = backup_path.stat()
        if not stat.S_ISREG(backup_stat.st_mode):
            return ([f"{backup_path.name}: not a regular file"], [])
        try:
            same_as_live = os.path.samefile(backup_path, live_database)
        except FileNotFoundError:
            same_as_live = False
        if same_as_live or backup_stat.st_nlink > 1:
            return (
                [f"{backup_path.name}: hard-linked files are not independent backups"],
                [],
            )
        with closing(_connect_read_only(backup_path)) as source:
            failures, warnings = _connection_issues(
                source,
                expected_schema_version=expected_schema_version,
            )
            if failures or not restore:
                return (
                    [f"{backup_path.name}: {failure}" for failure in failures],
                    [f"{backup_path.name}: {warning}" for warning in warnings],
                )
            with tempfile.TemporaryDirectory(prefix="pokertrainer-restore-") as temp_dir:
                restored_path = Path(temp_dir) / "restored.sqlite3"
                with closing(sqlite3.connect(restored_path)) as restored:
                    source.backup(restored)
                with closing(_connect_read_only(restored_path)) as restored:
                    restore_failures, _ = _connection_issues(
                        restored,
                        expected_schema_version=None,
                    )
            return (
                [
                    f"{backup_path.name} restore: {failure}"
                    for failure in restore_failures
                ],
                [f"{backup_path.name}: {warning}" for warning in warnings],
            )
    except (OSError, sqlite3.Error) as exc:
        return ([f"{backup_path.name}: {type(exc).__name__}: {exc}"], [])


def _connection_issues(
    connection: sqlite3.Connection,
    *,
    expected_schema_version: int | None,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    quick_rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    if quick_rows != ["ok"]:
        failures.extend(f"quick_check: {item}" for item in quick_rows)
    foreign_count = 0
    for row in connection.execute("PRAGMA foreign_key_check"):
        foreign_count += 1
        if len(failures) < DETAIL_LIMIT:
            failures.append(
                f"foreign_key_check: {row[0]} row {row[1]} references {row[2]}"
            )
    if foreign_count > DETAIL_LIMIT:
        failures.append(
            f"foreign_key_check: ... and {foreign_count - DETAIL_LIMIT} more"
        )

    schema = _schema_check(connection, expected_schema_version)
    if schema.status == "fail":
        failures.append(f"schema_version: {schema.message}")
        failures.extend(f"schema_version: {detail}" for detail in schema.details)
    elif schema.status == "warning":
        warnings.append(f"schema_version: {schema.message}")
        warnings.extend(f"schema_version: {detail}" for detail in schema.details)

    contract = _schema_contract_check(connection)
    if contract.status == "fail":
        failures.append(f"schema_contract: {contract.message}")
        failures.extend(f"schema_contract: {detail}" for detail in contract.details)
    return failures, warnings


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_has_columns(
    connection: sqlite3.Connection,
    table: str,
    required_columns: set[str],
) -> bool:
    if not _table_exists(connection, table):
        return False
    return required_columns <= _table_columns(connection, table)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _append_limited(details: list[str], value: str) -> None:
    if len(details) < DETAIL_LIMIT:
        details.append(value)


def _details_with_hidden_count(
    visible_details: Sequence[str],
    total_count: int,
) -> tuple[str, ...]:
    details = list(visible_details)
    hidden = total_count - len(details)
    if hidden > 0:
        details.append(f"... and {hidden} more")
    return tuple(details)


def _limited_details(details: Sequence[str]) -> tuple[str, ...]:
    values = list(details)
    if len(values) <= DETAIL_LIMIT:
        return tuple(values)
    hidden = len(values) - DETAIL_LIMIT
    return (*values[:DETAIL_LIMIT], f"... and {hidden} more")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit PokerTrainer's local database, artifacts, and backups."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument(
        "--restore-backups",
        action="store_true",
        help="Copy each retained backup into memory and verify the restored database.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from poker_tracker.persistence.db import SCHEMA_VERSION

    report = audit_data_health(
        args.db,
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        expected_schema_version=SCHEMA_VERSION,
        restore_backups=args.restore_backups,
    )
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_report(report)
    return 0 if report.healthy else 1


def _print_report(report: HealthReport) -> None:
    state = "HEALTHY" if report.healthy else "UNHEALTHY"
    print(f"PokerTrainer data health: {state}")
    print(f"Database: {report.database_path}")
    print(f"Data directory: {report.data_dir}")
    print(f"Backup directory: {report.backup_dir}")
    for check in report.checks:
        print(f"[{check.status.upper()}] {check.name}: {check.message}")
        for detail in check.details:
            print(f"  - {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
