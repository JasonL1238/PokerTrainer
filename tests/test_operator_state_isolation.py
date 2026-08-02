"""The suite must never open, migrate, or write the operator's own state.

Round 12 found that running the documented ``pytest`` command applied the
IRREVERSIBLE schema v13 migration to ``<repo>/poker_tracker.db``: collecting
``tests/test_app_shell.py`` runs ``AppTest.from_file("app.py")``, which opens
``PokerDatabase(DEFAULT_DB_PATH)`` and calls ``init_db()``. On a pre-Phase-1
operator database that rewrote ``review_status`` from ``reviewed`` to
``needs_correction`` on every reconstructed hand, and the pinned pre-migration
snapshot -- documented in ``persistence/backup.py`` as "the only artifact that
can undo an irreversible migration" -- was written into the ``isolated_backup_dir``
temp tree and deleted with it, so ``data/backups`` stayed empty.

The repair is not "redirect the database too". Redirecting one more constant is
the per-field patch this program has spent eleven rounds learning not to make:
the hazard had already been recognised for backups and missed for the database,
the videos, the frames, the exports, the ROI previews, the CV timelines, the job
logs and the solver runs. What ``tests/conftest.py`` does instead is claim the
two environment variables every operator root in this product resolves from,
before the first ``poker_tracker`` import in the process, so a root added later
that follows the same convention is redirected without anyone editing a list.

The tests below check that property rather than the redirect: any module-level
path constant that does NOT move when those two variables move, and is not a
source asset shipped in the repository, is a hole in it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poker_tracker.maintenance import data_health
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence import db as db_module
from poker_tracker.solver import storage as solver_storage
from poker_tracker.suite_quality import flake
from poker_tracker.ui import video_storage

REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = r"""
import importlib
import json
import sys
from pathlib import Path

import poker_tracker

# Walked from the filesystem, not with pkgutil: four of this package's
# subpackages (`persistence`, `ui`, `math`, `coaching`) carry no `__init__.py`,
# so `pkgutil.iter_modules` reports none of them -- which silently excluded
# `persistence.db` and `ui.video_storage`, the two modules this whole file is
# about, from an earlier version of this probe.
root = Path(poker_tracker.__file__).resolve().parent
names = []
for path in sorted(root.rglob("*.py")):
    if path.stem in {"__init__", "__main__"}:  # a CLI entrypoint runs on import
        parts = path.relative_to(root).parts[:-1]
    else:
        parts = (*path.relative_to(root).parts[:-1], path.stem)
    if not parts:
        continue
    names.append("poker_tracker." + ".".join(parts))

found = {}
imported = 0
for module_name in sorted(set(names)):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        sys.stderr.write(f"{module_name}: {exc}\n")
        continue
    imported += 1
    for attribute, value in vars(module).items():
        if attribute.startswith("_") or not isinstance(value, (str, Path)):
            continue
        text = str(value)
        if not text.startswith("/"):
            continue
        found[f"{module_name}.{attribute}"] = text
sys.stdout.write(json.dumps({"imported": imported, "constants": found}))
"""


def _module_path_constants(db_path: Path, data_dir: Path) -> tuple[int, dict[str, str]]:
    """Every module-level absolute path constant, as one sandbox resolves it."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(data_dir),
            "POKER_DB_PATH": str(db_path),
            "POKER_DATA_DIR": str(data_dir),
        },
    )
    probed = json.loads(completed.stdout)
    return int(probed["imported"]), dict(probed["constants"])


def test_the_suite_cannot_reach_the_operator_database() -> None:
    """The demonstrated defect: pytest migrated <repo>/poker_tracker.db in place.

    BEFORE the conftest redirect, ``DEFAULT_DB_PATH`` was
    ``<repo>/poker_tracker.db`` for the whole run and ``test_app_shell`` opened
    and migrated it.
    """
    resolved = Path(db_module.DEFAULT_DB_PATH).resolve()
    assert not resolved.is_relative_to(REPO_ROOT), (
        f"The suite's default database path is inside the repository: {resolved}"
    )


def test_every_operator_owned_root_is_redirected_out_of_the_repository() -> None:
    """The database was one of eight roots; only the backups had been redirected."""
    roots = {
        "database": Path(db_module.DEFAULT_DB_PATH),
        "backups": backup_module.BACKUPS_DIR,
        "backup data dir": backup_module.DATA_DIR,
        "videos": video_storage.VIDEOS_DIR,
        "frames": video_storage.FRAMES_DIR,
        "exports": video_storage.EXPORTS_DIR,
        "roi previews": video_storage.ROI_PREVIEWS_DIR,
        "cv timelines": video_storage.CV_TIMELINES_DIR,
        "job logs": video_storage.JOB_LOGS_DIR,
        "solver runs": solver_storage.SOLVER_RUNS_DIR,
        "data health database": data_health.DEFAULT_DATABASE_PATH,
        "data health data dir": data_health.DEFAULT_DATA_DIR,
    }
    inside = {
        label: str(path)
        for label, path in roots.items()
        if Path(path).resolve().is_relative_to(REPO_ROOT)
    }
    assert not inside, f"Operator state roots still inside the repository: {inside}"


def test_no_operator_path_constant_escapes_the_environment_redirect(
    tmp_path: Path,
) -> None:
    """The family rule, with no module list in it.

    Import every ``poker_tracker`` module in two subprocesses whose
    ``POKER_DB_PATH`` and ``POKER_DATA_DIR`` differ, and compare every
    module-level absolute path constant. A constant that does not move is only
    acceptable when it is a source asset shipped in the repository -- the
    package root, or a script the pipeline runs. Anything else is a location the
    product writes that this suite cannot redirect, which is exactly the shape
    the database had.
    """
    modules = len(
        [
            path
            for path in (REPO_ROOT / "poker_tracker").rglob("*.py")
            if path.stem != "__main__"
        ]
    )
    imported_first, first = _module_path_constants(
        tmp_path / "a" / "poker.db", tmp_path / "a" / "data"
    )
    imported_second, second = _module_path_constants(
        tmp_path / "b" / "poker.db", tmp_path / "b" / "data"
    )
    # A probe that quietly imports nothing proves nothing. An earlier version
    # walked with `pkgutil`, which reports no namespace package, so it never
    # imported `persistence`, `ui`, `math` or `coaching` at all -- the two modules
    # the finding is about were both in that gap, and the test passed anyway.
    assert imported_first == imported_second >= modules - 4, (
        f"The probe imported {imported_first} of about {modules} modules."
    )
    assert first and second, "The probe found no module-level path constants at all."
    assert "poker_tracker.persistence.db.DEFAULT_DB_PATH" in first
    assert "poker_tracker.ui.video_storage.VIDEOS_DIR" in first

    unmoved = {name: value for name, value in first.items() if second.get(name) == value}
    operator_state = {
        name: value
        for name, value in unmoved.items()
        # A constant that does not move is a hole unless it is a path the
        # repository actually ships, and no shipped source asset lives under the
        # operator's data tree.
        if not Path(value).exists()
        or Path(value).resolve().is_relative_to(REPO_ROOT / "data")
    }
    assert not operator_state, (
        "These module-level paths are not redirected by POKER_DB_PATH / "
        f"POKER_DATA_DIR and are not shipped source assets: {operator_state}"
    )


def test_running_the_app_shell_leaves_the_operator_database_untouched() -> None:
    """End to end, on the real file, exactly as the finding demonstrated it.

    ``AppTest.from_file("app.py")`` is what ``test_app_shell`` runs. Before the
    redirect this opened, WAL-ified and migrated the repository's own database;
    afterwards the file is not touched at all.
    """
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    operator_db = REPO_ROOT / "poker_tracker.db"
    if not operator_db.exists():
        pytest.skip("No operator database in this checkout to protect.")
    before = operator_db.stat()
    sidecars_before = sorted(p.name for p in REPO_ROOT.glob("poker_tracker.db-*"))

    AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=20).run()

    after = operator_db.stat()
    assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
    assert sorted(p.name for p in REPO_ROOT.glob("poker_tracker.db-*")) == sidecars_before


def _collect_only(plugin: str | None, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Collect one cheap module in a child pytest, optionally loading a plugin."""
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q"]
    if plugin is not None:
        command += ["-p", plugin]
    command.append(str(REPO_ROOT / "tests" / "test_icm.py"))
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTEST_ADDOPTS": "", "TMPDIR": str(tmp_path)},
    )


def test_a_plugin_loaded_from_inside_the_package_cannot_reach_the_real_database(
    tmp_path: Path,
) -> None:
    """The redirect assumes conftest runs first, and ``-p`` breaks that assumption.

    pytest imports a ``-p`` plugin during argument preparsing, before any
    conftest. Naming a module inside ``poker_tracker`` therefore executes
    ``poker_tracker/__init__.py`` -- which imports ``persistence.db`` and
    ``ui.video_storage`` -- while ``POKER_DB_PATH`` and ``POKER_DATA_DIR`` are
    still unset, so every operator root freezes on the real one and the suite
    migrates ``<repo>/poker_tracker.db``. Observed: a shuffled full run took the
    operator's database from schema 15 to 18, and the pre-migration snapshot the
    migration writes went to wherever the unredirected data directory pointed.

    The run must be refused rather than allowed to proceed, because by the time
    a test opens the path the damage is a completed migration.
    """
    result = _collect_only("poker_tracker.suite_quality.random_order", tmp_path)
    assert result.returncode != 0, (
        "A plugin that imports the application before conftest was allowed to run:\n"
        + result.stdout
    )
    combined = result.stdout + result.stderr
    assert "before tests/conftest.py could redirect" in combined, combined


def test_the_supported_shuffle_plugin_leaves_the_redirect_intact(tmp_path: Path) -> None:
    """The shim exists so the flake hunt is still runnable, not merely refused."""
    result = _collect_only(flake.PLUGIN, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert flake.PLUGIN == "sq_random_order", (
        "The flake harness must load the top-level shim, not a module inside the "
        f"package: {flake.PLUGIN}"
    )
