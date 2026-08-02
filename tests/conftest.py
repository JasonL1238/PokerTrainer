"""Shared test isolation for the whole suite.

The redirect below runs at IMPORT time, before the first ``poker_tracker``
import in this file, and that ordering is the whole mechanism. Every
operator-owned root in this product is a module-level constant resolved from an
environment variable exactly once, when its module is first imported:
``db.DEFAULT_DB_PATH`` from ``POKER_DB_PATH``, and ``backup``,
``ui.video_storage``, ``solver.storage`` and ``maintenance.data_health`` from
``POKER_DATA_DIR``. Setting both variables before any of those modules load
therefore redirects all of them, and every root added later that follows the
same convention, with no list of modules to keep current.

It exists because the suite reached the operator's real database. ``pytest``
collects ``tests/test_app_shell.py``, which runs ``AppTest.from_file("app.py")``,
which opens ``PokerDatabase(DEFAULT_DB_PATH)`` and calls ``init_db()``. On a
pre-Phase-1 file that applied the IRREVERSIBLE v13 migration to
``<repo>/poker_tracker.db`` -- rewriting ``review_status`` from ``reviewed`` to
``needs_correction`` on every reconstructed hand -- while the pinned rollback
snapshot the migration writes went to the ``isolated_backup_dir`` temp tree and
was deleted with it, leaving ``data/backups`` empty. The hazard had been
recognised for backups alone; the database, the videos, the frames, the exports,
the CV timelines, the job logs and the solver runs were all still live.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

# The redirect only works if nothing has already imported the application, and
# this file is not the earliest code pytest runs. A plugin loaded with `-p` is
# imported during argument preparsing, before any conftest, so `-p
# poker_tracker.<anything>` executes `poker_tracker/__init__.py` -- which
# imports `persistence.db` and `ui.video_storage` -- with the variables below
# still unset. Every operator root then freezes on the real one, and the suite
# migrates `<repo>/poker_tracker.db` exactly as it did in round 12. Refusing
# here costs a run; discovering it afterwards costs the rollback point, because
# the pre-migration snapshot is written wherever POKER_DATA_DIR pointed at
# import time.
if "poker_tracker" in sys.modules:
    raise RuntimeError(
        "poker_tracker was imported before tests/conftest.py could redirect the "
        "operator's state, so the suite would read and migrate the real database "
        "and data directory. This is what loading a plugin from inside the "
        "poker_tracker package with -p does. Load the shuffle plugin as "
        "`-p sq_random_order` (the top-level shim, which imports nothing), or set "
        "POKER_DB_PATH and POKER_DATA_DIR outside the repository before invoking "
        "pytest."
    )

# Claimed before the first poker_tracker import below, and never after: these
# constants are read once per process, at module import.
_OPERATOR_STATE_SANDBOX = Path(tempfile.mkdtemp(prefix="pokertrainer-test-state-"))
os.environ["POKER_DB_PATH"] = str(_OPERATOR_STATE_SANDBOX / "poker_tracker.db")
os.environ["POKER_DATA_DIR"] = str(_OPERATOR_STATE_SANDBOX / "data")

import pytest  # noqa: E402

from poker_tracker.persistence import backup as backup_module  # noqa: E402
from poker_tracker.persistence.db import PokerDatabase  # noqa: E402
from poker_tracker.services.hand_accounting import (  # noqa: E402
    attest_assumption,
    reconcile_persisted_hand,
)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove the sandbox the suite's operator state was redirected into."""
    shutil.rmtree(_OPERATOR_STATE_SANDBOX, ignore_errors=True)


def attest_declared_assumptions(
    db: PokerDatabase, hand_id: int, *, only: str | None = None
) -> tuple[str, ...]:
    """Press 'Confirm this assumption' for every measured dependence, as the operator does.

    A reconstructed hand's pot awards are typed into the Accounting
    reconciliation panel -- the CV exporter emits no settlement rows at all -- so
    they are a declared settlement input like the rake and the dead money, and
    they are measured and blocked like one. A fixture that wants a study-READY
    reconstructed hand therefore has to perform the attestation the product asks
    for, exactly as it already has to tick the whole-hand confirmation checkbox.

    ``only`` restricts the attestation to one ``input_name``, which is how a test
    isolates one declaration's blocker from another's.

    Returns the codes it confirmed, so a caller can assert on what it answered
    rather than assuming.
    """
    accounting = reconcile_persisted_hand(db, hand_id)
    confirmed: list[str] = []
    for dependence in accounting.assumption_dependence:
        if only is not None and dependence.input_name != only:
            continue
        # Through the same door the control uses, which measures before writing.
        attest_assumption(db, hand_id, dependence.code)
        confirmed.append(dependence.code)
    return tuple(confirmed)


@pytest.fixture(autouse=True)
def isolated_backup_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Keep migration and CV-import snapshots out of the operator's real backup set.

    init_db() snapshots any real file database before migrating it, and the
    retention policy keeps only the newest five. Without this redirect, running
    the suite would rotate away genuine backups in data/backups.

    The redirect owns a PRIVATE MonkeyPatch rather than taking the shared
    ``monkeypatch`` fixture: pytest hands the same MonkeyPatch instance to this
    fixture and to the test using it, so a test that calls ``monkeypatch.undo()``
    mid-body -- the migration-rollback tests do, to restore a patched migration --
    would also revert this redirect and send the next real migration's snapshot to
    the operator's data/backups.
    """
    directory = tmp_path_factory.mktemp("backups")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(backup_module, "BACKUPS_DIR", directory)
    try:
        yield directory
    finally:
        patcher.undo()


@pytest.fixture(autouse=True)
def _clear_streamlit_resource_cache() -> Iterator[None]:
    """Stop one AppTest's cached database from being handed to the next one.

    ``st.cache_resource`` is process-global, and ``app.py`` caches its
    ``PokerDatabase`` there. A UI test that clears the cache on the way IN but
    not on the way out leaves its own sandbox database cached for every later
    AppTest in the process -- still readable through the open SQLite handle long
    after pytest has deleted the tmp_path file behind it. The next AppTest then
    renders against the previous test's sessions: with a session present,
    ``create_hand_form`` stops returning early and the Hands page grows a second
    radio, so ``test_product_shell_navigation_smoke``'s positional lookup finds
    the manual-entry form's control instead of the navigation one.

    That is order-dependent rather than constant, which is what made it a flake:
    running ``test_the_session_hand_list_performs_the_deletion_the_blockers_name``
    immediately before ``test_product_shell_navigation_smoke`` reproduces it in
    about a second, and a shuffled full run reproduces it whenever the two land
    in that order. Clearing after every test makes the cache a per-test resource,
    which is what every test already assumes it is.
    """
    yield
    streamlit = sys.modules.get("streamlit")
    if streamlit is not None:
        streamlit.cache_resource.clear()
