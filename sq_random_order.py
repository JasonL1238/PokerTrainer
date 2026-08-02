"""The `-p` entrypoint for the suite-quality collection shuffle.

The shuffle itself lives in ``poker_tracker.suite_quality.random_order``. This
module exists because of *when* pytest loads a ``-p`` plugin: during argument
preparsing, before any conftest runs. Naming a module inside the
``poker_tracker`` package there executes ``poker_tracker/__init__.py`` first,
which imports ``persistence.db`` and ``ui.video_storage`` -- the modules that
resolve the operator's database and data directory once, at import. That
happens before ``tests/conftest.py`` can redirect them, so a shuffled run read
and migrated ``<repo>/poker_tracker.db``: the round-12 hazard, reintroduced
through the plugin loader rather than through the app.

So this module imports nothing at load time. ``pytest_addoption`` needs only a
string, and the ordering rule is imported inside the collection hook, which
runs long after conftest has claimed the variables.
"""

from __future__ import annotations

from typing import Any

SEED_OPTION = "--sq-seed"


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("suite-quality")
    group.addoption(
        SEED_OPTION,
        action="store",
        default=None,
        type=int,
        help="Shuffle module order and within-module test order with this seed.",
    )


def pytest_collection_modifyitems(session: Any, config: Any, items: list[Any]) -> None:
    seed = config.getoption("sq_seed", default=None)
    if seed is None:
        return
    # Imported here, not at module scope: at module scope this is the defect
    # the file's docstring describes.
    from poker_tracker.suite_quality.random_order import shuffled

    items[:] = shuffled(items, seed=seed, key=lambda item: item.nodeid.partition("::")[0])


def pytest_report_header(config: Any) -> str | None:
    seed = config.getoption("sq_seed", default=None)
    if seed is None:
        return None
    return f"suite-quality: collection shuffled with seed {seed}"
