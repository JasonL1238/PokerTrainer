"""An opt-in pytest plugin that shuffles collection order.

Load it explicitly -- ``python -m pytest -p poker_tracker.suite_quality.random_order
--sq-seed 7`` -- so an ordinary ``pytest`` run is byte-for-byte the run it was
before. A plugin that reorders tests by default would make every other agent's
run different from the one they debugged.

Order is shuffled at two levels, not one: the modules are permuted, and the
tests inside each module are permuted among themselves. A flat shuffle across
all 2600 tests would tear apart module- and package-scoped fixtures -- this
suite has a module-scoped OCR template bank and a per-session state sandbox --
turning a ten-second module into a thousand fixture setups and measuring
nothing but the cost of the shuffle. Keeping each module contiguous still
surfaces the leakage that matters here: state one module leaves behind for
another, and order dependence between tests that share a fixture.

Without a seed the plugin does nothing, so it is safe to leave in a command
line that sometimes wants order and sometimes does not.
"""

from __future__ import annotations

import random
from collections import OrderedDict
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
    items[:] = shuffled(items, seed=seed, key=lambda item: item.nodeid.partition("::")[0])


def pytest_report_header(config: Any) -> str | None:
    seed = config.getoption("sq_seed", default=None)
    if seed is None:
        return None
    return f"suite-quality: collection shuffled with seed {seed}"


def shuffled(items: list[Any], *, seed: int, key: Any) -> list[Any]:
    """Permute groups, and permute each group's members, deterministically.

    Split out from the hook so the ordering rule is testable without running a
    pytest session, and so the same seed always produces the same order on any
    machine -- a flake report is worthless if its seed does not reproduce.
    """
    groups: OrderedDict[Any, list[Any]] = OrderedDict()
    for item in items:
        groups.setdefault(key(item), []).append(item)

    rng = random.Random(seed)
    names = list(groups)
    rng.shuffle(names)

    result: list[Any] = []
    for name in names:
        members = list(groups[name])
        rng.shuffle(members)
        result.extend(members)
    return result
