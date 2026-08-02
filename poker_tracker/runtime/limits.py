"""Resource bounds read from the environment, for anything that spawns heavy work.

The rule every function here exists to hold: a limit the operator asked for and
did not get is worse than no limit at all. They believe the host is protected
while a reconstruction or a solve grows until the OOM killer picks a victim,
which on this product means Streamlit and whatever was mid-write. So a value
that cannot be parsed, or a cap that cannot be installed, raises with the
variable named instead of being swallowed and the work started anyway.

The idiom was established on the solver side --
``poker_tracker.solver.jobs.configured_memory_limit_bytes`` and
``poker_tracker.solver.run_job._memory_limiter`` -- and is the same code with
``POKERTRAINER_SOLVER_MEMORY_GB`` hardcoded into its messages. This module is
that code with the variable name as a parameter so the CV side does not become
a second dialect of it; the solver pair should be migrated onto it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

try:
    import resource
except ImportError:  # Windows has no setrlimit at all.
    resource = None  # type: ignore[assignment]


def bounded_int_from_env(
    variable: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read a whole-number setting, or raise naming ``variable``.

    An unset or blank value takes the default. Anything else must parse and sit
    inside the range: a typo that lands outside it is a misconfiguration the
    operator can fix now, and running with the default instead would hide it
    until the bound mattered.
    """
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{variable} must be a whole number from {minimum} to {maximum}, "
            f"with no unit suffix; got {raw!r}."
        ) from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{variable} must be from {minimum} to {maximum}; got {value}."
        )
    return value


def memory_limit_bytes_from_env(variable: str) -> int | None:
    """Return the address-space cap in bytes, or None when the operator set none.

    Every way of asking for a cap that cannot be honoured raises here, while the
    operator can still change the environment, rather than at the point the
    child is already running without it.
    """
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    try:
        limit_gb = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{variable} must be a plain number of gigabytes such as 8 or 6.5, "
            f"with no unit suffix; got {raw!r}."
        ) from exc
    if not math.isfinite(limit_gb) or limit_gb <= 0:
        raise ValueError(
            f"{variable} must be a finite number of gigabytes greater than zero; "
            f"got {raw!r}."
        )
    if os.name != "posix":
        raise ValueError(
            f"{variable} cannot be enforced on this platform; unset it rather "
            "than run while believing a cap is in force."
        )
    return int(limit_gb * 1024**3)


def memory_limiter(limit_bytes: int, *, variable: str) -> Callable[[], None]:
    """Build the between-fork-and-exec hook that caps a child's address space.

    subprocess re-raises whatever this hook raises in the parent, which is the
    point: a cap that could not be applied fails the launch instead of leaving
    the operator believing a cap is in force. Callers have to restate the reason
    themselves, because subprocess reports a hook failure only as a bare
    "Exception occurred in preexec_fn".
    """

    def apply_limit() -> None:
        if resource is None:
            raise RuntimeError(
                f"{variable} was set but this platform has no RLIMIT_AS support."
            )
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return apply_limit


def format_gb(limit_bytes: int) -> str:
    """Render a byte cap the way the operator wrote it in the environment."""
    return f"{limit_bytes / 1024**3:g} GB"
