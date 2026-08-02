"""One implementation of "read a resource limit from the environment".

The family, not the instance: the rule that a limit which cannot be honoured
must be refused rather than silently skipped existed twice. The CV path read it
from ``poker_tracker.runtime.limits``; the solver path had its own copy in
``solver.jobs.configured_memory_limit_bytes`` and ``solver.run_job._memory_limiter``
with ``POKERTRAINER_SOLVER_MEMORY_GB`` written into the messages.

Two copies that agree today are the dangerous shape, not the harmless one. They
were byte-for-byte equivalent on 8, 6.5, " 4 ", "8GB", "eight", "-1", "0",
"nan", "inf" and empty, and a copy nobody is watching drifts first. The symptom
of that drift is the exact defect the previous round repaired: a memory cap the
operator asked for, believed was holding, and did not get.

So this file holds two things:

* the behavioural contract, asserted through the solver's own entry points, so
  the migration cannot have quietly dropped the thread-count validation, the
  platform refusal, or a message that names the variable; and
* a source scan that fails when a third copy appears, in the style of
  ``test_no_consumer_decides_on_is_authoritative_alone`` in
  ``tests/test_phase1_declared_inputs_and_consumers.py``. The scan is what keeps
  this fixed after everyone who remembers the reason has moved on.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poker_tracker.runtime import limits as runtime_limits
from poker_tracker.solver import jobs as solver_jobs
from poker_tracker.solver import run_job as solver_run_job
from poker_tracker.ui import run_cv_job

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every spelling a previous round proved the two copies agreed on, plus the
# blank string. Written once and driven through both variables, because the
# point is that the answer no longer depends on which caller asks.
REJECTED_MEMORY_VALUES = ["8GB", "8 GB", "eight", "0", "-0", "-1", "nan", "inf", "-inf"]
ACCEPTED_MEMORY_VALUES = {"8": 8 * 1024**3, "6.5": int(6.5 * 1024**3), " 4 ": 4 * 1024**3}
UNSET_MEMORY_VALUES = ["", "   "]


@pytest.fixture(autouse=True)
def clear_limit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No assertion here may inherit a bound from the machine it runs on."""
    for variable in (
        solver_jobs.SOLVER_MEMORY_ENV_VAR,
        solver_jobs.SOLVER_THREADS_ENV_VAR,
        run_cv_job.CV_MEMORY_ENV_VAR,
        # resolve_limits validates the timeout as well, so a bad one on this
        # host would fail a memory assertion for the wrong reason.
        run_cv_job.CV_TIMEOUT_ENV_VAR,
    ):
        monkeypatch.delenv(variable, raising=False)


# --- The behaviour the migration had to preserve ----------------------------


@pytest.mark.parametrize("value", REJECTED_MEMORY_VALUES)
def test_the_solver_still_refuses_a_memory_value_it_cannot_honour(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusal, and the variable named, are what the operator acts on."""
    monkeypatch.setenv(solver_jobs.SOLVER_MEMORY_ENV_VAR, value)
    with pytest.raises(ValueError) as rejected:
        solver_jobs.configured_memory_limit_bytes()
    assert solver_jobs.SOLVER_MEMORY_ENV_VAR in str(rejected.value)
    assert repr(value) in str(rejected.value), "the operator cannot see what was rejected"


@pytest.mark.parametrize("value", REJECTED_MEMORY_VALUES)
def test_both_callers_reject_the_same_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CV path and the solver path answer the same question the same way.

    A parity assertion rather than two copies of the table: if one path ever
    starts accepting something the other refuses, the divergence has to fail
    here rather than at the moment a job runs uncapped.
    """
    monkeypatch.setenv(solver_jobs.SOLVER_MEMORY_ENV_VAR, value)
    monkeypatch.setenv(run_cv_job.CV_MEMORY_ENV_VAR, value)
    with pytest.raises(ValueError):
        solver_jobs.configured_memory_limit_bytes()
    with pytest.raises(ValueError):
        run_cv_job.resolve_limits()


@pytest.mark.parametrize(("value", "expected"), sorted(ACCEPTED_MEMORY_VALUES.items()))
def test_both_callers_accept_the_same_values_and_agree_on_the_bytes(
    value: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(solver_jobs.SOLVER_MEMORY_ENV_VAR, value)
    monkeypatch.setenv(run_cv_job.CV_MEMORY_ENV_VAR, value)
    assert solver_jobs.configured_memory_limit_bytes() == expected
    assert run_cv_job.resolve_limits().memory_limit_bytes == expected


@pytest.mark.parametrize("value", UNSET_MEMORY_VALUES)
def test_a_blank_memory_value_means_no_cap_on_both_paths(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(solver_jobs.SOLVER_MEMORY_ENV_VAR, value)
    monkeypatch.setenv(run_cv_job.CV_MEMORY_ENV_VAR, value)
    assert solver_jobs.configured_memory_limit_bytes() is None
    assert run_cv_job.resolve_limits().memory_limit_bytes is None


def test_an_unset_memory_variable_means_no_cap() -> None:
    assert solver_jobs.configured_memory_limit_bytes() is None


def test_a_platform_that_cannot_enforce_the_cap_refuses_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The macOS/Windows difference survived the migration.

    Returning the byte count on a platform that will reject ``RLIMIT_AS`` would
    hand the caller a cap it can only fail to install, which is the silent
    version of this bug rather than the visible one.
    """
    monkeypatch.setenv(solver_jobs.SOLVER_MEMORY_ENV_VAR, "8")
    monkeypatch.setattr(runtime_limits.os, "name", "nt")
    with pytest.raises(ValueError) as refused:
        solver_jobs.configured_memory_limit_bytes()
    assert solver_jobs.SOLVER_MEMORY_ENV_VAR in str(refused.value)
    assert "cannot be enforced on this platform" in str(refused.value)


def test_the_limiter_hook_names_the_solver_variable_when_the_platform_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hook failure reaches the parent as a bare preexec_fn error, so the
    reason has to be inside the exception the hook raises."""
    monkeypatch.setattr(runtime_limits, "resource", None)
    hook = runtime_limits.memory_limiter(
        8 * 1024**3, variable=solver_jobs.SOLVER_MEMORY_ENV_VAR
    )
    with pytest.raises(RuntimeError) as unsupported:
        hook()
    assert solver_jobs.SOLVER_MEMORY_ENV_VAR in str(unsupported.value)


@pytest.mark.skipif(sys.platform == "darwin", reason="Darwin refuses setrlimit(RLIMIT_AS).")
def test_the_solver_limiter_hook_really_caps_address_space() -> None:
    """The shared hook still installs a real cap for the solver's caller."""
    cap = 3 * 1024**3
    completed = subprocess.run(
        [sys.executable, "-c", "import resource;print(resource.getrlimit(resource.RLIMIT_AS)[0])"],
        preexec_fn=runtime_limits.memory_limiter(
            cap, variable=solver_jobs.SOLVER_MEMORY_ENV_VAR
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(completed.stdout.strip()) == cap


@pytest.mark.parametrize("value", ["abc", "2.5", "4GB", ""])
def test_an_unusable_thread_count_is_refused_by_name(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value == "":
        # An empty variable is how a compose file spells "I set this and left it
        # blank". The shared reader treats it as unset, which is what the memory
        # variable has always done; before the migration the solver's private
        # copy raised on it instead. Nothing runs unbounded either way -- the
        # default thread count still applies and is still written into the
        # command file -- so the two variables agreeing is the better answer.
        monkeypatch.setenv(solver_jobs.SOLVER_THREADS_ENV_VAR, value)
        assert 1 <= solver_jobs._configured_thread_count() <= solver_jobs.MAX_SOLVER_THREADS
        return
    monkeypatch.setenv(solver_jobs.SOLVER_THREADS_ENV_VAR, value)
    with pytest.raises(ValueError, match="must be an integer"):
        solver_jobs._configured_thread_count()


@pytest.mark.parametrize("value", ["0", "-1", "5", "64"])
def test_a_thread_count_outside_the_range_is_refused_rather_than_clamped(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TexasSolver reads the count out of the run's own command file, so a
    clamp would put a number in the recorded input the operator never chose."""
    monkeypatch.setenv(solver_jobs.SOLVER_THREADS_ENV_VAR, value)
    with pytest.raises(ValueError) as rejected:
        solver_jobs._configured_thread_count()
    assert solver_jobs.SOLVER_THREADS_ENV_VAR in str(rejected.value)


@pytest.mark.parametrize("value", ["1", "2", " 3 ", "4"])
def test_a_usable_thread_count_is_taken_as_written(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(solver_jobs.SOLVER_THREADS_ENV_VAR, value)
    assert solver_jobs._configured_thread_count() == int(value)


def test_the_default_thread_count_tracks_the_host_and_stays_inside_the_range() -> None:
    threads = solver_jobs._configured_thread_count()
    assert solver_jobs.MIN_SOLVER_THREADS <= threads <= solver_jobs.MAX_SOLVER_THREADS
    assert threads <= max(1, os.cpu_count() or 1)


def test_the_cap_is_rendered_the_way_the_operator_wrote_it() -> None:
    """Both restated messages read back the operator's own number."""
    assert runtime_limits.format_gb(8 * 1024**3) == "8 GB"
    assert runtime_limits.format_gb(int(6.5 * 1024**3)) == "6.5 GB"


# --- The scan that fails when a third copy appears --------------------------

# Where a resource limit may be read out of the environment and turned into a
# number, with the reason. Anything else is a new dialect of the rule.
_ENV_NUMBER_PARSERS = {
    # The one implementation. Every resource bound goes through these two.
    ("poker_tracker/runtime/limits.py", "bounded_int_from_env"),
    ("poker_tracker/runtime/limits.py", "memory_limit_bytes_from_env"),
    # Not a resource bound on a job: SQLite's busy timeout is a connection
    # setting read once at import, and its failure mode is "database is locked"
    # at startup rather than work that runs believing it is bounded.
    ("poker_tracker/persistence/db.py", "<module>"),
    # Retention windows are a deletion policy, not a process bound. It has its
    # own refusal rule -- `_validate_window` rejects a window that would purge
    # immediately -- revalidated on every read rather than only at parse.
    ("poker_tracker/services/retention.py", "from_env"),
    # The login throttle deliberately falls back to its default on a bad value
    # instead of raising, because the alternative to a throttle it cannot parse
    # is no sign-in at all. Not the limits rule; the opposite trade, on purpose.
    ("poker_tracker/ui/login_throttle.py", "_number"),
}

# Where a process limit may actually be installed. One place, because this is
# the call whose failure the caller has to convert into a refusal.
_SETRLIMIT_CALLERS = {("poker_tracker/runtime/limits.py", "apply_limit")}


class _EnvNumberWalker(ast.NodeVisitor):
    """Scopes that read the environment and parse the result as a number.

    Two passes over each scope: names bound from an environment read first,
    then ``int``/``float`` applied to one of them or to an environment read
    inline. Attributing to the innermost function keeps a nested helper from
    hiding under the name of the function that encloses it.
    """

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.env_derived: dict[tuple[str, str], set[str]] = {}
        self.parsers: set[tuple[str, str]] = set()
        self.setrlimit_callers: set[tuple[str, str]] = set()
        self._collecting = True

    def _here(self) -> tuple[str, str]:
        return (self.module, self.scope[-1] if self.scope else "<module>")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._collecting and _reads_environment(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env_derived.setdefault(self._here(), set()).add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "setrlimit":
            self.setrlimit_callers.add(self._here())
        if (
            not self._collecting
            and isinstance(func, ast.Name)
            and func.id in {"int", "float"}
            and node.args
        ):
            argument = node.args[0]
            known = self.env_derived.get(self._here(), set())
            if _reads_environment(argument) or _mentions_any_name(argument, known):
                self.parsers.add(self._here())
        self.generic_visit(node)

    def run(self, tree: ast.AST) -> None:
        self._collecting = True
        self.visit(tree)
        self._collecting = False
        self.visit(tree)


def _reads_environment(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "environ":
            return True
        if isinstance(child, ast.Attribute) and child.attr == "getenv":
            return True
    return False


def _mentions_any_name(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in names for child in ast.walk(node)
    )


def _scan() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Walk the product source: app.py plus every module under poker_tracker.

    Same reach as the declared-consumers scan. ``cv_lab`` is a research record
    rather than a runtime path and is excluded for the same reason it is there.
    """
    parsers: set[tuple[str, str]] = set()
    setrlimit_callers: set[tuple[str, str]] = set()
    for path in [REPO_ROOT / "app.py", *sorted((REPO_ROOT / "poker_tracker").rglob("*.py"))]:
        walker = _EnvNumberWalker(str(path.relative_to(REPO_ROOT)))
        walker.run(ast.parse(path.read_text(encoding="utf-8")))
        parsers |= walker.parsers
        setrlimit_callers |= walker.setrlimit_callers
    return parsers, setrlimit_callers


def test_no_second_implementation_of_a_resource_limit_read() -> None:
    """The family regression: a third copy fails here instead of on a host.

    The solver's copy and the CV path's copy were provably identical the day
    this was written, which is exactly why it needed fixing -- nothing would
    have reported the day they stopped being identical, and the report would
    have been a job that ran uncapped while its assumptions line said it did
    not. Adding an entry below means stating why that reader is not this rule.
    """
    parsers, _ = _scan()
    assert parsers == _ENV_NUMBER_PARSERS


def test_only_the_shared_module_installs_a_process_limit() -> None:
    """A caller that calls setrlimit itself owns the "it was refused" case
    itself, and that is the case every version of this bug got wrong."""
    _, setrlimit_callers = _scan()
    assert setrlimit_callers == _SETRLIMIT_CALLERS


def test_the_solver_worker_holds_no_private_limiter() -> None:
    """The named copy is gone, not shadowed by an import of the same name."""
    assert not hasattr(solver_run_job, "_memory_limiter")
    assert solver_run_job.memory_limiter is runtime_limits.memory_limiter
