"""Phase 8 failure injection: the solver's failure matrix, driven through the worker.

PLAN.md requires timeout, cancellation, process-group termination, missing
output, corrupt output and memory exhaustion to be tested. Every solver test
that reached process termination replaced it with a recorder, so none of them
proved a solver ever dies. These drive stub binaries that genuinely misbehave --
sleep past the deadline, ignore SIGTERM until SIGKILL, fork a child that
outlives its parent, exit 0 writing nothing, write truncated JSON, write a
result and then delete it -- through run_solver_job.

The invariant is not that a run succeeds. It is that a failed run is terminal,
says why in words an operator can act on, retains no partial strategy as though
it were an answer, and leaves nothing still solving -- checked against the
process group, because the process that survives is the one the solve forked,
not the one the worker launched.

Modelled on tests/test_job_failure_injection.py, which does this for the CV
pipeline; the solver should not grow a second failure-testing idiom.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Hand, ProcessingJob, Session, SolverRun, VideoRecord
from poker_tracker.solver import jobs, run_job
from poker_tracker.solver.jobs import (
    SolverJobAlreadyRunningError,
    cancel_solver_run,
    configured_memory_limit_bytes,
    start_solver_job,
)
from poker_tracker.solver.models import ResolvedRange, SolverSpot
from poker_tracker.solver.run_job import run_solver_job

# Above every macOS and Linux pid, so a cancellation test can name a worker
# without any chance of signalling something real.
_UNUSED_PID = 4_194_303

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the solver's process supervision is POSIX-only"
)


# --- Stub solvers -----------------------------------------------------------

# Each stub runs from the run directory, so it records the pids it creates as
# plain files the test can read after the worker has given up on it.

HANGS_PAST_THE_DEADLINE = """
import os, pathlib, time
pathlib.Path("solver.pid").write_text(str(os.getpid()))
time.sleep(300)
"""

IGNORES_SIGTERM = """
import os, pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path("solver.pid").write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""

FORKS_A_CHILD_THAT_OUTLIVES_IT = """
import os, pathlib, subprocess, sys, time
forked = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time\\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    "while True: time.sleep(0.05)",
])
pathlib.Path("forked.pid").write_text(str(forked.pid))
pathlib.Path("solver.pid").write_text(str(os.getpid()))
time.sleep(300)
"""

EXITS_ZERO_WRITING_NOTHING = """
pass
"""

WRITES_TRUNCATED_JSON = """
import pathlib
pathlib.Path("result.json").write_text('{"strategy": {"actions": ["CHE')
"""

WRITES_NON_JSON_BYTES = """
import pathlib
pathlib.Path("result.json").write_bytes(b"\\x00\\x01segmentation fault\\n")
"""

WRITES_A_RESULT_THEN_DELETES_IT = """
import json, pathlib
result = pathlib.Path("result.json")
result.write_text(json.dumps({"strategy": {"actions": ["CHECK"], "strategy": {}}}))
result.unlink()
"""

WRITES_AN_EMPTY_DUMP = """
import pathlib
pathlib.Path("result.json").write_text("{}")
"""

EXITS_NONZERO = """
import sys
sys.stdout.write("terminate called after throwing an instance of 'std::bad_alloc'\\n")
sys.exit(3)
"""

REPORTS_ITS_ADDRESS_SPACE_LIMIT = """
import json, pathlib, resource
pathlib.Path("rlimit.txt").write_text(str(resource.getrlimit(resource.RLIMIT_AS)[0]))
pathlib.Path("result.json").write_text(json.dumps({
    "node_type": "action_node",
    "actions": ["CHECK", "BET 3.75"],
    "childrens": {},
    "strategy": {
        "actions": ["CHECK", "BET 3.75"],
        "strategy": {"AhQs": [0.7, 0.3]},
    },
}))
print("Total exploitability 0.33 percent")
"""

ALLOCATES_UNTIL_IT_IS_KILLED = """
import pathlib
pathlib.Path("solver.pid").write_text("allocating")
held = []
while True:
    held.append(bytearray(256 * 1024 * 1024))
"""


# --- Workspace --------------------------------------------------------------


def _install_stub_solver(tmp_path: Path, script: str, monkeypatch) -> Path:
    binary = tmp_path / "console_solver"
    binary.write_text(f"#!{sys.executable}\n{script}", encoding="utf-8")
    binary.chmod(0o755)
    resources = tmp_path / "resources"
    (resources / "compairer").mkdir(parents=True, exist_ok=True)
    (resources / "compairer" / "card5_dic_sorted.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(binary))
    monkeypatch.setenv("TEXAS_SOLVER_RESOURCE_DIR", str(resources))
    return binary


def _solver_spot(hand_id: int) -> dict[str, object]:
    return {
        "hand_id": hand_id,
        "table_size": 6,
        "street": "flop",
        "board": "Ah7d2c",
        "pot": 10.0,
        "effective_stack": 90.0,
        "pot_type": "single_raised",
        "preflop_aggressor_key": "hero",
        "oop": {
            "player_key": "hero",
            "player_name": "Hero",
            "position": "BB",
            "role": "oop",
            "is_hero": True,
        },
        "ip": {
            "player_key": "villain",
            "player_name": "Villain",
            "position": "BTN",
            "role": "ip",
        },
        "hero_cards": "AhQs",
        "recorded_line": [
            {
                "player_key": "hero",
                "player_name": "Hero",
                "street": "flop",
                "action_type": "check",
            }
        ],
    }


def _resolved_range(player_key: str, role: str) -> dict[str, object]:
    return {
        "player_key": player_key,
        "player_name": player_key.title(),
        "position": "BB" if role == "oop" else "BTN",
        "role": role,
        "source": "custom",
        "profile_name": "Failure injection range",
        "notation": "AhQs",
        "solver_notation": "AhQs",
        "combo_count": 1,
        "range_percent": 0.001,
    }


def _made(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_database(tmp_path: Path) -> tuple[PokerDatabase, Path, int]:
    db_path = tmp_path / "solver.db"
    db = PokerDatabase(db_path)
    db.init_db()
    session = db.create_session(Session(name="Solver failure injection"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="NLHE cash",
            table_size=6,
        )
    )
    assert hand.id is not None
    return db, db_path, hand.id


def _queued_run(db: PokerDatabase, hand_id: int, run_dir: Path) -> SolverRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    command_path = run_dir / "input.txt"
    command_path.write_text("set_pot 10\ndump_result result.json\n", encoding="utf-8")
    return db.create_solver_run(
        SolverRun(
            hand_id=hand_id,
            input_hash="f" * 64,
            backend_version="stub",
            spot=_solver_spot(hand_id),
            range_ip=_resolved_range("villain", "ip"),
            range_oop=_resolved_range("hero", "oop"),
            command_path=str(command_path),
            result_path=str(run_dir / "result.json"),
            log_path=str(run_dir / "solver.log"),
        )
    )


@pytest.fixture
def solver_workspace(tmp_path: Path):
    """A queued run whose stub binary is chosen per test."""

    def build(script: str) -> tuple[PokerDatabase, Path, Path, int]:
        db, db_path, hand_id = _make_database(tmp_path)
        run_dir = tmp_path / "run_1"
        run = _queued_run(db, hand_id, run_dir)
        assert run.id is not None
        return db, db_path, run_dir, run.id

    return build


# --- Shared assertions ------------------------------------------------------


def _assert_safe_terminal_state(
    db_path: Path, run_id: int, *, status: str = "failed"
) -> SolverRun:
    """A finished-badly run is terminal, explains itself, and keeps no answer."""
    db = PokerDatabase(db_path)
    try:
        saved = db.fetch_solver_run(run_id)
        assert saved is not None
        assert saved.status == status, f"run left in {saved.status!r}"
        assert saved.pid is None, "a terminal run must not still claim a pid"
        assert not saved.evidence, "a failed run must not retain strategy evidence"
        assert saved.exploitability_pct is None
        if status == "failed":
            message = (saved.error_message or "").strip()
            assert message, "a failure must say why"
            assert "\n" not in message
        return saved
    finally:
        db.close()


def _recorded_pid(run_dir: Path, name: str, *, timeout: float = 10.0) -> int:
    path = run_dir / name
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            time.sleep(0.02)
    raise AssertionError(f"the stub solver never recorded {name}")


def _is_running(pid: int) -> bool:
    """Whether a pid is still a live process rather than an unreaped corpse."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "darwin" or Path("/proc").is_dir():
        return not _is_zombie(pid)
    return True


def _is_zombie(pid: int) -> bool:
    state = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    return state.stdout.strip().startswith("Z")


def _assert_stopped(pid: int, *, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    raise AssertionError(f"{what} (pid {pid}) survived the worker")


# --- Timeout ----------------------------------------------------------------


def test_a_solver_that_hangs_past_the_deadline_is_killed_and_the_run_fails(
    solver_workspace, tmp_path, monkeypatch
):
    _install_stub_solver(tmp_path, HANGS_PAST_THE_DEADLINE, monkeypatch)
    db, db_path, run_dir, run_id = solver_workspace(HANGS_PAST_THE_DEADLINE)

    # One second, not zero: the deadline has to fall after the solver is up,
    # or the test proves only that a process can be killed before it starts.
    with pytest.raises(TimeoutError) as excinfo:
        run_solver_job(db, run_id, timeout_seconds=1)

    solver_pid = _recorded_pid(run_dir, "solver.pid")
    assert "timeout" in str(excinfo.value).lower()
    _assert_stopped(solver_pid, what="the hung solver")
    saved = _assert_safe_terminal_state(db_path, run_id)
    assert "timeout" in saved.error_message.lower()
    assert "could not be confirmed stopped" not in saved.error_message


def test_a_solver_that_ignores_sigterm_is_escalated_to_sigkill(
    solver_workspace, tmp_path, monkeypatch
):
    """SIGTERM alone leaves a solve holding its memory for the rest of the day."""
    _install_stub_solver(tmp_path, IGNORES_SIGTERM, monkeypatch)
    monkeypatch.setattr(run_job, "TERMINATION_GRACE_SECONDS", 0.5)
    db, db_path, run_dir, run_id = solver_workspace(IGNORES_SIGTERM)

    with pytest.raises(TimeoutError):
        run_solver_job(db, run_id, timeout_seconds=1)

    _assert_stopped(
        _recorded_pid(run_dir, "solver.pid"), what="the SIGTERM-ignoring solver"
    )
    _assert_safe_terminal_state(db_path, run_id)


def test_a_solver_that_forks_leaves_no_process_behind(
    solver_workspace, tmp_path, monkeypatch
):
    """Process-group termination, not direct-child termination.

    The stub dies on SIGTERM like a well-behaved binary. Its forked child does
    not, and is reparented away the moment the stub goes. Signalling only the
    pid the worker launched reports success while that child keeps running.
    """
    _install_stub_solver(tmp_path, FORKS_A_CHILD_THAT_OUTLIVES_IT, monkeypatch)
    monkeypatch.setattr(run_job, "TERMINATION_GRACE_SECONDS", 1.0)
    db, db_path, run_dir, run_id = solver_workspace(FORKS_A_CHILD_THAT_OUTLIVES_IT)

    with pytest.raises(TimeoutError):
        run_solver_job(db, run_id, timeout_seconds=1)

    _assert_stopped(_recorded_pid(run_dir, "solver.pid"), what="the solver")
    _assert_stopped(
        _recorded_pid(run_dir, "forked.pid"), what="the process the solver forked"
    )
    _assert_safe_terminal_state(db_path, run_id)


# --- Output that is missing, corrupt, or empty -------------------------------


@pytest.mark.parametrize(
    ("label", "script", "expected"),
    [
        ("nothing written", EXITS_ZERO_WRITING_NOTHING, "without writing a result"),
        ("written then deleted", WRITES_A_RESULT_THEN_DELETES_IT, "without writing a result"),
        ("truncated json", WRITES_TRUNCATED_JSON, ""),
        ("non-json bytes", WRITES_NON_JSON_BYTES, ""),
        ("semantically empty dump", WRITES_AN_EMPTY_DUMP, ""),
        ("nonzero exit", EXITS_NONZERO, "exited with status 3"),
    ],
)
def test_unusable_solver_output_fails_the_run(
    solver_workspace, tmp_path, monkeypatch, label, script, expected
):
    """None of these may be retained as a strategy the product will explain."""
    _install_stub_solver(tmp_path, script, monkeypatch)
    db, db_path, _run_dir, run_id = solver_workspace(script)

    with pytest.raises((RuntimeError, ValueError)):
        run_solver_job(db, run_id, timeout_seconds=30)

    saved = _assert_safe_terminal_state(db_path, run_id)
    assert expected in saved.error_message, f"{label}: {saved.error_message!r}"


def test_a_worker_that_fails_after_launching_the_solver_does_not_orphan_it(
    solver_workspace, tmp_path, monkeypatch
):
    """The solver outlives the worker unless every exit path reaps it.

    A store failure between launching the solve and claiming the run used to
    return through a handler that never touched the process, leaving a
    multi-gigabyte solve running with its pid recorded nowhere.
    """
    _install_stub_solver(tmp_path, HANGS_PAST_THE_DEADLINE, monkeypatch)
    monkeypatch.setattr(run_job, "TERMINATION_GRACE_SECONDS", 1.0)
    db, db_path, run_dir, run_id = solver_workspace(HANGS_PAST_THE_DEADLINE)
    original = db.update_solver_run

    def explode_on_claim(target_id, **changes):
        if changes.get("status") == "running":
            _recorded_pid(run_dir, "solver.pid")  # fail once the solve is up
            raise RuntimeError("the store rejected the claim")
        return original(target_id, **changes)

    monkeypatch.setattr(db, "update_solver_run", explode_on_claim)
    with pytest.raises(RuntimeError, match="rejected the claim"):
        run_solver_job(db, run_id, timeout_seconds=30)

    _assert_stopped(_recorded_pid(run_dir, "solver.pid"), what="the abandoned solver")


# --- Cancellation ------------------------------------------------------------


def test_a_run_cancelled_mid_solve_lands_stale_and_stops_the_solver(
    solver_workspace, tmp_path, monkeypatch
):
    _install_stub_solver(tmp_path, HANGS_PAST_THE_DEADLINE, monkeypatch)
    monkeypatch.setattr(run_job, "TERMINATION_GRACE_SECONDS", 1.0)
    db, db_path, run_dir, run_id = solver_workspace(HANGS_PAST_THE_DEADLINE)
    solver_pid_holder: list[int] = []

    def request_cancellation() -> None:
        solver_pid_holder.append(_recorded_pid(run_dir, "solver.pid"))
        canceller = PokerDatabase(db_path)
        try:
            canceller.update_solver_run(run_id, status="cancelling")
        finally:
            canceller.close()

    canceller = threading.Thread(target=request_cancellation)
    canceller.start()
    run_solver_job(db, run_id, timeout_seconds=120)
    canceller.join(timeout=10)

    assert solver_pid_holder
    _assert_stopped(solver_pid_holder[0], what="the cancelled solver")
    _assert_safe_terminal_state(db_path, run_id, status="stale")


def test_cancelling_a_queued_run_with_no_worker_yet_cancels_it_outright(tmp_path):
    db, db_path, hand_id = _make_database(tmp_path)
    try:
        run = _queued_run(db, hand_id, tmp_path / "run_queued")
        assert run.pid is None
        cancelled = cancel_solver_run(db, run.id)
        assert cancelled.status == "cancelled"
        assert cancelled.pid is None
        assert "Cancelled by user." in cancelled.error_message
    finally:
        db.close()


def test_a_cancel_the_host_refuses_parks_the_run_instead_of_claiming_success(
    tmp_path, monkeypatch
):
    """A refused signal is the one case the OS will not let a test produce.

    Only the kill call is replaced, and only with the refusal the kernel would
    return for a process this user may not signal. The decision under test --
    that a cancellation which could not stop anything must not report the run
    as cancelled -- is the real one.
    """
    db, db_path, hand_id = _make_database(tmp_path)
    try:
        run = _queued_run(db, hand_id, tmp_path / "run_refused")
        running = db.update_solver_run(run.id, status="running", pid=_UNUSED_PID)

        def refuse(*_args, **_kwargs):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(jobs.os, "killpg", refuse)
        monkeypatch.setattr(jobs.os, "kill", refuse)
        parked = cancel_solver_run(db, running.id)

        assert parked.status == "cancelling"
        assert "waiting for the solver worker to stop" in parked.error_message
    finally:
        db.close()


def test_a_cancel_after_the_run_already_finished_does_not_revive_it(
    tmp_path, monkeypatch
):
    """Parking a terminal row back in 'cancelling' would resurrect a dead run."""
    db, db_path, hand_id = _make_database(tmp_path)
    try:
        run = _queued_run(db, hand_id, tmp_path / "run_finished")
        db.update_solver_run(run.id, status="running", pid=_UNUSED_PID)

        def refuse(*_args, **_kwargs):
            # The worker finishes in the window between reading the run and
            # signalling it, which is the only way this race can be staged.
            db.update_solver_run(run.id, status="failed", error_message="already over")
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(jobs.os, "killpg", refuse)
        monkeypatch.setattr(jobs.os, "kill", refuse)
        settled = cancel_solver_run(db, run.id)

        assert settled.status == "failed"
        assert settled.error_message == "already over"
    finally:
        db.close()


# --- Process-group termination, without the worker in the way ----------------


def _spawn_group(child_script: str) -> tuple[subprocess.Popen[bytes], int]:
    """A session leader with one forked child, as a solve's tree looks."""
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(300)\n",
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().decode().strip())
    return leader, child_pid


SLEEPS = "import time\nwhile True: time.sleep(0.05)"
IGNORES_TERM = (
    "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "while True: time.sleep(0.05)"
)


@pytest.mark.parametrize(
    ("label", "child_script"),
    [("cooperative child", SLEEPS), ("child ignoring SIGTERM", IGNORES_TERM)],
)
def test_terminating_a_solver_group_reaps_the_whole_tree(label, child_script):
    leader, child_pid = _spawn_group(child_script)
    try:
        assert jobs._terminate_solver_group(leader.pid, grace_seconds=1.0)
        _assert_stopped(child_pid, what=f"the {label}")
        leader.wait(timeout=10)
    finally:
        for pid in (leader.pid, child_pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        leader.wait(timeout=10)


# --- Memory --------------------------------------------------------------


def _address_space_limit_is_enforceable() -> bool:
    """Whether this host actually honours RLIMIT_AS; macOS rejects it outright."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import resource\n"
            "resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))\n",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


ENFORCEABLE = _address_space_limit_is_enforceable()


@pytest.mark.parametrize("value", ["8GB", "8 GB", "eight", "0", "-1", "nan", "inf"])
def test_a_memory_cap_that_cannot_be_parsed_is_refused_before_a_run_exists(
    tmp_path, monkeypatch, value
):
    """The operator has to learn the cap is not real while they can still fix it.

    Swallowing the parse error left console_solver running with no RLIMIT_AS
    while the run looked ordinary, so a deep river tree grew until the host OOM
    killer took Streamlit and whatever was mid-write with it.
    """
    _install_stub_solver(tmp_path, EXITS_ZERO_WRITING_NOTHING, monkeypatch)
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", value)
    monkeypatch.setattr(
        "poker_tracker.solver.jobs.solver_run_directory",
        lambda run_id: _made(tmp_path / f"run_{run_id}"),
    )
    db, db_path, hand_id = _make_database(tmp_path)
    try:
        spot = SolverSpot.model_validate(_solver_spot(hand_id))
        with pytest.raises(ValueError, match="POKERTRAINER_SOLVER_MEMORY_GB"):
            start_solver_job(
                db,
                spot,
                ResolvedRange.model_validate(_resolved_range("villain", "ip")),
                ResolvedRange.model_validate(_resolved_range("hero", "oop")),
            )
        assert db.fetch_active_solver_runs() == []
        assert db.fetch_solver_runs_by_hand(hand_id) == []
    finally:
        db.close()


def test_an_unset_memory_cap_stays_unset(monkeypatch):
    monkeypatch.delenv("POKERTRAINER_SOLVER_MEMORY_GB", raising=False)
    assert configured_memory_limit_bytes() is None
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", "  ")
    assert configured_memory_limit_bytes() is None
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", "6.5")
    assert configured_memory_limit_bytes() == int(6.5 * 1024**3)


@pytest.mark.skipif(not ENFORCEABLE, reason="this host rejects RLIMIT_AS")
def test_a_configured_memory_cap_is_in_force_inside_the_solver(
    solver_workspace, tmp_path, monkeypatch
):
    _install_stub_solver(tmp_path, REPORTS_ITS_ADDRESS_SPACE_LIMIT, monkeypatch)
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", "2")
    db, db_path, run_dir, run_id = solver_workspace(REPORTS_ITS_ADDRESS_SPACE_LIMIT)

    run_solver_job(db, run_id, timeout_seconds=30)

    observed = int((run_dir / "rlimit.txt").read_text(encoding="utf-8").strip())
    assert observed == 2 * 1024**3, "the child ran without the configured cap"


@pytest.mark.skipif(not ENFORCEABLE, reason="this host rejects RLIMIT_AS")
def test_a_solver_killed_by_its_memory_cap_fails_visibly(
    solver_workspace, tmp_path, monkeypatch
):
    _install_stub_solver(tmp_path, ALLOCATES_UNTIL_IT_IS_KILLED, monkeypatch)
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", "1")
    db, db_path, _run_dir, run_id = solver_workspace(ALLOCATES_UNTIL_IT_IS_KILLED)

    with pytest.raises(RuntimeError, match="exited with status"):
        run_solver_job(db, run_id, timeout_seconds=60)

    _assert_safe_terminal_state(db_path, run_id)


@pytest.mark.skipif(ENFORCEABLE, reason="this host enforces RLIMIT_AS")
def test_a_memory_cap_this_host_cannot_apply_stops_the_solve(
    solver_workspace, tmp_path, monkeypatch
):
    """Never run uncapped while the operator believes a cap is holding."""
    _install_stub_solver(tmp_path, REPORTS_ITS_ADDRESS_SPACE_LIMIT, monkeypatch)
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", "2")
    db, db_path, run_dir, run_id = solver_workspace(REPORTS_ITS_ADDRESS_SPACE_LIMIT)

    with pytest.raises(RuntimeError, match="POKERTRAINER_SOLVER_MEMORY_GB"):
        run_solver_job(db, run_id, timeout_seconds=30)

    assert not (run_dir / "rlimit.txt").exists(), "the solver ran without its cap"
    saved = _assert_safe_terminal_state(db_path, run_id)
    assert "2 GB" in saved.error_message


def test_an_applied_memory_cap_is_recorded_on_the_run(tmp_path, monkeypatch):
    """A cap nobody can read afterwards is a cap nobody can audit."""
    _install_stub_solver(tmp_path, EXITS_ZERO_WRITING_NOTHING, monkeypatch)
    monkeypatch.setenv("POKERTRAINER_SOLVER_MEMORY_GB", "3")
    monkeypatch.setattr(
        "poker_tracker.solver.jobs.solver_run_directory",
        lambda run_id: _made(tmp_path / f"run_{run_id}"),
    )

    class DummyProcess:
        pid = 424242

    monkeypatch.setattr(
        "poker_tracker.solver.jobs.subprocess.Popen",
        lambda *args, **kwargs: DummyProcess(),
    )
    db, db_path, hand_id = _make_database(tmp_path)
    try:
        run = start_solver_job(
            db,
            SolverSpot.model_validate(_solver_spot(hand_id)),
            ResolvedRange.model_validate(_resolved_range("villain", "ip")),
            ResolvedRange.model_validate(_resolved_range("hero", "oop")),
        )
        assert any("capped at 3 GB" in item for item in run.assumptions)
    finally:
        db.close()


# --- One heavy job at a time -------------------------------------------------


def test_a_solve_is_refused_while_a_reconstruction_job_is_active(tmp_path, monkeypatch):
    """The untested direction of the reservation: CV holds, the solver waits."""
    _install_stub_solver(tmp_path, EXITS_ZERO_WRITING_NOTHING, monkeypatch)
    db, db_path, hand_id = _make_database(tmp_path)
    try:
        video = db.create_video(
            VideoRecord(
                original_filename="session.mov",
                stored_path=str(tmp_path / "session.mov"),
                file_size_bytes=1,
            )
        )
        job = db.create_processing_job(
            ProcessingJob(
                video_id=video.id,
                job_type="cv_reconstruction",
                status="running",
            )
        )
        with pytest.raises(SolverJobAlreadyRunningError) as excinfo:
            start_solver_job(
                db,
                SolverSpot.model_validate(_solver_spot(hand_id)),
                ResolvedRange.model_validate(_resolved_range("villain", "ip")),
                ResolvedRange.model_validate(_resolved_range("hero", "oop")),
            )
        message = str(excinfo.value)
        assert "cv reconstruction" in message
        assert f"#{job.id}" in message
        assert db.fetch_solver_runs_by_hand(hand_id) == []
    finally:
        db.close()


# --- Reconciliation must not leave a solve running ---------------------------


def test_reconciling_a_dead_worker_also_stops_the_solver_it_left_behind(tmp_path):
    """The worker can be killed outright; the detached solve then has no parent.

    Its pid is recorded beside the run's command file precisely so this path can
    still reach it, the same way the reconstruction pipeline's sidecar pid is
    recorded.
    """
    db, db_path, hand_id = _make_database(tmp_path)
    orphan = subprocess.Popen(
        [sys.executable, "-c", SLEEPS], start_new_session=True
    )
    try:
        run_dir = tmp_path / "run_orphaned"
        run = _queued_run(db, hand_id, run_dir)
        jobs.solver_child_pid_path(run_dir).write_text(
            str(orphan.pid), encoding="utf-8"
        )
        db.update_solver_run(
            run.id,
            status="running",
            pid=999_999,
            heartbeat_at=None,
        )
        failed = jobs.reconcile_stale_solver_runs(db)

        assert failed == [run.id]
        _assert_stopped(orphan.pid, what="the orphaned solver")
        assert not jobs.solver_child_pid_path(run_dir).exists()
    finally:
        try:
            os.kill(orphan.pid, 9)
        except OSError:
            pass
        orphan.wait(timeout=10)
        db.close()


def test_the_worker_removes_its_pid_file_when_the_solve_ends(
    solver_workspace, tmp_path, monkeypatch
):
    """A stale pid file is a loaded gun aimed at whatever inherits that pid."""
    _install_stub_solver(tmp_path, EXITS_ZERO_WRITING_NOTHING, monkeypatch)
    db, db_path, run_dir, run_id = solver_workspace(EXITS_ZERO_WRITING_NOTHING)

    with pytest.raises(RuntimeError):
        run_solver_job(db, run_id, timeout_seconds=30)

    assert not jobs.solver_child_pid_path(run_dir).exists()


def test_a_failed_run_retains_no_strategy_json_as_an_answer(
    solver_workspace, tmp_path, monkeypatch
):
    """The dump can survive on disk; what must not survive is calling it evidence."""
    _install_stub_solver(tmp_path, WRITES_AN_EMPTY_DUMP, monkeypatch)
    db, db_path, run_dir, run_id = solver_workspace(WRITES_AN_EMPTY_DUMP)

    with pytest.raises(ValueError):
        run_solver_job(db, run_id, timeout_seconds=30)

    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8")) == {}
    saved = _assert_safe_terminal_state(db_path, run_id)
    assert not saved.evidence
