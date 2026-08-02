from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import SolverRun
from poker_tracker.runtime.limits import (
    bounded_int_from_env,
    format_gb,
    memory_limit_bytes_from_env,
)
from poker_tracker.solver.models import ResolvedRange, SolverSpot
from poker_tracker.solver.storage import solver_run_directory
from poker_tracker.solver.texassolver import (
    BACKEND_NAME,
    PINNED_CONSOLE_COMMIT,
    build_command_file,
    configured_binary,
    configured_resource_dir,
    input_hash,
    parse_strategy_result,
)

DEFAULT_STALE_AFTER = timedelta(minutes=5)
SOLVER_CHILD_PID_FILE = "solver_child.pid"
# Named once here because the worker restates both of them to the operator, and
# a message that names a variable the reader cannot find is worse than no
# message. The CV path holds its own pair the same way in ui/run_cv_job.py.
SOLVER_MEMORY_ENV_VAR = "POKERTRAINER_SOLVER_MEMORY_GB"
SOLVER_THREADS_ENV_VAR = "POKERTRAINER_SOLVER_THREADS"
MIN_SOLVER_THREADS = 1
MAX_SOLVER_THREADS = 4


class SolverJobAlreadyRunningError(RuntimeError):
    pass


def start_solver_job(
    db: PokerDatabase,
    spot: SolverSpot,
    range_ip: ResolvedRange,
    range_oop: ResolvedRange,
    *,
    assumptions: list[str] | None = None,
) -> SolverRun:
    binary = configured_binary()
    configured_resource_dir(binary)
    thread_count = _configured_thread_count()
    memory_limit_bytes = configured_memory_limit_bytes()
    digest = input_hash(spot, range_ip, range_oop)
    combined_assumptions = list(
        dict.fromkeys(
            [
                "Heads-up postflop TexasSolver strategy.",
                "No-rake equilibrium approximation.",
                "Built-in betting-size abstraction.",
                (
                    "Suit isomorphism is disabled because board/Hero blocker filtering "
                    "uses exact suit-specific combinations."
                ),
                (
                    f"IP range '{range_ip.profile_name}' is an estimated study input, "
                    "not solved preflop GTO."
                    if range_ip.source in {"default", "builtin"}
                    else f"IP range '{range_ip.profile_name}' was explicitly supplied by the user."
                ),
                (
                    f"OOP range '{range_oop.profile_name}' is an estimated study input, "
                    "not solved preflop GTO."
                    if range_oop.source in {"default", "builtin"}
                    else f"OOP range '{range_oop.profile_name}' was explicitly supplied by the user."
                ),
                *(
                    [f"Solver process memory was capped at {format_gb(memory_limit_bytes)}."]
                    if memory_limit_bytes is not None
                    else []
                ),
                *(assumptions or []),
                *range_ip.mismatches,
                *range_oop.mismatches,
            ]
        )
    )
    cached_source: SolverRun | None = None
    with db.transaction(immediate=True):
        cached = db.fetch_cached_solver_run(digest)
        active_solver = db.fetch_active_solver_runs()
        if active_solver:
            current = active_solver[0]
            raise SolverJobAlreadyRunningError(
                f"Solver run #{current.id} is already {current.status}."
            )
        active_jobs = db.fetch_active_jobs()
        if active_jobs:
            current = active_jobs[0]
            raise SolverJobAlreadyRunningError(
                f"Offline {current.job_type.replace('_', ' ')} job #{current.id} is "
                f"already {current.status}."
            )
        if cached is not None and Path(cached.result_path).is_file():
            cache_assumptions = [
                *combined_assumptions,
                f"Reused strategy artifacts from solver run #{cached.id}.",
            ]
            run = db.create_solver_run(
                SolverRun(
                    hand_id=spot.hand_id,
                    status="queued",
                    backend_name=BACKEND_NAME,
                    backend_version=PINNED_CONSOLE_COMMIT,
                    input_hash=digest,
                    spot=spot.model_dump(mode="json"),
                    range_ip=range_ip.model_dump(mode="json"),
                    range_oop=range_oop.model_dump(mode="json"),
                    assumptions=cache_assumptions,
                    exploitability_pct=cached.exploitability_pct,
                    runtime_seconds=cached.runtime_seconds,
                )
            )
            cached_source = cached
        else:
            run = db.create_solver_run(
                SolverRun(
                    hand_id=spot.hand_id,
                    status="queued",
                    backend_name=BACKEND_NAME,
                    backend_version=PINNED_CONSOLE_COMMIT,
                    input_hash=digest,
                    spot=spot.model_dump(mode="json"),
                    range_ip=range_ip.model_dump(mode="json"),
                    range_oop=range_oop.model_dump(mode="json"),
                    assumptions=combined_assumptions,
                )
            )
    if run.id is None:
        raise RuntimeError("The solver run could not be saved.")
    if cached_source is not None:
        try:
            return _materialize_cached_run(
                db,
                run,
                cached_source,
                spot=spot,
                range_ip=range_ip,
                range_oop=range_oop,
                thread_count=thread_count,
            )
        except Exception as exc:
            _fail_prelaunch_run(
                db,
                run.id,
                f"Could not materialize cached solver run: {exc}",
            )
            raise

    try:
        run_dir = solver_run_directory(run.id)
        command_path = run_dir / "input.txt"
        result_path = run_dir / "result.json"
        log_path = run_dir / "solver.log"
        command_path.write_text(
            build_command_file(
                spot,
                range_ip,
                range_oop,
                output_path=result_path,
                thread_count=thread_count,
            ),
            encoding="utf-8",
        )
        run = db.update_solver_run(
            run.id,
            command_path=str(command_path),
            result_path=str(result_path),
            log_path=str(log_path),
        )
    except Exception as exc:
        _fail_prelaunch_run(db, run.id, f"Could not prepare solver run: {exc}")
        raise
    worker_command = [
        sys.executable,
        "-m",
        "poker_tracker.solver.run_job",
        "--run-id",
        str(run.id),
        "--db",
        db.db_path,
    ]
    try:
        process = subprocess.Popen(
            worker_command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        _fail_prelaunch_run(db, run.id, f"Could not start solver worker: {exc}")
        raise RuntimeError("Could not start the solver worker.") from exc
    try:
        launched = db.update_solver_run(
            run.id,
            expected_statuses=("queued",),
            status="running",
            pid=process.pid,
            heartbeat_at=datetime.now(UTC),
            started_at=datetime.now(UTC),
        )
    except Exception:
        _terminate_solver_group(process.pid)
        # The worker dies before its own cleanup runs, so anything it managed to
        # launch in the moment it was alive has to be collected from here.
        _terminate_solver_child_process(run)
        raise
    if launched.status != "running" or launched.pid != process.pid:
        _terminate_solver_group(process.pid)
        _terminate_solver_child_process(run)
        if launched.status == "cancelling":
            db.update_solver_run(
                run.id,
                expected_statuses=("cancelling",),
                status="stale",
                pid=None,
                completed_at=datetime.now(UTC),
            )
        raise ValueError(
            "The hand changed while the solver worker was launching; the run was stopped."
        )
    return launched


def _materialize_cached_run(
    db: PokerDatabase,
    run: SolverRun,
    source: SolverRun,
    *,
    spot: SolverSpot,
    range_ip: ResolvedRange,
    range_oop: ResolvedRange,
    thread_count: int,
) -> SolverRun:
    if run.id is None:
        raise RuntimeError("The cached solver run could not be saved.")
    run_dir = solver_run_directory(run.id)
    command_path = run_dir / "input.txt"
    result_path = run_dir / "result.json"
    log_path = run_dir / "solver.log"
    try:
        command_path.write_text(
            build_command_file(
                spot,
                range_ip,
                range_oop,
                output_path=result_path,
                thread_count=thread_count,
            ),
            encoding="utf-8",
        )
        shutil.copy2(source.result_path, result_path)
        if source.log_path and Path(source.log_path).is_file():
            shutil.copy2(source.log_path, log_path)
        evidence = parse_strategy_result(
            result_path,
            spot=spot,
            range_ip=range_ip,
            range_oop=range_oop,
            backend_version=run.backend_version,
            exploitability_pct=source.exploitability_pct,
            runtime_seconds=source.runtime_seconds,
            assumptions=run.assumptions,
        )
        completed = db.update_solver_run(
            run.id,
            expected_statuses=("queued",),
            status="completed",
            command_path=str(command_path),
            result_path=str(result_path),
            log_path=str(log_path),
            evidence=evidence.model_dump(mode="json"),
            completed_at=datetime.now(UTC),
        )
        if completed.status == "cancelling":
            return db.update_solver_run(
                run.id,
                expected_statuses=("cancelling",),
                status="stale",
                pid=None,
                completed_at=datetime.now(UTC),
            )
        return completed
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _fail_prelaunch_run(db: PokerDatabase, run_id: int, message: str) -> SolverRun:
    saved = db.update_solver_run(
        run_id,
        expected_statuses=("queued",),
        status="failed",
        error_message=message,
        pid=None,
        completed_at=datetime.now(UTC),
    )
    if saved.status == "cancelling":
        return db.update_solver_run(
            run_id,
            expected_statuses=("cancelling",),
            status="stale",
            pid=None,
            completed_at=datetime.now(UTC),
        )
    return saved


def cancel_solver_run(db: PokerDatabase, run_id: int) -> SolverRun:
    run = db.fetch_solver_run(run_id)
    if run is None:
        raise ValueError("Solver run not found.")
    if run.status not in {"queued", "running", "cancelling"}:
        return run
    # The solver runs in its own session, so a killpg aimed at the worker no
    # longer reaches it. Stop the expensive process first: if the worker then
    # refuses to die the run parks in 'cancelling' without a multi-gigabyte
    # solve still burning the host behind it.
    _terminate_solver_child_process(run)
    if run.pid and not _terminate_solver_group(run.pid):
        current = db.fetch_solver_run(run_id)
        if current is not None and current.status not in {"queued", "running", "cancelling"}:
            return current
        return db.update_solver_run(
            run_id,
            expected_statuses=("queued", "running", "cancelling"),
            status="cancelling",
            error_message="Cancellation requested; waiting for the solver worker to stop.",
        )
    return db.update_solver_run(
        run_id,
        status="cancelled",
        error_message="Cancelled by user.",
        pid=None,
        completed_at=datetime.now(UTC),
    )


def reconcile_stale_solver_runs(
    db: PokerDatabase,
    *,
    stale_after: timedelta = DEFAULT_STALE_AFTER,
    now: datetime | None = None,
) -> list[int]:
    current = now or datetime.now(UTC)
    failed: list[int] = []
    for run in db.fetch_active_solver_runs():
        if run.id is None:
            continue
        reference = run.heartbeat_at or run.started_at or run.created_at
        alive = _pid_is_alive(run.pid)
        stale = current - _as_utc(reference) > stale_after
        if run.status == "queued" and not stale:
            continue
        if not stale and alive:
            continue
        # Past this point the run is finished one way or another, so nothing may
        # be left solving on its behalf. The solver has its own session and
        # outlives a killed worker unless it is reached through its pid file.
        _terminate_solver_child_process(run)
        if run.status == "cancelling" and not alive:
            db.update_solver_run(
                run.id,
                status="stale",
                pid=None,
                completed_at=current,
            )
            failed.append(run.id)
            continue
        if stale and alive and not _terminate_solver_group(run.pid):
            continue
        db.update_solver_run(
            run.id,
            status="failed",
            error_message="Solver worker stopped or its heartbeat expired.",
            completed_at=current,
        )
        failed.append(run.id)
    return failed


def _process_group_is_alive(pgid: int) -> bool:
    """True while any member of the process group survives, leader or not."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return not _refusal_means_the_group_is_gone(pgid)
    return True


def _refusal_means_the_group_is_gone(pid: int) -> bool:
    """Whether a refused group signal means nothing but corpses is left.

    Darwin refuses killpg with EPERM once every remaining member is an unreaped
    corpse, which reads identically to a group this user may not signal. The
    leader tells them apart: a pid this process can still signal, or that has
    already gone, was one of ours, so the refusal was about the dead. Reading
    that as a live process instead used to park every cancellation of an
    already-exited worker in 'cancelling' until the stale sweep ran.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def solver_child_pid_path(run_dir: Path) -> Path:
    """Where the worker records the pid of the solver it launched.

    The solver gets its own session so the worker can kill everything the solve
    forks without signalling itself, which also puts it out of reach of a
    killpg aimed at the worker's group. Cancellation and reconciliation follow
    this file to it, the same way the reconstruction pipeline's sidecar pid file
    works.
    """
    return run_dir / SOLVER_CHILD_PID_FILE


def _terminate_solver_child_process(run: SolverRun) -> None:
    """Kill the detached solver recorded beside a run's command file."""
    if not run.command_path:
        return
    pid_path = solver_child_pid_path(Path(run.command_path).parent)
    try:
        child_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        # No pid file means no solver was ever launched for this run, or the
        # worker already reaped it and cleaned up. Both are the normal case.
        return
    _terminate_solver_group(child_pid)
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        # A pid file we cannot remove is harmless; the run directory is
        # per-run, so nothing else will follow it to a recycled pid.
        pass


def configured_memory_limit_bytes() -> int | None:
    """Return the address-space cap for the solver child, or None when uncapped.

    A cap the operator asked for and did not get is worse than no cap at all:
    they believe the container's limit is holding while a deep river tree grows
    until the host OOM killer fires and takes Streamlit and any concurrent
    write with it. So every way of asking for a cap that cannot be honoured
    raises, before a run row exists and while the operator can still fix the
    environment.

    The rule itself lives in ``poker_tracker.runtime.limits`` because the CV
    reconstruction path needs exactly the same one. This kept its name: callers
    and tests import it, and the solver's variable belongs to the solver.
    """
    return memory_limit_bytes_from_env(SOLVER_MEMORY_ENV_VAR)


def _configured_thread_count() -> int:
    """The solver's thread ceiling, or raise naming the variable that is wrong.

    The default tracks the host rather than the ceiling, so a two-core machine
    is not asked for four threads. A value outside the range is refused instead
    of clamped: TexasSolver takes the count from the command file, so a silent
    clamp would put a number in the run's own input that the operator never
    chose.
    """
    return bounded_int_from_env(
        SOLVER_THREADS_ENV_VAR,
        default=min(MAX_SOLVER_THREADS, max(MIN_SOLVER_THREADS, os.cpu_count() or 1)),
        minimum=MIN_SOLVER_THREADS,
        maximum=MAX_SOLVER_THREADS,
    )


def _terminate_solver_group(pid: int | None, *, grace_seconds: float = 0.5) -> bool:
    """Send TERM then KILL to a process group; True only when the pid is gone.

    ``grace_seconds`` is how long the leader is given to exit on its own. The
    worker allows the solver longer than the UI allows the worker, because a
    solve inside a CFR iteration answers a signal late while a Python worker
    answers immediately.
    """
    if pid is None:
        return True
    signalled_group = True
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        signalled_group = False
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    except PermissionError:
        return _refusal_means_the_group_is_gone(pid)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        # Waiting on the leader alone declared success while everything it
        # forked was still solving: a group member outlives its parent and is
        # reparented away, and that member is what still holds the memory and
        # the threads. Escalation has to be driven by the group.
        alive = _process_group_is_alive(pid) if signalled_group else _pid_is_alive(pid)
        if not alive:
            return True
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
    except PermissionError:
        return _refusal_means_the_group_is_gone(pid)
    return True


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
