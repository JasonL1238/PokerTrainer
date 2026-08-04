from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.runtime.limits import format_gb, memory_limiter
from poker_tracker.safety.redaction import safe_error_message
from poker_tracker.solver.jobs import (
    SOLVER_MEMORY_ENV_VAR,
    _terminate_solver_group,
    configured_memory_limit_bytes,
    solver_child_pid_path,
)
from poker_tracker.solver.models import ResolvedRange, SolverSpot
from poker_tracker.solver.texassolver import (
    DEFAULT_TIMEOUT_SECONDS,
    configured_binary,
    configured_resource_dir,
    parse_final_exploitability,
    parse_strategy_result,
)

# A solver inside a CFR iteration answers a signal late, so it is given longer
# to exit than the UI gives the worker before escalating.
TERMINATION_GRACE_SECONDS = 5.0


# Asked by both the pre-flight check and the failure handler; two copies that
# drifted would let the handler overwrite a finished run's outcome.
_TERMINAL_STATUSES = frozenset({"cancelled", "stale", "failed", "completed"})


def _settle_cancellation(db: PokerDatabase, run_id: int) -> None:
    """Agree that a requested cancellation has been honoured.

    The compare-and-swap expects ``cancelling``, so it returns either the ``stale``
    row it just wrote or -- on a lost race -- a row that by that same predicate is
    no longer ``cancelling``, and nothing moves a run back to ``cancelling``. A
    retry guarded on a returned ``cancelling`` was therefore unreachable. Widening
    the expected statuses here would revive that dead guard.

    Four paths reach this: the pre-flight check, the ownership claim, the poll
    loop, and the failure handler. Each has already stopped its child process by
    the time it arrives, so settling the row is all that is left -- which is why
    they can share one writer rather than four copies of the same CAS.
    """
    db.update_solver_run(
        run_id,
        expected_statuses=("cancelling",),
        status="stale",
        pid=None,
        completed_at=datetime.now(UTC),
    )


def run_solver_job(
    db: PokerDatabase,
    run_id: int,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    run = db.fetch_solver_run(run_id)
    if run is None:
        raise ValueError("Solver run not found.")
    if run.status in _TERMINAL_STATUSES:
        db.close()
        return
    if run.status == "cancelling":
        _settle_cancellation(db, run_id)
        db.close()
        return
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    pid_path: Path | None = None
    try:
        # Setup lives inside the try because a worker that cannot start is a
        # failure the operator has to see now. Raising above it left the row
        # queued and healthy-looking until the stale reconciler noticed minutes
        # later, which is the same silence a missing binary or a rejected
        # memory cap would otherwise hide behind.
        binary = configured_binary()
        resource_dir = configured_resource_dir(binary)
        memory_limit_bytes = configured_memory_limit_bytes()
        spot = SolverSpot.model_validate(run.spot)
        range_ip = ResolvedRange.model_validate(run.range_ip)
        range_oop = ResolvedRange.model_validate(run.range_oop)
        command_path = Path(run.command_path)
        result_path = Path(run.result_path)
        log_path = Path(run.log_path)
        if not command_path.is_file():
            raise FileNotFoundError(f"Solver command file was not found: {command_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path = solver_child_pid_path(command_path.parent)
        with log_path.open("ab") as log:
            try:
                process = subprocess.Popen(
                    [str(binary), "-i", str(command_path), "-r", str(resource_dir)],
                    cwd=str(command_path.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    # Its own session makes the solve a process group this worker
                    # can terminate whole, including anything the solve forks,
                    # without signalling itself out of existence before it can
                    # record why the run ended.
                    start_new_session=os.name == "posix",
                    preexec_fn=(
                        memory_limiter(memory_limit_bytes, variable=SOLVER_MEMORY_ENV_VAR)
                        if os.name == "posix" and memory_limit_bytes is not None
                        else None
                    ),
                )
            except subprocess.SubprocessError as exc:
                if memory_limit_bytes is None:
                    raise
                # subprocess reports a hook failure as a bare "Exception occurred
                # in preexec_fn", so the reason the operator needs has to be
                # restated here. Refusing to solve is the point: a cap that was
                # asked for and not applied is how a river tree reaches the host
                # OOM killer while the operator believes it is bounded.
                raise RuntimeError(
                    f"The {format_gb(memory_limit_bytes)} solver memory cap from "
                    f"{SOLVER_MEMORY_ENV_VAR} could not be applied, so the solve "
                    "was not started. macOS rejects RLIMIT_AS outright; unset the "
                    "variable to solve without a cap, or run the solver in the Linux "
                    "container where the cap holds."
                ) from exc
            pid_path.write_text(str(process.pid), encoding="utf-8")
            owner_pid = run.pid or process.pid
            claimed = db.update_solver_run(
                run_id,
                expected_statuses=("queued", "running"),
                status="running",
                pid=owner_pid,
                heartbeat_at=datetime.now(UTC),
                started_at=run.started_at or datetime.now(UTC),
            )
            if claimed.status != "running" or claimed.pid != owner_pid:
                _terminate_solver_child(process)
                if claimed.status == "cancelling":
                    _settle_cancellation(db, run_id)
                return
            deadline = started + timeout_seconds
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    stopped = _terminate_solver_child(process)
                    message = f"TexasSolver exceeded the {timeout_seconds}-second timeout."
                    if not stopped:
                        # An orphaned solver keeps holding its memory and
                        # threads, so the operator has to be told to go find it
                        # rather than assume the timeout cleaned up after itself.
                        message += (
                            f" The solver process (pid {process.pid}) could not be "
                            "confirmed stopped and may still be running."
                        )
                    raise TimeoutError(message)
                time.sleep(2)
                current = db.fetch_solver_run(run_id)
                if current is None:
                    _terminate_solver_child(process)
                    return
                if current.status != "running":
                    _terminate_solver_child(process)
                    if current.status == "cancelling":
                        _settle_cancellation(db, run_id)
                    return
                db.update_solver_run(
                    run_id,
                    expected_statuses=("running",),
                    heartbeat_at=datetime.now(UTC),
                )
        if process.returncode != 0:
            raise RuntimeError(f"TexasSolver exited with status {process.returncode}.")
        if not result_path.is_file():
            raise RuntimeError("TexasSolver completed without writing a result JSON file.")
        runtime = time.monotonic() - started
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        exploitability = parse_final_exploitability(log_text)
        evidence = parse_strategy_result(
            result_path,
            spot=spot,
            range_ip=range_ip,
            range_oop=range_oop,
            backend_version=run.backend_version,
            exploitability_pct=exploitability,
            runtime_seconds=runtime,
            assumptions=run.assumptions,
        )
        db.update_solver_run(
            run_id,
            expected_statuses=("running",),
            status="completed",
            evidence=evidence.model_dump(mode="json"),
            exploitability_pct=exploitability,
            runtime_seconds=runtime,
            pid=None,
            heartbeat_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
    except Exception as exc:
        current = db.fetch_solver_run(run_id)
        if current is None:
            return
        if current.status == "cancelling":
            _settle_cancellation(db, run_id)
            return
        if current.status in _TERMINAL_STATUSES:
            return
        if current.status in {"queued", "running"}:
            db.update_solver_run(
                run_id,
                expected_statuses=("queued", "running"),
                status="failed",
                # The store scrubs this column whatever a writer hands it; this
                # call also bounds and flattens the message, so a solver
                # traceback does not arrive as a multi-line wall of text.
                error_message=safe_error_message(exc),
                runtime_seconds=max(0.0, time.monotonic() - started),
                pid=None,
                heartbeat_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
        raise
    finally:
        # Every exit from here is an exit of the worker, so nothing may be left
        # solving: a DB error while claiming the run used to leave console_solver
        # running with its pid recorded nowhere.
        if process is not None:
            _terminate_solver_child(process)
        if pid_path is not None:
            pid_path.unlink(missing_ok=True)
        db.close()


def _terminate_solver_child(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float | None = None,
) -> bool:
    """Stop the solver and everything it forked. True only when it is reaped.

    Signalling the direct child alone is not enough: a solve that forks leaves
    a grandchild that is reparented away and keeps the memory and threads the
    run was supposed to release. The solver is a session leader, so its pid is
    also its group id and one killpg reaches the whole tree.
    """
    grace = TERMINATION_GRACE_SECONDS if grace_seconds is None else grace_seconds
    if process.poll() is not None:
        return True
    if os.name == "posix":
        _terminate_solver_group(process.pid, grace_seconds=grace)
    else:
        try:
            process.terminate()
        except ProcessLookupError:
            return True
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return True
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    db = PokerDatabase(args.db)
    db.init_db()
    run_solver_job(db, args.run_id, timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
    main()
