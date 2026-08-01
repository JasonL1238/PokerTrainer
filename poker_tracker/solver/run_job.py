from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.safety.redaction import safe_error_message
from poker_tracker.solver.models import ResolvedRange, SolverSpot
from poker_tracker.solver.texassolver import (
    DEFAULT_TIMEOUT_SECONDS,
    configured_binary,
    configured_resource_dir,
    parse_final_exploitability,
    parse_strategy_result,
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
    if run.status in {"cancelled", "stale", "failed", "completed"}:
        db.close()
        return
    if run.status == "cancelling":
        completed = db.update_solver_run(
            run_id,
            expected_statuses=("cancelling",),
            status="stale",
            pid=None,
            completed_at=datetime.now(UTC),
        )
        if completed.status == "cancelling":
            db.update_solver_run(
                run_id,
                expected_statuses=("cancelling",),
                status="stale",
                pid=None,
                completed_at=datetime.now(UTC),
            )
        db.close()
        return
    binary = configured_binary()
    resource_dir = configured_resource_dir(binary)
    spot = SolverSpot.model_validate(run.spot)
    range_ip = ResolvedRange.model_validate(run.range_ip)
    range_oop = ResolvedRange.model_validate(run.range_oop)
    command_path = Path(run.command_path)
    result_path = Path(run.result_path)
    log_path = Path(run.log_path)
    if not command_path.is_file():
        raise FileNotFoundError(f"Solver command file was not found: {command_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [str(binary), "-i", str(command_path), "-r", str(resource_dir)],
                cwd=str(command_path.parent),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                preexec_fn=_memory_limit if os.name == "posix" else None,
            )
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
                _terminate_process(process)
                if claimed.status == "cancelling":
                    db.update_solver_run(
                        run_id,
                        expected_statuses=("cancelling",),
                        status="stale",
                        pid=None,
                        completed_at=datetime.now(UTC),
                    )
                return
            deadline = started + timeout_seconds
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    _terminate_process(process)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError(
                        f"TexasSolver exceeded the {timeout_seconds}-second timeout."
                    )
                time.sleep(2)
                current = db.fetch_solver_run(run_id)
                if current is None:
                    _terminate_process(process)
                    return
                if current.status != "running":
                    _terminate_process(process)
                    if current.status == "cancelling":
                        db.update_solver_run(
                            run_id,
                            expected_statuses=("cancelling",),
                            status="stale",
                            pid=None,
                            completed_at=datetime.now(UTC),
                        )
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
            db.update_solver_run(
                run_id,
                expected_statuses=("cancelling",),
                status="stale",
                pid=None,
                completed_at=datetime.now(UTC),
            )
            return
        if current.status in {"cancelled", "stale", "failed", "completed"}:
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
        db.close()


def _memory_limit() -> None:
    try:
        import resource

        configured = os.environ.get("POKERTRAINER_SOLVER_MEMORY_GB", "").strip()
        if not configured:
            return
        limit_gb = float(configured)
        if limit_gb <= 0:
            return
        limit = int(limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError):
        return


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except ProcessLookupError:
        return


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
