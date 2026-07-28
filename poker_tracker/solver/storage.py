from __future__ import annotations

import os
import shutil
from pathlib import Path

from poker_tracker.persistence.models import SolverRun

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
SOLVER_RUNS_DIR = DATA_DIR / "solver_runs"


def solver_run_directory(run_id: int) -> Path:
    SOLVER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = SOLVER_RUNS_DIR / f"run_{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_solver_run_artifacts(run: SolverRun) -> bool:
    paths = [run.command_path, run.result_path, run.log_path]
    directories = {Path(path).resolve().parent for path in paths if path}
    root = SOLVER_RUNS_DIR.resolve()
    removed = False
    for directory in directories:
        if directory == root or root not in directory.parents:
            continue
        if directory.exists():
            shutil.rmtree(directory)
            removed = True
    return removed
