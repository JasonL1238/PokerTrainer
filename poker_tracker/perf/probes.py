"""The individual measurements the performance harness can take.

Every probe answers one of the quantities Phase 13 asks for, and every probe has
the same obligation: produce a number with the conditions that produced it, or
say plainly that it did not run and why. A probe never substitutes a plausible
value, and never lets "could not measure" arrive as a zero.

Heavy probes run in a child process launched through :func:`run_child`, which
wraps the work so the child reports its own wall time and its own peak resident
set. Measuring a child's memory from the parent is not possible accurately --
``RUSAGE_CHILDREN`` is a maximum over every child the process ever waited on, so
it would attribute the CV pipeline's footprint to whichever probe ran last.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_tracker.perf.measurement import (
    UNIT_BYTES,
    UNIT_COUNT,
    UNIT_FPS,
    UNIT_MB_PER_SECOND,
    UNIT_SECONDS,
    Measurement,
    MeasurementSpec,
    measured,
    not_taken,
)
from poker_tracker.release_gate.environment import video_duration_seconds
from poker_tracker.release_gate.models import resolve_models
from poker_tracker.safety.redaction import redact_text

PIPELINE_SCRIPT = Path("cv_lab/scripts/pipeline/run_two_model_pipeline.py")
APP_SCRIPT = Path("app.py")
VALIDATION_ROOT_ENV = "POKER_VALIDATION_ROOT"
# The phase's own bound. A representative session must finish inside it.
SESSION_LIMIT_SECONDS = 3600.0

_MARKER = "@@PERF@@"
_SAMPLED_FRAMES = re.compile(r"sampled\s+(\d+)\s+frames")


# --------------------------------------------------------------------------
# Metric declarations
# --------------------------------------------------------------------------

IMPORT_CORE = MeasurementSpec(
    name="import.core_seconds",
    unit=UNIT_SECONDS,
    group="imports",
    description="Cold import of the persistence layer in a fresh interpreter.",
)
IMPORT_STREAMLIT = MeasurementSpec(
    name="import.streamlit_seconds",
    unit=UNIT_SECONDS,
    group="imports",
    description="Cold import of Streamlit in a fresh interpreter.",
)
IMPORT_CV_STACK = MeasurementSpec(
    name="import.cv_stack_seconds",
    unit=UNIT_SECONDS,
    group="imports",
    description="Cold import of torch, ultralytics, av and cv2 in a fresh interpreter.",
)
IMPORT_CV_STACK_RSS = MeasurementSpec(
    name="import.cv_stack_peak_rss_bytes",
    unit=UNIT_BYTES,
    group="imports",
    description="Peak resident set of an interpreter that only imported the CV stack.",
)

STARTUP_HEALTH = MeasurementSpec(
    name="startup.app_health_seconds",
    unit=UNIT_SECONDS,
    group="startup",
    description="Streamlit launch to the first healthy /_stcore/health response.",
)

UI_FIRST_RENDER = MeasurementSpec(
    name="ui.first_render_seconds",
    unit=UNIT_SECONDS,
    group="ui",
    description="First full script run of app.py under AppTest.",
)
UI_SLOWEST_PAGE = MeasurementSpec(
    name="ui.slowest_page_render_seconds",
    unit=UNIT_SECONDS,
    group="ui",
    description="Slowest single navigation rerun across every primary page.",
)
UI_PEAK_RSS = MeasurementSpec(
    name="ui.peak_rss_bytes",
    unit=UNIT_BYTES,
    group="ui",
    description="Peak resident set of an interpreter rendering every page once.",
)

UPLOAD_SECONDS = MeasurementSpec(
    name="upload.store_seconds",
    unit=UNIT_SECONDS,
    group="upload",
    description="Storing an uploaded recording through the vault's atomic writer.",
)
UPLOAD_THROUGHPUT = MeasurementSpec(
    name="upload.store_megabytes_per_second",
    unit=UNIT_MB_PER_SECOND,
    group="upload",
    description="Sustained write rate of the vault store path, fsync included.",
    lower_is_better=False,
)

MODEL_DETECTOR_INIT = MeasurementSpec(
    name="model_init.detector_seconds",
    unit=UNIT_SECONDS,
    group="models",
    description="Region detector from construction to its first inference, imports excluded.",
)
MODEL_CLASSIFIER_INIT = MeasurementSpec(
    name="model_init.classifier_seconds",
    unit=UNIT_SECONDS,
    group="models",
    description="Card classifier from construction to its first inference; its weights load lazily.",
)
MODEL_INIT_RSS = MeasurementSpec(
    name="model_init.peak_rss_bytes",
    unit=UNIT_BYTES,
    group="models",
    description="Peak resident set of an interpreter that loaded both models.",
)

RECONSTRUCTION_SECONDS = MeasurementSpec(
    name="reconstruction.wall_seconds",
    unit=UNIT_SECONDS,
    group="reconstruction",
    description="End-to-end two-model reconstruction of one recording.",
)
RECONSTRUCTION_FRAMES = MeasurementSpec(
    name="reconstruction.frames_processed",
    unit=UNIT_COUNT,
    group="reconstruction",
    description="Frames the pipeline sampled during the measured reconstruction.",
    lower_is_better=False,
)
RECONSTRUCTION_FPS = MeasurementSpec(
    name="reconstruction.frames_per_second",
    unit=UNIT_FPS,
    group="reconstruction",
    description="Sampled-frame throughput of the reconstruction pipeline.",
    lower_is_better=False,
)
RECONSTRUCTION_RSS = MeasurementSpec(
    name="reconstruction.peak_rss_bytes",
    unit=UNIT_BYTES,
    group="reconstruction",
    description="Peak resident set of the reconstruction process.",
)
RECONSTRUCTION_TIMELINE_BYTES = MeasurementSpec(
    name="reconstruction.timeline_bytes",
    unit=UNIT_BYTES,
    group="reconstruction",
    description="Size of the timeline artifact the measured reconstruction wrote.",
)

SOLVER_RUNS = MeasurementSpec(
    name="solver.recorded_runs",
    unit=UNIT_COUNT,
    group="solver",
    description="Completed solver runs with a recorded runtime in the database.",
    lower_is_better=False,
)
SOLVER_MEDIAN = MeasurementSpec(
    name="solver.recorded_runtime_median_seconds",
    unit=UNIT_SECONDS,
    group="solver",
    description="Median runtime of the solver runs this installation has recorded.",
)
SOLVER_MAX = MeasurementSpec(
    name="solver.recorded_runtime_max_seconds",
    unit=UNIT_SECONDS,
    group="solver",
    description="Slowest solver run this installation has recorded.",
)

HARNESS_PEAK_RSS = MeasurementSpec(
    name="memory.harness_peak_rss_bytes",
    unit=UNIT_BYTES,
    group="resources",
    description="Peak resident set of the harness process itself.",
)
DISK_WORKSPACE_GROWTH = MeasurementSpec(
    name="disk.workspace_growth_bytes",
    unit=UNIT_BYTES,
    group="resources",
    description="Signed change in workspace bytes across the run, not a snapshot.",
)
DISK_DATA_ROOT = MeasurementSpec(
    name="disk.data_root_bytes",
    unit=UNIT_BYTES,
    group="resources",
    description="Current size of the configured operator data root.",
)
DISK_FREE = MeasurementSpec(
    name="disk.free_bytes",
    unit=UNIT_BYTES,
    group="resources",
    description="Free space on the volume holding the workspace, after the run.",
    lower_is_better=False,
)
TEMP_LEAKED_COUNT = MeasurementSpec(
    name="tempfiles.leaked_count",
    unit=UNIT_COUNT,
    group="resources",
    description="Temporary entries the measured work left behind in its own TMPDIR.",
)
TEMP_LEAKED_BYTES = MeasurementSpec(
    name="tempfiles.leaked_bytes",
    unit=UNIT_BYTES,
    group="resources",
    description="Bytes held by leftover temporary entries after the run.",
)
LOG_GROWTH = MeasurementSpec(
    name="logs.growth_bytes",
    unit=UNIT_BYTES,
    group="resources",
    description="Signed change in log bytes written under the workspace.",
)


GROUP_SPECS: dict[str, tuple[MeasurementSpec, ...]] = {
    "imports": (IMPORT_CORE, IMPORT_STREAMLIT, IMPORT_CV_STACK, IMPORT_CV_STACK_RSS),
    "startup": (STARTUP_HEALTH,),
    "ui": (UI_FIRST_RENDER, UI_SLOWEST_PAGE, UI_PEAK_RSS),
    "upload": (UPLOAD_SECONDS, UPLOAD_THROUGHPUT),
    "models": (MODEL_DETECTOR_INIT, MODEL_CLASSIFIER_INIT, MODEL_INIT_RSS),
    "reconstruction": (
        RECONSTRUCTION_SECONDS,
        RECONSTRUCTION_FRAMES,
        RECONSTRUCTION_FPS,
        RECONSTRUCTION_RSS,
        RECONSTRUCTION_TIMELINE_BYTES,
    ),
    "solver": (SOLVER_RUNS, SOLVER_MEDIAN, SOLVER_MAX),
    "resources": (
        HARNESS_PEAK_RSS,
        DISK_WORKSPACE_GROWTH,
        DISK_DATA_ROOT,
        DISK_FREE,
        TEMP_LEAKED_COUNT,
        TEMP_LEAKED_BYTES,
        LOG_GROWTH,
    ),
}

# Groups a default run drives. ``resources`` is not listed because the harness
# always brackets the run with it rather than scheduling it as a probe.
PROBE_GROUPS: tuple[str, ...] = (
    "imports",
    "startup",
    "ui",
    "upload",
    "models",
    "reconstruction",
    "solver",
)

ALL_SPECS: tuple[MeasurementSpec, ...] = tuple(
    spec for group in sorted(GROUP_SPECS) for spec in GROUP_SPECS[group]
)

DEFAULT_TIMEOUTS: dict[str, float] = {
    "imports": 600.0,
    "startup": 180.0,
    "ui": 600.0,
    "models": 900.0,
    # Deliberately above the phase's one-hour bound: a reconstruction killed at
    # exactly 3600s could never be distinguished from one that took 61 minutes,
    # and the harness exists to tell those two apart.
    "reconstruction": 7200.0,
}


@dataclass
class ProbeContext:
    """Everything a probe needs, and nothing it may reach around.

    ``workspace`` is the only directory a probe may write to. The operator's
    real data root and database are read at most, never written: the harness
    must be safe to run against a live installation.
    """

    repo_root: Path
    workspace: Path
    db_path: Path
    data_root: Path
    manifest_path: Path
    video: Path | None = None
    upload_bytes: int = 32 * 1024 * 1024
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))

    @property
    def log_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.workspace / "tmp"

    @property
    def child_data_root(self) -> Path:
        return self.workspace / "data"

    def timeout(self, group: str) -> float:
        return float(self.timeouts.get(group, DEFAULT_TIMEOUTS.get(group, 600.0)))

    def prepare(self) -> None:
        for path in (self.workspace, self.log_dir, self.tmp_dir, self.child_data_root):
            path.mkdir(parents=True, exist_ok=True)


def child_peak_rss_bytes(ru_maxrss: int, child_platform: str) -> int:
    """Normalize a child's reported ``ru_maxrss`` to bytes.

    Same rule as ``release_gate.resources._maxrss_bytes`` -- Linux reports
    kilobytes, macOS and the BSDs report bytes -- applied here because the child
    reports the raw number and must not import this application to convert it.
    A test pins the two implementations together.
    """
    if child_platform == "darwin":
        return int(ru_maxrss)
    return int(ru_maxrss) * 1024


_CHILD_PROLOGUE = """
import json as _pj, resource as _pr, sys as _ps, time as _pt
_extra = {}
_started = _pt.perf_counter()
"""

_CHILD_EPILOGUE = """
_elapsed = _pt.perf_counter() - _started
_usage = _pr.getrusage(_pr.RUSAGE_SELF)
_ps.stdout.write("\\n@@PERF@@" + _pj.dumps({
    "seconds": _elapsed,
    "ru_maxrss": int(_usage.ru_maxrss),
    "platform": _ps.platform,
    "extra": _extra,
}) + "\\n")
"""


@dataclass(frozen=True)
class ChildResult:
    ok: bool
    seconds: float | None
    peak_rss_bytes: int | None
    extra: dict[str, Any]
    error: str | None
    output: str
    log_path: Path


def child_env(ctx: ProbeContext, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a probe child.

    Operator state is redirected into the workspace so no probe can write to the
    real database, the real vault or the real backups, and TMPDIR is redirected
    so leftover temporary files are attributable to this run rather than mixed
    in with every other process on the machine.
    """
    env = dict(os.environ)
    env["POKER_DB_PATH"] = str(ctx.child_data_root / "poker_tracker.db")
    env["POKER_DATA_DIR"] = str(ctx.child_data_root / "data")
    env["TMPDIR"] = str(ctx.tmp_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ctx.repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    for key, value in (overrides or {}).items():
        env[key] = value
    return env


def run_child(
    ctx: ProbeContext,
    *,
    name: str,
    body: str,
    timeout: float,
    env_overrides: dict[str, str] | None = None,
    drop_env: tuple[str, ...] = (),
) -> ChildResult:
    """Run ``body`` in a fresh interpreter that reports its own cost."""
    ctx.prepare()
    log_path = ctx.log_dir / f"{name}.log"
    env = child_env(ctx, env_overrides)
    for key in drop_env:
        env.pop(key, None)
    source = _CHILD_PROLOGUE + body + _CHILD_EPILOGUE
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ctx.repo_root),
            env=env,
        )
    except subprocess.TimeoutExpired:
        log_path.write_text(f"probe {name} exceeded {timeout}s\n", encoding="utf-8")
        return ChildResult(
            ok=False,
            seconds=None,
            peak_rss_bytes=None,
            extra={},
            error=f"probe exceeded its {timeout:.0f}s timeout after {time.perf_counter() - started:.0f}s",
            output="",
            log_path=log_path,
        )
    except OSError as exc:
        return ChildResult(
            ok=False,
            seconds=None,
            peak_rss_bytes=None,
            extra={},
            error=f"probe could not start: {exc}",
            output="",
            log_path=log_path,
        )
    output = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(redact_text(output), encoding="utf-8")
    if completed.returncode != 0:
        tail = " | ".join((completed.stderr or "").strip().splitlines()[-3:])
        return ChildResult(
            ok=False,
            seconds=None,
            peak_rss_bytes=None,
            extra={},
            error=f"probe exited {completed.returncode}: {redact_text(tail)[:400]}",
            output=output,
            log_path=log_path,
        )
    payload = _parse_marker(completed.stdout or "")
    if payload is None:
        return ChildResult(
            ok=False,
            seconds=None,
            peak_rss_bytes=None,
            extra={},
            error="probe produced no measurement record",
            output=output,
            log_path=log_path,
        )
    return ChildResult(
        ok=True,
        seconds=float(payload.get("seconds") or 0.0),
        peak_rss_bytes=child_peak_rss_bytes(
            int(payload.get("ru_maxrss") or 0), str(payload.get("platform") or "")
        ),
        extra=dict(payload.get("extra") or {}),
        error=None,
        output=output,
        log_path=log_path,
    )


def _parse_marker(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_MARKER):
            try:
                return json.loads(line[len(_MARKER) :])
            except json.JSONDecodeError:
                return None
    return None


def group_not_taken(
    group: str, reason: str, *, probe: str | None = None
) -> list[Measurement]:
    """Every metric in a group, withheld for one shared reason."""
    return [
        not_taken(spec, reason=reason, probe=probe or f"{group}_probe")
        for spec in GROUP_SPECS[group]
    ]


def complete_group(
    group: str, results: list[Measurement], *, probe: str
) -> list[Measurement]:
    """Fill in any metric the probe returned no record for.

    A probe that emits some of its group and forgets the rest would leave the
    report short a key, which reads as "nothing to see" rather than as a gap.
    The filler says the probe produced no record, which is the truth.
    """
    seen = {m.spec.name for m in results}
    return results + [
        not_taken(
            spec,
            reason=f"the {group} probe produced no record for this metric",
            probe=probe,
        )
        for spec in GROUP_SPECS[group]
        if spec.name not in seen
    ]


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


_IMPORT_SETS: tuple[tuple[MeasurementSpec, tuple[str, ...]], ...] = (
    (IMPORT_CORE, ("poker_tracker.persistence.db",)),
    (IMPORT_STREAMLIT, ("streamlit",)),
    (IMPORT_CV_STACK, ("torch", "ultralytics", "av", "cv2")),
)


def probe_imports(ctx: ProbeContext) -> list[Measurement]:
    """Cold import cost, one fresh interpreter per module set."""
    results: list[Measurement] = []
    for spec, modules in _IMPORT_SETS:
        body = (
            "import importlib\n"
            f"_mods = {list(modules)!r}\n"
            "_started = _pt.perf_counter()\n"
            "for _m in _mods:\n"
            "    importlib.import_module(_m)\n"
        )
        child = run_child(
            ctx,
            name=f"import_{spec.name.split('.')[1]}",
            body=body,
            timeout=ctx.timeout("imports"),
        )
        conditions = {
            "modules": list(modules),
            "interpreter": sys.executable,
            "excludes": "interpreter startup and site initialization",
        }
        if not child.ok or child.seconds is None:
            results.append(
                not_taken(
                    spec,
                    reason=child.error or "import probe produced no timing",
                    probe="probe_imports",
                    conditions=conditions,
                )
            )
            if spec is IMPORT_CV_STACK:
                results.append(
                    not_taken(
                        IMPORT_CV_STACK_RSS,
                        reason=child.error or "import probe produced no timing",
                        probe="probe_imports",
                        conditions=conditions,
                    )
                )
            continue
        results.append(
            measured(
                spec,
                value=round(child.seconds, 4),
                probe="probe_imports",
                conditions=conditions,
            )
        )
        if spec is IMPORT_CV_STACK:
            results.append(
                measured(
                    IMPORT_CV_STACK_RSS,
                    value=child.peak_rss_bytes or 0,
                    probe="probe_imports",
                    conditions=conditions,
                )
            )
    return complete_group("imports", results, probe="probe_imports")


def _free_port() -> int | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        return None


def probe_startup(ctx: ProbeContext) -> list[Measurement]:
    """Launch the real server and time it to its first healthy response."""
    app_path = ctx.repo_root / APP_SCRIPT
    if not app_path.is_file():
        return group_not_taken("startup", f"{APP_SCRIPT} not found", probe="probe_startup")
    port = _free_port()
    if port is None:
        return group_not_taken(
            "startup", "no loopback port could be reserved", probe="probe_startup"
        )
    timeout = ctx.timeout("startup")
    ctx.prepare()
    log_path = ctx.log_dir / "startup.log"
    env = child_env(ctx)
    # An authenticated instance still serves the health endpoint, but clearing
    # these keeps the measurement about server start rather than about whichever
    # credentials happen to be exported.
    for key in ("APP_PASSWORD", "POKERTRAINER_REQUIRE_AUTH"):
        env.pop(key, None)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats=false",
    ]
    endpoint = f"http://127.0.0.1:{port}/_stcore/health"
    conditions: dict[str, Any] = {
        "endpoint": endpoint,
        "headless": True,
        "poll_interval_s": 0.1,
        "includes": "interpreter startup, imports and the first server bind",
        # The same endpoint the container healthcheck uses, so this is the
        # operationally meaningful figure -- but it answers as soon as the
        # server binds, before any script run, and must not be read as "the
        # app is rendered and ready".
        "excludes": "the first script run; /_stcore/health answers on bind",
    }
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            started = time.perf_counter()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(ctx.repo_root),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                return group_not_taken(
                    "startup", f"server could not start: {exc}", probe="probe_startup"
                )
            deadline = started + timeout
            healthy: float | None = None
            while time.perf_counter() < deadline:
                if process.poll() is not None:
                    return group_not_taken(
                        "startup",
                        f"server exited {process.returncode} before becoming healthy",
                        probe="probe_startup",
                    )
                try:
                    with urllib.request.urlopen(endpoint, timeout=2) as response:
                        if response.status == 200:
                            healthy = time.perf_counter()
                            break
                except (urllib.error.URLError, OSError, ValueError):
                    pass
                time.sleep(0.1)
            if healthy is None:
                return group_not_taken(
                    "startup",
                    f"server did not answer /_stcore/health within {timeout:.0f}s",
                    probe="probe_startup",
                )
            return [
                measured(
                    STARTUP_HEALTH,
                    value=round(healthy - started, 3),
                    probe="probe_startup",
                    conditions=conditions,
                )
            ]
    finally:
        _terminate(process)


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


_UI_BODY = """
import sys
from streamlit.testing.v1 import AppTest

from poker_tracker.ui.navigation import Page

_app = AppTest.from_file({app_path!r}, default_timeout=240)
_started = _pt.perf_counter()
_app.run()
_first = _pt.perf_counter() - _started
if list(_app.exception):
    raise SystemExit("app raised an exception during first render")
_pages = {{}}
_slowest = None
for _page in list(Page):
    _app.radio[0].set_value(_page)
    _t0 = _pt.perf_counter()
    _app.run()
    _dt = _pt.perf_counter() - _t0
    _pages[str(_page)] = round(_dt, 4)
    if _slowest is None or _dt > _pages[_slowest]:
        _slowest = str(_page)
    if list(_app.exception):
        raise SystemExit("app raised an exception on page " + str(_page))
_extra = {{
    "first_render_seconds": round(_first, 4),
    "page_seconds": _pages,
    "slowest_page": _slowest,
}}
"""


def probe_ui(ctx: ProbeContext) -> list[Measurement]:
    """Render every primary page once and time the reruns.

    This is a study-view responsiveness figure taken on an empty database. It
    says how fast the shell renders, not how fast a page renders over a full
    library, and the conditions say so rather than letting the number imply more
    than it covers.
    """
    app_path = ctx.repo_root / APP_SCRIPT
    if not app_path.is_file():
        return group_not_taken("ui", f"{APP_SCRIPT} not found", probe="probe_ui")
    child = run_child(
        ctx,
        name="ui_render",
        body=_UI_BODY.format(app_path=str(app_path)),
        timeout=ctx.timeout("ui"),
        drop_env=("APP_PASSWORD", "POKERTRAINER_REQUIRE_AUTH"),
    )
    conditions: dict[str, Any] = {
        "harness": "streamlit.testing.v1.AppTest",
        "database": "empty workspace database, not the operator's library",
        "covers": "shell and page render only; no CV job, no solve",
    }
    if not child.ok:
        return group_not_taken(
            "ui", child.error or "UI probe produced no timing", probe="probe_ui"
        )
    pages = child.extra.get("page_seconds") or {}
    conditions["pages"] = pages
    conditions["slowest_page"] = child.extra.get("slowest_page")
    results = [
        measured(
            UI_FIRST_RENDER,
            value=float(child.extra.get("first_render_seconds") or 0.0),
            probe="probe_ui",
            conditions=conditions,
        ),
        measured(
            UI_PEAK_RSS,
            value=child.peak_rss_bytes or 0,
            probe="probe_ui",
            conditions=conditions,
        ),
    ]
    if pages:
        results.append(
            measured(
                UI_SLOWEST_PAGE,
                value=max(float(v) for v in pages.values()),
                probe="probe_ui",
                conditions=conditions,
            )
        )
    else:
        results.append(
            not_taken(
                UI_SLOWEST_PAGE,
                reason="no page rerun was timed",
                probe="probe_ui",
                conditions=conditions,
            )
        )
    return complete_group("ui", results, probe="probe_ui")


def probe_upload(ctx: ProbeContext) -> list[Measurement]:
    """Time the vault's atomic store path on a synthetic recording.

    This measures what the application does with an upload once it has it --
    chunked copy, fsync, atomic rename -- and deliberately not the browser
    transfer, which the harness cannot observe. Reporting a browser-inclusive
    number from a local file copy would be the fabrication this harness exists
    to avoid.
    """
    from poker_tracker.ui.video_storage import save_video_file

    ctx.prepare()
    videos_dir = ctx.child_data_root / "data" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    source = ctx.workspace / "synthetic_upload.mp4"
    payload = os.urandom(1024 * 1024)
    size = int(ctx.upload_bytes)
    conditions = {
        "bytes": size,
        "source": "synthetic incompressible bytes written to the workspace first",
        "excludes": "browser transfer; measures the local store path only",
        "includes": "1 MiB chunked copy, fsync and atomic rename",
        # The store path calls os.fsync, which on macOS does not force the
        # device cache the way F_FULLFSYNC does. A figure well above the drive's
        # sustained rate is the page cache, not the disk.
        "fsync": "os.fsync as the application calls it; platform semantics apply",
    }
    try:
        with source.open("wb") as handle:
            written = 0
            while written < size:
                chunk = payload[: min(len(payload), size - written)]
                handle.write(chunk)
                written += len(chunk)
    except OSError as exc:
        return group_not_taken(
            "upload", f"synthetic recording could not be written: {exc}", probe="probe_upload"
        )
    try:
        with source.open("rb") as handle:
            started = time.perf_counter()
            stored = save_video_file(handle, "perf_synthetic.mp4", videos_dir)
            elapsed = time.perf_counter() - started
    except (OSError, ValueError) as exc:
        return group_not_taken(
            "upload", f"store path failed: {exc}", probe="probe_upload"
        )
    finally:
        source.unlink(missing_ok=True)
    conditions["stored_bytes"] = stored.stat().st_size
    results = [
        measured(
            UPLOAD_SECONDS,
            value=round(elapsed, 4),
            probe="probe_upload",
            conditions=conditions,
        )
    ]
    if elapsed > 0:
        results.append(
            measured(
                UPLOAD_THROUGHPUT,
                value=round((size / 1_000_000) / elapsed, 3),
                probe="probe_upload",
                conditions=conditions,
            )
        )
    else:
        results.append(
            not_taken(
                UPLOAD_THROUGHPUT,
                reason="store completed below the clock's resolution",
                probe="probe_upload",
                conditions=conditions,
            )
        )
    return complete_group("upload", results, probe="probe_upload")


_MODEL_BODY = """
import sys
sys.path.insert(0, {repo_root!r})
import numpy as np
from cv_lab.scripts.pipeline.card_classifier import CardClassifier
from cv_lab.scripts.pipeline.evaluate_yolo_cards import (
    DEFAULT_YOLOV12_VENDOR,
    _load_yolo_class,
    _resolve_vendor_path,
)

_vendor = _resolve_vendor_path(str(DEFAULT_YOLOV12_VENDOR))
_yolo = _load_yolo_class(_vendor)
_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
_crop = np.zeros((96, 64, 3), dtype=np.uint8)

_t0 = _pt.perf_counter()
_detector = _yolo({detector!r})
_detector_construct = _pt.perf_counter() - _t0
_detector.predict(_frame, imgsz=640, conf=0.35, verbose=False)
_detector_ready = _pt.perf_counter() - _t0

_t1 = _pt.perf_counter()
_classifier = CardClassifier(weights={classifier!r}, vendor=_vendor, imgsz=128, device="")
_classifier_construct = _pt.perf_counter() - _t1
_classifier.classify(_crop)
_classifier_ready = _pt.perf_counter() - _t1

_extra = {{
    "detector_seconds": round(_detector_ready, 4),
    "classifier_seconds": round(_classifier_ready, 4),
    "detector_construct_seconds": round(_detector_construct, 4),
    "classifier_construct_seconds": round(_classifier_construct, 4),
}}
"""


def probe_models(ctx: ProbeContext) -> list[Measurement]:
    """Time from a cold process to a model that can process its first frame.

    Construction alone is the wrong quantity: ``CardClassifier`` defers loading
    its weights until the first ``classify`` call, so a construction-only figure
    reads as zero and hides the whole cost. Each model here is constructed and
    then run once on a synthetic frame, and the split between the two is kept in
    the conditions so a reader can see which part is lazy.
    """
    resolved = resolve_models(ctx.repo_root)
    missing = [role for role, entry in resolved.items() if not entry.get("present")]
    if missing:
        return group_not_taken(
            "models",
            f"weights not installed for: {', '.join(sorted(missing))}",
            probe="probe_models",
        )
    detector = str(ctx.repo_root / str(resolved["region_detector"]["path"]))
    classifier = str(ctx.repo_root / str(resolved["card_classifier"]["path"]))
    child = run_child(
        ctx,
        name="model_init",
        body=_MODEL_BODY.format(
            repo_root=str(ctx.repo_root), detector=detector, classifier=classifier
        ),
        timeout=ctx.timeout("models"),
    )
    conditions: dict[str, Any] = {
        "region_detector": resolved["region_detector"],
        "card_classifier": resolved["card_classifier"],
        "device": "default (unset)",
        "includes": "construction and one warm-up inference on a synthetic frame",
        "excludes": "torch/ultralytics import cost, measured separately",
        "warmup_frame": "1280x720 zeros for the detector, 96x64 zeros for the classifier",
    }
    if not child.ok:
        return group_not_taken(
            "models", child.error or "model probe produced no timing", probe="probe_models"
        )
    conditions["construct_only_seconds"] = {
        MODEL_DETECTOR_INIT.name: child.extra.get("detector_construct_seconds"),
        MODEL_CLASSIFIER_INIT.name: child.extra.get("classifier_construct_seconds"),
    }
    return [
        measured(
            MODEL_DETECTOR_INIT,
            value=float(child.extra.get("detector_seconds") or 0.0),
            probe="probe_models",
            conditions=conditions,
        ),
        measured(
            MODEL_CLASSIFIER_INIT,
            value=float(child.extra.get("classifier_seconds") or 0.0),
            probe="probe_models",
            conditions=conditions,
        ),
        measured(
            MODEL_INIT_RSS,
            value=child.peak_rss_bytes or 0,
            probe="probe_models",
            conditions=conditions,
        ),
    ]


def representative_case(ctx: ProbeContext) -> tuple[Path, dict[str, Any]] | None:
    """The recording a representative-session measurement should use.

    An explicitly supplied video wins. Otherwise the first release-scored case
    in the manifest whose recording is present under the validation root -- the
    same resolution the release gate performs, so both talk about the same
    corpus.
    """
    if ctx.video is not None:
        path = ctx.video.expanduser().resolve()
        if path.is_file():
            return path, {"case_id": "operator_supplied", "recording": path.name}
        return None
    root_value = os.environ.get(VALIDATION_ROOT_ENV)
    if not root_value:
        return None
    root = Path(root_value).expanduser()
    if not ctx.manifest_path.is_file():
        return None
    try:
        document = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for case in document.get("cases") or []:
        if not isinstance(case, dict) or case.get("counts_toward_release") is False:
            continue
        recording = case.get("recording")
        if not isinstance(recording, dict):
            continue
        logical = recording.get("logical_name")
        if not isinstance(logical, str) or not logical:
            continue
        candidate = (root / logical).resolve()
        if candidate.is_file():
            return candidate, {
                "case_id": case.get("case_id"),
                "recording": logical,
                "duration_s": recording.get("duration_s"),
                "split": case.get("split"),
            }
    return None


_RECONSTRUCTION_BODY = """
import runpy
import sys

sys.argv = {argv!r}
_started = _pt.perf_counter()
runpy.run_path({script!r}, run_name="__main__")
"""


def probe_reconstruction(ctx: ProbeContext) -> list[Measurement]:
    """Reconstruct one real recording end to end and time it.

    Run through ``runpy`` inside the measured child rather than as a grandchild
    process, so the peak resident set reported belongs to the reconstruction and
    not to whatever else the harness has waited on.
    """
    script = ctx.repo_root / PIPELINE_SCRIPT
    if not script.is_file():
        return group_not_taken(
            "reconstruction",
            f"pipeline script not found at {PIPELINE_SCRIPT}",
            probe="probe_reconstruction",
        )
    resolved = resolve_models(ctx.repo_root)
    missing = [role for role, entry in resolved.items() if not entry.get("present")]
    if missing:
        return group_not_taken(
            "reconstruction",
            f"weights not installed for: {', '.join(sorted(missing))}",
            probe="probe_reconstruction",
        )
    selected = representative_case(ctx)
    if selected is None:
        return group_not_taken(
            "reconstruction",
            (
                "no representative recording is available: set "
                f"{VALIDATION_ROOT_ENV} to a vault holding a manifest case, or pass --video"
            ),
            probe="probe_reconstruction",
        )
    video, case = selected
    ctx.prepare()
    timeline_path = ctx.workspace / "artifacts" / "perf_timeline.json"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    duration = case.get("duration_s")
    if isinstance(duration, (int, float)):
        end = float(duration)
        duration_source = "manifest"
    else:
        probed = video_duration_seconds(video)
        end = probed if probed else 86_400.0
        duration_source = "container probe" if probed else "unbounded default"
    argv = [
        str(script),
        "--video",
        str(video),
        "--start",
        "0",
        "--end",
        str(end),
        "--interval",
        "1",
        "--out",
        str(timeline_path),
        "--model1",
        str(ctx.repo_root / str(resolved["region_detector"]["path"])),
        "--model2",
        str(ctx.repo_root / str(resolved["card_classifier"]["path"])),
    ]
    conditions = {
        "case": case,
        "video_bytes": video.stat().st_size,
        "sample_interval_s": 1.0,
        "end_s": end,
        "duration_source": duration_source,
        "region_detector": resolved["region_detector"],
        "card_classifier": resolved["card_classifier"],
        "timeout_s": ctx.timeout("reconstruction"),
        # Wall time and throughput are end to end, model loading included,
        # because that is what a job costs. Subtract model_init.* to separate
        # the two rather than reading either number as the other.
        "includes": "model initialization, decode, inference and spine assembly",
    }
    child = run_child(
        ctx,
        name="reconstruction",
        body=_RECONSTRUCTION_BODY.format(argv=argv, script=str(script)),
        timeout=ctx.timeout("reconstruction"),
    )
    if not child.ok or child.seconds is None:
        return group_not_taken(
            "reconstruction",
            child.error or "reconstruction produced no timing",
            probe="probe_reconstruction",
        )
    results = [
        measured(
            RECONSTRUCTION_SECONDS,
            value=round(child.seconds, 3),
            probe="probe_reconstruction",
            conditions=conditions,
        ),
        measured(
            RECONSTRUCTION_RSS,
            value=child.peak_rss_bytes or 0,
            probe="probe_reconstruction",
            conditions=conditions,
        ),
    ]
    if timeline_path.is_file():
        results.append(
            measured(
                RECONSTRUCTION_TIMELINE_BYTES,
                value=timeline_path.stat().st_size,
                probe="probe_reconstruction",
                conditions=conditions,
            )
        )
    else:
        results.append(
            not_taken(
                RECONSTRUCTION_TIMELINE_BYTES,
                reason="the reconstruction wrote no timeline",
                probe="probe_reconstruction",
                conditions=conditions,
            )
        )
    match = _SAMPLED_FRAMES.search(child.output)
    if match is None:
        reason = "the pipeline did not report a sampled frame count"
        results.append(
            not_taken(
                RECONSTRUCTION_FRAMES,
                reason=reason,
                probe="probe_reconstruction",
                conditions=conditions,
            )
        )
        results.append(
            not_taken(
                RECONSTRUCTION_FPS,
                reason=f"throughput needs a frame count and {reason}",
                probe="probe_reconstruction",
                conditions=conditions,
            )
        )
        return complete_group("reconstruction", results, probe="probe_reconstruction")
    frames = int(match.group(1))
    results.append(
        measured(
            RECONSTRUCTION_FRAMES,
            value=frames,
            probe="probe_reconstruction",
            conditions=conditions,
        )
    )
    if child.seconds > 0:
        results.append(
            measured(
                RECONSTRUCTION_FPS,
                value=round(frames / child.seconds, 4),
                probe="probe_reconstruction",
                conditions={**conditions, "frames": frames},
            )
        )
    else:
        results.append(
            not_taken(
                RECONSTRUCTION_FPS,
                reason="reconstruction reported no elapsed time",
                probe="probe_reconstruction",
                conditions=conditions,
            )
        )
    return complete_group("reconstruction", results, probe="probe_reconstruction")


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open an operator's database in a mode that cannot write to it.

    Structural rather than a promise: ``mode=ro`` makes a stray write fail with
    "attempt to write a readonly database" instead of migrating, vacuuming or
    lock-upgrading a library the harness was only supposed to look at.
    """
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def probe_solver(ctx: ProbeContext) -> list[Measurement]:
    """Summarize the solver runtimes this installation has actually recorded.

    Read-only by construction: the database is opened with ``mode=ro`` so the
    harness cannot migrate, lock-upgrade or otherwise touch an operator's
    library while measuring it. Recorded history is used rather than a fresh
    solve because a synthetic solve on an arbitrary tree would answer a question
    nobody asked.
    """
    db_path = ctx.db_path
    conditions = {
        "database": str(db_path),
        "source": "solver_runs rows with status='completed' and a recorded runtime",
        "access": "sqlite mode=ro",
    }
    if not db_path.is_file():
        return group_not_taken(
            "solver",
            f"no database at {db_path}",
            probe="probe_solver",
        )
    try:
        connection = open_readonly(db_path)
        try:
            # 'completed' is the only status this product ever writes for a run
            # that produced a result; SolverRunStatus has never contained
            # 'succeeded', so the predicate this used to carry matched nothing
            # and every installation reported zero recorded solver runs.
            rows = connection.execute(
                "SELECT runtime_seconds FROM solver_runs "
                "WHERE status = 'completed' AND runtime_seconds IS NOT NULL"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return group_not_taken(
            "solver", f"database could not be read: {exc}", probe="probe_solver"
        )
    runtimes = [float(row[0]) for row in rows if row[0] is not None]
    results = [
        measured(
            SOLVER_RUNS,
            value=len(runtimes),
            probe="probe_solver",
            conditions=conditions,
        )
    ]
    if not runtimes:
        reason = "this installation has recorded no completed solver run"
        results.append(
            not_taken(SOLVER_MEDIAN, reason=reason, probe="probe_solver", conditions=conditions)
        )
        results.append(
            not_taken(SOLVER_MAX, reason=reason, probe="probe_solver", conditions=conditions)
        )
        return complete_group("solver", results, probe="probe_solver")
    results.append(
        measured(
            SOLVER_MEDIAN,
            value=round(statistics.median(runtimes), 3),
            probe="probe_solver",
            conditions={**conditions, "runs": len(runtimes)},
        )
    )
    results.append(
        measured(
            SOLVER_MAX,
            value=round(max(runtimes), 3),
            probe="probe_solver",
            conditions={**conditions, "runs": len(runtimes)},
        )
    )
    return complete_group("solver", results, probe="probe_solver")


PROBES: dict[str, Callable[[ProbeContext], list[Measurement]]] = {
    "imports": probe_imports,
    "startup": probe_startup,
    "ui": probe_ui,
    "upload": probe_upload,
    "models": probe_models,
    "reconstruction": probe_reconstruction,
    "solver": probe_solver,
}
