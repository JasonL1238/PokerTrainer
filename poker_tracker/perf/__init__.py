"""Local performance and resource measurement (Phase 13).

Kept out of ``release_gate`` because it measures the whole product -- server
startup, UI render, upload, model init, reconstruction, solver history -- rather
than the gate's own run, and because a gate verdict must not depend on how busy
the machine was. It reuses the gate's resource and environment accounting so a
performance report and a release report describe a host the same way, and a
later change can fold its output into the gate's report without moving code.

See docs/PERFORMANCE.md.
"""

from poker_tracker.perf.harness import HarnessOptions, empty_baseline, run_harness
from poker_tracker.perf.measurement import Measurement, MeasurementSpec, PerfReport

__all__ = [
    "HarnessOptions",
    "Measurement",
    "MeasurementSpec",
    "PerfReport",
    "empty_baseline",
    "run_harness",
]
