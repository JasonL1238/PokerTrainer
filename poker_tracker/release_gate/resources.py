"""Peak-memory and disk accounting for release reports.

The report has to state what a release run actually cost, so a regression in
memory or artifact size is visible in the diff between two reports rather than
only when a machine runs out of room.
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def _maxrss_bytes(usage: resource.struct_rusage) -> int:
    """Normalize ``ru_maxrss`` across platforms.

    Linux reports kilobytes, macOS and the BSDs report bytes. Treating one as
    the other is a 1024x error in whichever direction, so the platform check is
    not optional.
    """
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def peak_memory_bytes() -> int:
    """Peak resident set of this process and every child it waited on."""
    return max(
        _maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF)),
        _maxrss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN)),
    )


def directory_bytes(path: Path) -> int:
    """Total size of regular files under ``path``.

    Symlinks are counted as their own link size, never followed: a link into the
    video vault would otherwise report the whole corpus as release-run output.
    """
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            candidate = Path(root) / name
            try:
                total += candidate.lstat().st_size
            except OSError:
                continue
    return total


@dataclass(frozen=True)
class DiskUsage:
    total_bytes: int
    free_bytes: int


def disk_usage(path: Path) -> DiskUsage | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    return DiskUsage(total_bytes=usage.total, free_bytes=usage.free)


def collect_resources(*, artifact_dirs: dict[str, Path]) -> dict[str, object]:
    """Resource footprint for the report's ``resources`` section."""
    artifacts = {name: directory_bytes(path) for name, path in artifact_dirs.items()}
    root = next(iter(artifact_dirs.values()), Path.cwd())
    usage = disk_usage(root)
    return {
        "peak_memory_bytes": peak_memory_bytes(),
        "artifact_bytes": artifacts,
        "artifact_bytes_total": sum(artifacts.values()),
        "disk_total_bytes": usage.total_bytes if usage else None,
        "disk_free_bytes": usage.free_bytes if usage else None,
    }
