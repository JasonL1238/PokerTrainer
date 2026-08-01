"""Find every pinned dependency that cannot install from a wheel on a target architecture.

linux/arm64 is a stated deployment target, and it was impossible: eval7 ships no
aarch64 wheel, it is a Cython extension, and the runtime image has no compiler.
That was found by hand. Finding it by hand does not tell anyone whether the next
dependency bump reintroduces it, or whether some other package has the same
problem, so this reproduces the check.

For each architecture it asks pip to resolve the requirements with wheels only,
and reports whatever cannot be satisfied. Anything it names has to be built from
source, which means the container build needs a compiler stage that handles it --
see deploy/docker/build_python_env.sh.

    python deploy/check_wheel_availability.py
    python deploy/check_wheel_availability.py --arch aarch64

Needs network access. It is not part of the unit suite for that reason; the
static half of the same contract lives in
deploy/tests/test_container_build_contract.py.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11"

# Packages the container build is known to compile from source. Anything else
# turning up is a new problem, not a known one.
SOURCE_BUILT_IN_CONTAINER = {"eval7"}

# pip matches --platform tags literally, so the list has to name every manylinux
# baseline the pinned wheels actually use. polars, for one, publishes only
# manylinux_2_24 and is invisible to a shorter list.
_MANYLINUX_MINORS = (5, 12, 17, 24, 26, 27, 28, 31, 34, 35, 36, 38, 39)


def platform_tags(arch: str) -> list[str]:
    tags = [f"manylinux_2_{minor}_{arch}" for minor in _MANYLINUX_MINORS]
    tags += [f"manylinux2014_{arch}", f"manylinux1_{arch}", f"linux_{arch}"]
    return tags


def requirements_for(arch: str, excluded: set[str]) -> tuple[Path, Path]:
    """Both requirement files with markers resolved for ``arch``.

    pip evaluates ``platform_machine`` against the running interpreter even when
    --platform names another architecture, so the marked torch lines have to be
    selected here instead.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="wheelcheck-"))
    out_paths = []
    for name in ("requirements.txt", "requirements-cv.txt"):
        lines: list[str] = []
        for raw in (REPO_ROOT / name).read_text(encoding="utf-8").splitlines():
            line = re.sub(r"\s+#.*$", "", raw).strip()
            if not line or line.startswith("#"):
                lines.append(raw)
                continue
            spec, _, marker = line.partition(";")
            if marker and "platform_machine" in marker:
                if arch not in marker:
                    continue
                line = spec.strip()
            project = re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower()
            if project in excluded:
                continue
            lines.append(line)
        path = temp_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_paths.append(path)
    return out_paths[0], out_paths[1]


_UNSATISFIED = re.compile(
    r"No matching distribution found for ([A-Za-z0-9._-]+)|"
    r"Could not find a version that satisfies the requirement ([A-Za-z0-9._-]+)"
)


def unsatisfiable(arch: str) -> list[str]:
    """Every requirement with no usable wheel, found by excluding and retrying.

    pip stops at the first failure, so one run names one package.
    """
    failures: list[str] = []
    excluded: set[str] = set()
    for _ in range(12):
        base, cv = requirements_for(arch, excluded)
        with tempfile.TemporaryDirectory() as target:
            command = [
                sys.executable, "-m", "pip", "install",
                "--dry-run", "--no-cache-dir", "--only-binary=:all:",
                "--python-version", PYTHON_VERSION, "--implementation", "cp",
                "--target", target,
                "-r", str(base), "-r", str(cv),
            ]
            for tag in platform_tags(arch):
                command += ["--platform", tag]
            completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            return failures
        match = _UNSATISFIED.search(completed.stdout + completed.stderr)
        if match is None:
            raise RuntimeError(
                "pip failed for a reason other than a missing wheel:\n"
                + (completed.stderr or completed.stdout)[-2000:]
            )
        name = (match.group(1) or match.group(2)).lower()
        if name in excluded:
            raise RuntimeError(f"excluding {name} did not change the outcome")
        failures.append(name)
        excluded.add(name)
    raise RuntimeError("too many unsatisfiable requirements to enumerate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", action="append", choices=["x86_64", "aarch64"])
    args = parser.parse_args(argv)
    architectures = args.arch or ["x86_64", "aarch64"]

    status = 0
    for arch in architectures:
        print(f"resolving requirements for linux/{arch} with wheels only ...")
        failures = set(unsatisfiable(arch))
        if not failures:
            print(f"  every requirement has a {arch} wheel")
            continue
        for name in sorted(failures):
            known = name in SOURCE_BUILT_IN_CONTAINER
            print(f"  {'known' if known else 'NEW  '}  {name}: no {arch} wheel, must be compiled")
        unknown = failures - SOURCE_BUILT_IN_CONTAINER
        if unknown:
            print()
            print(f"  {sorted(unknown)} must be built from source on {arch}.")
            print("  deploy/docker/build_python_env.sh handles only "
                  f"{sorted(SOURCE_BUILT_IN_CONTAINER)}; extend it, or the "
                  f"linux/{arch} image will not build.")
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
