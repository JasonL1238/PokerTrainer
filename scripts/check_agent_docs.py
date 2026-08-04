#!/usr/bin/env python3
"""Verify the agent instruction files have not drifted from the canonical docs.

A repo hook keeps every ``AGENTS.md``/``CLAUDE.md`` pair byte-identical, so
vendor drift between the two names is already structural. What a hook cannot
check is whether the adapters still describe the docs that exist: a canonical doc
can be renamed, or added and never linked, and both adapters stay happily
identical while pointing at nothing.

So this checks the other axis. Adapters must reference every canonical doc, every
canonical doc must be referenced by something, and the product's non-negotiable
constraints must be readable without opening a second file -- an agent that has
to follow a link to learn it may not build a real-time assistant has already been
told too late.

Stdlib only, no third-party imports, so it runs anywhere the repo does.

Usage: python scripts/check_agent_docs.py [--quiet]
Exit:  0 all checks passed, 1 a check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The four documents the adapters exist to point at.
CANONICAL = (
    "docs/agent-guidelines.md",
    "docs/architecture.md",
    "docs/repository-map.md",
    "docs/testing.md",
)

# Distinctive fragments of the product boundary. Matched as substrings so wording
# around them can change without tripping this; the constraint itself may not
# quietly vanish.
NON_NEGOTIABLE = (
    "real-time poker assistance",
    "live table capture",
    "poker-client overlay",
    "current-hand recommendations",
)

ADAPTER_NAMES = ("AGENTS.md", "CLAUDE.md")


def _adapter_dirs() -> list[Path]:
    """Every directory holding at least one adapter file."""
    found: set[Path] = set()
    for name in ADAPTER_NAMES:
        for path in REPO.rglob(name):
            # Skip anything inside ignored data or dependency trees.
            parts = set(path.relative_to(REPO).parts)
            if parts & {".git", "node_modules", ".venv", "__pycache__"}:
                continue
            found.add(path.parent)
    return sorted(found)


def check() -> list[str]:
    failures: list[str] = []

    # 1. Both adapter names exist in each directory and match byte for byte.
    for directory in _adapter_dirs():
        rel = directory.relative_to(REPO)
        texts: dict[str, bytes | None] = {}
        for name in ADAPTER_NAMES:
            candidate = directory / name
            texts[name] = candidate.read_bytes() if candidate.is_file() else None
        missing = [name for name, body in texts.items() if body is None]
        if missing:
            failures.append(
                f"{rel}/: has {', '.join(n for n in ADAPTER_NAMES if n not in missing)} "
                f"but not {', '.join(missing)} -- an agent reading the missing name "
                f"gets no instructions at all"
            )
            continue
        if texts["AGENTS.md"] != texts["CLAUDE.md"]:
            failures.append(
                f"{rel}/: AGENTS.md and CLAUDE.md differ. They are mirrors; edit "
                f"one and let the hook copy it, or re-run the hook"
            )

    root_adapter = REPO / "AGENTS.md"
    if not root_adapter.is_file():
        return failures + ["AGENTS.md is missing from the repository root"]
    root_text = root_adapter.read_text(encoding="utf-8")

    # 2. Every canonical doc exists and the root adapter points at it.
    for doc in CANONICAL:
        if not (REPO / doc).is_file():
            failures.append(f"{doc}: referenced as canonical but the file is missing")
        elif doc not in root_text:
            failures.append(
                f"{doc}: exists but the root adapter never references it, so an "
                f"agent will not find it"
            )

    # 3. No canonical doc is orphaned -- something must link to each one.
    linkers = [root_text] + [
        (REPO / doc).read_text(encoding="utf-8")
        for doc in CANONICAL
        if (REPO / doc).is_file()
    ]
    for doc in CANONICAL:
        name = Path(doc).name
        referenced = sum(
            1
            for text in linkers
            if name in text and not text.startswith(f"# {name}")
        )
        if referenced == 0:
            failures.append(f"{doc}: orphaned -- no adapter or sibling doc links to it")

    # 4. The product boundary is stated in the adapter itself.
    for phrase in NON_NEGOTIABLE:
        if phrase not in root_text:
            failures.append(
                f'root adapter no longer states the constraint "{phrase}" -- it '
                f"must be readable without following a link"
            )

    return failures


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    failures = check()
    if failures:
        print("agent-doc check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if not quiet:
        dirs = len(_adapter_dirs())
        print(
            f"agent-doc check passed: {dirs} adapter pair(s), "
            f"{len(CANONICAL)} canonical docs, boundary stated inline."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
