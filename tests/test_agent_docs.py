"""The agent instruction files must keep describing the docs that exist.

A hook mirrors every ``AGENTS.md``/``CLAUDE.md`` pair, so the two names can never
disagree. The failure it cannot catch is the adapters going stale against
``docs/``: rename a canonical doc, or add one and forget to link it, and both
adapters stay byte-identical while pointing at a file that is not there. An agent
follows the dead link, finds nothing, and falls back to reading the repository --
which is the token cost the docs were written to remove.

These run the same check that CI and ``python scripts/check_agent_docs.py`` run,
so the guard cannot pass locally and fail in CI for a different reason.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_agent_docs.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_agent_docs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_checker_script_exists_and_is_importable():
    assert SCRIPT.is_file(), "scripts/check_agent_docs.py is the guard; it must exist"
    assert _load_checker().CANONICAL, "the canonical doc list must not be empty"


def test_adapters_and_canonical_docs_agree():
    failures = _load_checker().check()
    assert failures == [], "\n".join(failures)


def test_every_canonical_doc_is_a_real_file():
    checker = _load_checker()
    missing = [doc for doc in checker.CANONICAL if not (REPO / doc).is_file()]
    assert missing == [], f"canonical docs missing from the repo: {missing}"


def test_adapters_stay_short_enough_to_be_cheap():
    """An adapter is read on every agent's first turn, so its length is a tax.

    The number is a ceiling, not a target. It exists because the previous
    instruction file grew to carry the complete rules in both copies, which is
    exactly what moving them into ``docs/`` was meant to stop.
    """

    for name in ("AGENTS.md", "CLAUDE.md"):
        lines = (REPO / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 120, (
            f"{name} is {len(lines)} lines. Adapters point at docs/; put the rule "
            f"in docs/ rather than growing the file every agent reads first."
        )


def test_the_checker_notices_a_missing_canonical_doc(tmp_path, monkeypatch):
    """The guard has to fail when it should, or it is decoration.

    Pointed at a scratch tree containing a mirrored adapter pair that names a doc
    which is not there, `check()` must report it.
    """

    checker = _load_checker()
    adapter = "docs/agent-guidelines.md docs/architecture.md\n"
    (tmp_path / "AGENTS.md").write_text(adapter, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(adapter, encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", tmp_path)

    failures = checker.check()
    assert any("missing" in failure for failure in failures), failures


def test_the_checker_notices_adapters_that_disagree(tmp_path, monkeypatch):
    checker = _load_checker()
    (tmp_path / "AGENTS.md").write_text("one\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("two\n", encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", tmp_path)

    failures = checker.check()
    assert any("differ" in failure for failure in failures), failures
