# Repository agent instructions

Adapter file. The rules live in `docs/` so there is exactly one copy to keep
current.

`AGENTS.md` and `CLAUDE.md` are **byte-identical mirrors of this same text** — a
repo hook copies either onto the other on save. Edit one; the other follows. That
makes vendor drift structurally impossible rather than merely audited, so this
file addresses every agent at once: Claude Code, Codex, Cursor, and anything else
that reads one of the two names.

This is a local-first, post-session poker study and review platform.

## Read these

| Read | When |
|---|---|
| [docs/agent-guidelines.md](docs/agent-guidelines.md) | **Always.** Exploration, editing, validation, parallelism, completion rules |
| [docs/repository-map.md](docs/repository-map.md) | **Always.** Directory purposes, where a change belongs, `app.py` line map, commands |
| [docs/architecture.md](docs/architecture.md) | Before a change crossing package boundaries |
| [docs/testing.md](docs/testing.md) | Before running or writing tests |

Then read the nearest nested `AGENTS.md`. They exist at
`poker_tracker/persistence/`, `tests/`, and `cv_lab/`.

`README.md` is the operator setup guide; `PLAN.md` is status and roadmap. Neither
is an agent brief — do not read them for orientation.

## Non-negotiable, repeated so it never needs a second file read

- Never build real-time poker assistance.
- Never build live table capture.
- Never build a poker-client overlay.
- Never provide current-hand recommendations.
- Analysis is for completed hands/sessions only.
- Never identify yourself, an AI assistant, or an automated agent as a commit
  author, co-author, contributor, signer, or attribution in Git metadata or
  commit messages.
- Do not change database schemas without explaining migration impact.
- Preserve existing API response formats unless explicitly authorized.
- Do not commit, push, or deploy unless asked.

## Fast start

```bash
git ls-files | grep <thing>           # 423 tracked files; do not `find` (4+ GB ignored data)
grep -n "def <symbol>" <path>         # never read app.py (11.7k) or db.py (6.9k) whole
pytest tests/test_<x>.py::<test> -q   # targeted first
ruff check <path> && python -m mypy   # then lint and types
pytest -q                             # full suite last (~3.5 min, 3823 tests)
```

Install: `pip install -r requirements-dev.txt` · App: `streamlit run app.py`

## Keeping the adapters and docs in step

The hook guarantees the two adapters match each other. It cannot guarantee they
still match `docs/`, so `python scripts/check_agent_docs.py` verifies that both
adapters are identical, that both reference every canonical doc, that no canonical
doc is orphaned, and that the non-negotiable constraints above appear in both.
`tests/test_agent_docs.py` runs it under `pytest`.

Put shared rules in `docs/`. Keep these adapters short — everything here is paid
for on every agent's first turn.
