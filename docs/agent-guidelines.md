# Agent guidelines

Canonical rules for any coding agent working in this repository. Vendor-neutral:
`AGENTS.md` and `CLAUDE.md` are thin adapters that point here, so there is one
copy of the rules to keep current.

Read this file plus [repository-map.md](repository-map.md). Read
[architecture.md](architecture.md) only when a change crosses package
boundaries, and [testing.md](testing.md) before running anything.

## Product boundary (non-negotiable)

Never build real-time poker assistance, live table capture, a poker-client
overlay, or current-hand recommendations. Analysis is for completed
hands and sessions only. A request that needs any of these is refused, not
worked around.

## Exploration

- Start with the smallest relevant module. Do not read a package to change a
  function.
- Read the nearest applicable `AGENTS.md`/`CLAUDE.md` first. Nested ones exist
  at `poker_tracker/persistence/`, `tests/`, and `cv_lab/`.
- Search for the symbol before opening the file. `grep -n "def name" path` then
  read a window. Two files make a whole-file read a mistake: `app.py`
  (11.7k lines) and `poker_tracker/persistence/db.py` (6.9k lines). Line
  anchors for both are in [repository-map.md](repository-map.md).
- Avoid repo-wide scans. Scope every search to a package or `tests/`.
- Never glob or scan these — they are gigabytes of data, not code:
  `cv_lab/datasets/`, `cv_lab/frames/`, `cv_lab/crops/`, `cv_lab/runs/`,
  `cv_lab/models/`, `data/`, `artifacts/`, `.hypothesis/`, `*.pt`, `*.log`,
  `poker_tracker.db*`, any `.*_cache/`. `.gitignore` excludes most of them from
  git, which does **not** stop a filesystem tool from walking them.
- Prefer `git ls-files` (423 tracked files) over `find`/`ls` when surveying.

## Editing

- Reuse the existing type, utility, service, or pattern. Search before adding an
  abstraction; near-duplicate helpers already exist in this codebase and adding
  a parallel one has been a repeated mistake.
- Keep edits inside the requested scope. Do not rewrite unrelated working code,
  reformat untouched lines, or "improve" adjacent functions.
- Do not change a database schema without stating the migration impact.
  `SCHEMA_VERSION` is 20; see `poker_tracker/persistence/AGENTS.md`.
- Do not change persisted API/response shapes without explicit authorization.
- Match the surrounding code's comment density and naming. This repo documents
  *why* in prose comments; keep that when editing near them.
- Never attribute a commit to an AI, agent, or assistant in any Git metadata,
  message, trailer, or signature.

## Engineering style

- Build incrementally. Prefer small, testable modules over large ones.
- Keep database, CV, OCR, analytics, equity, and coaching concerns in separate
  modules.
- Store videos, frames, timelines and exports as **files**; structured data goes
  in SQLite. Never put a video in a SQL column.
- Add tests for core behaviour. Keep functions focused and avoid adding
  dependencies.
- Ignore development labour cost when weighing implementation choices. Optimize
  for product quality, maintainability, and runtime/hosting cost.

## Runtime and deployment posture

- The app runs and is tested **locally**. There is no active hosted deployment.
- Do not deploy, provision cloud resources, or add a hosting-provider dependency
  unless explicitly asked.
- Keep the project continuously deployment-ready — containerizable without a
  feature rewrite. The Dockerfile, container healthcheck, non-root runtime, and
  `linux/amd64` + `linux/arm64` compatibility are first-class supported paths.
- Stay provider-neutral. Runtime settings belong in environment variables.
  Never commit a secret.
- Keep SQLite, videos, models, timelines and backups on explicit persistent
  mounts or external storage, never on the container filesystem.
- When a change touches dependencies, startup, storage, auth, networking, or the
  CV/model runtime, verify **both** the local Streamlit path and the Docker path
  where feasible. See [CONTAINER.md](CONTAINER.md).

## Validation

Run the cheapest check that can fail first. The ladder, in order:

```bash
pytest tests/test_<file>.py::test_<name>      # the single test you affected
pytest tests/test_<file>.py -q                # its file
ruff check <path>                             # lint the paths you touched
python -m mypy                                # typed subset (pyproject files list)
pytest -q                                     # full suite, ~3.5 min, 3823 tests
```

- Never claim a check passed unless you ran it and it passed. Quote the output.
- Report a pre-existing failure as pre-existing, with evidence that it predates
  your change. `tests/test_sbom.py` has 4 known environment-dependent failures —
  see [testing.md](testing.md#known-failures).
- Keep command output concise: `-q`, `| tail`, no full logs pasted back.

## Parallelism and subagents

- Do not use subagents for simple work. A single grep, a one-file edit, or a
  targeted test run is not subagent work.
- Give each subagent a narrow, non-overlapping scope. Two agents editing the
  same file is a conflict, not parallelism.
- Batch independent read-only tool calls into one turn instead of serializing
  them.

## Completion

- Finish the whole requested scope, or state plainly which part you did not do
  and why.
- Update the affected doc when you change architecture, entry points, commands,
  or directory layout. Documentation ownership:
  - `docs/` — canonical for agents (this file, architecture, repository map,
    testing) and for operators (`CONTAINER.md`, `PERFORMANCE.md`, `RUNBOOKS.md`).
  - `README.md` — product overview, setup, operator entrypoint.
  - `PLAN.md` — current status, release plan, definition of done.
  - `cv_lab/notes/` — chronological research record; later findings supersede
    earlier experiments. Keep historical experiments out of the roadmap.
- Do not commit, push, deploy, or provision anything unless asked.
