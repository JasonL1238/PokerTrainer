# Repository map

Where things are, where a change belongs, and what to run. Read this before
opening files.

## Directories

| Path | What it is | Read it? |
|---|---|---|
| `app.py` | Streamlit UI, 11.7k lines, 221 top-level defs | **Never whole.** Use the line map below |
| `poker_tracker/` | Product package, 104 files. See [architecture.md](architecture.md#major-components) | Per-package |
| `tests/` | 196 files, 87k lines. Has its own `AGENTS.md` | Targeted only |
| `deploy/` | Dockerfile support, container contract tests, model provisioning | Yes, small |
| `docs/` | Canonical docs (this file, architecture, testing, agent-guidelines) + operator runbooks | Yes |
| `cv_lab/scripts/`, `cv_lab/notes/` | CV research **code and notes**. Has its own `AGENTS.md` | Only for CV work |
| `cv_lab/datasets/ frames/ crops/ runs/ models/` | **3.3 GB of data.** Never scan | No |
| `data/` | **1.0 GB** operator videos, frames, exports | No |
| `artifacts/`, `.hypothesis/`, `.*_cache/` | Generated output | No |
| `validation/` (root) | Frozen corpus manifest + answer key (`clubwpt_v1.json`, `truth/`) | Read, never edit casually |
| `poker_tracker.db*` | The operator's live database. Untracked | Never read or migrate |

Only **423 files are tracked**. `git ls-files` is the reliable survey tool; `find`
and `ls` walk 4+ GB of ignored data.

## Where common changes belong

| Change | Goes in |
|---|---|
| New UI surface / workspace tab | `app.py` (the matching `show_*_workspace`) |
| Reusable UI helper, view model | `poker_tracker/ui/` |
| Chip/pot/rake maths, ledger rules | `poker_tracker/math/accounting.py` |
| Analytics, coverage, population metrics | `poker_tracker/math/analytics.py` |
| Reconciliation, readiness, import gating | `poker_tracker/services/` |
| Table, column, migration | `poker_tracker/persistence/` — read its `AGENTS.md` first |
| Solver spot, ranges, job lifecycle | `poker_tracker/solver/` |
| Prompt, provider, grounding | `poker_tracker/coaching/` |
| Detector, classifier, OCR, timeline | `cv_lab/scripts/` — read its `AGENTS.md` first |
| Backup, retention, health audit | `poker_tracker/maintenance/` |

## `app.py` line map

Line numbers drift. Regenerate with:

```bash
grep -nE "^def (main|show|render)_?" app.py
```

Entry: `main()` at **983** — builds the sidebar and dispatches to one workspace.

| Workspace | Entry | Major surfaces within |
|---|---|---|
| Overview | `show_product_overview` **1512** | `render_storage_health` 1240, `render_health_report` 1317, `render_data_state_axes` 1452 |
| Sessions | `show_sessions_workspace` **1789** | `show_session_library` 5814, `show_session_hand_browser` 6468, `show_session_videos` 6637, `show_session_dashboard` 7066, `show_saved_hands` 7591 |
| Hands | `show_hands_workspace` **1848** | `show_hand_issue_queue` 1958, `show_player_editor` 7729, `show_action_editor` 7810 |
| Study | `show_study_workspace` **2171** | `render_study_replay` 2401, `render_validation_edit_and_approve` 2882, `render_study_analysis` 3267, `show_study_coach_review` 3754, `show_solver_review` 3952, `show_accounting_editor` 4613 |
| Insights | `show_insights_workspace` **5181** | `render_population_metric` 5062, `render_evidence_split` 5115, `render_study_themes` 5156 |
| Import | `show_import_workspace` **5387** | `show_video_processing` 10062, `show_cv_reconstruction` 10195, `render_pinned_hand_repair` 10404, `show_reconstruction_evidence_review` 10693 |
| Settings | `show_settings_workspace` **5420** | `show_storage_and_diagnostics` 5474, `show_solver_settings` 5700, `show_coach_review` 9625, `show_roi_calibration` 11519 |

Frequently needed helpers: `save_hand_coaching` (the coaching write boundary) and
`save_generated_hand_coaching` beside it (build-then-persist, used by all three
generating surfaces), `show_prompt_safety`, `render_study_readiness`,
`hand_history_text` (the one assembler of a hand's history text — takes
already-fetched records, never re-queries), `_reconcile_cached` /
`new_accounting_cache` (per-render reconciliation cache; pass it down rather than
reconciling again), `BLOCKER_CATEGORY_LABELS`.

Regenerate their line numbers with the `grep` above rather than trusting these —
they drift on every edit, which is why only the workspace anchors are numbered.

## Deletion writers

Every destructive product write goes through one writer in `app.py`, each of
which takes its rollback snapshot before touching anything. Add a delete control
by calling one of these, never by reaching `db.delete_*` directly —
`tests/test_phase10_insights_settings.py` fails a second call site.

| Writer | Removes | Notes |
|---|---|---|
| `delete_hand_and_artifacts` | one hand + solver runs | snapshot per hand |
| `delete_hands_and_artifacts` | several hands of one session | **one** snapshot for the batch |
| `delete_video_and_artifacts` | one recording, its files, its jobs' artifacts | refuses while a job is live or still launching |
| `_render_session_danger_zone` | a session | hands cascade; recordings are unlinked, not deleted |

`render_video_danger_zone(db, video, *, key_prefix)` is the shared UI control for
the third; it is mounted from the session recording list and from the legacy
frame-extraction panel on Import, so pass a distinct `key_prefix`.

The four standalone study calculators (realization, multiway equity, outs, ICM)
now live in `poker_tracker/ui/equity_tools.py` and are re-exported from `app.py`,
so `app.show_icm_tool` still resolves.

## Validation commands

```bash
# targeted -> package -> full  (always in this order)
pytest tests/test_solver.py::test_name       # one test
pytest tests/test_solver.py -q               # one file
ruff check poker_tracker/solver              # lint what you touched
python -m mypy                               # typed subset only (see pyproject)
pytest -q                                    # full: 3823 tests, ~3.5 min

# subsystem CLIs
python -m poker_tracker.suite_quality skips tests
python -m poker_tracker.suite_quality flake --passes 3 --seeds 20260801,20260802
python -m poker_tracker.suite_quality coverage
python -m poker_tracker.validation --help
python -m poker_tracker.release_gate --mode fixture
python -m poker_tracker.maintenance --json
python -m poker_tracker.perf run

# container contract (no daemon needed)
pytest deploy/tests -q

# run the app
streamlit run app.py

# agent-doc drift guard
python scripts/check_agent_docs.py
```

See [testing.md](testing.md) for what needs which environment, and for the
4 known-failing tests.
