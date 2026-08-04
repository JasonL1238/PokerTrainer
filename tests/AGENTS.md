# tests/ — agent instructions

Canonical rules: [../docs/agent-guidelines.md](../docs/agent-guidelines.md) and
[../docs/testing.md](../docs/testing.md). This file adds only what is different
here.

196 files, 87k lines. Never scan the directory to find a test — grep for the
behaviour or the symbol.

## The sandbox invariant

`conftest.py` claims `POKER_DB_PATH` and `POKER_DATA_DIR` into a temp tree **at
import time**, before the first `poker_tracker` import. Every operator root is a
module-level constant resolved once, so that ordering is the entire mechanism.

- It raises `RuntimeError` if `poker_tracker` was already imported. That is what
  `-p poker_tracker.<anything>` causes. Use `-p sq_random_order` (the root shim).
- Do not add a module-level `poker_tracker` import above the redirect in
  `conftest.py`.
- Never point a test at the real `poker_tracker.db`.

## Files that are records, not living tests

`test_phase1_adversarial_round*.py` (13 files) are point-in-time regression
records — each opens "Every test here failed before its fix." Do not refactor them
for style, deduplicate across them, or delete an apparently duplicated test:
two rounds asserting the same thing usually record two different arguments about
it, and the docstrings carry the reasoning even when the bodies match. Later
commits have touched some of them, so they are not literally sealed — but change
them for a *reason*, not for tidiness.

## Tests that read source text

Several tests assert invariants by reading `app.py` (or all of `poker_tracker/`)
as text or AST, and they are load-bearing safety guards, not style checks:

| Test | Pins |
|---|---|
| `test_study_readiness_ui.py:950,960` | exactly one raw `db.update_hand_status(`; ≥4 guarded writes |
| `test_phase10_insights_settings.py:973,996` | one call site per destructive writer (`db.delete_session/roi_profile/hand/video`); `_remove_hand_and_artifacts` keeps its required `snapshot` keyword and exactly two callers |
| `test_phase10_hands_study.py:436` | one definition, one call of `hand_evidence_badges` |
| `test_coaching_response_write_boundary.py` | every persisted response was built by `build_coaching_response` |
| `test_provenance_source_and_duplicate_totals.py:166` | a symbol appears inside one `app.py` region and not outside |
| `test_phase1_adversarial_round15.py:1166,1316` | sweeps `app.py` + all of `poker_tracker/` |

**If you move code between files, these break.** Update them to scan the new
corpus in a way that *preserves the guarantee* — "exactly one unguarded write in
the codebase", not "exactly one per file", which is a weaker claim wearing the
same assertion.

## Conventions

- A skip needs a reason naming the condition, or
  `python -m poker_tracker.suite_quality skips tests` fails.
- Duplication between test files is often deliberate: independent setup keeps a
  failure diagnosable. Do not hoist a helper into `conftest.py` just because two
  files share it — check whether the copies really behave identically first.
  Several near-identical helpers differ in a timeout or in what they search.
- Assert the invariant, not the symptom.
