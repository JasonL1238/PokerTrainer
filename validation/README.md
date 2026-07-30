# Private validation corpus

This directory holds the **committed** Phase 2 corpus artifacts:

- `clubwpt_v1.json` — hash-locked manifest (case IDs, splits, truth hashes, coverage tags)
- `truth/` — answer-key JSON documents

Raw ClubWPT recordings are **not** stored here. Put them under
`POKER_VALIDATION_ROOT` and reference them by `recording.logical_name`.

## Commands

```bash
# Release-shaped check (fails until 10 sessions / 100 completed / 10 partial, etc.)
python -m poker_tracker.validation --manifest validation/clubwpt_v1.json

# Schema + hash plumbing only (used by unit tests / early development)
python -m poker_tracker.validation --manifest validation/clubwpt_v1.json --skip-release-minimums

# Freeze an answer-key hash after intentional edits
python -m poker_tracker.validation --hash-truth validation/truth/fixture_synthetic_hu_01.json

# Local release: also verify raw video digests
export POKER_VALIDATION_ROOT=/path/to/private/videos
python -m poker_tracker.validation --manifest validation/clubwpt_v1.json --require-recordings
```

Exit codes: `0` pass, `1` product/content gate failure (e.g. hash mismatch,
illegal truth), `2` setup / incomplete corpus (below Phase 2 floors, missing
adjudication, missing recordings when required).
