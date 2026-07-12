# CV Lab — Findings 06 (pipeline decision: VLM is validation-only)

## Decision (from the user)
The VLM / vision-model subagent reads are ONLY a testing & validation oracle (and an answer-key
generator). They MUST NOT be in the actual shipping pipeline. The production "read" step must be
DETERMINISTIC classic CV. This supersedes the "VLM-first hybrid" recommendation in Findings 02.

## What this changes
Nothing about the deterministic scaffold already built and validated — that IS the real pipeline:
- PyAV frame decode (`extract_*.py`)                         [KEEP]
- Region-aware key-frame selection (`select_keyframes.py`)   [KEEP — deterministic]
- Green-coin landmark anchoring (`landmark_anchor.py`)       [KEEP — deterministic]
- Arithmetic reconciliation (pot/stack consistency)         [KEEP — deterministic]

Only the READ step changes: replace "VLM structured read" with deterministic extractors on the
anchored ROIs.

## The VLM's remaining (allowed) role
- Validation oracle during my testing.
- Answer-key generation: the Experiment-A hand reconstruction (Qs Jh, t300-396) and the per-frame
  baseline reads are now the ground-truth-ish reference to score the deterministic reader against.

## Revised production pipeline (all deterministic)
1. Decode (PyAV).
2. Screen classify (table vs lobby vs transition) — cheap, e.g. coin-count / histogram. [TODO]
3. Key-frame selection (region-aware diff) on anchored ROIs. [DONE]
4. Landmark anchoring (green-coin constellation -> similarity transform). [DONE]
5. **Deterministic READ per key-frame** [TODO — the new focus]:
   - POT number: OCR the anchored pot ROI (clean light text on dark bg). Linchpin for the
     arithmetic reconciler.
   - Board + hero cards: template-match rank glyphs + suit shapes on the anchored card slots
     (also fixes the spade-vs-club ambiguity the VLM had).
   - Player stacks / bets: OCR anchored per-seat number ROIs (already BB-denominated).
   - Action pills: classify by pill color (gray=check/fold, green=call/bet, orange=raise,
     white=BB) + short OCR of the pill text.
   - Active player: detect the blue timer ring around a seat.
   - Dealer button: detect the white 'D' disc.
6. Hand segmentation (whole-frame boundary + hero-card/board/pot resets). [partial — boundary DONE]
7. Stitch street-by-street + arithmetic reconcile. [DONE for one hand]

## Validation method for the deterministic reader
Score each deterministic extractor against the VLM answer key on the Experiment-A frames:
- POT OCR vs the known pot sequence (5.5 -> 32.5 -> 65.1 -> 162.2 -> 347.5).
- Card templates vs the known board/hole cards.
- Pills/stacks vs the known per-seat values.
Only where the deterministic reader can't be made to work would we revisit needing more/less VLM
help for building ground truth (per the user's original allowance) — but not in the pipeline.
