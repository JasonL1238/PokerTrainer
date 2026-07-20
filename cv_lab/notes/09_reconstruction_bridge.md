# CV Lab — Findings 09 (CV → hand-reconstruction bridge, measured)

## Goal and result
Close the bridge: given CV detections, reconstruct complete hands with **at most
1 thing wrong per hand**. Result on the v00 session (clubwpt_session_01.mov,
283 frames @ 2s, 7 fully-observed hands):

| path | errors/hand | hands ≤1 error |
|---|---|---|
| GT boxes (human labels + OCR) → spine | **0.14** (one label-coverage artifact) | 7/7 |
| Live two-model (detector+classifier+OCR) → spine | **0.0** | 7/7 |

Every scoreable field matches ground truth: hero cards, boards, dealer, full
per-street action sequences with amounts, final pots, winners, hero net.
All 10 timeline hands (7 complete + 3 partially observed) export through
`export_yolo_card_hands_for_app.py` into the app session payload.

## Ground truth + eval harness (new)
- `cv_lab/results/ground_truth/v00_hands.json` — hand-level answer key,
  stitched from the human-labeled boxes + template OCR, cross-checked with pot
  arithmetic (every street's pot delta reconciles against stack deltas), plus
  targeted visual verification. Includes full action lists with amounts.
- `cv_lab/scripts/eval_hand_reconstruction.py` — counts discrete errors per
  hand (segmentation, cards, dealer, pot, winner, hero net, per-street
  multiset action alignment with amount tolerances).

## What was broken and fixed
1. **Model 1 was dead.** The v6 mps training run had NaN box_loss from epoch 1
   and its best.pt silently overwrote the previously-working
   `region_spine_v1.pt`. Retrained (`region_spine_v7`, yolo11s, mps with
   `PYTORCH_ENABLE_MPS_FALLBACK=1`): P .997 / R .993 / mAP50 .993 /
   mAP50-95 .943. Promoted back over `cv_lab/models/region_spine_v1.pt`
   (+ kept as `region_spine_v7.pt`).
2. **OCR was never wired into the live path.** `frame_from_models` now runs the
   template OCR (`ocr_readers`) on pot/stack/bet boxes and pills (word first,
   colour fallback); fixture-provided attrs pass through untouched.
3. **SEAT_RING was backwards** — action order is ascending seat index
   (verified against live blind posts). Fixed; positions (BTN/SB/BB/…) are now
   correct.
4. **Seat assignment flapping** — each HUD element renders at its own per-seat
   position; nearest-avatar assignment flapped between adjacent seats.
   Added `SEAT_ANCHORS_BY_CLASS` (k-means over the labeled boxes, v00,
   per-file image dims).
5. **Spine hardening** (`build_yolo_hand_timeline.py`):
   - card/board debounce with reset-on-empty + accept-if-extended (boards only
     grow); hero/board majority voting per hand
   - per-hand revert-only debounce for stacks/bets/pot (A→B→A blips die, a
     call followed by the winner's sweep survives)
   - segmentation: hero change + evidence (board reset / pot drop / dealer
     move / gap) measured against the last state showing the old hero's cards
   - actions: attributed to the PREVIOUS state's street (money closes streets);
     stack-delta amounts with bet_text fallback; fold latch + pills as the only
     hero-fold signal; "fold with no facing bet" = showdown reveal, suppressed;
     pre-observed actions extracted from the hand's first state (standing bets
     + pills); closing checks synthesized (a street that ended with no bet was
     checked through; unraised BB gets its free check) with all-in awareness
   - settlement cut (`_settle_index`): after the pot sweeps to the winner,
     everything is next-deal noise (antes/blinds) — excluded from actions/pot
   - reconciliation: `initial_pot + observed contributions ≈ final_pot`
     (blinds/antes are pre-debited; uncalled over-shove chips measured against
     pre-settlement stacks)
6. **Validator** (`validate_yolo_card_timeline.py`): terminal board sweep is
   not a regression; transient per-state duplicate reads resolved by the
   hand's final voted cards no longer warn.

## Known gaps
- 2 partially-observed hands (recording start/end) + 1 hand under-sampled by
  the labeled-frame gap export as `needs_correction` drafts — by design.
- OCR misreads like "12"→0.12 (chip-icon glyph joining the digit run) still
  occur in bet_text; the spine routes around them (stack deltas + debounce),
  but `read_number` could be tightened.
- Anchors/zones are v00-layout-specific; other table layouts need re-learning
  (same k-means, `SEAT_ANCHORS_BY_CLASS` docstring).

## Repro
```
conda run -n poker-cv python cv_lab/scripts/run_two_model_pipeline.py \
    --start 0 --end 564 --interval 2 --device mps \
    --out cv_lab/results/two_model_timeline_v00.json
conda run -n poker-cv python -m cv_lab.scripts.eval_hand_reconstruction \
    --timeline cv_lab/results/two_model_timeline_v00.json \
    --truth cv_lab/results/ground_truth/v00_hands.json
```
