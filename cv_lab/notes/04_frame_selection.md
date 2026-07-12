# CV Lab — Findings 04 (automatic key-frame selection)

## Goal
Replace the manual "pick N timestamps to read" step (Experiment A) with an automatic
selector that finds the settled decision/state-change frames on its own. Validate it against
the Experiment-A hand, which serves as a known-good ANSWER KEY (its 12 real events are known).
Script: `cv_lab/scripts/select_keyframes.py`. No ground truth used.

## v1 (whole-frame motion diff) — FAILED, and the failure is instructive
Idea: a table is static except during animations, so diff consecutive downscaled gray frames;
stable stretches = states, motion bursts = transitions.

Result: **inadequate.** Poker state changes vary enormously in pixel footprint:
- 3-card flop deal: whole-frame diff ~6.8 (big)
- single turn card / a call chip / an action pill / a pot-digit change: ~1-2 (tiny)
- per-second timer-ring countdown tick: ~1-2 (noise)

So the turn card, most bets/calls, and the showdown sink BELOW the timer-tick noise floor. A
single global threshold cannot separate "turn card dealt" (2.05) from "timer ticked" (~1.5).
Measured: of 12 real events, whole-frame diff cleanly caught only the big ones (flop deal,
hero's raise). Verdict: whole-frame diff is only good for BIG events (deal / pot-push / screen
change) — useful as a hand-boundary signal, useless for per-action selection.

## v2 (region-aware differencing) — WORKS
Diff SEMANTIC REGIONS instead of the whole frame:
- **POT-number region** -> changes on every bet/call/raise (the pot updates)
- **BOARD region** -> changes on every street (flop/turn/river)
Fire a keyframe when ANY watched region changes; the representative is the settled frame right
AFTER the change burst. Keep whole-frame diff only as a hand-boundary hint.

Why it works: these regions are DEAD QUIET when idle (pot diff p50=0.00, p90=0.11) and spike
5-70 on real events -> clean signal/noise separation. Thresholds: pot 1.5, board 3.0 (both
>10x the noise floor).

### A second, subtle bug found + fixed: region-based DEDUP
The first v2 pass merged near-identical representatives using a WHOLE-FRAME thumbnail compare —
which reintroduced the v1 blindness: it wrongly merged the flop-deal state, the flop-bet state,
and the flop-call state into one (each pot change is tiny across the whole frame). Fix: dedup on
the REGION signatures (pot+board), not the whole thumbnail. After the fix all three flop states
are correctly kept distinct.

## Validation against the Experiment-A answer key
Window t300-396, 3 fps -> 289 sampled frames.

- **12 / 12 required events recovered** (every preflop raise/call, the 3-bet, flop deal, flop
  bet, flop call, turn, turn bet, turn raise, all-in, showdown, result) — each within <=2.3s.
- **289 sampled frames -> 18 keyframes (94% reduction).**
- VLM audit of the 18 chosen frames: [see results/frameselect_validation.json].

The one initially-missed event (the min open-raise, pot 5.5->8.5, diff 1.72) was captured after
lowering the pot threshold from 3.0 to 1.5 — still ~14x the noise floor. Its content was already
preserved in the next keyframe even before that, so it was never truly lost.

### VLM audit of the 18 chosen frames — defects found (honest)
Core is good: the 18 form ONE coherent hand with a monotonic settled-pot sequence
(8.5 -> 11.5 -> 23.5 -> 32.5 -> 48.8 -> 65.1 -> 86.6 -> 162.2 -> 347.5), flop/turn betting
captured cleanly. But ~5-7 of the 18 are low-value:
1. **UI-overlay false positives** (t325, t339, t346.67): the player accidentally opened
   "Profile" popups mid-hand; the popup covers the board/pot ROIs, so region-diff fires and the
   rep lands on the popup. **This also explains the earlier "board fires during preflop" mystery
   (333/339/346) — popups, not chip-slides.** Fix: overlay detection / require parseable POT text.
2. **Mid-animation frames** (t386.33, t388.67): all-in coin-splash; transient pot readouts (385,
   450.60) are in-flux, not the true 347.5. The "first quiet sample after burst" heuristic can
   land inside a long animation. Fix: require ~1s of consecutive quiet before declaring settled.
3. **Duplicates** (326~310, 340~334) and a **coverage gap** (preflop 3-bet resolution not shown
   as a clean frame — folds don't move the pot). Fix: add a per-seat action-pill band ROI (pills
   change on fold/call even when the pot doesn't) + tighter dedup.

Cross-experiment note: the transient all-in pot values (385, 450.60) are exactly the per-frame
garbage that Experiment-A's arithmetic reconciliation catches — reinforcing that cross-check.
Full record: `cv_lab/results/frameselect_validation.json`.

## Hand-boundary detection (bonus)
On a two-hand window (t388-440), the whole-frame signal flagged a boundary at t399.33 — exactly
the new-hand deal animation — with a clean keyframe gap (391.67 -> 400.67) separating the hands.
=> whole-frame diff, useless for per-action selection, is a good HAND SEGMENTER. The two signals
are complementary: region diff picks intra-hand states, whole-frame spikes mark hand boundaries.

## Limitations / generalization TODO
- ROIs are normalized fractions calibrated to THIS recording's geometry (table fills the frame).
  Because edges vary between recordings (user's constraint), these must later be re-anchored to
  on-screen LANDMARKS (locate the "POT"/"BLINDS" text, derive ROIs relative to them) instead of
  absolute fractions. This is the top generalization task.
- Mild over-selection remains (a fast multi-sample animation can yield 2 adjacent keyframes).
  Harmless (over-selection just costs a spare VLM read) but could be trimmed with a min-gap rule.
- Board region also fires on preflop chip-slides passing through it (~20). Harmless for
  selection (pot fires too); the VLM read determines the real board content downstream.

## Where this leaves the pipeline
Locate/segment (whole-frame boundaries) -> select key states (region diff) -> VLM read (Exp A) ->
stitch + arithmetic reconcile (Exp A). The two remaining gaps before a full-session run:
1. **Landmark-anchored ROIs** (edge-tolerance) — top priority.
2. **Screen classifier** (table vs lobby vs transition) so non-table frames are skipped entirely.
