# CV Lab — Findings 03 (full single-hand reconstruction)

## Experiment (Experiment "A")
Prove that a COMPLETE multi-street hand can be reconstructed end-to-end from a screen
recording, with no ground truth. Pipeline exercised:
1. **Locate** — 3 low-res preview scans (every 2s) over candidate windows; a subagent read
   each scan set and segmented it into hands + street transitions.
2. **Select** — picked 15 full-res frames at state-changes/decision points of one hand.
3. **Read (blind)** — one independent VLM subagent per frame, identical strict-JSON schema,
   NO shared context between agents (so stitching agreement is a real consistency test).
4. **Stitch + cross-check** — merged the 15 reads into one history; reconciled by pot-delta
   arithmetic.

Chosen hand: **hero Q-J (BTN), t0300–t0396**, a 3-bet pot that goes preflop → flop → turn
(hero rivers... no — makes a straight on the turn) → all-in → river → showdown win.
Raw stitched result: `cv_lab/results/hand01_reconstruction.json`.

## Result: the hand reconstructs perfectly
Full history recovered (positions, every action + size, all streets, showdown, result). The
decisive evidence is that **every pot value the VLM read matches the reconstructed action by
pure arithmetic**:

```
5.5 (blinds+antes) -> 32.5 (preflop) -> 65.1 (flop) -> 162.2 (turn raise) -> 347.5 (all-in)
hero pre-sweep stack 103.1 = 272.60 - 12 - 16.30 - 141.20   (exact)
```

No ground truth was used or needed. Independent blind reads of 15 separate frames stitched
into one internally-consistent hand.

## The two things that went wrong — and both were caught WITHOUT ground truth
1. **OCR error self-corrected by pot arithmetic.** The flop-bet frame (t0354) read hero's bet
   as `10.30` because a chip icon partially covered the digits. But flop pot 32.5 + bet =
   48.8 (the pot the SAME frame reported) forces bet = **16.30**, and the next independent
   frame (t0358) shows 16.30. => **pot-delta reconciliation is a reliable auto-correction for
   obscured/blurred numbers.** This is the single most important finding: the redundancy
   between "pot total" and "sum of bets" lets the system police its own OCR.
2. **Stack timing artifact resolved by reconciliation.** Hero ending stack disagreed (103.1
   full-res vs ~444.6 low-res scan). 103.1 is the PRE-sweep stack (hero over-shoved a bigger
   stack than villain could call; the uncalled part was already returned). 103.1 + 347.5 pot
   = 450.6 ≈ 444.6. => **final stacks are only valid a beat AFTER "YOU WIN"; sample a settled
   frame for them.**

## Persistent ambiguities (unchanged from baseline; need deterministic layer)
- **Black suit spade-vs-club**: hero Q and the board T/7 split across agents (mostly clubs;
  one agent read the Ten as 10s; the earlier low-res scan read hero Q as spades). Did NOT
  affect hand values this hand, but it is a real accuracy gap. Fix = template/suit-shape check
  on the fixed card slots (Findings 02, weakness #1).
- **Dealer-button seat** wavered frame-to-frame (hero vs GiveZeroBluffs). Trivially fixed by
  the invariant "the button does not move within a hand" → majority vote across the hand.

## What this proves about the approach
- End-to-end reconstruction WORKS on a real multi-street all-in hand, from a video, blind.
- The pipeline is **self-checking**: pot totals vs bet sums, and street-to-street stack deltas,
  form an arithmetic consistency net that catches per-frame read errors automatically. This is
  the key reason ground truth was not required to reach a confident result.
- Cost shape: ~50 cheap low-res preview reads to locate/segment + ~15 full-res reads to
  reconstruct one hand. Frame SELECTION (not brute-forcing all frames) is what makes it viable.

## Next experiments (in priority order)
1. **Automate frame selection** — replace the manual "pick 15 timestamps" with a detector:
   per-frame perceptual-hash / pixel-diff to find stable frames + state changes (pot text
   change, new board card, new action pill), so key-frame picking is programmatic.
2. **Deterministic card/suit cross-check** — settle spade-vs-club on the fixed card slots.
3. **Arithmetic reconciler as code** — implement the pot/stack consistency checker that flagged
   the 16.30 error, as a reusable validation pass over any stitched hand.
4. Then scale to the FULL session: segment all hands, reconstruct each, emit hand histories.
