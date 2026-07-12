# CV Lab — Findings 08 (full deterministic pipeline, VLM fully removed)

Completes the deterministic READ step (Findings 06/07) and wires the whole
pipeline end-to-end. **No vision model runs at any point in the pipeline.** The
VLM was used only offline, to label glyph templates and as the hand-01 answer
key — never at runtime.

## Pipeline (all deterministic)
`run_session.py`: PyAV decode (sequential, sample every `--stride` frames, ~2fps)
-> `classify_screen` (skip lobby/transition) -> `landmark_anchor` (green-coin
constellation similarity transform) -> `read_table` unified READ -> hand
segmentation -> street stitch + arithmetic reconciliation.

### READ extractors (all new/validated this round)
- `read_pot.py` — pot number (Findings 07). Digit bank completed with '9' from stacks.
- `read_cards.py` — board cards; now tight-crops each glyph to its bbox before
  template match (consistent framing for board AND hero index corners).
- `read_seats.py` — per-seat STACKS (anchored to each seat coin, reuse pot digit
  bank): **8/8 exact** on t0360. Bets best-effort (chip clusters on the felt
  ring); betting is otherwise recovered from street-to-street stack deltas.
  0.0 reads treated as all-in/animation misreads (a 0 stack is never displayed).
- `read_pills.py` — action pill colour {orange=raise, green=call/bet,
  neutral=check/fold/BB}, fraction-thresholded so it's scale-invariant.
- `read_markers.py` — active seat via the blue timer RING (hollow-circle test
  rejects filled blue card-backs) and dealer via the solid white 'D' disc
  (size+fill test rejects text). Both correct on t0328 (active=seat5, dealer=seat7).
- `read_hero.py` — hero hole cards: fixed fan geometry, per-card deskew, then the
  shared card templates on the index corner. Reads Qc/Jh correctly.
- `classify_screen.py` — table vs lobby/transition off the anchor fit residual
  (table resid ~0.001-0.002; lobby ~0.11, no pot coin). Clean separation.

### Card template bank (complete)
`harvest_cards.py` sampled the session at 1fps, deduped 72 distinct card faces;
labelled offline by eye (`build_bank_from_labels.py`) -> all 13 ranks + 4 suits.
Self-consistency re-reading the labelled faces: **59/60**.

## Hand segmentation + reconstruction
- DEAL boundary = board falling-edge (>=3 cards -> 0 after a flop) with a
  pot-reset-to-blinds fallback for fold-around hands. Median-smoothed board
  counts + denoised pot stream reject single-frame animation/transition spikes.
- Per hand: consensus hero cards + board, monotone pot sequence up to the settled
  final pot (mode of the last frames -> rejects award-animation over-counts),
  winner = seat that recovers the most stack, and a soft reconciliation check
  (sum of contributions ~ final pot).

## Validation on the answer-key hand (Qc Jh, t299-399)
Fully deterministic reconstruction matched the VLM answer key exactly:
- hero **Jh Qc**, board **8d Tc 7c 9h Kd**
- pot sequence **5.5 -> 8.5 -> 11.5 -> 23.5 -> 32.5 -> 48.8 -> 65.1 -> 86.6 ->
  162.2 -> 347.5** (the key's 5.5->32.5->65.1->162.2->347.5 street ends are all
  present)
- winner **seat7 (hero) +341.5 ~ final pot 347.5**, reconciled, complete=True.

## Full-session result (2fps, stride 30)
`results/session_full.json`. **1130 samples, 1085 classified as table** (lobby
tail correctly skipped), segmented into **10 hands, 5 fully reconstructed**
(self-consistent: hero cards + board + strictly-monotonic pot progression +
identified winner + soft stack reconciliation + NO duplicate cards).

Complete hands (winner seat index in the coin constellation):
- t142-217 hero **Kh Ks** (preflop win, pot 15.0, seat7)
- t218-298 hero **9h Qh**, board 6h 2c Ks Ts Tc, pot->96.3, seat4
- t299-399 hero **Jh Qc**, board 8d Tc 7c 9h Kd, pot->347.5, seat7 (== answer key)
- t399-488 hero **Qd Qs**, board Td 3s 9c 6c Jc, pot->78.2, seat7
- t488-548 hero **4h Kh**, board 9d 9h Jc, pot->108.0, seat5

The 5 not marked complete, with the reason each was correctly rejected:
- t0-22 & t549-556: partial hands at the recording edges (truncated).
- t22-40: hero 7h 9d folded preflop (no showdown / no winner recovered).
- t40-78: hero not dealt in (hero=None) -> no hole cards by definition.
- t78-142: **duplicate-card self-check fired** -- hero read 8c/Tc while the board
  had Tc (a hero rank misread). The consistency gate caught the error rather
  than emitting a wrong hand. This is the arithmetic/consistency net working.

## Honesty notes / limitations
- Bets (chip clusters) are best-effort (~40%); betting is reconstructed from
  street-to-street STACK deltas instead, which reconcile against the pot.
- Preflop-only / folded hands segment less cleanly than postflop hands (the
  board falling-edge is the strong boundary; the pot-reset fallback is weaker
  when pots are tiny). A couple of short folded hands may merge.
- Card reads are limited to the template bank's coverage; the duplicate-card
  gate flags the rare misread instead of trusting it.
- **No VLM anywhere in the pipeline** -- confirmed end to end.
