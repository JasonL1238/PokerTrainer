# 10 — Adversarial repair, round 1

Chronological record. Later findings supersede earlier experiments.

Three adversaries attacked the post-geometry-fix pipeline and produced 27 findings.
Every one was reproduced independently before any code changed. This note records
what the measurements showed, what was fixed, and — importantly — the findings whose
proposed remedy the measurements **disproved**, plus the gaps that are still open.

## Corpus discipline

The measurements below are from the 5 development recordings only:

| tag | geometry | AR |
|---|---|---|
| g0621 | 2062x1178 | 1.750 |
| g0711 | 2722x1832 | 1.486 |
| g0715 | 2132x1378 | 1.547 |
| g0723a | 2054x1470 | 1.397 (reference basis) |
| g0723b | 1272x896 | 1.420 (smallest) |

**A corpus-split violation was found and removed.** `landmark_anchor.REF_POT_RX` and
`REF_POT_SIDE_RY_MIN` were justified in-code as "measured across 9 geometries (the 5
development recordings plus the 8 archived fixture sets)", and the load-bearing
rationale for the rx guard was an off-column HUD element on a 2796x1914 archive. By
geometry, 7 of the 8 archived fixture sets in `cv_lab/results/` belong to locked-test
or validation recordings; only `frames_v05` (2722x1832 = 07-11) is development.
2796x1914 exists **only** in the held-out split. The measurement was also transcribed
verbatim into a permanent test.

Both constants have been re-measured on development material (1267 anchored
`pot_text` detections: every genuine pot at rx 0.4916-0.5076, main ry 0.3410-0.3810
n=1258, side ry 0.5419-0.5460 n=9 on one recording), the held-out justification is
gone, and the test now states the rule structurally with synthetic coordinates.

`cv_lab/results/frames_v0*.json` are **not** measurement material. Anything derived
from them, other than v05, is derived from held-out recordings.

A separate process point: `scratchpad/probe.json` from the adversarial round records
sha256/geometry/fps/duration for all 5 locked-test and all 3 validation recordings.
That is container metadata rather than frame content, so the damage is bounded, but it
is what made the geometry→recording mapping above possible and it should not have been
collected. No script, test or fixture in the repo decodes those files.

## What was wrong, and where it is now caught

### Reader (`ocr_readers.py`)

* **Run completeness.** `classify_digit` deliberately pools the affix glyphs (B/P/T,
  chip) with the digits so a real affix *breaks* the digit run. When a digit's best
  match is a letter the run truncates and the **fragment is returned as a confident
  value**. At 12px glyph height the `8` of "218 BB" scores 0.677 as `B` against 0.826
  as `8`: 16 consecutive samples of one seat read 21.0, and 21.0 shipped as a player's
  `starting_stack` at confidence 1.0. The same mechanism turns a clipped detector box
  ("71.20 BB" cropped to "20 BB") into a confident 0.0, indistinguishable from a
  genuine all-in.

  `_run_is_truncated` now rejects a run with a value-height glyph inside the numeral's
  own letter-spacing (`< 0.28` of digit height — every measured intra-numeral gap,
  including the decimal's own, is `<= 0.250`, while "POT:" and "BB" are separated by
  0.55 and up). Candidate selection prefers a *complete* run, which also fixes
  "POT: 9 BB" reading 0.0 (the `O` of POT and the `9` were both one-glyph runs and the
  tie broke on score).

  Corpus effect, 14212 real crops: **40 reads change (0.28%)**. 7 become correct
  (0.0 → 9.0 pot reads). 33 become unknown, of which the measured cost is 15 reads
  whose value was numerically right by luck (a trailing `0` that the chip template
  out-scored). **0 reads change to a different wrong value.**

* **Comma thousands separator.** A group separator sits where a decimal point sits and
  has the same silhouette: "12,345" read 12.345. ClubWPT renders 0, 1 or 2 fractional
  places and never 3 (measured over all 14390 reads), so a split leaving 3 or more is
  refused. Latent on this corpus; live the moment a chip-denominated or tournament
  layout is ingested.

* **Dead code.** `tokenize()` had exactly one occurrence in the repo — its own
  definition — while `read_number_detail`'s docstring described the gap tokenization it
  performs. Removed, docstring corrected.

### Zoning (`region_detections.py`, `landmark_anchor.py`)

* **Partial board-row yield was invisible.** `_board_row_missed` fired only when
  *every* card of a detected community row missed the zone. All-or-nothing was never
  the failure mode: a row that straddles a band edge yields most of its cards, and 3,
  4 and 5 are all legal board counts. A 5-card river board whose last card sat 0.0001
  of reference-y below the edge exported as a completed 4-card **turn** board at
  confidence 1.0, asserting a showdown result on a board missing its river card.
  `_board_row_partial` now covers it.

* **The row-shape test was scale-dependent.** `_BOARD_ROW_RX_SPAN` was a window on the
  *reference* x-span, i.e. the frame divided by the fitted anchor scale — so a wrong
  scale inflated the span straight out of the window and **switched the net off at
  exactly the moment the anchor was worst**. Measured: a real 5-card board spans 0.2447
  of reference-x, so the net disabled itself below 0.816x true scale; at 0.80x an
  entire recording's boards were lost with the net silent on all 39 frames and a
  mathematically perfect residual. The span is now measured in units of the row's own
  median card width — a ratio of two quantities carrying the same scale factor cannot
  be moved by it. Measured on 561 community rows: 3-card 3.45-4.58, 4-card 5.79-6.95,
  5-card 7.49-8.88 card widths; the only non-board same-row strip spans 26.99-27.21.

* **`ANCHOR_MAX_RESID` cannot bound zoning error** and the comment above it claimed it
  could. Any translation or uniform scaling of the landmark constellation *is* a
  similarity, so the fit absorbs it exactly: sweeping a rigid dy offset from -0.040 to
  +0.040 of frame height leaves the residual identical to 13 significant figures while
  every card's reference-y moves by the full offset. Card zoning tolerates about
  ±0.025 of rigid offset and the residual contributes nothing to that budget. The
  comment now says so and a test pins it.

* **The board band's calibration comment was stale.** It recorded board ry as observed
  0.4373..0.4513; the real development maximum is **0.4583** (g0711 t=329, 9d). The
  0.466 ceiling therefore left 0.0077 of margin — a third of the rigid offset the
  zones tolerate. Applying the documented confuser-midpoint rule to the corrected
  measurement (nearest confuser above: a stray reveal at 0.4811, g0723a t=143) gives
  **0.470**, symmetric at 0.0114 either side. The margin is thin by construction,
  which is why the partial-yield net now sits under it.

### Reconstruction (`build_yolo_hand_timeline.py`)

* **No stack-ledger invariant existed.** Two exported hands contained a player whose
  stack rose mid-hand, both at confidence 1.0 with `warnings=none`. The invariant is
  *not* "a stack must never rise" — a rise is legal, the client returns the uncalled
  part of an over-shove before the sweep (measured twice on the baseline: 0.0 → 18.9
  and 0.0 → 128.4). The exact, threshold-free form is **a player never holds more than
  it started the hand with, before settlement**. Folded seats are out of the ledger
  because the client tops them back up to the buy-in without waiting for the hand to
  end (baseline seat 6: folds on the turn, 164.0 → 200.0 while the river is played).
  Fires on exactly 1 of 21 pre-fix hands (the 21.0 truncation) and 0 of 21 post-fix.

* **`bet_text == 0` is now unread**, mirroring the pot rule. The client renders no bet
  row at all when a seat has nothing in front of it, and a crop holding only chip
  sprites reads a confident 0.0 (a chip's white annulus matches the `0` template).
  `stack_text == 0` stays a value — an all-in seat genuinely shows "0 BB".

### Gate (`validate_yolo_card_timeline.py`, `export_yolo_card_hands_for_app.py`)

* **No action-sequence legality check on the reachable path.** `_betting_round_reopened`
  was gated behind `board_is_empty`, so it was dead code on every hand with a board.
  **5 of 15 exported hands contradicted themselves**, all at confidence 1.0 with tags
  [], three of them at `completion_status: complete`. `_action_sequence_illegal` now
  runs on every hand, with three threshold-free rules: acting after all-in, folding
  immediately after one's own call with no intervening aggression, and checking into
  an unmatched bet. Verified on the corpus: 4 hands flagged, each with a named
  violation; a limped pot and a legal fold-to-a-reraise are negative controls.

* **Two `SPINE_FATAL_CODES` were unprotected.** `amount_scale_implausible` and
  `anchor_unavailable` could be deleted from the frozenset with all 794 tests green.
  Now pinned per code, named literally — a loop over the frozenset stops testing a
  code the moment someone deletes it.

* **Unrecognised spine codes failed OPEN** at the export gate while the same record
  listed them under `rejection_codes`. `RECOGNISED_SPINE_CODES` closes it.

* **Confidence overstated truncated and evidence-poor hands.** `partial_start`,
  `partial_end`, an unread terminal event, and `amounts_unknown` were all recorded and
  then read by nothing: a hand that opened mid-flop with 18.1 BB already in the pot,
  and a hand with 18 failed numeric reads, both scored 1.0 with tags []. All four now
  cap the score at 0.80. The `amounts_unknown` threshold is a *rate* (0.5 unread reads
  per state); measured per hand, 18 of 21 sit at 0.06-0.29 and the other three at 0.57,
  0.86 and 1.23.

## Findings whose proposed remedy the measurement disproved

* **"`_reject_stack_outliers` should reject in either direction."** It should not. A
  short stack is an ordinary poker fact: the development corpus contains genuine ones
  at **10.53x** below the sibling median (18.90 BB against 199.0) and **5.93x** (31.20
  against 185.0), both legible on screen. A symmetric 6.0x net discards both. The
  downward failure mode — a truncated digit run — is caught where the evidence for it
  exists: at the reader, and at the stack ledger. Pinned by
  `test_a_genuine_short_stack_is_not_rejected_as_an_outlier`.

## Still open

* **Side-pot refusal depends on one low-confidence detection.** `side_pot_unsupported`
  fires only when the detector finds a second `pot_text` box; on the one development
  recording that has a side pot those boxes score 0.382-0.758. Suppress them and the
  unrepresentable hand exports clean. The obvious corroborating evidence (two seats
  all-in with a third live) is **not usable on this corpus** — the side-pot recording
  yields one all-in action, while two baseline hands show two all-in seats each and
  have no side pot. Closing this needs a recording with both a measured side pot and a
  measured all-in ledger.

* **The reader is unreliable below ~17px glyph height, and nothing measures render
  size.** A synthetic sweep over real ClubWPT glyphs puts 5.0% of 3-digit values wrong
  at 12px and 0/258 at 19px and above. Measured real glyph heights: g0723a p50 22,
  g0621 22, g0711 28, g0715 21 — and **g0723b p50 12**, i.e. one supported development
  geometry sits entirely inside the unreliable band. The run-completeness net catches
  the specific truncations measured there, but it is a net, not a floor: the pipeline
  still has no notion of a minimum supported render size.

* **A confident 0.0 from a crop with no text is not separable at the glyph level.** The
  `AmountRead` docstring promises `None` and `0.0` are distinct channels. For
  `bet_text` this is now handled downstream; for `stack_text` the clipped-box case is
  caught by run completeness, but a crop of pure chip sprites still reads 0.0 with
  nothing but `score` to distinguish it, and `attr_score` has no consumer.

* **An unknown stack is dropped, not carried.** `stacks` omits the seat entirely, so 0
  of 102 exported players carry `starting_stack=None` — downstream then picks a value
  up from a different state. Unknown is neither zero nor unknown; it is invisible.

* **`build_yolo_hand_timeline.py` cannot be run as a script from the repo root**
  (`ModuleNotFoundError: No module named 'cv_lab'`) — it lacks the `sys.path` insert
  its sibling `export_yolo_card_hands_for_app.py` has. Pre-existing; only affects the
  documented CLI, not the importable path.

## Export effect, measured end to end

Re-running the whole pipeline over the 5 development recordings (video → detector →
classifier → OCR → spine → validator → exporter):

* boards, heroes, pots and `hero_bb_won` are **identical** on all 21 hands;
* hand segmentation is unchanged (7/3/6/1/4);
* exported hands **15 → 11**. The 4 removed are exactly the internally contradictory
  ones, each with a named violation:
  * g0723a hand 1 — preflop, seat 7 calls 2.0 then folds, nothing raised between
  * g0723a hand 2 — river, seat 3 goes all-in for 148.1 then folds
  * g0621 hand 1 — river, seat 5 calls 75.0 then folds
  * g0621 hand 3 — preflop, seat 3 checks facing a 6.0 raise it never matched
* the phantom all-in on g0711 hand 2 is **gone at the source**: the clipped stack read
  is now unknown, so no all-in is invented and the hand reconstructs correctly.
