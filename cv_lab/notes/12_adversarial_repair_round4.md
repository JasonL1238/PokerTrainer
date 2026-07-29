# Adversarial repair, round 4

Chronological record. Later findings supersede earlier experiments.

All measurement below is on the five DEVELOPMENT recordings only. No locked-test or
validation recording was opened.

## Measured throughput

Detector frames are frozen per run, so the two fixture sets are not comparable to
each other — only before/after within one set is.

| fixture set | before this round | after this round |
|---|---|---|
| fresh detector frames (1s, all 5 recordings) | 11 / 21 | 11 / 21 |
| the older `geom_test2` frames | 9 / 21 (adversary C) | 11 / 21 |

Throughput is unchanged on fresh frames. What changed is what the exported hands
say about themselves: the one confirmed wrong board in the corpus now ships at
confidence 0.65 with `LOW_CONFIDENCE` instead of 1.0 with no tag, and the hand
whose action ledger named a seat it did not list now imports instead of rolling
back the whole session.

## What was repaired

### Card zoning — `region_detections`

`_community_row` tested only the LARGEST same-reference-y card set and returned []
when that set failed the board-shape test. On the real g0723a t=265 showdown frame
a four-card villain-reveal strip outnumbered the three-card flop, failed the span
test, and the flop was then never examined — both board nets reported False with
the correct 8-point anchor in place. Replaced by `_community_rows`, which
enumerates every candidate row and splits each same-y group into horizontally
contiguous runs (`_BOARD_ROW_MAX_ADJACENT_GAP`, derived from the measured 1.71-2.32
card-width rhythm of 562 real board rows against the 26-card-width separation of
the two reveal pairs).

`_BOARD_ROW_RY_TOL` was 0.02 against a board band 0.061 tall, so a card leaving the
band vertically was 0.030-0.042 from its row-mates, dropped from the row, and the
partial net's `0 < zoned < len(row)` was False by construction on 557 of 562 real
rows. Re-derived from two independent measurements: the tolerance REQUIRED so a
card exiting either band edge still groups with every row-mate is at most 0.0423,
and the CLOSEST any non-board card comes to a board row's edge is 0.0621. 0.05 sits
inside that window and is 3.9x the largest observed within-row scatter (0.0129).

The `>= 3 states` gate on `board_zone_yield_zero` / `board_zone_yield_partial`
exceeded the evidence a real street supplies — exported hands hold their final
board for as few as 1 distinct state after state collapse — and
`board_zone_yield_zero` additionally required `not board`, which silenced it
entirely on a hand whose flop was captured and whose later rows were all missed.
Both nets now fire on one state. Cost: 0, because neither flag is raised on any of
the 1309 real card-bearing frames, deal animations included.

`ANCHOR_MIN_TRUSTED_POINTS = 5`. The residual is a SHAPE test and shape needs
redundancy; the similarity has three degrees of freedom, so a three-point fit has
none. Enumerating every stack_text subset of every card-bearing g0723a frame and
re-zoning through each gate-passing fit:

| landmarks | subsets | pass the gate | silently mis-zone |
|---|---|---|---|
| 3 | 3206 | 1773 | 15 |
| 4 | 3990 | 2883 | 15 |
| 5 | 3178 | 3008 | 0 |
| 6+ | 2088 | 2088 | 0 |

The worst three-point case zones hero's own hole cards as the BOARD and the real
flop as "other". Requiring five costs nothing measured — the fewest stack_text
boxes on any card-bearing frame across the five recordings is 6. Whole-recording
silent-wrong subset anchors: 329 before, 1 after, and that one produces a 1-card
board which `invalid_board_count` catches downstream.

`REF_BOARD_RX` was (0.295, 0.640) — margins of -0.055 / +0.036 against the ±0.030
its own comment states, with 0.640 a round number rather than a derivation. Board
rx re-measured over 2073 anchored detections confirms the extremes the comment
cites (0.3497..0.6043), so the constant is now what the stated rule produces:
(0.3197, 0.6343). Both edges are pinned; neither had a confuser pin before.

### OCR — `ocr_readers`

The `dot_unreconciled` fail-closed guard was unreachable in exactly the case it was
written for: the decimal-GAP arm ran first and always found a split. On the real
"18.30 BB" crop at 1.10x of the 1272x896 client the "BB" suffix is absorbed as
"88", the true dot's split leaves four fractional places and is rejected on arity,
and the gap arm then split at the WORD SPACE before the suffix and returned a
confident 1830.88. The guard now precedes the gap arm, and fires only for a
separator located BETWEEN two digits in a gap at least `_INTRA_NUMERAL_GAP` wide —
the interior test is what keeps it from swallowing the frozen "343.60" crop at
0.90x, where the real dot is lost and a 2px speck lands in a 2px gap.

`_MIN_CALIBRATED_RUN_H` 9 -> 12. The old value's justification ("every read stays
correct or fails closed down to run height 9, and EVERY confident wrong value in
the whole sweep sits at 7 or 8") was false: over 14390 real crops x 17 render
scales, 893 reads return a confident value disagreeing with the audited 1.0x
reference and 757 of them sit at run heights 9-11, inside the declared calibrated
range. 12 is the smallest run height any value-producing read has at native size on
the smallest supported client. Raising it fails closed on 0 of 14193 native reads
and removes 757 of the 893 wrong values — including all 77 false zeros in the
sweep, which sit at run heights 7-10.

`_AFFIX_MAX_REL_H = 0.78`'s stated basis ("the 'BB' suffix renders at 0.59-0.77 of
the digits' height") is false on one of the five recordings: on g0715 the suffix
renders at the FULL band height, 385 of 435 'B' glyphs at or above 0.78, median
1.000. The constant is unchanged — above ~1.0 it starts absorbing real digits — but
the comment now states honestly that height alone does not separate the suffix on
every supported client, and that the separation there rests on the 0.28 adjacency
gap alone. The 100x consequence is closed by the guard-ordering fix above.

### Spine — `build_yolo_hand_timeline`

`_reject_stack_outliers` exempted every read whose `attr_source` was not `"none"`,
which handed total immunity to `"gap"` — a decimal INFERRED from spacing, not seen.
Injecting a 1985.088 BB read into a 200 BB game with that source exported at
confidence 1.0 with warnings=none. Only a LOCATED separator exempts a read now;
`None` (no quality channel at all) stays inert as before.

`starting_stack` was the first SURVIVING stack reading, not the first reading. A
seat whose stack failed to read early in the hand is dropped from those states
entirely, so the field published a mid-hand stack as fact — measured, BTN 123.4 ->
115.4 and SB 218.0 -> 210.5, with warnings=none. It is now the reading on the
hand's first observed state or unknown. Cost: 3 of 148 player rows.

The player set was card_back-only, so a seat that folded BEFORE the hand came into
view — no card_back, only a FOLD pill — was absent from `players` while
`_reconstruct_actions` booked its fold. The export then emitted an action whose
`player_key` had no HandPlayer row, and `import_session` rolled the entire payload
back: 0 sessions, 0 hands, three good hands destroyed by one malformed one. A
persistent pill is now participation evidence at the same two-state bar as a
card_back.

`_flag_identity_splits` records a SAME-LENGTH card-list disagreement before
`_debounce_cards` absorbs it. Measured, the debounce's majority is not always
right: the g0723a turn card reads 8d eight times and 8h twice, the card on screen
is the eight of hearts, and 'Js 6h 4c 8d' exported at confidence 1.0 with tags [].
4 of 21 development hands carry a contest and 1 of the 4 is confirmed wrong on the
pixels, so the codes are review flags at severity 0.35 rather than rejections —
discarding three good hands to hold one bad one is the wrong trade at that rate.

### Export — `export_yolo_card_hands_for_app`

`_assert_actions_reference_players` refuses a hand whose ledger names a seat it does
not list, so an inconsistent hand is skipped alone rather than aborting the session
import.

`stack_conflicts`, `stack_outlier_checks_skipped` and `pot_text_off_column` were
recorded, serialized as `cv_*` evidence fields, and read by nothing. All three are
evidence gaps of the kind `amounts_unknown` already caps for, and now cap the score
at `LOW_CONFIDENCE_AT`.

## Known limits, not repaired

- **Card classifier suit accuracy.** 1 wrong card in 33 audited exported board
  cards (3.0%). The identity-split net flags the contested case, but a card the
  classifier reads WRONG unanimously is invisible to it. Closing this needs
  retraining, not code.
- **A leading decimal on the run's baseline is invisible to the dot search.**
  `dot_centers` filters `x0 < centre < x1`, so a decimal preceding the first
  accepted digit is discarded. When the integer part is hidden by an OPAQUE sprite
  rather than clipped, no remnant glyph exists for `_run_is_truncated` to fire on.
  1 occurrence in 14390 native crops (g0711 t=257 pot_text, ".50 BB" -> 50.0); it
  falls between two hands and reaches no export.
- **A seat-timer badge can outscore the real stack in the same crop.** g0621 t=177:
  "212.90 BB" plus a timer ring containing "12" reads a confident 12.0, because the
  5-digit run is judged a fragment and loses the completeness key to the 2-digit
  badge. Does not reach an export on this corpus.
- **The two-digit `"1" not in digits` exclusion** leaves a 10x hole at every gap
  width for any two-digit numeral containing a '1' on a one-decimal client. Its
  stated measured cost of 0 reads is a statement about the corpus, not about the
  exclusion's safety.
- **`clubwpt_session_01.mov`, a locked-test recording, is in
  `cv_lab/datasets/yolo_cards_autolabel_v1/manifest.csv`** (228 rows, both train and
  val). Pre-existing and untouched by this phase — `git diff HEAD -- cv_lab/datasets/`
  is empty — but the region detector's reported generalization has already seen a
  locked recording.
