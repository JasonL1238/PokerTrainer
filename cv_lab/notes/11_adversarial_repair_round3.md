# Adversarial repair, round 3

Later findings supersede earlier ones. This note records what round 3 changed, what
it measured, and — first, because the previous round's summary was wrong about it —
how the measurement itself was produced.

## 0. Provenance of the numbers below (the round-2 summary was stale)

The round-2 hand-off reported "OCR implausible 2.64% -> 0.00%, exports 14/21 ->
15/21, no regression on any of the four strong parts". That table was produced by
artifacts written roughly two hours **before** the code it described. Re-reading all
14,390 stored numeric attributes with the reader as it actually stood found 49
disagreements (0.34%), and re-running one whole recording end to end changed all
three of its exported hands. The headline "+1 exported hand" was not a measurement
of the shipped code; the repo's own round-1 note said 11, and 11 is what the code
produced.

Every number in this note comes from re-decoding the five development recordings
with PyAV and re-reading all 11,996 amount crops at the detector's own boxes with
the code as committed here. The re-read is stored under the session scratchpad; the
frozen inputs the test suite replays are in `tests/fixtures/`.

One honest gap: on the 06-21 recording the decode reached 2,078 of its amount
detections before the container ran out of frames, so 171 of its sampled timestamps
kept their stored reads. Its export outcome (0 of 3 hands) is identical under the
stale reads, the crop-based re-read and the video re-read, so the gap does not move
any conclusion here — but it is a gap.

## 1. Measurement, current code

| recording | layout_profile | amount reads | differ from round-2 artifacts | timeline hands | exported | exported confidences |
|---|---|---|---|---|---|---|
| 07-23 3.21.47 PM | 2054x1470 | 4004 | 9 | 7 | 4 | 1.0, 1.0, 1.0, 1.0 |
| 06-21 12.41.17 AM | 2062x1178 | 2078 | 2 | 3 | 0 | — |
| 07-11 12.45.27 PM | 2722x1832 | 3539 | 11 | 6 | 4 | 1.0, 1.0, 0.8*, 0.55* |
| 07-15 10.16.07 PM | 2132x1378 | 214 | 1 | 1 | 0 | — |
| 07-23 3.33.54 PM | 1272x896 | 2161 | 20 | 4 | 3 | 0.8*, 1.0, 1.0 |
| **total** | | **11996** | **43** | **21** | **11** | |

`*` carries the `LOW_CONFIDENCE` tag.

Throughput is 11 of 21, the same figure the round-1 note recorded and four below the
round-2 summary's claim. Hands are held back by named, checkable violations:
`action_sequence_illegal` (4), `board_regression` + `street_order_issue` (5),
`side_pot_unsupported` (1). None is a stub; each is a hand whose reconstruction
contradicts itself.

The confidence score now separates on the path a user actually sees. It did not
before: the export gate skips any hand carrying a validation code, so the severity
deductions could never reach a shipped hand, and the two evidence caps that could —
an unobserved boundary and a high unreadable-amount rate — both clamp to exactly
0.80 while the `LOW_CONFIDENCE` tag tested `< 0.8`. Every shipped hand therefore
read 1.0 with no tag. The cap and the tag share one constant now
(`export_yolo_card_hands_for_app.LOW_CONFIDENCE_AT`) and the comparison is
inclusive.

## 2. The reader

Six defects, one root cause between three of them: the run-completeness net measured
the gap from a rejected glyph to the digit run **through** the decimal point.

- A numeral whose integer part is clipped away renders as `.60 BB`. The dot search
  requires `x0 < dot < x1` where `x0` is the truncated run's own left edge, so the
  rendered decimal is invisible; the completeness net did not fire either, because
  the dot's width pushed the measured gap to 0.43 of run height, above the 0.28
  word-break floor. Confident `60.0` on the baseline geometry for three consecutive
  samples of a seat holding 180.6.
- At 0.90x the `9` of `19.50` scores 0.541, under the 0.55 floor; the same
  dot-inflated gap declared the surviving `50` a complete numeral. Confident `50.0`,
  non-monotonic in scale (0.85x and 0.95x both read 19.5).
- `_bridged_gap` now subtracts the numeral's own baseline dots before measuring, so
  both cases sit at 0.08-0.17 of run height and fail closed. Bridging must use
  baseline dots only: a 1px speck on the frozen `218 BB` crop's top edge otherwise
  bridges the word space to the `BB` caps and truncates a complete numeral.

The net also erred the other way. `POT: 212.50 BB`'s trailing fractional `0` scores
0.861 as the chip affix against 0.846 as `0` and sits 2px from the run, so the whole
read was discarded — for the eight samples covering an exported hand's settlement,
and for 42 previously-correct reads corpus-wide. A glyph inside the numeral's own
letter spacing **is** part of the numeral, which is the net's own premise; so the
affix label cannot be right, and the reader now takes the glyph's best digit reading
when it clears the confidence floor and fails closed only when it does not. Same
rule recovers `218 BB`, which used to fail closed at best and read `21.0` at worst.

Two guards were wrong at their own boundary. `_MAX_DIGIT_ASPECT` was inclusive at
1.0, which is exactly where a chip's pale annulus lands at reduced render size, so a
crop containing no text read a confident `0.0` — and `stack_text` 0.0 is trusted by
the spine as all-in. Re-measured over the 276,307 glyphs that actually reach a
winning run (not the 47,121-glyph figure, which pooled the confusers in with the
digits), real digits reach 0.900 and every glyph above it is a chip or the `O` of
`POT:`; the gate is 0.95. The completeness net's height test excluded any short
neighbour, which silenced it on a leading digit **clipped** by a tight detector box —
short precisely because it is cut. Height alone cannot be the test, and neither can
the gap: the `BB` caps butt up against the value at 0.07-0.27 of run height, inside
the letter-spacing floor. The suffix is now identified as what it is — a short glyph
the bank matches confidently as a **non-digit** — and deleting the height test
outright (which the first attempt did) absorbed the `B` of `198 BB` as an `8` on 55
real reads.

`_INTRA_NUMERAL_GAP` was shared by two rules that want opposite margins. Split. The
comment's claimed "1.12x separation" holds only against integers: the measured
integer and decimal gap bands **touch** at 0.250, and that is now stated.

Two-digit numerals could never reach the decimal-gap fallback — it needs
`max(gaps) >= 1.7 * median(gaps)` and with one gap the median is the max — so a
one-decimal client's `7.5 BB` with a lost dot returned a confident 10x inflation.
Extending the fallback is not available (no median to compare against, and the
bands touch), so the rule is the fail-closed half, excluding numerals containing a
`1` because `1` inks a bare stroke inside a full-width advance and inflates the gap
beside it. Measured cost on the corpus: 0 reads.

### The calibration range

Nothing in the pipeline recorded the render geometry as a supported or unsupported
fact. `layout_profile` was read at export from timeline metadata that nothing in
`cv_lab` ever wrote, so it was always `""`. Below the calibrated range the reader did
not degrade to unknowns; it degraded to confident wrong values — `314.90` to `90.0`,
`19.50` to `50.0`, `343.60` to `360.0`, and a 343.6 BB stack to `0.0`, which the
spine reads as all-in.

The guard is stated in **rendered digit height**, because that is the quantity that
breaks and it is independent of detector box padding, crop dimensions and client
resolution. Measured over 11,800 value-producing reads, run height spans 12-32px;
re-read at reduced scale every read stays correct or fails closed down to 9, and
every confident wrong value in the whole 0.60x-2.10x sweep sits at 7 or 8.
`_MIN_CALIBRATED_RUN_H = 9` is 0.75x the smallest calibrated render and 1.125x the
largest render that produced a wrong value. `_layout_profile` records the modal frame
size on the timeline and marks it `-unsupported` below the smallest calibrated client.

Sweeping all 15 numeric fixtures at 31 scales from 0.60x to 2.10x — 465 reads —
produces **zero** confident wrong values. Before this round the same sweep produced
dozens, and the test that claimed to cover it swept one fixture at four scales.

### Corpus effect

Re-reading all 11,996 real crops, reader before vs after:

- 11,959 identical
- 30 unknown -> value, all correct recoveries (`218.0` x16, `212.5` x11, `35.3` x3)
- 7 value -> unknown, every one of which was previously **wrong**: `60.0` for a
  clipped `.60`, `9.0` for a crop containing only the player name `LugoMax`, `24.0`
  for a `240.9` pot occluded by a chip stack, `0.0` for a chip-covered bet, `182.0`
  for `182.50`
- 0 values changed to a different value

## 3. The green pill

The client paints CALL and BET on the same green. The word template misses on 964 of
2,439 `action_pill` reads across the five geometries (39.5%), 412 of them green, and
the colour fallback resolved green to `"call"` as "the safe default". It is not a
default: it asserts that somebody had already bet on that street.

The label was the smaller half of the damage. On the baseline recording's hand 7 the
flop went check-check, so seat 3 opens the turn and a call is physically impossible;
the forced `call` left `street_has_bet` False, so the fold handler — which correctly
refuses folds on a street with nothing to fold to — **discarded** seat 2's observed
FOLD pill, and the round-completion pass synthesised a CHECK in its place. The
exported record contained an action that never happened and omitted one that did, at
confidence 0.8 with no warning and no rejection code. Five of the 21 reconstructed
hands carried such a call.

Colour cannot resolve it; structure can. A seat putting chips in on a street where
nobody has bet is betting. `region_detections.read_pill_action` now returns
`PILL_BET_OR_CALL` and the spine resolves it from the betting history — preflop it is
a call, because the blinds are a standing bet and a preflop "bet" does not exist.
`validate_yolo_card_timeline` gained `call_with_nothing_to_call` as the permanent net
under it.

## 4. Accounting and evidence

- **Hero net.** The client renders stacks pre-debited, so `series[-1] - series[0]`
  silently excludes everything committed before the first sample — the blinds on
  every hand, and the whole preflop action on any hand picked up late. The 07-23
  3.33.54 PM session's first hand showed the hero's 24.0 BB raise standing for 18
  consecutive states while the stack never moved, and published `hero_bb_won = 52.8`
  for a hand whose net is 28.8. `_committed_at_start` reads the standing bet while
  the stack still holds its first value; the same value fills the pre-observed
  action's amount, which used to publish `raise amount=None` with the size legible on
  screen throughout.
- **Two stack boxes, one seat.** Resolved by iteration order, including when the
  later read is `None`, destroying a good one. Measured: 3 of 1,309 frames, and all 3
  disagree. Two contradictory readings are a conflict; the seat is unknown and the
  conflict is counted.
- **Hero-seat cross-check.** A majority vote over the hero zone's own cards, so with
  an empty hero zone it evaluates `0 > 0` and reports "confirmed" — and an empty hero
  zone is exactly what a layout drift produces (a 1.24x vertical stretch
  re-attributes hero's cards to two villain seats as showdown reveals while the board
  still zones 5/5 and the anchor residual stays in tolerance). `layout_supported` now
  requires positive confirmation, not the absence of a contradiction.
- **Counters for silence.** A pot box dropped by the centre-column guard, and a frame
  with too few stack reads for the sibling-median net, were both indistinguishable
  from "checked and clean". `_reject_stack_outliers` is still inert below four reads
  by design — a median of two is not a measurement — but the skip is on the record,
  which matters because ClubWPT runs 6-max and heads-up tables where it is the normal
  case and all five development recordings render eight stack boxes.
- **`_STACK_OUTLIER_RATIO`.** Described as "the measured midpoint" of 4.06 and 10.03.
  It is not (arithmetic 7.045, geometric 6.383). The comment now says what it is: a
  round value inside the measured gap, chosen toward the conservative end.

## 5. Training triage

`mine_yolo_card_hard_examples` still bucketed cards with the legacy unanchored
rectangles, which lose 100% of the community row at aspect ratio 1.750 — so on that
family the miner called every board card "other" and mis-prioritised exactly the
geometry that was broken. It reads a card-only CSV and has no landmarks to fit an
anchor from, but the community-row **shape** test needs neither: it measures the row's
span in units of the row's own median card width, a ratio in which the frame's scale
cancels. The legacy rectangle is kept as the fallback for cards the shape test cannot
place, deliberately: a row needs three cards, and "one or two board cards" is exactly
what the `partial_board_count` triage signal exists to surface.

## 6. Known limits carried forward

- The reader fails closed below 9px digit height rather than reading; a client
  rendering smaller than the 1272x896 development minimum will produce unknowns and a
  `-unsupported` layout profile, not values.
- `_reject_stack_outliers` remains inert on tables with fewer than four readable
  stacks. The skip is recorded; the net is not replaced.
- `layout_supported` now requires `hero_seat_confirmed`, which timelines built before
  this round do not carry. Rebuild a timeline rather than re-export an old one.
- The centre-column pot guard's reject branch is still unexercised on real material
  (0 of 1,267 boxes off-column). It is counted now, not validated.
