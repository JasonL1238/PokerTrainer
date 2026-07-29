# 13 — Phase 5 verification: geometry anchoring and OCR scale

Chronological record. This note does not revise notes 01-12; where it disagrees
with an earlier number, the earlier number stands as what was true when it was
written and this one records what the final code measures.

Everything below is measured on the five DEVELOPMENT recordings only. No
locked-test and no validation recording was opened, decoded, or inspected during
this phase.

## 0. How the measurement was produced, and why it is comparable

Notes 11 and 12 both had to caveat that their before/after populations came from
different inference runs. That caveat does not apply here.

The pre-fix artifacts (`scratchpad/geom_test/`) and the final artifacts
(`scratchpad/final5/`) were produced by the same command on the same five files
at the same sample times. Comparing them detection by detection:

| recording | frames | detection boxes | boxes identical before/after |
|---|---|---|---|
| g0723a | 361 | 7728 | yes |
| g0621 | 361 | 8111 | yes |

Region-detector output is bit-identical across runs on this hardware, so every
difference reported below is attributable to the reconstruction/OCR code and to
nothing else. Per-class detection counts are identical on all five recordings
(2885/767/352 numeric boxes on g0723a, 158/29/27 on g0715, and so on), which is
the same check run the other way round.

Board-zone recall is scored against `scratchpad/board_gt.py`, which defines a
community card purely from detector box geometry (the widest CONTIGUOUS same-row
run of >= 3 similarly-sized face_card boxes) and uses no zone constant from
either the old or the new code, so it can score both fairly.

The five geometries:

| tag | file | geometry | AR |
|---|---|---|---|
| g0723a | 07-23 3.21.47 PM | 2054x1470 | 1.397 (reference basis) |
| g0621 | 06-21 12.41.17 AM | 2062x1178 | 1.750 |
| g0711 | 07-11 12.45.27 PM | 2722x1832 | 1.486 |
| g0715 | 07-15 10.16.07 PM | 2132x1378 | 1.547 |
| g0723b | 07-23 3.33.54 PM | 1272x896 | 1.420 (smallest) |

## 1. Defect 1 — hardcoded card zones did not anchor

`build_yolo_card_timeline._zone_for_box` tested raw frame-normalized rectangles.
The board window required `0.36 <= cy <= 0.55`.

Raw normalized cy of the independently-derived community cards, current
detections, n = 2067:

| recording | cy range | inside the old 0.36..0.55 window |
|---|---|---|
| g0723a | 0.4405..0.4436 | yes |
| g0621 | **0.3343..0.3388** | **no — the whole row is below the floor** |
| g0711 | 0.4109..0.4148 | yes |
| g0715 | 0.4324..0.4476 | yes |
| g0723b | 0.4528..0.4551 | yes |
| all | 0.3343..0.4551 | spread 0.1208 against a 0.19-tall window |

Board-card recall, same detections, old hardcoded test vs the anchored
`landmark_anchor.zone_for_ref` the spine now uses:

| recording | community cards | old zoning | anchored zoning | old false "board" | anchored false "board" |
|---|---|---|---|---|---|
| g0723a | 642 | 642 (100.0%) | 642 (100.0%) | 6 | 3 |
| g0621 | 435 | **0 (0.0%)** | **435 (100.0%)** | 1 | 0 |
| g0711 | 617 | 617 (100.0%) | 617 (100.0%) | 4 | 2 |
| g0715 | 39 | 39 (100.0%) | 39 (100.0%) | 1 | 1 |
| g0723b | 334 | 334 (100.0%) | 334 (100.0%) | 0 | 0 |
| **total** | **2067** | **1632 (79.0%)** | **2067 (100.0%)** | **12** | **6** |

Frames on which at least 3 board cards were zoned as board: 458 of 562 before,
562 of 562 after.

After anchoring, the same 2067 community cards collapse from a 0.1208 spread to
0.0141 in reference-y:

| | ry range across all 5 geometries | smallest margin to a band edge |
|---|---|---|
| board (band 0.409..0.470) | 0.4372..0.4513 | 0.0187 |
| hero (band 0.644..0.717) | 0.6737..0.6874 | 0.0296 |

The reported pre-fix hero margin of 0.004 on g0621 is gone: the tightest hero
margin anywhere in the corpus is now 0.0296, and the tightest board margin
0.0187 (g0715, whose river card is the corpus extreme).

End-to-end consequence on g0621, the recording the defect destroyed:

| | before | after |
|---|---|---|
| hand 1 board | `[]` | `['8d','Kd','Jc','9h','Jh']` |
| hand 1 action streets | 13 actions, all `preflop` | 12 actions across flop/turn/river |
| hand 2 board | `['4h']` (`invalid_board_count`) | `['Ad','7c','4h','As']` |
| timeline `card_complete_hands` | 2 of 3 | 3 of 3 |
| hands exported with an empty board and a played-out street | 2 | 0 |
| shipped session confidence | 0.989 | 0.0 |

`_zone_for_box` still exists but is now confined to the offline card-only CSV
builder and the hard-example miner, is documented as legacy and unanchored, and
the spine reaches zones only through `zone_for_ref`. `region_detections` is the
single caller.

## 2. Defect 2 — OCR decimal inference in absolute pixels, 2-decimal assumption

Every one of the 14390 numeric crops in the corpus was re-read at the same
detector boxes. Exact crop-level before/after:

| recording | numeric crops | value changed | scale corrections (10x/100x) | other value corrections | now fails closed | unknown -> value |
|---|---|---|---|---|---|---|
| g0723a | 4004 | 29 | 20 | 0 | 9 | 0 |
| g0621 | 4472 | 2 | 0 | 0 | 2 | 0 |
| g0711 | 3539 | 11 | 0 | 8 | 3 | 0 |
| g0715 | 214 | 15 | 4 | 10 | 1 | 0 |
| g0723b | 2161 | 365 | 345 | 16 | 4 | 0 |
| **total** | **14390** | **422** | **369** | **34** | **19** | **0** |

`unknown -> value` is 0 across the whole corpus: no fix in this phase converted a
read that previously failed closed into a confident value. Every change is either
a correction or a new refusal.

The 21 distinct corrections, with multiplicity:

```
g0723b stack_text  34410.0 -> 344.1  x70     g0723b bet_text    1830.0 -> 18.3   x14
g0723b stack_text  20410.0 -> 204.1  x52     g0715  pot_text      24.09 -> 240.9  x10
g0723b bet_text       50.0 -> 0.5    x48     g0723b stack_text  11490.0 -> 114.9  x9
g0723b stack_text  29720.0 -> 297.2  x47     g0711  pot_text        0.0 -> 9.0    x7
g0723b pot_text      750.0 -> 7.5    x30     g0723b bet_text    1950.0 -> 19.5    x5
g0723a bet_text       50.0 -> 0.5    x20     g0715  pot_text      891.0 -> 89.1    x3
g0723b stack_text  19760.0 -> 197.6  x20     g0723b pot_text      950.0 -> 9.5     x3
g0723b stack_text  31490.0 -> 314.9  x17     g0723b stack_text  40760.0 -> 407.6   x2
g0723b stack_text     21.0 -> 218.0  x16     g0715  stack_text    797.0 -> 79.7    x1
g0723b stack_text  40710.0 -> 407.1  x14     g0711  pot_text        0.5 -> 50.0    x1
g0723b stack_text  34310.0 -> 343.1  x14
```

Twenty of the twenty-one are corrections. The twenty-first is a **regression**
and is recorded as one in section 6.

Per-defect, using the implausibility metric (a BB-denominated read >= 1000; no
field in this corpus legitimately exceeds ~400 BB):

**(a) absolute-pixel decimal-gap threshold, smallest client (1272x896)**

| | before | after |
|---|---|---|
| stack reads >= 1000 | 245 / 1554 (15.8%) | 0 / 1553 (0.0%) |
| bet reads >= 1000 | 19 / 381 (5.0%) | 0 / 378 (0.0%) |
| largest stack read | 40760.0 | 407.6 |
| all numeric reads >= 1000 | 264 / 2126 (12.42%) | 0 / 2122 (0.00%) |
| frames failing max/median stack >= 6.0 | 155 / 197 | 0 / 197 |
| stack max/median ratio, median | 139.255 | 1.744 |
| stack max/median ratio, worst | 211.206 | 2.112 |

**(b) hardcoded 2-decimal split, one-decimal client (07-15)**

| crop | true | before | after |
|---|---|---|---|
| POT: 89.1 BB | 89.1 | 891.0 | 89.1 |
| POT: 240.9 BB | 240.9 | 24.09 | 240.9 |
| POT: 165 BB | 165 | 165.0 | 165.0 |
| a seat stack | 79.7 | 797.0 | 79.7 |

g0715 pot median 24.09 -> 89.1, pot max 891.0 -> 240.9, stack max 797.0 -> 323.0,
stack outlier frames 1/21 -> 0/21.

**(c) dropped leading "0." on the BASELINE recording**

20 consecutive `bet_text` reads at t=321..340 on g0723a read 50.0; all 20 now
read 0.5. This is the case that reached an exported hand as
`preflop seat:3 call 50.0` in a 200 BB game. 48 further occurrences on g0723b.

**(d) genuine "0 BB" all-in reads**

`stack_text` 0.0 is now a positive fact rather than a discard: `ocr_readers`
separates "no value" from a real zero, and the spine trusts a located zero. On
g0715 the 25 of 154 (16.2%) stack reads that are genuinely 0 survive into all 15
timeline states in both runs; what changed is that they are no longer
indistinguishable from a failed read.

**Cost of the new refusals.** Fail-closed numeric reads rose from 178/14390
(1.24%) to 197/14390 (1.37%) — 19 crops, +0.13 points. Twelve of the nineteen
were previously shipping a confident **0.0** — a value the spine reads as a
genuine all-in — and the remaining seven were 60.0 (x3), 9.0, 24.0, 195.0 and
182.0.

## 3. Defect 3 — pot was a single scalar; side pots unrepresentable

`assign_regions` now splits `pot_text` candidates on the anchored main/side row
boundary and reports both. On g0715, the one development recording with a side
pot:

| | before | after |
|---|---|---|
| hand 1 pot | 1.0 | 240.9 |
| hand 1 side_pot | field did not exist | 0.2 |
| hand 1 t_end | 21.0 (past the settlement, latched the next deal's blind pot) | 18.0 |
| warning | `pot_not_reconciled` | `side_pot_unsupported` |
| exported | 0 of 1 | 0 of 1 |

The hand is still not exported, and that is the intended outcome: PLAN.md's
"Pot, winner, and result reconstruction" section requires side-pot outcomes to be
**rejected explicitly** until representative truth cases exist. The change is
that the rejection now names the real reason instead of hiding a 240x pot error
behind a generic reconciliation failure.

## 4. Defect 4 — confidence tracked detection quality, not correctness

Session confidence was a density-style figure that averaged across a timeline, so
a session whose every board had been destroyed reported 0.989. It is now
`min(per-hand confidence)` — a session is only as trustworthy as its worst hand —
and is reported for operators only; the gate is per hand.

Shipped export figures, before vs after:

| recording | timeline hands | exported before | exported after | after: carrying LOW_CONFIDENCE | session conf before | session conf after |
|---|---|---|---|---|---|---|
| g0723a | 7 | 6 | 4 | 2 | 0.994 | 0.1 |
| g0621 | 3 | 2 | 0 | — | 0.989 | 0.0 |
| g0711 | 6 | 4 | 4 | 2 | 0.985 | 0.0 |
| g0715 | 1 | 0 | 0 | — | 0.978 | 0.4 |
| g0723b | 4 | 2 | 3 | 2 | 0.987 | 0.4 |
| **total** | **21** | **14** | **11** | **6** | | |

Before, all 14 exported hands shipped at `confidence_score` 0.95 with `tags=[]`
— the pipeline had no way to say "exported, but check this". After, 6 of 11
carry `LOW_CONFIDENCE` and per-hand scores separate (1.0 / 0.8 / 0.65 / 0.55).

The same **pre-fix timelines**, re-scored by the current validator, carry 23
warnings on 13 of 21 hands; the pre-fix validator reported 15 warnings on 7
hands. Two of the codes it raises on that old material did not exist at all
before this phase — `action_sequence_illegal` (7 occurrences) and
`board_empty_but_streets_advanced` (1) — and `side_pot_unsupported` is likewise
new. `invalid_board_count` (2) already existed.

## 5. Coverage went DOWN, 14 -> 11, and that is the reported result

Three hands stopped exporting and one started.

**Lost, g0621 hands 1-2.** Both shipped with `board_cards=""` while their action
ledgers described a played-out hand. Hand 1 shipped `result="Hero wins +116 BB"`,
`warnings=none`, session confidence 0.989. These are the destroyed boards; they
are now reconstructed (section 1) but their street reconstruction still trips
`board_regression` / `street_order_issue`, so they are held for review rather
than shipped.

**Lost, g0723a hands 1-2**, both to `action_sequence_illegal`, a code that did
not exist before this phase. Their reconstructions contradict themselves on the
pixels of their own ledger:

- hand 1 preflop books `seat 7 call 2.0` immediately followed by `seat 7 fold`,
  and on the turn books `seat 0 call 16.0` against a raise to 32.0;
- hand 2 river books `seat 3 all-in 148.1` immediately followed by `seat 3 fold`.

Both previously exported at confidence 0.95 with `tags=[]`. Hand 2's ledger
actually **improved** in this phase — `flop seat:3 call 11.8` became
`flop seat:3 bet 11.8` and `turn seat:0 bet / seat:3 call` became
`turn seat:3 bet / seat:0 call`, both correct after a checked-through street —
and it is still rejected, because the river contradiction is real.

**Gained, g0723b hand 2** (`Jh Ac`, board `Tc Kc 7c 4s`, pot 170.6), which the
100x stack inflation had previously made unreconstructable.

Alongside that, g0723b's shipped stacks stopped being nonsense: 29720.0 ->
297.2, 20410.0 -> 204.1, 40710.0 -> 407.1, 11490.0 -> 114.9, 34310.0 -> 343.1.
Exported player rows with a starting stack >= 1000 BB: 5 before, 0 after. And
hand 1's `hero_bb_won` moved 52.8 -> 28.8, the pre-debited-stack correction.

Net: 14 exports of which at least 6 carried a confirmed material defect and none
said so, versus 11 exports of which 6 say they need checking.

## 6. What did NOT improve, or got worse

**One OCR read regressed.** g0711 t=257, `pot_text`, 0.5 -> 50.0. The crop is a
seat panel whose integer part is hidden by an OPAQUE sprite, so the visible glyphs
are ".50 BB" and no remnant exists for the truncation net to fire on;
`dot_centers` filters `x0 < centre < x1`, so a decimal preceding the first
accepted digit is discarded. Both readings are wrong — the true value is some
X.50 — but the new one is wrong by 100x. 1 occurrence in 14390 native crops. It
falls at t=257, between hand 4 (ends t=252) and hand 5 (starts t=258), and
reaches no exported hand in this corpus. Not repaired.

**g0621's warning count went UP, 6 -> 10.** This is the honest cost of fixing
Defect 1: before, the boards were empty so the board-regression and street-order
nets had nothing to examine. Now that the boards exist, they flag three hands.
g0621 exports 0 of 3 hands. The recording is reconstructed further than it was;
it is not reconstructed correctly.

**`board_regression` and `street_order_issue` rose corpus-wide**, 5 and 6 before
to 8 and 11 after, for the same reason. Street reconstruction on a recovered
board is the largest open reconstruction defect in this corpus.

**Card classifier suit accuracy is unchanged and is a model problem.** 1 wrong
card in 33 audited exported board cards (3.0%) — g0723a's turn reads 8d eight
times and 8h twice against an eight of hearts on screen. The identity-split net
flags it (that hand ships at 0.65 with `LOW_CONFIDENCE`), but a card the
classifier reads wrong *unanimously* is invisible to every check in the pipeline.
Closing this needs retraining, not code.

**Known reader limits carried forward from note 12**, all unrepaired: the
two-digit `"1" not in digits` exclusion leaves a 10x hole at every gap width on a
one-decimal client; a seat-timer badge can outscore the real stack in the same
crop (g0621 t=177, "212.90 BB" + a timer ring reading "12" -> 12.0; that read now
fails closed rather than shipping 12.0, but the underlying scoring flaw stands);
`_AFFIX_MAX_REL_H` does not separate the "BB" suffix by height at all on g0715.

**The region detector's training manifest still contains a locked-test
recording.** `cv_lab/datasets/yolo_cards_autolabel_v1/manifest.csv` holds 228
rows from `clubwpt_session_01.mov`, in both train and val. Pre-existing;
`git diff HEAD -- cv_lab/datasets/` is empty for this phase. Every generalization
claim about the region detector in notes 10-13 is therefore made about a model
that has seen a locked recording, and the locked test cannot measure that model
cleanly until it is retrained from a clean manifest.

## 7. What generalizes — regression guard, before vs after

These are the parts notes 10-12 identified as strong. All are unchanged.

| recording | det/frame | empty frames | card labels read | unknown labels | frames with a duplicate label | states | hands | hero-seat mismatches |
|---|---|---|---|---|---|---|---|---|
| g0723a | 21.4 -> 21.4 | 0 -> 0 | 1380 -> 1380 | 0 -> 0 | 0 -> 0 | 169 -> 169 | 7 -> 7 | 0 -> 0 |
| g0621 | 22.5 -> 22.5 | 1 -> 1 | 1152 -> 1152 | 0 -> 0 | 0 -> 0 | 65 -> 65 | 3 -> 3 | 0 -> 0 |
| g0711 | 19.3 -> 19.3 | 0 -> 0 | 1287 -> 1287 | 0 -> 0 | 0 -> 0 | 140 -> 140 | 6 -> 6 | 0 -> 0 |
| g0715 | 16.8 -> 16.8 | 8 -> 8 | 118 -> 118 | 0 -> 0 | 0 -> 0 | 15 -> 15 | 1 -> 1 | 0 -> 0 |
| g0723b | 20.7 -> 20.7 | 0 -> 0 | 705 -> 705 | 0 -> 0 | 0 -> 0 | 82 -> 81 | 4 -> 4 | 0 -> 0 |

Hand segmentation: 21 hands before, 21 after, on all five geometries including
the tournament layout the detector was never trained on. Card classifier: 0
unknowns and 0 duplicate-label frames across 4642 card reads. Seat assignment: 0
hero-seat mismatches across 470 states. The one state-count change (g0723b
82 -> 81) is a collapse of two identical adjacent states after the stack reads
stopped disagreeing 100x.

## 8. Suite and lint, as run

```
$ python -m pytest -q tests/test_ocr_readers.py tests/test_two_model_spine.py \
    tests/test_yolo_card_timeline.py tests/test_yolo_hand_timeline.py \
    tests/test_yolo_card_app_export.py tests/test_yolo_card_timeline_validation.py \
    tests/test_video_processing.py
200 passed, 1 skipped in 0.43s

$ python -m pytest -q            # whole repo, both concurrent workstreams
1017 passed, 1 skipped in 20.81s

$ python -m ruff check cv_lab/
All checks passed!
```

Each confirmed defect above has a permanent regression under
`tests/test_cv_recording_regressions.py` (29 tests) built from frozen frames and
crops in `tests/fixtures/cv/` and `tests/fixtures/ocr/`, with sources recorded in
`tests/fixtures/PROVENANCE.json`. All fixtures are development-corpus only.

## 9. What this phase does NOT establish

- **No locked-test threshold in the Hard Release Gates table has been
  evaluated.** No answer key exists for any recording, locked or development. The
  gates for card accuracy, completion classification, hand boundaries,
  reconstruction error budget, accounting, and safe rejection are all unmeasured,
  not passed.
- **No validation recording has been opened either**, so anchor confidence,
  reader calibration, and the confidence scale remain measured on the same five
  files the constants were derived from. That is development-loop discipline, not
  held-out evidence.
- **Exported-hand correctness is not measured.** Everything in section 5 is
  self-consistency: a hand that survives every net may still be wrong. The only
  independent ground truth used anywhere in this note is the box-geometry
  community-row definition in section 1 and the distinct-hero-pair hand count.
- **The confidence scale is still hand-set.** `WARNING_SEVERITY` and
  `_REJECTION_SEVERITY` are constants, not calibrated thresholds. The scores in
  section 4 separate hands usefully; they are not probabilities.
- **Eleven of twenty-one hands is not a coverage claim.** It is what this corpus
  and this code produce today, on five recordings totalling 21 hands — far below
  the 10 sessions / 100 completed hands the corpus gate requires.
