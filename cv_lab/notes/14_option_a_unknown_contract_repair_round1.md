# 14 — Option A (fail loudly) and adversarial repair round 1

Chronological record. Notes 01-13 stand as written; this note records (a) the
Option A reader-contract phase that note 13 predates, and (b) the round-1
adversarial repair run against it. Everything below is measured on the SIX
development recordings only (the five of note 13 plus `clubwpt_session_01.mov`,
which is development material: it supplied 228 frames to the region detector's
training manifest). No locked-test and no validation recording was opened.

## 0. The contract, restated

The heuristic numeric reader hit its ceiling: successive repair rounds kept
finding input combinations that produced a CONFIDENT WRONG NUMBER. The contract
changed to Option A: a value is returned only when the read is PROVABLY
unambiguous; every other outcome is UNKNOWN with a named refusal code. UNKNOWN
is first-class — not 0.0, not None-meaning-zero, never silently dropped.
Coverage loss is an accepted outcome; a confident wrong number is not.

## 1. Round-1 findings and what was done

Three adversaries (A: confident-wrong hunter; B: UNKNOWN integrity downstream;
C: mutation testing / honesty of stated evidence) filed 22 findings. Per-finding
disposition, with the general-vs-specific verdict that drove each fix:

### Reader (`ocr_readers.py`)

**A1/A3/A4 — occlusion and box-clipping undefended (general).** Ink smaller
than the glyph band was routed into buckets no predicate policed (`dots` and an
unnamed remainder), so a menu sliding over a seat panel read 50.0 for a 99.50 BB
stack (cwpt01 t=554), a card sprite over a pot's integer digit read 50.0 for a
6.50 BB pot (g0711 t=257 — note 13's "not repaired" regression), and left-edge
clipping produced 60,982 confident wrong values. The general fix is structural:

- P2 now polices EVERY component that vertically overlaps the numeral's row
  (`_policed_ink`); the only explained non-run ink is a proven affix (`named`)
  or a baseline separator candidate strictly inside the run.
- P4 additionally refuses a run whose left margin is inside the numeral's own
  letter spacing (reusing `_RUN_ADJACENCY_GAP`; no new constant): a cut landing
  in the inter-digit gap leaves at most one letter space (≤ 0.250 run_h) of
  margin, so a grazing edge is indistinguishable from a severed leading digit.
  Unconditional — no ink inside the crop can prove what lies beyond its edge
  (a first draft waived the margin when ink survived left of the run; a border
  speck defeated it and the waiver was removed).

**A5 — the P5 gap test was in an `elif` (general).** Once a separator was
located, no gap was examined anywhere in the run — 78.3% of production reads
take the dot branch — so a painted-out interior digit read 96.2% confident
wrong. The gap test now runs on every read: the located separator is subtracted
(`_largest_hole`) and every remaining hole must stay inside measured letter
spacing. The boundary differs by branch because the ambiguity differs:
`_DECIMAL_BAND_MIN_GAP` (3/13, ≥) with no separator (lost-dot ambiguity),
`_INTRA_NUMERAL_MAX_GAP` (0.250, >, the measured intra-numeral extremum) with
one (missing-digit ambiguity; the lost-dot ambiguity is spent once the client's
single separator is located). A first draft used 3/13 for both and refused 331
legitimate reads, 270 of them on the smallest client where 3px letter gaps in a
13px run sit exactly on the decimal band's floor; the two-boundary form costs 0.

**B9 — P5(a) ranked instead of refusing (latent, general).** The
widest-gap-first loop broke at the first reconcilable separator candidate, so
two candidates reconciling to different values were resolved by a ranking key
and a non-reconcilable candidate behind the winner was never examined. Replaced:
every candidate must reconcile, all to the same split, else
`separator_unreconciled`. 0 corpus reads affected (no real read carries two
candidates); the ordering machinery is deleted.

**A2 — the debounce was the only defence (downstream of A1).** Verified dead
end-to-end: re-running the two-frame menu-occlusion arm, both arms now
reconstruct identically (the crop refuses at the source; no `call 49.5`, no 10x
pot). The single-overwrite-in-18,006-reads debounce is no longer load-bearing
for this failure.

**A6/C3 — P7's claim overstated; zero band margin.** Comment corrected: 12 is
the smallest run height measured correct on a NATIVE client (exactly one
client), resampled crops show the bank's discrimination is thin at 11-12px, the
band has zero margin below and one pixel above, and coverage figures measured
inside it do not extrapolate to unseen geometries. The value is unchanged
(refusal-only; nothing to weaken).

**A8 — knife-edge refusals under scale perturbation. OPEN, documented.** 5,360
of 17,768 crops flip value→UNKNOWN→value within one 5% synthetic scale step
(suffix_not_bb and integer_over_decimal_band firing on 1px quantization noise).
This is instability of REFUSALS, not of values — the failure direction is
coverage, not correctness — and any fix is hysteresis/tolerance tuning, i.e.
exactly the cleverness this phase deletes. Left open on the record.

**C7/C8 — justification tables did not reproduce.** Both constants' comment
tables re-measured against the shipped reader and replaced (decimal-gap
populations n=13669/1541; affix table now covers all six recordings from the
current samples, g0621 n=4020 not the stale 8808, cwpt01 included). Constant
values unchanged in both cases.

**C5 — `_AFFIX_MAX_REL_H` mis-described.** The "refusal-only" comment was false:
passing the gate admits values through the P2 exemption (57 development reads
exist only because of it). Comment rewritten at both sites; the admitting
direction is now pinned by a frozen "198 BB" crop
(`stack_198_suffix_named_at_2054x1470.png`), so the K-affix0 ablation no longer
survives the suite.

### Spine (`build_yolo_hand_timeline.py`)

**A7 — pre-debited starting stacks (general).** `starting_stack` published the
raw first-state read while the ledger booked the standing money as an action:
31 understated player rows, one arithmetically impossible (97.5 published,
99.5 contributed, confidence 1.0, warnings=[]). `_observed_starting_stack` is
now first-state read + `_committed_at_start`, unknown when either half is.

**B1 — refused first-state bet backfilled (instance of the general "refusal
treated as absent" defect).** The pre-observed-action branch now consults
`first["bets_unknown"]`; a refused first-state bet emits the action with
amount=None (an unknown money action), never a number scanned from a later
state. `_committed_at_start` was rebuilt around an explicit constancy proof:
the client pre-debits, so the standing bet is constant while the seat's stack
reads its starting value; a read inside that proven window is a measurement
(used for starting_stack / hero net), the scan STOPS at a refused stack read
(a debit can hide inside one — the synthetic arm that booked 24.0 for a true
4.0 dies there), absent-stack states neither extend the proof nor supply
readings, and a window with a refusal and no read returns None, not 0.0.

**B2 — `reconciled: true` over a ledger unknown.** The contribution estimator
now starts holed whenever `amounts_unknown_in_ledger` holds (computed before
the estimators, which it previously was not). Hands where pot text and the
winner's sweep — two estimators genuinely independent of the action ledger —
agree still reconcile, and every such hand carries the fatal
`amounts_unknown_in_ledger` code regardless.

**B3 — corpus-split violations.** `extract_gallery_frames.py` no longer maps or
decodes any held-out recording (entries deleted, not commented).
`_FOLD_MIN_RUN_CALIB`'s comment no longer cites v01#9: re-measured on the six
development recordings, confirmed folds run 6-64 samples (at 1s), no raw dim
run reverts mid-hand, so development data cannot distinguish 1 from 3 and 3 is
retained as the conservative (refusal-side) resolution. The phantom-winner
demotion comment now cites the 13 development hero-fold hands whose reconciled
villain sweeps it keeps (e.g. g0723a hand 3) instead of v01#1.

**B4 — unknown starting stack silently "complete".** New fatal code
`starting_stack_unknown` (spine warning → validator mirror → rejection code →
completion "uncertain" → confidence ≤ 0.5). Fires on 9 of 31 hands: 6 with the
first-state stack read refused (including the finding's g0711 hand 5 seat 5),
3 via `committed_at_start_unknown` (a refused bet in the committed window with
no proven read).

**B5 — `settle_scan_skipped` consumed by nothing.** Now reaches
CompletionEvidence.extra as `cv_settle_scan_skipped` and caps confidence at
0.80 (g0723a hand 5's six blind spots are no longer invisible; sens.py's
counterfactual showed the chosen settle index is unchanged on all six affected
hands, so this is visibility, not a wrong winner).

**B6 — one treatment of an unread pot in `_settle_index`.** The suffix max is
documented and computed as a lower bound over KNOWN pots feeding a one-sided,
proven-evidence-only rejection guard; the refused case skips (as before), and
the absent-box substitution is documented as guard-input-only. Measured against
the pre-Option-A scan: settle index identical on all 31 hands.

**B7 — `side_pot_unknown` read by nothing.** A refused side-pot read now fires
`side_pot_unsupported` (a detected second pot with an unreadable amount is not
"no side pot"). Latent on this corpus (0 of 716 states), tested synthetically.

**B8 — count vs breakdown mismatch.** `pot_zero_impossible` /
`bet_zero_impossible` now increment `amounts_unknown` alongside the by-code
map, so the exporter's "per-reason breakdown of the count above" sums to the
count and pot dropouts reach the unread-amount confidence cap.

### Tests (adversary C)

The four suite-surviving mutants are dead, each killed by a regression that
fails first against the mutant:

| mutant | kill |
|---|---|
| U-det0 (refusal → confident 0.0 at region_detections) | `test_frame_from_models_preserves_the_three_way_amount_distinction` (real bank, real crops, at the production line) |
| U-detabsent (refusal reason dropped → ABSENT) | same test |
| P4-clip (`if False:` the clip refusal) | `test_left_clip_family_never_yields_a_different_value` (also requires `run_clipped` to actually fire) |
| K-affix0 (`_AFFIX_MAX_REL_H = 0.0`) | `test_affix_gate_is_value_admitting_and_pinned` |

New spine regressions (each verified to fail against the reverted behaviour):
standing-stack inclusion, flicker-positive control, refused-first-bet,
committed-scan-stop, reconciled-abstention, starting-stack fatal code,
refused-side-pot, count-vs-breakdown, settle-scan evidence + cap, and
`starting_stack_unknown` added to the literal fatal-code export gate test.

## 2. Family verification (not instance verification)

Re-running the adversaries' own harnesses against the repaired reader, whole
corpus:

| family | before | after |
|---|---|---|
| left-edge clip, 1..60px, 17,469 crops | 60,982 confident wrong (9,628 crops) | **0** |
| right/top/bottom clip | 0 | 0 |
| interior digit painted out (n=39,953) | 38,439 wrong (96.2%) | 1,254 (3.1%); 38,305 UNKNOWN |
| leading digit painted out (n=15,408) | 13,485 wrong (87.5%) | 12,746 (82.7%) |
| trailing digit painted out (n=15,408) | — | 1,540 (10.0%) |
| two-sample menu occlusion (A2 end-to-end) | fabricated `call 49.5`, 10x pot, warnings=[] | identical clean reconstruction in both arms |

**Stated limit, deliberately.** A trace-free occluder that removes an EDGE
digit — an opaque background-coloured paint over the leading (or trailing)
digit that leaves no fragment, no sliver, and letter-space-consistent geometry
— produces a crop that IS a legal rendering of a smaller numeral ("191.3" with
the '1' painted out is pixel-legal "91.3"). No reader can refuse it from the
pixels without refusing every legitimate read. Production occluders measured on
this corpus (menu panels, card sprites, chip sprites) all leave fragments and
are refused; the synthetic residual above is the information-theoretic floor of
the family, not an unfixed bucket. Interior removal, which does leave evidence
(a hole), is refused at 96.9%; the 3.1% residual is paint rectangles that
happen to also erase the separator and re-form a legal shorter numeral.

## 3. Corpus effect of the whole round

18,006 amount crops, six development recordings, byte-identical detector boxes:

- UNKNOWN → value: **0**. value → different value: **0**.
- value → UNKNOWN: **10** — the two adversary-demonstrated confident-wrong
  occlusion reads (now refused at source), and 8 `run_clipped` refusals of a
  "0.50 BB" bet whose numeral grazes its crop's left edge (a grazed 0.50 is
  indistinguishable from a clipped 10.50; refusal is the contract).
- refusals 537 → 547; census: dot 13,669 / integer 3,790 values;
  suffix_not_bb 221, no_digit_run 202, unexplained_ink_in_numeral 49,
  integer_over_decimal_band 47, above/below_calibrated_render_size 16/1,
  run_clipped 8, ambiguous_longest_run 3.
- Exported hands: **12 → 9 of 31** (g0723a 4→3, g0711 2→1, cwpt01 6→5). All
  three losses are hands whose ledger or starting stacks now carry a named
  unknown (`amounts_unknown_in_ledger` from a refused standing bet;
  `starting_stack_unknown`). Every exported hand carries zero ledger unknowns
  and zero starting-stack unknowns. That trade — three fewer exports, no
  invisible unknowns — is the phase's stated purpose.

## 4. Corrections to the standing record

- **Note 13 §8's suite figure is superseded**: the owned CV surface is now 270
  passed / 1 skipped (was 200/1 in note 13, 253/1 entering this round).
- **The "541 legitimate >= 1000 reads on cwpt01" figure (C9)** was the
  before-arm's number; the shipped reader's is 528 (13 of the 541 are refused).
  The argument it supported — that a ">= 1000 BB is implausible" proxy is dead,
  since 1093/1131.90 BB are real stacks — is unaffected.
- **PLAN.md's phase entry is stale** (it still reads "14 of 21 hands moved to
  11 of 21" with per-hand attributions from the five-recording corpus). The
  current figures are this note's §3: 9 of 31 over six recordings. PLAN.md is
  being edited concurrently by the accounting workstream and is not touched
  from this one; its CV-coverage paragraph should be updated to cite this note.
- **Coverage disclosure (C3)**: every coverage figure in this note is measured
  at exactly the render sizes the calibrated band [12, 32] was fitted to, with
  zero margin below and 1px above. It does not extrapolate to unseen clients;
  an unseen geometry fails closed toward UNKNOWN (the intended direction).

## 5. What this round does NOT establish

Unchanged from note 13 §9: no locked-test gate has been evaluated, no
validation recording opened, exported-hand correctness is self-consistency (not
ground truth), the confidence scale is hand-set, and 9 of 31 is what this
corpus and this code produce today — not a coverage claim.
