# 15 — Option A contract, adversarial repair round 2

Chronological record. Notes 01-14 stand as written. Everything below is measured
on the SIX development recordings only (the five of note 13 plus
`clubwpt_session_01.mov`, which is development material — 228 of its frames are
in the region detector's training manifest). No locked-test and no validation
recording was opened, decoded, or inspected.

## 0. How this round differs from the four that did not converge

Notes 10-12 record three adversarial rounds that failed to converge because each
repair fixed the exact input the adversary demonstrated, so the next round found
a neighbouring input. This round's discipline, applied to every finding before
any code changed:

1. **Reproduce independently**, from the repo's own committed state, without
   trusting the finding's numbers.
2. **State a general-vs-specific verdict in writing.** If general, fix the
   general case and enumerate the whole family rather than the instance.
3. **Prefer widening UNKNOWN over adding a threshold.** No constant was added in
   this round, and none was weakened.

The baseline was re-derived rather than assumed: rebuilding all six timelines
from the retained byte-identical detector frames reproduces note 14 §3 exactly —
**9 of 31 hands exported** (g0723a 3, g0621 0, g0711 1, g0715 0, g0723b 0,
cwpt01 5) — so every measurement below is comparable to that note.

## 1. Fifteen findings, and what each turned out to be

Nine of the fifteen were **already repaired** in the committed state and were
re-verified as closed rather than taken on faith. Three needed work. Three are
disclosed residuals with no action.

| # | finding | verdict | disposition |
|---|---|---|---|
| 1 | dot forgery: a 2x2 baseline speck fabricates a decimal | already fixed | verified closed |
| 2 | wide occluder defeats P2's adjacency window | already fixed | verified closed |
| 3 | resampled crops at the band floor read wrong | disclosed residual | no action, see §5 |
| 4 | refused first-state bet on a pill-less seat deleted an action | already fixed | verified closed |
| 5 | contrib pot estimator did not abstain on a refused first pot | already fixed | verified closed |
| 6 | `pot_text_dropped` consumed by nothing | already fixed | verified closed |
| 7 | export gate failed open with no `states` list | already fixed | verified closed |
| 8 | `generate_hand_history` crashed on an UNKNOWN starting stack | already fixed | verified closed |
| 9 | `_result_contradicts_hero_net` never checked `"Hero folds"` | **general, open** | **fixed** |
| 10 | P1 (`suffix_not_bb`) has no killing test | **general, open** | **fixed** |
| 11 | `_committed_at_start` refused-window `None` → `0.0` mutant survives | already fixed | verified killed |
| 12 | committed-scan stop-at-refusal unpinned | already fixed | verified killed |
| 13 | PLAN.md carries superseded CV coverage claims | outside this workstream | see §6 |
| 14 | `_AFFIX_MAX_REL_H` survives as value-ADMITTING | documented deviation | verified pinned both ways |
| 15 | row-band split (`0.6 * max_h`) unpinned | **specific, open** | **fixed** |

### Closed-already, as verified

Findings 1 and 2 are refused at the source by the shipped reader, on the frozen
crops the findings name:

```
stack_197_at_2054x1470                        197.0  integer
stack_197_speck_forged_decimal_at_2054x1470   None   unexplained_ink_in_numeral
stack_343_60_at_1272x896                      343.6  dot
stack_343_60_sprite_far_fragments_at_1272x896 None   unexplained_ink_in_numeral
```

Findings 4-8 are closed structurally: the pre-observed scan iterates
`first["bets_unknown"]` on the pill-less path, `initial_pot_refused` removes the
contribution estimator when the first pot read in the window was refused,
`pot_text_dropped` reaches the exporter's severity table through
`_split_source_codes`, the exporter validates a stateless timeline as
`{**timeline, "states": []}` instead of skipping the gate, and the hand-history
renderer prints `stack unread` / `pot unread` and skips SPR when any live stack
is unknown. Findings 11 and 12 are killed by their regressions:
`S-committed-refusal-to-0.0` and `S-committed-scan-past` both fail the suite.

## 2. Finding 10 — P1 unpinned. **General**, and the general form is what was fixed

**Verdict: instance of a general defect.** The specific complaint is that P1 has
no killing test. The general defect is *an acceptance-path predicate pinned by
nothing* — and a fix that wrote one test for P1 would have left the rest of the
class exactly where the four non-converging rounds left their neighbours.

So every predicate in `read_number_detail`'s acceptance path was ablated
independently and the owned CV suite run against each. Nineteen predicates,
before the repair:

```
P1-suffix          SURVIVED  280 passed
band-split-0.6     SURVIVED  280 passed
P2-near, P2-far, P3-unique, P4-clip, P4-margin-only, P5a-reconcile,
P5a-agree, P5b-integer, P5b-dot, P6-leading0, P7-low, P7-high,
sep-hostgap, sep-baseline, digit-aspect, digit-height, affix-named-h
                   killed
```

Exactly two holes, and finding 15 is the second of them. After the repair, all
nineteen are killed.

**Reproduction of the P1 hole.** With `if False and suffix[:2] != ["B", "B"]:`
the suite stays green and **218 of the 219** `suffix_not_bb` refusals in the
18,006 retained development crops come back as confident values, across five
recordings (g0621 109, cwpt01 63, g0723b 41, g0711 4, g0723a 2) and all three
numeric classes (stack 203, bet 14, pot 2).

**The regression that fails first.** `stack_212_90_timer_badge_at_2062x1178.png`
— a real g0621 seat panel carrying the stack the screen renders "212.90 BB"
*and* a circular action-timer badge reading "12" over the avatar. The badge's
2-digit run wins the read and a timer carries no "BB", because a timer is not an
amount. With P1 off this crop reads a confident **12.0 at score 0.796**, a 17.7x
under-read of a real stack — the same crop and the same 12.0 note 12 recorded
back when run completeness was a ranking key. `test_p1_refuses_the_timer_badge_that_outscores_the_stack`
asserts the value is unknown, is not 12.0, and that `suffix_not_bb` is the
predicate that refuses it (so P1 stops being pinned the moment another predicate
starts covering the crop instead).

**Three further instances, constructed here and verified.**
`test_p1_fires_on_every_malformed_terminator` covers the ways a terminator can be
malformed while the numeral stays clean; each returns a confident 312.0 with P1
ablated:

| instance | shipped | P1 ablated |
|---|---|---|
| no terminator at all | `suffix_not_bb` | **312.0** |
| one cap instead of two (`"B"`) | `suffix_not_bb` | **312.0** |
| second cap reads as a digit (`"B8"`) | `suffix_not_bb` | **312.0** |
| `"8B"` | `unexplained_ink_in_numeral` (P2 first) | refused |
| `"88"` | `suffix_not_bb` | refused (`integer_over_decimal_band`) |
| control `"BB"` | **312.0** | 312.0 |

The last three rows are recorded and deliberately **not** asserted as P1: "8B"
is refused earlier by P2 and "88" cannot kill the mutant because a later
predicate catches it. Claiming a predicate fires where a different one actually
fires is precisely how P1 came to be unpinned — the old in-code claim at
`tests/test_ocr_readers.py` that "P1's own firing is pinned by
`test_bb_suffix_is_never_absorbed_into_the_numeral` and by the 221
`suffix_not_bb` refusals in the corpus census" was false in both halves (that
test only asserts the caps are not absorbed INTO the run, and a census is not a
test). That comment is corrected in place.

**No threshold was touched.** P1 is unchanged; only its pin is new.

### A false positive in the finding's evidence, with the disproving crop

The finding cites "two visually confirmed confident-wrong reads". **Only one of
the two is confident-wrong.** Rendering the crops at 6x and reading them:

- `g0621 t=177 stack` — screen shows `212.90 BB` beside a timer badge `12`;
  P1-off reads 12.0. **Confirmed confident-wrong**, and it is the fixture above.
- `g0723a t=0 bet` — screen shows a chip stack sprite and a rounded box reading
  **`10 BB`**. P1-off reads 10.0, which is **correct**. This is a coverage cost,
  not a confident-wrong read.

Sampling one crop per recording from the refusal population and reading the
pixels the same way: `cwpt01 t=82` is `292.20 BB` → 292.2 correct;
`g0723b t=158` is `POT: 78 BB` → 78.0 correct; `g0711 t=284` is `12 BB` with a
chip sprite sitting on the second `B` → 12.0 correct; `g0621 t=3` is
`3.60 BB` → 3.6 correct. So P1's 219 refusals are **predominantly coverage
cost**, and one demonstrated confident-wrong read — which does not change the
disposition, because a numeral the client did not terminate with "BB" is not
provably unambiguous and the contract refuses it either way. It does change the
honest description of what P1 buys, which is why it is recorded here.

## 3. Finding 9 — the sign check enumerated two of four results. **General**

**Verdict: instance of a general defect.** `_result_contradicts_hero_net`
enumerated `"Hero wins"` and `"Villain wins"` and returned `False` for
everything else, so it could not cover the space of results — the specific gap is
`"Hero folds"`, but the shape of the bug is the enumeration.

Fixed as a table over the producer's **closed** vocabulary rather than by adding
a third `if`:

```python
_RESULT_FORBIDS_NET_SIGN = {
    "Hero wins": -1,      # a hero who won the pot cannot net negative
    "Villain wins": +1,   # a hero who lost the pot cannot net positive
    "Hero folds": +1,     # a hero who folded cannot net positive
}
```

`""` is absent deliberately: an unresolved hand asserts nothing about who won, so
no net contradicts it. A hero who bets everyone off the pot is `winner_seat == 0`
(i.e. `"Hero wins"`), so the fold branch is reached only when the hero
surrendered, and a positive net there is the same two-fields-off-one-ledger
contradiction the other two catch. `result_contradicts_hero_net` is a
`SPINE_FATAL_CODE`, so firing blocks export.

**Latent on this corpus, and pinned anyway.** All 5 development `"Hero folds"`
hands net `-0.5`, `0.0`, or unknown, so no pipeline run would have caught the
omission:

```
cwpt01 h6 -0.5 | cwpt01 h10 0.0 | g0621 h3 None | g0723b h2 None | g0723b h4 None
Hero folds with POSITIVE net: 0
```

`test_a_folded_hero_cannot_net_positive` covers the family: two positive nets
(65.2 and a 0.5 BB jitter-sized one) contradict, the three legal directions
(loss, free fold, unread net) do not, and the vocabulary set itself is asserted
so a new result value added to the producer cannot silently get no sign check.
Both mutation directions die: `S-drop-hero-folds-rule` and `S-sign-check-off`.

## 4. Finding 15 — the row-band split. **Specific**, pinned, direction verified

**Verdict: genuinely specific** — a single inline mechanics literal, not a
family. It is fixed anyway, because "unpinned but currently harmless" is the
state P1 was in.

Corpus direction of the single-band mutant (`> 9999 * max_h`) over all 18,006
retained crops, re-measured here:

```
same 17157 | value->UNKNOWN 236 | UNKNOWN->value 0 | value->DIFFERENT 0
```

The mutant only destroys coverage; it never fabricates a number, which is why the
finding is not release-blocking and why it survived a suite that pins
confident-wrong rather than coverage. (The finding reported 224 value→UNKNOWN;
this round measures 236 on the retained crop set.)

`test_the_row_band_split_keeps_a_name_row_out_of_the_numeral` uses a real cwpt01
seat panel whose top edge carries the tail of the player's NAME row above
`124.80 BB`. It asserts structurally that the fixture really does carry two glyph
rows before asserting the read — without that, a future fixture swap leaving one
row would keep the test green while exercising nothing.

## 5. Findings 3 and 14 — disclosed residuals, no action

**Finding 3 (resampled crops at the band floor).** Retained and disclosed in
`_MIN_CALIBRATED_RUN_H`'s comment, which already carries this round's
measurement: 8 confident-wrong reads across a 523,770-read AREA/LINEAR scale
sweep plus a 2,500-crop interpolation sample, every one on a RESAMPLED crop at
run height 12-13, all single-scale knife edges whose neighbouring scales read
correctly. Raising the floor to exclude the zone would delete the entire
calibrated 1272x896 native client (1,138 reads at exactly 12) against zero native
failures. The precondition — native hinted rendering — is not enforceable from
the pixels. An operator-resampled or transcoded recording is outside this bank's
calibration whatever P7 measures.

**Finding 14 (`_AFFIX_MAX_REL_H` admits values).** A documented deviation from
"retained constants may only force UNKNOWN", disclosed in the admitting direction
at both sites, and pinned in **both** directions — `K-affix-0.0` and
`K-affix-100` are each killed. No action.

## 6. Corpus effect of this round: nothing moved

18,006 amount crops, six development recordings, byte-identical detector boxes,
rebuilt end to end:

- exported hands **9 of 31**, unchanged (g0723a 3, g0621 0, g0711 1, g0715 0,
  g0723b 0, cwpt01 5);
- hand-level diffs against the pre-change baseline on `warnings`, `result`,
  `pot` and `hero_bb_won`: **none**.

That is the expected result and it is the point: two of the three repairs are
test-only, and the third closes a latent contradiction this corpus does not
contain. **No coverage was gained and none was lost.** A round that moves the
export count is not automatically better than one that does not.

**PLAN.md (finding 13) is still stale** and is still not edited from this
workstream — it is owned by the concurrent accounting workstream and reads "14 of
21 hands moved to 11 of 21" against the true 9 of 31. Note 14 §4 disclosed this;
this note repeats it because it remains true, and because the round-2 coverage
report circulated to this round (12 of 31 exports, 537 UNKNOWN, `suffix_not_bb`
222) describes the arm ENTERING the round-1 repair, not shipped code. Shipped is
9 of 31, 547 refusals, `suffix_not_bb` 221 corpus-wide / 219 in the retained
crops.

## 7. Suite and lint, as run

```
$ python -m pytest -q tests/test_ocr_readers.py tests/test_two_model_spine.py \
    tests/test_yolo_card_timeline.py tests/test_yolo_hand_timeline.py \
    tests/test_yolo_card_app_export.py tests/test_yolo_card_timeline_validation.py \
    tests/test_video_processing.py tests/test_cv_recording_regressions.py
284 passed, 1 skipped in 1.53s          # was 280/1 entering this round

$ python -m pytest -q                   # whole repo, both concurrent workstreams
1412 passed, 1 skipped in 26.49s

$ python -m ruff check cv_lab/ tests/
All checks passed!
```

Two new fixtures, both development-corpus only, both recorded in
`tests/fixtures/PROVENANCE.json` with source recording, timestamp and true value:
`stack_212_90_timer_badge_at_2062x1178.png` and
`stack_124_80_name_row_above_at_2138x1402.png`.

## 8. What this round does NOT establish

Unchanged from note 13 §9 and note 14 §5, and none of it was addressed here:

- **No locked-test gate has been evaluated** and no validation recording opened.
  Anchor confidence, reader calibration and the confidence scale remain measured
  on the same six files the constants were derived from.
- **Exported-hand correctness is still self-consistency**, not ground truth. A
  hand that survives every net may still be wrong.
- **Nineteen killed predicate mutants is a statement about the suite**, not about
  the reader's correctness. It says every acceptance-path predicate now has a
  test that fails when it is disabled; it does not say the predicates are placed
  correctly, and it cannot say anything about a predicate nobody thought to write.
- **The two adversarial clean rounds required by the phase gate are not met.**
  This round found three genuine open findings in code that had already survived
  a repair round, and one factual error in an adversary's own evidence. Neither
  is the profile of a converged surface.
- **9 of 31 is not a coverage claim.** It is what this corpus and this code
  produce today, far below the 10 sessions / 100 completed hands the corpus gate
  requires.
