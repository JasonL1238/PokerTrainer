# 17 — Whole-product adversarial rounds 1–6: the record

Recorded 2026-08-02 for rounds 1 through 3 and round 5. Rounds 4 and 6, and the
coverage re-run inside round 5, were added on 2026-08-03 when they were moved out
of `PLAN.md`; until then the plan's own status table was their only record, which
is the gap this file exists to close.

`16_phase1_adversarial_rounds.md` records the fifteen rounds run against Phase 1
alone. This note records the rounds run against the *whole product* under
"Repeated adversarial-agent release loop" in `PLAN.md` — two fresh agents per
round, Adversary A white-box and Adversary B black-box, against the candidate as
a whole rather than against one phase.

**Read it in order, and read a later finding as superseding an earlier one.**
Round 2 and round 3 both landed criticals in the accounting module, and in both
cases at least one of them was *introduced by the repair to the round before*.
That sequence is the most useful thing in this file and it only reads correctly
in order.

Each round below names the commits that carry its repairs. The round attribution
comes from those commit messages; where a commit does not state a round number,
it is placed by what it repaired and that is said rather than implied.

The current status of the gate — that the stopping rule requires two consecutive
clean rounds and that the counter stands at 0 — is in `PLAN.md`. This note does
not carry status.

---

## Round 1 — the release gate could not be trusted to describe its own run

Repairs: `88e24b6`.

Two findings made `container` mode's verdict untrustworthy in opposite
directions, both from Adversary A.

1. **Container mode read a verdict out of a report it could not prove it had
   produced** (critical). The mounted report directory was reused across runs,
   so a `docker` exit 0 with no report written left a leftover report standing
   as this run's result — and its case list fed the aggregate, publishing hand
   counts and precision figures the run never measured. The directory is
   recreated per run now, and a token generated on the host is passed into the
   container and must come back inside the report.
2. **Container mode was not executing the image's code** (critical). Running
   `-m` with the repository bind-mounted at the working directory put the
   host's `poker_tracker` ahead of the image's on `sys.path`, so the image
   contributed an interpreter and nothing else — an image with a broken
   application copy passed. The corpus is mounted as data at `/corpus` and the
   working directory is `/app`, so the code under test is the image's.

The rest of the round, all in the same family of "a report that says less than
it appears to":

- Answer-key actions marked uncertain or unobservable were dropped from scoring
  silently. A key could mark all 300 money actions across 100 hands uncertain
  and the report would show a clean score with three skipped checks named. Each
  dropped action is reported now, as is an unscored partial action line.
- `amount_semantics: "unknown"` alongside an annotated amount excluded a check
  the key could actually answer. Rejected now, matching the rule that already
  rejects a declared-unobservable fact carrying a value.
- A case pinning the detector and saying nothing about the classifier read as
  fully pinned while the classifier ran whatever was installed. Enforcement
  enumerates the roles the run uses, not the roles the case happens to claim.
- A crash in the mode stage propagated out of the CLI, exiting 1 with no report
  at all — indistinguishable from an accuracy failure and leaving nothing to
  diagnose. Caught and reported as a setup failure.
- Failures during execution (a container timeout, an inner gate exiting 2) were
  classified as accuracy failures. That is the exact inversion the exit-code
  split exists to prevent.
- Unpinned models set the models stage to `failed` in fixture mode while the
  verdict said the gate passed. Fixture mode marks the stage skipped and
  certifying modes fail on it; a report may no longer claim a pass while
  carrying a failed stage, and drift between the two fails closed.
- `CRITICAL_CATEGORIES` had no callers and disagreed with the severity strings
  actually in use — a second, silently wrong source of truth.
- Out-of-order action messages named the preceding action rather than the one at
  the running-max position they were compared against.
- `directory_bytes` followed a symlinked top, so a report directory pointing at
  the video vault billed the whole corpus as this run's output.

## Round 2 — the accounting repair's own two criticals, and redaction that made things worse

Repairs: `e8c0e49`, then `ed0f403`.

### The two criticals, both from one mistake

Both came from narrowing an input to live contributions while the pot still held
live *and* dead money.

1. **A player all-in for their ante was eligible for no pot.** Pot layers derive
   their contributor set from live contributions, so a player whose entire
   commitment was a forced post appeared in no layer — while their chips sat in
   a pot they could not be declared the winner of. A hand the short stack won
   became unrecordable: the ledger refused the winner declaration outright.
   That is routine once a stack is at or below the ante. Eligibility on the
   layer holding the dead money follows the chips now. A folded ante poster is
   still ineligible: dead money reaching the pot does not buy back a claim on
   it.
2. **Split granularity regressed and certified as authoritative.** The quantum
   is the finest denomination the hand demonstrates, but the dead money was
   passed as one summed figure: four 0.25 antes became 1.00 and the hundredths
   those antes prove were destroyed. An exactly even 21.0 pot then chopped
   11.0/10.0, the extra chip going to whichever seat award `entry_order`
   happened to name. Worse, a hand whose award rows carry no amount — the app's
   own default — reconciled as authoritative with no issues, while an operator
   declaring the honest 10.5/10.5 got a permanent mismatch. Passing each dead
   contribution individually restores the even chop, identical under either
   odd-chip order.

### The rest of round 2

- **Retention deleted regression fixtures.** `regression_cases.fixture_path` and
  `report_path` were missing from `ARTIFACT_PATH_COLUMNS`, whose comment says to
  keep it exhaustive — and a regression fixture is frequently a frame or a
  recording under a managed directory. The rule that a referenced file is never
  deletable was broken outright, irreversibly.
- **The orphan-video reason claimed a dormancy it cannot measure.** The age is
  the file's mtime, not how long nothing has pointed at it, so a recording
  orphaned one second ago by a session delete was described as unused for 600
  days. It says what it actually knows now.
- **Quoting a secret made redaction strictly worse than leaving it bare.** The
  value class stopped at the first space or comma, the closing-quote
  backreference then failed, and the whole match failed silently.
  Passphrase-style secrets were fully exposed.
- **Only `Bearer` and `Basic` authorization schemes were redacted.** `Token`,
  `ApiKey`, `Digest` and the rest printed `<redacted>` beside the intact
  credential, which reads as scrubbed.
- **The issue bundle redacted its own serialized output**, but `json.dumps`
  escapes the quotes in every string field and an escaped key stops matching. A
  credential pasted in as JSON — the most likely paste — survived. Redaction
  runs over the structure now, before the encoder.
- `safe_error_message` sliced from the *end* of the string for `limit <= 0`.

### Closing the round at the boundary rather than the call site (`ed0f403`)

- **Retention compared paths as strings**, so on a case-insensitive filesystem a
  recording stored as `Session.MOV` and recorded as `session.mov` looked like an
  orphan — a file the database still points at, offered for deletion. Path
  identity is `(st_dev, st_ino)` where the file exists and a normalized
  case-folded key where it does not, through one helper every comparison in the
  module goes through. The textual fallback over-matches on purpose: keeping an
  orphan costs disk, deleting a live file costs the recording.
- **An audit was treated as an authorization.** A job completing between the
  audit and the sweep made the database start pointing at a file already
  classified as garbage, and the sweep unlinked it anyway. The reference check
  travels with the audit and is re-confirmed immediately before each unlink,
  through the same code path, against a database re-read whenever it changed
  underneath.
- **A zero retention window meant "expire everything now"** and was reachable by
  an unset environment variable. Refused on every construction path and on every
  read, naming the variable. Operators who mean it say `--purge-now`.
- **Job failure messages were scrubbed by whichever writer remembered to**; the
  solver worker did not. Redaction moved down to the single write boundary, so
  no future writer can store a credential by forgetting.
- **Pot layers were labelled by index**, which called a dead-money layer a side
  pot — a false statement in a tool people study from. They are labelled by what
  created them. The property suite written to catch the ante-refund bug could
  not generate an ante; it now generates antes, blinds, straddles, dead blinds
  and players all-in for a forced post.
- **A failed import showed its last progress reading**, which reads as "most of
  my hands got in". Terminal states report what was actually committed, and the
  log file that always existed is finally shown.

## Round 3 — pot layering, three more criticals in the same module

Repairs: `3c3144e`, `5e1cec8`, `ba3cb2d`. These commits do not carry a round
number; they are placed here because each repairs a critical in the module round
2 had just repaired, and `3c3144e` says so in as many words: "This is the third
critical in this module and the second introduced by a repair to the previous
one."

### A seat all-in on forced posts alone contested the whole first live layer (`3c3144e`)

Pot levels came from live contributions only, so a seat whose entire commitment
was an ante or a dead blind produced no level of its own and no layer was ever
capped at what they covered. Three-handed, ante 1, a 1 BB stack against two
players who bet 10: one pot of 23, the short stack recorded as winning 22
instead of 2.

Every gate passed. Chips were conserved, so the hand was balanced; no legality
rule looks at eligibility, so it was legal; the declared-award cross-check
compares the operator's award against the product's own derived payout, so
agreeing with the number on screen is what made it reconcile. It reached the
narrowest population the product has, labelled derived from the reconciled
ledger, and printed 2200 bb/100. The only truthful declaration an operator could
have entered was rejected as referencing a pot that did not exist.

Levels are built at every dead level as well as every live one now, and a
randomized sweep of 3,337 settled hands with forced posts finds no seat paid more
than it could match, against 197 violations before.

The eligibility override that caused it was added in round 2 to stop a
short-stacked hand being unrecordable — it turned a loud refusal into a quiet
eleven-fold error, which is the worse of the two failures.

Separately in the same commit: **an unfinished betting line settled with an
invented fold.** A wager nobody answered was refunded as though it had been
folded to, and the hand derived a hero result from a line that never closed.
Manual entry refuses it before any write and a second guard catches every
non-manual writer, while a genuine fold to a bet still refunds and reconciles as
before.

### A stale award raised out of the reconciler, and a weight was judged after rendering (`5e1cec8`)

- **An award row naming a pot index this build no longer derives** raised out of
  `reconcile_persisted_hand`, which its callers outside `app.py` do not catch.
  Pot indices are ordinals into a derived structure, so any change to layering
  can strand them. The hand is rebuilt with the unusable awards withdrawn now,
  and the mismatch is reported as a correction naming the stale claim and the
  layer count the hand actually has. Nothing is renumbered on the operator's
  behalf and no schema changed: a record that no longer matches what the code
  derives is a question for the operator, not something to quietly rewrite. A
  hand that is impossible even without its awards still raises, because the
  award-free rebuild is what tells the two apart.
- **A range weight below printing precision was compared against zero after
  being formatted for display**, so 5e-7 rounded to 0.0, slipped the guard, and
  was emitted as a token weighted zero — the silent drop the guard exists to
  prevent. The value is validated and the rendering is separate, driven by one
  constant so the two cannot drift.
- **A reconciliation save was not idempotent**: saving twice moved timestamps
  and silently dropped stale warnings, so the operator could not tell settled
  from one step along. A second identical save changes nothing now.
- **A result was labelled derived-from-the-ledger on the strength of the write
  guard** that once permitted a substitution, rather than on what the value is.
  The Overview consumer was corrected in an earlier round; the helper was not,
  and it kept the right name and the wrong logic for whoever reached for it
  next. Deleted rather than left beside the correct path.
- Found while verifying the above: **a line that stops while a seat still owes
  chips reconciled as authoritative** whenever no refund existed to key on —
  blinds 1 and 2, button calls 2, small blind never acts. Two seats tie at the
  top, nothing is uncalled, and the previous guard never ran. Closure is
  measured off the ledger directly now rather than inferred from a refund.

### Unequal dead money manufactured a side pot nobody was all-in for (`ba3cb2d`)

Layers were sliced at every distinct total commitment. A 5 ante against a 3 dead
blind, with two seats owing nothing dead and all four calling 20, derived a main
pot of 80 and a side pot of 8 that the two undead seats could not win. Declaring
the truth was refused, leaving pot 1 undeclared left the hand unbalanced, and the
settlement editor requires a winner for every derived pot — so the only
declaration the product accepted was the wrong one, and it reconciled as
authoritative with the hero's result short by the dead money.

The rule is the **live line**. A seat is capped only when it left less live money
in than a seat still in the hand: it declined or could not answer chips an
opponent actually risked. Nobody can decline a forced post, so unequal dead money
never opens a layer. Being all-in does not either, if the all-in covered the
wager. A cut is drawn only where somebody was short, and once drawn it applies to
every seat by that seat's own total, forced posts included — which is what still
caps a seat all-in for nothing but its ante.

Two more silent overpayments fell out of getting there. A level only one seat
reached was merged downward into a layer a capped seat could win, paying a
one-chip ante-only all-in four chips on a total the table matched three of. And
exempting the seats a cut would drop, rather than applying the cut uniformly, let
a seat that covered the wager win an ante that a shorter seat was refused.

A 150,000-hand sweep finds no seat paid more than the table matched of its own
commitment. The same sweep against the previous code finds 5,286.

**The honest verdict on this module is not-yet-caught rather than correct.** It
rests on a modelling choice no rulebook settles cleanly: "dead money is owed to
the table" and "you win from each opponent only what they matched of your
commitment" contradict each other whenever forced posts are unequal. The two
questions are scoped apart deliberately, and the argument is written down in the
module where the next reader will find it.

---

## What these three rounds establish

Three rounds, three sets of criticals, and in the accounting module a repair
introduced the next round's critical twice. The pattern Phase 1's fifteen rounds
recorded — a repair stated as an argument but applied to the shape that was
demonstrated — recurs here at a coarser grain: round 2 fixed eligibility for the
hand it was shown, round 3 found the layering rule underneath it was wrong for a
whole family of hands.

The counter is 0 of the required 2 and every round to date has found a critical.
`PLAN.md` carries that status; this note carries only what was found.

---

## Round 4 — two criticals inside the blind-structure repair

Repairs: `b58b7e3`.

Round 3 closed with the ledger deriving the amount to call from the largest
observed contribution. Blinds 5/10 with the big blind all-in for 4 made the small
blind's 5 the largest post anybody could see, so a truthful call of 10 was
reported illegal, the operator obeyed the message, entered 5, and the hand
reconciled with a 14-chip pot where the truth is 24. The product's own error
routed them there. The repair made the structure a declared input — a fact about
the room the action line cannot demonstrate, like the rake policy and the dead
money — floored the preflop amount to call at the largest structural forced bet,
and refused an undeclared structure rather than inferring one.

Both criticals were in that repair rather than in what it replaced.

**Shortness was judged at the instant a row was reduced** rather than against the
seat's final commitment, so moving an ante below a blind flipped `is_legal` with
identical chips: a seat's ante recorded below its own blind silently turned a
blocked hand into a reconciled one, around a pot 10 chips short.

**A transposed structure reached disk through `model_copy`**, which skips
validators, and the degraded reader then probed the columns one at a time and
kept the smaller half — a floor of 5 covering a big blind all-in for 4. The
salvage produced a structure that was *valid*, and whose floor covered the very
post the declaration existed to expose.

A third finding is real and only partly repaired, and is stated in `PLAN.md` as a
limit rather than as a guarantee: the refusal keys on action kind, so a short
blind booked as an `all-in` escapes it entirely. The CV spine books any seat whose
stack reads zero as a plain all-in and drops the markers that would identify the
post as forced, so on that path no reducer-level rule can tell a short blind from
an ordinary short shove. Changing the spine's classification is a corpus change
against a spine at zero errors per hand, so the limit is documented where it is
claimed.

The evidence that the declaration moves no chips is worth keeping: 20,000 hands
built under both builds differ in nothing but `is_legal` — 1,755 demotions and no
promotions — under any structure, valid or absurd. That is what made the schema 19
migration safe.

---

## Round 5 — the live-level pot model

The operator replaced the layering rule outright: boundaries are cut at distinct
LIVE contribution levels after refunds; all dead money goes whole into the lowest
layer; a seat is eligible for a layer if its own live contribution reaches that
layer's level, and every unfolded seat that put any chip up contests the main
pot. Four worked examples were handed down as acceptance criteria. All four are
reproduced exactly by the shipped reducer and by an independently written
harness: (a) big-blind ante 58/8 with the poster net +32, (b) ante-only seat 7/14
net 0, (c) 3/20 net +2, (d) one pot of 88 any of the four may win.

### The critical: the money classifier was still keyed on the action kind

`_build_pots` was faithful. What it was HANDED was not. Liveness was decided by

    action.kind in _BETTING_COMMITMENT_KINDS and not (
        action.kind == "post_blind" and not action.is_live_post
    )

so a forced post booked as `all-in` carrying `forced_bet_type="ante"` — the shape
the hand editor writes from its two selectboxes and the shape a post which took
its poster's last chip normally has — was counted as chosen live money. The
module had already ruled on this exact row twice, 370 lines further down, for the
blind-structure refusal; the money classifier was the one place not widened.

Under the OLD total-commitment layering the misclassification was nearly
harmless, because moving a chip between the live and dead columns left the
boundary where it was. Under the live-level model live money is the only thing
that opens a boundary and the only thing that decides eligibility above the main
pot, so the same relabel moved chips between layers: worked example (c) paid its
ante-only seat +4 instead of +2, and a 10-ante hand paid a 10-chip stack 30 chips
its three opponents never wagered against it — settled, balanced, legal,
warning-free, `is_authoritative=True`, `status="reconciled"`.

Repaired by `_is_live_money`, which asks `_is_forced_post` / `_is_live_structural_post`
rather than the kind, plus the mirror of the same test in `build_ledger_from_records`
(the raise-to baseline, where a relabelled ante made "raise to 40" mean two
different chip amounts). A row the recording NAMES as a dead forced bet is dead
even when the separate post-status field was left at its live default.

The property suite could not see any of this: its generator emitted no
`forced_bet_type` at all and classified its own live/dead bookkeeping from the
kind, so the input family was unreachable. With the generator widened, four
independent properties fail against the pre-repair reducer.

### Two findings NOT repaired, and why

Two adversaries independently reproduced the same thing: a seat all-in for less
than *another* seat's forced post is paid that post in full. Antes of 100 with a
40-chip stack short of its own ante pays it 540 where five opponents covered 40
apiece; a button ante of 200 against a one-chip all-in pays that seat 204. Both
reproduce end to end as authoritative and both are regressions against the old
total-commitment layering, which got them right.

Neither is a deviation from the specification. They are rule 2 — "ALL dead money
goes entirely into the LOWEST layer" — doing exactly what it says, in a
configuration none of the four worked examples reaches: in (a) and (b) the
unmatched post is the short seat's OWN, in (c) each opponent's ante exactly equals
the short seat's whole commitment, in (d) nobody is short. Rules 2 and 4 as
written also make (a) and (d) impossible any other way: (a) requires the big
blind's unmatched 10 ante to sit in a main pot the two DEEP seats may win, so
unmatched dead money demonstrably is shared.

A model does exist that satisfies all four worked examples and both hands: cap
each contributor's dead chips, for placement in the lowest layer, at the smallest
TOTAL commitment among that layer's eligible seats, and push the excess up. It
opens no new boundary, so it is compatible with rule 1 and with (d)'s single pot.
It is not implemented here. Four of the five criticals in this module were
introduced by an in-session repair that argued its way past the previous model,
and this is the fifth invitation to do it again. It needs an operator ruling.

In the meantime the hand is not published. No chip figure changes; a warning
names the seat, the poster and both numbers, and `_cross_check` folds ledger
warnings into its issues, so the hand is `needs_correction` and never
authoritative. A wrong prediction that is visibly rejected is a coverage
limitation.

### What the replacement invariant is and is not

`_model_payout_cap` states the payout in the spec's own terms and is genuinely
independent of `_build_pots` for AMOUNTS: eight mutations of the reducer were
tried and the six that are not semantic no-ops were all caught, including a
revert to the round-4 total-commitment cut. It was NOT independent for
ELIGIBILITY — it read the main pot's eligible set off the ledger and then asserted
the cap for that set, so widening the main pot widened the cap with it. Rule 3 is
now derived in the suite and asserted, not borrowed.

Two limits remain. The generator still cannot produce an unfolded seat that put
no chip up, so the eligibility assertion is not exercised by fuzzing (one
hand-written test covers it). And `_model_payout_cap` encodes rule 2 as written,
so it actively REJECTS the capped alternative above — it is evidence that the
code matches the model, never that the model is right. That distinction is the
whole reason this module has produced five consecutive criticals.

### The coverage re-run on `bc597c5`

Repairs: `cdb0e65`, `1f8f01d`.

The round was re-run against the frozen tree at `bc597c5` with a different
mandate: cover the half of the product the accounting-focused rounds had left
alone. It is counted inside round 5 rather than as a round of its own because it
attacked the same candidate. It found a critical and four highs, none of them in
the pot model, and all five were independently reproduced before repair.

- **The coaching grounding check was written, tested, and never called by the
  product.** Its only non-test caller was the offline evaluator; `app.py` never
  imported the module. A provider's answer went from the network into storage with
  no comparison against the prompt that produced it, under a green banner
  confirming the outgoing *prompt* was safe — which an operator reads as "this
  answer was checked" — and then promoted the hand to reviewed. A coach could
  invent a card that was never dealt and the hand would be marked studied. The
  check moved into the one constructor that turns provider text into a persistable
  row, so a future caller inherits it rather than having to remember it. A rejected
  answer is kept verbatim and marked stale.
- **Container mode certified coverage it had not executed.** The docker argv
  hard-codes the fixture gate while the coverage statement was derived from the
  *requested* mode, so a container run claimed video decoding with the pinned
  weights and end-to-end reconstruction of every corpus recording having done
  neither. This is the third instance of one pattern — the gate once reported zero
  errors on a run that measured nothing, and the perf harness once reported
  untaken measurements as zero — so certification is computed from a record of the
  stages that actually ran.
- **Retention would delete live CV timelines.** No database column names one, so
  retention called them orphans from the moment they were written, while the
  recovery drill and the backup inventory both know a missing timeline permanently
  blocks every later import for that job, and nothing deletes a `processing_jobs`
  row.
- **The schema-20 migration staled the wrong population.** It keyed on the
  declared external dead-money column while the rule change that moves chips
  applies to *recorded* posts, so hands with a recorded dead blind or an ante
  republished a different hero result under text written about the old one.
- **A `forced_bet_type` on a `call` row exempted it from the legality check that
  reads `to_call`.** Tagging an ordinary call a straddle turned an illegal action
  line clean, and tagging it a dead blind moved chips as well.

The repair to that last one carried its own defect and `1f8f01d` reverses it. The
ruling had been that the label decides what a row is; that reading let a label
take a row *out* of the pool its kind puts it in. Tagging the only ante row a
straddle, a big blind or a dead blind emptied the ante pool, which switched off
the undeclared-ante-mode refusal and turned a hand the product exists to refuse
into one it accepted, with no chip moving to make it visible — reachable by one
mis-click on a selectbox drawn beside every row. The rule now is that the kind
decides what a row is and the label may only refine which forced post it is;
where the two disagree the hand is refused and derives byte-identically to the
same row with the field cleared. Seven of the fifty-six kind/label pairs are newly
refused. The migration docstring asserting the old reading was corrected — its own
query already counted by kind, so the prose had been contradicting the SQL beside
it.

One hole was left open and named in the tests so it could not drift: a live blind
typed as a *dead* blind still silenced the structure refusal and moved chips,
where no blind structure is declared — which is every hand migrated from schema
19. Round 6 closed it by treating the forced-bet name and the post status as two
statements of one fact, so a row where they disagree is reported rather than
resolved. See round 6 below.

Ten lesser findings from the same pass were triaged as disclosure-only. Each was
re-reproduced against `cdb0e65` before being recorded, and each is a code defect
rather than a documentation one; what was done about them was documentation, which
is not closing them. All ten — `B2` through `B5` and `A2-3` through `A2-9` — are
still open and are listed in `PLAN.md` under "Known open items", which is where an
open finding belongs. Round 6 closed none of them.

---

## Round 6 — the product's claims about itself outrunning what it checks

Repairs: landed in the round-6 repair pass; see the working tree and the commits
that follow `1f8f01d`.

Six findings, four confirmed as high. None was in code an earlier round had
already repaired, and five of the six were in surfaces no earlier round had
examined at all. That is the useful thing about this round: after five rounds
spent almost entirely inside pot accounting and the release gate's reporting, the
defects moved to wherever nobody had looked, and they shared a shape.

The shape is a surface asserting something about itself that it does not check.
Every one of them passes the suite while saying something false on screen or in a
report.

- **`poker_tracker/validation/corpus.py` — the locked-test seal was a
  declaration.** The corpus report answers "has the locked test been used for
  tuning" from `used_for_tuning`, a boolean each case sets about itself, and
  `split_integrity` compared *normalized logical names* between splits. The
  manifest already carries a SHA-256 for every recording and the check never
  looked at it, so the same recording copied into development under a second
  filename left the seal reading clean. Repaired by comparing digests as well as
  names, requiring a declared digest for every locked case under
  `--require-recordings`, and naming in the report what content equality cannot
  answer — a re-encoded or trimmed copy of a locked recording, and adjacent
  segments captured from the same session — so a clean seal reads as the narrow
  finding it is.
- **`cv_lab/scripts/pipeline/run_two_model_pipeline.py::_sample_times` — the
  emitted series was shifted, and its docstring called it the honest record.**
  Seeking to `t` returns the first frame at or after `t`, which on a
  variable-rate screen recording can be seconds later; the sampler stamped that
  picture with `t`, which moves it backwards in time and closes the hole it came
  out of. The spine measures `prior_gap_s` as the difference between consecutive
  state times, so a pair straddling a real unobserved stretch reported as one
  ordinary interval apart, which disarms every refusal keyed on coverage —
  `mid_hand_coverage_gap`, the roster-shrink and continuous-presence refusals —
  and a hand nobody watched exports as complete. This is the one finding of the
  six in a surface an earlier round had touched: `d835ad6` bounded the same
  sampler and wrote the docstring that was wrong.
- **`poker_tracker/ui/frame_extraction.py` — the same shift on the diagnostic
  frame path**, in the stored `timestamp_seconds` and in the filename. Found
  beside it: a failed `cv2.imwrite` was counted as an extracted frame, so a row
  pointed at a file that is not there and read downstream as a frame the run had
  kept.
- **`poker_tracker/math/icm.py::icm_risk_premium` — a metric asserted without the
  input it needs.** It computed Hero's equity after *deleting* the risked chips
  from the table. ICM equity is a function of stack shares, so removing the chips
  shrinks the denominator and inflates Hero's post-loss share: on
  `[5000, 3000, 2000]` paying `[50, 30, 20]` it returned 1.57 where every outcome
  the tournament can produce lies between 2.73 and 2.96. The premium genuinely
  depends on which opponent wins the chips, which is the input the single number
  did not have. It transfers the chips now, defaults to the largest single-winner
  premium (conservative, and a figure some real outcome produces, which an average
  is not), and exposes the per-opponent span. Two limits are stated rather than
  modelled: a multiway split can cost Hero more than any single winner, and an
  opponent shorter than the risked amount describes a run of pots rather than one
  confrontation.
- **`poker_tracker/math/equity.py` — the method claim and the sampler's
  termination.** The class docstring said postflop uses exact enumeration; multiway
  is Monte-Carlo at every street. And multiway sampling rejects deals that reuse a
  card, so a range set card removal makes undealable had no bound on how long it
  would try. It works to an attempt budget now and returns `no_valid_combos` with
  a reason instead of an equity figure.
- **`app.py` — a blocker naming a clearing action the product would not draw.**
  Every control that clears a trust blocker is hosted by
  `render_validation_edit_and_approve`, whose other caller hangs off a completed
  reconstruction job whose timeline is still on disk. A manually entered hand, or
  a reconstructed one whose recording was later deleted, therefore read blockers
  naming a screen no page in the product offered — including the ante-mode
  refusal that schema 20 raises on migrated hands. A route from the hand itself
  now reaches the same workspace. This is the sixth distinct way the standing rule
  "a blocker never names an action the product cannot perform" has been broken,
  and the fifth and sixth were both cases where the guard test passes because the
  named control *is drawn* somewhere.

Three more repairs landed in the same pass, each found while attacking the ground
around one of the six.

- **A hole in the forced-post refusal, and only the half of it that shows.** A
  live blind typed as a *dead*
  blind — by the row's post status, or by naming it `dead_blind` — silenced the
  blind-structure refusal and moved chips where no structure is declared, which is
  every hand migrated from schema 19. A big blind all-in for 4 with blinds
  undeclared is refused and lays out 12/2; one selectbox on the row the warning
  names, in the panel that warning auto-opens, moved 8 chips of a 14-chip pot and
  presented the hand as study-ready with zero blockers. The name and the status
  are two statements of one fact, so a row where they disagree is reported rather
  than resolved. That is `1f8f01d`'s rule for the kind-versus-label axis, extended
  to the liveness axis. It closes only what a contradiction reveals: a row that
  marks the post dead by ONE field and leaves the other unstated says nothing that
  can be checked, and still moves the same chips into the same wrong layers with
  no blocker raised. `PLAN.md` item 6 carries it.
- **A cancelled reconstruction left its partial artifacts on disk.** Cancellation
  sends SIGTERM to the worker's process group and CPython's default disposition
  exits without unwinding, so the `finally` that `_discard_partial_artifacts`
  names in its own docstring never ran on that path: a partial timeline and one
  JPEG per sampled second survived. Best effort by design — two seconds before
  SIGKILL is ample for the unlink and not a guarantee.
- **`ensure_hand_imported` accepted a non-completed job's timeline.** The gate
  lived in `app.py`, which filters the review surface to completed jobs, while the
  recovery scans and the draft path call the service directly. A timeline covering
  part of a recording reads exactly like a whole one.

What the round says about the guard tests is the same thing the accounting rounds
said about the accounting gates: they are not independent of the code they check.
A test that asserts a named control exists cannot see that no path reaches it. A
report field derived from the mode string cannot see that the mode did less. A
docstring is not a check at all.
