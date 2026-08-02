# 17 — Whole-product adversarial rounds 1–3: the record

Recorded 2026-08-02.

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
