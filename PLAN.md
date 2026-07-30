# PokerTrainer Master Completion Plan

This is the canonical current-status, implementation, validation, and release
plan for PokerTrainer. It replaces short feature wish lists with an executable
program for completing the first trustworthy local/private beta.

The plan intentionally includes all known remaining work: product behavior,
computer vision, OCR, hand reconstruction, accounting, correction, coaching,
TexasSolver integration, persistence, security, recovery, performance,
container readiness, documentation, acceptance testing, and repeated
adversarial review.

## Product boundary

PokerTrainer analyzes saved recordings and completed hands after play.

It will never:

- capture a live poker table;
- watch or analyze a hand while it is being played;
- display a poker-client overlay;
- recommend an action for a current hand;
- expose an API intended for real-time poker assistance.

Every CV, solver, equity, and coaching workflow must operate on a recording or
hand that the user has already finished playing.

## First-release claim

The first release is a local-first, single-user, private beta for saved ClubWPT
desktop recordings and manually entered completed hands.

The release claim is deliberately narrower than “universal poker video
reconstruction”:

- only explicitly validated ClubWPT layouts and capture geometries are certified;
- every CV import begins as a draft;
- uncertain, partial, inconsistent, or unsupported evidence is rejected or
  routed to correction instead of being silently trusted;
- manual confirmation and accounting reconciliation remain required before a
  reconstructed hand can be treated as study-ready;
- coaching and TexasSolver output are evidence attached to confirmed completed
  hands, not substitutes for correct source facts.

The application currently runs locally. There is no active hosted deployment.
The repository must remain provider-neutral and continuously container-ready,
but provisioning or deploying a cloud host is not part of this plan unless the
user explicitly requests it later.

## Verified implementation baseline

As of July 28, 2026, the repository contains:

- a seven-workflow Streamlit application: Overview, Sessions, Hands, Study,
  Insights, Import, and Settings;
- SQLite persistence for sessions, hands, players, actions, settlements,
  coaching, corrections, issues, videos, processing jobs, extracted frames,
  ROI profiles, review evidence, range profiles, solver runs, and explicit hand
  completion status plus its versioned reconstruction evidence (schema 13);
- schema migration protection, WAL operation, JSON import/export version 5,
  pinned pre-migration and rotating pre-import backups, and a read-only
  data-health/restore audit;
- derived, never-persisted study readiness with per-blocker reasons and clearing
  actions, enforced at the store and behind a single guarded UI writer;
- manual entry and editing for sessions, hands, players, actions, settlements,
  and review state;
- poker math for cards, ranges, equity, pot odds, EV, ICM, analytics, and
  authoritative action-ledger accounting;
- post-session Theory and Exploit coaching through configured providers, with
  prompt safety checks and retained provider/model/source evidence;
- optional TexasSolver-backed heads-up postflop cash-game analysis with
  eligibility checks, exact ranges, retained assumptions, convergence, logs,
  strategy JSON, stale-result handling, and solver-grounded explanation checks;
- offline video upload, metadata validation, frame extraction, detached CV
  jobs, heartbeat/recovery behavior, timelines, and draft session import;
- an eight-region reconstruction model plus card recognition/OCR readers,
  temporal reconstruction, labeling tools, hard-example queues, and evaluation
  scripts;
- a guided three-mode Study workflow (Replay, Fix & confirm, Analyze) that keeps
  hand selection compact, lets any saved action drive one shared reconstructed
  table replay, groups failing trust checks into concrete fix steps, hides
  advanced corrections until opened, and includes inline TexasSolver setup/use
  guidance while retaining before/after correction history and staling derived
  coaching/solver evidence;
- a persistent cross-session issue inbox that freezes debugging evidence and
  retains resolution notes;
- authenticated local/container operation, a non-root Docker runtime, pinned
  CV dependencies, healthchecks, and CI;
- 1509 passing tests with one intentionally skipped test at the latest inventory
  (see "Verification record" below, which is the authoritative count and which
  this bullet must be updated to match), with Ruff and the configured MyPy target
  green.

These checks establish a strong component baseline. They do not yet prove that
real recordings reconstruct reliably across a representative held-out set.

## Current reconstruction truth

The July 23 recording is sufficient for continued closed-loop development, not
for unattended or release-trusted import.

Observed evidence:

- the recording contains seven completed hands plus an unfinished eighth;
- the default export recovered six of seven completed hands;
- it also included the unfinished eighth hand;
- representative hero and board cards were visually correct;
- review of a recording that began mid-preflop exposed a roster defect: players
  who folded before frame zero were dropped, shifting BTN/SB/BB labels and
  same-frame action order; reconstruction now recovers stable opening occupancy,
  anchors labels to the dealer button, and retains those seats as pre-observation
  folds (see `cv_lab/notes/15_mid_hand_button_position_repair.md`);
- reconstructed action ledgers still contained non-authoritative and
  overcommit cases.

Consequences:

- the largest remaining product risk is recording reconstruction, not solver or
  coaching presentation;
- one recording cannot establish generalization;
- a green unit suite cannot replace real-model, real-video acceptance tests;
- until the held-out gates in this plan pass, reconstructed hands remain
  `needs_correction` drafts.

## Program rules

- Build each phase incrementally and keep the normal test suite green.
- Preserve existing public response shapes; new fields must be additive or
  introduced through a versioned format.
- Keep database, CV, OCR, reconstruction, accounting, solver, coaching, and UI
  responsibilities separated.
- Store raw videos and large validation artifacts as files, never SQL blobs.
- Store durable structured application data in SQLite first.
- Never tune a model, threshold, or reconstruction heuristic against the locked
  test split.
- Every confirmed defect must gain the smallest permanent regression test that
  reproduces it.
- No issue may be marked resolved merely because the output looks better; its
  regression must pass.
- An unavailable dependency, skipped full-video run, failed agent setup, or
  missing validation artifact is not a passing result.

# Completion program

## Phase 0 — Consolidate and freeze the beta baseline

### Work

- Inventory and preserve all current user-owned changes before touching
  overlapping files.
- Consolidate accounting, settlement, solver, session-library, reconstruction
  review, correction, issue-queue, maintenance, and deployment-readiness work.
- Remove obsolete roadmap claims and keep later CV research findings out of the
  current product contract.
- Confirm `AGENTS.md` and `CLAUDE.md` remain byte-for-byte identical.
- Run the full unit/integration/UI suite, Ruff, configured MyPy, and
  `git diff --check`.
- Run the application against:
  - a fresh database;
  - a migrated historical database;
  - the existing local database without destructive writes.
- Record the exact Python, FFmpeg, model, dependency, OS, and architecture
  versions used for the baseline.
- Identify intentionally skipped tests and either justify them in the release
  report or make them runnable.

### Exit gate

- The component suite is green.
- The application starts from a clean database.
- Existing structured data opens without loss.
- Documentation describes the code that actually exists.
- No unexplained test skip or untracked release-critical artifact remains.

## Phase 1 — Make hand completion and study readiness explicit

**Status: implementation complete; adversarial certification not met.** Schema
version 13 and JSON export version 5 are landed, readiness is derived in
`poker_tracker/services/study_readiness.py`, and every UI path that can mark a
hand reviewed is routed through one guarded writer. Both exit-gate bullets below
are satisfied and re-verified against the code that exists.

What is **not** satisfied is this plan's own stopping rule (see "Repetition and
stopping rule"). Fifteen adversarial rounds have been run against Phase 1 and
every one of them produced at least one valid blocking finding. Rounds 10 through
15 are recorded in their own sections below; the summary that follows covers
rounds 7 through 9 — including, in round 9,
three independent demonstrations that round 8's chip-denomination fix closed only
the exact hand shape it was demonstrated on, so as soon as more seats contributed
than won, the same declared field still doubled the derived hero result and landed
a fabricated `hands.hero_bb_won` as reconciled and study-ready, from the
settlement editor and from one `import_session` call; a rake allocation that could
charge a pot more than that pot contained, paying its winner a negative amount
that `is_balanced` certified and that no operator could ever declare away; and
`STALE_COACHING_EVIDENCE` naming "re-run coaching" as its only clearing action
when the store had no discard writer at all and every imported hand starts staled,
so an operator with no configured provider held a permanently unstudyable hand —
and, in round 8, a chip denomination whose second, undocumented job was to divide
a chopped pot,
so a display setting with no upper bound redirected the whole pot to one seat and
made a fabricated hero result reconcile by the settlement editor and by a single
`import_session` call; an acknowledged rake attestation that survived the rake
going from the whole pot to nothing, because the policy tuple it was compared
against left out the field that rounds it; a declared-award audit snapshot that
excluded the odd-chip order column deciding who received the chips; and a
confirmation checkbox reading "I have read the evidence above" over a page that
rendered none of the nine evidence fields it was asking about — and, in round 7,
a rake policy that moved the *derived* side of the hero cross-check instead of
widening its tolerance (so comparing exactly bought nothing, and any fabricated
hero result could be made to reconcile), an import payload that declared its own
coaching staleness, a recorded rake and net pot that an import payload could
restate by a quarter of the pot because only the settlement editor ever rewrites
them, a documented 30-second busy timeout that `PRAGMA journal_mode = WAL` does
not honour at all, and a read-only backup audit that wrote sidecars into the
operator's backup directory and failed intact WAL backups on an archival mount.
Each finding reset the counter, so the
consecutive-clean-round count stands at **0 of the required 2**. Phase 1 is
therefore done as implementation and undone as certification: it must not be
counted toward the release claim until two consecutive rounds come back clean
with no code, schema, or dependency change in between.

### Close-out record

A close-out pass was run after round 9. It is **not** an adversarial round and
does not advance the counter: it re-attacked already-repaired findings rather
than searching for new ones, and the stopping rule requires *fresh* agents with
varied attack prompts. What it establishes is that round 9's repairs are real
and that the earlier rounds' repairs did not decay under them.

Re-attacked on the working tree, each one landing nothing:

- the settlement `Chip unit` swept across `[0.001 … 1e6]` on a three-seat
  contribution against a two-way chop — the shape round 8's fix could not see —
  produced one identical derived hero payout at every value and reconciled a
  fabricated `hands.hero_bb_won` at none of them, while a genuine 21-chip pot
  still chopped 11/10;
- an `import_session` payload declaring its own `rake_rounding_unit`,
  `rake_rate`, `rake_cap`, `rake_amount`, `net_pot`, `completion_status` and
  `review_status: reviewed` landed `unreviewed`, non-authoritative and not
  study-ready, on `COMPLETION_NOT_COMPLETE`, `ACCOUNTING_NOT_AUTHORITATIVE` and
  `UNRESOLVED_SOURCE_WARNING`;
- 2000 chips of declared dead money against a 24-chip action line wrote
  `declared_unobserved_chips` into the evidence and still failed the accounting
  gate rather than clearing it;
- `INVALID_HERO_OR_BOARD_CARDS` fires on the reachable hand-edited-column path,
  and both retained defensive branches still return their message when driven
  through `Hand.model_construct`;
- the v13 migration leaves a manual hand's `review_status` untouched at
  `reviewed`, `unreviewed` and `needs_correction` alike.

Four mutants were introduced and reverted to prove those are pinned rather than
merely passing: promoting manual hands inside `_migrate_to_v13` is killed by 3
tests, feeding the declared unit back into `_split_pot` is killed by 55,
restoring the `min(rounding_unit, rake, gross − rake) / 2` slack on the recorded
rake and net pot is killed by 6 across rounds 5, 6 and 7, and dropping
`backup_database`'s `wal_checkpoint` / `journal_mode=DELETE` pair is killed by
`test_a_snapshot_is_self_contained_and_read_only_verifiable` — that last one
specifically because its seed database is in WAL mode, so the assertion is a real
check and not the tautology the round-4 migration test turned out to be. Every
mutant was reverted and the full suite, Ruff and MyPy re-run green afterwards.

The two recurring themes were re-probed directly: a hand promoted to `reviewed`
and then given a contradicting `pot_size` through `update_hand_facts` came back
`needs_correction` / `uncertain` with `source_facts_corrected` in its evidence
and all three blockers restored, and both named stale-evidence escape hatches
(`db.discard_stale_coaching`, `db.delete_solver_run`) exist as writers.

### Structural repair after the close-out (the dependence rule)

The close-out above established that rounds 7-9's repairs held under re-attack.
It did not change the fact that each of those repairs closed one *shape* of one
defect. Rounds 7, 8 and 9 all landed findings in the same family — a declared
settlement input moving the DERIVED side of a cross-check — and each repair
taught the disclosure gate about one more field, so the next round found one more
field combination. Blocking findings oscillated between 2 and 5 per round and did
not converge.

That mechanism has now been replaced rather than extended. Study readiness is
gated by `ACCOUNTING_ASSUMPTION_DEPENDENT`, derived per read from a dual
reconciliation with no field list in it (see "Assumption-dependent
reconciliation"), and the writer-side codes are raised by the same measurement.
Six defects the field-list design could not see are pinned in
`tests/test_phase1_assumption_dependence.py`, each of which left the hand
study-ready with an EMPTY blocker tuple beforehand: a settlement row written by a
plain `UPDATE`; an attestation inherited by the same policy taking 8000x more
chips off a corrected action line; a forged import payload; three chopped-pot
shapes including the ones rounds 8 and 9 repaired one at a time; declared dead
money at a zero rake rate; and a rake whose waiver flipped because the BOARD was
corrected, with no settlement write of any kind. Three combinations the old gate
disclosed while nothing rested on them are now silent, which matters for the same
reason: an operator trained to click Acknowledge past meaningless disclosures is
an operator who clicks past the one that matters.

This is a code change, so it resets the stopping rule again: the counter stays at
**0 of 2**. The prior on a clean round should be revised upward relative to
rounds 7-9 only to the extent that a rule with no field list has no per-field hole
left to find — which is a claim about this repair, not evidence about it.

### Round 10, and what it found

Round 10 has now run. It found no per-field hole in the dependence rule, which is
the first time in ten rounds that the recurring defect family did not recur. It
found four holes *around* the rule, and each was repaired as a family rather than
as the shape demonstrated:

1. **The attestation shared the pipeline channel** (critical). Written into
   `warning_codes` / `acknowledged_codes` for auditability, it was offered to the
   generic one-click **Acknowledge**, reachable both by an ordinary export/import
   round trip of a legitimately attested hand and by a payload that simply listed
   the measured code. Repaired by giving the attestation its own channel and
   enforcing the separation in `parse_completion_evidence`, which every reader
   goes through, in both directions.
2. **The measurement asked about the verdict, not the figures** (critical). A
   hand recording none of the cross-checked figures reconciles under every
   policy; a declared 90% rake moved the reported hero result by 72 of 80 chips
   undisclosed. Repaired by measuring chip movement as well as the verdict.
3. **The exemption was a string a payload could choose** (high). Repaired by
   scoping it on `requires_assumption_attestation` — reconstructed **or**
   imported — instead of on `source_type`.
4. **The attestation named the chips but not the declaration** (medium). Repaired
   by fingerprinting the declaration and the hand's gross pot into the code.

And three defects elsewhere in Phase 1: the DERIVED hero result rendered into the
'Correct hand facts' form and persisted into the OBSERVED `hands.hero_bb_won`
column by saving any unrelated field (high — the editor now reads the stored hand,
and `Hand.derived_result_substituted` makes every writer refuse a display copy); a
settlement row this build cannot validate raising a `ValidationError` out of
`fetch_hand_settlement` into the Study and Insights pages, which catch
`LedgerError` only (medium — the row degrades on read); and Coach Review
presenting an assumption-dependent reconciliation to a provider as reconciled
fact (medium — every non-Study surface now reads `_accounting_is_established`).

Round 10 also reported six mechanisms that survived mutation with no killing test.
All six are now pinned: `_declared_chips_taken`'s fail-closed branch, the
per-input replacement of an attestation, the exempt-hand no-op, the blocker's
`detail` tuple, `BLOCKER_ORDER`, and the payout term in the measured movement. The
`JOINT_INPUT` branch is documented as what it is — defence in depth that no hand
shape is known to reach, kept because its absence would silently return `()` — and
is pinned by an injected test rather than by a hand shape.

Every repair above is a code change, so the counter stays at **0 of 2** and round
11 is the first round eligible to count once no further change lands.

### Round 11, and what it found

Round 11 has now run. It broke the dependence rule itself for the first time, and
it broke it in the same shape the rule was created to end: the neutralisation set
was two values named at a call site, and everything outside those two was held
constant and therefore unmeasurable. Four repairs, each made as a family:

1. **The declared pot awards were outside the measured set** (critical). On a
   reconstructed hand the CV exporter emits no `settlement` key at all, so every
   award row is typed into the Accounting reconciliation panel by an operator,
   and it is the single input the derived payouts and the reported hero result
   are computed from. With `pot_size` and `hero_bb_won` null - the ordinary state
   of a freshly imported hand - one dropdown moved the recorded hero result by
   the whole pot in either direction with no measurement, no disclosure, no
   correction record and an EMPTY blocker tuple. Repaired by making
   `_Declaration` the complete set of inputs a cross-check pass takes from a
   declaration rather than from the recording, with the completeness asserted as
   a property (see "Assumption-dependent reconciliation") rather than as a list.
2. **"Which predicate does this consumer read?" was itself an enumerated list**
   (high). `_accounting_is_established` was applied to the surfaces round 10
   demonstrated; six consumers stayed on `is_authoritative`, and PLAN.md recorded
   the repair as complete anyway. Repaired by one service-level definition and an
   AST regression that fails on any new reader of the raw flag.
3. **Confirming an assumption cleared the blocker and re-enabled nothing**
   (critical). The measurement is re-derived on every read, so a predicate keyed
   on its existence was permanently False for any hand whose rake really takes
   chips - and on a manual hand, where no attestation control is drawn at all,
   coaching was disabled with no clearing action anywhere in the product.
   Repaired by making one expression answer "does this hand still owe an answer?"
   for the blocker and for every gate.
4. **Two mechanisms survived mutation with no killing test** (high): the
   fingerprint's gross-pot term, and the verdict half of `_is_dependent`. Both
   are now pinned, and the verdict half is reachable by a real hand shape rather
   than by injection. A third, smaller one - the measured movement string
   collapsing every sub-microchip quantity to "+0" - is fixed by writing the
   movement in a form that round-trips to the float it was measured from.

Round 11 also reported that the "mypy clean" claim covered eight files, none of
which held the accounting code. `poker_tracker/math/analytics.py`,
`poker_tracker/services/hand_accounting.py` and `poker_tracker/solver/eligibility.py`
are now checked as well (11 files). `app.py`, `persistence/db.py`,
`persistence/import_export.py` and `math/accounting.py` are still NOT type-checked,
and `follow_imports = "skip"` means the checked modules are checked against stubs
of their imports; the claim in "Verification record" says so rather than implying
coverage it does not have.

Every repair above is a code change, so the counter stays at **0 of 2** and round
12 is the first round eligible to count once no further change lands.

### Round 12, and what it found

Round 12 has now run. It found no per-field hole in the dependence rule either,
and it found nothing wrong with the neutralisation set round 11 closed. What it
broke was, once again, the *surroundings* — and one of its findings is the same
rule inverted, which is the most useful thing this round produced. Four blocking
repairs, each made as a family:

1. **The documented `pytest` command migrated the operator's own database**
   (high). `tests/test_app_shell.py` runs `AppTest.from_file("app.py")`, which
   opens `PokerDatabase(DEFAULT_DB_PATH)` and calls `init_db()`. On a
   pre-Phase-1 file that applied the irreversible v13 migration to
   `<repo>/poker_tracker.db`, rewriting `review_status` from `reviewed` to
   `needs_correction` on every reconstructed hand, while the pinned rollback
   snapshot went into the `isolated_backup_dir` temp tree and was deleted with
   it. Repaired at the family: `tests/conftest.py` claims `POKER_DB_PATH` and
   `POKER_DATA_DIR` before its first `poker_tracker` import, which is every
   operator root in the product — the database, the videos, the frames, the
   exports, the ROI previews, the CV timelines, the job logs, the solver runs,
   the backups — with no module list to keep current. The hazard had been
   recognised for the backups alone.
2. **A blocker whose clearing action was performable and could never clear it**
   (high). `INVALID_HERO_OR_BOARD_CARDS` on a hand whose *correct* card value is
   empty: the reader blanks an unreadable column, the form is filled from the
   reader's view, and `update_hand_facts` compared the submission against that
   same degraded view — so `before_state == after_state`, the UPDATE never fired,
   no correction was recorded, and the UI reported "Corrected facts saved". The
   family is not "unreadable cards": it is a writer deciding "nothing changed" in
   the MODEL's space about a statement it makes in the COLUMN's space. Both sides
   of the no-op test are now the values the UPDATE binds, and the statement is
   built from one list so the two cannot be spelled differently again.
3. **The disclosure half of the channel separation** (medium, blocking).
   `declared_unobserved_rake` and `declared_unobserved_chips` were written into
   `warning_codes`, the pipeline's channel, so declaring a rake on a hand whose
   reconstruction evidence was complete and clean demoted it to `uncertain` and
   told the operator "the pipeline could not prove this hand was fully
   reconstructed", pointing at a form with no rake field in it. Round 10 gave the
   ATTESTATION its own channel for exactly this reason and left this half behind.
   Repaired the same way: `declared_settlement_codes`, enforced in
   `parse_completion_evidence` on every read, load-bearing for nothing.
4. **Six clearing actions naming an operation the product does not perform**
   (medium, blocking). "Re-import this hand" — there is no writer that re-imports
   an existing hand. `import_hands_into_session` APPENDS, renumbering collisions,
   so the operator performed the named action verbatim, kept a byte-identical
   blocker, gained a duplicate of every hand in the session, and had a completed
   hand counted twice in that session's statistics with nothing telling them to
   delete the original. There is now one literal describing what the action
   actually does and what else it requires, an AST regression fails on any second
   literal naming a reconstruction, and a behavioural test performs the action and
   pins the append.

And the inverted finding, which is not release-blocking and matters anyway:
`declared_pot_awards` was measured as a dependence on **100% of authoritative
reconstructed hands** (144 of 144 across an independent 696-state sweep), because
withdrawing the awards to "nobody won anything" is not a state any recording can
produce — an award-less ledger is never `is_settled`, so the verdict half fired
unconditionally. A mandatory press of "Confirm this assumption" on every hand,
including hands declaring no rake and no dead money at all, is precisely the
click-through fatigue this rule exists to prevent, and the second option PLAN
promises ("or withdrawing the declaration in the same panel") does not exist for
an award. The withdrawn state is now the award declaration the RECORDING FORCES:
a pot exactly one seat is still eligible for is answered by the action line, not
by the operator, so declaring it moves nothing and is silent; a pot two or more
seats are eligible for is a showdown, where the winner genuinely is a declaration
nothing corroborates, and is still measured, named and blocked. The same sweep
after the repair: 117 named, 27 silent, and every silent one provably forced —
zero hands where a silent award declaration disagrees with the only eligible
seat. An award to a seat that folded is refused by the ledger itself, so the
exemption cannot launder a hero result.

Three smaller repairs: `db.acknowledge_accounting_assumption` accepted any
well-formed code, filed a `hand_corrections` row attesting to it, and — because
at most one attestation survives per input — EVICTED the operator's genuine
attestation on the way past; the check needs a ledger, which that layer cannot
have, so it lives in `services.hand_accounting.attest_assumption` and an AST
regression pins that writer to exactly one caller. The Confirm control's
"the write was refused" branch, the only survivor of round 11's 45 mutants, is
now pinned — and was unobservable in the first place because the branch called
`st.rerun()` and discarded its own message. And `data_health` now reports a
`confirmed_assumption_codes` entry with no matching `hand_corrections` row: the
attestation is the one half of the mechanism that is not re-derived on read, a
forged one cannot be disproven, but an uncorroborated one can be named.

Every repair above is a code change, so the counter stays at **0 of 2** and round
13 is the first round eligible to count once no further change lands.

### Round 13, and what it found

Round 13 has now run. The dependence rule's core held for a third consecutive
round; what broke was the boundary the rule is scoped ON — the manual exemption
— and, once again, the perimeter. One family in two costumes, and four defects
beside it, each repaired as a family:

1. **The manual exemption was two strings** (critical + high, the same family).
   Round 10 stated the closing argument — a hand that was not entered in this
   database may not claim an exemption that rests on being entered here — and
   enforced it on exactly one consumer, `requires_assumption_attestation`. Every
   other reader of the exemption still asked `is_reconstructed(source_type,
   completion_status)`, which is satisfied by writing two strings. An import
   payload relabelled `source_type: manual` with blank evidence bypassed every
   remaining reconstructed-hand blocker at once and was re-promoted straight to
   `reviewed` by `_enforce_review_status_floor` — the floor trusted the same
   payload's `manual` claim as the reason to honour its `reviewed` claim. A
   payload that instead bumped `evidence_version` to 2 defeated the
   manual-payload refusal (which checked `is_known`, 1..1) while KEEPING the
   pipeline's rejection codes in the stored row. And the READER accepted the
   pair import refuses verbatim: one hand-edited UPDATE relabelled a blocked CV
   hand `('manual', 'not_applicable')` with its reconstruction evidence still
   attached, every blocker stopped being emitted, the measured dependence went
   unacted-on, and `update_hand_status` accepted the `reviewed` promotion.
   Repaired at the argument, not at the shapes: `claims_reconstruction` (ANY
   nonzero `evidence_version`, readable or not) is what "carries reconstruction
   evidence" means to import, to `_hand_from_row` — which now normalises a
   manual claim carrying it to `cv_import`, the same verdict import reaches —
   and to `update_hand_status`, which refuses it on the raw row;
   `requires_user_confirmation` (reconstructed OR imported, delegating to the
   round-10 predicate) scopes USER_CONFIRMATION_MISSING and the checkbox that
   clears it, so an imported hand owes this operator's confirmation whatever
   `source_type` it declares; and no hand landing from a payload keeps a
   declared `reviewed`, for any source type — a genuine manual export is
   byte-identical to the forgery, so it lands `needs_correction` too, and the
   label is one tick and one save away for the operator who now vouches for it.
   (The v13 MIGRATION still keeps manual review statuses: a migrated database is
   the same operator's own data, not somebody's JSON.)
2. **Nine clearing actions named a deletion no control performed** (high).
   `NEW_RECONSTRUCTION_STEPS` ended with "delete this one from the session's hand
   list", and the running app had no reachable delete-hand control at all: the
   only `db.delete_hand` call site sat in `show_saved_hands`, which nothing
   invokes, so the operator who performed the named action verbatim ended
   holding a duplicated session and an instruction the product could not
   perform — the exact failure the round-12 repair that wrote that sentence set
   out to eliminate, one layer up. Every hand row rendered by
   `render_hand_results` (Sessions → Hands, and the Hand library) now carries a
   confirm-gated 'Delete hand' control routed through one writer that stops and
   removes solver runs first, the clearing action names it, and an AppTest
   regression performs the deletion on the page the text names.
3. **An evidence write could walk a hand INTO the exemption** (medium).
   `update_hand_completion` re-derived `completion_status` from `source_type`,
   which returns `not_applicable` for any manual row, so one press of the
   generic Acknowledge on a hand-edited `('manual', 'complete')` pair was a
   promotion into the exemption — the hazard
   `db.acknowledge_accounting_assumption` documents and closes by not
   recomputing at all. This writer must recompute (it is the promotion path for
   acknowledged warnings), so it pins the boundary instead: a derived
   `not_applicable` never replaces a stored status that was anything else.
4. **The pipeline's code channels were writable through the evidence blob**
   (medium, defence in depth). `update_hand_completion` trusted a caller's blob
   for `rejection_codes` and `warning_codes` while already refusing to let the
   same blob touch `confirmed_assumption_codes` and `partial`. Both channels are
   now preserved from the stored row — a caller may add a code, which only ever
   demotes, and may never remove one.
5. **The attestation fingerprint did not bind the action line** (low). The
   measured deltas are declared-minus-neutral, so the contributions cancel out
   of every one of them: two seats committing 40 each and four seats committing
   20 each produced byte-identical dependence codes — same gross, same
   declaration, same movement — while the derived hero result differed by 20
   chips. The settled per-seat contribution vector and the hero identity are now
   context terms, so the documented binding ("the declaration and the hand as
   well as the chips") is the implemented one. Stored attestations from earlier
   builds lapse when the code is next measured, the blocker reappears, and the
   operator re-confirms — the same behaviour as every earlier fingerprint
   strengthening, and no schema change.
6. **The dependence rule's non-reconciling-baseline gate had no killing test**
   (medium). Replacing `if not baseline.reconciles: return ()` with `pass`
   survived the full suite while drawing Confirm-this-assumption controls on a
   hand whose chips do not balance. Pinned by a regression that was run against
   the mutant and observed to kill it.

Round 13 also reported CV-suite test failures from the concurrent Phase 5
workflow (out of this effort's scope). That workflow was still editing
`cv_lab/scripts/pipeline/ocr_readers.py` and its tests while these repairs
landed, so those files' results drifted run to run; every run with the
in-flight CV files excluded was green, the Phase 1 files are unaffected, and
the final full run after its edits settled is green end to end. Round-14
eligibility requires the whole tree green at the moment the counting rounds
run, which is a Phase 5 close-out condition, not a Phase 1 one.

The regressions live in `tests/test_phase1_adversarial_round13.py`, one per
family plus the constructed instances, and the two import tests that pinned "a
manual hand's declared review status is never rewritten by the import" now pin
the opposite — that sentence described the defect.

Every repair above is a code change, so the counter stays at **0 of 2**.

### Round 14, and what it found

Round 14 has run. It found no hole in the dependence rule itself for a fourth
consecutive round. What it found, again, was enumerated-list decay AROUND the
rules, in six families, each repaired as a family with several independent
instances pinned rather than the one shape demonstrated:

1. **Row readers raised instead of degrading** (high). `_hand_from_row` guarded
   four columns by hand while `table_size = 99` in the fifth raised a
   `ValidationError` out of `fetch_hands_by_session` and took the entire
   application down on load; `_action_from_row` and its siblings were not
   guarded at all. Every reader now salvages column-by-column through one helper
   (`db._salvaged_row`) driven by the model's own field set, and every
   degradation is conservative: it can only ever add blockers. The columns it had
   to give up are recorded under `UNREADABLE_HAND_COLUMNS_KEY` and reported by a
   new `UNREADABLE_HAND_COLUMNS` blocker.
2. **`update_hand_completion` took the caller's blob as the base** (high).
   Pinning three code channels from the stored row left every OTHER field
   caller-writable: a blob manufactured the pipeline's boundary observations and
   promoted an unprovable hand, and a blob that dropped the
   `imported_from_payload` stamp walked an imported hand into the manual
   exemption. The merge is inverted — the stored evidence is the base and the
   caller may only ADD codes.
3. **The promotion gate deny-listed values instead of allow-listing them**
   (high). Any unrecognised `terminal_event` counted as observed, and any finite
   `boundary_confidence` — including 0.0 and 42.0 — counted as a measurement.
   Both are allow-lists now (`OBSERVED_TERMINAL_EVENTS`, a 0..1 range).
4. **Non-finite floats passed the validating boundary** (medium). `ge=0` admits
   `inf`, so a hostile payload landed `dead_money=Infinity` and the session
   export stopped being RFC 8259 JSON. `PersistedModel` sets
   `allow_inf_nan=False` once, for every float field of every persisted model.
5. **The raw attestation writer trusted a shape-valid code** (high). A forged
   `declared_settlement_dependence:...` string was recorded, evicted the
   operator's genuine attestation for the same input, and filed an audit row for
   an attestation nobody made. `db.acknowledge_accounting_assumption` now
   re-measures and refuses a code naming no current dependence, failing closed
   when the measurement cannot be taken.
6. **The dependence tolerance was unpinned** (medium). Widening
   `_FLOAT_TOLERANCE` from 1e-9 to 1e-6 survived the whole suite.

The regressions live in `tests/test_phase1_adversarial_round14.py` (20 tests).
Every repair is a code change, so the counter stays at **0 of 2**.

### Round 15, and what it found

Round 15 has run, with three adversaries (accounting/assumption bypass;
readiness bypass outside accounting; mutation testing and honesty). The
dependence rule's core held for a fifth consecutive round. Twenty-two findings
were reported and twenty-one reproduced; they collapse into eight families, each
repaired at the general case with a sweep of independent instances:

1. **A SQL predicate classified a row that a reader reclassifies** (high, four
   reported instances, six found). `discard_stale_coaching` filtered
   `WHERE is_stale = 1` while `_coaching_response_from_row` answers
   `bool(is_stale)`, so a stored `2`, `-1` or `'yes'` raised
   STALE_COACHING_EVIDENCE, drew the control the blocker names, matched nothing,
   and flashed "Discarded 0 stale coaching review(s)." as a SUCCESS — a
   permanently unstudyable hand. `resolve_hand_issue` had the mirror image:
   `_hand_issue_from_row` forces `status='open'` on a row it cannot fully read,
   the writer filtered `AND status = 'open'`, and the blocker's own clearing form
   answered "Open hand issue not found." `update_hand_status`'s documented
   unbypassable floor and `fetch_hand_issues`' status filter were blind to the
   same row, so the store ACCEPTED a `reviewed` promotion readiness refuses and
   the unresolved-issue inbox did not list it. `_validate_single_hero` accepted a
   second hero, and `fetch_cached_solver_run` served a run the reader calls
   `stale` as a cache hit. Every site now selects candidates by identity and
   classifies through the reader;
   `test_no_sql_predicate_classifies_a_row_the_reader_reclassifies` fails on a
   new raw-column predicate, with an allow-list of the seven places where the
   column, not the verdict, is the right subject.
2. **Two stored timestamps could be incomparable** (high). `fromisoformat`
   returns a NAIVE datetime for an offset-less string, and one such value made
   `max(stale)` raise `TypeError` out of `_coaching_blockers`, taking the Study
   page down for that hand and Insights down for the ENTIRE database.
   `PersistedModel` normalises every naive datetime field to UTC, so every
   comparison in the product is fixed rather than the two demonstrated.
3. **An attestation travelled in the payload carrying its evidence** (high). One
   JSON field flipped a debugging issue to `resolved`, in a state
   `resolve_hand_issue` refuses to create, and cleared OPEN_DEBUGGING_ISSUE.
   Imported issues land `open`, exactly as imported coaching lands stale and a
   declared `reviewed` is floored; the exporting database's resolution notes are
   carried into the reopened issue's description so nothing is lost.
4. **A read-time degradation marker was laundered by a round trip** (medium).
   The card marker was restored from round 5; `UNREADABLE_HAND_COLUMNS`, added in
   round 14, had no equivalent, so an ordinary export/import was a third,
   undocumented clearing action that repairs by discarding — including for the
   columns the blocker says cannot be repaired at all. Restoration is keyed on
   `DERIVED_EVIDENCE_KEYS` and on the table's own PRAGMA columns, and round 8's
   "a marker may not overwrite a readable column" guard is generalised from the
   two card columns to every column.
5. **Two degradations on one row reached opposite `review_status` verdicts**
   (medium). `_degraded_hand` forced `needs_correction`; `_degrade_unreadable_cards`
   — which blanks the columns that ARE the study material — did not, so a hand
   hand-edited to a two-card board counted as `reviewed` in
   `compute_session_stats`, in the Insights "Unresolved" KPI and in every list
   row while Study refused it. One `_demote_degraded_hand` step keyed on "does
   this hand carry ANY read-time marker?" now applies the contract, which also
   fixes the writer-side instance (`restore_unreadable_columns`) with no second
   guard.
6. **A consumer prescribed an acknowledgement for a pipeline REJECTION**
   (medium). `_source_warning_blockers` was repaired for this in round 13 and
   `_layout_blockers` was not, so three blockers on the same page said
   contradictory things about one code and the operator was told the
   unperformable one was the fix. The split is now
   `CompletionEvidence.unresolved_warning_codes` /
   `unresolved_rejection_codes`, and the mixed property's readers are enforced.
7. **A derived figure was written into an observed-fact column** (high). The
   settlement editor's "Replace observed final pot/result with the derived ledger
   values" was gated on `reconciled.ledger.is_settled`, strictly weaker than
   `accounting_is_established`. On a reconstructed hand recording nothing with a
   declared 25% rake it wrote the declaration-derived +20 into
   `hands.hero_bb_won` — the honest result is +40 — and because Study correctly
   refused the hand, `compute_session_stats` fell back to `hands.hero_bb_won` and
   published the fabrication as an OBSERVED result. On a hand with genuine
   observations it replaced `pot_size` 80 and `hero_bb_won` +40 with 155 and
   +115. The gate now lives in `services.settlement_sync`, the db writer refuses
   again on its own single-pass measurement when nothing is attested, and
   `test_no_ui_call_site_writes_the_recorded_pot_or_hero_result` fails on a new
   UI call site. This is PLAN.md:679 ("Distinguish observed facts from
   user-entered settlement assumptions") violated literally, and the repair is
   scoped so no tenth call site can reopen it.
8. **`AssumptionDependence.describe()` stated every direction backwards**
   (medium). The deltas are declared-minus-neutral, which is right for the CODE,
   and the sentence's subject is the REMOVAL, so 5 of 6 printed terms on a
   50%-rake hand were inverted — in the blocker detail, in the caption above
   "Confirm this assumption", and in the line handed to the coaching provider. An
   operator applying the blocker's own clearing action would have gone and edited
   correct data. Repaired in `describe()` and not in `_ledger_deltas`, which
   would have lapsed every stored attestation for a display defect.

Beside the families, round 15's mutation adversary demonstrated eight behaviours
surviving the whole suite with no killing test — both coaching gates, the
measured movement inside `describe()`, three declared fields of the attestation
fingerprint (`rake_rounding_unit`, `no_flop_no_drop`, the declared award amount),
the producer of `derived_result_substituted`, the cross-check's
"No persisted settlement assumptions or awards." issue, and the `unbuildable`
measure label — and four documentation claims nothing enforced: this file's
missing round-14 record and stale verification table, README's false "no value
you type in `Chip unit` changes a derived payout", a comment crediting
`_declared_chips_taken` with a dual reconciliation it does not perform, a comment
citing a test that does not exist, and an understated cost ceiling (a fold win
costs five extra ledger builds, not four). All are pinned or corrected, and each
mutant was re-run against its regression and observed to be killed.

The regressions live in `tests/test_phase1_adversarial_round15.py` (90 tests),
grouped by family. Every repair above is a code change, so the counter stays at
**0 of 2** and round 16 is the first round eligible to count once no further
change lands.

**What remains before Phase 1 may be counted toward the release claim** — and
nothing below is optional:

1. Two consecutive adversarial rounds, each with fresh agents and varied attack
   prompts, reporting zero critical, zero high, zero release-blocking medium, and
   zero unresolved safety/data-loss/stale-evidence/silent-acceptance findings.
   Rounds 10 through 15 have been run and all six produced blocking findings;
   their repairs land here, so the counter is at 0 and round 16 is the first
   round eligible to count.
2. No code, schema, dependency, or release-configuration change between those two
   rounds. The round-15 repairs land *before* round 16 starts.
3. The exit-gate bullets below stay satisfied through both rounds; they are
   re-derived from the code each round, not carried forward. Where this document
   states which call sites exist, the statement is now backed by a test rather
   than by a reading - round 11's medium was this file recording a repair as
   complete while six consumers still read the old predicate.

Fifteen rounds have each broken a mechanism the previous round's document
claimed was closed, twice in the same place (the chip denomination, rounds 8 and
9; the accounting tolerance, rounds 4 through 7). Round 10 broke the
*surroundings* of a mechanism whose core it could not break. Round 11 broke the
core, by the rule's own argument: the neutralisation set was two values chosen
by hand, so the largest declared input on every reconstructed hand was never in
it. Round 12 found the core intact under a fresh attack and broke the
surroundings again — including, for the first time, the rule's own anti-fatigue
property, which was false for 100% of authoritative hands. Round 13 found the
core intact for a third time and broke the SCOPE it is gated on: the manual
exemption was decided by two strings a payload or a hand-edited row could write,
and round 10's own closing argument ("neither of them may claim an exemption
that rests on being entered here") had been enforced on exactly one of its
consumers. Round 14 found the core intact for a fourth time and broke the
READERS around it — a single unreadable column took the whole application down
on load, and the evidence writer took the caller's blob as its base. Round 15
found the core intact for a fifth time and broke the SQL beneath it: five
predicates classified rows in the column's space while every blocker, list and
gate read the model, so a clearing action reported success while clearing
nothing and a store-level floor was blind to the row it is the floor for. It also
found the one place the whole dependence rule can be circumvented without
touching the rule — writing the derived figure into the observed-fact column the
rule's own fallback reads.

That is the fifteenth consecutive round to find a real blocking defect, and the
prior on a clean round 16 should be set accordingly. The recurring lesson of
rounds 10 through 15 is that a repair stated as an argument but applied to an
enumerated list of consumers decays into the next round's finding; the round-13
repairs are therefore scoped on shared predicates (`claims_reconstruction`,
`requires_user_confirmation`) rather than on the call sites that were
demonstrated, and the round-15 repairs go one step further — where a family
cannot be collapsed into a single predicate, the SET of legitimate call sites is
now asserted by a test that fails when a new one appears
(`test_no_sql_predicate_classifies_a_row_the_reader_reclassifies`,
`test_no_consumer_prescribes_an_action_from_unresolved_codes`,
`test_no_ui_call_site_writes_the_recorded_pot_or_hero_result`, joining
`test_no_consumer_decides_on_is_authoritative_alone`).

The current `review_status` is not enough to distinguish “the hand ended,” “the
pipeline reconstructed enough evidence,” “the ledger balances,” and “the hand
is safe for study.” Those concepts must be represented separately.

### Readiness blocker vocabulary

Readiness is derived per render and never persisted. Each blocker carries a
stable code, a category, a plain-language reason, and the exact clearing action.
Blockers are emitted, and categories rendered, in this order:

| Code | Category | Applies to |
| --- | --- | --- |
| `COMPLETION_NOT_COMPLETE` | completion | reconstructed hands |
| `COMPLETION_EVIDENCE_MISSING` | completion | reconstructed hands |
| `INVALID_HERO_OR_BOARD_CARDS` | cards | all hands |
| `UNREADABLE_HAND_COLUMNS` | facts | all hands |
| `UNSUPPORTED_TABLE_LAYOUT` | layout | reconstructed hands |
| `ACCOUNTING_NOT_AUTHORITATIVE` | accounting | all hands |
| `ACCOUNTING_ASSUMPTION_DEPENDENT` | accounting | reconstructed hands |
| `OPEN_DEBUGGING_ISSUE` | issues | all hands |
| `UNRESOLVED_SOURCE_WARNING` | completion | reconstructed hands |
| `STALE_COACHING_EVIDENCE` | coaching | all hands |
| `STALE_SOLVER_EVIDENCE` | solver | all hands |
| `USER_CONFIRMATION_MISSING` | confirmation | reconstructed and imported hands |

A manual hand (`source_type == "manual"` and `completion_status ==
"not_applicable"`) can only ever emit the cards, `UNREADABLE_HAND_COLUMNS`,
`ACCOUNTING_NOT_AUTHORITATIVE`, issue, coaching, and solver blockers, so the
pre-Phase-1 manual workflow is unchanged — with two qualifications, and they are the two blockers scoped on
"entered here" rather than on the pair. `ACCOUNTING_ASSUMPTION_DEPENDENT` is
scoped by `requires_assumption_attestation`: a hand this operator entered *in
this database* is exempt, because a declared ante or rake is that operator's own
observation and there is no pipeline claim for it to outrank; a hand that
arrived through import is not exempt, whatever `source_type` it declares,
because the importing operator entered nothing. The dependence is still measured
and still reported on the reconciliation for an exempt hand; it just never
blocks, and it never withholds the hand's figures from the win rate, the list
views, a coaching prompt or the solver either - an exempt hand's declaration is
already answered by the person who made it, which is the whole argument for the
exemption. `USER_CONFIRMATION_MISSING` is scoped by
`requires_user_confirmation`, which delegates to the same predicate for the same
reason: an imported hand declaring `source_type: manual` owes this operator's
explicit confirmation exactly as a reconstructed hand does, and round 13 showed
that scoping it on the reconstructed pair alone let such a payload land
study-ready with an empty blocker tuple. See "Who must attest, and why it is not
`source_type`".

### Assumption-dependent reconciliation

`ACCOUNTING_ASSUMPTION_DEPENDENT` is derived, per read, by reconciling one hand
twice: once with the stored settlement declaration and once with that declaration
withdrawn, against the *same* fetched records. Everything that is the hand rather
than the declaration — the action line, the board, whether a flop was seen, the
recorded pot and hero result — is held constant across both passes.

The declaration is `hand_accounting._Declaration`: the rake policy, the declared
dead money, **and the declared pot awards**, which is the complete set of inputs a
cross-check pass takes from what somebody declared rather than from what was
recorded. The awards used to be on the other side of that line, documented as
"the hand, not the policy". They are not: the CV exporter emits no `settlement`
key at all, so on a reconstructed hand every award row was typed into the same
panel, by the same operator, in the same save as the rake — and it is the single
input the derived payouts, and therefore the reported hero result, are computed
from. On a hand recording a null `pot_size` and a null `hero_bb_won` (the ordinary
state of a freshly imported hand) one dropdown moved the recorded hero result by
the whole 80-chip pot in either direction, with no measurement, no disclosure, no
correction record and an EMPTY blocker tuple, while a declared rake of the same 40
chips was named, measured and blocked. Round 11 called that "a two-entry field
list wearing a different hat", which is exactly what it was.

The completeness of that set is a property rather than a promise:
`test_a_neutral_declaration_derives_a_ledger_from_the_recording_alone` sweeps
wildly different settlement rows and award sets over one unchanged recording and
asserts every fully-neutral ledger is identical, so an input added to the derived
side later without being added to `_Declaration` fails a test instead of opening
the next per-field hole.

The hand is assumption-dependent when removing the declaration changes either the
**verdict** (the neutral pass stops reconciling, or the ledger cannot be built
without it) or the **figures** (it reconciles too, but derives a different gross
pot, rake, net pot, payout, or hero result). Each declared input is then
neutralised on its own, by the same two-part test, to attribute it.

The second half is not a refinement; without it the rule has a hole the size of
the product. A hand that records none of the figures the cross-check compares —
no `gross_pot`, `rake_amount` or `net_pot` on the settlement, a null
`hands.pot_size`, a null `hands.hero_bb_won`, award rows with no amount —
reconciles under *every* policy, because nothing is left for a policy to
contradict, and that is the ordinary state of a freshly imported hand:
`import_session` never calls `persist_reconciliation`. Round 10 landed a declared
90% rake on such a hand and measured it as assumption-INDEPENDENT while it moved
the hero result by 72 of the pot's 80 chips — the result `_hands_with_accounting_results`
and `math.analytics` substitute into every list, every stat and every prompt. A
sweep of 585 authoritative shapes (2-4 seats × 13 declarations × 3 winner sets ×
5 recording states) now agrees exactly, in both directions, with "do the reported
figures move?": 145 of them were unblocked with an empty blocker tuple before.

The VERDICT half is not decorative either, which round 11 correctly reported it
as: over 27,000 shapes it was never the sole reason for a dependence, because the
five figures `_ledger_deltas` compares covered every disagreement the cross-check
could produce. Withdrawing the declared awards changes that. Declare a policy
that takes the whole pot and every payout is zero already, so removing the
winners leaves gross, rake, net, payout and hero all standing still while the
ledger goes unsettled - a dependence with no chip movement at all, measured as
`verdict-only` and pinned by
`test_withdrawing_the_awards_is_a_dependence_the_figures_cannot_show`, which
fails if the branch is deleted.

**What the awards are withdrawn TO** is the round-12 repair, and it is the same
argument the rest of the rule rests on. "Withdraw the declaration" has to name a
state the hand could actually be in without it. For the rake and the dead money
that state is obvious. For the awards it is NOT "nobody won anything": no
recording produces that, an award-less ledger is never `is_settled`, and
comparing against it therefore made every hand whose baseline reconciles
award-dependent by construction — measured at 144 of 144 authoritative states in
an independent sweep, 300 of a wider one where it was the ONLY dependence, i.e. a
compulsory press of `Confirm this assumption` on a hand declaring no rake and no
dead money at all. That is the click-through fatigue the first property below
exists to prevent, arrived at from the other side, and the alternative clearing
action this document promises ("or withdrawing the declaration in the same
panel") is not available for an award at all.

What a recording CAN determine is the winner of a pot exactly one seat is still
eligible for: everyone else folded, so the action line names the winner and the
operator's declaration of it moves nothing. `_forced_winners` reads that off the
award-less ledger's own `PotLayer.eligible_players`, and it is the state the
awards are compared against whenever every pot has one. A pot two or more seats
are eligible for is a showdown, where who was pushed the chips genuinely is a
declaration nothing in the recording corroborates: still measured, still named,
still blocking. The exemption cannot launder anything, because an award to a seat
that folded is refused by the ledger itself (`Player is not eligible for pot`)
before any of this runs. The same 696-state sweep after the repair: 117 named, 27
silent, every silent one provably forced, zero disagreements between a silent
declaration and the only eligible seat.

Four properties follow, and they are the reason this replaced eight rounds of
per-field disclosure conditions:

- **There is no field list, so there is no per-field hole.** A rate with a zero
  cap, a no-flop-no-drop waiver on a hand that saw no flop, and a chip unit
  coarser than the whole rake all take zero chips, derive the same ledger the
  action line derives alone, and are silent. Any combination that does move chips
  is named without anyone having enumerated it.
- **The attestation is bound to a quantity AND to the declaration.** The blocker
  clears only when the operator confirms a code carrying both a fingerprint of the
  declared inputs (over this hand's gross pot, its settled per-seat contribution
  vector, and which seat is the hero) and the measured chip movement
  (`declared_settlement_dependence:<input>:<fingerprint>:<movement>`). The four
  original context terms are pinned by
  `test_the_fingerprint_separates_every_context_term_it_digests`, including the
  gross-pot term, which survived mutation with no killing test in round 11 and is
  the only thing telling apart two declarations whose measured movement is
  byte-identical because a rake cap pins the rake either way; the contribution
  and hero terms are round 13's addition — the deltas are declared-minus-neutral,
  so the contributions cancel out of every one of them, and two seats committing
  40 each shared a code with four seats committing 20 each while the derived hero
  result differed by 20 chips — and are pinned by
  `tests/test_phase1_adversarial_round13.py`. The movement is
  written with `format(value, "+")`, which round-trips to the float it was
  measured from, so no two distinct measurements share a string; the previous
  `f"{value:+.6f}"` rendered every movement in (1e-9, 5e-7) as "+0". An
  attestation earned while a policy destroyed 0.01 chips is a different string
  from one covering the same policy destroying 80.01 chips off a corrected action
  line, and — round 10 — a 50% rake on an 80-chip pot is a different string from
  a 25% rake on a corrected 160-chip pot, which removes the same 40 chips and
  moves every headline figure identically.
- **The attestation has its own channel and its own control.** It is stored in
  `completion_evidence.confirmed_assumption_codes`, never in `warning_codes` or
  `acknowledged_codes`, and `parse_completion_evidence` enforces the separation in
  both directions on every read, so no writer, payload, or hand-edited row can mix
  them. It used to share the pipeline channel "for auditability": an ordinary
  export → import round trip then delivered a legitimate attestation as an
  *unacknowledged pipeline warning*, which demoted `completion_status`, was
  reported by `UNRESOLVED_SOURCE_WARNING` as a field to fix in Correct hand facts,
  and was answered by the generic one-click **Acknowledge** — a control captioned
  as a pipeline note, stating no chip figure, that cleared this blocker in one
  press. The audit trail is the `hand_corrections` row the attestation writes.
- **The verdict lives with the reader, not a writer.** `hand_settlements` has no
  CHECK constraint, and every earlier disclosure was raised inside
  `upsert_hand_settlement`; a settlement row written any other way reached
  `reconcile_persisted_hand` undisclosed. The measurement runs on every read that
  readiness consults, so no writer can bypass it. A row this build cannot validate
  at all (a negative rate, a zero rounding unit, a NaN) is degraded on read to a
  non-reconciled settlement naming the unreadable columns, rather than raising a
  `ValidationError` out of the fetch and taking the session's hand list with it.

Confirming the hand as a whole (`USER_CONFIRMATION_MISSING`) does **not** clear
it: that checkbox asks whether the reconstruction is right, and this asks whether
specific unobserved chips were really taken, added, or pushed to a particular
seat. The clearing action is `Confirm this assumption` in Study → Summary →
Accounting reconciliation, or correcting the declaration in the same panel until
it matches what happened — a declaration that changes nothing is never disclosed
at all. "Withdrawing" is the right word for a rake or a dead-money amount, which
can be set to zero; it is not available for a pot award, which is why the awards
are compared against the winner the recording forces rather than against no
winner (see above) instead of being offered a clearing action that does not
exist.

Confirming is also what re-opens every gate the dependence closed. The
measurement is re-derived from the chips on every read and is never erased by the
answer, so a predicate keyed on its mere existence stayed False forever: an
operator who did exactly what the blocker asked held a study-ready hand whose
coaching button was permanently disabled above a message naming the action they
had just performed, and on a manual hand — where no attestation control is drawn,
because there is nothing to attest to that the operator did not already state —
an ordinary room rake disabled coaching with no clearing action in the product at
all. `unattested_assumption_dependence` is the one place that decides, and
"answered" means attested *or* exempt.

Every surface that publishes a derived figure reads one predicate,
`services.study_readiness.accounting_is_established` — authoritative *and* no
measured dependence this hand still owes an answer for — rather than
`is_authoritative`. "Owes an answer" is the same expression the readiness blocker
uses (`unattested_assumption_dependence`), so the gate and the blocker cannot
drift: an attested declaration and a hand exempt from attestation are both
answered, and answering re-publishes the figure everywhere at once.

Writing a derived figure INTO an observed-fact column takes the same gate, and
that is round 15's finding. "Every surface that publishes a derived figure" was
read as every surface that *renders* one; the settlement editor's "Replace
observed final pot/result with the derived ledger values" writes one into
`hands.pot_size` and `hands.hero_bb_won`, which is the strongest form of
publishing there is — those columns are the independent evidence the cross-check
compares against, they are exported as observed facts, and
`math.analytics.compute_session_stats` falls back to `hands.hero_bb_won`
precisely when the derived figure is refused, so an unattested declaration
written there is republished as an OBSERVED result on the hand Study is refusing.
The write now goes through `services.settlement_sync`, which takes
`accounting_is_established` and refuses by name otherwise;
`db.update_hand_accounting_evidence` refuses again on its own single-pass
`_declared_chips_taken` measurement when the hand owes an attestation and carries
none; and `test_no_ui_call_site_writes_the_recorded_pot_or_hero_result` fails if
any module outside that service calls the writer.

That list of consumers is enforced rather than written down.
`test_no_consumer_decides_on_is_authoritative_alone` walks the AST of `app.py`
and every `poker_tracker` module and fails on any read of `.is_authoritative`
outside six named places, each recorded with the reason the raw ledger verdict
is the right question there. The count is asserted against the enforced set
rather than written here, because this paragraph said "seven" for two rounds
while the set held six. The previous version of this paragraph claimed the
repair was complete when six consumers were still on the raw flag — the session
win rate and its confidence interval, the hero-result column of every list view,
the Overview featured pot, `_hero_ledger_result`, the Math Review defaults and
solver eligibility — so a declared 90% rake published −32 BB as a reconciled
result, and a −3200 bb/100 win rate, on a hand Study refused for exactly that
reason. A claim about which call sites exist is now a test.

The writer-side codes `declared_unobserved_chips` and `declared_unobserved_rake`
are retained as the audit trail. They are raised by `_declared_chips_taken`
(does this declaration actually move chips on this hand?) rather than by
`dead_money > 0` / `rake_rate > 0` — a *different*, single-pass measurement from
the dependence rule above, with a strictly smaller input set: its own docstring
says "Winners are deliberately not fetched", because `upsert_hand_settlement`
calls it before the award rows may exist.

They are stored in `completion_evidence.declared_settlement_codes`, never in
`warning_codes` or `acknowledged_codes`, and `parse_completion_evidence` enforces
that separation on every read exactly as it does for the attestation channel — a
misfiled declaration is relocated rather than dropped, because it is an audit
record with nothing resting on it and every database written before round 12
holds its declarations in `warning_codes`. Round 12's finding was that they were
in the pipeline's channel while being described here as "no longer load-bearing
for readiness", and they were both: `derive_completion_status` demotes on an
unresolved entry in `warning_codes`, so declaring a 50% rake on a hand whose
reconstruction evidence was complete and clean turned it `uncertain` and raised
COMPLETION_NOT_COMPLETE ("The pipeline could not prove this hand was fully
reconstructed") and UNRESOLVED_SOURCE_WARNING ("The pipeline flagged 1 unresolved
source warning(s)") about a figure the pipeline never claimed — naming Correct
hand facts, a form with no rake field, as the place to fix a value that lives
only in the Accounting reconciliation panel. Now they are load-bearing for
nothing: no blocker reads them, no Acknowledge control offers them, and
`completion_status` does not move when one is written.

### Who must attest, and why it is not `source_type`

The exemption is for a hand **this operator entered in this database**: an ante, a
dead blind, a straddle from a seat that left, and the room's rake are all that
person's own entry, and there is no pipeline claim for a declaration to outrank.
`manual` + `not_applicable` is how such a hand is stored, but it is not the
argument, and round 10 showed the difference: an import payload declaring
`source_type: manual` with its `completion_evidence` removed satisfies import's
manual-payload guard (which refuses only *readable* reconstruction evidence),
derives `not_applicable`, and landed a fabricated hero result study-ready with an
empty blocker tuple before any click at all. Such a payload is byte-identical to a
genuine manual export, so no guard can disprove the claim — and none has to. What
a payload cannot manufacture is having been entered here: `import_session` stamps
every hand it lands, and `requires_assumption_attestation` (reconstructed **or**
imported) is the single predicate consulted by the blocker, by the control that
clears it, and by the writer behind that control. The stamp carries no
`evidence_version`, so it is not reconstruction evidence and the manual-payload
guard is unaffected; it is idempotent, so repeated round trips do not accumulate.

That one predicate also fixed a control that lied: the writer was scoped on
`source_type == 'manual'` while the blocker was scoped on the reconstructed pair,
so on a `manual` row stored with a completion status other than `not_applicable`
the button was drawn, the write was silently discarded, and the page flashed
"Confirmed". The writer now returns whether it recorded anything, and it never
re-derives `completion_status` — re-deriving it turned one press of this button
into a promotion that cleared three unrelated blockers.

Round 13 finished what round 10 started, because the argument had been enforced
on one consumer and the exemption's every OTHER consumer still read the pair of
strings. Three more places now reach the same verdict. The reader:
`_hand_from_row` normalises a `manual` claim carrying a reconstruction claim
(`claims_reconstruction`, ANY nonzero `evidence_version` — gating on
readability let a bumped version smuggle rejection codes past import's refusal)
to `cv_import`, exactly as it already normalised the reverse pair, so a
hand-edited relabel cannot walk a blocked CV hand out of the blockers while its
evidence stays attached. The confirmation gate: `USER_CONFIRMATION_MISSING` and
the checkbox that clears it are scoped by `requires_user_confirmation` —
reconstructed or imported, delegating to `requires_assumption_attestation` so
the two cannot drift — because the importing operator has not vouched for a
hand whatever its payload declares. And the review-status floor:
`_enforce_review_status_floor` demotes every declared `reviewed` to
`needs_correction`, for any source type — it used to re-promote any payload
claiming `manual`, trusting one string from the payload as the reason to honour
a second string from the same payload. A genuine manual export is byte-identical
to the forgery and gets the same honest treatment; the label is one tick and one
save away for the operator who now vouches for it, and the v13 migration is
unaffected because a migrated database is the same operator's own data.

Nothing this build stores or exports can be a value RFC 8259 JSON cannot express.
`_serialize_json` writes with `allow_nan=False` after replacing non-finite floats
with `None`, `_parse_json_object` sanitises on read so a row an older build already
wrote is cleaned rather than re-emitted, and `completion._as_float_or_none` treats
NaN and infinity as unreadable. A `boundary_confidence` of NaN used to pass the
`is None` gate, derive `complete`, accept a promotion to `reviewed`, and then leave
a version 5 export that Python's lenient reader could re-import — restoring the
same NaN — and no standards-compliant parser could read at all.

### Additive persistence model

Add a schema version 13 migration with:

- `hands.completion_status`, with:
  - `complete`;
  - `partial`;
  - `uncertain`;
  - `not_applicable` for manual hands;
- `hands.completion_evidence`, stored as versioned JSON containing:
  - `partial_start`;
  - `partial_end`;
  - terminal-event type;
  - first/last source timestamps;
  - preceding/following boundary evidence;
  - boundary confidence;
  - source frame references;
  - warning/rejection codes;
  - pipeline/model versions.

Derive, rather than persist, a `study_readiness` result with a list of blockers.
For a reconstructed hand, readiness requires:

- `completion_status == complete`;
- valid and unique hero/board cards;
- supported table/layout evidence;
- authoritative reconciled accounting;
- no open debugging issue;
- no unresolved source warning;
- no stale retained coaching or solver result being represented as current;
- explicit user confirmation.

### Migration impact

- The migration is additive and must not delete or rewrite correction, issue,
  coaching, settlement, video, or solver history.
- Exactly one process runs the chain. `init_db` re-reads the stored version under
  SQLite's write reservation and stops if another opener already migrated, so two
  concurrent opens cannot both replay v13 and reset operator confirmations. A
  stamp that is not a readable, non-negative version number is refused with the
  same clear message the newer-database path uses, rather than replaying the whole
  chain (`-1`) or escaping as a bare `int()` error.
- A **missing** stamp is not the same as a fresh database, and is refused too.
  `_readable_schema_version` reports `0` for a genuinely new file, for a deleted
  `schema_version` row, and for a dropped `schema_metadata` table; only the first
  may migrate. `_physical_schema_floor` reads the file's own schema for the one
  artefact of the one migration that is not safe to replay — `hands.completion_status`
  — and a stamp behind that floor is refused with a restore-from-backup message.
  Migrations 6–12 are additive DDL and idempotent row repair and are not part of
  the discriminator, so a genuine pre-versioning database still migrates: the test
  is what the schema physically contains, never whether a stamp happens to exist.
  The floor is measured on the file as it arrived, before `_create_base_tables`
  writes the current schema, and it is compared against the version re-read under
  the write reservation, so neither a fresh file nor a concurrent opener that saw
  a stale pre-migration version is refused.
- A database this build refuses is never written to. The `journal_mode = WAL`
  pragma runs only after the version check, so a refused open leaves a restored
  `journal_mode=delete` snapshot byte-identical and does not fail on a read-only
  mount before the operator sees the message.
- The pre-migration snapshot is pinned: it is written under a name the five-slot
  rotation never matches, so routine per-import snapshots cannot delete the only
  rollback point for an irreversible migration. Pinned is not unaudited and not
  unbounded. `data_health` scans `backup.PINNED_GLOB` alongside the rotating glob
  and restore-drills pinned snapshots without the current-version comparison,
  because they are stamped at the pre-upgrade version; and pinned snapshots keep
  their own `PINNED_KEEP_COUNT` slots, so a repeatedly failing migration cannot
  write one unbounded full copy per start.
- A failed pre-migration snapshot names the directory it could not write and says
  the database is unchanged. SQLite reports an unwritable backups mount as
  `unable to open database file`, whose plain reading is that the operator's
  database is corrupt — the opposite of the truth, and the message a container
  operator gets at startup on a read-only or full `data/` volume.
- Existing manual hands become `not_applicable` so their current workflows keep
  working.
- Existing `cv_import` and `corrected_cv` hands migrate conservatively to
  `uncertain` and `needs_correction`; they require confirmation rather than
  being silently promoted.
- JSON export becomes version 5 and includes completion evidence.
- Import versions 1–4 remain accepted and receive safe defaults.
- Older application versions continue refusing to open a newer database.
- A consistent backup must be created before migrating a real file database.

### UI behavior

- Show completion, accounting, issue, coaching, and solver blockers separately.
- Never use one ambiguous percentage as proof that the whole hand is correct. The
  landing hero's proof point reads "N% marked reviewed", not a bare "N% reviewed":
  it is the first number a user sees, and `review_status` is a workflow label
  everywhere else in the app.
- Prevent a partial or uncertain CV hand from being marked reviewed.
- Allow the user to inspect and correct retained partial hands without treating
  them as completed study records.
- Explain why a hand is blocked and what exact action clears each blocker.
- Show the reconstruction evidence the confirmation gate refers to. The checkbox
  reads "I have read the evidence above and confirm this hand is correct", and
  until round 8 there was no evidence above it: the boundaries, terminal event,
  boundary confidence, source timestamps and frames, layout profile and
  pipeline/model versions had a producer, a persistence layer, an export format
  and a digest consumer, and no display consumer at all, so on a hand where the
  tick was the only remaining gate the whole content above it was the sentence
  saying it had not been ticked. `view_models.completion_evidence_rows` is the
  pure transformation and `app.show_reconstruction_evidence` draws it on all three
  promotion surfaces — Study, the Sessions hand list, and Settings → Coach.

### The completion invariant

`completion_status == "complete"` is true only when
`derive_completion_status` says so, on every writer and every reader:

- the CV exporter derives it from the evidence it just built;
- `import_session` re-derives it and ignores whatever the payload declared, at
  every export version;
- `update_hand_completion` re-derives it on every evidence write;
- `update_hand_status` refuses `reviewed` when the stored column and the stored
  evidence disagree;
- `evaluate_study_readiness` emits `COMPLETION_NOT_COMPLETE` when they disagree.

A **warning code** is an operator-acknowledgeable note. A **rejection code** is
the pipeline refusing the hand: `acknowledge_codes` will not accept one, and
`derive_completion_status` blocks on `rejection_codes` directly so a hand-edited
`acknowledged_codes` list cannot launder one into a promotion. A rejection clears
only by producing new evidence without it — a `hand_corrections.csv` row appends
`manual_hand_correction` to the timeline's warnings and never replaces them.

The exporter's severity table is built from the validator's, not maintained
beside it: a code the exporter does not recognise is classified as a rejection, so
two drifting tables silently turned nine validator findings — including its two
mildest — into permanent, unclearable rejections.

A source-fact correction is recorded in the evidence, not only in the column.
`_invalidate_hand_derivatives` appends `source_facts_corrected`, and
`_flag_hand_for_debugging` appends `flagged_for_debugging`, each as an
acknowledgeable warning, so the demoted column agrees with its own evidence, the
Source warnings panel renders and gives the operator the action the blocker text
promises, and replaying the unchanged evidence cannot silently restore `complete`.
`partial` stays `partial` throughout: no correction restores missing footage.

A blocker never names an action the product cannot perform. When the evidence
carries a rejection code, `COMPLETION_NOT_COMPLETE` says plainly that only a new
reconstruction clears it, because `acknowledge_codes` refuses rejections and no
correction writer rewrites `rejection_codes`. It says the same, for the same
reason, when the evidence is *unreadable* and when it is readable but carries no
code at all. Every hand the v13 migration classified is in the first case: the
migration leaves `completion_evidence` at `{}` rather than fabricating evidence for
historical hands, so there is nothing to correct and nothing to acknowledge, and no
writer reachable from the UI attaches evidence to an existing hand. Telling those
operators to "acknowledge each remaining source warning in the Source warnings
panel" named a panel that is not even drawn — it renders only when the evidence
carries a code — and following it added `ACCOUNTING_NOT_AUTHORITATIVE` instead of
removing anything. `STALE_SOLVER_EVIDENCE` names a
Delete stale run control that now exists (`db.delete_solver_run`, refused while
the run is live), because re-running the solve is unavailable on a hand a
correction left solver-ineligible. "Live" is one definition shared by
`delete_solver_run` and `fetch_active_solver_runs` (`queued`, `running`,
`cancelling`); they used to disagree about `cancelling`, so a run the store itself
called active could be deleted from under the worker winding it down. A run that is
only `cancelling` gets its own blocker text: nothing was invalidated by a
correction, there is no saved result, and the Delete stale run control is not drawn
while a cancellation is in flight, so it says to wait for the cancellation instead.
`STALE_COACHING_EVIDENCE` needed the identical escape hatch and did not have it
until round 9. Its only stated clearing action was "re-run coaching in Study →
Coach", and the only writer that satisfies it is `db.create_coaching_response` —
the store had no discard writer for a coaching review at all, and the Coach button
is disabled with no LLM provider configured. That is not a corner case:
`import_session` stales every imported coaching row by construction, so an
operator importing a colleague's session before setting up a provider — or
offline, or with a rotated key — held a hand that could never become study-ready
behind a blocker naming an action they could not take. `db.discard_stale_coaching`
is the twin of `delete_solver_run`: it deletes only the stale rows, covers both
retained tables (`coaching_reviews` and the legacy `hand_reviews`, since the
blocker considers both and clearing one would leave it standing), and never
touches a current review. The Discard stale coaching control is drawn from the
blocker rather than from the tab's own list, because a hand staled only in the
legacy table shows the blocker on a tab that does not render those rows.
`STALE_COACHING_EVIDENCE` and `STALE_SOLVER_EVIDENCE` also now use the same
tie-break (`>=`): on equal timestamps the solver blocker used to stand while
naming a re-run that had just been performed.

The rule cuts the other way too, and `UNSUPPORTED_TABLE_LAYOUT` was breaking it in
that direction: one fixed sentence covered four different causes and was false for
two of them. "Only a new reconstruction clears this … Correcting the table size by
hand does not clear it" is exactly right about the evidence-borne causes, because
no writer reachable from the UI rewrites `layout_supported` or the evidence's own
`table_size`. It was drawn verbatim over two causes that are not evidence-borne at
all: `hand.table_size` is an ordinary editable column, so typing it in Correct hand
facts removed that line and, when it was the only one, cleared the blocker while
the text said that action does nothing; and `hero_seat_mismatch` is an
acknowledgeable warning, so one press of Acknowledge cleared it the same way.
Withholding the action that works while naming one that does not reads as "this
hand is beyond repair" to an operator holding the fix, so the clearing action is
now composed from the causes actually present. The recorded table size is also
compared against the reconstructed one, which nothing did: the two columns could
disagree outright — 9 seats recorded against evidence for 6 — and the gate was
satisfied by any typed value, so "record the table size" was a box to tick rather
than a fact to state.

A `partial` column its own evidence contradicts states the disagreement instead of
inventing a truncation. The import ceiling deliberately honours a declared
`partial` over a weaker re-derivation and `update_hand_completion` is sticky on it,
so the column can legitimately outrank evidence recording
`partial_start=False, partial_end=False`; asserting "it starts mid-hand" about that
hand was a fact about the operator's recording the product did not have, and the
clearing action then told them to re-import from a complete recording, which is
exactly what they had done.

`import_session` applies the v13 migration's own rule. It re-derives completion
from the evidence and never upgrades past what the payload declared, refuses a
payload that claims `source_type: manual` while carrying readable reconstruction
evidence, and never lands `reviewed` on a reconstructed hand — readiness requires
explicit user confirmation, which is derived per render and cannot travel in a
payload.

**An acknowledgement cannot travel in a payload either**, for exactly the same
reason: it is an operator of *this* database attesting to a code they have read.
`acknowledged_codes` is an input to `derive_completion_status` through
`unresolved_codes`, so a payload that declared a warning *and* acknowledged it was
internally consistent and slipped straight past the "may only weaken" ceiling — it
derived `complete` with an empty blocker tuple, attested to by nobody, and a real
`hero_seat_mismatch` arriving pre-acknowledged silenced `UNSUPPORTED_TABLE_LAYOUT`
while the UI drew no Acknowledge control for it. `import_session` resets the list
and keeps every code, so nothing is lost and the importing operator re-acknowledges
what they accept. This is a deliberate, documented round-trip asymmetry: a v5
export of an acknowledged hand re-imports unacknowledged.

`unreadable_card_columns` is likewise never stored. `_hand_from_row` injects it
when a hand-edited card column cannot be read back, so it describes the *current*
contents of a column and is a derivation, not evidence — but every writer that
round-tripped a fetched hand's evidence persisted it, and once stored it was
permanent, because `_card_blockers` consults it before the live column and no
writer removed it. A hand could carry `INVALID_HERO_OR_BOARD_CARDS` forever while
its board was valid, with the blocker's stated clearing action doing nothing.
`create_hand`, `update_hand_completion` and `import_session` strip it on write, so
it appears the moment a column is unreadable and disappears the moment it is fixed.

`reviewed` never outlives the evidence it was granted on. Every writer that
changes a hand's players, actions, or settlement returns a promoted hand to
`needs_correction`, and the settlement writers additionally stale the retained
coaching and solver output the corrected award invalidates — demoting the column
alone was cosmetic, because one click restored it while wrong-winner coaching was
still labelled current. That includes `update_hand_completion`: the evidence is
precisely what a promotion was granted on, so an evidence write that re-derives
below `complete` demotes too. It used to re-derive the column and leave
`review_status` alone, landing a hand at `uncertain` — with a pipeline *rejection*
in its own evidence — still labelled `reviewed`, still counted in the landing
hero's "N% marked reviewed", in a pair `update_hand_status` refuses to create.
Acknowledging a warning that leaves the hand `complete` is not an invalidation and
does not demote.

Import may only ever *weaken* what a payload declared, across the whole ordering
`complete` → `uncertain` → `partial`, not just at the `complete` step. A declared
`partial` therefore survives a weaker re-derivation, which is what every stripped,
corrupt, pre-v5 or future-`evidence_version` payload produces, and
`update_hand_completion` is sticky on `partial` so no acknowledgement can launder
it back up.

Every demotion is written into the evidence, not only into the column. That
includes the debugging-issue flag, which previously demoted `complete` to
`uncertain` while the stored evidence still derived `complete`, leaving a hand
whose blocker named a Source warnings panel that was never drawn.

### Tests

- Fresh schema and every historical migration path.
- Rollback from a migration failure without partial schema state.
- Manual-hand compatibility.
- Existing CV-hand conservative migration.
- Export/import v1–v5.
- Readiness truth table covering every blocker and combination.
- UI attempts to bypass readiness through direct status controls.
- Forged-payload bypasses (`tests/test_phase1_readiness_bypass.py`): a declared
  completion status, a declared `reviewed` over an open issue, a laundered
  rejection acknowledgement, and a `complete` column its evidence contradicts.
- Adversarial round-2 regressions (`tests/test_phase1_adversarial_round2.py`):
  the rejection-code contract across a corrections CSV, the exporter/validator
  severity-table agreement, concurrent migration, a NULL `source_type`, a refused
  newer database left byte-identical, an unreadable version stamp, corrupt
  `completion_status` and `source_type` columns, `reviewed` outliving a settlement
  or roster edit, import promotion and completion upgrades, session-level coaching
  survival through `import_hands_into_session`, the correction/acknowledge loop,
  the layout blocker's stated clearing action, and the two UI surfaces that
  presented a blocked hand as study-ready.
- Adversarial round-3 regressions (`tests/test_phase1_adversarial_round3.py`): a
  pre-v5 `roi_profiles` table that bricked every open, the import ceiling across
  the whole completion ordering, a `RecursionError` escaping the evidence parser,
  a settlement award change staling coaching and solver runs, the rejection-code
  and stale-solver clearing actions, the debugging flag's evidence record, the
  pinned pre-migration snapshot surviving rotation, both halves of
  `is_reconstructed_hand`, the exporter's real layout evidence, and the Study
  page's own readiness composition and confirmation lifetime.
- Adversarial round-4 regressions (`tests/test_phase1_adversarial_round4.py`): the
  chip denomination that was also the reconciliation tolerance, an import payload
  that supplied that tolerance itself, declared dead money disclosed as a source
  warning, non-finite evidence floats and strict-JSON export, a hand-edited card
  column that took the whole session's hand list down, the resolved-issue and
  live-solver-run contracts, the two stale-evidence tie-breaks, the exporter's
  layout rule in its negative direction, the audited and bounded pinned snapshot,
  the unwritable-backup message, and the documented retention/evidence constants.
  `tests/test_schema_v13_migration_paths.py` now seeds manual rows at three
  different review statuses: with only a `reviewed` one, "manual hands keep their
  review_status untouched" was pinned by a tautology, and a migration that
  promoted every manual hand passed the whole suite.
- Adversarial round-5 regressions (`tests/test_phase1_adversarial_round5.py`): the
  rake-rate/chip-unit product that still reached half the pot and was still applied
  to pre-rake quantities, the waived rake that still bought slack, that tolerance
  arriving through an import payload, an honest whole-chip rake that must still
  absorb its own rounding, a payload that pre-acknowledged its own warnings, the
  read-time unreadable-card marker that became a permanent unclearable blocker after
  one round trip, a live schema-13 database whose stamp row was gone (and the
  pre-versioning database that must still migrate), an evidence write that weakened
  a hand while it stayed `reviewed` (and the acknowledgement that must not demote
  one), a recorded hero result with no hero seat, awards exported without a
  settlement row, and the migrated hand told to use a panel that never renders.
  Three mechanisms that no test pinned are now pinned directly, each having survived
  deletion against the whole suite: the `is_known` gate inside
  `derive_completion_status`, the `source_type` half of `is_reconstructed_hand`
  (the round-3 test named for it varies only `completion_status`, so it proved that
  half twice), and the `hero_seat_mismatch` branch of `_layout_blockers`.
  `tests/test_hand_issue_queue.py` and `tests/test_persistence_integrity.py` now
  drop the v13 columns when they rewind a database to schema 11 and 7: leaving them
  in place fabricated a file whose schema was ahead of its own stamp, which is
  precisely the state a lost stamp produces and which `init_db` now refuses.
- Adversarial round-6 regressions (`tests/test_phase1_adversarial_round6.py`): an
  interrupted FIRST start on a brand-new file (the base-table DDL committed
  outside the migration transaction, so an interruption left schema-13 structures
  with no stamp and every later start refused the file forever, naming a backup
  the product had correctly never taken); the residual `min(unit, rake taken,
  gross − rake) / 2` slack on the hero result and on declared awards, by the
  direct settlement route and through a single `import_session` call; the three
  pre-rake comparisons PLAN.md names as the round-4 fix, each of which could be
  handed the slack with the whole suite green; one blank `Observed payout` cell
  disabling the declared-award check for every other row on the hand; a
  re-declared pot winner recorded as a `HandCorrection` and disclosed in the
  completion evidence, with the reconciler's own derived-refund write proved not
  to count as one; an unreadable card column that must still block after an
  export/import round trip; a rejection code that must not be described as an
  acknowledgeable source warning; the ledger-error clearing action; and four
  mechanisms that no test pinned — the `hand.table_size is None` branch of
  `_layout_blockers`, `app.hand_study_readiness` (the single composer behind every
  readiness surface, which had no test at all), the multi-hero accounting
  cross-check, and rotation's promise never to delete a snapshot the product did
  not write.
- Adversarial round-7 regressions (`tests/test_phase1_adversarial_round7.py`): a
  declared rake policy carrying a fabricated hero result on a reconstructed hand,
  by the settlement route and across an export/import round trip, with the
  zero-rake and acknowledged cases pinned so the disclosure can still be cleared
  and cannot fire on a policy that takes nothing; an imported hand-level, legacy
  and session-level coaching review that declared `is_stale: false` in the
  payload; an imported settlement restating its own `rake_amount` and `net_pot`
  4.9 chips off its own ledger, and the same restatement at four chip
  denominations by the direct route; a dead-money amount raised by four orders of
  magnitude after its acknowledgement, alongside the idempotent re-save that must
  keep it; a concurrent writer on a rollback-journal database, which
  `PRAGMA journal_mode = WAL` reported as a raw "database is locked" without ever
  consulting the 30-second busy timeout, plus three real processes opening one
  fresh file together; a read-only mount named instead of reported as a lock; a
  WAL-mode backup audited without writing sidecars beside it and without being
  failed on a read-only mount; a deleted hand that must stale the session
  coaching which summarised it; an unreadable `tags` and `hand_settlements.warnings`
  column that must degrade instead of raising out of a fetch; a version stamp
  ahead of the schema it describes; and `create_hand_player`'s settlement
  invalidation and coaching/solver staling, which nothing pinned.
- Adversarial round-8 regressions (`tests/test_phase1_adversarial_round8.py`): the
  chip denomination that was also the pot-splitting granularity, swept across the
  whole dial by the settlement route and through a single `import_session` call,
  with the genuine odd chip pinned in both directions so the fix cannot be
  mistaken for deleting the mechanic; an acknowledged rake attestation that must
  re-raise when only the rounding unit moves, alongside the idempotent re-save
  that must keep it; a swapped odd-chip order recorded as a `HandCorrection` and
  disclosed in the evidence, with the unchanged re-save proved not to count as
  one; `create_settlement_entry`'s award disclosure and its demotion/staling,
  neither of which anything reached; the layout blocker's clearing action in all
  three of its shapes and the recorded/reconstructed table-size disagreement; the
  reconstruction evidence the confirmation checkbox names, both as the pure
  transformation and rendered on the Study page; the import card-restore guard,
  whose removal the whole suite used to survive; `app.hand_study_readiness`'s
  legacy `hand_reviews` input, which feeds two promotion surfaces and was the one
  input of that helper with no test; a pinned pre-migration snapshot stamped
  behind the live schema, which the audit must not fail; the landing hero's
  "marked reviewed" wording; and the Add-hand writer's stored completion column,
  read raw because `_hand_from_row` repairs the pair on every read.

- Round-11 regressions (`tests/test_phase1_declared_inputs_and_consumers.py`):
  a first declared pot winner measured, disclosed and blocked in either
  direction, plus three further shapes of the same family (a re-ordered chop that
  moves the odd chip with every settlement field identical, a side-pot winner
  declared on pot 1, and a winner arriving in an import payload); the
  neutral-declaration completeness property, which sweeps six unrelated
  settlement rows and award sets over one recording and asserts the fully neutral
  ledger never moves; the verdict half of `_is_dependent`, reached by a real hand
  shape for the first time; the injectivity and round-trip of the measured
  movement string; each of the four terms the attestation fingerprint digests,
  including the gross-pot term at a binding rake cap; the AST scan that fails on
  any consumer reading `is_authoritative` outside six named places, with the count
  derived from the set; every
  derived-figure surface (session stats, the list-view substitution,
  `_hero_ledger_result`, the prompt math facts, solver eligibility) refusing an
  unattested reconciliation and publishing it once answered; a manual hand
  carrying an ordinary room rake being established with no control to press; and
  the attestation lapsing when the declaration changes.

### The accounting tolerance

There is one tolerance, and it is float-representation noise.

**Every recorded figure on a settled hand is compared exactly.** The
gross pot, the observed final pot (`hands.pot_size`), declared refunds — uncalled
bets are returned before the drop — the hero's net result, and every declared
award are judged against float-representation noise alone. No rake policy
whatsoever can excuse a disagreement about how many chips went into the pot, how
many came back out, or which seat they went to.

Making only the *pre-rake* quantities exact was not enough, and an earlier
revision of this section claimed it was: "that single change is what closes the
whole family: the attacks all worked by making the *pot* check permissive". The
round-6 attack never touched a pot check. It set `rake_rate` to 1.0, a `rake_cap`
to half the pot and a `rake_rounding_unit` to match, which maximises
`min(rounding_unit, rake taken, gross pot − rake taken)` at `gross_pot / 2` and
so the halved slack at `gross_pot / 4`, and then spent it on `hands.hero_bb_won`
and on a declared award — landing a study-ready hand, by the settlement editor
and by a single `import_session` call, whose recorded hero result was a quarter
of the pot away from its own action line.

**Round 7 removed the last of the slack.** Rounds 4–6 left the recorded
`rake_amount` and the recorded `net_pot` a tolerance of
`min(rounding_unit, rake taken, gross pot − rake taken) / 2`, on the argument
that both are restatements of the stored rake policy rather than observations of
the hand, and that `persist_reconciliation` rewrites both from the ledger so the
slack could only excuse reading the same policy one rounding step earlier. The
second half of that argument was false off the settlement-editor path.
`import_session` never calls `persist_reconciliation`, and every readiness surface
reads through the read-only `reconcile_persisted_hand`, so on an imported payload
the recorded pair was never rewritten and was judged against a tolerance the same
payload supplied — `rake_rate`, `rake_cap` and `rake_rounding_unit` together.
One `import_session` call landed a hand recording a rake 24.5% of its own gross
pot away from its own action line as reconciled, authoritative and study-ready
with an empty blocker tuple, and re-exported the forged pair. No honest producer
writes a disagreeing row: the settlement editor nulls both figures and the
reconciler writes them from the ledger. A stored policy that disagrees with the
stored amount beside it now fails closed onto the same one-click clearing action
— save the settlement — and `tests/test_phase1_adversarial_round4.py`,
`round5.py`, `round6.py` and `round7.py` all fail if the tolerance is restored,
which no test detected before. Four earlier versions failed here. Feeding
`rake_rounding_unit` in directly made a chip denomination with no upper bound the
tolerance for every check at once. Replacing it with
`min(rounding_unit, gross_pot × rake_rate) / 2` only bounded it: `Rake %` accepts
100, so the product still reached `gross_pot / 2`, and a hand whose recorded pot
was 50% larger than its own action line still reconciled. Deriving it from
`ledger.rake` halved the ceiling again and fixed `no_flop_no_drop` — a stock 5% /
whole-chip / no-flop-no-drop policy had been buying slack on a hand it
definitionally rakes at zero — but left it gating the two anchors above.
Narrowing *which comparisons it reaches* was the round-6 step, and round 7
narrowed it to nothing: bounding a tolerance whose width is set by the data it is
judging never closes the family.

**A declared award is checked per identity, and a blank amount disables nothing.**
`Observed payout` is an optional column, so an award row may name a winner without
an amount. The comparison used to be guarded by "every award on the hand has an
amount", so one blank cell anywhere skipped the whole per-identity check and a
declared payout of 9999 against a derived 250 reconciled. An identity whose award
amounts are all present is compared exactly; an identity with a blank among them
must still satisfy the half of the claim it made — the amounts it *did* declare
cannot already exceed its whole derived payout, because the blank rows can only
add more.

**Re-declaring who won a pot is a source-fact correction.** The declared winner is
the sole input the derived payouts, and therefore the hero-result cross-check, are
computed from, so flipping it in the settlement editor could clear
`ACCOUNTING_NOT_AUTHORITATIVE` while leaving no `hand_corrections` row, no
evidence disclosure and `completion_status` still `complete`.
`replace_settlement_entries` now records a `settlement_award_update` correction
and writes `source_facts_corrected` into the evidence whenever a hand's existing
award declarations change. The reconciler's own derived-refund write is excluded
by construction — it compares award rows only, and an empty "before" is a first
declaration rather than a re-declaration — because treating that as an operator
correction would demote every hand the reconciler touched.

**A recorded hero result with no Hero seat is not a passed check.** The hero
cross-check used to be skipped entirely when no roster row was flagged, so
unticking `Hero` in the player editor — an ordinary correction — *deleted* it: a
hand recording a fabricated `hero_bb_won` reconciled against nothing, became
authoritative, and stayed renderable at the fabricated value everywhere, because
the derived result is only substituted when a hero row exists. A hand that records
a hero result or hero cards with no hero seat now raises an issue naming the
control that fixes it.

**The chip unit had a second job nobody had named, and it took two rounds to
take it away.** `rake_rounding_unit` is an unbounded operator field ("Chip unit"
in the settlement editor, and a verbatim value in an import payload). Its
documented job is rounding the rake, which is a real room rule — a house that
drops whole dollars against a 0.50 blind is ordinary. Its undocumented job was to
be the granularity a *chopped pot* was divided at: `_split_pot` rounded each
winner's share down to it and gave every leftover chip to the first name in
`odd_chip_order`, so a unit at or above the pot collapsed the base share to zero
and pushed the whole pot to one seat. With the rake rate at zero, neither
declared-chips disclosure fires and no correction is recorded, so raising one
number redirected a chop, moved the derived hero result by half the pot, and made
a fabricated `hands.hero_bb_won` reconcile exactly — study-ready, promotable, and
reproducible in a fresh database from a single `import_session` call. It was a
continuous dial, not a switch: on a 20-chip chop, 3 paid the hero 11, 7 paid 13,
20 paid the lot.

Round 8's fix bounded the declared unit by the greatest common divisor of the
observed contributions, honouring it when it divided that gcd and falling back to
the gcd otherwise, and this document claimed the dial was gone. **That claim was
wrong, and round 9 found it wrong three times independently.** The rule still let
the declared field *select* a split, because every divisor of the gcd stayed
reachable and divisors do not agree once the pot is anything but two equal halves.
The round-8 regression tests used exactly two equal contributions, where every
divisor halves the pot evenly, so they could not see it. On three seats
contributing 8 each and a two-way chop of the 24, units 0.01/1/2/4 paid 12/12
while 3, 5, 8 and 100 paid 16/8 — the hero's derived result doubled, the
fabricated `hero_bb_won` reconciled, `study_ready` came back true with an empty
blocker tuple, and the same landed from one `import_session` call. The fallback
made it worse rather than safer: a unit that did *not* divide the gcd fell back to
the gcd itself, which is the most distorting denomination available, so the
attacker's best move was to declare a number that fit nothing. And the bound was
read off per-player **totals**, not off the action line, so a hand of six 5-chip
bets admitted a unit of 10 — a denomination no amount on the hand ever showed.

The direction of the bound was the underlying error. A gcd is an *upper* bound on
the denomination in play: every amount is a whole multiple of the real chip, so
the real chip divides the gcd and may be far finer, and three seats each
committing 8 demonstrate nothing whatsoever about 8-chips. Splitting at an upper
bound maximises the redistribution, because rounding the base share down sheds up
to a whole unit from *every* winner before the round-robin hands those chips back
from the front of the order.

So the granularity is now derived from the evidence alone and the declared unit is
confined to the rake. `_split_granularity` takes the settled contributions and the
declared dead money — never the unit, and nothing computed from it — and returns
the **finest** decimal place any of those amounts is written in, capped at one
whole chip. An amount of 49.75 demonstrates that hundredths exist, so a chop of
that hand is derived in hundredths; whole-chip amounts derive a whole-chip chop.
The odd chip itself is real and is kept: chips are indivisible, so a 21-chip pot
chopped two ways is genuinely pushed 11/10, and deriving 10.5/10.5 would raise a
false blocker against an honest declared award. What it is no longer is a dial. The
derived split is now identical at every declared unit, at every rake rate, and the
most any granularity can move a seat is under one chip — against half the pot,
which is what the declared field could move. This is also what makes the
zero-rate disclosure gate honest: `upsert_hand_settlement` records
`DECLARED_RAKE_CODE` only when `rake_rate > 0`, on the stated grounds that "a cap
or a rounding unit on a zero rate takes nothing", and until round 9 that sentence
was false and switched the one audit channel off exactly where the attack was
cheapest. `_split_pot` always distributes the full amount, so no granularity can
change the gross, the rake, or chip conservation; only who receives an odd chip,
and that is decided by the audited `odd_chip_order`. The fix is in the ledger,
which is the one place both the settlement editor and `import_session` read
through.

**No pot may be raked past its own size.** `_allocate_rake` rounded every
non-final pot's proportional share down to the declared unit and charged the whole
leftover to the last pot with no cap at that pot's amount. An ordinary hand under
an ordinary policy — a 149.25 main pot and a 0.50 side pot, 5% capped at 5 with a
whole-dollar drop — took 4 from the main pot and charged the leftover 1 to a pot
of 0.50, so that layer rendered as amount 0.50 / rake 1.00 / net −0.50 and paid
its winner minus half a chip. `is_balanced` stayed true because the negative
preserved `paid + rake == gross`, and `is_legal` and the warning list stayed
empty, so a hand whose side-pot winner also won the main pot certified as
authoritative and study-ready. Where the two pots had different winners the hand
became *permanently* unreconcilable: the derived payout was negative,
`SettlementEntry.amount` is `ge=0` so the operator could not declare it, and
`ACCOUNTING_NOT_AUTHORITATIVE` named a save that could never clear. Each share is
now capped at its own pot and the rounding leftover is offered to the layers in
order, each taking only what it still has room for; the rate is bounded at one and
the total at the gross, so a feasible allocation always exists.

**The odd-chip order is a declared fact and is audited as one.**
`_declared_award_state` sorted its claims alphabetically and documented entry
order as deliberately excluded, but `reconcile_persisted_hand` sorts award rows by
`(pot_index, entry_order)` to build `odd_chip_order`, so on a pot that cannot
divide evenly, swapping two rows in the editor's Order column moved the odd chip
between seats and flipped the derived hero result with `before == after`: no
`settlement_award_update` correction, no `source_facts_corrected` in the evidence,
and `ACCOUNTING_NOT_AUTHORITATIVE` cleared with the audit trail recording nothing.
The snapshot now carries the declared order. Re-saving an untouched editor writes
the same orders back, so an idempotent save still compares equal.

**Both public writers of `settlement_entries` disclose a re-declared winner.**
`create_settlement_entry` demoted and staled but recorded no correction and wrote
no evidence code, so adding a second award row for a pot — which turns a single
winner into a chop and moves every derived payout — cost an acknowledgement
through `replace_settlement_entries` and nothing through its sibling. The public
writer now takes the same before/after snapshot; `replace_settlement_entries` uses
a private inserter, because the public writer's per-row snapshot would record a
correction per entry for an unchanged declaration.

Dead money is the other input that can move the derived pot, and unlike the chip
unit it does so honestly — the ledger models externally-contributed chips exactly
as declared. On a **reconstructed** hand it is still an operator assertion the
pipeline cannot corroborate, and it is the one free parameter that can always be
tuned until the recorded pot equals the derived one. Declaring it therefore writes
`declared_unobserved_chips` into the hand's completion evidence as an
acknowledgeable warning, so `UNRESOLVED_SOURCE_WARNING` surfaces the reliance and
the operator accepts it as an auditable correction rather than the reconciled
verdict resting on it silently. Manual hands are untouched: antes, dead blinds and
straddles are ordinary facts there, with no pipeline claim to contradict.

**A declared rake is the mirror image, and round 7 gave it the same disclosure.**
Dead money *creates* chips the observed action line never saw; a rake policy
*destroys* them. The rake is the more dangerous of the two, because it does not
widen a tolerance — it moves the DERIVED side of the hero-result and declared-award
cross-checks, so comparing those exactly (the round-6 fix) detects nothing at all.
`rake_rate` is bounded only by 1.0, `rake_cap` is unbounded above, and both arrive
verbatim in an import payload; choosing `rake = gross − contribution − target`
makes any `hands.hero_bb_won` in `[−contribution, gross − contribution]` reconcile
exactly. A reconstructed hand recording a hero *loss* of 10 on a pot the hero won
was authoritative, study-ready with zero blockers, and promotable to `reviewed`.
A settlement whose rake policy actually takes chips now writes
`declared_unobserved_rake` into the completion evidence of a reconstructed hand,
exactly as dead money does: the operator may attest to it, and the reconciled
verdict may rest on it, but not silently. The rake taken from a pot is never
observable from the action line, which is precisely why it needs the attestation.

**Rounds 7, 8 and 9 each repaired one shape of this and the next round found
another, so round 10's repair replaced the mechanism.** Every disclosure above was
raised by a list of per-field conditions, and a field list can only ever describe
the shapes already demonstrated: `rake_rate > 0` announced policies that took no
chips at all while missing every combination not yet drawn, and the whole gate
lived inside one writer, so a settlement row written any other way was disclosed
nothing. Readiness is now gated by `ACCOUNTING_ASSUMPTION_DEPENDENT`, derived per
read from a dual reconciliation (see "Assumption-dependent reconciliation"), and
the codes below are retained as the writer-side audit trail with their triggers
re-derived from the same measurement. The paragraphs that follow describe why the
disclosure exists; what raises it is no longer a field.

**Both disclosures attest to a quantity, not to the act of declaring one.**
`_record_declared_chip_adjustment` compares the settlement row as it was *before*
the write, so re-saving the same policy — which `persist_reconciliation` does on
every reconcile — keeps the acknowledgement, while changing the declared amount
re-raises the warning. Before round 7 both sides of that comparison were booleans:
acknowledging 0.5 chips of dead money licensed every later figure, and raising the
declaration to 4980 to make a fabricated 5000-chip pot reconcile re-raised nothing.
Round 8 found the rake half of that invariant broken by an omission rather than a
boolean: `_rake_policy` returned the rate, the cap and `no_flop_no_drop` and left
out `rake_rounding_unit`, which `_compute_rake` rounds the raw rake *down* to. On
an 80-chip pot at a declared 100% rate, moving the unit alone from 0.01 to 81 moved
the rake from 80 to 0 and flipped the derived hero result by the whole pot, while
the policy compared equal, nothing was re-disclosed, and the acknowledgement earned
against an 80-chip rake described a rake that was no longer in force. The tuple is
the set of fields that decide *how many chips the ledger takes out*, so every such
field belongs in it.

Manual hands are exempt from both, and that is a stated decision rather than an
oversight: `_record_source_correction_in_evidence` returns early on them, a manual
hand carries no completion evidence to write into (import *refuses* a payload that
declares `manual` while carrying reconstruction evidence), and every figure on
such a hand — the action line, the rake and the hero result alike — is the same
operator's own entry, so there is no independent claim for a disclosure to
protect. The consequence is recorded under "Known non-blocking gaps".

### Exit gate

- Every reconstructed hand has an explicit completion classification. **Satisfied**
  — the v13 migration classifies every existing row, `Hand`'s source-aware
  validator classifies every new one, and the CV exporter attaches versioned
  evidence so a genuinely complete reconstruction can reach `complete` instead of
  being stranded at `uncertain`.
- No partial, uncertain, unreconciled, open-issue, or stale-evidence hand can be
  presented as study-ready. **Satisfied** — `update_hand_status` refuses the
  promotion in the store, `guarded_update_hand_status` is the single UI writer
  behind every review-status surface, and the Study status control does not
  offer `reviewed` while any blocker stands.

### Verification record

Re-run after the mid-hand button-position repair, on the working tree, macOS 24.6.0
(darwin/arm64). Re-derived from the code, not carried forward: the previous
revision of this table recorded `1326 passed` while the tree produced `1412`,
because the round-14 repairs and their 20 regressions landed without this
section being re-run — which is round 15's own finding about this document, and
the reason the count below was taken from a fresh run rather than edited.

The full tree was stable during this run. OpenCV 5 changed the synthetic text
rasterization used by six calibrated OCR tests, so the declared dependency is
now capped below 5; the result below uses OpenCV 4.11 and NumPy 2.2.6.

| Command | Result |
| --- | --- |
| `python -m pytest -q` | `1509 passed, 1 skipped` |
| `python -m ruff check .` | `All checks passed!` |
| `python -m mypy` | `Success: no issues found in 11 source files` |
| `git diff --check` | no output, exit 0 |
| `cmp AGENTS.md CLAUDE.md` | no output, exit 0 |

What that mypy line does and does not cover, stated rather than implied: the 11
files are `persistence/completion.py`, `services/study_readiness.py`,
`services/hand_accounting.py`, `math/analytics.py`, `solver/eligibility.py` and
the six `ui/` modules. `app.py`, `persistence/db.py`,
`persistence/import_export.py` and `math/accounting.py` are NOT type-checked, and
`follow_imports = "skip"` means even the checked modules are checked against
stubs of their imports. Widening it further is a Phase 0 hygiene item, not a
Phase 1 gate.

A fresh database initialises at schema version 13 with `hands.completion_status`
and `hands.completion_evidence` present. A **copy** of the repository database
(`poker_tracker.db`, already at 13) re-opens with every table's row count
unchanged, no table added or dropped; the real file was SHA-256-hashed before and
after and is byte-identical. Reading its version stamp still leaves `-shm`/`-wal`
sidecars beside it, which is the documented gap below, not a write to the database.

The suite grew from the 442 tests of the Phase 0 baseline to 1509. That count
includes the CV suites, so the number will drift; the Phase 1 files in it are
`test_study_readiness*.py`,
`test_completion_evidence.py`, `test_schema_v13_migration_paths.py`,
`test_phase1_readiness_bypass.py`, `test_review_promotion_surfaces_ui.py`,
`test_phase1_assumption_dependence.py`,
`test_phase1_declared_inputs_and_consumers.py` and
`test_phase1_adversarial_round2.py` … `round15.py`, and
`test_operator_state_isolation.py`. The single skip
is `tests/test_ocr_readers.py::test_without_chip_template_chip_would_join_run`,
a negative control that documents why the chip affix exists; it skips when the
synthetic chip glyph lands below the classifier's confidence floor, in which case
the misread it is demonstrating does not occur and there is nothing to assert.
It guards no product behaviour on its own — `test_chip_breaks_out_of_digit_run`
and `test_genuine_decimal_still_reads` cover the shipped path unconditionally.

`data/backups` was hashed before and after a full suite run and is byte-identical
across it, and so is `poker_tracker.db` itself: since round 12 the suite claims
`POKER_DB_PATH` and `POKER_DATA_DIR` before its first `poker_tracker` import, so
no test can reach any operator root, and `test_operator_state_isolation.py` fails
if one stops being redirected. MyPy's file list is deliberately narrow (11 files, configured in
`pyproject.toml`, and `follow_imports = "skip"` on top of that); it is not
whole-repo type coverage and must not be reported as such. `app.py`,
`persistence/db.py`, `persistence/import_export.py` and `math/accounting.py` are
not type-checked at all.

### Known non-blocking gaps

Recorded here rather than fixed, with no effect on correctness, safety, data
integrity, or the release claim:

- The dependence rule costs a read up to four extra ledger builds on a SHOWDOWN
  hand declaring a rake, dead money and pot awards, and five on a FOLD WIN, where
  `_forced_winners` fires and a second neutral pass is built against the forced
  winners — the commonest hand shape there is. (The measured 429.7us against
  163.0us for the pre-rule shape, a factor of 2.64, was taken on the showdown
  shape. A previous revision of this entry stated the ceiling as four for every
  shape; the ceiling is now derived from a build counter in
  `test_the_documented_dependence_cost_ceiling_is_the_measured_one` rather than
  from prose.) `math.analytics.compute_session_stats`
  calls `reconcile_persisted_hand` once per hand with no cache, so the Insights
  page pays that multiple on every hand of a session. The per-render
  `AccountingCache` in `app.py` does not reach it. It is a hot-path caching
  question rather than a correctness one — every figure it produces is the same
  figure — and it belongs with the other performance work rather than in the
  Phase 1 gate.
- The `payout` term of a measured dependence is an unsigned magnitude — the
  largest absolute per-seat change — while its four siblings (`gross`, `rake`,
  `net`, `hero`) are signed differences of a single figure. Two seats' payouts can
  move in opposite directions under one declaration, so there is no single
  direction to state, and the term also collapses a per-seat vector into one
  scalar, so which seat moved is not recorded in the acknowledgement code.
  `describe()` now words it as the magnitude it is ("the largest payout for any
  seat by N chips"), so the display defect — the same rendered token meaning
  "40 lower" under a rake and "75 higher" under dead money — is fixed. The
  information-content limit is not, and is recorded rather than repaired: making
  the term a signed per-seat vector would change every dependence code and lapse
  every stored attestation, which is a conservative failure but not one worth
  taking for a low-severity display property. A round-15 sweep of 69,156 states
  found zero code-set collisions covering different figures, because the signed
  `hero` term and the awards fingerprint pin the payout distribution
  independently, so there is no known attestation-inheritance consequence.
- `app.show_saved_hands` is currently unreferenced. It is routed through
  `guarded_update_hand_status` and covered by
  `tests/test_review_promotion_surfaces_ui.py`, and it holds the only
  per-hand-export control, so it is retained rather than deleted. Four of the
  five review-status surfaces are reachable in the running app today. A previous
  revision of this entry recorded that it also held the only delete-hand
  control and judged that fact non-blocking; round 13 showed the judgement was
  wrong — nine clearing actions depended on that control existing — and every
  hand row rendered by `render_hand_results` now carries a reachable 'Delete
  hand' control of its own, pinned by an AppTest regression.
- The first hand of every recording is classified `partial`: the segmenter cuts a
  hand only at a positively detected fresh deal, so the recording's opening hand
  has no observed start boundary. This is deliberate under-claiming, not a
  detection failure, and Phase 5 may add a pre-deal boundary read that proves it.
- A version 5 JSON export cannot be read by the previous release. The payload is
  strictly additive, but `SUPPORTED_IMPORT_VERSIONS` there stops at 4 and there is
  no option to emit a v4 payload. Documented in `README.md`.
- `completion_evidence.layout_supported` is not wired to a registry of certified
  ClubWPT geometries. The CV exporter sets it from "a table size was resolved and
  no state disagreed about the hero seat", and no code anywhere compares the
  recording's resolution, crop, scale, or client skin against a validated
  profile. This entry previously claimed the blocker over-blocks — "the current
  card-only YOLO path resolves no player rows, so `table_size` is `None` and every
  hand it exports carries `UNSUPPORTED_TABLE_LAYOUT`" — and that was simply wrong
  about the code that exists. The spine does resolve player rows: all 10 hands
  across both committed recording fixtures export `layout_supported=True` with a
  resolved `table_size`, so `UNSUPPORTED_TABLE_LAYOUT` fires on none of them.
  `tests/test_phase1_adversarial_round3.py` pins that real behaviour so the two
  cannot drift apart again, and `tests/test_phase1_adversarial_round4.py` pins the
  rule's negative direction directly on `_completion_evidence_for_hand` — with only
  the positive fixtures, replacing the whole expression with `layout_supported=True`
  passed the entire suite. The blocker is therefore inert on the current
  pipeline, not conservative: it will only start doing work when the certified
  registry lands. Nothing else changes as a result — the hands in question are
  still gated by completion, accounting, source-warning and confirmation
  blockers — but the layout gate must not be counted as protection it does not
  currently provide. `UNSUPPORTED_TABLE_LAYOUT`
  now says plainly that only a new reconstruction clears it — correcting the table
  size by hand does not, because `update_hand_facts` cannot write pipeline
  evidence and typing a seat count is not proof the geometry was read correctly.
  Building the certified-geometry registry is Phase 2 corpus work ("at least one
  unsupported geometry used to test safe rejection") and Phase 5 region-detection
  work.
- A pre-v5 payload that carries no completion evidence at all has nothing but its
  declared `source_type` to record provenance, exactly as the v13 migration had.
  Import refuses a payload that declares `source_type: manual` while carrying
  readable reconstruction evidence, which covers everything this build produces,
  but a legacy v1–v4 CV payload hand-edited to `manual` is indistinguishable from
  a genuine manual hand.
- A previous revision of this entry recorded that a manual hand's declared
  `review_status` survives import unchanged. Round 13 removed that behaviour:
  `reviewed` is this database operator's attestation and cannot travel in a
  payload for any declared `source_type`, so every imported hand declaring it
  lands `needs_correction`. The v13 migration still leaves manual rows
  untouched — a migrated database is the same operator's own data, not
  somebody's JSON.
- `_completion_reason` describes a hand stored as `manual` + `uncertain` as one
  that "could not prove this hand was fully reconstructed". The wording is wrong
  for a manual hand, but the pair is only constructible outside `import_session`
  and `_hand_from_row`, it blocks correctly in every case — now proven rather than
  asserted, by `tests/test_phase1_adversarial_round3.py` — and no reachable writer
  produces it.
- `app.show_insights_workspace` computes its "Not study-ready" KPI by running one
  full `reconcile_persisted_hand` plus four per-hand fetches for every hand in the
  database, on every render — roughly nine SQL statements per hand, with no cache,
  no `LIMIT`, and no short-circuit. `study_readiness.py` stays pure and is not the
  problem; the caller is. It is a list-view cost, not a correctness one, and is
  left for the Phase 13 performance pass rather than fixed here.
- The repository database `poker_tracker.db` is already at schema 13 and holds
  **no sessions and no hands** — 1 video, 1 processing job and 18
  `reconstruction_frame_reviews`. Re-opening a copy of it therefore proves the
  file still opens with every row intact, but it exercises no hand-level v12→v13
  migration and is not evidence about a populated operator database. The
  hand-level migration evidence is
  `tests/test_schema_v13_migration_paths.py`, which builds real old-shaped files
  at no stamp, 5 and 11 and asserts the classification and the manual
  `review_status` invariant across three distinct statuses. A populated
  historical database remains untested against this build, and Phase 0's
  "migrated historical database" walkthrough is the place that gets fixed.
- `data/backups` currently holds 33 files, recounted at the Phase 1 close-out
  rather than estimated or carried forward (the previous revision of this entry
  said 29, taken before rounds 5–9 ran; it was 27 at the start of the close-out
  and the six added files are accounted for at the end of this entry): five rotating
  `poker_tracker_<timestamp>.sqlite3` snapshots that are pytest fixtures
  ("Legacy session" at schema 5, "Legacy completion" at schema 12, 4 hands each)
  and not operator data, each now carrying its own `-shm`/`-wal` pair;
  five pinned `poker_tracker-premigration-*.sqlite3` files that are also pytest
  fixtures (three hold a test session at schema 12 with 5 hands — "Legacy session"
  twice and "m" once; two hold an unstamped session "R5" with 1 hand, which is the
  refused-stamp path, correctly snapshotted and correctly not migrated). Pinned
  residue is now at its own `PINNED_KEEP_COUNT` ceiling of five, which is the
  bound working as designed rather than growth. Then eight orphaned `-shm`/`-wal`
  files across four stems whose parent snapshots were rotated away; and three
  genuine pre-Phase-1 operator backups that predate this work and must not be
  deleted
  (`poker_tracker-v7-20260724.db`, `poker_tracker-v8-20260724.db`,
  `poker_tracker-before-real-data-cleanup-20260728.sqlite3`, the last of which
  also carries a stale `-shm`/`-wal` pair from before round 7 stopped the audit
  writing sidecars beside a WAL backup). An earlier revision
  of this entry said "four orphaned sidecars" and omitted the operator backups
  entirely; the counts above were taken from the directory. The cause — an
  autouse redirect registered on pytest's shared `monkeypatch`, which the
  migration-rollback tests reverted mid-body — is fixed in `tests/conftest.py`,
  which now owns a private `MonkeyPatch`; a full suite run leaves the directory
  byte-identical, re-confirmed at the close-out by hashing every file before and
  after `python -m pytest -q` twice.

  The six files the close-out added came from a **manual** read-only `sqlite3`
  inspection of the snapshots, not from the product or the suite, and they are
  worth recording because of what they demonstrate. Those five rotating fixtures
  are still in `journal_mode=wal`, so merely reading one creates its `-shm`/`-wal`
  pair — which is precisely the failure `backup_database` was changed to prevent:
  it now checkpoints and sets `journal_mode=DELETE` on every snapshot it writes,
  and the five pinned pre-migration files on disk are all `delete` and took no
  sidecars under the same inspection. The WAL rotating files are pre-fix residue
  written by an older build, so this is evidence the fix works, not evidence
  against it. It is also the reason the audit must never open a backup for
  writing: on a read-only archival mount that same read would have failed on
  backups that are intact.

  The already-lost snapshots cannot be recovered, and the residue
  is left for the operator to delete rather than removed automatically, because
  nothing in the product may delete a rollback point it did not write. That
  promise had one hole, now closed: rotation selected victims with the glob
  `poker_tracker_*.sqlite3`, which also matches an operator's own
  `poker_tracker_manual_keepme.sqlite3`, so `_rotate` now matches the exact
  timestamped name `backup_database` writes and skips everything else. Pinned
  snapshots are now audited and restore-drilled by `data_health` and retained under
  their own `PINNED_KEEP_COUNT` slots, so this residue is bounded and visible
  rather than invisible and unbounded.
- `_card_problem`'s board-count and hero/board-duplicate branches remain
  unreachable from any writer: `Hand` refuses those values, and `_hand_from_row`
  now blanks a column it cannot read back and records what it held under
  `completion.UNREADABLE_CARDS_KEY`, which is the path a hand-edited row actually
  takes to `INVALID_HERO_OR_BOARD_CARDS`. The branches are retained as defence in
  depth against a future lenient parser and are exercised directly through
  `Hand.model_construct` in `tests/test_phase1_adversarial_round4.py`, so deleting
  one no longer leaves the suite green.
- `_hand_from_row`'s "one damaged row must not make every hand in the session
  unreadable" defence covers the JSON columns and the classification columns, not
  the numeric and enum ones. `tags` and `hand_settlements.warnings` were fixed in
  round 7 — `_parse_json_list` now degrades through `_parse_json_object` like the
  evidence blob beside it — but a hand-edited `review_status`, `pot_size`,
  `table_size` or `confidence_score`, and the equivalent columns on
  `hand_settlements`, `settlement_entries`, `hand_players` and `actions`, still
  raise a pydantic `ValidationError` out of the fetch. No in-product writer can
  produce those values, so this is defence in depth that is partly present rather
  than a live corruption path; degrading them is not free, because the
  conservative value for a numeric observation is not obviously `None` (dropping
  `pot_size` would *remove* a cross-check rather than tighten one), and that
  design decision is deferred rather than guessed at here.
- An imported settlement's `status` is still taken from the payload. It buys
  nothing on its own — `is_authoritative` additionally requires the ledger to be
  settled, balanced, legal and issue-free, and every recorded figure is now
  compared exactly against the derived ledger — but `import_session` does not
  re-derive it the way `persist_reconciliation` would.
- `parse_completion_evidence`'s BLOB-decoding branch is unreachable from every
  in-repo caller — each passes a `dict` from `db._parse_json_object` or a validated
  `Hand.completion_evidence` — but it is the documented behaviour of a parser whose
  contract is that nothing in it raises, so it is retained and now covered by a
  direct test rather than left as unverified defensive code.
- Clearing both `Observed final pot (BB)` and `Hero result (BB)` to Unknown is a
  reachable way to satisfy the accounting gate, and it is disclosed only by the
  generic `source_facts_corrected` warning. On the UI path `persist_reconciliation`
  overwrites `gross_pot`, `rake_amount`, `net_pot` and `is_balanced` with the
  derived values, so those two columns are the only independent observations left;
  deleting both leaves a reconciled verdict resting on nothing observed. It is an
  auditable correction — it demotes `reviewed`, writes a `hand_corrections` row,
  and costs an acknowledgement — but it does not get the dedicated code
  `declared_unobserved_chips` got, so the readiness output cannot distinguish
  "reconciled against the recording" from "reconciled against nothing".
- Clearing both observed columns is still only disclosed generically, but a
  reconciliation that then rests on a declared rake or declared dead money is
  caught by `ACCOUNTING_ASSUMPTION_DEPENDENT`. That no longer depends on
  `persist_reconciliation` having written the recorded gross, rake and net for the
  neutral pass to disagree with: round 10 showed that precondition was the rule's
  boundary rather than a protection, since an imported or hand-edited row leaves
  all three NULL, and the measurement now compares the derived FIGURES as well as
  the verdict.
- A hand entered here by its own operator can record a hero result its own action
  line contradicts and reconcile, by choosing the rake, and it is never blocked
  for it: the dependence is measured and reported but the exemption applies. Every
  figure on such a hand is the same person's own entry, with no independent
  observation for a disclosure to protect. The consequence is that its accounting
  gate cannot detect a self-consistent forgery by its author, which is true of
  every field on such a hand and is not specific to the rake. The exemption is no
  longer granted by the `manual` label: an imported hand attests to its own
  declared assumptions whatever it calls itself (round 10).
- A hand stored as `manual` + a completion status other than `not_applicable` is
  reconstructed for readiness (correctly — the pair is unproven) but three of its
  blockers name CV reconstruction, ROI calibration and re-import, none of which
  exist for a hand typed in by hand, and `_completion_reason` asserts a source the
  hand does not have. `_hand_from_row` normalises only the mirror pair. The state
  is not produced by any reachable writer; see the `_completion_reason` entry
  above, which this extends to the clearing actions. Round 10 found a fourth
  blocker on this hand — `ACCOUNTING_ASSUMPTION_DEPENDENT` — that was not merely
  mis-worded but *unclearable*, because the writer behind its control refused
  every `manual` row while the blocker was emitted for the pair, and the page
  flashed "Confirmed" over the discarded write. That is repaired: both consult
  `requires_assumption_attestation`, and the writer reports whether it recorded
  anything. The three mis-worded clearing actions remain as described.
- `_apply_completion_import_defaults`'s `review_status` downgrade is redundant in
  outcome: `_enforce_review_status_floor` writes `needs_correction` for every
  reconstructed source a few lines later in the same transaction, and the condition
  can never be true for `manual`. Removing it passes the whole suite; removing the
  floor is killed by three tests. It is retained as a local invariant at the point
  the status is decided, and its comment no longer claims to be the guard.
- `_hand_settlement_from_row` is now total: every column it converts, including
  the timestamps, is inside the guard, and an unreadable row is degraded to a
  non-reconciled settlement naming the columns it could not read. The other row
  readers (`_session_from_row`, `_hand_from_row`, the action and player readers)
  still call `_parse_datetime` outside any guard, so a hand-edited `created_at`
  of `'never'` raises a `ValueError` out of their fetch. It is not repaired here
  because the conservative degradation the settlement reader uses has no honest
  equivalent for a timestamp: a settlement can be reported unreadable and blocked,
  whereas a hand would have to be shown with an invented creation date. Recorded
  as a known exposure of the same family rather than silently left as a claim of
  coverage.
- Reading the version stamp on a WAL database creates `-shm`/`-wal` sidecars beside
  it even when the open is then refused, so "a database this build refuses is never
  written to" holds for the database file itself but not for the directory. The
  refused file stays byte-identical. The second half of this entry — a bare
  `sqlite3.OperationalError` out of `PokerDatabase.__init__` on a read-only mount —
  was fixed in round 7: `_enter_wal_mode` names the mount, states that the database
  is intact, and says a container needs the whole directory writable. The same
  helper now waits out a concurrent writer for the documented busy timeout instead
  of dying immediately, because SQLite never runs the busy handler for a
  journal-mode change.
- `export_hand` emits no `solver_runs`, so a v5 export is not a faithful record of
  a hand's solver history and an export/import round trip silently discards every
  solver run, including live ones. `STALE_SOLVER_EVIDENCE` is therefore cleared by
  the round trip — defensibly, since the imported database genuinely holds no
  solver run and nothing stale is being presented as current, which is why this is
  a gap and not a bypass. The asymmetry with coaching (which *is* exported and
  whose stale blocker survives) is the part worth closing, and closing it means
  adding a section to the export payload rather than editing a check.
- Whether a source-fact change is written into the completion evidence depends, for
  two writers, on unrelated state. `update_hand_player` and `delete_action` call
  `_invalidate_hand_derivatives(force_review_status=True)` and so always append
  `source_facts_corrected` and demote `complete` to `uncertain`; `create_hand_player`
  and `create_action` do not force it, so the `force_review_status or has_review`
  guard skips the evidence write unless the hand happens to carry a saved coaching
  or hand review. Neither create path is reachable from the UI on an existing hand
  today — `save_player_rows`/`save_action_rows` serve the new-hand form, and the
  Study editor adds actions through `create_corrected_action`, which does force it —
  and the naive fix is actively wrong: `import_session` and the CV exporter build
  every hand through `create_hand_player`/`create_action`, so forcing it there would
  demote every freshly imported hand to `uncertain` on creation. Closing this
  properly means distinguishing "populating a new hand" from "adding to an existing
  one" rather than flipping a flag.
- The `BLOCKER_ORDER` sort in `evaluate_study_readiness` is a no-op against the
  code as it stands: the sequence of `blockers.extend(...)` calls already emits the
  documented order, so deleting the sort changes nothing observable and no test can
  kill that mutant. `tests/test_phase1_adversarial_round6.py` pins the observable
  order instead, which is what the contract actually promises; the sort is retained
  as defence in depth for a future reordering of those calls.

## Phase 2 — Build the private validation corpus

### Minimum corpus

Create a hash-locked ClubWPT corpus containing:

- 10 saved completed-session recordings;
- at least 100 fully annotated completed hands;
- at least 10 partial or deliberately truncated hands in addition to the 100;
- five development sessions;
- two validation sessions;
- three locked-test sessions.

Split by whole recording. No frames, clips, near-duplicates, or adjacent
recording segments from one session may cross splits.

### Required coverage

The corpus must include:

- five, six, seven, and eight dealt-in players;
- hero folds preflop;
- opponents folding before the hero acts;
- limped pots;
- single-raised and three-bet pots;
- postflop folds;
- flop, turn, and river completions;
- showdowns;
- voluntary shown cards;
- all-ins;
- uncalled bet returns;
- split pots if the client displays one;
- side pots if supported by the observed client behavior;
- short stacks and large stacks;
- player join/leave or empty-seat changes between hands;
- table/lobby transitions;
- long pauses and animation-heavy transitions;
- blurry, occluded, or dropped frames;
- recording start during a hand;
- recording end during a hand;
- supported resolution, window-size, scale, and crop profiles;
- at least one unsupported geometry used to test safe rejection.

### Artifact storage

- Keep raw private videos under a configurable `POKER_VALIDATION_ROOT`.
- Do not commit raw recordings to ordinary Git.
- Use logical case IDs instead of absolute user paths in manifests.
- Record SHA-256, byte size, duration, FPS, dimensions, and expected layout.
- Commit answer keys, manifests, compact cropped fixtures, cached deterministic
  detections where appropriate, and expected reports.
- Make the July 23 recording the first named corpus case.
- Retain the original recording; do not pretend that a derived JSON fixture can
  expose failures in video decoding, model inference, anchoring, or sampling.

### Ground-truth process

- Annotate each recording in two independent passes, with at least one human
  pass.
- Every hand answer key records:
  - first and last timestamps;
  - whether the start/end is partial;
  - terminal event;
  - hero cards;
  - final board;
  - dealer and dealt-in seats;
  - starting stacks where visible;
  - ordered actions with amount semantics and certainty;
  - final pot;
  - winner/result;
  - hero net;
  - evidence timestamps for each critical fact.
- Reconcile annotation disagreements through a separate adjudication pass.
- Use pot/stack arithmetic as a cross-check, not as a replacement for visible
  evidence.
- Freeze answer keys by version and hash before scoring the locked test.
- Any later answer-key correction must be separately reviewed, explained, and
  versioned; never silently edit truth to make the model pass.

### Exit gate

- Corpus counts and coverage pass automatically.
- Every file hash matches.
- Every truth record passes structural, card-uniqueness, action-legality, and
  arithmetic validation.
- The locked test has never been used for tuning.

## Phase 3 — Create the executable release-gate framework

### Operator interface

Add:

```bash
python -m poker_tracker.release_gate \
  --manifest validation/clubwpt_v1.json \
  --mode full \
  --report-dir data/release_reports
```

Supported modes:

- `fixture`: fast deterministic CI using compact retained inputs;
- `full`: actual local video decoding, real models, reconstruction, import, and
  downstream validation against the artifact vault;
- `container`: the same acceptance path inside the pinned Docker image.

Do not add a provider-specific hosted mode while there is no explicitly
requested deployment.

### Versioned interfaces

The corpus manifest must include:

- manifest schema version;
- case ID and recording hash;
- split;
- expected platform/layout profile;
- video metadata;
- truth-file path and hash;
- coverage tags;
- expected runtime class;
- model/configuration allowlist.

The answer-key schema must include the hand facts listed in Phase 2 and explicit
certainty/observability fields.

The release report must include:

- pass/fail verdict;
- commit identifier and dirty-state marker;
- manifest, answer-key, and model hashes;
- Python, dependency, FFmpeg, OS, CPU, memory, and architecture information;
- configuration with secrets redacted;
- per-stage and per-recording timing;
- peak memory and disk use;
- every per-hand mismatch;
- aggregate metrics;
- skipped/unavailable checks;
- generated artifact paths;
- adversarial-round summaries when present.

Exit codes:

- `0`: every mandatory gate passed;
- `1`: a product or accuracy gate failed;
- `2`: corpus, dependency, model, or environment setup was invalid.

### Evaluator corrections

Strengthen the existing hand evaluator so:

- missed, spurious, split, merged, and duplicate hands all count as failures;
- completed-versus-partial classification is scored directly;
- partial first/last hands are never omitted from completion precision/recall;
- action ordering is scored when the answer key is complete;
- missing predicted amounts do not automatically count as a match;
- amount tolerances are explicit and street-aware;
- unsupported/unobservable answer-key facts are excluded explicitly, not by
  implicit `None` behavior;
- critical and noncritical errors are categorized;
- threshold failures return nonzero;
- empty truth, empty predictions, missing artifacts, and zero scored hands fail
  closed;
- report generation is deterministic.

### CI integration

- Run unit, integration, UI, fixture-release, Ruff, and MyPy checks on pull
  requests.
- Validate all corpus schemas and hashes available to CI.
- Do not require private raw videos in public CI.
- Require the full local video gate before a release candidate is accepted.
- Retain CI and local release reports for comparison.

### Exit gate

- One command produces a complete, reproducible verdict.
- A release cannot be declared from a hand-written checklist alone.

## Phase 4 — Harden video ingestion and job execution

### Upload and storage

- Validate extension, actual container/codec, file size, duration, frame count,
  FPS, width, and height before scheduling CV.
- Reject empty, corrupt, unsupported, path-traversal, symlink, hardlink, and
  decompression-bomb-style inputs safely.
- Copy uploads atomically to the configured data root.
- Store video metadata and stable file paths in SQLite; keep bytes on disk.
- Detect missing, replaced, truncated, or size/hash-mismatched videos.
- Define retention behavior for source videos, frames, timelines, logs, and
  exports.
- Provide a user-visible storage audit before deleting anything.

### Processing jobs

- Keep one heavy CV or solver job active at a time.
- Make queue, start, heartbeat, progress, timeout, completion, cancellation,
  and failure transitions explicit and tested.
- Ensure a killed worker cannot leave a job permanently running.
- Ensure restart reconciliation never marks an alive worker failed or runs a
  duplicate worker.
- Write progress snapshots atomically.
- Bound logs and error messages; never expose secrets or raw provider keys.
- Terminate child process groups on timeout/cancellation.
- Keep failed timelines/logs for debugging without importing partial hands.
- Make import transactional after a successful consistent backup.
- If backup creation fails, do not import.
- If import fails, retain the prior database and diagnostic export.

### Failure injection

Test:

- missing FFmpeg;
- unsupported codec;
- corrupt recording;
- unreadable model;
- model exception;
- process kill;
- app restart;
- timeout;
- disk full;
- permission denied;
- progress-file corruption;
- database locked;
- backup failure;
- import validation failure;
- simultaneous CV/solver requests.

### Exit gate

- Every failure has a safe terminal state.
- No interrupted or failed job creates a partially imported session.
- The prior database remains recoverable.

## Phase 5 — Generalize and harden the CV/OCR pipeline

**Status: geometry and OCR-scale defects repaired and measured; exit gate NOT
evaluated and therefore NOT met.** The exit gate below requires locked-test
thresholds to pass. Those thresholds cannot be evaluated at all today, because
no answer key exists for any recording — locked, validation, or development.
Phase 5 must not be counted toward the release claim.

Four measured defects were repaired and re-measured across all five development
geometries. Full numbers are in `cv_lab/notes/13_phase5_geometry_and_ocr_verification.md`;
the detector produces bit-identical boxes across runs, so the before/after
populations are the same detections.

- Card zones were raw frame-normalized rectangles and did not anchor. At aspect
  ratio 1.750 the community row rendered at cy 0.334-0.339 against a band
  starting at 0.36, so **0 of 435** board-card detections were classified as
  board and a four-street showdown exported with `board_cards=""`, all actions
  labelled preflop, `result="Hero wins +116 BB"`, no warnings, and session
  confidence 0.989. Anchored zoning recovers **2067 of 2067** community cards
  across the five geometries (79.0% before), with false "board" classifications
  halved, 12 to 6.
- Numeric OCR inferred decimals from an absolute-pixel gap and assumed exactly
  two decimal places. On the smallest supported client 12.42% of all numeric
  reads were inflated 10-100x; that is now 0.00%, worst stack read 40760.0 ->
  407.6, worst per-frame stack outlier ratio 211.2 -> 2.1. On the one-decimal
  client `POT: 240.9 BB` read 24.09 and now reads 240.9. A dropped leading "0."
  that shipped `call 50.0` for a 0.50 BB call on the baseline recording is
  fixed. Of 14390 crops, 422 changed value; **0** moved from unknown to a
  confident value.
- The pot was a single scalar. Both pots are now read and reported; the one
  development hand with a side pot moves from a silently wrong final pot of 1.0
  to 240.9 main + 0.2 side and is explicitly rejected as
  `side_pot_unsupported`, which is what the "Pot, winner, and result
  reconstruction" section above requires until representative truth cases exist.
- Session confidence averaged across a timeline and reported 0.989 for a session
  whose every board was destroyed. It is now `min(per-hand confidence)`,
  reported for operators only, and never used as a gate.

Exported coverage on the development corpus moved **14 of 21 hands to 11 of 21**.
That is reported as a decrease, not defended as an improvement: the four lost
hands are two with destroyed boards and two whose own action ledgers contradict
themselves (a seat that calls and then folds; a seat that goes all-in and then
folds), all four of which previously shipped at confidence 0.95 with no tags. One
hand was gained. Six of the eleven surviving exports now carry `LOW_CONFIDENCE`;
before, none of the fourteen could say anything about itself.

Newly discovered release blockers, none of them closed:

- **No answer key exists for any recording.** Until two-pass adjudicated truth
  exists, no accuracy, completion, boundary, accounting, or safe-rejection gate
  in the Hard Release Gates table can return a result. This blocks Phase 5's
  exit gate outright and is upstream of Phase 2.
- **The region detector's training manifest contains a locked-test recording.**
  `cv_lab/datasets/yolo_cards_autolabel_v1/manifest.csv` holds 228 rows from
  `clubwpt_session_01.mov`, in both the train and val splits. This is
  pre-existing and untouched by this phase, but it violates the Split integrity
  gate: the locked test cannot cleanly measure the shipping detector until the
  model is retrained from a clean manifest.
- **Street reconstruction on a recovered board is the largest open
  reconstruction defect.** Fixing the zone anchoring gave the board-regression
  and street-order nets material they never had; those codes rose corpus-wide
  from 5 and 6 occurrences to 8 and 11, and the aspect-ratio-1.750 recording now
  exports 0 of its 3 hands.
- **The card classifier misreads suits and can do so unanimously.** One wrong
  card in 33 audited exported board cards. A card every frame agrees on is
  invisible to every check in the pipeline; closing this needs retraining.
- **One OCR read regressed.** A pot crop whose integer part is hidden by an
  opaque sprite moved from 0.5 to 50.0 (both wrong; the new one wrong by 100x).
  One occurrence in 14390 crops, falling between two hands and reaching no
  export. Unrepaired and recorded.
- **The confidence scale is still hand-set.** `WARNING_SEVERITY` and
  `_REJECTION_SEVERITY` are constants. The scores separate hands usefully but
  are not calibrated probabilities, and the calibration bullet in the
  "Confidence and rejection" section below remains open.

### Screen and layout classification

- Distinguish table, lobby, transition, modal, and unsupported screens.
- Identify the supported ClubWPT layout profile before applying coordinates.
- Use normalized landmarks and validated transforms rather than assuming one
  absolute desktop resolution.
- Measure anchor confidence and reject implausible transforms.
- Detect moved, resized, cropped, letterboxed, or partially obscured tables.
- Provide a calibration preview and explicit unsupported-layout result.
- Do not silently run a known-bad layout through default anchors.

### Frame selection and video decoding

- Validate timestamps and decoding monotonicity.
- Ensure first/last relevant frames are not lost to interval rounding.
- Use coarse screen classification plus change detection to retain deal,
  action, street, pot, showdown, and settlement transitions.
- Increase sampling around suspected boundaries and fast action changes.
- Avoid spending model time on long static/lobby stretches.
- Make selection deterministic for a pinned configuration.
- Retain enough source frames to explain every imported fact.

### Region detection

- Evaluate all eight reconstruction classes by precision, recall, seat/zone
  assignment, layout profile, scale, brightness, and obstruction.
- Audit the current model against the new corpus.
- Label false positives, false negatives, and geometry drift.
- Retrain only from development data.
- Maintain rehearsal data to prevent card/layout drift.
- Freeze model weights, class ordering, preprocessing, and thresholds by hash.
- Add an explicit compatibility check between model outputs and reconstruction
  code.

### Card recognition

- Measure hero, board, and shown-villain cards separately.
- Score rank, suit, full-card, and duplicate-card failures.
- Calibrate confidence by card class and capture profile.
- Require temporal agreement for accepted cards.
- Reject one-frame excursions and cards overlapping hero/board/villain zones.
- Preserve unknown cards rather than guessing.
- Mine every corrected card as a hard example.
- Prevent training/locked-test leakage.
- Confirm the classifier and template fallbacks agree on retained evidence.

### Numeric OCR

- Validate stack, bet, pot, blind, and ante reads independently.
- Handle decimals, commas, BB/currency markers, empty values, animation, and
  partially occluded digits.
- Use temporal debounce without carrying accepted values across hand boundaries.
- Reject impossible jumps through stack and pot invariants.
- Retain original crop, parsed text, numeric value, confidence, and fallback
  source for every critical amount.
- Calibrate templates only from development/validation data.
- Treat absent amounts as unknown, not zero.

### Action-pill and turn-order reading

- Validate action type, seat assignment, active-turn evidence, and amount
  semantics.
- Distinguish check from fold through retained-card state and street context.
- Prevent stale pills from becoming duplicate later actions.
- Handle instant folds that appear for only one sampled frame.
- Preserve uncertainty when observed order cannot be recovered.
- Do not invent an action merely to make the pot balance.

### Temporal state construction

- Make cross-frame state association deterministic.
- Prevent values, folded states, cards, pots, and stacks leaking into the next
  hand.
- Require stable dealt-in evidence without inventing phantom players.
- Track seat changes only at valid boundaries.
- Retain state-level provenance for every reconstructed event.
- Detect impossible board rollback, duplicate cards, and street-order changes.
- Detect lobby/transition frames inside a candidate hand.

### Hand segmentation

- Detect:
  - first observed deal;
  - next deal;
  - hero fold;
  - showdown;
  - pot award;
  - recording end;
  - lobby transition.
- Mark partial start and partial end independently.
- Prevent one physical hand becoming multiple predicted hands.
- Prevent adjacent physical hands being merged.
- Exclude next-hand blinds/antes and auto top-ups from prior settlement.
- Retain incomplete segments for review but never export them as complete.
- Make completed/partial classification a separately scored output.

### Action reconstruction

- Reconstruct forced bets, folds, checks, calls, bets, raises, all-ins, refunds,
  shows, and wins using explicit amount semantics.
- Maintain stable player keys through all streets.
- Enforce legal street progression and turn order.
- Use stack deltas, bet text, pot text, and action pills as independent evidence.
- Never let accounting “correct” an unsupported observed action silently.
- Preserve source timestamp, image, state index, derivation, and confidence per
  action.
- Mark an incomplete sequence non-authoritative.
- Improve preflop-only, sparse, folded, fast, and under-sampled hands.

### Pot, winner, and result reconstruction

- Keep pot OCR, contribution arithmetic, and winner stack-sweep estimates
  independent.
- Require corroboration before declaring a winner or final pot.
- Handle hero-fold-only records without fabricating unseen villain resolution.
- Handle uncalled returns before pot settlement.
- Detect auto top-ups and next-hand stack resets.
- Support side-pot/split-pot outcomes only after representative truth cases
  exist; otherwise reject them explicitly.
- Never convert an unreconciled estimate into an authoritative result.

### Confidence and rejection

Phase 1 already replaced the old `0.95 - 0.15 * len(warnings)` subtraction with a
severity-weighted score built from the validator's `WARNING_SEVERITY` table plus
`_EXPORT_ONLY_SEVERITY`, and introduced the `_REJECTION_SEVERITY = 0.5` cutoff
that decides which validator codes become permanently unclearable rejections
(currently eight of them). Those constants are hand-set, not calibrated, and
`hands.confidence_score` consequently holds two scales: rows written before
Phase 1 on the 0.95 baseline, rows written since on the severity scale. Both are
bucketed by the same `confidence_label()`, so the column is only safe to read as
a coarse label, never as a probability or as a cross-era comparison.

- Calibrate the per-fact and per-hand confidence the exporter now computes, and
  restate `_REJECTION_SEVERITY` as a measured threshold rather than a constant.
- Measure calibration on validation data, not the locked test.
- Decide whether to rescale or segregate pre-Phase-1 `confidence_score` values
  once a calibrated scale exists; do not compare the two eras before then.
- Define critical warning codes for cards, boundaries, layout, actions,
  accounting, and results.
- Any critical uncertainty forces `completion_status=uncertain` or
  `review_status=needs_correction`.
- Optimize for accepted-hand precision and safe rejection, not maximum automatic
  coverage.
- Report coverage separately so rejecting every hand cannot count as success.

### Exit gate

- All locked-test accuracy, completion, accounting, and rejection thresholds in
  the Hard Release Gates section pass.

**Evaluated: no. Met: no.** Not one threshold in that table has been given a
value. The blocking reason is that no answer key exists for any recording, so
there is nothing to score a prediction against; the locked-test recordings
remain unopened by design. Development-corpus self-consistency, which is what
the repairs above measured, is not a substitute — a hand that survives every
internal net can still be wrong, and this corpus (5 recordings, 21 hands) is far
below the 10 sessions / 100 completed hands / 10 partial cases the Corpus gate
requires. Phase 2 must deliver the corpus and adjudicated truth before this gate
can return anything at all.

## Phase 6 — Complete the correction and regression feedback loop

### Save-now/debug-later behavior

- Keep flagging available directly from Study.
- Save issue category, user description, immutable structured snapshot, source
  video/timeline identifiers, model/config hashes, and relevant timestamps.
- Never duplicate the full video into SQLite.
- Show all unresolved issues in a cross-session queue.
- Allow export/download of a self-contained structured issue bundle.
- Preserve resolved issues and resolution notes permanently.

### Correction behavior

- Allow auditable correction of:
  - hand boundaries/completion status;
  - hero and board cards;
  - players, positions, and stacks;
  - actions and amount semantics;
  - pot, result, and hero net;
  - settlement assumptions and awards.
- Require a correction reason.
- Retain before/after values.
- Mark source type `corrected_cv`.
- Invalidate settlement and all affected derived evidence transactionally.
- Prevent unrelated session/hand evidence from being staled.

### Regression promotion

- Add an operator workflow that converts a confirmed issue into:
  - a minimal cached-state fixture when the bug is reconstruction-only;
  - a cropped-frame fixture when the bug is OCR/detection;
  - a full-video corpus case when decoding, anchoring, sampling, or boundaries
    are involved.
- Link the issue ID, correction ID, regression case, fixing change, and passing
  report.
- Do not resolve the issue until the new regression fails before the fix and
  passes after it.
- Re-run the affected corpus slice and the entire locked acceptance set.

### Derived reruns

- Corrections stale prior hand/session coaching and completed solver runs while
  retaining them.
- The UI clearly distinguishes current from historical/stale output.
- Coaching may be rerun from corrected facts.
- Solver may be rerun only when the corrected hand remains eligible.
- A stale result can never satisfy the review gate.

### Exit gate

- A user can flag without diagnosing, leave the app, and later reproduce and
  resolve the issue without losing source evidence.
- Every closed release-blocking issue has a passing regression.

## Phase 7 — Complete authoritative accounting

### Ledger correctness

- Validate forced posts, antes, straddles, dead blinds, calls, bets, raises,
  all-ins, refunds, folds, side pots, split awards, rake, and no-flop-no-drop.
- Preserve amount semantics (`incremental`, `raise_to`, `unknown`) through CV,
  editing, export/import, coaching, and solver eligibility.
- Enforce stack caps and reject overcommit.
- Reconcile gross pot, refunds, rake, net pot, awards, and hero net.
- Distinguish observed facts from user-entered settlement assumptions.
- Retain warnings and the exact inputs used for each calculation.
- Keep fold-only hands honest when later villain resolution was not observed.

### Editing and invalidation

- Any player/action/hand-fact change invalidates the prior settlement.
- Settlement edits do not rewrite recorded actions.
- Stable player keys survive display-name changes.
- Deleted/reassigned players cannot leave orphaned actions or awards.
- Review, Insights, coaching, and solver read authoritative ledger results when
  available and visibly fall back otherwise.

### Tests

- Golden ledgers for every supported pot structure.
- Property/fuzz tests for conservation of chips.
- Illegal ordering and amount-semantics tests.
- Multiway all-in, side-pot, refund, split, and rake boundaries.
- Round-trip and migration preservation.
- Streamlit settlement editor workflows.

### Exit gate

- Every study-ready CV hand is authoritative and balanced.
- Any imbalance has a visible blocker and cannot be silently reviewed.

## Phase 8 — Finish and certify TexasSolver review

TexasSolver feature implementation is substantially present. Remaining work is
release certification and boundary hardening, not expanding it into universal
GTO analysis.

### Functional certification

- Validate pinned binary/resource discovery on macOS and in both container
  architectures.
- Confirm eligible completed, reconciled, heads-up postflop cash spots from
  five- through eight-handed source tables.
- Test built-in, saved, estimated, and custom weighted ranges.
- Preserve exact suit-specific blocker filtering.
- Continue disabling unsafe suit-isomorphism optimization and record that
  assumption.
- Retain command, input hashes, ranges, board, stack/pot geometry, abstraction,
  convergence, exploitability, raw JSON, logs, runtime, and backend version.
- Validate output frequency vectors and reject malformed/partial results.
- Test timeout, cancellation, process-group termination, stale heartbeat,
  missing output, corrupt output, and memory exhaustion.
- Keep one heavy job active at a time.

### Honest scope

- Exclude preflop-only, multiway postflop, tournament/ICM, unsupported game,
  missing-action, and unreconciled spots.
- Label raked source hands as no-rake equilibrium approximations when the
  backend does not model recorded rake.
- Do not claim action EV, BB loss, or regret when the retained output does not
  expose those values.
- Keep built-in preflop ranges labeled as study estimates.
- Show solver assumptions next to all explanations.

### Licensing/distribution gate

- Treat TexasSolver’s license as a release blocker for any distributed or
  hosted solver-enabled image.
- Before publishing an image or enabling a hosted solver, retain either:
  - written maintainer permission for the intended use/distribution; or
  - a documented, reviewed AGPL compliance approach with required source,
    notices, and offer mechanics.
- Until that gate passes, keep the solver as an optional user-installed local
  dependency or locally built image and do not publish the bundled image.
- Generate third-party notices and an SBOM for any distributable artifact.
- Do not present this plan as legal advice; obtain qualified review before
  public distribution.

### Exit gate

- All functional, resource, failure, honesty, and licensing gates appropriate to
  the chosen local distribution path pass.

## Phase 9 — Finish and certify coaching

### Provider behavior

- Fail closed when no provider/key is configured.
- Keep secrets in environment variables and out of SQLite, prompts, logs,
  exports, and reports.
- Retain provider, model, raw prompt, raw response, parsed sections, source
  hand/session, safety mode, and timestamp.
- Handle provider timeout, rate limit, invalid JSON/shape, refusal, partial
  output, network failure, and malformed content.
- Make retries bounded and user-visible; never duplicate a successful review
  silently.

### Grounding

- Build prompts only from completed post-session evidence.
- Label manual estimates, approximate equity, accounting facts, and solver facts
  separately.
- Prevent invented hole cards, board cards, actions, pots, frequencies, action
  EV, or BB loss.
- Require solver-specific claims to match retained parsed solver evidence.
- State important assumptions and missing facts.
- Never treat provider prose as authoritative reconstruction or settlement.

### Staleness and reruns

- Any relevant correction stales old hand and session coaching.
- Stale coaching remains inspectable but cannot be shown as current.
- Corrected coaching reruns use the latest database state and retain history.
- Session coaching stales when any included hand changes.
- Review completion requires stale coaching to be rerun or explicitly removed
  from the current evidence set.

### Evaluation

- Expand deterministic golden coaching cases across supported hand types.
- Run structural/fabrication/safety scoring in CI with a fake provider.
- Run opt-in live-provider smoke tests without making wording equality a gate.
- Check that prompts contain only intended completed-session facts.

### Exit gate

- No unsupported model statement is presented as solver or recorded truth.
- Corrections always invalidate and rerun affected coaching correctly.

## Phase 10 — Finish every product workflow

### Overview

- Show accurate session, hand, review, issue, reconciliation, and job counts.
- Distinguish completed, partial, uncertain, corrected, stale, and reviewed data.
- Show failed/interrupted jobs with a useful next action.
- Surface storage/database health without exposing secrets.
- Avoid presenting CV draft performance as confirmed analytics.

### Sessions

- Validate create/edit behavior from a clean database.
- Support multiple recordings attached to one session without duplicate hands.
- Preserve hand ordering and stable identifiers.
- Make manual hand entry explicit and safe.
- Show session-level open issues, stale coaching, unresolved hands, and data
  provenance.
- Validate empty, large, and partially processed sessions.

### Hands

- Keep cross-session search/filtering accurate for cards, dates, stakes,
  position, source, tags, review state, and issues.
- Show current evidence status without requiring entry into Study.
- Keep the unresolved debugging inbox accessible and navigable.
- Prevent stale analytics badges from surviving corrections.
- Validate pagination/large-list performance.

### Study

- Make the readiness blockers the primary workflow.
- Retain table replay, action history, source evidence, accounting, correction,
  issue reporting, coaching, math, and solver without conflating their sources.
- Ensure every edit has validation, confirmation, audit history, and targeted
  invalidation.
- Keep historical/stale coaching and solver output visually distinct.
- Prevent marking reviewed through any UI path while blockers remain.
- Test previous/next navigation and session queue behavior after mutations.

### Insights

- Compute metrics only from the appropriate confirmed/reconciled population.
- Show denominators and coverage.
- Separate manual, CV draft, corrected CV, and reviewed evidence.
- Exclude stale coaching themes from current conclusions.
- Explain when the sample is too small.
- Verify every metric against hand-level source rows.

### Import

- Make post-session-only scope explicit.
- Validate file before copying or queueing.
- Show layout/profile support, metadata, progress, failure, backup, timeline, and
  import summary.
- Never claim a failed/skipped/partial hand was imported successfully.
- Allow retry without duplicating a completed import.
- Keep the original video linked to all imported/corrected hands.

### Settings

- Validate data paths, storage health, ROI profiles, export/import, coaching
  configuration, math tools, and solver availability.
- Provide environment-variable guidance without displaying secret values.
- Show configured model hashes and supported layout profiles.
- Make ROI calibration previews reproducible.
- Add a release/diagnostics download containing redacted configuration and
  health output.
- Keep destructive operations explicit, narrowly scoped, and recoverable.

### Cross-cutting UX

- Test narrow/mobile-width rendering for study views without implying mobile
  live use.
- Add useful loading, empty, error, and recovery states.
- Ensure buttons/forms are idempotent across Streamlit reruns.
- Provide keyboard-accessible labels and readable contrast.
- Avoid relying on color alone for status.
- Keep warnings concise but specific.

### Exit gate

- Every workflow passes a clean-database end-to-end acceptance script and
  representative AppTest coverage.

## Phase 11 — Persistence, portability, backup, and recovery

### SQLite integrity

- Test foreign keys, WAL settings, busy timeouts, transaction boundaries, and
  concurrent UI/worker access.
- Verify all rows survive every supported migration.
- Reject databases newer than the running application.
- Detect missing required tables/columns/indexes and foreign-key violations.
- Confirm corrections, issues, coaching, settlements, solver records, and
  provenance cascade only where intended.

### Import/export

- Preserve current API shapes through versioning.
- Round-trip sessions, hands, players, actions, settlements, reviews, coaching,
  corrections, issues, completion evidence, and relevant provenance.
- Validate malformed types, duplicate identifiers, missing relationships,
  unsupported versions, oversized JSON, and partial payloads.
- Ensure import validation completes before application-data writes.
- Make duplicate-import behavior explicit.
- Never overwrite an existing session silently.

### Backup and restore

- Create a consistent SQLite backup before each CV import and schema migration.
- Retain a documented rotating policy.
- Verify every retained backup through isolated restore.
- Test restoring to a clean data root.
- Verify session/hand counts, schema, foreign keys, issue evidence, and one
  completed hand after restore.
- Keep source videos and generated artifacts in the backup inventory even when
  they are backed up separately from SQLite.
- Report missing video/frame/timeline/solver files.
- Never overwrite the only live database during a drill.

### Exit gate

- A fresh machine with the repository, environment configuration, persistent
  data directory, and verified backup can recover the complete study history.

## Phase 12 — Security, privacy, and safety hardening

### Authentication and sessions

- Fail closed when authentication is required but no password is configured.
- Use constant-time password comparison.
- Prevent authentication state from leaking across users/browser sessions.
- Rate-limit or otherwise bound repeated login attempts when exposed beyond
  localhost.
- Never log the password or provider keys.

### Filesystem safety

- Confine uploads, frames, timelines, exports, logs, backups, and solver results
  to explicit data roots.
- Reject traversal, symlink, hardlink, and unexpected device/file types.
- Sanitize user-provided filenames while retaining the original display name.
- Use atomic writes for critical artifacts.
- Apply least-privilege permissions in containers.

### Data privacy

- Keep raw videos local by default.
- Document exactly which hand facts leave the machine for external coaching.
- Require explicit provider configuration before transmitting prompts.
- Keep diagnostics and release reports redacted.
- Provide clear deletion/export behavior without broad destructive commands.

### Post-session safety

- Test all prompts and UI entrypoints for completed-session-only language.
- Reject any request path that attempts to analyze an in-progress hand.
- Keep CV workers file-based and detached from poker-client capture.
- Do not add window capture, screen polling, overlay, hotkey, or live API
  integration.
- Ensure adversarial review explicitly tests these boundaries.

### Dependency and supply-chain safety

- Pin release-critical CV/model and solver inputs.
- Generate dependency inventory/SBOM for distributable containers.
- Scan dependencies and images for known critical vulnerabilities.
- Record model origins, hashes, and applicable licenses.
- Remove unused high-risk dependencies.
- Keep secrets out of image layers and Git history.

### Exit gate

- No critical/high security, privacy, or post-session-safety finding remains.

## Phase 13 — Performance, resource, and container readiness

### Local runtime

- Measure startup time, upload time, frame throughput, model initialization,
  reconstruction time, import time, UI responsiveness, and solver runtime.
- Profile peak resident memory, disk growth, temporary files, and log growth.
- Reuse model instances safely where it materially improves runtime.
- Bound caches and retained artifacts.
- Ensure SQLite remains responsive while a heavy job runs.
- Verify a representative full session completes within one hour on the
  supported local reference machine.

### Docker

- Build from a clean context.
- Run as a non-root user.
- Keep all durable data on explicit mounts.
- Verify healthcheck, authentication, environment configuration, model loading,
  FFmpeg, CV, solver discovery, and shutdown behavior.
- Build and test both `linux/amd64` and `linux/arm64`.
- Record image size and startup/peak memory.
- Ensure runtime writes do not depend on the immutable application layer.
- Ensure the base application remains usable when TexasSolver is absent.
- Do not publish a solver-enabled image until its licensing gate is satisfied.

### Resource coordination

- Enforce one heavy CV/solver job at a time across UI and worker entrypoints.
- Keep solver address-space limits and thread limits configurable.
- Bound CV timeouts and memory through documented environment variables.
- Fail with actionable messages instead of allowing host-wide exhaustion.

### Provider-neutral deployment readiness

- Keep runtime paths, authentication, ports, limits, and secrets in environment
  variables.
- Keep data on mountable persistent storage.
- Validate restart and health behavior through local Compose or an equivalent
  provider-neutral container runner.
- Maintain provider-specific documents only as optional references.
- Do not provision or deploy a host during this plan.

### Exit gate

- Local Streamlit and both container architectures pass the same functional,
  persistence, recovery, and representative workload checks.

## Phase 14 — Testing and continuous quality

### Unit tests

- Cards, ranges, equity, EV, pot odds, ICM, and analytics.
- OCR parsing, debounce, card voting, seat assignment, and state transitions.
- Hand segmentation, completion classification, action reconstruction, and
  pot/winner consensus.
- Accounting legality, side pots, refunds, rake, and settlement.
- Readiness, staleness, issue, correction, and solver eligibility rules.
- Import/export and migrations.

### Property and fuzz tests

- Card uniqueness and normalization.
- Range parser normalization and weights.
- Accounting chip conservation.
- Import JSON validation.
- OCR numeric parsing.
- Timeline state ordering.
- Malformed solver output.
- Path and filename validation.

### Integration tests

- Video metadata → stored video → job → timeline → export → backup → import.
- Timeline → SQLite → accounting → correction → stale derivatives → rerun.
- Reconstruction → coaching prompt.
- Reconciled hand → solver spot → retained result → grounded explanation.
- Issue save → regression → fix → resolution → export/import.
- Migration → application reads → backup → isolated restore.

### UI tests

- Every tab from a clean database.
- Authentication success/failure/misconfiguration.
- Manual entry and editing.
- CV job states.
- Readiness blockers.
- Correction and issue queue.
- Settlement.
- Coaching rerun.
- Solver eligibility/run/history.
- Import/export and diagnostics.

### Real-video tests

- Development/validation iteration.
- Locked full-model evaluation.
- Repeat determinism.
- Unsupported layout rejection.
- Corrupt/truncated input.
- Runtime and peak-memory measurement.

### Regression policy

- A bug fix without a reproducing test is incomplete.
- Hand-specific bugs must link to the saved issue/evidence.
- Tests must fail for the original defect, pass for the fix, and not weaken an
  existing assertion to obtain green.
- Update expected fixtures only with reviewed source evidence.

### Exit gate

- All mandatory suites pass without unexplained skips or flaky reruns.

## Phase 15 — Documentation and operator readiness

### README

- Keep the product boundary, supported workflows, setup, environment variables,
  storage layout, solver scope, coaching configuration, tests, data health, and
  container commands current.
- State the certified ClubWPT layout/corpus scope.
- State that CV imports are drafts until confirmed.
- State that no hosted deployment is active.

### PLAN

- Update phase status and gates as work lands.
- Keep failed experiments and obsolete alternatives out of the current roadmap.
- Record newly discovered release blockers.

### CV research notes

- Add chronological notes for corpus construction, model evaluations,
  calibration decisions, reconstruction changes, and final held-out results.
- Never rewrite older findings to imply later success.

### Operator runbooks

- Clean local install.
- Model/FFmpeg/TexasSolver diagnostics.
- Full release-gate execution.
- Corpus vault setup.
- Database migration.
- Backup and isolated restore.
- Failed job recovery.
- Data-health audit.
- Docker build/run on both architectures.
- Upgrade and rollback.
- Issue-to-regression debugging workflow for a future Codex agent.

### Release evidence

- Archive the final release report, corpus/model hashes, dependency versions,
  migration/restore results, container results, and adversarial findings.
- Keep secrets and raw private videos out of committed reports.

### Exit gate

- Another engineer or agent can install, validate, diagnose, back up, restore,
  and continue the issue loop without undocumented tribal knowledge.

# Hard release gates

The release command must fail if any mandatory gate below fails.

| Area | Mandatory private-beta result |
|---|---|
| Component quality | Pytest, Ruff, configured MyPy, and diff checks pass |
| Corpus | 10 hash-verified sessions, at least 100 completed hands, and at least 10 partial cases |
| Split integrity | No recording/clip/frame leakage into the locked test |
| Truth quality | Two-pass annotation, adjudicated disagreements, timestamped evidence, valid arithmetic |
| Completion classification | 100% precision and 100% recall for completed versus partial/unfinished hands on locked test |
| Hand boundaries | Zero missed, split, merged, duplicate, or spurious locked-test hands |
| Boundary timing | Matched boundaries fall within the documented sampling tolerance |
| Raw card recognition | At least 98% exact full-card accuracy on locked test |
| Accepted cards | 100% exact hero/board cards among hands allowed to become study-ready |
| Reconstruction | Every locked completed hand has at most one noncritical scored error |
| Critical facts | Zero silently accepted wrong boundary, card, winner, result, or materially wrong action |
| Actions | No illegal accepted sequence, stack overcommit, or unsupported invented action |
| Accounting | 100% balanced or explicitly blocked as needing correction |
| Safe rejection | 100% of materially uncertain/inconsistent hands are blocked from study-ready |
| Coverage | Report accepted/rejected coverage; rejecting every hand cannot pass |
| Provenance | Every critical reconstructed fact links to retained source evidence |
| Corrections | Every mutation retains before/after state and targets derivative invalidation correctly |
| Issues | Every release-blocking closed issue has a passing permanent regression |
| Coaching | Zero stale or fabricated output represented as current/solver truth |
| Solver | Eligible spots, ranges, parser, job recovery, assumptions, and resource limits pass |
| Determinism | Repeated pinned runs produce semantically identical timelines and verdicts |
| Job safety | Every injected failure ends safely with no partial import |
| Persistence | Fresh install and every supported migration/round-trip pass |
| Recovery | All retained backups pass isolated restore and data-health checks |
| Authentication | Required-auth fail-closed, session, and secret-handling tests pass |
| Filesystem | Traversal/link/atomic-write/permission tests pass |
| Post-session safety | No live capture, overlay, or current-hand path exists |
| Local runtime | Representative session completes within one hour and documented memory/disk limits |
| Heavy jobs | CV and solver never overlap |
| Containers | Clean AMD64 and ARM64 builds pass health and representative workloads |
| Licensing | No solver-enabled image is distributed without resolved permission/compliance |
| Documentation | Setup, validation, backup, restore, and debugging runbooks match the release |

The governing safety principle is:

> A wrong prediction that is visibly rejected is a coverage limitation. A wrong
> prediction silently accepted as study-ready is a release blocker.

# Repeated adversarial-agent release loop

The adversarial review is mandatory after all normal tests and release gates
first pass.

## Candidate preparation

1. Freeze the exact candidate commit, dependency lock state, model hashes,
   corpus manifest, and release report.
2. Confirm the normal suite, full real-video gate, migration/restore gate, and
   both container gates pass.
3. Create disposable databases, data directories, copied recordings, and
   container environments for attacks.
4. Never let adversarial testing mutate the user’s live database or sole video
   copy.

## Adversary A — white-box integrity and security

Spawn a fresh agent with instructions to:

- inspect code and invariants without editing product files;
- attack schema migration, transactions, foreign keys, concurrency, backup,
  restore, import/export, stale derivatives, job recovery, filesystem
  confinement, authentication, secret handling, dependency boundaries, and
  post-session safety;
- run destructive experiments only on disposable copies;
- look for silent data loss, corruption, unsafe acceptance, privilege/path
  bypasses, and misleading output;
- return structured findings with severity, exact reproduction, expected versus
  actual behavior, evidence, and affected gate.

## Adversary B — black-box product and reconstruction

Spawn a second fresh agent with instructions to:

- use the application, test interfaces, corpus, and mutated disposable inputs
  without editing product files;
- attack corrupt video, unusual metadata, partial hands, split/merge boundaries,
  fast folds, ambiguous cards/OCR, stale UI state, repeated clicks, job
  interruption, unsupported layouts, accounting edge cases, solver/coaching
  eligibility, resource exhaustion, and workflow bypasses;
- attempt to make an incorrect hand appear completed, reconciled, reviewed, or
  current;
- return the same structured finding format.

## Triage and repair

1. Reproduce every reported finding independently.
2. Classify:
   - critical: safety violation, exploitable auth/path issue, data loss, or
     silently trusted materially wrong hand;
   - high: core workflow failure, unrecoverable job/data state, stale evidence
     presented as current, or release-gate bypass;
   - medium: supported case fails with a safe recovery path;
   - low: nonblocking usability, maintainability, or cosmetic issue.
3. Record false positives with concrete reproduction evidence.
4. Save hand-specific valid findings in the debugging issue queue.
5. Add the smallest regression that fails before the fix.
6. Fix every critical, high, and release-blocking medium issue.
7. Run targeted tests, the entire normal suite, the full release gate, restore
   drills, and affected container checks.
8. Do not weaken a release threshold to make a finding disappear.

## Repetition and stopping rule

- Any valid blocking finding resets the clean-round counter to zero.
- After fixes pass, spawn two new adversarial agents with fresh contexts and
  varied attack prompts/seeds.
- An agent setup failure, unavailable artifact, timeout without a verdict, or
  skipped attack does not count as clean.
- Low findings may remain only when documented in `PLAN.md` with rationale and
  no effect on correctness, safety, data integrity, or the release claim.
- Completion requires two consecutive rounds in which both agents report:
  - zero critical findings;
  - zero high findings;
  - zero release-blocking medium findings;
  - zero unresolved safety, data-loss, stale-evidence, or silent-acceptance
    concerns.
- If code, models, schema, dependencies, or release configuration change after a
  clean round, the clean-round count resets.

# Final local/private-beta acceptance sequence

Run these in order against the exact candidate:

1. Clean environment installation.
2. Fresh-database application walkthrough.
3. Historical database migration.
4. Unit/integration/UI/fixture suite.
5. Ruff, MyPy, and repository consistency checks.
6. Full 10-session real-video release gate.
7. Repeat-determinism run.
8. Failure-injection suite.
9. Correction/issue/regression/rerun workflow.
10. Coaching provider failure and grounding checks.
11. TexasSolver functional/resource/failure checks.
12. Data-health audit and isolated restore of every retained backup.
13. Clean AMD64 container build and workload.
14. Clean ARM64 container build and workload.
15. Provider-neutral restart and persistent-volume drill.
16. First adversarial-agent round.
17. Repair/regression/full rerun if needed.
18. Repeat with fresh agents until two consecutive clean rounds.
19. Archive redacted release evidence.
20. Tag/identify the local private-beta candidate.

No cloud resources are provisioned and no hosted deployment is performed by
this plan. If deployment is explicitly requested later, repeat the relevant
acceptance, security, resource, backup, restore, and adversarial gates in a
staging environment before exposing the application.

# Definition of done

The first local/private beta is complete only when:

- a user can import a saved supported ClubWPT session recording;
- completed, partial, and uncertain hands are classified correctly;
- reconstructed drafts retain inspectable source evidence;
- uncertain or inconsistent evidence is blocked rather than silently trusted;
- the user can immediately correct a hand or save it for later debugging;
- every correction updates SQLite transactionally and preserves audit history;
- accounting reconciles the confirmed hand or blocks review;
- stale coaching and solver evidence is retained but never shown as current;
- current post-session coaching can be regenerated from corrected facts;
- eligible confirmed hands can be analyzed with honestly scoped TexasSolver
  evidence;
- export/import and backup/restore preserve the complete structured history;
- local Streamlit and supported containers pass the same core workflows;
- the full corpus satisfies every hard release threshold;
- two consecutive two-agent adversarial rounds are clean;
- all operator and debugging workflows are documented.

# After the first release

These are not allowed to weaken or delay the first-release correctness gates.
They begin only after the closed-loop reconstruction system is dependable.

## Higher-confidence reconstruction

- Expand from 10 sessions/100 hands to at least 30 sessions/300 hands.
- Add additional ClubWPT capture geometries as separately certified profiles.
- Improve automatic coverage while preserving zero silent critical acceptance.
- Add model calibration/drift reports across releases.

## Optional analysis improvements

- Expand vetted range-profile coverage.
- Add action-EV evidence only if a validated backend exposes it.
- Evaluate a local LLM only when it improves privacy or recurring runtime cost
  without reducing coaching quality.
- Add retrieval over user-owned notes after source/correction semantics are
  dependable.

## Additional clients/layouts

- Treat each poker client and layout as a new product surface.
- Require its own corpus, model/layout profile, answer keys, locked evaluation,
  runtime measurements, and adversarial passes.
- Never infer support merely because one sample appears to work.

## Deployment

- Remain provider-neutral.
- Select a host only after an explicit user request.
- Resolve solver licensing/distribution first.
- Require persistent storage, TLS, authentication, healthchecks, restart,
  backup, restore, resource, and adversarial staging gates.

## Explicitly out of scope

- real-time assistance;
- live table capture;
- poker-client overlays;
- current-hand recommendations;
- PostgreSQL or multi-user accounts before a real multi-user requirement;
- universal client/layout support without separate validation;
- claims of universal or mathematical 100% CV accuracy.
