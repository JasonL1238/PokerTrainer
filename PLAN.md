# PokerTrainer Master Completion Plan

This is the canonical current-status, implementation, validation, and release
plan for PokerTrainer. It replaces short feature wish lists with an executable
program for completing the first trustworthy local/private beta.

The plan intentionally includes all known remaining work: product behavior,
computer vision, OCR, hand reconstruction, accounting, correction, coaching,
TexasSolver integration, persistence, security, recovery, performance,
container readiness, documentation, acceptance testing, and repeated
adversarial review.

**This file holds the current plan and the open items. It does not hold
history.** The round-by-round adversarial record — what each round found, the
argument behind each repair, and the regression that pins it — lives in
`cv_lab/notes/16_phase1_adversarial_rounds.md` (Phase 1's fifteen rounds) and
`cv_lab/notes/17_release_adversarial_rounds.md` (the whole-product rounds). Those
notes are chronological and later findings supersede earlier ones; nothing in
them is a claim about what is true today. A finding that is still open appears
here, as an open item, in the phase that owns it.

Every status line in this file is written to be falsifiable. Where a claim has
not been executed — an image nobody has built, a solver nobody has run, a
contrast ratio nobody has computed — it says so in those words. A status line
that overstates what works is worse than no status line, because someone will
rely on it.

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
- every CV hand that lands in a session begins as a needs-correction draft
  (auto-add only for frame-validated full hands that did not start mid-hand;
  incomplete/mid-start hands require an explicit draft add);
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

As of August 2, 2026, the repository contains:

- a seven-workflow Streamlit application: Overview, Sessions, Hands, Study,
  Insights, Import, and Settings, with every published figure carrying a declared
  population, a denominator, a coverage figure and an evidence class (manual / CV
  draft / corrected CV / reviewed) rather than a bare rate;
- SQLite persistence at **schema 20** for sessions, hands, players, actions,
  settlements, coaching, corrections, issues, videos and their content hashes,
  processing jobs, extracted frames, per-action source-frame provenance, ROI
  profiles, review evidence, range profiles, solver runs and the parameters they
  were produced under, regression cases linking an issue to the regression that
  proves it stays closed, explicit hand completion status plus its versioned
  reconstruction evidence, and per-hand study inclusion preference;
- schema migration protection, WAL operation, JSON import/export **version 6**,
  pinned pre-migration, pre-import and pre-delete backups written beside the
  database they roll back and each carrying an artifact inventory, a read-only
  data-health/restore audit, and an isolated fresh-machine recovery drill
  (`python -m poker_tracker.maintenance.recovery`);
- derived, never-persisted study readiness with per-blocker reasons and clearing
  actions, enforced at the store and behind a single guarded UI writer;
- compact manual solver-spot entry (single or multi-hand paste with `x/b3.5/c`
  lines) plus full editing for sessions, hands, players, actions, settlements, and
  review state;
- poker math for cards, ranges, equity, pot odds, EV, ICM, analytics, and
  authoritative action-ledger accounting;
- post-session Theory and Exploit coaching through configured providers, with
  prompt safety checks, retained provider/model/source evidence, and a grounding
  check that compares the response against the prompt that produced it;
- optional TexasSolver-backed heads-up postflop cash-game analysis with
  eligibility checks, exact ranges, retained assumptions and run parameters,
  convergence, logs, strategy JSON, stale-result handling, a bounded recorded-action
  mapping that refuses rather than substitutes past its limit, a positive
  definition of a usable result, and solver-grounded explanation checks;
- offline video upload, metadata validation, frame extraction, detached CV jobs
  bounded by documented environment variables for wall-clock time and address
  space, heartbeat/recovery behavior, timelines, and validate-then-edit-import;
- an eight-region reconstruction model plus card recognition/OCR readers,
  temporal reconstruction, labeling tools, hard-example queues, and evaluation
  scripts;
- a guided two-mode Study workflow (Replay, Analyze) for approved hands only,
  with explicit BB units on every stack, pot and result, session-scoped loading,
  inline TexasSolver guidance, before/after correction history, and staling of
  derived coaching/solver evidence;
- a persistent cross-session issue inbox that freezes debugging evidence, carries
  the identity of the source recording, frames, models and environment in its
  exportable bundle, retains resolution notes, and deep-links back to Import
  validation;
- an executable release gate in three modes (`fixture`, `full`, `container`), a
  performance and resource harness that reports an unmeasured figure as `null`
  with a reason rather than as zero, suite-quality tooling for skip policy, flake
  detection and coverage, a dependency inventory/SBOM, and CI;
- authenticated local/container operation, a non-root Docker runtime, pinned CV
  dependencies, healthchecks, and a container verification script;
- **3066 passing tests with 6 skips** and no failures at the latest inventory
  (see "Verification record" under Phase 1, which is the authoritative count and
  which this bullet must be updated to match), with Ruff and the configured MyPy
  target green.

These checks establish a strong component baseline. They do not prove that real
recordings reconstruct reliably across a representative held-out set, and several
of the claims above have never been executed end to end: no image has been built
on either architecture, no TexasSolver binary exists on this machine, and no
answer key exists for any recording. Each is stated where it belongs, in the
phase that owns it.

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

## Where each phase actually stands

One line per phase. "Gate met" means the phase's own exit gate has been evaluated
and passed, not that its code is written. A phase whose code is complete and
whose gate has never been evaluated is **not** met, and is not counted toward the
release claim.

| Phase | Implementation | Exit gate | What the gate is waiting on |
| --- | --- | --- | --- |
| 0 Baseline freeze | done | not met | the migrated-historical-database walkthrough |
| 1 Completion & readiness | done | bullets satisfied, phase **uncertified** | two consecutive clean adversarial rounds; counter 0 of 2 |
| 2 Validation corpus | **blocked** | not met | the operator's annotation pass — no adjudicated answer key exists for any recording |
| 3 Release-gate framework | done | not met | Phase 2; the framework cannot return 0 without truth |
| 4 Ingestion & job execution | done | met for everything testable locally | — |
| 5 CV/OCR pipeline | repairs measured | **not evaluated** | Phase 2 |
| 6 Correction & regression loop | done | not met | Phase 2 (the locked acceptance set) |
| 7 Authoritative accounting | done; verdict **not-yet-caught** | not met | Phase 2; a critical in each of four consecutive rounds |
| 8 TexasSolver certification | bounded, honest, retained | **not met** | a real binary to certify against; AGPL licensing |
| 9 Coaching | grounding implemented | not met | live-provider evaluation is opt-in and unrun |
| 10 Product workflows | seven surfaces done | **partly unverified** | narrow-width rendering and contrast are unmeasured |
| 11 Persistence & recovery | done | not met | the drill has only run against synthetic histories |
| 12 Security & privacy | done | met, except licensing | AGPL (`ultralytics`) blocks publishing an image |
| 13 Performance & containers | code done | **not met** | the image has never been built on either architecture |
| 14 Testing & quality | done | not met | the one-hour representative-session gate has never run |
| 15 Documentation | runbooks written | not met | a release that can pass, which requires Phase 2 |

**Phase 2 is the critical path.** It is blocked on the operator's annotation
pass, which is human work no code can substitute for, and it holds the exit gates
of Phases 3, 5, 6 and 7 shut regardless of how much code those phases contain.

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

**Status: implementation complete; adversarial certification not met.**

The completion columns landed with schema 13, readiness is derived per read in
`poker_tracker/services/study_readiness.py`, and every UI path that can mark a
hand reviewed is routed through one guarded writer. Both exit-gate bullets below
are satisfied against the code that exists.

What is not satisfied is this plan's own stopping rule. Fifteen adversarial
rounds have been run against Phase 1 and every one of them produced at least one
valid blocking finding, so the consecutive-clean-round counter stands at **0 of
the required 2**. Phase 1 is done as implementation and undone as certification.
It must not be counted toward the release claim.

The round-by-round record — what each round found, what was repaired, the
argument behind each repair, and the regression that pins it — is in
`cv_lab/notes/16_phase1_adversarial_rounds.md`. That is history and is not
repeated here. What stays here is the contract that history produced, and the
items that are still open.

These fifteen rounds are **phase-scoped and numbered separately** from the
whole-product rounds under "Where the adversarial gate stands", which are at
round 3. Both counters are 0 and neither substitutes for the other: passing this
phase's certification would not satisfy the release stopping rule, which is the
whole-product one.

### What remains before Phase 1 may be counted

Nothing below is optional.

1. Two consecutive adversarial rounds, each with fresh agents and varied attack
   prompts, reporting zero critical, zero high, zero release-blocking medium, and
   zero unresolved safety, data-loss, stale-evidence or silent-acceptance
   findings. Round 16 is the first round eligible to count.
2. No code, schema, dependency, or release-configuration change between those two
   rounds.
3. The exit-gate bullets stay satisfied through both rounds, re-derived from the
   code each round rather than carried forward. Where this document states which
   call sites exist, the statement is backed by a test rather than by a reading:
   `test_no_consumer_decides_on_is_authoritative_alone`,
   `test_no_sql_predicate_classifies_a_row_the_reader_reclassifies`,
   `test_no_consumer_prescribes_an_action_from_unresolved_codes` and
   `test_no_ui_call_site_writes_the_recorded_pot_or_hero_result` each fail when a
   new consumer appears. Recording a repair as complete while six consumers still
   read the old predicate is a defect this phase has already shipped once.

Set the prior on a clean round 16 from the record rather than from the size of
the last repair: fifteen consecutive rounds found a real blocking defect, and
rounds 10 through 15 each broke a mechanism the previous round's document
described as closed.

### Readiness blocker vocabulary

Readiness is derived per render and never persisted. Each blocker carries a
stable code, a category, a plain-language reason, and the exact clearing action.
Blockers are emitted, and categories rendered, in this order:

| Code | Category | Applies to |
| --- | --- | --- |
| `STUDY_EXCLUDED_BY_OPERATOR` | study_preference | all hands |
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
"not_applicable"`) can only ever emit the study-preference, cards,
`UNREADABLE_HAND_COLUMNS`, `ACCOUNTING_NOT_AUTHORITATIVE`, issue, coaching, and
solver blockers, so the pre-Phase-1 manual workflow is unchanged — with two
qualifications, and they are the two blockers scoped on "entered here" rather
than on the pair. `ACCOUNTING_ASSUMPTION_DEPENDENT` is scoped by
`requires_assumption_attestation`, and `USER_CONFIRMATION_MISSING` by
`requires_user_confirmation`, which delegates to it. See "Who must attest".

**A blocker never names an action the product cannot perform.** This is a
standing rule, not an aspiration, and it has been violated in at least four
distinct ways over the fifteen rounds — a named panel that is not drawn, a
deletion no control performed, a re-import that appends instead of replacing, a
discard writer that did not exist. Every clearing action must name a control that
exists on a page the operator can reach, and adding a blocker means checking
that.

### Assumption-dependent reconciliation

`ACCOUNTING_ASSUMPTION_DEPENDENT` is derived, per read, by reconciling one hand
twice: once with the stored settlement declaration and once with that declaration
withdrawn, against the same fetched records. Everything that is the hand rather
than the declaration — the action line, the board, whether a flop was seen, the
recorded pot and hero result — is held constant across both passes.

The declaration is `hand_accounting._Declaration`: the rake policy, the declared
dead money, and the declared pot awards. That is the complete set of inputs a
cross-check pass takes from what somebody declared rather than from what was
recorded, and its completeness is a property rather than a promise —
`test_a_neutral_declaration_derives_a_ledger_from_the_recording_alone` sweeps
wildly different settlement rows and award sets over one unchanged recording and
asserts every fully neutral ledger is identical, so an input added to the derived
side later without being added to `_Declaration` fails a test instead of opening
the next per-field hole.

The hand is assumption-dependent when removing the declaration changes either the
**verdict** (the neutral pass stops reconciling, or the ledger cannot be built
without it) or the **figures** (it reconciles too, but derives a different gross
pot, rake, net pot, payout, or hero result). Both halves are load-bearing. A hand
that records none of the figures the cross-check compares reconciles under every
policy — that is the ordinary state of a freshly imported hand, since
`import_session` never calls `persist_reconciliation` — so without the figures
half a declared 90% rake moves the hero result by most of the pot and measures as
independent. Without the verdict half, a policy that takes the whole pot leaves
every figure standing still while the ledger goes unsettled. Each declared input
is then neutralised on its own, by the same two-part test, to attribute it.

**What the awards are withdrawn to** is not "nobody won anything": no recording
produces that state, an award-less ledger is never `is_settled`, and comparing
against it made every reconciling hand award-dependent by construction — a
compulsory confirmation on hands declaring no rake and no dead money at all,
which is the click-through fatigue this rule exists to prevent. The awards are
compared against the winner the recording forces: `_forced_winners` reads the
award-less ledger's own `PotLayer.eligible_players`, and a pot exactly one seat
is still eligible for is answered by the action line rather than by the operator.
A pot two or more seats are eligible for is a showdown, where who was pushed the
chips genuinely is a declaration nothing corroborates: still measured, still
named, still blocking. An award to a seat that folded is refused by the ledger
itself before any of this runs, so the exemption cannot launder a hero result.

Four properties follow, and they are why this replaced eight rounds of per-field
disclosure conditions:

- **There is no field list, so there is no per-field hole.** A rate with a zero
  cap, a no-flop-no-drop waiver on a hand that saw no flop, and a chip unit
  coarser than the whole rake all take zero chips and are silent. Any combination
  that does move chips is named without anyone having enumerated it.
- **The attestation is bound to a quantity and to the declaration.** The blocker
  clears only when the operator confirms a code carrying both a fingerprint of
  the declared inputs — over this hand's gross pot, its settled per-seat
  contribution vector, and which seat is the hero — and the measured chip
  movement: `declared_settlement_dependence:<input>:<fingerprint>:<movement>`.
  The movement is written with `format(value, "+")`, which round-trips to the
  float it was measured from, so no two distinct measurements share a string. A
  declaration that later changes, or a correction that grows the pot the same
  policy applies to, lapses the attestation and the operator is asked again.
- **The attestation has its own channel and its own control.** It is stored in
  `completion_evidence.confirmed_assumption_codes`, never in `warning_codes` or
  `acknowledged_codes`, and `parse_completion_evidence` enforces the separation in
  both directions on every read, so no writer, payload, or hand-edited row can mix
  them. The audit trail is the `hand_corrections` row the attestation writes.
- **The verdict lives with the reader, not a writer.** `hand_settlements` has no
  CHECK constraint, so a settlement row written any other way would reach
  `reconcile_persisted_hand` undisclosed. The measurement runs on every read that
  readiness consults. A row this build cannot validate at all is degraded on read
  to a non-reconciled settlement naming the unreadable columns, rather than
  raising out of the fetch.

Confirming the hand as a whole (`USER_CONFIRMATION_MISSING`) does not clear it:
that checkbox asks whether the reconstruction is right, and this asks whether
specific unobserved chips were really taken, added, or pushed to a particular
seat. The clearing action is **Confirm this assumption** in Study → Summary →
Accounting reconciliation, or correcting the declaration in the same panel until
it matches what happened. "Withdrawing" is available for a rake or a dead-money
amount and is not available for a pot award, which is why the awards are compared
against the winner the recording forces.

Confirming is also what re-opens every gate the dependence closed. The
measurement is re-derived on every read and is never erased by the answer, so
`unattested_assumption_dependence` — "does this hand still owe an answer?" — is
the one expression that decides, and "answered" means attested *or* exempt.

Every surface that publishes a derived figure reads one predicate,
`services.study_readiness.accounting_is_established` (authoritative **and** no
measured dependence this hand still owes an answer for) rather than
`is_authoritative`, so the gate and the blocker cannot drift. **Writing a derived
figure into an observed-fact column takes the same gate**: the settlement
editor's "Replace observed final pot/result with the derived ledger values"
writes into `hands.pot_size` and `hands.hero_bb_won`, which are the independent
evidence the cross-check compares against and the fallback
`math.analytics.compute_session_stats` reads precisely when the derived figure is
refused. That write goes through `services.settlement_sync`, which takes
`accounting_is_established` and refuses by name otherwise;
`db.update_hand_accounting_evidence` refuses again on its own single-pass
measurement; and `test_no_ui_call_site_writes_the_recorded_pot_or_hero_result`
fails if any module outside that service calls the writer.

The writer-side codes `declared_unobserved_chips` and `declared_unobserved_rake`
are retained as an audit trail with nothing resting on them. They are raised by
`_declared_chips_taken` — a different, single-pass measurement with a strictly
smaller input set, because `upsert_hand_settlement` calls it before the award rows
may exist — and stored in `completion_evidence.declared_settlement_codes`, never
in the pipeline's channel. No blocker reads them, no Acknowledge control offers
them, and `completion_status` does not move when one is written.

### Who must attest, and why it is not `source_type`

The exemption is for a hand **this operator entered in this database**: an ante, a
dead blind, a straddle from a seat that left, and the room's rake are all that
person's own entry, and there is no pipeline claim for a declaration to outrank.
`manual` + `not_applicable` is how such a hand is stored, but it is not the
argument. An import payload can write those two strings, and a payload that does
is byte-identical to a genuine manual export, so no guard can disprove the claim
— and none has to. What a payload cannot manufacture is having been entered here:
`import_session` stamps every hand it lands, and `requires_assumption_attestation`
(reconstructed **or** imported) is the single predicate consulted by the blocker,
by the control that clears it, and by the writer behind that control. The stamp
carries no `evidence_version`, so it is not reconstruction evidence, and it is
idempotent, so repeated round trips do not accumulate.

Three further readers reach the same verdict, because enforcing the argument on
one consumer is what let a relabelled payload through:

- `_hand_from_row` normalises a `manual` claim carrying a reconstruction claim
  (`claims_reconstruction`, ANY nonzero `evidence_version`, readable or not) to
  `cv_import`, so a hand-edited relabel cannot walk a blocked CV hand out of its
  blockers while its evidence stays attached; `update_hand_status` refuses the
  same pair on the raw row.
- `USER_CONFIRMATION_MISSING` and the checkbox that clears it are scoped by
  `requires_user_confirmation`, delegating to the same predicate, because the
  importing operator has not vouched for a hand whatever its payload declares.
- `_enforce_review_status_floor` demotes every declared `reviewed` to
  `needs_correction`, for any source type. The label is one tick and one save away
  for the operator who now vouches for it. The v13 migration still keeps manual
  review statuses: a migrated database is the same operator's own data, not
  somebody's JSON.

Nothing this build stores or exports can be a value RFC 8259 JSON cannot express.
`_serialize_json` writes with `allow_nan=False`, `_parse_json_object` sanitises on
read, `completion._as_float_or_none` treats NaN and infinity as unreadable, and
`PersistedModel` sets `allow_inf_nan=False` once for every float field of every
persisted model.

### Persistence and migration impact

Phase 1's migration is schema version 13:

- `hands.completion_status`: `complete`, `partial`, `uncertain`, or
  `not_applicable` for manual hands;
- `hands.completion_evidence`, versioned JSON holding `partial_start`,
  `partial_end`, terminal-event type, first/last source timestamps,
  preceding/following boundary evidence, boundary confidence, source frame
  references, warning/rejection codes, and pipeline/model versions.

Study readiness is derived from those rather than persisted. For a reconstructed
hand it requires `completion_status == complete`, valid and unique hero/board
cards, supported table/layout evidence, authoritative reconciled accounting, no
open debugging issue, no unresolved source warning, no stale retained coaching or
solver result being represented as current, and explicit user confirmation.

The migration chain has since run past 13 and is now at **schema 20**: 14 adds
video content hashes, 15 per-hand study inclusion, 16 per-action source-frame
provenance, 17 the `regression_cases` table, 18 `solver_runs.run_parameters`, 19
the declared blind structure (`hand_settlements.small_blind`, `big_blind`,
`straddles`), 20 the declared ante mode (`hand_settlements.ante_mode`). Every one
is additive, and 19 and 20 are deliberately unbackfilled: a big blind inferred
from the largest observed post is exactly the defect 19 exists to end, and an
ante mode inferred from one seat having anted is exactly the defect 20 exists to
end.

**Schema 20 is the first migration that visibly demotes hands that previously
reconciled**, and that is the ruling rather than a side effect. Every stored hand
containing an ante reads `ante_mode IS NULL`, which is ambiguous rather than
defaulted, so it gains one legality issue naming the anteing seats, `is_legal`
goes False, `persist_reconciliation` writes `needs_correction`, and study
readiness blocks on `ACCOUNTING_NOT_AUTHORITATIVE`. No row is rewritten and no
chip figure moves: the layers published beside the refusal are the capped
(PER_PLAYER) reading, which is the strict direction and byte-for-byte what the
product derived before the column existed. The clearing action is one ordinary
settlement save. **Hands with no ante rows are never asked for a declaration** —
`NONE` is not a guess for them, so the absent declaration resolves silently.
Count what will block with `SELECT COUNT(DISTINCT hand_id) FROM actions WHERE
action_type = 'ante' OR forced_bet_type IN ('ante','big_blind_ante')`.
**v20 reaches a second population, for a different reason, and this one does
move chips.** Ruling 5 ships in the same release and caps operator-typed
external dead money against each collecting seat's own commitment; a stored hand
whose declared amount exceeds the smallest commitment contesting the main pot
keeps its gross, its pot count and its eligible sets while the distribution — and
therefore the hero result — changes, so no existing cross-check can see it. Such
a hand may contain no ante and asks for no declaration. The new figure is the
correct one, so the migration does not touch the settlement, the awards or
`review_status`; what it does is mark the coaching and solver output retained
beside those hands stale, because that analysis was written against a result
this build no longer produces and `is_stale` is a stored flag that a change in
the derivation rule otherwise never sets. Study readiness then blocks on
`STALE_COACHING_EVIDENCE`. The predicate is `dead_money > 0`, deliberately
over-strict because a schema migration cannot run the reducer to find the floor;
count it with `SELECT COUNT(*) FROM hand_settlements WHERE dead_money > 0`. A
hand in neither population is untouched in every respect.
JSON export is at **version 6** and imports accept 1
through 6.

The migration rules that Phase 1 established and every later migration inherits:

- Additive only. No migration deletes or rewrites correction, issue, coaching,
  settlement, video, or solver history.
- Exactly one process runs the chain. `init_db` re-reads the stored version under
  SQLite's write reservation and stops if another opener already migrated. A stamp
  that is not a readable, non-negative version number is refused with the same
  clear message the newer-database path uses.
- A **missing** stamp is not a fresh database and is refused too.
  `_physical_schema_floor` reads the file's own schema for the one artefact of the
  one migration that is not safe to replay (`hands.completion_status`), and a
  stamp behind that floor is refused with a restore-from-backup message. The test
  is what the schema physically contains, never whether a stamp happens to exist,
  so a genuine pre-versioning database still migrates.
- A database this build refuses is never written to. The `journal_mode = WAL`
  pragma runs only after the version check.
- A consistent backup is created before migrating a real file database, and it is
  **pinned**: written under a name the rotating five-slot pool never matches, kept
  in its own `PINNED_KEEP_COUNT` slots, audited and restore-drilled by
  `data_health`. It is written to `backups_dir_for(db_path)` — beside the database
  it can roll back — so opening a fixture, a restored copy, or a backup under
  audit cannot evict a rollback point of the live database. An artifact inventory
  is written beside it.
- A failed pre-migration snapshot names the directory it could not write and says
  the database is unchanged, rather than surfacing SQLite's `unable to open
  database file`, whose plain reading is that the operator's database is corrupt.
- Existing manual hands become `not_applicable`. Existing `cv_import` and
  `corrected_cv` hands migrate conservatively to `uncertain` and
  `needs_correction`; they require confirmation rather than being silently
  promoted.
- Import versions 1–4 remain accepted and receive safe defaults. Older
  application versions continue refusing to open a newer database.

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
  reads "I have read the evidence above and confirm this hand is correct", so the
  evidence has to be above it: `view_models.completion_evidence_rows` is the pure
  transformation and `app.show_reconstruction_evidence` draws it on all three
  promotion surfaces — Study, the Sessions hand list, and Settings → Coach.

### The completion invariant

`completion_status == "complete"` is true only when `derive_completion_status`
says so, on every writer and every reader:

- the CV exporter derives it from the evidence it just built;
- `import_session` re-derives it and ignores whatever the payload declared, at
  every export version, and may only ever *weaken* it across the whole ordering
  `complete` → `uncertain` → `partial`;
- `update_hand_completion` re-derives it on every evidence write, takes the
  stored evidence as its base so a caller may only ADD codes, is sticky on
  `partial`, and never lets a derived `not_applicable` replace a stored status
  that was anything else;
- `update_hand_status` refuses `reviewed` when the stored column and the stored
  evidence disagree;
- `evaluate_study_readiness` emits `COMPLETION_NOT_COMPLETE` when they disagree.

A **warning code** is an operator-acknowledgeable note. A **rejection code** is
the pipeline refusing the hand: `acknowledge_codes` will not accept one, and
`derive_completion_status` blocks on `rejection_codes` directly so a hand-edited
`acknowledged_codes` list cannot launder one into a promotion. A rejection clears
only by producing new evidence without it. The exporter's severity table is built
from the validator's rather than maintained beside it, and a code the exporter
does not recognise is classified as a rejection.

**An acknowledgement cannot travel in a payload**, because it is an operator of
*this* database attesting to a code they have read. `import_session` resets the
list and keeps every code, so nothing is lost and the importing operator
re-acknowledges what they accept. A v5 or v6 export of an acknowledged hand
re-imports unacknowledged; this is a deliberate, documented round-trip asymmetry,
and the same rule applies to a declared `reviewed`, a resolved debugging issue,
an assumption attestation, and a coaching review's staleness.

Read-time degradation markers — `unreadable_card_columns`,
`UNREADABLE_HAND_COLUMNS` — are derivations rather than evidence, so they are
never stored, are stripped on every write, and are restored after a round trip
keyed on `DERIVED_EVIDENCE_KEYS` and the table's own PRAGMA columns. A marker may
never overwrite a readable column, and any read-time marker demotes the hand
through one `_demote_degraded_hand` step, so two degradations on one row cannot
reach opposite `review_status` verdicts.

`reviewed` never outlives the evidence it was granted on. Every writer that
changes a hand's players, actions, settlement or evidence returns a promoted hand
to `needs_correction`, and the settlement writers additionally stale the retained
coaching and solver output the corrected award invalidates. Acknowledging a
warning that leaves the hand `complete` is not an invalidation and does not
demote. Every demotion is written into the evidence, not only into the column.

Every row reader degrades rather than raises. `db._salvaged_row` is driven by the
model's own field set and pydantic's error report, so a column added later is
covered by having been added, and a degraded row can only ever ADD blockers. No
SQL predicate classifies a row that the reader would reclassify: candidates are
selected by identity and classified through the reader, with an allow-list of the
seven places where the column, not the verdict, is the right subject.

### The accounting tolerance

There is one tolerance and it is float-representation noise.

**Every recorded figure on a settled hand is compared exactly.** The gross pot,
the observed final pot (`hands.pot_size`), declared refunds — uncalled bets are
returned before the drop — the hero's net result, and every declared award are
judged against float-representation noise alone. No rake policy whatsoever can
excuse a disagreement about how many chips went into the pot, how many came back
out, or which seat they went to. Four earlier versions of this rule bounded a
tolerance whose width was set by the data it was judging; each was defeated. The
history is in `cv_lab/notes/16_phase1_adversarial_rounds.md` §6.

The other standing rules in this area:

- **A declared award is checked per identity, and a blank amount disables
  nothing.** An identity whose award amounts are all present is compared exactly;
  an identity with a blank among them must still satisfy the half of the claim it
  made.
- **Re-declaring who won a pot is a source-fact correction.** Both public writers
  of `settlement_entries` take a before/after snapshot, record a
  `settlement_award_update` correction and write `source_facts_corrected` into the
  evidence. The declared odd-chip order is part of that snapshot, because it
  decides who receives an indivisible chip. The reconciler's own derived-refund
  write is excluded by construction.
- **A recorded hero result with no hero seat is not a passed check.** A hand that
  records a hero result or hero cards with no hero seat raises an issue naming the
  control that fixes it.
- **The chip unit rounds the rake and nothing else.** The granularity a chopped
  pot is divided at is derived from the evidence alone — `_split_granularity`
  takes the settled contributions and each declared dead-money contribution
  individually, and returns the finest decimal place any of those amounts is
  written in, capped at one whole chip. The odd chip is real and is kept; what it
  is no longer is a dial. What the operator types in `Chip unit` *does* still
  change derived payouts, because rounding the rake changes the net pot every
  payout is drawn from — it is a declared settlement input like any other and is
  measured like one.
- **No pot may be raked past its own size.** Each share is capped at its own pot
  and the rounding leftover is offered to the layers in order, each taking only
  what it still has room for.
- **A pot layer is cut at a LIVE contribution level or at a dead-money cap, and
  nowhere else.** Live boundaries are cut at live contribution levels after
  refunds, and apply to every seat by that seat's own live contribution: nobody
  can decline a forced post, so unequal dead money never opens a live layer, and
  a short seat's own forced posts never raise the level its opponents are charged
  into the main pot at. Cutting at a seat's TOTAL commitment instead was the
  round-19 critical: a seat live-short behind an ante was paid live chips no
  opponent had wagered against it, settled, balanced and legal with no warning.
  Dead money starts in the lowest layer and the part of it above the smallest
  TOTAL commitment among that layer's eligible seats rises into a layer of its
  own, eligible by total. The model is written down in
  `poker_tracker/math/accounting.py::_build_pots`, its acceptance criteria are
  `tests/test_accounting_pot_layering_model.py`, and the history is in
  `cv_lab/notes/17_release_adversarial_rounds.md`. The honest verdict on the
  module is still not-yet-caught rather than correct.
- **WHICH dead chips that cap governs is a DECLARED input, never inferred.** The
  ante mode (`hand_settlements.ante_mode`) is one of `NONE`, `PER_PLAYER` or
  `SINGLE_PAYER_TABLE_ANTE`. Under `PER_PLAYER` every ante is capped, which is
  the rule shipped in round 20 and is retained unchanged. Under
  `SINGLE_PAYER_TABLE_ANTE` the consolidated ante is table money: it goes whole
  into the main pot and is never capped against a shorter blind. The two give
  different pots on the same recording — blinds 1/2 with a 2-chip big-blind ante
  and an all-in 1-chip small blind is main 5 one way and main 4 the other — and
  nothing in the action line tells them apart, so a hand containing any ante with
  no declared mode is refused rather than defaulted. The mode names ANTES only: a
  dead blind, missed blind or penalty post is capped under every mode, so one
  hand can run both rules at once on disjoint pools.
- **Externally declared dead money is capped exactly like a recorded forced
  post**, under whichever rule the mode selects for the capped pool. It used to
  join the main pot whole and unwarned, which paid a seat that had committed 2
  chips as much as 312.
- **A folded seat's forced post that no surviving seat could cover belongs to the
  pot.** It no longer blocks the hand; a button that antes 50,000 and folds
  against two 20,000 stacks settles as one 90,000 pot. The layering already
  produced that pot, so this is a study-readiness change and not a layering
  change. What still refuses is the shape the rulings do not reach: dead money
  with no unfolded contributor at all, where rule 3 leaves nobody eligible for
  the main pot and there is no layer to award.
- **What is dead is decided by what the row IS, not by the kind that carries it.**
  A forced post which took its poster's last chip is routinely booked as `all-in`
  with `actions.forced_bet_type` / `actions.is_live_post` carrying the truth, and
  both columns are operator-editable on every action row. The money classifier
  read `action_type` alone, so that row was counted as chosen live money — and
  under the live-level model live money is the only thing that opens a boundary,
  so a dead ante became a live level and the seat was paid live chips no opponent
  had wagered against it. `_is_live_money` now decides it through the same
  `_is_forced_post` / `_is_live_structural_post` pair the blind-structure refusal
  already used, so every spelling of one event derives byte-identical chips.
- **OPEN, and refused rather than answered: an unmatched forced post larger than
  a main-pot seat's whole commitment.** Rule 2 is unconditional and worked
  examples (a) and (d) both require it — in (a) the big blind's unmatched 10 ante
  sits in a main pot the two deep seats may win. But in all four worked examples
  every forced post is within reach of every seat that may win it, so none of them
  decides the case where it is not: antes of 100 with a 40-chip stack short of its
  own ante pays that stack all five opponents' full antes (540, where each covered
  40), and a button ante of 200 against a one-chip all-in pays that seat 204.
  Capping a forced post at the shortest main-pot seat's total commitment
  reproduces all four worked examples AND both of those hands, so the reading is
  genuinely open and only the operator can close it. Until then the ledger changes
  no chip and emits a named warning; `_cross_check` folds ledger warnings into its
  issues, so such a hand is `needs_correction` and never authoritative. Pinned by
  `tests/test_accounting_pot_layering_model.py::test_a_forced_post_no_main_pot_seat_could_cover_is_not_study_ready`
  and
  `tests/test_hand_accounting_service.py::test_a_forced_post_no_seat_could_cover_is_refused_as_study_ready`.
  Note that `_model_payout_cap` in the property suite encodes rule 2 as written,
  so the suite will actively reject the capped alternative; it is evidence that
  the code matches the model, never that the model is right.
- **Dead money and a declared rake are mirror images and both are measured.** Dead
  money creates chips the observed action line never saw; a rake policy destroys
  them. Neither widens a tolerance — both move the derived side of the
  cross-check, which is why the dependence rule measures chips rather than
  inspecting fields.

### Tests

- Fresh schema and every historical migration path, from real physical DDL per
  version, seeded in every table that version had, asserted row-for-row intact
  after migrating (`tests/test_migration_matrix.py`,
  `tests/legacy_schema_fixtures.py`).
- Rollback from a migration failure without partial schema state.
- Manual-hand compatibility and the conservative CV-hand migration.
- Export/import v1–v6, including the fuzz and validation suites.
- Readiness truth table covering every blocker and combination.
- UI attempts to bypass readiness through direct status controls.
- Forged-payload bypasses (`tests/test_phase1_readiness_bypass.py`).
- The dependence rule and its consumers
  (`tests/test_phase1_assumption_dependence.py`,
  `tests/test_phase1_declared_inputs_and_consumers.py`).
- Per-round adversarial regressions, `tests/test_phase1_adversarial_round2.py`
  through `round15.py`. Which round pinned which finding is recorded in
  `cv_lab/notes/16_phase1_adversarial_rounds.md` §7 rather than here.
- `tests/test_operator_state_isolation.py`, which fails if any operator root
  stops being redirected away from the real database during a test run.

### Exit gate

- Every reconstructed hand has an explicit completion classification.
  **Satisfied** — the v13 migration classifies every existing row, `Hand`'s
  source-aware validator classifies every new one, and the CV exporter attaches
  versioned evidence.
- No partial, uncertain, unreconciled, open-issue, or stale-evidence hand can be
  presented as study-ready. **Satisfied** — `update_hand_status` refuses the
  promotion in the store, `guarded_update_hand_status` is the single UI writer
  behind every review-status surface, and the Study status control does not offer
  `reviewed` while any blocker stands.

Both bullets are satisfied and neither is sufficient: the phase is gated on the
stopping rule, not on these two.

### Verification record

Taken on 2026-08-02 against the tree at commit `ba3cb2d`, macOS 24.6.0
(darwin/arm64), Python 3.13.5, OpenCV 4.11, NumPy 2.2.6. Re-derived from a fresh
run, not carried forward.

| Command | Result |
| --- | --- |
| `python -m pytest -q` | `3066 passed, 6 skipped` in 199s |
| `python -m ruff check .` | `All checks passed!` |
| `python -m mypy` | `Success: no issues found in 14 source files` |
| `git diff --check` | no output, exit 0 |
| `cmp AGENTS.md CLAUDE.md` | no output, exit 0 |

The six skips each name an external condition a reader can evaluate, which is
what `python -m poker_tracker.suite_quality skip-policy` enforces: four are
`RLIMIT_AS` on Darwin (`test_cv_resource_bounds`,
`test_runtime_limits_single_implementation`, two in
`test_solver_failure_injection`), one is the migration matrix's
"newest version has no later migration to lack", and one is
`tests/test_ocr_readers.py::test_without_chip_template_chip_would_join_run`, a
negative control that skips when the synthetic chip glyph lands below the
classifier's confidence floor, in which case the misread it demonstrates does not
occur. The four `RLIMIT_AS` skips mean the memory-cap paths are **unexercised on
this host** and are covered only on Linux; that is a real coverage limit of this
run, not a clean result.

What mypy does and does not cover, stated rather than implied: 14 files, listed
in `pyproject.toml`, with `follow_imports = "skip"`, so even the checked modules
are checked against stubs of their imports. `app.py`, `persistence/db.py`,
`persistence/import_export.py` and `math/accounting.py` are **not** type-checked.
Widening it is Phase 0 hygiene, not a Phase 1 gate, and it must not be reported
as whole-repository type coverage.

A fresh database initialises at schema 20. The suite claims `POKER_DB_PATH` and
`POKER_DATA_DIR` before its first `poker_tracker` import, so no test reaches any
operator root; `data/backups` and `poker_tracker.db` are byte-identical across a
full run. Loading a plugin from inside the `poker_tracker` package with `-p`
defeats that redirect and `tests/conftest.py` refuses to run when it detects it.

### Known open gaps

Recorded rather than fixed. None of these affects correctness, safety, data
integrity, or the release claim on its own; several are upstream of a phase that
has not run yet.

- **The dependence rule costs a read up to four extra ledger builds on a showdown
  hand and five on a fold win**, the commonest shape there is. The ceiling is
  derived from a build counter in
  `test_the_documented_dependence_cost_ceiling_is_the_measured_one` rather than
  from prose. `math.analytics.compute_session_stats` reconciles once per hand with
  no cache, so Insights pays that multiple per hand of a session, and
  `app.show_insights_workspace` computes its "Not study-ready" KPI with a full
  reconcile plus four per-hand fetches for every hand in the database on every
  render. Both are caching questions rather than correctness ones — every figure
  produced is the same figure — and belong to the Phase 13 performance pass.
- **The `payout` term of a measured dependence is an unsigned magnitude** — the
  largest absolute per-seat change — while its four siblings are signed
  differences of a single figure. Two seats' payouts can move in opposite
  directions under one declaration, so there is no single direction to state, and
  the term collapses a per-seat vector into one scalar. `describe()` words it as
  the magnitude it is. The information-content limit is recorded rather than
  repaired: making it a signed per-seat vector would lapse every stored
  attestation. A sweep of 69,156 states found zero code-set collisions covering
  different figures.
- **`completion_evidence.layout_supported` is not wired to a registry of certified
  ClubWPT geometries.** The CV exporter sets it from "a table size was resolved
  and no state disagreed about the hero seat"; nothing compares the recording's
  resolution, crop, scale, or client skin against a validated profile. All 10
  hands across both committed recording fixtures export `layout_supported=True`,
  so `UNSUPPORTED_TABLE_LAYOUT` fires on none of them. The blocker is therefore
  **inert on the current pipeline, not conservative**, and must not be counted as
  protection it does not provide. Building the registry is Phase 2 corpus work
  ("at least one unsupported geometry used to test safe rejection") and Phase 5
  region-detection work.
- **The first hand of every recording is classified `partial`**, because the
  segmenter cuts a hand only at a positively detected fresh deal. Deliberate
  under-claiming; Phase 5 may add a pre-deal boundary read that proves it.
- **A version 6 JSON export cannot be read by an older release.** The payload is
  strictly additive but there is no option to emit an older payload. Documented in
  `README.md`.
- **`export_hand` emits no `solver_runs`**, so an export/import round trip
  silently discards every solver run, including live ones, and
  `STALE_SOLVER_EVIDENCE` is cleared by the round trip. Defensible — the imported
  database genuinely holds no solver run — but it is an asymmetry with coaching,
  which is exported and whose stale blocker survives. Closing it means adding a
  section to the export payload rather than editing a check.
- **An imported settlement's `status` is still taken from the payload.** It buys
  nothing on its own, because `is_authoritative` additionally requires the ledger
  to be settled, balanced, legal and issue-free and every recorded figure is
  compared exactly, but `import_session` does not re-derive it the way
  `persist_reconciliation` would.
- **A pre-v5 payload carrying no completion evidence has nothing but its declared
  `source_type` to record provenance.** Import refuses a payload that declares
  `source_type: manual` while carrying readable reconstruction evidence, which
  covers everything this build produces, but a legacy v1–v4 CV payload hand-edited
  to `manual` is indistinguishable from a genuine manual hand.
- **A hand entered here by its own operator can record a hero result its own
  action line contradicts and reconcile, by choosing the rake, and is never
  blocked for it.** Every figure on such a hand is the same person's own entry,
  with no independent observation for a disclosure to protect. Its accounting gate
  cannot detect a self-consistent forgery by its author, which is true of every
  field on such a hand.
- **Clearing both `Observed final pot (BB)` and `Hero result (BB)` to Unknown** is
  a reachable way to satisfy the accounting gate, disclosed only by the generic
  `source_facts_corrected` warning. It is auditable — it demotes `reviewed`,
  writes a `hand_corrections` row, and costs an acknowledgement — but readiness
  cannot distinguish "reconciled against the recording" from "reconciled against
  nothing". A reconciliation that then rests on a declared rake or dead money is
  caught by `ACCOUNTING_ASSUMPTION_DEPENDENT`.
- **A hand stored as `manual` with a completion status other than
  `not_applicable`** is treated as reconstructed for readiness (correctly — the
  pair is unproven), but three of its blockers name CV reconstruction, ROI
  calibration and re-import, and `_completion_reason` asserts a source the hand
  does not have. No reachable writer produces the state; `_hand_from_row`
  normalises only the mirror pair.
- **Whether a source-fact change is written into the completion evidence depends,
  for two writers, on unrelated state.** `create_hand_player` and `create_action`
  do not force the invalidation, so the evidence write is skipped unless the hand
  happens to carry a saved coaching or hand review. Neither is reachable from the
  UI on an existing hand today, and the naive fix is actively wrong:
  `import_session` and the CV exporter build every hand through those writers, so
  forcing it would demote every freshly imported hand on creation. Closing this
  means distinguishing "populating a new hand" from "adding to an existing one".
- **Reading the version stamp on a WAL database creates `-shm`/`-wal` sidecars
  beside it even when the open is then refused**, so "a database this build
  refuses is never written to" holds for the file and not for the directory. The
  refused file stays byte-identical.
- **`app.show_saved_hands` is unreferenced.** It is routed through
  `guarded_update_hand_status`, covered by
  `tests/test_review_promotion_surfaces_ui.py`, and holds the only per-hand-export
  control, so it is retained rather than deleted. Four of the five review-status
  surfaces are reachable in the running app.
- **Retained defensive code with no reachable producer**, kept and pinned rather
  than deleted: `_card_problem`'s board-count and hero/board-duplicate branches,
  `parse_completion_evidence`'s BLOB-decoding branch, the `JOINT_INPUT` branch of
  the dependence measurement, `_apply_completion_import_defaults`'s redundant
  `review_status` downgrade, and the `BLOCKER_ORDER` sort (a no-op against the
  current sequence of `blockers.extend` calls; the observable order is what the
  regression pins).
- **The repository database `poker_tracker.db` holds no sessions and no hands.**
  Re-opening a copy proves the file still opens with every row intact; it
  exercises no hand-level migration. The hand-level evidence is
  `tests/test_migration_matrix.py` and `tests/test_schema_v13_migration_paths.py`,
  which build real old-shaped files and assert the classification and the manual
  `review_status` invariant. **A populated historical operator database remains
  untested against this build**, and Phase 0's "migrated historical database"
  walkthrough is where that gets fixed.

## Phase 2 — Build the private validation corpus

**Status: BLOCKED on the operator's annotation pass. Nothing here is waiting on
code.** No *adjudicated* answer key exists for any recording — locked,
validation, or development. One single-pass artifact does exist and should not be
mistaken for the corpus: `cv_lab/results/ground_truth/v00_hands.json` is a
hand-level key for `clubwpt_session_01.mov`, hand-stitched by one labeler from
human-labeled boxes and template OCR, and the spine's 0-err/hand figure is
measured against it. It is one development recording, it was never second-passed,
and that recording is itself named below as contaminating the detector's training
manifest — so it can support a regression check and cannot support a gate.
Two-pass adjudicated truth is human work, and until it exists there
is nothing for any accuracy, completion, boundary, accounting or safe-rejection
gate to score a prediction against.

This is the critical path for the whole release. Phase 3's framework cannot
return a passing verdict, Phase 5's exit gate cannot be evaluated at all, Phase
6's "re-run the entire locked acceptance set" cannot be satisfied, and Phase 7's
"every study-ready CV hand is authoritative and balanced" is a statement about a
corpus that does not exist. Those four phases are blocked here no matter how much
code they contain.

Two further blockers live inside this phase and are unresolved:

- **The region detector's training manifest contains a locked-test recording.**
  `cv_lab/datasets/yolo_cards_autolabel_v1/manifest.csv` holds 228 rows from
  `clubwpt_session_01.mov`, in both the train and val splits. This violates the
  Split integrity gate: the locked test cannot cleanly measure the shipping
  detector until the model is retrained from a clean manifest.
- **No certified-geometry registry exists**, so `layout_supported` is set from a
  weak heuristic and `UNSUPPORTED_TABLE_LAYOUT` is inert on the current pipeline.
  "At least one unsupported geometry used to test safe rejection" is corpus work
  and is where that gets fixed.

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

**Status: framework implemented; exit gate NOT met because it cannot yet be
exercised.** All three modes, the evaluator corrections, the report contract, and
CI integration are built and tested. `fixture` mode scores retained prediction
timelines; `full` mode decodes real recordings with the pinned models; `container`
mode re-runs the gate inside the pinned image and compares verdicts. Every mode
fails closed, and setup failures (absent vault, absent weights, absent Docker)
exit 2 while genuine accuracy misses exit 1.

What the framework cannot do yet is produce a *passing* verdict, because Phase 2
has produced no answer keys. The committed corpus therefore exits 2, and CI
asserts that it does — a green CI explicitly does not mean a passing release
gate. The exit gate below is met when a real corpus makes the command capable of
returning 0.

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

**Status: implemented and tested; exit gate met for everything testable
locally.** Upload validation, atomic storage, and replacement/truncation
detection were already present. This phase added explicit retention behavior
(`poker_tracker/services/retention.py`, `python -m
poker_tracker.maintenance.retention_cli`), a user-visible storage audit that is
a dry run by default, and the nine failure-injection scenarios that had no
coverage.

Retention has since been repaired four times by adversarial rounds 2 and 3, and
the shape of those defects is worth keeping in view because every one of them
deleted, or offered to delete, a file nothing could rebuild:

- `ARTIFACT_PATH_COLUMNS` missed `regression_cases.fixture_path` and
  `report_path`, so retention deleted regression fixtures — irreversibly, and in
  direct contradiction of the rule that a referenced file is never deletable. The
  list is derived from the columns now rather than kept by hand.
- Paths were compared as strings, so on a case-insensitive filesystem a recording
  stored as `Session.MOV` and recorded as `session.mov` looked like an orphan.
  Identity is `(st_dev, st_ino)` where the file exists and a normalized
  case-folded key where it does not; the textual fallback over-matches on purpose,
  because keeping an orphan costs disk and deleting a live file costs the
  recording.
- The audit was treated as an authorization. A job completing between the audit
  and the sweep made the database start pointing at a file already classified as
  garbage. The reference check travels with the audit and is re-confirmed
  immediately before each unlink, against a database re-read whenever it changed
  underneath.
- A zero window meant "expire everything now" and was reachable by an unset
  environment variable. It is refused on every construction path and every read,
  naming the variable; `--purge-now` is how an operator says they mean it.

Job error messages are redacted at the single write boundary rather than by
whichever writer remembers to — the solver worker did not — and a terminal job
state reports what was actually committed rather than its last progress reading,
which read as "most of my hands got in".


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

Exported coverage on the development corpus moved **14 of 21 hands to 11 of 21**
at the time of that measurement. That was reported as a decrease, not defended as
an improvement: the four lost hands are two with destroyed boards and two whose
own action ledgers contradict themselves (a seat that calls and then folds; a
seat that goes all-in and then folds), all four of which previously shipped at
confidence 0.95 with no tags. One hand was gained.

**That figure is superseded.** The Option A reader contract — a value only when
provably unambiguous, else a named UNKNOWN — and two further adversarial repair
rounds against it moved the measured baseline to **9 of 31 hands exported across
six development recordings** — `cv_lab/notes/14_option_a_unknown_contract_repair_round1.md`
and `15_option_a_unknown_contract_repair_round2.md` carry the current numbers,
and note 13 is the earlier measurement they supersede.
Coverage falling as the reader gets stricter is the intended direction: the
governing principle counts a visibly rejected wrong prediction as a coverage
limitation and a silently accepted one as a release blocker.

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
  Incomplete hero-preflop segments stay in the timeline export for evidence
  review; they land in the session only after an explicit draft add (as
  ``partial`` / ``uncertain`` ``needs_correction`` drafts) so the operator can
  fill blanks and finalize them; hands where hero never played preflop are
  skipped.
  Completeness still requires ``derive_completion_status`` (or an explicit
  operator finalize attestation) — incomplete never lands as ``complete``.
  Finalize is the path for late-joined recordings where the operator still
  reconstructed the whole hand; rejection codes stay in the audit trail but no
  longer permanently block that attestation.
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

**Status: the regression link and issue bundle are implemented; "re-run the
affected corpus slice and the entire locked acceptance set" cannot be satisfied
until Phase 2 exists.** A release-blocking issue can no longer be closed without
a regression observed both failing for the defect and passing after the fix
(schema 17, `regression_cases`). Issue bundles carry the identity of the source
recording, frames, models and environment so an issue stays reproducible after
the models move on.


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

**Status: the ledger has produced a critical in four consecutive adversarial
rounds; the exit gate depends on Phase 2 and cannot be evaluated.**

The property and golden coverage the test bullets ask for exists: chip
conservation across generated stacks, all-in layering, splits and rake policies,
and golden ledgers pinning straddles, dead blinds and three-way side pots. The
property suite that could not generate an ante now generates antes, blinds,
straddles, dead blinds and players all-in for a forced post, because it was
written to catch a bug it structurally could not reach.

What that coverage has not bought is confidence in the module. Seven criticals
have been found here since it was declared correct: an uncalled-bet refund
measured against total contributions, so an unmatched ante or dead blind was
refunded out of the pot; a seat all-in for its ante eligible for no pot at all;
split granularity destroyed by summing the dead money before measuring it; a seat
all-in on forced posts alone contesting a whole live layer and being paid 22
chips of a 23-chip pot instead of 2; and unequal dead money manufacturing a side
pot nobody was all-in for; a short-post refusal decided at the instant the blind
row was reduced, so a seat's ante recorded *below* its blind silently un-blocked a
hand around a pot 10 chips short; and a transposed blind structure salvaged by the
reader into a smaller valid one whose floor then covered the very post it was
declared to expose. **Three of those were introduced by the repair to the previous
one.** Each is recorded, with its sweep, in
`cv_lab/notes/17_release_adversarial_rounds.md`.

One finding from the same round is only PARTLY repaired and is stated here as a
limit rather than a guarantee: the refusal reaches a forced post the recording
*identifies* as one — by an action type of `post_blind`, or by a `forced_bet_type`
naming a live structural bet on a row booked under another type — and nothing
else. `cv_lab/scripts/pipeline/build_yolo_hand_timeline.py` books any seat whose
stack reads zero as a plain `all-in` and drops both markers, so a short blind on
that path is indistinguishable from an ordinary short shove and is **not**
refused. Declaring the structure derives such a hand correctly; the product does
not ask for it. Closing that needs the spine to keep the forced-post identity,
which is a CV-corpus change, not an accounting one.

Every one of them passed `is_balanced`, `is_legal` and the declared-award
cross-check, reached the narrowest population the product has, and printed a
figure an operator would have believed. The current rule — a layer is cut only
where somebody was short of *live* money, and once cut it applies to every seat by
that seat's own total, forced posts included — is verified by a 150,000-hand sweep
finding no seat paid more than the table matched of its own commitment, against
5,286 violations under the previous code. It also rests on a modelling choice no
rulebook settles cleanly, so **the honest verdict on this module is not-yet-caught
rather than correct**, and it is written that way in
`poker_tracker/math/accounting.py` for the next reader.

The exit gate ("every study-ready CV hand is authoritative and balanced") is a
statement about a corpus, so it cannot be evaluated until answer keys exist.


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

**Status: honesty and boundary work done; NOT certified against a real binary,
and the licensing gate is unresolved.**

What landed:

- **The recorded-action mapping is bounded and refuses.** It used to descend into
  the nearest size in the tree with no bound and no warning: a 25 BB bet into a
  5 BB pot, against a tree offering only CHECK / BET 1.65 / BET 3.75, returned
  Hero's frequencies for facing 3.75 with warnings empty, and the retained
  evidence then read "recorded_action: call 25 BB" beside them while the coaching
  prompt was handed both. The limit is proportional to the pot
  (`ACTION_MAPPING_MAX_POT_FRACTION`), because 2 BB of error means one thing into
  5 BB and another into 200, and past it the answer is a refusal rather than a
  quieter wrong answer. Actions carrying no size are matched by name or not at
  all: a check is not a small bet and a raise is not a call.
- **A usable result is defined positively.** A result file of `{}` was accepted
  as a completed run. A usable result now requires an action node, a non-empty
  action list, and coverage of the submitted range
  (`MIN_STRATEGY_RANGE_COVERAGE`); anything else is rejected by name.
- **The failure matrix is exercised**: timeout, cancellation, process-group
  termination, stale heartbeat, missing output, corrupt output and memory
  exhaustion (`tests/test_solver_failure_injection.py`).
- **The honesty labels are asserted rather than asserted-about**
  (`tests/test_solver_honesty.py`): no-rake equilibrium approximation on raked
  source hands, no action-EV or BB-loss claim the retained output cannot support,
  built-in preflop ranges labelled study estimates, suit isomorphism disabled and
  recorded in each run's assumptions.
- **The run parameters are retained on the row**, not only in the run directory
  (schema 18, `solver_runs.run_parameters`, write-once at INSERT). The betting
  abstraction, accuracy target and iteration cap used to exist only as text inside
  a directory the product deletes on hand or session delete, that operators prune,
  and that a container without a persistent mount never has — while the row went
  on presenting its frequencies as evidence. Existing rows read `{}` and the UI
  says the abstraction is unknown; backfilling them with today's tree would be
  exactly the failure the column exists to stop. Deliberately unbackfilled.

**What has not happened.** No TexasSolver binary exists on this machine, so
**not one functional or resource gate in this phase has been executed against a
real solver**. Everything above is verified against fakes, fixtures and injected
failures. "Validate pinned binary/resource discovery on macOS and in both
container architectures", the eligible-spot confirmations across five- through
eight-handed source tables, the range variants, and the one-heavy-job-at-a-time
resource behaviour under a real solve are all unexecuted. Do not read the list
above as certification.

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
- Refuse rather than substitute when the recorded action is not in the tree, and
  reject rather than accept a result that does not describe the submitted range.

### Licensing/distribution gate

**Unresolved, and it blocks publishing an image.** `ultralytics` is AGPL-3.0 and
the reconstruction pipeline depends on it, so the same distribution question that
applies to TexasSolver applies to the *base* image, not only to a solver-enabled
one. Local use is unaffected; publishing an image is what triggers the
obligation. `python -m poker_tracker.maintenance.sbom --format notices` lists it,
and `--fail-on-review` exits nonzero while any such component is present. Resolve
by obtaining an Ultralytics commercial license, replacing the inference
dependency, or satisfying AGPL source-offer obligations. This is not legal
advice; obtain qualified review.

- Treat TexasSolver's license as a release blocker for any distributed or hosted
  solver-enabled image.
- Before publishing an image or enabling a hosted solver, retain either written
  maintainer permission for the intended use/distribution, or a documented,
  reviewed AGPL compliance approach with required source, notices, and offer
  mechanics.
- Until that gate passes, keep the solver as an optional user-installed local
  dependency or locally built image and do not publish the bundled image.
- Generate third-party notices and an SBOM for any distributable artifact.
- Do not present this plan as legal advice; obtain qualified review before
  public distribution.

### Exit gate

- All functional, resource, failure, honesty, and licensing gates appropriate to
  the chosen local distribution path pass.

**Met: no.** The honesty and failure halves pass against fakes. The functional
and resource halves have never been run against a real binary, and the licensing
half is unresolved.

## Phase 9 — Finish and certify coaching

**Status: grounding implemented; live-provider evaluation not run.** Responses
are now checked against the prompt that produced them, so an invented card or a
solver-shaped frequency with no retained solver evidence is caught. Provider
retention, fail-closed behavior and staleness were already present. The opt-in
live-provider smoke test is not part of any automated run.


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

**Status: all seven surfaces built and driven by an acceptance script; narrow-width
rendering and contrast are UNVERIFIED.**

What landed:

- **A population layer.** Insights no longer computes anything over the whole
  hands table. Every figure is drawn from a declared population — confirmed,
  confirmed and reconciled, or all saved — whose rule is written in schema
  vocabulary, and each metric carries its denominator, its coverage, its evidence
  split (manual / CV draft / corrected CV / reviewed), the provenance of every
  hero result it summed, and a sample verdict that refuses to print a rate below
  a 30-hand floor. A win rate that blends a reviewed hand, an unreconciled CV
  draft and a manually entered one is a single figure standing for four different
  kinds of knowledge; it no longer exists.
- **Overview carries three independent axes and staleness**, counts jobs from
  every job on file rather than the six most recent, names the open issues it
  cannot show a row for, and distinguishes a ledger-derived hero result from an
  observed one.
- **Sessions gained the issue, stale-coaching, unresolved-hand and provenance
  panel** the plan asked for, built from counts analytics already computed and
  nothing consumed.
- **Health is surfaced in-product.** Settings → Storage & health: resolved paths,
  the previously CLI-only health audit behind an explicit button, runtime
  configuration reported by name and set/unset only, model weight hashes,
  supported and observed table layouts, retained snapshots with their restore
  procedure. Each check reports in words rather than by colour alone.
- **A redacted diagnostics bundle**, scrubbed through `redact_structure` before
  serialization: resolved configuration, dependency and model identity, layout
  support, row counts and the health report, carrying no hand history, note,
  coaching text, video filename or environment value.
- **Every destructive control writes a purpose-scoped `predelete` snapshot**
  through one helper and refuses to delete when the snapshot cannot be written.
- **Two surfaces can no longer disagree about one hand.** The session browser
  called the shared badge builder without the inputs the library passed it, so a
  hand read "open issue, stale" in one place and clean in the other, and neither
  screen looked wrong on its own.

**What is unverified, and must not be reported as passing:**

- **Narrow/mobile-width rendering.** `AppTest` has no viewport. It executes the
  script and inspects the element tree; it cannot tell whether anything wraps,
  overflows or truncates at any width. Nothing in the suite measures a rendered
  layout, so "test narrow/mobile-width rendering for study views" has not been
  done — it has been written about.
- **Readable contrast.** Nothing computes a contrast ratio anywhere in this
  repository. The theme tokens are chosen by eye. "Provide keyboard-accessible
  labels and readable contrast" is therefore half met: the labels exist and are
  asserted, the contrast claim rests on nobody having measured it.

Both need a real browser — a headless render at fixed widths, and a computed
contrast ratio per token pair against WCAG thresholds. Until then this phase's
Cross-cutting UX bullets are partly unevaluated and the exit gate below is not
met.

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

### What the acceptance script covers

`tests/test_phase10_overview_sessions.py`,
`tests/test_phase10_hands_study.py` and
`tests/test_phase10_insights_settings.py` drive all seven surfaces from `init_db`
forward, first over an empty store and then over a corpus holding every state the
product must keep apart, and assert data-derived claims rather than headings.
Import states the post-session-only boundary in the interface, warns when a run's
table geometry is outside the calibrated range, names the recording behind each
reconstructed hand including after a correction, and says out loud that a
rollback snapshot precedes the first hand import.

What that script cannot do is look at the screen. It has no viewport, so it
cannot see a layout, and it does not compute colours, so it cannot see contrast.

### Still open in this phase

- **Narrow/mobile-width rendering is unverified.** No test measures a rendered
  layout at any width.
- **Readable contrast is unverified.** Nothing computes a contrast ratio; the
  theme tokens in `ui_theme.py` were chosen by eye.
- **ROI calibration preview reproducibility**: preview paths are not yet derived
  from frame and region geometry, so a preview is not reproducible from its
  inputs.
- **The Solver tab echoes the `TEXAS_SOLVER_PATH` it could not find.** A local
  diagnostic rather than a secret, but it is an absolute filesystem path on
  screen.

## Phase 11 — Persistence, portability, backup, and recovery

**Status: migrations verified generatively, backups pinned and inventoried, the
recovery drill executed — against synthetic histories only.**

What landed:

- **Migrations are verified against real historical schemas, generatively.** They
  used to be verified from three hand-picked starting versions against a database
  that already had the current shape, which is why migrations 14 through 17 had
  never been observed doing any work. Every version now has its real physical DDL
  (`tests/legacy_schema_fixtures.py`), is seeded in every table it had, and is
  asserted row-for-row intact after migrating, against an invariant that names no
  table of its own and so covers the migrations not yet written
  (`tests/test_migration_matrix.py`).
- **A snapshot lives with the database it can roll back.** A pre-migration
  snapshot used to go to the operator's backup directory whatever database was
  being opened, so migrating a temporary fixture or a restored copy evicted a
  rollback point of the live database.
- **Every snapshot carries an artifact inventory.** Rows are half the study
  history; the recordings, frames, timelines and solver outputs they point at are
  deliberately not copied, so the inventory is what makes a missing artifact
  reportable rather than discovered during a session.
- **`data_health` derives its artifact list from the columns.** It checked three
  of the nine artifact path columns and reported that all references were present;
  a column added later is now covered by having been added.
- **Recovery has an execution, not only a definition.**
  `python -m poker_tracker.maintenance.recovery` restores a chosen snapshot in
  isolation, refuses to target the live root, and verifies what recovery has to
  mean: schema, foreign keys, counts against the inventory, issue evidence, one
  completed hand read *through the application* rather than by raw select, and
  which artifacts are missing — reported as `PARTIAL RECOVERY` with each file
  named, not as a warning. Its refusal to run is exit 2 and means nothing was
  checked. The procedure is in `docs/RUNBOOKS.md`.

**What has not happened.** The drill has only ever been run against synthetic
histories — fixtures this suite built. No real operator database has been backed
up, restored on a second machine, and opened. The repository's own
`poker_tracker.db` holds no sessions and no hands, so it cannot stand in for one.
"A fresh machine with the repository, environment configuration, persistent data
directory, and verified backup can recover the complete study history" is a claim
about a machine nobody has used.

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

**Status: implemented; no known critical or high finding open, but redaction has
been repaired since this phase was declared done.** Added a process-wide bound on
repeated sign-in attempts (a per-session counter is reset by opening a new tab,
so it would have been decoration), credential scrubbing by shape and by
configured value, and a dependency inventory/SBOM.

Adversarial round 2 then found three redaction defects, all of which made the
product *worse* than doing nothing, because output labelled scrubbed was read as
scrubbed:

- Quoting a secret made redaction strictly worse than leaving it bare — the value
  class stopped at the first space or comma, the closing-quote backreference
  failed, and the whole match failed silently, so passphrase-style secrets were
  fully exposed.
- Only `Bearer` and `Basic` authorization schemes were redacted; `Token`,
  `ApiKey`, `Digest` and the rest printed `<redacted>` beside the intact
  credential.
- The issue bundle redacted its own serialized output, but `json.dumps` escapes
  the quotes in every string field and an escaped key stops matching, so a
  credential pasted in as JSON — the most likely paste — survived. Redaction now
  runs over the structure, before the encoder.

Writing that redaction also surfaced a false positive worth remembering:
`session_id` matched the credential key pattern, and it is a plain foreign key on
almost every row here, so ordinary data was being scrubbed.

**The licensing blocker is unresolved.** The SBOM surfaced it and it was
previously unrecorded anywhere: **`ultralytics` is AGPL-3.0 and the
reconstruction pipeline depends on it**, so publishing the base image raises the
same question TexasSolver raises for a solver-enabled one. See the licensing gate
under Phase 8. This is the one item standing between this phase and its exit
gate, and it is not a code change.


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

**Status: the two container criticals are fixed and the image has NEVER BEEN
BUILT, on either architecture.**

State this plainly, because a "container-ready" claim nobody has executed is
exactly the kind of thing this plan exists to prevent. There is no Docker daemon
on this host. `docker build` has not been run for `linux/amd64` or for
`linux/arm64`, no image has been started, no healthcheck has answered, no
non-root runtime has been observed, and no representative workload has been run
inside a container. Everything in this phase's Docker section is static work.

The two blockers that would have stopped a build are repaired but unproven:

- **The Dockerfile copied four model artifacts that `.gitignore` excludes**, so
  the build only ever succeeded on the machine that trained them. The image ships
  without the large CV weights; they are resolved at runtime through symlinks in
  `/app/cv_lab/models` pointing into `/data/models` on the persistent mount, and
  `deploy/provision_models.py` installs them against `deploy/model_manifest.json`
  by SHA-256, writing to a temporary name and renaming only after the digest
  matches.
- **`eval7` has no aarch64 wheel** and the runtime image had no compiler, so an
  ARM64 build would have failed at pip. `deploy/check_wheel_availability.py` and
  the build-stage split address it.

`deploy/verify_container.sh` and `deploy/tests/test_container_build_contract.py`
exist so the claim becomes executable the moment a Docker host does: the contract
test asserts that every Dockerfile `COPY` source exists in a clean checkout and
that the stage running pip has a compiler, and it runs in the ordinary suite. It
is a check on the *recipe*, not evidence about an image.

**Measurement exists and reports honestly.** `python -m poker_tracker.perf`
measures startup, import, UI render, upload, model initialization, reconstruction
throughput, solver runtime, peak memory, disk growth, temporary files and log
growth, states the host and the conditions once, and reports a figure it could
not take as `null` with a reason — never as `0`, which is the mistake the release
gate already made once. A comparison reports `missing_baseline`,
`missing_current` or `incomparable_host` rather than inventing a verdict. See
`docs/PERFORMANCE.md`.

**CV jobs are bounded.** Wall-clock time and address space are environment
variables, refused loudly when malformed, and a cap that cannot be installed on
this platform stops the job with that explanation rather than running uncapped. A
run killed by its timeout discards its partial timeline, because a partial
timeline that survives is something that can be mistaken for a result. The two
resource-limit implementations that agreed on every input have been collapsed into
one, with a guard that fires if a third appears. Note that `RLIMIT_AS` is refused
by Darwin, so the memory-cap path is **unexercised on this host** and its four
tests skip here.

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

**Status: the suite is built and measured; the one-hour representative-session
gate has never been run.**

What landed:

- **Five integration chains, joined rather than segmented.** They existed as
  disjoint segments whose joins had never been crossed, and crossing them found
  real defects: an explanation grounded in a solver result read back out of SQLite
  rather than the object still in memory, a correction's derivatives asserted
  stale as a set rather than a sample, and a regression link that has to survive
  an export for the resolution gate to mean anything on the other side
  (`tests/test_chain_*.py`).
- **Seven property suites that did not exist now do**, and they found a destroyed
  board scoring clean at full confidence under two of six input orderings, and a
  range weight below printing precision rounding to zero and being emitted as a
  token — the exact silent drop its guard exists to prevent.
- **A flake seen once in twelve runs has a name.** A cached Streamlit resource
  surviving between tests, ordered by 192 of 399 seeds. The mechanism was proven
  by clearing the cache and watching the failing seed pass, not by correlation.
  `poker_tracker/suite_quality/flake.py` reproduces it by shuffling collection
  order between passes through the `sq_random_order` plugin shim.
- **Skip reasons are enforced.** `skip_policy` reads every skip declaration out of
  the source and judges whether its reason names an external condition a reader
  can evaluate; an unconditional skip, a missing reason, a placeholder, or a
  condition nobody has reviewed is a violation. `addopts = "-ra"` puts the reasons
  on screen for every run.
- **Coverage is measured and reported, not gated.** `coverage_report` says which
  core modules the suite actually executes. There is deliberately no `fail_under`:
  a floor rewards executing lines instead of asserting on them, and the finding
  worth acting on is the name of the module nothing runs.
- **The sampler is bounded**, which is a Phase 5 defect this phase's work found.
  Frame sampling ran past the end of the recording — a three-second clip asked for
  ten seconds yielded eleven timestamped states from four distinct images, the
  last frame re-emitted under eight later timestamps as though it were eight
  observations — and the release gate made it worse by passing 86400 whenever a
  manifest case had no duration, which the only committed case does not have. A
  full-mode run would have inferred on roughly 86,398 duplicates of one frame.
  Sampling is bounded by what the decoder has, one decoded frame is never emitted
  under two timestamps, and an unknown duration is probed or fails the case: it is
  missing information, not a day.

**What has not happened.** "Verify a representative full session completes within
one hour on the supported local reference machine" has never been executed. The
harness supports it (`--require-session-check`, exit 4) and no run has been made,
so the Local runtime row of the Hard Release Gates table has no value. Real-video
tests remain blocked on Phase 2.

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

**Status: the operator documentation is written; the release evidence archive
cannot be produced.** `docs/RUNBOOKS.md` covers install, diagnostics, the release
gate, corpus vault, migration, backup and isolated restore, the fresh-machine
recovery drill, failed-job recovery, storage audit, containers, upgrade and
rollback, licensing before distribution, and the issue-to-regression loop.
`docs/CONTAINER.md` covers what a build needs that a `git clone` does not contain
and how to prove an image works — written against an image nobody has built.
`docs/PERFORMANCE.md` covers the measurement harness and its three rules. The
release evidence archive requires a release that can pass, which requires Phase
2.

`cv_lab/notes/` holds the chronological research and adversarial record,
including the Phase 1 rounds (`16`) and the whole-product rounds (`17`) that used
to live inline in this file. Findings there are history: they explain decisions
and are not claims about what is true today.


### README

- Keep the product boundary, supported workflows, setup, environment variables,
  storage layout, solver scope, coaching configuration, tests, data health, and
  container commands current.
- State the certified ClubWPT layout/corpus scope.
- State that CV hands enter the session only after frame validation (auto) or
  an explicit draft add, remain ``needs_correction`` drafts until confirmed in
  Study, and that study readiness and study inclusion stay separate.
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

**As of August 2, 2026 exactly one row below has been evaluated: Component
quality.** Everything from Corpus down either has no value because Phase 2 has
produced no answer key, or has never been executed on this machine — the
container rows because no Docker daemon exists here, the Solver row because no
binary exists here, the Local runtime row because the one-hour representative
session has never been run, and the Licensing row because the AGPL question is
open. The release gate command reflects this: the committed corpus exits `2`, and
CI asserts that it still does, so a green CI explicitly does not mean a passing
release gate.

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

## Where the adversarial gate stands

**Five whole-product rounds have run. Every one of them found criticals. The
clean-round counter is 0 of the required 2.**

| Round | Repairs carried by | What it found |
| --- | --- | --- |
| 1 | `88e24b6` | Container mode read its verdict from a report it could not prove it produced, and was not executing the image's code at all. Plus nine reporting defects that made a run say less than it appeared to. |
| 2 | `e8c0e49`, `ed0f403` | Two criticals in pot accounting, both introduced by this program's own repair to the previous defect. Plus retention deleting regression fixtures, and redaction that made a quoted secret strictly *more* exposed than an unquoted one. |
| 3 | `3c3144e`, `5e1cec8`, `ba3cb2d` | Three more criticals in the same module: a forced-post-only seat contesting a whole live layer, a stale award raising out of the reconciler, and unequal dead money manufacturing a side pot nobody was all-in for. |
| 4 | uncommitted (blind structure) | Two criticals in the blind-structure repair itself: the short-post refusal was decided at the instant the blind row was reduced, so moving a seat's ante below its blind silently turned a blocked hand into a reconciled one; and a transposed structure written through `model_copy` (which skips validators) was salvaged by the reader into a smaller *valid* structure, whose floor then covered the very post it was declared to expose. A third finding — the refusal never reaches a forced post the recording does not identify as one, which is every short blind the CV spine emits — is real, is only partly repaired, and is now stated as a limit rather than a guarantee. |
| 5 | uncommitted (live-level pot model) | One critical in the same module, again in the repair to the previous defect: the live/dead money classifier still keyed on `action_type` alone, so a forced post booked as `all-in` with its `forced_bet_type` recorded — the shape the hand editor and the CV spine both produce — was counted as chosen live money. Under the new live-level model that opened a boundary and paid a seat live chips no opponent had wagered against it, settled, balanced, legal and warning-free. Two further findings are real, reproduced, and **not** repaired: they are consequences of rule 2 of the operator's pot model, not deviations from it, and they need an operator ruling rather than a fifth in-session model rewrite. They are refused as study-ready in the meantime. |

The findings themselves are recorded in
`cv_lab/notes/17_release_adversarial_rounds.md`.

**This is the counter the release stopping rule reads.** Phase 1 ran its own
fifteen-round, phase-scoped series (`cv_lab/notes/16_phase1_adversarial_rounds.md`),
numbered separately, also at 0 of 2. Neither counter substitutes for the other,
and neither has ever reached 1.

Two things about this record matter more than the count:

1. **The accounting module has now produced a critical in five consecutive
   rounds, and four times the critical was introduced by the repair to the one
   before.** Round 2's eligibility override was added to stop a short-stacked hand
   being unrecordable, and it turned a loud refusal into a quiet elevenfold
   overpayment — the worse of the two failures. A repair in this module is not
   evidence that the module is now correct.
2. **Every one of these criticals passed every gate the product has.** Chips were
   conserved, so the hand balanced; no legality rule looks at eligibility, so it
   was legal; the declared-award cross-check compares the operator's award against
   the product's own derived payout, so agreeing with the wrong number on screen
   is what made it reconcile. The gates are not independent of the code they are
   checking, and that is why the stopping rule is two consecutive clean rounds
   with fresh agents rather than a green suite.

Round 6 is the first round eligible to count, once no further change lands.
Repairs land *before* the next round starts, and any code, schema, dependency or
release-configuration change between two rounds resets the counter.
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
