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
  ROI profiles, review evidence, range profiles, and solver runs;
- schema migration protection, WAL operation, JSON import/export, rotating
  pre-import backups, and a read-only data-health/restore audit;
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
- Study controls for correcting hand facts, players, and actions while retaining
  before/after audit history and staling derived coaching/solver evidence;
- a persistent cross-session issue inbox that freezes debugging evidence and
  retains resolution notes;
- authenticated local/container operation, a non-root Docker runtime, pinned
  CV dependencies, healthchecks, and CI;
- 442 passing tests with one intentionally skipped test at the latest inventory,
  with Ruff and the configured MyPy target green.

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

The current `review_status` is not enough to distinguish “the hand ended,” “the
pipeline reconstructed enough evidence,” “the ledger balances,” and “the hand
is safe for study.” Those concepts must be represented separately.

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
- Never use one ambiguous percentage as proof that the whole hand is correct.
- Prevent a partial or uncertain CV hand from being marked reviewed.
- Allow the user to inspect and correct retained partial hands without treating
  them as completed study records.
- Explain why a hand is blocked and what exact action clears each blocker.

### Tests

- Fresh schema and every historical migration path.
- Rollback from a migration failure without partial schema state.
- Manual-hand compatibility.
- Existing CV-hand conservative migration.
- Export/import v1–v5.
- Readiness truth table covering every blocker and combination.
- UI attempts to bypass readiness through direct status controls.

### Exit gate

- Every reconstructed hand has an explicit completion classification.
- No partial, uncertain, unreconciled, open-issue, or stale-evidence hand can be
  presented as study-ready.

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

- Replace the current simple warning subtraction with calibrated per-fact and
  per-hand confidence.
- Measure calibration on validation data, not the locked test.
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
