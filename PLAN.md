# PokerTrainer Plan

This is the canonical implementation and release plan. It describes the current
product, not the original pre-implementation proposal.

## Product boundary

PokerTrainer analyzes saved recordings and completed hands after play. It will
not capture a live table, display a poker-client overlay, recommend actions for
a current hand, or provide real-time assistance.

## Current product

The repository already contains:

- a seven-workflow Streamlit application for overview, sessions, hands, study,
  insights, import, and settings;
- SQLite persistence for sessions, hands, actions, players, settlements,
  coaching responses, videos, CV jobs, ROI profiles, review evidence, and
  auditable reconstruction corrections;
- manual entry plus JSON import and export;
- poker math for cards, ranges, equity, pot odds, EV, ICM, and accounting;
- post-session Theory and Exploit coaching through configured providers;
- optional TexasSolver-backed heads-up postflop cash-game review with selectable
  ranges, saved convergence/assumptions, and solver-grounded AI explanation;
- offline video storage, detached CV reconstruction jobs, recovery, retained
  timelines, backups, draft import, and human correction;
- Study-page correction controls that update the hand database, retain
  before/after facts, stale superseded coaching, require reconciliation, and
  support corrected-hand coaching reruns;
- a persistent cross-session debugging inbox that lets users flag an issue
  without diagnosing it, freezes the current evidence, and retains resolution
  notes for a later developer or agent;
- a two-model CV pipeline, labeling tools, reconstruction validation, and
  regression tests;
- an authenticated, provider-neutral Docker image plus an optional OCI
  deployment reference.

## Release plan

### 1. Stabilize the beta

- Consolidate the current accounting, settlement, session-library, and
  reconstruction-review changes.
- Keep all tests, lint, and type checks green.
- Exercise manual entry, import/export, coaching, and correction workflows from
  a clean database.
- Exercise the flag-now/debug-later queue from Study through resolution and
  export/import.
- Turn the July 23 completed-session recording into a checked-in or reproducible
  regression fixture with explicit expected hand boundaries and action labels.
- Verify interrupted CV jobs fail safely and imports always create a backup.

### 2. Generalize reconstruction

- Validate representative completed-session recordings across supported
  resolutions and crops.
- Relearn or calibrate layout anchors when a supported table layout changes.
- Improve weak preflop-only, folded, truncated, and under-sampled hands.
- Reject uncertain or internally inconsistent hands as `needs_correction`
  instead of silently accepting them.
- Use saved issues and corrections as evaluation and training data, closing
  each issue only after its regression case passes.

### Current reconstruction evidence

The July 23 validation recording is sufficient to continue closed-loop beta
iteration, but not sufficient for unattended or release-trusted import. The
observed session contained seven completed hands plus an unfinished eighth. The
default export returned six of the seven completed hands and also included the
unfinished hand. Representative hero/board cards were visually correct, while
action ledgers still contained non-authoritative and overcommit cases.

Until held-out recordings meet the gates below, reconstructed hands remain
`needs_correction` drafts and must be reconciled before coaching can be treated
as current.

### 3. Meet quality gates

- Card recognition: target at least 98% on a held-out representative set.
- Complete hands: target no more than one scored reconstruction error per hand.
- Accounting: require balanced pots or an explicit correction warning.
- Coaching: keep provider, model, source facts, and confidence visible; never
  represent approximate equity or LLM output as solver truth.
- Runtime: finish one representative session within the deployment timeout and
  memory limits.

### 4. Release and operate

- Build and run the pinned image on `linux/amd64` and `linux/arm64`.
- Validate the pinned TexasSolver binary, one representative 5–8 handed
  heads-up cash spot, timeout recovery, and the 8 GB memory ceiling.
- Keep hosted execution and binary distribution behind a release gate until
  written maintainer permission for that scope or documented AGPL compliance
  is retained with the release.
- Pass authentication, healthcheck, restart, backup, and restore drills.
- Keep local Streamlit and Docker operation as the supported runtime posture.
- If deployment is explicitly requested, choose and validate a host at that
  time; the OCI runbook is one optional reference rather than an active target.

## Later, optional work

- Expand vetted range-profile coverage and add action-EV evidence only if a
  validated solver backend exposes it.
- Local LLM evaluation when it improves privacy or recurring cost without
  reducing coaching quality.
- Additional poker clients or table layouts, each with its own validation set.
- Retrieval over user-owned notes after the core correction and evaluation loop
  is dependable.

PostgreSQL, multi-user accounts, live capture, overlays, and real-time advice are
not part of the current release plan.

## Definition of done

The first release is done when a user can upload a completed-session recording,
receive reconstructed draft hands, correct uncertain evidence, reconcile and
study the confirmed hands, generate clearly sourced post-session coaching, and
recover the application and its structured data from a tested backup.
