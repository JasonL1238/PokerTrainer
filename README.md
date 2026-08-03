# PokerTrainer

PokerTrainer is a local-first, post-session poker study and review workspace. It organizes completed sessions and hands, runs offline video reconstruction, provides poker math and coaching tools, and keeps source confidence visible throughout review.

It never provides real-time poker assistance, live table capture, poker-client overlays, or current-hand recommendations.

The current implementation status, release gates, and remaining work live in
[PLAN.md](PLAN.md).

## Product workspace

The Streamlit application is organized around seven workflows:

- **Overview** — portfolio metrics, recent sessions, and processing jobs.
- **Sessions** — session summaries and compact multi-hand solver-spot entry
  (`x/b3.5/c` lines, single or paste).
- **Hands** — searchable cross-session hand library.
- **Study** — hand replay, recorded math, optional TexasSolver analysis,
  auditable correction, coaching reruns, notes, and review state.
- **Insights** — metrics over a declared population (confirmed / confirmed and
  reconciled / all saved), each figure carrying its denominator, its evidence
  split, and a sample verdict instead of a bare rate.
- **Import** — completed-session video upload and offline CV reconstruction.
- **Settings** — storage and database health, runtime configuration, model
  hashes and supported table layouts, a redacted diagnostics bundle, ROI
  calibration, data transfer, math tools, and coaching configuration.

CV, equity, solver, and coaching output remain separately labeled by source and
confidence. The application does not turn approximate inputs into a universal
GTO score.

Study shows approved hands only (edit and approve happen on Import validation):

1. **Replay** — inspect the full saved action history in a readable vertical
   action list with expanded pot, stack, SPR, and note details. Choose any action
   to update the single table replay to that moment, including the street board,
   pot, remaining stacks, folded seats, and highlighted actor. Table stacks, pot,
   and results always show an explicit **BB** unit.
2. **Analyze** — use quick math, TexasSolver, coaching, or notes.

## Local setup

Requirements:

- Python 3.11+
- SQLite
- FFmpeg for video workflows
- Optional TexasSolver `console_solver` for solver-backed postflop review

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The base requirements support the review application. The cloud CV container additionally installs the pinned dependencies in `requirements-cv.txt`.

By default, structured data is stored in `poker_tracker.db` and file data under `data/`. Override those locations with:

```bash
export POKER_DB_PATH=/path/to/poker_tracker.db
export POKER_DATA_DIR=/path/to/data
```

`POKER_DB_BUSY_TIMEOUT_MS` (default `30000`) is how long a second process — a CV
job, a solver job, or a second app start — waits for a lock instead of failing
with "database is locked". Raise it if a migration on a very large database
outlasts it.

## Shared-password authentication

Local development remains available without a password. Web deployments fail closed when authentication is required:

```bash
export APP_PASSWORD='use-a-long-random-secret'
export POKERTRAINER_REQUIRE_AUTH=true
streamlit run app.py
```

Secrets are read from the environment and are never stored in SQLite.

## Completed-session reconstruction

The Import workflow connects a saved video to the existing two-model CV pipeline:

1. Save and validate a completed-session video.
2. Start one detached `cv_reconstruction` job.
3. Track PID, heartbeat, progress, and safe error state in SQLite.
4. Produce a retained timeline and session export; create or link a destination
   session (no bulk hand import on job finish).
5. Create a consistent SQLite backup.
6. Review frames in Import: opening a hand drafts it into the session so you can
   edit actions, cards, players, and gaps beside the frames. Mark a debugging
   issue to hold a hand out of Study without fixing it now.
7. Finish validation with no open issues to auto-approve the hand for Study
   (partials are fine once you fill missing chunks yourself and readiness clears).
8. Study is study-only (Replay + Analyze). Hands with open issues stay in the
   Hands Issues inbox and deep-link back to Import validation.
9. Retain before/after audit records; reconcile the ledger during validation;
   rerun coaching in Study after approval when needed.

Only one local processing job can run at a time — the Import launcher, the solver
launcher and the reconstruction worker itself all refuse to start heavy work
while another heavy job holds the machine. On restart, dead or stale workers are
marked failed instead of remaining stuck. SQLite uses WAL mode so the UI can
continue reading while the worker writes. Reconstruction uses Apple MPS or CUDA
when available (same models/weights); set `POKER_CV_DEVICE=cpu` to force CPU.

### Bounding a reconstruction

| Variable | Default | What it bounds |
| --- | --- | --- |
| `POKERTRAINER_CV_TIMEOUT_SECONDS` | `3600` | Wall-clock time for one reconstruction, from 60 to 86400 |
| `POKERTRAINER_CV_MEMORY_GB` | unset (no cap) | Address space of the CV pipeline process |
| `POKERTRAINER_CV_MAX_EXTRACTED_FRAMES` | `2000` | Frames one diagnostic frame extraction may retain on disk |

```bash
export POKERTRAINER_CV_TIMEOUT_SECONDS=7200   # a slow CPU-only host
export POKERTRAINER_CV_MEMORY_GB=6            # Linux only; see below
```

A value that cannot be honoured is refused rather than ignored: a malformed
number, a timeout outside the range, or a memory cap on a platform that cannot
install one fails the job immediately with the variable named, instead of
running on the default while you believe your setting is in force.

`POKERTRAINER_CV_MEMORY_GB` sets `RLIMIT_AS`, so it bounds *address space*, not
resident memory — PyTorch reserves far more of the former than it uses, so set
it well above the resident figure you expect. macOS refuses `RLIMIT_AS`
outright; setting the variable there stops the reconstruction with that
explanation rather than running it uncapped. Leave it unset for local macOS use
and set it in the Linux container.

When a reconstruction stops on its timeout, the job says which limit it hit,
which variable sets that limit, and that its partial timeline and review frames
were discarded — a killed run leaves nothing behind that could be mistaken for a
result. Only a completed job keeps its review frames, because the export and the
validated hands point at them.

Sampling is bounded by what the decoder actually has. A run never samples past
the end of a recording and never emits one decoded frame under two timestamps, so
a short clip cannot produce a long timeline out of repeated stills, and a
recording whose duration cannot be determined is probed or fails rather than
being treated as a day long.

Corrections are written transactionally to SQLite. Editing hand facts, players,
or actions changes CV imports to `corrected_cv`, records the original and
corrected values in `hand_corrections`, invalidates settlement, and marks prior
hand/session coaching stale without deleting it. Deleting a hand marks its
session's coaching stale for the same reason.

Everything you declare in the Accounting reconciliation panel — the **blind
structure** (Small blind / Big blind / Straddles), which sets what every seat
owed before the first voluntary action; the **rake
policy**, which removes chips the recording never showed; **dead money**, which
adds them; and the **pot awards**, which decide who is paid and therefore what
the hero result is — is checked by reconciling the hand twice: once with what you
declared, and once with the whole declaration withdrawn. The hand is
**assumption-dependent** when withdrawing it changes either the verdict (it stops
reconciling) or the figures (it still reconciles, but the pot, the rake, a payout
or the hero result come out different). Both halves matter: a freshly imported
hand often records none of the figures the cross-check compares, so it reconciles
under every policy while the declaration still moves the hero result shown in
every list, stat and prompt.

The blind structure is in that set for a different reason: it is the one
declaration that can **block** a hand outright rather than move a figure. A
forced post that took its poster's last chip does not show the size of the bet it
was paying — blinds 5/10 with a big blind all-in for 4 leaves the small blind's 5
as the largest post anybody can see — so where the recording identifies such a
post, the ledger refuses to name an amount to call, `is_legal` goes False, and
the hand blocks on `ACCOUNTING_NOT_AUTHORITATIVE` until you fill the fields in.
Every chip figure is still derived and displayed while it is blocked; nothing is
hidden and nothing is guessed. It is a floor and never a ceiling, so a declared
structure can only ever raise what a seat owed and can never excuse an under-call
the recording proves. Two limits are worth knowing: the refusal only reaches a
post the recording *identifies* as forced (a reconstructed all-in that carries no
forced-bet type is indistinguishable from an ordinary short shove and is not
refused), and no existing hand was backfilled with a structure, because inferring
one from the largest observed post is the defect itself.

The **ante mode** is in that set for both reasons at once, and it is the only
declaration that does both. It says how this hand's antes were taken — *no
antes*, *per-player antes*, or *one consolidated table ante* (a big-blind or
button ante) — and the two ante readings lay the same recording out differently.
A per-player ante is capped at the shortest seat's total commitment in the layer
it sits in and the excess rises; a consolidated table ante is table money and
sits whole in the main pot, uncapped. Blinds 1/2 with a 2-chip big-blind ante and
an all-in 1-chip small blind is a 5-chip main pot one way and a 4-chip main pot
the other, and nothing in the action line tells them apart. So a hand that
contains any ante and does not declare a mode is **refused**: `is_legal` goes
False and the hand blocks on `ACCOUNTING_NOT_AUTHORITATIVE` until you choose the
mode in Edit settlement. It is never inferred — one seat anting looks identical
whether it is a big-blind ante or a late-entry seat posting its own. A hand with
**no antes at all is never asked for a declaration**: *no antes* is not a guess
for it, so the absent mode resolves silently and nothing blocks. (That is the
*declaration*. A hand with no antes can still be re-derived by the amended
external-dead-money rule shipped in the same release — see the migration note
below.) The mode names antes only; a dead blind, a missed
blind or a penalty post is capped under every mode, so a hand can run both rules
at once.

Externally declared **dead money** is now capped like a recorded forced post: a
seat collects it only up to its own total commitment and the rest rises to the
seats that committed more. It used to join the main pot whole, which paid a seat
that had committed 2 chips as much as 312.

The pot awards are in that set because on a reconstructed hand nothing observed
them. The CV pipeline emits no settlement rows at all, so the winner of every pot
is typed in — in the same panel, in the same save, as the rake — and it is the
single input the payouts and the hero result are computed from. On a hand with no
recorded pot size and no recorded result, choosing the other winner moves the
reported result by the whole pot.

An assumption-dependent hand is blocked from study until you press **Confirm this
assumption** beside each listed declaration in Study → Summary → Accounting
reconciliation, which records the exact chip movement you are attesting to.
Nothing else clears it: not the "I confirm this hand is correct" tick, which is a
judgement about the reconstruction rather than a claim about the room, and not
the **Acknowledge** button in the Source warnings panel, which answers pipeline
codes and is never offered a settlement assumption.

Confirming is also what re-enables everything the block disabled. Until every
measured declaration is answered, the hand's derived result is left out of the
session win rate, out of the hero-result column of every list, and out of the
math facts handed to a coaching provider, and the hand is not solver-eligible;
once answered, it is published everywhere at once. A hand you entered yourself is
answered by definition and is never held back.

Because the check measures chips rather than inspecting fields, a declaration that
moves nothing is never raised: a rate with a zero cap, no-flop-no-drop on a hand
that saw no flop, or a chip unit coarser than the whole rake are all silent. And
because the confirmation carries both the measured amount and the declaration it
was measured on, it covers that declaration only — if a later correction grows the
pot the same policy applies to, or the policy changes to one that happens to
remove the same chips, you are asked again.

A separate, simpler measurement — "does this declaration move chips at all?" —
records `declared_unobserved_chips` and `declared_unobserved_rake` as an audit
note on the hand's own evidence, shown as **Declared settlement inputs**. They are
not pipeline warnings and never appear in the Source warnings panel: what the
pipeline could not prove and what you declared are different claims, and nothing
in readiness reads the second.

The exemption is for hands **you entered here**, where every figure is your own
entry: those are measured and reported but never blocked, and never withheld from
coaching, the win rate or the solver either — an ordinary room rake on a hand you
typed in is your own observation, and there is no attestation for the product to
ask you for. It is not granted by the
`manual` label, because an import payload can write that label. Every hand that
arrives through import is stamped as imported and attests to its own declared
assumptions in the database it lands in — the same reason import never lands a
`reviewed` status and never carries your acknowledgements across.

The settlement editor's **Replace observed final pot/result with the derived
ledger values** rewrites `pot_size` and `hero_bb_won` — the hand's *observed*
summary — from the reconciled ledger. It is offered because a freshly
reconstructed hand often records neither, and once you have verified the action
line and the winners the ledger is the best record there is. It is refused while
the reconciliation is not established: an unbalanced or illegal ledger, or a
declared settlement input you have not confirmed. Confirm the assumption (or
correct the declaration) and save again, and it writes; otherwise the settlement
is saved and the recorded pot and result are left exactly as they were, with the
reason named. A derived figure that rests on an unconfirmed declaration must not
become an observation, because the recorded pair is the independent evidence the
cross-check compares the ledger against.

The settlement editor's **Chip unit** rounds the rake and nothing else. It is a
room rule — a house that drops whole dollars against a 0.50 blind is ordinary —
and it no longer decides the granularity a chopped pot is divided at. That
granularity is derived from the hand's own amounts: the finest decimal place any
observed contribution or declared dead-money figure is written in, capped at one
whole chip. Each declared ante or dead blind counts individually, so four 0.25
antes prove the table deals in hundredths rather than summing to a whole chip and
destroying them. Indivisible chips are still real, so a 21-chip pot chopped two
ways is still pushed 11/10 in the audited `Order` column's order, and a rake share
is never charged to a pot beyond what that pot holds.

**Side pots are cut only where somebody was short of live money** — where a seat
declined or could not answer chips an opponent actually risked. Nobody can decline
an ante or a blind, so unequal dead money never creates a side pot, and being
all-in does not either when the all-in covered the wager. Once a cut is drawn it
applies to every seat by that seat's own **live** contribution: an opponent never
matches your ante, so your own forced posts cannot raise the level the rest of the
table is charged into the pot below at. Every dead chip — antes, dead blinds and
any dead money you declare — is owed to the table rather than wagered, so it is
never handed back as an uncalled bet, and a seat all-in for nothing but its ante
stays eligible for the layer holding its chips. It does not all land in the main
pot, though: as described above, each seat collects dead money only up to its own
total commitment and the excess rises into the layer above, so nobody is paid out
of chips they could never have matched.
Which chips are dead is decided by **what the row is, not by how it was spelled**:
a forced post that took its poster's last chip is often recorded as an all-in, and
the `Forced post` and `Post status` fields on an action row are what the ledger
reads, so relabelling one row cannot move a chip. The `Forced post` field says
*which* forced post a row is; it cannot make one. Only `ante`, `post_blind` and
`all-in` rows can be posts — a bet, a call or a raise answers a wager level,
which is the one thing a forced post never does — so a `Forced post` value on one
of those is a contradiction between two things you told the product. It derives
the row from its action type, exactly as if the field were blank, and reports the
contradiction as a legality issue naming the row. Clearing the field or
correcting the action type in `Edit actions` resolves it.

**One case is deliberately refused rather than answered.** When a seat is all-in
for less than *another* seat's forced post — a stack short of its own ante, or a
one-chip all-in against a button ante — the main pot hands that seat the whole
forced post, more than any opponent covered of it. Whether an unmatched forced
post should be capped at what the winner covered is a question about the pot model
itself, and the ledger does not answer it: the chips are still derived and the
hand is still legal and balanced, but it is reported as **not study-ready** with
the seat, the poster and both numbers named, so it can never be published as
authoritative while the question is open. If the editor shows you a pot
you cannot explain from the action line, that is worth reporting rather than
declaring around: the honest verdict on this part of the ledger is *not yet
caught* rather than *correct*, and five separate adversarial rounds have found
criticals in it.

What you type in `Chip unit` **does** change derived payouts, because rounding the
rake changes the net pot every payout is drawn from. On an 80-chip pot at a
declared 50% rake, a unit of 1 charges 40 and pays the winner 40; a unit of 3
charges 39 and pays 41; a unit of 81 charges nothing and pays 80. Read it as
carefully as the rake rate itself before you press **Confirm this assumption** —
it is a declared settlement input like any other, it is measured like one, and a
value that moves chips is named and blocks the hand until you attest to the
movement. (A previous revision of this paragraph said no value you type here
changes a derived payout. That was false, and false about the one field two
adversarial rounds landed critical findings on.)

If the cause is not known yet, use **Flag this hand for future debugging**.
PokerTrainer saves the issue categories, description, and a snapshot of the
hand, session, players, actions, and correction history in `hand_issues`. The
Hands page provides a cross-session inbox of every unresolved issue. A later
debugging pass can download the snapshot, fix the pipeline, and record resolution
notes without losing the original evidence.

A hand is only presented as study-ready when every blocker clears: explicit
completion, valid and unique hero/board cards, confirmed table layout,
authoritative reconciled accounting, no open debugging issue, no unresolved
source warning, no stale coaching or solver result being shown as current, and —
for a reconstructed hand — your explicit confirmation. The Study page lists each
blocker by category with the exact action that clears it. Reconstruction
confidence is shown as a bucketed label only; no single percentage is ever
presented as proof the whole hand is correct. Partial and uncertain hands stay
fully inspectable, replayable, and correctable — they simply cannot be marked
reviewed.

Every blocker names an action you can actually take. The two stale-evidence
blockers each carry an escape hatch for the case where re-running is not
available to you: **Delete stale run** removes a solver result a correction
invalidated when the hand is no longer solver-eligible, and **Discard stale
coaching** deletes a stale retained review when no LLM provider is configured.
Importing a session marks every coaching review stale by construction, so without
that control an imported hand would sit behind a blocker naming a button you
cannot press. Neither control touches a current result. The solver control is not
drawn while a cancellation is still in flight — that blocker says to wait for the
cancellation instead.

`Reviewed` is a workflow label, not a readiness verdict, and PokerTrainer never
presents it as one: readiness is re-derived on every render, and any edit to a
hand's players, actions, or settlement returns a reviewed hand to
`needs_correction` so a promotion cannot outlive the evidence it was granted on.

JSON export version 6 carries correction, issue, coaching, completion history and
per-action source-frame provenance through backup/import workflows. Import still
accepts versions 1-5 and gives them safe conservative defaults. Importing a
session lands every hand as
`needs_correction`, whatever review status and whatever `source_type` the payload
declares: your confirmation that a hand is correct is deliberately per-render and
never persisted, so it cannot travel in a file, and the importing operator has not
yet seen the evidence. (A genuine manual export loses its `reviewed` label too,
because it is byte-identical to a forgery of one, and the label is one tick and
one save away for the operator who now vouches for it. The v13 migration is
different: it keeps manual review statuses, because a migrated database is your
own data rather than somebody's JSON.) For the same reason, source-warning
acknowledgements do not travel either: an export of a hand whose warnings you
accepted re-imports with those warnings unaccepted, so the importing operator
accepts them themselves. The codes are preserved in full — only your attestation
to them is dropped. Settlement-assumption confirmations are reset for the same
reason, and the dependence is simply re-measured and asked again. A debugging
issue you resolved re-imports **open**, with your resolution notes carried into
its description as history: resolving is an assertion that somebody looked at the
hand and fixed the thing, and the importing operator has looked at nothing.
Retained coaching travels the same way: the text of every saved review is imported
in full and marked *stale*, because a review describes the hand, ledger and winners
of the database that wrote it and nothing in the importing database can verify that
claim. Re-run coaching there to make it current.

Re-importing a file you already imported still appends a second session — import
never addresses or replaces an existing one — but the copy is recognised from the
rows it wrote, renamed `<name> (re-imported copy #<id>)`, and annotated with the
session it duplicates. **A labelled copy is left out of every portfolio and
Insights total**: Overview names it and states the sessions, hands and BB it
dropped, the session list keeps a row for it marked "Not counted", and Insights
reports it under "In a session the importer labelled a re-imported copy" beside
the denominator, the same way every other population exclusion is reported. The
copy is still fully browsable and studiable; delete it (Sessions → Delete session)
to remove it entirely.

The database is at **schema version 20** and every migration is additive: v10
added correction history and review staleness, v11 solver records, v12 the
debugging issue queue, v13 explicit hand completion
(`complete`/`partial`/`uncertain`/`not_applicable`) and its versioned
reconstruction evidence, v14 video content hashes, v15 per-hand study inclusion,
v16 the source frame behind each reconstructed action, v17 the regression that
proves a closed issue stays closed, and v18 the tree, accuracy target and
iteration cap a solver run was produced under, v19 the declared blind
structure (`hand_settlements.small_blind`, `big_blind`, `straddles`), and v20 the
declared ante mode (`hand_settlements.ante_mode`). Existing manual hands became
`not_applicable` at v13; existing reconstructed hands migrated conservatively to
`uncertain` and `needs_correction` and must be re-confirmed. Existing rows remain
intact, and the later migrations are deliberately unbackfilled: an old solver run
reports its abstraction as unknown rather than being labelled with today's
settings, and v19 declares no blind structure for any existing hand rather than
guessing one from the posts it can see. **v20 is the one migration that visibly
blocks stored hands**: every hand containing an ante reads `ante_mode IS NULL`,
which is ambiguous rather than defaulted, so hands that previously reconciled
start refusing, with the anteing seats named, and stop being study-ready. No row
is rewritten and the layers shown beside the refusal are the capped reading, and
one ordinary settlement save clears each hand.

Three things about that population are worth knowing before you go looking,
because the surfaces disagree about it:

- **`review_status` is not demoted, by design** — the migration will not discard
  a confirmation to announce a change. A hand you had marked `reviewed` still
  reads *Reviewed* in the Hands library. The library's badges cover evidence
  class, open issues, stale analysis and reconstruction confidence, and none of
  them is readiness — so unless the hand carried retained coaching or solver
  output (which v20 does mark *Stale analysis*), nothing on that row says the
  ledger stopped reconciling. Insights still admits it to **Confirmed hands**,
  whose "Admits" sentence claims the promotion guard only allows a hand in "once
  every readiness blocker is clear". For this population that sentence is false:
  the guard never ran on it. The same Insights page counts the hand under its
  *Not study-ready* tile. **Read the readiness tile, not the review badge**, and
  prefer the **Confirmed and reconciled** population, which does require the
  ledger.
- **The Study checklist drops the detail.** It shows the blocker's headline —
  "The chip ledger does not reconcile" — and points at *Edit this hand → actions
  or Chip stacks*. The blocker itself names the ante mode, the anteing seats and
  the three choices; that text is only rendered in the accounting panel. Open the
  hand's accounting panel to see what to declare.
- **A manually-entered hand with no linked recording has no route to the
  settlement editor.** The editor lives inside the CV reconstruction validation
  flow, and Import shows only "No videos linked yet" for such a hand.

Count the ante population with `SELECT
COUNT(DISTINCT hand_id) FROM actions WHERE action_type = 'ante' OR
forced_bet_type IN ('ante','big_blind_ante')`.

**v20 touches a second, wider population, and this one does move chips.** The
same release caps dead money against each collecting seat's own total
commitment instead of dropping it whole into the lowest layer. That covers dead
money the recording holds — an ante, a dead or missed blind, a penalty post —
*and* the external amount you typed into `dead_money`; a hand can carry the
first with the second at zero. Where a dead contribution exceeds the smallest
commitment contesting the layer it sits in, the pot count, the eligible sets and
the gross all stay the same while the *distribution* changes — so every
cross-check still passes and the hero result moves anyway. Those hands need no
declaration and may contain no ante at all. The new figure is the right one, but
coaching and solver output retained beside them was written against the old one,
so v20 marks that analysis stale and study readiness blocks on
`STALE_COACHING_EVIDENCE` until you rerun it. Your `review_status` is not
touched and nothing is deleted — only the freshness flags move. The predicate is
"this hand holds dead money" rather than "a dead contribution exceeds the floor",
because a schema migration cannot run the reducer to find the floor, so it is
deliberately over-strict: some hands whose figures did not move ask for a
coaching rerun. A hand whose forced posts are all live blinds — every ordinary
cash-game hand — holds no dead money and is untouched in every respect. Startup
migrates older
databases automatically; older application versions intentionally refuse to open
a newer database, and a version 6 export cannot be read back by an older release,
so keep a copy of any pre-upgrade export you may need to restore. A database whose
schema version stamp is missing, unreadable, behind the schema the file actually
contains, or ahead of it is refused rather than re-migrated, and the message says to restore from a
backup: re-running the version 13 migration against a live database would discard
every review confirmation recorded since it first ran. A brand-new database cannot
reach that state: the base tables, the migration chain, and the version stamp are
written in one transaction, so an interrupted first start — a power loss, an OOM
kill, a container restart, Ctrl-C — leaves an empty file the next start simply
creates again.

A consistent, self-contained backup is written before any real file database is
migrated, and each snapshot is left in `journal_mode=delete` so it can be
verified and restore-drilled from a read-only or archival mount. A snapshot goes
to the backups directory of **the database it can roll back** — `data/backups`
for your live database, and a `backups/` directory beside the file for anything
else — so opening a fixture, a restored copy, or a backup you are auditing can
never evict a rollback point of your real database.
**Migration fails closed:** if the snapshot cannot be written — a read-only
container filesystem, an unmounted data volume, or a full disk — startup raises
and no migration runs. The error names the backup directory it could not write
and states that the database is unchanged, rather than reporting SQLite's
`unable to open database file`, which reads as database corruption. Make the
`data/` mount writable before the first start after an upgrade.

Pre-migration snapshots are **pinned**: they are named
`poker_tracker-premigration-<timestamp>.sqlite3` and the five-slot per-import
rotation never matches them, so routine snapshots cannot delete the only rollback
point for an irreversible migration. They keep their own five retention slots, so
a repeatedly failing migration — which snapshots the whole database on every
start — cannot fill the backup mount either. They are stamped at the pre-upgrade
schema version, so `data_health` verifies and restore-drills them without the
current-version comparison it applies to rotating snapshots. Delete them yourself
once the upgrade is proven.
Pre-delete snapshots are pinned the same way. Deleting a session, a hand, or an
ROI profile writes `poker_tracker-predelete-<scope>-<timestamp>.sqlite3` first,
in its own five-slot pool, and the deletion does not happen if the snapshot
cannot be written. Settings -> Storage & health lists every retained snapshot and
states the restore procedure. Snapshots hold rows only: videos, frames, timelines
and solver outputs are deliberately not copied, so a snapshot restored after those
were removed will reference files that are gone. Each snapshot therefore carries
an **artifact inventory** beside it, recording what the rows pointed at, so a
restore can report which files are missing instead of leaving you to find out
during a session.

Per-import snapshots keep the rotating `poker_tracker_<timestamp>.sqlite3` name
and the newest five are retained. Rotation matches that exact timestamped name and
nothing else, so your own files in `data/backups` are never deleted even when they
start with `poker_tracker_`. Opening a database this app refuses — one
written by a newer release, or one whose version stamp is unreadable — never
writes to the file, so an archival or read-only restore mount is safe to point at.

File layout:

```text
data/
  backups/       pinned pre-migration, pre-import and pre-delete SQLite backups
  cv_timelines/  retained reconstruction timelines
  exports/       generated session exports
  frames/        diagnostic extracted frames
  job_logs/      detached worker logs
  roi_previews/  ROI crop previews
  solver_runs/   TexasSolver commands, logs, and raw strategy JSON
  videos/        uploaded completed-session videos
```

Videos remain files; SQLite stores structured records and paths.

## Coaching providers

Coaching uses a configured external provider. If no key is available, coaching
stays unavailable instead of generating substitute content:

```bash
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
```

Every generated prompt passes the post-session safety check and can be inspected
before use. **The answer is checked too, and the two checks are not the same
verdict.** The prompt check confirms the question was a post-session one; the
grounding check compares the response against the prompt that produced it. It
covers two fabrication classes: **cards the prompt never contained**, and
**solver-shaped quantities** — an action frequency, an EV figure or an
exploitability number — asserted with no retained solver evidence carrying that
figure. It does **not** check invented pot sizes, invented actions, or a bare
"loses 4.2 BB" that does not name EV; those pass.

That check runs where the response becomes a database row, so no coaching
surface can skip it. A response that fails is still saved and still shown
verbatim — you paid for it, and a rejection you cannot read is no better than no
check — but it is saved **stale**, with the reason on the row:

- it is kept out of the current-review list and shown under retained stale
  evidence, labelled `STALE` with its reason above the text;
- it does **not** promote the hand to `reviewed`, whatever the hand's readiness
  says. The message names the rejected review rather than the hand's readiness,
  because those are different problems;
- it raises `STALE_COACHING_EVIDENCE`, so the hand is not study-ready until you
  re-run coaching or discard the retained review in **Analyze → AI coach**.

The detector is deliberately biased toward silence on ambiguous text, so a
reported failure is worth investigating rather than routine; it is not a proof
that a response that passed is correct.

One limit worth knowing: a provider *failure* is displayed as the client library
worded it, without the redaction pass the health-check and diagnostics surfaces
apply. No failure reachable offline echoes a key, so this is an unguarded path
rather than a known leak — but do not paste a coaching error into a bug report
without reading it first.

## TexasSolver postflop review

The Study workspace can run TexasSolver against a completed, reconciled,
heads-up NLHE cash-game postflop spot. Five- through eight-handed tables are
supported through their input ranges; the solver itself receives the two
remaining postflop ranges.

### Using the integration

1. On **Import**, validate frames and edit the hand beside them. Use **Other
   fixes** for cards, players, or chip accounting. A late-joined recording can
   still be finalized if you reconstructed the whole hand — use **Finalize
   incomplete hand**. Mark a debugging issue to hold a hand out of Study.
2. Press **Finish validation — send to Study** (or let auto-approve run when
   every frame is Correct and readiness clears).
3. Open **Study** → **Analyze → TexasSolver**. The app checks eligibility and
   explains every item that still needs correction.
5. Confirm the automatically selected heads-up street, pot, and effective stack.
6. Start with **Default** ranges for both players, or choose a premade/custom
   range when you have a better assumption.
7. Press **Run TexasSolver analysis**. Solver work runs in the background; use
   **Refresh** to check it or **Cancel** to stop it.
8. Review Hero's combo frequencies, convergence, assumptions, the tree the run
   used, and the mapped recorded action. Optionally generate an AI explanation
   grounded in that saved solver evidence.

### When the solver refuses instead of answering

The solve tree offers a handful of discrete sizes, so a recorded bet rarely
equals one of them and some substitution is unavoidable. **An unbounded
substitution is not.** The gap between your recorded action and the nearest
branch is measured against the pot the action was made into — 2 BB of error means
one thing into a 5 BB pot and another into a 200 BB one — and past a quarter of
the pot the app refuses the mapping and says so, rather than answering a
different question quietly. A 25 BB bet into a 5 BB pot, against a tree offering
CHECK / BET 1.65 / BET 3.75, used to return Hero's frequencies for facing 3.75
with no warning at all, while the retained evidence read "recorded_action: call
25 BB" beside them and the coaching prompt was handed both.

An action carrying no size is matched by name or not at all: a check is not a
small bet and a raise is not a call, so when the tree does not offer your action
there is no nearer branch, only a different one.

A finished run is also checked before it counts as a result. A usable result
needs an action node, a non-empty action list, and coverage of the range you
submitted; an empty or partial strategy dump is rejected by name instead of being
retained as a solve. Every retained run also records the betting abstraction,
accuracy target and iteration cap it was produced under, on the run row itself
rather than only in its directory — a frequency vector with no record of the tree
it came from is a claim that cannot be checked. Runs recorded before that column
existed say the abstraction is unknown; they are deliberately not backfilled with
today's settings.

For local use, compile or install the pinned `console` implementation and
configure its absolute path:

```bash
# Apple Silicon source builds require CMake and an OpenMP runtime.
brew install cmake libomp
export TEXAS_SOLVER_PATH=/absolute/path/to/console_solver
# Optional when the installed resources directory is not beside the binary:
export TEXAS_SOLVER_RESOURCE_DIR=/absolute/path/to/resources
export POKERTRAINER_SOLVER_THREADS=4
streamlit run app.py
```

The Dockerfile compiles commit `42313c9c` and configures the resulting binary for
both `linux/amd64` and `linux/arm64`. Container defaults limit the solver to two
threads, 8 GB of address space, a 30-minute run, and one heavy CV/solver job at a
time. **That build has not been performed on either architecture** — see
[Container](#container) — so treat the solver-in-container path as untested.

Each player can use an automatic estimated range, a built-in or saved premade
range, or validated custom weighted notation. Custom syntax accepts both eval7
weights such as `50%(AJs+)` and TexasSolver-style `AJs+:0.5`. Exact range
snapshots, convergence, assumptions, logs, and raw JSON are retained with the
run. Built-in ranges are study estimates, not solved preflop GTO.
The blocker-filtered adapter sends exact suit-specific combinations; the pinned
console cannot safely apply suit isomorphism to those ranges, so this optimization
is explicitly disabled and recorded in each run's assumptions.

TexasSolver output is used only as evidence for an optional AI explanation.
Raked cash results are labeled no-rake equilibrium approximations, entirely
multiway and tournament/ICM spots are excluded, and the app does not report
exact BB loss because the standard strategy JSON does not provide action EV.
Hosted execution and binary distribution remain a release gate until written
maintainer permission for that scope or documented AGPL compliance is retained
with the release.

## Release gate

One command produces a reproducible verdict on whether this build is releasable:

```bash
python -m poker_tracker.release_gate --mode fixture --report-dir data/release_reports
```

Modes: `fixture` scores retained prediction timelines without decoding video
(fast, and what CI runs); `full` decodes every corpus recording with the pinned
weights; `container` re-runs the acceptance path inside the pinned image and
compares verdicts. Exit `0` means every mandatory gate passed, `1` means a
product or accuracy gate failed, and `2` means the run could not be performed at
all — a setup problem, not a measurement.

Two report fields are load-bearing and easy to misread. `aggregate.measured`
is `false` when nothing was scored, and the counts beside it read `null` rather
than `0`, because zero errors over zero measurements is not a result.
`certification` is computed from `certification.executed`, the list of acts the
run actually performed, so a mode that quietly does less than its name says
cannot inflate its own claim. `release_certifying` is `true` only when the run
decoded video, loaded the pinned weights, reconstructed the recordings and
scored the result. `fixture` mode does none of the first three. `container` mode
runs the *fixture* gate inside the image, so it certifies that the image
reproduces the host's verdict, not that the product reconstructs video.

The committed corpus currently exits `2`: Phase 2 has produced no answer keys,
so no accuracy claim can be evaluated. CI asserts that it still fails closed, so
a green CI explicitly does not mean a passing release gate.

## Storage retention

```bash
python -m poker_tracker.maintenance.retention_cli              # dry run
python -m poker_tracker.maintenance.retention_cli --apply
```

Frames, timelines, job logs, exports and ROI previews expire on per-category
windows (`POKER_RETAIN_*_DAYS`). A file the product still expects is never
offered for deletion at any age, and the audit always prints its plan before
`--apply` acts. Source recordings need an explicit `--include-orphan-videos`,
because a recording is the one artifact nothing can rebuild.

Four things about "still expects" are worth knowing, because each of them was
once wrong in a way that deleted or offered to delete a file:

- **The reference list covers every artifact column, including regression
  fixtures.** A fixture attached to a resolved issue is frequently a frame or a
  recording under a managed directory, and it used to be missing from the list.
- **Some artifacts have no column at all, and the columns alone were the whole
  list.** A completed reconstruction job's CV timeline is addressed by job id,
  not by a stored path, so it counted as unreferenced from the moment it was
  written and its 90-day window deleted it — while nothing in the product deletes
  a `processing_jobs` row, so the completed job goes on expecting that timeline
  forever and every remaining validated-hand import for it stays blocked with no
  way to rebuild. The frames a timeline's states name have no column either until
  the operator reviews one, so the same question expired exactly the frames still
  waiting to be reviewed. Both are now resolved through the same naming rule the
  snapshot inventory and the health audit use.
- **Files are matched by identity, not by spelling.** On a case-insensitive
  filesystem a recording stored as `Session.MOV` and recorded as `session.mov`
  looked like an orphan. Matching is `(st_dev, st_ino)` where the file exists and
  a normalized case-folded key where it does not, and the textual fallback
  deliberately over-matches: keeping an orphan costs disk, deleting a live file
  costs the recording.
- **The audit is not an authorization.** A job that finishes between the audit and
  the sweep makes the database start pointing at a file the audit already
  classified as garbage. Every reference is re-confirmed immediately before each
  unlink, against a re-read of the database when it changed underneath.

A window must be positive. A zero or negative `POKER_RETAIN_*_DAYS` is refused
with an error naming the variable, because an unset variable expanding to empty
would otherwise mean "expire everything immediately". When you genuinely want
that, `--purge-now` says so explicitly; it still keeps referenced files and
still requires `--include-orphan-videos` for recordings.

Exit codes are the same whatever `--json` does to the output: `0` nothing to do,
would delete, or deleted; `1` a deletion failed; `2` the configuration is
invalid; `3` the sweep refused to act.

That contract holds for everything the CLI examines, and **not** for failures
that stop it examining anything. Opening the database and preparing the data
directory happen before the reporting path, so an unopenable database or an
unusable `--data-dir` leaves an unhandled traceback on stderr, exits `1`, and
under `--json` writes a zero-byte document. An unattended caller reading `1`
will look for a filesystem problem when what it has is a configuration one —
the case `2` is reserved for — and a `--json` consumer that treats an empty
document as "no findings" will read a sweep that examined nothing as a clean
one. Check for an empty document before trusting the code.

The age shown is the file's modification time, not how long it has been
unreferenced — nothing records when a row stopped pointing at a file. Backups are
outside retention's scope; they rotate on their own fixed count.

## Dependency inventory and licensing

```bash
python -m poker_tracker.maintenance.sbom --format notices > NOTICES.txt
python -m poker_tracker.maintenance.sbom --format cyclonedx > sbom.json
```

**`ultralytics` is AGPL-3.0 and the reconstruction pipeline depends on it.**
Local use is unaffected; publishing an image is what triggers the obligation, and
it applies to the base image, not only a solver-enabled one. TexasSolver carries
a separate blocker of the same kind. `--fail-on-review` exits nonzero while any
such component is present. This is not legal advice.

## Data health

Run the operator audit against the configured SQLite database and data directory:

```bash
python -m poker_tracker.maintenance --restore-backups
```

The command runs SQLite structural and foreign-key checks, validates the schema
version and core schema, verifies recorded videos and review images still exist,
compares stored video sizes, and checks every retained backup. Its artifact check
covers **every** path column the schema has, derived from the columns rather than
from a hand-kept list, so a column added later is covered by having been added.
`--restore-backups` copies each backup into an isolated temporary database for a
safe restore drill; it never replaces the live database. The audit does not
issue application-data writes, although SQLite may create or update its normal
WAL shared-memory sidecars while opening an active database. Add `--json` for
automation. The command exits nonzero when a check fails, while a fresh
installation with no backups reports a warning.

**This audit is the only surface that notices a missing recording.** Nothing on
the Import or Sessions card checks that the file behind a `videos` row still
exists, so a recording deleted from disk keeps rendering its duration,
resolution and size as current fact with **Run CV reconstruction** and **Extract
frames** enabled. Both buttons then fail truthfully and name the missing path, so
no wrong result is produced — but the browse surfaces will not tell you until you
either press one or run this check.

The same audit runs in-product from **Settings -> Storage & health**, behind an
explicit *Run health check* button so a page repaint never pays for it. Each
check reports in words rather than by colour alone. That tab also builds a
redacted **diagnostics bundle**: resolved configuration, dependency and model
identity, layout support, row counts and the health report, with every string
scrubbed through the same redaction before serialization — over the structure,
not over the encoded JSON, because `json.dumps` escapes the quotes in every
string field and an escaped key stops matching. It carries no hand history, note,
coaching text, video filename or environment value: environment variables are
reported by name and set/unset only.

## Recovery drill

The audit above verifies a *file*. This verifies a *recovery*: it restores a
chosen snapshot into a throwaway location, migrates it, and answers whether your
study history came back.

```bash
python -m poker_tracker.maintenance.recovery \
  --backup "$POKER_DATA_DIR/backups/<snapshot>.sqlite3" \
  --data-dir "$POKER_DATA_DIR" \
  --target "$(mktemp -d)"
```

It checks what recovery has to mean: the schema, foreign keys, issue evidence,
one completed hand read *through the application* rather than by raw select, and
which recordings, frames, timelines and solver outputs are missing. Missing
artifacts are reported as a **partial recovery** with each file named, not as a
warning.

**It does not compare row counts against the snapshot, so a `RECOVERED` verdict
does not mean the complete history came back.** The drill is written to compare
its restored session, hand, completed-hand and issue totals against the
inventory beside the snapshot, but `build_inventory` records artifacts only and
writes no counts, so there is nothing on the other side of that comparison. It
says so on every run — "artifact references are verified; totals are
self-reported" — and it still fails outright on a restore that holds no sessions
or no hands, which is the one loss it can prove. Anything short of total
emptiness is unproven. Read `RECOVERED` as "it restored, migrated, and reads
back through the application".

Exit `0` is `RECOVERED`. Exit `1` is `PARTIAL` (something is provably gone) or
`UNVERIFIED` (it restored cleanly but no inventory accompanied the snapshot, so
completeness is unproven). Exit `2` means the drill did not run and **nothing was
checked** — it refuses outright if its target overlaps `POKER_DATA_DIR`, contains
`POKER_DB_PATH`, or already holds a database, because a drill that restored an
old snapshot over your live file would destroy the history it was run to protect.

Because the counts comparison is inert, those first two are currently inverted
against the evidence: a snapshot carrying **no** inventory exits 1 as
`UNVERIFIED`, while one carrying an inventory exits 0 — keeping the completeness
reference produces the weaker verdict, not the stronger one.

Run it before you need it, on the machine you would actually recover onto. The
full procedure, including what to bring to a fresh machine and what each failing
check means, is in [docs/RUNBOOKS.md](docs/RUNBOOKS.md).

## Test

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
```

The suite covers persistence, migrations, hand completion and study readiness,
solver ranges/eligibility/CLI parsing, the debugging issue queue, correction
audit/export, Study UI mutation, coaching safety, math, video handling, CV
reconstruction, job recovery, backups, view models, authentication, and the
Streamlit product shell. It also carries dedicated readiness-bypass and
adversarial-round regressions that try to make an unproven hand look
study-ready.

Every skip has to name an external condition a reader can evaluate, and
`python -m poker_tracker.suite_quality skip-policy` fails the run when one does
not. `-ra` is on by default so the reasons print. On macOS six tests skip: four
because Darwin refuses `setrlimit(RLIMIT_AS)`, so the memory-cap paths are
exercised only on Linux; one because the newest schema version has no later
migration to lack; and
`tests/test_ocr_readers.py::test_without_chip_template_chip_would_join_run`, a
negative control documenting why the chip affix exists, which skips when the
synthetic chip glyph falls below the classifier confidence floor, because the
misread it demonstrates then does not occur. Any other skip is a real problem.

Two more tools live beside the suite:

```bash
python -m poker_tracker.suite_quality flake      # repeat runs, shuffled order
python -m poker_tracker.suite_quality coverage   # which core modules run at all
```

`flake` names every test whose result was not the same in every pass; a verdict
that cannot be reproduced is worth less than the name of the test that produced
it. `coverage` reports and does not gate — the useful output is the name of
important code nothing runs, not a percentage.

`python -m mypy` uses the narrow file list configured in `pyproject.toml`; it is
not whole-repository type checking.

## Container

```bash
docker build --platform linux/amd64 -t pokertrainer .
docker run --rm -p 8501:8501 \
  -e APP_PASSWORD='replace-me' \
  -v pokertrainer-data:/data \
  pokertrainer
```

The image runs as a non-root user and exposes a Streamlit healthcheck.

**No image has been built from this repository, on either architecture.** There
is no Docker daemon on the development machine, so the Dockerfile, the
healthcheck, the non-root runtime, the model provisioning and the entrypoint are
all unexecuted. Two blockers that would have stopped a build have been repaired
and are likewise unproven: the Dockerfile used to `COPY` model weights that
`.gitignore` excludes, so the build only ever succeeded on the machine that
trained them, and `eval7` has no aarch64 wheel against a runtime image with no
compiler. Build both `linux/amd64` and `linux/arm64` and run
`deploy/verify_container.sh` before accepting any deployment architecture, and
treat every container claim in this README as a description of the recipe rather
than a report on an image.

The image deliberately ships **without** the large CV weights; they are resolved
at runtime from `/data/models` on the persistent mount and installed by
`deploy/provision_models.py` against `deploy/model_manifest.json`, each file
renamed into place only after its SHA-256 matches. Full detail is in
[docs/CONTAINER.md](docs/CONTAINER.md).

## Deployment

- **Local-first:** run the application directly with `streamlit run app.py` or
  use the Docker command above.
- **Provider-neutral container:** the image keeps runtime configuration in
  environment variables and persistent state under `/data`.
- **Optional OCI reference:** [deploy/oci/README.md](deploy/oci/README.md)
  documents one possible Oracle Cloud setup. It is not an active deployment
  target and should only be used when deployment is explicitly requested.

## Repository guidance

Project guidance has one source per purpose:

- [AGENTS.md](AGENTS.md) contains durable engineering instructions for Codex and
  other coding agents that support the convention.
- [CLAUDE.md](CLAUDE.md) contains the same complete instructions for Claude
  Code. It must remain byte-for-byte identical to `AGENTS.md`.
- [PLAN.md](PLAN.md) contains current status, priorities, release gates, and the
  definition of done.
- [docs/RUNBOOKS.md](docs/RUNBOOKS.md) contains the operator procedures: install,
  diagnostics, release gate, corpus vault, migration, backup and isolated
  restore, the fresh-machine recovery drill, failed-job recovery, storage audit,
  containers, upgrade and rollback, licensing before distribution, and the
  issue-to-regression debugging loop.
- [docs/CONTAINER.md](docs/CONTAINER.md) covers what a container build needs that
  a `git clone` does not contain, and how to verify an image.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) covers the measurement harness and
  the rules it follows, chief among them that an unmeasured figure is reported as
  `null` with a reason and never as zero.
- [cv_lab/notes/README.md](cv_lab/notes/README.md) indexes the chronological CV
  research and adversarial record. Those findings explain decisions, later ones
  supersede earlier ones, and none of them is a claim about what is true today.
- [deploy/oci/README.md](deploy/oci/README.md) is the Oracle deployment
  runbook.

There are currently no repository-local skills, custom subagents, or Codex
project settings. Add a skill only for a repeatable workflow that needs its own
instructions or scripts; ordinary project conventions belong in `AGENTS.md`.
