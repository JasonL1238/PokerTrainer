# Performance and resource measurement

`python -m poker_tracker.perf` measures what a session costs on a specific
machine: startup, import, UI render, upload, model initialization,
reconstruction throughput, solver runtime, peak memory, disk growth, temporary
files and log growth. It writes one JSON report that a later run can be compared
against.

Code lives in `poker_tracker/perf/`. It reuses the release gate's resource and
environment accounting (`poker_tracker/release_gate/resources.py`,
`environment.py`) so a performance report and a release report describe a host
the same way.

## The three rules

**A measurement that was not taken is `null`, never `0`.** Every metric in the
report carries a `status` of `measured` or `not_taken`, and a `not_taken` entry
carries the reason. This is the same discipline the release gate applies to its
error counts (`runner._aggregate_metrics` withholds counts when nothing was
scored) and it exists for the same reason: a row of zeros from a run that
measured nothing is indistinguishable from a perfect one.

**Every number carries its host and its conditions.** The report states the
machine once — system, architecture, CPU count, memory, Python, dependency
versions, git commit and dirty state, load average at start — and each metric
states the conditions it was taken under: what it includes, what it excludes,
which weights, which recording, how many bytes.

**A comparison never invents a verdict.** `compare` reports `missing_baseline`
or `missing_current` when either side has no value, and `incomparable_host` when
the two reports came from different machines. Neither is ever counted as a
regression or an improvement.

## Running it

```bash
# everything this machine can measure
python -m poker_tracker.perf run

# a subset, into a chosen report path
python -m poker_tracker.perf run --groups imports,ui,upload --out data/perf_reports/perf_report.json

# compare a fresh run against a recorded baseline, failing on a regression
python -m poker_tracker.perf run --baseline data/perf_reports/baseline.json --fail-on-regression

# judge two existing reports
python -m poker_tracker.perf compare --baseline a.json --report b.json
```

Probe groups: `imports`, `startup`, `ui`, `upload`, `models`, `reconstruction`,
`solver`. Resource accounting always runs; it brackets the whole run.

Exit codes: `0` success (including a run where measurements could not be taken —
that is not a product failure), `1` usage error, `2` a regression against the
baseline with `--fail-on-regression`, `4` the representative-session requirement
not satisfied with `--require-session-check`.

### It never touches operator state

Probe children run with `POKER_DB_PATH`, `POKER_DATA_DIR` and `TMPDIR`
redirected into the run's workspace, so no probe can write to the real database,
video vault or backups. The solver probe opens the configured database with
SQLite `mode=ro`, so a stray write fails rather than migrating a live library.
The workspace is a temp directory by default and is kept after the run because
it holds the probe logs; `--clean-workspace` discards it.

## What is measured

| Metric | Unit | Better | Meaning |
| --- | --- | --- | --- |
| `import.core_seconds` | seconds | lower | Cold import of the persistence layer in a fresh interpreter. |
| `import.streamlit_seconds` | seconds | lower | Cold import of Streamlit in a fresh interpreter. |
| `import.cv_stack_seconds` | seconds | lower | Cold import of torch, ultralytics, av and cv2 in a fresh interpreter. |
| `import.cv_stack_peak_rss_bytes` | bytes | lower | Peak resident set of an interpreter that only imported the CV stack. |
| `startup.app_health_seconds` | seconds | lower | Streamlit launch to the first healthy `/_stcore/health` response. |
| `ui.first_render_seconds` | seconds | lower | First full script run of `app.py` under `AppTest`. |
| `ui.slowest_page_render_seconds` | seconds | lower | Slowest single navigation rerun across every primary page. |
| `ui.peak_rss_bytes` | bytes | lower | Peak resident set of an interpreter rendering every page once. |
| `upload.store_seconds` | seconds | lower | Storing an uploaded recording through the vault's atomic writer. |
| `upload.store_megabytes_per_second` | MB/s | higher | Sustained write rate of the vault store path, fsync included. |
| `model_init.detector_seconds` | seconds | lower | Region detector from construction to its first inference. |
| `model_init.classifier_seconds` | seconds | lower | Card classifier from construction to its first inference. |
| `model_init.peak_rss_bytes` | bytes | lower | Peak resident set of an interpreter that loaded both models. |
| `reconstruction.wall_seconds` | seconds | lower | End-to-end two-model reconstruction of one recording. |
| `reconstruction.frames_processed` | count | higher | Frames the pipeline sampled during that reconstruction. |
| `reconstruction.frames_per_second` | fps | higher | Sampled-frame throughput. |
| `reconstruction.peak_rss_bytes` | bytes | lower | Peak resident set of the reconstruction process. |
| `reconstruction.timeline_bytes` | bytes | lower | Size of the timeline artifact it wrote. |
| `solver.recorded_runs` | count | higher | Completed solver runs with a recorded runtime in the database. |
| `solver.recorded_runtime_median_seconds` | seconds | lower | Median recorded solver runtime. |
| `solver.recorded_runtime_max_seconds` | seconds | lower | Slowest recorded solver run. |
| `memory.harness_peak_rss_bytes` | bytes | lower | Peak resident set of the harness process and its waited children. |
| `disk.workspace_growth_bytes` | bytes | lower | Signed change in workspace bytes across the run. |
| `disk.data_root_bytes` | bytes | lower | Size of the configured operator data root (read only). |
| `disk.free_bytes` | bytes | higher | Free space on the workspace volume after the run. |
| `tempfiles.leaked_count` | count | lower | Temporary entries left behind in the run's own `TMPDIR`. |
| `tempfiles.leaked_bytes` | bytes | lower | Bytes held by those leftovers. |
| `logs.growth_bytes` | bytes | lower | Signed change in log bytes written under the workspace. |

Things worth knowing about individual numbers, all of which the report also
states in each metric's `conditions`:

- `startup.app_health_seconds` measures the same endpoint the container
  healthcheck uses. It answers as soon as the server binds, **before** the first
  script run — it is not "the app is rendered and ready".
- `ui.*` renders against an empty workspace database, so it describes the shell,
  not a page over a full hand library.
- `upload.*` measures the local store path — chunked copy, fsync, atomic rename.
  It excludes the browser transfer, which the harness cannot observe.
- `model_init.*` includes one warm-up inference, because `CardClassifier` loads
  its weights lazily and a construction-only figure for it would read as zero.
- `reconstruction.wall_seconds` and `frames_per_second` are end to end, model
  loading included, because that is what a job costs. Subtract `model_init.*` to
  separate the two.
- `solver.*` summarizes runs this installation has already recorded rather than
  launching a synthetic solve, so it describes real trees an operator actually
  solved. Zero recorded runs is reported as `recorded_runs: 0` with the
  percentiles withheld.
- Disk and log figures are signed before/after deltas, not snapshots, and the
  temp-file figure counts only what the run's own redirected `TMPDIR` retained.

## Baselines

`poker_tracker/perf/baselines/local_reference.json` is shipped **empty**: every
metric is `never_measured` with a `null` value, and the representative-session
check is `never_run`. No number in it was ever taken. A fabricated baseline
would make the first real run look like a pass or a regression depending on
which way the fiction leaned, and neither verdict would mean anything.

To record a real one, run the harness on the machine in question and keep the
report:

```bash
python -m poker_tracker.perf run --out data/perf_reports/baseline.json
```

Comparison tolerance defaults to 25% (`--tolerance`); a metric that moves less
than that is `unchanged`. Direction of improvement is per metric — throughput
and free space regress by falling, everything else by rising.

Regenerate the empty baseline after adding a metric, or the comparison will
silently have nothing to say about it:

```bash
python -m poker_tracker.perf new-baseline --out poker_tracker/perf/baselines/local_reference.json
```

`tests/test_perf_harness.py` fails if the shipped baseline drifts from the
declared metric set, or if any value in it stops being `null`.

## The supported local reference machine

**No reference machine has been designated.** Nothing in `README.md`, `PLAN.md`
or `docs/RUNBOOKS.md` names one, so no measurement anywhere in this repository
certifies a release gate today. Reports say so rather than implying otherwise:
`host.is_designated_reference` is `false` and the representative-session check
reports `certifies_release_gate: false` even when it finishes well inside the
limit.

To designate one, set both variables on that machine and record its report:

```bash
export POKERTRAINER_PERF_HOST_LABEL="oci-a1-4c-24g"       # this machine's identity
export POKERTRAINER_PERF_REFERENCE_HOST="oci-a1-4c-24g"   # the designated reference
```

A run certifies only when the two agree. Document the designated label wherever
the release gates are recorded so the label is not just an environment variable
someone exported once.

## The one-hour representative-session requirement

PLAN.md Phase 13 requires that "a representative full session completes within
one hour on the supported local reference machine". The harness makes that check
executable:

```bash
export POKER_VALIDATION_ROOT=/path/to/recording/vault
python -m poker_tracker.perf run --groups upload,reconstruction --require-session-check
```

It reconstructs the first release-scored manifest case whose recording is
present under the vault (or `--video PATH` for an explicit one), sums the upload
and reconstruction time, and compares it to 3600 seconds. The check covers the
machine work: storing one recording and reconstructing it. It excludes operator
review time, solver runs and any second recording, and says so in the report.

The reconstruction probe's own timeout is deliberately above the limit (7200s by
default): a run killed at exactly 3600s could not be told apart from one that
took 61 minutes, and distinguishing those is the point.

**This check has never been run.** It needs a corpus recording under
`POKER_VALIDATION_ROOT`, both model weights installed under `cv_lab/models/`,
and a designated reference machine. Until all three exist, the report says
`never_run` and names what is missing.

## What this harness does not measure

Stated plainly so nobody reads coverage into a gap:

- **Container image size, container startup-to-healthy time, and in-container
  peak RSS.** These need a Docker daemon and a buildable image. No image has
  ever been built from this repository; the daemon is unreachable on the
  development machine, and the build has open defects of its own.
- **`linux/amd64` and `linux/arm64` build or runtime figures.** Same blocker.
- **Concurrent-load behaviour.** Every probe runs alone; the harness measures no
  contention between a heavy CV job and the UI.
- **Solver runtime on a fresh tree.** Only runtimes already recorded in the
  database are summarized.

When a container build exists, the natural extension is a `container` probe
group that records `docker image inspect --format {{.Size}}`, container
startup-to-healthy seconds and in-container peak RSS through the same
`Measurement` contract — withheld with a reason on a machine with no daemon,
exactly as the corpus-dependent probes are withheld today.
