# PokerTrainer

PokerTrainer is a local-first, post-session poker study and review workspace. It organizes completed sessions and hands, runs offline video reconstruction, provides poker math and coaching tools, and keeps source confidence visible throughout review.

It never provides real-time poker assistance, live table capture, poker-client overlays, or current-hand recommendations.

## Product workspace

The Streamlit application is organized around seven workflows:

- **Overview** — portfolio metrics, recent sessions, and processing jobs.
- **Sessions** — session summaries and manual hand entry.
- **Hands** — searchable cross-session hand library.
- **Study** — hand replay, recorded math, coaching results, notes, and review state.
- **Insights** — evidence-backed review coverage and tagged study themes.
- **Import** — completed-session video upload and offline CV reconstruction.
- **Settings** — ROI calibration, data transfer, math tools, and coaching configuration.

CV and coaching output is labeled by source and confidence. The application does not claim solver-perfect or GTO-scored analysis.

## Local setup

Requirements:

- Python 3.11+
- SQLite
- FFmpeg for video workflows

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
4. Produce a retained timeline and session export.
5. Create a consistent SQLite backup.
6. Import reconstructed hands as needs-correction drafts with confidence and provenance.

Only one local processing job can run at a time. On restart, dead or stale workers are marked failed instead of remaining stuck. SQLite uses WAL mode so the UI can continue reading while the worker writes.

File layout:

```text
data/
  backups/       rotating pre-import SQLite backups
  cv_timelines/  retained reconstruction timelines
  exports/       generated session exports
  frames/        diagnostic extracted frames
  job_logs/      detached worker logs
  roi_previews/  ROI crop previews
  videos/        uploaded completed-session videos
```

Videos remain files; SQLite stores structured records and paths.

## Coaching providers

Mock coaching works offline. Optional providers read keys from environment variables:

```bash
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
```

Every generated prompt passes the post-session safety check and can be inspected before use.

## Test

```bash
python -m pytest -q
```

The suite covers persistence, migrations, math, coaching safety, video handling, CV reconstruction, job recovery, backups, view models, authentication, and the Streamlit product shell.

## Container

```bash
docker build --platform linux/amd64 -t pokertrainer .
docker run --rm -p 8501:8501 \
  -e APP_PASSWORD='replace-me' \
  -v pokertrainer-data:/data \
  pokertrainer
```

The image runs as a non-root user and exposes a Streamlit healthcheck. Build both `linux/amd64` and `linux/arm64` before accepting a deployment architecture.

## Deployment

- **Free-first:** follow [deploy/oci/README.md](deploy/oci/README.md) for one Oracle Cloud Always Free A1 VM, persistent block storage, Docker Compose, and Caddy TLS.
- **Fallback:** `fly.toml` deploys the identical image to an always-on Fly Machine with a persistent volume.

Oracle deployment is accepted only after ARM64 dependencies, representative CV runtime, memory, restart, and restore gates pass. Fly is the paid operational fallback when those gates fail.
