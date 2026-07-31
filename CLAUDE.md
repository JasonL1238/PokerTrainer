# Repository agent instructions

## Instruction-file synchronization
- `AGENTS.md` and `CLAUDE.md` must remain byte-for-byte identical.
- Whenever either instruction file changes, apply the same change to the other file in the same work.
- Keep the complete instructions in both files; do not replace either file with an import or symlink.
- After changing either file, verify that `AGENTS.md` and `CLAUDE.md` are identical.

## Project
This is a local-first post-session poker study and review platform.

## Non-negotiable constraints
- Never build real-time poker assistance.
- Never build live table capture.
- Never build a poker-client overlay.
- Never provide current-hand recommendations.
- Analysis is for completed hands/sessions only.

## Engineering style
- Build incrementally.
- Prefer small, testable modules.
- Keep database, CV, OCR, analytics, equity, and coaching modules separate.
- Store videos as files, not in SQL.
- Store structured data in SQLite first.
- Add tests for core behavior.
- Ignore development labor costs when evaluating or prioritizing implementation choices; optimize for product quality, maintainability, and runtime/hosting cost.
- Never identify yourself, an AI assistant, or an automated agent as a commit author, co-author, contributor, signer, or attribution in Git metadata or commit messages.
- Do not change database schemas without explaining migration impact.
- Preserve existing API response formats unless explicitly authorized.
- Search for existing implementations before creating new abstractions.
- Keep functions focused and avoid unnecessary dependencies.

## Runtime and deployment posture
- The application is currently intended to run and be tested locally; there is no active hosted deployment.
- Do not deploy, provision cloud resources, or add a hosting-provider dependency unless the user explicitly requests it.
- Keep the project continuously deployment-ready so it can be containerized and hosted without a feature rewrite.
- Treat the Dockerfile, container healthcheck, non-root runtime, and `linux/amd64` and `linux/arm64` compatibility as first-class supported paths.
- Keep deployment provider-neutral. Runtime settings belong in environment variables, and secrets must never be committed.
- Keep SQLite, videos, models, timelines, and backups on explicit persistent mounts or external storage rather than the container filesystem.
- When a change affects dependencies, startup, storage, authentication, networking, or CV/model runtime, verify both the local Streamlit path and the Docker path when feasible.

## Documentation
- `README.md` is the canonical product overview, setup guide, and operator entrypoint.
- `PLAN.md` is the canonical current status, release plan, and definition of done.
- `cv_lab/notes/` is a chronological research record. Later findings supersede earlier experiments.
- Keep historical experiments out of the current roadmap.
- Update documentation when behavior, commands, deployment, or release gates change.

## Commands
- Install: pip install -r requirements.txt
- Test: pytest
- App: streamlit run app.py

## Cursor Cloud specific instructions
- Dependencies live in a virtualenv at `.venv` (the system Python 3.12 is externally managed, so a venv is used). Run tools through it: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, `.venv/bin/mypy`, and `.venv/bin/streamlit run app.py`.
- The env installs the full CV/video stack (`requirements-cv.txt` plus `ultralytics==8.3.203 --no-deps`) and system `ffmpeg`, not just base `requirements.txt`. The CV/video tests (`tests/test_cv_jobs.py`, `tests/test_video_ingest.py`, etc.) hard-fail with `ModuleNotFoundError: av` without this stack, so keep it installed.
- The app serves on `:8501` with health at `/_stcore/health`. The SQLite DB and `data/` subdirs are created lazily on the first browser load of the UI (not by the health check). Override locations with `POKER_DB_PATH` / `POKER_DATA_DIR`; no password is needed locally.
- Pre-existing repo state (not an environment problem): `main` currently fails CI at the `ruff check .` lint step (10 errors) before tests run, and a handful of tests fail on the current loose-pinned dependency set (resolves to streamlit 1.60, ruff 0.16, torch 2.8). The multi-hand paste "Save hands" flow raises a `StreamlitAPIException` under streamlit 1.60, though the hand still persists. Do not treat these as setup regressions.
- End-to-end CV video reconstruction additionally needs model weights under `cv_lab/models/` (`region_spine_v1.pt`, `card_cls_v1.pt`, `card_templates.npz`, `pot_digits.npz`); these are gitignored and absent from a fresh clone (only `ocr_templates.npz` is committed), so the Import video pipeline cannot run fully without supplying them.
