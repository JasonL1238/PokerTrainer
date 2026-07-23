# AGENTS.md

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
- Never add yourself, an AI assistant, or an automated agent as a commit co-author.

## Commands
- Install: pip install -r requirements.txt
- Test: pytest
- App: streamlit run app.py
