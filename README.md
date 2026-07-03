# Poker Trainer

Local-first, post-session poker study and review tool.

This Milestone 1 app is intentionally limited to manual entry of completed hands. It does not provide real-time assistance, live table capture, poker-client overlays, hotkeys, or current-hand recommendations.

## Requirements

- Python 3.11+
- SQLite

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

By default, the app stores data in `poker_tracker.db` in the project root.

## Run Tests

```bash
pytest
```

## Current Scope

The current app includes:

- Manual session creation
- Manual hand entry
- Street-by-street action entry
- SQLite persistence
- Saved-hand viewing
- Review statuses and hand tags
- Clean hand-history text formatting
- Basic/manual session stats
- Mock hand reviews with theory, exploit, EV/math, lesson, and next-question sections
- Sample session data
- JSON export for one hand or a full session
- JSON import for a full session

## Sample Data

Run the app and click `Load sample data` in the sidebar.

You can also load sample data from Python:

```bash
python -c "from poker_tracker.db import PokerDatabase; from poker_tracker.seed_data import create_sample_data; db=PokerDatabase(); db.init_db(); create_sample_data(db); db.close()"
```

## Import / Export

In the app, open the `Import / Export` tab:

- `Export full session JSON` downloads the selected session.
- `Import session JSON` imports a previously exported session.

Individual hands can be exported from the `Review Hands` tab.

Future milestones can add CV, OCR, equity calculators, solver integration, video processing, RAG, and real LLM review generation as separate modules.
