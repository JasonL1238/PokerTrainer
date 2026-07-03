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
- Poker math helpers for pot odds, bluff thresholds, and simple EV estimates
- Low-confidence placeholder equity abstraction for future real calculators
- Structured post-session coaching prompt generation for future LLM integration
- Offline mock LLM provider for structured post-session coach reviews
- Optional OpenAI-compatible cloud provider configured by environment variables
- Sample session data
- JSON export for one hand or a full session
- JSON import for a full session

## Math Review

Run the app and open the `Math Review` tab:

- Select a saved hand.
- Enter pot size before call, call amount, bet size, fold frequency, and villain range label.
- Review required calling equity, break-even bluff frequency, approximate call EV, and approximate bluff EV.
- Optionally generate a math-aware mock review.
- Expand `Structured future-LLM prompt` to inspect the prompt that a future provider abstraction can use.

Equity is currently a deterministic placeholder with low confidence. It is not solver output and should not be treated as exact poker truth.

## Coach Review

Run the app and open the `Coach Review` tab:

- Choose `Mock` for fully offline deterministic provider reviews.
- Choose `Cloud` only after configuring environment variables.
- Select hand-level or session-level review.
- Inspect the exact post-session-only prompt before generation.
- Generate and save the provider response to SQLite.

Cloud provider configuration:

```bash
export POKER_TRACKER_LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key
export POKER_TRACKER_LLM_MODEL=gpt-4o-mini
streamlit run app.py
```

If the cloud provider is selected without an API key, the app falls back to the mock provider. API keys are read from the environment and are not stored in SQLite.

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
