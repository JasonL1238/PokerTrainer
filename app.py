from __future__ import annotations

import json
from datetime import date

import streamlit as st
from pydantic import ValidationError

from poker_tracker.analytics import compute_session_stats
from poker_tracker.coaching_prompts import build_hand_review_prompt, build_session_review_prompt
from poker_tracker.db import DEFAULT_DB_PATH, PokerDatabase
from poker_tracker.equity import PlaceholderEquityCalculator
from poker_tracker.ev import bluff_ev, call_ev
from poker_tracker.hand_history import format_hand_history
from poker_tracker.import_export import export_hand, export_session, import_session
from poker_tracker.llm_providers import (
    LLMProviderError,
    build_coaching_response,
    get_provider_from_env,
    provider_config_from_env,
)
from poker_tracker.models import Action, HAND_TAGS, Hand, HandPlayer, Session
from poker_tracker.pot_odds import break_even_bluff_frequency, format_percentage, required_equity_to_call
from poker_tracker.ranges import RANGE_LABELS, estimate_villain_range_label
from poker_tracker.review import generate_mock_review
from poker_tracker.safety import validate_post_session_prompt
from poker_tracker.seed_data import create_sample_data


STREETS = ["preflop", "flop", "turn", "river", "showdown"]
ACTION_TYPES = ["fold", "check", "call", "bet", "raise", "all-in", "post_blind", "show", "win"]
POSITIONS = ["", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
REVIEW_STATUSES = ["unreviewed", "reviewed", "needs_correction"]
COACHING_MODES = ["Theory + Exploit", "Theory Only", "Exploit Only", "Leak Finder"]


@st.cache_resource
def get_database() -> PokerDatabase:
    db = PokerDatabase(DEFAULT_DB_PATH)
    db.init_db()
    return db


def main() -> None:
    st.set_page_config(page_title="Poker Trainer", layout="wide")
    st.title("Manual Post-Session Review")
    st.caption("Completed-hand study only. No real-time assistance, capture, overlays, or hotkeys.")

    db = get_database()

    with st.sidebar:
        st.header("Session")
        create_session_form(db)
        if st.button("Load sample data"):
            create_sample_data(db)
            st.success("Sample session loaded.")
            st.rerun()
        selected_session = select_session(db)

    if selected_session is None:
        st.info("Create or load a session to start reviewing completed hands.")
        return

    dashboard_tab, entry_tab, hands_tab, math_tab, coach_tab, transfer_tab = st.tabs(
        ["Dashboard", "Enter Hand", "Review Hands", "Math Review", "Coach Review", "Import / Export"]
    )
    with dashboard_tab:
        show_session_dashboard(db, selected_session)
    with entry_tab:
        create_hand_form(db, selected_session.id)
    with hands_tab:
        show_saved_hands(db, selected_session)
    with math_tab:
        show_math_review(db, selected_session)
    with coach_tab:
        show_coach_review(db, selected_session)
    with transfer_tab:
        show_import_export(db, selected_session)


def create_session_form(db: PokerDatabase) -> None:
    with st.form("create_session", clear_on_submit=True):
        name = st.text_input("Session name", placeholder="Friday review")
        date_played = st.date_input("Date played", value=date.today())
        platform = st.text_input("Platform", value="Manual")
        stakes = st.text_input("Stakes", placeholder="1/2 NL")
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Create session")

    if submitted:
        if not name.strip():
            st.error("Session name is required.")
            return
        db.create_session(
            Session(
                name=name.strip(),
                date_played=date_played,
                platform=platform.strip() or "Manual",
                stakes=stakes.strip(),
                notes=notes.strip(),
            )
        )
        st.success("Session created.")
        st.rerun()


def select_session(db: PokerDatabase) -> Session | None:
    sessions = db.fetch_sessions()
    if not sessions:
        return None

    labels = {
        session.id: (
            f"{session.date_played.isoformat()} - {session.name} "
            f"({session.platform}, {session.stakes or 'no stakes'})"
        )
        for session in sessions
        if session.id is not None
    }
    selected_id = st.selectbox(
        "Select session",
        options=list(labels.keys()),
        format_func=lambda session_id: labels[session_id],
    )
    return next(session for session in sessions if session.id == selected_id)


def show_session_dashboard(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    stats = compute_session_stats(db, session.id)
    st.subheader(f"{session.name} - basic/manual stats")
    first, second, third, fourth = st.columns(4)
    first.metric("Hands", stats.hand_count)
    second.metric("Hero result", f"{stats.total_hero_bb:g} BB")
    third.metric("Average / hand", f"{stats.average_hero_bb:g} BB")
    fourth.metric("Reviewed", stats.hands_by_review_status.get("reviewed", 0))

    st.write(
        {
            "Unreviewed": stats.hands_by_review_status.get("unreviewed", 0),
            "Needs correction": stats.hands_by_review_status.get("needs_correction", 0),
            "Aggressive actions": stats.aggression_count,
            "Passive actions": stats.passive_count,
        }
    )
    st.markdown("##### Biggest Winning Hands")
    st.dataframe(_hand_summary_rows(stats.biggest_winning_hands), hide_index=True)
    st.markdown("##### Biggest Losing Hands")
    st.dataframe(_hand_summary_rows(stats.biggest_losing_hands), hide_index=True)
    st.markdown("##### Tag Counts")
    st.write(stats.hands_by_tag or {})
    st.markdown("##### Action Counts")
    st.write(stats.action_counts_by_type or {})


def create_hand_form(db: PokerDatabase, session_id: int | None) -> None:
    if session_id is None:
        st.error("Select a saved session before adding hands.")
        return

    with st.form("create_hand", clear_on_submit=True):
        st.markdown("#### Hand Setup")
        setup_left, setup_right = st.columns(2)
        with setup_left:
            hand_number = st.number_input("Hand number", min_value=1, step=1)
            game_type = st.text_input("Game type", value="No-limit Hold'em")
            blinds_antes = st.text_input("Blinds / antes", placeholder="1/2 NL, 0.25 ante")
            table_size = st.number_input("Table size", min_value=2, max_value=10, value=6, step=1)
            effective_stack = st.number_input("Effective stack", min_value=0.0, step=1.0)
            source_type = st.selectbox("Source", ["manual", "cv_import", "corrected_cv"])
        with setup_right:
            hero_position = st.selectbox("Hero position", POSITIONS, index=6)
            hero_cards = st.text_input("Hero cards", placeholder="Ah Qs")
            board_cards = st.text_input("Board cards", placeholder="Qd 7s 2c 9h 3s")
            pot_size = st.number_input("Final pot size", min_value=0.0, step=1.0)
            hero_bb_won = st.number_input("Final result in BB", step=0.5)
            result = st.text_input("Result text", placeholder="Hero wins")
            review_status = st.selectbox("Review status", REVIEW_STATUSES)

        tags = st.multiselect("Tags", sorted(HAND_TAGS))
        notes = st.text_area("Hand notes / pro-style hand history", height=100)

        st.markdown("#### Players In The Hand")
        player_rows = collect_player_inputs()

        st.markdown("#### Action Line")
        action_rows = collect_action_inputs()
        submitted = st.form_submit_button("Save hand")

    if submitted:
        try:
            saved_hand = db.create_hand(
                Hand(
                    session_id=session_id,
                    hand_number=int(hand_number),
                    game_type=game_type.strip(),
                    blinds_antes=blinds_antes.strip(),
                    table_size=int(table_size),
                    effective_stack=float(effective_stack),
                    hero_position=hero_position,
                    hero_cards=hero_cards,
                    board_cards=board_cards,
                    pot_size=float(pot_size),
                    result=result.strip(),
                    hero_bb_won=float(hero_bb_won),
                    review_status=review_status,
                    source_type=source_type,
                    tags=tags,
                    notes=notes.strip(),
                )
            )
            save_player_rows(db, saved_hand.id, player_rows)
            save_action_rows(db, saved_hand.id, action_rows)
        except (ValidationError, ValueError) as exc:
            st.error(f"Could not save hand: {exc}")
            return
        st.success("Hand saved.")
        st.rerun()


def collect_player_inputs() -> list[dict]:
    edited_rows = st.data_editor(
        [
            {"Player": "Hero", "Position": "BTN", "Starting stack": 0.0, "Hero?": True, "Notes": ""},
            {"Player": "", "Position": "", "Starting stack": 0.0, "Hero?": False, "Notes": ""},
            {"Player": "", "Position": "", "Starting stack": 0.0, "Hero?": False, "Notes": ""},
        ],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
            "Starting stack": st.column_config.NumberColumn("Starting stack", min_value=0),
            "Hero?": st.column_config.CheckboxColumn("Hero?"),
        },
        key="players_editor",
    )
    return [
        {
            "player_name": str(row.get("Player") or "").strip(),
            "position": str(row.get("Position") or "").strip(),
            "starting_stack": _optional_float(row.get("Starting stack")),
            "is_hero": bool(row.get("Hero?")),
            "notes": str(row.get("Notes") or "").strip(),
        }
        for row in edited_rows
    ]


def collect_action_inputs() -> list[dict]:
    rows: list[dict] = []
    for street in STREETS:
        with st.expander(street.title(), expanded=street == "preflop"):
            edited_rows = st.data_editor(
                _default_action_rows(street),
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player"),
                    "Position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
                    "Action": st.column_config.SelectboxColumn("Action", options=ACTION_TYPES),
                    "Amount": st.column_config.NumberColumn("Amount", min_value=0),
                    "Pot before": st.column_config.NumberColumn("Pot before", min_value=0),
                    "Stack before": st.column_config.NumberColumn("Stack before", min_value=0),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                key=f"{street}_actions_editor",
            )
            for row in _non_empty_action_rows(edited_rows):
                row["Street"] = street
                rows.append(row)
    return [
        {
            "street": str(row.get("Street") or "preflop"),
            "player_name": str(row.get("Player") or "").strip(),
            "position": str(row.get("Position") or "").strip(),
            "action_type": str(row.get("Action") or "fold"),
            "amount": _optional_float(row.get("Amount")),
            "pot_before": _optional_float(row.get("Pot before")),
            "stack_before": _optional_float(row.get("Stack before")),
            "notes": str(row.get("Notes") or "").strip(),
        }
        for row in rows
    ]


def save_player_rows(db: PokerDatabase, hand_id: int | None, player_rows: list[dict]) -> None:
    if hand_id is None:
        raise ValueError("Hand must be saved before players can be saved.")
    for row in player_rows:
        if not row["player_name"]:
            continue
        db.create_hand_player(HandPlayer(hand_id=hand_id, **row))


def save_action_rows(db: PokerDatabase, hand_id: int | None, action_rows: list[dict]) -> None:
    if hand_id is None:
        raise ValueError("Hand must be saved before actions can be saved.")
    for row in action_rows:
        if not row["player_name"]:
            continue
        db.create_action(Action(hand_id=hand_id, **row))


def show_saved_hands(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return

    hands = db.fetch_hands_by_session(session.id)
    if not hands:
        st.info("No hands saved for this session yet.")
        return

    for hand in hands:
        if hand.id is None:
            continue
        players = db.fetch_players_by_hand(hand.id)
        actions = db.fetch_actions_by_hand(hand.id)
        with st.expander(
            f"Hand #{hand.hand_number}: {hand.hero_cards or 'unknown cards'} "
            f"{hand.hero_bb_won or 0:g} BB [{hand.review_status}]"
        ):
            st.code(format_hand_history(session, hand, actions, players), language="text")

            status = st.selectbox(
                "Review status",
                REVIEW_STATUSES,
                index=REVIEW_STATUSES.index(hand.review_status),
                key=f"status_{hand.id}",
            )
            if st.button("Update status", key=f"status_button_{hand.id}"):
                db.update_hand_status(hand.id, status)
                st.rerun()

            show_action_editor(db, actions)

            if st.button("Generate mock review", key=f"review_{hand.id}"):
                review = generate_mock_review(hand, actions, players)
                db.create_hand_review(review)
                db.update_hand_status(hand.id, "reviewed")
                st.success("Mock review saved.")
                st.rerun()

            for review in db.fetch_reviews_by_hand(hand.id):
                st.markdown("##### Hand Summary")
                st.write(review.hand_summary)
                st.markdown("##### Theory Coach")
                st.write(review.theory_coach)
                st.markdown("##### Exploit Coach")
                st.write(review.exploit_coach)
                st.markdown("##### EV / Math Notes")
                st.write(review.ev_math_notes)
                st.markdown("##### Study Lesson")
                st.write(review.study_lesson)
                st.markdown("##### Next Review Question")
                st.write(review.next_review_question)

            st.download_button(
                "Export this hand JSON",
                data=json.dumps(export_hand(db, hand.id), indent=2),
                file_name=f"hand_{hand.hand_number}.json",
                mime="application/json",
                key=f"export_hand_{hand.id}",
            )

            confirm_delete = st.checkbox(
                "Confirm delete this hand and all related rows", key=f"confirm_delete_{hand.id}"
            )
            if st.button("Delete hand", key=f"delete_hand_{hand.id}", disabled=not confirm_delete):
                db.delete_hand(hand.id)
                st.rerun()


def show_action_editor(db: PokerDatabase, actions: list[Action]) -> None:
    st.markdown("##### Edit / Delete Actions")
    if not actions:
        st.caption("No actions saved.")
        return

    for action in actions:
        if action.id is None:
            continue
        with st.form(f"edit_action_{action.id}"):
            cols = st.columns([0.9, 0.8, 1, 1, 0.8, 0.8, 1.2])
            street = cols[0].selectbox("Street", STREETS, index=STREETS.index(action.street))
            position = cols[1].selectbox(
                "Position",
                POSITIONS,
                index=POSITIONS.index(action.position) if action.position in POSITIONS else 0,
            )
            player_name = cols[2].text_input("Player", value=action.player_name)
            action_type = cols[3].selectbox(
                "Action", ACTION_TYPES, index=ACTION_TYPES.index(action.action_type)
            )
            amount = cols[4].number_input("Amount", min_value=0.0, value=action.amount or 0.0)
            action_index = cols[5].number_input(
                "Order", min_value=1, value=action.action_index or 1, step=1
            )
            notes = cols[6].text_input("Notes", value=action.notes)
            update, delete = st.columns(2)
            submitted_update = update.form_submit_button("Update action")
            submitted_delete = delete.form_submit_button("Delete action")

        if submitted_update:
            db.update_action(
                Action(
                    id=action.id,
                    hand_id=action.hand_id,
                    street=street,
                    action_index=int(action_index),
                    player_name=player_name,
                    position=position,
                    action_type=action_type,
                    amount=amount,
                    notes=notes,
                )
            )
            st.rerun()
        if submitted_delete:
            db.delete_action(action.id)
            st.rerun()


def show_import_export(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    st.download_button(
        "Export full session JSON",
        data=json.dumps(export_session(db, session.id), indent=2),
        file_name=f"session_{session.id}.json",
        mime="application/json",
    )
    uploaded = st.file_uploader("Import session JSON", type=["json"])
    if uploaded is not None and st.button("Import uploaded session"):
        payload = json.loads(uploaded.getvalue().decode("utf-8"))
        imported = import_session(db, payload)
        st.success(f"Imported session: {imported.name}")
        st.rerun()


def show_math_review(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    hands = db.fetch_hands_by_session(session.id)
    if not hands:
        st.info("Save or load hands before using Math Review.")
        return

    labels = {
        hand.id: f"Hand #{hand.hand_number}: {hand.hero_cards or 'unknown'} ({hand.hero_bb_won or 0:g} BB)"
        for hand in hands
        if hand.id is not None
    }
    selected_hand_id = st.selectbox(
        "Select hand",
        options=list(labels.keys()),
        format_func=lambda hand_id: labels[hand_id],
    )
    hand = next(item for item in hands if item.id == selected_hand_id)
    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)

    st.code(format_hand_history(session, hand, actions, players), language="text")

    default_range = estimate_villain_range_label(hand.tags, hand.notes)
    range_options = sorted(RANGE_LABELS)
    range_label = st.selectbox(
        "Villain range label",
        options=range_options,
        index=range_options.index(default_range),
    )

    first, second = st.columns(2)
    with first:
        pot_before_call = st.number_input("Pot before call", min_value=0.0, step=1.0)
        call_amount = st.number_input("Call amount", min_value=0.0, step=1.0)
        use_placeholder_equity = st.checkbox("Use placeholder equity estimate", value=True)
    with second:
        pot_size = st.number_input("Pot size for bet/bluff", min_value=0.0, step=1.0)
        bet_size = st.number_input("Bet size", min_value=0.0, step=1.0)
        fold_frequency_pct = st.slider("Estimated fold frequency", 0, 100, 40)

    math_facts: dict[str, float | str] = {}
    equity_result = None
    errors: list[str] = []

    if call_amount > 0 and pot_before_call > 0:
        try:
            required = required_equity_to_call(call_amount, pot_before_call)
            math_facts["required_equity_to_call"] = required
            st.write(f"Required equity to call: {format_percentage(required)}")
        except ValueError as exc:
            errors.append(str(exc))

    if bet_size > 0 and pot_size > 0:
        try:
            bluff_frequency = break_even_bluff_frequency(bet_size, pot_size)
            math_facts["break_even_bluff_frequency"] = bluff_frequency
            st.write(f"Break-even bluff frequency: {format_percentage(bluff_frequency)}")
        except ValueError as exc:
            errors.append(str(exc))

    if use_placeholder_equity and hand.hero_cards:
        try:
            equity_result = PlaceholderEquityCalculator().calculate_equity(
                hand.hero_cards,
                hand.board_cards,
                range_label,
            )
            math_facts["placeholder_equity"] = equity_result.equity or "unavailable"
            st.write(
                f"Placeholder equity: {format_percentage(equity_result.equity or 0)} "
                f"(confidence {format_percentage(equity_result.confidence)})"
            )
            st.caption(equity_result.notes)
        except ValueError as exc:
            errors.append(str(exc))

    if equity_result is not None and equity_result.equity is not None and call_amount > 0 and pot_before_call > 0:
        ev_value = call_ev(equity_result.equity, pot_before_call, call_amount)
        math_facts["call_ev"] = round(ev_value, 3)
        st.write(f"Approximate call EV: {ev_value:.2f}")

    fold_frequency = fold_frequency_pct / 100
    if bet_size > 0 and pot_size > 0:
        bluff_value = bluff_ev(fold_frequency, pot_size, bet_size)
        math_facts["bluff_ev"] = round(bluff_value, 3)
        st.write(f"Approximate bluff EV at {fold_frequency_pct}% folds: {bluff_value:.2f}")

    for error in errors:
        st.error(error)

    prompt = build_hand_review_prompt(
        session,
        hand,
        actions,
        players,
        pot_odds_facts=math_facts,
        equity_result=equity_result,
        villain_range_label=range_label,
    )

    if st.button("Generate math-aware mock review"):
        review = generate_mock_review(
            hand,
            actions,
            players,
            math_facts=math_facts,
            equity_result=equity_result,
            villain_range_label=range_label,
        )
        db.create_hand_review(review)
        db.update_hand_status(hand.id, "reviewed")
        st.success("Math-aware mock review saved.")
        st.rerun()

    with st.expander("Structured future-LLM prompt"):
        st.code(prompt, language="text")


def show_coach_review(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    st.subheader("Post-Session Coach Review")
    st.caption("Provider reviews are for completed hands/sessions only.")

    hands = db.fetch_hands_by_session(session.id)
    if not hands:
        st.info("Save or load hands before generating coach reviews.")
        return

    config = provider_config_from_env()
    provider_choice = st.selectbox("Provider", ["Mock", "Cloud"])
    if provider_choice == "Cloud" and not config.has_api_key:
        st.warning("Cloud provider selected but no OPENAI_API_KEY is configured. Falling back to Mock.")
    provider = get_provider_from_env("cloud" if provider_choice == "Cloud" else "mock")
    st.write({"Active provider": provider.provider_name, "Model": provider.model_name})

    coaching_mode = st.selectbox("Coaching mode", COACHING_MODES)
    review_scope = st.radio("Review scope", ["Hand", "Session"], horizontal=True)

    if review_scope == "Hand":
        show_hand_coach_review(db, session, hands, provider, coaching_mode)
    else:
        show_session_coach_review(db, session, hands, provider, coaching_mode)


def show_hand_coach_review(
    db: PokerDatabase,
    session: Session,
    hands: list[Hand],
    provider,
    coaching_mode: str,
) -> None:
    labels = {
        hand.id: f"Hand #{hand.hand_number}: {hand.hero_cards or 'unknown'} ({hand.hero_bb_won or 0:g} BB)"
        for hand in hands
        if hand.id is not None
    }
    selected_hand_id = st.selectbox(
        "Select hand",
        options=list(labels.keys()),
        format_func=lambda hand_id: labels[hand_id],
        key="coach_hand_select",
    )
    hand = next(item for item in hands if item.id == selected_hand_id)
    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    history = format_hand_history(session, hand, actions, players)
    st.code(history, language="text")

    range_label = st.selectbox(
        "Villain range label",
        options=sorted(RANGE_LABELS),
        index=sorted(RANGE_LABELS).index(estimate_villain_range_label(hand.tags, hand.notes)),
        key="coach_range_label",
    )
    pot_before_call = st.number_input("Optional pot before call", min_value=0.0, step=1.0)
    call_amount = st.number_input("Optional call amount", min_value=0.0, step=1.0)
    math_facts = _optional_prompt_math_facts(pot_before_call, call_amount)

    prompt = build_hand_review_prompt(
        session,
        hand,
        actions,
        players,
        pot_odds_facts=math_facts,
        villain_range_label=range_label,
        coaching_mode=coaching_mode,
    )
    show_prompt_safety(prompt)
    with st.expander("Exact prompt sent to provider"):
        st.code(prompt, language="text")

    if st.button("Generate and save post-session hand review"):
        try:
            raw_response = provider.generate_hand_review(prompt)
            saved = db.create_coaching_response(
                build_coaching_response(
                    provider=provider,
                    prompt=prompt,
                    raw_response=raw_response,
                    review_type="hand",
                    hand_id=hand.id,
                    session_id=session.id,
                )
            )
            db.update_hand_status(hand.id, "reviewed")
            st.success(f"Saved provider review #{saved.id}.")
            st.rerun()
        except (LLMProviderError, ValueError) as exc:
            st.error(f"Could not generate review: {exc}")

    show_saved_provider_reviews(db.fetch_coaching_reviews_by_hand(hand.id))


def show_session_coach_review(
    db: PokerDatabase,
    session: Session,
    hands: list[Hand],
    provider,
    coaching_mode: str,
) -> None:
    stats = compute_session_stats(db, session.id)
    selected_hands = select_session_review_hands(hands)
    histories = [
        format_hand_history(
            session,
            hand,
            db.fetch_actions_by_hand(hand.id),
            db.fetch_players_by_hand(hand.id),
        )
        for hand in selected_hands
        if hand.id is not None
    ]
    st.write(
        {
            "Selected hands": [hand.hand_number for hand in selected_hands],
            "Tag counts": stats.hands_by_tag,
            "Review statuses": stats.hands_by_review_status,
        }
    )
    prompt = build_session_review_prompt(
        session,
        stats,
        histories,
        coaching_mode=coaching_mode,
    )
    show_prompt_safety(prompt)
    with st.expander("Exact prompt sent to provider"):
        st.code(prompt, language="text")

    if st.button("Generate and save post-session session review"):
        try:
            raw_response = provider.generate_session_review(prompt)
            saved = db.create_coaching_response(
                build_coaching_response(
                    provider=provider,
                    prompt=prompt,
                    raw_response=raw_response,
                    review_type="session",
                    session_id=session.id,
                )
            )
            st.success(f"Saved provider session review #{saved.id}.")
            st.rerun()
        except (LLMProviderError, ValueError) as exc:
            st.error(f"Could not generate session review: {exc}")

    show_saved_provider_reviews(db.fetch_coaching_reviews_by_session(session.id))


def select_session_review_hands(hands: list[Hand]) -> list[Hand]:
    """Pick relevant completed hands for a session-level provider prompt."""
    selected: list[Hand] = []
    losing = sorted(
        [hand for hand in hands if hand.hero_bb_won is not None and hand.hero_bb_won < 0],
        key=lambda hand: hand.hero_bb_won or 0,
    )[:3]
    winning = sorted(
        [hand for hand in hands if hand.hero_bb_won is not None and hand.hero_bb_won > 0],
        key=lambda hand: hand.hero_bb_won or 0,
        reverse=True,
    )[:2]
    tagged = [
        hand
        for hand in hands
        if set(hand.tags) & {"MISSED_VALUE", "RIVER_DECISION", "MULTIWAY", "BIG_POT"}
    ]
    unreviewed = [hand for hand in hands if hand.review_status == "unreviewed"][:3]
    for hand in [*losing, *winning, *tagged, *unreviewed]:
        if hand.id is not None and hand.id not in {item.id for item in selected}:
            selected.append(hand)
    return selected[:8]


def show_saved_provider_reviews(reviews) -> None:
    st.markdown("##### Saved Provider Reviews")
    if not reviews:
        st.caption("No provider reviews saved yet.")
        return
    for review in reviews:
        with st.expander(
            f"{review.created_at.isoformat()} - {review.provider_name}/{review.model_name}"
        ):
            st.write({"Review type": review.review_type, "Safety mode": review.safety_mode})
            st.write(review.parsed_sections or {})
            st.code(review.raw_response, language="text")


def show_prompt_safety(prompt: str) -> None:
    result = validate_post_session_prompt(prompt)
    if result.is_safe:
        st.success("Prompt safety check passed: post-session review only.")
    else:
        st.error("Prompt safety check failed: " + "; ".join(result.errors))


def _optional_prompt_math_facts(
    pot_before_call: float,
    call_amount: float,
) -> dict[str, float | str]:
    if pot_before_call <= 0 or call_amount <= 0:
        return {}
    return {"required_equity_to_call": required_equity_to_call(call_amount, pot_before_call)}


def _hand_summary_rows(hands: list[Hand]) -> list[dict]:
    return [
        {
            "Hand": hand.hand_number,
            "Hero": hand.hero_cards,
            "Result BB": hand.hero_bb_won,
            "Tags": ", ".join(hand.tags),
            "Status": hand.review_status,
        }
        for hand in hands
    ]


def _default_action_rows(street: str) -> list[dict]:
    row_count = 4 if street == "preflop" else 1
    return [
        {
            "Player": "",
            "Position": "",
            "Action": "fold",
            "Amount": 0.0,
            "Pot before": 0.0,
            "Stack before": 0.0,
            "Notes": "",
        }
        for _ in range(row_count)
    ]


def _non_empty_action_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("Player") or "").strip()]


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


if __name__ == "__main__":
    main()


# TODO: Future CV/OCR output should import corrected hands through this same manual schema.
# TODO: Keep video files outside SQL when video processing is introduced later.
