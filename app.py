from __future__ import annotations

import json
from datetime import date

import streamlit as st
from pydantic import ValidationError

from poker_tracker.analytics import compute_session_stats
from poker_tracker.coaching_prompts import build_hand_review_prompt, build_session_review_prompt
from poker_tracker.db import DEFAULT_DB_PATH, PokerDatabase
from poker_tracker.equity import get_equity_calculator
from poker_tracker.ev import bluff_ev, call_ev
from poker_tracker.frame_extraction import (
    delete_extracted_frames,
    extract_frames_for_video,
    select_representative_frames,
)
from poker_tracker.hand_history import format_hand_history
from poker_tracker.import_export import export_hand, export_session, import_session
from poker_tracker.image_utils import image_dimensions, save_roi_crop_preview
from poker_tracker.llm_providers import (
    LLMProviderError,
    build_coaching_response,
    get_provider_from_env,
    provider_config_from_env,
)
from poker_tracker.models import Action, HAND_TAGS, Hand, HandPlayer, ROIProfile, ROIRegion, Session, VideoRecord
from poker_tracker.pot_odds import break_even_bluff_frequency, format_percentage, required_equity_to_call
from poker_tracker.ranges import RANGE_LABELS, estimate_villain_range_label
from poker_tracker.review import generate_mock_review
from poker_tracker.roi import ROI_TYPES, validate_roi_bounds
from poker_tracker.roi_profiles import (
    create_starter_clubwpt_profile,
    duplicate_roi_profile,
    export_roi_profile,
    generate_roi_crop_previews,
    import_roi_profile,
)
from poker_tracker.safety import validate_post_session_prompt
from poker_tracker.seed_data import create_sample_data
from poker_tracker.video_metadata import extract_video_metadata
from poker_tracker.video_storage import ensure_data_directories, save_video_file, validate_video_extension


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

    dashboard_tab, entry_tab, hands_tab, math_tab, coach_tab, video_tab, roi_tab, transfer_tab = st.tabs(
        [
            "Dashboard",
            "Enter Hand",
            "Review Hands",
            "Math Review",
            "Coach Review",
            "Video Processing",
            "ROI Calibration",
            "Import / Export",
        ]
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
    with video_tab:
        show_video_processing(db, selected_session)
    with roi_tab:
        show_roi_calibration(db)
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
        compute_equity = st.checkbox("Compute Hero equity vs range", value=True)
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

    if compute_equity and hand.hero_cards:
        try:
            equity_result = get_equity_calculator().calculate_equity(
                hand.hero_cards,
                hand.board_cards,
                range_label,
            )
            math_facts["equity"] = equity_result.equity or "unavailable"
            if equity_result.equity is None:
                st.write(f"Equity vs {range_label}: unavailable ({equity_result.method})")
            else:
                st.write(
                    f"Equity vs {range_label}: {format_percentage(equity_result.equity)} "
                    f"({equity_result.method}, confidence {format_percentage(equity_result.confidence)})"
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


def show_video_processing(db: PokerDatabase, session: Session) -> None:
    st.subheader("Completed Session Video Processing")
    st.caption("Upload completed session videos only. This does not capture or analyze live tables.")
    ensure_data_directories()

    uploaded = st.file_uploader("Upload completed session video", type=["mp4", "mov", "mkv", "avi"])
    link_to_session = st.checkbox("Link upload to selected session", value=True)
    video_notes = st.text_area("Video notes", height=80)
    if uploaded is not None and st.button("Save uploaded video"):
        try:
            validate_video_extension(uploaded.name)
            uploaded.seek(0)
            stored_path = save_video_file(uploaded, uploaded.name)
            metadata = extract_video_metadata(stored_path)
            saved_video = db.create_video(
                VideoRecord(
                    session_id=session.id if link_to_session else None,
                    original_filename=uploaded.name,
                    stored_path=str(stored_path),
                    file_size_bytes=stored_path.stat().st_size,
                    duration_seconds=metadata.duration_seconds,
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                    frame_count=metadata.frame_count,
                    notes=video_notes.strip(),
                )
            )
            if metadata.error:
                st.warning(metadata.error)
            st.success(f"Saved video #{saved_video.id}.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("#### Stored Videos")
    all_videos = db.fetch_videos()
    if not all_videos:
        st.info("No videos stored yet.")
        return

    labels = {
        video.id: f"Video #{video.id}: {video.original_filename} ({_format_bytes(video.file_size_bytes)})"
        for video in all_videos
        if video.id is not None
    }
    selected_video_id = st.selectbox(
        "Select stored video",
        options=list(labels.keys()),
        format_func=lambda video_id: labels[video_id],
    )
    video = db.fetch_video(selected_video_id)
    if video is None:
        st.error("Selected video no longer exists.")
        return
    show_video_metadata(video)
    show_video_jobs_and_frames(db, video)


def show_video_metadata(video: VideoRecord) -> None:
    st.write(
        {
            "Original filename": video.original_filename,
            "Stored path": video.stored_path,
            "Size": _format_bytes(video.file_size_bytes),
            "Duration": _format_optional_seconds(video.duration_seconds),
            "FPS": video.fps,
            "Resolution": _format_resolution(video.width, video.height),
            "Frame count": video.frame_count,
            "Uploaded": video.uploaded_at.isoformat(),
            "Notes": video.notes,
        }
    )


def show_video_jobs_and_frames(db: PokerDatabase, video: VideoRecord) -> None:
    if video.id is None:
        return

    st.markdown("#### Frame Extraction")
    settings_left, settings_right = st.columns(2)
    with settings_left:
        frames_per_second = st.number_input(
            "Frames per second",
            min_value=0.1,
            value=2.0,
            step=0.5,
            key=f"extract_fps_{video.id}",
        )
        start_time = st.number_input(
            "Start time seconds",
            min_value=0.0,
            value=0.0,
            step=0.5,
            key=f"extract_start_{video.id}",
        )
    with settings_right:
        max_frames = st.number_input(
            "Max frames",
            min_value=1,
            value=20,
            step=1,
            key=f"extract_max_{video.id}",
        )
        end_time = st.number_input(
            "End time seconds (0 = no limit)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            key=f"extract_end_{video.id}",
        )

    if st.button("Extract frames", key=f"extract_button_{video.id}"):
        try:
            summary = extract_frames_for_video(
                db,
                video.id,
                frames_per_second=float(frames_per_second),
                max_frames=int(max_frames),
                start_time_seconds=float(start_time),
                end_time_seconds=float(end_time) if end_time > 0 else None,
            )
            st.success(f"Extracted {summary.frames_extracted} frames to {summary.output_dir}.")
            if summary.errors:
                st.warning("; ".join(summary.errors))
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(f"Frame extraction failed: {exc}")

    jobs = db.fetch_jobs_by_video(video.id)
    if jobs:
        st.markdown("##### Jobs")
        st.dataframe(
            [
                {
                    "ID": job.id,
                    "Type": job.job_type,
                    "Status": job.status,
                    "Progress": job.progress_percent,
                    "Message": job.message,
                    "Error": job.error_message,
                    "Created": job.created_at.isoformat(),
                }
                for job in jobs
            ],
            hide_index=True,
            use_container_width=True,
        )

    frames = db.fetch_frames_by_video(video.id)
    st.write({"Extracted frame count": len(frames)})
    confirm_delete = st.checkbox(
        "Confirm delete extracted frames for this video",
        key=f"delete_frames_confirm_{video.id}",
    )
    if st.button("Delete extracted frames", key=f"delete_frames_{video.id}", disabled=not confirm_delete):
        deleted = delete_extracted_frames(db, video.id)
        st.success(f"Deleted {deleted} extracted frame records/files.")
        st.rerun()
    show_frame_preview(frames)


def show_frame_preview(frames) -> None:
    if not frames:
        st.caption("No frames extracted yet.")
        return
    st.markdown("##### Frame Preview")
    representative = select_representative_frames(frames, limit=12)
    columns = st.columns(4)
    for index, frame in enumerate(representative):
        with columns[index % 4]:
            st.image(frame.image_path, caption=f"{frame.timestamp_seconds:.2f}s")

    frame_labels = {
        frame.id: f"{frame.timestamp_seconds:.2f}s - {frame.image_path}"
        for frame in frames
        if frame.id is not None
    }
    selected_frame_id = st.selectbox(
        "Select full frame",
        options=list(frame_labels.keys()),
        format_func=lambda frame_id: frame_labels[frame_id],
    )
    selected = next(frame for frame in frames if frame.id == selected_frame_id)
    st.write({"Timestamp": selected.timestamp_seconds, "Path": selected.image_path})
    st.image(selected.image_path)


def show_roi_calibration(db: PokerDatabase) -> None:
    st.subheader("ROI Calibration")
    st.caption(
        "Manual calibration for completed-session extracted frames only. "
        "No card detection, OCR, live capture, or action reconstruction is performed."
    )
    ensure_data_directories()
    # TODO: Add interactive rectangle drawing later if a stable dependency is worth it.

    videos = db.fetch_videos()
    if not videos:
        st.info("Upload a completed session video and extract frames before calibrating ROIs.")
        return

    video_labels = {
        video.id: f"Video #{video.id}: {video.original_filename} ({_format_resolution(video.width, video.height)})"
        for video in videos
        if video.id is not None
    }
    selected_video_id = st.selectbox(
        "Select video for calibration",
        options=list(video_labels.keys()),
        format_func=lambda video_id: video_labels[video_id],
        key="roi_video_select",
    )
    video = db.fetch_video(selected_video_id)
    if video is None or video.id is None:
        st.error("Selected video no longer exists.")
        return

    frames = db.fetch_frames_by_video(video.id)
    if not frames:
        st.info("Extract frames for this video before ROI calibration.")
        return

    frame_labels = {
        frame.id: f"{frame.timestamp_seconds:.2f}s - frame {frame.frame_index}"
        for frame in frames
        if frame.id is not None
    }
    selected_frame_id = st.selectbox(
        "Select calibration frame",
        options=list(frame_labels.keys()),
        format_func=lambda frame_id: frame_labels[frame_id],
        key="roi_frame_select",
    )
    frame = next(item for item in frames if item.id == selected_frame_id)
    st.image(frame.image_path, caption=f"Calibration frame at {frame.timestamp_seconds:.2f}s")
    try:
        frame_width, frame_height = image_dimensions(frame.image_path)
        st.write({"Frame width": frame_width, "Frame height": frame_height, "Frame path": frame.image_path})
    except ValueError as exc:
        st.error(str(exc))
        return

    show_roi_profile_tools(db, video, frame_width, frame_height)
    profiles = db.fetch_roi_profiles()
    if not profiles:
        st.info("Create a profile or starter preset to begin adding regions.")
        return

    profile_labels = {
        profile.id: f"{'* ' if profile.is_active else ''}Profile #{profile.id}: {profile.name}"
        for profile in profiles
        if profile.id is not None
    }
    selected_profile_id = st.selectbox(
        "Select ROI profile",
        options=list(profile_labels.keys()),
        format_func=lambda profile_id: profile_labels[profile_id],
        key="roi_profile_select",
    )
    profile = db.fetch_roi_profile(selected_profile_id)
    if profile is None or profile.id is None:
        st.error("Selected profile no longer exists.")
        return

    st.write(
        {
            "Platform": profile.platform,
            "Layout": profile.table_layout,
            "Profile dimensions": _format_resolution(profile.video_width, profile.video_height),
            "Active": profile.is_active,
        }
    )
    if st.button("Mark selected profile active", key=f"roi_active_{profile.id}"):
        db.mark_roi_profile_active(profile.id)
        st.rerun()
    if st.button("Duplicate selected profile", key=f"roi_duplicate_{profile.id}"):
        duplicate_roi_profile(db, profile.id)
        st.success("Duplicated ROI profile.")
        st.rerun()

    show_roi_import_export(db, profile)
    show_add_roi_region_form(db, profile, frame_width, frame_height)
    show_roi_regions(db, profile, frame, frame_width, frame_height)


def show_roi_profile_tools(
    db: PokerDatabase,
    video: VideoRecord,
    frame_width: int,
    frame_height: int,
) -> None:
    st.markdown("#### Profiles")
    left, right = st.columns(2)
    with left.form("create_roi_profile"):
        name = st.text_input("New profile name", value="ClubWPT Gold custom")
        description = st.text_area("Description", height=70)
        platform = st.text_input("Platform", value="ClubWPT Gold")
        table_layout = st.text_input("Table layout", value="9-max")
        use_frame_dims = st.checkbox("Use selected frame dimensions", value=True)
        submitted = st.form_submit_button("Create empty profile")
    if submitted:
        profile = ROIProfile(
            name=name.strip(),
            description=description.strip(),
            platform=platform.strip() or "ClubWPT Gold",
            table_layout=table_layout.strip(),
            video_width=frame_width if use_frame_dims else video.width,
            video_height=frame_height if use_frame_dims else video.height,
        )
        db.create_roi_profile(profile)
        st.success("ROI profile created.")
        st.rerun()

    with right:
        st.markdown("##### Starter preset")
        st.caption("Creates editable placeholder regions for common ClubWPT Gold table elements.")
        seats = st.number_input("Seats", min_value=6, max_value=9, value=9, step=1, key="roi_preset_seats")
        if st.button("Create ClubWPT Gold starter preset"):
            create_starter_clubwpt_profile(
                db,
                video_width=frame_width,
                video_height=frame_height,
                max_seats=int(seats),
            )
            st.success("Starter ROI profile created.")
            st.rerun()


def show_roi_import_export(db: PokerDatabase, profile: ROIProfile) -> None:
    if profile.id is None:
        return
    st.markdown("#### Import / Export ROI Profile")
    st.download_button(
        "Export selected ROI profile JSON",
        data=json.dumps(export_roi_profile(db, profile.id), indent=2),
        file_name=f"roi_profile_{profile.id}.json",
        mime="application/json",
        key=f"roi_export_{profile.id}",
    )
    uploaded = st.file_uploader("Import ROI profile JSON", type=["json"], key="roi_import_upload")
    if uploaded is not None and st.button("Import ROI profile"):
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))
            imported = import_roi_profile(db, payload)
            st.success(f"Imported ROI profile: {imported.name}")
            st.rerun()
        except (ValueError, ValidationError, KeyError, json.JSONDecodeError) as exc:
            st.error(f"Could not import ROI profile: {exc}")


def show_add_roi_region_form(
    db: PokerDatabase,
    profile: ROIProfile,
    frame_width: int,
    frame_height: int,
) -> None:
    if profile.id is None:
        return
    st.markdown("#### Add ROI Region")
    with st.form("add_roi_region", clear_on_submit=True):
        left, right = st.columns(2)
        with left:
            roi_key = st.text_input("ROI key", placeholder="hero_card_1")
            roi_type = st.selectbox("ROI type", ROI_TYPES, index=ROI_TYPES.index("unknown"))
            label = st.text_input("Label", placeholder="Hero card 1")
            notes = st.text_area("Notes", height=70)
        with right:
            x = st.number_input("X", min_value=0, value=0, step=1)
            y = st.number_input("Y", min_value=0, value=0, step=1)
            width = st.number_input("Width", min_value=1, value=40, step=1)
            height = st.number_input("Height", min_value=1, value=40, step=1)
            seat_index = st.number_input("Seat index (0 = none)", min_value=0, max_value=10, value=0, step=1)
            card_index = st.number_input("Card index (0 = none)", min_value=0, max_value=5, value=0, step=1)
        submitted = st.form_submit_button("Add region")
    if submitted:
        try:
            region = ROIRegion(
                profile_id=profile.id,
                roi_key=roi_key.strip(),
                roi_type=roi_type,
                label=label.strip(),
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                seat_index=int(seat_index) or None,
                card_index=int(card_index) or None,
                notes=notes.strip(),
            )
            validate_roi_bounds(region, image_width=frame_width, image_height=frame_height)
            db.create_roi_region(region)
            st.success("ROI region added.")
            st.rerun()
        except (ValueError, ValidationError) as exc:
            st.error(f"Could not add ROI region: {exc}")


def show_roi_regions(
    db: PokerDatabase,
    profile: ROIProfile,
    frame,
    frame_width: int,
    frame_height: int,
) -> None:
    if profile.id is None:
        return
    regions = db.fetch_roi_regions_by_profile(profile.id)
    st.markdown("#### Regions")
    if not regions:
        st.caption("No ROI regions saved for this profile yet.")
        return

    st.dataframe(
        [
            {
                "Key": region.roi_key,
                "Type": region.roi_type,
                "Label": region.label,
                "X": region.x,
                "Y": region.y,
                "W": region.width,
                "H": region.height,
                "Seat": region.seat_index,
                "Card": region.card_index,
            }
            for region in regions
        ],
        hide_index=True,
        use_container_width=True,
    )

    if st.button("Generate all crop previews", key=f"roi_generate_all_{profile.id}_{frame.id}"):
        try:
            results = generate_roi_crop_previews(db, profile.id, frame.id)
            st.success(f"Generated {len(results)} crop previews.")
        except ValueError as exc:
            st.error(str(exc))

    for region in regions:
        if region.id is None:
            continue
        with st.expander(f"{region.roi_key} [{region.roi_type}]"):
            show_edit_roi_region_form(db, region, frame, frame_width, frame_height)


def show_edit_roi_region_form(
    db: PokerDatabase,
    region: ROIRegion,
    frame,
    frame_width: int,
    frame_height: int,
) -> None:
    with st.form(f"edit_roi_region_{region.id}"):
        left, right = st.columns(2)
        with left:
            roi_key = st.text_input("ROI key", value=region.roi_key)
            roi_type = st.selectbox(
                "ROI type",
                ROI_TYPES,
                index=ROI_TYPES.index(region.roi_type) if region.roi_type in ROI_TYPES else ROI_TYPES.index("unknown"),
            )
            label = st.text_input("Label", value=region.label)
            notes = st.text_area("Notes", value=region.notes, height=70)
        with right:
            x = st.number_input("X", min_value=0, value=region.x, step=1)
            y = st.number_input("Y", min_value=0, value=region.y, step=1)
            width = st.number_input("Width", min_value=1, value=region.width, step=1)
            height = st.number_input("Height", min_value=1, value=region.height, step=1)
            seat_value = region.seat_index or 0
            card_value = region.card_index or 0
            seat_index = st.number_input("Seat index (0 = none)", min_value=0, max_value=10, value=seat_value, step=1)
            card_index = st.number_input("Card index (0 = none)", min_value=0, max_value=5, value=card_value, step=1)
        update, preview, delete = st.columns(3)
        submitted_update = update.form_submit_button("Update")
        submitted_preview = preview.form_submit_button("Preview crop")
        submitted_delete = delete.form_submit_button("Delete")

    updated = ROIRegion(
        id=region.id,
        profile_id=region.profile_id,
        roi_key=roi_key.strip(),
        roi_type=roi_type,
        label=label.strip(),
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
        seat_index=int(seat_index) or None,
        card_index=int(card_index) or None,
        notes=notes.strip(),
        created_at=region.created_at,
    )
    if submitted_update:
        try:
            validate_roi_bounds(updated, image_width=frame_width, image_height=frame_height)
            db.update_roi_region(updated)
            st.success("ROI region updated.")
            st.rerun()
        except (ValueError, ValidationError) as exc:
            st.error(f"Could not update ROI region: {exc}")
    if submitted_preview:
        try:
            result = save_roi_crop_preview(frame, updated)
            st.write(
                {
                    "Crop": result.crop_path,
                    "Size": f"{result.crop_width}x{result.crop_height}",
                    "Source timestamp": result.source_timestamp_seconds,
                }
            )
            st.image(result.crop_path)
        except ValueError as exc:
            st.error(f"Could not preview ROI crop: {exc}")
    confirm_delete = st.checkbox(
        f"Confirm delete {region.roi_key}",
        key=f"confirm_delete_roi_{region.id}",
    )
    if submitted_delete:
        if not confirm_delete:
            st.warning("Check the delete confirmation box first.")
            return
        db.delete_roi_region(region.id)
        st.success("ROI region deleted.")
        st.rerun()


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _format_optional_seconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}s"


def _format_resolution(width: int | None, height: int | None) -> str:
    if width is None or height is None:
        return "unknown"
    return f"{width}x{height}"


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
