from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

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
)
from poker_tracker.models import Action, HAND_TAGS, Hand, HandPlayer, ROIProfile, ROIRegion, Session, VideoRecord
from poker_tracker.pot_odds import (
    break_even_bluff_frequency,
    format_percentage,
    minimum_defense_frequency,
    required_equity_to_call,
)
from poker_tracker.icm import icm_equities, icm_risk_premium
from poker_tracker.preflop_ranges import available_ranges
from poker_tracker.ranges import RANGE_LABELS, estimate_villain_range_label
from poker_tracker.study_math import (
    REALIZATION_FACTOR_GUIDE,
    bluff_to_value_ratio,
    optimal_bluff_fraction,
    outs_to_equity_exact,
    outs_to_equity_rule,
    realized_equity,
)
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
from poker_tracker.ui_theme import brand_header, inject_theme
from poker_tracker.seed_data import create_sample_data
from poker_tracker.video_metadata import extract_video_metadata
from poker_tracker.video_storage import ensure_data_directories, save_video_file, validate_video_extension


STREETS = ["preflop", "flop", "turn", "river", "showdown"]
ACTION_TYPES = ["fold", "check", "call", "bet", "raise", "all-in", "post_blind", "show", "win"]
POSITIONS = ["", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
REVIEW_STATUSES = ["unreviewed", "reviewed", "needs_correction"]
COACHING_MODES = ["Theory + Exploit", "Theory Only", "Exploit Only", "Leak Finder"]
MAX_IMPORT_BYTES = 10 * 1024 * 1024  # sane ceiling for JSON imports; videos have their own path


@st.cache_resource
def get_database() -> PokerDatabase:
    db = PokerDatabase(DEFAULT_DB_PATH)
    db.init_db()
    return db


@st.cache_data(show_spinner=False)
def _cached_equity(hero_cards: str, board_cards: str, range_label: str):
    """Cache equity results: exact enumeration/MC is CPU-heavy and pure."""
    return get_equity_calculator().calculate_equity(hero_cards, board_cards, range_label)


@st.cache_data(show_spinner=False)
def _cached_multiway_equity(hero_cards: str, board_cards: str, villain_ranges: tuple[str, ...]):
    calculator = get_equity_calculator()
    if not hasattr(calculator, "calculate_equity_multiway"):
        return None  # placeholder engine (eval7 missing) has no multiway support
    return calculator.calculate_equity_multiway(hero_cards, board_cards, list(villain_ranges))


def flash(message: str, icon: str = "✅") -> None:
    """Queue a confirmation that survives the st.rerun() after a state change."""
    st.session_state["_flash"] = (message, icon)


def show_flash() -> None:
    queued = st.session_state.pop("_flash", None)
    if queued:
        message, icon = queued
        st.toast(message, icon=icon)


def main() -> None:
    st.set_page_config(page_title="PokerTrainer", page_icon="♠", layout="wide")
    inject_theme()
    brand_header()
    show_flash()

    db = get_database()

    with st.sidebar:
        st.header("Session")
        create_session_form(db)
        if st.button("Load sample data"):
            if any(session.name == "Sample post-session review" for session in db.fetch_sessions()):
                st.warning("Sample data is already loaded.")
            else:
                create_sample_data(db)
                flash("Sample session loaded.")
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
        flash("Session created.")
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
    st.subheader(session.name)
    st.caption(f"{session.date_played} · {session.stakes or 'stakes not set'} · {session.platform or 'platform not set'}")

    first, second, third, fourth = st.columns(4)
    first.metric("Hands", stats.hand_count)
    second.metric("Hero result", f"{stats.total_hero_bb:g} BB")
    winrate_help = (
        f"{stats.average_hero_bb:+.2f} BB/hand over {stats.hands_with_result} hands with recorded results"
    )
    if stats.bb_per_100_ci is not None:
        ci_low, ci_high = stats.bb_per_100_ci
        winrate_help += (
            f". 95% CI: {ci_low:+.0f} to {ci_high:+.0f} bb/100 — small samples say little about a true winrate."
        )
    third.metric("Winrate", f"{stats.bb_per_100:+.0f} bb/100", help=winrate_help)
    fourth.metric("Reviewed", stats.hands_by_review_status.get("reviewed", 0))

    fifth, sixth, seventh, eighth = st.columns(4)
    fifth.metric("Unreviewed", stats.hands_by_review_status.get("unreviewed", 0))
    sixth.metric("Needs correction", stats.hands_by_review_status.get("needs_correction", 0))
    seventh.metric("Aggressive actions", stats.aggression_count)
    eighth.metric("Passive actions", stats.passive_count)

    if stats.hand_count == 0:
        st.info("No hands recorded yet. Add hands in the Enter Hand tab to see session stats.")
        return

    winning_col, losing_col = st.columns(2)
    with winning_col:
        st.markdown("##### Biggest Winning Hands")
        st.dataframe(_hand_summary_rows(stats.biggest_winning_hands), hide_index=True, width="stretch")
    with losing_col:
        st.markdown("##### Biggest Losing Hands")
        st.dataframe(_hand_summary_rows(stats.biggest_losing_hands), hide_index=True, width="stretch")

    tags_col, actions_col = st.columns(2)
    with tags_col:
        st.markdown("##### Tag Counts")
        if stats.hands_by_tag:
            st.dataframe(
                [{"Tag": tag, "Hands": count} for tag, count in sorted(stats.hands_by_tag.items(), key=lambda item: -item[1])],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No tags applied yet.")
    with actions_col:
        st.markdown("##### Action Counts")
        if stats.action_counts_by_type:
            st.dataframe(
                [{"Action": action, "Count": count} for action, count in sorted(stats.action_counts_by_type.items(), key=lambda item: -item[1])],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No street actions recorded yet.")

    with st.expander("Danger zone: delete this session"):
        st.warning(
            f"Deleting **{session.name}** removes all {stats.hand_count} hands, actions, and "
            "reviews in it. Uploaded videos are kept but unlinked. This cannot be undone."
        )
        confirm = st.checkbox(
            "I understand this permanently deletes the session and its hands.",
            key=f"confirm_delete_session_{session.id}",
        )
        if st.button("Delete session", disabled=not confirm, key=f"delete_session_{session.id}"):
            db.delete_session(session.id)
            flash(f"Session '{session.name}' deleted.", icon="🗑️")
            st.rerun()


def create_hand_form(db: PokerDatabase, session_id: int | None) -> None:
    if session_id is None:
        st.error("Select a saved session before adding hands.")
        return

    existing_hands = db.fetch_hands_by_session(session_id)
    next_hand_number = max((hand.hand_number for hand in existing_hands), default=0) + 1

    # clear_on_submit=False so a validation error does not wipe the user's work;
    # the form and editors are reset explicitly after a successful save.
    with st.form("create_hand", clear_on_submit=False):
        st.markdown("#### Hand Setup")
        setup_left, setup_right = st.columns(2)
        with setup_left:
            hand_number = st.number_input("Hand number", min_value=1, step=1, value=next_hand_number)
            game_type = st.text_input("Game type", value="No-limit Hold'em")
            blinds_antes = st.text_input("Blinds / antes", placeholder="1/2 NL, 0.25 ante")
            table_size = st.number_input("Table size", min_value=2, max_value=10, value=6, step=1)
            effective_stack = st.number_input("Effective stack (BB)", min_value=0.0, step=1.0)
            source_type = st.selectbox("Source", ["manual", "cv_import", "corrected_cv"])
        with setup_right:
            hero_position = st.selectbox("Hero position", POSITIONS, index=6)
            hero_cards = st.text_input("Hero cards", placeholder="Ah Qs")
            board_cards = st.text_input("Board cards", placeholder="Qd 7s 2c 9h 3s")
            pot_size = st.number_input("Final pot size (BB)", min_value=0.0, step=1.0)
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
        if any(hand.hand_number == int(hand_number) for hand in existing_hands):
            st.error(f"Hand #{int(hand_number)} already exists in this session. Pick a different number.")
            return
        try:
            # One transaction: a validation error in any player/action row rolls
            # back the whole hand instead of persisting a partial save.
            with db.transaction():
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
        # Reset the data editors explicitly: clear_on_submit does not cover them,
        # and stale rows would leak into the next hand.
        for editor_key in ["players_editor", *(f"{street}_actions_editor" for street in STREETS)]:
            st.session_state.pop(editor_key, None)
        flash(f"Hand #{int(hand_number)} saved.")
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
        width="stretch",
        column_config={
            "Position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
            "Starting stack": st.column_config.NumberColumn("Starting stack (BB)", min_value=0),
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
                width="stretch",
                column_config={
                    "Player": st.column_config.TextColumn("Player"),
                    "Position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
                    "Action": st.column_config.SelectboxColumn("Action", options=ACTION_TYPES),
                    "Amount": st.column_config.NumberColumn("Amount (BB)", min_value=0),
                    "Pot before": st.column_config.NumberColumn("Pot before (BB)", min_value=0),
                    "Stack before": st.column_config.NumberColumn("Stack before (BB)", min_value=0),
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

    filter_col, page_col = st.columns([3, 1])
    with filter_col:
        status_filter = st.multiselect(
            "Filter by review status", REVIEW_STATUSES, default=[], placeholder="All statuses"
        )
    if status_filter:
        hands = [hand for hand in hands if hand.review_status in status_filter]
    if not hands:
        st.caption("No hands match the selected filters.")
        return

    # Paginate so large sessions do not render (and query) every hand each rerun.
    page_size = 10
    total_pages = (len(hands) + page_size - 1) // page_size
    with page_col:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (int(page) - 1) * page_size
    st.caption(f"Showing hands {start + 1}–{min(start + page_size, len(hands))} of {len(hands)}")

    for hand in hands[start : start + page_size]:
        if hand.id is None:
            continue
        players = db.fetch_players_by_hand(hand.id)
        actions = db.fetch_actions_by_hand(hand.id)
        with st.expander(
            f"Hand #{hand.hand_number}: {hand.hero_cards or 'unknown cards'} "
            f"{hand.hero_bb_won or 0:+g} BB [{hand.review_status}]"
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
                flash("Mock review saved.")
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
            amount = cols[4].number_input("Amount (BB)", min_value=0.0, value=action.amount)
            action_index = cols[5].number_input(
                "Order", min_value=1, value=action.action_index or 1, step=1
            )
            notes = cols[6].text_input("Notes", value=action.notes)
            pot_col, stack_col = st.columns(2)
            pot_before = pot_col.number_input("Pot before (BB)", min_value=0.0, value=action.pot_before)
            stack_before = stack_col.number_input("Stack before (BB)", min_value=0.0, value=action.stack_before)
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
                    pot_before=pot_before,
                    stack_before=stack_before,
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
        raw = uploaded.getvalue()
        if len(raw) > MAX_IMPORT_BYTES:
            st.error(f"Import file is too large ({len(raw) / 1_048_576:.0f} MB). Limit is 10 MB.")
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            st.error(f"That file is not valid JSON: {exc}")
            return
        try:
            imported = import_session(db, payload)
        except (ValidationError, KeyError, ValueError) as exc:
            st.error(f"Import failed — the file does not match the expected session format. {exc}")
            return
        flash(f"Imported session: {imported.name}")
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
    baseline_options = {"(none)": None} | {
        f"{chart.position} {chart.scenario.replace('_', ' ')} — {chart.description}": chart
        for chart in available_ranges()
    }
    baseline_choice = st.selectbox(
        "Positional baseline (optional, overrides the label)",
        options=list(baseline_options.keys()),
        help="100bb 9-max study charts — a defensible starting point, not solver output.",
    )
    custom_range = st.text_input(
        "Custom villain range (optional, overrides both)",
        placeholder="e.g. 22+,ATs+,KQo or KK",
        help="Standard range notation. Leave empty to use the label or baseline.",
    )
    baseline_chart = baseline_options[baseline_choice]
    range_display = range_label
    if baseline_chart is not None:
        range_label = baseline_chart.notation
        range_display = f"{baseline_chart.position} {baseline_chart.scenario.replace('_', ' ')}"
    if custom_range.strip():
        range_label = custom_range.strip()
        range_display = "custom range"

    first, second = st.columns(2)
    with first:
        pot_before_call = st.number_input("Pot before call (BB)", min_value=0.0, step=1.0)
        call_amount = st.number_input("Call amount (BB)", min_value=0.0, step=1.0)
        compute_equity = st.checkbox("Compute Hero equity vs range", value=True)
    with second:
        pot_size = st.number_input("Pot size for bet/bluff (BB)", min_value=0.0, step=1.0)
        bet_size = st.number_input("Bet size (BB)", min_value=0.0, step=1.0)
        fold_frequency_pct = st.slider("Estimated fold frequency", 0, 100, 40)

    math_facts: dict[str, float | str] = {}
    equity_result = None
    errors: list[str] = []
    call_metrics: list[tuple[str, str, str | None]] = []
    bet_metrics: list[tuple[str, str, str | None]] = []

    if call_amount > 0 and pot_before_call > 0:
        try:
            required = required_equity_to_call(call_amount, pot_before_call)
            math_facts["required_equity_to_call"] = required
            call_metrics.append(("Required equity to call", format_percentage(required), None))
        except ValueError as exc:
            errors.append(str(exc))

    if bet_size > 0 and pot_size > 0:
        try:
            bluff_frequency = break_even_bluff_frequency(bet_size, pot_size)
            mdf = minimum_defense_frequency(bet_size, pot_size)
            bluff_fraction = optimal_bluff_fraction(bet_size, pot_size)
            bluffs_per_value = bluff_to_value_ratio(bet_size, pot_size)
            math_facts["break_even_bluff_frequency"] = bluff_frequency
            math_facts["minimum_defense_frequency"] = mdf
            math_facts["optimal_river_bluff_fraction"] = round(bluff_fraction, 4)
            bet_metrics.append(("Break-even bluff freq", format_percentage(bluff_frequency), None))
            bet_metrics.append(("Min defense freq (MDF)", format_percentage(mdf), None))
            bet_metrics.append(
                (
                    "River bluff fraction",
                    format_percentage(bluff_fraction),
                    f"Optimal share of a polarized river betting range that is bluffs "
                    f"({bluffs_per_value:.2f} bluffs per value bet). Earlier streets support more bluffs.",
                )
            )
        except ValueError as exc:
            errors.append(str(exc))

    if compute_equity and hand.hero_cards:
        try:
            with st.spinner("Computing Hero equity vs range..."):
                equity_result = _cached_equity(hand.hero_cards, hand.board_cards, range_label)
            math_facts["equity"] = equity_result.equity or "unavailable"
            if equity_result.equity is None:
                call_metrics.append((f"Equity vs {range_display}", "unavailable", equity_result.method))
            else:
                equity_help = (
                    f"{equity_result.method}, confidence {format_percentage(equity_result.confidence)}"
                )
                if equity_result.std_error:
                    ci_low, ci_high = (
                        max(0.0, equity_result.equity - 1.96 * equity_result.std_error),
                        min(1.0, equity_result.equity + 1.96 * equity_result.std_error),
                    )
                    equity_help += (
                        f". Monte-Carlo 95% CI: {format_percentage(ci_low)}–{format_percentage(ci_high)}"
                    )
                call_metrics.append(
                    (
                        f"Equity vs {range_display}",
                        format_percentage(equity_result.equity),
                        equity_help,
                    )
                )
        except ValueError as exc:
            errors.append(str(exc))

    if equity_result is not None and equity_result.equity is not None and call_amount > 0 and pot_before_call > 0:
        ev_value = call_ev(equity_result.equity, pot_before_call, call_amount)
        math_facts["call_ev"] = round(ev_value, 3)
        call_metrics.append(("Call EV (approx)", f"{ev_value:+.2f}", "assumes no future betting"))

    fold_frequency = fold_frequency_pct / 100
    if bet_size > 0 and pot_size > 0:
        bluff_value = bluff_ev(fold_frequency, pot_size, bet_size)
        math_facts["bluff_ev"] = round(bluff_value, 3)
        bet_metrics.append(
            ("Bluff EV (approx)", f"{bluff_value:+.2f}", f"at {fold_frequency_pct}% folds, zero equity when called")
        )

    if call_metrics or bet_metrics:
        st.markdown("##### Calling Math")
        if call_metrics:
            for column, (metric_label, value, help_text) in zip(st.columns(max(len(call_metrics), 3)), call_metrics):
                column.metric(metric_label, value, help=help_text)
        else:
            st.caption("Enter a pot and call amount to see calling math.")
        st.markdown("##### Betting Math")
        if bet_metrics:
            for column, (metric_label, value, help_text) in zip(st.columns(max(len(bet_metrics), 3)), bet_metrics):
                column.metric(metric_label, value, help=help_text)
        else:
            st.caption("Enter a pot and bet size to see betting math.")
    if equity_result is not None:
        st.caption(equity_result.notes)

    for error in errors:
        st.error(error)

    st.markdown("##### Study Tools")
    realization_tab, multiway_tab, outs_tab, icm_tab = st.tabs(
        ["Equity Realization", "Multiway Equity", "Outs & Draws", "Tournament ICM"]
    )
    with realization_tab:
        show_equity_realization_tool(equity_result)
    with multiway_tab:
        show_multiway_equity_tool(hand, range_label, range_display)
    with outs_tab:
        show_outs_tool()
    with icm_tab:
        show_icm_tool()

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
        flash("Math-aware mock review saved.")
        st.rerun()

    with st.expander("Structured future-LLM prompt"):
        st.code(prompt, language="text")


def show_equity_realization_tool(equity_result) -> None:
    st.caption(
        "Raw equity assumes every hand reaches showdown. Out of position or with a capped "
        "range, Hero realizes less of it. Factors are study heuristics, not solver output."
    )
    if equity_result is None or equity_result.equity is None:
        st.info("Compute Hero equity vs a range above to estimate realized equity.")
        return
    scenario = st.selectbox(
        "Realization scenario",
        options=list(REALIZATION_FACTOR_GUIDE.keys()),
        format_func=lambda key: key.replace("_", " "),
    )
    factor = REALIZATION_FACTOR_GUIDE[scenario]
    realized = realized_equity(equity_result.equity, factor)
    raw_col, factor_col, realized_col = st.columns(3)
    raw_col.metric("Raw equity", format_percentage(equity_result.equity))
    factor_col.metric("Realization factor", f"{factor:.2f}×")
    realized_col.metric("Realized equity (est.)", format_percentage(realized))


def show_multiway_equity_tool(hand: Hand, range_label: str, range_display: str) -> None:
    st.caption(
        "Pot-share equity vs two or more ranges. Villain 1 uses the range selected above; "
        "add at least one more villain. Multiway pots need stronger hands to continue."
    )
    if not hand.hero_cards:
        st.info("This hand has no Hero cards recorded.")
        return
    second = st.text_input(
        "Villain 2 range", value="standard", help="A range label or standard notation."
    )
    third = st.text_input("Villain 3 range (optional)", value="")
    villain_ranges = [range_label, second.strip(), *( [third.strip()] if third.strip() else [] )]
    if not second.strip():
        st.info("Enter a Villain 2 range to compute multiway equity.")
        return
    try:
        with st.spinner("Computing multiway pot share..."):
            result = _cached_multiway_equity(hand.hero_cards, hand.board_cards, tuple(villain_ranges))
    except ValueError as exc:
        st.error(str(exc))
        return
    if result is None:
        st.warning("Multiway equity needs the eval7 engine (not available).")
        return
    if result.equity is None:
        st.warning(f"Could not compute: {result.notes}")
        return
    share_col, fair_col = st.columns(2)
    help_text = result.notes
    if result.std_error:
        low, high = (
            max(0.0, result.equity - 1.96 * result.std_error),
            min(1.0, result.equity + 1.96 * result.std_error),
        )
        help_text += f" 95% CI: {format_percentage(low)}–{format_percentage(high)}."
    share_col.metric(f"Hero pot share ({len(villain_ranges) + 1}-way)", format_percentage(result.equity), help=help_text)
    fair_col.metric(
        "Fair share",
        format_percentage(1 / (len(villain_ranges) + 1)),
        help="An equal split of the pot. Above this, Hero is profiting from the multiway pot.",
    )
    st.caption(f"Villain 1: {range_display} · ranges: {result.villain_range_label}")


def show_outs_tool() -> None:
    st.caption("Draw equity from counted outs — the rule of 2 and 4 next to the exact odds.")
    outs = st.number_input("Outs", min_value=0, max_value=20, value=9, step=1)
    street = st.radio(
        "Cards to come",
        options=["Flop → river (2 cards)", "Turn → river (1 card)"],
        horizontal=True,
    )
    streets_to_come = 2 if street.startswith("Flop") else 1
    unseen = 47 if streets_to_come == 2 else 46
    if outs == 0:
        st.info("Count Hero's outs to estimate draw equity.")
        return
    rule = outs_to_equity_rule(int(outs), streets_to_come)
    exact = outs_to_equity_exact(int(outs), unseen, streets_to_come)
    rule_col, exact_col = st.columns(2)
    rule_col.metric("Rule of 2 and 4", format_percentage(min(rule, 1.0)))
    exact_col.metric(
        "Exact",
        format_percentage(exact),
        help=f"{outs} outs among {unseen} unseen cards, {streets_to_come} card(s) to come.",
    )


def show_icm_tool() -> None:
    st.caption(
        "Malmuth-Harville ICM: converts tournament chip stacks into prize equity. "
        "Chips lost hurt more than chips won help — the risk premium quantifies that."
    )
    stacks_text = st.text_input("Stacks (comma-separated chips)", value="5000, 3000, 2000")
    payouts_text = st.text_input("Payouts (comma-separated, best first)", value="50, 30, 20")
    try:
        stacks = [float(part) for part in stacks_text.split(",") if part.strip()]
        payouts = [float(part) for part in payouts_text.split(",") if part.strip()]
        equities = icm_equities(stacks, payouts)
    except ValueError as exc:
        st.error(f"Could not compute ICM: {exc}")
        return
    st.dataframe(
        [
            {
                "Player": index + 1,
                "Stack": f"{stack:g}",
                "Chip share": format_percentage(stack / sum(stacks)),
                "ICM equity": f"{equity:.2f}",
                "Prize share": format_percentage(equity / sum(payouts)),
            }
            for index, (stack, equity) in enumerate(zip(stacks, equities))
        ],
        hide_index=True,
        width="stretch",
    )
    hero_col, risk_col = st.columns(2)
    hero_seat = hero_col.number_input("Hero player #", min_value=1, max_value=len(stacks), value=1, step=1)
    max_risk = stacks[int(hero_seat) - 1]
    risk_amount = risk_col.number_input(
        "Chips at risk", min_value=0.0, max_value=float(max_risk), value=min(1000.0, max_risk / 2), step=100.0
    )
    if risk_amount > 0 and risk_amount < max_risk:
        premium = icm_risk_premium(stacks, payouts, int(hero_seat) - 1, risk_amount)
        st.metric(
            "ICM cost of losing those chips",
            f"{premium:.2f}",
            help=(
                "Prize equity lost if Hero loses this many chips. Compare against the prize "
                "equity gained by winning the same pot — the gap is the ICM risk premium."
            ),
        )


def show_coach_review(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    st.subheader("Post-Session Coach Review")
    st.caption("Provider reviews are for completed hands/sessions only.")

    hands = db.fetch_hands_by_session(session.id)
    if not hands:
        st.info("Save or load hands before generating coach reviews.")
        return

    provider_choice = st.selectbox("Provider", ["Mock", "Claude (Anthropic)", "Cloud (OpenAI)"])
    provider_key = {
        "Mock": "mock",
        "Claude (Anthropic)": "anthropic",
        "Cloud (OpenAI)": "cloud",
    }[provider_choice]
    provider = get_provider_from_env(provider_key)
    # get_provider_from_env falls back to Mock when the selected provider's key is absent.
    if provider_key in {"anthropic", "cloud"} and provider.provider_name == "mock":
        key_name = "ANTHROPIC_API_KEY" if provider_key == "anthropic" else "OPENAI_API_KEY"
        st.warning(f"{provider_choice} selected but no {key_name} is configured. Falling back to Mock.")
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
            with st.spinner("Generating hand review..."):
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
            flash(f"Saved provider review #{saved.id}.")
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
    st.caption(
        f"Selected hands: {', '.join(f'#{hand.hand_number}' for hand in selected_hands) or 'none'}"
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
            with st.spinner("Generating session review..."):
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
            flash(f"Saved provider session review #{saved.id}.")
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
            flash(f"Saved video #{saved_video.id}.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("#### Stored Videos")
    filter_to_session = st.checkbox(
        f"Show only videos linked to this session (#{session.id})",
        value=False,
        key="video_filter_session",
        disabled=session.id is None,
    )
    all_videos = db.fetch_videos(session.id if (filter_to_session and session.id is not None) else None)
    if not all_videos:
        if filter_to_session:
            st.info("No videos are linked to the selected session yet.")
        else:
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
            with st.spinner("Extracting frames — this can take a while for long videos..."):
                summary = extract_frames_for_video(
                    db,
                    video.id,
                    frames_per_second=float(frames_per_second),
                    max_frames=int(max_frames),
                    start_time_seconds=float(start_time),
                    end_time_seconds=float(end_time) if end_time > 0 else None,
                )
            if summary.errors:
                flash(
                    f"Extracted {summary.frames_extracted} frames with "
                    f"{len(summary.errors)} warnings: {'; '.join(summary.errors)}",
                    icon="⚠️",
                )
            else:
                flash(f"Extracted {summary.frames_extracted} frames to {summary.output_dir}.")
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
            width="stretch",
        )

    frames = db.fetch_frames_by_video(video.id)
    st.caption(f"Extracted frames: {len(frames)}")
    confirm_delete = st.checkbox(
        "Confirm delete extracted frames for this video",
        key=f"delete_frames_confirm_{video.id}",
    )
    if st.button("Delete extracted frames", key=f"delete_frames_{video.id}", disabled=not confirm_delete):
        deleted = delete_extracted_frames(db, video.id)
        flash(f"Deleted {deleted} extracted frame records/files.")
        st.rerun()
    show_frame_preview(frames)

    with st.expander("Danger zone: delete this video"):
        st.warning(
            "Removes the stored video file, all extracted frames, and job history. "
            "Hands and sessions are unaffected. This cannot be undone."
        )
        confirm_video = st.checkbox(
            "I understand this permanently deletes the video and its files.",
            key=f"confirm_delete_video_{video.id}",
        )
        if st.button("Delete video", key=f"delete_video_{video.id}", disabled=not confirm_video):
            delete_extracted_frames(db, video.id)  # frame files first; rows would cascade anyway
            db.delete_video(video.id)
            Path(video.stored_path).unlink(missing_ok=True)
            flash(f"Deleted video {video.original_filename}.", icon="🗑️")
            st.rerun()


def _safe_image(path: str, caption: str | None = None) -> None:
    """Render an image, degrading gracefully if the file was moved or deleted."""
    if Path(path).is_file():
        st.image(path, caption=caption)
    else:
        st.warning(f"Image file missing: {path}")


def show_frame_preview(frames) -> None:
    if not frames:
        st.caption("No frames extracted yet.")
        return
    st.markdown("##### Frame Preview")
    representative = select_representative_frames(frames, limit=12)
    columns = st.columns(4)
    for index, frame in enumerate(representative):
        with columns[index % 4]:
            _safe_image(frame.image_path, caption=f"{frame.timestamp_seconds:.2f}s")

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
    st.caption(f"{selected.timestamp_seconds:.2f}s · {selected.image_path}")
    _safe_image(selected.image_path)


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
    _safe_image(frame.image_path, caption=f"Calibration frame at {frame.timestamp_seconds:.2f}s")
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
        flash("Duplicated ROI profile.")
        st.rerun()
    with st.expander("Danger zone: delete this profile"):
        st.warning(
            f"Deleting **{profile.name}** removes all its calibrated regions. This cannot be undone."
        )
        confirm_profile = st.checkbox(
            "I understand this permanently deletes the profile and its regions.",
            key=f"confirm_delete_profile_{profile.id}",
        )
        if st.button("Delete profile", key=f"roi_delete_{profile.id}", disabled=not confirm_profile):
            db.delete_roi_profile(profile.id)
            flash(f"Deleted ROI profile '{profile.name}'.", icon="🗑️")
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
        flash("ROI profile created.")
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
            flash("Starter ROI profile created.")
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
        raw = uploaded.getvalue()
        if len(raw) > MAX_IMPORT_BYTES:
            st.error(f"Import file is too large ({len(raw) / 1_048_576:.0f} MB). Limit is 10 MB.")
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            imported = import_roi_profile(db, payload)
            flash(f"Imported ROI profile: {imported.name}")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Could not import ROI profile: it contains duplicate ROI keys.")
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
            flash("ROI region added.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error(f"An ROI region with the key '{roi_key.strip()}' already exists in this profile.")
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
        width="stretch",
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

    def build_updated_region() -> ROIRegion:
        # Constructed lazily inside each submit branch: an invalid field (e.g. an
        # empty ROI key) must surface as a form error, not a raw traceback.
        return ROIRegion(
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
            updated = build_updated_region()
            validate_roi_bounds(updated, image_width=frame_width, image_height=frame_height)
            db.update_roi_region(updated)
            flash("ROI region updated.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error(f"An ROI region with the key '{roi_key.strip()}' already exists in this profile.")
        except (ValueError, ValidationError) as exc:
            st.error(f"Could not update ROI region: {exc}")
    if submitted_preview:
        try:
            result = save_roi_crop_preview(frame, build_updated_region())
            st.write(
                {
                    "Crop": result.crop_path,
                    "Size": f"{result.crop_width}x{result.crop_height}",
                    "Source timestamp": result.source_timestamp_seconds,
                }
            )
            st.image(result.crop_path)
        except (ValueError, ValidationError) as exc:
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
        flash("ROI region deleted.")
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
