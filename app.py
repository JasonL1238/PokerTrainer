from __future__ import annotations

import json
from datetime import date

import streamlit as st
from pydantic import ValidationError

from poker_tracker.analytics import compute_session_stats
from poker_tracker.db import DEFAULT_DB_PATH, PokerDatabase
from poker_tracker.hand_history import format_hand_history
from poker_tracker.import_export import export_hand, export_session, import_session
from poker_tracker.models import Action, HAND_TAGS, Hand, HandPlayer, Session
from poker_tracker.review import generate_mock_review
from poker_tracker.seed_data import create_sample_data


STREETS = ["preflop", "flop", "turn", "river", "showdown"]
ACTION_TYPES = ["fold", "check", "call", "bet", "raise", "all-in", "post_blind", "show", "win"]
POSITIONS = ["", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
REVIEW_STATUSES = ["unreviewed", "reviewed", "needs_correction"]


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

    dashboard_tab, entry_tab, hands_tab, transfer_tab = st.tabs(
        ["Dashboard", "Enter Hand", "Review Hands", "Import / Export"]
    )
    with dashboard_tab:
        show_session_dashboard(db, selected_session)
    with entry_tab:
        create_hand_form(db, selected_session.id)
    with hands_tab:
        show_saved_hands(db, selected_session)
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
