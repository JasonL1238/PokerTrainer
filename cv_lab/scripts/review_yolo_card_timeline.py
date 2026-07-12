"""Streamlit UI for reviewing reconstructed YOLO card hands.

This is an offline, post-session correction tool. It reviews saved timeline
JSON and writes hand-level corrections; it does not capture live tables or give
poker advice.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv_lab.scripts.export_yolo_card_hands_for_app import (  # noqa: E402
    HAND_CORRECTION_FIELDS,
    export_timeline,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", default="cv_lab/results/yolo_card_timeline_card_changes_v1.json")
    parser.add_argument("--validation", default="cv_lab/results/yolo_card_timeline_validation_card_changes_v1.json")
    parser.add_argument("--corrections", default="cv_lab/results/yolo_card_hand_corrections_card_changes_v1.csv")
    parser.add_argument("--export-out", default="cv_lab/results/yolo_card_hands_corrected_draft_session_card_changes_v1.json")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_corrections(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("hand_number"):
            out[int(row["hand_number"])] = row
    return out


def _write_corrections(path: Path, rows_by_hand: dict[int, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_hand[key] for key in sorted(rows_by_hand)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HAND_CORRECTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cards(cards: list[str] | None) -> str:
    return " ".join(cards or [])


def _validation_by_hand(validation: dict) -> dict[int, dict]:
    out = {}
    for hand in validation.get("hands", []):
        if hand.get("hand_number") is not None:
            out[int(hand["hand_number"])] = hand
    return out


def _set_hand(hand_idx: int) -> None:
    st.query_params["hand"] = str(hand_idx)
    st.rerun()


def _install_keyboard_nav(prev_idx: int, next_idx: int) -> None:
    script = f"""
    <script>
    const prevHand = {json.dumps(str(prev_idx))};
    const nextHand = {json.dumps(str(next_idx))};
    function go(hand) {{
      const params = new URLSearchParams(window.parent.location.search);
      params.set("hand", hand);
      window.parent.location.search = "?" + params.toString();
    }}
    function handler(event) {{
      const tag = (event.target && event.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (event.key === "ArrowLeft") {{
        event.preventDefault();
        go(prevHand);
      }}
      if (event.key === "ArrowRight") {{
        event.preventDefault();
        go(nextHand);
      }}
    }}
    try {{
      window.parent.document.removeEventListener("keydown", window.__yoloHandNavHandler);
    }} catch (e) {{}}
    window.__yoloHandNavHandler = handler;
    window.parent.document.addEventListener("keydown", handler);
    </script>
    """
    components.html(script, height=0)


def _dataset_path(timeline: dict) -> Path | None:
    dataset = timeline.get("metadata", {}).get("dataset")
    return Path(dataset) if dataset else None


def main() -> None:
    args = _args()
    timeline_path = Path(args.timeline)
    validation_path = Path(args.validation)
    corrections_path = Path(args.corrections)
    export_out = Path(args.export_out)

    st.set_page_config(page_title="YOLO Hand Timeline Review", layout="wide")
    st.title("YOLO Hand Timeline Review")
    st.caption("Offline completed-session reconstruction review. No live capture, overlays, or current-hand advice.")

    if not timeline_path.exists():
        st.error(f"Missing timeline: {timeline_path}")
        st.stop()

    timeline = _read_json(timeline_path)
    validation = _read_json(validation_path) if validation_path.exists() else {"hands": []}
    corrections = _read_corrections(corrections_path)
    hands = timeline.get("hands", [])
    validation_lookup = _validation_by_hand(validation)
    dataset = _dataset_path(timeline)

    if not hands:
        st.info("No provisional hands in this timeline.")
        st.stop()

    with st.sidebar:
        st.header("Files")
        st.code(str(timeline_path.resolve()))
        st.code(str(corrections_path.resolve()))
        show_warning_only = st.checkbox("Only hands with warnings", value=False)
        query = st.text_input("Filter", placeholder="hand number, card, warning")

        options = []
        for idx, hand in enumerate(hands):
            hand_no = int(hand.get("hand_number", idx + 1))
            warning_count = validation_lookup.get(hand_no, {}).get("warning_count", 0)
            corrected = " corrected" if hand_no in corrections else ""
            label = (
                f"{idx:03d} | hand {hand_no} | t={hand.get('t_start')}..{hand.get('t_end')} | "
                f"hero={_cards(hand.get('hero'))} board={_cards(hand.get('board'))} | "
                f"warnings={warning_count}{corrected}"
            )
            if show_warning_only and warning_count == 0:
                continue
            if query.strip() and query.strip().lower() not in label.lower():
                continue
            options.append((idx, label))

        if not options:
            st.info("No hands match the filter.")
            st.stop()

        requested = st.query_params.get("hand")
        try:
            selected = int(requested) if requested is not None else options[0][0]
        except ValueError:
            selected = options[0][0]
        option_indices = [idx for idx, _ in options]
        if selected not in option_indices:
            selected = option_indices[0]

        selected = st.selectbox(
            "Hand",
            options=option_indices,
            index=option_indices.index(selected),
            format_func=lambda idx: dict(options).get(idx, str(idx)),
        )

    hand = hands[selected]
    hand_no = int(hand.get("hand_number", selected + 1))
    warning_report = validation_lookup.get(hand_no, {"warnings": [], "warning_count": 0})
    current = corrections.get(hand_no, {})
    prev_idx = max(0, selected - 1)
    next_idx = min(len(hands) - 1, selected + 1)
    _install_keyboard_nav(prev_idx, next_idx)

    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    with nav_left:
        if st.button("Previous", disabled=selected <= 0):
            _set_hand(prev_idx)
    with nav_mid:
        st.caption("Use left/right arrow keys when you are not typing in a field.")
    with nav_right:
        if st.button("Next", disabled=selected >= len(hands) - 1):
            _set_hand(next_idx)

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader(f"Hand {hand_no}")
        stats = st.columns(4)
        stats[0].metric("States", hand.get("n_states", 0))
        stats[1].metric("Warnings", warning_report.get("warning_count", 0))
        stats[2].metric("Hero", _cards(hand.get("hero")) or "none")
        stats[3].metric("Board", _cards(hand.get("board")) or "none")

        if warning_report.get("warnings"):
            st.warning("Validation warnings")
            for warning in warning_report["warnings"][:20]:
                st.write(f"`{warning.get('code')}` {warning.get('image') or ''} {warning.get('message')}")
            if len(warning_report["warnings"]) > 20:
                st.caption(f"{len(warning_report['warnings']) - 20} more warnings not shown.")
        else:
            st.success("No validator warnings for this hand.")

        st.subheader("Source Frames")
        source_images = hand.get("source_images", [])
        if dataset and source_images:
            cols = st.columns(3)
            for i, rel_image in enumerate(source_images[:9]):
                review_image = dataset / "review" / f"{Path(rel_image).stem}.jpg"
                target = review_image if review_image.exists() else dataset / rel_image
                with cols[i % 3]:
                    if target.exists():
                        st.image(str(target), caption=f"{Path(rel_image).name}", width="stretch")
                    else:
                        st.caption(rel_image)
        else:
            st.info("No source images available.")

    with right:
        st.subheader("Hand Correction")
        with st.form(f"hand_{hand_no}_correction"):
            hero_cards = st.text_input(
                "Hero cards",
                value=current.get("hero_cards") or _cards(hand.get("hero")),
                placeholder="Ah Qs",
            )
            board_cards = st.text_input(
                "Board cards",
                value=current.get("board_cards") or _cards(hand.get("board")),
                placeholder="Qd 7s 2c 9h Kc",
            )
            action_options = ["keep", "drop"]
            action = current.get("action", "keep") or "keep"
            if action not in action_options:
                action = "keep"
            action_choice = st.selectbox("Export action", action_options, index=action_options.index(action))
            notes = st.text_area("Notes", value=current.get("notes", ""), height=120)
            save = st.form_submit_button("Save Correction")
            save_next = st.form_submit_button("Save + Next")

        if save or save_next:
            corrections[hand_no] = {
                "hand_number": str(hand_no),
                "hero_cards": hero_cards.strip(),
                "board_cards": board_cards.strip(),
                "action": "" if action_choice == "keep" else action_choice,
                "notes": notes.strip(),
            }
            _write_corrections(corrections_path, corrections)
            st.success("Saved.")
            if save_next and selected < len(hands) - 1:
                _set_hand(next_idx)

        if st.button("Clear This Correction", disabled=hand_no not in corrections):
            corrections.pop(hand_no, None)
            _write_corrections(corrections_path, corrections)
            st.rerun()

        st.divider()
        st.subheader("Export")
        if st.button("Write Corrected Draft JSON"):
            payload = export_timeline(
                timeline_path,
                export_out,
                session_name="YOLO corrected card draft session",
                hand_corrections_path=corrections_path,
            )
            st.success(
                f"Wrote {export_out} with {payload['cv_import_summary']['exported_hands']} exported hands "
                f"and {payload['cv_import_summary']['skipped_hands']} skipped."
            )


if __name__ == "__main__":
    main()
