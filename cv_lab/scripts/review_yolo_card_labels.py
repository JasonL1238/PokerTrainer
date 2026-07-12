"""Streamlit UI for correcting YOLO card autolabels.

This is an offline/post-session review tool. It edits corrections.csv for a
generated YOLO dataset; it does not capture live tables or provide hand advice.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cv_lab/datasets/yolo_cards_autolabel_v3")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_optional_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _read_csv(path)


def _load_classes(dataset: Path) -> list[str]:
    return [line.strip() for line in (dataset / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def _frame_key(row: dict) -> str:
    return row["image"]


def _frame_label(row: dict) -> str:
    return f"{row['split']} | t={row['time_s']}s | {Path(row['image']).name} | {row['detections']} det"


def _status_for_frame(rows: list[dict]) -> str:
    changed = 0
    deleted = 0
    for row in rows:
        if row.get("action", "").strip().lower() in {"delete", "drop", "remove"}:
            deleted += 1
        elif row.get("correct_label", "").strip():
            changed += 1
    if changed or deleted:
        return f"{changed} changed, {deleted} deleted"
    return "no corrections"


def _row_signature(row: dict) -> tuple[str, str]:
    return row["label"], row["detection_index"]


def _set_frame(frame_idx: int) -> None:
    st.query_params["frame"] = str(frame_idx)
    st.rerun()


def _write_missing_labels(path: Path, rows: list[dict]) -> None:
    fieldnames = ["image", "split", "time_s", "missing_cards", "notes"]
    _write_csv(path, rows, fieldnames)


def _install_keyboard_nav(prev_idx: int, next_idx: int) -> None:
    script = f"""
    <script>
    const prevFrame = {json.dumps(str(prev_idx))};
    const nextFrame = {json.dumps(str(next_idx))};
    function go(frame) {{
      const params = new URLSearchParams(window.parent.location.search);
      params.set("frame", frame);
      window.parent.location.search = "?" + params.toString();
    }}
    function handler(event) {{
      const tag = (event.target && event.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (event.key === "ArrowLeft") {{
        event.preventDefault();
        go(prevFrame);
      }}
      if (event.key === "ArrowRight") {{
        event.preventDefault();
        go(nextFrame);
      }}
    }}
    try {{
      window.parent.document.removeEventListener("keydown", window.__yoloCardNavHandler);
    }} catch (e) {{}}
    window.__yoloCardNavHandler = handler;
    window.parent.document.addEventListener("keydown", handler);
    </script>
    """
    components.html(script, height=0)


def main() -> None:
    args = _args()
    dataset = Path(args.dataset)
    corrections_path = dataset / "corrections.csv"
    missing_path = dataset / "missing_labels.csv"
    manifest_path = dataset / "manifest.csv"
    review_dir = dataset / "review"

    st.set_page_config(page_title="YOLO Card Label Review", layout="wide")
    st.title("YOLO Card Label Review")
    st.caption("Offline completed-session label correction. No live capture, overlays, or poker advice.")

    if not corrections_path.exists() or not manifest_path.exists():
        st.error(f"Missing dataset files under {dataset}")
        st.stop()

    classes = _load_classes(dataset)
    class_options = ["keep"] + classes
    action_options = ["keep", "delete"]
    rows = _read_csv(corrections_path)
    missing_rows = _read_optional_csv(missing_path)
    manifest = _read_csv(manifest_path)
    fieldnames = list(rows[0].keys()) if rows else []

    by_frame: dict[str, list[dict]] = {}
    for row in rows:
        by_frame.setdefault(_frame_key(row), []).append(row)
    missing_by_frame = {row["image"]: row for row in missing_rows}

    searchable = []
    for i, frame in enumerate(manifest):
        frame_rows = by_frame.get(_frame_key(frame), [])
        status = _status_for_frame(frame_rows)
        labels = frame.get("labels", "")
        searchable.append((i, f"{i:04d} | {_frame_label(frame)} | {status} | {labels}"))

    with st.sidebar:
        st.header("Dataset")
        st.code(str(dataset.resolve()))
        only_uncorrected = st.checkbox("Show only uncorrected", value=False)
        query = st.text_input("Filter", placeholder="AS, 10D, train, t=...")
        options = searchable
        if only_uncorrected:
            options = [(i, label) for i, label in options if "no corrections" in label]
        if query.strip():
            q = query.strip().lower()
            options = [(i, label) for i, label in options if q in label.lower()]
        if not options:
            st.info("No frames match the filter.")
            st.stop()
        option_indices = [i for i, _ in options]
        requested_frame = st.query_params.get("frame")
        try:
            requested_idx = int(requested_frame) if requested_frame is not None else option_indices[0]
        except ValueError:
            requested_idx = option_indices[0]
        if requested_idx not in option_indices:
            requested_idx = option_indices[0]
        selected = st.selectbox(
            "Frame",
            options=[i for i, _ in options],
            index=option_indices.index(requested_idx),
            format_func=lambda idx: dict(options).get(idx, str(idx)),
        )
        st.write(f"{len(options)} frames shown")

    frame = manifest[selected]
    frame_rows = by_frame.get(_frame_key(frame), [])
    review_image = review_dir / f"{Path(frame['image']).stem}.jpg"
    prev_idx = max(0, selected - 1)
    next_idx = min(len(manifest) - 1, selected + 1)
    _install_keyboard_nav(prev_idx, next_idx)

    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    with nav_left:
        if st.button("Previous", disabled=selected <= 0):
            _set_frame(prev_idx)
    with nav_mid:
        st.caption("Use left/right arrow keys when you are not typing in a field.")
    with nav_right:
        if st.button("Next", disabled=selected >= len(manifest) - 1):
            _set_frame(next_idx)

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader(_frame_label(frame))
        if review_image.exists():
            st.image(str(review_image), width="stretch")
        else:
            st.warning(f"Missing review image: {review_image}")

    with right:
        st.subheader("Corrections")
        st.write("Use the detection numbers drawn on the image.")
        updated: dict[tuple[str, str], dict[str, str]] = {}
        with st.form(f"frame_{selected}_corrections"):
            for det in frame_rows:
                sig = _row_signature(det)
                st.markdown(f"**{det['detection_index']}: {det['pred_label']}**  conf `{det['conf']}`")
                current_label = det.get("correct_label", "").strip() or "keep"
                if current_label not in class_options:
                    current_label = "keep"
                current_action = det.get("action", "").strip().lower() or "keep"
                if current_action not in action_options:
                    current_action = "keep"
                label_choice = st.selectbox(
                    "Correct label",
                    class_options,
                    index=class_options.index(current_label),
                    key=f"label_{selected}_{det['detection_index']}",
                )
                action_choice = st.selectbox(
                    "Action",
                    action_options,
                    index=action_options.index(current_action),
                    key=f"action_{selected}_{det['detection_index']}",
                )
                updated[sig] = {
                    "correct_label": "" if label_choice == "keep" else label_choice,
                    "action": "" if action_choice == "keep" else action_choice,
                }
                st.divider()
            current_missing = missing_by_frame.get(frame["image"], {})
            missing_cards = st.text_input(
                "Missing cards / boxes",
                value=current_missing.get("missing_cards", ""),
                placeholder="Example: QH, 7S",
            )
            missing_notes = st.text_area(
                "Missing notes",
                value=current_missing.get("notes", ""),
                placeholder="Example: board card not boxed, hero left card missing",
                height=80,
            )
            saved = st.form_submit_button("Save This Frame")
            saved_next = st.form_submit_button("Save + Next")

        if saved or saved_next:
            for row in rows:
                sig = _row_signature(row)
                if row["image"] == frame["image"] and sig in updated:
                    row["correct_label"] = updated[sig]["correct_label"]
                    row["action"] = updated[sig]["action"]
            _write_csv(corrections_path, rows, fieldnames)
            missing_rows = [row for row in missing_rows if row["image"] != frame["image"]]
            if missing_cards.strip() or missing_notes.strip():
                missing_rows.append({
                    "image": frame["image"],
                    "split": frame["split"],
                    "time_s": frame["time_s"],
                    "missing_cards": missing_cards.strip(),
                    "notes": missing_notes.strip(),
                })
            _write_missing_labels(missing_path, missing_rows)
            st.success("Saved corrections.csv")
            if saved_next and selected < len(manifest) - 1:
                _set_frame(next_idx)
            st.rerun()

        if st.button("Clear Corrections For This Frame"):
            for row in rows:
                if row["image"] == frame["image"]:
                    row["correct_label"] = ""
                    row["action"] = ""
            _write_csv(corrections_path, rows, fieldnames)
            missing_rows = [row for row in missing_rows if row["image"] != frame["image"]]
            _write_missing_labels(missing_path, missing_rows)
            st.rerun()

    st.markdown("### Current Frame Rows")
    st.dataframe(
        [
            {
                "det": row["detection_index"],
                "pred": row["pred_label"],
                "conf": row["conf"],
                "correct_label": row.get("correct_label", ""),
                "action": row.get("action", ""),
            }
            for row in frame_rows
        ],
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Apply After Review")
    st.code(f"python cv_lab/scripts/apply_yolo_card_corrections.py --dataset {dataset}")
    st.caption(
        f"Missing boxes are logged in {missing_path}. Those need real boxes before they can be used for YOLO training."
    )


if __name__ == "__main__":
    main()
