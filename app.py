from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from poker_tracker.coaching.coaching_prompts import (
    build_hand_review_prompt,
    build_session_review_prompt,
)
from poker_tracker.coaching.hand_history import format_hand_history
from poker_tracker.coaching.llm_providers import (
    LLMProviderError,
    build_coaching_response,
    get_provider_from_env,
)
from poker_tracker.coaching.safety import validate_post_session_prompt
from poker_tracker.coaching.solver_grounding import (
    validate_solver_coaching_response,
)
from poker_tracker.math.accounting import LedgerError
from poker_tracker.math.analytics import compute_session_stats
from poker_tracker.math.equity import get_equity_calculator
from poker_tracker.math.ev import (
    bluff_ev,
    call_ev,
    semi_bluff_break_even_fold_frequency,
    semi_bluff_ev,
)
from poker_tracker.math.icm import icm_equities, icm_risk_premium
from poker_tracker.math.pot_odds import (
    break_even_bluff_frequency,
    format_percentage,
    minimum_defense_frequency,
    pot_odds_offered_by_bet,
    rake_amount,
    required_equity_to_call,
    required_equity_to_call_after_rake,
    stack_to_pot_ratio,
)
from poker_tracker.math.preflop_ranges import available_ranges
from poker_tracker.math.ranges import RANGE_LABELS, estimate_villain_range_label, range_notation
from poker_tracker.math.study_math import (
    REALIZATION_FACTOR_GUIDE,
    bluff_to_value_ratio,
    geometric_bet_fraction,
    optimal_bluff_fraction,
    outs_to_equity_exact,
    outs_to_equity_rule,
    realized_equity,
)
from poker_tracker.persistence.completion import (
    OBSERVED_TERMINAL_EVENTS,
    CompletionEvidence,
    acknowledge_codes,
    dump_completion_evidence,
    has_operator_manual_completion,
    is_assumption_dependence_code,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import DEFAULT_DB_PATH, PokerDatabase
from poker_tracker.persistence.import_export import export_hand, export_session, import_session
from poker_tracker.persistence.models import (
    HAND_TAGS,
    Action,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    ReconstructionFrameReview,
    ROIProfile,
    ROIRegion,
    Session,
    SettlementEntry,
    SolverRangeProfile,
    StudyInclusion,
    VideoRecord,
    utc_now,
)
from poker_tracker.player_labels import actor_label
from poker_tracker.services.action_provenance import backfill_action_provenance
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    attest_assumption,
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.manual_spot_entry import (
    ManualSpotDefaults,
    ManualSpotInput,
    parse_manual_spot_lines,
    parse_postflop_line,
    save_manual_spot,
    save_manual_spots,
    validate_manual_spot,
)
from poker_tracker.services.settlement_sync import (
    SettlementSyncRefused,
    sync_recorded_figures_from_ledger,
)
from poker_tracker.services.study_readiness import (
    BlockerCategory,
    StudyBlocker,
    StudyReadiness,
    accounting_is_established,
    evaluate_study_readiness,
    hand_requires_assumption_attestation,
    hand_requires_user_confirmation,
    is_reconstructed_hand,
    unattested_assumption_dependence,
)
from poker_tracker.services.validated_hand_import import (
    CV_TIMELINE_IDENTITY_KEY,
    autonomous_import_blockers,
    ensure_draft_for_review,
    ensure_hand_imported,
    find_existing_imported_hand,
    hand_frames_validated,
    import_all_autonomous_eligible,
    related_cv_job_ids,
)
from poker_tracker.solver.eligibility import prepare_solver_spot
from poker_tracker.solver.jobs import (
    SolverJobAlreadyRunningError,
    cancel_solver_run,
    reconcile_stale_solver_runs,
    start_solver_job,
)
from poker_tracker.solver.models import ResolvedRange, SolverEvidence
from poker_tracker.solver.profile_io import export_range_profiles, import_range_profiles
from poker_tracker.solver.ranges import (
    BUILTIN_RANGE_PROFILES,
    default_scenario,
    normalize_weighted_notation,
    resolve_custom_range,
    resolve_profile,
    resolve_selected_profile,
)
from poker_tracker.solver.storage import remove_solver_run_artifacts
from poker_tracker.solver.texassolver import (
    PINNED_CONSOLE_COMMIT,
    configured_binary,
    configured_resource_dir,
)
from poker_tracker.ui.auth import check_password, logout_button
from poker_tracker.ui.components import (
    coverage_bar,
    data_callout,
    empty_state,
    frequency_bars,
    kpi_card,
    page_header,
    panel,
    product_hero,
    section_header,
    section_header_with_meta,
    status_badge,
    trust_badge,
    workflow_step,
)
from poker_tracker.ui.cv_jobs import (
    CVJobAlreadyRunningError,
    cancel_processing_job,
    reconcile_stuck_jobs,
    start_cv_job,
)
from poker_tracker.ui.frame_extraction import (
    delete_extracted_frames,
    extract_frames_for_video,
    select_representative_frames,
)
from poker_tracker.ui.image_utils import image_dimensions, save_roi_crop_preview
from poker_tracker.ui.jobs import LOG_TAIL_LINES, describe_job_outcome, describe_job_outcomes
from poker_tracker.ui.navigation import Page, navigate_to, render_navigation
from poker_tracker.ui.poker_visuals import (
    action_replay_state,
    equity_meter_html,
    inject_poker_visual_styles,
    poker_table_html,
    range_cells_from_notation,
    range_matrix_html,
    render_action_timeline,
    render_poker_table,
)
from poker_tracker.ui.reconstruction_review import (
    ACTION_MAY_NOT_BELONG,
    ISSUE_GUIDANCE,
    MONEY_ACTION_TYPES,
    STACK_VALUE_KINDS,
    STREET_BY_BOARD_COUNT,
    ActionCvIssue,
    FrameIssueTarget,
    ValidationFrameContext,
    cv_issues_for_timeline_action,
    empty_hands_review_message,
    first_unreviewed_frame_index,
    frame_issue_targets,
    hand_frame_progress,
    hand_validation_label,
    history_impacts,
    job_id_from_hand_notes,
    load_timeline_for_job,
    match_db_action_to_frame_target,
    match_db_action_to_timeline_action,
    observed_facts,
    seat_holds_cards,
    seat_refusal_code,
    seat_value,
    states_for_hand,
    timeline_action_by_frame_and_seat,
    timeline_actions_for_image,
    timeline_path_for_job,
    timeline_source_image_for_slot,
    unknown_read_text,
)
from poker_tracker.ui.roi import ROI_TYPES, validate_roi_bounds
from poker_tracker.ui.roi_profiles import (
    create_starter_clubwpt_profile,
    duplicate_roi_profile,
    export_roi_profile,
    generate_roi_crop_previews,
    import_roi_profile,
)
from poker_tracker.ui.session_library import (
    date_session_name,
    filter_hands,
    filter_sessions,
    session_dates,
    sessions_on_date,
)
from poker_tracker.ui.ui_theme import brand_header, inject_theme
from poker_tracker.ui.video_ingest import ingest_uploaded_video
from poker_tracker.ui.video_storage import (
    ensure_data_directories,
    validate_video_extension,
)
from poker_tracker.ui.view_models import (
    build_job_rows,
    build_portfolio_summary,
    build_session_rows,
    completion_evidence_rows,
    confidence_label,
)

STREETS = ["preflop", "flop", "turn", "river", "showdown"]
ACTION_TYPES = [
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "all-in",
    "ante",
    "post_blind",
    "show",
    "win",
]
POSITIONS = ["", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
REVIEW_STATUSES = ["unreviewed", "reviewed", "needs_correction"]
STUDY_INCLUSION_OPTIONS: list[StudyInclusion] = ["auto", "study", "skip"]
STUDY_INCLUSION_LABELS = {
    "auto": "Auto (follow readiness)",
    "study": "Study hand",
    "skip": "Non-study hand",
}
TERMINAL_EVENT_OPTIONS = sorted(OBSERVED_TERMINAL_EVENTS)
COACHING_MODES = ["Theory + Exploit", "Theory Only", "Exploit Only", "Leak Finder"]
HAND_ISSUE_LABELS = {
    "hand_boundary": "Hand boundary / missing hand",
    "cards": "Cards",
    "players": "Players / seats",
    "stacks": "Stacks",
    "actions": "Actions",
    "pot_or_result": "Pot / result",
    "accounting": "Accounting",
    "coaching": "Coaching",
    "other": "Other",
}
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
    return calculator.calculate_equity_multiway(hero_cards, board_cards, list(villain_ranges))


def flash(message: str) -> None:
    """Queue a confirmation that survives the st.rerun() after a state change."""
    st.session_state["_flash"] = message


def show_flash() -> None:
    queued = st.session_state.pop("_flash", None)
    if queued:
        st.toast(queued)


BLOCKER_CATEGORY_LABELS: dict[BlockerCategory, str] = {
    "completion": "Completion",
    "cards": "Cards",
    "facts": "Stored facts",
    "layout": "Table layout",
    "accounting": "Accounting",
    "issues": "Debugging issues",
    "coaching": "Coaching evidence",
    "solver": "Solver evidence",
    "confirmation": "Your confirmation",
}


def study_confirmation_key(
    hand: Hand, accounting: AccountingReconciliation | None
) -> str:
    """Scope the never-persisted confirmation to the hand AND the evidence shown.

    The tick reads "I have read the evidence above and confirm this hand is
    correct", so it may not outlive that evidence. Keyed on the hand id alone it
    did: after a correction that changed the hero's stack, invalidated the
    settlement and staled the coaching, the box was still ticked, so
    USER_CONFIRMATION_MISSING never came back. The only case that did reset was an
    accident of widget garbage collection -- the acknowledge handler reruns before
    the checkbox is drawn -- which every control rendered below the checkbox
    missed. Changing the key retires the old widget's state instead.
    """
    return f"study_confirm_{hand.id}_{_study_evidence_digest(hand, accounting)}"


def _study_evidence_digest(
    hand: Hand, accounting: AccountingReconciliation | None
) -> str:
    """A short, stable digest of exactly the facts the confirmation attests to."""
    ledger = None if accounting is None else accounting.ledger
    material: list[object] = [
        hand.completion_status,
        json.dumps(hand.completion_evidence, sort_keys=True, default=str),
        hand.source_type,
        hand.hero_cards,
        hand.board_cards,
        hand.hero_position,
        hand.table_size,
        hand.pot_size,
        hand.hero_bb_won,
    ]
    if accounting is None:
        material.append("accounting-unavailable")
    else:
        material.extend(
            [
                accounting.is_authoritative,
                tuple(accounting.issues),
                tuple(
                    sorted(
                        (
                            entry.entry_type,
                            entry.pot_index,
                            entry.player_key or "",
                            entry.amount,
                        )
                        for entry in accounting.entries
                    )
                ),
            ]
        )
    if ledger is not None:
        material.extend(
            [
                ledger.gross_pot,
                ledger.rake,
                ledger.net_pot,
                tuple(sorted(ledger.net_results.items())),
            ]
        )
    return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()[:16]


def hand_study_readiness(
    db: PokerDatabase,
    hand: Hand,
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    *,
    user_confirmed: bool = False,
) -> StudyReadiness:
    """Fetch the evidence readiness composes for a surface that does not already hold it."""

    if hand.id is None:
        return evaluate_study_readiness(
            hand,
            accounting=accounting,
            accounting_error=accounting_error,
            user_confirmed=user_confirmed,
        )
    return evaluate_study_readiness(
        hand,
        accounting=accounting,
        accounting_error=accounting_error,
        hand_issues=db.fetch_hand_issues(hand_id=hand.id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand.id),
        # Legacy hand_reviews rows are staled by the same correction path and are
        # still rendered in the Hands workspace, so they are retained coaching
        # evidence too and must be able to block.
        hand_reviews=db.fetch_reviews_by_hand(hand.id),
        solver_runs=db.fetch_solver_runs_by_hand(hand.id),
        user_confirmed=user_confirmed,
    )


def _readiness_allows_reviewed(readiness: StudyReadiness) -> bool:
    return readiness.is_ready or (
        readiness.codes() == ("STUDY_EXCLUDED_BY_OPERATOR",)
    )


def approve_hand_for_study(
    db: PokerDatabase,
    hand: Hand,
    readiness: StudyReadiness,
    *,
    announce: bool = True,
) -> bool:
    """Promote one hand to reviewed through the same guard as every other surface.

    Caller must pass readiness evaluated with ``user_confirmed=True`` when the
    hand requires confirmation — finishing Import validation is that confirmation.
    """
    return guarded_update_hand_status(
        db, hand, readiness, "reviewed", announce=announce
    )


def _cv_timeline_identity(hand: Hand) -> tuple[int | None, int | None]:
    """Return ``(job_id, timeline_hand_number)`` from stamped evidence or notes."""
    evidence = parse_completion_evidence(hand.completion_evidence)
    identity = evidence.extra.get(CV_TIMELINE_IDENTITY_KEY)
    job_id: int | None = None
    timeline_hand_number: int | None = None
    if isinstance(identity, dict):
        try:
            job_id = int(identity["job_id"])
        except (KeyError, TypeError, ValueError):
            job_id = None
        try:
            timeline_hand_number = int(identity["timeline_hand_number"])
        except (KeyError, TypeError, ValueError):
            timeline_hand_number = None
    if job_id is None:
        job_id = job_id_from_hand_notes(hand.notes)
    if timeline_hand_number is None:
        from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
            timeline_hand_number_from_notes,
        )

        timeline_hand_number = timeline_hand_number_from_notes(hand.notes)
    if timeline_hand_number is None:
        timeline_hand_number = hand.hand_number
    return job_id, timeline_hand_number


def try_approve_hand_after_validation(
    db: PokerDatabase,
    hand: Hand,
    *,
    announce: bool = True,
) -> bool:
    """Confirm + promote after Import validation when readiness clears.

    Finishing validation (with or without edits) is user confirmation. Open
    HandIssue rows and other readiness blockers still refuse promotion.
    """
    if hand.id is None:
        return False
    if hand.review_status == "reviewed":
        return True
    accounting, accounting_error = _reconcile_cached(db, hand.id, None)
    readiness = hand_study_readiness(
        db,
        hand,
        accounting,
        accounting_error,
        user_confirmed=True,
    )
    return approve_hand_for_study(db, hand, readiness, announce=announce)


def _open_hand_for_validation(db: PokerDatabase, hand: Hand) -> None:
    """Deep-link Issues / library openers to Import frame validation for a hand."""
    if hand.session_id is None:
        _open_hand_for_study(hand)
        return
    job_id, timeline_hand_number = _cv_timeline_identity(hand)
    videos = db.fetch_videos(session_id=hand.session_id)
    preferred = videos[0] if videos else None
    if job_id is not None and videos:
        for video in videos:
            jobs = [
                item
                for item in db.fetch_jobs_by_video(video.id)
                if item.job_type == "cv_reconstruction" and item.id == job_id
            ]
            if jobs:
                preferred = video
                break
    _activate_session(hand.session_id)
    if preferred is not None and preferred.id is not None:
        st.session_state["video_context_id"] = preferred.id
        if job_id is not None:
            st.session_state[f"cv_review_job_{preferred.id}"] = job_id
            if timeline_hand_number is not None:
                st.session_state[f"evidence_hand_pending_{job_id}"] = timeline_hand_number
    navigate_to(Page.IMPORT)
    flash("Opened Import validation for this hand.")


def guarded_update_hand_status(
    db: PokerDatabase,
    hand: Hand,
    readiness: StudyReadiness,
    status: str,
    *,
    announce: bool = True,
) -> bool:
    """Single choke point for review-status writes; refuses to promote a blocked hand."""

    trust_ready = _readiness_allows_reviewed(readiness)
    if status == "reviewed" and not trust_ready:
        if announce:
            st.error(
                "This hand is not study-ready. Clear the blockers listed above before "
                "marking it reviewed."
            )
        return False
    if hand.id is None:
        if announce:
            st.error("This hand has not been saved yet.")
        return False
    try:
        db.update_hand_status(hand.id, status)
    except ValueError as exc:
        if announce:
            st.error(str(exc))
        return False
    return True


def review_status_options(hand: Hand, readiness: StudyReadiness) -> tuple[list[str], int]:
    """Offer 'reviewed' only when nothing blocks, and never re-add it as a fallback.

    A hand whose stored status is 'reviewed' while a blocker stands -- imported
    from an older database, hand-edited, or promoted before a later edit
    invalidated it -- used to have 'reviewed' re-appended purely because the
    stored value had to appear in the option list. The control then offered, and
    preselected, the one value the page's own blocker list said was false.

    Non-study preference alone does not block archival review: exclusion from
    coaching is not the same as "this hand's facts are untrustworthy."
    """
    trust_ready = _readiness_allows_reviewed(readiness)
    options = [
        item for item in REVIEW_STATUSES if item != "reviewed" or trust_ready
    ]
    if hand.review_status in options:
        return options, options.index(hand.review_status)
    st.caption(
        f"Stored status: {hand.review_status.replace('_', ' ')}. It is not offered "
        "while the blockers above stand."
    )
    return options, options.index("needs_correction")


def render_study_readiness(readiness: StudyReadiness) -> None:
    """Show every blocker grouped by category with the exact action that clears it.

    Deliberately renders no aggregate score or percentage: a single number would
    read as proof the whole hand is correct.
    """

    if readiness.is_ready:
        st.markdown(
            status_badge("reviewed", label="Study-ready · 0 blockers"),
            unsafe_allow_html=True,
        )
        st.caption(
            "Completion, cards, layout, accounting, issues, coaching, and solver "
            "evidence all pass."
        )
        return
    st.markdown(
        status_badge(
            "needs_correction",
            label=f"Not study-ready · {len(readiness.blockers)} blocker(s)",
        ),
        unsafe_allow_html=True,
    )
    for category, blockers in readiness.by_category().items():
        st.markdown(
            status_badge(
                "needs_correction",
                label=f"{BLOCKER_CATEGORY_LABELS[category]} · {len(blockers)} blocker(s)",
            ),
            unsafe_allow_html=True,
        )
        for blocker in blockers:
            st.markdown(f"**{blocker.reason}**")
            st.caption(f"Clears when: {blocker.clearing_action}")
            for item in blocker.detail:
                st.caption(f"· {item}")


def render_study_workflow(readiness: StudyReadiness) -> None:
    """Explain readiness as concrete checks and where to clear them."""

    with st.container(key="study_workflow_guide"):
        st.markdown("#### Replay, then analyze")
        if readiness.is_ready:
            st.caption("This hand passed every trust check and is ready to analyze.")
            return
        fix_groups = study_fix_groups(readiness)
        st.caption(
            f"{len(readiness.blockers)} trust check(s) are failing. Clear them on "
            f"Import validation ({len(fix_groups)} step group(s)), then return here "
            "to study. Replay still works; trusted analysis waits."
        )
        with st.expander(
            f"What needs fixing · {len(fix_groups)} step(s)",
            expanded=False,
        ):
            for index, (title, destination, blockers) in enumerate(fix_groups, start=1):
                st.markdown(f"**{index}. {title}**")
                for blocker in blockers:
                    st.write(f"• {blocker.reason}")
                st.caption(f"Open: {destination}")


def study_fix_groups(
    readiness: StudyReadiness,
) -> list[tuple[str, str, list[StudyBlocker]]]:
    """Group blockers that one user action commonly clears together."""

    definitions = {
        "study_preference": (
            "Choose study vs non-study",
            "Import validation → Edit this hand → Study inclusion "
            "(or Hands library inclusion)",
        ),
        "evidence": (
            "Verify the reconstructed hand",
            "Import validation → Edit this hand → Cards / Source warnings",
        ),
        "accounting": (
            "Reconcile the chips",
            "Import validation → Edit this hand → actions or Chip stacks",
        ),
        "issues": (
            "Resolve saved debugging issues",
            "Import validation → Edit this hand → Debugging issues "
            "(or Hands Issues inbox)",
        ),
        "coaching": (
            "Refresh coaching",
            "Analyze → AI coach",
        ),
        "solver": (
            "Refresh solver evidence",
            "Analyze → TexasSolver",
        ),
        "confirmation": (
            "Finish Import validation",
            "Import validation → Finish validation — send to Study",
        ),
    }
    evidence_codes = {
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
        "INVALID_HERO_OR_BOARD_CARDS",
        "UNREADABLE_HAND_COLUMNS",
        "UNSUPPORTED_TABLE_LAYOUT",
        "UNRESOLVED_SOURCE_WARNING",
    }
    grouped: dict[str, list[StudyBlocker]] = {}
    for blocker in readiness.blockers:
        key = (
            "evidence"
            if blocker.code in evidence_codes
            else blocker.category
        )
        grouped.setdefault(key, []).append(blocker)
    return [
        (definitions[key][0], definitions[key][1], blockers)
        for key, blockers in grouped.items()
    ]


def show_reconstruction_evidence(hand: Hand, evidence: CompletionEvidence) -> None:
    """Draw the evidence the confirmation checkbox asks the operator to attest to.

    The checkbox says "I have read the evidence above". Every one of these fields
    had a producer (the CV exporter), a persistence layer, an export format and a
    digest consumer, and no display consumer at all, so on a hand where the tick
    was the only remaining gate the entire content above it was the one sentence
    saying it had not been ticked. That made the final Phase 1 control a rubber
    stamp while its own label claimed otherwise.
    """

    if hand.id is None:
        return
    rows = completion_evidence_rows(evidence)
    with st.expander("Reconstruction evidence", expanded=False):
        st.caption(
            "Written by the reconstruction pipeline and never edited by hand. "
            "This is the evidence the confirmation below refers to."
        )
        for label, value in rows:
            st.markdown(f"**{label}** · {value}")


def show_source_warning_controls(
    db: PokerDatabase,
    hand: Hand,
    evidence: CompletionEvidence,
    *,
    force_open: bool = False,
) -> None:
    """Acknowledge pipeline source codes; the only user path from uncertain to complete."""

    # A measured settlement-assumption dependence is never offered here. It has
    # its own channel and its own control, which states the chip movement being
    # attested to; this panel's button says only "Acknowledge" and is captioned
    # as a pipeline note. `parse_completion_evidence` already keeps the two
    # channels apart on every read, so this filter is the second lock on the same
    # door rather than the only one.
    codes = [
        code
        for code in (*evidence.warning_codes, *evidence.rejection_codes)
        if not is_assumption_dependence_code(code)
    ]
    if hand.id is None or not codes:
        if force_open:
            st.caption("No source warnings on this hand.")
        return
    acknowledged = set(evidence.acknowledged_codes)
    rejections = set(evidence.rejection_codes)
    unresolved = len(evidence.unresolved_codes)
    with _study_panel(
        f"Source warnings · {unresolved} unresolved of {len(codes)}",
        force_open=force_open,
    ):
        st.caption(
            "Acknowledging records the accepted warning as an auditable correction. "
            "It can never make a partial recording complete, and it is not offered "
            "for a rejection code: a rejection is the pipeline refusing the hand, "
            "and only a new reconstruction can clear it."
        )
        for index, code in enumerate(codes):
            code_col, action_col = st.columns([2.2, 1])
            is_acknowledged = code in acknowledged
            is_rejection = code in rejections
            if is_rejection:
                label = f"{code} · rejected by the pipeline"
            else:
                label = f"{code} · {'acknowledged' if is_acknowledged else 'unresolved'}"
            code_col.markdown(
                status_badge(
                    "reviewed" if is_acknowledged and not is_rejection else "needs_correction",
                    label=label,
                ),
                unsafe_allow_html=True,
            )
            if is_rejection:
                action_col.caption("Re-run the reconstruction")
                continue
            if is_acknowledged:
                continue
            if action_col.button(
                "Acknowledge",
                key=f"study_ack_{hand.id}_{index}",
                width="stretch",
            ):
                updated = acknowledge_codes(evidence, [code])
                db.update_hand_completion(
                    hand.id,
                    completion_evidence=dump_completion_evidence(updated),
                    notes=f"Acknowledged source warning {code}.",
                )
                flash(f"Acknowledged source warning {code}.")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="PokerTrainer", layout="wide")
    inject_theme()
    inject_poker_visual_styles()
    if not check_password():
        return
    show_flash()

    db = get_database()
    reconciled = reconcile_stuck_jobs(db)
    if reconciled:
        st.toast(f"Recovered {len(reconciled)} interrupted processing job(s).")
    recovered_solver_runs = reconcile_stale_solver_runs(db)
    if recovered_solver_runs:
        st.toast(f"Recovered {len(recovered_solver_runs)} interrupted solver run(s).")
    sessions = db.fetch_sessions()

    with st.sidebar:
        brand_header()
        page = render_navigation()
        st.divider()
        trust_badge()
        selected_session = None
        if sessions:
            st.caption("SESSION CONTEXT")
            selected_session = select_session(db)
        with st.expander("New session", expanded=not sessions):
            create_session_form(db)
        logout_button()

    if page == Page.OVERVIEW:
        show_product_overview(db)
    elif page == Page.SESSIONS:
        show_sessions_workspace(db, selected_session)
    elif page == Page.HANDS:
        show_hands_workspace(db)
    elif page == Page.STUDY:
        show_study_workspace(db, selected_session)
    elif page == Page.INSIGHTS:
        show_insights_workspace(db)
    elif page == Page.IMPORT:
        show_import_workspace(db, selected_session)
    else:
        show_settings_workspace(db, selected_session)


AccountingCache = dict[int, tuple["AccountingReconciliation | None", str | None]]


def new_accounting_cache() -> AccountingCache:
    """One render pass's reconciliations, keyed by hand id.

    Deliberately created by the caller and never module-global. Every surface
    that uses one builds it, reads through it, and drops it inside a single
    render, so a settlement write in a form handler -- which is always followed
    by ``st.rerun()`` -- can never be answered from a cache the write invalidated.

    Insights and Study each reconciled the same hand twice per render: once
    through ``_hands_with_accounting_results`` to substitute the derived hero
    result into the list, and again through ``_accounting_or_error`` (Insights)
    or the Study page's own call. A reconciliation is now two ledger builds on
    a hand that declares a settlement policy, so paying for it twice is the cost
    this change would otherwise have added.
    """
    return {}


def _reconcile_cached(
    db: PokerDatabase, hand_id: int, cache: AccountingCache | None
) -> tuple[AccountingReconciliation | None, str | None]:
    if cache is not None and hand_id in cache:
        return cache[hand_id]
    try:
        entry: tuple[AccountingReconciliation | None, str | None] = (
            reconcile_persisted_hand(db, hand_id),
            None,
        )
    except LedgerError as exc:
        entry = (None, str(exc))
    if cache is not None:
        cache[hand_id] = entry
    return entry


def _hands_with_accounting_results(
    db: PokerDatabase, hands: list[Hand], cache: AccountingCache | None = None
) -> list[Hand]:
    resolved: list[Hand] = []
    for hand in hands:
        result = hand.hero_bb_won
        substituted = False
        if hand.id is not None:
            accounting, _ = _reconcile_cached(db, hand.id, cache)
            # Not `is_authoritative`. This substitution is where a derived figure
            # becomes the hand's result in every list, the Overview panel, the
            # portfolio summary and the Insights KPIs, so it is exactly the place
            # that must not publish a number an unanswered declaration produced.
            if _accounting_is_established(hand, accounting):
                players = db.fetch_players_by_hand(hand.id)
                hero = next((player for player in players if player.is_hero), None)
                if hero is not None:
                    result = accounting.ledger.net_results.get(hero.player_key, result)
                    substituted = result != hand.hero_bb_won
        # The copy is marked so a writer can refuse it. These objects are display
        # values -- a DERIVED hero result standing in for an observed one -- and
        # one of them reached 'Correct hand facts', where saving an unrelated
        # field persisted the derivation into `hands.hero_bb_won`.
        resolved.append(
            hand.model_copy(
                update={
                    "hero_bb_won": result,
                    "derived_result_substituted": substituted,
                }
            )
        )
    return resolved


def _accounting_or_error(
    db: PokerDatabase, hand: Hand, cache: AccountingCache | None = None
) -> tuple[AccountingReconciliation | None, str | None]:
    """Reconcile one hand for a surface that renders many, mirroring the Study page."""
    if hand.id is None:
        return None, None
    return _reconcile_cached(db, hand.id, cache)


def _format_persisted_hand_history(db: PokerDatabase, session: Session, hand: Hand) -> str:
    if hand.id is None:
        return format_hand_history(session, hand, [], [])
    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    try:
        accounting = reconcile_persisted_hand(db, hand.id)
        error = None
    except LedgerError as exc:
        accounting = None
        error = str(exc)
    return format_hand_history(
        session,
        hand,
        actions,
        players,
        ledger=None if accounting is None else accounting.ledger,
        accounting_issues=_accounting_prompt_issues(accounting, error),
        accounting_authoritative=_accounting_is_established(hand, accounting),
    )


def show_product_overview(db: PokerDatabase) -> None:
    sessions = db.fetch_sessions()
    hands_by_session = {
        session.id: _hands_with_accounting_results(db, db.fetch_hands_by_session(session.id))
        for session in sessions
        if session.id is not None
    }
    all_hands = [hand for hands in hands_by_session.values() for hand in hands]
    summary = build_portfolio_summary(all_hands, len(sessions))

    featured = all_hands[-1] if all_hands else None
    if featured is not None and featured.id is not None:
        featured_actions = db.fetch_actions_by_hand(featured.id)
        featured_players = db.fetch_players_by_hand(featured.id)
        try:
            featured_accounting = reconcile_persisted_hand(db, featured.id)
        except LedgerError:
            featured_accounting = None
        table_html = poker_table_html(
            hero_cards=featured.hero_cards,
            board_cards=featured.board_cards,
            pot_size=(
                featured_accounting.ledger.gross_pot
                if _accounting_is_established(featured, featured_accounting)
                else featured.pot_size
            ),
            players=featured_players,
            result_bb=featured.hero_bb_won,
            label=f"Completed hand #{featured.hand_number}",
        )
        hand_label = f"HAND #{featured.hand_number}"
    else:
        featured_actions = []
        featured_players = []
        featured_accounting = None
        table_html = poker_table_html(
            hero_cards="",
            board_cards="",
            pot_size=None,
            players=[],
            label="No completed hands recorded",
        )
        hand_label = "NO HAND"

    product_hero(
        "Turn completed hands into sharper decisions.",
        "Reconstruct sessions, replay decision paths, and study the math in one private poker analysis workspace.",
        table_html,
        proof_points=(
            (str(summary.hand_count), "saved hands"),
            # "marked reviewed", never a bare "reviewed": review_status is a
            # workflow label and this is the first number a user sees. Unqualified,
            # one percentage read as proof that every saved hand is correct, which
            # is exactly what the rest of the app is careful never to claim.
            (f"{summary.review_percent:.0f}%", "marked reviewed"),
            (hand_label, "replay surface"),
        ),
    )

    with st.container(key="overview_actions"):
        action_left, action_right, _ = st.columns([1, 1, 3])
        if action_left.button("Import completed session", type="primary", width="stretch"):
            navigate_to(Page.IMPORT)
            st.rerun()
        if action_right.button("Open hand library", width="stretch"):
            navigate_to(Page.HANDS)
            st.rerun()

    with st.container(key="overview_metrics"):
        columns = st.columns(4)
        with columns[0]:
            kpi_card("Sessions", str(summary.session_count), "Completed sessions on file")
        with columns[1]:
            kpi_card("Hands", str(summary.hand_count), "Across every saved session")
        with columns[2]:
            # A workflow label, never a readiness verdict: a hand can be marked
            # reviewed and still be blocked on accounting, issues, coaching, or
            # solver evidence. Insights carries the readiness count.
            kpi_card(
                "Review coverage",
                f"{summary.review_percent:.0f}%",
                f"{summary.reviewed_count} hands marked reviewed",
                tone="positive" if summary.review_percent >= 75 else "default",
            )
        with columns[3]:
            result_tone = (
                "positive"
                if summary.net_bb > 0
                else "negative"
                if summary.net_bb < 0
                else "default"
            )
            kpi_card(
                "Recorded result",
                f"{summary.net_bb:+g} BB",
                "Recorded hands only",
                tone=result_tone,
            )

    if featured_actions:
        section_header_with_meta(
            "Latest decision history",
            "Street-by-street action from the featured completed hand.",
            f"{len(featured_actions)} ACTIONS",
        )
        render_action_timeline(
            featured_actions,
            players=featured_players,
            effective_stack=featured.effective_stack,
            initial_pot=(
                featured_accounting.settlement.dead_money
                if featured_accounting is not None and featured_accounting.settlement is not None
                else None
            ),
            ledger=(None if featured_accounting is None else featured_accounting.ledger),
        )

    section_header_with_meta(
        "Recent sessions",
        "Newest completed sessions",
        f"{len(session_rows := build_session_rows(sessions, hands_by_session))} TOTAL",
    )
    if not session_rows:
        empty_state(
            "No sessions yet",
            "Create a session or import a completed-session video to begin your study library.",
        )
    else:
        st.dataframe(
            [
                {
                    "Session": row.name,
                    "Date": row.date_played,
                    "Platform": row.platform,
                    "Stakes": row.stakes,
                    "Hands": row.hand_count,
                    "Reviewed": f"{row.reviewed_count}/{row.hand_count}",
                    "Result": f"{row.net_bb:+g} BB",
                }
                for row in session_rows[:8]
            ],
            hide_index=True,
            width="stretch",
        )

    section_header("Processing", "Recent offline reconstruction activity")
    recent_jobs = db.fetch_recent_jobs(6)
    job_rows = build_job_rows(
        recent_jobs,
        db.fetch_videos(),
        outcomes=describe_job_outcomes(db, recent_jobs),
    )
    if not job_rows:
        empty_state("No processing jobs", "Uploaded video reconstruction jobs will appear here.")
    else:
        st.dataframe(
            [
                {
                    "Video": row.filename,
                    "Type": row.job_type,
                    "Status": row.status.replace("_", " ").title(),
                    "Progress": row.progress_label,
                    "Outcome": row.outcome_statement,
                    # The worker log is the only account of why a job stopped,
                    # so it travels with the outcome instead of living one page
                    # deeper where nobody knows to look for it.
                    "Log": row.outcome.log_path or "—",
                    "Created": row.age_label,
                }
                for row in job_rows
            ],
            hide_index=True,
            width="stretch",
        )


def show_sessions_workspace(db: PokerDatabase, session: Session | None) -> None:
    page_header(
        "Sessions",
        "Browse by date, keep multiple recordings together, and jump straight into any hand.",
    )
    sessions = db.fetch_sessions()
    show_session_library(db, sessions, session)
    if session is None:
        empty_state(
            "Create your first session",
            "Create one below. Date played defaults to today and can be changed later.",
        )
        create_session_form(db, form_key="create_first_session")
        return
    summary_tab, hands_tab, videos_tab, add_tab = st.tabs(
        ["Overview", "Hands", "Videos", "Add hands"]
    )
    with summary_tab:
        show_session_dashboard(db, session)
    with hands_tab:
        show_session_hand_browser(db, session)
    with videos_tab:
        show_session_videos(db, session)
    with add_tab:
        create_hand_form(db, session.id)


def show_hands_workspace(db: PokerDatabase) -> None:
    page_header(
        "Hand library",
        "Find high-impact and unresolved decisions across every completed session.",
    )
    sessions = db.fetch_sessions()
    hands = _hands_with_accounting_results(db, db.fetch_all_hands())
    if not hands:
        empty_state(
            "No hands to review", "Import a completed session or add a hand manually first."
        )
        return
    sessions_by_id = {session.id: session for session in sessions if session.id is not None}
    hands_by_id = {hand.id: hand for hand in hands if hand.id is not None}
    show_hand_issue_queue(
        db,
        db.fetch_hand_issues(status="open"),
        hands_by_id,
        sessions_by_id,
    )

    with st.container(key="hand_filters"):
        search_col, status_col, result_col = st.columns([2, 1, 1])
        query = search_col.text_input(
            "Find a hand",
            placeholder="Cards, date, session, stakes, position, tag…",
            key="hand_library_search",
        )
        review_status = status_col.segmented_control(
            "Review",
            options=["all", *REVIEW_STATUSES],
            default="all",
            format_func=lambda value: value.replace("_", " ").title(),
            key="hand_library_status",
        )
        result_filter = result_col.segmented_control(
            "Result",
            options=["all", "wins", "losses", "unknown"],
            default="all",
            format_func=str.title,
            key="hand_library_result",
        )
    filtered = filter_hands(
        hands,
        sessions_by_id,
        query=query,
        review_status=review_status or "all",
        result_filter=result_filter or "all",
    )
    if not filtered:
        empty_state("No matching hands", "Clear one or more filters to broaden the result set.")
        return
    render_hand_results(db, filtered, sessions_by_id, key_prefix="library")


def show_hand_issue_queue(
    db: PokerDatabase,
    issues: list[HandIssue],
    hands_by_id: dict[int, Hand],
    sessions_by_id: dict[int, Session],
) -> None:
    """Render the cross-session inbox an agent can inspect later."""

    with st.expander(f"Saved debugging issue queue ({len(issues)} open)", expanded=bool(issues)):
        if not issues:
            st.caption(
                "No unresolved hand issues. Flag one during Import validation."
            )
            return
        for issue in issues:
            hand = hands_by_id.get(issue.hand_id)
            if hand is None:
                continue
            session = sessions_by_id.get(hand.session_id)
            label = ", ".join(
                HAND_ISSUE_LABELS.get(issue_type, issue_type.replace("_", " ").title())
                for issue_type in issue.issue_types
            )
            summary, action = st.columns([6, 1])
            with summary:
                st.markdown(
                    f"**{session.name if session else 'Unknown session'} · "
                    f"Hand #{hand.hand_number}** — {label}"
                )
                st.caption(issue.description)
                st.caption(f"Saved {issue.created_at.isoformat()}")
            if action.button(
                "Open",
                key=f"issue_queue_open_{issue.id}",
                width="stretch",
            ):
                _open_hand_for_validation(db, hand)
                st.rerun()


def _sort_study_queue(hands: list[Hand]) -> list[Hand]:
    return sorted(
        (hand for hand in hands if hand.id is not None),
        key=lambda item: (
            0 if item.study_inclusion == "study" else 1,
            item.hand_number,
            item.id or 0,
        ),
    )


def _study_queue_for_session(hands: list[Hand]) -> list[Hand]:
    """Approved Study hands only — validation owns editing and promotion."""

    return _sort_study_queue(
        [
            hand
            for hand in hands
            if hand.id is not None
            and hand.study_inclusion != "skip"
            and hand.review_status == "reviewed"
        ]
    )


def load_study_session_hands(
    db: PokerDatabase,
    sessions: list[Session],
    preferred_session: Session | None,
    requested_hand_id: int | None,
) -> tuple[Session | None, list[Hand], Hand | None]:
    """Load one session's Study queue without reconciling every hand in the DB.

    Returns ``(hand_session, ordered_queue, forced_skip_hand)``. When the
    operator opened a non-study hand, ``forced_skip_hand`` is that row so
    inclusion can be changed without silently swapping hands.
    """

    if requested_hand_id is not None:
        requested_hand = db.fetch_hand(requested_hand_id)
        if requested_hand is not None and (
            requested_hand.study_inclusion == "skip"
            or requested_hand.review_status != "reviewed"
        ):
            hand_session = next(
                (item for item in sessions if item.id == requested_hand.session_id),
                None,
            )
            return hand_session, [requested_hand], requested_hand

    session_order: list[Session] = []
    if preferred_session is not None:
        session_order.append(preferred_session)
    session_order.extend(
        item
        for item in sessions
        if preferred_session is None or item.id != preferred_session.id
    )
    if requested_hand_id is not None:
        requested_hand = db.fetch_hand(requested_hand_id)
        if requested_hand is not None:
            match = next(
                (item for item in sessions if item.id == requested_hand.session_id),
                None,
            )
            if match is not None:
                session_order = [match] + [
                    item for item in session_order if item.id != match.id
                ]

    for candidate in session_order:
        if candidate.id is None:
            continue
        ordered = _study_queue_for_session(db.fetch_hands_by_session(candidate.id))
        if ordered:
            return candidate, ordered, None
    return preferred_session or (sessions[0] if sessions else None), [], None


def _batch_approve_candidates(ordered: list[Hand]) -> list[Hand]:
    return [
        hand
        for hand in ordered
        if hand.id is not None
        and hand.study_inclusion != "skip"
        and hand.review_status != "reviewed"
        and hand.completion_status in {"complete", "not_applicable"}
    ]


def _session_hands_ready_to_approve(
    db: PokerDatabase,
    candidates: list[Hand],
    cache: AccountingCache,
) -> list[tuple[Hand, StudyReadiness]]:
    """Evaluate batch candidates once; confirmation is implied by the approve action."""

    ready: list[tuple[Hand, StudyReadiness]] = []
    for hand in candidates:
        if hand.id is None:
            continue
        accounting, accounting_error = _reconcile_cached(db, hand.id, cache)
        readiness = hand_study_readiness(
            db,
            hand,
            accounting,
            accounting_error,
            user_confirmed=True,
        )
        if readiness.is_ready:
            ready.append((hand, readiness))
    return ready


def approve_ready_hands_in_session(
    db: PokerDatabase,
    candidates: list[Hand],
    cache: AccountingCache,
) -> tuple[int, int]:
    """Approve session hands that are study-ready with confirmation implied."""

    ready = _session_hands_ready_to_approve(db, candidates, cache)
    approved = 0
    for hand, readiness in ready:
        if approve_hand_for_study(db, hand, readiness, announce=False):
            approved += 1
    skipped = len(candidates) - approved
    return approved, skipped


def show_study_workspace(db: PokerDatabase, session: Session | None) -> None:
    page_header(
        "Study",
        "Replay approved hands and turn one completed decision into a reusable lesson.",
    )
    sessions = db.fetch_sessions()
    if not sessions:
        empty_state(
            "Nothing queued for study",
            "Validate hands on Import first — Study only shows approved hands.",
        )
        return
    accounting_cache = new_accounting_cache()
    requested = st.session_state.get("study_hand_id")
    hand_session, ordered, forced_hand = load_study_session_hands(
        db, sessions, session, requested if isinstance(requested, int) else None
    )
    if forced_hand is not None and forced_hand.study_inclusion == "skip":
        st.warning(
            f"Hand #{forced_hand.hand_number} is marked non-study. "
            "Change Study inclusion below to return it to the queue after approval."
        )
        show_study_inclusion_controls(db, forced_hand, force_open=True)
        return
    if forced_hand is not None and forced_hand.review_status != "reviewed":
        st.warning(
            f"Hand #{forced_hand.hand_number} is not approved for study yet. "
            "Finish editing and validation on Import — Study is study-only."
        )
        _offer_frame_validation_link(db, forced_hand)
        show_study_inclusion_controls(db, forced_hand, force_open=True)
        return
    if not ordered:
        empty_state(
            "No approved study hands queued",
            "Finish Import validation to send hands here, or pick another session. "
            "Hands with open issues stay in the Hands Issues inbox.",
        )
        return
    available_ids = {hand.id for hand in ordered if hand.id is not None}
    if requested not in available_ids:
        requested = ordered[0].id
        _set_study_hand_id(requested)
    hand = next(item for item in ordered if item.id == requested)
    if hand_session is None:
        hand_session = next(item for item in sessions if item.id == hand.session_id)

    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    # Reconcile only the open hand — not every hand in the database.
    accounting, accounting_error = _reconcile_cached(db, hand.id, accounting_cache)
    coaching_reviews = db.fetch_coaching_reviews_by_hand(hand.id)
    hand_issues = db.fetch_hand_issues(hand_id=hand.id)
    solver_runs = db.fetch_solver_runs_by_hand(hand.id)
    # Approved Study hands are already confirmed during Import validation.
    user_confirmed = True
    readiness = evaluate_study_readiness(
        hand,
        accounting=accounting,
        accounting_error=accounting_error,
        hand_issues=hand_issues,
        coaching_reviews=coaching_reviews,
        # Legacy hand_reviews rows are staled by the same correction path and are
        # blocking evidence too. Omitting them here made this page -- which feeds
        # review-status writers -- report "Study-ready · 0 blockers" on a hand
        # every other surface refused.
        hand_reviews=db.fetch_reviews_by_hand(hand.id),
        solver_runs=solver_runs,
        user_confirmed=user_confirmed,
    )

    render_study_workflow(readiness)
    if not readiness.is_ready:
        st.info(
            "This approved hand picked up new trust blockers. "
            "Return to Import validation to edit or resolve issues."
        )
        _offer_frame_validation_link(db, hand)
    render_study_hand_navigation(ordered, hand, hand_session)

    with st.container(key="study_workspace"):
        replay_tab, analyze_tab = st.tabs(["1 · Replay", "2 · Analyze"])
        with replay_tab:
            render_study_replay(
                hand_session,
                hand,
                actions,
                players,
                accounting,
                accounting_error,
                readiness,
            )
        with analyze_tab:
            render_study_analysis(
                db,
                hand_session,
                hand,
                actions,
                players,
                accounting,
                accounting_error,
                readiness,
                coaching_reviews,
            )


def study_hand_label(hand: Hand) -> str:
    result = "—" if hand.hero_bb_won is None else f"{hand.hero_bb_won:+g} BB"
    inclusion = STUDY_INCLUSION_LABELS.get(hand.study_inclusion, hand.study_inclusion)
    completion = hand.completion_status.replace("_", " ")
    return (
        f"Hand #{hand.hand_number} · {hand.hero_cards or 'Unknown cards'} · "
        f"{result} · {completion} · {inclusion}"
    )


def render_study_hand_navigation(
    ordered: list[Hand],
    hand: Hand,
    session: Session,
) -> None:
    """Keep hand selection in one compact row shared by every Study mode."""

    active_index = next(index for index, item in enumerate(ordered) if item.id == hand.id)
    hand_ids = [item.id for item in ordered if item.id is not None]
    hands_by_id = {item.id: item for item in ordered if item.id is not None}
    # Never force ``study_hand_picker`` from ``hand.id`` here. Streamlit writes the
    # operator's dropdown choice into that key before this function runs; clobbering
    # it snaps the selection back and blocks ``study_hand_id`` from updating.
    # Arrows / external openers keep both keys in sync via ``_set_study_hand_id``.
    if "study_hand_picker" not in st.session_state:
        st.session_state["study_hand_picker"] = hand.id
    with st.container(key="study_hand_navigation"):
        previous_col, chooser_col, next_col = st.columns([1.1, 2.2, 1.1])
        previous_col.button(
            "← Previous hand",
            key="study_previous_hand",
            disabled=active_index == 0,
            help="Go to the previous hand in this session",
            width="stretch",
            on_click=_set_study_hand_id,
            args=(ordered[active_index - 1].id if active_index > 0 else hand.id,),
        )
        selected_id = chooser_col.selectbox(
            "Choose a completed hand",
            hand_ids,
            index=active_index,
            format_func=lambda hand_id: study_hand_label(hands_by_id[hand_id]),
            key="study_hand_picker",
        )
        if selected_id != hand.id:
            st.session_state["study_hand_id"] = selected_id
            st.rerun()
        next_col.button(
            "Next hand →",
            key="study_next_hand",
            disabled=active_index >= len(ordered) - 1,
            help="Go to the next hand in this session",
            width="stretch",
            type="primary",
            on_click=_set_study_hand_id,
            args=(
                ordered[active_index + 1].id
                if active_index < len(ordered) - 1
                else hand.id,
            ),
        )
        st.caption(
            f"Hand {active_index + 1} of {len(ordered)} · {session.name} · "
            f"{hand.source_type.replace('_', ' ').title()} · "
            f"{hand.completion_status.replace('_', ' ').title()}"
        )
        st.caption(
            f"Reconstruction confidence · {confidence_label(hand.confidence_score)}"
        )


def _set_study_hand_id(hand_id: int | None) -> None:
    """Arrow callbacks update both the page hand and the stable picker widget."""
    if hand_id is None:
        return
    st.session_state["study_hand_id"] = hand_id
    st.session_state["study_hand_picker"] = hand_id


def render_study_replay(
    session: Session,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    readiness: StudyReadiness,
) -> None:
    """Show only the completed-hand replay and its recorded decisions."""

    replay_key = f"study_replay_action_{hand.id}"
    selected_index = st.session_state.get(replay_key)
    if not isinstance(selected_index, int) or not 0 <= selected_index < len(actions):
        selected_index = None
    initial_pot = (
        accounting.settlement.dead_money
        if accounting is not None and accounting.settlement is not None
        else None
    )
    replay_state = (
        action_replay_state(
            actions,
            selected_index,
            players=players,
            board_cards=hand.board_cards,
            initial_pot=initial_pot,
            ledger=None if accounting is None else accounting.ledger,
        )
        if selected_index is not None
        else None
    )
    if actions:
        previous_col, current_col, next_col, final_col = st.columns(
            [0.65, 2.7, 0.65, 0.9]
        )
        if previous_col.button(
            "←",
            key=f"study_replay_previous_{hand.id}",
            disabled=selected_index == 0,
            help="Previous action",
            width="stretch",
        ):
            st.session_state[replay_key] = (
                len(actions) - 1 if selected_index is None else selected_index - 1
            )
            st.rerun()
        current_col.markdown(
            "**Final hand**"
            if selected_index is None
            else f"**{study_action_label(actions[selected_index], selected_index)}**"
        )
        if next_col.button(
            "→",
            key=f"study_replay_next_{hand.id}",
            disabled=selected_index is None,
            help="Next action",
            width="stretch",
        ):
            st.session_state[replay_key] = (
                None if selected_index == len(actions) - 1 else selected_index + 1
            )
            st.rerun()
        if final_col.button(
            "Final hand",
            key=f"study_replay_final_{hand.id}",
            disabled=selected_index is None,
            width="stretch",
        ):
            st.session_state[replay_key] = None
            st.rerun()

    table_col, summary_col = st.columns([1.7, 0.75], gap="large")
    with table_col:
        section_header_with_meta(
            f"Hand #{hand.hand_number}",
            (
                "Completed-hand replay"
                if selected_index is None
                else f"Table immediately after action {selected_index + 1}"
            ),
            (
                hand.game_type.upper()
                if selected_index is None and hand.game_type
                else "FINAL HAND"
                if selected_index is None
                else f"ACTION {selected_index + 1} OF {len(actions)}"
            ),
        )
        render_poker_table(
            hero_cards=hand.hero_cards,
            board_cards=(
                hand.board_cards if replay_state is None else replay_state.board_cards
            ),
            pot_size=(
                replay_state.pot_size
                if replay_state is not None
                else accounting.ledger.gross_pot
                if _accounting_is_established(hand, accounting)
                else hand.pot_size
            ),
            players=players if replay_state is None else replay_state.players,
            result_bb=(
                _hero_ledger_result(hand, accounting, players, hand.hero_bb_won)
                if replay_state is None
                else None
            ),
            label=(
                f"{session.name} · {hand.hero_position or 'Position not recorded'}"
                if selected_index is None
                else study_action_label(actions[selected_index], selected_index)
            ),
            actor_player_key=(
                None if replay_state is None else replay_state.actor_player_key
            ),
            folded_player_keys=(
                frozenset() if replay_state is None else replay_state.folded_player_keys
            ),
        )
    with summary_col:
        st.markdown("#### Hand snapshot")
        data_callout(
            "View",
            "Final hand"
            if selected_index is None
            else f"After action {selected_index + 1}",
        )
        data_callout(
            "Hero position",
            hand.hero_position or "Not recorded",
        )
        data_callout(
            "Pot",
            (
                f"{replay_state.pot_size:g} BB · replay"
                if replay_state is not None and replay_state.pot_size is not None
                else "—"
                if replay_state is not None
                else
                f"{accounting.ledger.gross_pot:g} BB · reconciled"
                if _accounting_is_established(hand, accounting)
                else "—"
                if hand.pot_size is None
                else f"{hand.pot_size:g} BB · observed"
            ),
        )
        if readiness.is_ready:
            st.success("The saved hand is ready for analysis.")
        else:
            st.warning(
                "Replay is available. Trusted analysis is paused until the "
                "checklist above is cleared."
            )
        st.caption("Next: open Analyze when you want post-session tools.")

    section_header_with_meta(
        "Decision history",
        "Choose one action per row to update the table above.",
        f"{len(actions)} ACTIONS",
    )
    if actions:
        st.radio(
            "Replay action",
            options=list(range(len(actions))),
            index=None,
            format_func=lambda index: study_action_label(actions[index], index),
            key=replay_key,
            label_visibility="collapsed",
            width="stretch",
        )
        st.caption(
            "The gold-outlined seat acted. Dimmed seats had already folded. "
            "Choose Final hand to return to the completed result."
        )
        with st.expander(
            "Full action details · pot, stack, SPR, and notes",
            expanded=True,
        ):
            render_action_timeline(
                actions,
                players=players,
                effective_stack=hand.effective_stack,
                initial_pot=initial_pot,
                ledger=None if accounting is None else accounting.ledger,
            )
    else:
        render_action_timeline(actions)
    with st.expander("Show raw hand history"):
        st.code(
            format_hand_history(
                session,
                hand,
                actions,
                players,
                ledger=None if accounting is None else accounting.ledger,
                accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
                accounting_authoritative=_accounting_is_established(hand, accounting),
            ),
            language="text",
        )


def study_action_label(
    action: Action,
    index: int,
    *,
    issue_badge: str = "",
) -> str:
    """Return a compact but complete label for an action-replay control."""

    actor = actor_label(action.player_name, action.position) or "Unknown player"
    amount = "" if action.amount is None else f" · {action.amount:g} BB"
    badge = f" · ⚑ {issue_badge}" if issue_badge else ""
    return (
        f"{index + 1:02d} · {action.street.title()} · {actor} · "
        f"{action.action_type.replace('-', ' ').replace('_', ' ').title()}"
        f"{amount}{badge}"
    )


_BLOCKER_OTHER_FIX_LABELS: dict[str, str] = {
    "STUDY_EXCLUDED_BY_OPERATOR": "Study inclusion",
    "COMPLETION_NOT_COMPLETE": "Finalize incomplete hand",
    "UNRESOLVED_SOURCE_WARNING": "Source warnings",
    "INVALID_HERO_OR_BOARD_CARDS": "Cards, board, or pot",
    "UNREADABLE_HAND_COLUMNS": "Cards, board, or pot",
    "UNSUPPORTED_TABLE_LAYOUT": "Cards, board, or pot",
    "ACCOUNTING_NOT_AUTHORITATIVE": "Chip stacks / accounting",
    "ACCOUNTING_ASSUMPTION_DEPENDENT": "Chip stacks / accounting",
}


def _validation_focus_action_key(hand_id: int) -> str:
    return f"validation_focus_action_{hand_id}"


def _validation_other_fix_key(hand_id: int) -> str:
    return f"validation_fc_tool_{hand_id}"


def _jump_to_frame(frame_context: ValidationFrameContext, frame_index: int) -> None:
    st.session_state[frame_context.pending_hand_key] = frame_context.hand_number
    st.session_state[frame_context.cursor_key] = frame_index
    states = frame_context.states
    if 0 <= frame_index < len(states):
        when = float(states[frame_index].get("time_s", 0))
        flash(
            f"Showing frame {frame_index + 1} @ {when:.2f}s — scroll up to the "
            "frame evidence panel."
        )


def _validation_other_fixes_expander_key(hand_id: int) -> str:
    return f"validation_other_fixes_expander_{hand_id}"


def _validation_action_expander_key(hand_id: int, action_id: int) -> str:
    return f"validation_action_expander_{hand_id}_{action_id}"


def _jump_to_other_fix(hand_id: int, tool_label: str) -> None:
    st.session_state[_validation_other_fix_key(hand_id)] = tool_label
    st.session_state[_validation_other_fixes_expander_key(hand_id)] = True


def _jump_to_action(hand_id: int, action_id: int) -> None:
    # One-shot: open the expander once; do not keep re-forcing it open on reruns.
    st.session_state[_validation_focus_action_key(hand_id)] = action_id
    st.session_state[_validation_action_expander_key(hand_id, action_id)] = True


def _study_other_fix_options(
    readiness: StudyReadiness,
    *,
    is_reconstructed: bool,
    hand: Hand,
    hand_issues: list[HandIssue],
) -> list[tuple[str, str]]:
    """Return non-action Fix tools. Actions are always edited on the main Fix surface."""

    suggested: list[tuple[str, str]] = []
    codes = set(readiness.codes())
    if "STUDY_EXCLUDED_BY_OPERATOR" in codes:
        suggested.append(("inclusion", "Study inclusion"))
    if {
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
    } & codes:
        suggested.append(("finalize", "Finalize incomplete hand"))
    if "UNRESOLVED_SOURCE_WARNING" in codes:
        suggested.append(("warnings", "Source warnings"))
    if {
        "INVALID_HERO_OR_BOARD_CARDS",
        "UNREADABLE_HAND_COLUMNS",
        "UNSUPPORTED_TABLE_LAYOUT",
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
    } & codes:
        suggested.append(("facts", "Cards, board, or pot"))
    if {
        "ACCOUNTING_NOT_AUTHORITATIVE",
        "ACCOUNTING_ASSUMPTION_DEPENDENT",
    } & codes:
        suggested.append(("accounting", "Chip stacks / accounting"))
    if "OPEN_DEBUGGING_ISSUE" in codes or hand_issues:
        suggested.append(("issues", "Debugging issues"))

    always: list[tuple[str, str]] = [
        ("facts", "Cards, board, or pot"),
        ("players", "Players / seats"),
        ("accounting", "Chip stacks / accounting"),
        ("inclusion", "Study inclusion"),
    ]
    if is_reconstructed:
        always.extend(
            [
                ("warnings", "Source warnings"),
                ("finalize", "Finalize incomplete hand"),
                ("frames", "Jump to frame validation"),
            ]
        )
    always.extend(
        [
            ("issues", "Debugging issues"),
            ("history", "Correction history"),
        ]
    )
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for tool_id, label in [*suggested, *always]:
        if tool_id in seen:
            continue
        if tool_id == "finalize" and hand.completion_status not in {
            "partial",
            "uncertain",
        }:
            continue
        seen.add(tool_id)
        ordered.append((tool_id, label))
    return ordered


def _other_fixes_should_expand(readiness: StudyReadiness) -> bool:
    """Open Other fixes when blockers are not action-line edits."""

    return bool(
        {
            "STUDY_EXCLUDED_BY_OPERATOR",
            "COMPLETION_NOT_COMPLETE",
            "COMPLETION_EVIDENCE_MISSING",
            "INVALID_HERO_OR_BOARD_CARDS",
            "UNREADABLE_HAND_COLUMNS",
            "UNSUPPORTED_TABLE_LAYOUT",
            "ACCOUNTING_NOT_AUTHORITATIVE",
            "ACCOUNTING_ASSUMPTION_DEPENDENT",
            "OPEN_DEBUGGING_ISSUE",
            "UNRESOLVED_SOURCE_WARNING",
            "STALE_COACHING_EVIDENCE",
            "STALE_SOLVER_EVIDENCE",
        }
        & set(readiness.codes())
    )


def _render_study_fix_tool(
    tool_id: str,
    db: PokerDatabase,
    hand: Hand,
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    hand_issues: list[HandIssue],
    is_reconstructed: bool,
    completion_evidence: CompletionEvidence,
) -> None:
    """Render one Other-fixes tool. Actions are edited on the main Fix surface."""

    if tool_id == "inclusion":
        show_study_inclusion_controls(db, hand, force_open=True)
    elif tool_id == "finalize":
        show_finalize_incomplete_hand(db, hand, completion_evidence, force_open=True)
    elif tool_id == "warnings":
        show_source_warning_controls(
            db, hand, completion_evidence, force_open=True
        )
    elif tool_id == "facts":
        show_hand_fact_editor(db, hand, force_open=True)
    elif tool_id == "players":
        show_player_editor(db, players, force_open=True)
    elif tool_id == "accounting":
        _render_accounting_status(accounting, accounting_error)
        show_accounting_editor(
            db,
            hand,
            players,
            accounting,
            accounting_error,
            force_open=True,
        )
    elif tool_id == "issues":
        show_hand_issue_controls(db, hand, hand_issues, force_open=True)
    elif tool_id == "history":
        show_correction_history(db, hand.id, force_open=True)
    elif tool_id == "frames":
        if is_reconstructed:
            _offer_frame_validation_link(db, hand)
        else:
            st.caption("Frame validation is only available for reconstructed hands.")
    else:
        st.caption("Unknown fix tool.")


def owning_job_for_hand(hand: Hand) -> int | None:
    """Which reconstruction job this hand was imported from.

    Reads the stamped identity in preference to the notes: the stamp is what
    chose this hand for that job, while the notes are free text the operator
    can rewrite from this very panel.
    """

    return _cv_timeline_identity(hand)[0]


def frame_context_belongs_to_hand(
    hand: Hand,
    frame_context: ValidationFrameContext,
) -> bool:
    """Whether this job's frames may be used to explain this hand's rows."""

    owning = owning_job_for_hand(hand)
    return owning is None or owning == frame_context.job_id


def prepare_hand_frames(
    db: PokerDatabase,
    hand: Hand,
    frame_context: ValidationFrameContext | None,
) -> tuple[ValidationFrameContext | None, str]:
    """Settle which frames may explain this hand, and repair its provenance.

    Both steps together, and in this order, because the second writes to the
    database: backfilling from a foreign job's timeline records frames the
    hand never came from, and the write is one-shot, so it cannot be undone
    through the app. Kept out of the Streamlit function so the wiring itself
    is testable — deleting either step here must fail a test, not just
    deleting the helpers it calls.
    """

    usable, notice = usable_frame_context(hand, frame_context)
    if usable is not None and hand.id is not None:
        # Rows imported before schema 16 have no recorded source frame; fill
        # it now that the timeline is open.
        backfill_action_provenance(db, hand.id, usable.timeline_hand)
    return usable, notice


def usable_frame_context(
    hand: Hand,
    frame_context: ValidationFrameContext | None,
) -> tuple[ValidationFrameContext | None, str]:
    """Drop a frame context that belongs to a different reconstruction job.

    Several jobs can run on one video and all resolve to this same hand.
    Explaining its rows with another job's frames produces claims that are
    true of neither, so the panel says which job owns the hand and derives
    nothing. Returned rather than applied inline so the decision is testable.
    """

    if frame_context is None or frame_context_belongs_to_hand(hand, frame_context):
        return frame_context, ""
    owning = owning_job_for_hand(hand)
    return None, (
        f"This hand was imported from job {owning}, but you are viewing job "
        f"{frame_context.job_id}'s frames. Frame-based notes are hidden — "
        f"open job {owning} to validate it."
    )


def render_validation_edit_and_approve(
    db: PokerDatabase,
    hand: Hand,
    *,
    frames_validated: bool,
    frame_context: ValidationFrameContext | None = None,
) -> None:
    """Edit the session draft beside frame review; approve when validation finishes."""

    if hand.id is None:
        return
    # Always re-fetch so edits from this same render cycle are visible next rerun.
    hand = db.fetch_hand(hand.id) or hand
    frame_context, foreign_job_notice = prepare_hand_frames(db, hand, frame_context)
    if foreign_job_notice:
        st.warning(foreign_job_notice)
    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    accounting, accounting_error = _reconcile_cached(db, hand.id, None)
    hand_issues = db.fetch_hand_issues(hand_id=hand.id)
    completion_evidence = parse_completion_evidence(hand.completion_evidence)
    is_reconstructed = is_reconstructed_hand(hand)
    readiness = hand_study_readiness(
        db,
        hand,
        accounting,
        accounting_error,
        user_confirmed=True,
    )
    open_issues = [issue for issue in hand_issues if issue.status == "open"]
    issue_targets: list[FrameIssueTarget] = []
    if frame_context is not None:
        issue_targets = frame_issue_targets(
            frame_context.timeline_hand,
            frame_context.states,
            frame_context.reviews_by_image,
        )

    st.markdown("### Fix this hand")
    st.caption(
        "Jump a blocker to open its frame, action, or other-fix control. "
        "Finish with no open debugging issues to send to Study."
    )

    if open_issues:
        st.error(
            f"{len(open_issues)} open debugging issue(s) — held out of Study."
        )
    approved_ready = hand.review_status == "reviewed" and readiness.is_ready
    if approved_ready:
        st.success("Approved for Study. Open Study to replay and analyze.")
        if not frames_validated and issue_targets:
            flagged_n = sum(1 for t in issue_targets if t.status == "incorrect")
            unreviewed_n = sum(1 for t in issue_targets if t.status == "unreviewed")
            parts = []
            if flagged_n:
                parts.append(f"{flagged_n} flagged")
            if unreviewed_n:
                parts.append(f"{unreviewed_n} unreviewed")
            st.caption(
                (" / ".join(parts) if parts else f"{len(issue_targets)} frame(s)")
                + " — optional cleanup, not a Study blocker."
            )
    else:
        if hand.review_status == "reviewed" and not readiness.is_ready:
            st.warning(
                "Marked reviewed, but new trust blockers appeared. Clear them below."
            )
        _render_validation_fix_targets(
            hand_id=hand.id,
            readiness=readiness,
            frames_validated=frames_validated,
            issue_targets=issue_targets,
            frame_context=frame_context,
            actions=actions,
            open_issue_count=len(open_issues),
            other_tool_labels={
                label
                for _tool_id, label in _study_other_fix_options(
                    readiness,
                    is_reconstructed=is_reconstructed,
                    hand=hand,
                    hand_issues=hand_issues,
                )
                if _tool_id not in {"frames", "issues"}
            },
        )

    show_action_editor(
        db,
        actions,
        players,
        force_open=True,
        issue_targets=issue_targets,
        frame_context=frame_context,
        hand_id=hand.id,
    )

    # Issues are hosted once below (not also under Other fixes) to avoid duplicate forms.
    other_tools = [
        (tool_id, label)
        for tool_id, label in _study_other_fix_options(
            readiness,
            is_reconstructed=is_reconstructed,
            hand=hand,
            hand_issues=hand_issues,
        )
        if tool_id not in {"frames", "issues"}
    ]
    other_expander_key = _validation_other_fixes_expander_key(hand.id)
    if other_expander_key not in st.session_state:
        st.session_state[other_expander_key] = _other_fixes_should_expand(readiness)
    with st.expander(
        "Other fixes (cards, players, chips, finalize…)",
        key=other_expander_key,
        on_change="rerun",
    ):
        tool_labels = [label for _tool_id, label in other_tools]
        label_to_tool = {label: tool_id for tool_id, label in other_tools}
        tool_key = _validation_other_fix_key(hand.id)
        if st.session_state.get(tool_key) not in tool_labels:
            st.session_state[tool_key] = tool_labels[0]
        selected_label = st.selectbox(
            "What else needs fixing?",
            tool_labels,
            key=tool_key,
        )
        selected_tool = label_to_tool[selected_label]
        with st.container(border=True):
            _render_study_fix_tool(
                selected_tool,
                db,
                hand,
                players,
                accounting,
                accounting_error,
                hand_issues,
                is_reconstructed,
                completion_evidence,
            )

    with st.expander(
        f"Hold out of Study · {len(open_issues)} open",
        expanded=bool(open_issues),
    ):
        show_hand_issue_controls(db, hand, hand_issues, force_open=True)

    auto_key = f"validation_auto_approve_attempted_{hand.id}"
    if (
        frames_validated
        and not open_issues
        and hand.review_status != "reviewed"
        and readiness.is_ready
        and not st.session_state.get(auto_key)
    ):
        st.session_state[auto_key] = True
        if try_approve_hand_after_validation(db, hand, announce=False):
            flash(
                f"Hand #{hand.hand_number} validated and approved for Study."
            )
            st.rerun()

    if hand.review_status == "reviewed":
        return
    if open_issues:
        st.caption(
            "Resolve open issues or leave them for the Issues inbox — "
            "this hand stays out of Study while they remain open."
        )
        return
    finish_help = (
        "Frames are all Correct. Send to Study if the edited hand looks right."
        if frames_validated
        else (
            "You can send to Study after clearing trust checks above — "
            "frame labels can stay incomplete."
        )
    )
    st.caption(finish_help)
    if st.button(
        "Finish validation — send to Study",
        key=f"validation_finish_approve_{hand.id}",
        type="primary",
        width="stretch",
    ):
        if try_approve_hand_after_validation(db, hand):
            flash(f"Hand #{hand.hand_number} approved for Study.")
            st.rerun()


def _render_validation_fix_targets(
    *,
    hand_id: int,
    readiness: StudyReadiness,
    frames_validated: bool,
    issue_targets: list[FrameIssueTarget],
    frame_context: ValidationFrameContext | None,
    actions: list[Action],
    open_issue_count: int,
    other_tool_labels: set[str] | None = None,
) -> None:
    """Compact, clickable list of what blocks Study and where to fix it."""

    flagged = [target for target in issue_targets if target.status == "incorrect"]
    unreviewed = [target for target in issue_targets if target.status == "unreviewed"]
    available_tools = other_tool_labels or set()
    blocker_count = len(readiness.blockers)
    # Frame labels help editing but do not gate Study approval.
    if blocker_count == 0 and open_issue_count == 0:
        st.success("No Study blockers. Finish validation when the hand looks right.")
        if not frames_validated and (flagged or unreviewed):
            st.caption(
                f"{len(flagged)} flagged / {len(unreviewed)} unreviewed frame(s) — "
                "optional cleanup; open a row below to jump there."
            )
            with st.expander("Frame cleanup shortcuts", expanded=bool(flagged)):
                for target in flagged:
                    _render_frame_fix_row(
                        hand_id=hand_id,
                        target=target,
                        frame_context=frame_context,
                        actions=actions,
                        show_thumbnail=True,
                    )
                for target in unreviewed:
                    _render_frame_fix_row(
                        hand_id=hand_id,
                        target=target,
                        frame_context=frame_context,
                        actions=actions,
                        show_thumbnail=False,
                    )
        return

    title_bits = []
    if blocker_count:
        title_bits.append(f"{blocker_count} check(s)")
    if open_issue_count:
        title_bits.append(f"{open_issue_count} issue(s)")
    heading = "What's blocking Study · " + " · ".join(title_bits)
    # Markdown keeps AppTest/readiness assertions able to see the heading; the
    # expander label stays short so the title is not duplicated in the chrome.
    st.markdown(f"**{heading}**")
    with st.expander("Jump to the frame, action, or other fix", expanded=True):
        if not frames_validated and (flagged or unreviewed):
            st.markdown("**Frames (optional cleanup)**")
            for target in flagged:
                _render_frame_fix_row(
                    hand_id=hand_id,
                    target=target,
                    frame_context=frame_context,
                    actions=actions,
                    show_thumbnail=True,
                )
            if unreviewed:
                st.caption(f"{len(unreviewed)} unreviewed frame(s)")
                for target in unreviewed:
                    _render_frame_fix_row(
                        hand_id=hand_id,
                        target=target,
                        frame_context=frame_context,
                        actions=actions,
                        show_thumbnail=False,
                    )
        if readiness.blockers:
            for category, blockers in readiness.by_category().items():
                st.markdown(
                    status_badge(
                        "needs_correction",
                        label=(
                            f"{BLOCKER_CATEGORY_LABELS[category]} · "
                            f"{len(blockers)} blocker(s)"
                        ),
                    ),
                    unsafe_allow_html=True,
                )
                for blocker in blockers:
                    st.markdown(f"**{blocker.reason}**")
                    st.caption(f"Clears when: {blocker.clearing_action}")
                    for item in blocker.detail:
                        st.caption(f"· {item}")
                    tool_label = _BLOCKER_OTHER_FIX_LABELS.get(blocker.code)
                    cols = st.columns([1.2, 1.2, 2])
                    if tool_label and tool_label in available_tools:
                        if cols[0].button(
                            f"Open {tool_label}",
                            key=f"validation_jump_blocker_{hand_id}_{blocker.code}",
                            width="stretch",
                        ):
                            _jump_to_other_fix(hand_id, tool_label)
                            st.rerun()
                    elif tool_label:
                        cols[0].caption(f"{tool_label} is not available on this hand.")
                    if blocker.code == "ACCOUNTING_NOT_AUTHORITATIVE" and actions:
                        first = next((a for a in actions if a.id is not None), None)
                        if first is not None and cols[1].button(
                            "Edit first action",
                            key=f"validation_jump_acct_action_{hand_id}",
                            width="stretch",
                        ):
                            _jump_to_action(hand_id, first.id)
                            st.rerun()
                    if blocker.code == "OPEN_DEBUGGING_ISSUE":
                        cols[0].caption("Use Hold out of Study below.")
                    if blocker.code == "USER_CONFIRMATION_MISSING":
                        cols[0].caption("Use Finish validation at the bottom.")


def _linked_db_actions_for_frame(
    target: FrameIssueTarget, actions: list[Action]
) -> list[Action]:
    """All saved actions that match timeline lines on this frame."""

    linked: list[Action] = []
    for action in actions:
        if action.id is None:
            continue
        if (
            match_db_action_to_frame_target(
                street=action.street,
                action_type=action.action_type,
                player_name=action.player_name,
                position=action.position,
                amount=action.amount,
                targets=[target],
            )
            is not None
        ):
            linked.append(action)
    return linked


def _render_frame_fix_row(
    *,
    hand_id: int,
    target: FrameIssueTarget,
    frame_context: ValidationFrameContext | None,
    actions: list[Action],
    show_thumbnail: bool = True,
) -> None:
    """One flagged/unreviewed frame with jump controls and linked actions."""

    if show_thumbnail:
        thumb_col, body_col = st.columns([0.7, 2.3], gap="small")
        with thumb_col:
            if Path(target.source_image).is_file():
                st.image(target.source_image, width=120)
            else:
                st.caption(f"Frame {target.frame_index + 1}")
    else:
        body_col = st.container()
    with body_col:
        st.markdown(f"**{target.summary()}**")
        if target.notes:
            st.caption(target.notes)
        if target.action_labels():
            st.caption("Actions from this frame: " + " · ".join(target.action_labels()))
        else:
            st.caption("No action line attributed to this frame.")
        if frame_context is not None and st.button(
            f"Open frame {target.frame_index + 1}",
            key=(
                f"validation_jump_frame_{frame_context.job_id}_"
                f"{hand_id}_{target.frame_index}"
            ),
            width="stretch",
            type="primary" if target.status == "incorrect" else "secondary",
        ):
            _jump_to_frame(frame_context, target.frame_index)
            st.rerun()
        linked_actions = _linked_db_actions_for_frame(target, actions)
        for action in linked_actions:
            label = study_action_label(action, max(0, (action.action_index or 1) - 1))
            if st.button(
                f"Edit · {label}",
                key=(
                    f"validation_jump_frame_action_{hand_id}_"
                    f"{target.frame_index}_{action.id}"
                ),
                width="stretch",
            ):
                _jump_to_action(hand_id, action.id)
                st.rerun()


def render_study_analysis(
    db: PokerDatabase,
    session: Session,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    readiness: StudyReadiness,
    coaching_reviews,
) -> None:
    """Group analysis tools separately from reconstruction and correction."""

    st.markdown("### Analyze")
    if readiness.is_ready:
        st.success("This hand is confirmed and ready for post-session analysis.")
    else:
        st.warning(
            f"Analysis is limited while {len(readiness.blockers)} trust check(s) "
            "remain. Return to Import validation to edit or resolve issues."
        )

    math_tab, solver_tab, coach_tab, notes_tab = st.tabs(
        ["Quick math", "TexasSolver", "AI coach", "Notes"]
    )
    with math_tab:
        st.caption(
            "Fast calculations from the saved ledger only—no live or current-hand "
            "recommendations."
        )
        call_snapshots = (
            [
                snapshot
                for snapshot in accounting.ledger.snapshots
                if snapshot.kind in {"call", "all-in"} and snapshot.call_increment > 0
            ]
            if accounting is not None and accounting.ledger.is_legal
            else []
        )
        if call_snapshots:
            decision = call_snapshots[-1]
            required = required_equity_to_call(
                decision.call_increment, decision.pot_before
            )
            st.markdown(
                equity_meter_html(required, label="Required equity"),
                unsafe_allow_html=True,
            )
            st.caption(
                f"Calling {decision.call_increment:g} BB into a "
                f"{decision.pot_before:g} BB pot from the reconciled action ledger."
            )
        else:
            empty_state(
                "No call math available",
                accounting_error
                or "Reconcile a legal call action to calculate required equity.",
            )
    with solver_tab:
        show_solver_review(
            db,
            session,
            hand,
            actions,
            players,
            accounting,
            accounting_error,
            readiness,
        )
    with coach_tab:
        show_study_coach_review(
            db,
            session,
            hand,
            actions,
            players,
            accounting,
            accounting_error,
            coaching_reviews,
            readiness,
        )
    with notes_tab:
        st.markdown("##### Hand notes")
        st.write(hand.notes or "No notes recorded.")
        st.caption(
            "Edit notes on Import validation → Edit this hand → Cards, board, or pot."
        )


@contextmanager
def _study_panel(
    title: str,
    *,
    force_open: bool = False,
    expanded: bool = False,
) -> Iterator[None]:
    """Expander when browsing; plain panel when Fix already chose this tool."""

    if force_open:
        st.markdown(f"##### {title}")
        yield
        return
    with st.expander(title, expanded=expanded):
        yield


def show_hand_issue_controls(
    db: PokerDatabase,
    hand: Hand,
    issues: list[HandIssue],
    *,
    force_open: bool = False,
) -> None:
    """Flag a hand now and leave a self-contained report for future debugging."""

    if hand.id is None:
        return
    open_issues = [issue for issue in issues if issue.status == "open"]
    if open_issues:
        st.error(f"This hand has {len(open_issues)} unresolved debugging issue(s).")

    with _study_panel(
        "Flag this hand for future debugging",
        force_open=force_open,
        expanded=False,
    ):
        st.caption(
            "This does not require you to diagnose or fix it now. The current hand, "
            "players, actions, and correction history are saved as a snapshot."
        )
        with st.form(f"flag_hand_issue_{hand.id}"):
            issue_types = st.multiselect(
                "What looks wrong?",
                list(HAND_ISSUE_LABELS),
                format_func=lambda value: HAND_ISSUE_LABELS[value],
            )
            description = st.text_area(
                "What did you notice?",
                placeholder="Example: the third completed hand is missing from the import.",
            )
            submitted = st.form_submit_button("Save issue for later")
        if submitted:
            try:
                db.create_hand_issue(
                    HandIssue(
                        hand_id=hand.id,
                        issue_types=issue_types,
                        description=description.strip(),
                    )
                )
            except (ValidationError, ValueError) as exc:
                st.error(f"Could not save issue: {exc}")
            else:
                flash("Issue saved to the debugging queue.")
                st.rerun()

    if not issues:
        return
    st.markdown("##### Saved debugging reports")
    for issue in issues:
        label = ", ".join(
            HAND_ISSUE_LABELS.get(issue_type, issue_type.replace("_", " ").title())
            for issue_type in issue.issue_types
        )
        with st.container(border=True, key=f"study_issue_{issue.id}"):
            state = "Open" if issue.status == "open" else "Resolved"
            st.markdown(f"**{state} · {label}**")
            st.write(issue.description)
            st.caption(f"Saved {issue.created_at.isoformat()}")
            st.download_button(
                "Download debug snapshot",
                data=json.dumps(issue.model_dump(mode="json"), indent=2),
                file_name=f"hand_{hand.hand_number}_issue_{issue.id}.json",
                mime="application/json",
                key=f"download_hand_issue_{issue.id}",
            )
            if issue.status == "resolved":
                st.success(issue.resolution_notes or "Resolved.")
                continue
            with st.form(f"resolve_hand_issue_{issue.id}"):
                resolution_notes = st.text_input(
                    "Resolution notes",
                    placeholder="What was fixed or why this is no longer an issue?",
                )
                resolved = st.form_submit_button("Mark issue resolved")
            if resolved:
                try:
                    db.resolve_hand_issue(
                        issue.id,
                        resolution_notes=resolution_notes,
                    )
                except ValueError as exc:
                    st.error(f"Could not resolve issue: {exc}")
                else:
                    flash("Debugging issue resolved.")
                    st.rerun()


def show_study_inclusion_controls(
    db: PokerDatabase,
    hand: Hand,
    *,
    force_open: bool = False,
) -> None:
    """Let the operator mark any hand as study or non-study."""
    if hand.id is None:
        return
    with _study_panel(
        "Study inclusion",
        force_open=force_open,
        expanded=hand.study_inclusion != "auto",
    ):
        st.caption(
            "Study keeps this hand in the study queue. Non-study excludes it from "
            "coaching. Auto follows the usual readiness checks."
        )
        current = hand.study_inclusion if hand.study_inclusion in STUDY_INCLUSION_OPTIONS else "auto"
        choice = st.radio(
            "Include in study?",
            STUDY_INCLUSION_OPTIONS,
            index=STUDY_INCLUSION_OPTIONS.index(current),
            format_func=lambda value: STUDY_INCLUSION_LABELS[value],
            key=f"study_inclusion_{hand.id}",
            horizontal=True,
        )
        if st.button("Save study inclusion", key=f"save_study_inclusion_{hand.id}"):
            try:
                db.update_study_inclusion(hand.id, choice)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(f"Study inclusion set to {STUDY_INCLUSION_LABELS[choice]}.")
                st.rerun()


def show_finalize_incomplete_hand(
    db: PokerDatabase,
    hand: Hand,
    evidence: CompletionEvidence,
    *,
    force_open: bool = False,
) -> None:
    """Allow the operator to complete an incomplete/partial CV draft by hand."""
    if hand.id is None:
        return
    if hand.source_type == "manual" and hand.completion_status == "not_applicable":
        return
    if has_operator_manual_completion(evidence) and hand.completion_status == "complete":
        st.caption("This draft was finalized by operator attestation.")
        return
    if hand.completion_status not in {"partial", "uncertain"}:
        if force_open:
            st.caption("This hand is already complete — nothing to finalize.")
        return
    with _study_panel(
        "Finalize incomplete hand",
        force_open=force_open,
        expanded=True,
    ):
        st.caption(
            "Use this when the recording missed the start or end, but you still "
            "reconstructed the whole hand (for example you joined late on preflop "
            "and filled every action). Fill hero cards, edit missing actions, "
            "acknowledge remaining source warnings, then attest how the hand ended. "
            "Pipeline rejection codes stay in the audit trail; finalize overrides "
            "them for study."
        )
        if evidence.rejection_codes:
            st.info(
                "Pipeline rejected: "
                + ", ".join(evidence.rejection_codes)
                + ". Finalize is allowed if you filled those gaps yourself."
            )
        if not (hand.hero_cards or "").strip():
            st.warning("Fill in hero cards before finalizing.")
        if evidence.unresolved_warning_codes:
            st.warning(
                "Acknowledge remaining source warnings first: "
                + ", ".join(evidence.unresolved_warning_codes)
            )
        default_terminal = (
            evidence.terminal_event
            if evidence.terminal_event in TERMINAL_EVENT_OPTIONS
            else "hero_fold"
        )
        terminal = st.selectbox(
            "How did this hand end?",
            TERMINAL_EVENT_OPTIONS,
            index=TERMINAL_EVENT_OPTIONS.index(default_terminal),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"finalize_terminal_{hand.id}",
        )
        notes = st.text_input(
            "Finalize notes (optional)",
            key=f"finalize_notes_{hand.id}",
            placeholder=(
                "Example: joined mid-preflop; reconstructed full action from the table"
                if evidence.rejection_codes
                else "Example: folded preflop after reviewing the video"
            ),
        )
        if st.button(
            "Mark complete from my fill-in",
            key=f"finalize_incomplete_{hand.id}",
            type="primary",
            disabled=not (hand.hero_cards or "").strip(),
        ):
            try:
                finalized = db.finalize_incomplete_hand(
                    hand.id,
                    terminal_event=terminal,
                    notes=notes,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if finalized.completion_status == "complete":
                    flash(
                        "Incomplete hand finalized as complete. Confirm evidence "
                        "and mark reviewed when ready."
                    )
                else:
                    flash(
                        f"Finalize saved (status: {finalized.completion_status}). "
                        "Clear remaining blockers under Import validation before study."
                    )
                st.rerun()


def show_hand_fact_editor(
    db: PokerDatabase,
    hand: Hand,
    *,
    force_open: bool = False,
) -> None:
    """Edit reconstructed evidence in place and retain the original values.

    Reads the STORED hand rather than the one it was handed. Every Study surface
    builds its hand list through ``_hands_with_accounting_results``, which
    replaces ``hero_bb_won`` with the derived ledger result for display, and that
    same object reached this form: the 'Hero result (BB)' input rendered
    pre-filled with the derivation while the column was NULL, and saving any
    other field on the form -- a note, a tag -- wrote the derivation into the
    OBSERVED column and recorded it in ``hand_corrections`` as a fact the
    operator stated. With a rake declared, the number written was the operator's
    own rake policy applied to the action line: a settlement assumption laundered
    into the observation column that the whole accounting cross-check treats as
    independent evidence (see ``hand_accounting._cross_check``, which compares
    ``hands.hero_bb_won`` EXACTLY for exactly that reason).
    """

    if hand.id is not None:
        stored = db.fetch_hand(hand.id)
        if stored is not None:
            hand = stored
    with _study_panel("Hand facts", force_open=force_open, expanded=False):
        st.caption(
            "Saving changes updates this hand in SQLite, records before/after evidence, "
            "and marks prior coaching stale."
        )
        with st.form(f"correct_hand_facts_{hand.id}"):
            game_type = st.text_input("Game type", value=hand.game_type)
            blinds_antes = st.text_input("Blinds / antes", value=hand.blinds_antes)
            first, second = st.columns(2)
            table_size = first.number_input(
                "Table size",
                min_value=2,
                max_value=10,
                value=hand.table_size,
                placeholder="Unknown",
            )
            effective_stack = second.number_input(
                "Effective stack (BB)",
                min_value=0.0,
                value=hand.effective_stack,
                placeholder="Unknown",
            )
            hero_position = st.selectbox(
                "Hero position",
                POSITIONS,
                index=POSITIONS.index(hand.hero_position) if hand.hero_position in POSITIONS else 0,
            )
            hero_cards = st.text_input("Hero cards", value=hand.hero_cards)
            board_cards = st.text_input("Board cards", value=hand.board_cards)
            pot_col, result_col = st.columns(2)
            pot_size = pot_col.number_input(
                "Observed final pot (BB)",
                min_value=0.0,
                value=hand.pot_size,
                placeholder="Unknown",
            )
            hero_bb_won = result_col.number_input(
                "Hero result (BB)",
                value=hand.hero_bb_won,
                placeholder="Unknown",
            )
            result = st.text_input("Result label", value=hand.result)
            tags = st.multiselect("Tags", sorted(HAND_TAGS), default=hand.tags)
            notes = st.text_area("Hand notes", value=hand.notes)
            correction_reason = st.text_input(
                "Why is this correction needed?",
                placeholder="Example: turn card was read as 8h; video shows 6h",
            )
            submitted = st.form_submit_button("Save corrected facts")
        if submitted:
            if not correction_reason.strip():
                st.error("Add a correction reason so this example can improve reconstruction.")
                return
            try:
                corrected = Hand(
                    **{
                        **hand.model_dump(),
                        "game_type": game_type.strip(),
                        "blinds_antes": blinds_antes.strip(),
                        "table_size": None if table_size is None else int(table_size),
                        "effective_stack": _optional_float(effective_stack),
                        "hero_position": hero_position,
                        "hero_cards": hero_cards.strip(),
                        "board_cards": board_cards.strip(),
                        "pot_size": _optional_float(pot_size),
                        "result": result.strip(),
                        "hero_bb_won": _optional_float(hero_bb_won),
                        "tags": tags,
                        "notes": notes.strip(),
                    }
                )
                db.update_hand_facts(corrected, correction_notes=correction_reason)
            except (sqlite3.IntegrityError, ValidationError, ValueError) as exc:
                st.error(f"Could not save corrected facts: {exc}")
            else:
                flash("Corrected facts saved; derived coaching is now stale.")
                st.rerun()


def show_correction_history(
    db: PokerDatabase,
    hand_id: int,
    *,
    force_open: bool = False,
) -> None:
    corrections = db.fetch_hand_corrections(hand_id)
    with _study_panel(
        f"Correction history ({len(corrections)})",
        force_open=force_open,
    ):
        if not corrections:
            st.caption("No corrections recorded yet.")
            return
        for correction in corrections:
            changed = sorted(
                set(correction.before_state) | set(correction.after_state)
            )
            st.markdown(
                f"**{correction.correction_type.replace('_', ' ').title()}** · "
                f"{correction.created_at.isoformat()}"
            )
            st.caption(correction.notes or "No correction reason recorded.")
            st.caption(f"Changed evidence · {', '.join(changed) or 'new/deleted row'}")


def show_study_coach_review(
    db: PokerDatabase,
    session: Session,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    reviews,
    readiness: StudyReadiness,
) -> None:
    """Show current coaching and allow an explicit post-correction rerun."""

    current_reviews = [review for review in reviews if not review.is_stale]
    stale_reviews = [review for review in reviews if review.is_stale]
    if current_reviews:
        latest = current_reviews[0]
        st.caption(f"{latest.provider_name} · {latest.model_name} · current")
        for title, content in latest.parsed_sections.items():
            st.markdown(f"**{title.replace('_', ' ').title()}**")
            st.write(content)
    else:
        empty_state(
            "No current coaching review",
            (
                "The saved review is stale because the hand changed."
                if stale_reviews
                else "Generate a post-session review below."
            ),
        )
    if stale_reviews:
        st.warning(
            f"{len(stale_reviews)} prior coaching review(s) are retained as stale evidence."
        )
    # The second clearing action STALE_COACHING_EVIDENCE names, and the only one
    # that works without a configured provider -- which is the state every
    # imported hand starts in, because import_session stales all of them. The
    # solver twin has had Delete stale run for the same reason since round 8.
    # Driven by the blocker, not by `stale_reviews`: the legacy hand_reviews rows
    # block too and are not in this tab's list, so a hand staled only there would
    # otherwise show the blocker with no control anywhere that clears it.
    if hand.id is not None and readiness.has("STALE_COACHING_EVIDENCE"):
        if st.button(
            "Discard stale coaching",
            key=f"coach_discard_stale_{hand.id}",
            width="stretch",
        ):
            discarded = db.discard_stale_coaching(hand.id)
            flash(f"Discarded {discarded} stale coaching review(s).")
            st.rerun()

    provider_choice = st.selectbox(
        "Rerun provider",
        ["Claude (Anthropic)", "Cloud (OpenAI)"],
        key=f"study_coach_provider_{hand.id}",
    )
    provider_key = {
        "Claude (Anthropic)": "anthropic",
        "Cloud (OpenAI)": "cloud",
    }[provider_choice]
    provider = None
    try:
        provider = get_provider_from_env(provider_key)
        st.caption(f"Model · {provider.model_name}")
    except LLMProviderError as exc:
        st.warning(str(exc))

    coaching_mode = st.selectbox(
        "Coaching mode",
        COACHING_MODES,
        key=f"study_coach_mode_{hand.id}",
    )
    range_options = sorted(RANGE_LABELS)
    range_label = st.selectbox(
        "Villain range label",
        range_options,
        index=range_options.index(estimate_villain_range_label(hand.tags, hand.notes)),
        key=f"study_coach_range_{hand.id}",
    )
    is_authoritative = _accounting_is_established(hand, accounting)
    solver_evidence = None
    if hand.id is not None:
        completed_runs = [
            run
            for run in db.fetch_solver_runs_by_hand(hand.id)
            if run.status == "completed" and run.evidence
        ]
        if completed_runs:
            try:
                solver_evidence = SolverEvidence.model_validate(completed_runs[0].evidence)
                st.caption(
                    f"Solver evidence attached · run #{completed_runs[0].id} · "
                    f"{solver_evidence.backend} {solver_evidence.backend_version}"
                )
            except ValidationError:
                st.warning("Latest saved solver evidence is invalid and was not attached.")
    if unattested_assumption_dependence(hand, accounting):
        # Naming the ledger here was false on exactly the hands that reach it:
        # the ledger IS legal and balanced, which is why a dependence could be
        # measured at all, and the action the operator needs is in a different
        # panel from the one this sentence used to send them to.
        st.warning(
            "Coaching is disabled until you confirm the declared settlement "
            "assumptions this hand's reconciliation rests on, in Summary → "
            "Accounting reconciliation."
        )
    elif not is_authoritative:
        st.warning(
            "Reconcile a legal, balanced ledger before generating coaching from this hand."
        )
    prompt = build_hand_review_prompt(
        session,
        hand,
        actions,
        players,
        pot_odds_facts=_accounting_prompt_math_facts(hand, accounting),
        villain_range_label=range_label,
        coaching_mode=coaching_mode,
        ledger=None if accounting is None else accounting.ledger,
        accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
        accounting_authoritative=is_authoritative,
        solver_evidence=solver_evidence,
    )
    with st.expander("Exact post-session prompt"):
        st.code(prompt, language="text")
    if readiness.has("OPEN_DEBUGGING_ISSUE"):
        st.warning("Resolve the open debugging issue before generating coaching.")
    if readiness.has("STUDY_EXCLUDED_BY_OPERATOR"):
        st.warning("This hand is marked non-study; coaching is disabled.")
    if st.button(
        "Generate and save corrected-hand coaching",
        key=f"study_rerun_coaching_{hand.id}",
        disabled=(
            provider is None
            or not is_authoritative
            or readiness.has("OPEN_DEBUGGING_ISSUE")
            or readiness.has("STUDY_EXCLUDED_BY_OPERATOR")
        ),
        width="stretch",
    ):
        try:
            safety = validate_post_session_prompt(prompt)
            if not safety.is_safe:
                raise ValueError("; ".join(safety.errors))
            with st.spinner("Generating corrected-hand coaching..."):
                raw_response = provider.generate_hand_review(prompt)
            if solver_evidence is not None:
                validate_solver_coaching_response(raw_response, solver_evidence)
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
            # The coaching is kept either way; only the promotion is gated. flash()
            # is used so the outcome survives the rerun below.
            if readiness.is_ready and guarded_update_hand_status(
                db, hand, readiness, "reviewed"
            ):
                flash(f"Saved current coaching review #{saved.id}.")
            else:
                flash(
                    f"Saved coaching review #{saved.id}. Review status unchanged: "
                    "this hand is not study-ready."
                )
            st.rerun()
        except (LLMProviderError, ValueError) as exc:
            st.error(f"Could not generate coaching: {exc}")

    if stale_reviews:
        st.markdown("##### Retained stale coaching")
        show_saved_provider_reviews(stale_reviews)


def show_solver_review(
    db: PokerDatabase,
    session: Session,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    readiness: StudyReadiness,
) -> None:
    """Configure and run one auditable post-session TexasSolver analysis."""

    st.markdown("##### TexasSolver postflop analysis")
    st.caption(
        "TexasSolver analyzes one saved heads-up postflop decision. It does not "
        "solve preflop, multiway, tournament, or live hands."
    )
    with st.expander("How to use TexasSolver", expanded=False):
        workflow_step(
            1,
            "Make the hand eligible",
            "On Import validation, correct the cards, players, positions, and "
            "actions; then reconcile the chip ledger.",
        )
        workflow_step(
            2,
            "Choose both starting ranges",
            "Use Default first. Premade and Custom are available when you have a "
            "better range assumption.",
        )
        workflow_step(
            3,
            "Run and inspect",
            "Run the analysis, refresh while it works, then review Hero's saved "
            "combo frequencies and assumptions.",
        )
        st.caption(
            "Optional: use Explain solver result with AI only after the solver run "
            "finishes. The explanation is grounded in the saved frequencies."
        )

    prepared = prepare_solver_spot(hand, players, actions, accounting)
    for warning in prepared.eligibility.warnings:
        st.warning(warning)
    runs = db.fetch_solver_runs_by_hand(hand.id) if hand.id is not None else []
    if not prepared.eligibility.eligible or prepared.spot is None:
        st.warning("This hand is not ready for TexasSolver yet.")
        st.markdown("**Fix these items on Import validation:**")
        for reason in prepared.eligibility.reasons:
            st.markdown(f"- {reason}")
        _show_solver_runs(
            db, session, hand, players, actions, accounting, accounting_error, runs, readiness
        )
        return

    spot = prepared.spot
    st.success("Eligible heads-up postflop spot found.")
    st.markdown("###### 1. Confirm the selected spot")
    st.markdown(
        f"**{spot.street.title()} · {spot.oop.player_name} (OOP) vs "
        f"{spot.ip.player_name} (IP)**"
    )
    st.caption(
        f"{spot.pot:g} BB pot · {spot.effective_stack:g} BB effective · "
        f"{spot.pot_type.replace('_', ' ')}"
    )
    st.markdown("###### 2. Choose starting ranges")
    st.caption(
        "Default is the recommended first pass. Built-in ranges are transparent "
        "study estimates, not solved preflop GTO ranges."
    )
    user_profiles = db.fetch_solver_range_profiles()
    oop_range, oop_error = _solver_range_selector(
        db, spot, spot.oop, user_profiles, key_prefix=f"solver_oop_{hand.id}"
    )
    ip_range, ip_error = _solver_range_selector(
        db, spot, spot.ip, user_profiles, key_prefix=f"solver_ip_{hand.id}"
    )
    range_errors = [error for error in (oop_error, ip_error) if error]
    for error in range_errors:
        st.error(error)
    if oop_range is not None:
        _show_resolved_range(oop_range)
    if ip_range is not None:
        _show_resolved_range(ip_range)

    st.markdown("###### 3. Run TexasSolver")
    binary_ready = True
    try:
        binary = configured_binary()
        configured_resource_dir(binary)
        st.success("TexasSolver is installed and ready.")
        st.caption(f"Pinned backend · {PINNED_CONSOLE_COMMIT} · {binary}")
    except (FileNotFoundError, PermissionError) as exc:
        binary_ready = False
        st.warning(str(exc))
        with st.expander("One-time local TexasSolver setup", expanded=True):
            st.write(
                "Install or compile the pinned console solver, keep its resources "
                "directory beside the binary, then restart PokerTrainer with:"
            )
            st.code(
                "export TEXAS_SOLVER_PATH=/absolute/path/to/console_solver\n"
                "# Only needed when resources is not beside the binary:\n"
                "export TEXAS_SOLVER_RESOURCE_DIR=/absolute/path/to/resources\n"
                "streamlit run app.py",
                language="bash",
            )
            st.caption("You can verify this later in Settings → Solver.")
    if readiness.has("STUDY_EXCLUDED_BY_OPERATOR"):
        st.warning("This hand is marked non-study; solver analysis is disabled.")
    if st.button(
        "Run TexasSolver analysis",
        key=f"solver_analyze_{hand.id}",
        type="primary",
        width="stretch",
        disabled=bool(range_errors)
        or oop_range is None
        or ip_range is None
        or not binary_ready
        or readiness.has("STUDY_EXCLUDED_BY_OPERATOR"),
    ):
        try:
            run = start_solver_job(
                db,
                spot,
                ip_range,
                oop_range,
                assumptions=prepared.eligibility.warnings,
            )
            flash(
                f"Solver run #{run.id} is {run.status}."
                if run.status != "completed"
                else f"Reused completed solver run #{run.id}."
            )
            st.rerun()
        except (
            SolverJobAlreadyRunningError,
            FileNotFoundError,
            PermissionError,
            RuntimeError,
            ValueError,
        ) as exc:
            st.error(str(exc))
    _show_solver_runs(
        db, session, hand, players, actions, accounting, accounting_error, runs, readiness
    )


def _solver_range_selector(
    db: PokerDatabase,
    spot,
    player,
    user_profiles: list[SolverRangeProfile],
    *,
    key_prefix: str,
) -> tuple[ResolvedRange | None, str | None]:
    st.markdown(
        f"##### {player.player_name} · {player.position or 'position unknown'} · "
        f"{player.role.upper()}"
    )
    st.caption(
        "Default uses the built-in estimate matched to this seat and pot type."
    )
    mode = st.radio(
        "Range source",
        ["Default", "Premade", "Custom"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )
    try:
        if mode == "Default":
            return (
                resolve_profile(
                    spot,
                    player,
                    BUILTIN_RANGE_PROFILES,
                    source="default",
                ),
                None,
            )
        if mode == "Premade":
            scenario = default_scenario(spot, player)
            options = [
                profile
                for profile in [*BUILTIN_RANGE_PROFILES, *user_profiles]
                if profile.pot_type in {"", spot.pot_type}
                and profile.scenario in {"", scenario}
            ]
            selected_index = st.selectbox(
                "Premade range",
                options=range(len(options)),
                format_func=lambda index: (
                    f"{options[index].name}"
                    + (" · saved" if options[index].id is not None else "")
                ),
                key=f"{key_prefix}_profile",
            )
            selected = options[int(selected_index)]
            source = "user" if selected.id is not None else "builtin"
            return (
                resolve_selected_profile(
                    spot,
                    player,
                    selected,
                    source=source,
                ),
                None,
            )

        notation = st.text_area(
            "Custom weighted range",
            placeholder="TT+,AJs+:0.5,AQo+",
            help=(
                "Supports standard eval7 notation and TexasSolver-style HAND:weight tokens."
            ),
            key=f"{key_prefix}_custom",
        )
        if not notation.strip():
            return None, "Enter a custom range or select Default."
        resolved = resolve_custom_range(spot, player, notation)
        normalized = normalize_weighted_notation(notation)
        st.markdown(
            range_matrix_html(
                range_cells_from_notation(normalized),
                label=f"{player.player_name} custom range",
            ),
            unsafe_allow_html=True,
        )
        save_name = st.text_input(
            "Save name",
            placeholder=f"{spot.table_size}-max {player.position} custom",
            key=f"{key_prefix}_save_name",
        )
        if st.button(
            "Save as premade range",
            key=f"{key_prefix}_save",
            disabled=not save_name.strip(),
        ):
            profile = SolverRangeProfile(
                name=save_name.strip(),
                notation=normalized,
                table_size=spot.table_size,
                position=player.position,
                scenario=default_scenario(spot, player),
                pot_type=spot.pot_type,
                stack_bb=spot.effective_stack,
                description="User-saved post-session solver range.",
                updated_at=utc_now(),
            )
            db.create_solver_range_profile(profile)
            flash(f"Saved premade range '{profile.name}'.")
            st.rerun()
        return resolved, None
    except (ValueError, sqlite3.IntegrityError) as exc:
        return None, str(exc)


def _show_resolved_range(resolved: ResolvedRange) -> None:
    st.caption(
        f"{resolved.role.upper()} resolved range · {resolved.profile_name} · "
        f"{resolved.combo_count} blocker-valid combos · "
        f"{format_percentage(resolved.range_percent)} weighted coverage"
    )
    for mismatch in resolved.mismatches:
        st.warning(mismatch)


def _show_solver_runs(
    db: PokerDatabase,
    session: Session,
    hand: Hand,
    players: list[HandPlayer],
    actions: list[Action],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    runs,
    readiness: StudyReadiness,
) -> None:
    if not runs:
        return
    latest = runs[0]
    st.divider()
    st.markdown(f"##### Solver result · run #{latest.id}")
    st.caption(
        f"Status · {latest.status.replace('_', ' ').title()} · "
        f"backend {latest.backend_name} {latest.backend_version}"
    )
    if latest.status in {"queued", "running", "cancelling"}:
        st.info("Solver work is running in the background. Refresh to check progress.")
        refresh_col, cancel_col = st.columns(2)
        if refresh_col.button("Refresh", key=f"solver_refresh_{latest.id}", width="stretch"):
            st.rerun()
        if cancel_col.button(
            "Cancel",
            key=f"solver_cancel_{latest.id}",
            width="stretch",
            disabled=latest.status == "cancelling",
        ):
            cancel_solver_run(db, latest.id)
            flash(f"Cancelled solver run #{latest.id}.")
            st.rerun()
        return
    if latest.status in {"failed", "cancelled", "stale"}:
        st.warning(latest.error_message or f"Solver run is {latest.status}.")
        # The clearing action STALE_SOLVER_EVIDENCE names. Re-running is not always
        # possible: a correction can leave a hand solver-ineligible with a stale
        # run that would otherwise block study forever.
        if latest.id is not None and st.button(
            "Delete stale run",
            key=f"solver_delete_{latest.id}",
            width="stretch",
        ):
            try:
                db.delete_solver_run(latest.id)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(f"Deleted solver run #{latest.id}.")
                st.rerun()
        return
    try:
        evidence = SolverEvidence.model_validate(latest.evidence)
    except ValidationError as exc:
        st.error(f"Saved solver evidence is invalid: {exc}")
        return
    if evidence.exploitability_pct is not None:
        st.metric("Final exploitability", f"{evidence.exploitability_pct:g}% of starting pot")
    if evidence.runtime_seconds is not None:
        st.caption(f"Runtime · {evidence.runtime_seconds:.1f} seconds")
    if evidence.action_frequencies:
        st.markdown("**Recorded Hero combo strategy**")
        for item in evidence.action_frequencies:
            st.progress(
                item.frequency,
                text=f"{item.action} · {format_percentage(item.frequency)}",
            )
    else:
        st.warning("Combo-specific strategy was unavailable at the mapped node.")
    if evidence.range_action_frequencies:
        with st.expander("Input-weighted range strategy"):
            for item in evidence.range_action_frequencies:
                st.progress(
                    item.frequency,
                    text=f"{item.action} · {format_percentage(item.frequency)}",
                )
    if evidence.recorded_action:
        st.caption(
            f"Recorded · {evidence.recorded_action} · mapped solver branch · "
            f"{evidence.mapped_action or 'unavailable'}"
        )
    for item in [*evidence.assumptions, *evidence.warnings]:
        st.caption(f"· {item}")
    st.caption("Action EV and exact BB loss are unavailable and are not inferred.")

    provider_choice = st.selectbox(
        "Explanation provider",
        ["Claude (Anthropic)", "Cloud (OpenAI)"],
        key=f"solver_explain_provider_{hand.id}",
    )
    provider_key = {
        "Claude (Anthropic)": "anthropic",
        "Cloud (OpenAI)": "cloud",
    }[provider_choice]
    provider = None
    try:
        provider = get_provider_from_env(provider_key)
    except LLMProviderError as exc:
        st.warning(str(exc))
    if readiness.has("OPEN_DEBUGGING_ISSUE"):
        st.warning("Resolve the open debugging issue before explaining this solver result.")
    if readiness.has("STUDY_EXCLUDED_BY_OPERATOR"):
        st.warning("This hand is marked non-study; solver explanation is disabled.")
    if st.button(
        "Explain solver result with AI",
        key=f"solver_explain_{latest.id}",
        disabled=(
            provider is None
            or not _accounting_is_established(hand, accounting)
            or readiness.has("OPEN_DEBUGGING_ISSUE")
            or readiness.has("STUDY_EXCLUDED_BY_OPERATOR")
        ),
        width="stretch",
    ):
        prompt = build_hand_review_prompt(
            session,
            hand,
            actions,
            players,
            pot_odds_facts=_accounting_prompt_math_facts(hand, accounting),
            ledger=None if accounting is None else accounting.ledger,
            accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
            accounting_authoritative=_accounting_is_established(hand, accounting),
            solver_evidence=evidence,
        )
        try:
            safety = validate_post_session_prompt(prompt)
            if not safety.is_safe:
                raise ValueError("; ".join(safety.errors))
            with st.spinner("Explaining saved solver evidence..."):
                raw_response = provider.generate_hand_review(prompt)
            validate_solver_coaching_response(raw_response, evidence)
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
            # The explanation is kept either way; only the promotion is gated.
            if readiness.is_ready and guarded_update_hand_status(
                db, hand, readiness, "reviewed"
            ):
                flash(f"Saved solver-grounded coaching review #{saved.id}.")
            else:
                flash(
                    f"Saved solver-grounded coaching review #{saved.id}. Review status "
                    "unchanged: this hand is not study-ready."
                )
            st.rerun()
        except (LLMProviderError, ValueError) as exc:
            st.error(f"Could not explain solver result: {exc}")


def _hero_ledger_result(
    hand: Hand,
    accounting: AccountingReconciliation | None,
    players: list[HandPlayer],
    observed_result: float | None,
) -> float | None:
    if not _accounting_is_established(hand, accounting):
        return observed_result
    assert accounting is not None
    hero = next((player for player in players if player.is_hero), None)
    if hero is None:
        return observed_result
    return accounting.ledger.net_results.get(hero.player_key, observed_result)


def _render_accounting_status(
    accounting: AccountingReconciliation | None,
    error: str | None,
) -> None:
    if error is not None:
        st.warning(f"Accounting unavailable · {error}")
        return
    if accounting is None:
        st.warning("Accounting unavailable.")
        return
    if accounting.is_authoritative:
        st.success(
            f"Reconciled · {len(accounting.ledger.pots)} pot layer(s) · "
            f"{accounting.ledger.rake:g} BB rake · balanced and legal"
        )
        return
    label = (
        "Unsettled"
        if accounting.settlement is None
        else accounting.settlement.status.replace("_", " ").title()
    )
    st.warning(f"{label} · accounting is not authoritative.")
    for issue in accounting.issues[:4]:
        st.caption(f"· {issue}")
    if len(accounting.issues) > 4:
        st.caption(f"· {len(accounting.issues) - 4} additional issue(s)")


def _accounting_player_labels(
    players: list[HandPlayer],
) -> dict[str, HandPlayer]:
    """Return readable settlement labels without collapsing stable identities."""

    base_labels = [
        (
            (f"Seat {player.seat_index} · " if player.seat_index is not None else "")
            + (actor_label(player.player_name, player.position) or "position unknown")
        )
        for player in players
    ]
    duplicate_labels = {label for label in base_labels if base_labels.count(label) > 1}
    return {
        (f"{label} · {player.player_key}" if label in duplicate_labels else label): player
        for label, player in zip(base_labels, players, strict=True)
    }


def _settlement_entry_editor_label(
    entry: SettlementEntry,
    player_labels: dict[str, HandPlayer],
    label_by_key: dict[str, str],
) -> str:
    """Resolve an entry to an editor option or an explicit stale sentinel."""

    if entry.player_key is not None and entry.player_key in label_by_key:
        return label_by_key[entry.player_key]
    if entry.player_key is None:
        matches = [
            label
            for label, player in player_labels.items()
            if player.player_name == entry.player_name
        ]
        if len(matches) == 1:
            return matches[0]
    identifier = (
        entry.player_key[:8]
        if entry.player_key
        else f"entry-{entry.id if entry.id is not None else 'unsaved'}"
    )
    return f"[Needs reassignment] {entry.player_name or 'Unknown player'} · {identifier}"


def show_assumption_dependence_controls(
    db: PokerDatabase,
    hand: Hand,
    accounting: AccountingReconciliation | None,
) -> None:
    """Draw the exact control ACCOUNTING_ASSUMPTION_DEPENDENT names, and nothing else.

    It lives inside the Accounting reconciliation panel rather than in the Source
    warnings panel on purpose. A settlement assumption is not a pipeline warning:
    it is a claim the operator made in the four inputs directly below this, and
    the other clearing action -- withdrawing the claim -- is those same inputs. It
    is also deliberately separate from 'I have read the evidence above and confirm
    this hand is correct', which asks a broader question that a reader can answer
    honestly without ever having looked at the rake.

    The button records the MEASURED code, so what is stored is an attestation to
    a chip quantity. If the pot later changes under an unchanged policy, the
    measurement changes with it, the stored code no longer matches, and the
    blocker returns without anything needing to notice that the settlement row
    was untouched.
    """

    if hand.id is None or accounting is None or not accounting.assumption_dependence:
        return
    acknowledged = set(
        parse_completion_evidence(hand.completion_evidence).confirmed_assumption_codes
    )
    # The same predicate the blocker is emitted under and the same one the writer
    # behind this button enforces, so the control is drawn exactly when it can do
    # something. It used to be `is_reconstructed_hand`, while the writer refused
    # every `source_type == 'manual'` row: on a manual hand whose completion
    # status was not `not_applicable` -- a state `create_hand` accepts -- the
    # button was drawn, the write was discarded, and the page still flashed
    # "Confirmed".
    reconstructed = hand_requires_assumption_attestation(hand)
    st.markdown("**Declared settlement assumptions this reconciliation rests on**")
    st.caption(
        "Removing these declarations stops this hand reconciling, so the pot, the "
        "rake, and the hero result are not established by the recording alone. "
        "Confirming records the exact chip movement you are attesting to; it is "
        "not a confirmation of the hand as a whole."
        if reconstructed
        else "Recorded for audit. You entered this hand yourself, so a declared "
        "ante, dead blind, rake or pot award is your own observation: it is "
        "measured and shown here, and it never blocks study, coaching or the "
        "solver."
    )
    for index, dependence in enumerate(accounting.assumption_dependence):
        detail_col, action_col = st.columns([2.4, 1])
        is_confirmed = dependence.code in acknowledged
        detail_col.markdown(
            status_badge(
                "reviewed" if is_confirmed or not reconstructed else "needs_correction",
                label=f"{dependence.input_name} · "
                f"{'confirmed' if is_confirmed else 'unconfirmed'}",
            ),
            unsafe_allow_html=True,
        )
        detail_col.caption(dependence.describe())
        if is_confirmed or not reconstructed:
            continue
        if action_col.button(
            "Confirm this assumption",
            key=f"accounting_assumption_{hand.id}_{index}",
            width="stretch",
        ):
            if attest_assumption(db, hand.id, dependence.code):
                flash(f"Confirmed the declared {dependence.input_name} for this hand.")
                st.rerun()
            else:
                # Reachable, and reported honestly rather than assumed: a control
                # must never say "Confirmed" over a write that was discarded. The
                # measurement is re-derived inside `attest_assumption`, so a
                # dependence that stopped existing between this page rendering and
                # this button being pressed -- another tab saving the settlement,
                # a correction landing -- refuses here instead of recording an
                # attestation to a quantity that no longer exists.
                #
                # No rerun on this branch: rerunning discarded the message before
                # it was ever drawn, which is why the guard round 10 added had no
                # observable behaviour to test and survived mutation.
                st.error(
                    "Nothing was recorded: this hand no longer measures that "
                    "assumption, or it does not require an attestation. Reload the "
                    "hand to see the current measurement."
                )
    st.divider()


def show_accounting_editor(
    db: PokerDatabase,
    hand: Hand,
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    *,
    force_open: bool = False,
) -> None:
    if hand.id is None:
        return
    with _study_panel(
        "Accounting reconciliation",
        force_open=force_open,
        expanded=False,
    ):
        if accounting_error is not None:
            st.error(accounting_error)
            st.caption(
                "Correct unknown action amounts or player identities in Sessions before settling. "
                "Existing settlement entries remain editable so stale awards can be removed."
            )
        settlement = (
            accounting.settlement
            if accounting is not None and accounting.settlement is not None
            else db.fetch_hand_settlement(hand.id) or HandSettlement(hand_id=hand.id)
        )
        show_assumption_dependence_controls(db, hand, accounting)
        player_labels = _accounting_player_labels(players)
        if not player_labels:
            st.info("Add players and starting stacks before reconciling.")
            return
        entries = (
            list(accounting.entries)
            if accounting is not None
            else db.fetch_settlement_entries(hand.id)
        )
        awards = [entry for entry in entries if entry.entry_type == "award"]
        refunds = [entry for entry in entries if entry.entry_type == "refund"]
        label_by_key = {player.player_key: label for label, player in player_labels.items()}
        award_editor_labels = [
            _settlement_entry_editor_label(entry, player_labels, label_by_key) for entry in awards
        ]
        refund_editor_labels = [
            _settlement_entry_editor_label(entry, player_labels, label_by_key) for entry in refunds
        ]
        stale_labels = {
            label
            for label in [*award_editor_labels, *refund_editor_labels]
            if label not in player_labels
        }
        player_options = [*player_labels, *sorted(stale_labels)]
        if stale_labels:
            st.warning(
                "Some saved awards or refunds reference players no longer in this hand. "
                "Reassign or delete those rows before saving."
            )
        with st.form(f"accounting_{hand.id}"):
            assumption_cols = st.columns(4)
            dead_money = assumption_cols[0].number_input(
                "External dead money",
                min_value=0.0,
                value=float(settlement.dead_money),
                step=0.5,
                help="Only chips not represented by player actions.",
            )
            rake_rate_pct = assumption_cols[1].number_input(
                "Rake %",
                min_value=0.0,
                max_value=100.0,
                value=float(settlement.rake_rate * 100),
                step=0.1,
            )
            rake_cap = assumption_cols[2].number_input(
                "Rake cap",
                min_value=0.0,
                value=settlement.rake_cap,
                step=0.5,
                placeholder="Uncapped",
            )
            rounding = assumption_cols[3].number_input(
                "Chip unit",
                min_value=0.001,
                value=float(settlement.rake_rounding_unit),
                step=0.01,
            )
            no_flop_no_drop = st.checkbox(
                "No flop, no drop",
                value=settlement.no_flop_no_drop,
            )
            sync_observed = st.checkbox(
                "Replace observed final pot/result with the derived ledger values",
                value=False,
                help=(
                    "Use only after verifying the action line and winners. "
                    "This makes the reconciled ledger the saved summary."
                ),
            )
            # Name each layer from the ledger's own record of what created it.
            # Reading "1+ are side pots" off the index told the operator that a
            # layer split off by a blind that folded for less was a side pot,
            # which it is not.
            if accounting is not None and accounting.ledger.pots:
                layer_names = ", ".join(
                    f"{pot.index} {pot.label.lower()}" for pot in accounting.ledger.pots
                )
                st.caption(f"Awards declare winners by pot layer: {layer_names}.")
            else:
                st.caption(
                    "Awards declare winners by pot layer, numbered from 0 for the main pot."
                )
            award_rows = st.data_editor(
                [
                    {
                        "Pot": entry.pot_index,
                        "Winner": editor_label,
                        "Payout": entry.amount,
                        "Odd-chip order": entry.entry_order,
                    }
                    for entry, editor_label in zip(awards, award_editor_labels, strict=True)
                ],
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Pot": st.column_config.NumberColumn("Pot", min_value=0, step=1, required=True),
                    "Winner": st.column_config.SelectboxColumn(
                        "Winner", options=player_options, required=True
                    ),
                    "Payout": st.column_config.NumberColumn("Observed payout", min_value=0),
                    "Odd-chip order": st.column_config.NumberColumn(
                        "Order", min_value=1, step=1, required=True
                    ),
                },
                key=f"accounting_awards_{hand.id}",
            )
            refund_rows = st.data_editor(
                [
                    {
                        "Player": editor_label,
                        "Amount": entry.amount,
                        "Order": entry.entry_order,
                    }
                    for entry, editor_label in zip(refunds, refund_editor_labels, strict=True)
                ],
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Player": st.column_config.SelectboxColumn(
                        "Refunded player", options=player_options, required=True
                    ),
                    "Amount": st.column_config.NumberColumn(
                        "Uncalled refund", min_value=0, required=True
                    ),
                    "Order": st.column_config.NumberColumn(
                        "Order", min_value=1, step=1, required=True
                    ),
                },
                key=f"accounting_refunds_{hand.id}",
            )
            submitted = st.form_submit_button("Save and reconcile", type="primary")
        if not submitted:
            return
        try:
            saved_entries: list[SettlementEntry] = []
            for row in award_rows:
                label = str(row.get("Winner") or "")
                if not label and row.get("Pot") is None:
                    continue
                if label not in player_labels:
                    raise ValueError("Choose a current winner or delete each stale award row.")
                if row.get("Pot") is None:
                    raise ValueError("Every award needs a pot layer.")
                player = player_labels[label]
                saved_entries.append(
                    SettlementEntry(
                        hand_id=hand.id,
                        entry_type="award",
                        pot_index=int(row["Pot"]),
                        player_key=player.player_key,
                        player_name=player.player_name,
                        amount=_optional_float(row.get("Payout")),
                        entry_order=int(row.get("Odd-chip order") or 1),
                    )
                )
            for row in refund_rows:
                label = str(row.get("Player") or "")
                if not label and row.get("Amount") is None:
                    continue
                if label not in player_labels:
                    raise ValueError("Choose a current player or delete each stale refund row.")
                if row.get("Amount") is None:
                    raise ValueError("Every refund needs an amount.")
                player = player_labels[label]
                saved_entries.append(
                    SettlementEntry(
                        hand_id=hand.id,
                        entry_type="refund",
                        player_key=player.player_key,
                        player_name=player.player_name,
                        amount=float(row["Amount"]),
                        entry_order=int(row.get("Order") or 1),
                    )
                )
            configured = settlement.model_copy(
                update={
                    "status": "settled",
                    "dead_money": float(dead_money),
                    "rake_rate": float(rake_rate_pct) / 100,
                    "rake_cap": _optional_float(rake_cap),
                    "rake_rounding_unit": float(rounding),
                    "no_flop_no_drop": no_flop_no_drop,
                    "gross_pot": None,
                    "rake_amount": None,
                    "net_pot": None,
                    "is_balanced": False,
                    "warnings": [],
                }
            )
            sync_refusal: str | None = None
            with db.transaction():
                db.upsert_hand_settlement(configured)
                db.replace_settlement_entries(hand.id, saved_entries)
                reconciled = persist_reconciliation(db, hand.id)
                if sync_observed:
                    # The precondition is NOT `ledger.is_settled`. Replacing an
                    # observed-fact column with a derived figure is publishing that
                    # figure as an observation, so it takes the same gate every
                    # other consumer of a derived figure takes, and the gate lives
                    # in the service rather than here.
                    try:
                        reconciled = sync_recorded_figures_from_ledger(db, hand.id)
                    except SettlementSyncRefused as exc:
                        sync_refusal = str(exc)
            if sync_refusal is not None:
                st.error(f"Saved the settlement, but did not replace the record: {sync_refusal}")
            elif reconciled.is_authoritative:
                flash("Hand accounting reconciled and balanced.")
            else:
                flash("Accounting saved with issues that still need correction.")
            if sync_refusal is None:
                st.rerun()
        except (LedgerError, ValidationError, ValueError) as exc:
            st.error(f"Could not reconcile accounting: {exc}")


def show_insights_workspace(db: PokerDatabase) -> None:
    page_header(
        "Insights",
        "Evidence-backed patterns from your completed hands—without fabricated solver scores.",
    )
    accounting_cache = new_accounting_cache()
    hands = _hands_with_accounting_results(db, db.fetch_all_hands(), accounting_cache)
    if not hands:
        empty_state(
            "Not enough evidence", "Insights appear after completed hands are imported or recorded."
        )
        return
    tagged: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for hand in hands:
        statuses[hand.review_status] = statuses.get(hand.review_status, 0) + 1
        for tag in hand.tags:
            tagged[tag] = tagged.get(tag, 0) + 1
    unresolved = [hand for hand in hands if hand.review_status != "reviewed"]
    biggest = sorted(
        [hand for hand in unresolved if hand.hero_bb_won is not None],
        key=lambda hand: abs(hand.hero_bb_won or 0),
        reverse=True,
    )[:8]
    with st.container(key="session_metrics"):
        metric_cols = st.columns(3)
        with metric_cols[0]:
            kpi_card("Evidence base", str(len(hands)), "Completed hands")
        with metric_cols[1]:
            kpi_card(
                "Unresolved",
                str(len(unresolved)),
                "Review status is not 'reviewed'",
                tone="negative" if unresolved else "positive",
            )
        with metric_cols[2]:
            # A blocker count, not an aggregate confidence score: one percentage
            # would read as proof the whole hand is correct. It counts hands, not
            # blockers, and it consults every category rather than the completion
            # column alone -- counting completion only reported 0 for a hand whose
            # ledger did not reconcile. Per-render user confirmation cannot be
            # evaluated across a list, so it is excluded and named below.
            blocked = [
                hand
                for hand in hands
                if hand.id is not None
                and hand.study_inclusion != "skip"
                and not hand_study_readiness(
                    db,
                    hand,
                    *_accounting_or_error(db, hand, accounting_cache),
                    user_confirmed=True,
                ).is_ready
            ]
            kpi_card(
                "Not study-ready",
                str(len(blocked)),
                "Completion, cards, layout, accounting, issue, or evidence blockers",
                tone="negative" if blocked else "positive",
            )
    st.caption(
        "Review status is a workflow label. Study readiness is derived per hand "
        "and additionally requires your explicit confirmation on the Study page."
    )
    left, right = st.columns(2)
    with left:
        section_header("Study themes", "Tag frequency and sample size")
        if tagged:
            theme_rows = [
                {
                    "theme": tag.replace("_", " ").title(),
                    "hands": count,
                }
                for tag, count in sorted(tagged.items(), key=lambda item: -item[1])
            ]
            frequency_bars((str(row["theme"]), int(row["hands"])) for row in theme_rows)
        else:
            empty_state(
                "No tagged themes", "Apply tags during review to build a useful leak index."
            )
    with right:
        section_header("Review coverage", "Workflow status, not playing skill")
        coverage_rows = [
            {
                "status": status.replace("_", " ").title(),
                "hands": count,
                "tone": status,
            }
            for status, count in sorted(statuses.items())
        ]
        coverage_bar(
            (
                str(row["tone"]),
                int(row["hands"]),
            )
            for row in coverage_rows
        )
    section_header("Largest unresolved decisions", "Ranked by absolute recorded BB result")
    if biggest:
        st.dataframe(
            [
                {
                    "Hand": f"#{hand.hand_number}",
                    "Result": f"{hand.hero_bb_won:+g} BB",
                    "Position": hand.hero_position or "—",
                    "Status": hand.review_status.replace("_", " ").title(),
                    "Tags": ", ".join(hand.tags) or "—",
                }
                for hand in biggest
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        empty_state(
            "No unresolved recorded results",
            "Every hand with a recorded BB outcome is marked reviewed. That is a "
            "workflow status, not a study-readiness verdict.",
        )


def show_import_workspace(db: PokerDatabase, session: Session | None) -> None:
    with st.container(key="import_workspace"):
        page_header(
            "Import",
            "Add recordings for one completed session, reconstruct hands, then validate frames.",
            eyebrow="SESSION INGEST",
        )
        sessions = db.fetch_sessions()
        if not sessions:
            empty_state(
                "Create a session for these recordings",
                "Every session needs a played date (defaults to today). You can change it any time.",
            )
            create_session_form(db, form_key="create_import_session")
            return

        session = _choose_import_session(db, sessions, session)
        if session is None:
            empty_state(
                "Select or create a session",
                "Pick a session below, or create one with a played date before importing.",
            )
            create_session_form(db, form_key="create_import_session")
            return
        show_video_processing(db, session)


def show_settings_workspace(db: PokerDatabase, session: Session | None) -> None:
    page_header("Settings", "Calibration, portability, and advanced post-session tooling.")
    calibration_tab, data_tab, math_tab, solver_tab, coach_tab = st.tabs(
        ["ROI calibration", "Data transfer", "Math tools", "Solver", "Coaching"]
    )
    with calibration_tab:
        show_roi_calibration(db)
    with data_tab:
        if session is None:
            empty_state(
                "No session selected",
                "Create or select a session before importing or exporting JSON.",
            )
        else:
            show_import_export(db, session)
    with math_tab:
        if session is None:
            empty_state(
                "No session selected", "Select a session to use its saved hands in math review."
            )
        else:
            with st.container(key="math_workspace"):
                show_math_review(db, session)
    with solver_tab:
        show_solver_settings(db)
    with coach_tab:
        if session is None:
            empty_state(
                "No session selected",
                "Select a completed session before generating coaching review.",
            )
        else:
            show_coach_review(db, session)


def show_solver_settings(db: PokerDatabase) -> None:
    section_header("TexasSolver", "Optional post-session heads-up analysis backend")
    with st.expander("Setup and usage guide", expanded=False):
        st.markdown(
            "1. Install or compile the pinned `console_solver` build.\n"
            "2. Set `TEXAS_SOLVER_PATH` and, if necessary, "
            "`TEXAS_SOLVER_RESOURCE_DIR`.\n"
            "3. Restart PokerTrainer.\n"
            "4. Open **Study → TexasSolver** on an eligible completed hand."
        )
        st.code(
            "export TEXAS_SOLVER_PATH=/absolute/path/to/console_solver\n"
            "# Optional when resources is not beside console_solver:\n"
            "export TEXAS_SOLVER_RESOURCE_DIR=/absolute/path/to/resources\n"
            "export POKERTRAINER_SOLVER_THREADS=4\n"
            "streamlit run app.py",
            language="bash",
        )
    try:
        binary = configured_binary()
        configured_resource_dir(binary)
        st.success(f"Configured · {binary}")
        st.caption(f"Pinned console compatibility target · {PINNED_CONSOLE_COMMIT}")
    except (FileNotFoundError, PermissionError) as exc:
        st.warning(str(exc))
        st.caption("Open Setup and usage guide above for the complete configuration.")
    st.caption(
        "Built-in profiles are estimated study inputs. They are not solved preflop GTO ranges."
    )
    st.metric("Built-in estimated profiles", len(BUILTIN_RANGE_PROFILES))

    profiles = db.fetch_solver_range_profiles()
    st.markdown("##### Saved custom ranges")
    if profiles:
        st.dataframe(
            [
                {
                    "Name": profile.name,
                    "Position": profile.position or "Any",
                    "Scenario": profile.scenario or "Any",
                    "Pot": profile.pot_type.replace("_", " ").title() or "Any",
                    "Table": f"{profile.table_size}-max" if profile.table_size else "Any",
                    "Stack": f"{profile.stack_bb:g} BB" if profile.stack_bb else "Any",
                }
                for profile in profiles
            ],
            hide_index=True,
            width="stretch",
        )
        delete_id = st.selectbox(
            "Delete saved range",
            options=[None, *[profile.id for profile in profiles]],
            format_func=lambda profile_id: (
                "Select a range"
                if profile_id is None
                else next(profile.name for profile in profiles if profile.id == profile_id)
            ),
            key="solver_delete_profile",
        )
        if st.button(
            "Delete selected range",
            disabled=delete_id is None,
            key="solver_delete_profile_button",
        ):
            db.delete_solver_range_profile(int(delete_id))
            flash("Saved solver range deleted.")
            st.rerun()
    else:
        st.caption("No custom ranges saved yet.")

    st.download_button(
        "Export saved ranges JSON",
        data=json.dumps(export_range_profiles(profiles), indent=2),
        file_name="pokertrainer_solver_ranges.json",
        mime="application/json",
        disabled=not profiles,
    )
    uploaded = st.file_uploader(
        "Import saved ranges JSON",
        type=["json"],
        key="solver_range_import",
    )
    if uploaded is not None and st.button("Import range profiles"):
        try:
            raw = uploaded.getvalue()
            if len(raw) > MAX_IMPORT_BYTES:
                raise ValueError("Range-profile import exceeds the 10 MB limit.")
            imported = import_range_profiles(json.loads(raw))
            existing_names = {profile.name.casefold() for profile in profiles}
            collisions = [
                profile.name
                for profile in imported
                if profile.name.casefold() in existing_names
            ]
            if collisions:
                raise ValueError(
                    "These range names already exist: " + ", ".join(collisions)
                )
            with db.transaction():
                for profile in imported:
                    db.create_solver_range_profile(profile)
            flash(f"Imported {len(imported)} solver range profile(s).")
            st.rerun()
        except (json.JSONDecodeError, ValidationError, ValueError, sqlite3.IntegrityError) as exc:
            st.error(f"Could not import ranges: {exc}")


def _activate_session(session_id: int) -> None:
    """Keep page and sidebar session context synchronized."""

    st.session_state["active_session_id"] = session_id
    st.session_state["session_context_id"] = session_id


def show_session_library(
    db: PokerDatabase,
    sessions: list[Session],
    active_session: Session | None,
) -> None:
    """Render a searchable, visible session index instead of another dropdown."""

    with st.expander("Browse all sessions", expanded=active_session is None):
        search_tab, calendar_tab = st.tabs(["Search", "Calendar"])
        active_id = active_session.id if active_session is not None else None
        with calendar_tab:
            render_calendar_session_browser(
                sessions,
                key_prefix="session_library",
                active_id=active_id,
            )
        with search_tab:
            query = st.text_input(
                "Find a session",
                placeholder="Try “July 27”, “ClubWPT”, “1/2”, or a note",
                key="session_library_search",
            )
            matches = filter_sessions(sessions, query)
            if not matches:
                st.caption("No sessions match that search.")
            else:
                st.caption(f"{len(matches)} session{'s' if len(matches) != 1 else ''}")
                for item in matches[:20]:
                    if item.id is None:
                        continue
                    hands = db.fetch_hands_by_session(item.id)
                    videos = db.fetch_videos(item.id)
                    with st.container(border=True, key=f"session_card_{item.id}"):
                        detail, action = st.columns([5, 1])
                        with detail:
                            st.markdown(f"**{item.name}**")
                            st.caption(
                                f"{item.date_played.strftime('%A, %b')} {item.date_played.day} · "
                                f"{item.platform or 'Platform not set'} · "
                                f"{item.stakes or 'Stakes not set'} · "
                                f"{len(hands)} hands · {len(videos)} videos"
                            )
                        if action.button(
                            "Open"
                            if active_session is None or item.id != active_session.id
                            else "Current",
                            key=f"open_session_{item.id}",
                            type="primary"
                            if active_session is not None and item.id == active_session.id
                            else "secondary",
                            disabled=active_session is not None
                            and item.id == active_session.id,
                            width="stretch",
                        ):
                            _activate_session(item.id)
                            st.rerun()


def _open_hand_for_study(hand: Hand) -> None:
    if hand.id is None:
        return
    _activate_session(hand.session_id)
    _set_study_hand_id(hand.id)
    navigate_to(Page.STUDY)


def delete_hand_and_artifacts(db: PokerDatabase, hand_id: int) -> str | None:
    """Delete one hand after stopping and removing its solver runs.

    Returns an error message instead of deleting when an active solver cannot be
    stopped yet. This is the writer behind every 'Delete hand' control, and it
    exists because ``NEW_RECONSTRUCTION_STEPS`` names the deletion as part of its
    clearing action: an import ADDS the rebuilt hands beside the existing ones,
    so the superseded copy must be deletable from the session's hand list or the
    blocker names an action the product cannot perform.
    """
    for run in db.fetch_solver_runs_by_hand(hand_id):
        if run.status in {"queued", "running", "cancelling"}:
            cancelled = cancel_solver_run(db, run.id)
            if cancelled.status == "cancelling":
                return (
                    "The active solver could not be stopped yet. "
                    "Try deleting again after it exits."
                )
        remove_solver_run_artifacts(run)
    db.delete_hand(hand_id)
    return None


def render_hand_results(
    db: PokerDatabase,
    hands: list[Hand],
    sessions_by_id: dict[int, Session],
    *,
    key_prefix: str,
    page_size: int = 20,
) -> None:
    """Render scan-friendly hand rows with a direct Study action on every row."""

    page_key = f"{key_prefix}_hand_page"
    total_pages = max(1, (len(hands) + page_size - 1) // page_size)
    page = min(max(1, int(st.session_state.get(page_key, 1))), total_pages)
    st.session_state[page_key] = page
    start = (page - 1) * page_size

    nav_left, nav_label, nav_right = st.columns([1, 4, 1])
    if nav_left.button(
        "← Previous",
        key=f"{key_prefix}_previous_page",
        disabled=page == 1,
        width="stretch",
    ):
        st.session_state[page_key] = page - 1
        st.rerun()
    nav_label.caption(
        f"{len(hands)} hands · showing {start + 1}–{min(start + page_size, len(hands))}"
    )
    if nav_right.button(
        "Next →",
        key=f"{key_prefix}_next_page",
        disabled=page == total_pages,
        width="stretch",
    ):
        st.session_state[page_key] = page + 1
        st.rerun()

    for item in hands[start : start + page_size]:
        if item.id is None:
            continue
        session = sessions_by_id[item.session_id]
        result = "Result unknown" if item.hero_bb_won is None else f"{item.hero_bb_won:+g} BB"
        completion_label = item.completion_status.replace("_", " ").title()
        study_label = STUDY_INCLUSION_LABELS.get(
            item.study_inclusion, item.study_inclusion
        )
        with st.container(border=True, key=f"{key_prefix}_hand_{item.id}"):
            summary, action = st.columns([6, 1])
            with summary:
                st.markdown(
                    f"**Hand #{item.hand_number} · {item.hero_cards or 'Unknown cards'}**  "
                    f"`{result}`"
                )
                st.caption(
                    f"{session.name} · {session.date_played.isoformat()} · "
                    f"{item.hero_position or 'Position unknown'} · "
                    f"{item.review_status.replace('_', ' ').title()} · "
                    f"{completion_label} · {study_label} · "
                    f"{', '.join(item.tags) or 'No tags'}"
                )
            if action.button(
                "Study",
                key=f"{key_prefix}_study_{item.id}",
                type="primary",
                width="stretch",
            ):
                _open_hand_for_study(item)
                st.rerun()
            with st.expander("Study inclusion"):
                study_choice = st.radio(
                    "Include in study?",
                    STUDY_INCLUSION_OPTIONS,
                    index=STUDY_INCLUSION_OPTIONS.index(
                        item.study_inclusion
                        if item.study_inclusion in STUDY_INCLUSION_OPTIONS
                        else "auto"
                    ),
                    format_func=lambda value: STUDY_INCLUSION_LABELS[value],
                    key=f"{key_prefix}_study_inclusion_{item.id}",
                    horizontal=True,
                )
                if st.button(
                    "Save study inclusion",
                    key=f"{key_prefix}_save_study_inclusion_{item.id}",
                ):
                    try:
                        db.update_study_inclusion(item.id, study_choice)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        flash(
                            f"Hand #{item.hand_number}: "
                            f"{STUDY_INCLUSION_LABELS[study_choice]}."
                        )
                        st.rerun()
            # The control NEW_RECONSTRUCTION_STEPS depends on: comparing a
            # blocked hand against its rebuilt copy ends with deleting one of
            # them, and this row is the session's hand list where that happens.
            with st.expander("Delete hand"):
                st.warning(
                    f"Deleting hand #{item.hand_number} removes its actions, "
                    "players, settlement, reviews, issues, and solver runs. "
                    "This cannot be undone."
                )
                confirm_delete = st.checkbox(
                    "I understand this permanently deletes this hand and all "
                    "related rows.",
                    key=f"{key_prefix}_confirm_delete_{item.id}",
                )
                if st.button(
                    "Delete hand",
                    key=f"{key_prefix}_delete_{item.id}",
                    disabled=not confirm_delete,
                ):
                    error = delete_hand_and_artifacts(db, item.id)
                    if error:
                        st.error(error)
                    else:
                        flash(f"Hand #{item.hand_number} deleted.")
                        st.rerun()


def show_session_hand_browser(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    hands = _hands_with_accounting_results(db, db.fetch_hands_by_session(session.id))
    if not hands:
        empty_state(
            "No hands in this session",
            "Add one manually or reconstruct another video into this session.",
        )
    else:
        search_col, status_col = st.columns([2, 1])
        query = search_col.text_input(
            "Find within this session",
            placeholder="Cards, position, tag, notes…",
            key=f"session_hand_search_{session.id}",
        )
        review_status = status_col.segmented_control(
            "Review",
            options=["all", *REVIEW_STATUSES],
            default="all",
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"session_hand_status_{session.id}",
        )
        filtered = filter_hands(
            hands,
            {session.id: session},
            query=query,
            review_status=review_status or "all",
        )
        if filtered:
            render_hand_results(
                db,
                filtered,
                {session.id: session},
                key_prefix=f"session_{session.id}",
                page_size=15,
            )
        else:
            st.caption("No hands match those filters.")

    other_hands = [hand for hand in db.fetch_all_hands() if hand.session_id != session.id]
    if other_hands:
        with st.expander("Move an existing hand into this session"):
            st.caption(
                "Use this when a hand was filed under the wrong session. "
                "A duplicate hand number is renumbered automatically."
            )
            other_sessions = {item.id: item for item in db.fetch_sessions() if item.id is not None}
            for hand in other_hands[:15]:
                if hand.id is None:
                    continue
                source = other_sessions.get(hand.session_id)
                label_col, move_col = st.columns([5, 1])
                label_col.markdown(
                    f"**#{hand.hand_number} · {hand.hero_cards or 'Unknown cards'}**"
                )
                label_col.caption(source.name if source else "Unknown session")
                if move_col.button(
                    "Move here",
                    key=f"move_hand_{hand.id}_to_{session.id}",
                    width="stretch",
                ):
                    moved = db.move_hand_to_session(hand.id, session.id)
                    flash(f"Hand moved to {session.name} as hand #{moved.hand_number}.")
                    st.rerun()


def _save_video_upload(
    db: PokerDatabase,
    session: Session,
    *,
    key_prefix: str,
) -> None:
    ensure_data_directories()
    uploaded = st.file_uploader(
        "Add a completed-session video",
        type=["mp4", "mov", "mkv", "avi"],
        key=f"{key_prefix}_uploader",
        help="A session can contain as many recordings as you need.",
    )
    notes = st.text_area(
        "Source notes (optional)",
        height=68,
        key=f"{key_prefix}_notes",
        placeholder="Table, time range, or what this recording contains",
    )
    if uploaded is None:
        return
    if st.button(
        "Add video to this session",
        key=f"{key_prefix}_save",
        type="primary",
    ):
        try:
            validate_video_extension(uploaded.name)
            uploaded.seek(0)
            ingested = ingest_uploaded_video(uploaded, uploaded.name)
            metadata = ingested.metadata
            try:
                saved = db.create_video(
                    VideoRecord(
                        session_id=session.id,
                        original_filename=uploaded.name,
                        stored_path=str(ingested.path),
                        file_size_bytes=ingested.file_size_bytes,
                        content_sha256=ingested.content_sha256,
                        duration_seconds=metadata.duration_seconds,
                        fps=metadata.fps,
                        width=metadata.width,
                        height=metadata.height,
                        frame_count=metadata.frame_count,
                        notes=notes.strip(),
                    )
                )
            except Exception:
                ingested.path.unlink(missing_ok=True)
                raise
            flash(f"Added {saved.original_filename} to {session.name}.")
            if session.id is not None:
                st.session_state[_import_collect_panel_key(session.id)] = False
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not save video: {exc}")


def show_session_videos(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    videos = db.fetch_videos(session.id)
    section_header_with_meta(
        "Recording sources",
        "Every recording linked to this completed session.",
        f"{len(videos)} VIDEO{'S' if len(videos) != 1 else ''}",
    )
    if not videos:
        empty_state(
            "No videos attached",
            "Add the first recording below. More recordings can be added later.",
        )
    for video in videos:
        if video.id is None:
            continue
        with st.container(border=True, key=f"session_video_{video.id}"):
            detail, action = st.columns([5, 1])
            detail.markdown(f"**{video.original_filename}**")
            detail.caption(
                f"{_format_optional_seconds(video.duration_seconds)} · "
                f"{_format_bytes(video.file_size_bytes)} · "
                f"uploaded {video.uploaded_at.date().isoformat()}"
            )
            if action.button(
                "Process",
                key=f"process_video_{video.id}",
                type="primary",
                width="stretch",
            ):
                _activate_session(session.id)
                st.session_state["video_context_id"] = video.id
                navigate_to(Page.IMPORT)
                st.rerun()

    with st.expander("Add another video", expanded=not videos):
        _save_video_upload(db, session, key_prefix=f"session_{session.id}_video")

    other_videos = [video for video in db.fetch_videos() if video.session_id != session.id]
    if other_videos:
        with st.expander("Attach a video already in the library"):
            st.caption(
                "Unassigned videos can be attached; videos from another session are moved here."
            )
            session_names = {
                item.id: item.name for item in db.fetch_sessions() if item.id is not None
            }
            for video in other_videos[:15]:
                if video.id is None:
                    continue
                detail, action = st.columns([5, 1])
                detail.markdown(f"**{video.original_filename}**")
                detail.caption(
                    "Unassigned"
                    if video.session_id is None
                    else f"Currently in {session_names.get(video.session_id, 'another session')}"
                )
                if action.button(
                    "Attach",
                    key=f"attach_video_{video.id}_to_{session.id}",
                    width="stretch",
                ):
                    db.update_video_session(video.id, session.id)
                    flash(f"Attached {video.original_filename} to {session.name}.")
                    st.rerun()


def create_session_form(
    db: PokerDatabase,
    *,
    form_key: str = "create_session",
) -> None:
    with st.form(form_key, clear_on_submit=True):
        date_played = st.date_input(
            "Date played",
            value=date.today(),
            help="Defaults to today. Every session keeps a played date you can edit later.",
        )
        generated_name = date_session_name(date_played, db.fetch_sessions())
        name = st.text_input(
            "Custom name (optional)",
            placeholder=generated_name,
            help="Leave blank to use the memorable date-based name.",
        )
        st.caption(f"Default name · {generated_name}")
        platform = st.text_input("Platform", value="Manual")
        stakes = st.text_input("Stakes", placeholder="1/2 NL")
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Create session")

    if submitted:
        saved = db.create_session(
            Session(
                name=name.strip() or generated_name,
                date_played=date_played,
                platform=platform.strip() or "Manual",
                stakes=stakes.strip(),
                notes=notes.strip(),
            )
        )
        if saved.id is not None:
            _activate_session(saved.id)
        flash(f"Session created: {saved.name}.")
        st.rerun()


def edit_session_form(db: PokerDatabase, session: Session) -> None:
    """Edit session fields, including the played date, after creation."""

    if session.id is None:
        return
    with st.expander("Edit session details", expanded=False):
        with st.form(f"edit_session_{session.id}"):
            date_played = st.date_input(
                "Date played",
                value=session.date_played,
                help="Change the session date any time after creation.",
            )
            name = st.text_input("Name", value=session.name)
            platform = st.text_input("Platform", value=session.platform)
            stakes = st.text_input("Stakes", value=session.stakes)
            notes = st.text_area("Notes", value=session.notes, height=80)
            submitted = st.form_submit_button("Save session changes")
        if submitted:
            try:
                updated = db.update_session(
                    session.model_copy(
                        update={
                            "name": name.strip() or session.name,
                            "date_played": date_played,
                            "platform": platform.strip() or "Manual",
                            "stakes": stakes.strip(),
                            "notes": notes.strip(),
                        }
                    )
                )
                flash(f"Updated session: {updated.name}.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def _session_button_label(session: Session) -> str:
    return f"{session.date_played.strftime('%b')} {session.date_played.day} · {session.name}"


def render_calendar_session_browser(
    sessions: list[Session],
    *,
    key_prefix: str,
    active_id: int | None = None,
) -> None:
    """Pick a calendar day, then open a session played on that date."""

    known_dates = session_dates(sessions)
    default = date.today()
    if active_id is not None:
        active = next((item for item in sessions if item.id == active_id), None)
        if active is not None:
            default = active.date_played
    elif known_dates:
        default = max(known_dates)

    picked = st.date_input(
        "Calendar day",
        value=default,
        key=f"{key_prefix}_calendar_day",
        help="Use the calendar to find which day a session was played.",
    )
    matches = sessions_on_date(sessions, picked)
    day_label = f"{picked.strftime('%A, %b')} {picked.day}, {picked.year}"
    if matches:
        st.caption(f"{len(matches)} session{'s' if len(matches) != 1 else ''} on {day_label}")
    else:
        st.caption(f"No sessions on {day_label}.")
        return

    for item in matches:
        if item.id is None:
            continue
        if st.button(
            _session_button_label(item),
            key=f"{key_prefix}_cal_session_{item.id}",
            type="primary" if item.id == active_id else "secondary",
            disabled=item.id == active_id,
            width="stretch",
        ):
            _activate_session(item.id)
            st.rerun()


def select_session(db: PokerDatabase) -> Session | None:
    sessions = db.fetch_sessions()
    if not sessions:
        return None

    valid_ids = {session.id for session in sessions if session.id is not None}
    active_id = st.session_state.get("active_session_id")
    if active_id not in valid_ids:
        active_id = sessions[0].id
        if active_id is not None:
            _activate_session(active_id)

    recent = sessions[:6]
    if active_id is not None and all(item.id != active_id for item in recent):
        active = next(item for item in sessions if item.id == active_id)
        recent = [active, *recent[:5]]
    labels = {
        session.id: _session_button_label(session)
        for session in recent
        if session.id is not None
    }
    st.caption("RECENT SESSIONS")
    for session_id, label in labels.items():
        if st.button(
            label,
            key=f"session_context_{session_id}",
            type="primary" if session_id == active_id else "secondary",
            disabled=session_id == active_id,
            width="stretch",
        ):
            _activate_session(session_id)
            st.rerun()
    with st.expander("Calendar", expanded=False):
        render_calendar_session_browser(
            sessions,
            key_prefix="sidebar",
            active_id=active_id if isinstance(active_id, int) else None,
        )
    st.caption("Older sessions are searchable in Sessions.")
    return next(session for session in sessions if session.id == active_id)


def _choose_import_session(
    db: PokerDatabase,
    sessions: list[Session],
    current: Session | None,
) -> Session | None:
    """Let Import pick its target session, with calendar and create options."""

    if current is None and sessions:
        active_id = st.session_state.get("active_session_id")
        current = next((item for item in sessions if item.id == active_id), sessions[0])
        if current.id is not None:
            _activate_session(current.id)

    if current is None:
        return None

    with st.container(border=True, key="import_session_target"):
        title_col, change_col, new_col = st.columns([3.2, 1, 1])
        with title_col:
            st.markdown(f"**{current.name}**")
            st.caption(
                f"{current.date_played.strftime('%b')} {current.date_played.day}, "
                f"{current.date_played.year}"
                + (f" · {current.stakes}" if current.stakes else "")
                + " · target session"
            )
        change_open = change_col.toggle(
            "Change",
            value=False,
            key=f"import_change_session_toggle_{current.id}",
        )
        new_open = new_col.toggle(
            "New",
            value=False,
            key=f"import_new_session_toggle_{current.id}",
        )
        if change_open:
            recent_tab, calendar_tab = st.tabs(["Recent", "Calendar"])
            active_id = current.id
            with calendar_tab:
                render_calendar_session_browser(
                    sessions,
                    key_prefix="import_target",
                    active_id=active_id,
                )
            with recent_tab:
                for item in sessions[:12]:
                    if item.id is None:
                        continue
                    if st.button(
                        _session_button_label(item),
                        key=f"import_pick_session_{item.id}",
                        type="primary" if item.id == active_id else "secondary",
                        disabled=item.id == active_id,
                        width="stretch",
                    ):
                        _activate_session(item.id)
                        st.rerun()
        if new_open:
            create_session_form(db, form_key="create_import_target_session")
    return current


def _import_collect_panel_key(session_id: int) -> str:
    return f"import_collect_expanded_{session_id}"


def _import_collect_is_open(session_id: int, *, has_videos: bool) -> bool:
    key = _import_collect_panel_key(session_id)
    if key not in st.session_state:
        st.session_state[key] = not has_videos
    return bool(st.session_state[key])


def show_session_dashboard(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    stats = compute_session_stats(db, session.id)
    st.subheader(session.name)
    st.caption(
        f"{session.date_played} · {session.stakes or 'stakes not set'} · {session.platform or 'platform not set'}"
    )
    edit_session_form(db, session)

    winrate_help = f"{stats.average_hero_bb:+.2f} BB/hand over {stats.hands_with_result} hands with recorded results"
    if stats.bb_per_100_ci is not None:
        ci_low, ci_high = stats.bb_per_100_ci
        winrate_help += f". 95% CI: {ci_low:+.0f} to {ci_high:+.0f} bb/100 — small samples say little about a true winrate."
    with st.container(key="session_metrics"):
        first, second, third, fourth = st.columns(4)
        with first:
            kpi_card("Hands", str(stats.hand_count), "Completed hands in session")
        with second:
            result_tone = (
                "positive"
                if stats.total_hero_bb > 0
                else "negative"
                if stats.total_hero_bb < 0
                else "default"
            )
            kpi_card(
                "Hero result",
                f"{stats.total_hero_bb:+g} BB",
                "Recorded results only",
                tone=result_tone,
            )
        with third:
            kpi_card("Winrate", f"{stats.bb_per_100:+.0f} bb/100", winrate_help)
        with fourth:
            kpi_card(
                "Reviewed",
                str(stats.hands_by_review_status.get("reviewed", 0)),
                "Completed review workflow",
                tone="positive",
            )

        fifth, sixth, seventh, eighth = st.columns(4)
        with fifth:
            kpi_card(
                "Unreviewed",
                str(stats.hands_by_review_status.get("unreviewed", 0)),
                "Still in the study queue",
            )
        with sixth:
            kpi_card(
                "Needs correction",
                str(stats.hands_by_review_status.get("needs_correction", 0)),
                "Verify recorded evidence",
                tone="warning",
            )
        with seventh:
            kpi_card("Aggressive actions", str(stats.aggression_count), "Bet, raise, or all-in")
        with eighth:
            kpi_card("Passive actions", str(stats.passive_count), "Check or call")

    if stats.hand_count == 0:
        st.info("No hands recorded yet. Add hands in the Add hands tab to see session stats.")
        return

    winning_col, losing_col = st.columns(2)
    with winning_col:
        st.markdown("##### Biggest Winning Hands")
        st.dataframe(
            _hand_summary_rows(stats.biggest_winning_hands), hide_index=True, width="stretch"
        )
    with losing_col:
        st.markdown("##### Biggest Losing Hands")
        st.dataframe(
            _hand_summary_rows(stats.biggest_losing_hands), hide_index=True, width="stretch"
        )

    tags_col, actions_col = st.columns(2)
    with tags_col:
        st.markdown("##### Tag Counts")
        if stats.hands_by_tag:
            st.dataframe(
                [
                    {"Tag": tag, "Hands": count}
                    for tag, count in sorted(stats.hands_by_tag.items(), key=lambda item: -item[1])
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No tags applied yet.")
    with actions_col:
        st.markdown("##### Action Counts")
        if stats.action_counts_by_type:
            st.dataframe(
                [
                    {"Action": action, "Count": count}
                    for action, count in sorted(
                        stats.action_counts_by_type.items(), key=lambda item: -item[1]
                    )
                ],
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
            for session_hand in db.fetch_hands_by_session(session.id):
                if session_hand.id is not None:
                    for run in db.fetch_solver_runs_by_hand(session_hand.id):
                        if run.status in {"queued", "running", "cancelling"}:
                            cancelled = cancel_solver_run(db, run.id)
                            if cancelled.status == "cancelling":
                                st.error(
                                    "The active solver could not be stopped yet. "
                                    "Try deleting again after it exits."
                                )
                                return
                        remove_solver_run_artifacts(run)
            db.delete_session(session.id)
            flash(f"Session '{session.name}' deleted.")
            st.rerun()


_SOLVER_POSITIONS = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def create_hand_form(db: PokerDatabase, session_id: int | None) -> None:
    if session_id is None:
        st.error("Select a saved session before adding hands.")
        return

    existing_hands = db.fetch_hands_by_session(session_id)
    next_hand_number = max((hand.hand_number for hand in existing_hands), default=0) + 1
    used_numbers = {hand.hand_number for hand in existing_hands}

    st.caption(
        "Completed heads-up postflop spots. Blinds/preflop come from pot type; "
        "type the postflop line as `x/b3.5/c | x/b8/f` (OOP first; amounts in BB added)."
    )

    defaults = _manual_spot_defaults_controls()
    mode = st.radio(
        "Entry mode",
        ["Single hand", "Multi-hand paste"],
        horizontal=True,
        key="manual_spot_entry_mode",
    )

    if mode == "Multi-hand paste":
        _create_multi_hand_form(db, session_id, defaults, next_hand_number, used_numbers)
    else:
        _create_single_hand_form(db, session_id, defaults, next_hand_number, used_numbers)


def _manual_spot_defaults_controls() -> ManualSpotDefaults:
    with st.expander("Table defaults (applied to every hand)", expanded=False):
        pos_left, pos_right, stack_col = st.columns(3)
        with pos_left:
            hero_position = st.selectbox(
                "Hero",
                _SOLVER_POSITIONS,
                index=_SOLVER_POSITIONS.index("BB"),
                key="manual_spot_hero_pos",
            )
        with pos_right:
            villain_position = st.selectbox(
                "Villain",
                _SOLVER_POSITIONS,
                index=_SOLVER_POSITIONS.index("BTN"),
                key="manual_spot_villain_pos",
            )
        with stack_col:
            starting_stack = st.number_input(
                "Effective stack (BB)",
                min_value=1.0,
                value=100.0,
                step=1.0,
                key="manual_spot_stack",
            )

        struct_left, struct_right = st.columns(2)
        with struct_left:
            pot_type_label = st.radio(
                "Pot type",
                ["Single raised", "3-bet"],
                horizontal=True,
                key="manual_spot_pot_type",
            )
            opener_label = st.selectbox(
                "Preflop opener",
                ["Villain", "Hero"],
                index=0,
                key="manual_spot_opener",
            )
            three_bettor_label = st.selectbox(
                "3-bettor",
                ["Hero", "Villain"],
                index=0,
                key="manual_spot_three_bettor",
                help="Used only for 3-bet pots.",
            )
        with struct_right:
            open_to = st.number_input(
                "Open to (BB)",
                min_value=1.5,
                value=2.5,
                step=0.5,
                key="manual_spot_open_to",
            )
            three_bet_to = st.number_input(
                "3-bet to (BB)",
                min_value=2.0,
                value=9.0,
                step=0.5,
                key="manual_spot_three_bet_to",
                help="Used only for 3-bet pots.",
            )
            table_size = st.number_input(
                "Table size",
                min_value=5,
                max_value=8,
                value=6,
                step=1,
                key="manual_spot_table_size",
            )

    return ManualSpotDefaults(
        hero_position=hero_position,
        villain_position=villain_position,
        table_size=int(table_size),
        starting_stack=float(starting_stack),
        pot_type="three_bet" if pot_type_label == "3-bet" else "single_raised",
        opener="hero" if opener_label == "Hero" else "villain",
        three_bettor="hero" if three_bettor_label == "Hero" else "villain",
        open_to=float(open_to),
        three_bet_to=float(three_bet_to),
    )


def _create_single_hand_form(
    db: PokerDatabase,
    session_id: int,
    defaults: ManualSpotDefaults,
    next_hand_number: int,
    used_numbers: set[int],
) -> None:
    for key, default in (
        ("manual_spot_hero_cards", ""),
        ("manual_spot_board_cards", ""),
        ("manual_spot_line_draft", "x/b3.5/c"),
        ("manual_spot_notes", ""),
    ):
        if key not in st.session_state:
            st.session_state[key] = default
    # clear_on_submit=False so validation errors keep the typed line.
    with st.form("create_hand", clear_on_submit=False):
        cards_left, cards_right = st.columns(2)
        with cards_left:
            hero_cards = st.text_input(
                "Hero cards", placeholder="Ah Qs", key="manual_spot_hero_cards"
            )
        with cards_right:
            board_cards = st.text_input(
                "Board", placeholder="Qd 7s 2c", key="manual_spot_board_cards"
            )

        postflop_line = st.text_input(
            "Postflop line",
            placeholder="x/b3.5/c | x/b8/f",
            help=(
                "x=check f=fold c=call b3.5=bet r10=raise ai50=all-in. "
                "Streets: flop | turn | river. Optional h/v prefixes."
            ),
            key="manual_spot_line_draft",
        )
        st.caption(
            f"{defaults.hero_position} vs {defaults.villain_position} · "
            f"{'3-bet' if defaults.pot_type == 'three_bet' else 'SRP'} · "
            f"{defaults.starting_stack:g} BB · OOP acts first unless you prefix h/v"
        )

        outcome_left, outcome_right, notes_col = st.columns([1, 1, 2])
        with outcome_left:
            winner_label = st.selectbox("Winner", ["Hero", "Villain"], index=0)
        with outcome_right:
            hand_number = st.number_input(
                "Hand #", min_value=1, step=1, value=next_hand_number
            )
        with notes_col:
            notes = st.text_input("Notes (optional)", key="manual_spot_notes")

        submitted = st.form_submit_button("Save hand", type="primary")

    if not submitted:
        return

    if int(hand_number) in used_numbers:
        st.error(
            f"Hand #{int(hand_number)} already exists in this session. Pick a different number."
        )
        return

    actions, line_errors = parse_postflop_line(
        postflop_line,
        hero_position=defaults.hero_position,
        villain_position=defaults.villain_position,
    )
    if line_errors:
        for error in line_errors:
            st.error(error)
        return

    spot = ManualSpotInput(
        hand_number=int(hand_number),
        hero_cards=hero_cards.strip(),
        board_cards=board_cards.strip(),
        hero_position=defaults.hero_position,
        villain_position=defaults.villain_position,
        table_size=defaults.table_size,
        starting_stack=defaults.starting_stack,
        pot_type=defaults.pot_type,
        opener=defaults.opener,
        three_bettor=defaults.three_bettor,
        open_to=defaults.open_to,
        three_bet_to=defaults.three_bet_to,
        postflop_actions=actions,
        winner="hero" if winner_label == "Hero" else "villain",
        notes=notes.strip(),
    )
    errors = validate_manual_spot(spot)
    if errors:
        for error in errors:
            st.error(error)
        return

    try:
        saved_hand, accounting, warnings = save_manual_spot(db, session_id, spot)
    except (ValidationError, ValueError, LedgerError) as exc:
        st.error(f"Could not save hand: {exc}")
        return

    st.session_state["manual_spot_hero_cards"] = ""
    st.session_state["manual_spot_board_cards"] = ""
    st.session_state["manual_spot_line_draft"] = ""
    st.session_state["manual_spot_notes"] = ""
    _flash_manual_save_results([(saved_hand, accounting, warnings)])
    st.rerun()


def _create_multi_hand_form(
    db: PokerDatabase,
    session_id: int,
    defaults: ManualSpotDefaults,
    next_hand_number: int,
    used_numbers: set[int],
) -> None:
    st.caption(
        "One hand per line. Defaults above fill positions/stack/pot type. "
        "Example: `AhQs | Qd7s2c | x/b3.5/c | hero` — optional "
        "`BB vs BTN`, `SRP`/`3bet`, `open2.5`, `3b9`."
    )
    with st.form("create_hands_batch", clear_on_submit=False):
        batch_text = st.text_area(
            "Hands",
            height=180,
            placeholder=(
                "AhQs | Qd7s2c | x/b3.5/c | hero\n"
                "KdKh | Ah9c2s | x/x | c/b12/c | villain\n"
                "7h6h | BB vs BTN | Td9c2sJh | 3bet | x/b6/c | x/b14/f | hero"
            ),
            key="manual_spot_batch_text",
        )
        start_number = st.number_input(
            "First hand number",
            min_value=1,
            step=1,
            value=next_hand_number,
            help="Later lines use the next free numbers in order.",
        )
        submitted = st.form_submit_button("Save hands", type="primary")

    if not submitted:
        return

    parsed = parse_manual_spot_lines(
        batch_text,
        defaults,
        starting_hand_number=int(start_number),
    )
    if parsed.errors:
        for error in parsed.errors:
            st.error(error)
        return

    conflicts = [spot.hand_number for spot in parsed.spots if spot.hand_number in used_numbers]
    if conflicts:
        st.error(
            "Hand number"
            + ("s" if len(conflicts) != 1 else "")
            + f" already in this session: {', '.join(f'#{n}' for n in conflicts)}."
        )
        return

    try:
        results = save_manual_spots(db, session_id, parsed.spots)
    except (ValidationError, ValueError, LedgerError) as exc:
        st.error(f"Could not save hands: {exc}")
        return

    st.session_state["manual_spot_batch_text"] = ""
    _flash_manual_save_results(results)
    st.rerun()


def _flash_manual_save_results(
    results: list[tuple[Hand, AccountingReconciliation, list[str]]],
) -> None:
    ready = 0
    for saved_hand, accounting, warnings in results:
        for warning in warnings:
            st.warning(f"Hand #{saved_hand.hand_number}: {warning}")
        if _accounting_is_established(saved_hand, accounting):
            ready += 1
        elif accounting.issues:
            st.caption(
                f"Hand #{saved_hand.hand_number}: " + "; ".join(accounting.issues[:3])
            )
    if len(results) == 1:
        saved_hand, accounting, _ = results[0]
        if _accounting_is_established(saved_hand, accounting):
            flash(f"Hand #{saved_hand.hand_number} saved and ready for Study / solver.")
        else:
            flash(
                f"Hand #{saved_hand.hand_number} saved. Finish accounting on "
                "Import validation before running the solver."
            )
        return
    flash(
        f"Saved {len(results)} hands"
        + (f" ({ready} ready for Study / solver)." if ready else ".")
    )


def show_saved_hands(db: PokerDatabase, session: Session) -> None:
    # NOTE: currently unreferenced by the running app; recorded under
    # "Known non-blocking gaps" in PLAN Phase 1. It holds the only per-hand-export
    # control, so it is retained rather than deleted, and its review-status write
    # still goes through guarded_update_hand_status. The delete-hand control the
    # clearing actions depend on is NOT only here any more: every hand row
    # rendered by render_hand_results carries one.
    if session.id is None:
        return

    accounting_cache = new_accounting_cache()
    hands = _hands_with_accounting_results(
        db, db.fetch_hands_by_session(session.id), accounting_cache
    )
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
            f"{'unknown' if hand.hero_bb_won is None else f'{hand.hero_bb_won:+g} BB'} "
            f"[{hand.review_status}]"
        ):
            st.code(
                _format_persisted_hand_history(db, session, hand),
                language="text",
            )

            accounting, accounting_error = _reconcile_cached(
                db, hand.id, accounting_cache
            )
            readiness = hand_study_readiness(
                db,
                hand,
                accounting,
                accounting_error,
                user_confirmed=bool(
                    st.session_state.get(study_confirmation_key(hand, accounting), False)
                ),
            )
            render_study_readiness(readiness)
            if is_reconstructed_hand(hand):
                show_reconstruction_evidence(
                    hand, parse_completion_evidence(hand.completion_evidence)
                )
            if hand_requires_user_confirmation(hand):
                st.checkbox(
                    "I have read the evidence above and confirm this hand is correct",
                    key=study_confirmation_key(hand, accounting),
                )

            status_options, status_index = review_status_options(hand, readiness)
            status_key = f"status_{hand.id}"
            if st.session_state.get(status_key) not in status_options:
                st.session_state.pop(status_key, None)
            status = st.selectbox(
                "Review status",
                status_options,
                index=status_index,
                key=status_key,
            )
            if st.button("Update status", key=f"status_button_{hand.id}"):
                if guarded_update_hand_status(db, hand, readiness, status):
                    st.rerun()

            show_player_editor(db, players)
            show_action_editor(db, actions, players)

            for review in db.fetch_reviews_by_hand(hand.id):
                if review.is_stale:
                    # Retained history, not current coaching: this advice was
                    # generated against facts that have since been corrected.
                    st.markdown(
                        status_badge("needs_correction", label="Stale coaching · not current"),
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        review.stale_reason
                        or "Hand evidence changed after this review was saved."
                    )
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
                error = delete_hand_and_artifacts(db, hand.id)
                if error:
                    st.error(error)
                    return
                st.rerun()


def show_player_editor(
    db: PokerDatabase,
    players: list[HandPlayer],
    *,
    force_open: bool = False,
) -> None:
    editable = [player for player in players if player.id is not None]
    with _study_panel(
        "Players / seats",
        force_open=force_open,
        expanded=False,
    ):
        if not editable:
            st.caption("No players saved.")
            return
        hand_id = editable[0].hand_id
        options = {player.id: player for player in editable}
        selected_id = st.selectbox(
            "Which player needs a fix?",
            list(options),
            format_func=lambda player_id: (
                f"{options[player_id].player_name or 'Unknown'} · "
                f"{options[player_id].position or 'no seat'}"
                + (" · hero" if options[player_id].is_hero else "")
            ),
            key=f"study_edit_player_{hand_id}",
        )
        player = options[selected_id]
        with st.form(f"edit_player_{player.id}"):
            cols = st.columns([0.6, 1.2, 0.8, 1, 0.55, 1.4])
            seat_index = cols[0].number_input(
                "Seat",
                min_value=0,
                max_value=9,
                value=player.seat_index,
                placeholder="Unknown",
            )
            player_name = cols[1].text_input("Player", value=player.player_name)
            position = cols[2].selectbox(
                "Position",
                POSITIONS,
                index=(POSITIONS.index(player.position) if player.position in POSITIONS else 0),
            )
            starting_stack = cols[3].number_input(
                "Starting stack",
                min_value=0.0,
                value=player.starting_stack,
                placeholder="Unknown",
            )
            is_hero = cols[4].checkbox("Hero", value=player.is_hero)
            notes = cols[5].text_input("Notes", value=player.notes)
            correction_reason = st.text_input(
                "Correction reason",
                placeholder="What did reconstruction get wrong?",
                key=f"player_correction_reason_{player.id}",
            )
            submitted = st.form_submit_button("Update player")
        if not submitted:
            return
        if not correction_reason.strip():
            st.error("Add a correction reason so this example can be learned from.")
            return
        try:
            db.update_hand_player(
                player.model_copy(
                    update={
                        "seat_index": (None if seat_index is None else int(seat_index)),
                        "player_name": player_name.strip(),
                        "position": position,
                        "starting_stack": _optional_float(starting_stack),
                        "is_hero": is_hero,
                        "notes": notes.strip(),
                    }
                ),
                correction_notes=correction_reason,
            )
            st.rerun()
        except (sqlite3.IntegrityError, ValidationError, ValueError) as exc:
            st.error(f"Could not update player: {exc}")


def show_action_editor(
    db: PokerDatabase,
    actions: list[Action],
    players: list[HandPlayer],
    *,
    force_open: bool = False,
    issue_targets: list[FrameIssueTarget] | None = None,
    frame_context: ValidationFrameContext | None = None,
    hand_id: int | None = None,
) -> None:
    with _study_panel(
        "Edit each action",
        force_open=force_open,
        expanded=False,
    ):
        show_action_editor_contents(
            db,
            actions,
            players,
            issue_targets=issue_targets,
            frame_context=frame_context,
            hand_id=hand_id,
        )


def _action_player_options(players: list[HandPlayer]) -> dict[str, HandPlayer]:
    """Stable Who-select labels for action editing."""

    labels: dict[str, HandPlayer] = {}
    for player in players:
        base = f"{player.player_name} · {player.position or 'unknown'}"
        label = base if base not in labels else f"{base} · {player.player_key[:8]}"
        labels[label] = player
    return labels


def show_action_editor_contents(
    db: PokerDatabase,
    actions: list[Action],
    players: list[HandPlayer],
    *,
    issue_targets: list[FrameIssueTarget] | None = None,
    frame_context: ValidationFrameContext | None = None,
    hand_id: int | None = None,
) -> None:
    targets = issue_targets or []
    focus_key = (
        _validation_focus_action_key(hand_id) if hand_id is not None else None
    )
    # One-shot focus from a jump button: open once, then clear so collapse sticks.
    focus_id = st.session_state.pop(focus_key, None) if focus_key else None
    editable = [
        (index, action)
        for index, action in enumerate(actions)
        if action.id is not None
    ]
    if not editable:
        st.info("No actions saved yet. Add the missing lines below.")
    else:
        for index, action in editable:
            # Scope BOTH lookups to the row's own source frame: an unscoped
            # match can hand a row a neighbouring frame's flag, thumbnail and
            # jump button after a type correction.
            own_image = _timeline_source_image_for_action(
                action, frame_context
            ) or _slot_source_image_ignoring_identity_safe(action, frame_context)
            scoped_targets = (
                [target for target in targets if target.source_image == own_image]
                if own_image
                else targets
            )
            linked = match_db_action_to_frame_target(
                street=action.street,
                action_type=action.action_type,
                player_name=action.player_name,
                position=action.position,
                amount=action.amount,
                targets=scoped_targets,
            )
            if linked is None:
                # Frame flags belong to the frame, not the action's identity:
                # a type or amount correction must not detach the row from its
                # flagged source frame. Scope the relaxed match to THIS row's
                # own source frame — matching on identity alone across every
                # frame is ambiguous, and resolves either to nothing or to a
                # neighbouring frame's flag.
                own_targets = scoped_targets if own_image else []
                relaxed = (
                    match_db_action_to_frame_target(
                        street=action.street,
                        action_type=action.action_type,
                        player_name=action.player_name,
                        position=action.position,
                        amount=action.amount,
                        targets=own_targets,
                        identity_only=True,
                    )
                    if own_targets
                    else None
                )
                if relaxed is not None and relaxed.status == "incorrect":
                    linked = relaxed
            # Operator frame flags and machine CV reads carry distinct
            # provenance prefixes; "unreviewed" alone is progress, not an
            # issue, and stays out of the badge so real defects stand out.
            badge_parts: list[str] = []
            if linked is not None and linked.status == "incorrect":
                badge_parts.append(
                    "source frame flagged: "
                    + (", ".join(linked.issue_types) or "frame")
                )
            cv_issues = _ordered_by_severity(
                _cv_issues_for_db_action(
                    action,
                    frame_context,
                    fallback_frame_index=(
                        linked.frame_index if linked is not None else None
                    ),
                )
            )
            if cv_issues:
                cv_kinds = list(
                    dict.fromkeys(
                        issue.kind.lower()
                        for issue in cv_issues
                        if issue.kind != "Edited line"
                    )
                )
                shown = cv_kinds[:2]
                extra = len(cv_kinds) - len(shown)
                cv_text = "CV: " + " · ".join(shown)
                if extra > 0:
                    cv_text += f" · +{extra} more"
                badge_parts.append(cv_text)
            badge = " · ".join(badge_parts)
            label = study_action_label(action, index, issue_badge=badge)
            is_focused = action.id == focus_id
            show_association = linked is not None and (
                is_focused or linked.status == "incorrect"
            )
            if hand_id is not None:
                expander_key = _validation_action_expander_key(hand_id, action.id)
                if is_focused:
                    st.session_state[expander_key] = True
                elif expander_key not in st.session_state:
                    st.session_state[expander_key] = False
                action_expander = st.expander(
                    label,
                    key=expander_key,
                    on_change="rerun",
                )
            else:
                action_expander = st.expander(label, expanded=is_focused)
            with action_expander:
                if show_association and linked is not None:
                    _render_action_frame_association(
                        hand_id=hand_id,
                        action_id=action.id,
                        target=linked,
                        frame_context=frame_context,
                    )
                elif linked is not None:
                    st.caption(linked.summary())
                if cv_issues:
                    _render_action_cv_issues(
                        hand_id=hand_id,
                        action_id=action.id,
                        issues=cv_issues,
                        frame_context=frame_context,
                    )
                _render_edit_one_action(
                    db,
                    action,
                    players,
                    # Only open the stack field when an issue actually asks
                    # for a value. A row questioning whether the action
                    # happened must not invite the operator to legitimize it.
                    needs_stack_before=any(
                        _issue_requests_a_stack_value(issue)
                        for issue in cv_issues
                    ),
                    stack_value_not_supplied=any(
                        issue.kind in STACK_VALUE_KINDS
                        and not _issue_requests_a_stack_value(issue)
                        for issue in cv_issues
                    ),
                    stack_value_ruled_out=any(
                        _issue_rules_out_its_figure(issue) for issue in cv_issues
                    ),
                    action_may_not_have_happened=any(
                        issue.kind
                        in (
                            ACTION_MAY_NOT_BELONG,
                            "Seat not in the hand on this frame",
                        )
                        for issue in cv_issues
                    ),
                )

    with st.expander("Add a missing action", expanded=not editable):
        _show_add_corrected_action(db, players)


def _timeline_source_image_for_action(
    action: Action,
    frame_context: ValidationFrameContext | None,
) -> str | None:
    """Which frame this row came from, regardless of any later correction.

    Prefers the frame stored on the row at import. The slot lookup below is
    the fallback for rows saved before schema 16 (and for manual hands), and
    it follows the row's CURRENT street and order — so it goes wrong exactly
    when the operator moves a row, which is what the stored value fixes.
    """

    if frame_context is None:
        return None
    if action.source_image:
        return action.source_image
    return timeline_source_image_for_slot(
        frame_context.timeline_hand,
        street=action.street,
        action_index=action.action_index,
        position=action.position,
        player_name=action.player_name,
    )


def _slot_source_image_ignoring_identity(
    action: Action,
    frame_context: ValidationFrameContext,
) -> str | None:
    """Source frame for a row whose actor was reassigned.

    Used only for seat-independent frame facts (coverage gaps): those describe
    the frame itself, so losing them because the operator renamed a player is
    pure signal loss. Never used to attribute a seat-specific read.
    """

    return timeline_source_image_for_slot(
        frame_context.timeline_hand,
        street=action.street,
        action_index=action.action_index,
        position="",
        player_name="",
    )


# Phrases that mark a message as CONDEMNING the figure it names, rather than
# simply not offering one. Only these may produce the "do not copy" caption.
_RULED_OUT_MARKERS = (
    "cannot confirm itself",
    "not the stack before this action",
    "is the stack AFTER this action",
    "put chips in since then",
)


def _issue_rules_out_its_figure(issue: ActionCvIssue) -> bool:
    """Whether this message tells the operator its own figure is wrong."""

    if issue.kind == "Stack before looks post-action":
        return True
    return issue.kind in STACK_VALUE_KINDS and any(
        marker in issue.detail for marker in _RULED_OUT_MARKERS
    )


def _issue_requests_a_stack_value(issue: ActionCvIssue) -> bool:
    """Whether this issue hands the operator a value to type into the field.

    Reads the flag the issue sets, never its prose: inferring this from
    wording is how the caption came to tell the operator to re-enter the
    figure the warning directly above it had just condemned.
    """

    return issue.kind in STACK_VALUE_KINDS and issue.offers_a_value


def _slot_source_image_ignoring_identity_safe(
    action: Action,
    frame_context: ValidationFrameContext | None,
) -> str | None:
    """Identity-free slot lookup that tolerates a missing frame context."""

    if frame_context is None:
        return None
    if action.source_image:
        return action.source_image
    return _slot_source_image_ignoring_identity(action, frame_context)


def _frame_index_for_image(
    image: str | None,
    frame_context: ValidationFrameContext | None,
) -> int | None:
    """Position of a source image in the hand's frame list, for jump wiring."""

    if not image or frame_context is None:
        return None
    return next(
        (
            index
            for index, state in enumerate(frame_context.states)
            if str(state.get("image") or "") == image
        ),
        None,
    )


def _seat_index_for_action(
    action: Action,
    frame_context: ValidationFrameContext,
) -> int | None:
    """Seat number for a saved row, via the timeline hand's player list.

    Player name is the stronger key, so it is resolved across every player
    before falling back to position — otherwise an earlier player's matching
    position outranks a later player's matching name.
    """

    players = frame_context.timeline_hand.get("players") or []
    for player in players:
        name = str(player.get("player_name") or "")
        if action.player_name and name and name == action.player_name:
            seat = player.get("seat")
            return None if seat is None else int(seat)
    for player in players:
        position = str(player.get("position") or "")
        if action.position and position and position == action.position:
            seat = player.get("seat")
            return None if seat is None else int(seat)
    return None


def _cv_issues_for_db_action(
    action: Action,
    frame_context: ValidationFrameContext | None,
    *,
    fallback_frame_index: int | None = None,
) -> list[ActionCvIssue]:
    """CV read failures behind one saved action line, or [] without a timeline.

    Provenance decides which reconstructed line this row came from. The slot
    match is only a fallback for rows saved before schema 16: it follows the
    row's CURRENT street and order, so after a correction it can land on a
    different line — and that line's amounts and stacks belong to real frames,
    which is what makes borrowing them dangerous.
    """

    if frame_context is None:
        return []
    # The seat is resolved from the row's CURRENT actor, so an actor
    # correction moves this key. Accept a frame+seat hit only when it is
    # unambiguous: the frame carries a single reconstructed line, or the row
    # still occupies that line's slot (which an actor-only edit preserves).
    seat = _seat_index_for_action(action, frame_context)
    origin = timeline_action_by_frame_and_seat(
        frame_context.timeline_hand, action.source_image, seat
    )
    if origin is not None and not _origin_is_unambiguous(
        action, origin, frame_context
    ):
        origin = None
    detached = False
    if origin is None:
        origin = match_db_action_to_timeline_action(
            frame_context.timeline_hand,
            street=action.street,
            action_index=action.action_index,
            action_type=action.action_type,
            position=action.position,
            player_name=action.player_name,
        )
        # A stored frame that disagrees with the slot match means the row was
        # moved: the slot now holds someone else's line.
        if (
            origin is not None
            and action.source_image
            and str(origin.get("source_image") or "") != action.source_image
        ):
            origin = None
    else:
        # Keyed by frame and seat, so it stays correct through any edit. It is
        # "detached" only when the row no longer occupies that line's slot.
        detached = not _row_still_matches_origin(action, origin)

    if origin is None:
        issues = _issues_for_unattributable_row(
            action, frame_context, fallback_frame_index
        )
        retired = _retired_check_notice(action, frame_context, issues)
        if retired is not None:
            issues.append(retired)
        return issues

    issues = cv_issues_for_timeline_action(
        origin,
        frame_context.timeline_hand,
        frame_context.states,
        db_amount=action.amount,
        db_stack_before=action.stack_before,
        recording_start_s=frame_context.recording_start_s,
        db_street=action.street,
        db_action_type=action.action_type,
    )
    retired = _claims_retired_by_the_edit(action, origin, frame_context, issues)
    if retired is not None:
        issues.append(retired)
    if detached and issues:
        # Never let a correction silently clear a live warning: say the
        # warnings are re-derived from the frame this row came from.
        note = (
            "This line no longer matches the reconstructed line it came "
            "from, so the following is re-derived from its source frame."
        )
        issues = [
            ActionCvIssue(kind="Edited line", detail=note, frame_index=None),
            *issues,
        ]
    return issues


def _origin_is_unambiguous(
    action: Action,
    origin: dict[str, object],
    frame_context: ValidationFrameContext,
) -> bool:
    """Whether a frame+seat hit really identifies THIS row's origin.

    A frame usually carries several reconstructed lines, so after an actor
    correction the seat key can land on a different seat's line and lend it
    its amounts. Require either a frame with a single line, or agreement on
    the slot the row still occupies.

    This is deliberately conservative and fails safe: a row that was moved AND
    whose frame carries other lines falls back to the unattributable path,
    which still warns but explains less. Storing the seat alongside the frame
    would let attribution survive both edits at once.
    """

    if origin.get("action_index") == action.action_index and str(
        origin.get("street") or ""
    ).lower() == action.street.lower():
        return True
    same_frame = [
        line
        for line in frame_context.timeline_hand.get("actions") or []
        if str(line.get("source_image") or "") == action.source_image
    ]
    return len(same_frame) == 1


def _retired_check_notice(
    action: Action,
    frame_context: ValidationFrameContext,
    issues: list[ActionCvIssue],
) -> ActionCvIssue | None:
    """Warn when a row that the frames used to object to now says nothing.

    Provenance-independent on purpose: the rows the backfill legitimately
    refuses are exactly the ones that lose their origin on an edit, and going
    quiet reads as resolved while the objected-to value is still stored.
    """

    origin = timeline_action_by_frame_and_seat(
        frame_context.timeline_hand,
        _timeline_source_image_for_action(action, frame_context)
        or _slot_source_image_ignoring_identity(action, frame_context),
        _seat_index_for_action(action, frame_context),
    )
    if origin is None:
        return None
    return _claims_retired_by_the_edit(
        action, origin, frame_context, issues, only_if_silent=True
    )


def _claims_retired_by_the_edit(
    action: Action,
    origin: dict[str, object],
    frame_context: ValidationFrameContext,
    issues: list[ActionCvIssue],
    *,
    only_if_silent: bool = False,
) -> ActionCvIssue | None:
    """Warn when an edit stopped a check that had been flagging a saved value.

    Retyping a money action to a fold stops the amount and stack checks from
    running, but the value they were objecting to is still stored. Going quiet
    reads as resolved, so say which check no longer applies and what it had
    flagged.
    """

    kinds_now = {issue.kind for issue in issues}
    if only_if_silent and kinds_now - {"Frame checks no longer apply"}:
        return None
    before = cv_issues_for_timeline_action(
        origin,
        frame_context.timeline_hand,
        frame_context.states,
        db_amount=action.amount,
        db_stack_before=action.stack_before,
        recording_start_s=frame_context.recording_start_s,
        db_street=str(origin.get("street") or action.street),
        db_action_type=str(origin.get("action_type") or action.action_type),
    )
    lost = [
        issue
        for issue in before
        if issue.kind not in kinds_now
        and issue.kind in {"Stack before looks post-action", "Amount unknown"}
    ]
    if not lost:
        return None
    kinds = sorted({issue.kind.lower() for issue in lost})
    names = " and ".join(f'"{kind}"' for kind in kinds)
    verb = "check no longer runs" if len(kinds) == 1 else "checks no longer run"
    saved = (
        f" Its saved Stack before ({action.stack_before:g} BB) was what that "
        "check objected to."
        if action.stack_before is not None
        and any(i.kind == "Stack before looks post-action" for i in lost)
        else ""
    )
    return ActionCvIssue(
        kind="Check no longer applies",
        detail=(
            f"This line's type changed, so the import's {names} {verb} on "
            f"it — but nothing it flagged was corrected.{saved} "
            "Re-check this row against its frames yourself."
        ),
    )


def _row_is_the_same_line(action: Action, origin: dict[str, object]) -> bool:
    """Whether this row is still the line it came from, ignoring its order.

    Reordering does not change which line a row IS, so its own derivation
    still applies — losing it silently dropped the stack warning. Actor and
    type must still agree: those are what a borrowed derivation would
    misdescribe.
    """

    return (
        action.street.lower() == str(origin.get("street", "")).lower()
        and action.action_type.replace("-", "_")
        == str(origin.get("action_type") or "").replace("-", "_")
        and action.player_name == str(origin.get("player_name") or "")
    )


def _row_still_matches_origin(action: Action, origin: dict[str, object]) -> bool:
    """Whether a saved row still claims the same street, order, type and actor."""

    return (
        action.street.lower() == str(origin.get("street", "")).lower()
        and action.action_index == origin.get("action_index")
        and action.action_type.replace("-", "_")
        == str(origin.get("action_type") or "").replace("-", "_")
        and action.player_name == str(origin.get("player_name") or "")
    )


# Badge order: a row can only show two kinds, so a claim that a saved value
# is wrong must outrank a hedge about where the row now sits.
_BADGE_RANK = {
    ACTION_MAY_NOT_BELONG: 0,
    "Seat not in the hand on this frame": 1,
    "Stack before looks post-action": 2,
    "Amount unknown": 3,
    "Stack before unknown": 4,
    "Unmeasured transition": 5,
    "Coverage gap": 6,
    "Moved off its source street": 7,
    "Frame checks no longer apply": 8,
    "Check no longer applies": 2,
    "Edited line": 9,
}


def _issue_badge_rank(issue: ActionCvIssue) -> int:
    return _BADGE_RANK.get(issue.kind, 5)


def _ordered_by_severity(issues: list[ActionCvIssue]) -> list[ActionCvIssue]:
    """One order for the badge and the expanded body, so they never disagree.

    The badge shows two kinds; the body shows all of them. Sorting only the
    badge meant the item it promoted could appear last when expanded.
    """

    edited = [issue for issue in issues if issue.kind == "Edited line"]
    rest = sorted(
        (issue for issue in issues if issue.kind != "Edited line"),
        key=_issue_badge_rank,
    )
    return [*edited, *rest]


def _street_depicted_by_frame(
    frame_context: ValidationFrameContext,
    own_image: str | None,
) -> str:
    """The street a frame shows, from its board size."""

    if not own_image:
        return ""
    state = next(
        (
            candidate
            for candidate in frame_context.states
            if str(candidate.get("image") or "") == own_image
        ),
        None,
    )
    if state is None:
        return ""
    board = state.get("board_cards")
    if not isinstance(board, list):
        return ""
    return STREET_BY_BOARD_COUNT.get(len(board), "").lower()


def _frame_phrase(frame_index: int | None, action: Action) -> str:
    """Name a frame with only the confidence its provenance supports."""

    if frame_index is None:
        return "its source frame"
    if action.source_image:
        return f"frame {frame_index + 1}, the frame this line came from"
    return (
        f"frame {frame_index + 1}, the closest frame the reconstruction can "
        "attribute to this line"
    )


def _frame_bet_record(
    frame_context: ValidationFrameContext,
    own_image: str | None,
    action: Action,
) -> tuple[float | None, str]:
    """What the reader recorded for this seat's bet box: a value, or a refusal.

    A refusal is still the reader saying it saw a box, so denying one is as
    wrong as denying a read value.
    """

    if not own_image:
        return None, ""
    state = next(
        (
            candidate
            for candidate in frame_context.states
            if str(candidate.get("image") or "") == own_image
        ),
        None,
    )
    if state is None:
        return None, ""
    seat = _seat_index_for_action(action, frame_context)
    value = seat_value(state.get("bets"), seat)
    if value is not None:
        return value, ""
    code = seat_refusal_code(state.get("bets_unknown"), seat)
    return None, (unknown_read_text(code) if code else "")


def _issues_for_unattributable_row(
    action: Action,
    frame_context: ValidationFrameContext,
    fallback_frame_index: int | None,
) -> list[ActionCvIssue]:
    """Rows the reconstruction can no longer be tied to: added, or re-pointed.

    The row's reads no longer apply, but its source frame usually still
    resolves — so the frame's own evidence (missing stacks, coverage gaps,
    unmeasured transitions, a seat holding no cards) is re-derived from a stub
    carrying only THIS row's values. Nothing is copied from another line,
    which is what makes re-deriving safe rather than a fresh borrow.
    """

    own_image = _timeline_source_image_for_action(
        action, frame_context
    ) or _slot_source_image_ignoring_identity(action, frame_context)
    own_index = (
        _frame_index_for_image(own_image, frame_context)
        if own_image
        else fallback_frame_index
    )
    issues: list[ActionCvIssue] = []
    if (
        action.amount is None
        and action.action_type.replace("-", "_") in MONEY_ACTION_TYPES
    ):
        if _row_frame_shows_no_cards(action, frame_context, own_image):
            # Name the frame with the confidence it was actually earned, and
            # never deny a bet box the reader recorded a value for.
            box, box_refused = _frame_bet_record(frame_context, own_image, action)
            if box is not None:
                box_note = (
                    f", though the reader did read {box:g} BB in its bet box "
                    "there — that total belongs to an earlier action on this "
                    "street"
                )
            elif box_refused:
                box_note = (
                    ", though the reader did see a bet box there and declined "
                    f"to read it ({box_refused})"
                )
            else:
                box_note = ", so there was no bet box to read"
            where = (
                f"{_frame_phrase(own_index, action).capitalize()} shows no "
                f"cards for this seat{box_note}. Confirm this action happened "
                "before entering an amount for it."
            )
        elif own_index is None:
            where = "Read the amount off the frames and enter it below."
        else:
            where = (
                f"{_frame_phrase(own_index, action).capitalize()} — open it, "
                "read the chips this seat added, and enter that below."
            )
        issues.append(
            ActionCvIssue(
                kind="Amount unknown",
                detail=(
                    "This line was edited or added, so the import's amount "
                    "read no longer applies to it — and its amount is still "
                    f"empty. {where}"
                ),
                frame_index=own_index,
            )
        )
    frame_evidence = _frame_evidence_for_row(action, frame_context, own_image)
    issues.extend(frame_evidence)
    needs_amount = action.action_type.replace("-", "_") in MONEY_ACTION_TYPES
    if (
        not action.source_image
        and not frame_evidence
        and (action.stack_before is None or (needs_amount and action.amount is None))
    ):
        # Only when no frame-derived check actually ran: saying they cannot be
        # applied while two of them are applied above is a contradiction.
        fields = "Stack before" if not needs_amount else "Amount and Stack before"
        issues.append(
            ActionCvIssue(
                kind="Frame checks no longer apply",
                detail=(
                    "No source frame was recorded for this line and it no "
                    "longer matches any reconstructed line, so the import's "
                    f"frame-based checks cannot be applied to it. Check its "
                    f"{fields} against the frames yourself."
                ),
            )
        )
    return issues


def _row_frame_shows_no_cards(
    action: Action,
    frame_context: ValidationFrameContext,
    own_image: str | None,
) -> bool:
    """Whether the row's own frame shows this seat holding nothing."""

    if not own_image:
        return False
    state = next(
        (
            candidate
            for candidate in frame_context.states
            if str(candidate.get("image") or "") == own_image
        ),
        None,
    )
    if state is None:
        return False
    # A hand's last retained frame has already cleared the table, so absence
    # there is not evidence — the same guard the phantom check applies.
    if frame_context.states and state is frame_context.states[-1]:
        return False
    return not seat_holds_cards(state, _seat_index_for_action(action, frame_context))


def _frame_evidence_for_row(
    action: Action,
    frame_context: ValidationFrameContext,
    own_image: str | None,
) -> list[ActionCvIssue]:
    """Everything the row's own frame still says about it, minus its reads."""

    if not own_image:
        return []
    seat = _seat_index_for_action(action, frame_context)
    neighbour = timeline_action_by_frame_and_seat(
        frame_context.timeline_hand, own_image, seat
    )
    stub = {
        "source_image": own_image,
        "seat": seat,
        # Needed by the bet-box ownership guard, which compares this line's
        # position on the street against the seat's earlier ones.
        "action_index": (neighbour or {}).get("action_index", action.action_index),
        # The street the LINE was reconstructed on, so a moved row still gets
        # hedged here; falls back to the row's own street when unknown.
        "street": str(
            (neighbour or {}).get("street")
            or _street_depicted_by_frame(frame_context, own_image)
            or action.street
        ),
        "action_type": action.action_type,
        # Deliberately no amount or stack_before: this row has no
        # reconstructed reads any more, and copying a neighbour's would be
        # the borrowing this whole path exists to prevent.
        "amount": None,
        "stack_before": None,
        # Only the row's OWN line may lend its derivation: frames carry lines
        # with different derivations, so a neighbour's would accuse an
        # observed row of having been inferred.
        "derivation": (
            str((neighbour or {}).get("derivation") or "")
            if neighbour is not None and _row_is_the_same_line(action, neighbour)
            else ""
        ),
    }
    issues = [
        issue
        for issue in cv_issues_for_timeline_action(
            stub,
            frame_context.timeline_hand,
            frame_context.states,
            db_amount=action.amount,
            db_stack_before=action.stack_before,
            recording_start_s=frame_context.recording_start_s,
            db_street=action.street,
            db_action_type=action.action_type,
        )
        if issue.kind != "Amount unknown"
    ]
    if _row_frame_shows_no_cards(action, frame_context, own_image):
        # Whatever else this frame says, it does not show this seat in the
        # hand — so nothing derived from it may be offered as a value to type,
        # and the editor must not open the field to invite one.
        issues = [
            issue.__class__(
                kind=issue.kind,
                detail=issue.detail,
                frame_index=issue.frame_index,
                offers_a_value=False,
            )
            for issue in issues
        ]
        if not any(
            issue.kind in (ACTION_MAY_NOT_BELONG, "Seat not in the hand on this frame")
            for issue in issues
        ):
            index = _frame_index_for_image(own_image, frame_context)
            issues.insert(
                0,
                ActionCvIssue(
                    kind="Seat not in the hand on this frame",
                    detail=(
                        f"{_frame_phrase(index, action).capitalize()} shows no "
                        "cards for this seat, so it was not in the hand there. "
                        "Check that frame before entering anything for this "
                        "line."
                    ),
                    frame_index=index,
                ),
            )
    if not any(issue.kind == ACTION_MAY_NOT_BELONG for issue in issues):
        # The evidence is (frame, seat), both of which survive every edit, so
        # it must not depend on a derivation the edit knocked out. Four
        # consecutive rounds found an edit silently clearing this accusation.
        stranded = _stranded_seat_issue(action, frame_context, own_image)
        if stranded is not None:
            # It supersedes the weaker seat-absence notice rather than joining
            # it: one frame fact stated twice consumes both badge slots.
            issues = [
                issue
                for issue in issues
                if issue.kind != "Seat not in the hand on this frame"
            ]
            issues.insert(0, stranded)
    return issues


def _stranded_seat_issue(
    action: Action,
    frame_context: ValidationFrameContext,
    own_image: str,
) -> ActionCvIssue | None:
    """Flag an edited row whose source frame shows its seat holding nothing.

    Folds are excluded: a folding seat loses its cards on its own frame, so
    absence there is the expected observation rather than evidence against it.
    """

    if action.action_type.replace("-", "_") == "fold":
        return None
    if not action.source_image:
        # Without recorded provenance the frame was inferred from the row's
        # current street and order, so it moves with the edit. Recommending a
        # delete on evidence the edit produced would destroy real rows.
        return None
    if not _row_frame_shows_no_cards(action, frame_context, own_image):
        return None
    index = _frame_index_for_image(own_image, frame_context)
    where = f"frame {index + 1}" if index is not None else "its source frame"
    # Only claim origin when provenance actually recorded it; otherwise this
    # frame is merely the closest the reconstruction can attribute — and this
    # is the message that instructs a delete.
    opening = (
        f"The frame this line came from ({where})"
        if action.source_image
        else f"{where.capitalize()}, the closest frame the reconstruction can "
        "attribute to this line,"
    )
    seat = _seat_index_for_action(action, frame_context)
    origin = timeline_action_by_frame_and_seat(
        frame_context.timeline_hand, own_image, seat
    )
    if str((origin or {}).get("action_type") or "").replace("-", "_") == "fold":
        # The frame records a fold; retyping the row does not change that, and
        # a folding seat loses its cards on its own frame.
        return None
    held_earlier = _seat_held_cards_earlier(frame_context, own_image, seat)
    consequence = (
        "so it had already left the hand and cannot have acted here"
        if held_earlier
        else "and no frame in this hand shows it holding any — it was either "
        "never dealt in, or had already folded before the first retained "
        "frame"
    )
    return ActionCvIssue(
        kind=ACTION_MAY_NOT_BELONG,
        detail=(
            f"{opening} shows no cards for this seat, {consequence} — "
            "whatever this row now says. Check that frame and delete this "
            "line if the action did not happen."
        ),
        frame_index=index,
    )


def _seat_held_cards_earlier(
    frame_context: ValidationFrameContext,
    own_image: str,
    seat: int | None,
) -> bool:
    """Whether any frame before this one shows the seat holding cards."""

    index = _frame_index_for_image(own_image, frame_context)
    if index is None or seat is None:
        return False
    return any(
        seat_holds_cards(state, seat)
        for state in frame_context.states[:index]
    )


def _render_action_cv_issues(
    *,
    hand_id: int | None,
    action_id: int,
    issues: list[ActionCvIssue],
    frame_context: ValidationFrameContext | None,
) -> None:
    """List the CV read failures behind this action, with a jump to their frame."""

    st.warning("\n".join(f"- **{issue.kind}** — {issue.detail}" for issue in issues))
    if frame_context is None or hand_id is None:
        return
    jump_indexes = list(
        dict.fromkeys(
            issue.frame_index for issue in issues if issue.frame_index is not None
        )
    )
    for frame_index in jump_indexes:
        if st.button(
            f"Jump to frame {frame_index + 1}",
            key=f"cv_issue_jump_{hand_id}_{action_id}_{frame_index}",
            width="stretch",
        ):
            _jump_to_frame(frame_context, frame_index)
            st.rerun()


def _render_action_frame_association(
    *,
    hand_id: int | None,
    action_id: int,
    target: FrameIssueTarget,
    frame_context: ValidationFrameContext | None,
) -> None:
    """Show the source frame and issue types that produced this action line."""

    visual_col, text_col = st.columns([0.75, 2.25], gap="small")
    with visual_col:
        if Path(target.source_image).is_file():
            st.image(target.source_image, width=132)
        else:
            st.caption(f"Frame {target.frame_index + 1}")
    with text_col:
        st.caption(target.summary())
        if target.notes:
            st.caption(f"Correction note: {target.notes}")
        if frame_context is not None and hand_id is not None:
            if st.button(
                f"Jump to frame {target.frame_index + 1}",
                key=f"action_jump_frame_{hand_id}_{action_id}_{target.frame_index}",
                width="stretch",
            ):
                _jump_to_frame(frame_context, target.frame_index)
                st.rerun()


def _render_edit_one_action(
    db: PokerDatabase,
    action: Action,
    players: list[HandPlayer],
    *,
    needs_stack_before: bool = False,
    stack_value_not_supplied: bool = False,
    stack_value_ruled_out: bool = False,
    action_may_not_have_happened: bool = False,
) -> None:
    """Simple per-action editor: Who / What / Amount first; advanced fields optional.

    ``needs_stack_before`` opens the advanced block by default, because a CV
    issue on this row hands the operator a value to enter there.
    ``stack_value_not_supplied`` means a stack issue fired but gave no usable
    figure, so the block still opens but must not claim one was named.
    ``action_may_not_have_happened`` keeps the block shut outright: a row
    whose frame shows its seat was not in the hand must not be handed a field
    to legitimize it with, whatever else fired alongside.
    """

    if action.id is None:
        return
    player_options = _action_player_options(players)
    if not player_options:
        st.warning("Add the hand's players before editing actions.")
        return

    labels = list(player_options)
    current_label = next(
        (
            label
            for label, player in player_options.items()
            if player.player_key == action.player_key
        ),
        next(
            (
                label
                for label, player in player_options.items()
                if player.player_name == action.player_name
                and (
                    not action.position
                    or player.position == action.position
                )
            ),
            labels[0],
        ),
    )
    advanced_key = f"action_advanced_{action.id}"
    if (
        (needs_stack_before or stack_value_not_supplied or stack_value_ruled_out)
        and not action_may_not_have_happened
        and advanced_key not in st.session_state
    ):
        # A warning on this row asks for Stack before, which lives here.
        st.session_state[advanced_key] = True
    show_advanced = st.checkbox(
        "More fields (order, pot/stack before, forced posts)",
        key=advanced_key,
    )

    with st.form(f"edit_action_{action.id}"):
        street_col, who_col, did_col, amount_col = st.columns(4)
        street = street_col.selectbox(
            "Street",
            STREETS,
            index=STREETS.index(action.street) if action.street in STREETS else 0,
        )
        who_label = who_col.selectbox(
            "Who",
            labels,
            index=labels.index(current_label),
        )
        action_type = did_col.selectbox(
            "Action",
            ACTION_TYPES,
            index=(
                ACTION_TYPES.index(action.action_type)
                if action.action_type in ACTION_TYPES
                else 0
            ),
        )
        amount = amount_col.number_input(
            "Amount (BB)",
            min_value=0.0,
            value=action.amount,
            placeholder=(
                "Unknown — enter it"
                if action.action_type.replace("-", "_") in MONEY_ACTION_TYPES
                else "Not applicable"
            ),
        )

        action_index = action.action_index or 1
        notes = action.notes or ""
        semantics = action.amount_semantics
        pot_before = action.pot_before
        stack_before = action.stack_before
        forced_bet_type = action.forced_bet_type or ""
        live_post_value = (
            "unspecified"
            if action.is_live_post is None
            else "live"
            if action.is_live_post
            else "dead"
        )
        if show_advanced:
            if needs_stack_before:
                caption = (
                    "Stack before is requested by a warning on this action — "
                    "fill it in from the frame that warning names."
                )
            elif stack_value_ruled_out:
                caption = (
                    "The warning above rules out the figure it names — work "
                    "the value out from the frames before entering it here."
                )
            elif stack_value_not_supplied:
                caption = (
                    "A warning on this action concerns Stack before. Follow "
                    "its instruction: read the frame it names and enter what "
                    "you see."
                )
            else:
                caption = (
                    "Usually leave these alone unless the ledger still fails."
                )
            st.caption(caption)
            order_col, notes_col = st.columns([1, 2])
            action_index = order_col.number_input(
                "Order",
                min_value=1,
                value=int(action.action_index or 1),
                step=1,
            )
            notes = notes_col.text_input("Notes", value=action.notes or "")
            semantics_col, pot_col, stack_col = st.columns(3)
            semantics = semantics_col.selectbox(
                "Amount meaning",
                ["incremental", "raise_to", "unknown"],
                index=["incremental", "raise_to", "unknown"].index(
                    action.amount_semantics
                    if action.amount_semantics
                    in {"incremental", "raise_to", "unknown"}
                    else "unknown"
                ),
                help=(
                    "Incremental = chips added now. "
                    "Raise-to = total committed this street."
                ),
            )
            pot_before = pot_col.number_input(
                "Pot before (BB)",
                min_value=0.0,
                value=action.pot_before,
            )
            stack_before = stack_col.number_input(
                "Stack before (BB)",
                min_value=0.0,
                value=action.stack_before,
            )
            forced_options = [
                "",
                "small_blind",
                "big_blind",
                "ante",
                "big_blind_ante",
                "straddle",
                "dead_blind",
                "bring_in",
            ]
            post_col, status_col = st.columns(2)
            forced_bet_type = post_col.selectbox(
                "Forced post",
                forced_options,
                index=(
                    forced_options.index(action.forced_bet_type)
                    if action.forced_bet_type in forced_options
                    else 0
                ),
            )
            live_post_value = status_col.selectbox(
                "Post status",
                ["unspecified", "live", "dead"],
                index=(
                    0
                    if action.is_live_post is None
                    else 1
                    if action.is_live_post
                    else 2
                ),
            )

        correction_reason = st.text_input(
            "Why are you changing this?",
            placeholder="e.g. video shows a call, not a fold",
            key=f"action_correction_reason_{action.id}",
        )
        save_col, delete_col = st.columns(2)
        submitted_update = save_col.form_submit_button("Save", type="primary")
        submitted_delete = delete_col.form_submit_button("Delete this action")

    player = player_options[who_label]
    if submitted_update:
        if not correction_reason.strip():
            st.error("Say briefly why this line is wrong so the change is auditable.")
            return
        try:
            db.update_action(
                Action(
                    id=action.id,
                    hand_id=action.hand_id,
                    player_key=player.player_key,
                    street=street,
                    action_index=int(action_index),
                    player_name=player.player_name,
                    position=player.position,
                    action_type=action_type,
                    amount=amount,
                    amount_semantics=semantics,
                    forced_bet_type=forced_bet_type or None,
                    is_live_post=(
                        None
                        if live_post_value == "unspecified"
                        else live_post_value == "live"
                    ),
                    pot_before=pot_before,
                    stack_before=stack_before,
                    notes=notes,
                ),
                correction_notes=correction_reason,
            )
        except (ValidationError, ValueError) as exc:
            st.error(f"Could not update action: {exc}")
        else:
            flash("Action updated.")
            st.rerun()
    if submitted_delete:
        if not correction_reason.strip():
            st.error("Say briefly why you are deleting this action.")
            return
        db.delete_action(action.id, correction_notes=correction_reason)
        flash("Action deleted.")
        st.rerun()


def _show_add_corrected_action(db: PokerDatabase, players: list[HandPlayer]) -> None:
    if not players:
        st.warning("Add the hand's players before adding a corrected action.")
        return
    player_options = _action_player_options(players)
    with st.form(f"add_corrected_action_{players[0].hand_id}"):
        street_col, who_col, did_col, amount_col = st.columns(4)
        street = street_col.selectbox("Street", STREETS)
        who_label = who_col.selectbox("Who", list(player_options))
        action_type = did_col.selectbox("Action", ACTION_TYPES)
        amount = amount_col.number_input(
            "Amount (BB)", min_value=0.0, value=None, placeholder="Not applicable"
        )
        correction_reason = st.text_input(
            "Why are you adding this?",
            placeholder="e.g. missed Hero call between the flop bet and turn",
        )
        submitted = st.form_submit_button("Add action", type="primary")
    if submitted:
        if not correction_reason.strip():
            st.error("Say briefly why this action was missing.")
            return
        player = player_options[who_label]
        try:
            db.create_corrected_action(
                Action(
                    hand_id=player.hand_id,
                    player_key=player.player_key,
                    street=street,
                    player_name=player.player_name,
                    position=player.position,
                    action_type=action_type,
                    amount=_optional_float(amount),
                    amount_semantics="unknown",
                    notes="",
                ),
                correction_notes=correction_reason,
            )
        except (sqlite3.IntegrityError, ValidationError, ValueError) as exc:
            st.error(f"Could not add action: {exc}")
        else:
            flash("Action added.")
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
    accounting_cache = new_accounting_cache()
    hands = _hands_with_accounting_results(
        db, db.fetch_hands_by_session(session.id), accounting_cache
    )
    if not hands:
        st.info("Save or load hands before using Math Review.")
        return

    labels = {
        hand.id: (
            f"Hand #{hand.hand_number}: {hand.hero_cards or 'unknown'} "
            f"({'unknown' if hand.hero_bb_won is None else f'{hand.hero_bb_won:+g} BB'})"
        )
        for hand in hands
        if hand.id is not None
    }
    hand_col, range_col, baseline_col = st.columns([1.1, 0.75, 1.4])
    selected_hand_id = hand_col.selectbox(
        "Select hand",
        options=list(labels.keys()),
        format_func=lambda hand_id: labels[hand_id],
    )
    hand = next(item for item in hands if item.id == selected_hand_id)
    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    accounting, accounting_error = _reconcile_cached(db, hand.id, accounting_cache)

    with st.expander("Completed-hand history", expanded=False):
        st.code(
            format_hand_history(
                session,
                hand,
                actions,
                players,
                ledger=None if accounting is None else accounting.ledger,
                accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
                accounting_authoritative=_accounting_is_established(hand, accounting),
            ),
            language="text",
        )

    call_pot_default = 0.0
    call_default = 0.0
    bet_pot_default = 0.0
    bet_default = 0.0
    stack_default = float(hand.effective_stack or 0)
    rake_pct_default = 0.0
    rake_cap_default: float | None = None
    if _accounting_is_established(hand, accounting):
        assert accounting is not None
        hero = next((player for player in players if player.is_hero), None)
        hero_key = None if hero is None else hero.player_key
        hero_calls = [
            snapshot
            for snapshot in accounting.ledger.snapshots
            if snapshot.player == hero_key and snapshot.call_increment > 0
        ]
        hero_bets = [
            snapshot
            for snapshot in accounting.ledger.snapshots
            if snapshot.player == hero_key
            and snapshot.kind in {"bet", "all-in"}
            and snapshot.to_call_before == 0
        ]
        if hero_calls:
            call_pot_default = hero_calls[-1].pot_before
            call_default = hero_calls[-1].call_increment
            stack_default = hero_calls[-1].effective_stack_before
        if hero_bets:
            bet_pot_default = hero_bets[-1].pot_before
            bet_default = hero_bets[-1].amount
            stack_default = hero_bets[-1].effective_stack_before
        if accounting.settlement is not None:
            rake_pct_default = accounting.settlement.rake_rate * 100
            rake_cap_default = accounting.settlement.rake_cap
        st.caption(
            "Scenario inputs are prefilled from the reconciled ledger and remain editable for study."
        )
    else:
        st.warning(
            "This hand is not reconciled. Scenario inputs are manual and are not treated as recorded facts."
        )

    default_range = estimate_villain_range_label(hand.tags, hand.notes)
    range_options = sorted(RANGE_LABELS)
    range_label = range_col.selectbox(
        "Villain range label",
        options=range_options,
        index=range_options.index(default_range),
    )
    baseline_options = {"(none)": None} | {
        f"{chart.position} {chart.scenario.replace('_', ' ')} — {chart.description}": chart
        for chart in available_ranges()
    }
    baseline_choice = baseline_col.selectbox(
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

    matrix_notation = (
        range_label
        if baseline_chart is not None or custom_range.strip()
        else range_notation(range_label)
    )
    if matrix_notation:
        try:
            with st.expander("Range matrix", expanded=False):
                st.caption(f"{range_display.title()} · study range, not solver output")
                st.markdown(
                    range_matrix_html(
                        range_cells_from_notation(matrix_notation),
                        label=f"{range_display} preflop range",
                    ),
                    unsafe_allow_html=True,
                )
        except (KeyError, ValueError) as exc:
            st.warning(f"Range matrix unavailable: {exc}")

    with st.container(key="math_primary_inputs"):
        call_pot_col, call_col, bet_pot_col, bet_col = st.columns(4)
        pot_before_call = call_pot_col.number_input(
            "Pot facing Hero (BB)",
            min_value=0.0,
            value=float(call_pot_default),
            step=0.5,
            help="Pot after Villain's bet and before Hero calls.",
        )
        call_amount = call_col.number_input(
            "Call amount (BB)", min_value=0.0, value=float(call_default), step=0.5
        )
        pot_size = bet_pot_col.number_input(
            "Pot before Hero bets (BB)",
            min_value=0.0,
            value=float(bet_pot_default),
            step=0.5,
        )
        bet_size = bet_col.number_input(
            "Hero bet (BB)", min_value=0.0, value=float(bet_default), step=0.5
        )

    with st.container(key="math_secondary_inputs"):
        fold_col, called_equity_col, stack_col, streets_col = st.columns(4)
        fold_frequency_pct = fold_col.slider("Villain folds", 0, 100, 40)
        equity_when_called_pct = called_equity_col.slider(
            "Equity when called",
            0,
            100,
            30,
            help="Equity versus Villain's continuing range, not their full starting range.",
        )
        effective_stack = stack_col.number_input(
            "Effective stack behind (BB)",
            min_value=0.0,
            value=float(stack_default),
            step=1.0,
        )
        streets_remaining = streets_col.selectbox(
            "Betting streets left",
            [1, 2, 3],
            help="Include the current street.",
        )
    compute_equity = st.toggle("Compute blocker-aware equity", value=True)

    with st.expander("Rake and model assumptions", expanded=False):
        rake_col, cap_col, assumption_col = st.columns([1, 1, 2])
        rake_pct = rake_col.number_input(
            "Rake (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(rake_pct_default),
            step=0.1,
        )
        rake_cap = cap_col.number_input(
            "Rake cap (BB)",
            min_value=0.0,
            value=rake_cap_default,
            step=0.5,
            placeholder="Uncapped",
        )
        assumption_col.caption(
            "Heads-up, one-street EV unless stated otherwise. Use zero rake when "
            "no-flop-no-drop or another room rule means no rake is taken."
        )

    math_facts: dict[str, float | str] = {}
    inputs_match_ledger = bool(
        _accounting_is_established(hand, accounting)
        and pot_before_call == call_pot_default
        and call_amount == call_default
        and pot_size == bet_pot_default
        and bet_size == bet_default
    )
    math_facts["input_provenance"] = (
        "reconciled ledger" if inputs_match_ledger else "user-edited post-session study scenario"
    )
    equity_result = None
    errors: list[str] = []
    call_metrics: list[tuple[str, str, str | None]] = []
    bet_metrics: list[tuple[str, str, str | None]] = []
    rake_rate = rake_pct / 100
    modeled_rake_cap = _optional_float(rake_cap)

    if call_amount > 0 and pot_before_call > 0:
        try:
            required = required_equity_to_call(call_amount, pot_before_call)
            adjusted_required = required_equity_to_call_after_rake(
                call_amount,
                pot_before_call,
                rake_rate,
                modeled_rake_cap,
            )
            gross_final_pot = pot_before_call + call_amount
            modeled_rake = rake_amount(gross_final_pot, rake_rate, modeled_rake_cap)
            math_facts["required_equity_to_call"] = required
            math_facts["rake_adjusted_required_equity"] = adjusted_required
            call_metrics.extend(
                [
                    ("Required equity", format_percentage(required), "No-rake pot odds."),
                    (
                        "After-rake threshold",
                        f"{adjusted_required * 100:.1f}%",
                        f"Final pot {gross_final_pot:g} BB; modeled rake {modeled_rake:g} BB.",
                    ),
                ]
            )
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    if bet_size > 0 and pot_size > 0:
        try:
            bluff_frequency = break_even_bluff_frequency(bet_size, pot_size)
            mdf = minimum_defense_frequency(bet_size, pot_size)
            offered = pot_odds_offered_by_bet(bet_size, pot_size)
            bluff_fraction = optimal_bluff_fraction(bet_size, pot_size)
            bluffs_per_value = bluff_to_value_ratio(bet_size, pot_size)
            math_facts["break_even_bluff_frequency"] = bluff_frequency
            math_facts["minimum_defense_frequency"] = mdf
            math_facts["pot_odds_offered"] = offered
            math_facts["optimal_river_bluff_fraction"] = round(bluff_fraction, 4)
            bet_metrics.append(("Bet size", f"{bet_size / pot_size * 100:.1f}% pot", None))
            bet_metrics.append(
                ("Pot odds offered", format_percentage(offered), "Caller's break-even equity.")
            )
            bet_metrics.append(
                (
                    "Auto-profit folds",
                    format_percentage(bluff_frequency),
                    "Folds needed for a zero-equity bluff.",
                )
            )
            bet_metrics.append(
                (
                    "MDF baseline",
                    format_percentage(mdf),
                    "Heads-up, one-street baseline; not a solver strategy frequency.",
                )
            )
            bet_metrics.append(
                (
                    "River bluff fraction",
                    format_percentage(bluff_fraction),
                    f"Optimal share of a polarized river betting range that is bluffs "
                    f"({bluffs_per_value:.2f} bluffs per value bet). Earlier streets support more bluffs.",
                )
            )
            if effective_stack > 0:
                spr = stack_to_pot_ratio(effective_stack, pot_size)
                geometric = geometric_bet_fraction(spr, int(streets_remaining))
                math_facts["spr"] = round(spr, 3)
                math_facts["geometric_bet_fraction"] = round(geometric, 4)
                bet_metrics.extend(
                    [
                        ("SPR", f"{spr:.2f}", "Effective stack behind divided by current pot."),
                        (
                            "Geometric size",
                            f"{geometric * 100:.1f}% pot",
                            f"Equal sizing across {streets_remaining} street(s), assuming bet-call each street.",
                        ),
                    ]
                )
        except ValueError as exc:
            errors.append(str(exc))

    if compute_equity and hand.hero_cards:
        try:
            with st.spinner("Computing Hero equity vs range..."):
                equity_result = _cached_equity(hand.hero_cards, hand.board_cards, range_label)
            math_facts["equity"] = (
                equity_result.equity if equity_result.equity is not None else "unavailable"
            )
            if equity_result.equity is None:
                call_metrics.append(
                    (f"Equity vs {range_display}", "unavailable", equity_result.method)
                )
            else:
                equity_help = equity_result.method.replace("_", " ")
                if equity_result.valid_combos is not None:
                    equity_help += f", {equity_result.valid_combos} blocker-valid combos"
                if equity_result.samples is not None:
                    equity_help += f", {equity_result.samples:,} samples"
                if equity_result.std_error:
                    ci_low, ci_high = (
                        max(0.0, equity_result.equity - 1.96 * equity_result.std_error),
                        min(1.0, equity_result.equity + 1.96 * equity_result.std_error),
                    )
                    equity_help += f". Monte-Carlo 95% CI: {format_percentage(ci_low)}–{format_percentage(ci_high)}"
                call_metrics.append(
                    (
                        f"Equity vs {range_display}",
                        format_percentage(equity_result.equity),
                        equity_help,
                    )
                )
        except ValueError as exc:
            errors.append(str(exc))

    if (
        equity_result is not None
        and equity_result.equity is not None
        and call_amount > 0
        and pot_before_call > 0
    ):
        ev_value = call_ev(
            equity_result.equity,
            pot_before_call,
            call_amount,
            rake_rate=rake_rate,
            rake_cap=modeled_rake_cap,
        )
        adjusted_required = required_equity_to_call_after_rake(
            call_amount, pot_before_call, rake_rate, modeled_rake_cap
        )
        math_facts["call_ev"] = round(ev_value, 3)
        call_metrics.append(
            (
                "Equity edge",
                f"{(equity_result.equity - adjusted_required) * 100:+.1f} pp",
                "Equity minus the after-rake break-even threshold.",
            )
        )
        call_metrics.append(
            ("Call EV", f"{ev_value:+.2f} BB", "Incremental EV; assumes no future betting.")
        )

    fold_frequency = fold_frequency_pct / 100
    if bet_size > 0 and pot_size > 0:
        bluff_value = bluff_ev(fold_frequency, pot_size, bet_size)
        equity_when_called = equity_when_called_pct / 100
        semi_value = semi_bluff_ev(fold_frequency, equity_when_called, pot_size, bet_size)
        needed_folds = semi_bluff_break_even_fold_frequency(equity_when_called, pot_size, bet_size)
        math_facts["bluff_ev"] = round(bluff_value, 3)
        math_facts["semi_bluff_ev"] = round(semi_value, 3)
        bet_metrics.extend(
            [
                (
                    "Pure-bluff EV",
                    f"{bluff_value:+.2f} BB",
                    f"At {fold_frequency_pct}% folds, zero equity when called.",
                ),
                (
                    "Semi-bluff EV",
                    f"{semi_value:+.2f} BB",
                    f"At {equity_when_called_pct}% equity when called.",
                ),
                (
                    "Folds needed",
                    format_percentage(needed_folds),
                    "Break-even semi-bluff fold frequency.",
                ),
            ]
        )

    if call_metrics or bet_metrics:
        st.markdown("##### Calling Math")
        if call_metrics:
            _render_metric_rows(call_metrics)
        else:
            st.caption("Enter a pot and call amount to see calling math.")
        st.markdown("##### Betting Math")
        if bet_metrics:
            _render_metric_rows(bet_metrics)
        else:
            st.caption("Enter a pot and bet size to see betting math.")
    if equity_result is not None:
        st.caption(equity_result.notes)
    st.caption(
        "Equilibrium metrics are heads-up, one-street baselines. Full GTO strategy "
        "requires the exact game tree, positions, ranges, stack depth, rake, and allowed sizes."
    )

    for error in errors:
        st.error(error)

    with st.expander("Advanced study models", expanded=False):
        realization_tab, multiway_tab, outs_tab, icm_tab = st.tabs(
            ["Realization heuristic", "Multiway equity", "Outs", "Tournament ICM"]
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
        ledger=None if accounting is None else accounting.ledger,
        accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
        accounting_authoritative=_accounting_is_established(hand, accounting),
    )

    with st.expander("Structured coaching prompt"):
        st.code(prompt, language="text")


def _render_metric_rows(
    metrics: list[tuple[str, str, str | None]],
    *,
    per_row: int = 4,
) -> None:
    """Render compact metric rows without dropping or compressing values."""
    for start in range(0, len(metrics), per_row):
        batch = metrics[start : start + per_row]
        for column, (label, value, help_text) in zip(st.columns(len(batch)), batch, strict=True):
            column.metric(label, value, help=help_text)


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
    villain_ranges = [range_label, second.strip(), *([third.strip()] if third.strip() else [])]
    if not second.strip():
        st.info("Enter a Villain 2 range to compute multiway equity.")
        return
    try:
        with st.spinner("Computing multiway pot share..."):
            result = _cached_multiway_equity(
                hand.hero_cards, hand.board_cards, tuple(villain_ranges)
            )
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
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
    share_col.metric(
        f"Hero pot share ({len(villain_ranges) + 1}-way)",
        format_percentage(result.equity),
        help=help_text,
    )
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
            for index, (stack, equity) in enumerate(zip(stacks, equities, strict=True))
        ],
        hide_index=True,
        width="stretch",
    )
    hero_col, risk_col = st.columns(2)
    hero_seat = hero_col.number_input(
        "Hero player #", min_value=1, max_value=len(stacks), value=1, step=1
    )
    max_risk = stacks[int(hero_seat) - 1]
    risk_amount = risk_col.number_input(
        "Chips at risk",
        min_value=0.0,
        max_value=float(max_risk),
        value=min(1000.0, max_risk / 2),
        step=100.0,
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

    accounting_cache = new_accounting_cache()
    hands = _hands_with_accounting_results(
        db, db.fetch_hands_by_session(session.id), accounting_cache
    )
    if not hands:
        st.info("Save or load hands before generating coach reviews.")
        return

    provider_choice = st.selectbox("Provider", ["Claude (Anthropic)", "Cloud (OpenAI)"])
    provider_key = {
        "Claude (Anthropic)": "anthropic",
        "Cloud (OpenAI)": "cloud",
    }[provider_choice]
    try:
        provider = get_provider_from_env(provider_key)
    except LLMProviderError as exc:
        st.warning(str(exc))
        return
    st.write({"Active provider": provider.provider_name, "Model": provider.model_name})

    coaching_mode = st.selectbox("Coaching mode", COACHING_MODES)
    review_scope = st.radio("Review scope", ["Hand", "Session"], horizontal=True)

    if review_scope == "Hand":
        show_hand_coach_review(
            db, session, hands, provider, coaching_mode, accounting_cache
        )
    else:
        show_session_coach_review(db, session, hands, provider, coaching_mode)


def show_hand_coach_review(
    db: PokerDatabase,
    session: Session,
    hands: list[Hand],
    provider,
    coaching_mode: str,
    accounting_cache: AccountingCache | None = None,
) -> None:
    labels = {
        hand.id: (
            f"Hand #{hand.hand_number}: {hand.hero_cards or 'unknown'} "
            f"({'unknown' if hand.hero_bb_won is None else f'{hand.hero_bb_won:+g} BB'})"
        )
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
    accounting, accounting_error = _reconcile_cached(db, hand.id, accounting_cache)
    history = format_hand_history(
        session,
        hand,
        actions,
        players,
        ledger=None if accounting is None else accounting.ledger,
        accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
        accounting_authoritative=_accounting_is_established(hand, accounting),
    )
    st.code(history, language="text")

    range_label = st.selectbox(
        "Villain range label",
        options=sorted(RANGE_LABELS),
        index=sorted(RANGE_LABELS).index(estimate_villain_range_label(hand.tags, hand.notes)),
        key="coach_range_label",
    )
    math_facts = _accounting_prompt_math_facts(hand, accounting)
    # `unattested_...`, not `assumption_dependence`: a dependence the operator has
    # already confirmed -- or one on a hand they entered themselves, where no
    # attestation control is ever drawn -- is answered, and a message telling them
    # to go and perform an action they have already performed (or that the product
    # does not offer) is the failure PLAN.md's "a blocker never names an action the
    # product cannot perform" rule exists to prevent.
    if unattested_assumption_dependence(hand, accounting):
        st.warning(
            "Coaching is disabled until you confirm the declared settlement "
            "assumptions this hand's reconciliation rests on, in Study → Summary "
            "→ Accounting reconciliation. Until then its pot, rake, and hero "
            "result are not established by the recording."
        )
    elif accounting is None or not accounting.is_authoritative:
        st.warning(
            "Coaching is disabled until the completed hand has a legal, balanced reconciliation."
        )
    elif math_facts:
        st.caption("Math facts are sourced from the reconciled action ledger.")

    prompt = build_hand_review_prompt(
        session,
        hand,
        actions,
        players,
        pot_odds_facts=math_facts,
        villain_range_label=range_label,
        coaching_mode=coaching_mode,
        ledger=None if accounting is None else accounting.ledger,
        accounting_issues=_accounting_prompt_issues(accounting, accounting_error),
        accounting_authoritative=_accounting_is_established(hand, accounting),
    )
    show_prompt_safety(prompt)
    with st.expander("Exact prompt sent to provider"):
        st.code(prompt, language="text")

    # This surface does not otherwise fetch issue, coaching, or solver evidence, so
    # readiness is composed here before the promotion is offered.
    readiness = hand_study_readiness(
        db,
        hand,
        accounting,
        accounting_error,
        user_confirmed=bool(
            st.session_state.get(study_confirmation_key(hand, accounting), False)
        ),
    )
    st.markdown("##### Study readiness")
    render_study_readiness(readiness)
    if is_reconstructed_hand(hand):
        show_reconstruction_evidence(
            hand, parse_completion_evidence(hand.completion_evidence)
        )
    if hand_requires_user_confirmation(hand):
        st.checkbox(
            "I have read the evidence above and confirm this hand is correct",
            key=study_confirmation_key(hand, accounting),
        )

    if st.button(
        "Generate and save post-session hand review",
        disabled=(
            not _accounting_is_established(hand, accounting)
            or readiness.has("OPEN_DEBUGGING_ISSUE")
            or readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")
            or readiness.has("STUDY_EXCLUDED_BY_OPERATOR")
        ),
    ):
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
            # The review is kept either way; only the promotion is gated.
            if readiness.is_ready and guarded_update_hand_status(
                db, hand, readiness, "reviewed"
            ):
                flash(f"Saved provider review #{saved.id}.")
            else:
                flash(
                    f"Saved provider review #{saved.id}. Review status unchanged: "
                    "this hand is not study-ready."
                )
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
    study_hands = [
        hand
        for hand in hands
        if hand.study_inclusion != "skip"
        and hand.completion_status in {"complete", "not_applicable"}
    ]
    if not study_hands:
        st.warning(
            "No complete study hands in this session yet. Finalize incomplete drafts "
            "or change Study inclusion before generating session coaching."
        )
        return
    stats = compute_session_stats(db, session.id, hands=study_hands)
    selected_hands = select_session_review_hands(study_hands)
    histories = [
        _format_persisted_hand_history(db, session, hand)
        for hand in selected_hands
        if hand.id is not None
    ]
    st.caption(
        f"Selected hands: {', '.join(f'#{hand.hand_number}' for hand in selected_hands) or 'none'}"
    )
    excluded_notes: list[str] = []
    skipped = sum(1 for hand in hands if hand.study_inclusion == "skip")
    if skipped:
        excluded_notes.append(f"{skipped} non-study")
    drafts = sum(
        1
        for hand in hands
        if hand.study_inclusion != "skip"
        and hand.completion_status not in {"complete", "not_applicable"}
    )
    if drafts:
        excluded_notes.append(f"{drafts} incomplete draft")
    if excluded_notes:
        st.caption(
            "Excluded " + " and ".join(excluded_notes) + " hand(s) from session coaching."
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
        state = "STALE" if review.is_stale else "CURRENT"
        with st.expander(
            f"{state} · {review.created_at.isoformat()} - "
            f"{review.provider_name}/{review.model_name}"
        ):
            st.write({"Review type": review.review_type, "Safety mode": review.safety_mode})
            if review.is_stale:
                st.warning(review.stale_reason or "Underlying evidence changed.")
            st.write(review.parsed_sections or {})
            st.code(review.raw_response, language="text")


def show_prompt_safety(prompt: str) -> None:
    result = validate_post_session_prompt(prompt)
    if result.is_safe:
        st.success("Prompt safety check passed: post-session review only.")
    else:
        st.error("Prompt safety check failed: " + "; ".join(result.errors))


def _accounting_is_established(
    hand: Hand, accounting: AccountingReconciliation | None
) -> bool:
    """One-line delegate to the single service-level definition.

    It stays here only so every UI call site reads the same short name. The rule
    itself lives in ``services.study_readiness.accounting_is_established``,
    beside the readiness blocker it has to agree with, because it used to be a
    second expression in a second module: this file answered "is there any
    measured dependence?" while readiness answered "is there one the operator has
    not attested to, on a hand that owes an attestation at all?". The two
    disagreed in both directions -- an attested reconstructed hand stayed
    study-ready with coaching permanently disabled, and a manual hand carrying an
    ordinary room rake had coaching disabled with no control anywhere that could
    ever enable it.
    """
    return accounting_is_established(hand, accounting)


def _accounting_prompt_issues(
    accounting: AccountingReconciliation | None, accounting_error: str | None
) -> list[str]:
    """Everything a reader of the hand history must know about its accounting.

    A measured settlement-assumption dependence belongs in this list for the same
    reason a ledger warning does: it is a reason the figures below it are not
    established. It is stated in chips, so the sentence survives being read by a
    provider that has never seen the Study page.
    """
    if accounting_error:
        return [accounting_error]
    if accounting is None:
        return []
    return [
        *accounting.issues,
        *(
            f"Reconciles only under a declared settlement assumption — "
            f"{dependence.describe()}"
            for dependence in accounting.assumption_dependence
        ),
    ]


def _accounting_prompt_math_facts(
    hand: Hand, accounting: AccountingReconciliation | None
) -> dict[str, float | str]:
    if not _accounting_is_established(hand, accounting):
        return {}
    assert accounting is not None
    facts: dict[str, float | str] = {
        "accounting_status": "reconciled, chip-balanced, betting-sequence legal",
        "gross_pot_bb": accounting.ledger.gross_pot,
        "rake_bb": accounting.ledger.rake,
        "net_pot_bb": accounting.ledger.net_pot,
    }
    call_snapshots = [
        snapshot
        for snapshot in accounting.ledger.snapshots
        if snapshot.kind in {"call", "all-in"} and snapshot.call_increment > 0
    ]
    if not call_snapshots:
        return facts
    decision = call_snapshots[-1]
    facts.update(
        {
            "last_call_bb": decision.call_increment,
            "pot_before_last_call_bb": decision.pot_before,
            "required_equity_to_call": required_equity_to_call(
                decision.call_increment, decision.pot_before
            ),
            "effective_stack_low_bb": decision.effective_stack_range_before[0],
            "effective_stack_high_bb": decision.effective_stack_range_before[1],
        }
    )
    if decision.spr_range_before is not None:
        facts["spr_low"] = round(decision.spr_range_before[0], 3)
        facts["spr_high"] = round(decision.spr_range_before[1], 3)
    if (
        not any(
            snapshot.kind in {"ante", "post_blind", "bet", "call", "raise", "all-in"}
            for snapshot in accounting.ledger.snapshots[decision.index + 1 :]
        )
        and accounting.ledger.net_pot > 0
    ):
        facts["terminal_call_equity_after_recorded_rake"] = round(
            decision.call_increment / accounting.ledger.net_pot,
            6,
        )
    return facts


def show_video_processing(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    all_videos = db.fetch_videos(session.id)
    has_videos = bool(all_videos)
    collect_open = _import_collect_is_open(session.id, has_videos=has_videos)

    with st.container(key="import_collect_bar"):
        if has_videos and not collect_open:
            picker, reopen = st.columns([3.2, 1])
            labels = {
                video.id: (
                    f"{video.original_filename} · "
                    f"{_format_optional_seconds(video.duration_seconds)}"
                )
                for video in all_videos
                if video.id is not None
            }
            available_ids = set(labels)
            selected_video_id = st.session_state.get("video_context_id")
            if selected_video_id not in available_ids:
                selected_video_id = next(iter(labels))
                st.session_state["video_context_id"] = selected_video_id
            chosen = picker.selectbox(
                f"{len(all_videos)} recording{'s' if len(all_videos) != 1 else ''}",
                options=list(labels),
                format_func=lambda video_id: labels[video_id],
                index=list(labels).index(selected_video_id),
                key=f"collapsed_video_select_{session.id}",
            )
            if chosen != selected_video_id:
                st.session_state["video_context_id"] = chosen
                st.rerun()
            if reopen.button(
                "Add video",
                key=f"reopen_import_collect_{session.id}",
                width="stretch",
            ):
                st.session_state[_import_collect_panel_key(session.id)] = True
                st.rerun()
            video = next(video for video in all_videos if video.id == chosen)
            show_video_metadata(video)
            show_video_jobs_and_frames(db, video)
            return

        upload_col, hide_col = st.columns([3.2, 1])
        with upload_col:
            with st.expander(
                "Add recording" if has_videos else "Add first recording",
                expanded=not has_videos,
            ):
                _save_video_upload(db, session, key_prefix=f"import_{session.id}")
        if has_videos and hide_col.button(
            "Done",
            key=f"hide_import_collect_{session.id}",
            type="primary",
            width="stretch",
            help="Hide the upload panel and keep validating.",
        ):
            st.session_state[_import_collect_panel_key(session.id)] = False
            st.rerun()

    if not all_videos:
        st.caption("No videos linked yet — upload a completed-session recording above.")
        return

    video = _select_session_video(all_videos, key_prefix="import")
    if video is None:
        st.error("Selected video no longer exists.")
        return
    show_video_metadata(video)
    show_video_jobs_and_frames(db, video)


def _select_session_video(
    all_videos: list[VideoRecord],
    *,
    key_prefix: str,
) -> VideoRecord | None:
    """Pick the active recording for import / reconstruction."""

    available_ids = {video.id for video in all_videos if video.id is not None}
    selected_video_id = st.session_state.get("video_context_id")
    if selected_video_id not in available_ids:
        selected_video_id = all_videos[0].id
        st.session_state["video_context_id"] = selected_video_id
    labels = {
        video.id: (
            f"{video.original_filename} · "
            f"{_format_optional_seconds(video.duration_seconds)}"
        )
        for video in all_videos
        if video.id is not None
    }
    chosen = st.selectbox(
        f"{len(all_videos)} recording{'s' if len(all_videos) != 1 else ''}",
        options=list(labels),
        format_func=lambda video_id: labels[video_id],
        index=list(labels).index(selected_video_id),
        key=f"{key_prefix}_video_select",
    )
    if chosen != selected_video_id:
        st.session_state["video_context_id"] = chosen
        st.rerun()
    return next((video for video in all_videos if video.id == chosen), None)


def show_video_metadata(video: VideoRecord) -> None:
    fps = "—" if video.fps is None else f"{video.fps:g} FPS"
    st.markdown(
        '<div class="pt-import-meta">'
        f"<span><strong>{escape(_format_optional_seconds(video.duration_seconds))}</strong> duration</span>"
        f"<span><strong>{escape(_format_resolution(video.width, video.height))}</strong></span>"
        f"<span><strong>{escape(fps)}</strong></span>"
        f"<span><strong>{escape(_format_bytes(video.file_size_bytes))}</strong></span>"
        f"<span>{escape(video.original_filename)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Source path & notes", expanded=False):
        st.code(video.stored_path, language="text")
        st.caption(video.notes or "No source notes recorded.")


def show_video_jobs_and_frames(db: PokerDatabase, video: VideoRecord) -> None:
    if video.id is None:
        return

    show_cv_reconstruction(db, video)
    with st.expander("Advanced diagnostics · legacy frame extraction"):
        show_legacy_frame_extraction(db, video)


def show_cv_reconstruction(db: PokerDatabase, video: VideoRecord) -> None:
    """Render the non-blocking reconstruction launch and current job state."""
    if video.id is None:
        return
    workflow_step(
        1,
        "Reconstruct completed hands",
        "Export a timeline, then validate frames. Hands join the session only when "
        "validated or when you add a draft.",
        state="active",
    )
    linked_session = db.fetch_session(video.session_id) if video.session_id is not None else None
    if linked_session is not None:
        session_name = linked_session.name
        target_session_id = linked_session.id
        st.caption(
            f"Hands go to **{linked_session.name}** · numbers kept when possible."
        )
    else:
        target_session_id = None
        session_name = st.text_input(
            "Destination session name",
            value=Path(video.original_filename).stem,
            key=f"cv_session_name_{video.id}",
            help="This unassigned video will create a new empty session for later hand adds.",
        )
    latest_jobs = [
        job for job in db.fetch_jobs_by_video(video.id) if job.job_type == "cv_reconstruction"
    ]
    active = next(
        (job for job in latest_jobs if job.status in {"queued", "running", "cancelling"}),
        None,
    )
    start_col, cancel_col = st.columns([3.2, 1])
    with start_col:
        start_clicked = st.button(
            "Run CV reconstruction",
            type="primary",
            disabled=active is not None,
            key=f"cv_start_{video.id}",
            width="stretch",
        )
    with cancel_col:
        cancel_clicked = st.button(
            "Cancel",
            disabled=active is None,
            key=f"cv_cancel_{video.id}",
            width="stretch",
        )
    if start_clicked:
        try:
            started = start_cv_job(
                db,
                video.id,
                video.stored_path,
                session_name,
                target_session_id=target_session_id,
            )
            flash(f"Reconstruction job #{started.id} started.")
            st.rerun()
        except (CVJobAlreadyRunningError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if cancel_clicked and active is not None and active.id is not None:
        try:
            cancelled = cancel_processing_job(db, active.id)
            if cancelled.status == "cancelled":
                flash(f"Cancelled reconstruction job #{cancelled.id}.")
            elif cancelled.status == "cancelling":
                flash(
                    f"Cancellation requested for job #{cancelled.id}; "
                    "waiting for the worker to stop."
                )
            else:
                flash(
                    f"Reconstruction job #{cancelled.id} already finished "
                    f"as {cancelled.status}."
                )
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    latest_jobs = [
        job for job in db.fetch_jobs_by_video(video.id) if job.job_type == "cv_reconstruction"
    ]
    if latest_jobs:
        latest = latest_jobs[0]
        if latest.status in {"queued", "running", "cancelling"}:
            _show_live_cv_job_status(db, video.id)
        else:
            _render_cv_job_status(db, latest)
        completed_jobs = [job for job in latest_jobs if job.status == "completed"]
        if completed_jobs:
            review_job = _choose_frame_review_job(db, video.id, completed_jobs)
            if review_job is not None:
                show_reconstruction_evidence_review(db, review_job)
    else:
        st.caption("No reconstruction has been run for this source video.")

    hands_in_session = 0
    if video.session_id is not None and latest_jobs:
        # Prefer completed-job evidence presence over raw job status for step 4.
        for completed in [job for job in latest_jobs if job.status == "completed"]:
            if completed.id is None:
                continue
            related = related_cv_job_ids(db, video.id)
            try:
                timeline = load_timeline_for_job(completed.id)
            except (OSError, ValueError, json.JSONDecodeError):
                timeline = None
            if not timeline:
                continue
            for hand in timeline.get("hands") or []:
                number = int(hand.get("hand_number") or 0)
                if number < 1:
                    continue
                if find_existing_imported_hand(
                    db,
                    session_id=video.session_id,
                    job_id=completed.id,
                    timeline_hand_number=number,
                    related_job_ids=related,
                ):
                    hands_in_session += 1
            break
    step4_state = (
        "complete"
        if hands_in_session > 0
        else "active"
        if latest_jobs and latest_jobs[0].status == "completed"
        else "pending"
    )
    workflow_step(
        2,
        "Validate or add drafts",
        "Mark frames Correct to auto-add full hands, or Add draft for incomplete ones. "
        "Study stays separate until you confirm.",
        state=step4_state,
    )


@st.fragment(run_every=1.0)
def _show_live_cv_job_status(db: PokerDatabase, video_id: int) -> None:
    """Poll one active job and advance the full page as soon as it finishes."""
    latest_jobs = [
        job for job in db.fetch_jobs_by_video(video_id) if job.job_type == "cv_reconstruction"
    ]
    if not latest_jobs:
        st.info("Waiting for the reconstruction worker to start.")
        return
    latest = latest_jobs[0]
    if latest.status not in {"queued", "running", "cancelling"}:
        st.rerun()
    _render_cv_job_status(db, latest)
    st.caption("Updating automatically — you can leave this page open.")


def _offer_frame_validation_link(db: PokerDatabase, hand: Hand) -> None:
    """Study can jump to Import frame review without hunting the sidebar path."""
    if hand.session_id is None:
        return
    videos = db.fetch_videos(session_id=hand.session_id)
    if not videos:
        st.caption(
            "Frame validation lives on Import once a recording is attached to this session."
        )
        return
    job_id = job_id_from_hand_notes(hand.notes)
    preferred = videos[0]
    matched_job = False
    if job_id is not None:
        for video in videos:
            jobs = [
                job
                for job in db.fetch_jobs_by_video(video.id)
                if job.job_type == "cv_reconstruction" and job.id == job_id
            ]
            if jobs:
                preferred = video
                matched_job = True
                break
    if not matched_job and len(videos) > 1:
        # Prefer a recording that already has reconstruction evidence rather than
        # silently opening the newest upload in a multi-video session.
        scored = []
        for video in videos:
            completed = [
                job
                for job in db.fetch_jobs_by_video(video.id)
                if job.job_type == "cv_reconstruction" and job.status == "completed"
            ]
            labels = sum(
                len(db.fetch_reconstruction_frame_reviews(job.id))
                for job in completed
                if job.id is not None
            )
            scored.append((labels, len(completed), video.id or 0, video))
        scored.sort(reverse=True)
        preferred = scored[0][3]
        if scored[0][0] == 0 and scored[0][1] == 0:
            st.caption(
                "This session has multiple recordings; opening the newest one. "
                "Pick the correct video on Import if needed."
            )
    if st.button(
        "Open frame validation on Import",
        key=f"study_open_frame_validation_{hand.id}",
        width="stretch",
        help="Jump to the saved frame labels for this session's recording.",
    ):
        _activate_session(hand.session_id)
        st.session_state["video_context_id"] = preferred.id
        if matched_job and job_id is not None and preferred.id is not None:
            st.session_state[f"cv_review_job_{preferred.id}"] = job_id
        navigate_to(Page.IMPORT)
        flash("Opened Import frame validation. Progress is already saved per job.")
        st.rerun()


def _choose_frame_review_job(db: PokerDatabase, video_id: int, completed_jobs: list):
    """Pick a review job without hiding older partial progress behind a fresh empty job."""
    if not completed_jobs:
        return None
    by_id = {job.id: job for job in completed_jobs if job.id is not None}
    if not by_id:
        return completed_jobs[0]
    counts = {
        job_id: len(db.fetch_reconstruction_frame_reviews(job_id)) for job_id in by_id
    }
    preferred_id = max(by_id, key=lambda job_id: (counts.get(job_id, 0), job_id))
    newest_id = completed_jobs[0].id
    if (
        preferred_id != newest_id
        and counts.get(preferred_id, 0) > 0
        and counts.get(newest_id, 0) == 0
    ):
        st.warning(
            f"Newest job #{newest_id} has no frame labels yet. Showing job "
            f"#{preferred_id} which still has {counts.get(preferred_id, 0)} saved "
            "label(s). Choose another job below if you meant the newest run."
        )
    if len(completed_jobs) == 1:
        return completed_jobs[0]

    job_key = f"cv_review_job_{video_id}"
    job_ids = [job.id for job in completed_jobs if job.id is not None]
    if st.session_state.get(job_key) not in job_ids:
        st.session_state[job_key] = preferred_id
    selected_id = st.selectbox(
        "Reconstruction job to resume validating",
        job_ids,
        format_func=lambda job_id: (
            f"Job #{job_id} · {counts.get(job_id, 0)} saved labels · "
            f"{((by_id[job_id].completed_at or by_id[job_id].created_at).strftime('%Y-%m-%d %H:%M') if (by_id[job_id].completed_at or by_id[job_id].created_at) else 'unknown time')} · "
            f"{by_id[job_id].message or by_id[job_id].status}"
        ),
        key=job_key,
    )
    return by_id[selected_id]


def _render_cv_job_status(db: PokerDatabase, job) -> None:
    """Render what one reconstruction job is doing, or what it left behind.

    A stopped job gets no progress bar and no percentage. The figure it stopped
    at describes work in flight, and beside "Failed" it is read as a share of
    hands that made it in; the outcome resolver answers that question from the
    database instead. The wording is carried in text, not colour, so the state
    survives a screenshot, a colour-blind reader and a printed page.
    """
    outcome = describe_job_outcome(db, job)
    if outcome.is_live and outcome.progress_percent is not None:
        st.progress(
            max(0.0, min(1.0, outcome.progress_percent / 100)),
            text=f"{outcome.headline} · {outcome.statement}",
        )
    status_cols = st.columns(4)
    status_cols[0].metric("Status", outcome.headline)
    status_cols[1].metric("Progress", outcome.progress_label)
    status_cols[2].metric("Job", f"#{job.id}")
    heartbeat = job.heartbeat_at or job.started_at
    status_cols[3].metric(
        "Heartbeat", heartbeat.strftime("%H:%M:%S UTC") if heartbeat else "Waiting"
    )
    if not outcome.is_live:
        panel("Outcome", outcome.statement)
    if outcome.error_message:
        st.error(f"Failure detail: {outcome.error_message}")
    _render_job_log(outcome)


def _render_job_log(outcome) -> None:
    """Offer the worker log a failure wrote, which nothing else surfaces.

    The tail is redacted upstream, in the outcome resolver, because a worker log
    is where a provider key or a connection string ends up printed verbatim.
    """
    if not outcome.log_path:
        return
    data_callout("Worker log", outcome.log_path)
    if not outcome.log_tail:
        return
    with st.expander(f"Worker log · last {LOG_TAIL_LINES} lines (credentials redacted)"):
        st.code("\n".join(outcome.log_tail), language="text")


def show_reconstruction_evidence_review(db: PokerDatabase, job) -> None:
    """Review each retained reconstruction frame beside the history it produced."""
    if job.id is None:
        return
    try:
        timeline = load_timeline_for_job(job.id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Frame evidence is unavailable: {exc}")
        return
    if timeline is None:
        st.info(
            "This older job has no retained evidence timeline. Run reconstruction again "
            "to enable frame-by-frame validation."
        )
        return
    hands = timeline.get("hands", [])
    if not hands:
        st.warning(empty_hands_review_message(timeline))
        return

    st.markdown("#### Frame evidence review")
    st.caption(
        "Label frames, then fix linked actions below. Progress saves permanently."
    )
    video = db.fetch_video(job.video_id)
    session_id = video.session_id if video is not None else None
    related_jobs = related_cv_job_ids(db, job.video_id) if video is not None else {job.id}
    recovery_key = f"evidence_auto_import_scanned_{job.id}"
    if session_id is not None and not st.session_state.get(recovery_key):
        scan_results = import_all_autonomous_eligible(db, job.id)
        blocked_for_destination = any(
            "destination session" in ";".join(result.reasons)
            for result in scan_results
            if result.status == "blocked"
        )
        if not blocked_for_destination:
            st.session_state[recovery_key] = True
        imported = [result for result in scan_results if result.status == "imported"]
        if imported:
            st.success(
                f"Added {len(imported)} previously validated hand(s) to the session."
            )
    elif session_id is None:
        st.warning(
            "This video is not linked to a destination session yet, so hands cannot "
            "be added. Re-run reconstruction or attach the video to a session first."
        )
    reviews = db.fetch_reconstruction_frame_reviews(job.id)
    review_lookup = {(review.hand_number, review.source_image): review for review in reviews}
    hand_numbers = [int(hand.get("hand_number", 0)) for hand in hands]
    reviews_by_hand_image = {
        number: {
            review.source_image: review
            for review in reviews
            if review.hand_number == number
        }
        for number in hand_numbers
    }
    navigable_images_by_hand = {
        int(hand.get("hand_number", 0)): [
            str(state.get("image") or "")
            for state in states_for_hand(timeline, hand)
            if state.get("image")
        ]
        for hand in hands
    }
    total_frames = sum(len(images) for images in navigable_images_by_hand.values())
    # Job summary must use the same navigable image set as the per-hand table.
    # Counting every DB row inflated Validated / Needs improvement when orphan
    # reviews outlived dropped timeline frames.
    correct = 0
    incorrect = 0
    for number, images in navigable_images_by_hand.items():
        hand_reviews = reviews_by_hand_image.get(number, {})
        for image in images:
            review = hand_reviews.get(image)
            if review is None:
                continue
            if review.status == "correct":
                correct += 1
            elif review.status == "incorrect":
                incorrect += 1
    reviewed_frames = correct + incorrect
    summary_container = st.container(key=f"evidence_summary_{job.id}")
    with summary_container:
        summary = st.columns(4)
        summary[0].metric("Hands", len(hands))
        summary[1].metric("Used frames", total_frames)
        summary[2].metric("Validated", f"{reviewed_frames}/{total_frames}")
        summary[3].metric("Needs improvement", incorrect)

    progress_rows = []
    for hand in hands:
        number = int(hand.get("hand_number", 0))
        progress = hand_frame_progress(
            hand,
            reviews_by_hand_image.get(number, {}),
            countable_images=navigable_images_by_hand.get(number, []),
        )
        in_session = (
            session_id is not None
            and find_existing_imported_hand(
                db,
                session_id=session_id,
                job_id=job.id,
                timeline_hand_number=number,
                related_job_ids=related_jobs,
            )
            is not None
        )
        gate = autonomous_import_blockers(
            timeline,
            hand,
            timeline_path=timeline_path_for_job(job.id),
            reviews_by_image=reviews_by_hand_image.get(number, {}),
        )
        if in_session:
            session_status = "in session — edit here"
        elif session_id is None:
            session_status = "no destination session"
        elif gate.ok:
            session_status = "ready to auto-add"
        elif progress["flagged"]:
            session_status = "flagged — open hand to edit"
        elif progress["remaining"]:
            session_status = "frames remaining"
        else:
            session_status = "; ".join(gate.reasons[:2]) or "blocked"
        progress_rows.append(
            {
                "Hand": f"#{number}",
                "Hero": " ".join(hand.get("hero") or []) or "cards unknown",
                "Validated": f"{progress['reviewed']}/{progress['total']}",
                "Remaining": progress["remaining"],
                "Flagged": progress["flagged"],
                "Session": session_status,
            }
        )
    with st.expander("Saved progress by hand · resume anytime", expanded=reviewed_frames > 0):
        st.dataframe(progress_rows, hide_index=True, width="stretch")
        st.caption(
            "Leaving this page is safe. Re-open Import → this video → this job to "
            "continue or edit earlier labels."
        )

    hand_labels = {
        int(hand.get("hand_number", 0)): hand_validation_label(
            hand,
            reviews_by_hand_image.get(int(hand.get("hand_number", 0)), {}),
            countable_images=navigable_images_by_hand.get(
                int(hand.get("hand_number", 0)), []
            ),
        )
        for hand in hands
    }
    selected_key = f"evidence_hand_{job.id}"
    pending_key = f"evidence_hand_pending_{job.id}"
    if pending_key in st.session_state:
        st.session_state[selected_key] = st.session_state.pop(pending_key)
    if selected_key not in st.session_state:
        # Prefer the first hand that still has unreviewed frames.
        resume_hand = next(
            (
                int(hand.get("hand_number", 0))
                for hand in hands
                if hand_frame_progress(
                    hand,
                    reviews_by_hand_image.get(int(hand.get("hand_number", 0)), {}),
                    countable_images=navigable_images_by_hand.get(
                        int(hand.get("hand_number", 0)), []
                    ),
                )["remaining"]
                > 0
            ),
            hand_numbers[0],
        )
        st.session_state[selected_key] = resume_hand

    active_hand_index = (
        hand_numbers.index(st.session_state[selected_key])
        if st.session_state.get(selected_key) in hand_numbers
        else 0
    )
    prev_hand_col, chooser_col, next_hand_col = st.columns([1.1, 2.2, 1.1])
    prev_hand_col.button(
        "← Previous hand",
        key=f"evidence_prev_hand_{job.id}",
        disabled=active_hand_index == 0,
        width="stretch",
        help="Go to the previous timeline hand",
        on_click=_queue_evidence_hand,
        args=(
            pending_key,
            hand_numbers[active_hand_index - 1] if active_hand_index else hand_numbers[0],
            f"evidence_cursor_{job.id}_{hand_numbers[active_hand_index - 1] if active_hand_index else hand_numbers[0]}",
        ),
    )
    selected_hand_number = chooser_col.selectbox(
        "Hand to validate",
        hand_numbers,
        format_func=lambda number: hand_labels[number],
        key=selected_key,
    )
    next_hand_col.button(
        "Next hand →",
        key=f"evidence_next_hand_{job.id}",
        disabled=active_hand_index >= len(hand_numbers) - 1,
        width="stretch",
        type="primary",
        help="Go to the next timeline hand",
        on_click=_queue_evidence_hand,
        args=(
            pending_key,
            hand_numbers[active_hand_index + 1]
            if active_hand_index < len(hand_numbers) - 1
            else hand_numbers[-1],
            f"evidence_cursor_{job.id}_{hand_numbers[active_hand_index + 1] if active_hand_index < len(hand_numbers) - 1 else hand_numbers[-1]}",
        ),
    )
    st.caption(
        f"Hand {active_hand_index + 1} of {len(hand_numbers)} · "
        f"use Next hand → when this hand’s frames are done"
    )

    hand = next(item for item in hands if int(item.get("hand_number", 0)) == selected_hand_number)
    db_hand: Hand | None = None
    draft_result = None
    if session_id is None:
        st.warning(
            "Link this video to a destination session before validating hands."
        )
    else:
        try:
            draft_result = ensure_draft_for_review(
                db, job.id, selected_hand_number
            )
        except Exception as exc:  # noqa: BLE001 - keep the review UI usable
            st.error(f"Could not open session draft for editing: {exc}")
        else:
            if draft_result.status in {"imported", "already_present"}:
                if draft_result.status == "imported":
                    flash(
                        f"{draft_result.message} Edit beside the frames below."
                    )
                if draft_result.hand_id is not None:
                    db_hand = db.fetch_hand(draft_result.hand_id)
            else:
                st.warning(
                    draft_result.message
                    or "Could not draft this hand into the session for editing."
                )
        if db_hand is None:
            db_hand = find_existing_imported_hand(
                db,
                session_id=session_id,
                job_id=job.id,
                timeline_hand_number=selected_hand_number,
                related_job_ids=related_jobs,
            )
        if db_hand is not None:
            if db_hand.review_status == "reviewed":
                st.caption("Approved for Study.")
            else:
                st.caption(
                    "Draft ready — edit beside frames; finish with no open issues for Study."
                )
        elif draft_result is not None and draft_result.status == "blocked":
            st.caption(
                "Reasons: " + ("; ".join(draft_result.reasons) or "blocked")
            )

    states = states_for_hand(timeline, hand)
    if not states:
        st.warning("No retained source states could be matched to this hand.")
        return

    hand_reviews = reviews_by_hand_image.get(selected_hand_number, {})
    saved_by_image = {}
    for frame in states:
        image = str(frame.get("image") or "")
        review = review_lookup.get((selected_hand_number, frame.get("image")))
        if review is not None:
            saved_by_image[image] = review
    cursor_key = f"evidence_cursor_{job.id}_{selected_hand_number}"
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = first_unreviewed_frame_index(states, saved_by_image)
    st.session_state[cursor_key] = min(
        int(st.session_state.get(cursor_key, 0)), len(states) - 1
    )
    cursor = st.session_state[cursor_key]
    state = states[cursor]
    current_review = review_lookup.get((selected_hand_number, state["image"]))
    hand_progress = hand_frame_progress(
        hand,
        hand_reviews,
        countable_images=navigable_images_by_hand.get(selected_hand_number, []),
    )

    resume_col, done_col = st.columns([1.2, 1])
    if resume_col.button(
        "Jump to first unreviewed frame",
        key=f"evidence_resume_{job.id}_{selected_hand_number}",
        disabled=hand_progress["remaining"] == 0,
        width="stretch",
    ):
        st.session_state[cursor_key] = first_unreviewed_frame_index(states, saved_by_image)
        st.rerun()
    if hand_progress["remaining"] and current_review is not None and (
        current_review.status in {"correct", "incorrect"}
    ):
        done_col.warning(
            "This browser session left you on an already-labeled frame. "
            "Saved progress is fine — click Jump to first unreviewed to resume grading."
        )
    else:
        done_col.caption(
            f"Hand #{selected_hand_number}: {hand_progress['reviewed']}/"
            f"{hand_progress['total']} saved · progress stays if you leave"
        )

    verdict_label = (
        "✓ Correct"
        if current_review and current_review.status == "correct"
        else "⚑ Needs fix"
        if current_review and current_review.status == "incorrect"
        else "Unreviewed"
    )
    verdict_class = (
        "is-correct"
        if current_review and current_review.status == "correct"
        else "is-incorrect"
        if current_review and current_review.status == "incorrect"
        else "is-unreviewed"
    )
    navigation = st.container(
        key=f"evidence_navigation_{job.id}_{selected_hand_number}"
    )
    with navigation:
        st.markdown(
            (
                "<div class='pt-evidence-position'>"
                f"<span>Frame <strong>{cursor + 1}</strong> / "
                f"<strong>{len(states)}</strong></span>"
                f"<span>{float(state.get('time_s', 0)):.2f}s</span>"
                f"<span class='pt-evidence-verdict {verdict_class}'>{verdict_label}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        previous_col, next_col = st.columns(2)
        previous_col.button(
            "← Previous",
            key=f"evidence_prev_{job.id}_{selected_hand_number}",
            disabled=cursor == 0,
            width="stretch",
            on_click=_move_evidence_cursor,
            args=(cursor_key, -1, len(states)),
        )
        next_col.button(
            "Next →",
            key=f"evidence_next_{job.id}_{selected_hand_number}",
            disabled=cursor == len(states) - 1,
            width="stretch",
            on_click=_move_evidence_cursor,
            args=(cursor_key, 1, len(states)),
        )

    comparison = st.container(key=f"evidence_comparison_{job.id}_{selected_hand_number}")
    with comparison:
        frame_col, history_col = st.columns([1.45, 1], gap="large")
    with frame_col:
        _safe_image(
            str(state["image"]),
            caption=f"Source frame · {float(state.get('time_s', 0)):.2f}s",
        )
        with st.expander("What the models read in this frame"):
            for label, value in observed_facts(state):
                data_callout(label, value)

    with history_col:
        st.markdown("##### History built from this frame")
        for impact in history_impacts(hand, states, cursor):
            st.markdown(
                (
                    "<div class='pt-evidence-impact'>"
                    f"<span>{impact['kind']}</span>"
                    f"<strong>{impact['text']}</strong>"
                    f"<small>From {impact['source']}</small>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        st.button(
            "✓ Everything from this frame is correct",
            type="primary",
            width="stretch",
            key=f"evidence_correct_{job.id}_{selected_hand_number}_{cursor}",
            on_click=_mark_evidence_correct,
            args=(
                db,
                job.id,
                selected_hand_number,
                state,
                cursor_key,
                len(states),
            ),
        )
        with st.expander(
            "Flag a problem",
            expanded=bool(current_review and current_review.status == "incorrect"),
        ):
            with st.form(
                f"evidence_issue_{job.id}_{selected_hand_number}_{cursor}",
                clear_on_submit=False,
            ):
                issue_types = st.multiselect(
                    "What is wrong?",
                    list(ISSUE_GUIDANCE),
                    default=[] if current_review is None else current_review.issue_types,
                )
                notes = st.text_area(
                    "What should it have said?",
                    value="" if current_review is None else current_review.notes,
                    placeholder="Example: Seat 4 called 7 BB; this was not a raise.",
                    height=88,
                )
                submitted = st.form_submit_button(
                    "Save issue & go to next frame",
                    width="stretch",
                )
            if submitted:
                if not issue_types:
                    st.error("Choose at least one issue type.")
                else:
                    db.upsert_reconstruction_frame_review(
                        ReconstructionFrameReview(
                            job_id=job.id,
                            hand_number=selected_hand_number,
                            source_image=str(state["image"]),
                            timestamp_seconds=float(state.get("time_s", 0)),
                            status="incorrect",
                            issue_types=issue_types,
                            notes=notes.strip(),
                        )
                    )
                    _move_evidence_cursor(cursor_key, 1, len(states))
                    flash("Frame issue saved. You can leave and resume later.")
                    st.rerun()

    if db_hand is not None:
        frames_ok = hand_frames_validated(
            hand,
            hand_reviews,
            countable_images=navigable_images_by_hand.get(selected_hand_number, []),
        )
        render_validation_edit_and_approve(
            db,
            db_hand,
            frames_validated=frames_ok,
            frame_context=ValidationFrameContext(
                job_id=job.id,
                hand_number=selected_hand_number,
                timeline_hand=hand,
                states=states,
                reviews_by_image=hand_reviews,
                cursor_key=cursor_key,
                pending_hand_key=pending_key,
                recording_start_s=min(
                    (
                        float(state.get("time_s", 0.0))
                        for state in timeline.get("states", [])
                    ),
                    default=None,
                ),
            ),
        )

    # Count only navigable frames so the expander matches the summary metrics.
    navigable_images = {
        (number, image)
        for number, images in navigable_images_by_hand.items()
        for image in images
    }
    flagged = [
        review
        for review in reviews
        if review.status == "incorrect"
        and (review.hand_number, review.source_image) in navigable_images
    ]
    if flagged:
        with st.expander(f"Improvement queue · {len(flagged)} flagged frame(s)"):
            for index, review in enumerate(flagged):
                target_hand = next(
                    (
                        item
                        for item in hands
                        if int(item.get("hand_number", 0)) == review.hand_number
                    ),
                    None,
                )
                action_bits = []
                if target_hand is not None:
                    action_bits = [
                        ref.label()
                        for ref in timeline_actions_for_image(
                            target_hand, review.source_image
                        )
                    ]
                open_col, detail_col = st.columns([0.7, 2.5])
                detail_col.markdown(
                    f"**Hand #{review.hand_number} · {review.timestamp_seconds:.2f}s · "
                    + (", ".join(review.issue_types) or "unspecified")
                    + "**"
                )
                if action_bits:
                    detail_col.caption("Actions: " + " · ".join(action_bits))
                if review.notes:
                    detail_col.caption(review.notes)
                if open_col.button(
                    "Open frame",
                    key=f"evidence_open_flag_{job.id}_{index}",
                    width="stretch",
                ):
                    target_states = (
                        states_for_hand(timeline, target_hand)
                        if target_hand is not None
                        else []
                    )
                    target_cursor = next(
                        (
                            frame_index
                            for frame_index, frame in enumerate(target_states)
                            if str(frame.get("image") or "") == review.source_image
                        ),
                        None,
                    )
                    if target_cursor is None:
                        flash(
                            "That flagged frame is no longer in the retained timeline "
                            "for this hand."
                        )
                    else:
                        st.session_state[pending_key] = review.hand_number
                        st.session_state[
                            f"evidence_cursor_{job.id}_{review.hand_number}"
                        ] = target_cursor
                    st.rerun()
            counts: dict[str, int] = {}
            for review in flagged:
                for issue in review.issue_types:
                    counts[issue] = counts.get(issue, 0) + 1
            if counts:
                st.markdown("**Issue types**")
                for issue, count in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                ):
                    module, next_step = ISSUE_GUIDANCE.get(
                        issue,
                        ("Reconstruction pipeline", "Inspect the flagged source frames."),
                    )
                    st.caption(f"{issue} · {count} — {module}: {next_step}")


def _move_evidence_cursor(cursor_key: str, delta: int, frame_count: int) -> None:
    current = int(st.session_state.get(cursor_key, 0))
    st.session_state[cursor_key] = max(0, min(frame_count - 1, current + delta))


def _queue_evidence_hand(
    pending_key: str,
    hand_number: int,
    cursor_key: str | None = None,
) -> None:
    """Defer hand changes so the selectbox widget is not mutated after render."""
    st.session_state[pending_key] = hand_number
    # Clear the destination cursor so the hand resumes at first unreviewed.
    if cursor_key is not None:
        st.session_state.pop(cursor_key, None)


def _mark_evidence_correct(
    db: PokerDatabase,
    job_id: int,
    hand_number: int,
    state: dict,
    cursor_key: str,
    frame_count: int,
) -> None:
    lock_key = f"evidence_correct_lock_{job_id}_{hand_number}_{state.get('image')}"
    if st.session_state.get(lock_key):
        return
    st.session_state[lock_key] = True
    try:
        try:
            db.upsert_reconstruction_frame_review(
                ReconstructionFrameReview(
                    job_id=job_id,
                    hand_number=hand_number,
                    source_image=str(state["image"]),
                    timestamp_seconds=float(state.get("time_s", 0)),
                    status="correct",
                )
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Could not save frame verdict: {exc}")
            return
        _move_evidence_cursor(cursor_key, 1, frame_count)
        try:
            draft = ensure_draft_for_review(db, job_id, hand_number)
        except Exception as exc:  # noqa: BLE001
            flash(
                f"Frame marked correct, but session draft failed: {exc}."
            )
            return
        if draft.status == "imported":
            flash(f"Frame marked correct. {draft.message}")
        elif draft.status == "already_present":
            flash("Frame marked correct. Hand draft is in the session.")
        elif draft.status == "blocked" and draft.reasons:
            flash(
                "Frame marked correct. Draft not available yet: "
                + "; ".join(draft.reasons)
            )
        else:
            flash("Frame marked correct. Progress is saved — stop anytime.")
        # Auto-import path remains for full hands that were never drafted early.
        try:
            ensure_hand_imported(db, job_id, hand_number, mode="auto")
        except Exception:
            pass
        if draft.hand_id is not None:
            hand = db.fetch_hand(draft.hand_id)
            if hand is not None and hand.review_status != "reviewed":
                reviews = {
                    review.source_image: review
                    for review in db.fetch_reconstruction_frame_reviews(
                        job_id, hand_number=hand_number
                    )
                }
                timeline = load_timeline_for_job(job_id)
                timeline_hand = None
                if timeline is not None:
                    timeline_hand = next(
                        (
                            item
                            for item in timeline.get("hands") or []
                            if int(item.get("hand_number", 0)) == hand_number
                        ),
                        None,
                    )
                if timeline_hand is not None and hand_frames_validated(
                    timeline_hand, reviews
                ):
                    open_issues = [
                        issue
                        for issue in db.fetch_hand_issues(hand_id=hand.id)
                        if issue.status == "open"
                    ]
                    if not open_issues:
                        auto_key = f"validation_auto_approve_attempted_{hand.id}"
                        if not st.session_state.get(auto_key):
                            st.session_state[auto_key] = True
                            if try_approve_hand_after_validation(
                                db, hand, announce=False
                            ):
                                flash(
                                    f"Hand #{hand_number} validated and "
                                    "approved for Study."
                                )
    finally:
        st.session_state.pop(lock_key, None)


def show_legacy_frame_extraction(db: PokerDatabase, video: VideoRecord) -> None:
    """Keep the original diagnostic frame tool available without dominating Import."""

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
                    f"{len(summary.errors)} warnings: {'; '.join(summary.errors)}"
                )
            else:
                flash(f"Extracted {summary.frames_extracted} frames to {summary.output_dir}.")
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(f"Frame extraction failed: {exc}")

    jobs = db.fetch_jobs_by_video(video.id)
    if jobs:
        st.markdown("##### Jobs")
        # Same outcome resolver as the reconstruction panel: a diagnostics table
        # is still a place an operator reads "82%" as hands that got in, and the
        # error column is still a place a credential surfaces.
        diagnostic_rows = build_job_rows(jobs, [video], outcomes=describe_job_outcomes(db, jobs))
        created_at = {job.id: job.created_at for job in jobs if job.id is not None}
        st.dataframe(
            [
                {
                    "ID": row.job_id,
                    "Type": row.job_type,
                    "Status": row.status,
                    "Progress": row.progress_label,
                    "Outcome": row.outcome_statement,
                    "Error": row.outcome.error_message,
                    "Log": row.outcome.log_path or "—",
                    "Created": created_at[row.job_id].isoformat(),
                }
                for row in diagnostic_rows
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
    if st.button(
        "Delete extracted frames", key=f"delete_frames_{video.id}", disabled=not confirm_delete
    ):
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
            flash(f"Deleted video {video.original_filename}.")
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
        st.write(
            {
                "Frame width": frame_width,
                "Frame height": frame_height,
                "Frame path": frame.image_path,
            }
        )
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
        if st.button(
            "Delete profile", key=f"roi_delete_{profile.id}", disabled=not confirm_profile
        ):
            db.delete_roi_profile(profile.id)
            flash(f"Deleted ROI profile '{profile.name}'.")
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
        seats = st.number_input(
            "Seats", min_value=6, max_value=9, value=9, step=1, key="roi_preset_seats"
        )
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
            seat_index = st.number_input(
                "Seat index (0 = none)", min_value=0, max_value=10, value=0, step=1
            )
            card_index = st.number_input(
                "Card index (0 = none)", min_value=0, max_value=5, value=0, step=1
            )
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
            st.error(
                f"An ROI region with the key '{roi_key.strip()}' already exists in this profile."
            )
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
                index=ROI_TYPES.index(region.roi_type)
                if region.roi_type in ROI_TYPES
                else ROI_TYPES.index("unknown"),
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
            seat_index = st.number_input(
                "Seat index (0 = none)", min_value=0, max_value=10, value=seat_value, step=1
            )
            card_index = st.number_input(
                "Card index (0 = none)", min_value=0, max_value=5, value=card_value, step=1
            )
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
            st.error(
                f"An ROI region with the key '{roi_key.strip()}' already exists in this profile."
            )
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


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


if __name__ == "__main__":
    main()
