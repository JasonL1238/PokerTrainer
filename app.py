from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
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
    CompletionEvidence,
    acknowledge_codes,
    dump_completion_evidence,
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
    VideoRecord,
    utc_now,
)
from poker_tracker.player_labels import actor_label
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    attest_assumption,
    persist_reconciliation,
    reconcile_persisted_hand,
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
    product_hero,
    section_header,
    section_header_with_meta,
    status_badge,
    trust_badge,
    workflow_step,
)
from poker_tracker.ui.cv_jobs import CVJobAlreadyRunningError, reconcile_stuck_jobs, start_cv_job
from poker_tracker.ui.frame_extraction import (
    delete_extracted_frames,
    extract_frames_for_video,
    select_representative_frames,
)
from poker_tracker.ui.image_utils import image_dimensions, save_roi_crop_preview
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
    ISSUE_GUIDANCE,
    history_impacts,
    load_timeline_for_job,
    observed_facts,
    states_for_hand,
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
)
from poker_tracker.ui.ui_theme import brand_header, inject_theme
from poker_tracker.ui.video_metadata import extract_video_metadata
from poker_tracker.ui.video_storage import (
    ensure_data_directories,
    save_video_file,
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


def guarded_update_hand_status(
    db: PokerDatabase,
    hand: Hand,
    readiness: StudyReadiness,
    status: str,
) -> bool:
    """Single choke point for review-status writes; refuses to promote a blocked hand."""

    if status == "reviewed" and not readiness.is_ready:
        st.error(
            "This hand is not study-ready. Clear the blockers listed above before "
            "marking it reviewed."
        )
        return False
    if hand.id is None:
        st.error("This hand has not been saved yet.")
        return False
    try:
        db.update_hand_status(hand.id, status)
    except ValueError as exc:
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
    """
    options = [
        item for item in REVIEW_STATUSES if item != "reviewed" or readiness.is_ready
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
    """Explain readiness as concrete checks and grouped fix steps."""

    with st.container(key="study_workflow_guide"):
        st.markdown("#### Start with Replay, then fix, then analyze")
        if readiness.is_ready:
            st.caption("This hand passed every trust check and is ready to analyze.")
            return
        fix_groups = study_fix_groups(readiness)
        st.caption(
            f"{len(readiness.blockers)} trust check(s) are failing. Some clear "
            f"together, so there are {len(fix_groups)} concrete fix step(s). "
            "Replay still works; trusted analysis waits until these checks pass."
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
        "evidence": (
            "Verify the reconstructed hand",
            "Fix & confirm → Correct hand facts / Source warnings",
        ),
        "accounting": (
            "Reconcile the chips",
            "Fix & confirm → Accounting reconciliation",
        ),
        "issues": (
            "Resolve saved debugging issues",
            "Fix & confirm → Saved debugging issue",
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
            "Confirm the saved hand",
            "Fix & confirm → Confirm the saved hand",
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
        return
    acknowledged = set(evidence.acknowledged_codes)
    rejections = set(evidence.rejection_codes)
    unresolved = len(evidence.unresolved_codes)
    with st.expander(f"Source warnings · {unresolved} unresolved of {len(codes)}"):
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
    job_rows = build_job_rows(db.fetch_recent_jobs(6), db.fetch_videos())
    if not job_rows:
        empty_state("No processing jobs", "Uploaded video reconstruction jobs will appear here.")
    else:
        st.dataframe(
            [
                {
                    "Video": row.filename,
                    "Type": row.job_type,
                    "Status": row.status.replace("_", " ").title(),
                    "Progress": f"{row.progress_percent:.0f}%",
                    "Update": row.message,
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
            "Create one below. Its date becomes the memorable default name.",
        )
        create_session_form(db, form_key="create_first_session")
        return
    summary_tab, hands_tab, videos_tab, add_tab = st.tabs(
        ["Overview", "Hands", "Videos", "Add hand"]
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
            st.caption("No unresolved hand issues. Flag one from its Study page.")
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
                _open_hand_for_study(hand)
                st.rerun()


def show_study_workspace(db: PokerDatabase, session: Session | None) -> None:
    page_header(
        "Study",
        "Replay the hand, inspect the evidence, and turn one completed decision into a reusable lesson.",
    )
    sessions = db.fetch_sessions()
    accounting_cache = new_accounting_cache()
    all_hands = _hands_with_accounting_results(db, db.fetch_all_hands(), accounting_cache)
    if not all_hands:
        empty_state("Nothing queued for study", "Import or add a completed hand first.")
        return
    available_ids = {hand.id for hand in all_hands if hand.id is not None}
    requested = st.session_state.get("study_hand_id")
    if requested not in available_ids:
        session_hands = [hand for hand in all_hands if session and hand.session_id == session.id]
        requested = (session_hands or all_hands)[0].id
        st.session_state["study_hand_id"] = requested
    hand = next(item for item in all_hands if item.id == requested)
    hand_session = next(item for item in sessions if item.id == hand.session_id)
    ordered = sorted(
        (item for item in all_hands if item.session_id == hand.session_id and item.id is not None),
        key=lambda item: (item.hand_number, item.id or 0),
    )

    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    # The list above already reconciled this hand; a second build would be the
    # same two ledgers over the same records.
    accounting, accounting_error = _reconcile_cached(db, hand.id, accounting_cache)
    coaching_reviews = db.fetch_coaching_reviews_by_hand(hand.id)
    hand_issues = db.fetch_hand_issues(hand_id=hand.id)
    solver_runs = db.fetch_solver_runs_by_hand(hand.id)
    completion_evidence = parse_completion_evidence(hand.completion_evidence)
    is_reconstructed = is_reconstructed_hand(hand)
    # Read before the widget renders: on the rerun after the tick, session state
    # already holds the value, which is what makes the checkbox gate effective.
    user_confirmed = bool(st.session_state.get(study_confirmation_key(hand, accounting), False))
    readiness = evaluate_study_readiness(
        hand,
        accounting=accounting,
        accounting_error=accounting_error,
        hand_issues=hand_issues,
        coaching_reviews=coaching_reviews,
        # Legacy hand_reviews rows are staled by the same correction path and are
        # blocking evidence too. Omitting them here made this page -- which feeds
        # three of the five review-status writers -- report "Study-ready · 0
        # blockers" on a hand every other surface refused.
        hand_reviews=db.fetch_reviews_by_hand(hand.id),
        solver_runs=solver_runs,
        user_confirmed=user_confirmed,
    )

    render_study_workflow(readiness)
    render_study_hand_navigation(ordered, hand, hand_session)

    with st.container(key="study_workspace"):
        replay_tab, fix_tab, analyze_tab = st.tabs(
            ["1 · Replay", "2 · Fix & confirm", "3 · Analyze"]
        )
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
        with fix_tab:
            render_study_fix_and_confirm(
                db,
                hand,
                actions,
                players,
                accounting,
                accounting_error,
                readiness,
                hand_issues,
                is_reconstructed,
                completion_evidence,
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
    return f"Hand #{hand.hand_number} · {hand.hero_cards or 'Unknown cards'} · {result}"


def render_study_hand_navigation(
    ordered: list[Hand],
    hand: Hand,
    session: Session,
) -> None:
    """Keep hand selection in one compact row shared by every Study mode."""

    active_index = next(index for index, item in enumerate(ordered) if item.id == hand.id)
    hand_ids = [item.id for item in ordered if item.id is not None]
    hands_by_id = {item.id: item for item in ordered if item.id is not None}
    with st.container(key="study_hand_navigation"):
        previous_col, chooser_col, next_col = st.columns([0.45, 3.1, 0.45])
        if previous_col.button(
            "←",
            key=f"study_previous_{hand.id}",
            disabled=active_index == 0,
            help="Previous hand",
            width="stretch",
        ):
            st.session_state["study_hand_id"] = ordered[active_index - 1].id
            st.rerun()
        selected_id = chooser_col.selectbox(
            "Choose a completed hand",
            hand_ids,
            index=active_index,
            format_func=lambda hand_id: study_hand_label(hands_by_id[hand_id]),
            key=f"study_hand_picker_{hand.id}",
        )
        if selected_id != hand.id:
            st.session_state["study_hand_id"] = selected_id
            st.rerun()
        if next_col.button(
            "→",
            key=f"study_next_{hand.id}",
            disabled=active_index == len(ordered) - 1,
            help="Next hand",
            width="stretch",
        ):
            st.session_state["study_hand_id"] = ordered[active_index + 1].id
            st.rerun()
        st.caption(
            f"{active_index + 1} of {len(ordered)} · {session.name} · "
            f"{hand.source_type.replace('_', ' ').title()} · "
            f"{hand.completion_status.replace('_', ' ').title()}"
        )
        st.caption(
            f"Reconstruction confidence · {confidence_label(hand.confidence_score)}"
        )


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
        st.caption("Next: open the Fix & confirm tab.")

    section_header_with_meta(
        "Decision history",
        "Click any action to update the table above.",
        f"{len(actions)} ACTIONS",
    )
    if actions:
        st.pills(
            "Replay action",
            options=list(range(len(actions))),
            default=selected_index,
            format_func=lambda index: study_action_label(actions[index], index),
            key=replay_key,
            label_visibility="collapsed",
            width="stretch",
        )
        st.caption(
            "The gold-outlined seat acted. Dimmed seats had already folded. "
            "Choose Final hand to return to the completed result."
        )
        with st.expander("Show pot, stack, SPR, and notes for every action"):
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


def study_action_label(action: Action, index: int) -> str:
    """Return a compact but complete label for an action-replay control."""

    actor = actor_label(action.player_name, action.position) or "Unknown player"
    amount = "" if action.amount is None else f" · {action.amount:g} BB"
    return (
        f"{index + 1:02d} · {action.street.title()} · {actor} · "
        f"{action.action_type.replace('-', ' ').title()}{amount}"
    )


def render_study_fix_and_confirm(
    db: PokerDatabase,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer],
    accounting: AccountingReconciliation | None,
    accounting_error: str | None,
    readiness: StudyReadiness,
    hand_issues: list[HandIssue],
    is_reconstructed: bool,
    completion_evidence: CompletionEvidence,
) -> None:
    """Present blocker resolution first and keep technical editors opt-in."""

    st.markdown("### Fix & confirm")
    st.caption(
        "Check the short list below. Open a correction tool only when that part "
        "of the saved hand is wrong."
    )
    if readiness.is_ready:
        st.success("Everything required is confirmed. This hand is ready to analyze.")
    else:
        fix_groups = study_fix_groups(readiness)
        st.warning(
            f"{len(readiness.blockers)} trust check(s) are failing across "
            f"{len(fix_groups)} fix step(s)."
        )
        blocker_columns = st.columns(min(2, len(fix_groups)))
        for index, (title, destination, blockers) in enumerate(fix_groups):
            with blocker_columns[index % len(blocker_columns)]:
                with st.container(border=True):
                    st.markdown(f"**{index + 1}. {title}**")
                    for blocker in blockers:
                        st.write(f"• {blocker.reason}")
                    st.caption(f"Use: {destination}")
        with st.expander("Show exact requirements"):
            render_study_readiness(readiness)

    status_col, tools_col = st.columns([0.9, 1.45], gap="large")
    with status_col:
        st.markdown("#### Confirm the saved hand")
        if is_reconstructed:
            show_reconstruction_evidence(hand, completion_evidence)
            show_source_warning_controls(db, hand, completion_evidence)
        if hand_requires_user_confirmation(hand):
            st.checkbox(
                "I have read the evidence above and confirm this hand is correct",
                key=study_confirmation_key(hand, accounting),
            )
        with st.expander("Set review status", expanded=readiness.is_ready):
            status_options, status_index = review_status_options(hand, readiness)
            status_key = f"study_status_{hand.id}"
            if st.session_state.get(status_key) not in status_options:
                st.session_state.pop(status_key, None)
            status = st.selectbox(
                "Review status",
                status_options,
                index=status_index,
                key=status_key,
            )
            if st.button(
                "Save review status",
                key=f"study_save_{hand.id}",
                width="stretch",
            ):
                if guarded_update_hand_status(db, hand, readiness, status):
                    flash("Review status updated.")
                    st.rerun()
        show_hand_issue_controls(db, hand, hand_issues)

    with tools_col:
        st.markdown("#### Correction tools")
        st.caption("These are closed by default. Open only the section you need.")
        with st.expander("Accounting status"):
            _render_accounting_status(accounting, accounting_error)
        show_accounting_editor(
            db,
            hand,
            players,
            accounting,
            accounting_error,
        )
        show_hand_fact_editor(db, hand)
        show_player_editor(db, players)
        show_action_editor(db, actions, players)
        show_correction_history(db, hand.id)


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
            "remain. Open Fix & confirm and follow the grouped checklist."
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
        st.caption("Edit notes in Fix & confirm → Correct hand facts.")


def show_hand_issue_controls(
    db: PokerDatabase,
    hand: Hand,
    issues: list[HandIssue],
) -> None:
    """Flag a hand now and leave a self-contained report for future debugging."""

    if hand.id is None:
        return
    open_issues = [issue for issue in issues if issue.status == "open"]
    if open_issues:
        st.error(f"This hand has {len(open_issues)} unresolved debugging issue(s).")

    with st.expander("Flag this hand for future debugging", expanded=False):
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


def show_hand_fact_editor(db: PokerDatabase, hand: Hand) -> None:
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
    with st.expander("Correct hand facts", expanded=False):
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


def show_correction_history(db: PokerDatabase, hand_id: int) -> None:
    corrections = db.fetch_hand_corrections(hand_id)
    with st.expander(f"Correction history ({len(corrections)})"):
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
    if st.button(
        "Generate and save corrected-hand coaching",
        key=f"study_rerun_coaching_{hand.id}",
        disabled=(
            provider is None
            or not is_authoritative
            or readiness.has("OPEN_DEBUGGING_ISSUE")
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
            "In Fix & confirm, correct the cards, players, positions, and actions; "
            "then reconcile the chip ledger.",
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
        st.markdown("**Fix these items in Fix & confirm:**")
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
    if st.button(
        "Run TexasSolver analysis",
        key=f"solver_analyze_{hand.id}",
        type="primary",
        width="stretch",
        disabled=bool(range_errors)
        or oop_range is None
        or ip_range is None
        or not binary_ready,
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
    if st.button(
        "Explain solver result with AI",
        key=f"solver_explain_{latest.id}",
        disabled=(
            provider is None
            or not _accounting_is_established(hand, accounting)
            or readiness.has("OPEN_DEBUGGING_ISSUE")
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
) -> None:
    if hand.id is None:
        return
    with st.expander("Accounting reconciliation", expanded=False):
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
            st.caption("Awards declare winners by pot layer: 0 is the main pot; 1+ are side pots.")
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
    page_header(
        "Import a completed session",
        "Keep every recording from the same completed session together, then reconstruct its hands.",
    )
    if session is None:
        empty_state(
            "Create a session for these recordings",
            "The date becomes its default name; you can attach more videos at any time.",
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
        query = st.text_input(
            "Find a session",
            placeholder="Try “July 27”, “ClubWPT”, “1/2”, or a note",
            key="session_library_search",
        )
        matches = filter_sessions(sessions, query)
        if not matches:
            st.caption("No sessions match that search.")
            return

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
                    "Open" if active_session is None or item.id != active_session.id else "Current",
                    key=f"open_session_{item.id}",
                    type="primary"
                    if active_session is not None and item.id == active_session.id
                    else "secondary",
                    disabled=active_session is not None and item.id == active_session.id,
                    width="stretch",
                ):
                    _activate_session(item.id)
                    st.rerun()


def _open_hand_for_study(hand: Hand) -> None:
    if hand.id is None:
        return
    _activate_session(hand.session_id)
    st.session_state["study_hand_id"] = hand.id
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
        "Source notes",
        height=70,
        key=f"{key_prefix}_notes",
        placeholder="Optional: table, time range, or what this recording contains",
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
            stored_path = save_video_file(uploaded, uploaded.name)
            metadata = extract_video_metadata(stored_path)
            saved = db.create_video(
                VideoRecord(
                    session_id=session.id,
                    original_filename=uploaded.name,
                    stored_path=str(stored_path),
                    file_size_bytes=stored_path.stat().st_size,
                    duration_seconds=metadata.duration_seconds,
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                    frame_count=metadata.frame_count,
                    notes=notes.strip(),
                )
            )
            if metadata.error:
                st.warning(metadata.error)
            flash(f"Added {saved.original_filename} to {session.name}.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


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
        date_played = st.date_input("Date played", value=date.today())
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
        session.id: (
            f"{session.date_played.strftime('%b')} {session.date_played.day} · {session.name}"
        )
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
    st.caption("Older sessions are searchable in Sessions.")
    return next(session for session in sessions if session.id == active_id)


def show_session_dashboard(db: PokerDatabase, session: Session) -> None:
    if session.id is None:
        return
    stats = compute_session_stats(db, session.id)
    st.subheader(session.name)
    st.caption(
        f"{session.date_played} · {session.stakes or 'stakes not set'} · {session.platform or 'platform not set'}"
    )

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
        st.info("No hands recorded yet. Add hands in the Enter Hand tab to see session stats.")
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
            hand_number = st.number_input(
                "Hand number", min_value=1, step=1, value=next_hand_number
            )
            game_type = st.text_input("Game type", value="No-limit Hold'em cash")
            blinds_antes = st.text_input("Blinds / antes", placeholder="1/2 NL, 0.25 ante")
            table_size = st.number_input("Table size", min_value=2, max_value=10, value=6, step=1)
            effective_stack = st.number_input(
                "Declared effective stack (BB)",
                min_value=0.0,
                value=None,
                step=1.0,
                placeholder="Unknown",
                help="Optional summary evidence. Decision-level effective stacks are derived from players.",
            )
            source_type = st.selectbox("Source", ["manual", "cv_import", "corrected_cv"])
        with setup_right:
            hero_position = st.selectbox("Hero position", POSITIONS, index=6)
            hero_cards = st.text_input("Hero cards", placeholder="Ah Qs")
            board_cards = st.text_input("Board cards", placeholder="Qd 7s 2c 9h 3s")
            pot_size = st.number_input(
                "Observed final pot (BB)",
                min_value=0.0,
                value=None,
                step=1.0,
                placeholder="Unknown",
            )
            hero_bb_won = st.number_input(
                "Observed final result in BB",
                value=None,
                step=0.5,
                placeholder="Unknown",
            )
            result = st.text_input("Result text", placeholder="Hero wins")
            # A hand being created has no accounting and no review, so "reviewed"
            # is never a legitimate starting status here.
            review_status = st.selectbox(
                "Review status", ["unreviewed", "needs_correction"]
            )

        tags = st.multiselect("Tags", sorted(HAND_TAGS))
        notes = st.text_area("Hand notes / pro-style hand history", height=100)

        st.markdown("#### Players In The Hand")
        player_rows = collect_player_inputs()

        st.markdown("#### Action Line")
        action_rows = collect_action_inputs()
        submitted = st.form_submit_button("Save hand")

    if submitted:
        if any(hand.hand_number == int(hand_number) for hand in existing_hands):
            st.error(
                f"Hand #{int(hand_number)} already exists in this session. Pick a different number."
            )
            return
        # A hand entered by hand but declared reconstructed carries no evidence, so
        # it starts unproven instead of being mintable as reviewed.
        is_declared_reconstructed = source_type != "manual"
        completion_status = "uncertain" if is_declared_reconstructed else "not_applicable"
        if is_declared_reconstructed:
            review_status = "needs_correction"
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
                        effective_stack=_optional_float(effective_stack),
                        hero_position=hero_position,
                        hero_cards=hero_cards,
                        board_cards=board_cards,
                        pot_size=_optional_float(pot_size),
                        result=result.strip(),
                        hero_bb_won=_optional_float(hero_bb_won),
                        review_status=review_status,
                        source_type=source_type,
                        completion_status=completion_status,
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
            {
                "Player": "Hero",
                "Seat": None,
                "Position": "BTN",
                "Starting stack": None,
                "Hero?": True,
                "Notes": "",
            },
            {
                "Player": "",
                "Seat": None,
                "Position": "",
                "Starting stack": None,
                "Hero?": False,
                "Notes": "",
            },
            {
                "Player": "",
                "Seat": None,
                "Position": "",
                "Starting stack": None,
                "Hero?": False,
                "Notes": "",
            },
        ],
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Seat": st.column_config.NumberColumn("Seat", min_value=0, max_value=9, step=1),
            "Position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
            "Starting stack": st.column_config.NumberColumn("Starting stack (BB)", min_value=0),
            "Hero?": st.column_config.CheckboxColumn("Hero?"),
        },
        key="players_editor",
    )
    return [
        {
            "player_name": str(row.get("Player") or "").strip(),
            "seat_index": (None if row.get("Seat") is None else int(row["Seat"])),
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
            st.caption(
                "Amount is the additional BB committed by this action, not the total raise-to size."
            )
            edited_rows = st.data_editor(
                _default_action_rows(street),
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Player": st.column_config.TextColumn("Player"),
                    "Position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
                    "Action": st.column_config.SelectboxColumn("Action", options=ACTION_TYPES),
                    "Forced post": st.column_config.SelectboxColumn(
                        "Forced post",
                        options=[
                            "",
                            "small_blind",
                            "big_blind",
                            "ante",
                            "big_blind_ante",
                            "straddle",
                            "dead_blind",
                            "bring_in",
                        ],
                    ),
                    "Post status": st.column_config.SelectboxColumn(
                        "Post status", options=["", "live", "dead"]
                    ),
                    "Amount": st.column_config.NumberColumn(
                        "Increment committed (BB)",
                        min_value=0,
                        help="For a raise, enter only the additional chips moved into the pot.",
                    ),
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
            "forced_bet_type": str(row.get("Forced post") or "") or None,
            "is_live_post": (
                None if not row.get("Post status") else row.get("Post status") == "live"
            ),
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
    players = db.fetch_players_by_hand(hand_id)
    for row in action_rows:
        if not row["player_name"]:
            continue
        matches = [
            player
            for player in players
            if player.player_name == row["player_name"]
            and (not row["position"] or player.position == row["position"])
        ]
        if len(matches) != 1:
            matches = [player for player in players if player.player_name == row["player_name"]]
        if len(matches) != 1:
            raise ValueError(
                f"Action player {row['player_name']!r} must match exactly one saved player."
            )
        db.create_action(
            Action(
                hand_id=hand_id,
                player_key=matches[0].player_key,
                amount_semantics="incremental",
                **row,
            )
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


def show_player_editor(db: PokerDatabase, players: list[HandPlayer]) -> None:
    with st.expander("Edit players and starting stacks", expanded=False):
        for player in players:
            if player.id is None:
                continue
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
                continue
            if not correction_reason.strip():
                st.error("Add a correction reason so this example can be learned from.")
                continue
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
) -> None:
    with st.expander("Edit or add actions", expanded=False):
        st.caption(
            "Use this only when the saved action history does not match the recording."
        )
        show_action_editor_contents(db, actions, players)


def show_action_editor_contents(
    db: PokerDatabase,
    actions: list[Action],
    players: list[HandPlayer],
) -> None:
    st.markdown("##### Edit / Delete Actions")
    if not actions:
        st.caption("No actions saved.")

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
            amount = cols[4].number_input(
                "Amount (BB)",
                min_value=0.0,
                value=action.amount,
                placeholder="Not applicable",
            )
            action_index = cols[5].number_input(
                "Order", min_value=1, value=action.action_index or 1, step=1
            )
            notes = cols[6].text_input("Notes", value=action.notes)
            semantics_col, pot_col, stack_col, post_col = st.columns(4)
            semantics = semantics_col.selectbox(
                "Amount meaning",
                ["incremental", "raise_to", "unknown"],
                index=["incremental", "raise_to", "unknown"].index(action.amount_semantics),
                help="Incremental means chips added now. Raise-to means the total committed this street.",
            )
            pot_before = pot_col.number_input(
                "Pot before (BB)", min_value=0.0, value=action.pot_before
            )
            stack_before = stack_col.number_input(
                "Stack before (BB)", min_value=0.0, value=action.stack_before
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
            forced_bet_type = post_col.selectbox(
                "Forced post",
                forced_options,
                index=(
                    forced_options.index(action.forced_bet_type)
                    if action.forced_bet_type in forced_options
                    else 0
                ),
            )
            live_post_value = post_col.selectbox(
                "Post status",
                ["unspecified", "live", "dead"],
                index=(0 if action.is_live_post is None else 1 if action.is_live_post else 2),
            )
            correction_reason = st.text_input(
                "Correction reason",
                placeholder="What did reconstruction get wrong?",
                key=f"action_correction_reason_{action.id}",
            )
            update, delete = st.columns(2)
            submitted_update = update.form_submit_button("Update action")
            submitted_delete = delete.form_submit_button("Delete action")

        if submitted_update:
            if not correction_reason.strip():
                st.error("Add a correction reason so this example can be learned from.")
                continue
            matches = [
                player
                for player in players
                if player.player_name == player_name
                and (not position or player.position == position)
            ]
            if len(matches) != 1:
                matches = [player for player in players if player.player_name == player_name]
            if len(matches) != 1:
                st.error("The action must match exactly one saved player.")
                continue
            try:
                db.update_action(
                    Action(
                        id=action.id,
                        hand_id=action.hand_id,
                        player_key=matches[0].player_key,
                        street=street,
                        action_index=int(action_index),
                        player_name=player_name,
                        position=position,
                        action_type=action_type,
                        amount=amount,
                        amount_semantics=semantics,
                        forced_bet_type=forced_bet_type or None,
                        is_live_post=(
                            None if live_post_value == "unspecified" else live_post_value == "live"
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
                st.rerun()
        if submitted_delete:
            if not correction_reason.strip():
                st.error("Add a correction reason before deleting this action.")
                continue
            db.delete_action(action.id, correction_notes=correction_reason)
            st.rerun()

    with st.expander("Add missing action"):
        if not players:
            st.warning("Add the hand's players before adding a corrected action.")
            return
        player_labels = {
            f"{player.player_name} · {player.position or 'unknown'} · {player.player_key[:8]}": player
            for player in players
        }
        with st.form(f"add_corrected_action_{players[0].hand_id}"):
            player_label = st.selectbox("Player", list(player_labels))
            player = player_labels[player_label]
            first, second, third = st.columns(3)
            street = first.selectbox("Street", STREETS)
            action_type = second.selectbox("Action", ACTION_TYPES)
            amount = third.number_input(
                "Amount (BB)", min_value=0.0, value=None, placeholder="Not applicable"
            )
            semantics = st.selectbox(
                "Amount meaning", ["incremental", "raise_to", "unknown"]
            )
            pot_before = st.number_input(
                "Pot before (BB)", min_value=0.0, value=None, placeholder="Unknown"
            )
            stack_before = st.number_input(
                "Stack before (BB)", min_value=0.0, value=None, placeholder="Unknown"
            )
            notes = st.text_input("Action notes")
            correction_reason = st.text_input(
                "Correction reason",
                placeholder="Example: missed Hero call between the flop bet and turn",
            )
            submitted = st.form_submit_button("Add corrected action")
        if submitted:
            if not correction_reason.strip():
                st.error("Add a correction reason so this example can be learned from.")
                return
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
                        amount_semantics=semantics,
                        pot_before=_optional_float(pot_before),
                        stack_before=_optional_float(stack_before),
                        notes=notes.strip(),
                    ),
                    correction_notes=correction_reason,
                )
            except (sqlite3.IntegrityError, ValidationError, ValueError) as exc:
                st.error(f"Could not add action: {exc}")
            else:
                flash("Corrected action added.")
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
    stats = compute_session_stats(db, session.id)
    selected_hands = select_session_review_hands(hands)
    histories = [
        _format_persisted_hand_history(db, session, hand)
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
    workflow_step(
        1,
        "Collect the session recordings",
        "Add one or several finished videos to the same session.",
        state="active",
    )
    st.caption(
        f"Adding to **{session.name}** · this workflow never captures or analyzes a live table."
    )
    with st.expander("Add another recording", expanded=not db.fetch_videos(session.id)):
        _save_video_upload(db, session, key_prefix=f"import_{session.id}")

    workflow_step(
        2,
        "Validate the source",
        "Choose a recording with one click and confirm its metadata.",
    )
    all_videos = db.fetch_videos(session.id)
    if not all_videos:
        st.info("No videos are linked to this session yet.")
        return

    available_ids = {video.id for video in all_videos if video.id is not None}
    selected_video_id = st.session_state.get("video_context_id")
    if selected_video_id not in available_ids:
        selected_video_id = all_videos[0].id
        st.session_state["video_context_id"] = selected_video_id
    st.caption(f"{len(all_videos)} recording{'s' if len(all_videos) != 1 else ''} in this session")
    source_columns = st.columns(min(3, len(all_videos)))
    for index, item in enumerate(all_videos):
        if item.id is None:
            continue
        label = (
            f"{'✓ ' if item.id == selected_video_id else ''}"
            f"{item.original_filename}\n{_format_optional_seconds(item.duration_seconds)}"
        )
        if source_columns[index % len(source_columns)].button(
            label,
            key=f"choose_video_{item.id}",
            type="primary" if item.id == selected_video_id else "secondary",
            disabled=item.id == selected_video_id,
            width="stretch",
        ):
            st.session_state["video_context_id"] = item.id
            st.rerun()

    video = db.fetch_video(selected_video_id)
    if video is None:
        st.error("Selected video no longer exists.")
        return
    show_video_metadata(video)
    show_video_jobs_and_frames(db, video)


def show_video_metadata(video: VideoRecord) -> None:
    metadata = st.columns(4)
    metadata[0].metric("Duration", _format_optional_seconds(video.duration_seconds))
    metadata[1].metric("Resolution", _format_resolution(video.width, video.height))
    metadata[2].metric("Frame rate", "—" if video.fps is None else f"{video.fps:g} FPS")
    metadata[3].metric("File size", _format_bytes(video.file_size_bytes))
    st.caption(f"{video.original_filename} · uploaded {video.uploaded_at.isoformat()}")
    with st.expander("Source provenance"):
        st.code(video.stored_path, language="text")
        st.write(video.notes or "No source notes recorded.")


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
        3,
        "Reconstruct completed hands",
        "Append auditable draft hands from this recording to its session.",
        state="active",
    )
    linked_session = db.fetch_session(video.session_id) if video.session_id is not None else None
    if linked_session is not None:
        session_name = linked_session.name
        target_session_id = linked_session.id
        st.info(
            f"Reconstructed hands will be added to **{linked_session.name}**. "
            "Existing hand numbers are preserved when possible and collisions are renumbered."
        )
    else:
        target_session_id = None
        session_name = st.text_input(
            "Imported session name",
            value=Path(video.original_filename).stem,
            key=f"cv_session_name_{video.id}",
            help="This unassigned video will create a new session.",
        )
    latest_jobs = [
        job for job in db.fetch_jobs_by_video(video.id) if job.job_type == "cv_reconstruction"
    ]
    active = next((job for job in latest_jobs if job.status in {"queued", "running"}), None)
    if st.button(
        "Run CV reconstruction",
        type="primary",
        disabled=active is not None,
        key=f"cv_start_{video.id}",
    ):
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

    latest_jobs = [
        job for job in db.fetch_jobs_by_video(video.id) if job.job_type == "cv_reconstruction"
    ]
    if latest_jobs:
        latest = latest_jobs[0]
        if latest.status in {"queued", "running"}:
            _show_live_cv_job_status(db, video.id)
        else:
            _render_cv_job_status(latest)
        if latest.status == "completed":
            st.success("Reconstruction is complete. Validate its frame evidence below.")
            show_reconstruction_evidence_review(db, latest)
    else:
        st.caption("No reconstruction has been run for this source video.")

    workflow_step(
        4,
        "Review imported hands",
        "Open the generated drafts in Study and verify their source evidence.",
        state="complete" if latest_jobs and latest_jobs[0].status == "completed" else "pending",
    )
    st.caption(
        "Completed jobs create needs-correction drafts with source confidence and provenance."
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
    if latest.status not in {"queued", "running"}:
        st.rerun()
    _render_cv_job_status(latest)
    st.caption("Updating automatically — you can leave this page open.")


def _render_cv_job_status(job) -> None:
    """Render the latest persisted state of one reconstruction job."""
    st.progress(
        job.progress_percent / 100,
        text=f"{job.status.title()} · {job.message}",
    )
    status_cols = st.columns(4)
    status_cols[0].metric("Status", job.status.replace("_", " ").title())
    status_cols[1].metric("Progress", f"{job.progress_percent:.0f}%")
    status_cols[2].metric("Job", f"#{job.id}")
    heartbeat = job.heartbeat_at or job.started_at
    status_cols[3].metric(
        "Heartbeat", heartbeat.strftime("%H:%M:%S UTC") if heartbeat else "Waiting"
    )
    if job.error_message:
        st.error(job.error_message)


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
        st.info("The reconstruction did not produce any hands to validate.")
        return

    st.markdown("#### Frame evidence review")
    st.caption(
        "Validate the hand where it was assembled: each retained frame is paired with "
        "the exact history facts it created."
    )
    st.caption(
        "Your verdict is saved to this job's audit record. Correct frames count as "
        "verified evidence; flagged frames enter the improvement queue below. "
        "Validation does not silently rewrite the hand or retrain a model."
    )
    reviews = db.fetch_reconstruction_frame_reviews(job.id)
    review_lookup = {(review.hand_number, review.source_image): review for review in reviews}
    total_frames = sum(len(hand.get("source_images") or []) for hand in hands)
    correct = sum(review.status == "correct" for review in reviews)
    incorrect = sum(review.status == "incorrect" for review in reviews)
    summary = st.columns(4)
    summary[0].metric("Hands", len(hands))
    summary[1].metric("Used frames", total_frames)
    summary[2].metric("Validated", f"{correct + incorrect}/{total_frames}")
    summary[3].metric("Needs improvement", incorrect)

    hand_labels = {}
    for hand in hands:
        number = int(hand.get("hand_number", 0))
        count = len(hand.get("source_images") or [])
        hero = " ".join(hand.get("hero") or []) or "cards unknown"
        hand_labels[number] = f"Hand #{number} · {hero} · {count} source frames"
    selected_hand_number = st.selectbox(
        "Hand to validate",
        list(hand_labels),
        format_func=lambda number: hand_labels[number],
        key=f"evidence_hand_{job.id}",
    )
    hand = next(item for item in hands if int(item.get("hand_number", 0)) == selected_hand_number)
    states = states_for_hand(timeline, hand)
    if not states:
        st.warning("No retained source states could be matched to this hand.")
        return

    cursor_key = f"evidence_cursor_{job.id}_{selected_hand_number}"
    st.session_state[cursor_key] = min(int(st.session_state.get(cursor_key, 0)), len(states) - 1)
    cursor = st.session_state[cursor_key]
    state = states[cursor]
    current_review = review_lookup.get((selected_hand_number, state["image"]))

    previous_col, position_col, next_col = st.columns([1, 3, 1])
    previous_col.button(
        "← Previous",
        key=f"evidence_prev_{job.id}_{selected_hand_number}",
        disabled=cursor == 0,
        width="stretch",
        on_click=_move_evidence_cursor,
        args=(cursor_key, -1, len(states)),
    )
    verdict_label = (
        "✓ Correct"
        if current_review and current_review.status == "correct"
        else "⚑ Needs fix"
        if current_review and current_review.status == "incorrect"
        else "Unreviewed"
    )
    position_col.markdown(
        f"<div class='pt-evidence-position'>Frame <strong>{cursor + 1}</strong> of "
        f"<strong>{len(states)}</strong> · {float(state.get('time_s', 0)):.2f}s · "
        f"{verdict_label}</div>",
        unsafe_allow_html=True,
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
                    flash("Frame issue saved to the improvement queue.")
                    st.rerun()

    flagged = [review for review in reviews if review.status == "incorrect"]
    if flagged:
        with st.expander(f"Improvement queue · {len(flagged)} flagged frame(s)"):
            st.dataframe(
                [
                    {
                        "Hand": f"#{review.hand_number}",
                        "Time": f"{review.timestamp_seconds:.2f}s",
                        "Issue": ", ".join(review.issue_types),
                        "Correction": review.notes or "No correction note",
                    }
                    for review in flagged
                ],
                hide_index=True,
                width="stretch",
            )
            counts: dict[str, int] = {}
            for review in flagged:
                for issue in review.issue_types:
                    counts[issue] = counts.get(issue, 0) + 1
            for issue, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                module, next_step = ISSUE_GUIDANCE.get(
                    issue, ("Reconstruction pipeline", "Inspect the flagged source frames.")
                )
                st.markdown(f"**{issue} · {count}**")
                st.caption(f"{module} — {next_step}")


def _move_evidence_cursor(cursor_key: str, delta: int, frame_count: int) -> None:
    current = int(st.session_state.get(cursor_key, 0))
    st.session_state[cursor_key] = max(0, min(frame_count - 1, current + delta))


def _mark_evidence_correct(
    db: PokerDatabase,
    job_id: int,
    hand_number: int,
    state: dict,
    cursor_key: str,
    frame_count: int,
) -> None:
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job_id,
            hand_number=hand_number,
            source_image=str(state["image"]),
            timestamp_seconds=float(state.get("time_s", 0)),
            status="correct",
        )
    )
    _move_evidence_cursor(cursor_key, 1, frame_count)
    flash("Frame marked correct.")


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


def _default_action_rows(street: str) -> list[dict]:
    row_count = 4 if street == "preflop" else 1
    return [
        {
            "Player": "",
            "Position": "",
            "Action": "fold",
            "Forced post": "",
            "Post status": "",
            "Amount": None,
            "Pot before": None,
            "Stack before": None,
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
