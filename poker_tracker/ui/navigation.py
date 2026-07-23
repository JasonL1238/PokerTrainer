"""Primary product navigation for the Streamlit application."""
from __future__ import annotations

from enum import StrEnum

import streamlit as st


class Page(StrEnum):
    OVERVIEW = "overview"
    SESSIONS = "sessions"
    HANDS = "hands"
    STUDY = "study"
    INSIGHTS = "insights"
    IMPORT = "import"
    SETTINGS = "settings"


_PRIMARY_NAV: tuple[tuple[Page, str, str], ...] = (
    (Page.OVERVIEW, "Overview", "Home"),
    (Page.SESSIONS, "Sessions", "Sessions"),
    (Page.HANDS, "Hands", "Hands"),
    (Page.STUDY, "Study", "Study"),
    (Page.INSIGHTS, "Insights", "Insights"),
    (Page.IMPORT, "Import", "Import"),
    (Page.SETTINGS, "Settings", "Settings"),
)


def render_navigation() -> Page:
    """Render the persistent left-rail navigation and return the active page."""
    pages = [item[0] for item in _PRIMARY_NAV]
    labels = {page: label for page, label, _ in _PRIMARY_NAV}
    pending = st.session_state.pop("_pending_navigation", None)
    if pending is not None:
        st.session_state["primary_navigation"] = Page(pending)
    active = st.radio(
        "Workspace",
        options=pages,
        format_func=lambda page: labels[page],
        key="primary_navigation",
        label_visibility="collapsed",
    )
    return Page(active)


def navigate_to(page: Page) -> None:
    """Set the requested destination for the next Streamlit rerun."""
    st.session_state["_pending_navigation"] = page
