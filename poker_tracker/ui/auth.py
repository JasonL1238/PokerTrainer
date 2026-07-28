"""Small shared-password gate for the personal web deployment."""

from __future__ import annotations

import hmac
import os

import streamlit as st

from poker_tracker.ui.components import product_hero
from poker_tracker.ui.poker_visuals import poker_table_html
from poker_tracker.ui.ui_theme import brand_header


def check_password() -> bool:
    """Return True only after authentication when a password is configured."""
    configured = os.environ.get("APP_PASSWORD", "")
    required = os.environ.get("POKERTRAINER_REQUIRE_AUTH", "").lower() in {"1", "true", "yes"}

    if not configured:
        if required:
            _render_configuration_error()
            return False
        return True
    if st.session_state.get("authed") is True:
        return True

    with st.container(key="login_shell"):
        visual, sign_in = st.columns([1.45, 0.78], gap="large", vertical_alignment="center")
        with visual:
            table_html = poker_table_html(
                hero_cards="",
                board_cards="",
                pot_size=None,
                players=[],
                label="Completed-hand data appears after sign-in",
            )
            product_hero(
                "Review the decisions that define your game.",
                "A private command center for completed-session reconstruction, analysis, and deliberate practice.",
                table_html,
                proof_points=(("OFFLINE", "analysis"), ("PRIVATE", "workspace")),
            )
        with sign_in:
            brand_header()
            st.markdown("### Enter the decision lab")
            st.caption("Sign in to your private post-session study workspace.")
            with st.form("shared_password_form", clear_on_submit=True):
                candidate = st.text_input(
                    "Password", type="password", autocomplete="current-password"
                )
                submitted = st.form_submit_button("Continue", type="primary", width="stretch")
            if submitted:
                if hmac.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8")):
                    st.session_state["authed"] = True
                    st.rerun()
                else:
                    st.error("Unable to sign in with those credentials.")
            st.caption("Completed-session analysis only. No live assistance.")
    return False


def logout_button() -> None:
    """Render logout only when the password gate is active."""
    if not os.environ.get("APP_PASSWORD"):
        return
    if st.button("Log out", key="logout", width="stretch"):
        st.session_state.pop("authed", None)
        st.rerun()


def _render_configuration_error() -> None:
    st.error("Authentication is required, but APP_PASSWORD is not configured.")
    st.caption("Set the deployment secret and restart the application.")
