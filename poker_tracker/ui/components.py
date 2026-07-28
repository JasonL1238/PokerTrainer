"""Reusable presentation-only components for PokerTrainer."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st

_STATUS_TONES = {
    "completed": "positive",
    "reviewed": "positive",
    "running": "active",
    "queued": "neutral",
    "unreviewed": "neutral",
    "failed": "negative",
    "needs_correction": "warning",
}


def product_hero(
    title: str,
    description: str,
    visual_html: str,
    *,
    kicker: str = "POST-SESSION DECISION LAB",
    proof_points: Iterable[tuple[str, str]] = (),
) -> None:
    """Render the immersive first-screen product statement and replay surface."""
    proof_html = "".join(
        f"<span><strong>{escape(value)}</strong> {escape(label)}</span>"
        for value, label in proof_points
    )
    st.markdown(
        '<section class="pt-hero"><div class="pt-hero-grid"><div class="pt-hero-copy">'
        f'<div class="pt-hero-kicker">{escape(kicker)}</div><h1>{escape(title)}</h1>'
        f'<p>{escape(description)}</p><div class="pt-hero-proof">{proof_html}</div>'
        f'</div><div class="pt-hero-visual">{visual_html}</div></div></section>',
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str = "",
    *,
    eyebrow: str = "POST-SESSION STUDY",
) -> None:
    """Render a consistent page heading and optional supporting copy."""
    description_html = (
        f'<p class="pt-page-description">{escape(description)}</p>' if description else ""
    )
    st.markdown(
        '<section class="pt-page-header">'
        f'<div class="pt-eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1>"
        f"{description_html}"
        "</section>",
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "") -> None:
    """Render a compact section heading."""
    detail = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        '<div class="pt-section-header">'
        f'<div class="pt-section-header-copy"><h2>{escape(title)}</h2>{detail}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def section_header_with_meta(title: str, description: str, meta: str) -> None:
    """Render a section heading with a compact technical annotation."""
    st.markdown(
        '<div class="pt-section-header">'
        '<div class="pt-section-header-copy">'
        f"<h2>{escape(title)}</h2><p>{escape(description)}</p></div>"
        f'<div class="pt-section-meta">{escape(meta)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, detail: str = "", *, tone: str = "default") -> None:
    """Render a compact metric card without coupling to Streamlit's metric markup."""
    detail_html = f'<div class="pt-kpi-detail">{escape(detail)}</div>' if detail else ""
    st.markdown(
        f'<div class="pt-kpi pt-kpi-{escape(tone)}">'
        f'<div class="pt-kpi-label">{escape(label)}</div>'
        f'<div class="pt-kpi-value">{escape(value)}</div>'
        f"{detail_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def coverage_bar_html(rows: Iterable[tuple[str, int]]) -> str:
    """Return a compact workflow-status distribution without chart runtime state."""

    items = [(status, max(0, int(count))) for status, count in rows]
    total = sum(count for _, count in items)
    if total <= 0:
        return '<div class="pt-coverage pt-coverage-empty">No review statuses recorded.</div>'
    segments = "".join(
        (
            f'<span class="pt-coverage-segment pt-coverage-{_STATUS_TONES.get(status, "neutral")}" '
            f'style="width:{100 * count / total:.6f}%" '
            f'aria-label="{escape(status.replace("_", " ").title())}: {count}"></span>'
        )
        for status, count in items
        if count > 0
    )
    legend = "".join(
        (
            f'<span class="pt-coverage-key pt-coverage-{_STATUS_TONES.get(status, "neutral")}">'
            '<i aria-hidden="true"></i>'
            f'{escape(status.replace("_", " ").title())} <strong>{count}</strong></span>'
        )
        for status, count in items
    )
    return (
        f'<div class="pt-coverage" aria-label="Review coverage across {total} hands">'
        f'<div class="pt-coverage-track">{segments}</div>'
        f'<div class="pt-coverage-legend">{legend}</div></div>'
    )


def coverage_bar(rows: Iterable[tuple[str, int]]) -> None:
    """Render a review-status distribution in the product design language."""

    st.markdown(coverage_bar_html(rows), unsafe_allow_html=True)


def frequency_bars_html(rows: Iterable[tuple[str, int]]) -> str:
    """Return compact proportional bars for small categorical counts."""

    items = [(str(label), max(0, int(count))) for label, count in rows]
    maximum = max((count for _, count in items), default=0)
    if maximum <= 0:
        return '<div class="pt-frequency pt-frequency-empty">No frequency data recorded.</div>'
    rendered = "".join(
        (
            '<div class="pt-frequency-row">'
            f'<span>{escape(label)}</span>'
            '<div class="pt-frequency-track">'
            f'<i style="width:{100 * count / maximum:.6f}%"></i></div>'
            f"<strong>{count}</strong></div>"
        )
        for label, count in items
    )
    return f'<div class="pt-frequency" aria-label="Category frequencies">{rendered}</div>'


def frequency_bars(rows: Iterable[tuple[str, int]]) -> None:
    """Render compact categorical frequencies without a chart runtime."""

    st.markdown(frequency_bars_html(rows), unsafe_allow_html=True)


def status_badge(status: str, *, label: str | None = None) -> str:
    """Return safe badge HTML for use inside richer component markup."""
    normalized = status.lower().strip()
    tone = _STATUS_TONES.get(normalized, "neutral")
    display = label or normalized.replace("_", " ").title()
    return (
        f'<span class="pt-badge pt-badge-{tone}">'
        f'<span aria-hidden="true" class="pt-badge-dot"></span>{escape(display)}</span>'
    )


def empty_state(title: str, description: str) -> None:
    """Render an intentional empty state instead of a bare informational alert."""
    st.markdown(
        '<div class="pt-empty">'
        '<div class="pt-empty-marker" aria-hidden="true"></div>'
        f"<h3>{escape(title)}</h3><p>{escape(description)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def data_callout(label: str, value: str) -> None:
    """Render a narrow factual callout for saved or computed data."""
    st.markdown(
        '<div class="pt-data-callout">'
        f"<span>{escape(label)}</span><strong>{escape(value)}</strong></div>",
        unsafe_allow_html=True,
    )


def panel(title: str, description: str = "") -> None:
    """Render a compact, non-interactive information panel."""
    detail = f'<div class="pt-panel-copy">{escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="pt-panel"><div class="pt-panel-title">{escape(title)}</div>{detail}</div>',
        unsafe_allow_html=True,
    )


def workflow_step(number: int, title: str, description: str, *, state: str = "pending") -> None:
    """Render one labeled stage in an offline-processing workflow."""
    st.markdown(
        f'<div class="pt-workflow-step pt-workflow-{escape(state)}">'
        f"<span>{number:02d}</span><div><strong>{escape(title)}</strong>"
        f"<p>{escape(description)}</p></div></div>",
        unsafe_allow_html=True,
    )


def trust_badge() -> None:
    """Keep the product's post-session safety boundary visible."""
    st.markdown(
        '<div class="pt-trust"><span></span>Post-session only</div>',
        unsafe_allow_html=True,
    )
