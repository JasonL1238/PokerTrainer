from inspect import getsource

from poker_tracker.ui.ui_theme import _THEME_CSS, brand_header


def test_theme_uses_product_tokens_and_responsive_breakpoints() -> None:
    assert "--pt-bg: #121418" in _THEME_CSS
    assert "--pt-accent: #D9A441" in _THEME_CSS
    assert "--pt-font-display: \"IBM Plex Sans\"" in _THEME_CSS
    assert "--pt-font-sans: \"IBM Plex Sans\"" in _THEME_CSS
    assert "Syne" not in _THEME_CSS
    assert "Arial Narrow" not in _THEME_CSS
    assert "max-width: 1760px" in _THEME_CSS
    assert "@media (max-width: 1100px)" in _THEME_CSS
    assert "@media (max-width: 900px)" in _THEME_CSS
    assert "@media (max-width: 720px)" in _THEME_CSS
    assert "@media (max-width: 520px)" in _THEME_CSS


def test_theme_preserves_button_contrast_and_scrollable_tabs() -> None:
    assert ".stButton > button p" in _THEME_CSS
    assert 'button[kind="primary"] p' in _THEME_CSS
    assert "overflow-x: auto" in _THEME_CSS
    assert "overscroll-behavior-x: contain" in _THEME_CSS


def test_theme_respects_reduced_motion() -> None:
    assert "@media (prefers-reduced-motion: reduce)" in _THEME_CSS
    assert "animation-duration: .001ms !important" in _THEME_CSS
    assert "transition-duration: .001ms !important" in _THEME_CSS


def test_shell_uses_text_brand_mark_and_compact_spacing() -> None:
    assert 'aria-hidden="true"><span>PT</span></div>' in getsource(brand_header)
    assert "transform: rotate(45deg)" in _THEME_CSS
    assert ".pt-empty-marker" in _THEME_CSS
    assert "border-radius: 999px" not in _THEME_CSS


def test_math_workspace_has_compact_scoped_density() -> None:
    assert ".st-key-math_workspace" in _THEME_CSS
    assert 'min-height: 54px' in _THEME_CSS
    assert 'max-height: 13rem' in _THEME_CSS


def test_import_workspace_organizes_without_crushing_type() -> None:
    assert ".st-key-import_workspace" in _THEME_CSS
    assert ".pt-import-meta" in _THEME_CSS
    assert ".st-key-import_session_target" in _THEME_CSS
    assert ".st-key-import_collect_bar" in _THEME_CSS
    # Import is the product's one long-form reading surface, so the compact scale
    # stops here: its body copy keeps a floor rather than following the shell all
    # the way down. These are the smallest sizes on the wiring-up path.
    assert ".pt-workflow-step strong" in _THEME_CSS
    assert "font-size: .82rem; line-height: 1.3" in _THEME_CSS
    assert "font-size: .72rem; line-height: 1.4" in _THEME_CSS
    assert "font-size: .74rem; line-height: 1.4" in _THEME_CSS


def test_shell_density_comes_from_one_scale_and_one_control_height() -> None:
    """Compact density is two tokens, not a sweep of per-widget numbers.

    A future edit that reaches for a literal height on a button or a wider gap on
    one workspace is what these pin against -- the whole point of the tokens is
    that the next density change stays a two-line edit.
    """
    assert "--pt-control-h: 30px" in _THEME_CSS
    assert "--pt-control-h-touch: 44px" in _THEME_CSS
    assert "min-height: var(--pt-control-h)" in _THEME_CSS
    assert "--pt-space-2: 0.36rem" in _THEME_CSS
    assert '[data-testid="stVerticalBlock"] { gap: var(--pt-space-2); }' in _THEME_CSS
    # A phone gets its 44px targets back even though the desktop shell is dense.
    assert "min-height: var(--pt-control-h-touch)" in _THEME_CSS


def test_markdown_block_owns_its_trailing_space() -> None:
    """Both halves of Streamlit's margin pair must be zeroed, or cards overlap.

    Streamlit ships ``p, ol, ul { margin-bottom: 1rem }`` against a -1rem on the
    markdown container. Those cancel for prose but not for a custom HTML block
    ending in a div -- every ``pt-*`` card -- where the stock 1rem vertical gap
    was silently absorbing the difference. Once that gap is tightened, dropping
    either half alone pulls the next element up on top of the card above it,
    which is a real rendering bug and not a spacing preference.
    """
    assert '[data-testid="stMarkdownContainer"] { margin-bottom: 0; }' in _THEME_CSS
    assert '[data-testid="stMarkdownContainer"] > :last-child { margin-bottom: 0; }' in _THEME_CSS


def test_evidence_review_stacks_at_narrow_widths() -> None:
    assert ".pt-evidence-impact" in _THEME_CSS
    assert ".pt-evidence-position > span { white-space: nowrap; }" in _THEME_CSS
    assert ".pt-evidence-verdict.is-correct" in _THEME_CSS
    assert '[class*="st-key-evidence_navigation_"]' in _THEME_CSS
    assert '[class*="st-key-evidence_summary_"]' in _THEME_CSS
    assert '[class*="st-key-evidence_comparison_"]' in _THEME_CSS


def test_sidebar_brand_aligns_with_main_content_before_scroll() -> None:
    assert '[data-testid="stSidebarUserContent"]' in _THEME_CSS
    assert "margin-top: -36px" in _THEME_CSS
    assert "margin-top: -44px" in _THEME_CSS
