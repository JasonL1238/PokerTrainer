import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

from poker_tracker.ui.navigation import Page


def test_shell_sets_a_compact_sidebar_width_in_pixels() -> None:
    """``initial_sidebar_state`` is an int on purpose: it is the rail's width.

    Streamlit reads that number as the *default* pixel width, so the rail matches
    the compact type scale while a reader's own drag still wins over it. The
    argument looks like a mistake next to the documented "auto"/"expanded"
    strings, which is exactly why reverting it needs to fail a test.
    """
    tree = ast.parse(Path("app.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_page_config"
    ]
    assert len(calls) == 1, "the shell configures the page exactly once"
    widths = [
        kw.value.value
        for kw in calls[0].keywords
        if kw.arg == "initial_sidebar_state" and isinstance(kw.value, ast.Constant)
    ]
    assert widths, "initial_sidebar_state must stay set"
    assert isinstance(widths[0], int) and not isinstance(widths[0], bool)
    # Streamlit clamps to 200-600; outside that the number is silently ignored.
    assert 200 <= widths[0] <= 600


def test_product_shell_navigation_smoke(monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    app = AppTest.from_file("app.py", default_timeout=20).run()

    assert not list(app.exception)
    assert app.radio[0].options == [
        "Overview",
        "Sessions",
        "Hands",
        "Study",
        "Insights",
        "Import",
        "Settings",
    ]
    assert any("Turn completed hands into sharper decisions" in item.value for item in app.markdown)
    assert any("pt-hero" in item.value for item in app.markdown)
    assert any("pt-poker-stage" in item.value for item in app.markdown)

    expected_headings = {
        Page.SESSIONS: "Sessions",
        Page.HANDS: "Hand library",
        Page.STUDY: "Study",
        Page.INSIGHTS: "Insights",
        Page.IMPORT: "Import",
        Page.SETTINGS: "Settings",
    }
    for page, heading in expected_headings.items():
        app.radio[0].set_value(page)
        app.run()
        assert not list(app.exception)
        assert any(heading in item.value for item in app.markdown)


def test_password_gate_blocks_data_until_valid_password(monkeypatch) -> None:
    monkeypatch.setenv("APP_PASSWORD", "correct-horse")
    monkeypatch.setenv("POKERTRAINER_REQUIRE_AUTH", "true")
    app = AppTest.from_file("app.py", default_timeout=20).run()

    assert not list(app.exception)
    assert len(app.radio) == 0
    assert app.text_input[0].label == "Password"

    app.text_input[0].set_value("correct-horse")
    next(button for button in app.button if button.label == "Continue").click()
    app.run()

    assert not list(app.exception)
    assert app.session_state["authed"] is True
    assert len(app.radio) == 1


def test_required_auth_fails_closed_without_secret(monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("POKERTRAINER_REQUIRE_AUTH", "true")
    app = AppTest.from_file("app.py", default_timeout=20).run()

    assert not list(app.exception)
    assert len(app.radio) == 0
    assert any("APP_PASSWORD" in error.value for error in app.error)


def test_password_gate_uses_product_hero(monkeypatch) -> None:
    monkeypatch.setenv("APP_PASSWORD", "correct-horse")
    monkeypatch.setenv("POKERTRAINER_REQUIRE_AUTH", "true")
    app = AppTest.from_file("app.py", default_timeout=20).run()

    assert not list(app.exception)
    assert any("Review the decisions that define your game" in item.value for item in app.markdown)
    assert any(
        "Completed-hand data appears after sign-in" in item.value
        for item in app.markdown
    )
