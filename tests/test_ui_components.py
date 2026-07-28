from poker_tracker.ui.components import coverage_bar_html, frequency_bars_html


def test_coverage_bar_uses_exact_counts_and_safe_status_labels() -> None:
    rendered = coverage_bar_html(
        [("reviewed", 3), ("unreviewed", 1), ("needs_correction", 1)]
    )

    assert 'aria-label="Review coverage across 5 hands"' in rendered
    assert "Reviewed <strong>3</strong>" in rendered
    assert "Unreviewed <strong>1</strong>" in rendered
    assert "Needs Correction <strong>1</strong>" in rendered
    assert "width:60.000000%" in rendered


def test_coverage_bar_handles_an_empty_distribution() -> None:
    rendered = coverage_bar_html([("reviewed", 0)])

    assert "No review statuses recorded." in rendered


def test_frequency_bars_scale_to_the_largest_count() -> None:
    rendered = frequency_bars_html([("River Decision", 4), ("Big Pot", 2)])

    assert "River Decision" in rendered
    assert "width:100.000000%" in rendered
    assert "width:50.000000%" in rendered


def test_frequency_bars_handle_empty_counts() -> None:
    assert "No frequency data recorded." in frequency_bars_html([])
