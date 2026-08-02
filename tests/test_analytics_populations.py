"""The metric-population layer, verified by recomputing from the hand rows.

Analytics used to compute over the whole ``hands`` table with no population
filter. A win rate that mixes a reviewed hand, an unreconciled CV draft and a
manually entered hand is one number standing for four epistemic states, and the
product printed it as though it meant one thing.

Every metric test here computes the same figure a second time, independently,
straight out of SQLite -- fetch the rows, apply the population rule by hand, sum
the results -- and asserts the two agree. That is the only assertion shape that
catches the failure this layer is most likely to have: a filter that silently
drops hands still returns a plausible number, and only a second computation from
the source rows disagrees with it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from poker_tracker.math.accounting import LedgerError
from poker_tracker.math.analytics import (
    DEFAULT_POPULATION,
    EVIDENCE_CLASSES,
    MIN_HANDS_FOR_INTERVAL,
    MIN_HANDS_FOR_RATE,
    POPULATIONS,
    RATE_CAVEAT,
    Metric,
    PopulationKey,
    aggregate_study_themes,
    build_hand_evidence,
    classify_evidence,
    compute_population_metrics,
    compute_session_stats,
    normalize_theme,
    population_exclusion,
    resolve_hero_result,
    select_population,
)
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import accounting_is_established

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _open_db(tmp_path: Path, name: str = "populations.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _clean_evidence() -> dict[str, object]:
    return dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="fold_win",
            boundary_confidence=0.93,
            layout_supported=True,
            table_size=6,
        )
    )


def _add_hand(
    db: PokerDatabase,
    session_id: int,
    hand_number: int,
    *,
    source_type: str = "manual",
    review_status: str = "unreviewed",
    completion_status: str | None = None,
    study_inclusion: str = "auto",
    hero_bb_won: float | None = None,
    tags: list[str] | None = None,
) -> Hand:
    if completion_status is None:
        completion_status = "not_applicable" if source_type == "manual" else "complete"
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=hand_number,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            hero_bb_won=hero_bb_won,
            review_status=review_status,  # type: ignore[arg-type]
            source_type=source_type,  # type: ignore[arg-type]
            completion_status=completion_status,  # type: ignore[arg-type]
            completion_evidence={} if source_type == "manual" else _clean_evidence(),
            tags=tags or [],
        )
    )
    # ``create_hand`` always starts at 'auto'; study inclusion is an operator
    # preference set after the hand exists.
    if study_inclusion != "auto":
        assert hand.id is not None
        hand = db.update_study_inclusion(hand.id, study_inclusion)
    return hand


def _add_fold_line(db: PokerDatabase, hand: Hand, *, bet: float = 40.0) -> None:
    """Hero bets, villain folds: the recording, not a declaration, names the winner.

    A showdown winner is an operator declaration nothing corroborates, and
    ``accounting_is_established`` refuses it until it is attested. A fold line is
    the shortest hand whose reconciliation is established on its own evidence,
    which is what the ``reconciled`` population is supposed to admit.
    """
    assert hand.id is not None
    for key, name, hero in (("hero", "Hero", True), ("villain", "Villain", False)):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                is_hero=hero,
                starting_stack=1000,
            )
        )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="villain",
            street="preflop",
            action_index=1,
            player_name="Villain",
            action_type="bet",
            amount=bet,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="hero",
            street="preflop",
            action_index=2,
            player_name="Hero",
            action_type="call",
            amount=bet,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="villain",
            street="flop",
            action_index=3,
            player_name="Villain",
            action_type="fold",
        )
    )


def _reconcile_to_established(db: PokerDatabase, hand: Hand) -> float:
    """Settle a fold-line hand and return the ledger's hero result."""
    assert hand.id is not None
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=None,
            )
        ],
    )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    accounting = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert accounting_is_established(stored, accounting), (
        "fixture precondition: this hand must reconcile without an attestation"
    )
    return accounting.ledger.net_results["hero"]


def _coach(
    db: PokerDatabase, hand_id: int, lesson: str, *, stale: bool = False
) -> None:
    db.create_coaching_response(
        CoachingResponse(
            provider_name="test",
            model_name="test-model",
            raw_prompt="prompt",
            raw_response=f"Study Lesson: {lesson}",
            review_type="hand",
            hand_id=hand_id,
            parsed_sections={"Study Lesson": lesson},
            is_stale=stale,
        )
    )


@pytest.fixture
def mixed_library(tmp_path: Path) -> tuple[PokerDatabase, dict[str, Hand], float]:
    """One session holding every state the population rule has to tell apart."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Mixed", date_played=date(2026, 3, 1)))
    assert session.id is not None
    hands = {
        # Confirmed, result observed on the row.
        "manual_reviewed": _add_hand(
            db,
            session.id,
            1,
            review_status="reviewed",
            hero_bb_won=12,
            tags=["BIG_POT"],
        ),
        # A CV draft: never corrected, never confirmed, reconstruction partial.
        "cv_draft": _add_hand(
            db,
            session.id,
            2,
            source_type="cv_import",
            completion_status="partial",
            hero_bb_won=-5,
            tags=["LOW_CONFIDENCE", "BIG_POT"],
        ),
        # Corrected but not yet marked reviewed.
        "corrected": _add_hand(
            db,
            session.id,
            3,
            source_type="corrected_cv",
            hero_bb_won=3,
            tags=["BIG_POT"],
        ),
        # Confirmed and reconciled; its result comes from the ledger. Promoted
        # after the settlement below, because a settlement write demotes a
        # reviewed hand to 'needs_correction' by design.
        "cv_reviewed": _add_hand(
            db,
            session.id,
            4,
            source_type="cv_import",
            tags=["BIG_POT", "RIVER_DECISION"],
        ),
        # Confirmed, but the operator excluded it from study.
        "skipped": _add_hand(
            db,
            session.id,
            5,
            review_status="reviewed",
            study_inclusion="skip",
            hero_bb_won=100,
        ),
        # review_status = 'reviewed' on a row whose reconstruction is uncertain:
        # the shape a payload or a hand-edited row can have, and the reason the
        # population tests completion_status rather than trusting one column.
        "forged_reviewed": _add_hand(
            db,
            session.id,
            6,
            source_type="cv_import",
            review_status="reviewed",
            completion_status="uncertain",
            hero_bb_won=50,
        ),
    }
    _add_fold_line(db, hands["cv_reviewed"])
    derived = _reconcile_to_established(db, hands["cv_reviewed"])
    assert hands["manual_reviewed"].id is not None
    assert hands["cv_reviewed"].id is not None
    db.update_hand_status(hands["cv_reviewed"].id, "reviewed")
    _coach(db, hands["manual_reviewed"].id, "Fold more turns out of position")
    _coach(db, hands["cv_reviewed"].id, "Bluff the river more", stale=True)
    yield db, hands, derived
    db.close()


# ---------------------------------------------------------------------------
# The independent recomputation the phase asks for
# ---------------------------------------------------------------------------


def _recompute_from_rows(
    db: PokerDatabase, population: PopulationKey
) -> tuple[list[int], list[float]]:
    """The population and its results, computed straight off the stored rows.

    Deliberately does not import any of the population helpers under test: it
    re-states the rule from ``POPULATIONS[...].rule`` against the columns, so a
    filter that drops a hand disagrees with this rather than with itself.
    """
    member_ids: list[int] = []
    results: list[float] = []
    for hand in db.fetch_all_hands():
        assert hand.id is not None
        if hand.study_inclusion == "skip":
            continue
        if hand.review_status != "reviewed":
            continue
        if hand.completion_status not in ("complete", "not_applicable"):
            continue
        if population == "reconciled":
            try:
                accounting = reconcile_persisted_hand(db, hand.id)
            except LedgerError:
                accounting = None
            if not accounting_is_established(hand, accounting):
                continue
        resolved = resolve_hero_result(db, hand)
        member_ids.append(hand.id)
        if resolved.value is not None:
            results.append(resolved.value)
    return sorted(member_ids), results


def test_confirmed_population_matches_a_recomputation_from_the_hand_rows(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """The filter and a hand-written recomputation must select the same rows.

    Before the population layer existed every metric ran over ``fetch_all_hands``
    and this test had nothing to compare: six hands in four states all counted
    once, so any filter and no filter produced the same number.
    """
    db, hands, _ = mixed_library
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")

    expected_ids, _ = _recompute_from_rows(db, "confirmed")
    assert sorted(member.hand_id or 0 for member in snapshot.members) == expected_ids
    assert expected_ids == sorted(
        [hands["manual_reviewed"].id or 0, hands["cv_reviewed"].id or 0]
    )
    assert snapshot.considered_count == 6


def test_every_excluded_hand_is_accounted_for_with_a_reason(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """A hand that leaves the population must say which clause dropped it."""
    db, _, _ = mixed_library
    evidence = build_hand_evidence(db, db.fetch_all_hands())
    snapshot = select_population(evidence, "confirmed")

    assert snapshot.excluded_by_reason == {
        "operator_excluded": 1,
        "not_reviewed": 2,
        "incomplete": 1,
    }
    assert snapshot.size + sum(snapshot.excluded_by_reason.values()) == len(evidence)


def test_a_reviewed_row_with_an_uncertain_reconstruction_is_still_excluded(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """review_status alone is one column, and one column is one forgery away.

    The promotion guard will not let an uncertain hand reach 'reviewed' through
    the product, but an import payload and a hand-edited row are not the product.
    """
    db, hands, _ = mixed_library
    evidence = {
        item.hand_id: item for item in build_hand_evidence(db, db.fetch_all_hands())
    }
    forged = evidence[hands["forged_reviewed"].id]

    assert forged.review_status == "reviewed"
    assert population_exclusion(forged, "confirmed") == "incomplete"
    assert population_exclusion(forged, "all_saved") is None


def test_metrics_agree_with_a_sum_over_the_source_rows(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """Net, average and bb/100 recomputed from the rows, not from each other."""
    db, _, derived = mixed_library
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    metrics = compute_population_metrics(snapshot)

    _, expected_results = _recompute_from_rows(db, "confirmed")
    assert sorted(expected_results) == sorted([12.0, derived])

    assert metrics.net_result.value == pytest.approx(sum(expected_results))
    assert metrics.average_result.value == pytest.approx(
        sum(expected_results) / len(expected_results)
    )
    assert metrics.bb_per_100.value == pytest.approx(
        100 * sum(expected_results) / len(expected_results)
    )
    assert metrics.hand_count.value == 2


def test_the_all_saved_population_is_the_number_the_product_used_to_print(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """The unfiltered figure still exists, and it disagrees with the confirmed one.

    If these two ever matched, the population layer would be decorative.
    """
    db, _, derived = mixed_library
    evidence = build_hand_evidence(db, db.fetch_all_hands())
    unfiltered = compute_population_metrics(select_population(evidence, "all_saved"))
    confirmed = compute_population_metrics(select_population(evidence, "confirmed"))

    assert unfiltered.net_result.value == pytest.approx(12 - 5 + 3 + derived + 100 + 50)
    assert confirmed.net_result.value == pytest.approx(12 + derived)
    assert unfiltered.net_result.value != pytest.approx(confirmed.net_result.value)
    assert "unconfirmed CV drafts" in " ".join(unfiltered.net_result.caveats)


def test_the_reconciled_population_keeps_only_ledger_established_hands(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    db, hands, derived = mixed_library
    evidence = build_hand_evidence(db, db.fetch_all_hands())
    snapshot = select_population(evidence, "reconciled")

    assert [member.hand_id for member in snapshot.members] == [hands["cv_reviewed"].id]
    assert snapshot.excluded_by_reason["accounting_not_established"] == 1
    metrics = compute_population_metrics(snapshot)
    assert metrics.net_result.value == pytest.approx(derived)
    assert metrics.net_result.basis_mix["reconciled"] == 1
    assert metrics.net_result.basis_mix["observed"] == 0


# ---------------------------------------------------------------------------
# Denominators, coverage and the small-sample verdict
# ---------------------------------------------------------------------------


def test_every_metric_carries_a_denominator_and_a_coverage_statement(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """The return type is the enforcement: there is no metric without context.

    A bare float invites a caption written from memory, which is how a
    draft-inflated win rate got printed as a fact.
    """
    db, _, _ = mixed_library
    evidence = build_hand_evidence(db, db.fetch_all_hands())
    for key in POPULATIONS:
        snapshot = select_population(evidence, key)  # type: ignore[arg-type]
        metrics = compute_population_metrics(snapshot)
        assert metrics.metrics, key
        for metric in metrics.metrics:
            assert isinstance(metric, Metric)
            assert metric.population.key == key
            assert metric.coverage.eligible == snapshot.size or metric.key == "hand_count"
            assert metric.coverage.considered == 6
            assert metric.support
            assert str(metric.coverage.included) in metric.support
            assert str(metric.coverage.considered) in metric.support
            assert metric.sample.verdict in ("empty", "below_threshold", "adequate")


def test_a_metric_the_sample_cannot_carry_refuses_to_name_a_figure(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """Two hands is below the rate floor, so the rate is not printed at all."""
    db, _, _ = mixed_library
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    metrics = compute_population_metrics(snapshot)

    assert metrics.bb_per_100.value is not None
    assert metrics.bb_per_100.sample.verdict == "below_threshold"
    assert metrics.bb_per_100.sample.threshold == MIN_HANDS_FOR_RATE
    assert metrics.bb_per_100.is_reportable is False
    assert metrics.bb_per_100.headline == "Not enough evidence"
    # The support line is still there: the caller says how little it had.
    assert metrics.bb_per_100.support
    assert RATE_CAVEAT in metrics.bb_per_100.caveats
    # A count is not an estimate, so it prints at any non-zero sample size.
    assert metrics.hand_count.is_reportable is True
    assert metrics.hand_count.headline == "2 hands"


def test_an_adequate_sample_prints_the_rate_and_still_carries_the_caveat(
    tmp_path: Path,
) -> None:
    db = _open_db(tmp_path, "adequate.db")
    session = db.create_session(Session(name="Long", date_played=date(2026, 3, 2)))
    assert session.id is not None
    for number in range(1, MIN_HANDS_FOR_RATE + 1):
        _add_hand(
            db,
            session.id,
            number,
            review_status="reviewed",
            hero_bb_won=float(number % 5) - 2,
        )
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    metrics = compute_population_metrics(snapshot)

    expected = [float(number % 5) - 2 for number in range(1, MIN_HANDS_FOR_RATE + 1)]
    assert metrics.bb_per_100.sample.verdict == "adequate"
    assert metrics.bb_per_100.is_reportable is True
    assert metrics.bb_per_100.value == pytest.approx(100 * sum(expected) / len(expected))
    assert "bb/100" in metrics.bb_per_100.headline
    assert RATE_CAVEAT in metrics.bb_per_100.caveats
    assert metrics.bb_per_100.interval is not None
    low, high = metrics.bb_per_100.interval
    assert low < metrics.bb_per_100.value < high
    db.close()


def test_a_result_free_population_reports_zero_coverage_rather_than_zero_bb(
    tmp_path: Path,
) -> None:
    """No recorded result is not a result of zero, and the coverage says so."""
    db = _open_db(tmp_path, "resultless.db")
    session = db.create_session(Session(name="Blank", date_played=date(2026, 3, 3)))
    assert session.id is not None
    _add_hand(db, session.id, 1, review_status="reviewed")
    _add_hand(db, session.id, 2, review_status="reviewed")
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    metrics = compute_population_metrics(snapshot)

    assert metrics.net_result.value is None
    assert metrics.net_result.headline == "Not enough evidence"
    assert metrics.net_result.coverage.included == 0
    assert metrics.net_result.coverage.eligible == 2
    assert metrics.net_result.coverage.is_complete is False
    assert metrics.hand_count.value == 2
    db.close()


def test_interval_needs_two_results(
    tmp_path: Path,
) -> None:
    db = _open_db(tmp_path, "one.db")
    session = db.create_session(Session(name="One", date_played=date(2026, 3, 4)))
    assert session.id is not None
    _add_hand(db, session.id, 1, review_status="reviewed", hero_bb_won=4)
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    metrics = compute_population_metrics(snapshot)

    assert MIN_HANDS_FOR_INTERVAL == 2
    assert metrics.bb_per_100.interval is None
    assert metrics.bb_per_100.interval_label == "No interval at this sample size"
    db.close()


# ---------------------------------------------------------------------------
# Evidence classes
# ---------------------------------------------------------------------------


def test_evidence_classes_partition_the_population(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """The parts sum to the whole, which is what lets them be printed beside it."""
    db, _, _ = mixed_library
    evidence = build_hand_evidence(db, db.fetch_all_hands())
    snapshot = select_population(evidence, "all_saved")

    mix = snapshot.evidence_mix
    assert set(mix) == set(EVIDENCE_CLASSES)
    assert sum(mix.values()) == snapshot.size == 6
    assert mix == {
        "reviewed": 4,
        "corrected_cv": 1,
        "cv_draft": 1,
        "manual": 0,
    }


def test_a_reviewed_cv_hand_is_classified_by_its_review_not_its_source(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    db, hands, _ = mixed_library
    stored = db.fetch_hand(hands["cv_reviewed"].id or 0)
    assert stored is not None
    assert stored.source_type == "cv_import"
    assert classify_evidence(stored) == "reviewed"

    draft = db.fetch_hand(hands["cv_draft"].id or 0)
    corrected = db.fetch_hand(hands["corrected"].id or 0)
    assert draft is not None and corrected is not None
    assert classify_evidence(draft) == "cv_draft"
    assert classify_evidence(corrected) == "corrected_cv"


def test_result_basis_separates_a_derived_figure_from_an_observed_one(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    db, hands, derived = mixed_library
    evidence = {
        item.hand_id: item for item in build_hand_evidence(db, db.fetch_all_hands())
    }

    reconciled = evidence[hands["cv_reviewed"].id]
    assert reconciled.result_basis == "reconciled"
    assert reconciled.result_value == pytest.approx(derived)
    # Nothing was ever recorded on this row; the figure is entirely a derivation.
    stored = db.fetch_hand(hands["cv_reviewed"].id or 0)
    assert stored is not None and stored.hero_bb_won is None

    observed = evidence[hands["manual_reviewed"].id]
    assert observed.result_basis == "observed"
    assert observed.result_value == 12

    blank = evidence[hands["cv_draft"].id]
    assert blank.result_basis == "observed"


def test_session_stats_provenance_counts_are_the_same_notion(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """``SessionStats`` already counted this split; the layer builds on it.

    Both now read one resolver, so the session dashboard and Insights cannot
    disagree about which of a hand's two possible results is the one being shown.
    """
    db, hands, _ = mixed_library
    session_id = hands["manual_reviewed"].session_id
    stats = compute_session_stats(db, session_id)
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "all_saved")

    assert stats.reconciled_result_count == snapshot.result_basis_mix["reconciled"] == 1
    assert stats.observed_result_count == snapshot.result_basis_mix["observed"] == 5


# ---------------------------------------------------------------------------
# Themes and stale coaching
# ---------------------------------------------------------------------------


def test_theme_counts_carry_their_denominator(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """A theme on 3 of 4 hands and 3 of 400 must not render identically."""
    db, _, _ = mixed_library
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    themes = aggregate_study_themes(snapshot)

    by_name = {theme.theme: theme for theme in themes.themes}
    assert by_name["Big Pot"].hands == 2
    assert by_name["Big Pot"].denominator == 2
    assert by_name["Big Pot"].share == pytest.approx(100.0)
    assert by_name["River Decision"].hands == 1
    assert by_name["River Decision"].denominator == 2
    assert by_name["River Decision"].share == pytest.approx(50.0)
    # The draft's LOW_CONFIDENCE tag is outside the confirmed population.
    assert "Low Confidence" not in by_name


def test_stale_coaching_is_excluded_from_the_theme_index_and_reported(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """Retained history is not current evidence.

    A coaching review a correction invalidated is kept deliberately, but letting
    it into the index means a conclusion the operator has already overruled keeps
    voting on the current one.
    """
    db, _, _ = mixed_library
    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    themes = aggregate_study_themes(snapshot)

    names = {theme.theme for theme in themes.themes}
    assert "Fold more turns out of position" in names
    assert "Bluff the river more" not in names
    assert themes.stale_coaching_reviews_excluded == 1
    assert themes.stale_coaching_hands == 1
    assert themes.stale_only_themes_excluded == ("Bluff the river more",)
    assert "1 stale coaching review(s)" in themes.exclusion_statement


def test_a_current_coaching_review_puts_its_lesson_back_in_the_index(
    mixed_library: tuple[PokerDatabase, dict[str, Hand], float],
) -> None:
    """The exclusion is about staleness, not about coaching."""
    db, hands, _ = mixed_library
    assert hands["cv_reviewed"].id is not None
    _coach(db, hands["cv_reviewed"].id, "Bluff the river more")

    snapshot = select_population(build_hand_evidence(db, db.fetch_all_hands()), "confirmed")
    themes = aggregate_study_themes(snapshot)

    by_name = {theme.theme: theme for theme in themes.themes}
    assert by_name["Bluff the river more"].hands == 1
    assert by_name["Bluff the river more"].source == "coaching"
    assert themes.stale_only_themes_excluded == ()
    assert themes.stale_coaching_reviews_excluded == 1


def test_theme_normalization_keeps_the_claim_and_drops_the_argument() -> None:
    """Without this every hand gets a unique theme and every count is 1."""
    assert (
        normalize_theme("Fold more turns. Villain is polarised and you are capped.")
        == "Fold more turns"
    )
    assert normalize_theme("  spaced   out \n lesson ") == "spaced out lesson"
    assert normalize_theme("") == ""
    long = normalize_theme("word " * 40)
    assert long.endswith("…")
    assert len(long) <= 81


def test_default_population_is_the_narrow_one() -> None:
    """The default cannot be the population that mixes four epistemic states."""
    assert DEFAULT_POPULATION == "confirmed"
    assert POPULATIONS[DEFAULT_POPULATION].rule.startswith("review_status = 'reviewed'")
    assert "CV drafts" in POPULATIONS["confirmed"].excludes
