"""Phase 9: a coaching response may not assert facts it was never given.

A fabricated card or frequency reads as recorded truth in a way that vague prose
does not, which is why these two get a mechanical check and general waffle does
not. The detector is deliberately biased toward silence on ambiguous text: a
reported failure should be worth investigating, not routine.
"""

from __future__ import annotations

import pytest

from poker_tracker.coaching.grounding import (
    AMBIGUOUS_CARD_TOKENS,
    cards_in_text,
    check_grounding,
    grounding_stale_reason,
    response_attributes_to_solver,
    solver_claims,
    ungrounded_cards,
    ungrounded_solver_claims,
)

PROMPT = (
    "Completed hand review.\n"
    "Hero cards: Kd Qd\n"
    "Board: 2c 7d 9h Ts Jc\n"
    "Preflop: Hero raises to 3bb, BB calls.\n"
    "Flop: checked through.\n"
)


# --- Card fabrication -------------------------------------------------------


def test_a_card_the_hand_never_contained_is_flagged():
    response = "Your flush draw with Qh was strong here."
    assert ungrounded_cards(PROMPT, response) == ["Qh"]


def test_cards_from_the_prompt_are_not_flagged():
    response = "With Kd Qd on a 2c 7d 9h board you have two overcards."
    assert ungrounded_cards(PROMPT, response) == []


def test_hand_classes_are_not_mistaken_for_cards():
    """"AKs" and "T9o" are range notation, not specific cards."""
    response = "Your range here is AKs, KQs, T9o and similar broadways."
    assert ungrounded_cards(PROMPT, response) == []


@pytest.mark.parametrize(
    "response",
    [
        "As the pot grows, your equity shifts.",
        "Ah, but the turn changes things.",
        "That line is an ad for a wider range.",
    ],
)
def test_ordinary_english_is_not_read_as_an_ace(response):
    """"As", "Ah" and "ad" are words before they are cards."""
    assert ungrounded_cards(PROMPT, response) == []


def test_an_ambiguous_token_inside_a_card_run_is_still_read_as_a_card():
    """Surrounded by other cards, "As" is unmistakably the ace of spades."""
    response = "He turned over As Kh for the nut flush."
    flagged = ungrounded_cards(PROMPT, response)
    assert "As" in flagged
    assert "Kh" in flagged


def test_the_ambiguity_list_is_exactly_the_english_words():
    assert AMBIGUOUS_CARD_TOKENS == {"As", "Ah", "Ad"}


def test_uppercase_suits_are_not_cards():
    """"AD" is an abbreviation; the notation this project uses is "Ad"."""
    assert cards_in_text("The AD and the KH were not dealt.") == set()


# --- Solver claims ----------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        "You should check-raise 32% of the time here.",
        "This spot is a fold 78% of the time.",
        "Bet 25% frequency on this texture.",
    ],
)
def test_a_frequency_without_solver_evidence_is_flagged(response):
    claims = ungrounded_solver_claims(response, solver_evidence=None)
    assert claims, f"no claim detected in {response!r}"
    # The figure itself is what makes it a quantitative claim.
    assert any(char.isdigit() for claim in claims for char in claim)


def test_an_ev_claim_without_solver_evidence_is_flagged():
    response = "Betting has an EV of 1.4bb over checking."
    claims = ungrounded_solver_claims(response, solver_evidence=None)
    assert claims, "an EV figure with no solver run is invented"


def test_a_frequency_matching_retained_evidence_is_accepted():
    response = "The solver checks here 65% of the time."
    evidence = '{"check": 65.0, "bet": 35.0}'
    assert ungrounded_solver_claims(response, solver_evidence=evidence) == []


def test_a_frequency_the_solver_never_produced_is_still_flagged():
    """Evidence existing is not the same as the claim matching it."""
    response = "The solver checks here 65% of the time."
    evidence = '{"check": 12.0, "bet": 88.0}'
    assert ungrounded_solver_claims(response, solver_evidence=evidence)


def test_prose_about_the_solver_without_a_figure_is_not_a_claim():
    response = "A solver would probably prefer a smaller size here."
    assert ungrounded_solver_claims(response, solver_evidence=None) == []
    # It is still recognizably an attribution, which the caller may care about.
    assert response_attributes_to_solver(response) is True


def test_exploitability_is_a_solver_claim():
    assert solver_claims("The run converged to exploitability of 0.4%.")


# --- Combined ---------------------------------------------------------------


def test_a_clean_response_passes():
    response = (
        "With Kd Qd on 2c 7d 9h you hold two overcards and a backdoor draw. "
        "Checking back is reasonable given the board texture."
    )
    report = check_grounding(PROMPT, response)
    assert report.ok is True
    assert report.failures == []


def test_failures_name_what_was_invented():
    response = "Your Qh gave you a flush draw, so bet 70% of the time."
    report = check_grounding(PROMPT, response, solver_evidence=None)
    assert report.ok is False
    joined = " ".join(report.failures)
    assert "Qh" in joined
    assert "solver-specific" in joined


def test_solver_evidence_only_excuses_the_solver_half():
    response = "Your Qh is a flush draw; solver bets 70% of the time."
    report = check_grounding(PROMPT, response, solver_evidence='{"bet": 70.0}')
    assert report.ungrounded_solver_claims == []
    assert report.ungrounded_cards == ["Qh"]
    assert report.ok is False


# --- What a rejected response says about itself -----------------------------


def test_a_passing_report_leaves_the_retained_reason_empty():
    """Empty is what "nothing is wrong with this row" already means in that column."""
    report = check_grounding(PROMPT, "Kd Qd is two overcards on 2c 7d 9h.")
    assert grounding_stale_reason(report) == ""


def test_the_retained_reason_leads_with_the_disposition_then_the_detail():
    """A reader who does not yet know the answer was rejected cannot use a card list."""
    report = check_grounding(PROMPT, "Your Qh gave you the flush draw.")
    reason = grounding_stale_reason(report)
    assert reason.startswith("Not current analysis:")
    assert "Qh" in reason
    assert reason.endswith(".")
