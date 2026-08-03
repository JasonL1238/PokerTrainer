"""Check a coaching response against the facts it was actually given.

PLAN.md Phase 9: "Prevent invented hole cards, board cards, actions, pots,
frequencies, action EV, or BB loss," and "Require solver-specific claims to
match retained parsed solver evidence."

The existing eval scores structure, unsafe phrasing, and one equity heuristic.
None of that notices a review that discusses a queen of hearts the hand never
contained, or quotes a check-raise frequency when no solver ever ran. Those are
the fabrications that matter, because a card or a frequency reads as recorded
truth in a way that vague prose does not.

Everything here compares the RESPONSE against the PROMPT. The prompt is the
whole of what the model was told, so a specific claim that is not traceable to
it was invented. This is a detector, not a proof: it is deliberately biased
toward silence on ambiguous text (see AMBIGUOUS_CARD_TOKENS) so that a reported
failure is worth investigating rather than routine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Rank+suit in the notation the prompts use. Case matters: ranks are uppercase
# and suits lowercase, which is what keeps "AD" (an abbreviation) out.
_CARD_RE = re.compile(r"\b([2-9TJQKA])([cdhs])\b")

# Card tokens that are also ordinary English words. "As the pot grows", "Ah,
# but", "the ad" would each otherwise read as a fabricated ace. These are only
# reported when they sit in an unmistakably card-shaped context, which costs
# some recall on three of the fifty-two cards and buys silence on ordinary prose.
AMBIGUOUS_CARD_TOKENS: frozenset[str] = frozenset({"As", "Ah", "Ad"})

# A card run: two or more card tokens separated only by spaces, commas or slashes.
_CARD_RUN_RE = re.compile(
    r"(?:\b[2-9TJQKA][cdhs]\b)(?:\s*[,/]?\s*\b[2-9TJQKA][cdhs]\b)+"
)

# Language that asserts a solver-derived quantity.
_FREQUENCY_RE = re.compile(
    r"(?i)\b\d{1,3}(?:\.\d+)?\s*%\s*(?:of the time|frequency|\bfreq\b)"
    r"|\b(?:checks?|bets?|raises?|calls?|folds?|shoves?)\s+\d{1,3}(?:\.\d+)?\s*%"
)
_EV_RE = re.compile(
    r"(?i)\bEV\b\s*(?:of|=|:)?\s*[-+]?\d|\bexpected value\b\s*(?:of|=|:)?\s*[-+]?\d"
    r"|[-+]?\d+(?:\.\d+)?\s*(?:bb|big blinds)\s+(?:of\s+)?EV"
)
_EXPLOITABILITY_RE = re.compile(r"(?i)\bexploitab\w*\b\s*(?:of|=|:)?\s*[-+]?\d")

# Phrases that name the solver as the source of a claim.
_SOLVER_ATTRIBUTION_RE = re.compile(
    r"(?i)\b(?:solver|gto|equilibrium|texassolver)\b"
)


@dataclass
class GroundingReport:
    """What the response asserted that the prompt does not support."""

    ungrounded_cards: list[str] = field(default_factory=list)
    ungrounded_solver_claims: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded_cards and not self.ungrounded_solver_claims

    @property
    def failures(self) -> list[str]:
        messages = [
            f"Response names card {card!r}, which the prompt never mentions"
            for card in self.ungrounded_cards
        ]
        messages.extend(
            f"Response makes a solver-specific claim with no retained solver "
            f"evidence: {claim!r}"
            for claim in self.ungrounded_solver_claims
        )
        return messages


def cards_in_text(text: str) -> set[str]:
    """Every unambiguous card token, plus ambiguous ones inside a card run."""
    found = {
        match.group(0)
        for match in _CARD_RE.finditer(text)
        if match.group(0) not in AMBIGUOUS_CARD_TOKENS
    }
    # An ambiguous token surrounded by other cards is a card, not a word.
    for run in _CARD_RUN_RE.finditer(text):
        found.update(match.group(0) for match in _CARD_RE.finditer(run.group(0)))
    return found


def ungrounded_cards(prompt: str, response: str) -> list[str]:
    """Cards the response names that the prompt does not contain.

    The prompt side deliberately uses the permissive scan: a card the prompt
    mentions anywhere counts as supplied, because the question is only whether
    the model could have seen it.
    """
    supplied = {match.group(0) for match in _CARD_RE.finditer(prompt)}
    return sorted(cards_in_text(response) - supplied)


def solver_claims(response: str) -> list[str]:
    """Specific quantities that only a solver run could justify."""
    claims: list[str] = []
    for pattern in (_FREQUENCY_RE, _EV_RE, _EXPLOITABILITY_RE):
        claims.extend(match.group(0).strip() for match in pattern.finditer(response))
    # Deduplicate while keeping order, so the report reads in text order.
    return list(dict.fromkeys(claims))


def ungrounded_solver_claims(response: str, *, solver_evidence: str | None) -> list[str]:
    """Solver-shaped claims made without retained solver evidence to match.

    When evidence exists, a claim is accepted if its numeric figure appears in
    that evidence. A frequency the solver never produced is exactly as invented
    as one produced with no solver at all.
    """
    claims = solver_claims(response)
    if not claims:
        # A bare attribution with no figure is prose, not a quantitative claim.
        return []
    if not solver_evidence:
        return claims
    return [claim for claim in claims if not _figures_appear_in(claim, solver_evidence)]


def _figures_appear_in(claim: str, evidence: str) -> bool:
    figures = re.findall(r"\d+(?:\.\d+)?", claim)
    if not figures:
        return True
    return all(figure in evidence for figure in figures)


def check_grounding(
    prompt: str,
    response: str,
    *,
    solver_evidence: str | None = None,
) -> GroundingReport:
    """Compare one response against the prompt that produced it."""
    return GroundingReport(
        ungrounded_cards=ungrounded_cards(prompt, response),
        ungrounded_solver_claims=ungrounded_solver_claims(
            response, solver_evidence=solver_evidence
        ),
    )


# What a retained-but-rejected review says about itself wherever it is rendered.
# It leads with the disposition rather than the detail because the list of
# invented cards means nothing to an operator who does not yet know the answer
# was rejected.
UNGROUNDED_STALE_PREFIX = (
    "Not current analysis: this answer asserted facts the prompt it was "
    "generated from does not support. "
)


def grounding_stale_reason(report: GroundingReport) -> str:
    """The retained-review note for a response that failed its own prompt.

    Empty for a response that passed, so a caller can assign it unconditionally
    and let the empty string mean what it already means everywhere else in the
    coaching tables: nothing is wrong with this row.
    """
    if report.ok:
        return ""
    return UNGROUNDED_STALE_PREFIX + "; ".join(report.failures) + "."


def response_attributes_to_solver(response: str) -> bool:
    """Whether the response credits a solver for anything at all."""
    return bool(_SOLVER_ATTRIBUTION_RE.search(response))
