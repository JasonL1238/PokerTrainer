from poker_tracker.coaching.safety import ensure_post_session_prompt


class FakeLLMProvider:
    """Deterministic provider used only to test parsing and persistence."""

    provider_name = "test"
    model_name = "deterministic-fixture"

    def generate_hand_review(self, prompt: str) -> str:
        ensure_post_session_prompt(prompt)
        return _structured_response(
            {
                "Hand Summary": "Post-session summary based on the supplied hand history.",
                "Theory Coach": "Review range, position, pot odds, and sizing.",
                "Exploit Coach": "Use only provided player notes and stored actions.",
                "EV / Math Notes": "Use supplied math facts only. Do not invent equities.",
                "Mistake Severity": "Medium.",
                "Best Alternative Line": "Compare two recorded-hand alternatives.",
                "Study Lesson": "Write down the key decision and its assumptions.",
                "Next Review Question": "Which stored fact most changes the best line?",
            }
        )

    def generate_session_review(self, prompt: str) -> str:
        ensure_post_session_prompt(prompt)
        return _structured_response(
            {
                "Session Summary": "Post-session summary from supplied session data.",
                "Biggest Leaks": "Prioritize repeated tags and large losing hands.",
                "Best Played Spots": "Review the biggest wins for value maximization.",
                "Theory Study Priorities": "Study range construction and street planning.",
                "Exploit Study Priorities": "Use only provided notes and stored hands.",
                "Hands To Review Again": "Revisit unreviewed and large-pot hands.",
                "Next Study Plan": "Review three hands and compare the recorded math.",
            }
        )


def _structured_response(sections: dict[str, str]) -> str:
    return "\n\n".join(f"{heading}:\n{body}" for heading, body in sections.items())
