"""
agents/progress_agent.py — Tracks progress and generates adaptive session summaries

Responsibilities:
  1. Analyze the user's vocabulary record and answer history
  2. Identify weak spots (low-accuracy words) and strengths (mastered words)
  3. Return a short natural-language progress summary and recommended next focus
  4. (Optionally) advise the orchestrator on difficulty adjustments
"""

from __future__ import annotations

from state.models import UserState
from agents.base import BaseAgent


PROGRESS_PROMPT = """You are a progress tracker inside a language learning multi-agent system.

Here is the user's current vocabulary data:
{vocab_summary}

Session stats:
- Current streak:  {streak}
- Total words seen: {total_seen}
- Mastered words:  {mastered_count}
- Current level:   {level}

Tasks:
1. Write a 1-sentence encouraging progress summary.
2. Identify up to 3 words the user should review (lowest accuracy, at least 1 attempt).
3. Suggest a difficulty adjustment: "increase", "maintain", or "decrease".

Return ONLY valid JSON:
{{
  "summary":              "<1-sentence progress summary>",
  "words_to_review":      ["word1", "word2"],
  "difficulty_adjustment": "increase | maintain | decrease"
}}"""


class ProgressAgent(BaseAgent):
    name = "progress-agent"

    def run(self, user_state: UserState) -> dict:
        self.clear_logs()
        self.log(f"Analyzing {len(user_state.vocab)} words, level={user_state.level}")

        # Build vocab summary for the prompt
        rows = []
        for word, rec in user_state.vocab.items():
            rows.append(
                f"  {word} ({rec.lang}): {rec.attempts} attempts, "
                f"{rec.correct} correct, accuracy={rec.accuracy}, "
                f"mastered={rec.mastered}"
            )
        vocab_summary = "\n".join(rows) if rows else "No words yet."

        prompt = PROGRESS_PROMPT.format(
            vocab_summary=vocab_summary,
            streak=user_state.streak,
            total_seen=user_state.total_seen,
            mastered_count=user_state.mastered_count,
            level=user_state.level,
        )

        raw = self._call(prompt, max_tokens=400)
        data = self._parse_json(raw)
        self.log(f"Adjustment recommendation: {data.get('difficulty_adjustment')}")
        return data
