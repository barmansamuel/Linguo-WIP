"""
agents/orchestrator.py — Central coordinator for the multi-agent workflow

Workflow per session turn:
  generate_sentence()
    └─ SentenceAgent.run()       → GeneratedSentence
        └─ RAGDictionary.lookup() (context enrichment)
        └─ RAGDictionary.add_entry() (new word registration)

  check_answer()
    └─ EvaluatorAgent.run()      → EvaluationResult
        └─ RAGDictionary.lookup() (synonym checking)
    └─ UserState.record_answer() → updates vocab/streak/history

  get_hint()
    └─ HintAgent.run()           → str (hint text)
        └─ RAGDictionary.exact_lookup() (context)

  get_progress()
    └─ ProgressAgent.run()       → dict (summary + recommendations)

All agent logs are collected here and returned to the UI.
"""

from __future__ import annotations

from rag.dictionary import RAGDictionary
from state.models import UserState, GeneratedSentence, EvaluationResult
from agents.sentence_agent import SentenceAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.hint_agent import HintAgent
from agents.progress_agent import ProgressAgent


class Orchestrator:
    def __init__(self):
        self.rag   = RAGDictionary()
        self.state = UserState()

        # Instantiate all agents, sharing the same RAG dictionary
        self.sentence_agent  = SentenceAgent(self.rag)
        self.evaluator_agent = EvaluatorAgent(self.rag)
        self.hint_agent      = HintAgent(self.rag)
        self.progress_agent  = ProgressAgent()

        self._current: GeneratedSentence | None = None
        self._current_lang: str = "Spanish"

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_sentence(self, language: str, topic: str) -> tuple[GeneratedSentence, list[str]]:
        """
        Route to SentenceAgent. Returns the sentence and all collected logs.
        """
        self._current_lang = language
        logs = [f"[orchestrator] routing to sentence-agent (lang={language}, topic={topic}, level={self.state.level})"]

        result = self.sentence_agent.run(
            language=language,
            topic=topic,
            user_state=self.state,
        )
        logs += self.sentence_agent.logs

        # Register word in user state (for vocab tracking)
        self.state.record_word(result.foreign_word, result.english_meaning, language)
        logs.append(f"[orchestrator] word registered: '{result.foreign_word}' → '{result.english_meaning}'")

        self._current = result
        logs.append(f"[orchestrator] ready for user input")
        return result, logs

    def check_answer(self, guess: str) -> tuple[EvaluationResult, list[str]]:
        """
        Route to EvaluatorAgent, then update UserState. Returns result + logs.
        """
        if self._current is None:
            raise RuntimeError("No active sentence — call generate_sentence() first.")

        logs = [f"[orchestrator] routing to evaluator-agent"]

        result = self.evaluator_agent.run(
            language=self._current_lang,
            foreign_word=self._current.foreign_word,
            correct_meaning=self._current.english_meaning,
            guess=guess,
        )
        logs += self.evaluator_agent.logs

        self.state.record_answer(self._current.foreign_word, result.correct)
        logs.append(
            f"[orchestrator] state updated: streak={self.state.streak}, "
            f"mastered={self.state.mastered_count}"
        )

        return result, logs

    def get_hint(self) -> tuple[str, list[str]]:
        """Route to HintAgent. Returns hint text + logs."""
        if self._current is None:
            raise RuntimeError("No active sentence.")

        logs = ["[orchestrator] routing to hint-agent"]
        hint = self.hint_agent.run(
            language=self._current_lang,
            foreign_word=self._current.foreign_word,
            correct_meaning=self._current.english_meaning,
            sentence=self._current.sentence,
        )
        logs += self.hint_agent.logs
        return hint, logs

    def get_progress(self) -> tuple[dict, list[str]]:
        """Route to ProgressAgent. Returns analysis dict + logs."""
        logs = ["[orchestrator] routing to progress-agent"]
        analysis = self.progress_agent.run(self.state)
        logs += self.progress_agent.logs
        return analysis, logs

    # ── Convenience accessors ──────────────────────────────────────────────────

    @property
    def user_state(self) -> UserState:
        return self.state

    @property
    def current_sentence(self) -> GeneratedSentence | None:
        return self._current

    def reset_session(self) -> None:
        """Clear user state but keep the RAG dictionary."""
        self.state = UserState()
        self._current = None
