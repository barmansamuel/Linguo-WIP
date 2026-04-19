"""
agents/sentence_agent.py — Generates contextual sentences with an embedded foreign word

Responsibilities:
  1. Receive user level, language, topic, and vocabulary history
  2. Query the RAG dictionary for existing words to avoid repetition
  3. Prompt the LLM to produce a new sentence with one embedded foreign word
  4. Return a validated GeneratedSentence model
  5. Register the new word into the RAG dictionary
"""

from __future__ import annotations

from rag.dictionary import RAGDictionary, DictionaryEntry
from state.models import GeneratedSentence, UserState
from agents.base import BaseAgent


SENTENCE_PROMPT = """You are a language learning sentence generator inside a multi-agent system.

Task: Generate ONE English sentence about "{topic}" for a {level} learner of {language}, \
with exactly ONE key {language} word embedded inline.

Rules:
- Replace exactly one English word with its {language} equivalent.
- The {language} word should be inferable from context but not trivially obvious.
- Level guidance:
    beginner     → very common, concrete nouns or verbs (food, body, places)
    intermediate → everyday vocabulary, basic emotions, common verbs
    advanced     → richer, nuanced vocabulary; idioms acceptable
- Do NOT reuse any of these recently mastered words: {avoid_words}
- RAG context (already-known translations for reference): {rag_context}
- Return ONLY valid JSON with no markdown, no preamble.

JSON schema:
{{
  "sentence":        "<English sentence with one {language} word>",
  "foreign_word":    "<the {language} word used>",
  "english_meaning": "<its English meaning, one word or short phrase>",
  "part_of_speech":  "<noun|verb|adjective|adverb|phrase>",
  "example_context": "<one short sentence in {language} showing the word in use>",
  "difficulty":      "<easy|medium|hard>"
}}"""


class SentenceAgent(BaseAgent):
    name = "sentence-agent"

    def __init__(self, rag: RAGDictionary):
        super().__init__()
        self.rag = rag

    def run(
        self,
        language: str,
        topic: str,
        user_state: UserState,
    ) -> GeneratedSentence:
        self.clear_logs()
        level = user_state.level
        mastered = user_state.mastered_words[-10:]  # avoid last 10 mastered words

        # RAG: look up semantically related words already in the dictionary
        rag_results = self.rag.lookup(topic, language, top_k=5)
        rag_context = (
            ", ".join(f"{e.foreign_word}={e.english_meaning}" for e in rag_results)
            or "none yet"
        )

        self.log(f"level={level}, topic={topic}, lang={language}")
        self.log(f"RAG context ({len(rag_results)} entries): {rag_context}")
        self.log(f"Avoiding {len(mastered)} mastered words")

        prompt = SENTENCE_PROMPT.format(
            topic=topic,
            level=level,
            language=language,
            avoid_words=", ".join(mastered) or "none",
            rag_context=rag_context,
        )

        self.log("Calling LLM...")
        raw = self._call(prompt, max_tokens=600)
        data = self._parse_json(raw)
        result = GeneratedSentence(**data)

        # Register new word in the RAG dictionary
        self.rag.add_entry(DictionaryEntry(
            foreign_word=result.foreign_word,
            language=language,
            english_meaning=result.english_meaning,
            part_of_speech=result.part_of_speech,
            example_context=result.example_context,
        ))
        self.log(f"Added '{result.foreign_word}' to RAG dictionary (size={self.rag.size()})")

        return result
