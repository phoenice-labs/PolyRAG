"""
Phase 5: Conversation context tracking and context-aware query building.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from core.query.llm_client import LMStudioClient


@dataclass
class Turn:
    """A single conversation turn."""

    query: str
    answer: str = ""
    metadata: dict = field(default_factory=dict)


class ConversationContextTracker:
    """
    Maintains a sliding window of conversation turns.
    Used to resolve pronoun references and topic continuity across turns.

    Parameters
    ----------
    max_turns : maximum turns to keep in context window (default: 5)
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._history: Deque[Turn] = deque(maxlen=max_turns)

    def add_turn(self, query: str, answer: str = "", metadata: dict | None = None) -> None:
        self._history.append(Turn(query=query, answer=answer, metadata=metadata or {}))

    def get_history(self) -> List[Turn]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def as_text(self) -> str:
        """Format history as a readable string for LLM context injection."""
        lines = []
        for i, turn in enumerate(self._history, start=1):
            lines.append(f"[Turn {i}] User: {turn.query}")
            if turn.answer:
                lines.append(f"[Turn {i}] Assistant: {turn.answer[:200]}")
        return "\n".join(lines)


class ContextualQueryBuilder:
    """
    Uses conversation history + LLM to produce a fully self-contained query.
    Resolves pronouns, fills in implied context from prior turns.
    """

    SYSTEM_PROMPT = (
        "You are a query reformulation assistant. "
        "Given a conversation history and a follow-up question, "
        "rewrite the follow-up question to be fully self-contained "
        "(resolve all pronouns and references). "
        "Output ONLY the rewritten question."
    )

    def __init__(self, client: LMStudioClient) -> None:
        self.client = client

    def build(self, query: str, history: List[Turn]) -> str:
        if not history:
            return query

        history_text = "\n".join(
            f"Q: {t.query}\nA: {t.answer[:150]}" for t in history[-3:]
        )
        prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {query}\n\n"
            f"Self-contained question:"
        )
        return self.client.complete(prompt=prompt, system=self.SYSTEM_PROMPT)


class QueryIntelligencePipeline:
    """
    Full query intelligence pipeline:
      1. Context resolution (conversation history → self-contained query)
      2. Query rewriting (acronyms, filler, synonyms)
      3. Step-back abstraction (broader recall)
      4. Multi-query generation (paraphrases for higher recall)
      5. HyDE expansion (hypothetical document embedding)

    Each stage is individually togglable.
    """

    def __init__(
        self,
        client: LMStudioClient,
        embedder,
        context_tracker: Optional[ConversationContextTracker] = None,
        enable_rewrite: bool = True,
        enable_stepback: bool = True,
        enable_multi_query: bool = True,
        enable_hyde: bool = True,
        n_paraphrases: int = 3,
    ) -> None:
        from core.query.rewriter import (
            MultiQueryGenerator,
            QueryExpander,
            QueryRewriter,
            StepBackPrompter,
        )

        self.client = client
        self.embedder = embedder
        self.context_tracker = context_tracker or ConversationContextTracker()
        self.rewriter = QueryRewriter(client) if enable_rewrite else None
        self.stepback = StepBackPrompter(client) if enable_stepback else None
        self.multi_query = MultiQueryGenerator(client, n_paraphrases) if enable_multi_query else None
        self.hyde = QueryExpander(client, embedder) if enable_hyde else None

    def process(
        self,
        raw_query: str,
        update_history: bool = True,
    ) -> "QueryBundle":
        """
        Transform a raw query into a QueryBundle with all enriched forms.

        Parameters
        ----------
        raw_query      : The original user query.
        update_history : Whether to add this turn to the context tracker.

        Returns
        -------
        QueryBundle with rewritten, expanded, and multi-query variants.
        """
        # Step 1: Context-aware reformulation
        history = self.context_tracker.get_history()
        builder = ContextualQueryBuilder(self.client)
        contextualized = builder.build(raw_query, history) if history else raw_query

        # Step 2: Rewrite
        rewritten = self.rewriter.rewrite(contextualized) if self.rewriter else contextualized

        # Step 3: Step-back
        stepback_q = self.stepback.step_back(rewritten) if self.stepback else None

        # Step 4: Multi-query
        paraphrases = self.multi_query.generate(rewritten) if self.multi_query else [rewritten]

        # Step 5: HyDE embedding
        hyde_embedding = self.hyde.expand(rewritten) if self.hyde else None

        if update_history:
            self.context_tracker.add_turn(raw_query)

        return QueryBundle(
            raw=raw_query,
            contextualized=contextualized,
            rewritten=rewritten,
            stepback=stepback_q,
            paraphrases=paraphrases,
            hyde_embedding=hyde_embedding,
        )


@dataclass
class QueryBundle:
    """All enriched forms of a user query produced by QueryIntelligencePipeline."""

    raw: str
    contextualized: str
    rewritten: str
    stepback: Optional[str]
    paraphrases: List[str]
    hyde_embedding: Optional[List[float]]

    @property
    def primary_query(self) -> str:
        """The best single query to use for retrieval."""
        return self.rewritten or self.contextualized or self.raw
