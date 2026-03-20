"""
Phase 5: Query rewriting and expansion.
All LLM calls go through LMStudioClient (localhost:1234 / mistralai/ministral-3b).
Prompts are loaded from core.prompt_registry (config/prompts.yaml) with fallback
to the hardcoded class constants if the registry is unavailable.
"""
from __future__ import annotations

from typing import List, Optional

from core.query.llm_client import LMStudioClient


def _get_prompt(key: str, fallback: str) -> str:
    """Load prompt from registry with fallback to hardcoded constant."""
    try:
        from core.prompt_registry import PromptRegistry
        return PromptRegistry.instance().get_prompt(key)
    except Exception:
        return fallback


class QueryRewriter:
    """
    Rewrites a raw user query into a retrieval-optimised form.
    - Expands acronyms
    - Removes conversational filler
    - Adds domain-relevant terms
    """

    SYSTEM_PROMPT = (
        "You are a search query optimiser. "
        "Rewrite the given query to maximise document retrieval quality. "
        "Expand acronyms, clarify ambiguities, and add relevant synonyms. "
        "Output ONLY the rewritten query — no explanation."
    )

    def __init__(self, client: LMStudioClient) -> None:
        self.client = client

    def rewrite(self, query: str) -> str:
        if not query.strip():
            return query
        system = _get_prompt("query_rewriter", self.SYSTEM_PROMPT)
        return self.client.complete(
            prompt=f"Original query: {query}\n\nRewritten query:",
            system=system,
            trace_method="Query Rewriting",
        )


class QueryExpander:
    """
    HyDE — Hypothetical Document Embeddings.
    Generates a hypothetical answer passage, embeds it, and uses that
    embedding for retrieval (often outperforms embedding the raw question).
    """

    SYSTEM_PROMPT = (
        "You are a document writing assistant. "
        "Write a short, factual passage (2-4 sentences) that would directly answer the question. "
        "Write the passage as if it came from an authoritative source. "
        "Output ONLY the passage — no preamble."
    )

    def __init__(self, client: LMStudioClient, embedder) -> None:
        self.client = client
        self.embedder = embedder

    def expand(self, query: str) -> List[float]:
        """Return embedding of a hypothetical answer document."""
        system = _get_prompt("hyde", self.SYSTEM_PROMPT)
        hypothesis = self.client.complete(
            prompt=f"Question: {query}\n\nAnswer passage:",
            system=system,
            trace_method="HyDE Expansion",
        )
        return self.embedder.embed_one(hypothesis)

    def generate_hypothesis(self, query: str) -> str:
        """Return the hypothetical answer text (for debugging / logging)."""
        system = _get_prompt("hyde", self.SYSTEM_PROMPT)
        return self.client.complete(
            prompt=f"Question: {query}\n\nAnswer passage:",
            system=system,
            trace_method="HyDE Expansion",
        )


class MultiQueryGenerator:
    """
    Generates N paraphrases of the query, retrieves for each, then
    union + RRF fuses the results for higher recall.
    """

    SYSTEM_PROMPT = (
        "Generate {n} distinct search query paraphrases for the question below. "
        "Each paraphrase should approach the topic from a different angle. "
        "Output ONLY the paraphrases, one per line, no numbering."
    )

    def __init__(self, client: LMStudioClient, n_queries: int = 3) -> None:
        self.client = client
        self.n_queries = n_queries

    def generate(self, query: str) -> List[str]:
        """Return a list of paraphrased queries (including the original)."""
        system_template = _get_prompt("multi_query", self.SYSTEM_PROMPT)
        prompt_text = system_template.format(n=self.n_queries)
        response = self.client.complete(
            prompt=f"Original question: {query}\n\nParaphrases:",
            system=prompt_text,
            trace_method="Multi-Query Generation",
        )
        paraphrases = [
            line.strip()
            for line in response.splitlines()
            if line.strip() and line.strip() != query
        ]
        # Always include original
        return [query] + paraphrases[: self.n_queries - 1]


class StepBackPrompter:
    """
    Abstracts the query to a broader/more general form for improved recall.
    'Step-back prompting' (Zheng et al., 2023).
    """

    SYSTEM_PROMPT = (
        "Given a specific question, generate a more general, abstract question "
        "that covers the same topic. This broader question should help retrieve "
        "relevant background information. Output ONLY the broader question."
    )

    def __init__(self, client: LMStudioClient) -> None:
        self.client = client

    def step_back(self, query: str) -> str:
        system = _get_prompt("step_back", self.SYSTEM_PROMPT)
        return self.client.complete(
            prompt=f"Specific question: {query}\n\nBroader question:",
            system=system,
            trace_method="Step-Back Prompting",
        )
