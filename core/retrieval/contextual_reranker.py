"""
Phase 11: Contextual LLM Re-ranker (Contextual Fusion).

After retrieval, uses one LLM call to rank all candidate chunks by relevance
to the query, then blends the LLM rank with the original retrieval score.

Why one batched call (not N individual calls):
  - 1 call vs N calls → dramatically lower latency with local LLM
  - LLM can compare chunks against each other in context
  - Returns a ranking, not individual scores

Score fusion:
  final_score = llm_weight * llm_rank_score + (1 - llm_weight) * retrieval_score

Gracefully degrades to original order when LM Studio is offline.

Config (config.yaml):
  advanced_retrieval:
    contextual_reranker:
      enabled: true
      llm_weight: 0.4           # 0.0 = pure retrieval, 1.0 = pure LLM
      max_chunks_to_rank: 10    # cap for latency control
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from core.store.models import SearchResult

logger = logging.getLogger(__name__)

_RANK_PROMPT = """\
You are a search relevance expert. Rank these passages from MOST to LEAST relevant to the question.

Question: {query}

{passages}

Return ONLY a JSON object with passage numbers ranked best-first:
{{"ranking": [<most_relevant>, ..., <least_relevant>]}}
Include all {n} numbers exactly once. Example for 3 passages: {{"ranking": [2, 1, 3]}}
"""


class ContextualReranker:
    """
    Batched LLM re-ranker: one call ranks all top-K candidates.

    Parameters
    ----------
    llm_client          : LMStudioClient
    llm_weight          : weight of LLM ranking signal (0.0–1.0)
    max_chunks_to_rank  : max chunks sent to LLM per query (latency cap)
    max_chunk_chars     : max chars per chunk excerpt sent to LLM
    """

    def __init__(
        self,
        llm_client,
        llm_weight: float = 0.4,
        max_chunks_to_rank: int = 10,
        max_chunk_chars: int = 600,
    ) -> None:
        self._client = llm_client
        self.llm_weight = llm_weight
        self.retrieval_weight = 1.0 - llm_weight
        self.max_chunks_to_rank = max_chunks_to_rank
        self.max_chunk_chars = max_chunk_chars

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Re-rank results using a single batched LLM call.

        Falls back to original order if LLM unavailable or returns invalid output.
        """
        if not results:
            return results

        if not self._client.is_available():
            logger.debug("ContextualReranker: LM Studio offline — passthrough")
            return results

        to_rank = results[:self.max_chunks_to_rank]
        tail    = results[self.max_chunks_to_rank:]

        ranking = self._llm_rank(query, to_rank)   # 1-based indices, best-first

        # Build LLM rank score: rank 1 → score 1.0, rank N → score ~0
        n = len(to_rank)
        llm_scores = {idx: (n - 1 - rank) / max(n - 1, 1) for rank, idx in enumerate(ranking)}

        reranked: List[SearchResult] = []
        for i, r in enumerate(to_rank):
            orig_score = min(1.0, max(0.0, float(r.score)))
            llm_score  = llm_scores.get(i + 1, 0.5)          # fallback if idx missing
            fused = self.llm_weight * llm_score + self.retrieval_weight * orig_score
            r.document.metadata["llm_rank_score"] = round(llm_score, 3)
            r.document.metadata["fused_score"]    = round(fused, 3)
            reranked.append(SearchResult(document=r.document, score=fused, rank=0))

        all_results = sorted(reranked, key=lambda x: x.score, reverse=True) + tail
        for i, r in enumerate(all_results, start=1):
            r.rank = i
        return all_results

    # ── Private ───────────────────────────────────────────────────────────────

    def _llm_rank(self, query: str, results: List[SearchResult]) -> List[int]:
        """
        Ask LLM to rank passages. Returns a list of 1-based passage indices
        ordered best-first. Falls back to [1, 2, ..., N] on any failure.
        """
        n = len(results)
        fallback = list(range(1, n + 1))

        passages = "\n\n".join(
            f"Passage {i+1}:\n{r.document.text[:self.max_chunk_chars]}"
            for i, r in enumerate(results)
        )

        # Load prompt from registry (with fallback to hardcoded _RANK_PROMPT)
        try:
            from core.prompt_registry import PromptRegistry
            prompt_template = PromptRegistry.instance().get_prompt("contextual_reranker")
        except Exception:
            prompt_template = _RANK_PROMPT

        prompt = prompt_template.format(query=query, passages=passages, n=n)

        try:
            raw = self._client.complete(
                prompt,
                system="You are a search relevance expert. Return only JSON.",
                max_tokens=200,
                temperature=0.0,
                trace_method="Contextual Re-ranking",
            )
        except Exception as exc:
            logger.debug("ContextualReranker LLM call failed: %s", exc)
            return fallback

        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return fallback

        try:
            data = json.loads(match.group())
            ranking = [int(x) for x in data.get("ranking", [])]
            # Validate: must contain all indices 1..n
            if sorted(ranking) == fallback:
                return ranking
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

        return fallback
