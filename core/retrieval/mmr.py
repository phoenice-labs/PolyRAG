"""
Phase 11: Maximal Marginal Relevance (MMR) — diversity-aware re-ranking.

Prevents the top-K results from being near-duplicate paraphrases of the
same passage. Balances relevance (similarity to query) against novelty
(dissimilarity from already-selected results).

Formula (Carbonell & Goldstein, 1998):
  MMR(d) = λ · relevance(d, query) - (1-λ) · max_sim(d, selected)

λ = 1.0 → pure relevance (identical to original top-K)
λ = 0.0 → pure diversity (maximum novelty, ignores relevance)
λ = 0.7 → recommended: slightly diversity-boosted ranking

Operates on document embeddings — no LLM required.
Falls back to original order if embeddings are missing.

Config (config.yaml):
  advanced_retrieval:
    mmr:
      enabled: true
      diversity_weight: 0.3   # (1 - λ). Higher = more diversity.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from core.store.models import SearchResult

logger = logging.getLogger(__name__)


class MMRReranker:
    """
    Diversity-aware re-ranker using Maximal Marginal Relevance.

    Parameters
    ----------
    diversity_weight : float in [0, 1]
        Controls diversity vs relevance trade-off.
        0.0 = pure relevance, 1.0 = pure diversity.
        Recommended: 0.3 (slight diversity boost preserving relevance).
    """

    def __init__(self, diversity_weight: float = 0.3) -> None:
        self.lmbda = 1.0 - diversity_weight   # λ in the MMR formula

    def rerank(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Apply MMR greedy selection.

        Returns up to top_k results, diversified.
        Falls back to original order if embeddings are absent.
        """
        if len(results) <= 1:
            return results[:top_k]

        # Build embedding matrix
        emb_list = [r.document.embedding for r in results]
        has_emb  = [bool(e) for e in emb_list]

        if not any(has_emb):
            logger.debug("MMR: no embeddings — returning original top_k")
            return results[:top_k]

        dim = len(next(e for e in emb_list if e))
        emb_matrix = np.array(
            [np.array(e, dtype=np.float32) if e else np.zeros(dim) for e in emb_list],
            dtype=np.float32,
        )

        # Normalise for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = emb_matrix / norms

        # Relevance scores (normalised to [0, 1])
        relevance = np.array([float(r.score) for r in results], dtype=np.float32)
        r_max = relevance.max()
        if r_max > 0:
            relevance /= r_max

        selected: List[int] = []
        remaining = list(range(len(results)))

        while len(selected) < min(top_k, len(results)) and remaining:
            if not selected:
                # First pick: most relevant document
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # MMR score for each remaining document
                sel_embs = normed[selected]                     # (k, dim)
                rem_embs = normed[remaining]                    # (r, dim)
                sim_to_sel = (rem_embs @ sel_embs.T).max(axis=1)  # (r,)

                mmr_scores = (
                    self.lmbda * relevance[np.array(remaining)]
                    - (1.0 - self.lmbda) * sim_to_sel
                )
                best = remaining[int(np.argmax(mmr_scores))]

            selected.append(best)
            remaining.remove(best)

        # Build re-ranked list
        reranked = []
        for rank, idx in enumerate(selected, start=1):
            r = results[idx]
            r.rank = rank
            reranked.append(r)

        return reranked
