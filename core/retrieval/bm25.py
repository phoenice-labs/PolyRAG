"""
Phase 3: Hybrid Search — BM25 (keyword) + dense vector + metadata filters + RRF fusion.

BM25 is always available via rank-bm25 (pure Python, no server needed).
Adapters that support native BM25 (Qdrant/Weaviate/Milvus/PGVector) use it natively.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from core.store.models import Document, SearchResult


class BM25Index:
    """
    Pure-Python BM25 index built on top of rank-bm25.
    Used as fallback for adapters without native keyword search (ChromaDB, FAISS).

    Requires: pip install rank-bm25
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: List[Document] = []
        self._bm25 = None

    def add(self, documents: List[Document]) -> None:
        self._docs.extend(documents)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("Install rank-bm25: pip install rank-bm25") from e

        corpus = [doc.text.lower().split() for doc in self._docs]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def clear(self) -> None:
        self._docs = []
        self._bm25 = None

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """
        BM25 keyword search with optional post-rank metadata filtering.

        Parameters
        ----------
        filters : AND-logic exact-match filter dict, e.g. {"source": "gutenberg"}.
                  Applied after BM25 scoring so filtered-out docs don't pollute
                  the fused RRF ranking in HybridRetriever.
        """
        if not self._bm25 or not self._docs:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        # Over-fetch so filtering doesn't shrink results below top_k
        over_k = top_k * 3 if filters else top_k
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:over_k]
        results = []
        for rank, (idx, score) in enumerate(scored, start=1):
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    document=self._docs[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        if filters:
            from core.retrieval.hybrid import MetadataFilter
            results = MetadataFilter.apply(results, filters)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._docs)
