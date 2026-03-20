"""
Phase 4: Multi-stage retrieval — broad recall → parent expansion → re-ranking.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

from core.store.models import Document, SearchResult

# Module-level cache: model_name → CrossEncoder instance (loaded once, reused forever)
_CE_MODEL_CACHE: Dict[str, object] = {}
_CE_MODEL_LOCK = threading.Lock()


def _get_cross_encoder(model_name: str):
    """Load CrossEncoder once per process; subsequent calls return cached instance."""
    if model_name in _CE_MODEL_CACHE:
        return _CE_MODEL_CACHE[model_name]
    with _CE_MODEL_LOCK:
        if model_name not in _CE_MODEL_CACHE:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError as e:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e
            _CE_MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _CE_MODEL_CACHE[model_name]


class CrossEncoderReRanker:
    """
    Cross-encoder re-ranker using sentence-transformers cross-encoder models.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (open-source, ~70 MB)

    The underlying CrossEncoder model is loaded once per process and cached —
    subsequent instantiations of CrossEncoderReRanker reuse the same weights.

    Requires: pip install sentence-transformers  (already in requirements.txt)
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        self._model = _get_cross_encoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Re-rank candidates using query-document cross-attention.

        Returns
        -------
        Sorted SearchResult list with updated scores and ranks.
        """
        if not candidates:
            return []
        self._load()
        pairs = [(query, r.document.text) for r in candidates]
        raw_scores = self._model.predict(pairs)

        # Normalise to [0,1] via sigmoid
        import math
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        scored = sorted(
            zip(raw_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        top = scored[:top_k] if top_k else scored
        results = []
        for rank, (score, result) in enumerate(top, start=1):
            results.append(SearchResult(
                document=result.document,
                score=sigmoid(float(score)),
                rank=rank,
            ))
        return results


class ParentExpander:
    """
    Fetches parent chunks for child-chunk hits to provide broader context.
    Requires a ChunkRegistry populated during ingestion.
    """

    def __init__(self, registry) -> None:
        self.registry = registry  # ChunkRegistry

    def expand(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Replace child chunk results with their parent chunk where available.
        Preserves original score and rank; deduplicates by parent_id.
        """
        seen_ids: set = set()
        expanded: List[SearchResult] = []

        for result in results:
            doc_id = result.document.id
            chunk = self.registry.get(doc_id)
            if chunk and chunk.parent_id:
                parent = self.registry.get(chunk.parent_id)
                if parent:
                    if parent.chunk_id not in seen_ids:
                        parent_doc = Document(
                            id=parent.chunk_id,
                            text=parent.text,
                            embedding=result.document.embedding,
                            metadata={
                                **result.document.metadata,
                                "expanded_from": doc_id,
                                "section_title": parent.section_title or "",
                            },
                        )
                        expanded.append(SearchResult(
                            document=parent_doc,
                            score=result.score,
                            rank=result.rank,
                        ))
                        seen_ids.add(parent.chunk_id)
                    continue  # always skip child, whether parent was new or already seen
            # No parent — use as-is
            if doc_id not in seen_ids:
                expanded.append(result)
                seen_ids.add(doc_id)

        for i, r in enumerate(expanded, start=1):
            r.rank = i
        return expanded


class CrossDocumentAggregator:
    """
    Groups results by source document, removes near-duplicate passages,
    and returns a de-duplicated, cross-document view.
    """

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self.similarity_threshold = similarity_threshold

    def aggregate(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove near-duplicate results (same doc_id + overlapping text hash).
        Returns re-ranked de-duplicated list.
        """
        seen_hashes: set = set()
        unique: List[SearchResult] = []

        for result in results:
            # Use first 64 chars as a lightweight fingerprint
            fingerprint = result.document.text[:64].strip().lower()
            if fingerprint not in seen_hashes:
                seen_hashes.add(fingerprint)
                unique.append(result)

        for i, r in enumerate(unique, start=1):
            r.rank = i
        return unique


class MultiStageRetriever:
    """
    Full multi-stage retrieval pipeline:
      1. Hybrid broad recall (vector + BM25)
      2. Parent context expansion
      3. Cross-encoder re-ranking
      4. Relevance threshold filtering
      5. Cross-document de-duplication

    Parameters
    ----------
    hybrid_retriever   : HybridRetriever (Phase 3)
    reranker           : CrossEncoderReRanker
    parent_expander    : ParentExpander (optional)
    aggregator         : CrossDocumentAggregator
    relevance_threshold: minimum reranker score to keep (default: 0.0)
    recall_multiplier  : broad_k = top_k × multiplier (default: 5)
    """

    def __init__(
        self,
        hybrid_retriever,
        reranker: CrossEncoderReRanker,
        parent_expander: Optional[ParentExpander] = None,
        aggregator: Optional[CrossDocumentAggregator] = None,
        relevance_threshold: float = 0.0,
        recall_multiplier: int = 5,
    ) -> None:
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.expander = parent_expander
        self.aggregator = aggregator or CrossDocumentAggregator()
        self.relevance_threshold = relevance_threshold
        self.recall_multiplier = recall_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        # Stage 1: broad hybrid recall
        broad_k = top_k * self.recall_multiplier
        candidates = self.hybrid.search(query, top_k=broad_k, filters=filters)

        # Stage 2: parent expansion
        if self.expander:
            candidates = self.expander.expand(candidates)

        # Stage 3: cross-encoder re-rank
        reranked = self.reranker.rerank(query, candidates, top_k=broad_k)

        # Stage 4: relevance threshold filter
        filtered = [r for r in reranked if r.score >= self.relevance_threshold]

        # Stage 5: cross-document de-dup
        final = self.aggregator.aggregate(filtered)

        return final[:top_k]
