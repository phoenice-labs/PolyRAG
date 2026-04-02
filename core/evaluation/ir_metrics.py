"""
IR retrieval metrics — MRR, Precision@k, Recall@k, NDCG@k.

These metrics evaluate *retrieval quality* independently of answer quality,
requiring only expected_sources ground truth (no LLM needed).

Design:
- Zero external dependencies (pure Python + math)
- Follows the same singleton + dataclass pattern as ragas_scorer.py
- Source matching: a retrieved chunk is "relevant" if any string in
  expected_sources appears (case-insensitive) in:
    chunk.metadata.get("source", "") OR chunk.text (first 200 chars)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IRMetricsResult:
    """Retrieval IR metrics for a single question evaluated at top-k."""

    k: int = 5
    mrr: Optional[float] = None           # Mean Reciprocal Rank
    precision_at_k: Optional[float] = None  # P@k
    recall_at_k: Optional[float] = None    # R@k
    ndcg_at_k: Optional[float] = None     # NDCG@k (binary relevance)
    relevant_count: int = 0               # number of relevant chunks in top-k
    total_expected: int = 0               # total expected sources provided
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "mrr": self.mrr,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "relevant_count": self.relevant_count,
            "total_expected": self.total_expected,
            "error": self.error,
        }


def _is_relevant(chunk: Dict[str, Any], expected_sources: List[str]) -> bool:
    """
    Return True if chunk is considered relevant given expected_sources.

    Matching is case-insensitive substring match against:
    - chunk["metadata"]["source"] (primary)
    - chunk["text"] first 200 chars (fallback for inline source mentions)
    """
    if not expected_sources:
        return False
    source = (chunk.get("metadata") or {}).get("source", "")
    text_snippet = (chunk.get("text") or "")[:200]
    haystack = (source + " " + text_snippet).lower()
    return any(exp.lower() in haystack for exp in expected_sources if exp)


class IRMetricsScorer:
    """
    Compute standard IR retrieval metrics for RAG evaluation.

    Usage::

        scorer = IRMetricsScorer()
        result = scorer.score(chunks, expected_sources=["policy.pdf"], k=5)
        print(result.mrr, result.ndcg_at_k)
    """

    def score(
        self,
        chunks: List[Dict[str, Any]],
        expected_sources: List[str],
        k: int = 5,
    ) -> IRMetricsResult:
        """
        Score a ranked list of retrieved chunks against ground-truth sources.

        Parameters
        ----------
        chunks:
            Ranked list of chunk dicts (index 0 = top result).
            Each dict must have at least ``"text"`` and optionally
            ``"metadata": {"source": ...}``.
        expected_sources:
            List of source identifiers that should appear in results.
            An empty list causes all metrics to return None.
        k:
            Cut-off rank.  Only the first *k* chunks are considered.
        """
        if not expected_sources:
            return IRMetricsResult(k=k, total_expected=0)

        top_k = chunks[:k]
        relevance = [_is_relevant(c, expected_sources) for c in top_k]

        relevant_count = sum(relevance)
        total_expected = len(expected_sources)

        # MRR — reciprocal rank of the FIRST relevant result
        mrr: Optional[float] = None
        for rank, rel in enumerate(relevance, start=1):
            if rel:
                mrr = round(1.0 / rank, 4)
                break
        if mrr is None:
            mrr = 0.0

        # Precision@k
        precision_at_k = round(relevant_count / k, 4) if k > 0 else 0.0

        # Recall@k — capped at 1.0 (can't recall more than 100%)
        recall_at_k = round(min(relevant_count / total_expected, 1.0), 4)

        # NDCG@k — binary relevance (1 if relevant, 0 if not)
        # DCG = sum( rel_i / log2(i+1) )  for i in 1..k
        # IDCG = DCG of ideal ranking (all relevant docs first)
        dcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(relevance, start=1)
        )
        ideal_hits = min(relevant_count, k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_at_k = round(dcg / idcg, 4) if idcg > 0 else 0.0

        return IRMetricsResult(
            k=k,
            mrr=mrr,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            relevant_count=relevant_count,
            total_expected=total_expected,
        )

    def score_batch(
        self,
        items: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[IRMetricsResult]:
        """
        Score a batch of evaluation items.

        Each item must have:
        - ``"chunks"``: ranked list of chunk dicts
        - ``"expected_sources"``: list of expected source strings
        """
        return [
            self.score(
                chunks=item.get("chunks", []),
                expected_sources=item.get("expected_sources", []),
                k=k,
            )
            for item in items
        ]


# Module-level singleton — same pattern as ragas_scorer.py
_ir_scorer: Optional[IRMetricsScorer] = None


def get_ir_scorer() -> IRMetricsScorer:
    """Return the module-level IRMetricsScorer singleton."""
    global _ir_scorer
    if _ir_scorer is None:
        _ir_scorer = IRMetricsScorer()
    return _ir_scorer
