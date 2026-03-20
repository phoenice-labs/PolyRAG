"""
TripleHybridRetriever — 3-way hybrid search + optional SPLADE (4-way).

Fuses independent ranked lists via Reciprocal Rank Fusion (RRF, k=60):
  1. Dense vector search        (semantic similarity)
  2. BM25 keyword search        (exact / lexical match)
  3. SPLADE sparse neural       (learned term expansion — optional)
  4. Knowledge Graph traversal  (relational / entity-based)

The result is a single ranked list where a document is boosted if it
appears in multiple signal sources — reducing both false positives and
hallucinations.

Formula (Cormack et al.):
  rrf_score(d) = Σ  w_i / (k + rank_i(d))
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from core.graph.traversal import GraphTraverser
from core.retrieval.hybrid import HybridRetriever, MetadataFilter
from core.store.models import SearchResult

logger = logging.getLogger(__name__)

_RRF_K = 60


class TripleHybridRetriever:
    """
    Extends HybridRetriever with a third Knowledge Graph signal.

    Parameters
    ----------
    hybrid_retriever : the existing 2-way (vector + BM25) HybridRetriever
    traverser        : GraphTraverser for KG-based candidates
    graph_weight     : RRF weight for graph signal (default: 1.0; same as vector and BM25)
    graph_top_k_mult : multiplier × top_k for graph candidate pool
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        traverser: GraphTraverser,
        graph_weight: float = 1.0,
        graph_top_k_mult: int = 3,
    ) -> None:
        self.hybrid = hybrid_retriever
        self.traverser = traverser
        self.graph_weight = graph_weight
        self.graph_top_k_mult = graph_top_k_mult

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        embedder=None,
        enable_dense: bool = True,
        enable_bm25: bool = True,
        enable_splade: bool = True,
        enable_graph: bool = True,
        enable_llm_graph: bool = False,
    ) -> Tuple[List[SearchResult], list]:
        """
        Up to 4-way retrieval — Hybrid (Dense+BM25+SPLADE) and Graph are
        independent, so they run in parallel via ThreadPoolExecutor.

        Returns
        -------
        results    : top-k SearchResult list after multi-way RRF
        graph_paths: list of GraphPath objects (for explainability in RAGResponse)
        """
        import concurrent.futures

        recall_k = top_k * 5   # broad recall pool
        run_hybrid = enable_dense or enable_bm25 or enable_splade

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futs = {}
            if run_hybrid:
                futs["hybrid"] = executor.submit(
                    self.hybrid.retrieve,
                    query=query, collection=collection, top_k=recall_k,
                    filters=filters, embedder=embedder,
                    enable_dense=enable_dense, enable_bm25=enable_bm25,
                    enable_splade=enable_splade,
                )
            if enable_graph:
                futs["graph"] = executor.submit(
                    self.traverser.traverse,
                    query=query, top_k=top_k * self.graph_top_k_mult,
                    use_llm_graph=enable_llm_graph,
                )
            hybrid_results = futs["hybrid"].result() if "hybrid" in futs else []
            graph_result   = futs["graph"].result()  if "graph"  in futs else ([], [])

        graph_results, graph_paths = graph_result if isinstance(graph_result, tuple) else (graph_result, [])

        # Graph-only mode: return graph results tagged with Knowledge Graph lineage.
        # Normalize r.score to the RRF scale so it is comparable to fused results
        # and does not mislead the confidence display (hop scores are 1.0/0.6/0.3).
        if not run_hybrid and enable_graph:
            for i, r in enumerate(graph_results[:top_k], start=1):
                rrf_score = self.graph_weight / (_RRF_K + i)
                r.score = rrf_score  # replace hop-distance score with RRF-normalised value
                lin = list(r.document.metadata.get("_method_lineage", []))
                if not any(e.get("method") == "Knowledge Graph" for e in lin):
                    lin.append({
                        "method": "Knowledge Graph",
                        "rank": i,
                        "rrf_contribution": round(rrf_score, 6),
                    })
                r.document.metadata["_method_lineage"] = lin
                r.rank = i
            return graph_results[:top_k], graph_paths

        # Hybrid-only mode (graph disabled or returned nothing)
        if not graph_results:
            return hybrid_results[:top_k], []

        logger.debug(
            "TripleHybridRetriever: hybrid=%d, graph=%d candidates",
            len(hybrid_results), len(graph_results),
        )

        # ── 3-way RRF fusion ──────────────────────────────────────────────────
        fused = self._rrf_fuse(hybrid_results, graph_results)

        # ── Apply metadata filters post-fusion ────────────────────────────────
        if filters:
            fused = MetadataFilter.apply(fused, filters)

        # Re-rank by final RRF score
        fused_sorted = sorted(fused, key=lambda r: r.score, reverse=True)[:top_k]
        for i, r in enumerate(fused_sorted, start=1):
            r.rank = i

        return fused_sorted, graph_paths


    # ── Private ───────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        hybrid_results: List[SearchResult],
        graph_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Combine hybrid (already RRF-fused vector+BM25) and graph signal.
        Extends the _method_lineage already set by HybridFuser with a Graph entry.
        """
        k = _RRF_K

        hybrid_ranks: Dict[str, int] = {
            r.document.id: i + 1 for i, r in enumerate(hybrid_results)
        }
        graph_ranks: Dict[str, int] = {
            r.document.id: i + 1 for i, r in enumerate(graph_results)
        }
        # Prefer hybrid documents (full text + lineage) over graph stubs for overlapping ids
        all_docs = {r.document.id: r.document for r in graph_results}
        all_docs.update({r.document.id: r.document for r in hybrid_results})

        all_ids = set(hybrid_ranks.keys()) | set(graph_ranks.keys())

        rrf_scores: Dict[str, float] = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in hybrid_ranks:
                score += 1.0 / (k + hybrid_ranks[doc_id])
            if doc_id in graph_ranks:
                score += self.graph_weight / (k + graph_ranks[doc_id])
            rrf_scores[doc_id] = score

        results: List[SearchResult] = []
        for doc_id, score in rrf_scores.items():
            doc = all_docs[doc_id]
            # Extend lineage already written by HybridFuser (Dense+BM25)
            lineage = list(doc.metadata.get("_method_lineage", []))
            if doc_id in graph_ranks:
                graph_rank = graph_ranks[doc_id]
                graph_contrib = self.graph_weight / (k + graph_rank)
                if not any(e.get("method") == "Knowledge Graph" for e in lineage):
                    lineage.append({
                        "method": "Knowledge Graph",
                        "rank": graph_rank,
                        "rrf_contribution": round(graph_contrib, 6),
                    })
            # Annotate retrieval_signals (existing behaviour)
            signals = []
            if doc_id in hybrid_ranks:
                signals.append("vector+bm25")
            if doc_id in graph_ranks:
                signals.append("graph")
            doc.metadata["retrieval_signals"] = ",".join(signals)
            doc.metadata["_method_lineage"] = lineage
            results.append(SearchResult(document=doc, score=score, rank=0))

        return results
