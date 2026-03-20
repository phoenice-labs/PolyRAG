"""
Hybrid Search Fusion — combines dense-vector, BM25, and SPLADE results via RRF.

Reciprocal Rank Fusion (RRF):
    score(d) = Σ w_i / (k + rank_i(d))
    k=60 (Cormack et al., 2009 default)

Signal weights (configurable, defaults):
    Dense Vector  : 1.0  — semantic similarity
    BM25 Keyword  : 0.8  — exact lexical match (slightly down-weighted when SPLADE present)
    SPLADE Sparse : 1.0  — learned sparse expansion (supersedes most of BM25)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from core.store.models import Document, SearchResult


class HybridFuser:
    """
    Fuses ranked lists from Dense Vector, BM25, and optionally SPLADE via RRF.

    Parameters
    ----------
    k        : RRF constant (default: 60)
    vector_w : weight for dense vector results (default: 1.0)
    bm25_w   : weight for BM25 results (default: 1.0; recommend 0.8 when SPLADE enabled)
    splade_w : weight for SPLADE sparse neural results (default: 1.0)
    """

    def __init__(
        self,
        k: int = 60,
        vector_w: float = 1.0,
        bm25_w: float = 1.0,
        splade_w: float = 1.0,
    ) -> None:
        self.k = k
        self.vector_w = vector_w
        self.bm25_w = bm25_w
        self.splade_w = splade_w

    def fuse(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int = 5,
        splade_results: Optional[List[SearchResult]] = None,
    ) -> List[SearchResult]:
        """
        Combine vector, BM25, and SPLADE result lists via RRF.
        Tracks per-method contributions in document.metadata["_method_lineage"].

        Returns
        -------
        Re-ranked list of SearchResult (score = RRF score), top_k length.
        """
        scores: Dict[str, float] = defaultdict(float)
        doc_lookup: Dict[str, Document] = {}
        # Per-doc lineage: method → {rank, rrf_contribution}
        lineage: Dict[str, list] = defaultdict(list)

        for result in vector_results:
            doc_id = result.document.id
            contrib = self.vector_w / (self.k + result.rank)
            scores[doc_id] += contrib
            lineage[doc_id].append({
                "method": "Dense Vector",
                "rank": result.rank,
                "rrf_contribution": round(contrib, 6),
            })
            doc_lookup[doc_id] = result.document

        for result in bm25_results:
            doc_id = result.document.id
            contrib = self.bm25_w / (self.k + result.rank)
            scores[doc_id] += contrib
            lineage[doc_id].append({
                "method": "BM25 Keyword",
                "rank": result.rank,
                "rrf_contribution": round(contrib, 6),
            })
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = result.document

        for result in (splade_results or []):
            doc_id = result.document.id
            contrib = self.splade_w / (self.k + result.rank)
            scores[doc_id] += contrib
            lineage[doc_id].append({
                "method": "SPLADE Sparse Neural",
                "rank": result.rank,
                "rrf_contribution": round(contrib, 6),
            })
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = result.document

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused: List[SearchResult] = []
        for rank, doc_id in enumerate(sorted_ids[:top_k], start=1):
            doc = doc_lookup[doc_id]
            doc.metadata["_method_lineage"] = lineage[doc_id]
            fused.append(
                SearchResult(
                    document=doc,
                    score=scores[doc_id],
                    rank=rank,
                )
            )
        return fused


class MetadataFilter:
    """
    Simple AND-logic metadata filter applied post-retrieval.
    Normalises filter dicts to each adapter's query dialect via FilterDialect.
    """

    @staticmethod
    def apply(results: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Filter results keeping only those whose metadata matches ALL filters."""
        if not filters:
            return results
        kept = []
        for r in results:
            if all(r.document.metadata.get(k) == v for k, v in filters.items()):
                kept.append(r)
        # Re-number ranks
        for i, r in enumerate(kept, start=1):
            r.rank = i
        return kept


class HybridRetriever:
    """
    Orchestrates hybrid search across a vector store adapter + BM25 index
    and (optionally) SPLADE sparse neural index.

    Usage
    -----
        retriever = HybridRetriever(store, bm25_index, embedder, collection,
                                    splade_index=splade_index)
        results   = retriever.search(query, top_k=10, filters={"source": "gutenberg"})
    """

    def __init__(
        self,
        store,                               # VectorStoreBase
        bm25_index: "BM25Index",
        embedder,                            # EmbeddingProviderBase
        collection: str,
        splade_index=None,                   # Optional[SparseNeuralIndex]
        fuser: Optional[HybridFuser] = None,
        splade_w: float = 1.0,
        bm25_w_with_splade: float = 0.8,
    ) -> None:
        from core.retrieval.bm25 import BM25Index

        self.store = store
        self.bm25 = bm25_index
        self.splade = splade_index
        self.embedder = embedder
        self.collection = collection
        self._splade_w = splade_w
        self._bm25_w_with_splade = bm25_w_with_splade
        self.fuser = fuser or HybridFuser()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        vector_k_multiplier: int = 3,
        enable_dense: bool = True,
        enable_bm25: bool = True,
        enable_splade: bool = True,
    ) -> List[SearchResult]:
        """
        Run hybrid search and return fused results.
        Dense vector, BM25, and SPLADE are independent — run in parallel via
        ThreadPoolExecutor (up to 3 workers).

        Parameters
        ----------
        query              : Natural language query.
        top_k              : Final number of results to return.
        filters            : Optional metadata filter dict.
        vector_k_multiplier: Over-fetch vector results (top_k × multiplier) then re-rank.
        enable_dense       : Include dense vector retrieval.
        enable_bm25        : Include BM25 keyword retrieval.
        enable_splade      : Include SPLADE sparse neural retrieval (if index available).
        """
        import concurrent.futures

        broad_k = top_k * vector_k_multiplier
        vector_results: List[SearchResult] = []
        bm25_results: List[SearchResult] = []
        splade_results: List[SearchResult] = []

        run_splade = enable_splade and self.splade is not None and len(self.splade) > 0

        if enable_dense:
            query_vec = self.embedder.embed_one(query)

        # Down-weight BM25 when SPLADE is active (SPLADE subsumes most of BM25)
        bm25_w = self._bm25_w_with_splade if run_splade else self.fuser.bm25_w
        fuser = HybridFuser(
            k=self.fuser.k,
            vector_w=self.fuser.vector_w,
            bm25_w=bm25_w,
            splade_w=self._splade_w,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futs = {}
            if enable_dense:
                futs["vector"] = executor.submit(
                    self.store.query, self.collection, query_vec, broad_k, filters
                )
            if enable_bm25:
                futs["bm25"] = executor.submit(self.bm25.search, query, broad_k, filters)
            if run_splade:
                futs["splade"] = executor.submit(self.splade.search, query, broad_k, filters)
            vector_results = futs["vector"].result() if "vector" in futs else []
            bm25_results   = futs["bm25"].result()   if "bm25"   in futs else []
            splade_results = futs["splade"].result() if "splade" in futs else []

        # Fuse all available signals
        fused = fuser.fuse(
            vector_results, bm25_results, top_k=top_k,
            splade_results=splade_results if run_splade else None,
        )

        # Post-retrieval metadata filter (catches what adapter filter missed)
        if filters:
            fused = MetadataFilter.apply(fused, filters)

        return fused[:top_k]

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
    ) -> List[SearchResult]:
        """
        Alias for search() that accepts the TripleHybridRetriever calling convention.
        The `collection` and `embedder` params are ignored — they are already
        bound in this retriever's constructor.
        """
        return self.search(query=query, top_k=top_k, filters=filters,
                           enable_dense=enable_dense, enable_bm25=enable_bm25,
                           enable_splade=enable_splade)
