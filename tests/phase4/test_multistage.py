"""
Phase 4 – Multi-stage Retrieval Tests
======================================
Run:  pytest tests/phase4/ -v
"""
from __future__ import annotations

import random
from typing import List

import pytest

from core.store.models import Document, SearchResult
from core.retrieval.multistage import (
    CrossEncoderReRanker,
    CrossDocumentAggregator,
    ParentExpander,
)


def _vec(dim=8):
    v = [random.random() for _ in range(dim)]
    n = sum(x**2 for x in v)**0.5
    return [x/n for x in v]


def _result(i, text, score, rank, **meta):
    return SearchResult(
        document=Document(id=f"doc_{i}", text=text, embedding=_vec(), metadata=meta),
        score=score, rank=rank,
    )


class TestCrossEncoderReRanker:
    @pytest.fixture(scope="class")
    def reranker(self):
        return CrossEncoderReRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_rerank_returns_correct_count(self, reranker):
        candidates = [
            _result(i, f"passage {i} about various topics", 0.5, i+1)
            for i in range(5)
        ]
        reranked = reranker.rerank("What is the meaning of life?", candidates, top_k=3)
        assert len(reranked) == 3

    def test_rerank_scores_are_normalised(self, reranker):
        candidates = [_result(i, f"text {i}", 0.5, i+1) for i in range(3)]
        reranked = reranker.rerank("life", candidates)
        for r in reranked:
            assert 0.0 <= r.score <= 1.0

    def test_rerank_relevant_passage_ranks_higher(self, reranker):
        """
        The passage directly about 'to be or not to be' should rank above
        a passage about cooking.
        """
        candidates = [
            _result(0, "To be or not to be, that is the question. Hamlet.", 0.5, 1),
            _result(1, "The recipe requires flour, butter, sugar and eggs.", 0.4, 2),
            _result(2, "Othello is jealous of Cassio and plots revenge.", 0.3, 3),
        ]
        reranked = reranker.rerank("What does Hamlet say about existence?", candidates)
        assert reranked[0].document.id == "doc_0"

    def test_rerank_ranks_sequential(self, reranker):
        candidates = [_result(i, f"text {i}", 0.5, i+1) for i in range(4)]
        reranked = reranker.rerank("test query", candidates)
        assert [r.rank for r in reranked] == list(range(1, len(reranked)+1))

    def test_empty_input_returns_empty(self, reranker):
        assert reranker.rerank("query", []) == []


class TestParentExpander:
    def test_expands_child_to_parent(self):
        from core.chunking.models import Chunk, ChunkRegistry

        registry = ChunkRegistry()
        parent = Chunk(chunk_id="parent_1", doc_id="d", text="Full section text about Hamlet.")
        child = Chunk(chunk_id="child_1", doc_id="d", text="Hamlet questions existence.",
                      parent_id="parent_1")
        registry.register(parent)
        registry.register(child)

        expander = ParentExpander(registry)
        results = [_result(0, "Hamlet questions existence.", 0.9, 1)]
        # Manually set doc id to match child
        results[0].document.id = "child_1"

        expanded = expander.expand(results)
        assert len(expanded) == 1
        assert expanded[0].document.text == "Full section text about Hamlet."

    def test_deduplicates_same_parent(self):
        from core.chunking.models import Chunk, ChunkRegistry

        registry = ChunkRegistry()
        parent = Chunk(chunk_id="p1", doc_id="d", text="Parent text.")
        child_a = Chunk(chunk_id="c1", doc_id="d", text="Child A.", parent_id="p1")
        child_b = Chunk(chunk_id="c2", doc_id="d", text="Child B.", parent_id="p1")
        registry.register_many([parent, child_a, child_b])

        expander = ParentExpander(registry)
        results = [
            SearchResult(document=Document(id="c1", text="Child A.", embedding=_vec(),
                                           metadata={}), score=0.9, rank=1),
            SearchResult(document=Document(id="c2", text="Child B.", embedding=_vec(),
                                           metadata={}), score=0.8, rank=2),
        ]
        expanded = expander.expand(results)
        # Both map to same parent → only one result
        assert len(expanded) == 1

    def test_no_parent_passes_through(self):
        from core.chunking.models import ChunkRegistry
        registry = ChunkRegistry()  # empty
        expander = ParentExpander(registry)
        results = [_result(0, "orphan text", 0.9, 1)]
        expanded = expander.expand(results)
        assert len(expanded) == 1
        assert expanded[0].document.text == "orphan text"


class TestCrossDocumentAggregator:
    def test_removes_near_duplicates(self):
        agg = CrossDocumentAggregator()
        dup_text = "To be or not to be, that is the question."
        results = [
            _result(0, dup_text, 0.9, 1),
            _result(1, dup_text, 0.8, 2),    # exact duplicate
            _result(2, "Romeo loves Juliet forever.", 0.7, 3),
        ]
        unique = agg.aggregate(results)
        assert len(unique) == 2

    def test_preserves_unique_results(self):
        agg = CrossDocumentAggregator()
        results = [
            _result(i, f"Unique passage {i} about different topic.", 0.9-i*0.1, i+1)
            for i in range(5)
        ]
        unique = agg.aggregate(results)
        assert len(unique) == 5

    def test_reranks_after_dedup(self):
        agg = CrossDocumentAggregator()
        results = [
            _result(0, "duplicate text here", 0.9, 1),
            _result(1, "duplicate text here", 0.8, 2),
            _result(2, "different content entirely", 0.7, 3),
        ]
        unique = agg.aggregate(results)
        ranks = [r.rank for r in unique]
        assert ranks == list(range(1, len(unique)+1))


class TestMultiStageRetrieverIntegration:
    """End-to-end multi-stage retrieval on Shakespeare corpus."""

    @pytest.fixture(scope="class")
    def embedder(self):
        from core.embedding.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})

    def test_multistage_retrieval_on_shakespeare(self, shakespeare_clean, embedder):
        from core.store.adapters.chromadb_adapter import ChromaDBAdapter
        from core.retrieval.bm25 import BM25Index
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.multistage import (
            CrossEncoderReRanker, CrossDocumentAggregator,
            ParentExpander, MultiStageRetriever,
        )
        from core.chunking.models import ChunkRegistry
        from core.store.models import Document
        from core.ingestion.loader import naive_chunk_text

        store = ChromaDBAdapter({"mode": "memory"})
        store.connect()
        store.create_collection("ms_test", embedder.embedding_dim)

        sample = shakespeare_clean[:30_000]
        texts = naive_chunk_text(sample, chunk_size=400, overlap=60)
        docs = []
        for i, text in enumerate(texts[:80]):
            docs.append(Document(
                id=f"c_{i}", text=text,
                embedding=embedder.embed_one(text),
                metadata={"source": "shakespeare"},
            ))
        store.upsert("ms_test", docs)

        bm25 = BM25Index()
        bm25.add(docs)
        hybrid = HybridRetriever(store=store, bm25_index=bm25,
                                  embedder=embedder, collection="ms_test")
        reranker = CrossEncoderReRanker()
        registry = ChunkRegistry()
        retriever = MultiStageRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
            parent_expander=ParentExpander(registry),
            aggregator=CrossDocumentAggregator(),
            recall_multiplier=3,
        )

        results = retriever.retrieve("What does Hamlet say about existence and death?", top_k=5)
        assert len(results) >= 1
        assert results[0].score <= 1.0
        store.close()
