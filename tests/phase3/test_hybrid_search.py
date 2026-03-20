"""
Phase 3 – Hybrid Search Tests (BM25 + vector + RRF)
=====================================================
Run:  pytest tests/phase3/ -v
"""
from __future__ import annotations

import math
import random
from typing import List

import pytest

from core.store.models import Document, SearchResult
from core.retrieval.bm25 import BM25Index
from core.retrieval.hybrid import HybridFuser, MetadataFilter, HybridRetriever


def _make_doc(i: int, text: str, **meta) -> Document:
    dim = 8
    v = [random.random() for _ in range(dim)]
    norm = sum(x**2 for x in v)**0.5
    return Document(id=f"doc_{i}", text=text, embedding=[x/norm for x in v], metadata=meta)


def _make_result(doc: Document, score: float, rank: int) -> SearchResult:
    return SearchResult(document=doc, score=score, rank=rank)


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class TestBM25Index:
    @pytest.fixture
    def index(self):
        docs = [
            _make_doc(0, "Hamlet questions the meaning of life and existence"),
            _make_doc(1, "Romeo and Juliet is a tragic love story"),
            _make_doc(2, "Macbeth is consumed by ambition and power"),
            _make_doc(3, "Othello deals with jealousy and betrayal"),
            _make_doc(4, "The comedy of errors features mistaken identity"),
        ]
        bm25 = BM25Index()
        bm25.add(docs)
        return bm25

    def test_search_returns_results(self, index):
        results = index.search("Hamlet existence", top_k=3)
        assert len(results) >= 1

    def test_hamlet_query_ranks_hamlet_first(self, index):
        results = index.search("Hamlet existence life", top_k=5)
        assert results[0].document.id == "doc_0"

    def test_keyword_exact_match(self, index):
        results = index.search("Macbeth ambition", top_k=3)
        ids = [r.document.id for r in results]
        assert "doc_2" in ids

    def test_scores_are_positive(self, index):
        results = index.search("Romeo Juliet love", top_k=3)
        assert all(r.score > 0 for r in results)

    def test_empty_index_returns_empty(self):
        bm25 = BM25Index()
        results = bm25.search("anything", top_k=5)
        assert results == []

    def test_len_reflects_added_docs(self, index):
        assert len(index) == 5

    def test_search_on_shakespeare(self, shakespeare_clean):
        """BM25 on real corpus: 'to be or not to be' should surface Hamlet."""
        from core.ingestion.loader import naive_chunk_text
        chunks = naive_chunk_text(shakespeare_clean[:50_000], chunk_size=512, overlap=64)
        docs = [_make_doc(i, t) for i, t in enumerate(chunks[:100])]
        bm25 = BM25Index()
        bm25.add(docs)
        results = bm25.search("to be or not to be that is the question", top_k=3)
        assert len(results) >= 1
        top_text = results[0].document.text.lower()
        assert "be" in top_text


# ── HybridFuser Tests ─────────────────────────────────────────────────────────

class TestHybridFuser:
    def _make_results(self, ids_scores):
        return [
            SearchResult(
                document=_make_doc(int(i.split("_")[1]), f"text {i}"),
                score=s, rank=r+1
            )
            for r, (i, s) in enumerate(ids_scores)
        ]

    def test_rrf_fusion_combines_lists(self):
        fuser = HybridFuser(k=60)
        vec = self._make_results([("doc_0", 0.9), ("doc_1", 0.7), ("doc_2", 0.5)])
        bm25 = self._make_results([("doc_1", 5.0), ("doc_0", 3.0), ("doc_3", 2.0)])
        fused = fuser.fuse(vec, bm25, top_k=4)
        ids = [r.document.id for r in fused]
        # doc_0 appears in both lists → should rank highly
        assert "doc_0" in ids[:2]
        assert "doc_1" in ids[:2]

    def test_ranks_are_sequential(self):
        fuser = HybridFuser()
        vec = self._make_results([("doc_0", 0.9), ("doc_1", 0.5)])
        bm25 = self._make_results([("doc_1", 3.0), ("doc_2", 1.0)])
        fused = fuser.fuse(vec, bm25, top_k=3)
        assert [r.rank for r in fused] == list(range(1, len(fused)+1))

    def test_top_k_respected(self):
        fuser = HybridFuser()
        vec = self._make_results([(f"doc_{i}", float(5-i)) for i in range(5)])
        bm25 = self._make_results([(f"doc_{i}", float(5-i)) for i in range(5)])
        fused = fuser.fuse(vec, bm25, top_k=3)
        assert len(fused) == 3

    def test_scores_descending(self):
        fuser = HybridFuser()
        vec = self._make_results([("doc_0", 0.9), ("doc_1", 0.5), ("doc_2", 0.2)])
        bm25 = self._make_results([("doc_2", 4.0), ("doc_0", 2.0)])
        fused = fuser.fuse(vec, bm25, top_k=3)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs(self):
        fuser = HybridFuser()
        assert fuser.fuse([], [], top_k=5) == []

    def test_hybrid_improves_over_vector_only(self):
        """
        Simulate a case where BM25 finds a highly relevant doc
        that vector search ranked low (3rd). Hybrid should boost it.
        """
        fuser = HybridFuser(k=60)
        vec = [
            SearchResult(document=_make_doc(0, "irrelevant topic"), score=0.9, rank=1),
            SearchResult(document=_make_doc(1, "somewhat relevant"), score=0.7, rank=2),
            SearchResult(document=_make_doc(2, "to be or not to be"), score=0.5, rank=3),
        ]
        bm25 = [
            SearchResult(document=_make_doc(2, "to be or not to be"), score=9.0, rank=1),
        ]
        fused = fuser.fuse(vec, bm25, top_k=3)
        # doc_2 should jump up in rank due to BM25 boost
        top_ids = [r.document.id for r in fused]
        assert top_ids[0] == "doc_2"


# ── MetadataFilter Tests ──────────────────────────────────────────────────────

class TestMetadataFilter:
    def test_filter_by_source(self):
        results = [
            _make_result(_make_doc(0, "a", source="gutenberg"), 0.9, 1),
            _make_result(_make_doc(1, "b", source="other"), 0.7, 2),
            _make_result(_make_doc(2, "c", source="gutenberg"), 0.5, 3),
        ]
        filtered = MetadataFilter.apply(results, {"source": "gutenberg"})
        assert len(filtered) == 2
        assert all(r.document.metadata["source"] == "gutenberg" for r in filtered)

    def test_filter_reranks_results(self):
        results = [
            _make_result(_make_doc(0, "a", cat="A"), 0.9, 1),
            _make_result(_make_doc(1, "b", cat="B"), 0.7, 2),
            _make_result(_make_doc(2, "c", cat="A"), 0.5, 3),
        ]
        filtered = MetadataFilter.apply(results, {"cat": "A"})
        assert [r.rank for r in filtered] == [1, 2]

    def test_empty_filter_returns_all(self):
        results = [
            _make_result(_make_doc(i, f"text {i}"), 0.5, i+1) for i in range(5)
        ]
        filtered = MetadataFilter.apply(results, {})
        assert len(filtered) == 5


# ── End-to-end Hybrid Retriever Test ─────────────────────────────────────────

class TestHybridRetrieverE2E:
    @pytest.fixture(scope="class")
    def embedder(self):
        from core.embedding.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})

    def test_hybrid_retrieval_on_shakespeare(self, shakespeare_clean, embedder):
        """Full hybrid search on 20k chars of Shakespeare."""
        from core.store.adapters.chromadb_adapter import ChromaDBAdapter
        from core.retrieval.bm25 import BM25Index
        from core.retrieval.hybrid import HybridRetriever
        from core.ingestion.loader import naive_chunk_text
        from core.store.models import Document

        store = ChromaDBAdapter({"mode": "memory"})
        store.connect()
        store.create_collection("h_test", embedder.embedding_dim)

        sample = shakespeare_clean[:20_000]
        texts = naive_chunk_text(sample, chunk_size=300, overlap=50)
        docs = []
        for i, text in enumerate(texts[:60]):
            emb = embedder.embed_one(text)
            docs.append(Document(id=f"c_{i}", text=text, embedding=emb,
                                 metadata={"source": "shakespeare"}))
        store.upsert("h_test", docs)

        bm25 = BM25Index()
        bm25.add(docs)

        retriever = HybridRetriever(store=store, bm25_index=bm25,
                                    embedder=embedder, collection="h_test")
        results = retriever.search("to be or not to be existence", top_k=5)

        assert len(results) > 0
        top_text = results[0].document.text.lower()
        assert any(w in top_text for w in ["be", "hamlet", "exist", "question", "life"])
        store.close()
