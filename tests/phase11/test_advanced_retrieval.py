"""
Phase 11: Advanced Retrieval & LLM-Enhanced Extraction — tests.

Tests cover:
  - LLMEntityExtractor   (mocked LM Studio)
  - RAPTOR indexer       (mocked LM Studio) + retriever
  - ContextualReranker   (mocked LM Studio, batched ranking)
  - MMRReranker          (pure numpy, no LLM)
  - build_raptor_index() pipeline integration
  - config.yaml advanced_retrieval section
  - Already-implemented HyDE and Multi-Query noted (not re-tested here)
"""
from __future__ import annotations

import json
import pytest
from typing import List
from unittest.mock import MagicMock, patch

from core.store.models import Document, SearchResult


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_result(doc_id: str, text: str, score: float, embedding=None) -> SearchResult:
    emb = embedding or [0.1, 0.2, 0.3]
    return SearchResult(
        document=Document(id=doc_id, text=text, embedding=emb, metadata={}),
        score=score,
        rank=0,
    )


def _make_llm_client(response: str = '{"entities": [], "relations": []}') -> MagicMock:
    client = MagicMock()
    client.is_available.return_value = True
    client.complete.return_value = response
    return client


def _make_offline_client() -> MagicMock:
    client = MagicMock()
    client.is_available.return_value = False
    return client


# ──────────────────────────────────────────────────────────────────────────────
# 1. LLMEntityExtractor tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMEntityExtractor:

    def test_basic_extraction(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        response = json.dumps({
            "entities": [
                {"text": "Apple Inc.", "type": "ORG"},
                {"text": "Steve Jobs", "type": "PERSON"},
            ],
            "relations": [
                {"subject": "Steve Jobs", "predicate": "founded", "object": "Apple Inc."}
            ],
        })
        client = _make_llm_client(response)
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Steve Jobs founded Apple Inc.", chunk_id="c1")

        assert result.entity_count >= 2
        entity_texts = {e.text for e in result.entities}
        assert "Apple Inc." in entity_texts
        assert "Steve Jobs" in entity_texts

    def test_relation_extracted(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        response = json.dumps({
            "entities": [
                {"text": "Google", "type": "ORG"},
                {"text": "Larry Page", "type": "PERSON"},
            ],
            "relations": [
                {"subject": "Larry Page", "predicate": "co_founded", "object": "Google"}
            ],
        })
        client = _make_llm_client(response)
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Larry Page co-founded Google.", chunk_id="c2")

        assert result.relation_count >= 1
        predicates = {t.predicate for t in result.triples}
        assert "co_founded" in predicates

    def test_empty_text_returns_empty(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        client = _make_llm_client()
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("", chunk_id="c-empty")
        assert result.entity_count == 0
        assert result.relation_count == 0
        client.complete.assert_not_called()

    def test_offline_returns_empty(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        client = _make_offline_client()
        extractor = LLMEntityExtractor(llm_client=client)
        assert not extractor.is_available()
        result = extractor.extract("Some text about things.", chunk_id="c-offline")
        assert result.entity_count == 0

    def test_malformed_json_returns_empty(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        client = _make_llm_client("not json at all !!!")
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Some text.", chunk_id="c-bad")
        assert result.entity_count == 0

    def test_markdown_fenced_json_parsed(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        response = '```json\n{"entities": [{"text": "Microsoft", "type": "ORG"}], "relations": []}\n```'
        client = _make_llm_client(response)
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Microsoft released Windows.", chunk_id="c-md")
        assert result.entity_count >= 1

    def test_unknown_entity_type_normalised_to_concept(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        response = json.dumps({
            "entities": [{"text": "Quantum", "type": "UNKNOWN_TYPE"}],
            "relations": [],
        })
        client = _make_llm_client(response)
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Quantum physics.", chunk_id="c-type")
        assert result.entity_count == 1
        assert result.entities[0].entity_type == "CONCEPT"

    def test_entity_ids_are_normalised(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        response = json.dumps({
            "entities": [{"text": "Steve Jobs", "type": "PERSON"}],
            "relations": [],
        })
        client = _make_llm_client(response)
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Steve Jobs was a visionary.", chunk_id="c-norm")
        assert result.entities[0].entity_id == "PERSON:steve_jobs"

    def test_relation_missing_entity_skipped(self):
        """Relations referencing entities NOT in the entity list should be skipped."""
        from core.graph.llm_extractor import LLMEntityExtractor
        response = json.dumps({
            "entities": [{"text": "Apple", "type": "ORG"}],
            "relations": [
                {"subject": "apple", "predicate": "makes", "object": "unknown_entity"}
            ],
        })
        client = _make_llm_client(response)
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Apple makes things.", chunk_id="c-rel")
        # Relation skipped because "unknown_entity" not in entity_map
        assert result.relation_count == 0

    def test_llm_exception_returns_empty(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        client = MagicMock()
        client.is_available.return_value = True
        client.complete.side_effect = Exception("connection reset")
        extractor = LLMEntityExtractor(llm_client=client)
        result = extractor.extract("Some text.", chunk_id="c-exc")
        assert result.entity_count == 0

    def test_max_chunk_chars_truncation(self):
        from core.graph.llm_extractor import LLMEntityExtractor
        client = _make_llm_client('{"entities": [], "relations": []}')
        extractor = LLMEntityExtractor(llm_client=client, max_chunk_chars=50)
        long_text = "A" * 1000
        extractor.extract(long_text, chunk_id="c-long")
        call_args = client.complete.call_args[0][0]
        # Prompt should contain at most 50 chars of text
        assert "A" * 51 not in call_args


# ──────────────────────────────────────────────────────────────────────────────
# 2. RAPTOR indexer tests
# ──────────────────────────────────────────────────────────────────────────────

def _make_docs_with_embeddings(n: int = 8) -> List[Document]:
    import numpy as np
    rng = np.random.default_rng(0)
    docs = []
    for i in range(n):
        emb = rng.random(16).astype(float).tolist()
        docs.append(Document(
            id=f"doc-{i}", text=f"Document {i} about topic {i % 3}.",
            embedding=emb, metadata={}
        ))
    return docs


class TestRaptorIndexer:

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.collection_exists.return_value = False
        store.upsert.return_value = None
        return store

    @pytest.fixture
    def mock_embedder(self):
        import numpy as np
        embedder = MagicMock()
        embedder.embedding_dim = 16
        embedder.embed_one.return_value = np.random.default_rng(1).random(16).tolist()
        return embedder

    def test_offline_llm_returns_zero(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorIndexer
        client = _make_offline_client()
        indexer = RaptorIndexer(llm_client=client, embedder=mock_embedder, n_clusters=3)
        docs = _make_docs_with_embeddings(6)
        n = indexer.build(docs, "test_coll", mock_store)
        assert n == 0
        mock_store.upsert.assert_not_called()

    def test_build_creates_summaries(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorIndexer
        client = _make_llm_client("This is a cluster summary covering key topics.")
        indexer = RaptorIndexer(llm_client=client, embedder=mock_embedder, n_clusters=3)
        docs = _make_docs_with_embeddings(9)
        n = indexer.build(docs, "my_coll", mock_store)
        assert n > 0
        mock_store.upsert.assert_called()

    def test_empty_docs_returns_zero(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorIndexer
        client = _make_llm_client("summary")
        indexer = RaptorIndexer(llm_client=client, embedder=mock_embedder)
        n = indexer.build([], "coll", mock_store)
        assert n == 0

    def test_summary_doc_has_raptor_metadata(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorIndexer
        upserted_docs = []

        def capture_upsert(coll, docs):
            upserted_docs.extend(docs)

        mock_store.upsert.side_effect = capture_upsert
        client = _make_llm_client("A good summary.")
        indexer = RaptorIndexer(llm_client=client, embedder=mock_embedder, n_clusters=2)
        docs = _make_docs_with_embeddings(6)
        indexer.build(docs, "base_coll", mock_store)

        for doc in upserted_docs:
            assert doc.metadata.get("raptor_level") == 1
            assert "source_chunk_ids" in doc.metadata

    def test_raptor_collection_name(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorIndexer
        client = _make_llm_client("summary text")
        indexer = RaptorIndexer(llm_client=client, embedder=mock_embedder, n_clusters=2)
        docs = _make_docs_with_embeddings(4)
        indexer.build(docs, "polyrag_main", mock_store)
        # Should create collection with _raptor suffix
        mock_store.create_collection.assert_called_with("polyrag_main_raptor", 16)

    def test_kmeans_assigns_all_docs(self):
        from core.retrieval.raptor import RaptorIndexer
        import numpy as np
        embeddings = np.random.default_rng(5).random((12, 8)).astype(np.float32)
        labels = RaptorIndexer._kmeans(embeddings, k=4)
        assert len(labels) == 12
        assert all(0 <= l < 4 for l in labels)

    def test_kmeans_k_equals_n(self):
        from core.retrieval.raptor import RaptorIndexer
        import numpy as np
        embeddings = np.eye(3, dtype=np.float32)
        labels = RaptorIndexer._kmeans(embeddings, k=3)
        assert len(labels) == 3
        assert len(set(labels)) == 3   # each doc in its own cluster


class TestRaptorRetriever:

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.collection_exists.return_value = True
        # Leaf results
        leaf = [_make_result("leaf-1", "Leaf chunk text.", 0.9)]
        # RAPTOR summary results
        raptor = [_make_result("raptor-1", "RAPTOR summary.", 0.85)]
        store.query.side_effect = [leaf, raptor]
        return store

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed_one.return_value = [0.1] * 16
        return embedder

    def test_merges_leaf_and_summary(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorRetriever
        retriever = RaptorRetriever(vector_store=mock_store, embedder=mock_embedder)
        results = retriever.retrieve("query", "test_coll", top_k=5)
        ids = {r.document.id for r in results}
        assert "leaf-1" in ids
        assert "raptor-1" in ids

    def test_summary_score_weighted(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorRetriever
        retriever = RaptorRetriever(vector_store=mock_store, embedder=mock_embedder, summary_weight=0.5)
        results = retriever.retrieve("query", "test_coll", top_k=5)
        summary_r = next(r for r in results if r.document.id == "raptor-1")
        assert summary_r.score == pytest.approx(0.85 * 0.5, abs=1e-6)

    def test_no_raptor_collection(self, mock_embedder):
        from core.retrieval.raptor import RaptorRetriever
        store = MagicMock()
        store.collection_exists.return_value = False
        leaf = [_make_result("l1", "Leaf.", 0.8)]
        store.query.return_value = leaf
        retriever = RaptorRetriever(vector_store=store, embedder=mock_embedder)
        results = retriever.retrieve("q", "coll", top_k=5)
        assert len(results) >= 1
        # Only leaf results (no raptor query because collection doesn't exist)
        assert store.query.call_count == 1

    def test_summary_annotated(self, mock_store, mock_embedder):
        from core.retrieval.raptor import RaptorRetriever
        retriever = RaptorRetriever(vector_store=mock_store, embedder=mock_embedder)
        results = retriever.retrieve("query", "coll", top_k=5)
        summary_r = next(r for r in results if r.document.id == "raptor-1")
        assert summary_r.document.metadata.get("is_raptor_summary") is True


# ──────────────────────────────────────────────────────────────────────────────
# 3. ContextualReranker tests
# ──────────────────────────────────────────────────────────────────────────────

class TestContextualReranker:

    def _make_results(self, n: int = 3) -> List[SearchResult]:
        texts = ["Alpha passage.", "Beta passage.", "Gamma passage.", "Delta passage.", "Epsilon passage."]
        return [_make_result(f"r{i}", texts[i % len(texts)], 1.0 - i * 0.1) for i in range(n)]

    def test_offline_passthrough(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = _make_offline_client()
        reranker = ContextualReranker(llm_client=client)
        results = self._make_results(3)
        original_ids = [r.document.id for r in results]
        reranked = reranker.rerank("query", results)
        assert [r.document.id for r in reranked] == original_ids   # unchanged

    def test_valid_ranking_applied(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        # Ranking: passage 3 > 1 > 2 (0-based → 1-based: 3, 1, 2)
        client = _make_llm_client('{"ranking": [3, 1, 2]}')
        reranker = ContextualReranker(llm_client=client, llm_weight=1.0)
        results = self._make_results(3)
        reranked = reranker.rerank("query", results)
        # Best result should be original r2 (ranked 1st by LLM)
        assert reranked[0].document.id == "r2"

    def test_score_fusion(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = _make_llm_client('{"ranking": [1, 2, 3]}')
        reranker = ContextualReranker(llm_client=client, llm_weight=0.5)
        results = self._make_results(3)
        reranked = reranker.rerank("query", results)
        for r in reranked:
            assert "fused_score" in r.document.metadata
            assert 0.0 <= r.document.metadata["fused_score"] <= 1.0

    def test_invalid_ranking_falls_back(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = _make_llm_client('{"ranking": [1, 2]}')   # missing 3 → invalid
        reranker = ContextualReranker(llm_client=client, llm_weight=0.4)
        results = self._make_results(3)
        reranked = reranker.rerank("query", results)
        # Should not crash; returns some ordering
        assert len(reranked) == 3

    def test_llm_rank_score_metadata_set(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = _make_llm_client('{"ranking": [2, 1, 3]}')
        reranker = ContextualReranker(llm_client=client, llm_weight=0.4)
        results = self._make_results(3)
        reranked = reranker.rerank("query", results)
        scored = [r for r in reranked if "llm_rank_score" in r.document.metadata]
        assert len(scored) > 0

    def test_tail_results_untouched(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = _make_llm_client('{"ranking": [1, 2]}')
        reranker = ContextualReranker(llm_client=client, max_chunks_to_rank=2)
        results = self._make_results(4)   # 4 results, max_chunks_to_rank=2
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 4
        # Last 2 not scored by LLM
        tail = reranked[2:]
        assert all("llm_rank_score" not in r.document.metadata for r in tail)

    def test_empty_results(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = _make_llm_client()
        reranker = ContextualReranker(llm_client=client)
        assert reranker.rerank("q", []) == []

    def test_llm_exception_falls_back(self):
        from core.retrieval.contextual_reranker import ContextualReranker
        client = MagicMock()
        client.is_available.return_value = True
        client.complete.side_effect = Exception("timeout")
        reranker = ContextualReranker(llm_client=client, llm_weight=0.4)
        results = self._make_results(3)
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 3    # should not crash


# ──────────────────────────────────────────────────────────────────────────────
# 4. MMR re-ranker tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMMRReranker:

    def _make_results_with_embeddings(self, n: int, seed: int = 0) -> List[SearchResult]:
        """Create results with distinct embeddings (low similarity = diverse)."""
        import numpy as np
        rng = np.random.default_rng(seed)
        results = []
        for i in range(n):
            emb = rng.random(8).tolist()
            results.append(_make_result(f"doc-{i}", f"Text {i}.", score=1.0 - i * 0.05, embedding=emb))
        return results

    def _make_duplicate_results(self) -> List[SearchResult]:
        """Create results with near-identical embeddings."""
        emb = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        results = []
        for i in range(5):
            r = _make_result(f"dup-{i}", f"Nearly identical text {i}.", score=1.0 - i * 0.02, embedding=emb[:])
        # Add one outlier with very different embedding
        outlier_emb = [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]
        results = [_make_result(f"dup-{i}", f"Near-identical {i}.", 0.9 - i * 0.05, emb[:]) for i in range(5)]
        results.append(_make_result("outlier", "Very different content.", 0.6, outlier_emb))
        return results

    def test_returns_top_k_results(self):
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker(diversity_weight=0.5)
        results = self._make_results_with_embeddings(10)
        reranked = reranker.rerank(results, top_k=5)
        assert len(reranked) == 5

    def test_single_result_unchanged(self):
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker()
        results = [_make_result("only", "Only result.", 0.9)]
        reranked = reranker.rerank(results, top_k=5)
        assert len(reranked) == 1

    def test_no_embeddings_fallback(self):
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker()
        results = [_make_result(f"r{i}", f"Text {i}.", 0.9 - i * 0.1, embedding=None) for i in range(5)]
        reranked = reranker.rerank(results, top_k=3)
        assert len(reranked) == 3   # falls back to top_k slice

    def test_diversity_promotes_outlier(self):
        """MMR should include the diverse outlier even though its raw score is lower."""
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker(diversity_weight=0.7)
        results = self._make_duplicate_results()
        reranked = reranker.rerank(results, top_k=4)
        ids = [r.document.id for r in reranked]
        # Outlier has much lower score but should appear in top-4 due to diversity
        assert "outlier" in ids

    def test_zero_diversity_weight_is_pure_relevance(self):
        """diversity_weight=0.0 → λ=1.0 → pure relevance (same as top-k by score)."""
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker(diversity_weight=0.0)
        results = self._make_results_with_embeddings(8)
        reranked = reranker.rerank(results, top_k=4)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_assigned(self):
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker(diversity_weight=0.3)
        results = self._make_results_with_embeddings(6)
        reranked = reranker.rerank(results, top_k=4)
        assert [r.rank for r in reranked] == list(range(1, len(reranked) + 1))

    def test_lambda_property(self):
        from core.retrieval.mmr import MMRReranker
        r = MMRReranker(diversity_weight=0.3)
        assert r.lmbda == pytest.approx(0.7)

    def test_does_not_exceed_top_k(self):
        from core.retrieval.mmr import MMRReranker
        reranker = MMRReranker()
        results = self._make_results_with_embeddings(3)
        reranked = reranker.rerank(results, top_k=10)  # top_k > available
        assert len(reranked) == 3


# ──────────────────────────────────────────────────────────────────────────────
# 5. Phase 11 pipeline integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase11PipelineIntegration:

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Minimal config with FAISS (in-memory) and all Phase 11 features disabled."""
        return {
            "store": {"backend": "faiss", "faiss": {"mode": "memory"}},
            "embedding": {"provider": "sentence_transformer", "model": "all-MiniLM-L6-v2"},
            "ingestion": {"collection_name": "test_p11", "chunk_size": 200, "chunk_overlap": 20},
            "retrieval": {
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "relevance_threshold": 0.0, "recall_multiplier": 2,
            },
            "llm": {"base_url": "http://localhost:1234/v1", "model": "ministral-3b"},
            "access": {"user_clearance": "INTERNAL"},
            "quality": {"min_score": 0.0, "dedup_threshold": 0.99},
            "audit_log_path": str(tmp_path / "audit.jsonl"),
            "graph": {"enabled": False},
            "advanced_retrieval": {
                "raptor": {"enabled": False},
                "contextual_reranker": {"enabled": False},
                "mmr": {"enabled": True, "diversity_weight": 0.3},
            },
        }

    def test_pipeline_starts_with_mmr(self, minimal_config):
        from orchestrator.pipeline import RAGPipeline
        with RAGPipeline(minimal_config) as p:
            assert p._mmr_reranker is not None

    def test_pipeline_raptor_disabled_no_indexer(self, minimal_config):
        from orchestrator.pipeline import RAGPipeline
        with RAGPipeline(minimal_config) as p:
            assert p._raptor_indexer is None
            assert p._raptor_enabled is False

    def test_build_raptor_index_noop_when_disabled(self, minimal_config):
        from orchestrator.pipeline import RAGPipeline
        with RAGPipeline(minimal_config) as p:
            n = p.build_raptor_index()
            assert n == 0   # RAPTOR disabled → returns 0

    def test_mmr_applied_in_query(self, minimal_config):
        from orchestrator.pipeline import RAGPipeline
        with RAGPipeline(minimal_config) as p:
            p.ingest_text(
                "The quick brown fox jumps over the lazy dog. "
                "Artificial intelligence is transforming industries. "
                "Renewable energy sources include solar and wind power. "
                "Climate change affects ecosystems worldwide. "
                "Machine learning enables pattern recognition.",
                metadata={"source": "test"},
            )
            results = p.query("fox and dog", top_k=3)
            assert len(results) <= 3
            assert all(r.rank > 0 for r in results)

    def test_ingest_and_query_no_crash(self, minimal_config):
        from orchestrator.pipeline import RAGPipeline
        with RAGPipeline(minimal_config) as p:
            p.ingest_text(
                "The global economy depends on international trade and cooperation. "
                "Financial markets reflect investor confidence and economic indicators.",
                metadata={"source": "economy_test"},
            )
            results = p.query("economy and trade", top_k=3)
            assert isinstance(results, list)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Config validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigPhase11:

    def test_config_has_advanced_retrieval_section(self):
        import yaml, pathlib
        cfg_path = pathlib.Path("config/config.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "advanced_retrieval" in cfg
        adv = cfg["advanced_retrieval"]
        assert "raptor" in adv
        assert "contextual_reranker" in adv
        assert "mmr" in adv

    def test_config_raptor_defaults(self):
        import yaml, pathlib
        with open(pathlib.Path("config/config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        raptor = cfg["advanced_retrieval"]["raptor"]
        assert isinstance(raptor.get("n_clusters"), int)
        assert isinstance(raptor.get("summary_weight"), float)

    def test_config_mmr_defaults(self):
        import yaml, pathlib
        with open(pathlib.Path("config/config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mmr = cfg["advanced_retrieval"]["mmr"]
        assert "diversity_weight" in mmr
        assert 0.0 <= mmr["diversity_weight"] <= 1.0

    def test_config_graph_llm_extraction(self):
        import yaml, pathlib
        with open(pathlib.Path("config/config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        graph = cfg.get("graph", {})
        assert "llm_extraction" in graph
        llm_ext = graph["llm_extraction"]
        assert "enabled" in llm_ext
        assert "merge_with_spacy" in llm_ext


# ──────────────────────────────────────────────────────────────────────────────
# 7. Document: already-implemented HyDE + Multi-Query + Rewrite
# ──────────────────────────────────────────────────────────────────────────────

class TestAlreadyImplementedRetrievalMethods:
    """
    Verify the pre-existing Phase 5 advanced retrieval techniques are present.
    HyDE, Multi-Query, and Query Rewriting were implemented in Phases 5/11.
    These tests confirm the classes exist and are importable.
    """

    def test_hyde_is_importable(self):
        from core.query.rewriter import QueryExpander
        assert QueryExpander is not None

    def test_query_rewriter_is_importable(self):
        from core.query.rewriter import QueryRewriter
        assert QueryRewriter is not None

    def test_multi_query_generator_is_importable(self):
        from core.query.rewriter import MultiQueryGenerator
        assert MultiQueryGenerator is not None

    def test_query_intelligence_pipeline_is_importable(self):
        from core.query.context import QueryIntelligencePipeline
        assert QueryIntelligencePipeline is not None

    def test_hyde_produces_hypothetical_doc(self):
        from core.query.rewriter import QueryExpander
        client = MagicMock()
        client.is_available.return_value = True
        client.complete.return_value = "A hypothetical answer about the topic."
        embedder = MagicMock()
        embedder.embed_one.return_value = [0.1] * 384
        hyde = QueryExpander(client, embedder)
        hypothesis = hyde.generate_hypothesis("What is machine learning?")
        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0

    def test_multi_query_generates_paraphrases(self):
        from core.query.rewriter import MultiQueryGenerator
        client = MagicMock()
        client.is_available.return_value = True
        client.complete.return_value = "What is ML?\nExplain machine learning\nML definition"
        gen = MultiQueryGenerator(client, n_queries=3)
        paraphrases = gen.generate("What is machine learning?")
        assert isinstance(paraphrases, list)
        assert len(paraphrases) > 0
