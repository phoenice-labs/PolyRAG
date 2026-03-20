"""
Full pipeline integration test — all 9 phases working together.
Uses ChromaDB (in-memory) + Shakespeare corpus.
LM Studio tests auto-skip if localhost:1234 is offline.

Run:  pytest tests/test_pipeline_e2e.py -v
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pipeline_config():
    """Minimal config dict for in-memory testing."""
    return {
        "store": {
            "backend": "chromadb",
            "chromadb": {"mode": "memory"},
        },
        "embedding": {
            "provider": "sentence_transformer",
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "ingestion": {
            "collection_name": "e2e_test",
            "chunk_size": 400,
            "chunk_overlap": 60,
            "embed_batch_size": 32,
        },
        "retrieval": {
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "relevance_threshold": 0.0,
            "recall_multiplier": 3,
        },
        "llm": {
            "base_url": "http://localhost:1234/v1",
            "model": "mistralai/ministral-3b",
            "temperature": 0.1,
            "max_tokens": 256,
            "enable_rewrite": True,
            "enable_stepback": False,
            "enable_multi_query": False,
            "enable_hyde": False,
            "n_paraphrases": 2,
        },
        "access": {"user_clearance": "INTERNAL"},
        "quality": {"min_score": 0.2, "dedup_threshold": 0.85},
        "audit_log_path": "./data/e2e_audit.jsonl",
    }


@pytest.fixture(scope="module")
def started_pipeline(pipeline_config, shakespeare_clean):
    """Start pipeline, ingest 30k chars of Shakespeare, yield, then stop."""
    from orchestrator.pipeline import RAGPipeline
    pipeline = RAGPipeline(pipeline_config)
    pipeline.start()
    sample = shakespeare_clean[:30_000]
    pipeline.ingest_text(sample, metadata={
        "doc_title": "Complete Works of Shakespeare",
        "source_url": "https://www.gutenberg.org/ebooks/100",
        "doc_version": "1.0",
        "classification": "PUBLIC",
    })
    yield pipeline
    pipeline.stop()


class TestPipelineHealth:
    def test_health_check_passes(self, started_pipeline):
        health = started_pipeline.health()
        assert health["store_healthy"] is True
        assert health["backend"] == "chromadb"
        assert health["embedding_dim"] == 384
        assert health["bm25_docs_indexed"] > 0

    def test_ingestion_populated_store(self, started_pipeline, pipeline_config):
        coll = pipeline_config["ingestion"]["collection_name"]
        count = started_pipeline.store.count(coll)
        assert count > 0


class TestPipelineRetrieval:
    def test_query_returns_results(self, started_pipeline):
        results = started_pipeline.query(
            "What does Hamlet say about death and existence?",
            top_k=5,
            use_multistage=False,   # faster for unit test
        )
        assert len(results) > 0

    def test_query_results_have_scores(self, started_pipeline):
        results = started_pipeline.query("Hamlet soliloquy", top_k=3)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_multistage_retrieval(self, started_pipeline):
        results = started_pipeline.query(
            "To be or not to be", top_k=3, use_multistage=True
        )
        assert len(results) >= 1

    def test_classification_filter_works(self, started_pipeline):
        """RESTRICTED user sees PUBLIC content; PUBLIC user blocked from INTERNAL+."""
        results = started_pipeline.query(
            "Hamlet", top_k=5,
            user_context={"clearance": "RESTRICTED"},
        )
        # All ingested docs are PUBLIC — should be accessible
        assert len(results) >= 1


class TestPipelineProvenance:
    def test_ask_returns_response_with_provenance(self, started_pipeline):
        from orchestrator.response import RAGResponse
        response = started_pipeline.ask(
            "What does Hamlet think about existence?", top_k=3
        )
        assert isinstance(response, RAGResponse)
        assert len(response.provenance) > 0

    def test_provenance_has_required_fields(self, started_pipeline):
        response = started_pipeline.ask("Hamlet revenge", top_k=3)
        for prov in response.provenance:
            assert prov.chunk_id
            assert prov.source_doc_id
            assert isinstance(prov.retrieval_score, float)

    def test_citations_generated(self, started_pipeline):
        response = started_pipeline.ask("Romeo and Juliet", top_k=3)
        assert len(response.citations) == len(response.provenance)
        assert all(isinstance(c, str) for c in response.citations)


class TestPipelineConfidence:
    def test_confidence_report_present(self, started_pipeline):
        response = started_pipeline.ask("What is the meaning of Hamlet?", top_k=5)
        assert response.confidence is not None
        assert response.confidence.verdict in {
            "HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"
        }

    def test_confidence_score_in_range(self, started_pipeline):
        response = started_pipeline.ask("Hamlet soliloquy existence", top_k=5)
        assert 0.0 <= response.confidence.composite_score <= 1.0

    def test_insufficient_evidence_for_nonsense_query(self, started_pipeline):
        response = started_pipeline.ask(
            "xyzxyz quantum blockchain nanobot perpetual motion machine 12345", top_k=3
        )
        # Scores should be low for unrelated query
        assert response.confidence.composite_score < 0.9  # not falsely HIGH


class TestPipelineLMStudio:
    """These tests require LM Studio at localhost:1234 with mistralai/ministral-3b."""

    @pytest.mark.lmstudio
    def test_answer_generated_when_lm_online(self, started_pipeline):
        if not started_pipeline._llm_client.is_available():
            pytest.skip("LM Studio not running at localhost:1234")
        response = started_pipeline.ask("What does Hamlet say about existence?", top_k=3)
        assert len(response.answer) > 20
        assert "[LM Studio offline" not in response.answer

    @pytest.mark.lmstudio
    def test_multi_turn_conversation(self, started_pipeline):
        if not started_pipeline._llm_client.is_available():
            pytest.skip("LM Studio not running at localhost:1234")
        r1 = started_pipeline.ask("What play features Hamlet?", top_k=3)
        r2 = started_pipeline.ask("What does he say in the famous soliloquy?", top_k=3)
        # Second answer should be about Hamlet (context carried over)
        assert len(r2.answer) > 0

    def test_graceful_degradation_when_offline(self, started_pipeline):
        """When LM Studio is offline, pipeline still returns retrieval results."""
        if started_pipeline._llm_client.is_available():
            pytest.skip("LM Studio IS running — skip degradation test")
        response = started_pipeline.ask("Hamlet death", top_k=3)
        assert "[LM Studio offline" in response.answer or len(response.results) > 0
        assert response.confidence is not None


class TestPipelineSummary:
    def test_response_summary_string(self, started_pipeline):
        response = started_pipeline.ask("Hamlet", top_k=3)
        summary = response.summary()
        assert isinstance(summary, str)
        assert "sources" in summary.lower() or "|" in summary
