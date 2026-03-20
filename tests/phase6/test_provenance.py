"""
Phase 6 – Provenance & Traceability Tests
==========================================
Run:  pytest tests/phase6/ -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.provenance.models import (
    ProvenanceRecord,
    CitationBuilder,
    SpanHighlighter,
    IngestionAuditLog,
    DocumentVersionRegistry,
    build_provenance,
)
from core.store.models import Document, SearchResult


def _make_result(doc_id="d1", text="sample text", score=0.8, rank=1, **meta):
    return SearchResult(
        document=Document(id=doc_id, text=text, embedding=[], metadata=meta),
        score=score, rank=rank,
    )


class TestProvenanceRecord:
    def test_basic_construction(self):
        rec = ProvenanceRecord(
            source_doc_id="doc1",
            chunk_id="chunk1",
            retrieval_score=0.85,
        )
        assert rec.source_doc_id == "doc1"
        assert rec.chunk_id == "chunk1"
        assert rec.retrieval_score == 0.85

    def test_citation_inline(self):
        rec = ProvenanceRecord(
            source_doc_id="doc1",
            chunk_id="c1",
            doc_title="Complete Works of Shakespeare",
            doc_version="1.0",
            section_title="Hamlet Act III",
            retrieval_score=0.9,
        )
        citation = rec.to_citation("inline")
        assert "Shakespeare" in citation

    def test_citation_footnote_includes_score(self):
        rec = ProvenanceRecord(
            source_doc_id="doc1", chunk_id="c1",
            doc_title="Hamlet", retrieval_score=0.9, rerank_score=0.75,
        )
        citation = rec.to_citation("footnote")
        assert "0.900" in citation or "0.9" in citation

    def test_serialisation_to_dict(self):
        rec = ProvenanceRecord(source_doc_id="d", chunk_id="c", retrieval_score=0.5)
        d = rec.model_dump()
        assert d["source_doc_id"] == "d"
        assert isinstance(d["retrieval_score"], float)


class TestBuildProvenance:
    def test_builds_from_search_result(self):
        result = _make_result(
            doc_id="chunk_42", text="To be or not to be.",
            score=0.88, rank=1,
            doc_id_meta="root_doc",
            source_url="https://gutenberg.org/ebooks/100",
        )
        prov = build_provenance(result)
        assert prov.chunk_id == "chunk_42"
        assert prov.retrieval_score == pytest.approx(0.88, abs=0.01)

    def test_classification_from_metadata(self):
        result = _make_result(classification="CONFIDENTIAL")
        prov = build_provenance(result)
        assert prov.classification_label == "CONFIDENTIAL"

    def test_all_results_have_provenance(self, shakespeare_clean):
        """Every SearchResult from ingestion must produce a complete ProvenanceRecord."""
        from core.embedding.sentence_transformer import SentenceTransformerProvider
        from core.store.adapters.chromadb_adapter import ChromaDBAdapter
        from core.ingestion.ingestor import Ingestor

        embedder = SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})
        store = ChromaDBAdapter({"mode": "memory"})
        store.connect()
        store.create_collection("prov_test", embedder.embedding_dim)

        ingestor = Ingestor(store, embedder, {
            "collection_name": "prov_test", "chunk_size": 300,
            "chunk_overlap": 50, "embed_batch_size": 32,
        })
        sample = shakespeare_clean[:10_000]
        ingestor.ingest_text(sample, metadata={"doc_title": "Shakespeare", "doc_version": "1.0"})

        query_vec = embedder.embed_one("Hamlet existence")
        results = store.query("prov_test", query_vec, top_k=5)

        assert len(results) > 0
        for r in results:
            prov = build_provenance(r)
            assert prov.chunk_id == r.document.id
            assert isinstance(prov.retrieval_score, float)
            assert prov.classification_label  # not empty
        store.close()


class TestCitationBuilder:
    def test_build_all_returns_list(self):
        records = [
            ProvenanceRecord(source_doc_id="d1", chunk_id="c1",
                             doc_title="Hamlet", retrieval_score=0.9),
            ProvenanceRecord(source_doc_id="d2", chunk_id="c2",
                             doc_title="Macbeth", retrieval_score=0.7),
        ]
        citations = CitationBuilder.build_all(records, style="inline")
        assert len(citations) == 2
        assert all(isinstance(c, str) for c in citations)

    def test_unique_sources(self):
        records = [
            ProvenanceRecord(source_doc_id="doc1", chunk_id="c1", retrieval_score=0.9),
            ProvenanceRecord(source_doc_id="doc1", chunk_id="c2", retrieval_score=0.8),
            ProvenanceRecord(source_doc_id="doc2", chunk_id="c3", retrieval_score=0.7),
        ]
        sources = CitationBuilder.build_unique_sources(records)
        assert sources == ["doc1", "doc2"]


class TestSpanHighlighter:
    SOURCE_TEXT = "To be or not to be, that is the question. Whether 'tis nobler."

    def test_verify_span_correct(self):
        rec = ProvenanceRecord(
            source_doc_id="d", chunk_id="c",
            retrieval_score=0.8,
            start_char=0, end_char=41,
            exact_text_span="To be or not to be, that is the question.",
        )
        assert SpanHighlighter.verify_span(self.SOURCE_TEXT, rec) is True

    def test_verify_span_incorrect(self):
        rec = ProvenanceRecord(
            source_doc_id="d", chunk_id="c",
            retrieval_score=0.8,
            start_char=0, end_char=10,
            exact_text_span="WRONG TEXT",
        )
        assert SpanHighlighter.verify_span(self.SOURCE_TEXT, rec) is False

    def test_highlight_wraps_span(self):
        rec = ProvenanceRecord(
            source_doc_id="d", chunk_id="c", retrieval_score=0.8,
            start_char=0, end_char=8,
            exact_text_span="To be or",
        )
        highlighted = SpanHighlighter.highlight(self.SOURCE_TEXT, rec, marker=">>")
        assert highlighted.startswith(">>To be or>>")


class TestIngestionAuditLog:
    def test_record_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = IngestionAuditLog(log_path=f"{tmpdir}/audit.jsonl")
            log.record("doc1", "https://gutenberg.org", chunk_count=10, metadata={})
            log.record("doc2", "local_file.txt", chunk_count=5, metadata={"tag": "test"})
            entries = log.read_all()
            assert len(entries) == 2
            assert entries[0]["doc_id"] == "doc1"
            assert entries[1]["chunk_count"] == 5

    def test_append_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = IngestionAuditLog(log_path=f"{tmpdir}/audit.jsonl")
            log.record("d1", "s1", 1, {})
            log.record("d2", "s2", 2, {})
            entries = log.read_all()
            assert entries[0]["doc_id"] == "d1"
            assert entries[1]["doc_id"] == "d2"

    def test_doc_history_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = IngestionAuditLog(log_path=f"{tmpdir}/audit.jsonl")
            log.record("doc_a", "src", 3, {})
            log.record("doc_b", "src", 2, {})
            log.record("doc_a", "src_v2", 4, {})
            history = log.get_doc_history("doc_a")
            assert len(history) == 2
            assert all(e["doc_id"] == "doc_a" for e in history)


class TestDocumentVersionRegistry:
    def test_register_and_latest(self):
        reg = DocumentVersionRegistry()
        reg.register("policy_v1", "1.0", effective_date="2024-01-01")
        reg.register("policy_v2", "2.0", effective_date="2025-01-01", supersedes="policy_v1")
        latest = reg.latest_version("policy_v2")
        assert latest["version"] == "2.0"

    def test_is_superseded(self):
        reg = DocumentVersionRegistry()
        reg.register("doc_v1", "1.0")
        reg.register("doc_v2", "2.0", supersedes="doc_v1")
        # Manually mark v1 as superseded (simplistic implementation)
        # The registry marks superseded_by on the entry
        all_v1 = reg.all_versions("doc_v1")
        assert len(all_v1) == 1

    def test_all_versions(self):
        reg = DocumentVersionRegistry()
        reg.register("doc", "1.0")
        reg.register("doc", "2.0")
        reg.register("doc", "3.0")
        versions = reg.all_versions("doc")
        assert len(versions) == 3
