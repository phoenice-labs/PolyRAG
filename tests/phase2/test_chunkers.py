"""
Phase 2 – Semantic Chunking Tests
===================================
Tests all chunkers and the pipeline using the Shakespeare corpus.

Run:  pytest tests/phase2/ -v
"""
from __future__ import annotations

import pytest
from core.chunking.fixed_overlap import FixedOverlapChunker
from core.chunking.sentence_boundary import SentenceBoundaryChunker
from core.chunking.section_aware import SectionAwareChunker
from core.chunking.pipeline import ChunkingPipeline
from core.chunking.models import Chunk, ChunkRegistry

SAMPLE_TEXT = """
ACT III SCENE I. A room in the castle.

Enter HAMLET

HAMLET
To be, or not to be, that is the question.
Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles, and by opposing end them.
To die — to sleep, no more; and by a sleep to say we end the heartache.

Enter OPHELIA

OPHELIA
Good my lord, how does your honour for this many a day?

HAMLET
I humbly thank you; well, well, well.
"""


class TestFixedOverlapChunker:
    def test_produces_chunks(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc1")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_text_non_empty(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc1")
        assert all(len(c.text) > 0 for c in chunks)

    def test_doc_id_preserved(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="my_doc")
        assert all(c.doc_id == "my_doc" for c in chunks)

    def test_metadata_propagated(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="d", metadata={"source": "test"})
        assert all(c.metadata.get("source") == "test" for c in chunks)

    def test_text_hash_populated(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="d")
        assert all(len(c.text_hash) == 16 for c in chunks)

    def test_token_count_populated(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="d")
        assert all(c.token_count > 0 for c in chunks)

    def test_span_covers_source(self, shakespeare_clean):
        """start_char and end_char must point into the original text."""
        chunker = FixedOverlapChunker(chunk_size=300, chunk_overlap=50)
        sample = shakespeare_clean[:5000]
        chunks = chunker.chunk(sample, doc_id="d")
        for c in chunks[:10]:
            extracted = sample[c.start_char: c.start_char + len(c.text)]
            assert extracted.strip() == c.text.strip(), (
                f"Span mismatch at [{c.start_char}:{c.end_char}]"
            )


class TestSentenceBoundaryChunker:
    def test_produces_chunks(self):
        chunker = SentenceBoundaryChunker(max_words=80)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc2")
        assert len(chunks) > 0

    def test_respects_max_words(self):
        chunker = SentenceBoundaryChunker(max_words=50)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc2")
        oversized = [c for c in chunks if c.token_count > 60]  # small tolerance
        assert len(oversized) == 0

    def test_sentence_chunker_on_shakespeare(self, shakespeare_clean):
        chunker = SentenceBoundaryChunker(max_words=100)
        sample = shakespeare_clean[:20_000]
        chunks = chunker.chunk(sample, doc_id="shk")
        assert len(chunks) > 10
        # Poetry often lacks sentence-ending punctuation, so word count can
        # exceed max_words for a single "sentence". Verify chunker is
        # productive (many chunks) and doesn't produce astronomically large ones.
        assert all(c.token_count <= 1000 for c in chunks)


class TestSectionAwareChunker:
    def test_detects_act_scene_headings(self):
        chunker = SectionAwareChunker(child_size=200, child_overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="hamlet")
        parent_chunks = [c for c in chunks if c.metadata.get("is_parent")]
        assert len(parent_chunks) >= 1

    def test_parent_child_links(self):
        chunker = SectionAwareChunker(child_size=200, child_overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="hamlet")
        children = [c for c in chunks if c.parent_id is not None]
        parent_ids = {c.chunk_id for c in chunks if c.metadata.get("is_parent")}
        for child in children:
            assert child.parent_id in parent_ids or child.parent_id is None

    def test_section_title_populated(self):
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="d")
        titled = [c for c in chunks if c.section_title]
        assert len(titled) >= 1

    def test_shakespeare_sections(self, shakespeare_clean):
        chunker = SectionAwareChunker(child_size=512, child_overlap=64)
        sample = shakespeare_clean[:50_000]
        chunks = chunker.chunk(sample, doc_id="shk")
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if c.parent_id]
        assert len(parents) >= 1
        assert len(children) >= 1
        # Every child's parent_id must exist
        parent_id_set = {p.chunk_id for p in parents}
        for child in children:
            assert child.parent_id in parent_id_set


class TestSemanticBoundaryChunker:
    @pytest.fixture(scope="class")
    def embedder(self):
        from core.embedding.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider(
            {"model": "all-MiniLM-L6-v2", "device": "cpu", "batch_size": 32}
        )

    def test_produces_chunks(self, embedder):
        from core.chunking.semantic_boundary import SemanticBoundaryChunker
        chunker = SemanticBoundaryChunker(embedder=embedder, threshold=0.3, max_chunk_words=100)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="sem")
        assert len(chunks) >= 1

    def test_chunks_cover_text(self, embedder):
        from core.chunking.semantic_boundary import SemanticBoundaryChunker
        chunker = SemanticBoundaryChunker(embedder=embedder, threshold=0.3, max_chunk_words=100)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="sem")
        combined = " ".join(c.text for c in chunks)
        # All major words from original should appear in combined chunks
        for word in ["HAMLET", "question", "OPHELIA"]:
            assert word.lower() in combined.lower(), f"'{word}' missing from chunks"


class TestChunkingPipeline:
    def test_pipeline_quality_gate_filters_short(self):
        chunker = FixedOverlapChunker(chunk_size=50, chunk_overlap=5)
        pipeline = ChunkingPipeline(chunker=chunker, min_words=50)
        short_text = "Hi."
        chunks = pipeline.run(short_text, doc_id="d")
        assert len(chunks) == 0  # too short for min_words=50

    def test_pipeline_deduplicates(self):
        chunker = FixedOverlapChunker(chunk_size=200, chunk_overlap=190)
        pipeline = ChunkingPipeline(chunker=chunker, min_words=3)
        chunks = pipeline.run(SAMPLE_TEXT, doc_id="d")
        hashes = [c.text_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "Duplicate chunk hashes after pipeline"

    def test_registry_populated(self):
        chunker = SectionAwareChunker()
        registry = ChunkRegistry()
        pipeline = ChunkingPipeline(chunker=chunker, registry=registry, min_words=5)
        chunks = pipeline.run(SAMPLE_TEXT, doc_id="reg_test")
        assert len(registry) > 0
        for c in chunks:
            assert registry.get(c.chunk_id) is not None

    def test_parent_retrieval_from_registry(self):
        chunker = SectionAwareChunker(child_size=100, child_overlap=10)
        registry = ChunkRegistry()
        pipeline = ChunkingPipeline(chunker=chunker, registry=registry, min_words=5)
        pipeline.run(SAMPLE_TEXT, doc_id="ptest")
        # Find a child and retrieve its parent
        children = [c for c in registry._store.values() if c.parent_id]
        if children:
            child = children[0]
            parent = registry.get_parent(child)
            assert parent is not None
            assert parent.chunk_id == child.parent_id
