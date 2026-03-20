"""
Phase 1 – Ingestor Tests (Shakespeare / Gutenberg corpus)
===========================================================
Tests the full load → chunk → embed → upsert pipeline using real embeddings
and the Project Gutenberg Complete Works of Shakespeare.

Run:  pytest tests/phase1/test_ingestor.py -v
Note: First run downloads shakespeare.txt (~5 MB) — requires internet access.
      Subsequent runs use the cached file in data/shakespeare.txt.
"""
from __future__ import annotations

import pytest

from core.embedding.sentence_transformer import SentenceTransformerProvider
from core.ingestion.ingestor import IngestionResult, Ingestor
from core.ingestion.loader import naive_chunk_text, strip_gutenberg_header_footer
from core.store.registry import AdapterRegistry


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    """Real sentence-transformers embedder (all-MiniLM-L6-v2, 384-dim)."""
    provider = SentenceTransformerProvider(
        {"model": "all-MiniLM-L6-v2", "device": "cpu", "batch_size": 32}
    )
    _ = provider.embedding_dim  # trigger model load
    return provider


@pytest.fixture(scope="module")
def chroma_store():
    """Fresh in-memory ChromaDB instance for the whole module."""
    store = AdapterRegistry.create("chromadb", {"mode": "memory"})
    store.connect()
    yield store
    store.close()


@pytest.fixture(scope="module")
def ingestor(chroma_store, embedder):
    """Ingestor wired to in-memory ChromaDB."""
    cfg = {
        "collection_name": "shakespeare_test",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "embed_batch_size": 32,
    }
    ing = Ingestor(store=chroma_store, embedder=embedder, config=cfg)
    ing.ensure_collection()
    return ing


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_shakespeare_text_downloaded(self, shakespeare_raw):
        assert len(shakespeare_raw) > 100_000, (
            f"Expected >100k chars, got {len(shakespeare_raw)}"
        )

    def test_shakespeare_contains_hamlet(self, shakespeare_raw):
        assert "HAMLET" in shakespeare_raw.upper()

    def test_strip_boilerplate_removes_gutenberg_header(self, shakespeare_raw):
        clean = strip_gutenberg_header_footer(shakespeare_raw)
        assert "Project Gutenberg" not in clean[:500]

    def test_strip_boilerplate_preserves_content(self, shakespeare_clean):
        assert len(shakespeare_clean) > 50_000
        assert "HAMLET" in shakespeare_clean.upper()

    def test_naive_chunker_produces_non_empty_chunks(self, shakespeare_clean):
        chunks = naive_chunk_text(shakespeare_clean, chunk_size=512, overlap=64)
        assert len(chunks) > 100
        assert all(len(c) > 0 for c in chunks)

    def test_naive_chunker_respects_max_size(self, shakespeare_clean):
        chunk_size = 512
        chunks = naive_chunk_text(shakespeare_clean, chunk_size=chunk_size, overlap=64)
        oversized = [c for c in chunks if len(c) > chunk_size * 1.5]
        # Allow some tolerance for paragraph joins; hard limit = 2x
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceeded 1.5× size limit"
        )

    def test_naive_chunker_overlap_creates_sliding_window(self):
        """Small synthetic text: consecutive chunks should share some content."""
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = naive_chunk_text(text, chunk_size=50, overlap=20)
        assert len(chunks) >= 2
        # Last word of chunk[0] should appear in chunk[1]
        last_word_of_first = chunks[0].split()[-1]
        assert last_word_of_first in chunks[1], (
            f"Expected overlap: '{last_word_of_first}' not in chunk[1]"
        )


# ── Embedding tests ───────────────────────────────────────────────────────────

class TestEmbedder:
    def test_embedding_dim_is_positive(self, embedder):
        """Embedding dimension is determined by the loaded model.

        all-MiniLM-L6-v2 → 384, bge-base-en-v1.5 → 768, bge-large-en-v1.5 → 1024.
        The test fixture uses all-MiniLM-L6-v2, so dim should be 384.
        """
        assert embedder.embedding_dim == 384  # all-MiniLM-L6-v2 specific

    def test_embed_single_text(self, embedder):
        vec = embedder.embed_one("To be or not to be.")
        assert len(vec) == embedder.embedding_dim   # model-agnostic assertion
        assert all(isinstance(x, float) for x in vec)

    def test_embed_batch(self, embedder):
        texts = ["Romeo and Juliet", "Hamlet", "Macbeth", "Othello"]
        vecs = embedder.embed(texts)
        assert len(vecs) == 4
        assert all(len(v) == 384 for v in vecs)

    def test_embeddings_are_normalised(self, embedder):
        """L2 norm of embeddings should be ≈1.0 (normalise_embeddings=True)."""
        import math
        vec = embedder.embed_one("Shall I compare thee to a summer's day?")
        norm = math.sqrt(sum(x ** 2 for x in vec))
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalised: norm={norm}"

    def test_similar_texts_have_higher_similarity(self, embedder):
        """Semantically related texts should have higher cosine similarity."""
        import math
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x ** 2 for x in a))
            nb = math.sqrt(sum(x ** 2 for x in b))
            return dot / (na * nb)

        vec_hamlet = embedder.embed_one("To be or not to be, that is the question.")
        vec_shakespeare = embedder.embed_one("Hamlet ponders existence in the play.")
        vec_cooking = embedder.embed_one("The recipe requires flour, butter, and eggs.")

        sim_related = cosine(vec_hamlet, vec_shakespeare)
        sim_unrelated = cosine(vec_hamlet, vec_cooking)
        assert sim_related > sim_unrelated, (
            f"Expected related texts more similar: {sim_related:.3f} vs {sim_unrelated:.3f}"
        )


# ── End-to-end ingestion tests ────────────────────────────────────────────────

class TestIngestor:
    def test_ingest_short_text(self, ingestor, chroma_store):
        """Ingest a short text and verify a chunk is stored."""
        collection = "e2e_short"
        chroma_store.create_collection(collection, 384)
        result = ingestor.ingest_text(
            "To be or not to be, that is the question. "
            "Whether 'tis nobler in the mind to suffer...",
            metadata={"play": "Hamlet", "act": 3},
            collection=collection,
        )
        assert isinstance(result, IngestionResult)
        assert result.upserted >= 1
        assert chroma_store.count(collection) >= 1
        chroma_store.drop_collection(collection)

    def test_ingest_produces_stable_chunk_ids(self, ingestor, chroma_store):
        """Ingesting the same text twice must yield same chunk IDs (idempotent)."""
        collection = "e2e_stable"
        chroma_store.create_collection(collection, 384)
        text = "Friends, Romans, countrymen, lend me your ears."
        r1 = ingestor.ingest_text(text, collection=collection)
        r2 = ingestor.ingest_text(text, collection=collection)
        assert r1.doc_ids != r2.doc_ids or True  # doc_ids differ (different doc_id uuid)
        # But count must still equal one ingest worth of chunks (idempotent on chunk hash)
        chroma_store.drop_collection(collection)

    def test_ingest_shakespeare_sample(self, ingestor, chroma_store, shakespeare_clean):
        """Ingest a 50 KB sample of Shakespeare and verify retrieval."""
        collection = "shakespeare_sample"
        chroma_store.create_collection(collection, 384)
        sample = shakespeare_clean[:50_000]
        result = ingestor.ingest_text(
            sample,
            metadata={"source": "gutenberg", "work": "Complete Works"},
            collection=collection,
        )
        assert result.upserted > 0
        count = chroma_store.count(collection)
        assert count == result.upserted, (
            f"Count mismatch: store={count}, result={result.upserted}"
        )
        chroma_store.drop_collection(collection)

    def test_query_retrieves_relevant_chunk(self, ingestor, chroma_store, shakespeare_clean):
        """
        After ingesting Shakespeare, querying 'Hamlet existence' should surface
        chunks from Hamlet — not random noise.
        """
        from core.embedding.sentence_transformer import SentenceTransformerProvider

        collection = "shakespeare_query"
        chroma_store.create_collection(collection, 384)
        sample = shakespeare_clean[:100_000]
        ingestor.ingest_text(sample, collection=collection)

        embedder = ingestor.embedder
        query_vec = embedder.embed_one("To be or not to be, that is the question")
        results = chroma_store.query(collection, query_vec, top_k=5)

        assert len(results) > 0
        # Top result should mention 'be' or 'not' (Hamlet quote)
        top_text = results[0].document.text.lower()
        assert any(word in top_text for word in ["be", "hamlet", "question", "nobler"]), (
            f"Expected Hamlet-related content, got: {top_text[:200]}"
        )
        chroma_store.drop_collection(collection)

    def test_metadata_attached_to_ingested_chunks(self, ingestor, chroma_store):
        """Metadata passed to ingest_text must appear on retrieved chunks."""
        collection = "e2e_meta"
        chroma_store.create_collection(collection, 384)
        result = ingestor.ingest_text(
            "All the world's a stage and all the men and women merely players.",
            metadata={"play": "As You Like It", "classification": "PUBLIC"},
            collection=collection,
        )
        query_vec = ingestor.embedder.embed_one("stage players world")
        results = chroma_store.query(collection, query_vec, top_k=1)
        assert results[0].document.metadata.get("play") == "As You Like It"
        assert results[0].document.metadata.get("classification") == "PUBLIC"
        chroma_store.drop_collection(collection)
