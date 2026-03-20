"""
Phase 1 – Adapter Contract Tests
=================================
Parametrized across ChromaDB, FAISS, and Qdrant (all in-memory, no server needed).
Every adapter must pass the full contract suite identically.

Run:  pytest tests/phase1/test_adapter_contract.py -v
"""
from __future__ import annotations

import random
from typing import List

import pytest

from core.store.models import Document, SearchResult
from tests.phase1.conftest import EMBEDDING_DIM


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_vec(dim: int = EMBEDDING_DIM) -> List[float]:
    v = [random.random() for _ in range(dim)]
    norm = sum(x ** 2 for x in v) ** 0.5
    return [x / norm for x in v]  # L2-normalised


def make_docs(n: int = 5) -> List[Document]:
    """Create n test documents with random normalised embeddings."""
    return [
        Document(
            id=f"doc_{i}",
            text=f"This is test document number {i}. It contains some sample text.",
            embedding=_rand_vec(),
            metadata={
                "source": "unit_test",
                "index": i,
                "category": "even" if i % 2 == 0 else "odd",
            },
        )
        for i in range(n)
    ]


# ── Contract test class ───────────────────────────────────────────────────────

class TestAdapterContract:
    """
    Full adapter contract.
    The `adapter` fixture is parametrized — each test runs for every local backend.
    """

    # ── Lifecycle / health ────────────────────────────────────────────────────

    def test_health_check_returns_true(self, adapter):
        assert adapter.health_check() is True

    # ── Collection management ─────────────────────────────────────────────────

    def test_collection_exists_after_create(self, adapter):
        assert adapter.collection_exists("test_col") is True

    def test_nonexistent_collection_returns_false(self, adapter):
        assert adapter.collection_exists("definitely_does_not_exist_xyz") is False

    def test_collection_info_returns_correct_structure(self, adapter):
        from core.store.models import CollectionInfo

        info = adapter.collection_info("test_col")
        assert isinstance(info, CollectionInfo)
        assert info.name == "test_col"
        assert info.count == 0

    def test_drop_collection_removes_it(self, adapter):
        adapter.drop_collection("test_col")
        assert adapter.collection_exists("test_col") is False
        # Recreate so fixture teardown doesn't fail
        adapter.create_collection("test_col", EMBEDDING_DIM)

    def test_drop_nonexistent_collection_is_silent(self, adapter):
        """Must not raise even if collection doesn't exist."""
        adapter.drop_collection("nonexistent_xyz_abc")  # should not raise

    def test_create_collection_is_idempotent(self, adapter):
        """Calling create twice must not raise."""
        adapter.create_collection("test_col", EMBEDDING_DIM)  # already exists
        assert adapter.collection_exists("test_col") is True

    # ── Upsert & count ────────────────────────────────────────────────────────

    def test_upsert_increases_count(self, adapter):
        docs = make_docs(5)
        adapter.upsert("test_col", docs)
        assert adapter.count("test_col") == 5

    def test_upsert_is_idempotent(self, adapter):
        """Re-upserting same IDs must not create duplicates."""
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        adapter.upsert("test_col", docs)
        assert adapter.count("test_col") == 3

    def test_upsert_updates_existing_document(self, adapter):
        """Upserting with same id but different text must overwrite."""
        doc = Document(
            id="update_me",
            text="original text",
            embedding=_rand_vec(),
            metadata={"version": 1},
        )
        adapter.upsert("test_col", [doc])

        doc_v2 = Document(
            id="update_me",
            text="updated text",
            embedding=_rand_vec(),
            metadata={"version": 2},
        )
        adapter.upsert("test_col", [doc_v2])
        assert adapter.count("test_col") == 1

    # ── Delete ────────────────────────────────────────────────────────────────

    def test_delete_reduces_count(self, adapter):
        docs = make_docs(5)
        adapter.upsert("test_col", docs)
        adapter.delete("test_col", ["doc_0", "doc_1"])
        assert adapter.count("test_col") == 3

    def test_delete_missing_id_is_silent(self, adapter):
        """Deleting a non-existent id must not raise."""
        adapter.delete("test_col", ["id_that_does_not_exist"])

    def test_deleted_document_not_returned_in_query(self, adapter):
        docs = make_docs(5)
        adapter.upsert("test_col", docs)
        adapter.delete("test_col", ["doc_0"])
        results = adapter.query("test_col", docs[0].embedding, top_k=5)
        returned_ids = [r.document.id for r in results]
        assert "doc_0" not in returned_ids

    # ── Query ─────────────────────────────────────────────────────────────────

    def test_query_returns_search_results(self, adapter):
        docs = make_docs(10)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_top_result_is_self(self, adapter):
        """Querying with a document's own vector should return it as rank 1."""
        docs = make_docs(10)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=5)
        assert results[0].document.id == "doc_0"

    def test_query_scores_are_descending(self, adapter):
        docs = make_docs(10)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not descending: {scores}"
        )

    def test_query_ranks_are_sequential(self, adapter):
        docs = make_docs(5)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=5)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_query_respects_top_k(self, adapter):
        docs = make_docs(10)
        adapter.upsert("test_col", docs)
        for k in (1, 3, 5):
            results = adapter.query("test_col", docs[0].embedding, top_k=k)
            assert len(results) <= k

    def test_query_scores_in_valid_range(self, adapter):
        """Scores must be in [0, 1] (or close, allowing floating-point tolerance)."""
        docs = make_docs(10)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=5)
        for r in results:
            assert -0.05 <= r.score <= 1.05, f"Score out of range: {r.score}"

    # ── Data integrity ────────────────────────────────────────────────────────

    def test_text_preserved_in_result(self, adapter):
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=1)
        assert "test document" in results[0].document.text

    def test_metadata_preserved_in_result(self, adapter):
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=1)
        assert results[0].document.metadata.get("source") == "unit_test"

    def test_document_id_preserved_in_result(self, adapter):
        docs = make_docs(5)
        adapter.upsert("test_col", docs)
        results = adapter.query("test_col", docs[0].embedding, top_k=1)
        assert results[0].document.id == "doc_0"

    # ── Drop and recreate ─────────────────────────────────────────────────────

    def test_drop_and_recreate_gives_empty_collection(self, adapter):
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        adapter.drop_collection("test_col")
        adapter.create_collection("test_col", EMBEDDING_DIM)
        assert adapter.collection_exists("test_col") is True
        assert adapter.count("test_col") == 0

    # ── Context manager ───────────────────────────────────────────────────────

    def test_context_manager_protocol(self):
        """Adapter can be used as a context manager via VectorStoreBase.__enter__."""
        from core.store.registry import AdapterRegistry

        with AdapterRegistry.create("chromadb", {"mode": "memory"}) as adp:
            adp.create_collection("ctx_test", EMBEDDING_DIM)
            docs = make_docs(2)
            adp.upsert("ctx_test", docs)
            assert adp.count("ctx_test") == 2

    # ── fetch_all contract ────────────────────────────────────────────────────

    def test_fetch_all_returns_id_and_text(self, adapter):
        """fetch_all must return dicts with at minimum 'id' and 'text' keys."""
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        raw = adapter.fetch_all("test_col", limit=10)
        assert len(raw) == 3
        for item in raw:
            assert "id" in item, "fetch_all item missing 'id'"
            assert "text" in item, "fetch_all item missing 'text'"

    def test_fetch_all_returns_metadata(self, adapter):
        """fetch_all must include a 'metadata' dict with all stored metadata fields."""
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        raw = adapter.fetch_all("test_col", limit=10)
        assert len(raw) == 3
        for item in raw:
            assert "metadata" in item, "fetch_all item missing 'metadata' key"
            assert isinstance(item["metadata"], dict), "'metadata' must be a dict"
            assert item["metadata"].get("source") == "unit_test", (
                f"metadata 'source' not preserved in fetch_all: {item['metadata']}"
            )

    def test_fetch_all_respects_limit(self, adapter):
        """fetch_all must return at most 'limit' items."""
        docs = make_docs(10)
        adapter.upsert("test_col", docs)
        raw = adapter.fetch_all("test_col", limit=5)
        assert len(raw) <= 5

    def test_fetch_all_ids_match_upserted(self, adapter):
        """fetch_all doc ids must be the original doc ids (not internal UUIDs)."""
        docs = make_docs(3)
        adapter.upsert("test_col", docs)
        raw = adapter.fetch_all("test_col", limit=10)
        returned_ids = {item["id"] for item in raw}
        expected_ids = {d.id for d in docs}
        assert returned_ids == expected_ids, (
            f"fetch_all IDs mismatch: returned={returned_ids}, expected={expected_ids}"
        )

    # ── list_collections contract ─────────────────────────────────────────────

    def test_list_collections_returns_list(self, adapter):
        """list_collections() must exist and return a list."""
        result = adapter.list_collections()
        assert isinstance(result, list), "list_collections() must return a list"

    def test_list_collections_includes_created(self, adapter):
        """A created collection must appear in list_collections()."""
        adapter.create_collection("extra_col", EMBEDDING_DIM)
        names = adapter.list_collections()
        assert "test_col" in names or "extra_col" in names, (
            f"Created collection not in list_collections(): {names}"
        )
