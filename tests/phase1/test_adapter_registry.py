"""
Phase 1 – Adapter Registry Tests
==================================
Verifies that AdapterRegistry correctly resolves backend keys to adapter instances.

Run:  pytest tests/phase1/test_adapter_registry.py -v
"""
from __future__ import annotations

import pytest

from core.store.registry import AdapterRegistry
from core.store.base import VectorStoreBase


class TestAdapterRegistry:
    def test_available_returns_all_six_backends(self):
        backends = AdapterRegistry.available()
        assert set(backends) == {
            "chromadb", "faiss", "qdrant", "weaviate", "milvus", "pgvector"
        }

    def test_create_chromadb_returns_correct_type(self):
        from core.store.adapters.chromadb_adapter import ChromaDBAdapter
        adp = AdapterRegistry.create("chromadb", {"mode": "memory"})
        assert isinstance(adp, ChromaDBAdapter)
        assert isinstance(adp, VectorStoreBase)

    def test_create_faiss_returns_correct_type(self):
        from core.store.adapters.faiss_adapter import FAISSAdapter
        adp = AdapterRegistry.create("faiss", {"mode": "memory"})
        assert isinstance(adp, FAISSAdapter)
        assert isinstance(adp, VectorStoreBase)

    def test_create_qdrant_returns_correct_type(self):
        from core.store.adapters.qdrant_adapter import QdrantAdapter
        adp = AdapterRegistry.create("qdrant", {"mode": "memory"})
        assert isinstance(adp, QdrantAdapter)
        assert isinstance(adp, VectorStoreBase)

    def test_create_weaviate_returns_correct_type(self):
        from core.store.adapters.weaviate_adapter import WeaviateAdapter
        adp = AdapterRegistry.create("weaviate", {"mode": "embedded"})
        assert isinstance(adp, WeaviateAdapter)
        assert isinstance(adp, VectorStoreBase)

    def test_create_milvus_returns_correct_type(self):
        from core.store.adapters.milvus_adapter import MilvusAdapter
        adp = AdapterRegistry.create("milvus", {"mode": "local", "uri": ":memory:"})
        assert isinstance(adp, MilvusAdapter)
        assert isinstance(adp, VectorStoreBase)

    def test_create_pgvector_returns_correct_type(self):
        from core.store.adapters.pgvector_adapter import PGVectorAdapter
        adp = AdapterRegistry.create("pgvector", {"host": "localhost"})
        assert isinstance(adp, PGVectorAdapter)
        assert isinstance(adp, VectorStoreBase)

    def test_create_raises_on_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            AdapterRegistry.create("elasticsearch", {})

    def test_backend_keys_are_case_insensitive(self):
        """Registry lookup should be case-insensitive."""
        from core.store.adapters.chromadb_adapter import ChromaDBAdapter
        adp = AdapterRegistry.create("ChromaDB", {"mode": "memory"})
        assert isinstance(adp, ChromaDBAdapter)

    def test_each_adapter_satisfies_interface(self):
        """Every adapter must implement all abstract methods of VectorStoreBase."""
        import inspect
        abstract_methods = {
            name
            for name, method in inspect.getmembers(VectorStoreBase, predicate=inspect.isfunction)
            if getattr(method, "__isabstractmethod__", False)
        }
        for backend in AdapterRegistry.available():
            adp = AdapterRegistry.create(backend, {})
            for method_name in abstract_methods:
                assert hasattr(adp, method_name), (
                    f"{backend} adapter missing method: {method_name}"
                )
                assert callable(getattr(adp, method_name)), (
                    f"{backend}.{method_name} is not callable"
                )

    def test_each_adapter_config_is_stored(self):
        """Each adapter should store its config for later use."""
        cfg = {"mode": "memory", "extra_key": "extra_value"}
        adp = AdapterRegistry.create("chromadb", cfg)
        assert hasattr(adp, "config")
        assert adp.config == cfg
