"""
AdapterRegistry — maps string backend keys to concrete adapter classes.
Usage:
    adapter = AdapterRegistry.create("chromadb", {"mode": "memory"})
    adapter.connect()
"""
from __future__ import annotations

from typing import Dict, Type

from core.store.base import VectorStoreBase

_REGISTRY: Dict[str, str] = {
    "chromadb":  "core.store.adapters.chromadb_adapter.ChromaDBAdapter",
    "faiss":     "core.store.adapters.faiss_adapter.FAISSAdapter",
    "qdrant":    "core.store.adapters.qdrant_adapter.QdrantAdapter",
    "weaviate":  "core.store.adapters.weaviate_adapter.WeaviateAdapter",
    "milvus":    "core.store.adapters.milvus_adapter.MilvusAdapter",
    "pgvector":  "core.store.adapters.pgvector_adapter.PGVectorAdapter",
}


class AdapterRegistry:
    """Factory for vector-store adapters."""

    @staticmethod
    def create(backend: str, config: dict) -> VectorStoreBase:
        """
        Instantiate an adapter by name.

        Parameters
        ----------
        backend : one of chromadb | faiss | qdrant | weaviate | milvus | pgvector
        config  : backend-specific config dict (mirrors config.yaml section)
        """
        key = backend.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available: {sorted(_REGISTRY.keys())}"
            )

        module_path, class_name = _REGISTRY[key].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls: Type[VectorStoreBase] = getattr(module, class_name)
        return cls(config)

    @staticmethod
    def available() -> list[str]:
        """Return list of registered backend keys."""
        return sorted(_REGISTRY.keys())
