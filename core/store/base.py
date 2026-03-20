"""
Abstract base class that every vector-store adapter must implement.
The orchestration layer depends ONLY on this interface — never on a concrete adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.store.models import CollectionInfo, Document, SearchResult


class VectorStoreBase(ABC):
    """
    Unified vector-store contract.

    Implementors
    ------------
    ChromaDBAdapter, FAISSAdapter, QdrantAdapter,
    WeaviateAdapter, MilvusAdapter, PGVectorAdapter
    """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """Open connection / initialise client."""

    @abstractmethod
    def close(self) -> None:
        """Release resources and close connection."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and operational."""

    # ── Collection management ─────────────────────────────────────────────────

    @abstractmethod
    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        """Create a collection/index. No-op if it already exists."""

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete a collection and all its data."""

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Return True if the collection exists."""

    @abstractmethod
    def collection_info(self, name: str) -> CollectionInfo:
        """Return metadata about a collection."""

    # ── Data operations ───────────────────────────────────────────────────────

    @abstractmethod
    def upsert(self, collection: str, documents: List[Document]) -> None:
        """
        Insert or update documents.
        Each Document must have a non-None embedding.
        Existing documents with the same id are overwritten.
        """

    @abstractmethod
    def delete(self, collection: str, ids: List[str]) -> None:
        """Delete documents by id.  Missing ids are silently ignored."""

    @abstractmethod
    def count(self, collection: str) -> int:
        """Return the number of documents in the collection."""

    # ── Search ────────────────────────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Nearest-neighbour search.

        Parameters
        ----------
        collection:      Target collection name.
        query_embedding: Dense query vector.
        top_k:           Maximum results to return.
        filters:         Optional metadata filter dict (backend-specific dialect
                         normalised by each adapter).

        Returns
        -------
        List of SearchResult sorted by score descending, ranks 1..n.
        """

    def fetch_all(self, collection: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch up to ``limit`` raw documents (id + text) without vector search.

        Used by bulk operations such as LLM graph enhancement.
        Returns a list of dicts, each with at minimum ``"id"`` and ``"text"`` keys.
        Subclasses should override this for backend-specific bulk retrieval.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fetch_all(). "
            "Override this method in the adapter subclass."
        )

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "VectorStoreBase":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()
