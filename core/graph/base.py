"""
GraphStoreBase — abstract contract for all graph backends.
Mirrors the VectorStoreBase pattern for zero-change backend swaps.

Supported implementations:
  networkx  → NetworkXGraphStore  (in-memory, tests / fallback)
  kuzu      → KuzuGraphStore      (embedded persistent, production default)
  neo4j     → Neo4jGraphStore     (enterprise, @integration)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from core.graph.models import Entity, GraphPath, Relation


class GraphStoreBase(ABC):
    """Abstract Knowledge Graph store.

    Every backend must implement all methods below.
    Porting from Kuzu → Neo4j = swap implementation class in GraphStoreRegistry.
    Both use Cypher so query syntax stays identical.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """Initialise connection / open DB file."""

    @abstractmethod
    def close(self) -> None:
        """Flush and close."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all nodes and edges (used in tests / re-ingestion)."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the store is reachable and schema is ready."""

    # ── Write ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity node (idempotent by entity_id)."""

    @abstractmethod
    def upsert_relation(self, relation: Relation) -> None:
        """Insert or increment a relation edge (idempotent: accumulates weight)."""

    @abstractmethod
    def link_entity_to_chunk(
        self,
        entity_id: str,
        chunk_id: str,
        chunk_text: str,
        doc_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Associate an entity node with the chunk where it was found.

        Chunk text is stored in the graph so traversal results are self-contained
        (no round-trip to the vector store needed for graph-only results).
        """

    # ── Read ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Fetch a single entity node by ID."""

    @abstractmethod
    def find_entities_by_text(self, text: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Fuzzy/exact search for entity nodes by surface text."""

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 2,
    ) -> List[Tuple[Entity, int]]:
        """Return (entity, hop_distance) tuples reachable within max_hops.

        hop_distance=1 means direct neighbour; hop_distance=2 means 2 edges away.
        Implementations should deduplicate and sort by hop_distance ascending.
        """

    @abstractmethod
    def get_chunk_ids_for_entity(self, entity_id: str) -> List[Tuple[str, str]]:
        """Return [(chunk_id, chunk_text), ...] for all chunks mentioning this entity."""

    @abstractmethod
    def entity_count(self) -> int:
        """Number of entity nodes in the graph."""

    @abstractmethod
    def relation_count(self) -> int:
        """Number of relation edges in the graph."""

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
