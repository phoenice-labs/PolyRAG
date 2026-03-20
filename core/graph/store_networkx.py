"""
NetworkXGraphStore — in-memory Knowledge Graph backed by networkx.

Used for:
  - Unit tests (fast, no file system)
  - Fallback when kuzu is not installed
  - Development / prototyping

Note: data is NOT persistent across process restarts.
      For production use, switch to KuzuGraphStore.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.graph.base import GraphStoreBase
from core.graph.models import Entity, GraphPath, Relation


class NetworkXGraphStore(GraphStoreBase):
    """Pure-Python in-memory graph using networkx.

    Node types  : 'entity' (Entity model) and 'chunk' (text snippet)
    Edge types  : 'appears_in' (entity→chunk) and 'relates_to' (entity→entity)
    """

    def __init__(self) -> None:
        self._G = None              # networkx MultiDiGraph
        self._entities: Dict[str, Entity] = {}
        self._chunks: Dict[str, Tuple[str, str]] = {}   # chunk_id → (text, doc_id)
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("Install networkx: pip install networkx") from e
        self._G = nx.MultiDiGraph()
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def clear(self) -> None:
        if self._G is not None:
            self._G.clear()
        self._entities.clear()
        self._chunks.clear()

    def health_check(self) -> bool:
        return self._connected and self._G is not None

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity
        if not self._G.has_node(entity.entity_id):
            self._G.add_node(entity.entity_id, node_type="entity", entity=entity)

    def upsert_relation(self, relation: Relation) -> None:
        # Accumulate weight on repeated calls
        for _, _, data in self._G.out_edges(relation.source_id, data=True):
            if (data.get("edge_type") == "relates_to"
                    and data.get("relation_type") == relation.relation_type
                    and _ == relation.target_id):
                data["weight"] = data.get("weight", 1.0) + relation.weight
                return
        self._G.add_edge(
            relation.source_id,
            relation.target_id,
            edge_type="relates_to",
            relation_type=relation.relation_type,
            weight=relation.weight,
        )

    def link_entity_to_chunk(
        self,
        entity_id: str,
        chunk_id: str,
        chunk_text: str,
        doc_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        chunk_node = f"chunk:{chunk_id}"
        if not self._G.has_node(chunk_node):
            self._G.add_node(chunk_node, node_type="chunk", chunk_id=chunk_id, text=chunk_text, doc_id=doc_id)
            self._chunks[chunk_id] = (chunk_text, doc_id)
        # Only one APPEARS_IN edge per (entity, chunk) pair
        if not any(
            data.get("edge_type") == "appears_in"
            for _, v, data in self._G.out_edges(entity_id, data=True)
            if v == chunk_node
        ):
            self._G.add_edge(entity_id, chunk_node, edge_type="appears_in", confidence=confidence)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def find_entities_by_text(self, text: str, entity_type: Optional[str] = None) -> List[Entity]:
        text_lower = text.lower()
        results = [
            e for e in self._entities.values()
            if text_lower in e.text.lower()
            and (entity_type is None or e.entity_type == entity_type)
        ]
        return results

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> List[Tuple[Entity, int]]:
        """BFS over entity→entity 'relates_to' edges up to max_hops."""
        visited: Dict[str, int] = {entity_id: 0}  # node → hop distance
        queue = [(entity_id, 0)]
        results: List[Tuple[Entity, int]] = []

        while queue:
            current_id, hop = queue.pop(0)
            if hop >= max_hops:
                continue
            for _, neighbor, data in self._G.out_edges(current_id, data=True):
                if data.get("edge_type") != "relates_to":
                    continue
                if neighbor.startswith("chunk:"):
                    continue
                if neighbor not in visited:
                    visited[neighbor] = hop + 1
                    queue.append((neighbor, hop + 1))
                    entity = self._entities.get(neighbor)
                    if entity:
                        results.append((entity, hop + 1))

        return sorted(results, key=lambda x: x[1])

    def get_chunk_ids_for_entity(self, entity_id: str) -> List[Tuple[str, str]]:
        """Return [(chunk_id, chunk_text), ...] for all chunks this entity appears in."""
        results = []
        for _, chunk_node, data in self._G.out_edges(entity_id, data=True):
            if data.get("edge_type") == "appears_in":
                node_data = self._G.nodes[chunk_node]
                cid = node_data.get("chunk_id", "")
                txt = node_data.get("text", "")
                if cid:
                    results.append((cid, txt))
        return results

    def entity_count(self) -> int:
        # Count entity nodes in G to include nodes added via upsert_entity() AND
        # those added directly to _G (e.g., by the spaCy EntityRelationExtractor).
        if self._G is not None:
            return sum(1 for _, d in self._G.nodes(data=True) if d.get("node_type") == "entity")
        return len(self._entities)

    def relation_count(self) -> int:
        return sum(
            1 for _, _, data in self._G.edges(data=True)
            if data.get("edge_type") == "relates_to"
        )
