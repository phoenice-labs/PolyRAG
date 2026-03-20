"""
Phase 10: Knowledge Graph data models.
Entity nodes and relation edges — the building blocks of GraphRAG.
Designed for portability: same model works across NetworkX, Kuzu, and Neo4j.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Entity ─────────────────────────────────────────────────────────────────────

# Simplified type map: spaCy label → canonical type
SPACY_TYPE_MAP: Dict[str, str] = {
    "PERSON":       "PERSON",
    "ORG":          "ORG",
    "GPE":          "LOCATION",     # geopolitical entity
    "LOC":          "LOCATION",
    "FAC":          "LOCATION",     # facility
    "DATE":         "DATE",
    "TIME":         "DATE",
    "EVENT":        "EVENT",
    "WORK_OF_ART":  "CONCEPT",
    "LAW":          "LAW",
    "PRODUCT":      "PRODUCT",
    "MONEY":        "CONCEPT",
    "NORP":         "ORG",          # nationalities, religions, political groups
    "LANGUAGE":     "CONCEPT",
    "PERCENT":      "CONCEPT",
    "QUANTITY":     "CONCEPT",
    "ORDINAL":      "CONCEPT",
    "CARDINAL":     "CONCEPT",
}


def make_entity_id(entity_type: str, text: str) -> str:
    """Deterministic, normalised entity ID — same entity across chunks maps to same node."""
    normalised = text.lower().strip().replace(" ", "_")
    return f"{entity_type.upper()}:{normalised}"


class Entity(BaseModel):
    """A node in the Knowledge Graph."""
    entity_id: str                     # e.g. "PERSON:john_doe"
    text: str                          # canonical surface form
    entity_type: str                   # PERSON | ORG | LOCATION | DATE | EVENT | LAW | PRODUCT | CONCEPT
    aliases: List[str] = Field(default_factory=list)   # alternate surface forms
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_spacy(cls, text: str, label: str) -> "Entity":
        entity_type = SPACY_TYPE_MAP.get(label, "CONCEPT")
        return cls(
            entity_id=make_entity_id(entity_type, text),
            text=text,
            entity_type=entity_type,
        )


class Relation(BaseModel):
    """A directed edge in the Knowledge Graph."""
    source_id: str
    target_id: str
    relation_type: str   # e.g. "governs", "references", "co_occurs", "works_for"
    weight: float = 1.0  # higher = stronger / more frequently observed
    chunk_id: Optional[str] = None   # chunk where this relation was extracted


class Triple(BaseModel):
    """Subject-Predicate-Object triple — the atomic unit of knowledge."""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float = 1.0
    chunk_id: str = ""

    def to_relation(self) -> Relation:
        return Relation(
            source_id=self.subject.entity_id,
            target_id=self.object.entity_id,
            relation_type=self.predicate,
            weight=self.confidence,
            chunk_id=self.chunk_id,
        )


class ExtractionResult(BaseModel):
    """Output of EntityRelationExtractor for one chunk."""
    chunk_id: str
    entities: List[Entity] = Field(default_factory=list)
    triples: List[Triple] = Field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        self.entity_count = len(self.entities)
        self.relation_count = len(self.triples)


class GraphPath(BaseModel):
    """A traversal path found during query-time graph lookup."""
    query_entity: str          # the entity detected in the query
    path_entities: List[str]   # entity texts along the path
    path_types: List[str]      # relation types along the path
    chunk_ids: List[str]       # all chunks associated with path entities
    hop_distance: int          # 0 = direct, 1 = 1 hop, 2 = 2 hops
    relevance_score: float     # 1.0 / (1 + hop_distance)

    @property
    def explanation(self) -> str:
        if not self.path_entities:
            return self.query_entity
        parts = [self.query_entity]
        for rel, ent in zip(self.path_types, self.path_entities):
            parts.append(f"—[{rel}]→ {ent}")
        return " ".join(parts)
