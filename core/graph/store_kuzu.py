"""
KuzuGraphStore — embedded persistent Knowledge Graph using Kuzu.

Kuzu is an embedded property graph database (like SQLite for graphs).
  - No server required, runs in-process
  - ACID transactions, persistent on-disk storage
  - Uses Cypher query language (same as Neo4j)

Porting to Neo4j:
  - Replace `import kuzu` with `from neo4j import GraphDatabase`
  - Replace `kuzu.Database(path)` with `GraphDatabase.driver(uri, auth=...)`
  - Replace `conn.execute(cypher, params)` with `session.run(cypher, params)`
  - Cypher queries remain IDENTICAL — no orchestration changes needed

Install:
  pip install kuzu
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.graph.base import GraphStoreBase
from core.graph.models import Entity, Relation

logger = logging.getLogger(__name__)


class KuzuGraphStore(GraphStoreBase):
    """
    Production-grade, embedded, persistent Knowledge Graph.

    Schema
    ------
    NODE TABLE Entity  (id STRING PK, text STRING, entity_type STRING)
    NODE TABLE Chunk   (id STRING PK, text STRING, doc_id STRING)
    REL  TABLE APPEARS_IN   Entity → Chunk   (confidence FLOAT)
    REL  TABLE RELATES_TO   Entity → Entity  (relation_type STRING, weight DOUBLE)
    """

    def __init__(self, db_path: str = "./data/graph.kuzu") -> None:
        self.db_path = str(Path(db_path).resolve())
        self._db = None
        self._conn = None
        self._entity_cache: Dict[str, Entity] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            import kuzu
        except ImportError as e:
            raise ImportError("Install kuzu: pip install kuzu") from e

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(self.db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        self._rebuild_cache()
        logger.info("KuzuGraphStore connected: %s", self.db_path)

    def close(self) -> None:
        self._conn = None
        self._db = None

    def clear(self) -> None:
        if self._conn is None:
            return
        # Drop all data; tables are recreated on next connect
        for stmt in [
            "MATCH (e:Entity)-[r:APPEARS_IN]->(c:Chunk) DELETE r",
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) DELETE r",
            "MATCH (e:Entity) DELETE e",
            "MATCH (c:Chunk) DELETE c",
        ]:
            try:
                self._conn.execute(stmt)
            except Exception:
                pass
        self._entity_cache.clear()

    def health_check(self) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute("MATCH (e:Entity) RETURN count(e) LIMIT 1")
            return True
        except Exception:
            return False

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        stmts = [
            """CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING,
                text STRING,
                entity_type STRING,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Chunk(
                id STRING,
                text STRING,
                doc_id STRING,
                PRIMARY KEY(id)
            )""",
            """CREATE REL TABLE IF NOT EXISTS APPEARS_IN(
                FROM Entity TO Chunk,
                confidence FLOAT
            )""",
            """CREATE REL TABLE IF NOT EXISTS RELATES_TO(
                FROM Entity TO Entity,
                relation_type STRING,
                weight DOUBLE
            )""",
        ]
        for stmt in stmts:
            self._conn.execute(stmt)

    def _rebuild_cache(self) -> None:
        """Rebuild in-memory entity cache from persisted data."""
        try:
            result = self._conn.execute("MATCH (e:Entity) RETURN e.id, e.text, e.entity_type")
            while result.has_next():
                row = result.get_next()
                entity = Entity(entity_id=row[0], text=row[1], entity_type=row[2])
                self._entity_cache[entity.entity_id] = entity
        except Exception as exc:
            logger.warning("Could not rebuild entity cache: %s", exc)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity) -> None:
        if entity.entity_id in self._entity_cache:
            return  # already persisted
        self._conn.execute(
            "MERGE (e:Entity {id: $id}) "
            "ON CREATE SET e.text = $text, e.entity_type = $etype",
            {"id": entity.entity_id, "text": entity.text, "etype": entity.entity_type},
        )
        self._entity_cache[entity.entity_id] = entity

    def upsert_relation(self, relation: Relation) -> None:
        # Check if edge exists
        result = self._conn.execute(
            "MATCH (a:Entity {id: $src})-[r:RELATES_TO]->(b:Entity {id: $tgt}) "
            "WHERE r.relation_type = $rtype RETURN count(r)",
            {"src": relation.source_id, "tgt": relation.target_id, "rtype": relation.relation_type},
        )
        count = result.get_next()[0]
        if count == 0:
            self._conn.execute(
                "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                "CREATE (a)-[:RELATES_TO {relation_type: $rtype, weight: $w}]->(b)",
                {
                    "src": relation.source_id,
                    "tgt": relation.target_id,
                    "rtype": relation.relation_type,
                    "w": relation.weight,
                },
            )

    def link_entity_to_chunk(
        self,
        entity_id: str,
        chunk_id: str,
        chunk_text: str,
        doc_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        # Ensure Chunk node exists
        result = self._conn.execute(
            "MATCH (c:Chunk {id: $id}) RETURN count(c)",
            {"id": chunk_id},
        )
        if result.get_next()[0] == 0:
            self._conn.execute(
                "CREATE (:Chunk {id: $id, text: $text, doc_id: $doc_id})",
                {"id": chunk_id, "text": chunk_text[:2000], "doc_id": doc_id},
            )

        # Ensure APPEARS_IN relation exists (one per entity-chunk pair)
        result = self._conn.execute(
            "MATCH (e:Entity {id: $eid})-[r:APPEARS_IN]->(c:Chunk {id: $cid}) RETURN count(r)",
            {"eid": entity_id, "cid": chunk_id},
        )
        if result.get_next()[0] == 0:
            self._conn.execute(
                "MATCH (e:Entity {id: $eid}), (c:Chunk {id: $cid}) "
                "CREATE (e)-[:APPEARS_IN {confidence: $conf}]->(c)",
                {"eid": entity_id, "cid": chunk_id, "conf": confidence},
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        entity = self._entity_cache.get(entity_id)
        if entity:
            return entity
        result = self._conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN e.id, e.text, e.entity_type",
            {"id": entity_id},
        )
        if result.has_next():
            row = result.get_next()
            return Entity(entity_id=row[0], text=row[1], entity_type=row[2])
        return None

    def find_entities_by_text(self, text: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Case-insensitive substring match on entity text."""
        # Filter from cache for speed; cache is always full after connect
        text_lower = text.lower()
        return [
            e for e in self._entity_cache.values()
            if text_lower in e.text.lower()
            and (entity_type is None or e.entity_type == entity_type)
        ]

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> List[Tuple[Entity, int]]:
        """1-hop and 2-hop neighbours via RELATES_TO edges."""
        results: Dict[str, Tuple[Entity, int]] = {}  # entity_id → (entity, min_hop)

        # 1-hop
        r1 = self._conn.execute(
            "MATCH (a:Entity {id: $id})-[:RELATES_TO]->(b:Entity) "
            "RETURN b.id, b.text, b.entity_type",
            {"id": entity_id},
        )
        while r1.has_next():
            row = r1.get_next()
            eid = row[0]
            if eid != entity_id and eid not in results:
                results[eid] = (Entity(entity_id=eid, text=row[1], entity_type=row[2]), 1)

        # 2-hop
        if max_hops >= 2:
            r2 = self._conn.execute(
                "MATCH (a:Entity {id: $id})-[:RELATES_TO]->(m:Entity)-[:RELATES_TO]->(b:Entity) "
                "RETURN b.id, b.text, b.entity_type",
                {"id": entity_id},
            )
            while r2.has_next():
                row = r2.get_next()
                eid = row[0]
                if eid != entity_id and eid not in results:
                    results[eid] = (Entity(entity_id=eid, text=row[1], entity_type=row[2]), 2)

        return sorted(results.values(), key=lambda x: x[1])

    def get_chunk_ids_for_entity(self, entity_id: str) -> List[Tuple[str, str]]:
        result = self._conn.execute(
            "MATCH (e:Entity {id: $id})-[:APPEARS_IN]->(c:Chunk) RETURN c.id, c.text",
            {"id": entity_id},
        )
        chunks = []
        while result.has_next():
            row = result.get_next()
            chunks.append((row[0], row[1]))
        return chunks

    def entity_count(self) -> int:
        return len(self._entity_cache)

    def relation_count(self) -> int:
        result = self._conn.execute(
            "MATCH ()-[r:RELATES_TO]->() RETURN count(r)"
        )
        return result.get_next()[0] if result.has_next() else 0
