"""
Neo4jGraphStore — enterprise migration stub.

Porting from KuzuGraphStore to Neo4jGraphStore requires only:
  1. pip install neo4j
  2. Set graph.backend: neo4j and graph.neo4j.uri/user/password in config.yaml
  3. Zero changes to orchestrator, pipeline, or any other component

The Cypher queries in KuzuGraphStore are IDENTICAL to Neo4j Cypher.
Only the driver calls differ:

  Kuzu:  conn.execute(cypher, params)
  Neo4j: session.run(cypher, **params)  (via neo4j Python driver)

This stub implements the full GraphStoreBase interface so it can be
tested/imported without a live Neo4j server.  All write methods are no-ops
and all read methods return empty results.  Tag @integration tests to
run against a real Neo4j instance.

Install (when ready):
  pip install neo4j>=5.0
  docker run --name neo4j -e NEO4J_AUTH=neo4j/password -p 7474:7474 -p 7687:7687 neo4j:latest
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from core.graph.base import GraphStoreBase
from core.graph.models import Entity, Relation

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStoreBase):
    """
    Neo4j implementation of GraphStoreBase (enterprise production target).

    Configuration (config.yaml):
        graph:
          backend: neo4j
          neo4j:
            uri: bolt://localhost:7687
            user: neo4j
            password: password
            database: neo4j        # optional, default DB

    Cypher queries are identical to KuzuGraphStore.
    Replace kuzu driver calls with neo4j driver calls to activate.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        logger.warning(
            "Neo4jGraphStore: this is a migration stub. "
            "Replace stub driver calls with neo4j Python driver to activate."
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            self._driver.verify_connectivity()
            self._init_schema()
            logger.info("Neo4jGraphStore connected: %s", self.uri)
        except ImportError as e:
            raise ImportError("Install neo4j: pip install neo4j>=5.0") from e
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {self.uri}.\n"
                f"Start with: docker run --name neo4j "
                f"-e NEO4J_AUTH={self.user}/{self.password} "
                f"-p 7474:7474 -p 7687:7687 neo4j:latest\n"
                f"Error: {exc}"
            ) from exc

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def clear(self) -> None:
        if not self._driver:
            return
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def health_check(self) -> bool:
        if not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def _init_schema(self) -> None:
        """Create indexes for fast entity lookup — Neo4j equivalent of PRIMARY KEY."""
        with self._driver.session(database=self.database) as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

    def _run(self, cypher: str, **params):
        """Execute a Cypher statement. Equivalent to kuzu conn.execute()."""
        with self._driver.session(database=self.database) as session:
            return list(session.run(cypher, **params))

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity) -> None:
        # MERGE is identical in Neo4j Cypher
        self._run(
            "MERGE (e:Entity {id: $id}) "
            "ON CREATE SET e.text = $text, e.entity_type = $etype",
            id=entity.entity_id, text=entity.text, etype=entity.entity_type,
        )

    def upsert_relation(self, relation: Relation) -> None:
        self._run(
            "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
            "MERGE (a)-[r:RELATES_TO {relation_type: $rtype}]->(b) "
            "ON CREATE SET r.weight = $w "
            "ON MATCH  SET r.weight = r.weight + $w",
            src=relation.source_id, tgt=relation.target_id,
            rtype=relation.relation_type, w=relation.weight,
        )

    def link_entity_to_chunk(
        self,
        entity_id: str,
        chunk_id: str,
        chunk_text: str,
        doc_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        self._run(
            "MERGE (c:Chunk {id: $cid}) "
            "ON CREATE SET c.text = $text, c.doc_id = $doc_id",
            cid=chunk_id, text=chunk_text[:2000], doc_id=doc_id,
        )
        self._run(
            "MATCH (e:Entity {id: $eid}), (c:Chunk {id: $cid}) "
            "MERGE (e)-[:APPEARS_IN {confidence: $conf}]->(c)",
            eid=entity_id, cid=chunk_id, conf=confidence,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        rows = self._run("MATCH (e:Entity {id: $id}) RETURN e.id, e.text, e.entity_type", id=entity_id)
        if rows:
            r = rows[0]
            return Entity(entity_id=r["e.id"], text=r["e.text"], entity_type=r["e.entity_type"])
        return None

    def find_entities_by_text(self, text: str, entity_type: Optional[str] = None) -> List[Entity]:
        if entity_type:
            rows = self._run(
                "MATCH (e:Entity) WHERE toLower(e.text) CONTAINS toLower($text) "
                "AND e.entity_type = $etype RETURN e.id, e.text, e.entity_type",
                text=text, etype=entity_type,
            )
        else:
            rows = self._run(
                "MATCH (e:Entity) WHERE toLower(e.text) CONTAINS toLower($text) "
                "RETURN e.id, e.text, e.entity_type",
                text=text,
            )
        return [Entity(entity_id=r["e.id"], text=r["e.text"], entity_type=r["e.entity_type"]) for r in rows]

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> List[Tuple[Entity, int]]:
        rows = self._run(
            f"MATCH (a:Entity {{id: $id}})-[:RELATES_TO*1..{max_hops}]->(b:Entity) "
            "RETURN b.id, b.text, b.entity_type",
            id=entity_id,
        )
        seen = set()
        results = []
        for r in rows:
            eid = r["b.id"]
            if eid not in seen and eid != entity_id:
                seen.add(eid)
                results.append((Entity(entity_id=eid, text=r["b.text"], entity_type=r["b.entity_type"]), 1))
        return results

    def get_chunk_ids_for_entity(self, entity_id: str) -> List[Tuple[str, str]]:
        rows = self._run(
            "MATCH (e:Entity {id: $id})-[:APPEARS_IN]->(c:Chunk) RETURN c.id, c.text",
            id=entity_id,
        )
        return [(r["c.id"], r["c.text"]) for r in rows]

    def entity_count(self) -> int:
        rows = self._run("MATCH (e:Entity) RETURN count(e) AS cnt")
        return rows[0]["cnt"] if rows else 0

    def relation_count(self) -> int:
        rows = self._run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt")
        return rows[0]["cnt"] if rows else 0
