"""
GraphTraverser — query-time Knowledge Graph traversal.

At query time:
  1. Extract entities from the user's question (NER)
  2. For each entity, find it in the Knowledge Graph
  3. Expand N hops along RELATES_TO edges
  4. Collect all Chunk nodes reachable from those entities
  5. Return ranked (chunk_id, chunk_text, score) tuples for 3-way RRF fusion

Relevance scoring:
  hop 0 (direct mention)  → score = 1.0
  hop 1                   → score = 0.6
  hop 2                   → score = 0.3
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from core.graph.base import GraphStoreBase
from core.graph.extractor import EntityRelationExtractor
from core.graph.models import Entity, GraphPath
from core.store.models import Document, SearchResult

logger = logging.getLogger(__name__)

_HOP_SCORES = {0: 1.0, 1: 0.6, 2: 0.3}


class GraphTraverser:
    """
    Query-time graph traversal producing SearchResult candidates.

    Parameters
    ----------
    graph_store  : any GraphStoreBase implementation
    extractor    : EntityRelationExtractor for NER on the query
    max_hops     : maximum edge hops from query entities (default: 2)
    min_score    : minimum relevance score to include a chunk (default: 0.0)
    llm_client   : optional LLM client for richer query entity extraction
    """

    def __init__(
        self,
        graph_store: GraphStoreBase,
        extractor: EntityRelationExtractor,
        max_hops: int = 2,
        min_score: float = 0.0,
        llm_client=None,
    ) -> None:
        self.graph = graph_store
        self.extractor = extractor
        self.max_hops = max_hops
        self.min_score = min_score
        self._llm_client = llm_client

    def traverse(
        self,
        query: str,
        top_k: int = 20,
        use_llm_graph: bool = False,
    ) -> Tuple[List[SearchResult], List[GraphPath]]:
        """
        Traverse the graph from query entities.

        Returns
        -------
        results : List[SearchResult] — chunks found via graph traversal (with graph relevance scores)
        paths   : List[GraphPath]   — human-readable traversal paths for explainability
        """
        if not self.graph.health_check():
            return [], []

        # Step 1: detect entities in the query — LLM path first, spaCy as fallback
        query_entities = []
        llm_used = False

        if use_llm_graph and self._llm_client:
            try:
                query_entities = self._detect_query_entities_llm(query)
                if query_entities:
                    llm_used = True
                    logger.debug(
                        "GraphTraverser: LLM extracted %d entities: %s",
                        len(query_entities),
                        [e.text for e in query_entities],
                    )
            except Exception as exc:
                logger.debug("GraphTraverser: LLM entity extraction failed, falling back to spaCy: %s", exc)

        if not query_entities:
            query_entities = self._detect_query_entities(query)

        if not query_entities:
            logger.debug("GraphTraverser: no NER entities in query, trying keyword fallback")
            query_entities = self._keyword_entity_lookup(query)
        if not query_entities:
            logger.debug("GraphTraverser: no entities detected in query '%s'", query[:80])
            return [], []

        logger.debug(
            "GraphTraverser: detected %d entities (llm=%s): %s",
            len(query_entities),
            llm_used,
            [e.text for e in query_entities],
        )

        # Step 2: collect chunks via graph traversal
        chunk_scores: Dict[str, float] = {}   # chunk_id → best relevance score
        chunk_texts:  Dict[str, str] = {}     # chunk_id → text
        paths: List[GraphPath] = []

        for q_entity in query_entities:
            # Resolve entity in graph (handle partial text matches)
            graph_entities = self._resolve_entity(q_entity)
            if not graph_entities:
                continue

            for g_entity in graph_entities:
                # Direct chunks (hop 0)
                direct_chunks = self.graph.get_chunk_ids_for_entity(g_entity.entity_id)
                for cid, ctext in direct_chunks:
                    score = _HOP_SCORES[0]
                    if score > chunk_scores.get(cid, -1):
                        chunk_scores[cid] = score
                        chunk_texts[cid] = ctext
                paths.append(GraphPath(
                    query_entity=q_entity.text,
                    path_entities=[],
                    path_types=[],
                    chunk_ids=[cid for cid, _ in direct_chunks],
                    hop_distance=0,
                    relevance_score=_HOP_SCORES[0],
                ))

                # Neighbour chunks (hops 1..N)
                neighbours = self.graph.get_neighbors(g_entity.entity_id, self.max_hops)
                for neighbour, hop in neighbours:
                    hop_score = _HOP_SCORES.get(hop, 0.1)
                    if hop_score < self.min_score:
                        continue
                    neighbour_chunks = self.graph.get_chunk_ids_for_entity(neighbour.entity_id)
                    for cid, ctext in neighbour_chunks:
                        if hop_score > chunk_scores.get(cid, -1):
                            chunk_scores[cid] = hop_score
                            chunk_texts[cid] = ctext
                    if neighbour_chunks:
                        paths.append(GraphPath(
                            query_entity=q_entity.text,
                            path_entities=[neighbour.text],
                            path_types=["relates_to"],
                            chunk_ids=[cid for cid, _ in neighbour_chunks],
                            hop_distance=hop,
                            relevance_score=hop_score,
                        ))

        if not chunk_scores:
            return [], paths

        # Step 3: convert to SearchResult list, sorted by score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[SearchResult] = []
        for rank, (cid, score) in enumerate(sorted_chunks, start=1):
            doc = Document(
                id=cid,
                text=chunk_texts.get(cid, ""),
                embedding=[],
                metadata={"source": "graph_traversal", "graph_score": score},
            )
            results.append(SearchResult(document=doc, score=score, rank=rank))

        return results, paths

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_query_entities(self, query: str) -> List[Entity]:
        """Extract entities from the query text using spaCy NER + noun chunks."""
        if not self.extractor.is_available():
            return []
        return self.extractor.extract_entities_only(query)

    def _keyword_entity_lookup(self, query: str) -> List[Entity]:
        """Keyword + lemma fallback: find graph entities matching content words in the query.

        This fires when NER + noun chunk extraction both return nothing — e.g. when the
        query is a pure verb phrase.  Tries both raw tokens and their lemmas.
        """
        if not self.extractor.is_available():
            # Simple word split fallback when spaCy unavailable
            found: List[Entity] = []
            for word in query.split():
                w = word.lower().strip(".,;:!?\"'")
                if len(w) > 3:
                    for ent in self.graph.find_entities_by_text(w):
                        found.append(ent)
            return found

        try:
            self.extractor._load()
            doc = self.extractor._nlp(query[:2000])
        except Exception:
            return []

        seen_ids: set = set()
        found: List[Entity] = []

        candidates = set()
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if len(token.text) > 3:
                candidates.add(token.text.lower())
            # Also try the lemma — catches "importing" → "import", "lands" → "land"
            if len(token.lemma_) > 3:
                candidates.add(token.lemma_.lower())

        for word in candidates:
            for ent in self.graph.find_entities_by_text(word):
                if ent.entity_id not in seen_ids:
                    seen_ids.add(ent.entity_id)
                    found.append(ent)

        return found

    def _detect_query_entities_llm(self, query: str) -> List[Entity]:
        """Extract entities from the query using the LLM for richer semantic coverage.

        The LLM understands verbs, concepts, and implicit relationships that spaCy
        NER misses (e.g. "divide" → CONCEPT:division, "Sunday" + "week" → relation).
        Falls back gracefully: returns [] so caller can use spaCy instead.
        """
        from core.graph.llm_extractor import LLMEntityExtractor
        from core.graph.models import make_entity_id

        extractor = LLMEntityExtractor(
            llm_client=self._llm_client,
            max_chunk_chars=500,   # queries are short — no need for large window
        )
        result = extractor.extract(query, chunk_id="__query__")
        return result.entities

    def _resolve_entity(self, query_entity: Entity) -> List[Entity]:
        """Find graph entities matching the query entity (exact first, then substring)."""
        # Exact match by entity_id
        exact = self.graph.get_entity(query_entity.entity_id)
        if exact:
            return [exact]

        # Substring text match (handles casing, abbreviations)
        matches = self.graph.find_entities_by_text(
            query_entity.text,
            entity_type=query_entity.entity_type,
        )
        # Also try without type constraint for broader recall
        if not matches:
            matches = self.graph.find_entities_by_text(query_entity.text)
        return matches[:5]  # cap to top 5 fuzzy matches
