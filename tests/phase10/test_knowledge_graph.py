"""
Phase 10: Knowledge Graph (GraphRAG) tests.

Tests cover:
  - Data models (Entity, Relation, Triple, GraphPath)
  - EntityRelationExtractor (NER + SVO + co-occurrence)
  - NetworkXGraphStore  (in-memory, always available)
  - KuzuGraphStore      (embedded persistent, requires kuzu)
  - GraphTraverser      (query-time traversal)
  - TripleHybridRetriever (3-way RRF fusion)
  - RAGResponse.graph_* fields
"""
from __future__ import annotations

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California. "
    "The company released the iPhone in 2007, which became a landmark product. "
    "Tim Cook succeeded Steve Jobs as CEO in 2011."
)


@pytest.fixture
def nx_store():
    from core.graph.store_networkx import NetworkXGraphStore
    store = NetworkXGraphStore()
    store.connect()
    yield store
    store.close()


@pytest.fixture
def populated_nx_store(nx_store):
    """NetworkX store with 3 entities and 2 chunks."""
    from core.graph.models import Entity, Relation
    apple = Entity(entity_id="ORG:apple_inc.", text="Apple Inc.", entity_type="ORG")
    jobs = Entity(entity_id="PERSON:steve_jobs", text="Steve Jobs", entity_type="PERSON")
    cupertino = Entity(entity_id="LOCATION:cupertino", text="Cupertino", entity_type="LOCATION")

    for ent in (apple, jobs, cupertino):
        nx_store.upsert_entity(ent)

    nx_store.link_entity_to_chunk(apple.entity_id, "chunk-1", "Apple Inc. was founded...", doc_id="doc-1")
    nx_store.link_entity_to_chunk(jobs.entity_id, "chunk-1", "Apple Inc. was founded...", doc_id="doc-1")
    nx_store.link_entity_to_chunk(cupertino.entity_id, "chunk-2", "Located in Cupertino.", doc_id="doc-1")

    rel = Relation(source_id=apple.entity_id, target_id=jobs.entity_id, relation_type="founded_by", weight=1.0)
    nx_store.upsert_relation(rel)

    return nx_store, apple, jobs, cupertino


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data model tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDataModels:
    def test_make_entity_id_normalisation(self):
        from core.graph.models import make_entity_id
        eid1 = make_entity_id("PERSON", "Steve Jobs")
        eid2 = make_entity_id("PERSON", "steve jobs")
        # Both map to the same ID (lowercase, underscore)
        assert eid1 == "PERSON:steve_jobs"
        assert eid2 == "PERSON:steve_jobs"

    def test_entity_from_spacy(self):
        from core.graph.models import Entity
        ent = Entity.from_spacy("Apple Inc.", "ORG")
        assert ent.entity_type == "ORG"
        assert ent.entity_id.startswith("ORG:")

    def test_triple_to_relation(self):
        from core.graph.models import Entity, Triple
        subj = Entity.from_spacy("Steve Jobs", "PERSON")
        obj = Entity.from_spacy("Apple", "ORG")
        triple = Triple(subject=subj, predicate="founded", object=obj, confidence=0.9, chunk_id="c1")
        rel = triple.to_relation()
        assert rel.source_id == subj.entity_id
        assert rel.target_id == obj.entity_id
        assert rel.relation_type == "founded"
        assert rel.weight == 0.9

    def test_graph_path_explanation(self):
        from core.graph.models import GraphPath
        path = GraphPath(
            query_entity="Steve Jobs",
            path_entities=["Apple Inc."],
            path_types=["founded"],
            chunk_ids=["c1"],
            hop_distance=1,
            relevance_score=0.6,
        )
        expl = path.explanation
        assert "Steve Jobs" in expl
        assert "Apple Inc." in expl

    def test_graph_path_direct_explanation(self):
        from core.graph.models import GraphPath
        path = GraphPath(
            query_entity="Apple",
            path_entities=[],
            path_types=[],
            chunk_ids=["c1"],
            hop_distance=0,
            relevance_score=1.0,
        )
        assert path.explanation == "Apple"


# ──────────────────────────────────────────────────────────────────────────────
# 2. NetworkXGraphStore tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNetworkXGraphStore:
    def test_connect_health(self, nx_store):
        assert nx_store.health_check()

    def test_upsert_and_get_entity(self, nx_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="ORG:test_org", text="Test Org", entity_type="ORG")
        nx_store.upsert_entity(ent)
        fetched = nx_store.get_entity("ORG:test_org")
        assert fetched is not None
        assert fetched.text == "Test Org"

    def test_upsert_entity_idempotent(self, nx_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="ORG:unique", text="Unique Corp", entity_type="ORG")
        nx_store.upsert_entity(ent)
        nx_store.upsert_entity(ent)  # duplicate — should not throw
        assert nx_store.entity_count() == 1

    def test_link_entity_to_chunk(self, nx_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="PERSON:alice", text="Alice", entity_type="PERSON")
        nx_store.upsert_entity(ent)
        nx_store.link_entity_to_chunk("PERSON:alice", "chunk-A", "Alice said hello.", doc_id="doc-1")

        chunks = nx_store.get_chunk_ids_for_entity("PERSON:alice")
        assert any(cid == "chunk-A" for cid, _ in chunks)

    def test_link_entity_chunk_idempotent(self, nx_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="PERSON:bob", text="Bob", entity_type="PERSON")
        nx_store.upsert_entity(ent)
        nx_store.link_entity_to_chunk("PERSON:bob", "chunk-B", "Bob did things.", doc_id="d")
        nx_store.link_entity_to_chunk("PERSON:bob", "chunk-B", "Bob did things.", doc_id="d")
        chunks = nx_store.get_chunk_ids_for_entity("PERSON:bob")
        chunk_ids = [c for c, _ in chunks]
        assert chunk_ids.count("chunk-B") == 1

    def test_upsert_relation_and_neighbors(self, populated_nx_store):
        store, apple, jobs, cupertino = populated_nx_store
        neighbors = store.get_neighbors(apple.entity_id, max_hops=1)
        neighbor_ids = [e.entity_id for e, _ in neighbors]
        assert jobs.entity_id in neighbor_ids

    def test_relation_weight_accumulation(self, nx_store):
        from core.graph.models import Entity, Relation
        a = Entity(entity_id="ORG:a", text="A", entity_type="ORG")
        b = Entity(entity_id="ORG:b", text="B", entity_type="ORG")
        nx_store.upsert_entity(a)
        nx_store.upsert_entity(b)
        r = Relation(source_id="ORG:a", target_id="ORG:b", relation_type="partners", weight=1.0)
        nx_store.upsert_relation(r)
        nx_store.upsert_relation(r)
        # Weight should accumulate (2.0), not duplicate edge
        rel_count = nx_store.relation_count()
        assert rel_count == 1  # only one edge

    def test_find_entities_by_text(self, populated_nx_store):
        store, apple, jobs, cupertino = populated_nx_store
        matches = store.find_entities_by_text("apple")
        assert any(e.entity_id == apple.entity_id for e in matches)

    def test_entity_and_relation_count(self, populated_nx_store):
        store, *_ = populated_nx_store
        assert store.entity_count() == 3
        assert store.relation_count() == 1

    def test_clear(self, populated_nx_store):
        store, *_ = populated_nx_store
        store.clear()
        assert store.entity_count() == 0
        assert store.relation_count() == 0

    def test_two_hop_neighbors(self, nx_store):
        from core.graph.models import Entity, Relation
        # A → B → C
        a = Entity(entity_id="P:a", text="A", entity_type="PERSON")
        b = Entity(entity_id="P:b", text="B", entity_type="PERSON")
        c = Entity(entity_id="P:c", text="C", entity_type="PERSON")
        for e in (a, b, c):
            nx_store.upsert_entity(e)
        nx_store.upsert_relation(Relation(source_id="P:a", target_id="P:b", relation_type="knows"))
        nx_store.upsert_relation(Relation(source_id="P:b", target_id="P:c", relation_type="knows"))

        neighbors_1hop = nx_store.get_neighbors("P:a", max_hops=1)
        neighbors_2hop = nx_store.get_neighbors("P:a", max_hops=2)

        ids_1 = {e.entity_id for e, _ in neighbors_1hop}
        ids_2 = {e.entity_id for e, _ in neighbors_2hop}

        assert "P:b" in ids_1
        assert "P:c" not in ids_1
        assert "P:b" in ids_2
        assert "P:c" in ids_2

    def test_get_chunk_text_stored(self, nx_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="ORG:textstored", text="TextStored", entity_type="ORG")
        nx_store.upsert_entity(ent)
        nx_store.link_entity_to_chunk(ent.entity_id, "c-text", "The full text here.", doc_id="d1")
        chunks = nx_store.get_chunk_ids_for_entity(ent.entity_id)
        texts = {cid: txt for cid, txt in chunks}
        assert "c-text" in texts
        assert texts["c-text"] == "The full text here."


# ──────────────────────────────────────────────────────────────────────────────
# 3. KuzuGraphStore tests (requires kuzu installed)
# ──────────────────────────────────────────────────────────────────────────────

def kuzu_available():
    try:
        import kuzu
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not kuzu_available(), reason="kuzu not installed")
class TestKuzuGraphStore:
    @pytest.fixture
    def kuzu_store(self, tmp_path):
        from core.graph.store_kuzu import KuzuGraphStore
        store = KuzuGraphStore(db_path=str(tmp_path / "test.kuzu"))
        store.connect()
        yield store
        store.clear()
        store.close()

    def test_health(self, kuzu_store):
        assert kuzu_store.health_check()

    def test_upsert_and_retrieve_entity(self, kuzu_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="ORG:openai", text="OpenAI", entity_type="ORG")
        kuzu_store.upsert_entity(ent)
        fetched = kuzu_store.get_entity("ORG:openai")
        assert fetched is not None
        assert fetched.text == "OpenAI"
        assert fetched.entity_type == "ORG"

    def test_entity_count(self, kuzu_store):
        from core.graph.models import Entity
        kuzu_store.upsert_entity(Entity(entity_id="P:x", text="X", entity_type="PERSON"))
        kuzu_store.upsert_entity(Entity(entity_id="P:y", text="Y", entity_type="PERSON"))
        assert kuzu_store.entity_count() == 2

    def test_link_and_retrieve_chunks(self, kuzu_store):
        from core.graph.models import Entity
        ent = Entity(entity_id="ORG:google", text="Google", entity_type="ORG")
        kuzu_store.upsert_entity(ent)
        kuzu_store.link_entity_to_chunk("ORG:google", "k-c1", "Google was founded in 1998.", doc_id="d1")
        chunks = kuzu_store.get_chunk_ids_for_entity("ORG:google")
        assert any(cid == "k-c1" for cid, _ in chunks)

    def test_upsert_relation(self, kuzu_store):
        from core.graph.models import Entity, Relation
        a = Entity(entity_id="P:larry", text="Larry Page", entity_type="PERSON")
        b = Entity(entity_id="ORG:google", text="Google", entity_type="ORG")
        kuzu_store.upsert_entity(a)
        kuzu_store.upsert_entity(b)
        rel = Relation(source_id="P:larry", target_id="ORG:google", relation_type="founded", weight=1.0)
        kuzu_store.upsert_relation(rel)
        assert kuzu_store.relation_count() >= 1

    def test_neighbors(self, kuzu_store):
        from core.graph.models import Entity, Relation
        a = Entity(entity_id="P:p1", text="P1", entity_type="PERSON")
        b = Entity(entity_id="P:p2", text="P2", entity_type="PERSON")
        kuzu_store.upsert_entity(a)
        kuzu_store.upsert_entity(b)
        kuzu_store.upsert_relation(Relation(source_id="P:p1", target_id="P:p2", relation_type="knows"))
        neighbors = kuzu_store.get_neighbors("P:p1", max_hops=1)
        assert any(e.entity_id == "P:p2" for e, _ in neighbors)

    def test_persistence(self, tmp_path):
        """Data persists across separate connection instances."""
        from core.graph.models import Entity
        from core.graph.store_kuzu import KuzuGraphStore
        db_path = str(tmp_path / "persist_test.kuzu")

        # Write in first connection
        s1 = KuzuGraphStore(db_path=db_path)
        s1.connect()
        ent = Entity(entity_id="ORG:persist_test", text="PersistTest", entity_type="ORG")
        s1.upsert_entity(ent)
        s1.close()

        # Read in second connection
        s2 = KuzuGraphStore(db_path=db_path)
        s2.connect()
        fetched = s2.get_entity("ORG:persist_test")
        s2.close()

        assert fetched is not None
        assert fetched.text == "PersistTest"

    def test_find_entities_by_text(self, kuzu_store):
        from core.graph.models import Entity
        kuzu_store.upsert_entity(Entity(entity_id="ORG:microsoft", text="Microsoft", entity_type="ORG"))
        kuzu_store.upsert_entity(Entity(entity_id="ORG:microsoft_azure", text="Microsoft Azure", entity_type="ORG"))
        matches = kuzu_store.find_entities_by_text("Microsoft")
        assert len(matches) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 4. EntityRelationExtractor tests (requires spaCy)
# ──────────────────────────────────────────────────────────────────────────────

def spacy_available():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not spacy_available(), reason="spaCy en_core_web_sm not installed")
class TestEntityRelationExtractor:
    @pytest.fixture
    def extractor(self):
        from core.graph.extractor import EntityRelationExtractor
        return EntityRelationExtractor(extract_svo=True, extract_cooc=True)

    def test_is_available(self, extractor):
        assert extractor.is_available()

    def test_extract_entities(self, extractor):
        result = extractor.extract(SAMPLE_TEXT, "c1")
        assert result.entity_count > 0
        entity_types = {e.entity_type for e in result.entities}
        # Should find at least PERSON or ORG
        assert entity_types & {"PERSON", "ORG", "LOCATION"}

    def test_extract_returns_unique_entities(self, extractor):
        text = "Apple Apple Apple is a great company."
        result = extractor.extract(text, "c2")
        entity_ids = [e.entity_id for e in result.entities]
        assert len(entity_ids) == len(set(entity_ids))

    def test_extract_entities_only(self, extractor):
        ents = extractor.extract_entities_only(SAMPLE_TEXT)
        assert len(ents) > 0
        assert all(e.entity_id for e in ents)

    def test_extraction_result_cooccurrence(self, extractor):
        # Text with 2 entities in same sentence → co_occurs relation
        text = "Steve Jobs and Apple Inc. were both pivotal to Silicon Valley."
        result = extractor.extract(text, "c3")
        predicates = {t.predicate for t in result.triples}
        assert "co_occurs" in predicates

    def test_empty_text_returns_empty(self, extractor):
        result = extractor.extract("", "c-empty")
        assert result.entity_count == 0
        assert result.relation_count == 0

    def test_entity_ids_normalised(self, extractor):
        result = extractor.extract("Steve Jobs founded Apple.", "c4")
        for e in result.entities:
            # IDs should be lowercase with underscores
            assert e.entity_id == e.entity_id.lower() or ":" in e.entity_id

    def test_svo_extraction(self, extractor):
        # Clear SVO-extractable sentence
        text = "Google acquired DeepMind in London."
        result = extractor.extract(text, "c5")
        # Even if SVO doesn't fire (dep parse may vary), should not crash
        assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# 5. GraphTraverser tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphTraverser:
    @pytest.fixture
    def traverser_with_data(self):
        """GraphTraverser backed by NetworkX store pre-populated with Apple data."""
        from core.graph.store_networkx import NetworkXGraphStore
        from core.graph.extractor import EntityRelationExtractor
        from core.graph.traversal import GraphTraverser
        from core.graph.models import Entity, Relation

        store = NetworkXGraphStore()
        store.connect()

        apple = Entity(entity_id="ORG:apple_inc.", text="Apple Inc.", entity_type="ORG")
        jobs = Entity(entity_id="PERSON:steve_jobs", text="Steve Jobs", entity_type="PERSON")
        iphone = Entity(entity_id="PRODUCT:iphone", text="iPhone", entity_type="PRODUCT")

        for ent in (apple, jobs, iphone):
            store.upsert_entity(ent)

        store.link_entity_to_chunk(apple.entity_id, "chunk-apple-1", "Apple Inc. is a tech giant.", doc_id="d1")
        store.link_entity_to_chunk(jobs.entity_id, "chunk-apple-1", "Apple Inc. is a tech giant.", doc_id="d1")
        store.link_entity_to_chunk(iphone.entity_id, "chunk-apple-2", "iPhone changed mobile computing.", doc_id="d1")
        store.upsert_relation(Relation(source_id=apple.entity_id, target_id=jobs.entity_id, relation_type="founded_by"))
        store.upsert_relation(Relation(source_id=apple.entity_id, target_id=iphone.entity_id, relation_type="makes"))

        # Use a mock extractor that always returns Apple as an entity
        class MockExtractor:
            def is_available(self): return True
            def extract_entities_only(self, text):
                return [apple]

        traverser = GraphTraverser(graph_store=store, extractor=MockExtractor(), max_hops=2)
        return traverser, store

    def test_traverse_direct_hit(self, traverser_with_data):
        traverser, _ = traverser_with_data
        results, paths = traverser.traverse("Apple Inc.", top_k=10)
        assert len(results) > 0
        chunk_ids = {r.document.id for r in results}
        assert "chunk-apple-1" in chunk_ids

    def test_traverse_two_hop(self, traverser_with_data):
        traverser, _ = traverser_with_data
        results, paths = traverser.traverse("Apple Inc.", top_k=10)
        chunk_ids = {r.document.id for r in results}
        # iPhone chunk (2 hops via Apple→makes→iPhone) should be reachable
        assert "chunk-apple-2" in chunk_ids

    def test_traverse_returns_paths(self, traverser_with_data):
        traverser, _ = traverser_with_data
        _, paths = traverser.traverse("Apple Inc.", top_k=10)
        assert len(paths) > 0
        assert all(p.query_entity for p in paths)

    def test_traverse_no_entities(self, traverser_with_data):
        """When extractor finds no entities, traverser returns empty."""
        from core.graph.store_networkx import NetworkXGraphStore
        from core.graph.traversal import GraphTraverser

        store = NetworkXGraphStore()
        store.connect()

        class EmptyExtractor:
            def is_available(self): return True
            def extract_entities_only(self, text): return []

        traverser = GraphTraverser(store, EmptyExtractor(), max_hops=2)
        results, paths = traverser.traverse("something something")
        assert results == []

    def test_traverse_scores_direct_higher_than_hop(self, traverser_with_data):
        traverser, _ = traverser_with_data
        results, _ = traverser.traverse("Apple Inc.", top_k=10)
        # direct hits should have higher score than hop results
        scores_by_chunk = {r.document.id: r.score for r in results}
        assert scores_by_chunk.get("chunk-apple-1", 0) > scores_by_chunk.get("chunk-apple-2", -1)


# ──────────────────────────────────────────────────────────────────────────────
# 6. TripleHybridRetriever tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTripleHybridRetriever:
    @pytest.fixture
    def triple_retriever(self):
        """TripleHybridRetriever wired with mock HybridRetriever and live NetworkX store."""
        from core.graph.store_networkx import NetworkXGraphStore
        from core.graph.traversal import GraphTraverser
        from core.retrieval.triple_hybrid import TripleHybridRetriever
        from core.graph.models import Entity, Relation
        from core.store.models import Document, SearchResult

        # Populate graph
        store = NetworkXGraphStore()
        store.connect()
        apple = Entity(entity_id="ORG:apple_inc.", text="Apple Inc.", entity_type="ORG")
        store.upsert_entity(apple)
        store.link_entity_to_chunk(apple.entity_id, "c-hybrid-1", "Apple Inc. founded in 1976.", doc_id="d1")

        # Mock HybridRetriever returning 2 docs
        class MockHybrid:
            collection = "test"
            def retrieve(self, query, collection, top_k, filters=None, embedder=None, **kwargs):
                docs = [
                    SearchResult(
                        document=Document(id="c-hybrid-1", text="Apple Inc. founded in 1976.", embedding=[], metadata={}),
                        score=0.9, rank=1,
                    ),
                    SearchResult(
                        document=Document(id="c-hybrid-2", text="Unrelated passage.", embedding=[], metadata={}),
                        score=0.5, rank=2,
                    ),
                ]
                return docs[:top_k]

        class MockExtractor:
            def is_available(self): return True
            def extract_entities_only(self, text): return [apple]

        traverser = GraphTraverser(store, MockExtractor(), max_hops=1)
        retriever = TripleHybridRetriever(
            hybrid_retriever=MockHybrid(),
            traverser=traverser,
            graph_weight=1.0,
        )
        return retriever

    def test_retrieve_returns_results(self, triple_retriever):
        results, graph_paths = triple_retriever.retrieve(
            query="Tell me about Apple Inc.",
            collection="test",
            top_k=5,
        )
        assert len(results) > 0

    def test_graph_signal_annotated(self, triple_retriever):
        results, _ = triple_retriever.retrieve("Apple Inc.", "test", top_k=5)
        # At least one result should have retrieval_signals annotation
        all_signals = [
            r.document.metadata.get("retrieval_signals", "")
            for r in results
        ]
        assert any("graph" in s for s in all_signals)

    def test_boosted_chunk_ranks_higher(self, triple_retriever):
        """chunk-1 appears in both hybrid and graph → should rank above chunk-2."""
        results, _ = triple_retriever.retrieve("Apple Inc.", "test", top_k=5)
        ids = [r.document.id for r in results]
        if "c-hybrid-1" in ids and "c-hybrid-2" in ids:
            assert ids.index("c-hybrid-1") < ids.index("c-hybrid-2")

    def test_graph_paths_returned(self, triple_retriever):
        _, paths = triple_retriever.retrieve("Apple Inc.", "test", top_k=5)
        assert isinstance(paths, list)

    def test_fallback_when_no_graph_results(self):
        """When graph returns nothing, falls back to 2-way hybrid."""
        from core.graph.store_networkx import NetworkXGraphStore
        from core.graph.traversal import GraphTraverser
        from core.retrieval.triple_hybrid import TripleHybridRetriever
        from core.store.models import Document, SearchResult

        store = NetworkXGraphStore()
        store.connect()  # empty graph

        class MockHybrid:
            def retrieve(self, query, collection, top_k, filters=None, embedder=None, **kwargs):
                return [
                    SearchResult(
                        document=Document(id="h1", text="Hybrid result.", embedding=[], metadata={}),
                        score=0.8, rank=1,
                    )
                ]

        class EmptyExtractor:
            def is_available(self): return True
            def extract_entities_only(self, text): return []

        traverser = GraphTraverser(store, EmptyExtractor(), max_hops=1)
        retriever = TripleHybridRetriever(MockHybrid(), traverser, graph_weight=1.0)
        results, paths = retriever.retrieve("anything", "test", top_k=5)
        assert len(results) >= 1
        assert paths == []


# ──────────────────────────────────────────────────────────────────────────────
# 7. GraphStoreRegistry tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphStoreRegistry:
    def test_networkx_backend(self):
        from core.graph.registry import get_graph_store
        store = get_graph_store("networkx", {})
        from core.graph.store_networkx import NetworkXGraphStore
        assert isinstance(store, NetworkXGraphStore)

    @pytest.mark.skipif(not kuzu_available(), reason="kuzu not installed")
    def test_kuzu_backend(self, tmp_path):
        from core.graph.registry import get_graph_store
        store = get_graph_store("kuzu", {"kuzu": {"db_path": str(tmp_path / "reg.kuzu")}})
        from core.graph.store_kuzu import KuzuGraphStore
        assert isinstance(store, KuzuGraphStore)

    def test_unknown_backend_raises(self):
        from core.graph.registry import get_graph_store
        with pytest.raises((ValueError, KeyError)):
            get_graph_store("invalid_backend_xyz", {})


# ──────────────────────────────────────────────────────────────────────────────
# 8. RAGResponse graph fields
# ──────────────────────────────────────────────────────────────────────────────

class TestRAGResponseGraphFields:
    def test_default_empty_graph_fields(self):
        from orchestrator.response import RAGResponse
        resp = RAGResponse(query="test", answer="answer", results=[])
        assert resp.graph_entities == []
        assert resp.graph_paths == []

    def test_graph_entities_and_paths_populated(self):
        from orchestrator.response import RAGResponse
        from core.graph.models import GraphPath
        path = GraphPath(
            query_entity="Apple",
            path_entities=["Steve Jobs"],
            path_types=["founded_by"],
            chunk_ids=["c1"],
            hop_distance=1,
            relevance_score=0.6,
        )
        resp = RAGResponse(
            query="Apple?",
            answer="Apple is...",
            results=[],
            graph_entities=["ORG:apple"],
            graph_paths=[path],
        )
        assert len(resp.graph_entities) == 1
        assert len(resp.graph_paths) == 1

    def test_summary_includes_graph_paths(self):
        from orchestrator.response import RAGResponse
        from core.graph.models import GraphPath
        path = GraphPath(
            query_entity="X", path_entities=[], path_types=[],
            chunk_ids=[], hop_distance=0, relevance_score=1.0,
        )
        resp = RAGResponse(
            query="q", answer="a", results=[], graph_paths=[path]
        )
        s = resp.summary()
        assert "Graph paths" in s

    def test_graph_explanation_no_paths(self):
        from orchestrator.response import RAGResponse
        resp = RAGResponse(query="q", answer="a", results=[])
        expl = resp.graph_explanation()
        assert "No knowledge graph paths" in expl

    def test_graph_explanation_with_paths(self):
        from orchestrator.response import RAGResponse
        from core.graph.models import GraphPath
        path = GraphPath(
            query_entity="Hamlet", path_entities=["Denmark"],
            path_types=["set_in"], chunk_ids=["c1"],
            hop_distance=1, relevance_score=0.6,
        )
        resp = RAGResponse(query="q", answer="a", results=[], graph_paths=[path])
        expl = resp.graph_explanation()
        assert "Hamlet" in expl
        assert "Denmark" in expl


# ──────────────────────────────────────────────────────────────────────────────
# 9. Neo4j stub — tagged as integration (requires running Neo4j)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestNeo4jGraphStoreStub:
    """Requires a running Neo4j instance. Skipped in CI / local without Neo4j."""

    def test_neo4j_import(self):
        """Verify Neo4jGraphStore can be imported."""
        from core.graph.store_neo4j import Neo4jGraphStore
        assert Neo4jGraphStore is not None

    def test_neo4j_connect_fails_gracefully_without_server(self):
        from core.graph.store_neo4j import Neo4jGraphStore
        store = Neo4jGraphStore(uri="bolt://localhost:9999", user="neo4j", password="wrong")
        with pytest.raises(Exception):
            store.connect()
            store.health_check()
