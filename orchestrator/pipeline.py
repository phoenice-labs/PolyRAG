"""
Full 9-Phase RAGPipeline — all capabilities wired together.
The orchestration layer is backend-agnostic: swap config.yaml → different vector store.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.embedding.sentence_transformer import EmbeddingRegistry
from core.ingestion.ingestor import IngestionResult, Ingestor
from core.store.models import Document, SearchResult
from core.store.registry import AdapterRegistry
from core.observability.logging import pipeline_logger, metrics


class RAGPipeline:
    """
    Unified RAG orchestration across all 9 phases.

    Usage
    -----
        with RAGPipeline.from_config("config/config.yaml") as p:
            p.ingest_gutenberg()
            response = p.ask("What does Hamlet say about existence?")
            print(response.summary())
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._started = False

        # ── Phase 1: Vector store + embedder + ingestor ───────────────────────
        store_cfg = config["store"]
        backend = store_cfg["backend"]
        self.store = AdapterRegistry.create(backend, store_cfg.get(backend, {}))
        embed_cfg = config.get("embedding", {})
        self.embedder = EmbeddingRegistry.create(
            embed_cfg.get("provider", "sentence_transformer"), embed_cfg
        )
        self.ingestor = Ingestor(self.store, self.embedder, config.get("ingestion", {}))

        # ── Phase 2: Chunking pipeline ────────────────────────────────────────
        self._chunk_registry = None
        self._chunking_pipeline = None

        # ── Phase 3: Hybrid search ────────────────────────────────────────────
        from core.retrieval.bm25 import BM25Index
        self.bm25_index = BM25Index()
        self._hybrid_retriever = None

        # ── Phase 3b: SPLADE sparse neural index ─────────────────────────────
        splade_cfg = config.get("retrieval", {}).get("splade", {})
        splade_enabled = splade_cfg.get("enabled", False)
        self.splade_index = None
        if splade_enabled:
            from core.retrieval.splade import SparseNeuralIndex
            splade_model = splade_cfg.get("model", "naver/splade-v3")
            splade_persist = splade_cfg.get("persist_dir", "./data/splade")
            self.splade_index = SparseNeuralIndex(
                model_name=splade_model,
                persist_dir=splade_persist,
            )
            pipeline_logger.info("SPLADE sparse neural index created", model=splade_model)

        # ── Phase 4: Multi-stage retrieval ────────────────────────────────────
        self._multistage = None

        # ── Phase 5: Query intelligence (LM Studio) ──────────────────────────
        self._llm_client = None
        self._query_pipeline = None
        self._context_tracker = None

        # ── Phase 6: Provenance ───────────────────────────────────────────────
        from core.provenance.models import IngestionAuditLog, DocumentVersionRegistry
        self.audit_log = IngestionAuditLog(
            config.get("audit_log_path", "./data/ingestion_audit.jsonl")
        )
        self.version_registry = DocumentVersionRegistry()

        # ── Phase 7: Confidence ───────────────────────────────────────────────
        from core.confidence.signals import AnswerConfidenceAggregator
        self.confidence_engine = AnswerConfidenceAggregator(embedder=self.embedder)

        # ── Phase 8: Temporal + Classification ───────────────────────────────
        from core.temporal.filters import TemporalFilter, ClassificationFilter
        self.temporal_filter = TemporalFilter()
        self.clf_filter = ClassificationFilter(
            config.get("access", {}).get("user_clearance", "INTERNAL")
        )

        # ── Phase 9: Noise control ────────────────────────────────────────────
        from core.noise.filters import NoiseFilterPipeline
        self.noise_pipeline = NoiseFilterPipeline(
            quality_threshold=config.get("quality", {}).get("min_score", 0.3),
            dedup_threshold=config.get("quality", {}).get("dedup_threshold", 0.85),
        )

        # ── Phase 10: Knowledge Graph ─────────────────────────────────────────
        self._graph_store = None
        self._graph_extractor = None
        self._graph_traverser = None
        self._triple_retriever = None

        # ── Phase 11: Advanced Retrieval ──────────────────────────────────────
        self._llm_extractor = None          # LLM entity extractor (graph enhancement)
        self._llm_extractor_merge = True    # merge with spaCy output
        self._raptor_indexer = None
        self._raptor_retriever = None
        self._raptor_doc_buffer: List[Document] = []   # accumulates for RAPTOR build
        self._raptor_enabled = False
        self._contextual_reranker = None
        self._mmr_reranker = None

        # ── Retrieval trace (populated by query()) ────────────────────────────
        # Each entry: {"method": str, "candidates_before": int, "candidates_after": int}
        self._last_retrieval_trace: List[Dict] = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "RAGPipeline":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "RAGPipeline":
        self.store.connect()
        coll_name = self.ingestor.collection_name
        if not self.store.collection_exists(coll_name):
            self.store.create_collection(coll_name, self.embedder.embedding_dim)

        # Phase 2: chunking — honour chunk_strategy from config
        from core.chunking.models import ChunkRegistry
        from core.chunking.pipeline import ChunkingPipeline
        self._chunk_registry = ChunkRegistry()
        ingest_cfg = self.config.get("ingestion", {})
        chunk_size     = ingest_cfg.get("chunk_size", 512)
        chunk_overlap  = ingest_cfg.get("chunk_overlap", 64)
        chunk_strategy = ingest_cfg.get("chunk_strategy", "section").lower()

        if chunk_strategy == "sentence":
            from core.chunking.sentence_boundary import SentenceBoundaryChunker
            # Convert chars → words (÷5 avg), but cap at 120 words to prevent huge chunks.
            # A 120-word chunk ≈ 600 chars — safe for embedding recall.
            max_w = min(120, max(10, chunk_size // 5))
            chunker = SentenceBoundaryChunker(
                max_words=max_w,
                overlap_sents=max(0, chunk_overlap // 40),
            )
        elif chunk_strategy in ("fixed", "sliding"):
            # Character-level sliding window — chunk_size is exact char limit.
            from core.chunking.fixed_overlap import FixedOverlapChunker
            chunker = FixedOverlapChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif chunk_strategy == "paragraph":
            # Blank-line paragraph boundaries; sub-split if > chunk_size chars.
            from core.chunking.paragraph import ParagraphChunker
            chunker = ParagraphChunker(
                max_chars=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # "section" (default) — detect headings, store children of chunk_size chars.
            from core.chunking.section_aware import SectionAwareChunker
            chunker = SectionAwareChunker(
                child_size=chunk_size,
                child_overlap=chunk_overlap,
            )
        self._chunking_pipeline = ChunkingPipeline(
            chunker=chunker,
            registry=self._chunk_registry,
        )

        # Phase 3: hybrid retriever (Dense + BM25 + optional SPLADE)
        from core.retrieval.hybrid import HybridRetriever
        splade_cfg = self.config.get("retrieval", {}).get("splade", {})
        self._hybrid_retriever = HybridRetriever(
            store=self.store,
            bm25_index=self.bm25_index,
            embedder=self.embedder,
            collection=coll_name,
            splade_index=self.splade_index,
            splade_w=splade_cfg.get("splade_weight", 1.0),
            bm25_w_with_splade=splade_cfg.get("bm25_weight_with_splade", 0.8),
        )

        # Phase 3b: warm-start BM25 + SPLADE + ChunkRegistry from stored chunks (survives server restarts)
        self._rebuild_from_store(coll_name)

        # Phase 4: multi-stage
        from core.retrieval.multistage import (
            CrossEncoderReRanker, CrossDocumentAggregator,
            MultiStageRetriever, ParentExpander,
        )
        reranker = CrossEncoderReRanker(
            self.config.get("retrieval", {}).get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
        self._multistage = MultiStageRetriever(
            hybrid_retriever=self._hybrid_retriever,
            reranker=reranker,
            parent_expander=ParentExpander(self._chunk_registry),
            aggregator=CrossDocumentAggregator(),
            relevance_threshold=self.config.get("retrieval", {}).get("relevance_threshold", 0.0),
            recall_multiplier=self.config.get("retrieval", {}).get("recall_multiplier", 2),
        )

        # Phase 5: LLM / query intelligence
        llm_cfg = self.config.get("llm", {})
        from core.query.llm_client import LMStudioClient
        from core.query.context import ConversationContextTracker, QueryIntelligencePipeline
        self._llm_client = LMStudioClient(
            base_url=llm_cfg.get("base_url", "http://localhost:1234/v1"),
            model=llm_cfg.get("model", "mistralai/ministral-3b"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 512),
        )
        self._context_tracker = ConversationContextTracker(
            max_turns=llm_cfg.get("context_turns", 5)
        )
        self._query_pipeline = QueryIntelligencePipeline(
            client=self._llm_client,
            embedder=self.embedder,
            context_tracker=self._context_tracker,
            enable_rewrite=llm_cfg.get("enable_rewrite", True),
            enable_stepback=llm_cfg.get("enable_stepback", False),
            enable_multi_query=llm_cfg.get("enable_multi_query", True),
            enable_hyde=llm_cfg.get("enable_hyde", True),
            n_paraphrases=llm_cfg.get("n_paraphrases", 3),
        )

        self._started = True
        pipeline_logger.info("RAGPipeline started", backend=self.config["store"]["backend"])

        # Phase 10: Knowledge Graph (optional — graceful degradation if disabled/unavailable)
        graph_cfg = self.config.get("graph", {})
        if graph_cfg.get("enabled", True):
            try:
                from core.graph.registry import get_graph_store
                from core.graph.extractor import EntityRelationExtractor
                from core.graph.traversal import GraphTraverser
                from core.retrieval.triple_hybrid import TripleHybridRetriever

                graph_backend = graph_cfg.get("backend", "kuzu")
                self._graph_store = get_graph_store(graph_backend, graph_cfg)
                self._graph_store.connect()

                # ── Restore graph from snapshot (persists across restarts) ──
                _snapshot_path = (
                    Path(__file__).resolve().parent.parent
                    / "data" / "graphs"
                    / f"{self.ingestor.collection_name}.json"
                )
                if _snapshot_path.exists():
                    try:
                        self._load_graph_snapshot(_snapshot_path)
                        pipeline_logger.info(
                            "Graph snapshot restored",
                            path=str(_snapshot_path),
                            entities=self._graph_store.entity_count(),
                        )
                    except Exception as exc:
                        pipeline_logger.warning("Graph snapshot restore failed", error=str(exc))
                self._graph_extractor = EntityRelationExtractor(
                    model=graph_cfg.get("spacy_model", "en_core_web_sm"),
                    extract_svo=graph_cfg.get("extract_svo", True),
                    extract_cooc=graph_cfg.get("extract_cooccurrence", True),
                )
                self._graph_traverser = GraphTraverser(
                    graph_store=self._graph_store,
                    extractor=self._graph_extractor,
                    max_hops=graph_cfg.get("max_hops", 2),
                    llm_client=self._llm_client,
                )
                self._triple_retriever = TripleHybridRetriever(
                    hybrid_retriever=self._hybrid_retriever,
                    traverser=self._graph_traverser,
                    graph_weight=graph_cfg.get("graph_weight", 1.0),
                )
                pipeline_logger.info(
                    "Phase 10 KnowledgeGraph online",
                    backend=graph_backend,
                    spacy_available=self._graph_extractor.is_available(),
                )
            except Exception as exc:
                pipeline_logger.warning(
                    "Phase 10 KnowledgeGraph unavailable — falling back to 2-way hybrid",
                    error=str(exc),
                )
                self._graph_store = None
                self._triple_retriever = None

        # ── Phase 11a: LLM entity extractor (graph enhancement) ──────────────
        llm_ext_cfg = graph_cfg.get("llm_extraction", {})
        if llm_ext_cfg.get("enabled", False) and self._llm_client and self._graph_store:
            try:
                from core.graph.llm_extractor import LLMEntityExtractor
                self._llm_extractor = LLMEntityExtractor(
                    llm_client=self._llm_client,
                    max_chunk_chars=llm_ext_cfg.get("max_chunk_chars", 2000),
                )
                self._llm_extractor_merge = llm_ext_cfg.get("merge_with_spacy", True)
                pipeline_logger.info("Phase 11a LLM entity extractor ready")
            except Exception as exc:
                pipeline_logger.warning("Phase 11a LLM extractor init failed", error=str(exc))

        # ── Phase 11b: RAPTOR hierarchical index ─────────────────────────────
        raptor_cfg = self.config.get("advanced_retrieval", {}).get("raptor", {})
        if raptor_cfg.get("enabled", False) and self._llm_client:
            try:
                from core.retrieval.raptor import RaptorIndexer, RaptorRetriever
                self._raptor_indexer = RaptorIndexer(
                    llm_client=self._llm_client,
                    embedder=self.embedder,
                    n_clusters=raptor_cfg.get("n_clusters", 10),
                )
                self._raptor_retriever = RaptorRetriever(
                    vector_store=self.store,
                    embedder=self.embedder,
                    summary_weight=raptor_cfg.get("summary_weight", 0.7),
                )
                self._raptor_enabled = True
                pipeline_logger.info("Phase 11b RAPTOR hierarchical index ready")
            except Exception as exc:
                pipeline_logger.warning("Phase 11b RAPTOR init failed", error=str(exc))

        # ── Phase 11c: Contextual LLM re-ranker ──────────────────────────────
        ctx_cfg = self.config.get("advanced_retrieval", {}).get("contextual_reranker", {})
        if ctx_cfg.get("enabled", False) and self._llm_client:
            try:
                from core.retrieval.contextual_reranker import ContextualReranker
                self._contextual_reranker = ContextualReranker(
                    llm_client=self._llm_client,
                    llm_weight=ctx_cfg.get("llm_weight", 0.4),
                    max_chunks_to_rank=ctx_cfg.get("max_chunks_to_rank", 10),
                )
                pipeline_logger.info("Phase 11c Contextual re-ranker ready")
            except Exception as exc:
                pipeline_logger.warning("Phase 11c contextual reranker init failed", error=str(exc))

        # ── Phase 11d: MMR diversity re-ranker ───────────────────────────────
        mmr_cfg = self.config.get("advanced_retrieval", {}).get("mmr", {})
        if mmr_cfg.get("enabled", True):
            try:
                from core.retrieval.mmr import MMRReranker
                self._mmr_reranker = MMRReranker(
                    diversity_weight=mmr_cfg.get("diversity_weight", 0.3),
                )
                pipeline_logger.info("Phase 11d MMR diversity re-ranker ready")
            except Exception as exc:
                pipeline_logger.warning("Phase 11d MMR init failed", error=str(exc))

        return self

    def stop(self) -> None:
        self.store.close()
        if self._graph_store:
            self._graph_store.close()
        self._started = False
        pipeline_logger.info("RAGPipeline stopped")

    def _rebuild_from_store(self, collection: Optional[str] = None) -> int:
        """
        Warm-start both the BM25 index and the ChunkRegistry from all chunks
        currently in the vector store.  A single ``fetch_all()`` call populates
        both structures so neither is lost across server restarts.

        - BM25 enables keyword search without re-ingesting data.
        - ChunkRegistry enables ``ParentExpander`` to fetch surrounding sibling
          chunks (the "small-to-big" context window) for any retrieved child chunk.

        Returns the number of chunks loaded (0 if collection is empty or adapter
        does not support ``fetch_all``).
        """
        from core.chunking.models import Chunk as RegistryChunk

        coll = collection or self.ingestor.collection_name
        if not self.store.collection_exists(coll):
            return 0

        # Milvus enforces (offset+limit) <= 16384; paginate in chunks to stay within limits.
        _PAGE_SIZE = 10_000
        raw: list = []
        try:
            offset = 0
            while True:
                try:
                    page = self.store.fetch_all(coll, limit=_PAGE_SIZE, offset=offset)
                except TypeError:
                    # Adapter doesn't support offset — fall back to single-page fetch
                    page = self.store.fetch_all(coll, limit=_PAGE_SIZE)
                    raw.extend(page)
                    break
                if not page:
                    break
                raw.extend(page)
                if len(page) < _PAGE_SIZE:
                    break
                offset += _PAGE_SIZE
        except NotImplementedError:
            pipeline_logger.warning(
                "Warm-start skipped — adapter has no fetch_all()",
                backend=self.config["store"]["backend"],
            )
            return 0
        except Exception as exc:
            pipeline_logger.warning("Warm-start failed", error=str(exc))
            return 0

        if not raw:
            return 0

        def _is_parent(meta: dict) -> bool:
            """Handle both bool True and string 'True'/'true' stored by string-serializing backends."""
            val = meta.get("is_parent", False)
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes")

        # Try to load SPLADE from disk first (avoids re-encoding on restart)
        splade_loaded = False
        if self.splade_index is not None:
            splade_loaded = self.splade_index.load(coll)

        splade_docs_to_encode: List[Document] = []

        for r in raw:
            meta = r.get("metadata", {})
            doc = Document(id=r["id"], text=r["text"], embedding=[], metadata=meta)
            # BM25 index — skip parent-sized chunks (is_parent=True were not upserted
            # after the recent fix, but older collections may still have them).
            # Handles both bool True and string "True" (Milvus stores all metadata as strings).
            if not _is_parent(meta):
                self.bm25_index.add([doc])
                # Collect for SPLADE batch encoding if not loaded from disk
                if self.splade_index is not None and not splade_loaded:
                    splade_docs_to_encode.append(doc)
            # ChunkRegistry — reconstruct Chunk shell (text + parent_id only; no embedding needed)
            chunk_shell = RegistryChunk(
                chunk_id=r["id"],
                doc_id=meta.get("doc_id", ""),
                parent_id=meta.get("parent_id") or None,
                text=r["text"],
                start_char=int(meta.get("start_char", 0) or 0),
                end_char=int(meta.get("end_char", 0) or 0),
                section_title=meta.get("section_title") or None,
            )
            self._chunk_registry.register(chunk_shell)

        # Batch-encode with SPLADE if disk load was not available
        if splade_docs_to_encode:
            pipeline_logger.info(
                "SPLADE warm-start: encoding %d chunks (first startup, model may download ~400 MB)",
                len(splade_docs_to_encode),
                collection=coll,
            )
            self.splade_index.add(splade_docs_to_encode, collection=coll)
            pipeline_logger.info("SPLADE warm-start complete", collection=coll)

        n = len(raw)
        pipeline_logger.info(
            "Warm-start complete (BM25 + SPLADE + ChunkRegistry)",
            collection=coll,
            n_docs=n,
        )
        return n

    # Keep old name as alias for backward compatibility
    def _rebuild_bm25_from_store(self, collection: Optional[str] = None) -> int:
        return self._rebuild_from_store(collection)

    def __enter__(self) -> "RAGPipeline":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        collection: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest raw text through Phase 2 semantic chunking pipeline."""
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        coll = collection or self.ingestor.collection_name

        # Phase 2: semantic chunking
        chunks = self._chunking_pipeline.run(text, doc_id, meta)

        if not self.store.collection_exists(coll):
            self.store.create_collection(coll, self.embedder.embedding_dim)

        from core.store.models import Document
        all_docs = []
        # Parent chunks (is_parent=True) are kept in ChunkRegistry for context expansion
        # but are NOT upserted to the vector store — they are too large (10K–40K chars)
        # and dilute embedding similarity for all searches.
        search_chunks = [c for c in chunks if not c.metadata.get("is_parent", False)]
        for i in range(0, len(search_chunks), 32):
            batch = search_chunks[i: i + 32]
            embeddings = self.embedder.embed([c.text for c in batch])
            docs = [
                Document(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=emb,
                    metadata={
                        **chunk.metadata,
                        "doc_id": doc_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "section_title": chunk.section_title or "",
                        "parent_id": chunk.parent_id or "",
                    },
                )
                for chunk, emb in zip(batch, embeddings)
            ]
            self.store.upsert(coll, docs)
            self.bm25_index.add(docs)         # Phase 3
            if self.splade_index is not None:  # Phase 3b: SPLADE
                self.splade_index.add(docs, collection=coll)
            all_docs.extend(docs)

            # Phase 10: entity + relation extraction → populate KnowledgeGraph
            if self._graph_store and self._graph_extractor:
                for chunk, doc in zip(batch, docs):
                    try:
                        extraction = self._graph_extractor.extract(chunk.text, chunk.chunk_id)
                        for entity in extraction.entities:
                            self._graph_store.upsert_entity(entity)
                            self._graph_store.link_entity_to_chunk(
                                entity_id=entity.entity_id,
                                chunk_id=chunk.chunk_id,
                                chunk_text=chunk.text,
                                doc_id=doc_id,
                            )
                        for triple in extraction.triples:
                            self._graph_store.upsert_entity(triple.subject)
                            self._graph_store.upsert_entity(triple.object)
                            self._graph_store.upsert_relation(triple.to_relation())
                    except Exception as exc:
                        pipeline_logger.warning(
                            "Graph extraction failed for chunk", chunk_id=chunk.chunk_id, error=str(exc)
                        )

            # Phase 11a: LLM-enhanced entity extraction (supplements or replaces spaCy)
            if self._llm_extractor and self._graph_store and self._llm_client and self._llm_client.is_available():
                for chunk, doc in zip(batch, docs):
                    try:
                        llm_result = self._llm_extractor.extract(chunk.text, chunk.chunk_id)
                        for entity in llm_result.entities:
                            self._graph_store.upsert_entity(entity)
                            self._graph_store.link_entity_to_chunk(
                                entity_id=entity.entity_id,
                                chunk_id=chunk.chunk_id,
                                chunk_text=chunk.text,
                                doc_id=doc_id,
                            )
                        for triple in llm_result.triples:
                            self._graph_store.upsert_entity(triple.subject)
                            self._graph_store.upsert_entity(triple.object)
                            self._graph_store.upsert_relation(triple.to_relation())
                    except Exception as exc:
                        pipeline_logger.warning(
                            "LLM graph extraction failed", chunk_id=chunk.chunk_id, error=str(exc)
                        )

        # Phase 11b: accumulate docs for RAPTOR tree build
        if self._raptor_enabled:
            self._raptor_doc_buffer.extend(all_docs)

        # Phase 6: audit
        self.audit_log.record(
            doc_id=doc_id,
            source=meta.get("source_url", meta.get("source_file", "inline")),
            chunk_count=len(chunks),
            metadata=meta,
        )
        metrics.increment("ingest.chunks", len(chunks))

        # Phase 10b: persist graph to disk so graph router can load it after restart
        if self._graph_store and self._graph_store.entity_count() > 0:
            ec = self._graph_store.entity_count()
            rc = self._graph_store.relation_count()
            pipeline_logger.info(
                "Phase 10 graph extraction complete",
                entities=ec,
                relations=rc,
                collection=coll,
            )
            try:
                self._save_graph_snapshot(coll)
                pipeline_logger.info("Graph snapshot saved", collection=coll, entities=ec, relations=rc)
            except Exception as exc:
                pipeline_logger.warning("Graph snapshot save failed", error=str(exc))

        return IngestionResult(
            total_chunks=len(search_chunks),
            upserted=len(all_docs),
            skipped=len(chunks) - len(search_chunks),  # parent chunks excluded from vector DB
            collection=coll,
            doc_ids=[d.id for d in all_docs],
        )

    def ingest_file(self, path: str, metadata: Optional[Dict] = None) -> IngestionResult:
        from core.ingestion.loader import load_text_file
        text = load_text_file(path)
        return self.ingest_text(text, metadata={"source_file": path, **(metadata or {})})

    def _load_graph_snapshot(self, path: Path) -> None:
        """
        Restore the in-memory NetworkX graph from a JSON snapshot file.
        Rebuilds entities, relations, and chunk links so that graph-based retrieval
        works immediately after a process restart — without re-ingesting documents.
        """
        import json
        from core.graph.models import Entity, Relation

        with open(path, encoding="utf-8") as f:
            snap = json.load(f)

        store = self._graph_store

        # Re-add entities
        for n in snap.get("nodes", []):
            eid = n.get("id", "")
            if not eid:
                continue
            label = n.get("label", eid)
            etype = n.get("entity_type", n.get("type", "CONCEPT"))
            # Normalize type to internal format (snapshot may use frontend enum)
            _type_to_internal = {"LOC": "LOCATION", "ORG": "ORG", "PERSON": "PERSON",
                                  "CONCEPT": "CONCEPT", "OTHER": "CONCEPT"}
            etype = _type_to_internal.get(etype, etype)
            entity = Entity(
                entity_id=eid,
                text=label,
                entity_type=etype,
            )
            store.upsert_entity(entity)

            # Re-link chunk provenance
            for chunk_ref in n.get("chunks", []):
                cid = chunk_ref.get("chunk_id", "")
                snippet = chunk_ref.get("snippet", "")
                if cid:
                    store.link_entity_to_chunk(eid, cid, snippet)

        # Re-add entity→entity relations
        for e in snap.get("edges", []):
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src and tgt:
                try:
                    store.upsert_relation(Relation(
                        source_id=src,
                        target_id=tgt,
                        relation_type=e.get("relation", "co_occurs"),
                        weight=float(e.get("weight", 1.0)),
                    ))
                except Exception:
                    pass

    def _save_graph_snapshot(self, collection: str) -> None:
        """Persist the in-memory graph to data/graphs/{collection}.json so the graph
        router can load it across process restarts."""
        import json
        root = Path(__file__).resolve().parent.parent
        graph_dir = root / "data" / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        out = graph_dir / f"{collection}.json"
        gs = self._graph_store
        G = gs._G  # networkx MultiDiGraph

        # Map raw spaCy/model entity types → frontend type enum
        _TYPE_MAP = {
            "PERSON": "PERSON", "PER": "PERSON",
            "ORG": "ORG", "ORGANIZATION": "ORG",
            "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC", "FAC": "LOC",
            "CONCEPT": "CONCEPT", "NORP": "CONCEPT", "LAW": "CONCEPT",
            "EVENT": "CONCEPT", "WORK_OF_ART": "CONCEPT",
        }

        nodes = []
        # Build a lookup: entity_id → list of (chunk_id, chunk_text[:120])
        chunk_provenance: dict = {}
        for src, tgt, d in G.edges(data=True):
            if d.get("edge_type") == "appears_in":
                chunk_data = G.nodes[tgt]
                chunk_provenance.setdefault(src, []).append({
                    "chunk_id": chunk_data.get("chunk_id", ""),
                    "snippet": (chunk_data.get("text", "")[:140]).strip(),
                })

        for nid, d in G.nodes(data=True):
            if d.get("node_type") == "chunk":
                continue   # skip chunk nodes — visualise entities only
            raw_type = (d["entity"].entity_type if d.get("entity") else d.get("type", "CONCEPT"))
            ui_type = _TYPE_MAP.get(raw_type.upper(), "OTHER")
            nodes.append({
                "id": str(nid),
                "label": d["entity"].text if d.get("entity") else str(nid),
                "type": ui_type,
                "entity_type": raw_type,
                "frequency": d.get("frequency", 1),
                "chunks": chunk_provenance.get(str(nid), []),
            })

        edges = []
        for src, tgt, d in G.edges(data=True):
            if d.get("edge_type") == "relates_to":
                edges.append({
                    "source": str(src),
                    "target": str(tgt),
                    "relation": d.get("relation_type", "related"),
                    "weight": float(d.get("weight", 1.0)),
                })

        snapshot = {
            "collection": collection,
            "entity_count": len(nodes),
            "relation_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        pipeline_logger.info(
            "Graph snapshot saved",
            collection=collection,
            nodes=len(nodes),
            edges=len(edges),
            path=str(out),
        )

    def build_raptor_index(self, collection: Optional[str] = None) -> int:
        """
        Build RAPTOR hierarchical summary index from buffered ingested documents.

        Call this after all ingest_text() / ingest_file() calls are complete.
        Returns the number of summary chunks created (0 if RAPTOR is disabled or LLM offline).
        """
        if not self._raptor_indexer:
            return 0
        if not self._raptor_doc_buffer:
            pipeline_logger.warning("RAPTOR: no documents buffered — ingest first")
            return 0
        coll = collection or self.ingestor.collection_name
        n = self._raptor_indexer.build(self._raptor_doc_buffer, coll, self.store)
        self._raptor_doc_buffer.clear()
        pipeline_logger.info("RAPTOR index built", summaries=n, collection=coll)
        return n

    def ingest_gutenberg(
        self,
        url: Optional[str] = None,
        strip_boilerplate: bool = True,
        metadata: Optional[Dict] = None,
    ) -> IngestionResult:
        from core.ingestion.loader import (
            download_gutenberg, strip_gutenberg_header_footer, GUTENBERG_SHAKESPEARE_URL,
        )
        url = url or GUTENBERG_SHAKESPEARE_URL
        text = download_gutenberg(url)
        if strip_boilerplate:
            text = strip_gutenberg_header_footer(text)
        return self.ingest_text(
            text,
            metadata={"source_url": url, "source": "gutenberg", **(metadata or {})},
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        filters: Optional[Dict] = None,
        user_context: Optional[Dict] = None,
        use_multistage: bool = True,
        enable_dense: bool = True,
        enable_bm25: bool = True,
        enable_splade: bool = True,
        enable_graph: bool = True,
        enable_rerank: bool = True,
        enable_llm_graph: bool = False,
    ) -> List[SearchResult]:
        """Phase 1-4 + 10 + 11 retrieval (no LLM answer). Returns SearchResult list.

        Parallelism map
        ───────────────
        Level 1 (this method):   TripleHybrid  ║  RAPTOR  — parallel via ThreadPoolExecutor
        Level 2 (triple_hybrid): Hybrid(D+B+S) ║  Graph   — parallel via ThreadPoolExecutor
        Level 3 (hybrid):        Dense Vector  ║  BM25  ║  SPLADE — parallel via ThreadPoolExecutor
        After all retrievals: Cross-Encoder → MMR → Contextual Rerank (must be sequential)
        """
        import concurrent.futures

        coll = collection or self.ingestor.collection_name
        if self._hybrid_retriever:
            self._hybrid_retriever.collection = coll

        # Reset retrieval trace for this request
        self._last_retrieval_trace = []

        # ── Level 1: TripleHybrid + RAPTOR in parallel ────────────────────────
        graph_paths = []
        results: List[SearchResult] = []

        use_triple = (
            self._triple_retriever
            and self._graph_store
            and self._graph_store.entity_count() > 0
        )

        def _run_triple():
            try:
                return self._triple_retriever.retrieve(
                    query=query_text,
                    collection=coll,
                    top_k=top_k * 2,
                    filters=filters,
                    embedder=self.embedder,
                    enable_dense=enable_dense,
                    enable_bm25=enable_bm25,
                    enable_splade=enable_splade,
                    enable_graph=enable_graph,
                    enable_llm_graph=enable_llm_graph,
                )
            except Exception as exc:
                pipeline_logger.warning("TripleHybridRetriever failed, falling back", error=str(exc))
                return self._fallback_retrieve(query_text, coll, top_k * 2, filters, use_multistage,
                                               enable_dense=enable_dense, enable_bm25=enable_bm25,
                                               enable_splade=enable_splade), []

        def _run_fallback():
            return self._fallback_retrieve(query_text, coll, top_k * 2, filters, use_multistage,
                                           enable_dense=enable_dense, enable_bm25=enable_bm25,
                                           enable_splade=enable_splade), []

        def _run_raptor():
            if not self._raptor_retriever:
                return []
            try:
                return self._raptor_retriever.retrieve(query_text, coll, top_k=top_k, filters=filters)
            except Exception as exc:
                pipeline_logger.warning("RAPTOR retrieval failed", error=str(exc))
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_main   = executor.submit(_run_triple if use_triple else _run_fallback)
            fut_raptor = executor.submit(_run_raptor)
            (results, graph_paths) = fut_main.result()
            raptor_results         = fut_raptor.result()

        # Merge RAPTOR results (de-dup by id, tag lineage, then sort)
        if raptor_results:
            existing_ids = {r.document.id for r in results}
            for r in raptor_results:
                if r.document.id not in existing_ids:
                    lin = list(r.document.metadata.get("_method_lineage", []))
                    lin.append({"method": "RAPTOR", "rank": r.rank, "rrf_contribution": round(float(r.score), 6)})
                    r.document.metadata["_method_lineage"] = lin
                    results.append(r)
                    existing_ids.add(r.document.id)
            results = sorted(results, key=lambda x: x.score, reverse=True)

        # ── Parent expansion (Small-to-Big) ────────────────────────────────────
        # Each retrieved child chunk is replaced with its parent chunk so that the
        # LLM receives the full logical block (section/paragraph) instead of just
        # the matching 256-char window.  Parent text is taken from ChunkRegistry
        # (rebuilt from the vector store on startup via _rebuild_from_store).
        # If the registry is empty (e.g. adapter has no fetch_all) expansion is
        # a no-op — child chunks are returned as-is.
        if self._multistage and self._multistage.expander:
            _before = len(results)
            results = self._multistage.expander.expand(results)
            if len(results) != _before or any(
                r.document.metadata.get("expanded_from") for r in results
            ):
                self._last_retrieval_trace.append({
                    "method": "Parent Expansion",
                    "candidates_before": _before,
                    "candidates_after": len(results),
                })

        # Trace: initial retrieval candidates
        self._last_retrieval_trace.append({"method": "Initial Retrieval", "candidates_before": 0, "candidates_after": len(results)})

        # ── Sequential post-processing (each step depends on previous) ─────────

        # Phase 8: temporal + classification
        _before = len(results)
        results = self.temporal_filter.filter(results)
        if user_context:
            from core.temporal.filters import AccessPolicyEvaluator
            policy = AccessPolicyEvaluator(
                user_clearance=user_context.get("clearance", "INTERNAL")
            )
            results = policy.filter(results, user_context)
        else:
            results = self.clf_filter.filter(results)
        self._last_retrieval_trace.append({"method": "Temporal + Access Filter", "candidates_before": _before, "candidates_after": len(results)})

        # Phase 9: noise control
        _before = len(results)
        results, _ = self.noise_pipeline.run(results)
        self._last_retrieval_trace.append({"method": "Noise Filter", "candidates_before": _before, "candidates_after": len(results)})

        # Phase 11c: Cross-Encoder rerank (applied when available and enabled)
        if enable_rerank and self._multistage and self._multistage.reranker:
            _before = len(results)
            try:
                results = self._multistage.reranker.rerank(query_text, results, top_k=top_k * 2)
                for r in results:
                    pp = r.document.metadata.get("_post_processors", [])
                    if "Cross-Encoder Rerank" not in pp:
                        pp.append("Cross-Encoder Rerank")
                    r.document.metadata["_post_processors"] = pp
                self._last_retrieval_trace.append({"method": "Cross-Encoder Rerank", "candidates_before": _before, "candidates_after": len(results)})
            except Exception as exc:
                pipeline_logger.warning("Cross-Encoder rerank failed, skipping", error=str(exc))

        # Phase 11d: MMR diversity re-ranking (applied before final top_k trim)
        _before = len(results)
        if self._mmr_reranker:
            results = self._mmr_reranker.rerank(results, top_k=top_k)
            for r in results:
                pp = r.document.metadata.get("_post_processors", [])
                if "MMR Diversity" not in pp:
                    pp.append("MMR Diversity")
                r.document.metadata["_post_processors"] = pp
            self._last_retrieval_trace.append({"method": "MMR Diversity", "candidates_before": _before, "candidates_after": len(results)})
        else:
            results = results[:top_k]
            self._last_retrieval_trace.append({"method": "Top-K Trim", "candidates_before": _before, "candidates_after": len(results)})

        self._last_graph_paths = graph_paths
        return results


    def _fallback_retrieve(self, query_text, coll, top_k, filters, use_multistage,
                           enable_dense: bool = True, enable_bm25: bool = True,
                           enable_splade: bool = True):
        """Hybrid fallback when graph is not available."""
        if self._hybrid_retriever and not (enable_dense and enable_bm25):
            return self._hybrid_retriever.search(
                query=query_text, top_k=top_k, filters=filters,
                enable_dense=enable_dense, enable_bm25=enable_bm25,
                enable_splade=enable_splade,
            )
        if use_multistage and self._multistage:
            results = self._multistage.retrieve(query_text, top_k=top_k, filters=filters)
            # Tag Cross-Encoder rerank as post-processor (MultiStageRetriever always applies it)
            for r in results:
                pp = r.document.metadata.get("_post_processors", [])
                if "Cross-Encoder Rerank" not in pp:
                    pp.append("Cross-Encoder Rerank")
                r.document.metadata["_post_processors"] = pp
            return results
        q_vec = self.embedder.embed_one(query_text)
        return self.store.query(coll, q_vec, top_k=top_k, filters=filters)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        user_context: Optional[Dict] = None,
    ) -> "RAGResponse":
        """
        Full 9-phase RAG: query intelligence → retrieval → provenance → confidence → answer.
        Gracefully degrades if LM Studio is offline.
        """
        from orchestrator.response import RAGResponse

        cid = pipeline_logger.new_correlation_id()
        t_start = time.perf_counter()

        # Phase B: clear LLM traces for this request
        if self._llm_client:
            self._llm_client.begin_request()

        # Phase 5: query intelligence
        rewritten = question
        bundle = None
        lm_available = self._llm_client and self._llm_client.is_available()
        if lm_available and self._query_pipeline:
            try:
                bundle = self._query_pipeline.process(question)
                rewritten = bundle.primary_query
            except Exception as e:
                pipeline_logger.warning("Query intelligence skipped", error=str(e))

        # Core retrieval (also sets self._last_graph_paths)
        self._last_graph_paths = []
        results = self.query(
            rewritten, top_k=top_k, filters=filters, user_context=user_context
        )
        graph_paths = list(self._last_graph_paths)

        # Multi-query RRF if bundle available
        if bundle and bundle.paraphrases and len(bundle.paraphrases) > 1 and lm_available:
            from core.retrieval.hybrid import HybridFuser
            extra = []
            for para in bundle.paraphrases[1:]:
                para_res = self.query(para, top_k=top_k, filters=filters, user_context=user_context)
                extra.extend(para_res)
            if extra:
                fuser = HybridFuser()
                results = fuser.fuse(results, extra[:top_k], top_k=top_k)

        # Phase 11c: Contextual LLM re-ranking (one batched LLM call)
        if self._contextual_reranker and lm_available:
            results = self._contextual_reranker.rerank(question, results)

        # Phase 6: provenance + citations
        from core.provenance.models import build_provenance, CitationBuilder
        provenance = [
            build_provenance(r, self._chunk_registry, self.version_registry)
            for r in results
        ]
        citations = CitationBuilder.build_all(provenance, style="footnote")

        # Phase 7: confidence
        confidence = self.confidence_engine.assess(question, results)

        # Phase 5: LLM answer generation
        answer = ""
        if lm_available:
            context_text = "\n\n".join(
                f"[{i+1}] {r.document.text[:600]}" for i, r in enumerate(results[:5])
            )
            # Load answer generation prompt from registry
            try:
                from core.prompt_registry import PromptRegistry
                system = PromptRegistry.instance().get_prompt("answer_generation")
            except Exception:
                system = (
                    "You are a precise, trustworthy assistant. "
                    "Answer the question using ONLY the provided context passages. "
                    "If the context is insufficient, say 'Insufficient evidence'. "
                    "Cite passage numbers [1], [2], etc. in your answer."
                )
            prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            try:
                answer = self._llm_client.complete(prompt, system=system,
                                                   trace_method="Answer Generation")
                if self._context_tracker:
                    self._context_tracker.add_turn(question, answer)
            except Exception as e:
                pipeline_logger.warning("LLM answer failed", error=str(e))
                answer = f"[LLM error] {results[0].document.text[:300] if results else 'No results.'}"
        else:
            answer = (
                f"[LM Studio offline — retrieval only]\n"
                + "\n\n".join(f"[{i+1}] {r.document.text[:300]}" for i, r in enumerate(results[:3]))
                if results else "No results found."
            )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000)
        metrics.record("ask.latency_ms", elapsed_ms)
        metrics.increment("ask.requests")
        pipeline_logger.info(
            "ask completed",
            correlation_id=cid, elapsed_ms=elapsed_ms,
            result_count=len(results), confidence=confidence.verdict,
        )

        # Graph entities detected in question
        graph_entities: List[str] = []
        if self._graph_extractor and self._graph_extractor.is_available():
            try:
                ents = self._graph_extractor.extract_entities_only(question)
                graph_entities = [f"{e.entity_type}:{e.text}" for e in ents]
            except Exception:
                pass

        # Collect LLM traces captured during this request
        llm_traces = self._llm_client.get_traces() if self._llm_client else []

        return RAGResponse(
            query=question, answer=answer, results=results,
            provenance=provenance, citations=citations,
            confidence=confidence, rewritten_query=rewritten,
            graph_entities=graph_entities,
            graph_paths=graph_paths,
            backend=self.config["store"]["backend"],
            pipeline_metadata={"elapsed_ms": elapsed_ms, "top_k": top_k},
            llm_traces=llm_traces,
        )

    # ── Status ────────────────────────────────────────────────────────────────

    def ask_with_bundle(
        self,
        question: str,
        bundle,                          # pre-computed QueryBundle (from expand_query)
        expansion_traces: list,          # LLMTraceEntry list from shared expansion
        top_k: int = 5,
        filters: Optional[Dict] = None,
        user_context: Optional[Dict] = None,
        enable_dense: bool = True,
        enable_bm25: bool = True,
        enable_splade: bool = True,
        enable_graph: bool = True,
        enable_rerank: bool = True,
        enable_llm_graph: bool = False,
    ) -> "RAGResponse":
        """
        Retrieval + re-ranking + answer using a pre-computed QueryBundle.
        Skips query expansion LLM calls (already done once, shared across backends).
        Appends expansion_traces + backend-specific traces to the response.
        """
        from orchestrator.response import RAGResponse

        cid = pipeline_logger.new_correlation_id()
        t_start = time.perf_counter()

        # Start fresh backend-specific traces (contextual rerank + answer)
        if self._llm_client:
            self._llm_client.begin_request()

        lm_available = self._llm_client and self._llm_client.is_available()
        rewritten = bundle.primary_query if bundle else question

        # Core retrieval
        self._last_graph_paths = []
        results = self.query(rewritten, top_k=top_k, filters=filters, user_context=user_context,
                             enable_dense=enable_dense, enable_bm25=enable_bm25,
                             enable_splade=enable_splade,
                             enable_graph=enable_graph, enable_rerank=enable_rerank,
                             enable_llm_graph=enable_llm_graph)
        graph_paths = list(self._last_graph_paths)

        # Multi-query RRF using pre-computed paraphrases
        if bundle and bundle.paraphrases and len(bundle.paraphrases) > 1:
            from core.retrieval.hybrid import HybridFuser
            extra = []
            for para in bundle.paraphrases[1:]:
                para_res = self.query(para, top_k=top_k, filters=filters, user_context=user_context,
                                     enable_dense=enable_dense, enable_bm25=enable_bm25,
                                     enable_splade=enable_splade,
                                     enable_graph=enable_graph, enable_rerank=enable_rerank,
                                     enable_llm_graph=enable_llm_graph)
                # Tag paraphrase results as Multi-Query before fusion
                for r in para_res:
                    lin = list(r.document.metadata.get("_method_lineage", []))
                    if not any(e.get("method") == "Multi-Query" for e in lin):
                        lin.append({"method": "Multi-Query", "rank": r.rank, "rrf_contribution": round(float(r.score), 6)})
                    r.document.metadata["_method_lineage"] = lin
                extra.extend(para_res)
            if extra:
                fuser = HybridFuser()
                results = fuser.fuse(results, extra[:top_k], top_k=top_k)

        # Contextual re-ranking (backend-specific — operates on this backend's chunks)
        if self._contextual_reranker and lm_available:
            results = self._contextual_reranker.rerank(question, results)
            for r in results:
                pp = r.document.metadata.get("_post_processors", [])
                if "Contextual Rerank" not in pp:
                    pp.append("Contextual Rerank")
                r.document.metadata["_post_processors"] = pp

        # Provenance + citations
        from core.provenance.models import build_provenance, CitationBuilder
        provenance = [
            build_provenance(r, self._chunk_registry, self.version_registry)
            for r in results
        ]
        citations = CitationBuilder.build_all(provenance, style="footnote")

        # Confidence
        confidence = self.confidence_engine.assess(question, results)

        # Answer generation (backend-specific — uses this backend's retrieved chunks)
        answer = ""
        if lm_available:
            context_text = "\n\n".join(
                f"[{i+1}] {r.document.text[:600]}" for i, r in enumerate(results[:5])
            )
            try:
                from core.prompt_registry import PromptRegistry
                system = PromptRegistry.instance().get_prompt("answer_generation")
            except Exception:
                system = (
                    "You are a precise, trustworthy assistant. "
                    "Answer the question using ONLY the provided context passages. "
                    "If the context is insufficient, say 'Insufficient evidence'. "
                    "Cite passage numbers [1], [2], etc. in your answer."
                )
            prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            try:
                answer = self._llm_client.complete(prompt, system=system, trace_method="Answer Generation")
                if self._context_tracker:
                    self._context_tracker.add_turn(question, answer)
            except Exception as e:
                pipeline_logger.warning("LLM answer failed", error=str(e))
                answer = f"[LLM error] {results[0].document.text[:300] if results else 'No results.'}"
        else:
            answer = (
                "[LM Studio offline — retrieval only]\n"
                + "\n\n".join(f"[{i+1}] {r.document.text[:300]}" for i, r in enumerate(results[:3]))
                if results else "No results found."
            )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000)
        metrics.record("ask.latency_ms", elapsed_ms)
        metrics.increment("ask.requests")

        # Graph entities
        graph_entities: List[str] = []
        if self._graph_extractor and self._graph_extractor.is_available():
            try:
                ents = self._graph_extractor.extract_entities_only(question)
                graph_entities = [f"{e.entity_type}:{e.text}" for e in ents]
            except Exception:
                pass

        # Merge shared expansion traces + backend-specific traces
        backend_traces = self._llm_client.get_traces() if self._llm_client else []
        all_traces = list(expansion_traces) + backend_traces

        return RAGResponse(
            query=question, answer=answer, results=results,
            provenance=provenance, citations=citations,
            confidence=confidence, rewritten_query=rewritten,
            graph_entities=graph_entities,
            graph_paths=graph_paths,
            backend=self.config["store"]["backend"],
            pipeline_metadata={"elapsed_ms": elapsed_ms, "top_k": top_k},
            llm_traces=all_traces,
        )

    def expand_query(self, question: str):
        """
        Run query expansion LLM calls (rewrite, HyDE, multi-query, step-back).
        Returns (QueryBundle, list[LLMTraceEntry]) — shareable across all backends.
        """
        from core.query.llm_client import LLMTraceEntry as TraceEntry

        if self._llm_client:
            self._llm_client.begin_request()

        bundle = None
        lm_available = self._llm_client and self._llm_client.is_available()
        if lm_available and self._query_pipeline:
            try:
                bundle = self._query_pipeline.process(question)
            except Exception as e:
                pipeline_logger.warning("Query expansion skipped", error=str(e))

        traces = self._llm_client.get_traces() if self._llm_client else []
        return bundle, traces

    def health(self) -> dict:
        lm_ok = self._llm_client.is_available() if self._llm_client else False
        h = {
            "store_healthy": self.store.health_check() if self._started else False,
            "backend": self.config["store"]["backend"],
            "collection": self.ingestor.collection_name,
            "embedding_model": self.config.get("embedding", {}).get("model", "unknown"),
            "embedding_dim": self.embedder.embedding_dim,
            "bm25_docs_indexed": len(self.bm25_index),
            "lm_studio_available": lm_ok,
            "lm_studio_url": self.config.get("llm", {}).get("base_url", "http://localhost:1234/v1"),
            "metrics": metrics.summary(),
        }
        if self._graph_store:
            h["graph_entities"] = self._graph_store.entity_count()
            h["graph_relations"] = self._graph_store.relation_count()
            h["graph_backend"] = self.config.get("graph", {}).get("backend", "none")
        return h
