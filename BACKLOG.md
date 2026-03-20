# Phoenice-PolyRAG – Development Backlog

> **Vision**: A trustworthy, enterprise-grade RAG orchestration layer that works seamlessly
> across Weaviate, Milvus, ChromaDB, Qdrant, PGVector, and FAISS — with zero changes to
> orchestration code when switching vector backends.

---

## Legend
- `[ ]` Not started  |  `[x]` Complete  |  `[~]` In progress  |  `[!]` Blocked

---

## Phase 1 – Foundation: Unified Vector Store Interface + Basic Ingestion
> **Goal**: Establish the abstract adapter contract and working adapters for all 6 backends.
> Every subsequent phase builds on this contract.

### Design
- [x] Define `VectorStoreBase` abstract interface (Python ABC)
  - Methods: `connect()`, `create_collection()`, `upsert()`, `delete()`, `query()`, `health_check()`, `close()`
  - Standard `Document` and `SearchResult` data models (dataclasses / Pydantic)
- [x] Define `AdapterRegistry` — runtime selection by config key

### Adapters
- [x] **ChromaDB** adapter (local-first, easiest to get started)
- [x] **FAISS** adapter (in-memory, no server required)
- [x] **Qdrant** adapter (gRPC + REST)
- [x] **Weaviate** adapter (GraphQL schema + batch ingest)
- [x] **Milvus** adapter (pymilvus, partition-aware)
- [x] **PGVector** adapter (psycopg2/asyncpg + pgvector extension)

### Basic Ingestion Pipeline
- [x] `Ingestor` class: load raw text/file → embed → upsert via adapter
- [x] Embedding provider abstraction (sentence-transformers, local)
- [x] Batch upsert with progress tracking
- [x] Collection/index lifecycle management (create, reset, drop)

### Configuration
- [x] `config.yaml` schema for backend selection + connection params

### Tests – Phase 1
- [x] `tests/phase1/test_adapter_contract.py` — parametrized across all adapters ✅
- [x] `tests/phase1/test_ingestor.py` — end-to-end ingest of sample corpus ✅
- [x] `tests/phase1/test_adapter_registry.py` — registry lookup by key ✅

**✅ STATUS: COMPLETE** — 98 tests passing (81 contract + 17 ingestor), ChromaDB / FAISS / Qdrant
fully in-memory; Weaviate / Milvus / PGVector adapters implemented, tagged `@integration`.

---

## Phase 2 – Semantic Chunking Pipeline
> **Goal**: Replace naive splitting with meaning-preserving chunking that maintains
> provenance links from child chunks back to parent sections and source documents.

### Chunking Strategies
- [x] `ChunkerBase` abstract interface: `chunk(document) → List[Chunk]`
- [x] **Fixed-overlap chunker** — baseline (size + stride)
- [x] **Sentence-boundary chunker** — regex sentence segmentation
- [x] **Section-aware chunker** — Markdown / ALL-CAPS / drama heading detection
- [x] **Semantic boundary chunker** — embedding similarity breakpoint detection
- [x] **Parent-child chunker** — stores parent chunk, indexes child chunks with `parent_id`

### Chunk Data Model
- [x] `Chunk` model: `chunk_id`, `parent_id`, `doc_id`, `text`, `start_char`, `end_char`,
  `section_title`, `page_num`, `chunk_type`, `metadata`
- [x] `ChunkRegistry` — maps `chunk_id` → full chunk (for parent retrieval)

### Pipeline
- [x] `ChunkingPipeline` — configurable chain of chunkers + post-processors
- [x] Deduplication fingerprint on chunk text (SHA-256)
- [x] Chunk quality gate: min/max token length

### Tests – Phase 2
- [x] `tests/phase2/test_chunkers.py` — all strategies tested on Shakespeare corpus ✅

**✅ STATUS: COMPLETE** — 20/20 tests passing

---

## Phase 3 – Hybrid Search (Semantic + BM25 + Metadata Filters)
> **Goal**: Every query uses all available signal types — dense vectors, sparse keyword,
> and structured metadata — fused into a single ranked result list.

### Hybrid Search Components
- [x] `BM25Index` — rank-bm25 powered keyword search (universal fallback)
- [x] `MetadataFilter` — structured predicate on document metadata
- [x] `HybridFuser` — Reciprocal Rank Fusion (RRF, k=60)
- [x] `HybridRetriever` — vector + BM25 → RRF → metadata filter

### Tests – Phase 3
- [x] `tests/phase3/test_hybrid_search.py` — BM25, RRF, metadata filter, E2E ✅

**✅ STATUS: COMPLETE** — 19/19 tests passing

---

## Phase 4 – Multi-Stage Retrieval & Cross-Encoder Re-ranking
> **Goal**: Two-stage pipeline: broad ANN recall → precision re-ranking, with parent
> context expansion and cross-document merging.

### Components
- [x] `CrossEncoderReRanker` — `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [x] `ParentExpander` — fetch parent chunk for child-chunk hits
- [x] `CrossDocumentAggregator` — group hits by document, remove near-duplicate spans
- [x] `MultiStageRetriever` — recall → expand → rerank → filter → dedup → top-k

### Tests – Phase 4
- [x] `tests/phase4/test_multistage.py` — all components tested ✅

**✅ STATUS: COMPLETE** — 12/12 tests passing

---

## Phase 5 – Query Intelligence (Rewriting, Expansion, Context-Aware)
> **Goal**: Transform raw user queries into retrieval-optimized forms; use conversation
> history for context-aware retrieval. Uses LM Studio (mistralai/ministral-3b @ localhost:1234).

### Components
- [x] `LMStudioClient` — OpenAI-compatible client for local LLM (localhost:1234)
- [x] `QueryRewriter` — LLM-based ambiguity removal
- [x] `QueryExpander` — HyDE (Hypothetical Document Embeddings)
- [x] `MultiQueryGenerator` — N paraphrases → retrieve → RRF union
- [x] `ConversationContextTracker` — multi-turn history
- [x] `ContextualQueryBuilder` — inject history into current query
- [x] `QueryIntelligencePipeline` — full pipeline with graceful LLM degradation

### Tests – Phase 5
- [x] `tests/phase5/test_query_intelligence.py` — unit tests (LLM-dependent skipped if offline) ✅

**✅ STATUS: COMPLETE** — 8/8 non-lmstudio tests passing; LM Studio tests auto-skip when offline

---

## Phase 6 – Provenance & Traceability (NON-NEGOTIABLE)
> **Goal**: Every answer atom is traceable to an exact text span in a versioned document.

### Components
- [x] `ProvenanceRecord` model with full lineage fields
- [x] `CitationBuilder` — APA, inline, footnote styles
- [x] `SpanHighlighter` — marks exact retrieved spans in source text
- [x] `IngestionAuditLog` — append-only JSONL log (thread-safe)
- [x] `DocumentVersionRegistry` — tracks document versions, flags superseded
- [x] `build_provenance()` helper — SearchResult → ProvenanceRecord

### Tests – Phase 6
- [x] `tests/phase6/test_provenance.py` — full provenance chain tested ✅

**✅ STATUS: COMPLETE** — 18/18 tests passing

---

## Phase 7 – Confidence Estimation & Quality Signals
> **Goal**: Users always know how much to trust each answer.

### Confidence Signals
- [x] `RetrievalScoreAnalyser` — mean, std, min, max of retrieval scores
- [x] `SourceAgreementScorer` — semantic similarity across top-k sources
- [x] `QuestionCoverageScorer` — fraction of query terms covered by retrieved text
- [x] `MissingEvidenceFlagger` — detect when no chunk scores above threshold
- [x] `ConflictDetector` — flag chunks with contradictory polarity
- [x] `AnswerConfidenceAggregator` — weighted composite [0.0, 1.0] score
- [x] `ConfidenceReport` — all signals + verdict: HIGH/MEDIUM/LOW/INSUFFICIENT_EVIDENCE

### Tests – Phase 7
- [x] `tests/phase7/test_confidence.py` — all signals tested ✅

**✅ STATUS: COMPLETE** — 19/19 tests passing

---

## Phase 8 – Temporal Relevance, Versioning & Data Classification
> **Goal**: The system understands document lifecycle and data sensitivity.

### Components
- [x] `TemporalMetadata` model: `created_at`, `effective_date`, `expiry_date`, `superseded_by`
- [x] `TemporalFilter` — exclude expired/superseded chunks at query time
- [x] `TemporalRanker` — boost recency-weighted score
- [x] `ClassificationLabel` enum: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
- [x] `ClassificationFilter` — enforce label-based access at retrieval
- [x] `ClassificationPropagator` — child chunks inherit parent classification
- [x] `AccessPolicyEvaluator` — RBAC policy engine

### Tests – Phase 8
- [x] `tests/phase8/test_temporal_classification.py` — all filters tested ✅

**✅ STATUS: COMPLETE** — 24/24 tests passing

---

## Phase 9 – Noise Control, Cross-Document Aggregation & Production Hardening
> **Goal**: Filter junk, deduplicate, detect conflicts, production-ready observability.

### Components
- [x] `DuplicateDetector` — MinHash LSH (datasketch) + SHA-256 hash fallback
- [x] `QualityScorer` — word count, punctuation ratio, repetition rate, coherence heuristics
- [x] `ConflictResolver` — detect contradictory passages, surface both views
- [x] `NoiseFilterPipeline` — quality filter → dedup → conflict resolution
- [x] `StructuredLogger` — JSON logging with correlation IDs
- [x] `PipelineMetrics` — latency tracking per stage

### Tests – Phase 9
- [x] `tests/phase9/test_noise_production.py` — noise pipeline + observability ✅

**✅ STATUS: COMPLETE** — 23/23 tests passing

---

## End-to-End Integration
- [x] `orchestrator/pipeline.py` — full 9-phase `RAGPipeline` wiring all components
- [x] `orchestrator/response.py` — `RAGResponse` envelope
- [x] `tests/test_pipeline_e2e.py` — complete E2E integration tests ✅

---

## Cross-Cutting Concerns (tracked throughout all phases)

| Concern | Owner Phase | Status |
|---|---|---|
| Pydantic v2 data models for all contracts | Phase 1 | `[x]` |
| Type hints + mypy strict mode | Phase 1 | `[x]` |
| `requirements.txt` + `pyproject.toml` | Phase 1 | `[x]` |
| Docker Compose for all 6 backends (local dev) | Phase 1 | `[ ]` |
| GitHub Actions CI pipeline | Phase 1 | `[ ]` |
| Adapter feature matrix documentation | Phase 3 | `[x]` |
| RAGAS evaluation harness integration | Phase 4 | `[ ]` |
| End-to-end integration test suite | Phase 9 | `[x]` |
| Security review (injection, data leakage) | Phase 9 | `[ ]` |
| Load test (Locust / k6) | Phase 9 | `[ ]` |

---

## Adapter Feature Matrix

| Feature | ChromaDB | FAISS | Qdrant | Weaviate | Milvus | PGVector |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Native BM25 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ (tsvector) |
| Native Hybrid Search | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Metadata Filters | ✅ | ✅* | ✅ | ✅ | ✅ | ✅ |
| Persistent Storage | ✅ | ❌* | ✅ | ✅ | ✅ | ✅ |
| Distributed / Cluster | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ (Citus) |
| RBAC / Auth | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ (pg roles) |
| Sparse Vectors (SPLADE) | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |

> ✅ = native  |  ❌ = emulated by orchestration layer  |  ✅* = with index serialization

---

## Evaluation Benchmarks (tracked per phase)

| Metric | Phase Introduced | Target |
|---|---|---|
| Adapter contract pass rate | Phase 1 | 100% |
| Chunk span roundtrip accuracy | Phase 2 | 100% |
| Hybrid MRR vs. semantic-only | Phase 3 | ≥ +5% |
| NDCG@10 improvement (re-rank) | Phase 4 | ≥ +10% |
| Multi-query recall@10 improvement | Phase 5 | ≥ +8% |
| Provenance completeness | Phase 6 | 100% |
| Confidence-quality Spearman r | Phase 7 | ≥ 0.70 |
| Zero expired/denied retrieval | Phase 8 | 100% |
| p99 query latency | Phase 9 | < 3s |

---

## Phase 10 – Knowledge Graph Integration (GraphRAG)
> **Goal**: Augment the 2-way hybrid search (vector + BM25) with a third signal: Knowledge Graph traversal,
> forming a **3-way hybrid** that reduces hallucinations and surfaces relational context.
>
> Transform unstructured text → structured knowledge graph of nodes (entities) and edges (relations),
> then traverse this graph at query time to find chunks connected through entity relationships.

### Graph Store Layer (mirrors VectorStoreBase adapter pattern)
- [x] `GraphStoreBase` ABC — `connect/close/clear/health_check/upsert_entity/upsert_relation/link_entity_to_chunk/get_entity/find_entities_by_text/get_neighbors/get_chunk_ids_for_entity/entity_count/relation_count`
- [x] `NetworkXGraphStore` — in-memory, pure Python, for tests and development
- [x] `KuzuGraphStore` — embedded persistent graph DB (Kuzu), production default
  - No server required (embedded like SQLite for graphs)
  - ACID transactions, persists across restarts
  - Uses Cypher query language (identical to Neo4j)
- [x] `Neo4jGraphStore` — enterprise migration stub, fully working once `pip install neo4j` + server running
  - Same Cypher queries as Kuzu → zero orchestration changes to switch
- [x] `GraphStoreRegistry` — factory identical to `AdapterRegistry`

### Entity & Relation Extraction (ingestion-time)
- [x] `EntityRelationExtractor` — spaCy NER + dependency-parse SVO triples + co-occurrence edges
  - Entity types: PERSON, ORG, LOCATION, DATE, EVENT, LAW, PRODUCT, CONCEPT
  - SVO triples: Subject →[verb]→ Object (confidence 0.8)
  - Co-occurrence: any 2 entities in same sentence →[co_occurs]→ (confidence 0.5)
  - Graceful degradation: `is_available()` returns False if spaCy not installed
- [x] Graph schema: `Entity` nodes + `Chunk` nodes + `APPEARS_IN` + `RELATES_TO` edges
- [x] Chunk text stored in graph (self-contained traversal, no round-trip to vector store)

### Query-Time Traversal
- [x] `GraphTraverser` — NER on query → resolve in graph → N-hop BFS → rank chunks
  - Hop scores: direct=1.0, 1-hop=0.6, 2-hop=0.3
  - Fuzzy entity resolution (substring fallback)
  - Returns `List[SearchResult]` + `List[GraphPath]` for explainability

### 3-Way RRF Fusion
- [x] `TripleHybridRetriever` — extends `HybridRetriever` with graph signal
  - `rrf_score(d) = 1/(k + rank_hybrid) + graph_weight/(k + rank_graph)`, k=60
  - Annotates `metadata["retrieval_signals"]` per result ("vector+bm25", "graph", or both)
  - Falls back to 2-way hybrid when graph has no entities

### Orchestrator Integration
- [x] `pipeline.py` — graph store + extractor wired in `start()`, extraction in `ingest_text()`, traversal in `query()`
- [x] `response.py` — added `graph_entities: List[str]` + `graph_paths: List[GraphPath]` fields + `graph_explanation()`
- [x] `config.yaml` — `graph:` section with backend, spacy_model, max_hops, graph_weight, kuzu path, neo4j URI
- [x] `requirements.txt` — `spacy>=3.7`, `networkx>=3.3`, `kuzu>=0.7.0`

### Neo4j Enterprise Migration Path
Switching from embedded Kuzu to enterprise Neo4j requires:
1. `pip install neo4j`
2. Change `config.yaml`: `graph.backend: neo4j`, fill in `graph.neo4j.uri/user/password`
3. Zero other changes — orchestration, Cypher queries, and API remain identical

### Tests – Phase 10
- [x] `tests/phase10/test_knowledge_graph.py` — 53 tests across all components ✅
  - Data models: Entity, Relation, Triple, GraphPath
  - NetworkXGraphStore: upsert, link, neighbors, chunk retrieval, 2-hop, idempotency
  - KuzuGraphStore: upsert, persist/reload, neighbors, chunk retrieval
  - EntityRelationExtractor: NER, SVO, co-occurrence, empty text
  - GraphTraverser: direct hit, 2-hop, paths, empty graph fallback
  - TripleHybridRetriever: 3-way fusion, signal annotation, boosted ranking, fallback
  - RAGResponse: graph_entities, graph_paths, summary, graph_explanation
  - Integration stub: Neo4jGraphStore importable, graceful connect failure

**✅ STATUS: COMPLETE** — 315 tests passing, 1 skipped (LM Studio integration auto-skips when offline)

---

## Phase 11 – Advanced Retrieval + LLM-Enhanced Extraction

> **Goal**: Add LLM-powered entity extraction, RAPTOR hierarchical indexing, batched LLM
> contextual re-ranking, and MMR diversity re-ranking to push retrieval quality toward
> structured-data-grade precision.

### LLM Entity/Relation Extraction (via LM Studio)
- [x] `core/graph/llm_extractor.py` — `LLMEntityExtractor`
  - JSON-prompt extraction: one LM Studio call per chunk, structured `{"entities": [...], "relations": [...]}`
  - Strips markdown code fences before JSON parse; unknown types → CONCEPT
  - Configurable `merge_with_spacy: true` — complements Phase 10 spaCy extraction
  - Truncates to `max_chunk_chars` (default 2000) to stay within context window
  - Disabled by default (`graph.llm_extraction.enabled: false`) — high quality, but adds latency

### RAPTOR Hierarchical Retrieval
- [x] `core/retrieval/raptor.py` — `RaptorIndexer` + `RaptorRetriever`
  - `RaptorIndexer.build()`: pure-numpy k-means (no scipy), LLM cluster summarization → upsert to `<collection>_raptor`
  - `RaptorRetriever.retrieve()`: searches both leaf + summary collections, marks summaries with `is_raptor_summary=True`
  - Pipeline: explicit `build_raptor_index()` call after ingestion; doc buffer accumulated in `ingest_text()`
  - Disabled by default (`advanced_retrieval.raptor.enabled: false`) — requires LLM + explicit call

### Contextual LLM Re-ranking
- [x] `core/retrieval/contextual_reranker.py` — `ContextualReranker`
  - Batched: ONE LLM call per query ranks all top-K candidates simultaneously
  - Score formula: `(n-1-rank) / max(n-1, 1)` → guaranteed [0, 1] range
  - Fused score: `llm_weight * llm_score + (1-llm_weight) * retrieval_score`
  - Falls back to original order on invalid/incomplete LLM response
  - Disabled by default (`advanced_retrieval.contextual_reranker.enabled: false`)

### MMR Diversity Re-ranking
- [x] `core/retrieval/mmr.py` — `MMRReranker`
  - Greedy selection: `λ * relevance - (1-λ) * max_sim_to_selected`
  - `diversity_weight` config (default 0.3 → λ=0.7, slight diversity boost)
  - Pure numpy, no LLM, no external deps — **enabled by default**
  - Falls back to top-k slice when embeddings unavailable

### LLM Client Enhancement
- [x] `core/query/llm_client.py` — per-call `max_tokens` / `temperature` overrides
  - Required for RAPTOR summarization (max_tokens=300) and contextual reranker (max_tokens=200)

### Already-Implemented Retrieval (Phase 5, documented here for completeness)
- [x] **HyDE** — `core/query/rewriter.py`: generate hypothetical document → embed as query
- [x] **Multi-Query** — `core/query/context.py`: 3 paraphrases + RRF fusion
- [x] **Query Rewriting** — `core/query/rewriter.py`: step-back + clarity rewrite
- [x] **Step-back Prompting** — `core/query/rewriter.py`: broader conceptual question

### Retrieval Method Summary (all active in pipeline)
| Method | Signal Type | LLM? | Notes |
|---|---|---|---|
| Dense vector | Semantic similarity | Embeddings | Always on |
| BM25 | Keyword overlap | No | Always on |
| Graph traversal | Entity relationships | No (spaCy) | Phase 10 |
| HyDE | Hypothetical document | Yes | Phase 5 |
| Multi-Query | Paraphrase ensemble | Yes | Phase 5 |
| Query Rewriting | Clarity + step-back | Yes | Phase 5 |
| RAPTOR | Hierarchical cluster | Yes | Phase 11 (opt-in) |
| Contextual Reranker | LLM batch ranking | Yes | Phase 11 (opt-in) |
| Cross-Encoder | Neural passage score | No | Phase 4 |
| MMR | Diversity selection | No | Phase 11 (default on) |

### Orchestrator Integration
- [x] `orchestrator/pipeline.py` — full 11-phase wiring
  - `query()`: 3-way hybrid → RAPTOR augment → temporal/classification filter → noise control → MMR
  - `ask()`: multi-query RRF → contextual LLM rerank → LLM answer generation
  - `build_raptor_index()`: explicit method for post-ingestion RAPTOR build
- [x] `config/config.yaml` — `advanced_retrieval:` section (raptor, contextual_reranker, mmr)

### Tests – Phase 11
- [x] `tests/phase11/test_advanced_retrieval.py` — 53 tests across all Phase 11 components ✅
  - LLMEntityExtractor: JSON parse, fence stripping, type normalization, merge mode, truncation
  - RaptorIndexer: k-means clustering, LLM summarization, upsert, empty corpus
  - RaptorRetriever: leaf+summary merge, summary flag, deduplication
  - ContextualReranker: score formula [0,1], batch parse, fallback on bad LLM response
  - MMRReranker: diversity selection, lambda param, no-embedding fallback
  - Pipeline integration: MMR default on, RAPTOR opt-in, contextual reranker opt-in

**✅ STATUS: COMPLETE** — 368 tests passing, 1 skipped (LM Studio integration auto-skips when offline)

---

## Phase 12 – Backend Comparison Suite

> **Goal**: Validate that all vector backends produce identical, dependable retrieval results
> for the same queries on the same corpus — proving the orchestration layer is truly backend-agnostic.

### Comparison Script
- [x] `scripts/compare_backends.py` — end-to-end multi-backend comparison runner
  - Auto-probes backend availability (no external servers needed for chromadb/faiss/qdrant)
  - Ingests a standard generic enterprise policy corpus (not Shakespeare-specific)
  - Runs 5 standard queries covering: factual retrieval, entity lookup, retention, incidents, enforcement
  - Collects per-query and aggregate metrics: latency, top score, mean score, keyword hit rate
  - Generates console tables via `tabulate` (per-query + aggregate + consistency analysis)
  - Produces `data/comparison_chart.png` (5-panel matplotlib chart)
  - Produces `data/comparison_results.json` (raw data for further analysis)
  - UTF-8 safe on Windows (`sys.stdout.reconfigure`)

### Bug Fixes (discovered during comparison)
- [x] `core/chunking/section_aware.py` — child chunk IDs now globally unique per document
  - Was: `{doc_id}::fixed::0` repeated for each section body (ChromaDB rejected as duplicate)
  - Fixed: `{doc_id}::section::{idx}::child::{j}` — unique across all sections
- [x] `core/retrieval/hybrid.py` — added `HybridRetriever.retrieve()` alias
  - Was: `TripleHybridRetriever` called `.retrieve()` but only `.search()` existed → fallback warning
  - Fixed: `retrieve()` delegates to `search()`, matching `TripleHybridRetriever`'s calling convention
- [x] `core/store/adapters/pgvector_adapter.py` — fixed index name syntax error
  - Was: `CREATE INDEX IF NOT EXISTS "collection"_vec_idx` (quoted identifier cannot be used as prefix)
  - Fixed: unquoted sanitised name `collection_vec_idx`
- [x] `core/store/adapters/pgvector_adapter.py` — fixed numpy array boolean ambiguity
  - Was: `list(vector) if vector else []` — fails for numpy arrays (psycopg2+pgvector returns ndarray)
  - Fixed: `list(vector) if vector is not None else []`

### Comparison Results (All 6 Backends — Docker server mode for Qdrant, Weaviate, Milvus, PGVector)

| Backend | Ingest(ms) | Chunks | Avg Latency | Avg Top Score | KW Hits | Errors |
|---|---|---|---|---|---|---|
| **chromadb** | 1552 | 20 | 461ms | 0.0326 | **5/5** | 0 |
| **faiss** | 1045 | 20 | 463ms | 0.0326 | **5/5** | 0 |
| **qdrant** | 2266 | 20 | 567ms | 0.0323 | **5/5** | 0 |
| **weaviate** | 3338 | 20 | 543ms | 0.0325 | **5/5** | 0 |
| **milvus** | 2229 | 20 | 683ms | 0.0326 | **5/5** | 0 |
| **pgvector** | 3003 | 20 | 515ms | 0.0322 | **5/5** | 0 |

**Key finding: ALL 6 backends return identical top chunks for all 5 queries.
Score variance across backends ≈ 0.000 (HIGH consistency). 100% keyword recall on all backends.**

### Usage
```powershell
# Compare all locally-available backends (chromadb, faiss, qdrant by default)
python scripts/compare_backends.py

# Compare specific backends:
python scripts/compare_backends.py --backends chromadb faiss qdrant

# Use your own document corpus:
python scripts/compare_backends.py --corpus path/to/your_document.txt

# Quiet mode (tables only):
python scripts/compare_backends.py --quiet --no-chart
```

### Tests – Phase 12
- [x] `tests/test_backend_comparison.py` — 52 tests ✅
  - `TestBackendIngestion`: ingestion succeeds, chunk count > 0, consistent chunk counts
  - `TestQueryRetrieval`: all queries return results, scores in valid range, keyword hit rate ≥ 3/5, latency < 30s
  - `TestCrossBackendConsistency`: score variance ≤ 0.1, keyword agreement majority, result counts consistent
  - `TestReportGeneration`: JSON report valid, chart PNG created (> 5KB), full script subprocess runs

**✅ STATUS: COMPLETE** — 420 tests passing, 1 skipped (LM Studio integration auto-skips when offline)

---

_Last updated: All 12 phases ✅ COMPLETE — **420 tests passing**, 1 skipped (LM Studio integration, auto-skips when offline)_

---

## Phase 13 – Frontend: React 19 + FastAPI Dashboard
> **Goal**: Browser-based orchestration UI for all RAG functions — ingestion, search, comparison,
> knowledge graph visualization, evaluation, document library — backed by a FastAPI REST/SSE API.

### Architecture
- **Frontend**: React 19 + Vite 5 + TypeScript, Tailwind CSS, shadcn/ui, Zustand, TanStack Query, React Flow 12, D3.js v7
- **API Server**: FastAPI + uvicorn, SSE streaming, background job queue (port 8000)
- **Routing**: React Router 6 — 8 pages accessible from sidebar nav

### P13-A: FastAPI Backend API
- [x] `api/main.py` — FastAPI app, CORS, all routers mounted at `/api`
- [x] `api/schemas.py` — Pydantic request/response models for all endpoints
- [x] `api/jobs.py` — Thread-safe in-memory job store (asyncio.Lock)
- [x] `api/deps.py` — Dependency injection: per-backend pipeline config, pipeline factory
- [x] `api/routers/ingest.py` — POST /api/ingest, GET /api/ingest/{id}/stream (SSE), GET /api/ingest/{id}/status
- [x] `api/routers/search.py` — POST /api/search (multi-backend, all 10 methods, retrieval trace)
- [x] `api/routers/compare.py` — POST /api/compare (comparison matrix)
- [x] `api/routers/backends.py` — GET /api/backends, health, collections CRUD
- [x] `api/routers/graph.py` — GET /api/graph/{collection} (D3-ready nodes + edges)
- [x] `api/routers/evaluate.py` — POST/GET /api/evaluate (faithfulness, relevance, source_hit scoring)
- [x] `api/routers/feedback.py` — POST/GET /api/feedback (relevance thumbs up/down)
- [x] `api/routers/chunks.py` — POST /api/chunks/preview (dry-run, no actual ingest)
- [x] `api/routers/jobs.py` — GET /api/jobs, GET /api/jobs/{id}

### P13-B: React 19 + Vite Scaffolding
- [x] `frontend/` — Vite 5 + React 19 + TypeScript project
- [x] Tailwind CSS v3 dark theme (bg-gray-950 / text-gray-100)
- [x] Zustand store — selectedBackends, retrievalMethods (10 toggles), backendStatuses
- [x] TanStack Query + Axios API client layer
- [x] React Flow 12 + D3.js v7
- [x] React Router 6 with 8-route sidebar nav

### P13-C: Shared Components
- [x] `BackendSelector` — multi-select with green/red/gray status badges
- [x] `MethodToggle` — 10 toggles grouped: "Always Available" (5) + "LLM-Required" (5)
- [x] `ResultCard` — score bar, chunk text, metadata accordion, provenance, thumbs up/down
- [x] `LogStream` — SSE live log panel, color-coded INFO/SUCCESS/ERROR/WARN, auto-scroll
- [x] `BackendHealthBar` — top-of-page live status bar, polls /api/backends every 10s
- [x] `RetrievalTrace` — collapsible audit panel with SVG funnel diagram
- [x] `IngestionFlow` — React Flow DAG (Upload → Chunk → Embed → KG → Upsert)

### P13-D through P13-J: Pages
- [x] **Ingestion Studio** — file/text upload, drag-and-drop batch queue, chunking config, preview modal, React Flow DAG, SSE log stream, per-backend status cards
- [x] **Search Lab** — query + history, 10-method toggles, per-backend result columns, A/B mode, retrieval trace panel, React Flow retrieval visualization
- [x] **Comparison Matrix** — live run, sortable table (base/full top score, KW hits, ingest time), CSV/JSON export
- [x] **Knowledge Graph** — D3.js force-directed graph, entity type colors, click→source chunks, collection selector
- [x] **Document Library** — tabbed by backend, collections with chunk count + health, document CRUD, re-ingest, delete
- [x] **Evaluation Studio** — ground-truth Q&A manager, auto-score per backend+method (faithfulness/relevance/source_hit), results matrix
- [x] **Job History** — persistent job log, status filter, log line expansion, clear completed
- [x] **Settings** — backend connection strings, LM Studio URL, embedding model, localStorage persistence

### 10 Value Additions (built-in from day 1)
- [x] **Evaluation Studio** — RAGAS-style scoring per backend × method matrix
- [x] **Chunking Preview** — visual dry-run before committing ingestion (color-coded chunk boundaries)
- [x] **Document Library** — full CRUD for all collections across all backends
- [x] **Retrieval Trace Panel** — step-by-step audit trail showing which methods fired + score funneling
- [x] **Relevance Feedback Loop** — thumbs up/down on ResultCard → POST /api/feedback
- [x] **A/B Config Testing** — SearchLab A/B mode: two configs side-by-side with diff view
- [x] **Backend Health Monitor** — BackendHealthBar with ping, vector counts, status badges
- [x] **Export / Reporting** — CSV + JSON export from ComparisonMatrix
- [x] **Multi-file Batch Ingestion Queue** — drag-and-drop multiple files, per-file progress bars
- [x] **Notifications / Job History** — JobHistory page + persistent job log

### Tests – Phase 13
- [x] `tests/phase13/test_api.py` — 17 API integration tests ✅
  - GET /api/health, /api/backends, /api/jobs
  - POST /api/chunks/preview (dry-run chunking)
  - POST /api/ingest (FAISS backend, poll to completion)
  - POST /api/search (FAISS backend, results with trace)
  - POST /api/feedback (store entry)
- [x] `frontend/src/test/` — 29 Vitest + React Testing Library tests ✅
  - BackendSelector: renders 6 backends, toggles selection, status badges
  - MethodToggle: renders 10 methods, "Always Available" + "LLM-Required" groups, toggle
  - ResultCard: chunk text, score bar, metadata, expand/collapse, thumbs up/down
  - LogStream: renders lines, color-codes INFO/ERROR/WARN
  - ComparisonMatrix: table headers, sortable columns

### Running the UI
```powershell
# Terminal 1: Start API server
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start React UI
cd frontend
npm run dev   # → http://localhost:3000

# Terminal 3 (optional): Start vector DB containers
docker compose -f docker-compose.polyrag.yml up -d
```

**✅ STATUS: COMPLETE** — 420 backend tests passing + 17 API tests + 29 frontend tests

---
_Last updated: Phase 13 ✅ COMPLETE — **466 tests total** (420 backend + 17 API integration + 29 frontend)_
