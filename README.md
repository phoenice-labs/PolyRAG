# Phoenice-PolyRAG

> **Enterprise-grade, trustworthy RAG with a unified orchestration layer
> that works seamlessly across six vector store backends — zero code changes to switch.**

---

## What Is This?

Phoenice-PolyRAG is a production-ready Retrieval-Augmented Generation (RAG) framework built entirely on **open-source** tooling. Its core design principle:

> *Write your RAG orchestration once. Run it on any vector store.*

Switch from ChromaDB (local dev) to Weaviate (production cluster) by changing **one line in config.yaml**.

---

## Supported Vector Backends

| Backend | Local / No-Server | Native Hybrid Search | Persistence |
|---|:---:|:---:|:---:|
| **ChromaDB** | ✅ in-memory | ❌ emulated | ✅ |
| **FAISS** | ✅ in-memory | ❌ emulated | ✅ (file) |
| **Qdrant** | ✅ in-memory | ✅ native | ✅ |
| **Weaviate** | ✅ embedded | ✅ native | ✅ |
| **Milvus** | ✅ Lite (Linux/macOS) | ✅ native | ✅ |
| **PGVector** | ❌ needs PostgreSQL | ✅ (tsvector) | ✅ |

---

## All 11 Phases — COMPLETE ✅

| Phase | Feature | Tests |
|---|---|:---:|
| **1** | Unified vector store interface, 6 adapters, embedding abstraction, basic ingestion | 98 ✅ |
| **2** | Semantic chunking: fixed-overlap, sentence-boundary, section-aware (parent-child), semantic boundary | 20 ✅ |
| **3** | Hybrid search: vector + BM25 (rank-bm25) + metadata filters + RRF fusion | 19 ✅ |
| **4** | Multi-stage retrieval: broad recall → parent expansion → cross-encoder re-ranking | 12 ✅ |
| **5** | Query intelligence: rewriting, HyDE expansion, multi-query, step-back, conversation context | 8 ✅ |
| **6** | Provenance & traceability: citations, exact text spans, audit log, version registry | 18 ✅ |
| **7** | Confidence estimation: score distribution, source agreement, coverage, conflict detection | 19 ✅ |
| **8** | Temporal relevance, versioning, data classification (RBAC), access policy | 24 ✅ |
| **9** | Noise control (MinHash dedup, quality scoring), cross-doc aggregation, observability | 23 ✅ |
| **E2E** | Full pipeline integration: ingest → hybrid retrieve → rerank → answer → provenance | 13 ✅ |
| **10** | Knowledge graph (GraphRAG): entity/relation extraction, 3-way hybrid search (vector+BM25+graph) | 53 ✅ |
| **11** | RAPTOR hierarchical retrieval, LLM entity extraction, contextual re-ranking, MMR diversity | 53 ✅ |

**Total: 368 tests | 368 passing | 1 skipped (LM Studio auto-skip when offline)**

---

## Quick Start

### 1 — Install (Windows)

```powershell
git clone <repo-url> Phoenice-PolyRAG
cd Phoenice-PolyRAG
.\install.ps1
```

For optional backends (Weaviate, PGVector):
```powershell
.\install.ps1 -Full
```

### 2 — Activate environment

```powershell
.\start.ps1
```

### 3 — Run all tests

```powershell
# All phases (no external services required)
pytest tests/ -v -m "not integration and not lmstudio"

# With LM Studio running (localhost:1234)
pytest tests/ -v -m "not integration"

# Specific phase
pytest tests/phase2/ -v
```

---

## LM Studio Setup (Query Intelligence)

Phase 5 query rewriting and HyDE expansion require a local LLM via [LM Studio](https://lmstudio.ai/).

1. Download and install LM Studio
2. Load model: **mistralai/ministral-3b** (or any compatible model)
3. Start the local server on port **1234** (default)
4. Configure in `config/config.yaml`:

```yaml
llm:
  base_url: http://localhost:1234/v1
  model: mistralai/ministral-3b
  api_key: lm-studio   # dummy key required by OpenAI SDK
  temperature: 0.1
  max_tokens: 512
```

> **Graceful degradation**: if LM Studio is offline, the pipeline still works — query
> rewriting is skipped, raw queries are used directly for retrieval.

---

## Project Structure

```
Phoenice-PolyRAG/
├── core/
│   ├── store/
│   │   ├── base.py              ← VectorStoreBase abstract interface
│   │   ├── models.py            ← Document, SearchResult, CollectionInfo
│   │   ├── registry.py          ← AdapterRegistry (factory by key)
│   │   └── adapters/            ← 6 backend adapters
│   ├── embedding/
│   │   └── sentence_transformer.py   ← all-MiniLM-L6-v2, 384-dim, CPU
│   ├── ingestion/
│   │   └── ingestor.py          ← load → chunk → embed → upsert
│   ├── chunking/                ← Phase 2: semantic chunking pipeline
│   │   ├── fixed_overlap.py
│   │   ├── sentence_boundary.py
│   │   ├── section_aware.py     ← heading detection + parent-child
│   │   ├── semantic_boundary.py
│   │   └── pipeline.py          ← ChunkingPipeline + QualityGate
│   ├── retrieval/               ← Phase 3, 4 & 11: hybrid + multi-stage + advanced
│   │   ├── bm25.py              ← BM25Index (rank-bm25)
│   │   ├── hybrid.py            ← HybridFuser (RRF), HybridRetriever
│   │   ├── multistage.py        ← CrossEncoderReRanker, MultiStageRetriever
│   │   ├── triple_hybrid.py     ← TripleHybridRetriever (vector+BM25+graph, Phase 10)
│   │   ├── raptor.py            ← RaptorIndexer + RaptorRetriever (Phase 11)
│   │   ├── contextual_reranker.py ← LLM batched re-ranking (Phase 11)
│   │   └── mmr.py               ← MMR diversity re-ranking (Phase 11)
│   ├── query/                   ← Phase 5: query intelligence
│   │   ├── llm_client.py        ← LMStudioClient (OpenAI SDK → localhost)
│   │   ├── rewriter.py          ← QueryRewriter, QueryExpander (HyDE), MultiQueryGenerator
│   │   └── context.py           ← ConversationContextTracker, QueryIntelligencePipeline
│   ├── graph/                   ← Phase 10 & 11: knowledge graph (GraphRAG)
│   │   ├── models.py            ← Entity, Relation, Triple, GraphPath models
│   │   ├── base.py              ← GraphStoreBase abstract interface
│   │   ├── extractor.py         ← EntityRelationExtractor (spaCy NER + SVO)
│   │   ├── llm_extractor.py     ← LLMEntityExtractor (LM Studio JSON-prompt, Phase 11)
│   │   ├── store_networkx.py    ← NetworkXGraphStore (in-memory, tests)
│   │   ├── store_kuzu.py        ← KuzuGraphStore (embedded persistent, default)
│   │   ├── store_neo4j.py       ← Neo4jGraphStore (enterprise migration path)
│   │   ├── registry.py          ← GraphStoreRegistry factory
│   │   └── traversal.py         ← GraphTraverser (query → entity → N-hop BFS)
│   ├── provenance/              ← Phase 6: full traceability
│   │   └── models.py            ← ProvenanceRecord, CitationBuilder, SpanHighlighter
│   ├── confidence/              ← Phase 7: trust signals
│   │   └── signals.py           ← 7 signals + AnswerConfidenceAggregator
│   ├── temporal/                ← Phase 8: lifecycle + classification
│   │   └── filters.py           ← TemporalFilter, ClassificationFilter, AccessPolicyEvaluator
│   ├── noise/                   ← Phase 9: quality control
│   │   └── filters.py           ← DuplicateDetector, QualityScorer, NoiseFilterPipeline
│   └── observability/           ← Phase 9: production logging
│       └── logging.py           ← StructuredLogger, PipelineMetrics
│   ├── orchestrator/
│   ├── pipeline.py              ← RAGPipeline (all 11 phases wired)
│   └── response.py              ← RAGResponse envelope (with graph_entities + graph_paths)
│   ├── tests/
│   ├── conftest.py              ← Generic corpus fixture (auto-downloads test text)
│   ├── phase1/ ... phase11/     ← Phase-specific test suites
│   └── test_pipeline_e2e.py     ← Full integration test
├── config/
│   └── config.yaml              ← All configuration (one file to rule them all)
├── data/
│   └── shakespeare.txt          ← Auto-downloaded on first run (~5 MB)
├── BACKLOG.md                   ← Phased development backlog
├── requirements.txt
├── install.ps1
└── start.ps1
```

---

## Configuration

Edit `config/config.yaml` to switch backends. No code changes needed.

```yaml
store:
  backend: chromadb   # ← change this: chromadb | faiss | qdrant | weaviate | milvus | pgvector

embedding:
  provider: sentence_transformer
  model: all-MiniLM-L6-v2   # 384-dim, ~80 MB, runs on CPU

ingestion:
  collection_name: polyrag
  chunk_size: 512
  chunk_overlap: 64

llm:
  base_url: http://localhost:1234/v1   # LM Studio
  model: mistralai/ministral-3b
  api_key: lm-studio
  enable_rewrite: true
  enable_hyde: true
  enable_multi_query: false

graph:
  backend: kuzu           # networkx | kuzu | neo4j
  enabled: true
  spacy_model: en_core_web_sm
  max_hops: 2
  graph_weight: 0.3
  llm_extraction:
    enabled: false        # true = LLM extracts entities per chunk (higher quality, slower)

advanced_retrieval:
  raptor:
    enabled: false        # true = hierarchical cluster summaries (requires LLM + explicit build call)
  contextual_reranker:
    enabled: false        # true = one LLM call re-ranks all top-K candidates per query
  mmr:
    enabled: true         # MMR diversity re-ranking (numpy only, always recommended)
    diversity_weight: 0.3 # 0 = pure relevance, 1 = max diversity

access:
  user_clearance: INTERNAL   # PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED

quality:
  min_score: 0.2
  dedup_threshold: 0.85
```

---

## Using the Pipeline

```python
from orchestrator.pipeline import RAGPipeline

pipeline = RAGPipeline()   # loads config/config.yaml
pipeline.start()

# Ingest any plain-text corpus (not Shakespeare-specific)
pipeline.ingest_text("Your document text here...", metadata={"source": "my_doc", "version": "1.0"})

# Or auto-download the Gutenberg test corpus
pipeline.ingest_gutenberg()

# Optional: build RAPTOR hierarchical index after all ingestion is complete
# (only needed if advanced_retrieval.raptor.enabled: true in config)
pipeline.build_raptor_index()

# Simple query — uses 3-way hybrid search (vector + BM25 + knowledge graph)
results = pipeline.query("What does the policy say about data retention?", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.document.text[:120]}")
    print(f"  signals: {r.metadata.get('retrieval_signals', 'N/A')}")

# Full RAG ask (answer + citations + confidence + graph explanation)
from orchestrator.response import RAGResponse
response: RAGResponse = pipeline.ask("Summarize the key security requirements.", top_k=5)

print(response.answer)
print("Confidence:", response.confidence.verdict, response.confidence.composite_score)
for citation in response.citations:
    print(citation)

# Knowledge graph paths (Phase 10+)
print(response.graph_explanation())   # human-readable entity → relationship chains

pipeline.stop()
```

### RAGResponse envelope

```python
response.answer          # str — LLM answer (or retrieval summary if LLM offline)
response.results         # List[SearchResult] — top-k retrieved chunks
response.provenance      # List[ProvenanceRecord] — exact source attribution
response.citations       # List[str] — formatted citations
response.confidence      # ConfidenceReport — verdict + composite score + signals
response.graph_entities  # List[str] — entities detected in query (Phase 10+)
response.graph_paths     # List[GraphPath] — entity relationship chains used in retrieval
response.graph_explanation()  # str — human-readable graph traversal explanation
response.summary()       # str — one-line summary of result quality
```

---

## Retrieval Methods

All ten retrieval methods work in concert, automatically fused via Reciprocal Rank Fusion (RRF):

| Method | Signal | LLM Required | Default |
|---|---|:---:|:---:|
| Dense vector search | Semantic similarity | No | ✅ Always on |
| BM25 keyword search | Exact token overlap | No | ✅ Always on |
| Knowledge graph traversal | Entity relationships | No (spaCy) | ✅ When graph enabled |
| Cross-encoder re-ranking | Neural passage score | No | ✅ Always on |
| MMR diversity selection | Result diversification | No | ✅ Always on |
| Query rewriting | Clarity improvement | Yes | ✅ When LLM available |
| HyDE expansion | Hypothetical document | Yes | ✅ When LLM available |
| Multi-query ensemble | Paraphrase coverage | Yes | Config opt-in |
| RAPTOR hierarchical | Cluster summaries | Yes | Config opt-in |
| Contextual LLM re-ranking | Batched LLM ranking | Yes | Config opt-in |

---

## Confidence Verdicts

| Verdict | Composite Score | Meaning |
|---|---|---|
| `HIGH` | ≥ 0.75 | Strong evidence, high source agreement |
| `MEDIUM` | 0.45 – 0.75 | Moderate evidence, some agreement |
| `LOW` | 0.20 – 0.45 | Weak evidence, low agreement |
| `INSUFFICIENT_EVIDENCE` | < 0.20 | No reliable answer found |

---

## Data Classification

Documents can be tagged with classification labels. Users only see what their clearance allows:

```python
pipeline.ingest_text(text, metadata={"classification": "CONFIDENTIAL"})

# User with INTERNAL clearance won't see CONFIDENTIAL results
results = pipeline.query("...", user_context={"clearance": "INTERNAL"})
```

Levels: `PUBLIC` → `INTERNAL` → `CONFIDENTIAL` → `RESTRICTED`

---

## Test Corpus

Tests use **Project Gutenberg's Complete Works of William Shakespeare**
([pg100.txt](https://www.gutenberg.org/cache/epub/100/pg100.txt), ~5 MB plain text).

Downloaded automatically on first test run and cached to `data/shakespeare.txt`.
No Gutenberg account required. Text is in the public domain.

---

## Docker (for server-based backends)

```yaml
# docker-compose.yml  —  run locally for Qdrant / Weaviate / Milvus / PGVector
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]

  weaviate:
    image: semitechnologies/weaviate:latest
    ports: ["8080:8080", "50051:50051"]
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      DEFAULT_VECTORIZER_MODULE: none

  milvus:
    image: milvusdb/milvus:latest
    ports: ["19530:19530"]

  pgvector:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: polyrag
```

---

## Scalability & Production Readiness

Six hardening fixes shipped across three PRs, all living behind the existing API surface — zero breaking changes.

| Fix | What Changed | Where |
|-----|-------------|-------|
| **Async `/api/rag`** | Unified agentic endpoint — full traceability in one call | `api/routers/rag.py` |
| **Rate limiting** | `429 Too Many Requests` at configurable RPS cap | `api/main.py` |
| **LRU pipeline cache** | Max 10 cached pipelines; LRU eviction calls `pipeline.stop()` | `api/deps.py` |
| **Streaming chunker** | `stream_chunk_file()` generator — O(chunk_size) RAM regardless of file size | `core/ingestion/loader.py` |
| **Persistent job store** | `data/jobs.jsonl` write-through; survives API restart | `api/jobs.py` |
| **ScaleHints on RagProfile** | Per-profile: `embed_batch_size`, `max_doc_size_mb`, `bm25_persist`, `max_concurrent_requests` | `api/routers/rag.py` |
| **BM25 persistence** | Serialises `bm25_index._docs` to `data/bm25/<md5>.pkl` after warm-up | `api/deps.py` |
| **System endpoints** | `/api/system/health`, `/api/system/cache` (GET + DELETE) | `api/routers/system.py` |

### Unified Agentic RAG endpoint

The `/api/rag` endpoint is the single integration point for agentic AI flows:

```http
POST /api/rag
Content-Type: application/json

{
  "profile_id": "my-profile",   # optional — omit to use inline config
  "question": "What does Hamlet say about mortality?",
  "collection_name": "my_docs",
  "backend": "chromadb",
  "embedding_model": "all-MiniLM-L6-v2",
  "methods": {
    "enable_dense": true,
    "enable_bm25": true,
    "enable_graph": true,
    "enable_rerank": true,
    "enable_mmr": true,
    "enable_splade": false,
    "enable_rewrite": false,
    "enable_multi_query": false,
    "enable_hyde": false,
    "enable_raptor": false,
    "enable_contextual_rerank": false,
    "enable_llm_graph": false
  },
  "top_k": 5
}
```

Response includes full traceability:

```json
{
  "answer": "...",
  "confidence": { "verdict": "HIGH", "composite_score": 0.83, "signals": {...} },
  "citations": ["Document A, chunk 3 (chars 120–450)"],
  "provenance": [{"doc_id": "...", "source": "...", "span_start": 120, "span_end": 450}],
  "results": [
    {
      "score": 0.91,
      "text": "...",
      "metadata": {
        "source": "my_doc.pdf",
        "chunk_strategy": "sentence_boundary",
        "classification": "INTERNAL",
        "_method_lineage": ["Dense Vector", "BM25 Keyword", "Knowledge Graph"]
      }
    }
  ],
  "graph_entities": ["PERSON:Hamlet", "PERSON:Horatio"],
  "graph_paths": ["Hamlet —[speaks_to]→ Horatio"],
  "traces": [...]
}
```

### Load Testing

```powershell
# Requires API running on port 8000
.\scripts\load-test.ps1                          # headless, 50 users, 60 s
.\scripts\load-test.ps1 -Users 100 -Duration 120 # custom
.\scripts\load-test.ps1 -UI                      # opens browser dashboard
.\scripts\load-test.ps1 -Report ./reports/run1.html
```

Pass/fail thresholds (checked at exit): p50 < 2 s, p95 < 5 s, error rate < 1 %.

### RAGAS Evaluation (PR #5 — pending merge)

When merged, every `/api/evaluate` response will include LLM-judged RAGAS metrics alongside word-overlap scores:

```json
"ragas": {
  "faithfulness": 0.95,
  "answer_relevancy": 0.88,
  "context_precision": 0.75,
  "context_recall": 0.80
}
```

Check availability before running: `GET /api/evaluate/ragas-status`.
Requires LM Studio running on port 1234. Gracefully returns `null` when offline.

---

## CI / CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every PR to `main`:

| Job | What it checks |
|-----|---------------|
| `Tests (Python 3.10)` | Fast test subset + `tests/phase13/test_rag_router.py` (28 tests) |
| `Tests (Python 3.11)` | Same matrix leg on Python 3.11 |
| `Import sanity check` | All routers importable; no circular dependency |

```bash
# Reproduce CI locally
pytest tests/ -m "not integration and not lmstudio and not browser" --timeout=300
pytest tests/phase13/test_rag_router.py -v
```

Recommended branch protection: **Settings → Branches → Require status checks** → select all three jobs above.

---

## Backlog & Roadmap

See [BACKLOG.md](./BACKLOG.md) for the full 11-phase development plan with
per-phase test exit criteria and evaluation benchmarks.

---

## Future: Metadata + Faceted Filtering

The codebase stores rich per-chunk metadata (`doc_id`, `source`, `section_title`, `chunk_strategy`, `classification`, `start_char`, `end_char`, user-provided fields) and already supports `filters: Optional[Dict]` at the pipeline and adapter layers. The items below extend that foundation toward full faceted search.

### Already Implemented ✅
| Component | Status |
|-----------|--------|
| `filters=` on `pipeline.query()` / `pipeline.ask()` | ✅ |
| Pre-search native filtering — Qdrant, Weaviate, Milvus, PGVector, ChromaDB | ✅ |
| Post-fusion `MetadataFilter.apply()` in `HybridRetriever` | ✅ |
| `filters=` passed through `TripleHybridRetriever` → `HybridRetriever` | ✅ |
| **BM25 `filters=` parameter (post-rank, pre-fusion)** | ✅ fixed |

### Pending — Retrieval Layer
| Item | File | What to do |
|------|------|-----------|
| Pass `filters=` to `RaptorIndexer.retrieve()` | `core/retrieval/raptor.py` | Forward to `store.query()` on the `_raptor` collection |
| `pre_filters=` on `CrossEncoderReRanker.rerank()` | `core/retrieval/multistage.py` | Apply `MetadataFilter` before expensive cross-encoder scoring |
| `pre_filters=` on `MMRReranker.rerank()` | `core/retrieval/mmr.py` | Narrow diversity candidate pool before MMR selection |
| `pre_filters=` on `ContextualReranker.rerank()` | `core/retrieval/contextual_reranker.py` | Pre-filter before LLM context scoring |
| Forward `filters=` to `MultiQueryGenerator` sub-queries | `core/query/rewriter.py` | Each parallel sub-query should carry the same filter |

### Pending — Graph Layer
| Item | File | What to do |
|------|------|-----------|
| `filters=` on `GraphTraverser.traverse()` | `core/graph/traversal.py` | Apply `MetadataFilter` to traversal results post-graph-walk |
| Node-property filtering in Cypher | `core/graph/store_kuzu.py` | Add `WHERE n.source = $source` / `n.entity_type IN [...]` to Kuzu queries |

### Pending — Adapter Layer (Richer Operators)
| Item | What to do |
|------|-----------|
| Unified `FilterCondition` model (`field`, `op`, `value`) | New `core/store/filter_translator.py` — translate once per backend instead of per-adapter ad-hoc dicts |
| Range queries (`gt`, `gte`, `lt`, `lte`) | Qdrant `Range`, Weaviate `.greater_than()`, Milvus SQL `>`, PGVector `::numeric >`, FAISS in-memory |
| Multi-value (`in` operator) | Qdrant `MatchAny`, Weaviate `.contains_any()`, Milvus `IN [...]`, PGVector `= ANY(...)` |
| OR groups across conditions | `{"$or": [{...}, {...}]}` → backend-specific OR dialect |

### Pending — API Layer
| Item | File | What to do |
|------|------|-----------|
| Add `filters: List[FilterCondition]` to `SearchRequest` | `api/schemas.py` | Expose filtering to REST clients |
| Pass filters through `_run_search_with_bundle()` | `api/routers/search.py` | Wire `SearchRequest.filters` into the retrieval call |
| Add `facets: List[FacetRequest]` to `SearchRequest` | `api/schemas.py` | Fields the client wants count buckets for |
| Return `FacetResult[]` in `SearchResponse` | `api/schemas.py` | `[{field, buckets: [{value, count}]}]` |

### Pending — Facet Engine
| Item | File | What to do |
|------|------|-----------|
| `FacetEngine.compute_facets(results, fields)` | `core/retrieval/facet_engine.py` (new) | Count unique values per metadata field across result window |
| `FacetEngine.apply_facet_filter(results, selected)` | same | OR within a field, AND across fields (standard e-commerce behavior) |

### Pending — Query Intelligence (Optional / Advanced)
| Item | File | What to do |
|------|------|-----------|
| `QueryRewriter.extract_filters(query)` | `core/query/rewriter.py` | Use LLM to detect filter intent in natural language, e.g. *"Hamlet quotes from Act 3"* → `{section_title contains "Act 3"}` |

---

## Open-Source Stack

| Component | Library |
|---|---|
| Data models | [Pydantic v2](https://docs.pydantic.dev/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) |
| Keyword search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, Cormack et al.) |
| Cross-encoder re-ranking | [sentence-transformers](https://www.sbert.net/) (ms-marco-MiniLM) |
| Local LLM | [LM Studio](https://lmstudio.ai/) + mistralai/ministral-3b via OpenAI SDK |
| Local vector DB | [ChromaDB](https://www.trychroma.com/), [FAISS](https://faiss.ai/) |
| Scalable vector DB | [Qdrant](https://qdrant.tech/), [Weaviate](https://weaviate.io/), [Milvus](https://milvus.io/) |
| Relational vector | [PGVector](https://github.com/pgvector/pgvector) |
| Knowledge graph (embedded) | [Kuzu](https://kuzudb.com/) + [NetworkX](https://networkx.org/) |
| Knowledge graph (enterprise) | [Neo4j](https://neo4j.com/) (zero-code migration from Kuzu) |
| NLP / NER | [spaCy](https://spacy.io/) en_core_web_sm |
| Near-dup detection | [datasketch](https://github.com/ekzhu/datasketch) (MinHash LSH) |
| Test corpus | [Project Gutenberg #100](https://www.gutenberg.org/ebooks/100) |
| Test framework | [pytest](https://pytest.org/) |

All components are open-source with permissive licences. No OpenAI API key required.



---

## Phase 13 – Browser UI (React 19 + FastAPI)

A full-featured browser dashboard to orchestrate all RAG functions — ingestion, search, comparison, knowledge graph, evaluation, and more.

### Architecture

`
Browser (React 19 + Vite 5 + TypeScript)
  ├── React Flow 12    → ingestion DAG + retrieval flow diagrams
  ├── D3.js v7         → knowledge graph (entities + relations)
  └── Axios + SSE      → real-time log streaming from backend

        ↕  HTTP REST + Server-Sent Events (port 8000)

FastAPI + uvicorn  (api/ directory)
  ├── POST /api/rag                    → unified agentic endpoint (answer + full traceability)
  ├── GET/POST/PUT/DELETE /api/rag/profiles → named config profiles (persist tested settings)
  ├── POST /api/ingest                 → start batch ingestion job(s)
  ├── GET  /api/ingest/{id}/stream     → SSE live chunking/embedding logs
  ├── POST /api/search                 → multi-backend query with retrieval trace
  ├── POST /api/compare                → full comparison matrix
  ├── GET  /api/backends               → health + ping for all 6 backends
  ├── GET  /api/collections            → collection CRUD
  ├── GET  /api/graph/{coll}           → entity/relation nodes for D3
  ├── POST /api/evaluate               → ground-truth scoring (faithfulness / relevance / recall)
  ├── GET  /api/evaluate/ragas-status  → RAGAS availability check (LM Studio)
  ├── POST /api/feedback               → relevance thumbs up/down
  ├── POST /api/chunks/preview         → dry-run chunking (no ingest)
  ├── GET  /api/jobs                   → job history + status
  ├── GET  /api/system/health          → uptime, pipeline cache, job counts
  └── GET/DELETE /api/system/cache     → inspect / flush LRU pipeline cache

        ↕  Python import (zero changes to existing code)

orchestrator/pipeline.py  (existing — untouched)
`

### Pages

| Page | Key Features |
|---|---|
| **⬆ Ingestion Studio** | Drag-and-drop batch queue · Chunking strategy config · Chunk Preview modal (color-coded boundaries) · Backend multi-select · React Flow DAG (Upload→Chunk→Embed→KG→Upsert) · SSE live log stream · Per-backend job status |
| **🔍 Search Lab** | Query input + history · 10 retrieval method toggles · Per-backend result cards (score, text, provenance, confidence) · Retrieval Trace audit panel · A/B Config Testing mode · React Flow retrieval visualization |
| **📊 Compare** | Live comparison matrix · Sortable columns (base/full top score, KW hits, ingest time) · CSV + JSON export |
| **�� Knowledge Graph** | D3.js force-directed graph · Entity type colors (PERSON/ORG/LOC/CONCEPT) · Click node → source chunks · Collection selector |
| **📚 Document Library** | Tabbed by backend · Collection CRUD · Document versions · Re-ingest · Delete · Health indicators |
| **⚖ Evaluation Studio** | Ground-truth Q&A manager · Auto-score per backend × method (faithfulness / relevance / source_hit) · Results matrix |
| **🗂 Job History** | Persistent job log · Status filter · Log line expansion · Clear completed |
| **⚙ Settings** | Backend connection strings · LM Studio URL/port · Embedding model |

### 10 Enterprise-Grade Features (built-in from day 1)

1. **Evaluation Studio** — RAGAS-style ground-truth scoring across all backends × retrieval methods
2. **Chunking Preview** — visual dry-run before committing (color-coded chunk boundaries, parent-child, entity spans)
3. **Document Library** — full collection CRUD with version and health tracking across all backends
4. **Retrieval Trace Panel** — step-by-step audit trail: which of 10 methods fired, scores before/after fusion
5. **Relevance Feedback Loop** — thumbs up/down on result chunks → stored and fed back as future score bias
6. **A/B Config Testing** — same query against two config profiles side-by-side with diff view
7. **Backend Health Monitor** — always-visible top bar with ping latency, vector counts, live status
8. **Export / Reporting** — CSV, JSON export from comparison matrix; HTML provenance reports
9. **Multi-file Batch Ingestion Queue** — drag-and-drop multiple files, per-file progress, shared ingestion policy
10. **Notifications / Job History** — persistent job log, completion status, error tracking

### Starting the UI

`powershell
# Terminal 1: Start the API server
.\start.ps1
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start the React frontend
cd frontend
npm run dev
# → Open http://localhost:3000

# Terminal 3 (optional): Start Docker vector DBs
docker compose -f docker-compose.polyrag.yml up -d
`

### Running Tests

`powershell
# Backend tests (420 tests)
pytest tests/ -q

# API integration tests (17 tests)
pytest tests/phase13/ -q

# Frontend tests (29 tests, Vitest + React Testing Library)
cd frontend && npm test
`

### Technology Stack (Phase 13 additions)

| Component | Library |
|---|---|
| Frontend framework | [React 19](https://react.dev/) + [Vite 5](https://vitejs.dev/) |
| UI styling | [Tailwind CSS v3](https://tailwindcss.com/) |
| State management | [Zustand](https://zustand-demo.pmnd.rs/) |
| Data fetching | [TanStack Query](https://tanstack.com/query) + [Axios](https://axios-http.com/) |
| Flow diagrams | [React Flow 12](https://reactflow.dev/) |
| Graph visualization | [D3.js v7](https://d3js.org/) |
| API server | [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) |
| Live streaming | Server-Sent Events (SSE) via [sse-starlette](https://github.com/sysid/sse-starlette) |
| Frontend testing | [Vitest](https://vitest.dev/) + [React Testing Library](https://testing-library.com/) |
