# Phoenice-PolyRAG — Copilot Workspace Context

> **Auto-loaded by GitHub Copilot CLI on every session.**
> Keep this file updated when adding phases, endpoints, or major config changes.
> Last updated: 2026-03-20 — Scalability PRs #1–#3 (LRU cache, streaming chunker, persistent jobs, rate limiting, system endpoints, ScaleHints, BM25 persistence) + CI (PR #4) + RAGAS (PR #5, pending) + Load test (PR #6)

---

## Project Identity

| Field | Value |
|-------|-------|
| **Name** | Phoenice-PolyRAG |
| **Version** | 14.2.0 |
| **Type** | Full-stack RAG framework (FastAPI + React + 11-phase Python pipeline) |
| **Tagline** | "Write your RAG orchestration once. Run it on any vector store." |
| **Python** | 3.10+ |
| **Node** | 18+ (frontend) |

---

## Tech Stack

### Backend (Python)
- **Framework**: FastAPI 0.111+, uvicorn
- **Data validation**: Pydantic 2.5+
- **Embeddings**: sentence-transformers 5.x — 3 supported models, selectable from Settings UI:
  - `all-MiniLM-L6-v2` — 384-dim, ~80 MB, fast (default)
  - `BAAI/bge-base-en-v1.5` — 768-dim, ~440 MB, MTEB 63.6 (balanced)
  - `BAAI/bge-large-en-v1.5` — 1024-dim, ~1.3 GB, MTEB 64.2 (best quality)
- **Sparse neural retrieval**: sentence-transformers `SparseEncoder` → `naver/splade-v3` (Apache 2.0, ~440 MB, downloaded on first use)
- **Vector stores** (6 backends, pluggable): ChromaDB, FAISS, Qdrant, Weaviate, Milvus, PGVector
- **Hybrid search**: rank-bm25 (keyword), SPLADE (sparse neural), RRF fusion (3-way: Dense + SPLADE + BM25)
- **Knowledge graph**: spaCy (NER), Kuzu (embedded), Neo4j (optional)
- **LLM client**: OpenAI SDK → LM Studio at `localhost:1234` (optional; graceful degradation)
- **Reranking**: cross-encoder, MMR, RAPTOR hierarchical
- **Quality evaluation**: RAGAS 0.4.x (LLM-judged: faithfulness, answer_relevancy, context_precision, context_recall) — PR #5, pending merge
- **Load testing**: Locust 2.24+ (`tests/load/locustfile.py`, `scripts/load-test.ps1`)
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — Python 3.10 + 3.11 matrix, CPU-only PyTorch
- **Testing**: pytest 8.0+, pytest-asyncio, httpx

### Frontend (React/TypeScript)
- **Framework**: React 19.2 + TypeScript 5.9
- **Build**: Vite
- **Styling**: Tailwind CSS
- **State**: TanStack React Query + Zustand
- **Visualization**: D3.js, ReactFlow
- **Testing**: Vitest, @testing-library/react, Playwright

---

## Repository Layout

```
Phoenice-PolyRAG/
├── core/                    ← All 11 phase modules (pluggable, reusable)
│   ├── store/               ← VectorStoreBase + 6 adapters (Phase 1)
│   ├── embedding/           ← SentenceTransformerProvider (MiniLM, BGE-base, BGE-large)
│   ├── ingestion/           ← Chunk → embed → upsert; stream_chunk_file() for large docs
│   ├── chunking/            ← Fixed, sentence-boundary, semantic (Phase 2)
│   ├── retrieval/           ← Hybrid, SPLADE, multistage, RAPTOR, MMR, contextual reranker (Phase 3,3b,4,11)
│   │   ├── hybrid.py        ← HybridRetriever (3-way: Dense+SPLADE+BM25), HybridFuser (RRF)
│   │   ├── splade.py        ← SparseNeuralIndex (naver/splade-v3, disk persistence)
│   │   ├── bm25.py          ← BM25Index (with metadata filtering fix)
│   │   └── triple_hybrid.py ← TripleHybridRetriever (vector+hybrid+graph)
│   ├── query/               ← LLM rewriting, HyDE, multi-query (Phase 5)
│   ├── graph/               ← Entity extraction, Kuzu graph, 3-way hybrid (Phase 10,11)
│   ├── provenance/          ← Citations, traceability, audit (Phase 6)
│   ├── confidence/          ← 7-signal confidence aggregator (Phase 7)
│   ├── temporal/            ← Lifecycle filters, RBAC, access policies (Phase 8)
│   ├── noise/               ← Dedup, quality scoring, observability (Phase 9)
│   ├── classification/      ← Data classification & access control
│   ├── evaluation/          ← RagasScorer, RagasResult (PR #5 — pending merge)
│   └── observability/       ← Structured logging, metrics
│
├── orchestrator/
│   ├── pipeline.py          ← RAGPipeline (wires all 11 phases) ← MAIN ENTRY POINT
│   ├── response.py          ← RAGResponse envelope (answer + citations + graph)
│   └── prompt_registry.py  ← LLM prompt templates
│
├── api/                     ← FastAPI REST server (port 8000, v14.2.0)
│   ├── main.py              ← App entry, health check, CORS, rate limiting, startup events
│   ├── routers/
│   │   ├── rag.py           ← Unified agentic endpoint + RagProfile CRUD + ScaleHints
│   │   ├── system.py        ← /api/system/health, /api/system/cache (LRU inspect/flush)
│   │   ├── search.py        ← Multi-backend parallel search
│   │   ├── ingest.py        ← Background ingestion + SSE streaming
│   │   ├── evaluate.py      ← Quality metrics (F/R/S/G + RAGAS when PR #5 merged)
│   │   └── ...              ← Other routers (see API Routes section)
│   ├── deps.py              ← Pipeline factory, LRU cache, BM25 persistence helpers
│   ├── jobs.py              ← Async job queue (write-through to data/jobs.jsonl)
│   └── schemas.py           ← Pydantic request/response models
│
├── frontend/                ← React 19 SPA (port 3000, Vite)
│   └── src/
│       ├── components/      ← UI components
│       ├── pages/           ← Route pages
│       ├── api/             ← Axios HTTP client
│       └── store/           ← Zustand global state
│
├── config/
│   ├── config.yaml          ← SINGLE config file (all 11 phases, all backends)
│   └── prompts.yaml         ← LLM prompt templates
│
├── data/                    ← Runtime data (safe to delete & regenerate)
│   ├── chromadb/            ← ChromaDB persistence
│   ├── faiss/               ← FAISS index files
│   ├── splade/              ← SPLADE pre-encoded sparse vectors (per collection)
│   ├── bm25/                ← BM25 index pkl snapshots (data/bm25/<md5>.pkl)
│   ├── jobs.jsonl           ← Persistent job store (write-through, survives restart)
│   ├── graph.kuzu           ← Kuzu embedded graph DB
│   ├── ingestion_audit.jsonl
│   ├── retrieval_trails.jsonl
│   └── shakespeare.txt      ← Auto-downloaded test corpus (5 MB)
│
├── tests/
│   ├── phase1/ … phase11/   ← Phase-specific test suites
│   ├── phase13/             ← API integration tests (28 tests incl. RAG router)
│   ├── load/                ← Locust load test (locustfile.py)
│   ├── test_pipeline_e2e.py ← End-to-end integration test
│   └── conftest.py
│
├── scripts/
│   ├── compare_backends.py  ← Benchmark all 6 vector store backends
│   └── load-test.ps1        ← PowerShell Locust runner (pre-flight checks, report output)
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml           ← GitHub Actions CI (Python 3.10+3.11, fast pytest, lint)
│   └── copilot-instructions.md  ← THIS FILE (workspace context)
│
├── install.ps1              ← Windows setup script
├── start.ps1                ← Windows service manager (multi-worker aware)
├── docker-compose.polyrag.yml ← Docker: Qdrant, Weaviate, Milvus, PGVector
├── pyproject.toml           ← Python package metadata
├── requirements.txt         ← Python dependencies
├── pytest.ini               ← Test markers (integration, lmstudio, browser)
└── BACKLOG.md               ← Phase backlog
```

---

## API Routes (FastAPI, port 8000, v14.2.0)

| Method | Path | Router | Purpose |
|--------|------|--------|---------|
| GET | `/api/health` | `main.py` | Health check |
| POST | `/api/rag` | `routers/rag.py` | **Unified agentic endpoint** — answer + full traceability |
| GET | `/api/rag/profiles` | `routers/rag.py` | List named RagProfiles |
| POST | `/api/rag/profiles` | `routers/rag.py` | Create a RagProfile |
| GET | `/api/rag/profiles/{id}` | `routers/rag.py` | Get a RagProfile |
| PUT | `/api/rag/profiles/{id}` | `routers/rag.py` | Update a RagProfile |
| DELETE | `/api/rag/profiles/{id}` | `routers/rag.py` | Delete a RagProfile |
| GET | `/api/system/health` | `routers/system.py` | Uptime, pipeline cache %, job counts |
| GET | `/api/system/cache` | `routers/system.py` | Inspect LRU pipeline cache entries |
| DELETE | `/api/system/cache` | `routers/system.py` | Flush all cached pipelines |
| POST | `/api/ingest` | `routers/ingest.py` | Ingest documents (background + SSE) |
| GET | `/api/jobs/{id}/status` | `routers/jobs.py` | Async ingest job status |
| POST | `/api/search` | `routers/search.py` | Multi-backend parallel search |
| GET | `/api/backends` | `routers/backends.py` | Available vector store backends |
| GET | `/api/chunks` | `routers/chunks.py` | Stored chunks |
| GET/POST | `/api/graph/*` | `routers/graph.py` | Entity/relation graph queries |
| POST | `/api/evaluate` | `routers/evaluate.py` | Quality metrics (F/R/S/G scores + graph trail) |
| GET | `/api/evaluate/{eval_id}` | `routers/evaluate.py` | Fetch stored evaluation results |
| GET | `/api/evaluate/browse-chunks` | `routers/evaluate.py` | Browse chunks for Q&A generation |
| POST | `/api/evaluate/generate-qa` | `routers/evaluate.py` | Auto-generate Q&A from a chunk |
| GET | `/api/evaluate/ragas-status` | `routers/evaluate.py` | RAGAS LLM availability (PR #5 pending) |
| POST | `/api/feedback` | `routers/feedback.py` | User annotations |
| POST | `/api/compare` | `routers/compare.py` | A/B backend comparison (latency, graph A/B, chunk previews) |
| GET | `/api/compare/sample-queries` | `routers/compare.py` | Ready-to-use benchmark queries |
| GET | `/api/prompts` | `routers/prompts.py` | Prompt template registry |
| GET | `/api/traces` | `routers/traces.py` | Execution traces |
| GET | `/api/retrieval-trails` | `routers/retrieval_trails.py` | Retrieval logs |

**API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 11 RAG Phases

| Phase | Module Path | Key Classes |
|-------|-------------|-------------|
| 1 — Vector Store Abstraction | `core/store/` | `VectorStoreBase`, `AdapterRegistry`, `ChromaDBAdapter`, `QdrantAdapter`, etc. |
| 2 — Semantic Chunking | `core/chunking/` | `FixedOverlapChunker`, `SentenceBoundaryChunker`, `SemanticBoundaryChunker` |
| 3 — Hybrid Search | `core/retrieval/hybrid.py`, `bm25.py` | `HybridRetriever`, `BM25Index` (with metadata filters), `HybridFuser` |
| 3b — SPLADE Sparse Neural | `core/retrieval/splade.py` | `SparseNeuralIndex` (`naver/splade-v3`; disk persistence; inverted-index dot-product) |
| 4 — Multi-stage Reranking | `core/retrieval/multistage.py` | `MultiStageRetriever`, `CrossEncoderReRanker` |
| 5 — Query Intelligence | `core/query/` | `QueryRewriter`, `QueryExpander`, `LMStudioClient` |
| 6 — Provenance & Citations | `core/provenance/` | `ProvenanceRecord`, `CitationBuilder` |
| 7 — Confidence Signals | `core/confidence/` | `AnswerConfidenceAggregator` (7 signals) |
| 8 — Temporal/RBAC Filters | `core/temporal/` | `TemporalFilter`, `ClassificationFilter`, `AccessPolicyEvaluator` |
| 9 — Noise & Quality | `core/noise/` | `DuplicateDetector`, `QualityScorer` |
| 10 — Knowledge Graph | `core/graph/` | `EntityRelationExtractor`, `KuzuGraphStore`, `TripleHybridRetriever` |
| 11 — RAPTOR + MMR + Contextual | `core/retrieval/raptor.py`, `mmr.py` | `RaptorIndexer`, `MMRReranker`, `ContextualReranker` |

---

## Configuration

### Primary Config: `config/config.yaml`
Single YAML file drives all 11 phases. **Change one line to switch backends:**
```yaml
store:
  backend: chromadb   # Options: chromadb | faiss | qdrant | weaviate | milvus | pgvector
```

### Environment Variable Overrides (optional)
```bash
POLYRAG_BACKEND=qdrant
POLYRAG_COLLECTION=my_docs
POLYRAG_LLMMODEL=gpt-4
POLYRAG_GRAPHENABLED=true
```

### SPLADE Sparse Neural Retrieval (Phase 3b)
SPLADE (`naver/splade-v3`, Apache 2.0) is the 3rd signal in the RRF fusion pipeline alongside Dense Vector and BM25.

```yaml
retrieval:
  splade:
    enabled: false              # set true or toggle from frontend
    model: naver/splade-v3      # ~440 MB, downloaded once from HuggingFace
    persist_dir: ./data/splade  # CSR-format numpy arrays; loaded in <1 s on restart
    splade_weight: 1.0
    bm25_weight_with_splade: 0.8   # BM25 down-weighted when SPLADE active
```

**Activation flow:**
1. Frontend toggle → `enable_splade: bool` in `RetrievalMethods`
2. Router passes `enable_splade` to `build_pipeline_config()` → `retrieval.splade.enabled`
3. Pipeline cache key includes SPLADE flag → new pipeline created on first toggle
4. `SparseNeuralIndex.load(collection)` attempts disk load; encodes via BERT if not found
5. `HybridRetriever.search()` runs 3-worker thread pool: Dense + BM25 + SPLADE in parallel
6. `HybridFuser.fuse()` combines all 3 signals with configured RRF weights

**Disk persistence format** (`data/splade/{collection}/`):
- `docs.json` — list of `{id, text, metadata}` 
- `vectors.npz` — CSR sparse arrays: `term_ids` (int32), `weights` (float32), `offsets` (int32)

**RRF weights:** Dense=1.0, SPLADE=1.0, BM25=0.8 (BM25 down-weighted; SPLADE subsumes most of its signal plus adds term expansion)

**Benchmark:** MRR@10 = 40.2 vs BM25 ~18 (MS MARCO); BEIR nDCG@10 = 51.7

---

### Embedding Model Selection (Multi-Encoder)
Three dense encoders are supported. Selected via **Settings UI → Embedding Model** dropdown; persisted to localStorage and sent with every search/ingest request.

| Model | Dim | Size | Speed (CPU) | MTEB Score | Collection Suffix |
|-------|-----|------|-------------|-----------|-------------------|
| `all-MiniLM-L6-v2` | 384 | ~80 MB | ~5 ms/chunk | 56.3 | `_minilm` |
| `BAAI/bge-base-en-v1.5` | 768 | ~440 MB | ~15 ms/chunk | 63.6 | `_bge-base` |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3 GB | ~40 ms/chunk | 64.2 | `_bge-large` |

**Collection isolation (critical):** Each model writes to its own collection. The `embedding_model` field in `SearchRequest` / `IngestRequest` is passed to `build_pipeline_config(embedding_model=...)` → `_model_slug()` auto-appends the suffix. Example: `polyrag_docs` → `polyrag_docs_bge-base`. This prevents dimension mismatch across encoder switches.

**Code locations:**
- `core/embedding/sentence_transformer.py` — `SUPPORTED_MODELS` dict, `SentenceTransformerProvider` (lazy-loaded, thread-safe, module-level cache per `(model_name, device)`)
- `api/deps.py` — `_model_slug()`, `_MODEL_SLUGS`, `build_pipeline_config(embedding_model=...)`, pipeline cache key includes `embedding.model`
- `api/schemas.py` — `EmbeddingModel` Literal type + field on `SearchRequest` and `IngestRequest`
- `frontend/src/store/index.ts` — `EMBEDDING_MODELS`, `EmbeddingModelId`, `embeddingModel` Zustand state (persisted to localStorage)
- `frontend/src/pages/Settings.tsx` — dropdown UI with dim/size hints and collection suffix display
- `frontend/src/api/search.ts` + `ingest.ts` — auto-read `embeddingModel` from `useStore.getState()` and include in request body

**Switching encoder:** Select in Settings → re-ingest your docs → search. BGE models download on first use and cache locally. All 3 encoders can coexist — each has its own isolated collection.

**Pipeline cache key includes:** `embedding.model` → different encoders never share a pipeline instance. Dim mismatch at vector store level is impossible.

### LM Studio (Phase 5 — optional)
- Endpoint: `http://localhost:1234/v1`
- API key: `lm-studio` (dummy)
- Model: `mistralai/ministral-3b` (or any compatible)
- **Graceful degradation** if offline (Phase 5 features disabled)

### Server Backend Credentials (in `config.yaml`)
```yaml
pgvector:  { host: localhost, port: 5433, user: postgres, password: postgres, database: polyrag }
neo4j:     { uri: bolt://localhost:7687, user: neo4j, password: password }
qdrant:    { host: localhost, port: 6333 }
```

---

## Docker Services (`docker-compose.polyrag.yml`)

| Service | Ports | Volume |
|---------|-------|--------|
| Qdrant | 6333 (REST), 6334 (gRPC) | `polyrag_qdrant_data` |
| Weaviate | 8088 (HTTP), 50052 (gRPC) | `polyrag_weaviate_data` |
| Milvus (+etcd, MinIO) | 19530 (gRPC), 9091 (metrics) | `polyrag_milvus_data` |
| PGVector (PostgreSQL 16) | 5433 → 5432 | `polyrag_pgvector_data` |

ChromaDB and FAISS run **in-process** (no Docker needed).

---

## Common Commands

### Setup & Start
```powershell
.\install.ps1               # Initial setup (venv + deps + spaCy + npm)
.\install.ps1 -Full         # Also install Weaviate, Milvus, PGVector extras
.\start.ps1                 # Start Docker backends + API (8000) + Frontend (3000)
.\start.ps1 -NoDocker       # Skip Docker containers
.\start.ps1 -Action stop    # Stop all services
```

### Manual Start (individual services)
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000   # API server
cd frontend && npm run dev                   # Frontend (port 3000)
docker compose -f docker-compose.polyrag.yml up -d   # Server backends
```

### Testing
```powershell
# Fast (no external services) — mirrors CI
pytest tests/ -v -m "not integration and not lmstudio and not browser" --timeout=300

# API safety tests (28 tests — always run before committing)
pytest tests/phase13/test_rag_router.py -v

# With LM Studio running
pytest tests/ -v -m "not integration"

# Specific phase
pytest tests/phase2/ -v
pytest tests/phase10/ -v

# E2E
pytest tests/test_pipeline_e2e.py -v
```

### Load Testing (requires API running)
```powershell
.\scripts\load-test.ps1                          # headless, 50 users, 60 s
.\scripts\load-test.ps1 -Users 100 -Duration 120 # custom scale
.\scripts\load-test.ps1 -UI                      # browser dashboard
.\scripts\load-test.ps1 -Report .\reports\run.html
```

### Python API Usage
```python
from orchestrator.pipeline import RAGPipeline

pipeline = RAGPipeline.from_config("config/config.yaml")
pipeline.start()
pipeline.ingest_gutenberg()                          # Download & ingest Shakespeare
pipeline.ingest_text("Custom doc...", metadata={})   # Ingest custom text
response = pipeline.ask("What does Hamlet say?")
print(response.answer, response.confidence.verdict)
pipeline.stop()
```

### Backend Benchmarking
```powershell
python scripts/compare_backends.py   # Benchmark all 6 vector stores
```

---

## Data Storage

| Store | Path | Format | Regenerable? |
|-------|------|--------|--------------|
| ChromaDB | `data/chromadb/` | Chroma DB | ✅ Yes |
| FAISS | `data/faiss/` | Binary index | ✅ Yes |
| SPLADE vectors | `data/splade/` | CSR numpy npz + docs.json | ✅ Yes (re-encodes on ingest) |
| BM25 snapshots | `data/bm25/` | pickle (`<md5>.pkl`) | ✅ Yes (re-serialised after warm-up) |
| Persistent jobs | `data/jobs.jsonl` | JSON Lines (write-through) | ✅ Yes (or delete to clear history) |
| Kuzu graph | `data/graph.kuzu` | Kuzu DB | ✅ Yes |
| Ingestion audit | `data/ingestion_audit.jsonl` | JSON Lines | ✅ Yes |
| Retrieval trails | `data/retrieval_trails.jsonl` | JSON Lines | ✅ Yes |
| Test corpus | `data/shakespeare.txt` | Plain text | ✅ Auto-downloaded |

All `data/` contents can be safely deleted; re-ingestion rebuilds them.

---

## Test Suite

- **Phase tests**: 368 tests across 11 phases + E2E — all passing (1 skipped for LM Studio auto-skip)
- **API safety tests**: 28 tests in `tests/phase13/test_rag_router.py` — run before every commit
- **Load tests**: `tests/load/locustfile.py` — 5 weighted scenarios; run manually via `scripts/load-test.ps1`
- **CI**: GitHub Actions on every PR — Python 3.10 + 3.11 matrix, fast subset, import sanity
- **Markers** (`pytest.ini`): `integration`, `lmstudio`, `browser`
- **Fast subset** (no external services): `-m "not integration and not lmstudio and not browser"`

---

## Key Design Decisions

1. **Single config file** (`config/config.yaml`) drives all phases — no scattered env vars
2. **Adapter pattern** for vector stores — `VectorStoreBase` interface, swap backend by changing 1 line
3. **Graceful degradation** — LM Studio, Neo4j, server backends, SPLADE, RAGAS all optional
4. **No GPU required** — all embeddings run on CPU (MiniLM, BGE-base, BGE-large, SPLADE `naver/splade-v3`)
5. **Background ingestion** — POST `/api/ingest` returns a job ID; SSE streams progress; jobs persisted to `data/jobs.jsonl`
6. **All data regenerable** — `data/` directory can be wiped and rebuilt
7. **LRU pipeline cache** — max 10 cached pipelines; eviction calls `pipeline.stop()` for clean resource release; BM25 index pkl saved after warm-up
8. **Unified agentic endpoint** — `POST /api/rag` accepts a profile ID or inline config; returns answer + confidence + citations + provenance + graph paths in one call
9. **Evaluate uses `ask_with_bundle()`** — calls `expand_query()` once then `ask_with_bundle()` per backend (supports retrieval method flags); plain `pipeline.ask()` does NOT accept those flags
10. **Knowledge Graph is supplementary, not a 7th backend** — Graph retrieval is a signal in 3-way RRF (Dense + BM25 + Graph). It depends on the vector store for ingestion and cannot be evaluated as an independent backend. Use Graph A/B toggle in Compare instead.
11. **Score identity across backends** — All 6 backends use the same embedding model → identical RRF scores. Query latency is the only genuine differentiator in Compare.
12. **SPLADE pipeline cache key** — `retrieval.splade.enabled` is part of the pipeline cache key in `deps.py`. Toggling SPLADE from the frontend creates a new pipeline instance (with SPLADE index initialized). Toggle is per-query via `enable_splade` in `RetrievalMethods`.
13. **Encoder isolation via collection suffix** — `build_pipeline_config(embedding_model=...)` calls `_model_slug()` and appends it to `collection_name` (e.g. `polyrag_docs_bge-base`). Different-dimension models **never share a collection**. `embedding_model` is also in the pipeline cache key so different encoders never share a pipeline instance.
14. **Rate limiting** — `api/main.py` caps incoming request rate; returns HTTP 429 (load test treats this as success, not error)
15. **ScaleHints on RagProfile** — `embed_batch_size`, `max_doc_size_mb`, `bm25_persist`, `max_concurrent_requests` — all backward-compatible (defaults match prior behavior)

---

## Frontend Screens

| Route | Component | Purpose |
|-------|-----------|---------|
| `/search` | `SearchLab.tsx` | Interactive search with retrieval method toggles |
| `/ingest` | `IngestPanel.tsx` | Document ingestion with progress streaming |
| `/compare` | `ComparisonMatrix.tsx` | Side-by-side backend benchmark |
| `/evaluate` | `EvaluationStudio.tsx` | Ground-truth Q&A quality evaluation |
| `/graph` | `GraphExplorer.tsx` | Knowledge graph visualisation |
| `/traces` | `TraceViewer.tsx` | Execution trace logs |
| `/settings` | `Settings.tsx` | Backend connections, LM Studio, **Embedding Model selector** |

### Settings Screen (`/settings`)
- **Backend Connections** — host/port/apiKey per vector store backend
- **LM Studio** — URL + model name + connection test button
- **Embedding Model** — dropdown with all 3 supported encoders showing dim/size/MTEB score; persisted to localStorage; **auto-applied to every search and ingest request** via `useStore.getState().embeddingModel` in `api/search.ts` + `api/ingest.ts`

### Compare Screen Features (`/compare`)
1. **Chunk Preview Panel** — click per-query row → modal with retrieved chunks (👁) + Graph trail tab (🕸️)
2. **Bar Chart toggle** — table ↔ horizontal bar chart, metric selector
3. **Overlap Matrix** — Jaccard % of shared chunk IDs between backend pairs
4. **Winner Badges** — ⚡ Fastest · 🎯 Top Score · 📦 Most Results · 🕸️ Graph Boost
5. **Run History + Diff** — auto-saved to localStorage; diff two runs with ▲▼ indicators
6. **P95 Latency** — 1×/3×/5× repeat_runs toggle for P50/P95 percentile measurement
7. **Graph A/B toggle** — runs each query with graph ON vs OFF; shows Δ score column (green=helpful, red=harmful) + graph trail (entities as badges, paths as chains)

### Evaluate Screen Features (`/evaluate`)
- **F/R/S/G scoring**: Faithfulness · Relevance · Source Hit · **Graph Source Hit** (G%)
- **Graph trail** in expanded rows: entity badges (`PERSON:Hamlet`) + path chains (`Hamlet —[rel]→ Horatio`)
- **Chunk Browser** — browse existing chunks to auto-generate Q&A pairs
- **Score legend**: sky=F, purple=R, green=S, indigo=G

---

### Key Schema Types (`api/schemas.py`)

```python
EmbeddingModel = Literal[
    "all-MiniLM-L6-v2",        # default — 384-dim, fast
    "BAAI/bge-base-en-v1.5",   # 768-dim, balanced
    "BAAI/bge-large-en-v1.5",  # 1024-dim, best quality
]

IngestRequest:
  ...
  embedding_model: EmbeddingModel = "all-MiniLM-L6-v2"

SearchRequest:
  ...
  embedding_model: EmbeddingModel = "all-MiniLM-L6-v2"
```

### Compare
```python
CompareRequest:
  collection_name, corpus_text, corpus_path  # data source (choose one)
  backends: List[str]
  queries: List[str]
  full_retrieval: bool          # run LLM methods ON for full_top_score
  repeat_runs: int              # 1–10, enables P50/P95 latency
  compare_graph_ab: bool        # run each query graph ON vs OFF

CompareBackendResult:           # per-query row
  top_score, avg_score, kw_hits, result_count
  query_latency_ms, latency_p50_ms, latency_p95_ms
  chunk_ids, chunks             # for preview panel + overlap matrix
  graph_entities, graph_paths   # trail from graph-enabled run
  score_no_graph, score_delta   # A/B: delta = graph_on - graph_off
  latency_no_graph_ms, latency_with_graph_ms

CompareSummary:                 # per-backend aggregated row
  base_top_score, full_top_score, avg_score
  avg_query_latency_ms, latency_p50_ms, latency_p95_ms, total_result_count
  avg_score_no_graph, avg_score_delta   # graph A/B averages
  avg_latency_no_graph_ms, avg_latency_with_graph_ms
```

### Evaluate
```python
EvaluateRequest:
  questions: [{question, expected_answer, expected_sources}]
  backends: List[str]
  collection_name: str
  methods: RetrievalMethods     # enable_dense, enable_bm25, enable_graph, etc.

# Per-backend result (in stored eval, keyed by backend name):
{
  answer: str
  scores: { faithfulness, relevance, source_hit, graph_source_hit }
  graph_entities: List[str]     # "TYPE:entity_name"
  graph_paths: List[str]        # "Entity —[rel]→ Entity" strings
}
```

---

## Critical Implementation Notes

### `pipeline.ask()` vs `pipeline.ask_with_bundle()`
- `pipeline.ask(question, top_k, filters)` — simple, no retrieval flags
- `pipeline.ask_with_bundle(question, bundle, expansion_traces, top_k, enable_dense, enable_bm25, enable_splade, enable_graph, enable_rerank, enable_llm_graph)` — requires prior `expand_query()` call
- **Evaluate router** uses `expand_query()` once then `ask_with_bundle()` per backend
- **NEVER** pass `enable_dense`/`enable_bm25`/`enable_splade` etc. to `pipeline.ask()` — it will TypeError

### Graph Data in `RAGResponse`
- `resp.graph_entities: List[str]` — format `"ENTITY_TYPE:entity_text"` (e.g. `"PERSON:Hamlet"`)
- `resp.graph_paths: List[str]` — via `str(GraphPath)` → `.explanation` → `"Entity —[rel]→ Entity"`
- Per-chunk `metadata["_method_lineage"]` tags `"Knowledge Graph"` as source method
- `GraphPath` object: `query_entity`, `path_entities`, `path_types`, `chunk_ids`, `hop_distance`, `relevance_score`

### `build_pipeline_config()` in `api/deps.py`
- `enable_er: bool = True` — controls Knowledge Graph initialisation
- `enable_splade: bool = False` — controls SPLADE index initialisation; also part of pipeline cache key
- `embedding_model: str = "all-MiniLM-L6-v2"` — selects encoder; auto-scopes collection name via `_model_slug()`; part of cache key
- Pass `enable_er=False` for the "no graph" baseline in A/B comparison
- `full_retrieval: bool` — enables LLM query-intelligence methods
- **Cache key** includes: backend, scoped_collection, chunk_size, chunk_strategy, chunk_overlap, raptor.enabled, contextual_reranker.enabled, mmr.enabled, **splade.enabled**, **embedding.model**

### **DO NOT MODIFY** `orchestrator/pipeline.py` or `core/` modules carelessly
- All UI/API enhancements extract data already returned by the pipeline
- The pipeline is the stable, tested foundation — changes risk breaking all 11 phases
- Exception: SPLADE-related additions in `core/retrieval/splade.py` and `core/retrieval/hybrid.py` are the pattern for adding new retrieval signals
- Run `pytest tests/phase13/test_rag_router.py -v` (28 tests) before any commit to verify the API layer is intact

### Open PRs (do not merge out of order)
| PR | Branch | Covers |
|----|--------|--------|
| #1 | `feature/scalability-phase1` | Async /api/rag, rate limiting, multi-worker |
| #2 | `feature/scalability-phase2` | LRU cache, streaming chunker, persistent jobs |
| #3 | `feature/scalability-phase3` | System endpoints, ScaleHints, BM25 persistence |
| #4 | `feature/ci-pipeline` | GitHub Actions CI (merge first — unlocks branch protection) |
| #5 | `feature/ragas-eval` | RAGAS integration (LLM-judged metrics) |
| #6 | `feature/load-test` | Locust load test + PowerShell runner |

Recommended merge order: **#4 → #1 → #2 → #3 → #5 → #6**

---

## Updating This Context File

When making significant changes, update the relevant section(s) of this file:
- New API route → update **API Routes** table
- New phase/module → update **11 RAG Phases** table + **Repository Layout**
- New dependency → update **Tech Stack**
- New config option → update **Configuration**
- New Docker service → update **Docker Services**
- New frontend screen or feature → update **Frontend Screens**
- New schema field → update **Key Schema Types**
- New critical constraint → update **Critical Implementation Notes**
- New data file → update **Data Storage**
- New PR → update **Open PRs** table
