# PolyRAG

> **Stop guessing. Start knowing exactly why your RAG works — or doesn't.**

[![CI](https://github.com/anand08151947-dot/PolyRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/anand08151947-dot/PolyRAG/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/react-19-61DAFB)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)

---

Most teams building RAG-powered applications — whether for Agentic AI, enterprise search, or document Q&A — are flying blind. Retrieval is unreliable but there is no visibility into why. Multiple retrieval signals fire on every query but no one knows which ones are actually contributing. There is no ground-truth evaluation, so quality is judged by feel. When something breaks in production, there is no trace to follow.

**PolyRAG is the workbench that changes that.**

It is an open-source platform for any team that needs RAG to work reliably before they ship it — in an agentic pipeline, a product feature, or a proof of concept. Rather than adding RAG and hoping for the best, PolyRAG gives your team a structured environment to evaluate, compare, and validate retrieval quality with evidence:

- **See inside every query** — a step-by-step retrieval trace shows exactly which of 12 methods fired, what score each contributed before and after fusion, and why a chunk ranked where it did
- **Compare backends with data** — run the same query across all six vector stores simultaneously and see scored results side-by-side, not synthetic benchmarks
- **Measure before you ship** — build a ground-truth Q&A dataset and auto-score every backend × retrieval combination for faithfulness, relevance, and source accuracy
- **Ingest with confidence** — preview exactly how documents get chunked before a single byte is written to any vector store

Everything runs locally. No cloud account required. LLM is optional — retrieval, tracing, and evaluation all work without one.

---

## Contents

- [Quick Start](#quick-start)
- [The Workbench](#the-workbench)
  - [Search Lab](#search-lab)
  - [Comparison Matrix](#comparison-matrix)
  - [Evaluation Studio](#evaluation-studio)
  - [Ingestion Studio](#ingestion-studio)
  - [Knowledge Graph Explorer](#knowledge-graph-explorer)
- [Retrieval Pipeline](#retrieval-pipeline)
- [LLM Configuration](#llm-configuration)
- [Vector Backends](#vector-backends)
- [REST API](#rest-api)
- [Configuration Reference](#configuration-reference)
- [Python API](#python-api)
- [Test Coverage](#test-coverage)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the browser UI)
- Windows (PowerShell scripts provided; pip + npm work cross-platform)

### Install

```powershell
git clone https://github.com/anand08151947-dot/PolyRAG.git
cd PolyRAG
.\install.ps1          # core dependencies
.\install.ps1 -Full    # includes Weaviate, PGVector client libraries
```

### Run

```powershell
# Terminal 1 - API server (port 8000)
.\start.ps1

# Terminal 2 - React frontend (port 3000)
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

> **No LLM required for retrieval.** To enable LLM-assisted features (query rewriting, HyDE, contextual re-ranking, answer generation), go to **Settings -> LLM Configuration** and configure any supported provider, or start LM Studio locally with no API key.

### Optional: Docker vector backends

ChromaDB and FAISS run in-process with no setup. For Qdrant, Weaviate, Milvus, or PGVector:

```powershell
docker compose -f docker-compose.polyrag.yml up -d
```

---

## The Workbench

### Search Lab

Search Lab is the core research environment. It exposes the full retrieval pipeline so you can understand not just *what* was returned, but *why*.

**What you can do:**

- Toggle any combination of 12 retrieval methods on or off per query
- See per-result cards with: raw score, text snippet, source document, exact character span, and confidence verdict
- Open the **Retrieval Trace Panel** — a step-by-step audit showing which methods fired, their individual pre-fusion scores, the RRF-combined score, and the rerank delta
- Compare two retrieval configurations side-by-side on the same query using **A/B Config Testing** — save any configuration as a named profile and replay it any time
- Visualise the live retrieval pipeline as a React Flow diagram — which nodes activated and in what order

**Retrieval methods available:**

| Group | Methods |
|---|---|
| **Lexical** | BM25 keyword · SPLADE sparse neural (learned term expansion) |
| **Semantic** | Dense vector search (MiniLM / BGE embeddings) |
| **Graph** | Knowledge graph entity traversal (spaCy + Kuzu) |
| **Re-ranking** | Cross-encoder neural · MMR diversity · Contextual LLM |
| **Query intelligence** | Query rewriting · HyDE expansion · Multi-query ensemble |
| **Hierarchical** | RAPTOR cluster-summary retrieval |

All active signals are fused automatically via **Reciprocal Rank Fusion (RRF)**.

---

### Comparison Matrix

The Comparison Matrix answers the question: *which vector backend and retrieval configuration performs best on my data?*

Rather than relying on synthetic benchmarks, it runs your own query against all six backends using configurable method combinations and presents the results in a single scored table.

**What you get:**

- Results for all six backends — ChromaDB, FAISS, Qdrant, Weaviate, Milvus, PGVector — in one view
- Two score columns per backend: **base retrieval** (dense only) vs **full hybrid** (with re-ranking, MMR, and graph)
- Scored metrics: top relevance score, keyword hit rate, ingest latency
- Click any column header to rank backends by any metric
- Export the full matrix as **CSV** or **JSON** for offline analysis or reporting

---

### Evaluation Studio

Evaluation Studio gives you a systematic, repeatable quality measurement loop — essential when choosing between retrieval strategies or tuning configuration for a specific corpus.

**Workflow:**

1. Create a ground-truth Q&A dataset — questions, reference answers, and expected source documents
2. Click **Run Evaluation** — every question is automatically scored against every backend x method combination
3. Review the results matrix — a backend x method grid of all scores — to pinpoint exactly where retrieval succeeds or breaks down

**Quality metrics:**

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer assert only what the retrieved chunks actually support? |
| **Answer relevance** | Does the answer address the question that was asked? |
| **Source hit rate** | Did the expected source document appear in the top-K results? |
| **Context precision** | Are the retrieved chunks actually relevant to the question? (RAGAS) |
| **Context recall** | Did retrieval surface all the chunks needed to answer? (RAGAS) |

RAGAS scores (faithfulness, answer_relevancy, context_precision, context_recall) are computed when an LLM provider is configured. All other metrics run without an LLM.

---

### Ingestion Studio

Ingestion Studio manages the full document ingestion pipeline with real-time feedback.

- **Supported formats** — PDF (`.pdf`), PowerPoint (`.pptx`), and plain-text formats (`.txt`, `.md`, `.rst`, `.csv`, `.json`, `.xml`, `.html`). Legacy `.ppt` files must be saved as `.pptx` before ingesting. Scanned/image-only PDFs are not supported (no OCR).
- **Drag-and-drop batch queue** — upload multiple files at once with per-file progress tracking
- **Chunking strategy selector** — choose from fixed-overlap, sentence-boundary, section-aware, or semantic-boundary per batch
- **Chunk Preview** — dry-run any document through the selected strategy and inspect the output (color-coded boundaries, parent-child hierarchy, detected entity spans) before writing anything to a vector store
- **Backend multi-select** — ingest to one or multiple backends simultaneously
- **React Flow DAG** — a live pipeline diagram: Upload -> Chunk -> Embed -> Knowledge Graph -> Upsert
- **SSE live log stream** — real-time chunking and embedding events streamed to the browser

---

### Knowledge Graph Explorer

An interactive D3.js force-directed graph of all entities and relationships extracted from your corpus.

- Entities are color-coded by type: PERSON, ORG, LOCATION, CONCEPT
- Click any node to jump to the source chunks that mention that entity
- Select any collection from the sidebar — the graph updates instantly
- Graph traversal is automatically included in Search Lab queries when enabled, contributing an entity-relationship signal to RRF fusion

---

## Retrieval Pipeline

All retrieval methods are composable and fused via Reciprocal Rank Fusion (RRF). The default configuration is a **3-way hybrid** (Dense + BM25 + Knowledge Graph). Enabling SPLADE adds a fourth sparse-neural signal for the highest-recall configuration.

| Method | Signal | LLM needed | On by default |
|---|---|:---:|:---:|
| Dense vector | Semantic similarity via sentence embeddings (MiniLM / BGE) | No | Yes |
| BM25 | Exact token overlap, TF-IDF weighting | No | Yes |
| SPLADE sparse neural | Learned term expansion via masked-LM head — naver/splade-cocondenser-selfdistil (~110 MB, Apache 2.0) | No | Config opt-in |
| Knowledge graph traversal | Entity relationship chains extracted by spaCy + Kuzu, N-hop BFS | No | Yes (when graph enabled) |
| Cross-encoder re-ranking | Neural passage relevance score (ms-marco-MiniLM) | No | Yes |
| MMR diversity | Maximal Marginal Relevance — penalises near-duplicate results | No | Yes |
| Query rewriting | Rewrites ambiguous or verbose queries for clarity and specificity | Yes | Yes (when LLM available) |
| HyDE | Generates a hypothetical answer, embeds it, retrieves by embedding similarity | Yes | Yes (when LLM available) |
| Multi-query ensemble | Generates query paraphrases, retrieves for each, merges results | Yes | Config opt-in |
| RAPTOR | Hierarchical cluster-summary index for broad or conceptual queries | Yes | Config opt-in |
| Contextual LLM re-ranking | Batched LLM call scores all top-K candidates | Yes | Config opt-in |
| LLM graph extraction | LLM-assisted entity and relation extraction at ingest time | Yes | Config opt-in |

### Confidence scoring

Every response includes a composite confidence verdict derived from seven signals: score distribution, source agreement, coverage breadth, cross-document consistency, temporal relevance, classification compliance, and noise ratio.

| Verdict | Composite score | Meaning |
|---|---|---|
| `HIGH` | >= 0.75 | Strong evidence, high source agreement |
| `MEDIUM` | 0.45 - 0.75 | Moderate evidence, some agreement |
| `LOW` | 0.20 - 0.45 | Weak evidence, low agreement |
| `INSUFFICIENT_EVIDENCE` | < 0.20 | No reliable answer found |

---

## LLM Configuration

LLM interaction is **fully config-driven** — configure your provider, model, base URL, and API key from **Settings -> LLM Configuration** in the browser. Changes take effect immediately, no restart required.

| Provider | API key required | Default base URL |
|---|:---:|---|
| LM Studio | No | `http://localhost:1234/v1` |
| Ollama | No | `http://localhost:11434/v1` |
| OpenAI | Yes | `https://api.openai.com/v1` |
| Groq | Yes | `https://api.groq.com/openai/v1` |
| Azure OpenAI | Yes | Your deployment endpoint |
| Google Gemini | Yes | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Anthropic | Yes | `https://api.anthropic.com` |

**Graceful degradation:** if the configured LLM is offline or unreachable, all LLM-dependent retrieval methods are automatically skipped. Dense vector, BM25, SPLADE, cross-encoder re-ranking, and MMR continue to work — retrieval does not stop.

**Getting started with LM Studio (free, fully local):**

1. Download [LM Studio](https://lmstudio.ai/) and load any GGUF model (e.g. mistralai/ministral-3b)
2. Start the local server (default port 1234)
3. In the browser: **Settings -> LLM Configuration -> Provider: LM Studio** -> enter model name -> **Save**

---

## Vector Backends

All six backends share an identical query interface. Switch by changing one line in `config/config.yaml` or from the Settings UI — no code changes required.

| Backend | Run mode | Cloud option | Native hybrid search |
|---|---|---|:---:|
| **ChromaDB** | In-process or server | Chroma Cloud | No (RRF-emulated) |
| **FAISS** | In-process, file-based | — | No (RRF-emulated) |
| **Qdrant** | In-process, Docker, or remote | Qdrant Cloud | Yes |
| **Weaviate** | Embedded, Docker, or remote | Weaviate Cloud | Yes |
| **Milvus** | Milvus Lite, Docker, or remote | Zilliz Cloud | Yes |
| **PGVector** | Docker or any PostgreSQL host | Any managed PostgreSQL | Yes (tsvector) |

```yaml
# config/config.yaml
store:
  backend: qdrant   # chromadb | faiss | qdrant | weaviate | milvus | pgvector

  # Connection URLs for Docker or remote deployments:
  qdrant_url: http://localhost:6333
  weaviate_url: http://localhost:8080
  milvus_uri: http://localhost:19530
  pgvector_dsn: postgresql://postgres:postgres@localhost:5432/polyrag
```

```powershell
# Start all Docker-based backends at once:
docker compose -f docker-compose.polyrag.yml up -d
```

---

## REST API

The FastAPI server runs on port 8000. All browser dashboard features are built on these endpoints.

| Endpoint | Description |
|---|---|
| `POST /api/rag` | Unified RAG endpoint — answer + citations + confidence + graph paths |
| `GET/POST/PUT/DELETE /api/rag/profiles` | Named config profiles — save and replay retrieval configurations |
| `POST /api/search` | Multi-backend query with full retrieval trace |
| `POST /api/compare` | Cross-backend comparison matrix |
| `POST /api/ingest` | Start a batch ingestion job — accepts PDF (`.pdf`), PowerPoint (`.pptx`), and plain-text files |
| `GET /api/ingest/{id}/stream` | SSE stream of live chunking and embedding events |
| `POST /api/chunks/preview` | Dry-run chunking without ingesting |
| `POST /api/evaluate` | Score a ground-truth dataset (faithfulness / relevance / RAGAS) |
| `GET /api/evaluate/ragas-status` | Check RAGAS + LLM availability |
| `GET /api/backends` | Health check and ping for all six backends |
| `GET /api/collections` | Collection CRUD across backends |
| `GET /api/graph/{collection}` | Entity/relation graph for D3 visualisation |
| `GET /api/config/llm` | Get current LLM configuration |
| `PUT /api/config/llm` | Update provider, model, or API key (live reload) |
| `GET /api/config/llm/providers` | List all supported providers |
| `GET /api/config/llm/test` | Test connectivity to the configured LLM |
| `POST /api/feedback` | Submit thumbs-up / thumbs-down on a result chunk |
| `GET /api/jobs` | Job history and status |
| `GET /api/system/health` | Uptime, pipeline cache stats, job counts |
| `GET/DELETE /api/system/cache` | Inspect or flush the LRU pipeline cache |

### Request example

```json
POST /api/rag

{
  "question": "What are the key data retention requirements?",
  "collection_name": "my_docs",
  "backend": "qdrant",
  "embedding_model": "all-MiniLM-L6-v2",
  "methods": {
    "enable_dense": true,
    "enable_bm25": true,
    "enable_splade": true,
    "enable_graph": true,
    "enable_rerank": true,
    "enable_mmr": true,
    "enable_rewrite": true
  },
  "top_k": 5
}
```

### Response example

```json
{
  "answer": "Data must be retained for seven years under...",
  "confidence": {
    "verdict": "HIGH",
    "composite_score": 0.87
  },
  "citations": ["Policy v2.1, Section 4 (chars 1420-1680)"],
  "provenance": [
    { "doc_id": "policy_v2.1", "source": "policy.pdf", "span_start": 1420, "span_end": 1680 }
  ],
  "results": [
    {
      "score": 0.91,
      "text": "...",
      "metadata": {
        "source": "policy.pdf",
        "chunk_strategy": "sentence_boundary",
        "_method_lineage": ["Dense Vector", "SPLADE Sparse", "BM25 Keyword", "Knowledge Graph"]
      }
    }
  ],
  "graph_entities": ["ORG:Data Protection Authority", "CONCEPT:retention period"],
  "graph_paths": ["retention period --[governed_by]--> Data Protection Authority"]
}
```

---

## Configuration Reference

All configuration lives in `config/config.yaml`. LLM settings are also manageable live from the Settings page.

```yaml
store:
  backend: chromadb           # chromadb | faiss | qdrant | weaviate | milvus | pgvector

embedding:
  model: all-MiniLM-L6-v2    # 384-dim, CPU-only, ~80 MB
  # Also: BAAI/bge-base-en-v1.5 (768-dim) | BAAI/bge-large-en-v1.5 (1024-dim)

ingestion:
  collection_name: polyrag
  chunk_size: 512
  chunk_overlap: 64
  enable_rich_formats: true    # false to restrict ingestion to plain-text files only; requires pypdf + python-pptx

llm:
  provider: lm_studio         # lm_studio | openai | ollama | groq | azure_openai | gemini | anthropic
  base_url: http://localhost:1234/v1
  model: mistralai/ministral-3b
  api_key: ""                 # leave empty for LM Studio and Ollama
  enable_rewrite: true
  enable_hyde: true
  enable_multi_query: false

retrieval:
  splade:
    enabled: false            # true to enable sparse neural retrieval
    model: naver/splade-cocondenser-selfdistil   # ~110 MB, Apache 2.0, downloaded on first use
    persist_dir: ./data/splade

graph:
  backend: kuzu               # networkx (tests) | kuzu (default) | neo4j (enterprise)
  enabled: true
  max_hops: 2

advanced_retrieval:
  raptor:
    enabled: false            # hierarchical cluster summaries - requires LLM
  contextual_reranker:
    enabled: false            # batched LLM re-ranking - requires LLM
  mmr:
    enabled: true
    diversity_weight: 0.3     # 0.0 = pure relevance | 1.0 = pure diversity

access:
  user_clearance: INTERNAL    # PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED

quality:
  min_score: 0.2
  dedup_threshold: 0.85
```

---

## Python API

PolyRAG can also be used directly as a Python library, bypassing the REST API entirely.

```python
from orchestrator.pipeline import RAGPipeline

pipeline = RAGPipeline()   # loads config/config.yaml
pipeline.start()

# Ingest
pipeline.ingest_text(
    "Your document content...",
    metadata={"source": "my_doc.pdf", "classification": "INTERNAL"}
)

# Retrieve - returns a ranked list of SearchResult objects
results = pipeline.query("What are the data retention requirements?", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.document.text[:120]}")
    print(f"  via: {r.metadata.get('_method_lineage')}")

# Full RAG - answer + citations + confidence + graph paths
response = pipeline.ask("Summarise the key requirements.", top_k=5)
print(response.answer)
print("Confidence:", response.confidence.verdict, response.confidence.composite_score)
print(response.graph_explanation())

pipeline.stop()
```

### Access control

Tag documents with a classification level. Queries only surface results the requesting user is cleared to see:

```python
pipeline.ingest_text(text, metadata={"classification": "CONFIDENTIAL"})

# A user with INTERNAL clearance will not receive CONFIDENTIAL chunks
results = pipeline.query("...", user_context={"clearance": "INTERNAL"})
```

Levels (ascending): `PUBLIC` | `INTERNAL` | `CONFIDENTIAL` | `RESTRICTED`

---

## Test Coverage

**443 automated tests — all passing. Runs on every commit via GitHub Actions (Python 3.10 + 3.11 matrix).**

| Area | What is tested | Tests |
|---|---|:---:|
| Vector store adapters | CRUD, collection management, metadata filtering, batch upsert — all 6 backends | 98 |
| Semantic chunking | Fixed-overlap, sentence-boundary, section-aware, semantic-boundary, quality gate | 20 |
| Hybrid search and RRF fusion | Dense + BM25 + SPLADE retrieval, RRF fusion, metadata filters, per-backend parity | 19 |
| Multi-stage retrieval | Broad recall -> parent expansion -> cross-encoder re-ranking pipeline | 12 |
| Query intelligence | Query rewriting, HyDE, multi-query, conversation context, step-back prompting | 8 |
| Provenance and citations | Exact text span attribution, citation builder, audit log, version registry | 18 |
| Confidence scoring | 7-signal aggregator: score distribution, source agreement, coverage, conflict detection | 19 |
| Access control | Temporal relevance, data classification, access policy enforcement | 24 |
| Noise and quality control | MinHash dedup, quality scoring, cross-document aggregation, observability | 23 |
| Knowledge graph | Entity/relation extraction, Kuzu graph store, 3-way hybrid search, N-hop traversal | 53 |
| Advanced retrieval | RAPTOR hierarchical index, LLM entity extraction, contextual re-ranking, MMR | 53 |
| End-to-end pipeline | Full ingest -> hybrid retrieve -> rerank -> answer -> provenance -> confidence flow | 59 |
| Frontend components | React pages, API client mocks, routing, document library, comparison matrix | 35 |
| **Total** | | **443** |

No external services or API keys required for the core test suite.

```powershell
pytest tests/ -q -m "not integration and not lmstudio"
cd frontend && npm test
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| API server | [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) |
| Frontend framework | [React 19](https://react.dev/) + [Vite 5](https://vitejs.dev/) + TypeScript |
| UI styling | [Tailwind CSS v3](https://tailwindcss.com/) |
| State and data fetching | [Zustand](https://zustand-demo.pmnd.rs/) + [TanStack Query](https://tanstack.com/query) |
| Flow diagrams | [React Flow 12](https://reactflow.dev/) |
| Graph visualisation | [D3.js v7](https://d3js.org/) |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) — MiniLM-L6-v2, BGE-base, BGE-large |
| Sparse neural retrieval | [sentence-transformers SparseEncoder](https://www.sbert.net/) — naver/splade-cocondenser-selfdistil |
| Keyword search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| NLP / NER | [spaCy](https://spacy.io/) en_core_web_sm |
| Knowledge graph | [Kuzu](https://kuzudb.com/) (default) · [NetworkX](https://networkx.org/) (tests) · [Neo4j](https://neo4j.com/) (optional) |
| Deduplication | [datasketch](https://github.com/ekzhu/datasketch) MinHash LSH |
| RAG evaluation | [RAGAS](https://docs.ragas.io/) — faithfulness, answer relevancy, context precision/recall |
| Load testing | [Locust](https://locust.io/) |
| Frontend testing | [Vitest](https://vitest.dev/) + [React Testing Library](https://testing-library.com/) |
| Live streaming | Server-Sent Events via [sse-starlette](https://github.com/sysid/sse-starlette) |
| PDF parsing | [pypdf](https://github.com/py-pdf/pypdf) — text extraction from PDF documents |
| PowerPoint parsing | [python-pptx](https://python-pptx.readthedocs.io/) — slide text and speaker-notes extraction from `.pptx` files |

All components are open-source with permissive licences. A fully local, API-key-free setup is possible using LM Studio or Ollama for the LLM layer.

---

## Contributing

Contributions are welcome — open an issue or pull request.

```powershell
# Run before submitting
pytest tests/ -q -m "not integration and not lmstudio"
cd frontend && npm test
```

---

*MIT License*