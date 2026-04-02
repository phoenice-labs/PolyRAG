# PolyRAG API Integration Guide

> **For agentic AI workflows, LLM pipelines, and external integrations.**
> Base URL: `http://localhost:8000/api`
> Interactive docs (Swagger UI): http://localhost:8000/docs
> OpenAPI schema: http://localhost:8000/openapi.json

---

## ⭐ Unified Agentic RAG API — Start Here

> **If you're integrating PolyRAG into an agentic AI flow, this is the only endpoint you need.**

`POST /api/rag` — single call, full traceability, no configuration noise.

```python
import requests

# Step 1 — Create a profile once (after you've evaluated your optimal config)
profile = requests.post("http://localhost:8000/api/rag/profiles", json={
    "name": "production-v1",
    "description": "Tested 2026-03 — Qdrant + BGE-base, dense+bm25+rerank",
    "backend": "qdrant",
    "collection_name": "product_docs",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "top_k": 8,
    "methods": {
        "enable_dense": True,
        "enable_bm25": True,
        "enable_rerank": True,
        "enable_mmr": True
    },
    "confidence_thresholds": { "high": 0.8, "medium": 0.5, "low": 0.3 }
}).json()
profile_id = profile["id"]

# Step 2 — Use in your agent (one line, forever)
answer = requests.post("http://localhost:8000/api/rag", json={
    "query": "What is our refund policy for digital goods?",
    "profile_id": profile_id
}).json()

print(answer["answer"])            # The answer
print(answer["verdict"])           # HIGH | MEDIUM | LOW | INSUFFICIENT
print(answer["answer_confidence"]) # 0.87
print(answer["sources"][0]["metadata"]["source"])          # document name
print(answer["sources"][0]["method_lineage"])              # which methods found it
print(answer["pipeline_audit"]["methods_active"])          # what ran
print(answer["pipeline_audit"]["latency_ms"])              # 142.5
```

### The Developer Journey to Production

```
1. Ingest docs → POST /api/ingest
2. Experiment  → POST /api/search  (try all 12 methods, compare backends)
3. Evaluate    → POST /api/evaluate (measure faithfulness, relevance, accuracy)
4. Compare     → POST /api/compare  (A/B benchmark across backends)
5. Settle      → POST /api/rag/profiles  (save your tested config as a named profile)
6. Deploy      → POST /api/rag  { "query": "...", "profile_id": "..." }
```

### `/api/rag` Request

```json
{
  "query": "What is our refund policy for digital goods?",

  "profile_id": "abc-123",

  // All fields below are optional — override the profile at query time
  "backend": "qdrant",
  "collection_name": "product_docs",
  "embedding_model": "BAAI/bge-base-en-v1.5",
  "top_k": 10,
  "methods": {
    "enable_dense": true,
    "enable_bm25": true,
    "enable_splade": false,
    "enable_graph": false,
    "enable_rerank": true,
    "enable_mmr": true,
    "enable_rewrite": false,
    "enable_multi_query": false,
    "enable_hyde": false,
    "enable_raptor": false,
    "enable_contextual_rerank": false,
    "enable_llm_graph": false
  },
  "confidence_thresholds": {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3
  }
}
```

### `/api/rag` Response (`RagAnswer`)

```json
{
  "query": "What is our refund policy for digital goods?",
  "profile_id": "abc-123",
  "profile_name": "production-v1",

  // ── The Answer ────────────────────────────────────────────────────
  "answer": "Digital goods are non-refundable after download...",
  "answer_confidence": 0.871,
  "verdict": "HIGH",

  // ── Sources with Full Lineage ─────────────────────────────────────
  "sources": [
    {
      "chunk_id": "chunk-0042",
      "text": "All digital purchases are final once downloaded...",
      "score": 0.921,
      "confidence": 0.94,
      "metadata": {
        "source": "refund-policy.pdf",
        "section_title": "Digital Goods Policy",
        "start_char": 1820,
        "end_char": 2140
      },
      "method_lineage": [
        { "method": "Dense Vector",   "rank": 1, "rrf_contribution": 0.0164 },
        { "method": "BM25 Keyword",   "rank": 2, "rrf_contribution": 0.0154 },
        { "method": "Cross-Encoder Rerank", "rank": 1, "rrf_contribution": 0.0164 }
      ],
      "post_processors": ["CrossEncoderReRanker", "MMRReranker"]
    }
  ],

  // ── Pipeline Audit ────────────────────────────────────────────────
  "pipeline_audit": {
    "backend": "qdrant",
    "collection": "product_docs_bge-base",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "methods_active": ["Dense Vector", "BM25 Keyword", "Cross-Encoder Rerank", "MMR Diversity"],
    "funnel": [
      { "method": "Dense Vector",        "candidates_before": 50, "candidates_after": 20 },
      { "method": "BM25 Keyword",        "candidates_before": 50, "candidates_after": 15 },
      { "method": "RRF Fusion",          "candidates_before": 35, "candidates_after": 10 },
      { "method": "Cross-Encoder Rerank","candidates_before": 10, "candidates_after": 8  },
      { "method": "MMR Diversity",       "candidates_before": 8,  "candidates_after": 5  }
    ],
    "latency_ms": 142.5
  },

  // ── Knowledge Graph (if graph enabled) ───────────────────────────
  "graph": {
    "entities": ["digital goods", "refund", "download"],
    "paths": ["digital goods --[GOVERNED_BY]--> refund policy"]
  },

  // ── LLM Traces ───────────────────────────────────────────────────
  "llm_traces": [],

  "timestamp": "2026-03-20T15:11:00.000Z"
}
```

### Profile CRUD

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/rag/profiles` | Create a profile from your tested config |
| `GET` | `/api/rag/profiles` | List all saved profiles |
| `GET` | `/api/rag/profiles/{id}` | Get a specific profile |
| `PUT` | `/api/rag/profiles/{id}` | Update a profile (id + created_at preserved) |
| `DELETE` | `/api/rag/profiles/{id}` | Delete a profile |

### ScaleHints (`scale_hints`)

`scale_hints` are operational tuning parameters embedded in a profile. They do **not** affect retrieval quality — only resource usage and throughput.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_batch_size` | `int` | `32` | Chunks per embedding batch (range: 1–512). Increase for GPU throughput; decrease for memory-constrained CPU. |
| `max_doc_size_mb` | `float` | `200.0` | Streaming threshold (MB). Files larger than this are chunked via the streaming ingestion path. |
| `bm25_persist` | `bool` | `true` | Persist BM25 index to disk (`data/bm25/`) between restarts. Set `false` to rebuild from scratch each time. |
| `max_concurrent_requests` | `int` | `5` | Pipeline concurrency cap (range: 1–100). Controls how many simultaneous RAG requests a single pipeline instance handles. |

### Confidence Verdict Logic

The `verdict` field uses profile-configurable thresholds:

| Verdict | Meaning | Default threshold |
|---------|---------|-------------------|
| `HIGH` | Strong evidence, high certainty | score ≥ 0.8 |
| `MEDIUM` | Good evidence, some uncertainty | score ≥ 0.5 |
| `LOW` | Weak evidence, treat as a hint | score ≥ 0.3 |
| `INSUFFICIENT` | Not enough relevant content found | score < 0.3 |

Override per profile: `"confidence_thresholds": { "high": 0.85, "medium": 0.6, "low": 0.35 }`

---

## Rate Limits

> | Endpoint | Limit |
> |---|---|
> | `POST /api/rag` | 60 req/min |
> | `POST /api/search` | 30 req/min |
> | `POST /api/ingest` | 10 req/min |
> | All others | 120 req/min |
>
> HTTP 429 is returned when exceeded. A `Retry-After` header indicates when to retry.

---

## Table of Contents

1. [Unified Agentic RAG API ⭐](#-unified-agentic-rag-api--start-here) ← **Start here for integration**
2. [Quick Start — Ask a Question](#2-quick-start--ask-a-question)
3. [Ingest Documents](#3-ingest-documents)
4. [Search / Ask (Full API)](#4-search--ask-full-api)
   - [Request Parameters](#request-parameters)
   - [Choosing a Vector Backend](#choosing-a-vector-backend)
   - [Choosing Retrieval Methods](#choosing-retrieval-methods)
   - [Choosing an Embedding Model](#choosing-an-embedding-model)
   - [Full Response with Traceability](#full-response-with-traceability)
5. [Traceability Fields Reference](#5-traceability-fields-reference)
6. [Retrieval Trails (Audit Log)](#6-retrieval-trails-audit-log)
7. [LLM Trace History](#7-llm-trace-history)
8. [Knowledge Graph API](#8-knowledge-graph-api)
9. [Backend Health & Discovery](#9-backend-health--discovery)
10. [Evaluate Retrieval Quality](#10-evaluate-retrieval-quality)
11. [Compare Backends](#11-compare-backends)
12. [Agentic Integration Patterns](#12-agentic-integration-patterns)
13. [Method Combination Recipes](#13-method-combination-recipes)
14. [Chunks (Dry-run Preview)](#14-chunks-dry-run-preview)
15. [LLM Configuration API](#15-llm-configuration-api)
16. [Prompt Management](#16-prompt-management)
17. [Purge](#17-purge)
18. [System / Operations](#18-system--operations)
19. [Jobs](#19-jobs)
20. [Feedback](#20-feedback)
21. [API Reference Summary](#20-api-reference-summary)

---

## 1. Quick Start — Ask a Question

The minimal call to get an answer with full traceability:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does Hamlet say about mortality?",
    "backends": ["faiss"],
    "collection_name": "polyrag_docs",
    "top_k": 5
  }'
```

```python
import httpx

resp = httpx.post("http://localhost:8000/api/search", json={
    "query": "What does Hamlet say about mortality?",
    "backends": ["faiss"],
    "collection_name": "polyrag_docs",
    "top_k": 5,
})
data = resp.json()

# Top answer
print(data["results"]["faiss"]["answer"])

# Retrieved chunks with scores
for chunk in data["results"]["faiss"]["chunks"]:
    print(chunk["score"], chunk["text"][:100])
```

---

## 2. Ingest Documents

### POST `/api/ingest`

Ingestion runs as a background job. One job is created per backend.

#### Request Body

```json
{
  "text": "Full document text...",
  "backends": ["faiss", "chromadb"],
  "collection_name": "my_collection",
  "chunk_strategy": "sentence",
  "chunk_size": 400,
  "overlap": 50,
  "enable_er": true,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | — | Raw document text to ingest. Use this **or** `corpus_path`. |
| `corpus_path` | `string` | — | Server-side file path (admin use). |
| `backends` | `string[]` | `["faiss"]` | One or more vector backends to ingest into. |
| `collection_name` | `string` | `"polyrag_docs"` | Logical namespace for this data. The actual collection stored is `{name}_{model_slug}` (e.g. `my_collection_minilm`). |
| `chunk_strategy` | `string` | `"section"` | `section` \| `sliding` \| `sentence` \| `paragraph` |
| `chunk_size` | `int` | `400` | Target characters per chunk. |
| `overlap` | `int` | `50` | Overlap characters between adjacent chunks. |
| `enable_er` | `bool` | `true` | Extract named entities + relations into the knowledge graph during ingest. |
| `embedding_model` | `string` | `"all-MiniLM-L6-v2"` | Encoder used to embed chunks. **Must match the encoder used at query time.** |
| `enable_splade` | `bool` | `false` | Build a SPLADE sparse-neural index during ingest. Required before using `enable_splade` at search time. |

#### Chunk Strategy Guide

| Strategy | Best For | Notes |
|----------|----------|-------|
| `sentence` | Q&A, precise retrieval | Respects sentence boundaries; best semantic coherence |
| `section` | Long documents, books | Splits on paragraph/heading markers |
| `sliding` | Dense technical text | Fixed-size window with overlap; no boundary awareness |
| `paragraph` | Articles, wikis | Splits on blank lines |

#### Response

```json
{
  "job_ids": {
    "faiss": "job-abc123",
    "chromadb": "job-def456"
  }
}
```

### GET `/api/ingest/{job_id}/status`

Poll job status:

```python
status = httpx.get(f"http://localhost:8000/api/ingest/{job_id}/status").json()
# status: "pending" | "running" | "done" | "error"
print(status["status"], status["result"])
```

Response fields: `job_id`, `status`, `backend`, `created_at`, `updated_at`, `log_lines[]`, `result.upserted`, `result.total_chunks`, `error`.

### GET `/api/ingest/{job_id}/stream`

Server-Sent Events stream for real-time progress:

```python
import httpx

with httpx.stream("GET", f"http://localhost:8000/api/ingest/{job_id}/stream") as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:])
```

### Supported File Formats for `corpus_path`

In addition to plain-text files, `corpus_path` now accepts **PDF (`.pdf`) and PowerPoint (`.pptx`)** files directly.

| Format | Notes |
|--------|-------|
| `.txt`, `.md` | Plain text — always supported |
| `.pdf` | Text-layer PDFs supported. Scanned/image-only PDFs are **not** supported (no OCR). Requires `pypdf>=4.0` (included in `requirements.txt`). |
| `.pptx` | Slide text and notes extracted. Legacy `.ppt` files must be converted to `.pptx` first. Requires `python-pptx>=1.0` (included in `requirements.txt`). |

Controlled by the `enable_rich_formats` config flag in `config/config.yaml` (default: `true`).

### GET `/api/ingest/jobs`

Returns a list of all ingestion jobs across all backends, sorted newest-first.

```bash
curl http://localhost:8000/api/ingest/jobs
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Job identifier |
| `status` | `string` | `pending` \| `running` \| `done` \| `error` |
| `backend` | `string` | Target vector backend |
| `collection_name` | `string` | Target collection |
| `created_at` | `string` | ISO 8601 creation timestamp |
| `updated_at` | `string` | ISO 8601 last-update timestamp |

---

## 3. Search / Ask (Full API)

### POST `/api/search`

The core retrieval endpoint. Returns answers, chunks, scores, provenance, method lineage, and LLM traces — all in one call.

#### Full Request Schema

```json
{
  "query": "string — required",
  "backends": ["faiss"],
  "collection_name": "polyrag_docs",
  "top_k": 5,
  "embedding_model": "all-MiniLM-L6-v2",
  "methods": {
    "enable_dense": true,
    "enable_bm25": true,
    "enable_splade": false,
    "enable_graph": true,
    "enable_rerank": true,
    "enable_mmr": true,
    "enable_rewrite": false,
    "enable_multi_query": false,
    "enable_hyde": false,
    "enable_raptor": false,
    "enable_contextual_rerank": false,
    "enable_llm_graph": false
  }
}
```

### Request Parameters

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | **required** | The user's question in natural language. |
| `backends` | `string[]` | `["faiss"]` | Which vector databases to query. All run in parallel. See [Choosing a Vector Backend](#choosing-a-vector-backend). |
| `collection_name` | `string` | `"polyrag_docs"` | The collection to search. Must match the collection used during ingest. |
| `top_k` | `int` | `5` | Number of chunks to return per backend (range: 1–50). |
| `embedding_model` | `string` | `"all-MiniLM-L6-v2"` | **Must match the model used during ingest**. Different models use separate collections automatically. |

---

### Choosing a Vector Backend

| Backend | `backends` value | Docker Required | Best For |
|---------|------------------|-----------------|----------|
| FAISS | `"faiss"` | ❌ No | Fast in-process, dev/test, single node |
| ChromaDB | `"chromadb"` | ❌ No | Local persistence, easy setup |
| Qdrant | `"qdrant"` | ✅ Yes | Production, filtering, payload indexing |
| Weaviate | `"weaviate"` | ✅ Yes | Multi-tenant, hybrid search built-in |
| Milvus | `"milvus"` | ✅ Yes | High-scale, billion-vector datasets |
| PGVector | `"pgvector"` | ✅ Yes | Existing PostgreSQL infrastructure |

**Multi-backend** (runs all in parallel and returns all results):
```json
{ "backends": ["faiss", "chromadb", "qdrant"] }
```

Start Docker services for server backends:
```bash
docker compose -f docker-compose.polyrag.yml up -d
```

---

### Choosing Retrieval Methods

All methods contribute to **RRF (Reciprocal Rank Fusion)** — enabling more methods improves recall, not just takes the latest one.

#### Always-Available Methods (no LLM required)

| Flag | Default | Method | When to Enable |
|------|---------|--------|----------------|
| `enable_dense` | `true` | Dense Vector (semantic embedding similarity) | Always — core semantic retrieval |
| `enable_bm25` | `true` | BM25 Keyword | Always — captures exact terminology, IDs, names |
| `enable_splade` | `false` | SPLADE Sparse Neural | When you need term expansion beyond BM25; requires SPLADE index built at ingest |
| `enable_graph` | `true` | Knowledge Graph (spaCy NER + Kuzu) | When queries involve people, places, events, or named entities |
| `enable_rerank` | `true` | Cross-Encoder Re-ranking | Always recommended — re-scores top candidates for precision |
| `enable_mmr` | `true` | MMR Diversity | When results feel repetitive; penalises near-duplicate chunks |

#### LLM-Required Methods (requires LM Studio at `localhost:1234`)

| Flag | Default | Method | When to Enable |
|------|---------|--------|----------------|
| `enable_rewrite` | `false` | Query Rewrite | Noisy or ambiguous queries; typo correction |
| `enable_multi_query` | `false` | Multi-Query (generates 3 phrasings) | Complex multi-faceted questions; broadens recall |
| `enable_hyde` | `false` | HyDE (Hypothetical Document Embeddings) | Abstract or conceptual queries; retrieves by answer similarity |
| `enable_raptor` | `false` | RAPTOR (hierarchical summaries) | Document-level understanding; "big picture" questions |
| `enable_contextual_rerank` | `false` | Contextual Re-ranking (LLM) | Highest-quality re-ranking; use when answer precision is critical |
| `enable_llm_graph` | `false` | LLM Graph Entity Extraction | Complex relationship queries; more accurate than spaCy |

> **Dependency rule**: `enable_multi_query` requires `enable_rewrite` (auto-enforced server-side).
> `enable_llm_graph` requires `enable_graph` (auto-enforced server-side).

#### Recommended Method Combinations

| Use Case | Configuration |
|----------|--------------|
| **Fast baseline** | `dense=true, bm25=true, rerank=true` |
| **High-quality production** | `dense=true, bm25=true, graph=true, rerank=true, mmr=true` |
| **Agentic / maximum recall** | All base methods `true` + `rewrite=true, multi_query=true` |
| **Named-entity-heavy queries** | Add `graph=true, llm_graph=true` |
| **Exact terminology / code** | Ensure `bm25=true, splade=true` |
| **Conceptual / abstract** | Add `hyde=true` |
| **Document summary questions** | Add `raptor=true` |
| **Speed-critical (< 200ms)** | `dense=true, bm25=true, rerank=false, mmr=false` |

---

### Choosing an Embedding Model

| `embedding_model` | Dimensions | Size | MTEB Score | Best For |
|-------------------|-----------|------|------------|----------|
| `"all-MiniLM-L6-v2"` | 384 | ~80 MB | 56.3 | Speed-first, dev/test, CPU-constrained |
| `"BAAI/bge-base-en-v1.5"` | 768 | ~440 MB | 63.6 | Balanced quality/speed for production |
| `"BAAI/bge-large-en-v1.5"` | 1024 | ~1.3 GB | 64.2 | Maximum retrieval quality |

> ⚠ **Collection isolation**: each model stores into a separate collection suffix automatically:
> - `polyrag_docs_minilm`
> - `polyrag_docs_bge-base`
> - `polyrag_docs_bge-large`
>
> Always use the same `embedding_model` for ingest and search.

---

### Full Response with Traceability

```json
{
  "query": "What does Hamlet say about mortality?",
  "results": {
    "faiss": {
      "backend": "faiss",
      "answer": "Hamlet reflects on mortality in the 'To be, or not to be' soliloquy...",
      "latency_ms": 142.5,
      "error": null,

      "chunks": [
        {
          "chunk_id": "chunk-0042",
          "text": "To be, or not to be, that is the question...",
          "score": 0.8731,
          "confidence": 1.0,
          "provenance": null,
          "metadata": {
            "source": "shakespeare.txt",
            "doc_id": "hamlet_act3",
            "char_start": 18450,
            "char_end": 18950,
            "chunk_type": "sentence",
            "entities": ["Hamlet", "Denmark"]
          },
          "method_lineage": [
            {
              "method": "Dense Vector",
              "rank": 1,
              "rrf_contribution": 0.0164
            },
            {
              "method": "BM25 Keyword",
              "rank": 2,
              "rrf_contribution": 0.0154
            },
            {
              "method": "Knowledge Graph",
              "rank": 3,
              "rrf_contribution": 0.0145
            }
          ],
          "post_processors": ["CrossEncoderReRanker", "MMRReranker"]
        }
      ],

      "retrieval_trace": [
        {
          "method": "Dense Vector",
          "candidates_before": 50,
          "candidates_after": 20,
          "scores": []
        },
        {
          "method": "BM25 Keyword",
          "candidates_before": 50,
          "candidates_after": 15,
          "scores": []
        },
        {
          "method": "RRF Fusion",
          "candidates_before": 35,
          "candidates_after": 10,
          "scores": []
        },
        {
          "method": "CrossEncoderReRanker",
          "candidates_before": 10,
          "candidates_after": 5,
          "scores": []
        }
      ],

      "llm_traces": [
        {
          "method": "Query Rewrite",
          "system_prompt": "You are a query optimiser...",
          "user_message": "Rewrite: What does Hamlet say about mortality?",
          "response": "What philosophical views on death does Hamlet express?",
          "latency_ms": 312.4
        }
      ],

      "graph_entities": ["Hamlet", "Ophelia", "Claudius", "Denmark"],
      "graph_paths": ["Hamlet --[SPEAKS_TO]--> Ophelia", "Claudius --[MURDERED]--> King Hamlet"]
    }
  }
}
```

---

## 4. Traceability Fields Reference

Every chunk in `results[backend].chunks[]` carries full lineage:

### Per-Chunk Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `string` | Unique identifier for this chunk in the vector store |
| `text` | `string` | The raw chunk text returned |
| `score` | `float` | Final RRF fusion score (0.0–1.0+). Higher = more relevant. |
| `confidence` | `float` | Normalised score: `score / max_score`. 1.0 = top-ranked chunk. |
| `provenance` | `string\|null` | Source citation string when available |
| `metadata.source` | `string` | Original document filename or URL |
| `metadata.doc_id` | `string` | Document-level identifier |
| `metadata.char_start` | `int` | Character offset start in the original document |
| `metadata.char_end` | `int` | Character offset end in the original document |
| `metadata.chunk_type` | `string` | Chunking strategy used (`sentence`, `section`, etc.) |
| `metadata.entities` | `string[]` | Named entities extracted from this chunk by spaCy |

### Method Lineage (`method_lineage[]`)

Shows **which retrieval methods found this chunk** and how much they contributed to its final score:

| Field | Type | Description |
|-------|------|-------------|
| `method` | `string` | Retrieval method name (e.g. `"Dense Vector"`, `"BM25 Keyword"`, `"Knowledge Graph"`, `"SPLADE"`) |
| `rank` | `int` | Rank assigned by this method (1 = top result for that method) |
| `rrf_contribution` | `float` | Score contribution via `1/(k+rank)`. Sum of contributions = final score. |

**Interpretation**:
- A chunk with lineage from 3 methods is more trustworthy than one found by only 1 method.
- Low `rank` + high `rrf_contribution` = this method strongly agrees this chunk is relevant.
- If only `Dense Vector` appears in lineage, the result is semantically similar but may not contain exact keywords.

### Post-Processors (`post_processors[]`)

Lists which re-ranking stages this chunk survived:

| Value | Meaning |
|-------|---------|
| `CrossEncoderReRanker` | Survived cross-encoder re-scoring |
| `MMRReranker` | Selected for result diversity (not too similar to other top chunks) |
| `ContextualReranker` | Re-ranked by LLM in full query context |

### Retrieval Trace (`retrieval_trace[]`)

Shows the funnel — how many candidates each phase started and ended with:

| Field | Description |
|-------|-------------|
| `method` | Pipeline stage name |
| `candidates_before` | Chunks entering this stage |
| `candidates_after` | Chunks passing through (post-filter/re-rank) |

### Backend-Level Fields

| Field | Description |
|-------|-------------|
| `answer` | LLM-synthesised answer (empty string if LLM not available) |
| `latency_ms` | Total end-to-end retrieval + answer latency in milliseconds |
| `graph_entities` | Named entities extracted from the query or retrieved chunks |
| `graph_paths` | Relation paths from the knowledge graph (e.g. `"A --[relates_to]--> B"`) |
| `llm_traces[]` | Full prompt + response for each LLM call made during this search |
| `error` | Error message if this backend failed (other backends still return results) |

---

## 5. Retrieval Trails (Audit Log)

Every search automatically appends a record to `data/retrieval_trails.jsonl`.

### GET `/api/retrieval-trails`

```bash
# Last 50 trail records
curl "http://localhost:8000/api/retrieval-trails?limit=50"

# Filter by backend
curl "http://localhost:8000/api/retrieval-trails?backend=faiss&limit=100"
```

#### Trail Record Fields

```json
{
  "timestamp": "2026-03-20T14:30:00.123Z",
  "query": "What does Hamlet say about mortality?",
  "backend": "faiss",
  "methods_used": {
    "enable_dense": true,
    "enable_bm25": true,
    "enable_graph": true,
    "enable_rerank": true,
    "enable_mmr": true,
    "enable_rewrite": false,
    "enable_multi_query": false,
    "enable_hyde": false,
    "enable_raptor": false,
    "enable_contextual_rerank": false,
    "enable_llm_graph": false,
    "enable_splade": false
  },
  "query_variants": {
    "rewritten": "What philosophical views on death does Hamlet express?",
    "paraphrases": ["How does Hamlet discuss death?", "Hamlet mortality soliloquy meaning"],
    "hyde_text": "Hamlet contemplates death in the famous 'To be or not to be' speech..."
  },
  "retrieval_trace": [
    { "method": "Dense Vector",    "candidates_before": 50, "candidates_after": 20 },
    { "method": "BM25 Keyword",    "candidates_before": 50, "candidates_after": 15 },
    { "method": "RRF Fusion",      "candidates_before": 35, "candidates_after": 10 },
    { "method": "CrossEncoderReRanker", "candidates_before": 10, "candidates_after": 5 }
  ],
  "result_count": 5,
  "latency_ms": 142.5
}
```

### DELETE `/api/retrieval-trails`

Clear the audit log:
```bash
curl -X DELETE http://localhost:8000/api/retrieval-trails
```

### GET `/api/retrieval-trails/analysis`

Analyse all recorded trails and return per-method contribution statistics.

```bash
curl "http://localhost:8000/api/retrieval-trails/analysis?backend=faiss&min_trails=1"
```

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `backend` | `string` | — | Filter to a specific backend (optional) |
| `min_trails` | `int` | `1` | Minimum trail count threshold |

```json
{
  "total_trails_analysed": 42,
  "per_method": {
    "Dense Vector": {
      "avg_contribution_pct": 38.2,
      "total_chunks_contributed": 120,
      "appears_in_n_trails": 42
    }
  },
  "recommended": ["Dense Vector", "BM25 Keyword", "Cross-Encoder Rerank"],
  "never_contributed": ["SPLADE"],
  "interpretation": "Based on your query history, these 3 methods account for 85% of all retrieved chunks."
}
```

---

## 6. LLM Trace History

### GET `/api/traces`

Returns every LLM prompt + response captured during retrieval:

```bash
curl "http://localhost:8000/api/traces?limit=20"
curl "http://localhost:8000/api/traces?method=Query%20Rewrite"
```

#### Trace Record Fields

```json
{
  "timestamp": "2026-03-20T14:30:00.456Z",
  "method": "Query Rewrite",
  "system_prompt": "You are an expert query optimiser...",
  "user_message": "Original: What does Hamlet say about mortality?\nRewrite this query...",
  "response": "What philosophical views on death does Hamlet express in his soliloquies?",
  "latency_ms": 312.4
}
```

Available `method` filter values: `Query Rewrite`, `Multi-Query`, `HyDE`, `RAPTOR`, `Contextual Rerank`, `LLM Graph Extract`, `Answer Synthesis`.

### DELETE `/api/traces`

```bash
curl -X DELETE http://localhost:8000/api/traces
```

---

## 7. Knowledge Graph API

### GET `/api/graph/{collection}`

Returns the entity-relation graph built during ingestion:

```bash
curl http://localhost:8000/api/graph/polyrag_docs_minilm
```

#### Response

```json
{
  "collection": "polyrag_docs_minilm",
  "nodes": [
    {
      "id": "hamlet",
      "label": "Hamlet",
      "type": "ENTITY",
      "frequency": 42,
      "chunks": [
        { "chunk_id": "chunk-0042", "snippet": "Hamlet speaks to Ophelia..." }
      ],
      "relations": [
        {
          "target_id": "ophelia",
          "target_label": "Ophelia",
          "relation": "SPEAKS_TO",
          "weight": 1.0
        }
      ]
    }
  ],
  "edges": [
    {
      "source": "hamlet",
      "target": "claudius",
      "relation": "CONFRONTS",
      "weight": 1.0,
      "doc_ids": ["hamlet_act3"]
    }
  ]
}
```

### POST `/api/graph/{collection}/enhance`

Run LLM-enhanced entity extraction (background job, returns `job_id`):

```bash
curl -X POST http://localhost:8000/api/graph/polyrag_docs_minilm/enhance
```

### GET `/api/graph`

List collection names that have a persisted graph snapshot.

```bash
curl http://localhost:8000/api/graph
# → ["polyrag_docs_minilm", "product_docs_bge-base"]
```

### GET `/api/graph/{collection}/enhance-status`

Check the LLM enhancement status for a collection's graph.

```bash
curl http://localhost:8000/api/graph/polyrag_docs_minilm/enhance-status
```

```json
{
  "collection": "polyrag_docs_minilm",
  "graph_exists": true,
  "node_count": 142,
  "edge_count": 381,
  "llm_enhanced": true,
  "llm_enhanced_at": "2026-03-20T12:00:00.000Z"
}
```

### GET `/api/graph/{collection}/enhance/{job_id}/stream`

SSE stream for the LLM graph enhancement background job. Same format as the ingest stream: `data:` lines for log output, `STATUS:done` or `STATUS:error` events, and `# keepalive` comments every 10 seconds.

```python
with httpx.stream("GET", f"http://localhost:8000/api/graph/{collection}/enhance/{job_id}/stream") as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:])
```

### DELETE `/api/graph/{collection}`

> ⚠️ Destructive — deletes the graph snapshot JSON and clears the Kuzu embedded graph DB for this collection.

```bash
curl -X DELETE http://localhost:8000/api/graph/polyrag_docs_minilm
```

```json
{
  "deleted": true,
  "collection": "polyrag_docs_minilm",
  "kuzu_cleared": true,
  "pipelines_evicted": 1
}
```

### DELETE `/api/graph`

> ⚠️ Destructive — deletes **all** graph snapshots and clears the entire Kuzu embedded graph DB.

```bash
curl -X DELETE http://localhost:8000/api/graph
```

```json
{
  "deleted": ["polyrag_docs_minilm", "product_docs_bge-base"],
  "count": 2,
  "kuzu_cleared": true,
  "pipelines_evicted": 3
}
```

---

## 8. Backend Health & Discovery

### GET `/api/backends`

Returns status of all 6 backends:

```bash
curl http://localhost:8000/api/backends
```

```json
[
  {
    "name": "faiss",
    "status": "connected",
    "ping_ms": 0.3,
    "collection_count": 2,
    "requires_docker": false,
    "error": null
  },
  {
    "name": "qdrant",
    "status": "error",
    "ping_ms": null,
    "collection_count": 0,
    "requires_docker": true,
    "error": "Connection refused"
  }
]
```

**Use this before search** to discover which backends are live. Backends with `"status": "error"` will return an `error` field in search results without blocking other backends.

### GET `/api/health`

```bash
curl http://localhost:8000/api/health
# → {"status": "ok"}
```

### GET `/api/backends/{name}/health`

Probe a single named backend. Returns the same `BackendInfo` shape as `GET /api/backends`. Returns `404` for an unknown backend name.

```bash
curl http://localhost:8000/api/backends/qdrant/health
```

### GET `/api/collections/{backend}`

List all collections stored in a backend.

```bash
curl http://localhost:8000/api/collections/faiss
```

```json
[
  { "name": "polyrag_docs_minilm",   "chunk_count": 1420, "index_type": "flat" },
  { "name": "product_docs_bge-base", "chunk_count": 8730, "index_type": "flat" }
]
```

### DELETE `/api/collections/{backend}/{collection}`

> ⚠️ Destructive — deletes the specified collection from the backend, evicts its cached pipeline, and removes its SPLADE index.

```bash
curl -X DELETE http://localhost:8000/api/collections/faiss/polyrag_docs_minilm
```

### DELETE `/api/collections/{backend}`

> ⚠️ Destructive — deletes **all** collections from the specified backend (full wipe).

```bash
curl -X DELETE http://localhost:8000/api/collections/faiss
```

---

## 9. Evaluate Retrieval Quality

### POST `/api/evaluate`

Run ground-truth Q&A evaluation against one or more backends:

```json
{
  "questions": [
    {
      "question": "What themes does Shakespeare explore in Hamlet?",
      "expected_answer": "revenge, mortality, corruption, madness",
      "expected_sources": ["hamlet.txt"]
    },
    {
      "question": "Who is Ophelia?",
      "expected_answer": "Polonius's daughter and Hamlet's love interest",
      "expected_sources": []
    }
  ],
  "backends": ["faiss", "chromadb"],
  "collection_name": "polyrag_docs",
  "methods": {
    "enable_dense": true,
    "enable_bm25": true,
    "enable_rerank": true,
    "enable_mmr": true
  }
}
```

#### Response — Per-Question Scores

```json
[
  {
    "backend": "faiss",
    "question": "What themes does Shakespeare explore in Hamlet?",
    "faithfulness": 0.85,
    "relevance": 0.91,
    "source_accuracy": 1.0,
    "overall": 0.92,
    "verdict": "PASS",
    "retrieved_chunks": [...]
  }
]
```

| Score | Description |
|-------|-------------|
| `faithfulness` | Keyword overlap between retrieved answer and expected answer |
| `relevance` | Semantic similarity between question and retrieved chunks |
| `source_accuracy` | Whether expected sources appeared in results (0.0 if `expected_sources` is empty) |
| `overall` | Weighted aggregate |
| `verdict` | `PASS` (≥0.7) \| `PARTIAL` (≥0.4) \| `FAIL` (<0.4) |

### GET `/api/evaluate/ragas-status`

Check whether RAGAS LLM-judged scoring is available:

```bash
curl http://localhost:8000/api/evaluate/ragas-status
```

```json
{
  "available": true,
  "scorer": "RagasScorer",
  "llm": "lm_studio",
  "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
  "reason": null
}
```

When `available` is `false`, the `reason` field explains why (e.g. `"LLM endpoint unreachable"`).

### GET `/api/evaluate/browse-chunks`

Paginated chunk browser for building Q&A evaluation datasets.

```bash
curl "http://localhost:8000/api/evaluate/browse-chunks?backend=faiss&collection=polyrag_docs_minilm&limit=30&offset=0&search=mortality"
```

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `backend` | `string` | — | Backend to browse |
| `collection` | `string` | — | Collection name |
| `limit` | `int` | `30` | Page size |
| `offset` | `int` | `0` | Pagination offset |
| `search` | `string` | — | Optional keyword filter |

Response: `{ "chunks": [...], "total": 1420, "offset": 0, "limit": 30 }`

### POST `/api/evaluate/generate-qa`

Generate a Q&A pair from a chunk using the configured LLM. Falls back to a heuristic extractor if the LLM is unavailable.

```json
{ "chunk_text": "Digital goods are non-refundable after download...", "chunk_id": "chunk-0042" }
```

Response:

```json
{
  "question": "Are digital goods refundable after download?",
  "answer": "No, digital goods are non-refundable once downloaded.",
  "chunk_id": "chunk-0042",
  "source": "llm",
  "note": null
}
```

`source` is `"llm"` or `"heuristic"`.

### POST `/api/evaluate/import-qa`

Import Q&A pairs from a JSON array or CSV text.

**JSON body:**
```json
{
  "items": [
    {
      "question": "What is the refund policy?",
      "expected_answer": "Digital goods are non-refundable.",
      "expected_sources": ["refund-policy.pdf"]
    }
  ]
}
```

**CSV body:**
```json
{ "csv": "question,expected_answer,expected_sources\nWhat is the refund policy?,...,refund-policy.pdf" }
```

Response: `{ "imported_count": 5, "items": [...], "errors": [] }`

### GET `/api/evaluate/results`

List all evaluation run IDs.

```bash
curl http://localhost:8000/api/evaluate/results
# → { "eval_ids": ["eval-abc123", "eval-def456"] }
```

### GET `/api/evaluate/{eval_id}`

Full evaluation results including per-question, per-backend scores and RAGAS metrics.

```bash
curl http://localhost:8000/api/evaluate/eval-abc123
```

### GET `/api/evaluate/{eval_id}/export`

Export evaluation results as a file download.

| Query Param | Values |
|-------------|--------|
| `format` | `json` (default) \| `csv` |

```bash
curl "http://localhost:8000/api/evaluate/eval-abc123/export?format=csv" -o results.csv
```

---

## 10. Compare Backends

### POST `/api/compare`

Benchmark multiple backends against the same queries:

```json
{
  "collection_name": "polyrag_docs",
  "backends": ["faiss", "chromadb", "qdrant"],
  "queries": [
    "What does Hamlet say about mortality?",
    "Who is Ophelia in the play?"
  ],
  "full_retrieval": false,
  "repeat_runs": 3,
  "compare_graph_ab": false
}
```

| Field | Description |
|-------|-------------|
| `full_retrieval` | `true` = include LLM methods (requires LM Studio) |
| `repeat_runs` | 1–10; enables P50/P95 latency stats |
| `compare_graph_ab` | Run each query with graph ON and OFF to measure knowledge graph impact |

---

## 11. Agentic Integration Patterns

### Pattern 1 — Single-Shot RAG Answer

```python
import httpx

def ask_polyrag(question: str, backend: str = "faiss") -> dict:
    resp = httpx.post("http://localhost:8000/api/search", json={
        "query": question,
        "backends": [backend],
        "collection_name": "polyrag_docs",
        "top_k": 5,
        "methods": {
            "enable_dense": True,
            "enable_bm25": True,
            "enable_graph": True,
            "enable_rerank": True,
            "enable_mmr": True,
        }
    }, timeout=30)
    result = resp.json()["results"][backend]
    return {
        "answer": result["answer"],
        "confidence": result["chunks"][0]["confidence"] if result["chunks"] else 0.0,
        "sources": [c["metadata"].get("source") for c in result["chunks"]],
        "latency_ms": result["latency_ms"],
    }
```

### Pattern 2 — Multi-Backend Consensus (for high-stakes answers)

```python
def ask_with_consensus(question: str) -> dict:
    resp = httpx.post("http://localhost:8000/api/search", json={
        "query": question,
        "backends": ["faiss", "chromadb"],
        "collection_name": "polyrag_docs",
        "top_k": 5,
    }, timeout=30).json()

    # Find chunks that appear in both backends (high confidence)
    chunk_ids_per_backend = {
        backend: {c["chunk_id"] for c in data["chunks"]}
        for backend, data in resp["results"].items()
    }
    all_ids = list(chunk_ids_per_backend.values())
    consensus_ids = all_ids[0].intersection(*all_ids[1:])

    return {
        "query": question,
        "consensus_chunk_count": len(consensus_ids),
        "answers": {b: d["answer"] for b, d in resp["results"].items()},
    }
```

### Pattern 3 — Agentic Tool with Full Traceability

```python
def polyrag_tool(question: str, require_sources: bool = True) -> dict:
    """
    Tool for agentic workflows. Returns answer + full provenance.
    Raises if confidence is too low.
    """
    resp = httpx.post("http://localhost:8000/api/search", json={
        "query": question,
        "backends": ["faiss"],
        "collection_name": "polyrag_docs",
        "top_k": 8,
        "methods": {
            "enable_dense": True,
            "enable_bm25": True,
            "enable_graph": True,
            "enable_rerank": True,
            "enable_mmr": True,
            "enable_rewrite": True,   # clean up noisy agent queries
        }
    }, timeout=60).json()

    result = resp["results"]["faiss"]
    if result.get("error"):
        raise RuntimeError(f"Retrieval failed: {result['error']}")

    chunks = result["chunks"]
    if not chunks:
        return {"answer": "No relevant information found.", "confidence": 0.0, "sources": []}

    top_chunk = chunks[0]
    return {
        "answer": result["answer"],
        "confidence": top_chunk["confidence"],
        "sources": list({c["metadata"].get("source", "unknown") for c in chunks}),
        "chunk_lineage": [
            {
                "chunk_id": c["chunk_id"],
                "text_preview": c["text"][:200],
                "score": c["score"],
                "confidence": c["confidence"],
                "found_by": [m["method"] for m in c["method_lineage"]],
                "source": c["metadata"].get("source"),
                "char_range": f"{c['metadata'].get('char_start')}–{c['metadata'].get('char_end')}",
            }
            for c in chunks
        ],
        "retrieval_pipeline": [
            f"{t['method']}: {t['candidates_before']} → {t['candidates_after']} candidates"
            for t in result["retrieval_trace"]
        ],
        "latency_ms": result["latency_ms"],
        "graph_entities": result.get("graph_entities", []),
    }
```

### Pattern 4 — Ingest + Query Lifecycle

```python
import httpx, time

BASE = "http://localhost:8000/api"

# Step 1: Ingest
ingest_resp = httpx.post(f"{BASE}/ingest", json={
    "text": open("my_document.txt").read(),
    "backends": ["faiss"],
    "collection_name": "agent_docs",
    "chunk_strategy": "sentence",
    "chunk_size": 400,
    "overlap": 50,
    "embedding_model": "BAAI/bge-base-en-v1.5",
}).json()

job_id = ingest_resp["job_ids"]["faiss"]

# Step 2: Wait for completion
while True:
    status = httpx.get(f"{BASE}/ingest/{job_id}/status").json()
    if status["status"] in ("done", "error"):
        break
    time.sleep(2)

assert status["status"] == "done", status["error"]

# Step 3: Query with the same embedding model
result = httpx.post(f"{BASE}/search", json={
    "query": "What is the main conclusion?",
    "backends": ["faiss"],
    "collection_name": "agent_docs",
    "embedding_model": "BAAI/bge-base-en-v1.5",  # ← must match ingest
    "top_k": 5,
}).json()["results"]["faiss"]

print(result["answer"])
```

### Pattern 5 — Audit & Traceability Pull

```python
# After running queries, pull the full audit trail
trails = httpx.get(f"{BASE}/retrieval-trails?limit=10&backend=faiss").json()

for trail in trails:
    print(f"Query: {trail['query']}")
    print(f"Methods: {[k for k,v in trail['methods_used'].items() if v]}")
    print(f"Pipeline: {[(t['method'], t['candidates_before'], t['candidates_after']) for t in trail['retrieval_trace']]}")
    print(f"Results: {trail['result_count']} chunks in {trail['latency_ms']}ms")
    print()
```

---

## 12. Method Combination Recipes

### Recipe A — Fast Production (< 200 ms)
```json
{
  "methods": {
    "enable_dense": true, "enable_bm25": true,
    "enable_rerank": true, "enable_mmr": false,
    "enable_graph": false, "enable_splade": false
  }
}
```

### Recipe B — High-Quality Production (200–500 ms)
```json
{
  "methods": {
    "enable_dense": true, "enable_bm25": true, "enable_graph": true,
    "enable_rerank": true, "enable_mmr": true, "enable_splade": false
  }
}
```

### Recipe C — Maximum Recall (500 ms–2 s, with LM Studio)
```json
{
  "methods": {
    "enable_dense": true, "enable_bm25": true, "enable_splade": true,
    "enable_graph": true, "enable_rerank": true, "enable_mmr": true,
    "enable_rewrite": true, "enable_multi_query": true,
    "enable_hyde": false, "enable_raptor": false,
    "enable_contextual_rerank": false, "enable_llm_graph": false
  }
}
```

### Recipe D — Entity-Heavy / Knowledge Graph Focus
```json
{
  "methods": {
    "enable_dense": true, "enable_bm25": true, "enable_graph": true,
    "enable_rerank": true, "enable_mmr": true,
    "enable_llm_graph": true, "enable_splade": false
  }
}
```

### Recipe E — All Methods (maximum quality, requires LM Studio)
```json
{
  "methods": {
    "enable_dense": true, "enable_bm25": true, "enable_splade": true,
    "enable_graph": true, "enable_rerank": true, "enable_mmr": true,
    "enable_rewrite": true, "enable_multi_query": true,
    "enable_hyde": true, "enable_raptor": true,
    "enable_contextual_rerank": true, "enable_llm_graph": true
  }
}
```

---

## 13. Chunks (Dry-run Preview)

### POST `/api/chunks/preview`

Dry-run chunking without ingesting. Useful for tuning `chunk_size`, `overlap`, and `strategy` before committing to a full ingest.

#### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | — | Raw text to chunk. Use this **or** `corpus_path`. |
| `corpus_path` | `string` | — | Server-side file path (admin use). Same format support as `/api/ingest`. |
| `strategy` | `string` | `"section"` | `section` \| `sliding` \| `sentence` \| `paragraph` |
| `chunk_size` | `int` | `400` | Target characters per chunk. |
| `overlap` | `int` | `50` | Overlap characters between chunks. |

```bash
curl -X POST http://localhost:8000/api/chunks/preview \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text...", "strategy": "sentence", "chunk_size": 400, "overlap": 50}'
```

#### Response

```json
{
  "chunks": [
    {
      "index": 0,
      "text": "Your document text...",
      "tokens": 42,
      "char_start": 0,
      "char_end": 210,
      "parent_id": null,
      "chunk_type": "sentence",
      "entities": ["Hamlet", "Denmark"]
    }
  ],
  "total_chunks": 12,
  "total_chars": 4820,
  "avg_chunk_size": 401.7,
  "strategy": "sentence"
}
```

---

## 14. LLM Configuration API

### GET `/api/config/llm/providers`

List all supported LLM providers.

```bash
curl http://localhost:8000/api/config/llm/providers
```

```json
[
  { "id": "lm_studio",    "label": "LM Studio",    "protocol": "openai",    "default_base_url": "http://localhost:1234/v1",        "requires_api_key": false, "notes": "Local OpenAI-compatible server" },
  { "id": "openai",       "label": "OpenAI",        "protocol": "openai",    "default_base_url": "https://api.openai.com/v1",       "requires_api_key": true,  "notes": "" },
  { "id": "ollama",       "label": "Ollama",        "protocol": "openai",    "default_base_url": "http://localhost:11434/v1",       "requires_api_key": false, "notes": "Local Ollama server" },
  { "id": "groq",         "label": "Groq",          "protocol": "openai",    "default_base_url": "https://api.groq.com/openai/v1",  "requires_api_key": true,  "notes": "" },
  { "id": "azure_openai", "label": "Azure OpenAI",  "protocol": "azure",     "default_base_url": "",                               "requires_api_key": true,  "notes": "Requires deployment name in model field" },
  { "id": "gemini",       "label": "Google Gemini", "protocol": "google",    "default_base_url": "",                               "requires_api_key": true,  "notes": "" },
  { "id": "anthropic",    "label": "Anthropic",     "protocol": "anthropic", "default_base_url": "https://api.anthropic.com",      "requires_api_key": true,  "notes": "" }
]
```

### GET `/api/config/llm`

Returns the current LLM configuration. The `api_key` is **never** returned — only whether it is set.

```bash
curl http://localhost:8000/api/config/llm
```

```json
{
  "provider": "lm_studio",
  "base_url": "http://localhost:1234/v1",
  "api_key_set": false,
  "model": "local-model",
  "temperature": 0.2,
  "max_tokens": 512,
  "timeout": 60
}
```

### PUT `/api/config/llm`

Update the LLM configuration. Immediately flushes all cached pipelines so the new config takes effect on the next request. Returns `422` for an unknown `provider` value.

```json
{
  "provider": "openai",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "max_tokens": 512,
  "timeout": 60
}
```

### GET `/api/config/llm/test`

Probe the currently configured LLM endpoint with a lightweight request.

```bash
curl http://localhost:8000/api/config/llm/test
```

```json
{
  "reachable": true,
  "provider": "lm_studio",
  "base_url": "http://localhost:1234/v1",
  "model": "local-model",
  "error": null
}
```

When `reachable` is `false`, the `error` field contains the failure reason.

---

## 15. Prompt Management

### GET `/api/prompts`

List all prompt templates registered in the system with their metadata.

```bash
curl http://localhost:8000/api/prompts
```

```json
[
  {
    "key": "query_rewrite",
    "method_name": "Query Rewrite",
    "method_id": "rewrite",
    "pipeline_stage": "query",
    "description": "Rewrites noisy user queries into clean retrieval queries.",
    "template": "You are an expert query optimiser. Rewrite the following query..."
  }
]
```

### PUT `/api/prompts/{key}`

Update a prompt template. Changes take effect on the **next** search request (no restart required).

```bash
curl -X PUT http://localhost:8000/api/prompts/query_rewrite \
  -H "Content-Type: application/json" \
  -d '{"template": "Rewrite this query for better retrieval: {query}"}'
```

### POST `/api/prompts/{key}/reset`

Reset a prompt to its factory default.

```bash
curl -X POST http://localhost:8000/api/prompts/query_rewrite/reset
```

```json
{ "status": "reset", "key": "query_rewrite", "template": "You are an expert query optimiser..." }
```

---

## 16. Purge

### DELETE `/api/purge/{collection}`

> ⚠️ Destructive — deep purge a collection from **all** storage layers simultaneously:
> 1. All 6 vector backends
> 2. Knowledge graph JSON snapshot
> 3. Kuzu embedded graph DB
> 4. SPLADE index directory (`data/splade/`)
> 5. BM25 pkl files (`data/bm25/`)
> 6. Pipeline LRU cache

```bash
curl -X DELETE http://localhost:8000/api/purge/polyrag_docs_minilm
```

```json
{
  "collection": "polyrag_docs_minilm",
  "summary": {
    "backends_deleted": 2,
    "backends_skipped": 4,
    "backends_error": 0,
    "graph_snapshot_deleted": true,
    "kuzu_cleared": true,
    "splade_files_removed": 3,
    "bm25_pkls_removed": 2,
    "pipelines_evicted": 1
  },
  "details": { }
}
```

### DELETE `/api/purge`

> ⚠️ Destructive — purge **all** collections from **all** storage layers. This is a full system wipe.

```bash
curl -X DELETE http://localhost:8000/api/purge
```

```json
{
  "collections_purged": ["polyrag_docs_minilm", "product_docs_bge-base"],
  "count": 2,
  "kuzu_cleared": true,
  "bm25_pkls_removed": 5,
  "pipelines_evicted": 3,
  "per_collection": { }
}
```

---

## 17. System / Operations

> Both `GET /api/health` (basic) and `GET /api/system/health` (detailed) are available.

### GET `/api/system/health`

Detailed system health including uptime, version, rate limiting status, pipeline cache stats, job queue summary, and BM25 snapshot count.

```bash
curl http://localhost:8000/api/system/health
```

```json
{
  "status": "ok",
  "version": "14.2.0",
  "uptime_seconds": 3612.4,
  "rate_limiting": true,
  "pipeline_cache": {
    "cached": 2,
    "max": 10,
    "utilisation_pct": 20.0
  },
  "jobs": {
    "pending": 0,
    "running": 1,
    "done": 42,
    "error": 1
  },
  "bm25_snapshots": 3
}
```

### GET `/api/system/cache`

List pipeline LRU cache entries in eviction order (MRU first).

```bash
curl http://localhost:8000/api/system/cache
```

```json
{
  "max_pipelines": 10,
  "cached": 2,
  "lru_entries": [
    { "rank": 1, "backend": "faiss",  "collection": "polyrag_docs_minilm",   "embedding_model": "all-MiniLM-L6-v2" },
    { "rank": 2, "backend": "qdrant", "collection": "product_docs_bge-base", "embedding_model": "BAAI/bge-base-en-v1.5" }
  ],
  "note": "rank 1 = most recently used (last to be evicted)"
}
```

### DELETE `/api/system/cache`

> ⚠️ Use with caution — flushes all cached pipelines. The next query to each pipeline will trigger a cold start (~10–30 s per pipeline).

```bash
curl -X DELETE http://localhost:8000/api/system/cache
```

```json
{ "flushed": 2, "pipelines_stopped": 2, "message": "All pipeline caches cleared." }
```

---

## 18. Jobs

### GET `/api/jobs`

List all jobs (ingest and other background jobs) sorted newest-first.

```bash
curl http://localhost:8000/api/jobs
```

Returns `List[JobStatus]`. Each entry has the same fields as `GET /api/ingest/{job_id}/status`: `job_id`, `status`, `backend`, `created_at`, `updated_at`, `log_lines[]`, `result`, `error`.

### GET `/api/jobs/{job_id}`

Full job detail including all log lines. Returns `404` if the job ID is not found.

```bash
curl http://localhost:8000/api/jobs/job-abc123
```

---

## 19. Feedback

### POST `/api/feedback`

Submit relevance feedback on a retrieved chunk.

```json
{
  "query": "What does Hamlet say about mortality?",
  "chunk_id": "chunk-0042",
  "backend": "faiss",
  "collection_name": "polyrag_docs_minilm",
  "relevant": true
}
```

### GET `/api/feedback`

Retrieve all stored feedback entries.

```bash
curl http://localhost:8000/api/feedback
```

```json
{
  "count": 14,
  "entries": [
    {
      "query": "What does Hamlet say about mortality?",
      "chunk_id": "chunk-0042",
      "backend": "faiss",
      "collection_name": "polyrag_docs_minilm",
      "relevant": true,
      "timestamp": "2026-03-20T15:11:00.000Z"
    }
  ]
}
```

---

## 20. API Reference Summary

| Method | Path | Purpose |
|--------|------|---------|
| **Health & System** | | |
| `GET` | `/api/health` | Basic health check — `{"status": "ok"}` |
| `GET` | `/api/system/health` | Detailed health: uptime, version, cache stats, job counts, BM25 snapshots |
| `GET` | `/api/system/cache` | List pipeline LRU cache entries |
| `DELETE` | `/api/system/cache` | ⚠️ Flush all cached pipelines (cold start on next query) |
| **Unified Agentic RAG** | | |
| **`POST`** | **`/api/rag`** | **⭐ Unified agentic RAG — profile-based production endpoint** |
| `POST` | `/api/rag/profiles` | Create a saved RAG configuration profile |
| `GET` | `/api/rag/profiles` | List all saved profiles |
| `GET` | `/api/rag/profiles/{id}` | Get a profile by ID |
| `PUT` | `/api/rag/profiles/{id}` | Update a profile |
| `DELETE` | `/api/rag/profiles/{id}` | Delete a profile |
| **Search** | | |
| `POST` | `/api/search` | **Core RAG query** — multi-backend, returns answer + full traceability |
| **Ingest** | | |
| `POST` | `/api/ingest` | Ingest documents (async, per backend). Supports `.txt`, `.pdf`, `.pptx`. |
| `GET` | `/api/ingest/jobs` | List all ingest jobs, sorted newest-first |
| `GET` | `/api/ingest/{job_id}/status` | Poll ingest job status |
| `GET` | `/api/ingest/{job_id}/stream` | SSE real-time ingest log stream |
| **Chunks** | | |
| `POST` | `/api/chunks/preview` | Dry-run chunking without ingesting |
| **Retrieval Trails** | | |
| `GET` | `/api/retrieval-trails` | Audit log of all past searches |
| `DELETE` | `/api/retrieval-trails` | Clear audit log |
| `GET` | `/api/retrieval-trails/analysis` | Per-method contribution statistics across recorded trails |
| **LLM Traces** | | |
| `GET` | `/api/traces` | LLM call history (prompts + responses) |
| `DELETE` | `/api/traces` | Clear LLM trace log |
| **Knowledge Graph** | | |
| `GET` | `/api/graph` | List collections with a persisted graph snapshot |
| `GET` | `/api/graph/{collection}` | Entity-relation knowledge graph for a collection |
| `POST` | `/api/graph/{collection}/enhance` | LLM-enhanced graph extraction (background job) |
| `GET` | `/api/graph/{collection}/enhance-status` | LLM enhancement status |
| `GET` | `/api/graph/{collection}/enhance/{job_id}/stream` | SSE stream for the enhance background job |
| `DELETE` | `/api/graph/{collection}` | ⚠️ Delete graph snapshot + Kuzu DB for a collection |
| `DELETE` | `/api/graph` | ⚠️ Delete all graph snapshots + entire Kuzu DB |
| **Backends** | | |
| `GET` | `/api/backends` | List all backends with health status |
| `GET` | `/api/backends/{name}/health` | Probe a single named backend |
| `GET` | `/api/collections/{backend}` | List collections in a backend |
| `DELETE` | `/api/collections/{backend}/{collection}` | ⚠️ Delete a collection (evicts cache + SPLADE) |
| `DELETE` | `/api/collections/{backend}` | ⚠️ Delete all collections in a backend (full wipe) |
| **Evaluate** | | |
| `POST` | `/api/evaluate` | Ground-truth Q&A quality evaluation |
| `GET` | `/api/evaluate/ragas-status` | Check RAGAS scoring availability |
| `GET` | `/api/evaluate/browse-chunks` | Paginated chunk browser for building Q&A datasets |
| `POST` | `/api/evaluate/generate-qa` | Generate a Q&A pair from a chunk (LLM or heuristic fallback) |
| `POST` | `/api/evaluate/import-qa` | Import Q&A pairs from JSON array or CSV |
| `GET` | `/api/evaluate/results` | List all evaluation run IDs |
| `GET` | `/api/evaluate/{eval_id}` | Full evaluation results including RAGAS metrics |
| `GET` | `/api/evaluate/{eval_id}/export` | Export results as JSON or CSV file download |
| **Compare** | | |
| `POST` | `/api/compare` | Benchmark backends head-to-head |
| **LLM Configuration** | | |
| `GET` | `/api/config/llm/providers` | List all supported LLM providers |
| `GET` | `/api/config/llm` | Current LLM config (api_key is never returned) |
| `PUT` | `/api/config/llm` | Update LLM config (flushes all cached pipelines) |
| `GET` | `/api/config/llm/test` | Probe the configured LLM endpoint |
| **Prompts** | | |
| `GET` | `/api/prompts` | List all prompt templates with metadata |
| `PUT` | `/api/prompts/{key}` | Update a prompt template |
| `POST` | `/api/prompts/{key}/reset` | Reset a prompt to factory default |
| **Purge** | | |
| `DELETE` | `/api/purge/{collection}` | ⚠️ Deep purge a collection from all storage layers |
| `DELETE` | `/api/purge` | ⚠️ Purge ALL collections from ALL storage layers |
| **Jobs** | | |
| `GET` | `/api/jobs` | List all jobs, sorted newest-first |
| `GET` | `/api/jobs/{job_id}` | Full job detail including log lines (404 if not found) |
| **Feedback** | | |
| `POST` | `/api/feedback` | Submit relevance feedback on a chunk |
| `GET` | `/api/feedback` | List all feedback entries |

---

> **Interactive Exploration**: Open http://localhost:8000/docs for the full Swagger UI where you can try every endpoint live with request/response examples.
