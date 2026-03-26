"""
Unified Agentic RAG API — /api/rag

Single production-grade endpoint for integrating PolyRAG into agentic AI flows.
Replaces the complexity of /api/search with a profile-based, fully-traceable
single-call interface.

Profile Workflow:
  1. Experiment with /api/search + /api/compare to find optimal settings
  2. POST /api/rag/profiles to save a named profile with your tested config
  3. Call POST /api/rag with { "query": "...", "profile_id": "..." }
  4. Receive a complete, traceable answer with confidence verdict + lineage
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import build_pipeline_config
from api.routers.search import _expand_query, _run_search_with_bundle
from api.schemas import (
    EmbeddingModel,
    LLMTraceEntry,
    RetrievalMethods,
    RetrievalTraceEntry,
    SearchResultItem,
)

router = APIRouter(tags=["Agentic RAG"])

PROFILES_DIR = Path("data/profiles")
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


# ── Schema: Confidence Thresholds ─────────────────────────────────────────────


class ConfidenceThresholds(BaseModel):
    """Per-profile confidence verdict thresholds — configurable by developer."""

    high: float = Field(0.8, ge=0.0, le=1.0, description="Score ≥ this → HIGH verdict")
    medium: float = Field(0.5, ge=0.0, le=1.0, description="Score ≥ this → MEDIUM verdict")
    low: float = Field(0.3, ge=0.0, le=1.0, description="Score ≥ this → LOW verdict")


# ── Schema: Scale Hints ───────────────────────────────────────────────────────


class ScaleHints(BaseModel):
    """
    Per-profile operational tuning for ingestion and retrieval volume.

    These settings are passed through to the ingestion pipeline and streaming chunker.
    They do NOT affect retrieval quality — only throughput, memory usage, and durability.

    Usage:
      - Increase ``embed_batch_size`` on machines with more RAM for faster ingestion.
      - Decrease ``max_doc_size_mb`` to enforce strict document size limits for your use case.
      - Set ``bm25_persist=true`` to snapshot the BM25 index after warm-start (faster restarts).
    """

    embed_batch_size: int = Field(
        32, ge=1, le=512,
        description="Number of chunks embedded per batch during ingestion (higher = faster, more RAM)",
    )
    max_doc_size_mb: float = Field(
        200.0, gt=0,
        description="Maximum allowed document size in MB. Larger files raise a 400 error.",
    )
    bm25_persist: bool = Field(
        True,
        description="If true, snapshot the BM25 index to disk after warm-start for faster restarts.",
    )
    max_concurrent_requests: int = Field(
        5, ge=1, le=100,
        description="Soft concurrency hint — used for documentation; enforcement via rate limiting.",
    )


# ── Schema: RAG Profile ───────────────────────────────────────────────────────


class RagProfile(BaseModel):
    """
    A saved, developer-tested RAG configuration.

    After experimenting with /api/search and /api/compare, persist your chosen
    backend + embedding model + retrieval method combination as a named profile.
    The returned `id` is then passed as `profile_id` in /api/rag calls.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Auto-generated profile ID (UUID). Use this as profile_id in /api/rag.",
    )
    name: str = Field(..., description="Human-readable name, e.g. 'production-v1'")
    description: str = Field("", description="Notes about what this config was tested for")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Core configuration ─────────────────────────────────────────────────────
    backend: str = Field("chromadb", description="Vector store: chromadb|faiss|qdrant|weaviate|milvus|pgvector")
    collection_name: str = Field("polyrag_docs", description="Target collection to query")
    embedding_model: EmbeddingModel = Field(
        "all-MiniLM-L6-v2", description="Embedding model used at ingestion time"
    )
    top_k: int = Field(5, ge=1, le=50, description="Maximum chunks to retrieve")
    methods: RetrievalMethods = Field(
        default_factory=RetrievalMethods,
        description="Which retrieval methods to activate",
    )

    # ── Confidence thresholds (configurable per profile) ──────────────────────
    confidence_thresholds: ConfidenceThresholds = Field(
        default_factory=ConfidenceThresholds,
        description="Score boundaries for HIGH / MEDIUM / LOW / INSUFFICIENT verdict",
    )

    # ── Scale hints (operational tuning, does not affect retrieval quality) ───
    scale_hints: ScaleHints = Field(
        default_factory=ScaleHints,
        description="Ingestion throughput and durability tuning for this profile",
    )


# ── Schema: RAG Request ───────────────────────────────────────────────────────


class RagRequest(BaseModel):
    """
    Unified RAG query request.

    Three usage patterns:

    **A — Profile-based (recommended for production):**
    ```json
    { "query": "...", "profile_id": "abc-123" }
    ```

    **B — Inline config (exploration / first integration):**
    ```json
    { "query": "...", "backend": "qdrant", "collection_name": "...", "methods": {...} }
    ```

    **C — Profile + runtime overrides (A/B at query time):**
    ```json
    { "query": "...", "profile_id": "abc-123", "top_k": 10 }
    ```
    Any field set in the request overrides the corresponding profile value.
    """

    query: str = Field(..., description="Natural language question to answer")
    profile_id: Optional[str] = Field(
        None,
        description="ID of a saved profile. When provided, all profile config is used as base.",
    )

    # Optional per-request overrides (override profile or supply defaults)
    backend: Optional[str] = Field(None, description="Override backend from profile")
    collection_name: Optional[str] = Field(None, description="Override collection from profile")
    embedding_model: Optional[EmbeddingModel] = Field(
        None, description="Override embedding model from profile"
    )
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override top_k from profile")
    methods: Optional[RetrievalMethods] = Field(
        None, description="Override retrieval methods from profile"
    )
    confidence_thresholds: Optional[ConfidenceThresholds] = Field(
        None, description="Override confidence thresholds from profile"
    )


# ── Schema: Pipeline Audit ────────────────────────────────────────────────────


class PipelineAudit(BaseModel):
    """Full retrieval funnel audit — every stage, candidate count, and latency."""

    backend: str
    collection: str
    embedding_model: str
    methods_active: List[str] = Field(
        description="Human-readable list of retrieval methods that ran"
    )
    funnel: List[RetrievalTraceEntry] = Field(
        description="Per-stage candidate counts (before → after filtering)"
    )
    latency_ms: float


# ── Schema: Graph Summary ─────────────────────────────────────────────────────


class GraphSummary(BaseModel):
    """Knowledge graph context surfaced during retrieval."""

    entities: List[str]
    paths: List[str]


# ── Schema: RagAnswer (the unified response envelope) ─────────────────────────


class RagAnswer(BaseModel):
    """
    Complete, traceable RAG answer — designed for agentic AI consumption.

    All fields are always populated (no optional surprises). Agents can ignore
    fields they don't need (e.g., `llm_traces`, `pipeline_audit`) but the data
    is always there for debugging, evaluation, and compliance.
    """

    query: str
    profile_id: Optional[str] = Field(None, description="Profile used (if any)")
    profile_name: Optional[str] = Field(None, description="Profile name (if any)")

    # ── The Answer ─────────────────────────────────────────────────────────────
    answer: str = Field(..., description="LLM-generated or extractive answer")
    answer_confidence: float = Field(
        ..., description="Aggregate confidence score [0.0–1.0] from top retrieved chunks"
    )
    verdict: str = Field(
        ..., description="HIGH | MEDIUM | LOW | INSUFFICIENT — based on profile thresholds"
    )

    # ── Supporting Evidence with Full Lineage ──────────────────────────────────
    sources: List[SearchResultItem] = Field(
        default_factory=list,
        description=(
            "Top-k retrieved chunks. Each chunk carries: score, confidence, "
            "metadata (document name, char range, section), method_lineage (which "
            "retrieval methods found it + RRF contributions), and post_processors "
            "(which rerankers it passed through)."
        ),
    )

    # ── Pipeline Audit Trail ───────────────────────────────────────────────────
    pipeline_audit: PipelineAudit = Field(
        description="Complete retrieval funnel: backend, methods run, stage-by-stage counts, latency"
    )

    # ── Knowledge Graph Context ────────────────────────────────────────────────
    graph: Optional[GraphSummary] = Field(
        None,
        description="Entities and relation paths from the knowledge graph (null if graph disabled)",
    )

    # ── LLM Traces (full observability) ───────────────────────────────────────
    llm_traces: List[LLMTraceEntry] = Field(
        default_factory=list,
        description="One entry per LLM call: method, system prompt, user message, response, latency_ms",
    )

    timestamp: str = Field(description="ISO 8601 UTC timestamp of this response")


# ── Internal Helpers ──────────────────────────────────────────────────────────

_METHOD_LABELS: Dict[str, str] = {
    "enable_dense": "Dense Vector",
    "enable_bm25": "BM25 Keyword",
    "enable_splade": "SPLADE",
    "enable_graph": "Knowledge Graph",
    "enable_rerank": "Cross-Encoder Rerank",
    "enable_mmr": "MMR Diversity",
    "enable_rewrite": "Query Rewrite",
    "enable_multi_query": "Multi-Query",
    "enable_hyde": "HyDE",
    "enable_raptor": "RAPTOR",
    "enable_contextual_rerank": "Contextual Rerank",
    "enable_llm_graph": "LLM Entity Extraction",
}


def _profile_path(profile_id: str) -> Path:
    return PROFILES_DIR / f"{profile_id}.json"


def _load_profile(profile_id: str) -> RagProfile:
    p = _profile_path(profile_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return RagProfile.model_validate_json(p.read_text())


def _save_profile(profile: RagProfile) -> None:
    _profile_path(profile.id).write_text(profile.model_dump_json(indent=2))


def _compute_verdict(confidence: float, thresholds: ConfidenceThresholds) -> str:
    if confidence >= thresholds.high:
        return "HIGH"
    if confidence >= thresholds.medium:
        return "MEDIUM"
    if confidence >= thresholds.low:
        return "LOW"
    return "INSUFFICIENT"


def _active_method_labels(methods: RetrievalMethods) -> List[str]:
    return [label for key, label in _METHOD_LABELS.items() if getattr(methods, key, False)]


def _avg_confidence(chunks: List[SearchResultItem]) -> float:
    """Aggregate confidence from top-3 chunks. Uses chunk.confidence if set, else chunk.score."""
    if not chunks:
        return 0.0
    top = chunks[:3]
    scores = [
        c.confidence if c.confidence is not None else c.score
        for c in top
    ]
    return round(sum(scores) / len(scores), 4)


# ── Profile CRUD ──────────────────────────────────────────────────────────────


@router.post(
    "/rag/profiles",
    response_model=RagProfile,
    status_code=201,
    summary="Create a RAG profile",
)
def create_profile(profile: RagProfile):
    """
    Save a tested configuration as a named profile.

    After experimenting with `/api/search` and `/api/compare`, lock in your chosen
    backend + embedding model + retrieval methods here. The returned `id` is used
    as `profile_id` in `/api/rag` calls.

    The `id` field is auto-generated if omitted.
    """
    if _profile_path(profile.id).exists():
        raise HTTPException(
            status_code=409,
            detail=f"Profile '{profile.id}' already exists. Use PUT /api/rag/profiles/{profile.id} to update.",
        )
    _save_profile(profile)
    return profile


@router.get(
    "/rag/profiles",
    response_model=List[RagProfile],
    summary="List all RAG profiles",
)
def list_profiles():
    """Return all saved profiles, newest first."""
    profiles: List[RagProfile] = []
    for f in sorted(PROFILES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            profiles.append(RagProfile.model_validate_json(f.read_text()))
        except Exception:
            pass
    return profiles


@router.get(
    "/rag/profiles/{profile_id}",
    response_model=RagProfile,
    summary="Get a RAG profile by ID",
)
def get_profile(profile_id: str):
    return _load_profile(profile_id)


@router.put(
    "/rag/profiles/{profile_id}",
    response_model=RagProfile,
    summary="Update a RAG profile",
)
def update_profile(profile_id: str, updates: RagProfile):
    """
    Update an existing profile. The `id` and `created_at` are always preserved
    from the existing record — you cannot change a profile's identity.
    """
    existing = _load_profile(profile_id)
    update_data = updates.model_dump(exclude_unset=True)
    update_data.pop("id", None)
    update_data.pop("created_at", None)
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    merged = existing.model_copy(update=update_data)
    _save_profile(merged)
    return merged


@router.delete(
    "/rag/profiles/{profile_id}",
    status_code=204,
    summary="Delete a RAG profile",
)
def delete_profile(profile_id: str):
    p = _profile_path(profile_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    p.unlink()


# ── Unified Query Endpoint ────────────────────────────────────────────────────


@router.post(
    "/rag",
    response_model=RagAnswer,
    summary="Unified RAG query — production agentic endpoint",
)
async def rag_query(req: RagRequest):
    """
    **Production-grade single-call RAG endpoint for agentic AI integration.**

    ### Designed For
    Developers who have finished experimenting with PolyRAG's `/api/search`,
    `/api/compare`, and `/api/evaluate` — and are now ready to integrate a
    stable, traceable RAG call into an agentic AI flow.

    ### Three Usage Patterns

    **A — Profile-based (recommended for production agents):**
    ```json
    { "query": "What is our refund policy?", "profile_id": "prod-v1-abc123" }
    ```
    No configuration noise. The profile carries the developer's tested choices.

    **B — Inline config (first integration / exploration):**
    ```json
    {
      "query": "What is our refund policy?",
      "backend": "qdrant",
      "collection_name": "product_docs",
      "embedding_model": "BAAI/bge-base-en-v1.5",
      "top_k": 8,
      "methods": { "enable_dense": true, "enable_bm25": true, "enable_rerank": true }
    }
    ```

    **C — Profile + runtime overrides (dynamic A/B at query time):**
    ```json
    { "query": "...", "profile_id": "prod-v1-abc123", "top_k": 15 }
    ```
    Any field set in the request body overrides the corresponding profile value.

    ### What's Always Returned
    - `answer` — generated answer text
    - `verdict` — HIGH / MEDIUM / LOW / INSUFFICIENT (configurable thresholds per profile)
    - `sources[]` — each chunk with: score, document name, char range, `method_lineage`
      (which retrieval methods found it + their RRF contributions), `post_processors`
      (which rerankers it survived)
    - `pipeline_audit` — full funnel: which methods ran, candidate counts per stage, latency
    - `graph` — knowledge graph entities and relation paths (if graph methods enabled)
    - `llm_traces` — every LLM call made: prompt, response, latency (for observability)
    """
    import asyncio

    # ── 1. Resolve effective configuration ────────────────────────────────────
    profile: Optional[RagProfile] = None
    if req.profile_id:
        profile = _load_profile(req.profile_id)

    backend = req.backend or (profile.backend if profile else "chromadb")
    collection_name = req.collection_name or (profile.collection_name if profile else "polyrag_docs")
    embedding_model = req.embedding_model or (profile.embedding_model if profile else "all-MiniLM-L6-v2")
    top_k = req.top_k or (profile.top_k if profile else 5)
    methods = req.methods or (profile.methods if profile else RetrievalMethods())
    thresholds = req.confidence_thresholds or (
        profile.confidence_thresholds if profile else ConfidenceThresholds()
    )

    # ── 2. Build pipeline configuration ──────────────────────────────────────
    config = build_pipeline_config(
        backend=backend,
        collection_name=collection_name,
        embedding_model=embedding_model,
        enable_splade=methods.enable_splade,
        enable_er=methods.enable_graph or methods.enable_llm_graph,
        enable_rewrite=methods.enable_rewrite,
        enable_multi_query=methods.enable_multi_query,
        enable_hyde=methods.enable_hyde,
        enable_raptor=methods.enable_raptor,
        enable_contextual_rerank=methods.enable_contextual_rerank,
        enable_mmr=methods.enable_mmr,
    )

    # ── 3. Phase 1 — Query expansion (once, backend-agnostic LLM calls) ──────
    # Run in a thread so the FastAPI event loop stays unblocked
    _pipeline, bundle, expansion_traces = await asyncio.to_thread(
        _expand_query, config, req.query
    )

    # ── 4. Phase 2 — Retrieval + reranking + answer ───────────────────────────
    # Also offloaded to thread — CPU-bound embedding + BM25 must not block
    result = await asyncio.to_thread(
        _run_search_with_bundle,
        config,
        req.query,
        top_k,
        bundle,
        expansion_traces,
        methods.model_dump(),
    )

    if result.error:
        raise HTTPException(
            status_code=502,
            detail=f"Pipeline error for backend '{backend}': {result.error}",
        )

    # ── 5. Confidence verdict ─────────────────────────────────────────────────
    confidence = _avg_confidence(result.chunks)
    verdict = _compute_verdict(confidence, thresholds)

    # ── 6. Build unified traceable response ───────────────────────────────────
    return RagAnswer(
        query=req.query,
        profile_id=req.profile_id,
        profile_name=profile.name if profile else None,
        answer=result.answer,
        answer_confidence=confidence,
        verdict=verdict,
        sources=result.chunks,
        pipeline_audit=PipelineAudit(
            backend=backend,
            collection=collection_name,
            embedding_model=embedding_model,
            methods_active=_active_method_labels(methods),
            funnel=result.retrieval_trace,
            latency_ms=round(result.latency_ms, 2),
        ),
        graph=GraphSummary(
            entities=result.graph_entities,
            paths=result.graph_paths,
        ) if (result.graph_entities or result.graph_paths) else None,
        llm_traces=result.llm_traces,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
