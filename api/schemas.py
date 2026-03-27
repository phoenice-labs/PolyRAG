"""
Pydantic request/response models for the Phoenice-PolyRAG API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ── LLM Trace (Phase B) ────────────────────────────────────────────────────────

class LLMTraceEntry(BaseModel):
    """One LLM call captured during a search request — for full observability."""
    method: str              # human-readable retrieval method name
    system_prompt: str       # system prompt sent to LLM
    user_message: str        # user/context message sent to LLM
    response: str            # raw LLM response text
    latency_ms: float        # round-trip latency in milliseconds


# ── Retrieval method flags ─────────────────────────────────────────────────────

class RetrievalMethods(BaseModel):
    # ── Independent methods — freely combinable ───────────────────────────────
    enable_dense: bool = True
    enable_bm25: bool = True
    enable_splade: bool = False     # SPLADE sparse neural (off by default — downloads ~440 MB model on first use)
    enable_graph: bool = True
    enable_rerank: bool = True      # Cross-Encoder (post-retrieval)
    enable_mmr: bool = True         # MMR diversity (post-retrieval)

    # ── LLM-required methods — some have parent-child dependencies ────────────
    enable_rewrite: bool = False    # parent of enable_multi_query
    enable_multi_query: bool = False  # child: requires enable_rewrite
    enable_hyde: bool = False
    enable_raptor: bool = False
    enable_contextual_rerank: bool = False
    enable_llm_graph: bool = False  # child of enable_graph: LLM entity extraction at query time

    @model_validator(mode='after')
    def enforce_dependencies(self) -> 'RetrievalMethods':
        """Enforce parent-child retrieval dependencies server-side."""
        if self.enable_multi_query and not self.enable_rewrite:
            self.enable_rewrite = True
        if self.enable_llm_graph and not self.enable_graph:
            self.enable_graph = True
        return self


EmbeddingModel = Literal[
    "all-MiniLM-L6-v2",       # 384-dim, ~80 MB,  fast  — default
    "BAAI/bge-base-en-v1.5",  # 768-dim, ~440 MB, balanced
    "BAAI/bge-large-en-v1.5", # 1024-dim, ~1.3 GB, best quality
]

# ── Request models ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: Optional[str] = None
    corpus_path: Optional[str] = None
    backends: List[str] = Field(default_factory=lambda: ["faiss"])
    chunk_strategy: str = "section"  # section | sliding | sentence | paragraph
    chunk_size: int = 400
    overlap: int = 50
    enable_er: bool = True      # entity-relation extraction (Knowledge Graph)
    enable_splade: bool = False  # pre-build SPLADE sparse neural index during ingestion
    collection_name: str = "polyrag_docs"
    embedding_model: EmbeddingModel = "all-MiniLM-L6-v2"


class SearchRequest(BaseModel):
    query: str
    backends: List[str] = Field(default_factory=lambda: ["faiss"])
    collection_name: str = "polyrag_docs"
    top_k: int = 5
    methods: RetrievalMethods = Field(default_factory=RetrievalMethods)
    embedding_model: EmbeddingModel = "all-MiniLM-L6-v2"


class CompareRequest(BaseModel):
    # ── Data source (choose one) ────────────────────────────────────────────
    # Use an already-ingested collection that exists across backends
    collection_name: Optional[str] = None
    # Paste raw text to ingest on-the-fly for comparison
    corpus_text: Optional[str] = None
    # Legacy: server-side file path (admin use only)
    corpus_path: Optional[str] = None

    backends: List[str] = Field(default_factory=lambda: ["faiss", "chromadb"])
    queries: Optional[List[str]] = None
    full_retrieval: bool = False
    compare_modes: bool = False
    corpus_limit: Optional[int] = None
    repeat_runs: int = Field(default=1, ge=1, le=10)   # for P50/P95 latency
    compare_graph_ab: bool = False                      # run each query with graph ON vs OFF


class FeedbackRequest(BaseModel):
    query: str
    chunk_id: str
    backend: str
    collection_name: str
    relevant: bool


class EvaluateQuestionItem(BaseModel):
    question: str
    expected_answer: str
    expected_sources: List[str] = Field(default_factory=list)


class EvaluateRequest(BaseModel):
    questions: List[EvaluateQuestionItem]
    backends: List[str] = Field(default_factory=lambda: ["faiss"])
    collection_name: str = "polyrag_docs"
    methods: RetrievalMethods = Field(default_factory=RetrievalMethods)


class ChunkPreviewRequest(BaseModel):
    text: Optional[str] = None
    corpus_path: Optional[str] = None
    strategy: str = "section"  # section | sliding | sentence | paragraph
    chunk_size: int = 400
    overlap: int = 50


# ── Response models ────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | error
    backend: str
    created_at: str
    updated_at: str
    log_lines: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ChunkItem(BaseModel):
    index: int
    text: str
    tokens: int
    char_start: int
    char_end: int
    parent_id: Optional[str] = None
    chunk_type: str = "section"
    entities: List[str] = Field(default_factory=list)


class ChunkPreviewResponse(BaseModel):
    chunks: List[ChunkItem]
    total_chunks: int
    total_chars: int
    avg_chunk_size: float
    strategy: str


class RetrievalTraceEntry(BaseModel):
    method: str
    candidates_before: int
    candidates_after: int
    scores: List[float] = Field(default_factory=list)


class MethodContribution(BaseModel):
    """One retrieval method's contribution to a chunk's final RRF score."""
    method: str               # e.g. "Dense Vector", "BM25 Keyword", "Knowledge Graph"
    rank: int                 # rank this method assigned to the chunk (1 = top)
    rrf_contribution: float   # 1/(k+rank) component added to RRF score


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[str] = None
    confidence: Optional[float] = None
    method_lineage: List[MethodContribution] = Field(default_factory=list)
    post_processors: List[str] = Field(default_factory=list)


class BackendSearchResult(BaseModel):
    backend: str
    answer: str
    chunks: List[SearchResultItem] = Field(default_factory=list)
    retrieval_trace: List[RetrievalTraceEntry] = Field(default_factory=list)
    llm_traces: List[LLMTraceEntry] = Field(default_factory=list)
    graph_entities: List[str] = Field(default_factory=list)
    graph_paths: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    method_contributions: Dict[str, Any] = Field(default_factory=dict)
    methods_used: Dict[str, bool] = Field(default_factory=dict)  # full flag map from request
    query_variants: Dict[str, Any] = Field(default_factory=dict)  # rewritten/hyde/stepback text
    error: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: Dict[str, BackendSearchResult]


class GraphChunkRef(BaseModel):
    chunk_id: str
    snippet: str = ""


class GraphNodeRelation(BaseModel):
    target_id: str
    target_label: str
    relation: str
    weight: float = 1.0


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "ENTITY"
    frequency: int = 1
    chunks: List[GraphChunkRef] = Field(default_factory=list)
    relations: List[GraphNodeRelation] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0
    doc_ids: List[str] = Field(default_factory=list)


class GraphResponse(BaseModel):
    collection: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class BackendInfo(BaseModel):
    name: str
    status: str  # available | connected | error
    ping_ms: Optional[float] = None
    collection_count: int = 0
    requires_docker: bool = False
    error: Optional[str] = None


class CompareChunkPreview(BaseModel):
    chunk_id: str
    text: str
    score: float
    method_lineage: List[MethodContribution] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompareBackendResult(BaseModel):
    backend: str
    query: str
    top_score: float = 0.0
    kw_hits: int = 0
    avg_score: float = 0.0
    result_count: int = 0
    query_latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    chunk_ids: List[str] = Field(default_factory=list)
    chunks: List[CompareChunkPreview] = Field(default_factory=list)
    # Graph A/B fields — populated when compare_graph_ab=True
    graph_entities: List[str] = Field(default_factory=list)
    graph_paths: List[str] = Field(default_factory=list)
    score_no_graph: float = 0.0
    score_delta: float = 0.0          # top_score (graph ON) - score_no_graph (graph OFF)
    latency_no_graph_ms: float = 0.0
    latency_with_graph_ms: float = 0.0
    error: Optional[str] = None
    method_contributions: Dict[str, Any] = Field(default_factory=dict)


class CompareSummary(BaseModel):
    backend: str
    base_top_score: float = 0.0
    full_top_score: float = 0.0
    base_kw_hits: int = 0
    avg_score: float = 0.0
    ingest_time_s: float = 0.0
    avg_query_latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_result_count: int = 0
    # Graph A/B summary fields
    avg_score_no_graph: float = 0.0
    avg_score_delta: float = 0.0      # positive = graph helped, negative = graph hurt
    avg_latency_no_graph_ms: float = 0.0
    avg_latency_with_graph_ms: float = 0.0
    errors: int = 0


class CompareResponse(BaseModel):
    per_query: List[CompareBackendResult]
    summary: List[CompareSummary]
