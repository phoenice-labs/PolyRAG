"""
Dependency injection helpers: pipeline factory per backend, shared job store,
feedback store, and evaluation store.
"""
from __future__ import annotations

import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.jobs import JobStore

# ── Module-level singletons ───────────────────────────────────────────────────

_job_store: Optional[JobStore] = None
_feedback_store: List[Dict[str, Any]] = []
_eval_store: Dict[str, Any] = {}
_eval_store_loaded: bool = False   # disk load happens once on first get_eval_store() call

# Evaluation persistence directory (survives server restarts)
_EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluations"

# Shared temp directory for FAISS/ChromaDB persistent storage during server life
_tmp_dir: Optional[str] = None

# Pipeline cache: (backend, collection_name, ...) → RAGPipeline
# Pipelines are expensive to create (loads embedder, cross-encoder, etc.)
# Safe to share for read-only operations; each request uses begin_request/get_traces
# which are protected by per-pipeline lock below.
#
# LRU eviction: when the cache reaches MAX_CACHED_PIPELINES, the least-recently-used
# entry is stopped and removed before a new pipeline is added.
# This prevents unbounded memory growth when developers switch between many
# (backend, collection, model) combinations during experimentation.
MAX_CACHED_PIPELINES: int = 10  # ~10 distinct (backend × collection × model) combos

_pipeline_cache: Dict[tuple, Any] = {}          # cache_key → RAGPipeline
_pipeline_lru: "list[tuple]" = []               # ordered list: [oldest, ..., newest]
_pipeline_cache_lock = threading.Lock()

# ── Embedding model helpers ───────────────────────────────────────────────────

# Short slug appended to collection names so different-dimension models
# never mix vectors in the same collection.
_MODEL_SLUGS: Dict[str, str] = {
    "all-MiniLM-L6-v2": "minilm",
    "BAAI/bge-base-en-v1.5": "bge-base",
    "BAAI/bge-large-en-v1.5": "bge-large",
}


def _model_slug(model_name: str) -> str:
    """Return a short, filesystem-safe slug for the given embedding model."""
    if model_name in _MODEL_SLUGS:
        return _MODEL_SLUGS[model_name]
    # Fallback: last path component, lowercase, alphanumeric+hyphen only
    raw = model_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9\-]", "-", raw)


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


def evict_pipeline_cache(backend: str, collection: str) -> None:
    """Remove ALL cached pipeline entries for a given (backend, collection) pair.

    Must be called after a collection is deleted so that:
    - The stale BM25 index and ChunkRegistry on the cached pipeline are discarded.
    - Re-ingestion to the same collection name creates a fresh pipeline.
    """
    with _pipeline_cache_lock:
        to_remove = [k for k in _pipeline_cache if k[0] == backend and k[1] == collection]
        for key in to_remove:
            _evict_key(key)


def evict_all_pipelines_for_collection(collection: str) -> int:
    """Remove ALL cached pipeline entries for a collection across ALL backends.

    Used when the graph for a collection is cleared — the cached pipeline holds
    the in-memory graph store, so it must be discarded to prevent stale data.
    Returns the number of pipeline entries evicted.
    """
    with _pipeline_cache_lock:
        to_remove = [k for k in _pipeline_cache if k[1] == collection]
        for key in to_remove:
            _evict_key(key)
        return len(to_remove)


def evict_all_pipelines() -> int:
    """Evict the entire pipeline cache (all backends, all collections).

    Used when all graphs are cleared.
    Returns the number of pipeline entries evicted.
    """
    with _pipeline_cache_lock:
        keys = list(_pipeline_cache.keys())
        for key in keys:
            _evict_key(key)
        return len(keys)


def _evict_key(key: tuple) -> None:
    """Evict a single pipeline by cache key. Caller must hold _pipeline_cache_lock."""
    pipeline = _pipeline_cache.pop(key, None)
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass
    if key in _pipeline_lru:
        _pipeline_lru.remove(key)


def _touch_lru(key: tuple) -> None:
    """Mark key as most-recently-used. Caller must hold _pipeline_cache_lock."""
    if key in _pipeline_lru:
        _pipeline_lru.remove(key)
    _pipeline_lru.append(key)


def _enforce_lru_limit() -> None:
    """Evict the LRU pipeline if the cache is at capacity. Caller must hold lock."""
    while len(_pipeline_cache) >= MAX_CACHED_PIPELINES:
        if not _pipeline_lru:
            break
        oldest_key = _pipeline_lru[0]
        _evict_key(oldest_key)


def get_pipeline_cache_info() -> Dict[str, Any]:
    """Return a snapshot of the current pipeline cache state (for /api/system/cache)."""
    with _pipeline_cache_lock:
        entries = []
        for i, key in enumerate(_pipeline_lru):
            entries.append({
                "rank": i + 1,       # 1 = oldest (next to evict), N = newest
                "backend": key[0],
                "collection": key[1],
                "embedding_model": key[9] if len(key) > 9 else "unknown",
            })
        return {
            "max_pipelines": MAX_CACHED_PIPELINES,
            "cached": len(_pipeline_cache),
            "entries": entries,
        }


def get_feedback_store() -> List[Dict[str, Any]]:
    return _feedback_store


def get_eval_store() -> Dict[str, Any]:
    global _eval_store, _eval_store_loaded
    if not _eval_store_loaded:
        _load_eval_store()
        _eval_store_loaded = True
    return _eval_store


def _load_eval_store() -> None:
    """Load all persisted evaluation results from disk into the in-memory store."""
    import json
    if not _EVAL_DIR.exists():
        return
    for path in _EVAL_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if path.stem not in _eval_store:
                _eval_store[path.stem] = data
        except Exception:
            pass  # corrupt file — skip silently


def persist_eval(eval_id: str, data: Dict[str, Any]) -> None:
    """Write one evaluation result to disk. Best-effort — never raises."""
    import json
    try:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        (_EVAL_DIR / f"{eval_id}.json").write_text(
            json.dumps(data, default=str), encoding="utf-8"
        )
    except Exception:
        pass


def get_tmp_dir() -> str:
    """Return the persistent storage directory for in-process backends (FAISS, ChromaDB).

    Uses data/faiss/ and data/chromadb/ under the repo root so data survives
    API server restarts. The old tempfile-based approach caused data loss on restart.
    """
    global _tmp_dir
    if _tmp_dir is None:
        _root = Path(__file__).resolve().parent.parent
        _tmp_dir = str(_root / "data")
        Path(_tmp_dir).mkdir(parents=True, exist_ok=True)
    return _tmp_dir


def create_store_adapter(backend: str):
    """Instantiate and connect ONLY the store adapter — no embedder, no cross-encoder.

    Use for collection management (list/delete/count) that does not need a full pipeline.
    Returns a connected VectorStoreBase instance. Raises if the backend is unavailable.
    """
    from core.store.registry import AdapterRegistry

    _root = Path(__file__).resolve().parent.parent
    _data_dir = _root / "data"
    tmp_dir = get_tmp_dir()
    use_docker = _detect_docker(backend)

    if backend == "chromadb":
        store_cfg = {"mode": "persistent", "path": str(Path(tmp_dir) / "chromadb")}
    elif backend == "faiss":
        store_cfg = {"mode": "persistent", "path": str(Path(tmp_dir) / "faiss")}
    elif backend == "qdrant":
        store_cfg = (
            {"mode": "server", "url": "http://localhost:6333"}
            if use_docker else {"mode": "memory"}
        )
    elif backend == "weaviate":
        if not use_docker:
            raise RuntimeError("Weaviate is not reachable (Docker not running on port 8088)")
        store_cfg = {"mode": "server", "host": "localhost", "http_port": 8088, "grpc_port": 50052}
    elif backend == "milvus":
        store_cfg = (
            {"mode": "server", "host": "localhost", "port": 19530}
            if use_docker
            else {"mode": "local", "uri": str(_data_dir / "milvus_lite.db")}
        )
    elif backend == "pgvector":
        store_cfg = {
            "host": "localhost",
            "port": 5433 if use_docker else 5432,
            "database": "polyrag",
            "user": "postgres",
            "password": "postgres",
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    adapter = AdapterRegistry.create(backend, store_cfg)
    # Connect with a hard timeout so a slow backend doesn't block the API process
    exc_holder: list = []

    def _connect() -> None:
        try:
            adapter.connect()
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_connect, daemon=True)
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        raise RuntimeError(f"Connecting to {backend} timed out after 10 s")
    if exc_holder:
        raise exc_holder[0]
    return adapter


# ── Docker detection (same logic as compare_backends.py) ─────────────────────

_DOCKER_PORTS = {
    "qdrant": 6333,
    "weaviate": 8088,
    "milvus": 19530,
    "pgvector": 5433,
}


def _detect_docker(backend: str) -> bool:
    port = _DOCKER_PORTS.get(backend)
    if not port:
        return False
    try:
        s = socket.create_connection(("localhost", port), timeout=1)
        s.close()
        return True
    except OSError:
        return False


# ── Config builder (mirrors compare_backends._make_config) ───────────────────

def build_pipeline_config(
    backend: str,
    collection_name: str = "api_ingest",
    chunk_size: int = 400,
    chunk_strategy: str = "sentence",
    overlap: int = 50,
    enable_er: bool = True,
    full_retrieval: bool = False,
    enable_splade: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    # Per-method LLM retrieval flags (override full_retrieval when provided)
    enable_rewrite: Optional[bool] = None,
    enable_multi_query: Optional[bool] = None,
    enable_hyde: Optional[bool] = None,
    enable_stepback: Optional[bool] = None,
    enable_raptor: Optional[bool] = None,
    enable_contextual_rerank: Optional[bool] = None,
    enable_mmr: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a pipeline config dict for the given backend.

    Collection isolation: each embedding model gets its own collection suffix
    (e.g. polyrag_docs_minilm, polyrag_docs_bge-base) so different-dimension
    vectors never mix in the same collection.
    """
    _root = Path(__file__).resolve().parent.parent
    _data_dir = _root / "data"
    _data_dir.mkdir(parents=True, exist_ok=True)

    # ── Model-scoped collection name ──────────────────────────────────────────
    # Appending the model slug prevents dimension mismatch across encoder switches.
    # Idempotent: if the name already ends with the slug (because the UI shows
    # and re-selects the already-scoped collection name), don't append again.
    slug = _model_slug(embedding_model)
    if collection_name.endswith(f"_{slug}"):
        scoped_collection = collection_name   # already scoped — leave as-is
    else:
        scoped_collection = f"{collection_name}_{slug}"

    tmp_dir = get_tmp_dir()    # still needed for FAISS/ChromaDB index dirs
    use_docker = _detect_docker(backend)

    qdrant_cfg = (
        {"mode": "server", "url": "http://localhost:6333"}
        if use_docker and backend == "qdrant"
        else {"mode": "memory"}
    )
    weaviate_cfg = (
        {"mode": "server", "host": "localhost", "http_port": 8088, "grpc_port": 50052}
        if use_docker and backend == "weaviate"
        else {"mode": "embedded", "host": "localhost", "http_port": 8099, "grpc_port": 50060}
    )
    milvus_cfg = (
        {"mode": "server", "host": "localhost", "port": 19530}
        if use_docker and backend == "milvus"
        else {"mode": "local", "uri": str(_data_dir / "milvus_lite.db")}
    )
    pgvector_cfg = (
        {"host": "localhost", "port": 5433, "database": "polyrag",
         "user": "postgres", "password": "postgres"}
        if use_docker and backend == "pgvector"
        else {"host": "localhost", "port": 5432, "database": "polyrag",
              "user": "postgres", "password": "postgres"}
    )

    return {
        "store": {
            "backend": backend,
            "chromadb": {"mode": "persistent", "path": str(Path(tmp_dir) / "chromadb")},
            "faiss": {"mode": "persistent", "path": str(Path(tmp_dir) / "faiss")},
            "qdrant": qdrant_cfg,
            "weaviate": weaviate_cfg,
            "milvus": milvus_cfg,
            "pgvector": pgvector_cfg,
        },
        "embedding": {
            "provider": "sentence_transformer",
            "model": embedding_model,
            "device": "cpu",
            "batch_size": 32,
        },
        "ingestion": {
            "collection_name": scoped_collection,
            "chunk_size": chunk_size,
            "chunk_strategy": chunk_strategy,
            "chunk_overlap": overlap,
            "embed_batch_size": 32,
        },
        "retrieval": {
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "relevance_threshold": 0.0,
            "recall_multiplier": 3,
            "splade": {
                "enabled": enable_splade,
                "model": "naver/splade-cocondenser-selfdistil",  # public, non-gated, ~110 MB
                "persist_dir": "./data/splade",
                "splade_weight": 1.0,
                "bm25_weight_with_splade": 0.8,
            },
        },
        "llm": {
            "base_url": "http://localhost:1234/v1",
            "model": "mistralai/ministral-3b",
            "temperature": 0.2,
            "max_tokens": 256,
            # Per-method flags: explicit arg wins; fall back to full_retrieval toggle
            "enable_rewrite":     (enable_rewrite     if enable_rewrite     is not None else full_retrieval),
            "enable_stepback":    (enable_stepback     if enable_stepback    is not None else False),
            "enable_multi_query": (enable_multi_query  if enable_multi_query is not None else full_retrieval),
            "enable_hyde":        (enable_hyde         if enable_hyde        is not None else full_retrieval),
            "n_paraphrases": 2,
        },
        "access": {"user_clearance": "INTERNAL"},
        "quality": {"min_score": 0.1, "dedup_threshold": 0.85},
        "graph": {
            "enabled": enable_er,
            "backend": "networkx",
            "spacy_model": "en_core_web_sm",
            "max_hops": 1,
            "graph_weight": 0.5,
            "extract_svo": True,
            "extract_cooccurrence": True,
            "llm_extraction": {"enabled": False},
        },
        "advanced_retrieval": {
            "raptor": {"enabled": (enable_raptor if enable_raptor is not None else False), "n_clusters": 3, "max_tokens": 200},
            "contextual_reranker": {"enabled": (enable_contextual_rerank if enable_contextual_rerank is not None else False), "top_k": 5},
            "mmr": {"enabled": (enable_mmr if enable_mmr is not None else True), "diversity_weight": 0.3},
        },
        "audit_log_path": str(_data_dir / f"audit_{backend}.jsonl"),
    }


def create_pipeline(config: Dict[str, Any]):
    """
    Instantiate and start a RAGPipeline for the given config.
    Pipelines are cached per (backend, collection_name) — the heavy components
    (embedder, cross-encoder model, BM25 index) are loaded only once per process.
    """
    backend    = config["store"]["backend"]
    # collection_name in config["ingestion"] is already model-scoped (e.g. polyrag_docs_minilm)
    collection = config.get("ingestion", {}).get("collection_name",
                 config["store"].get("collection_name", "default"))
    # Cache key: only store-affecting settings create a new pipeline.
    # LLM method toggles (rewrite/multi_query/hyde/stepback) are reconfigured
    # on the existing pipeline rather than creating a new Milvus connection.
    # RAPTOR and Contextual Rerank require structural initialisation → stay in key.
    cache_key = (
        backend, collection,
        config.get("ingestion", {}).get("chunk_size", 512),
        config.get("ingestion", {}).get("chunk_strategy", "sentence"),
        config.get("ingestion", {}).get("chunk_overlap", 64),
        config.get("advanced_retrieval", {}).get("raptor", {}).get("enabled", False),
        config.get("advanced_retrieval", {}).get("contextual_reranker", {}).get("enabled", False),
        config.get("advanced_retrieval", {}).get("mmr", {}).get("enabled", True),
        config.get("retrieval", {}).get("splade", {}).get("enabled", False),  # SPLADE changes pipeline init
        config.get("embedding", {}).get("model", "all-MiniLM-L6-v2"),        # different dims = new pipeline
    )

    if cache_key in _pipeline_cache:
        # Reconfigure lightweight LLM flags on the cached pipeline without rebuilding
        pipeline = _pipeline_cache[cache_key]
        with _pipeline_cache_lock:
            _touch_lru(cache_key)
        _reconfigure_llm_flags(pipeline, config)
        return pipeline

    with _pipeline_cache_lock:
        if cache_key not in _pipeline_cache:
            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

            _enforce_lru_limit()  # evict LRU entry if at capacity

            from orchestrator.pipeline import RAGPipeline
            pipeline = RAGPipeline(config)
            pipeline.start()
            _pipeline_cache[cache_key] = pipeline
            _touch_lru(cache_key)
            # Persist BM25 snapshot after warm-start for faster future restarts
            _save_bm25_snapshot(cache_key, pipeline)

    _reconfigure_llm_flags(_pipeline_cache[cache_key], config)
    return _pipeline_cache[cache_key]


# ── BM25 index persistence ────────────────────────────────────────────────────
# The BM25 warm-start (pipeline._rebuild_bm25_from_store) fetches all chunk texts
# from the vector store on every process start, which can be slow for large collections.
#
# Strategy: after warm-start completes, snapshot the BM25 docs list to
#   data/bm25/{cache_key_hash}.pkl
# This provides a disk-side view of the index state (useful for diagnostics and as
# the foundation for a future "skip rebuild if pkl is fresh" optimisation).
#
# We do NOT modify orchestrator/pipeline.py to avoid touching the protected core.

_BM25_DIR = Path("data/bm25")


def _bm25_pkl_path(cache_key: tuple) -> Path:
    """Deterministic pkl path for a given cache key."""
    import hashlib
    key_hash = hashlib.md5(str(cache_key).encode()).hexdigest()[:16]
    return _BM25_DIR / f"{key_hash}.pkl"


def _save_bm25_snapshot(cache_key: tuple, pipeline) -> None:
    """
    Serialize the pipeline's BM25 docs list to disk after warm-start.
    Non-blocking — silently swallows all errors (persistence is best-effort).

    The snapshot path: data/bm25/{cache_key_hash}.pkl
    """
    try:
        import pickle

        bm25_index = getattr(pipeline, "bm25_index", None)
        if bm25_index is None:
            return
        docs = getattr(bm25_index, "_docs", [])
        if not docs:
            return

        _BM25_DIR.mkdir(parents=True, exist_ok=True)
        pkl_path = _bm25_pkl_path(cache_key)
        with pkl_path.open("wb") as f:
            pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # persistence is best-effort; warm-start already succeeded


def get_bm25_snapshot_info() -> list:
    """
    Return metadata about on-disk BM25 snapshots for /api/system/health.
    """
    if not _BM25_DIR.exists():
        return []
    snapshots = []
    for pkl in _BM25_DIR.glob("*.pkl"):
        stat = pkl.stat()
        snapshots.append({
            "file": pkl.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": stat.st_mtime,
        })
    return snapshots


def _reconfigure_llm_flags(pipeline, config: dict) -> None:
    """
    Reconfigure LLM query-intelligence flags on a cached pipeline.
    Avoids creating a new Milvus/store connection just because LLM toggles changed.
    """
    llm_cfg = config.get("llm", {})
    qp = getattr(pipeline, "_query_pipeline", None)
    if qp is None:
        return
    # Toggle each component on/off by replacing it with None or a live instance
    if llm_cfg.get("enable_rewrite", False):
        if qp.rewriter is None:
            from core.query.rewriter import QueryRewriter
            qp.rewriter = QueryRewriter(pipeline._llm_client)
    else:
        qp.rewriter = None

    if llm_cfg.get("enable_multi_query", False):
        if qp.multi_query is None:
            from core.query.rewriter import MultiQueryGenerator
            qp.multi_query = MultiQueryGenerator(pipeline._llm_client, llm_cfg.get("n_paraphrases", 3))
    else:
        qp.multi_query = None

    if llm_cfg.get("enable_hyde", False):
        if qp.hyde is None:
            from core.query.rewriter import QueryExpander
            qp.hyde = QueryExpander(pipeline._llm_client, pipeline.embedder)
    else:
        qp.hyde = None

    if llm_cfg.get("enable_stepback", False):
        if qp.stepback is None:
            from core.query.rewriter import StepBackPrompter
            qp.stepback = StepBackPrompter(pipeline._llm_client)
    else:
        qp.stepback = None


