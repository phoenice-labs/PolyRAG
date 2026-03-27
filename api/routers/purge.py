"""
Purge router: DELETE /api/purge/{collection}
             DELETE /api/purge  (purge ALL collections everywhere)

Holistic collection cleanup — sweeps every storage layer in one call:
  1. All 6 vector store backends (faiss, chromadb, qdrant, weaviate, milvus, pgvector)
  2. Knowledge Graph JSON snapshot (data/graphs/<collection>.json)
  3. Kuzu embedded graph DB (global, shared — cleared when any graph is purged)
  4. SPLADE index directory (data/splade/<collection>/)
  5. BM25 pkl files whose cache-key hash matches this collection
  6. Pipeline LRU cache eviction (all backends × collection)

Each layer is independent — failure in one layer does NOT stop others.
Returns a per-layer report so developers can see exactly what was cleaned.
"""
from __future__ import annotations

import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(tags=["purge"])

_ROOT = Path(__file__).resolve().parent.parent.parent
_GRAPHS_DIR  = _ROOT / "data" / "graphs"
_SPLADE_DIR  = _ROOT / "data" / "splade"
_BM25_DIR    = _ROOT / "data" / "bm25"
_KUZU_PATH   = _ROOT / "data" / "graph.kuzu"

ALL_BACKENDS = ["faiss", "chromadb", "qdrant", "weaviate", "milvus", "pgvector"]


# ── Layer helpers (sync, called via asyncio.to_thread) ───────────────────────

def _purge_vector_backend(backend: str, collection: str) -> Dict[str, Any]:
    """Delete collection from one vector backend. Returns per-backend result."""
    try:
        from api.deps import create_store_adapter
        store = create_store_adapter(backend)
        deleted = False
        try:
            if hasattr(store, "delete_collection"):
                store.delete_collection(collection)
                deleted = True
            elif hasattr(store, "reset"):
                store.reset(collection)
                deleted = True
        finally:
            try:
                store.close()
            except Exception:
                pass
        return {"deleted": deleted, "error": None}
    except Exception as exc:
        return {"deleted": False, "error": str(exc)}


def _purge_graph_snapshot(collection: str) -> Dict[str, Any]:
    """Delete data/graphs/<collection>.json."""
    snap = _GRAPHS_DIR / f"{collection}.json"
    if not snap.exists():
        return {"deleted": False, "path": str(snap), "error": None}
    try:
        snap.unlink()
        return {"deleted": True, "path": str(snap), "error": None}
    except Exception as exc:
        return {"deleted": False, "path": str(snap), "error": str(exc)}


def _purge_kuzu() -> Dict[str, Any]:
    """Clear ALL data from the shared Kuzu graph DB."""
    if not _KUZU_PATH.exists():
        return {"cleared": False, "note": "Kuzu DB not found", "error": None}
    try:
        import kuzu  # type: ignore[import]
        db = kuzu.Database(str(_KUZU_PATH))
        conn = kuzu.Connection(db)
        stmts = [
            "MATCH (e:Entity)-[r:APPEARS_IN]->(c:Chunk) DELETE r",
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) DELETE r",
            "MATCH (e:Entity) DELETE e",
            "MATCH (c:Chunk) DELETE c",
        ]
        for stmt in stmts:
            try:
                conn.execute(stmt)
            except Exception:
                pass
        conn = None
        db = None
        return {"cleared": True, "note": "Kuzu is global — all collections cleared", "error": None}
    except ImportError:
        return {"cleared": False, "note": "kuzu not installed — using NetworkX (in-memory only)", "error": None}
    except Exception as exc:
        return {"cleared": False, "note": None, "error": str(exc)}


def _purge_splade(collection: str) -> Dict[str, Any]:
    """Delete data/splade/<collection>/ directory."""
    splade_dir = _SPLADE_DIR / collection
    if not splade_dir.exists():
        return {"deleted": False, "path": str(splade_dir), "files_removed": 0, "error": None}
    try:
        files = list(splade_dir.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        shutil.rmtree(splade_dir)
        return {"deleted": True, "path": str(splade_dir), "files_removed": file_count, "error": None}
    except Exception as exc:
        return {"deleted": False, "path": str(splade_dir), "files_removed": 0, "error": str(exc)}


def _purge_bm25(collection: str) -> Dict[str, Any]:
    """
    Delete BM25 pkl files whose cache key contains this collection name.

    BM25 pkl paths are data/bm25/{md5(str(cache_key))[:16]}.pkl where cache_key
    is a 10-tuple. We can't reverse the hash, but we CAN enumerate all pkl files
    and check if any cache key with this collection name would hash to each file.

    Strategy: try all plausible cache-key combinations for this collection across
    all backends and known chunk sizes/strategies. Any pkl that matches is deleted.
    Remaining unmatched pkls are harmless orphans (they will never be re-used once
    the pipeline is evicted).
    """
    if not _BM25_DIR.exists():
        return {"deleted": 0, "scanned": 0, "error": None}

    pkl_files = list(_BM25_DIR.glob("*.pkl"))
    if not pkl_files:
        return {"deleted": 0, "scanned": 0, "error": None}

    # Build the set of hashes we expect for this collection
    # Parameters that vary per cache key — cover the combinations used in practice
    chunk_sizes    = [128, 256, 512, 1024, 2048]
    chunk_strategies = ["sentence", "fixed", "semantic"]
    overlaps       = [0, 32, 64, 128]
    raptor_opts    = [True, False]
    ctx_rerank_opts = [True, False]
    mmr_opts       = [True, False]
    splade_opts    = [True, False]
    embedding_models = [
        "all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]

    target_hashes: set = set()
    for backend in ALL_BACKENDS:
        for cs in chunk_sizes:
            for strategy in chunk_strategies:
                for overlap in overlaps:
                    for raptor in raptor_opts:
                        for ctx in ctx_rerank_opts:
                            for mmr in mmr_opts:
                                for splade in splade_opts:
                                    for model in embedding_models:
                                        key = (backend, collection, cs, strategy, overlap, raptor, ctx, mmr, splade, model)
                                        h = hashlib.md5(str(key).encode()).hexdigest()[:16]
                                        target_hashes.add(h)

    deleted = 0
    errors: List[str] = []
    for pkl in pkl_files:
        if pkl.stem in target_hashes:
            try:
                pkl.unlink()
                deleted += 1
            except Exception as exc:
                errors.append(str(exc))

    result: Dict[str, Any] = {"deleted": deleted, "scanned": len(pkl_files), "error": None}
    if errors:
        result["errors"] = errors
    return result


def _evict_pipelines(collection: str) -> Dict[str, Any]:
    """Evict all cached pipeline entries for this collection (all backends)."""
    try:
        from api.deps import evict_all_pipelines_for_collection
        count = evict_all_pipelines_for_collection(collection)
        return {"evicted": count, "error": None}
    except Exception as exc:
        return {"evicted": 0, "error": str(exc)}


# ── Purge endpoint ────────────────────────────────────────────────────────────

@router.delete("/purge/{collection}")
async def purge_collection(collection: str) -> Dict[str, Any]:
    """
    Holistically purge a collection from ALL storage layers.

    Sweeps:
      - All 6 vector store backends (non-fatal per backend if unavailable)
      - Knowledge Graph JSON snapshot
      - Kuzu embedded graph DB (global — clears all collections)
      - SPLADE index directory
      - BM25 pkl orphan files
      - Pipeline LRU cache

    Returns a per-layer report. Each layer is independent — failure in one
    does NOT stop the others.
    """
    # Run all vector backend deletes in parallel
    vector_tasks = [
        asyncio.to_thread(_purge_vector_backend, backend, collection)
        for backend in ALL_BACKENDS
    ]
    vector_results_list = await asyncio.gather(*vector_tasks)
    vector_results = dict(zip(ALL_BACKENDS, vector_results_list))

    # Run remaining layers concurrently
    graph_result, kuzu_result, splade_result, bm25_result = await asyncio.gather(
        asyncio.to_thread(_purge_graph_snapshot, collection),
        asyncio.to_thread(_purge_kuzu),
        asyncio.to_thread(_purge_splade, collection),
        asyncio.to_thread(_purge_bm25, collection),
    )

    # Pipeline eviction is fast — run inline
    pipeline_result = _evict_pipelines(collection)

    backends_deleted = [b for b, r in vector_results.items() if r["deleted"]]
    backends_skipped = [b for b, r in vector_results.items() if not r["deleted"] and not r["error"]]
    backends_error   = {b: r["error"] for b, r in vector_results.items() if r["error"]}

    return {
        "collection": collection,
        "summary": {
            "backends_deleted": backends_deleted,
            "backends_skipped": backends_skipped,
            "backends_error": backends_error,
            "graph_snapshot_deleted": graph_result["deleted"],
            "kuzu_cleared": kuzu_result["cleared"],
            "splade_files_removed": splade_result.get("files_removed", 0),
            "bm25_pkls_removed": bm25_result["deleted"],
            "pipelines_evicted": pipeline_result["evicted"],
        },
        "details": {
            "vector_stores": vector_results,
            "graph_snapshot": graph_result,
            "kuzu": kuzu_result,
            "splade": splade_result,
            "bm25": bm25_result,
            "pipeline_cache": pipeline_result,
        },
    }


@router.delete("/purge")
async def purge_all_collections() -> Dict[str, Any]:
    """
    Purge ALL collections from ALL storage layers.

    Discovers all collection names across all backends, then runs a full
    holistic purge for each. Also sweeps for orphaned graph/SPLADE/BM25
    files that may no longer have a corresponding vector store collection.
    """
    # Discover all collection names across all backends
    all_names: set = set()

    async def _collect_names(backend: str) -> None:
        try:
            from api.deps import create_store_adapter
            store = create_store_adapter(backend)
            try:
                names = store.list_collections() if hasattr(store, "list_collections") else []
                all_names.update(n for n in (names or []) if n not in {"__probe__"})
            finally:
                try:
                    store.close()
                except Exception:
                    pass
        except Exception:
            pass

    await asyncio.gather(*[_collect_names(b) for b in ALL_BACKENDS])

    # Also include any orphaned graph/SPLADE directories not in any live backend
    if _GRAPHS_DIR.exists():
        all_names.update(p.stem for p in _GRAPHS_DIR.glob("*.json"))
    if _SPLADE_DIR.exists():
        all_names.update(p.name for p in _SPLADE_DIR.iterdir() if p.is_dir())

    # Purge each collection
    per_collection: Dict[str, Any] = {}
    for name in sorted(all_names):
        result = await purge_collection(name)
        per_collection[name] = result["summary"]

    # Clear Kuzu once (global) and all BM25 pkls
    kuzu_result = await asyncio.to_thread(_purge_kuzu)
    bm25_all_deleted = 0
    if _BM25_DIR.exists():
        for pkl in _BM25_DIR.glob("*.pkl"):
            try:
                pkl.unlink()
                bm25_all_deleted += 1
            except Exception:
                pass

    # Evict entire pipeline cache
    try:
        from api.deps import evict_all_pipelines
        total_evicted = evict_all_pipelines()
    except Exception:
        total_evicted = 0

    return {
        "collections_purged": sorted(all_names),
        "count": len(all_names),
        "kuzu_cleared": kuzu_result["cleared"],
        "bm25_pkls_removed": bm25_all_deleted,
        "pipelines_evicted": total_evicted,
        "per_collection": per_collection,
    }
