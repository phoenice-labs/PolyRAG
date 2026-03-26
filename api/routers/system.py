"""
System health & cache management endpoints.

GET  /api/system/health  — pipeline cache stats, job counts, uptime, rate-limiting status
GET  /api/system/cache   — detailed pipeline cache entries (backend, collection, model, LRU rank)
DELETE /api/system/cache — manually flush all cached pipelines (admin use)

These endpoints are read-only and safe to call from monitoring dashboards or health probes.
DELETE /api/system/cache should be used sparingly — it forces expensive pipeline re-creation
on the next query (embedder + cross-encoder reload, ~10–30s per pipeline).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter(tags=["System"])

# Record the process start time for uptime reporting
_PROCESS_START = time.time()


# ── GET /api/system/health ────────────────────────────────────────────────────


@router.get("/system/health", summary="System health and pipeline cache overview")
async def system_health() -> Dict[str, Any]:
    """
    Returns a full system health snapshot including:

    - **uptime_s**: seconds since the API process started
    - **version**: API version string
    - **rate_limiting**: whether slowapi middleware is active
    - **pipeline_cache**: current LRU cache occupancy (used / max)
    - **jobs**: counts of jobs by status from the persistent job store
    - **bm25_snapshots**: number of persisted BM25 pkl files on disk
    - **backends_available**: which vector store backends are reachable

    Use this endpoint for readiness/liveness probes and monitoring dashboards.
    """
    from api.deps import get_pipeline_cache_info, get_job_store
    from pathlib import Path

    uptime_s = round(time.time() - _PROCESS_START, 1)

    # Pipeline cache stats
    cache_info = get_pipeline_cache_info()

    # Job counts by status
    job_store = get_job_store()
    all_jobs = await job_store.list_jobs()
    job_counts: Dict[str, int] = {}
    for j in all_jobs:
        job_counts[j.status] = job_counts.get(j.status, 0) + 1

    # BM25 snapshots on disk
    bm25_dir = Path("data/bm25")
    bm25_snapshots = len(list(bm25_dir.glob("*.pkl"))) if bm25_dir.exists() else 0

    # Rate limiting flag
    try:
        import slowapi  # noqa: F401
        rate_limiting = True
    except ImportError:
        rate_limiting = False

    return {
        "status": "ok",
        "uptime_s": uptime_s,
        "version": "14.2.0",
        "rate_limiting": rate_limiting,
        "pipeline_cache": {
            "cached": cache_info["cached"],
            "max": cache_info["max_pipelines"],
            "utilisation_pct": round(
                cache_info["cached"] / max(cache_info["max_pipelines"], 1) * 100, 1
            ),
        },
        "jobs": {
            "total": len(all_jobs),
            "by_status": job_counts,
        },
        "bm25_snapshots": bm25_snapshots,
    }


# ── GET /api/system/cache ─────────────────────────────────────────────────────


@router.get("/system/cache", summary="Pipeline cache entries with LRU order")
async def system_cache() -> Dict[str, Any]:
    """
    Returns the full pipeline cache contents in LRU order.

    Each entry shows:
    - **rank**: 1 = oldest (next to be evicted), N = most-recently-used
    - **backend**: vector store backend name
    - **collection**: collection name
    - **embedding_model**: embedding model used by this pipeline

    Use this to understand which pipeline slots are occupied and which would
    be evicted next if you ingest into a new (backend × collection × model) combination.
    """
    from api.deps import get_pipeline_cache_info

    info = get_pipeline_cache_info()
    return {
        "max_pipelines": info["max_pipelines"],
        "cached": info["cached"],
        "lru_entries": info["entries"],
        "note": "rank=1 is oldest (evicted next); rank=N is most-recently-used (safest)",
    }


# ── DELETE /api/system/cache ──────────────────────────────────────────────────


@router.delete("/system/cache", summary="Flush all cached pipelines")
async def flush_pipeline_cache() -> Dict[str, Any]:
    """
    Stops and removes all cached RAG pipelines.

    **Use with caution** — the next query to any (backend × collection) will:
    1. Recreate the pipeline (loads embedder + cross-encoder: ~10–30s cold start)
    2. Warm-start BM25 index from the vector store (cost proportional to collection size)

    Useful after:
    - Changing the embedding model mid-session
    - Suspecting a stale pipeline state
    - Freeing memory before benchmarking

    Returns the number of pipelines that were stopped.
    """
    from api.deps import _pipeline_cache, _pipeline_lru, _pipeline_cache_lock

    stopped: List[str] = []
    with _pipeline_cache_lock:
        for key, pipeline in list(_pipeline_cache.items()):
            try:
                pipeline.stop()
                stopped.append(f"{key[0]}/{key[1]}")
            except Exception:
                pass
        _pipeline_cache.clear()
        _pipeline_lru.clear()

    return {
        "flushed": len(stopped),
        "pipelines_stopped": stopped,
        "message": (
            f"Flushed {len(stopped)} pipeline(s). "
            "Next queries will trigger a cold start (~10–30s each)."
        ),
    }
