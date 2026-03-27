"""
Backends router: GET /api/backends, GET /api/backends/{name}/health,
GET /api/collections/{backend}, DELETE /api/collections/{backend}/{collection},
DELETE /api/collections/{backend}  (clear all in backend)
"""
from __future__ import annotations

import socket
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import BackendInfo

router = APIRouter(tags=["backends"])

ALL_BACKENDS = ["faiss", "chromadb", "qdrant", "weaviate", "milvus", "pgvector"]

_DOCKER_BACKENDS = {"qdrant", "weaviate", "milvus", "pgvector"}
_DOCKER_PORTS = {
    "qdrant": 6333,
    "weaviate": 8088,
    "milvus": 19530,
    "pgvector": 5433,
}


def _ping_backend(name: str) -> BackendInfo:
    requires_docker = name in _DOCKER_BACKENDS
    t0 = time.monotonic()
    try:
        if name in ("faiss", "chromadb"):
            if name == "faiss":
                import faiss  # noqa: F401
            else:
                import chromadb  # noqa: F401
            ping_ms = (time.monotonic() - t0) * 1000
            return BackendInfo(name=name, status="available", ping_ms=round(ping_ms, 2),
                               collection_count=0, requires_docker=False)
        port = _DOCKER_PORTS.get(name)
        if port:
            s = socket.create_connection(("localhost", port), timeout=2)
            s.close()
            ping_ms = (time.monotonic() - t0) * 1000
            return BackendInfo(name=name, status="connected", ping_ms=round(ping_ms, 2),
                               collection_count=0, requires_docker=True)
        return BackendInfo(name=name, status="error", requires_docker=requires_docker, error="Unknown backend")
    except ImportError as exc:
        return BackendInfo(name=name, status="error", requires_docker=requires_docker, error=f"Import error: {exc}")
    except OSError as exc:
        return BackendInfo(name=name, status="error", requires_docker=requires_docker, error=f"Not reachable: {exc}")
    except Exception as exc:
        return BackendInfo(name=name, status="error", requires_docker=requires_docker, error=str(exc))


_SYSTEM_NAMES = {"__probe__"}


def _list_collections_for_backend(backend: str) -> List[Dict[str, Any]]:
    """Return list of collection dicts for a backend.

    Uses a lightweight store-adapter-only connection — no embedder or cross-encoder
    is loaded, so this is fast (~50–200 ms instead of 3–8 s).
    """
    from api.deps import create_store_adapter
    store = create_store_adapter(backend)
    results = []
    try:
        names = store.list_collections() if hasattr(store, "list_collections") else []
        for name in (names or []):
            if name in _SYSTEM_NAMES:
                continue
            try:
                count = store.count(name) if hasattr(store, "count") else 0
            except Exception:
                count = 0
            results.append({"name": name, "chunk_count": count, "index_type": backend})
    finally:
        try:
            store.close()
        except Exception:
            pass
    return results


def _delete_collection_in_backend(backend: str, collection: str) -> bool:
    """Delete one collection from a backend and evict its cached pipeline.

    Uses a lightweight store-adapter-only connection. After deletion, evicts all
    cached pipeline entries for this collection so stale BM25 / ChunkRegistry state
    is discarded. Also cleans the SPLADE index directory for this collection
    (collection-scoped, shared across backends — safe to remove idempotently).
    Raises the underlying exception if the adapter fails.
    """
    from pathlib import Path
    import shutil
    from api.deps import create_store_adapter, evict_pipeline_cache
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

    if deleted:
        evict_pipeline_cache(backend, collection)

        # Clean SPLADE index dir (collection-scoped, not backend-scoped)
        # Safe to remove even if already cleaned by a parallel backend delete.
        _root = Path(__file__).resolve().parent.parent.parent
        splade_dir = _root / "data" / "splade" / collection
        if splade_dir.exists():
            try:
                shutil.rmtree(splade_dir)
            except Exception:
                pass  # non-fatal: stale index is harmless but orphaned

    return deleted


@router.get("/backends", response_model=List[BackendInfo])
async def list_backends() -> List[BackendInfo]:
    import asyncio
    results = await asyncio.gather(*[asyncio.to_thread(_ping_backend, b) for b in ALL_BACKENDS])
    return list(results)


@router.get("/backends/{name}/health", response_model=BackendInfo)
async def backend_health(name: str) -> BackendInfo:
    if name not in ALL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {name}")
    import asyncio
    return await asyncio.to_thread(_ping_backend, name)


@router.get("/collections/{backend}")
async def list_collections(backend: str) -> List[Dict[str, Any]]:
    """List all collections in the given backend."""
    if backend not in ALL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {backend}")
    import asyncio
    try:
        return await asyncio.to_thread(_list_collections_for_backend, backend)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/collections/{backend}/{collection}")
async def delete_collection(backend: str, collection: str) -> Dict[str, Any]:
    """Delete a specific collection from a backend."""
    if backend not in ALL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {backend}")
    import asyncio
    try:
        deleted = await asyncio.to_thread(_delete_collection_in_backend, backend, collection)
        return {"deleted": deleted, "backend": backend, "collection": collection}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/collections/{backend}")
async def clear_all_collections(backend: str) -> Dict[str, Any]:
    """Delete ALL collections in a backend (full wipe)."""
    if backend not in ALL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {backend}")
    import asyncio
    try:
        collections = await asyncio.to_thread(_list_collections_for_backend, backend)
        deleted = []
        for col in collections:
            ok = await asyncio.to_thread(_delete_collection_in_backend, backend, col["name"])
            if ok:
                deleted.append(col["name"])
        return {"deleted": deleted, "backend": backend, "count": len(deleted)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
