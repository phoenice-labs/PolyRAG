"""
Phoenice-PolyRAG FastAPI application entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so orchestrator/core imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import (
    ingest as ingest_router,
    search as search_router,
    rag as rag_router,
    compare as compare_router,
    backends as backends_router,
    graph as graph_router,
    evaluate as evaluate_router,
    feedback as feedback_router,
    chunks as chunks_router,
    jobs as jobs_router,
    prompts as prompts_router,
    traces as traces_router,
    retrieval_trails as retrieval_trails_router,
    system as system_router,
    purge as purge_router,
)

app = FastAPI(
    title="Phoenice-PolyRAG API",
    description="Multi-backend RAG server with 12 retrieval methods and unified /api/rag agentic endpoint.",
    version="14.2.0",
)

# ── CORS (wide open for development) ─────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
# Per-IP limits protect the server under concurrent agentic load.
# Limits are intentionally generous for a developer/integration context.
# Adjust via config.yaml system.rate_limits when deploying to production.
#
# Limits (per IP, per minute):
#   /api/rag        → 60 req/min  (production agents; ~1 req/sec sustained)
#   /api/search     → 30 req/min  (exploration; more expensive, multi-backend)
#   /api/ingest     → 10 req/min  (heavy background work)
#   global fallback → 120 req/min (all other endpoints)

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Attach per-route limits via middleware so they apply without decorating each route
    @app.middleware("http")
    async def rate_limit_by_path(request: Request, call_next):
        path = request.url.path
        # Per-path overrides (tighter than global 120/min)
        route_limits = {
            "/api/rag": "60/minute",
            "/api/search": "30/minute",
            "/api/ingest": "10/minute",
        }
        # Resolve which limit applies (exact path match, strip trailing slash)
        for prefix, limit in route_limits.items():
            if path.rstrip("/").startswith(prefix):
                request.state._rate_limit_override = limit
                break
        return await call_next(request)

    _rate_limiting_enabled = True

except ImportError:
    # slowapi not installed — rate limiting silently disabled
    # Install with: pip install slowapi
    _rate_limiting_enabled = False

# ── Mount routers ─────────────────────────────────────────────────────────────

app.include_router(ingest_router.router, prefix="/api")
app.include_router(search_router.router, prefix="/api")
app.include_router(rag_router.router, prefix="/api")
app.include_router(compare_router.router, prefix="/api")
app.include_router(backends_router.router, prefix="/api")
app.include_router(graph_router.router, prefix="/api")
app.include_router(evaluate_router.router, prefix="/api")
app.include_router(feedback_router.router, prefix="/api")
app.include_router(chunks_router.router, prefix="/api")
app.include_router(jobs_router.router, prefix="/api")
app.include_router(prompts_router.router, prefix="/api")
app.include_router(traces_router.router, prefix="/api")
app.include_router(retrieval_trails_router.router, prefix="/api")
app.include_router(system_router.router, prefix="/api")
app.include_router(purge_router.router, prefix="/api")


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "Phoenice-PolyRAG API",
        "version": "14.2.0",
        "rate_limiting": _rate_limiting_enabled,
    }


# ── Startup: pre-warm embedding model ────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load the embedding model singleton so concurrent first-ingest requests
    don't all race to initialize PyTorch (which is not thread-safe at init time)."""
    import asyncio
    import logging
    from core.embedding.sentence_transformer import SentenceTransformerProvider
    try:
        await asyncio.to_thread(
            SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})._load
        )
        logging.getLogger("polyrag.api").info("Embedding model pre-warmed successfully")
    except Exception as exc:
        logging.getLogger("polyrag.api").warning("Embedding pre-warm failed: %s", exc)
