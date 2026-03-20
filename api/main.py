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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
)

app = FastAPI(
    title="Phoenice-PolyRAG API",
    description="Multi-backend RAG server with 12 retrieval methods and unified /api/rag agentic endpoint.",
    version="14.0.0",
)

# ── CORS (wide open for development) ─────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "service": "Phoenice-PolyRAG API"}


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
