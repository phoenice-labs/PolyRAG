"""
Search router: POST /api/search

LLM call optimization:
  Phase 1 (once):   Query expansion — rewrite, HyDE, multi-query, step-back
                    These are query-only; the vector DB used is irrelevant.
  Phase 2 (×N backends, parallel):
                    Retrieval, contextual re-ranking, answer generation
                    These operate on backend-specific retrieved chunks.

This reduces LLM calls from (4 × N_backends) to (4 + 2 × N_backends).
"""
from __future__ import annotations

import asyncio
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from api.deps import build_pipeline_config, create_pipeline
from api.schemas import (
    BackendSearchResult,
    LLMTraceEntry,
    RetrievalTraceEntry,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

router = APIRouter(tags=["search"])

_TRAIL_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "retrieval_trails.jsonl"
_trail_lock = threading.Lock()


def _append_retrieval_trail(record: dict) -> None:
    """Thread-safe append of one retrieval trail record to JSONL log."""
    _TRAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _trail_lock:
        with _TRAIL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _build_real_trace(pipeline, chunks: list) -> list:
    """
    Read the real per-phase trace from pipeline._last_retrieval_trace.
    Fall back to a single summary entry if not available.
    """
    raw = getattr(pipeline, "_last_retrieval_trace", None)
    if raw:
        return [
            RetrievalTraceEntry(
                method=step["method"],
                candidates_before=step["candidates_before"],
                candidates_after=step["candidates_after"],
                scores=[],
            )
            for step in raw
        ]
    # Fallback: single summary
    return [RetrievalTraceEntry(
        method="hybrid",
        candidates_before=len(chunks),
        candidates_after=len(chunks),
        scores=[c.score for c in chunks],
    )]


def _extract_query_variants(bundle) -> dict:
    """Extract query expansion variants from a QueryBundle for trail enrichment.

    Returns a dict with only the non-empty fields so the JSONL stays compact.
    """
    if bundle is None:
        return {}
    variants: dict = {}
    primary = getattr(bundle, "primary_query", None)
    if primary:
        variants["rewritten"] = primary
    paraphrases = getattr(bundle, "paraphrases", None)
    if paraphrases:
        variants["paraphrases"] = list(paraphrases)
    hyde_text = getattr(bundle, "hyde_text", None)
    if hyde_text:
        variants["hyde_text"] = hyde_text
    stepback = getattr(bundle, "stepback_query", None)
    if stepback:
        variants["stepback"] = stepback
    return variants


def _append_trail(
    query: str,
    backend: str,
    methods_used: dict,
    pipeline,
    chunks: list,
    latency_ms: float,
    bundle,
    method_contributions: dict | None = None,
) -> None:
    """Build and persist one retrieval trail record."""
    _append_retrieval_trail({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "backend": backend,
        "methods_used": methods_used,
        "query_variants": _extract_query_variants(bundle),
        "retrieval_trace": [
            {
                "method": s["method"],
                "candidates_before": s["candidates_before"],
                "candidates_after": s["candidates_after"],
            }
            for s in getattr(pipeline, "_last_retrieval_trace", [])
        ],
        "method_contributions": method_contributions or {},
        "result_count": len(chunks),
        "latency_ms": round(latency_ms, 2),
    })


def _compute_method_contributions(pipeline, chunks: list, methods_used: dict) -> dict:
    """
    Compute per-method contribution to the final result set.
    Reads _last_retrieval_trace (set by pipeline.query()) and correlates
    with chunk metadata (_method_lineage) to produce marginal contribution stats.
    Returns a dict keyed by method name.
    """
    contributions: dict = {}
    trace = getattr(pipeline, "_last_retrieval_trace", []) or []

    # Build candidate-count contribution from retrieval trace steps
    for step in trace:
        method = step.get("method", "unknown")
        before = step.get("candidates_before", 0)
        after = step.get("candidates_after", 0)
        contributions[method] = {
            "candidates_before": before,
            "candidates_after": after,
            "delta": after - before,
        }

    # Compute per-method chunk lineage from result metadata
    method_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        lineage = getattr(chunk, "method_lineage", None) or []
        for entry in lineage:
            m = getattr(entry, "method", None) or entry.get("method", "")
            if m:
                method_chunk_counts[m] = method_chunk_counts.get(m, 0) + 1

    total_chunks = len(chunks) or 1
    for method, count in method_chunk_counts.items():
        if method not in contributions:
            contributions[method] = {"candidates_before": 0, "candidates_after": 0, "delta": 0}
        contributions[method]["chunks_contributed"] = count
        contributions[method]["contribution_pct"] = round(count / total_chunks * 100, 1)

    # Annotate which methods were requested but contributed nothing
    for flag, enabled in methods_used.items():
        method_key = flag.replace("enable_", "")
        if enabled and method_key not in contributions and method_key not in method_chunk_counts:
            contributions[method_key] = {
                "candidates_before": 0, "candidates_after": 0, "delta": 0,
                "chunks_contributed": 0, "contribution_pct": 0.0,
            }

    return contributions


def _expand_query(config: dict, question: str):
    """
    Phase 1 — run once for all backends.
    Returns (pipeline, bundle, expansion_traces).
    The pipeline is the one used for expansion (any backend works since
    query expansion does not touch the vector store).
    """
    pipeline = create_pipeline(config)
    bundle, traces = pipeline.expand_query(question)
    return pipeline, bundle, traces


def _run_search_with_bundle(
    config: dict,
    query: str,
    top_k: int,
    bundle,
    expansion_traces: list,
    methods_used: dict = None,
) -> BackendSearchResult:
    """
    Phase 2 — per-backend retrieval + re-ranking + answer.
    Accepts the shared QueryBundle so no repeated query expansion LLM calls.
    """
    backend = config["store"]["backend"]
    t0 = time.monotonic()
    try:
        pipeline = create_pipeline(config)
        response = pipeline.ask_with_bundle(
            question=query,
            bundle=bundle,
            expansion_traces=expansion_traces,
            top_k=top_k,
            enable_dense=config.get("_enable_dense", True),
            enable_bm25=config.get("_enable_bm25", True),
            enable_splade=config.get("_enable_splade", False),
            enable_graph=config.get("_enable_graph", True),
            enable_rerank=config.get("_enable_rerank", True),
            enable_llm_graph=config.get("_enable_llm_graph", False),
        )
        latency_ms = (time.monotonic() - t0) * 1000

        from api.schemas import MethodContribution as MC
        _max_score = max((r.score for r in response.results), default=1.0) or 1.0
        chunks = [
            SearchResultItem(
                chunk_id=str(r.document.id),
                text=r.document.text,
                score=float(r.score),
                metadata={k: v for k, v in (r.document.metadata or {}).items() if not k.startswith("_")},
                provenance=None,
                confidence=float(r.score / _max_score),
                method_lineage=[
                    MC(method=c["method"], rank=c["rank"], rrf_contribution=c["rrf_contribution"])
                    for c in r.document.metadata.get("_method_lineage", [])
                ],
                post_processors=r.document.metadata.get("_post_processors", []),
            )
            for r in response.results
        ]

        llm_traces = [
            LLMTraceEntry(
                method=t.method,
                system_prompt=t.system_prompt,
                user_message=t.user_message,
                response=t.response,
                latency_ms=t.latency_ms,
            )
            for t in (response.llm_traces or [])
        ]

        # Use real per-phase trace collected inside pipeline.query()
        trace = _build_real_trace(pipeline, chunks)

        # Compute contributions before building result so it appears in API response
        contributions = _compute_method_contributions(pipeline, chunks, methods_used or {})

        result = BackendSearchResult(
            backend=backend,
            answer=response.answer or "",
            chunks=chunks,
            retrieval_trace=trace,
            llm_traces=llm_traces,
            graph_entities=list(response.graph_entities or []),
            graph_paths=[str(p) for p in (response.graph_paths or [])],
            latency_ms=round(latency_ms, 2),
            method_contributions=contributions,
        )

        # Persist retrieval trail
        _append_trail(query, backend, methods_used or {}, pipeline, chunks, latency_ms, bundle,
                      contributions)

        return result
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        # If the Milvus gRPC channel was closed, evict the stale pipeline from
        # cache and retry once with a fresh connection.
        exc_msg = str(exc).lower()
        if "closed channel" in exc_msg or "channel is closed" in exc_msg:
            from api.deps import _pipeline_cache, _pipeline_cache_lock
            backend_  = config["store"]["backend"]
            collection_ = config.get("ingestion", {}).get("collection_name",
                          config["store"].get("collection_name", "default"))
            # Evict ALL cache entries for this backend+collection
            with _pipeline_cache_lock:
                stale_keys = [k for k in list(_pipeline_cache) if k[0] == backend_ and k[1] == collection_]
                for k in stale_keys:
                    p = _pipeline_cache.pop(k, None)
                    if p:
                        try:
                            p.stop()
                        except Exception:
                            pass
            t0 = time.monotonic()
            try:
                pipeline = create_pipeline(config)
                response = pipeline.ask_with_bundle(
                    question=query,
                    bundle=bundle,
                    expansion_traces=expansion_traces,
                    top_k=top_k,
                    enable_dense=config.get("_enable_dense", True),
                    enable_bm25=config.get("_enable_bm25", True),
                    enable_graph=config.get("_enable_graph", True),
                    enable_rerank=config.get("_enable_rerank", True),
                    enable_llm_graph=config.get("_enable_llm_graph", False),
                )
                latency_ms = (time.monotonic() - t0) * 1000
                from api.schemas import MethodContribution as MC
                _max_score = max((r.score for r in response.results), default=1.0) or 1.0
                chunks = [
                    SearchResultItem(
                        chunk_id=str(r.document.id),
                        text=r.document.text,
                        score=float(r.score),
                        metadata={k: v for k, v in (r.document.metadata or {}).items() if not k.startswith("_")},
                        provenance=None,
                        confidence=float(r.score / _max_score),
                        method_lineage=[
                            MC(method=c["method"], rank=c["rank"], rrf_contribution=c["rrf_contribution"])
                            for c in r.document.metadata.get("_method_lineage", [])
                        ],
                        post_processors=r.document.metadata.get("_post_processors", []),
                    )
                    for r in response.results
                ]
                llm_traces = [
                    LLMTraceEntry(
                        method=t.method,
                        system_prompt=t.system_prompt,
                        user_message=t.user_message,
                        response=t.response,
                        latency_ms=t.latency_ms,
                    )
                    for t in (response.llm_traces or [])
                ]
                trace = _build_real_trace(pipeline, chunks)
                result = BackendSearchResult(
                    backend=backend,
                    answer=response.answer or "",
                    chunks=chunks,
                    retrieval_trace=trace,
                    llm_traces=llm_traces,
                    graph_entities=list(response.graph_entities or []),
                    graph_paths=[str(p) for p in (response.graph_paths or [])],
                    latency_ms=round(latency_ms, 2),
                    method_contributions=_compute_method_contributions(pipeline, chunks, methods_used or {}),
                )
                _append_trail(query, backend, methods_used or {}, pipeline, chunks, latency_ms, bundle)
                return result
            except Exception as exc2:
                latency_ms = (time.monotonic() - t0) * 1000
                return BackendSearchResult(
                    backend=backend,
                    answer="",
                    latency_ms=round(latency_ms, 2),
                    error=str(exc2),
                )
        return BackendSearchResult(
            backend=backend,
            answer="",
            latency_ms=round(latency_ms, 2),
            error=str(exc),
        )



@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """
    Multi-backend search with shared query expansion.
    LLM expansion runs once; retrieval + answer run per backend in parallel.
    """
    if not req.backends:
        raise HTTPException(status_code=400, detail="Provide at least one backend")

    m = req.methods

    # ── Phase 1: Query expansion (once, any backend) ────────────────────────
    first_backend = req.backends[0]
    expansion_config = build_pipeline_config(
        backend=first_backend,
        collection_name=req.collection_name,
        enable_rewrite=m.enable_rewrite,
        enable_multi_query=m.enable_multi_query,
        enable_hyde=m.enable_hyde,
        enable_stepback=getattr(m, "enable_stepback", False),
        enable_raptor=m.enable_raptor,
        enable_contextual_rerank=False,  # expansion pipeline does not need reranker
        embedding_model=req.embedding_model,
    )
    _, bundle, expansion_traces = await asyncio.to_thread(
        _expand_query, expansion_config, req.query
    )

    # ── Phase 2: Per-backend retrieval + rerank + answer (parallel) ─────────
    tasks = []
    for backend in req.backends:
        config = build_pipeline_config(
            backend=backend,
            collection_name=req.collection_name,
            enable_rewrite=False,         # expansion already done
            enable_multi_query=False,
            enable_hyde=False,
            enable_stepback=False,
            enable_raptor=m.enable_raptor,
            enable_contextual_rerank=m.enable_contextual_rerank,
            enable_mmr=m.enable_mmr,
            enable_er=True,               # always init graph store (snapshot loads in <1s)
            enable_splade=m.enable_splade,  # propagates to retrieval.splade.enabled for pipeline init
            embedding_model=req.embedding_model,
        )
        # Stash method flags in config so _run_search_with_bundle can read them
        config["_enable_dense"]  = m.enable_dense
        config["_enable_bm25"]   = m.enable_bm25
        config["_enable_splade"] = m.enable_splade
        config["_enable_graph"]  = m.enable_graph
        config["_enable_rerank"] = m.enable_rerank
        config["_enable_llm_graph"] = m.enable_llm_graph
        tasks.append(asyncio.to_thread(
            _run_search_with_bundle,
            config, req.query, req.top_k, bundle, expansion_traces,
            m.model_dump(),
        ))

    backend_results = await asyncio.gather(*tasks)
    results: Dict[str, BackendSearchResult] = {r.backend: r for r in backend_results}
    return SearchResponse(query=req.query, results=results)
