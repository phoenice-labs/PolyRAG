"""
Compare router: POST /api/compare

Three data-source modes (in priority order):
  1. collection_name  — use an existing ingested collection (no re-ingestion)
  2. corpus_text      — paste raw text; ingest into a temporary compare_<backend> collection
  3. corpus_path      — legacy server-side file path (admin use)

Returns synchronous CompareResponse with per-backend summary rows and per-query rows.
base_top_score uses dense+BM25+rerank; full_top_score uses LLM query-intelligence (when full_retrieval=True).
repeat_runs > 1 enables P50/P95 latency measurement.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from api.deps import build_pipeline_config, create_pipeline
from api.schemas import (
    CompareBackendResult,
    CompareChunkPreview,
    CompareRequest,
    CompareResponse,
    CompareSummary,
)

router = APIRouter(tags=["compare"])

_DEFAULT_QUERIES = [
    "What are the main topics covered?",
    "Summarise the key points.",
    "Who are the main entities mentioned?",
]

_SAMPLE_QUERIES = [
    "What are the main topics covered?",
    "Summarise the key findings.",
    "Who are the main characters or entities?",
    "What events or actions are described?",
    "What conclusions or outcomes are mentioned?",
]


@router.get("/compare/sample-queries")
def sample_queries() -> List[str]:
    """Return a list of ready-to-use benchmark queries."""
    return _SAMPLE_QUERIES


def _run_backend_compare(
    backend: str,
    collection: str,
    text_to_ingest: Optional[str],
    queries: List[str],
    full_retrieval: bool,
    repeat_runs: int = 1,
    compare_graph_ab: bool = False,
) -> tuple[List[CompareBackendResult], CompareSummary]:
    """Synchronous compare run for one backend — called via asyncio.to_thread.

    When text_to_ingest is None the collection already exists; ingestion is skipped.
    When full_retrieval=True, each query also runs with LLM methods ON to populate
    full_top_score (different from base_top_score which uses dense+BM25 only).
    When repeat_runs > 1, each query runs N times to compute P50/P95 latency.
    When compare_graph_ab=True, each query runs twice — graph OFF then graph ON —
    to measure the graph's contribution to score and latency.
    """
    config_base = build_pipeline_config(
        backend=backend,
        collection_name=collection,
        full_retrieval=False,
        enable_er=True,    # graph capable; A/B control is per-query via ask_with_bundle
    )
    config_no_graph = build_pipeline_config(
        backend=backend,
        collection_name=collection,
        full_retrieval=False,
        enable_er=False,   # graph explicitly disabled for A/B baseline
    ) if compare_graph_ab else None

    config_full = (
        build_pipeline_config(backend=backend, collection_name=collection, full_retrieval=True)
        if full_retrieval
        else None
    )

    per_query: List[CompareBackendResult] = []
    ingest_time = 0.0
    errors = 0
    base_top_scores: List[float] = []
    full_top_scores: List[float] = []
    query_latencies: List[float] = []
    result_counts: List[int] = []
    graph_ab_deltas: List[float] = []
    latencies_no_graph: List[float] = []
    latencies_with_graph: List[float] = []

    try:
        t0 = time.monotonic()
        pipeline = create_pipeline(config_base)

        # Only ingest if caller supplied raw text (not using an existing collection)
        if text_to_ingest is not None:
            pipeline.ingest_text(text_to_ingest, {})
        ingest_time = time.monotonic() - t0

        # Separate pipeline instance for graph-OFF baseline (A/B only)
        pipeline_no_graph = create_pipeline(config_no_graph) if config_no_graph else None

        for q in queries:
            try:
                # ── Repeat runs for latency percentiles ───────────────────
                run_latencies: List[float] = []
                resp_base = None
                for _ in range(max(1, repeat_runs)):
                    t_query = time.monotonic()
                    resp_base = pipeline.ask(q, top_k=5)
                    run_latencies.append((time.monotonic() - t_query) * 1000)

                latency_ms = run_latencies[-1]
                p50 = statistics.median(run_latencies)
                p95 = sorted(run_latencies)[int(len(run_latencies) * 0.95)] if len(run_latencies) > 1 else latency_ms

                # ── Base retrieval metrics ────────────────────────────────
                base_scores = [float(getattr(r, "score", 0.0)) for r in (resp_base.results or [])]
                base_top = max(base_scores) if base_scores else 0.0
                base_avg = sum(base_scores) / len(base_scores) if base_scores else 0.0
                n_results = len(resp_base.results or [])

                # Word-level keyword hit count
                query_words = [w for w in q.lower().split() if len(w) > 2]
                base_kw_hits = sum(
                    1
                    for r in (resp_base.results or [])
                    if any(w in (getattr(r, "text", "") or "").lower() for w in query_words)
                )
                base_top_scores.append(base_top)
                query_latencies.append(p50)
                result_counts.append(n_results)

                # ── Chunk previews & IDs (for preview panel + overlap) ────
                chunk_previews = [
                    CompareChunkPreview(
                        chunk_id=str(r.document.id),
                        text=(r.document.text or "")[:300],
                        score=float(r.score),
                    )
                    for r in (resp_base.results or [])
                ]
                chunk_ids = [str(r.document.id) for r in (resp_base.results or [])]

                # ── Graph trail: entities + paths from graph-enabled run ──
                graph_entities: List[str] = list(resp_base.graph_entities or [])
                graph_paths: List[str] = [str(p) for p in (resp_base.graph_paths or [])]

                # ── Graph A/B: run same query without graph to get baseline ─
                score_no_graph = 0.0
                lat_no_graph = 0.0
                lat_with_graph = p50
                score_delta = 0.0

                if pipeline_no_graph is not None:
                    t_ng = time.monotonic()
                    resp_ng = pipeline_no_graph.ask(q, top_k=5)
                    lat_no_graph = (time.monotonic() - t_ng) * 1000
                    ng_scores = [float(getattr(r, "score", 0.0)) for r in (resp_ng.results or [])]
                    score_no_graph = max(ng_scores) if ng_scores else 0.0
                    score_delta = round(base_top - score_no_graph, 4)
                    graph_ab_deltas.append(score_delta)
                    latencies_no_graph.append(lat_no_graph)
                    latencies_with_graph.append(lat_with_graph)

                # ── Full retrieval (LLM methods ON, only when requested) ──
                if config_full is not None:
                    pipeline_full = create_pipeline(config_full)
                    resp_full = pipeline_full.ask(q, top_k=5)
                    full_scores = [float(getattr(r, "score", 0.0)) for r in (resp_full.results or [])]
                    full_top = max(full_scores) if full_scores else 0.0
                else:
                    full_top = base_top
                full_top_scores.append(full_top)

                per_query.append(
                    CompareBackendResult(
                        backend=backend,
                        query=q,
                        top_score=round(base_top, 4),
                        kw_hits=base_kw_hits,
                        avg_score=round(base_avg, 4),
                        result_count=n_results,
                        query_latency_ms=round(latency_ms, 1),
                        latency_p50_ms=round(p50, 1),
                        latency_p95_ms=round(p95, 1),
                        chunk_ids=chunk_ids,
                        chunks=chunk_previews,
                        graph_entities=graph_entities,
                        graph_paths=graph_paths,
                        score_no_graph=round(score_no_graph, 4),
                        score_delta=score_delta,
                        latency_no_graph_ms=round(lat_no_graph, 1),
                        latency_with_graph_ms=round(lat_with_graph, 1),
                    )
                )
            except Exception as exc:
                errors += 1
                per_query.append(
                    CompareBackendResult(backend=backend, query=q, error=str(exc))
                )
    except Exception as exc:
        errors += 1

    overall_avg = sum(base_top_scores) / len(base_top_scores) if base_top_scores else 0.0
    avg_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
    all_p50s = [r.latency_p50_ms for r in per_query if not r.error]
    all_p95s = [r.latency_p95_ms for r in per_query if not r.error]
    summary = CompareSummary(
        backend=backend,
        base_top_score=round(max(base_top_scores, default=0.0), 4),
        full_top_score=round(max(full_top_scores, default=0.0), 4),
        base_kw_hits=sum(r.kw_hits for r in per_query),
        avg_score=round(overall_avg, 4),
        ingest_time_s=round(ingest_time, 2),
        avg_query_latency_ms=round(avg_latency, 1),
        latency_p50_ms=round(statistics.median(all_p50s) if all_p50s else 0.0, 1),
        latency_p95_ms=round(max(all_p95s) if all_p95s else 0.0, 1),
        total_result_count=sum(result_counts),
        avg_score_no_graph=round(sum(r.score_no_graph for r in per_query if not r.error) / max(len(per_query), 1), 4),
        avg_score_delta=round(sum(graph_ab_deltas) / len(graph_ab_deltas) if graph_ab_deltas else 0.0, 4),
        avg_latency_no_graph_ms=round(sum(latencies_no_graph) / len(latencies_no_graph) if latencies_no_graph else 0.0, 1),
        avg_latency_with_graph_ms=round(sum(latencies_with_graph) / len(latencies_with_graph) if latencies_with_graph else 0.0, 1),
        errors=errors,
    )
    return per_query, summary


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest) -> CompareResponse:
    """Run side-by-side backend comparison and return structured results."""
    if not req.backends:
        raise HTTPException(status_code=400, detail="Provide at least one backend")

    # ── Resolve data source ──────────────────────────────────────────────────
    text_to_ingest: Optional[str] = None
    collection: str

    if req.collection_name:
        # Mode 1: use an existing collection — no ingestion
        collection = req.collection_name
    elif req.corpus_text:
        # Mode 2: user-pasted text — ingest into temp compare collection
        collection = "compare_run"
        text_to_ingest = req.corpus_text
    elif req.corpus_path:
        # Mode 3: legacy server file path
        try:
            with open(req.corpus_path, "r", encoding="utf-8") as f:
                text_to_ingest = f.read()
            if req.corpus_limit:
                text_to_ingest = text_to_ingest[: req.corpus_limit]
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read corpus_path: {exc}")
        collection = "compare_run"
    else:
        # Default: built-in sample text so the comparison always produces results
        text_to_ingest = (
            "Artificial intelligence and machine learning are transforming industries. "
            "Natural language processing enables computers to understand human language. "
            "Vector databases power semantic search and retrieval-augmented generation. "
            "RAG systems combine retrieval with language model generation for accurate answers."
        )
        collection = "compare_run"

    queries = req.queries or _DEFAULT_QUERIES

    tasks = [
        asyncio.to_thread(
            _run_backend_compare, b, collection, text_to_ingest, queries,
            req.full_retrieval, req.repeat_runs, req.compare_graph_ab,
        )
        for b in req.backends
    ]
    results = await asyncio.gather(*tasks)

    all_per_query: List[CompareBackendResult] = []
    all_summaries: List[CompareSummary] = []
    for per_query, summary in results:
        all_per_query.extend(per_query)
        all_summaries.append(summary)

    return CompareResponse(per_query=all_per_query, summary=all_summaries)

