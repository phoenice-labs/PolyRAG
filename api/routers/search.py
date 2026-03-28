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


def _normalize_confidence(score: float, min_s: float, max_s: float, idx: int, n: int) -> float:
    """Min-max normalize a retrieval score into a [0, 1] confidence value.

    When all results share the same score (e.g. graph-only direct-match hop=0 → 1.0)
    the range is zero, so we fall back to a rank-based decay: rank 0 → 1.0, last → 0.0.
    This fixes the bug where graph results always showed 100 % confidence.
    """
    rng = max_s - min_s
    if rng > 1e-6:
        return round((score - min_s) / rng, 4)
    # Flat-score fallback: rank-based linear decay
    return round(max(0.0, 1.0 - idx / max(n, 1)), 4)


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
    Returns a dict keyed by display method name with full diagnostic detail:
      - status: 'active' | 'zero' | 'disabled'
      - reason: human-readable explanation
      - chunks_contributed: unique chunks this method introduced (not found by any earlier method)
      - rrf_boost_total: total RRF score this method added across all final chunks (score boost even when chunk already found)
      - avg_rank: average rank this method assigned to the chunks it scored
      - contribution_pct, candidates_before/after, delta
    ALL 10 methods are always present so the UI can show a complete picture.
    """
    # Canonical map: flag name → display name
    FLAG_TO_DISPLAY = {
        "enable_dense":              "Dense Vector",
        "enable_bm25":               "BM25 Keyword",
        "enable_splade":             "SPLADE Sparse Neural",
        "enable_graph":              "Knowledge Graph",
        "enable_rerank":             "Cross-Encoder Rerank",
        "enable_mmr":                "MMR Diversity",
        "enable_rewrite":            "Query Rewriting",
        "enable_multi_query":        "Multi-Query",
        "enable_hyde":               "HyDE Expansion",
        "enable_raptor":             "RAPTOR",
        "enable_contextual_rerank":  "Contextual Rerank",
        "enable_llm_graph":          "LLM Graph Extraction",
    }

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

    # Compute per-method chunk lineage: unique chunks introduced + RRF boost + avg rank
    # A chunk is "unique" to a method only if it appeared in ONLY that method's lineage
    # across the final result set (exclusive contribution).
    # RRF boost = total rrf_contribution across all chunks this method touched.
    method_chunk_ids: dict[str, set] = {}    # method → set of chunk_ids it appears in
    method_rrf_boost: dict[str, float] = {}  # method → sum of rrf_contribution
    method_ranks: dict[str, list] = {}       # method → list of ranks it assigned

    for chunk in chunks:
        lineage = getattr(chunk, "method_lineage", None) or []
        methods_in_this_chunk = set()
        for entry in lineage:
            m = getattr(entry, "method", None) or (entry.get("method", "") if isinstance(entry, dict) else "")
            rrf = getattr(entry, "rrf_contribution", None) or (entry.get("rrf_contribution", 0.0) if isinstance(entry, dict) else 0.0)
            rank = getattr(entry, "rank", None) or (entry.get("rank", 0) if isinstance(entry, dict) else 0)
            if not m:
                continue
            methods_in_this_chunk.add(m)
            method_rrf_boost[m] = method_rrf_boost.get(m, 0.0) + float(rrf)
            method_ranks.setdefault(m, []).append(int(rank))
        chunk_id = getattr(chunk, "chunk_id", None) or str(id(chunk))
        for m in methods_in_this_chunk:
            method_chunk_ids.setdefault(m, set()).add(chunk_id)

    # All chunk_ids touched by Dense or BM25 (the "baseline" methods always present)
    baseline_ids = (method_chunk_ids.get("Dense Vector", set()) |
                    method_chunk_ids.get("BM25 Keyword", set()))

    total_chunks = len(chunks) or 1
    for method, chunk_ids in method_chunk_ids.items():
        if method not in contributions:
            contributions[method] = {"candidates_before": 0, "candidates_after": 0, "delta": 0}
        # Unique chunks = those NOT already found by the baseline (Dense+BM25)
        if method in ("Dense Vector", "BM25 Keyword"):
            unique_ids = chunk_ids  # baseline methods count all their chunks
        else:
            unique_ids = chunk_ids - (baseline_ids - chunk_ids)  # chunks exclusively new
        unique_count = len(unique_ids)
        rrf_boost = round(method_rrf_boost.get(method, 0.0), 6)
        ranks = method_ranks.get(method, [])
        avg_rank = round(sum(ranks) / len(ranks), 1) if ranks else 0.0
        # chunks_contributed = how many final chunks this method appears in (lineage presence)
        chunks_in_result = len(chunk_ids)
        contributions[method]["chunks_contributed"] = chunks_in_result
        contributions[method]["unique_chunks_added"] = unique_count
        contributions[method]["rrf_boost_total"] = rrf_boost
        contributions[method]["avg_rank"] = avg_rank
        contributions[method]["contribution_pct"] = round(chunks_in_result / total_chunks * 100, 1)

    # Annotate status + reason for every method that was enabled
    for flag, enabled in methods_used.items():
        display = FLAG_TO_DISPLAY.get(flag, flag.replace("enable_", "").replace("_", " ").title())
        if not enabled:
            # Not enabled — add as disabled entry so UI can show it
            if display not in contributions:
                contributions[display] = {
                    "candidates_before": 0, "candidates_after": 0, "delta": 0,
                    "chunks_contributed": 0, "unique_chunks_added": 0,
                    "rrf_boost_total": 0.0, "avg_rank": 0.0, "contribution_pct": 0.0,
                    "status": "disabled",
                    "reason": "Not enabled in this request. Toggle it ON in Method Settings to evaluate its impact.",
                }
            elif "status" not in contributions[display]:
                contributions[display]["status"] = "disabled"
                contributions[display]["reason"] = "Not enabled in this request. Toggle it ON in Method Settings to evaluate its impact."
            continue
        if display in contributions:
            entry = contributions[display]
            pct = entry.get("contribution_pct", 0.0)
            rrf = entry.get("rrf_boost_total", 0.0)
            unique = entry.get("unique_chunks_added", 0)
            chunks_in = entry.get("chunks_contributed", 0)
            if pct > 0 or rrf > 0:
                entry["status"] = "active"
                parts = [f"Present in {chunks_in}/{total_chunks} final chunks ({pct:.1f}%)"]
                if unique > 0:
                    parts.append(f"{unique} unique chunk(s) introduced beyond Dense+BM25")
                if rrf > 0:
                    parts.append(f"RRF score boost: +{rrf:.4f} total across all final chunks")
                if entry.get("avg_rank", 0):
                    parts.append(f"avg rank assigned: {entry['avg_rank']}")
                entry["reason"] = ". ".join(parts) + "."
            else:
                entry["status"] = "zero"
                entry["reason"] = _explain_zero_contribution(flag, entry)
        else:
            # Enabled but left no trace — explain why
            contributions[display] = {
                "candidates_before": 0, "candidates_after": 0, "delta": 0,
                "chunks_contributed": 0, "unique_chunks_added": 0,
                "rrf_boost_total": 0.0, "avg_rank": 0.0, "contribution_pct": 0.0,
                "status": "zero",
                "reason": _explain_zero_contribution(flag, {}),
            }

    # Ensure all known methods are present even if not mentioned in methods_used
    for flag, display in FLAG_TO_DISPLAY.items():
        if display not in contributions:
            contributions[display] = {
                "candidates_before": 0, "candidates_after": 0, "delta": 0,
                "chunks_contributed": 0, "unique_chunks_added": 0,
                "rrf_boost_total": 0.0, "avg_rank": 0.0, "contribution_pct": 0.0,
                "status": "disabled",
                "reason": "Not enabled in this request.",
            }

    return contributions


def _explain_zero_contribution(flag: str, entry: dict) -> str:
    """Return a developer-friendly explanation for why an enabled method contributed 0 chunks."""
    rrf = entry.get("rrf_boost_total", 0.0)
    splade_rrf_note = (
        f" Note: SPLADE did add an RRF score boost of +{rrf:.4f} to chunks already found by Dense/BM25 "
        f"— meaning it agreed with those results and elevated their ranking, but introduced no NEW chunks. "
        f"This is normal behaviour with small chunk sizes (≤256 chars) where Dense+BM25 together already surface all relevant chunks. "
        f"To force SPLADE to contribute unique chunks: (a) re-ingest with SPLADE enabled so the index is pre-built; "
        f"(b) try larger chunk sizes (≥512 chars) so SPLADE's term-expansion has more vocabulary to work with."
    ) if rrf > 0 else (
        " To fix: re-ingest your collection with 'Enable SPLADE Index' toggled ON in Ingestion Studio. "
        "This pre-encodes sparse vectors during ingestion so SPLADE can immediately participate in RRF fusion. "
        "Without a pre-built index, SPLADE cannot contribute to RRF fusion (the index must be built at ingest time). "
        "With small chunks (≤256 chars) Dense+BM25 often subsume all SPLADE results — try chunk size ≥512."
    )
    reasons = {
        "enable_splade": (
            "SPLADE was enabled but contributed 0 unique chunks after RRF fusion."
            + splade_rrf_note
        ),
        "enable_graph": (
            "Knowledge Graph was enabled but contributed 0 unique chunks. "
            "Possible causes: (a) No entities were extracted from the query; "
            "(b) Graph snapshot not found — run ingestion with ER enabled; "
            "(c) Graph traversal found no paths connected to query entities."
        ),
        "enable_rerank": (
            "Cross-Encoder Rerank is a post-processor — it re-orders existing chunks, "
            "not a retrieval source. 0 unique chunks is expected (it refines rankings, not adds results). "
            "Check the Pipeline tab to see how it changed candidate ordering."
        ),
        "enable_mmr": (
            "MMR Diversity is a post-processor that prunes near-duplicate chunks. "
            "0 unique chunks is expected — it filters for diversity, not retrieves. "
            "Check the Pipeline tab to see how many duplicates it removed."
        ),
        "enable_raptor": (
            "RAPTOR was enabled but found no hierarchical summary nodes. "
            "RAPTOR indexes need to be built during ingestion. "
            "Re-ingest with RAPTOR ON to create summary tree nodes."
        ),
        "enable_rewrite": (
            "Query Rewriting is a query-expansion step — it generates a better query form. "
            "It contributes indirectly: the rewritten query is used by Dense, BM25, and SPLADE. "
            "Check the Query tab to see the rewritten form."
        ),
        "enable_multi_query": (
            "Multi-Query generates paraphrase variants sent to the retriever in parallel. "
            "Requires enable_rewrite=true. "
            "0 unique chunks means the paraphrases found results already covered by the primary query. "
            "Check the Query tab to see the generated paraphrases."
        ),
        "enable_hyde": (
            "HyDE (Hypothetical Document Embedding) was enabled but added no unique chunks. "
            "The synthetic document embedding may have found results already covered by the primary query. "
            "Check the Query tab to see the HyDE-generated hypothesis text. "
            "Try with more abstract or open-ended queries for best HyDE gains."
        ),
        "enable_contextual_rerank": (
            "Contextual Rerank re-ranks using full-context window comparison via LLM (expensive: 1 LLM call/chunk). "
            "0 unique chunks is expected — it's a post-processor that reorders, not a retrieval source. "
            "Check the Pipeline tab to see if it changed the final ranking."
        ),
        "enable_llm_graph": (
            "LLM Graph Extraction uses the LLM to extract entities from the query at search time. "
            "0 unique chunks means: LLM offline, entity extraction returned no entities, "
            "or those entities had no matching paths in the knowledge graph."
        ),
    }
    return reasons.get(flag, (
        "Enabled but contributed 0 unique chunks to the final result set. "
        f"Candidates before: {entry.get('candidates_before', 0)}, after: {entry.get('candidates_after', 0)}. "
        "Consider checking if the index/model for this method is properly initialized."
    ))


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
        _res = response.results
        _max_s = max((r.score for r in _res), default=1.0) or 1.0
        _min_s = min((r.score for r in _res), default=0.0)
        chunks = [
            SearchResultItem(
                chunk_id=str(r.document.id),
                text=r.document.text,
                score=float(r.score),
                metadata={k: v for k, v in (r.document.metadata or {}).items() if not k.startswith("_")},
                provenance=None,
                confidence=_normalize_confidence(r.score, _min_s, _max_s, idx, len(_res)),
                method_lineage=[
                    MC(method=c["method"], rank=c["rank"], rrf_contribution=c["rrf_contribution"])
                    for c in r.document.metadata.get("_method_lineage", [])
                ],
                post_processors=r.document.metadata.get("_post_processors", []),
            )
            for idx, r in enumerate(_res)
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
            methods_used={k: bool(v) for k, v in (methods_used or {}).items()},
            query_variants=_extract_query_variants(bundle),
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
                _res = response.results
                _max_s = max((r.score for r in _res), default=1.0) or 1.0
                _min_s = min((r.score for r in _res), default=0.0)
                chunks = [
                    SearchResultItem(
                        chunk_id=str(r.document.id),
                        text=r.document.text,
                        score=float(r.score),
                        metadata={k: v for k, v in (r.document.metadata or {}).items() if not k.startswith("_")},
                        provenance=None,
                        confidence=_normalize_confidence(r.score, _min_s, _max_s, idx, len(_res)),
                        method_lineage=[
                            MC(method=c["method"], rank=c["rank"], rrf_contribution=c["rrf_contribution"])
                            for c in r.document.metadata.get("_method_lineage", [])
                        ],
                        post_processors=r.document.metadata.get("_post_processors", []),
                    )
                    for idx, r in enumerate(_res)
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

    backend_results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    results: Dict[str, BackendSearchResult] = {}
    for backend, res in zip(req.backends, backend_results_raw):
        if isinstance(res, BaseException):
            # Defensive: _run_search_with_bundle already catches all exceptions internally,
            # but return_exceptions=True ensures one rogue task never kills all backends.
            results[backend] = BackendSearchResult(backend=backend, answer="", error=str(res))
        else:
            results[backend] = res
    return SearchResponse(query=req.query, results=results)
