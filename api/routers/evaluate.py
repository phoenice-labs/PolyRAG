"""
Evaluate router: POST /api/evaluate, GET /api/evaluate/{eval_id}

IMPORTANT: static-path routes (browse-chunks, generate-qa, results) MUST be
declared BEFORE the parameterised GET /evaluate/{eval_id} route so FastAPI
does not treat the literal path segments as eval_id values.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException

from api.deps import build_pipeline_config, create_pipeline, get_eval_store
from api.schemas import EvaluateRequest, RetrievalMethods

router = APIRouter(tags=["evaluate"])


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_answer(
    answer: str,
    expected: str,
    question: str,
    expected_sources: List[str],
    chunks,
    graph_chunks=None,
) -> Dict[str, Any]:
    """Compute simple evaluation metrics.

    - faithfulness: word-overlap between expected answer and generated answer
    - relevance:    word-overlap between question and generated answer (answer on-topic?)
    - source_hit:   any expected source keyword appears in retrieved chunks
    - graph_source_hit: same check but only for chunks attributed to Knowledge Graph
    """
    answer_words = set(answer.lower().split())

    expected_words = set(expected.lower().split())
    faithfulness = len(expected_words & answer_words) / max(len(expected_words), 1)

    q_words = set(w for w in question.lower().split() if len(w) > 3)
    relevance = len(q_words & answer_words) / max(len(q_words), 1) if q_words else 0.0

    source_hit = False
    for src in expected_sources:
        for chunk in chunks:
            meta = getattr(chunk, "metadata", {}) or {}
            if src in str(meta) or src in getattr(chunk, "text", ""):
                source_hit = True
                break

    # graph_source_hit: did the Knowledge Graph surface the expected source?
    graph_source_hit = 0.0
    if graph_chunks:
        for src in expected_sources:
            for chunk in graph_chunks:
                meta = getattr(chunk, "metadata", {}) or {}
                doc = getattr(chunk, "document", None)
                text = (getattr(doc, "text", "") if doc else getattr(chunk, "text", "")) or ""
                if src in str(meta) or src in text:
                    graph_source_hit = 1.0
                    break
            if graph_source_hit:
                break

    return {
        "faithfulness":    round(min(faithfulness, 1.0), 4),
        "relevance":       round(min(relevance, 1.0), 4),
        "source_hit":      float(source_hit),
        "graph_source_hit": graph_source_hit,
    }


def _run_evaluate(
    question: str,
    expected_answer: str,
    expected_sources: List[str],
    backends: List[str],
    collection_name: str,
    methods: RetrievalMethods,
) -> Dict[str, Any]:
    """Run evaluation for a single question across all backends.

    Mirrors the search router pattern:
      Phase 1 — expand query once (LLM calls, backend-agnostic)
      Phase 2 — per-backend retrieval + answer via ask_with_bundle()
    """
    per_backend: Dict[str, Any] = {}

    # ── Phase 1: Query expansion (once, using first available backend) ────────
    first_backend = backends[0] if backends else "chromadb"
    expansion_config = build_pipeline_config(
        first_backend,
        collection_name=collection_name,
        enable_rewrite=methods.enable_rewrite,
        enable_multi_query=methods.enable_multi_query,
        enable_hyde=methods.enable_hyde,
        enable_raptor=methods.enable_raptor,
        enable_contextual_rerank=False,
    )
    try:
        expansion_pipeline = create_pipeline(expansion_config)
        bundle, expansion_traces = expansion_pipeline.expand_query(question)
    except Exception as exc:
        bundle, expansion_traces = None, []

    # ── Phase 2: Per-backend retrieval + answer ───────────────────────────────
    for backend in backends:
        try:
            config = build_pipeline_config(
                backend,
                collection_name=collection_name,
                enable_rewrite=False,
                enable_multi_query=False,
                enable_hyde=False,
                enable_raptor=methods.enable_raptor,
                enable_contextual_rerank=methods.enable_contextual_rerank,
                enable_mmr=methods.enable_mmr,
                enable_er=True,
            )
            pipeline = create_pipeline(config)

            if bundle is not None:
                resp = pipeline.ask_with_bundle(
                    question=question,
                    bundle=bundle,
                    expansion_traces=expansion_traces,
                    top_k=5,
                    enable_dense=methods.enable_dense,
                    enable_bm25=methods.enable_bm25,
                    enable_graph=methods.enable_graph,
                    enable_rerank=methods.enable_rerank,
                    enable_llm_graph=methods.enable_llm_graph,
                )
            else:
                # Fallback: plain ask() when expansion failed
                resp = pipeline.ask(question, top_k=5)

            scores = _score_answer(
                resp.answer, expected_answer, question, expected_sources, resp.results,
                graph_chunks=[
                    r for r in (resp.results or [])
                    if any(
                        (m.get("method") if isinstance(m, dict) else getattr(m, "method", "")) == "Knowledge Graph"
                        for m in (
                            r.document.metadata.get("_method_lineage", [])
                            if isinstance(r.document.metadata.get("_method_lineage"), list)
                            else []
                        )
                    )
                ],
            )
            per_backend[backend] = {
                "answer": resp.answer,
                "scores": scores,
                "graph_entities": list(resp.graph_entities or []),
                "graph_paths": [str(p) for p in (resp.graph_paths or [])],
            }
        except Exception as exc:
            per_backend[backend] = {"error": str(exc), "scores": {}}
    return per_backend


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/evaluate")
async def start_evaluate(req: EvaluateRequest) -> Dict:
    """Run evaluation and store results, return evaluation_id."""
    eval_id = str(uuid.uuid4())
    store = get_eval_store()

    tasks = [
        asyncio.to_thread(
            _run_evaluate,
            q.question,
            q.expected_answer,
            q.expected_sources,
            req.backends,
            req.collection_name,
            req.methods,
        )
        for q in req.questions
    ]
    results = await asyncio.gather(*tasks)

    eval_results = []
    for q, per_backend in zip(req.questions, results):
        eval_results.append({
            "question": q.question,
            "expected_answer": q.expected_answer,
            "expected_sources": q.expected_sources,
            "per_backend": per_backend,
        })

    store[eval_id] = {"eval_id": eval_id, "results": eval_results}
    return {"eval_id": eval_id, "question_count": len(req.questions)}


# Static-path routes BEFORE /{eval_id} — order matters in FastAPI

@router.get("/evaluate/browse-chunks")
async def browse_chunks(
    backend: str,
    collection: str,
    limit: int = 30,
    offset: int = 0,
    search: str = "",
) -> Dict:
    """Return paginated chunks from the specified backend/collection for browsing."""
    from api.deps import create_store_adapter
    try:
        adapter = create_store_adapter(backend)
        raw = await asyncio.to_thread(adapter.fetch_all, collection, limit=2000)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot fetch chunks from {backend}: {exc}")

    needle = search.strip().lower()
    if needle:
        raw = [r for r in raw if needle in (r.get("text") or "").lower()]

    total = len(raw)
    page = raw[offset: offset + limit]

    chunks_out = []
    for item in page:
        text = item.get("text") or ""
        meta = item.get("metadata") or {}
        chunks_out.append({
            "id":       item.get("id", ""),
            "text":     text,
            "preview":  text[:200] + ("\u2026" if len(text) > 200 else ""),
            "metadata": {k: v for k, v in meta.items() if k not in ("vector",)},
        })

    return {"chunks": chunks_out, "total": total, "offset": offset, "limit": limit}


@router.post("/evaluate/generate-qa")
async def generate_qa_from_chunk(body: Dict) -> Dict:
    """Use LM Studio to generate a Q&A pair from a chunk; falls back to heuristic."""
    chunk_text: str = body.get("chunk_text", "").strip()
    chunk_id: str = body.get("chunk_id", "")
    if not chunk_text:
        raise HTTPException(status_code=400, detail="chunk_text is required")

    def _generate() -> Dict[str, str]:
        try:
            from core.query.llm_client import LMStudioClient
            client = LMStudioClient(timeout=30)
            if not client.is_available():
                raise RuntimeError("LM Studio not available")

            system = (
                "You are an expert at creating high-quality evaluation questions for RAG systems. "
                "Given a passage of text, generate exactly ONE question that can be answered solely "
                "from the passage, and provide the answer in 1-3 sentences using only the passage. "
                "Reply in JSON with keys 'question' and 'answer'."
            )
            prompt = f"Passage:\n{chunk_text[:1500]}\n\nRespond with JSON only."
            raw = client.complete(prompt, system=system, max_tokens=256, trace_method="eval_generate_qa")

            import json as _json, re as _re
            match = _re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', raw, _re.DOTALL)
            if match:
                parsed = _json.loads(match.group())
                return {
                    "question": parsed.get("question", "").strip(),
                    "answer":   parsed.get("answer", "").strip(),
                    "chunk_id": chunk_id,
                    "source":   "llm",
                }
            raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")
        except Exception as exc:
            import re as _re
            sentences = _re.split(r"(?<=[.!?])\s+", chunk_text.strip())
            sentences = [s for s in sentences if len(s) > 20]
            if sentences:
                first = sentences[0].rstrip(".!?")
                q = f"What does the text say about: {first[:80]}?"
                a = " ".join(sentences[:3])
            else:
                q = "What is described in this passage?"
                a = chunk_text[:200]
            return {
                "question": q,
                "answer":   a,
                "chunk_id": chunk_id,
                "source":   "heuristic",
                "note":     str(exc),
            }

    return await asyncio.to_thread(_generate)


@router.get("/evaluate/results")
async def list_evaluations() -> Dict:
    """Return all stored evaluation IDs."""
    store = get_eval_store()
    return {"eval_ids": list(store.keys())}


# Parameterised route LAST so static paths above take priority
@router.get("/evaluate/{eval_id}")
async def get_evaluate(eval_id: str) -> Dict:
    """Return full evaluation results."""
    store = get_eval_store()
    result = store.get(eval_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return result
