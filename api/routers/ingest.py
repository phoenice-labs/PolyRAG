"""
Ingest router: POST /api/ingest, GET /api/ingest/{job_id}/stream, GET /api/ingest/{job_id}/status
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.deps import build_pipeline_config, create_pipeline, get_job_store, _pipeline_cache, _pipeline_cache_lock
from api.jobs import JobStore
from api.schemas import IngestRequest, JobStatus

router = APIRouter(tags=["ingest"])


def _is_closed_channel(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "closed channel" in msg or "channel is closed" in msg


async def _run_ingest(job_id: str, text: str, config: dict, store: JobStore) -> None:
    """Background task: run pipeline ingest with granular progress logging."""
    backend = config["store"]["backend"]
    await store.update_job(job_id, status="running")
    await store.append_log(job_id, f"[{backend}] Reading corpus: {len(text):,} characters")

    try:
        # ── Step 1: Start pipeline ───────────────────────────────────────────
        await store.append_log(job_id, f"[{backend}] Starting pipeline (loading embedder)...")
        t0 = time.perf_counter()
        pipeline = await asyncio.to_thread(create_pipeline, config)
        await store.append_log(job_id, f"[{backend}] Pipeline ready in {time.perf_counter()-t0:.1f}s")

        # ── Step 2: Chunking preview ─────────────────────────────────────────
        await store.append_log(job_id, f"[{backend}] Chunking text (strategy: {config['ingestion'].get('chunk_strategy','sentence')}, size: {config['ingestion']['chunk_size']})...")

        # ── Step 3: Run ingest (chunking + embedding + upsert) ───────────────
        await store.append_log(job_id, f"[{backend}] Embedding & upserting chunks — this may take 1-3 min for large files...")

        # Heartbeat: emit a '.' log every 15s while ingest_text() runs in a thread
        async def _run_ingest_task():
            ingest_task = asyncio.create_task(
                asyncio.to_thread(pipeline.ingest_text, text, {"source": "api", "backend": backend})
            )
            heartbeat_count = 0
            while not ingest_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(ingest_task), timeout=15.0)
                except asyncio.TimeoutError:
                    heartbeat_count += 1
                    elapsed = heartbeat_count * 15
                    await store.append_log(job_id, f"[{backend}] Still processing... ({elapsed}s elapsed, embedding in progress)")
            return ingest_task.result()

        try:
            result = await _run_ingest_task()
        except Exception as first_exc:
            if _is_closed_channel(first_exc):
                # Stale Milvus connection in cached pipeline — evict and retry once
                await store.append_log(job_id, f"[{backend}] ⚠ Milvus closed channel — reconnecting and retrying...")
                cache_key = (backend, config.get("ingestion", {}).get("collection_name", "api_ingest"))
                with _pipeline_cache_lock:
                    stale = _pipeline_cache.pop(cache_key, None)
                if stale:
                    try:
                        stale.stop()
                    except Exception:
                        pass
                pipeline = await asyncio.to_thread(create_pipeline, config)
                result = await _run_ingest_task()
            else:
                raise

        await store.append_log(job_id, f"[{backend}] Chunks upserted: {result.upserted} / {result.total_chunks}")

        # ── Step 4: Done ─────────────────────────────────────────────────────
        total_time = time.perf_counter() - t0
        await store.append_log(job_id, f"[{backend}] Ingestion complete in {total_time:.1f}s — {result.upserted} chunks stored")
        await store.update_job(
            job_id,
            status="done",
            result={"upserted": result.upserted, "total_chunks": result.total_chunks, "elapsed_s": round(total_time, 1)},
        )
        # STATUS:done is emitted by the SSE generator when it sees job.status == "done"

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        await store.append_log(job_id, f"[{backend}] ERROR: {exc}")
        # Log condensed traceback (last 20 lines) to help diagnose deep errors
        tb_lines = tb.splitlines()
        for line in tb_lines[-20:]:
            await store.append_log(job_id, f"[{backend}] TB: {line}")
        await store.update_job(job_id, status="error", error=str(exc))
        # STATUS:error is emitted by the SSE generator when it sees job.status == "error"


@router.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> Dict:
    """Accept ingest request, create jobs per backend, start background tasks."""
    if not req.text and not req.corpus_path:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'corpus_path'")

    text = req.text
    if not text and req.corpus_path:
        try:
            from core.ingestion.loader import load_document
            enable_rich = True  # honoured per config; extractor raises clearly if lib missing
            text = load_document(req.corpus_path, enable_rich_formats=enable_rich)
        except (OSError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read corpus_path: {exc}")
        except (ImportError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    store = get_job_store()

    # ── Collection lock check: reject if another job is already ingesting this collection ──
    running = await store.get_running_jobs_for_collection(req.collection_name)
    if running:
        conflicting = running[0]
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Collection is currently being ingested — concurrent ingestion may corrupt the index",
                "collection": req.collection_name,
                "blocking_job_id": conflicting.id,
                "blocking_backend": conflicting.backend,
                "blocking_status": conflicting.status,
            },
        )

    job_ids: Dict[str, str] = {}

    for backend in req.backends:
        config = build_pipeline_config(
            backend=backend,
            collection_name=req.collection_name,
            chunk_size=req.chunk_size,
            chunk_strategy=req.chunk_strategy,
            overlap=req.overlap,
            enable_er=req.enable_er,
            enable_splade=req.enable_splade,
            embedding_model=req.embedding_model,
        )
        # Store the scoped collection name so lock checks work correctly
        scoped_collection = config.get("ingestion", {}).get("collection_name", req.collection_name)
        job = await store.create_job(
            backend=backend,
            corpus_path=req.corpus_path,
            config=config,
            collection_name=scoped_collection,
        )
        job_ids[backend] = job.id
        background_tasks.add_task(_run_ingest, job.id, text, config, store)

    return {"job_ids": job_ids}


@router.get("/ingest/jobs")
async def list_ingest_jobs() -> list:
    """Return all ingest jobs (used by nav badge and multi-user lock awareness)."""
    store = get_job_store()
    jobs = await store.list_jobs()
    return [
        {
            "id": j.id,
            "status": j.status,
            "backend": j.backend,
            "collection_name": j.collection_name,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
        }
        for j in jobs
    ]


@router.get("/ingest/{job_id}/stream")
async def stream_ingest(job_id: str) -> EventSourceResponse:
    """SSE endpoint: stream log lines as they appear."""
    store = get_job_store()

    async def _generator():
        sent = 0
        while True:
            job = await store.get_job(job_id)
            if job is None:
                yield {"data": "ERROR: job not found"}
                return
            # Send any new log lines
            while sent < len(job.log_lines):
                yield {"data": job.log_lines[sent]}
                sent += 1
            if job.status in ("done", "error"):
                yield {"data": f"STATUS:{job.status}"}
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(_generator())


@router.get("/ingest/{job_id}/status", response_model=JobStatus)
async def ingest_status(job_id: str) -> JobStatus:
    """Return current status of an ingest job."""
    store = get_job_store()
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job.id,
        status=job.status,
        backend=job.backend,
        created_at=job.created_at,
        updated_at=job.updated_at,
        log_lines=job.log_lines,
        result=job.result,
        error=job.error,
    )
