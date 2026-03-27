"""
Graph router: GET /api/graph/{collection}
             POST /api/graph/{collection}/enhance  — LLM-enhanced ER background job
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.deps import build_pipeline_config, create_pipeline, get_job_store
from api.schemas import GraphChunkRef, GraphEdge, GraphNode, GraphNodeRelation, GraphResponse

router = APIRouter(tags=["graph"])

# Graph snapshots are persisted here by pipeline._save_graph_snapshot()
_GRAPH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "graphs"


def _get_graph_data(collection: str) -> GraphResponse:
    """Load graph from persisted JSON snapshot (written during ingest with ER enabled)."""
    snapshot_path = _GRAPH_DIR / f"{collection}.json"

    if not snapshot_path.exists():
        # No snapshot yet — return empty (ingest with ER enabled first)
        return GraphResponse(collection=collection, nodes=[], edges=[])

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load graph snapshot: {exc}")

    nodes: List[GraphNode] = []

    # Build id→label map for resolving relation targets
    id_to_label: dict = {str(n["id"]): str(n.get("label", n["id"])) for n in data.get("nodes", [])}

    # Build outgoing relations per node from the edges list
    node_relations: dict = {}
    for e in data.get("edges", []):
        src = str(e["source"])
        tgt = str(e["target"])
        if src == tgt:
            continue   # skip self-loops
        node_relations.setdefault(src, []).append({
            "target_id": tgt,
            "target_label": id_to_label.get(tgt, tgt),
            "relation": str(e.get("relation", "related")),
            "weight": float(e.get("weight", 1.0)),
        })

    for n in data.get("nodes", []):
        chunk_refs = [
            GraphChunkRef(chunk_id=c["chunk_id"], snippet=c.get("snippet", ""))
            for c in n.get("chunks", [])
        ]
        rel_refs = [
            GraphNodeRelation(**r) for r in node_relations.get(str(n["id"]), [])
        ]
        nodes.append(GraphNode(
            id=str(n["id"]),
            label=str(n.get("label", n.get("text", n["id"]))),
            type=str(n.get("type", n.get("entity_type", "OTHER"))),
            frequency=int(n.get("frequency", 1)),
            chunks=chunk_refs,
            relations=rel_refs,
        ))

    edges: List[GraphEdge] = []
    for e in data.get("edges", []):
        edges.append(GraphEdge(
            source=str(e["source"]),
            target=str(e["target"]),
            relation=str(e.get("relation", "related")),
            weight=float(e.get("weight", 1.0)),
            doc_ids=[],
        ))

    return GraphResponse(collection=collection, nodes=nodes, edges=edges)


@router.get("/graph", response_model=List[str])
async def list_graphs() -> List[str]:
    """Return collection names that have a persisted graph snapshot."""
    if not _GRAPH_DIR.exists():
        return []
    return sorted(p.stem for p in _GRAPH_DIR.glob("*.json"))


@router.delete("/graph/{collection}")
async def delete_graph(collection: str) -> Dict:
    """
    Delete the persisted graph snapshot for a collection.

    Called automatically when a collection is deleted from all vector backends,
    or explicitly via the Document Library UI 'Clear Graph' button.
    Returns {deleted: true} if the file existed, {deleted: false} if it was already gone.
    """
    snapshot_path = _GRAPH_DIR / f"{collection}.json"
    if snapshot_path.exists():
        try:
            snapshot_path.unlink()
            return {"deleted": True, "collection": collection}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete graph snapshot: {exc}")
    return {"deleted": False, "collection": collection}


@router.delete("/graph")
async def delete_all_graphs() -> Dict:
    """
    Delete ALL persisted graph snapshots (all collections).

    Used by the Document Library 'Clear All Graphs' button.
    Returns the list of collections whose graphs were deleted.
    """
    if not _GRAPH_DIR.exists():
        return {"deleted": [], "count": 0}
    deleted = []
    errors = []
    for snap in _GRAPH_DIR.glob("*.json"):
        try:
            snap.unlink()
            deleted.append(snap.stem)
        except Exception as exc:
            errors.append(f"{snap.stem}: {exc}")
    result: Dict = {"deleted": deleted, "count": len(deleted)}
    if errors:
        result["errors"] = errors
    return result


class EnhanceStatus(BaseModel):
    collection: str
    graph_exists: bool
    node_count: int
    edge_count: int
    llm_enhanced: bool
    llm_enhanced_at: Optional[str]


@router.get("/graph/{collection}/enhance-status", response_model=EnhanceStatus)
async def get_enhance_status(collection: str) -> EnhanceStatus:
    """Return whether LLM graph enhancement has been run for this collection."""
    snapshot_path = _GRAPH_DIR / f"{collection}.json"
    if not snapshot_path.exists():
        return EnhanceStatus(
            collection=collection, graph_exists=False,
            node_count=0, edge_count=0,
            llm_enhanced=False, llm_enhanced_at=None,
        )
    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return EnhanceStatus(
        collection=collection,
        graph_exists=True,
        node_count=len(data.get("nodes", [])),
        edge_count=len(data.get("edges", [])),
        llm_enhanced=bool(data.get("llm_enhanced", False)),
        llm_enhanced_at=data.get("llm_enhanced_at"),
    )


@router.get("/graph/{collection}", response_model=GraphResponse)
async def get_graph(collection: str) -> GraphResponse:
    """Return entity nodes and relation edges for D3 visualization."""
    return await asyncio.to_thread(_get_graph_data, collection)


# ── LLM Graph Enhancement ─────────────────────────────────────────────────────

class EnhanceGraphRequest(BaseModel):
    backend: str = "milvus"
    max_chunks: int = 500


async def _run_enhance_graph(
    job_id: str,
    collection: str,
    backend: str,
    max_chunks: int,
) -> None:
    """Background task: run LLM-based ER extraction on all chunks and update graph snapshot."""
    from api.deps import get_job_store
    from core.graph.llm_extractor import LLMEntityExtractor

    store = get_job_store()
    await store.update_job(job_id, status="running")
    await store.append_log(job_id, f"[enhance] Starting LLM graph enhancement for '{collection}' on {backend}")

    try:
        # Build a pipeline config with ER enabled (to get graph store + LLM client)
        config = build_pipeline_config(
            backend=backend,
            collection_name=collection,
            enable_er=True,
        )
        # Force LLM extraction flags on
        config["graph"]["llm_extraction"]["enabled"] = True

        t0 = time.perf_counter()
        await store.append_log(job_id, f"[enhance] Loading pipeline...")
        pipeline = await asyncio.to_thread(create_pipeline, config)

        # Check LLM availability
        if not pipeline._llm_client or not pipeline._llm_client.is_available():
            await store.append_log(job_id, "[enhance] ⚠ LLM (LM Studio) is not reachable. Start LM Studio and retry.")
            await store.update_job(job_id, status="error", error="LLM not available")
            return

        # Fetch chunks from the vector store
        await store.append_log(job_id, f"[enhance] Fetching up to {max_chunks} chunks from '{collection}'...")
        try:
            raw_chunks = await asyncio.to_thread(
                lambda: pipeline.store.fetch_all(collection, limit=max_chunks)
            )
        except Exception as exc:
            await store.append_log(job_id, f"[enhance] ERROR fetching chunks: {exc}")
            await store.update_job(job_id, status="error", error=str(exc))
            return

        if not raw_chunks:
            await store.append_log(job_id, "[enhance] No chunks found. Ingest the document first.")
            await store.update_job(job_id, status="error", error="No chunks found")
            return

        await store.append_log(job_id, f"[enhance] Processing {len(raw_chunks)} chunks with LLM...")

        extractor = LLMEntityExtractor(pipeline._llm_client, max_chunk_chars=2000)
        graph_store = pipeline._graph_store
        new_entities = 0
        new_relations = 0

        for i, row in enumerate(raw_chunks):
            chunk_id = str(row.get("id", f"chunk_{i}"))
            text = str(row.get("text", ""))
            if not text.strip():
                continue

            result = await asyncio.to_thread(extractor.extract, text, chunk_id)
            for entity in result.entities:
                graph_store.upsert_entity(entity)
                graph_store.link_entity_to_chunk(entity.entity_id, chunk_id, text[:140])
                new_entities += 1
            for triple in result.triples:
                graph_store.upsert_entity(triple.subject)
                graph_store.upsert_entity(triple.object)
                graph_store.upsert_relation(triple.to_relation())
                new_relations += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(raw_chunks):
                await store.append_log(
                    job_id,
                    f"[enhance] {i+1}/{len(raw_chunks)} chunks — +{new_entities} entities, +{new_relations} relations"
                )

        # Save updated graph snapshot and stamp llm_enhanced flag
        await asyncio.to_thread(pipeline._save_graph_snapshot, collection)
        snapshot_path = _GRAPH_DIR / f"{collection}.json"
        if snapshot_path.exists():
            try:
                snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                snap["llm_enhanced"] = True
                snap["llm_enhanced_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                snapshot_path.write_text(json.dumps(snap), encoding="utf-8")
            except Exception:
                pass  # non-fatal: status endpoint will show llm_enhanced=False until next run
        elapsed = time.perf_counter() - t0
        await store.append_log(
            job_id,
            f"[enhance] ✅ Done in {elapsed:.1f}s — added {new_entities} entities, {new_relations} relations"
        )
        await store.update_job(
            job_id,
            status="done",
            result={"new_entities": new_entities, "new_relations": new_relations, "elapsed_s": round(elapsed, 1)},
        )

    except Exception as exc:
        import traceback
        tb_lines = traceback.format_exc().splitlines()
        await store.append_log(job_id, f"[enhance] ERROR: {exc}")
        for line in tb_lines[-10:]:
            await store.append_log(job_id, f"[enhance] TB: {line}")
        await store.update_job(job_id, status="error", error=str(exc))


@router.post("/graph/{collection}/enhance")
async def enhance_graph(
    collection: str,
    req: EnhanceGraphRequest,
    background_tasks: BackgroundTasks,
) -> Dict:
    """Start a background LLM enhancement job for the knowledge graph of a collection."""
    store = get_job_store()
    job = await store.create_job(backend=req.backend, corpus_path=None, config={})
    background_tasks.add_task(_run_enhance_graph, job.id, collection, req.backend, req.max_chunks)
    return {"job_id": job.id}


@router.get("/graph/{collection}/enhance/{job_id}/stream")
async def stream_enhance(collection: str, job_id: str) -> EventSourceResponse:
    """SSE stream for LLM graph enhancement job progress."""
    store = get_job_store()

    async def _gen():
        sent = 0
        while True:
            job = await store.get_job(job_id)
            if job is None:
                yield {"data": "ERROR: job not found"}
                return
            while sent < len(job.log_lines):
                yield {"data": job.log_lines[sent]}
                sent += 1
            if job.status in ("done", "error"):
                yield {"data": f"STATUS:{job.status}"}
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(_gen())
