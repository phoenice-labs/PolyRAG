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
# Kuzu embedded graph database path (shared across all collections)
_KUZU_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "graph.kuzu"


def _clear_kuzu_store() -> None:
    """
    Clear ALL entities and relations from the shared Kuzu graph database.

    Kuzu is an embedded persistent graph DB stored at data/graph.kuzu. It is
    shared across all collections (no per-collection isolation). Clearing it
    removes all entity/relation data; JSON snapshots are independent and serve
    as the graph router's source of truth.

    This is a synchronous function — call via asyncio.to_thread() from async routes.
    Silently no-ops if kuzu is not installed or the DB does not exist yet.
    """
    if not _KUZU_DB_PATH.exists():
        return
    try:
        import kuzu  # type: ignore[import]
    except ImportError:
        return  # kuzu not installed — graph backend is NetworkX (no persistent store)

    try:
        db = kuzu.Database(str(_KUZU_DB_PATH))
        conn = kuzu.Connection(db)
        for stmt in [
            "MATCH (e:Entity)-[r:APPEARS_IN]->(c:Chunk) DELETE r",
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) DELETE r",
            "MATCH (e:Entity) DELETE e",
            "MATCH (c:Chunk) DELETE c",
        ]:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # table may not exist yet — ignore
        conn = None
        db = None
    except Exception:
        pass  # DB may be locked or corrupt — best-effort


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
    Delete the persisted graph snapshot for a collection AND clear the shared Kuzu DB.

    The Kuzu embedded graph database (`data/graph.kuzu`) is global — all collections
    share a single Kuzu instance with no per-collection isolation. Clearing one
    collection's graph therefore clears the Kuzu DB for all collections. JSON
    snapshots for other collections remain on disk and will be restored from those
    files on next pipeline startup.

    Also evicts all cached pipelines for this collection so the next request gets
    a fresh pipeline without stale in-memory graph data.
    """
    from api.deps import evict_all_pipelines_for_collection

    snapshot_path = _GRAPH_DIR / f"{collection}.json"
    deleted_json = False
    if snapshot_path.exists():
        try:
            snapshot_path.unlink()
            deleted_json = True
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete graph snapshot: {exc}")

    # Clear the shared Kuzu DB (best-effort — non-fatal if kuzu not installed)
    cleared_kuzu = False
    try:
        await asyncio.to_thread(_clear_kuzu_store)
        cleared_kuzu = True
    except Exception:
        pass  # kuzu not installed or already empty — not fatal

    # Evict cached pipelines so stale in-memory graph is discarded
    evicted = evict_all_pipelines_for_collection(collection)

    return {
        "deleted": deleted_json,
        "collection": collection,
        "kuzu_cleared": cleared_kuzu,
        "pipelines_evicted": evicted,
    }


@router.delete("/graph")
async def delete_all_graphs() -> Dict:
    """
    Delete ALL persisted graph snapshots AND clear the shared Kuzu DB.

    Used by the Document Library 'Clear All Graphs' button and the Graph page.
    Returns the list of collections whose JSON snapshots were deleted.
    """
    from api.deps import evict_all_pipelines

    if not _GRAPH_DIR.exists():
        deleted: list = []
    else:
        deleted = []
        errors = []
        for snap in _GRAPH_DIR.glob("*.json"):
            try:
                snap.unlink()
                deleted.append(snap.stem)
            except Exception as exc:
                errors.append(f"{snap.stem}: {exc}")

    # Clear the shared Kuzu DB
    cleared_kuzu = False
    try:
        await asyncio.to_thread(_clear_kuzu_store)
        cleared_kuzu = True
    except Exception:
        pass

    # Evict the entire pipeline cache
    evicted = evict_all_pipelines()

    result: Dict = {
        "deleted": deleted,
        "count": len(deleted),
        "kuzu_cleared": cleared_kuzu,
        "pipelines_evicted": evicted,
    }
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
    """Background task: build or enhance the knowledge graph for a collection.

    Strategy:
      1. If LLM (LM Studio) is reachable → use LLMEntityExtractor for rich extraction.
      2. If LLM is unavailable → fall back to spaCy EntityRelationExtractor (same as
         ingestion-time ER).  This means Enhance Graph works even without LM Studio.
    Both paths upsert idempotently, so running on an existing graph just adds/updates
    entities rather than overwriting — 'create if absent, enhance if present'.
    """
    from api.deps import get_job_store
    from core.graph.llm_extractor import LLMEntityExtractor
    from core.graph.extractor import EntityRelationExtractor

    store = get_job_store()
    await store.update_job(job_id, status="running")
    await store.append_log(job_id, f"[enhance] Starting graph build/enhance for '{collection}' on {backend}")

    try:
        # Build a pipeline config with ER enabled (to get graph store + LLM client)
        config = build_pipeline_config(
            backend=backend,
            collection_name=collection,
            enable_er=True,
        )
        config["graph"]["llm_extraction"]["enabled"] = True

        t0 = time.perf_counter()
        await store.append_log(job_id, "[enhance] Loading pipeline...")
        pipeline = await asyncio.to_thread(create_pipeline, config)

        # Decide extraction mode: LLM (rich) or spaCy fallback
        llm_available = pipeline._llm_client and pipeline._llm_client.is_available()
        if llm_available:
            extraction_mode = "LLM (rich entity + relation extraction)"
            extractor_obj = LLMEntityExtractor(pipeline._llm_client, max_chunk_chars=2000)
            use_llm = True
        else:
            extraction_mode = "spaCy NER (LM Studio not reachable — using fast NER fallback)"
            extractor_obj = EntityRelationExtractor()
            use_llm = False
        await store.append_log(job_id, f"[enhance] Extraction mode: {extraction_mode}")

        # Check whether the graph already has content
        graph_store = pipeline._graph_store
        existing_entities = graph_store.entity_count() if graph_store else 0
        if existing_entities > 0:
            await store.append_log(
                job_id,
                f"[enhance] Existing graph detected ({existing_entities} entities) — will enhance in-place"
            )
        else:
            await store.append_log(job_id, "[enhance] No existing graph — building from scratch")

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
            await store.append_log(
                job_id,
                "[enhance] No chunks found in this collection. Ingest documents first, then run Enhance Graph."
            )
            await store.update_job(job_id, status="error", error="No chunks found")
            return

        await store.append_log(job_id, f"[enhance] Processing {len(raw_chunks)} chunks...")

        new_entities = 0
        new_relations = 0

        for i, row in enumerate(raw_chunks):
            chunk_id = str(row.get("id", f"chunk_{i}"))
            text = str(row.get("text", ""))
            if not text.strip():
                continue

            if use_llm:
                result = await asyncio.to_thread(extractor_obj.extract, text, chunk_id)
            else:
                # spaCy extractor is synchronous and fast — run in thread to avoid blocking
                result = await asyncio.to_thread(extractor_obj.extract, text, chunk_id)

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

        # Save updated graph snapshot; stamp mode + enhanced flag
        await asyncio.to_thread(pipeline._save_graph_snapshot, collection)
        snapshot_path = _GRAPH_DIR / f"{collection}.json"
        if snapshot_path.exists():
            try:
                snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                snap["llm_enhanced"] = llm_available
                snap["llm_enhanced_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                snap["extraction_mode"] = "llm" if llm_available else "spacy"
                snapshot_path.write_text(json.dumps(snap), encoding="utf-8")
            except Exception:
                pass  # non-fatal
        elapsed = time.perf_counter() - t0
        mode_tag = "LLM" if llm_available else "spaCy"
        await store.append_log(
            job_id,
            f"[enhance] Done in {elapsed:.1f}s via {mode_tag} — "
            f"+{new_entities} entities, +{new_relations} relations "
            f"(total in graph: {graph_store.entity_count()} entities)"
        )
        await store.update_job(
            job_id,
            status="done",
            result={
                "new_entities": new_entities,
                "new_relations": new_relations,
                "elapsed_s": round(elapsed, 1),
                "extraction_mode": "llm" if llm_available else "spacy",
            },
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
