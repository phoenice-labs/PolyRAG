"""
Retrieval Trails router: GET /api/retrieval-trails

Returns persistently stored retrieval trail records from data/retrieval_trails.jsonl.
Every search appends a record with the per-phase candidates_before/after pipeline steps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

router = APIRouter(tags=["retrieval-trails"])

_TRAIL_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "retrieval_trails.jsonl"


@router.get("/retrieval-trails", response_model=List[Dict[str, Any]])
def list_retrieval_trails(
    limit: int = Query(100, ge=1, le=5000, description="Max records to return (newest first)"),
    backend: Optional[str] = Query(None, description="Filter by backend name"),
) -> List[Dict[str, Any]]:
    """
    Return persisted retrieval trail records from data/retrieval_trails.jsonl.

    Records are ordered newest-first. Each record contains:
    - timestamp: ISO-8601 UTC
    - query: the search query text
    - backend: vector store backend used
    - methods_used: dict of enabled/disabled retrieval method flags
    - retrieval_trace: list of {method, candidates_before, candidates_after}
    - result_count: number of final results returned
    - latency_ms: total retrieval latency
    """
    if not _TRAIL_LOG.exists():
        return []

    lines = _TRAIL_LOG.read_text(encoding="utf-8").splitlines()
    records: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if backend and rec.get("backend") != backend:
                continue
            records.append(rec)
        except json.JSONDecodeError:
            continue

    return records[-limit:][::-1]


@router.delete("/retrieval-trails", response_model=Dict[str, str])
def clear_retrieval_trails() -> Dict[str, str]:
    """Clear the persisted retrieval trail log (data/retrieval_trails.jsonl)."""
    if _TRAIL_LOG.exists():
        _TRAIL_LOG.write_text("", encoding="utf-8")
    return {"status": "cleared", "path": str(_TRAIL_LOG)}
