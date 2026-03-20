"""
LLM Trace History router: GET /api/traces

Returns persisted LLM call records from data/llm_traces.jsonl.
Every search that triggers an LLM call (Query Rewrite, HyDE, Multi-Query,
Contextual Re-rank, Answer Generation) is appended to this file in real-time.
The file survives server restarts so you can audit the full history.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

router = APIRouter(tags=["traces"])

_TRACE_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "llm_traces.jsonl"


@router.get("/traces", response_model=List[Dict[str, Any]])
def list_llm_traces(
    limit: int = Query(100, ge=1, le=5000, description="Max records to return (newest first)"),
    method: Optional[str] = Query(None, description="Filter by retrieval method name"),
) -> List[Dict[str, Any]]:
    """
    Return persisted LLM call history from data/llm_traces.jsonl.

    Records are ordered newest-first. Each record contains:
    - timestamp: ISO-8601 UTC
    - method: retrieval method that triggered the call
    - system_prompt / user_message / response: full prompt + LLM output
    - latency_ms: round-trip time
    """
    if not _TRACE_LOG.exists():
        return []

    lines = _TRACE_LOG.read_text(encoding="utf-8").splitlines()
    records: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if method and rec.get("method") != method:
                continue
            records.append(rec)
        except json.JSONDecodeError:
            continue

    # Return newest first, capped at limit
    return records[-limit:][::-1]


@router.delete("/traces", response_model=Dict[str, str])
def clear_llm_traces() -> Dict[str, str]:
    """Clear the persisted LLM trace log (data/llm_traces.jsonl)."""
    if _TRACE_LOG.exists():
        _TRACE_LOG.write_text("", encoding="utf-8")
    return {"status": "cleared", "path": str(_TRACE_LOG)}
