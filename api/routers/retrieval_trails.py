"""
Retrieval Trails router: GET /api/retrieval-trails

Returns persistently stored retrieval trail records from data/retrieval_trails.jsonl.
Every search appends a record with the per-phase candidates_before/after pipeline steps
and method_contributions for traceability.
"""
from __future__ import annotations

import json
from collections import defaultdict
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
    - method_contributions: per-method marginal contribution stats
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


@router.get("/retrieval-trails/analysis", response_model=Dict[str, Any])
def analyse_retrieval_trails(
    backend: Optional[str] = Query(None, description="Filter by backend name"),
    min_trails: int = Query(1, ge=1, description="Minimum trail records required for analysis"),
) -> Dict[str, Any]:
    """
    Analyse all recorded retrieval trails and return per-method contribution statistics.

    Returns:
    - per_method: average contribution_pct, total_chunks, appears_in_N_trails
    - recommended: methods with avg contribution_pct > 5% across all trails
    - never_contributed: methods that were requested but contributed 0 chunks every time
    - total_trails_analysed: number of trail records included
    """
    if not _TRAIL_LOG.exists():
        return {"error": "No trails found", "total_trails_analysed": 0}

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
            if rec.get("method_contributions"):
                records.append(rec)
        except json.JSONDecodeError:
            continue

    if len(records) < min_trails:
        return {
            "error": f"Only {len(records)} trails with contribution data (need {min_trails})",
            "total_trails_analysed": len(records),
        }

    # Aggregate per-method stats
    method_contribution_pcts: dict[str, list[float]] = defaultdict(list)
    method_chunk_totals: dict[str, int] = defaultdict(int)

    for rec in records:
        contribs = rec.get("method_contributions", {})
        for method, stats in contribs.items():
            pct = stats.get("contribution_pct", 0.0)
            chunks = stats.get("chunks_contributed", 0)
            method_contribution_pcts[method].append(pct)
            method_chunk_totals[method] += chunks

    per_method: dict[str, Any] = {}
    for method, pcts in method_contribution_pcts.items():
        avg_pct = round(sum(pcts) / len(pcts), 2)
        per_method[method] = {
            "avg_contribution_pct": avg_pct,
            "total_chunks_contributed": method_chunk_totals[method],
            "appears_in_n_trails": len(pcts),
        }

    recommended = [m for m, s in per_method.items() if s["avg_contribution_pct"] > 5.0]
    never_contributed = [m for m, s in per_method.items() if s["total_chunks_contributed"] == 0]

    return {
        "total_trails_analysed": len(records),
        "per_method": per_method,
        "recommended": sorted(recommended, key=lambda m: -per_method[m]["avg_contribution_pct"]),
        "never_contributed": never_contributed,
        "interpretation": (
            "Methods in 'recommended' contributed >5% of result chunks on average. "
            "Enable these for best coverage. Methods in 'never_contributed' add overhead "
            "without improving results for this dataset."
        ),
    }
