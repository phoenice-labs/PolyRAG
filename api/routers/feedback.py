"""
Feedback router: POST /api/feedback, GET /api/feedback
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter

from api.deps import get_feedback_store
from api.schemas import FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """Store a relevance feedback entry."""
    store = get_feedback_store()
    entry: Dict[str, Any] = {
        "query": req.query,
        "chunk_id": req.chunk_id,
        "backend": req.backend,
        "collection_name": req.collection_name,
        "relevant": req.relevant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    store.append(entry)
    return {"status": "stored", "index": len(store) - 1}


@router.get("/feedback")
async def get_feedback() -> Dict[str, Any]:
    """Return all feedback entries."""
    store = get_feedback_store()
    return {"count": len(store), "entries": store}
