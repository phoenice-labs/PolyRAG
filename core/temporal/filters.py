"""
Phase 8: Temporal relevance, versioning, and data classification.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from core.store.models import SearchResult


# ── Temporal ─────────────────────────────────────────────────────────────────

class TemporalMetadata(BaseModel):
    """Lifecycle metadata attached to a document or chunk."""

    created_at: Optional[str] = None          # ISO 8601
    effective_date: Optional[str] = None       # When the content became active
    expiry_date: Optional[str] = None          # When content expires
    version_tag: Optional[str] = None
    superseded_by: Optional[str] = None        # doc_id of superseding document
    change_summary: Optional[str] = None


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        # Ensure timezone-aware for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class TemporalFilter:
    """
    Excludes expired or superseded chunks at query time.

    Parameters
    ----------
    as_of : the reference datetime (default: now UTC)
    """

    def __init__(self, as_of: Optional[datetime] = None) -> None:
        self.as_of = as_of or datetime.now(timezone.utc)

    def is_active(self, result: SearchResult) -> bool:
        meta = result.document.metadata

        # Expired?
        expiry = _parse_iso(meta.get("expiry_date"))
        if expiry and expiry < self.as_of:
            return False

        # Superseded?
        if meta.get("superseded_by"):
            return False

        # Not yet effective?
        effective = _parse_iso(meta.get("effective_date"))
        if effective and effective > self.as_of:
            return False

        return True

    def filter(self, results: List[SearchResult]) -> List[SearchResult]:
        active = [r for r in results if self.is_active(r)]
        for i, r in enumerate(active, start=1):
            r.rank = i
        return active


class TemporalRanker:
    """
    Boosts recency for time-sensitive queries.
    Adds a recency bonus to the retrieval score.

    Parameters
    ----------
    recency_weight : weight of the recency bonus (0.0 = disabled, 1.0 = max)
    decay_days     : half-life in days for exponential decay (default: 365)
    """

    def __init__(self, recency_weight: float = 0.15, decay_days: int = 365) -> None:
        self.recency_weight = recency_weight
        self.decay_days = decay_days

    def rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        import math
        now = datetime.now(timezone.utc)

        def recency_score(result: SearchResult) -> float:
            created = _parse_iso(result.document.metadata.get("created_at"))
            if not created:
                return 0.5  # neutral
            age_days = (now - created).days
            return math.exp(-age_days / self.decay_days)

        boosted = []
        for r in results:
            bonus = recency_score(r) * self.recency_weight
            new_score = min(1.0, r.score * (1 - self.recency_weight) + bonus)
            boosted.append(SearchResult(document=r.document, score=new_score, rank=r.rank))

        boosted.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(boosted, start=1):
            r.rank = i
        return boosted


# ── Data Classification ───────────────────────────────────────────────────────

class ClassificationLabel(str, Enum):
    PUBLIC       = "PUBLIC"
    INTERNAL     = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED   = "RESTRICTED"
    UNCLASSIFIED = "UNCLASSIFIED"

    @classmethod
    def rank(cls, label: str) -> int:
        """Higher rank = more sensitive."""
        order = {
            cls.PUBLIC: 0,
            cls.UNCLASSIFIED: 0,
            cls.INTERNAL: 1,
            cls.CONFIDENTIAL: 2,
            cls.RESTRICTED: 3,
        }
        try:
            return order[cls(label)]
        except (ValueError, KeyError):
            return 0


class ClassificationFilter:
    """
    Enforces data classification at retrieval time.
    Users may only see content at or below their clearance level.

    Parameters
    ----------
    user_clearance : the highest label the user is allowed to access
    """

    def __init__(self, user_clearance: str = "INTERNAL") -> None:
        self.clearance_rank = ClassificationLabel.rank(user_clearance)

    def is_accessible(self, result: SearchResult) -> bool:
        label = result.document.metadata.get("classification", "UNCLASSIFIED")
        return ClassificationLabel.rank(label) <= self.clearance_rank

    def filter(self, results: List[SearchResult]) -> List[SearchResult]:
        accessible = [r for r in results if self.is_accessible(r)]
        for i, r in enumerate(accessible, start=1):
            r.rank = i
        return accessible


class ClassificationPropagator:
    """
    Ensures child chunks inherit the classification of their parent document.
    Call this after chunking, before upsert.
    """

    @staticmethod
    def propagate(chunks: list, doc_classification: str) -> list:
        """Set classification on all chunks that don't already have one."""
        for chunk in chunks:
            if not chunk.metadata.get("classification"):
                chunk.metadata["classification"] = doc_classification
        return chunks


class AccessPolicyEvaluator:
    """
    Pluggable RBAC/ABAC policy engine.
    Accepts a callable policy function: (user_context, result) → bool.
    Default policy: classification-based access only.
    """

    def __init__(
        self,
        policy_fn: Optional[Callable[[dict, SearchResult], bool]] = None,
        user_clearance: str = "INTERNAL",
    ) -> None:
        self._clf_filter = ClassificationFilter(user_clearance)
        self._policy_fn = policy_fn

    def allows(self, user_context: dict, result: SearchResult) -> bool:
        # Base classification check
        if not self._clf_filter.is_accessible(result):
            return False
        # Additional custom policy
        if self._policy_fn:
            return self._policy_fn(user_context, result)
        return True

    def filter(
        self,
        results: List[SearchResult],
        user_context: dict,
    ) -> List[SearchResult]:
        allowed = [r for r in results if self.allows(user_context, r)]
        for i, r in enumerate(allowed, start=1):
            r.rank = i
        return allowed
