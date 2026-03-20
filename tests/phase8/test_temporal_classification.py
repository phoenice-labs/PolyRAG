"""
Phase 8 – Temporal Relevance & Data Classification Tests
==========================================================
Run:  pytest tests/phase8/ -v
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from core.store.models import Document, SearchResult
from core.temporal.filters import (
    TemporalFilter,
    TemporalRanker,
    ClassificationLabel,
    ClassificationFilter,
    ClassificationPropagator,
    AccessPolicyEvaluator,
)


def _res(i, score=0.8, rank=None, **meta):
    return SearchResult(
        document=Document(id=f"d{i}", text=f"text {i}", embedding=[], metadata=meta),
        score=score, rank=rank or i+1,
    )


# ── Temporal Filter ───────────────────────────────────────────────────────────

class TestTemporalFilter:
    def test_active_result_passes(self):
        tf = TemporalFilter()
        result = _res(0, effective_date="2020-01-01")
        assert tf.is_active(result) is True

    def test_expired_result_filtered(self):
        tf = TemporalFilter()
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        result = _res(0, expiry_date=past)
        assert tf.is_active(result) is False

    def test_superseded_result_filtered(self):
        tf = TemporalFilter()
        result = _res(0, superseded_by="newer_doc_v2")
        assert tf.is_active(result) is False

    def test_future_effective_filtered(self):
        tf = TemporalFilter()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        result = _res(0, effective_date=future)
        assert tf.is_active(result) is False

    def test_filter_removes_expired_keeps_active(self):
        tf = TemporalFilter()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        results = [
            _res(0, score=0.9),                   # active
            _res(1, score=0.8, expiry_date=past),  # expired
            _res(2, score=0.7),                    # active
        ]
        filtered = tf.filter(results)
        assert len(filtered) == 2
        assert all(r.document.id != "d1" for r in filtered)

    def test_filter_reranks_after_removal(self):
        tf = TemporalFilter()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        results = [_res(0), _res(1, expiry_date=past), _res(2)]
        filtered = tf.filter(results)
        assert [r.rank for r in filtered] == [1, 2]

    def test_no_temporal_metadata_passes(self):
        tf = TemporalFilter()
        result = _res(0)  # no dates at all
        assert tf.is_active(result) is True


class TestTemporalRanker:
    def test_recent_doc_boosted(self):
        ranker = TemporalRanker(recency_weight=0.2, decay_days=30)
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
        results = [
            _res(0, score=0.8, rank=1, created_at=old),
            _res(1, score=0.7, rank=2, created_at=recent),
        ]
        reranked = ranker.rerank(results)
        # recent doc (d1) should now rank above old (d0) despite lower base score
        assert reranked[0].document.id == "d1"

    def test_rerank_preserves_count(self):
        ranker = TemporalRanker()
        results = [_res(i) for i in range(5)]
        reranked = ranker.rerank(results)
        assert len(reranked) == 5

    def test_scores_stay_in_range(self):
        ranker = TemporalRanker(recency_weight=0.5)
        results = [_res(i, score=0.9) for i in range(3)]
        reranked = ranker.rerank(results)
        assert all(0.0 <= r.score <= 1.0 for r in reranked)


# ── Data Classification ───────────────────────────────────────────────────────

class TestClassificationLabel:
    def test_rank_ordering(self):
        assert ClassificationLabel.rank("PUBLIC") < ClassificationLabel.rank("INTERNAL")
        assert ClassificationLabel.rank("INTERNAL") < ClassificationLabel.rank("CONFIDENTIAL")
        assert ClassificationLabel.rank("CONFIDENTIAL") < ClassificationLabel.rank("RESTRICTED")

    def test_unclassified_equals_public(self):
        assert ClassificationLabel.rank("UNCLASSIFIED") == ClassificationLabel.rank("PUBLIC")


class TestClassificationFilter:
    def test_internal_clearance_blocks_restricted(self):
        clf = ClassificationFilter(user_clearance="INTERNAL")
        result = _res(0, classification="RESTRICTED")
        assert clf.is_accessible(result) is False

    def test_internal_clearance_allows_public(self):
        clf = ClassificationFilter(user_clearance="INTERNAL")
        result = _res(0, classification="PUBLIC")
        assert clf.is_accessible(result) is True

    def test_internal_clearance_allows_internal(self):
        clf = ClassificationFilter(user_clearance="INTERNAL")
        result = _res(0, classification="INTERNAL")
        assert clf.is_accessible(result) is True

    def test_restricted_clearance_allows_all(self):
        clf = ClassificationFilter(user_clearance="RESTRICTED")
        for label in ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]:
            result = _res(0, classification=label)
            assert clf.is_accessible(result) is True

    def test_public_clearance_blocks_internal(self):
        clf = ClassificationFilter(user_clearance="PUBLIC")
        result = _res(0, classification="INTERNAL")
        assert clf.is_accessible(result) is False

    def test_filter_removes_restricted_docs(self):
        clf = ClassificationFilter(user_clearance="INTERNAL")
        results = [
            _res(0, classification="PUBLIC"),
            _res(1, classification="RESTRICTED"),
            _res(2, classification="INTERNAL"),
            _res(3, classification="CONFIDENTIAL"),
        ]
        accessible = clf.filter(results)
        ids = [r.document.id for r in accessible]
        assert "d0" in ids
        assert "d1" not in ids   # RESTRICTED blocked
        assert "d2" in ids
        assert "d3" not in ids   # CONFIDENTIAL blocked for INTERNAL clearance

    def test_no_classification_treated_as_unclassified(self):
        clf = ClassificationFilter(user_clearance="INTERNAL")
        result = _res(0)  # no classification key
        assert clf.is_accessible(result) is True


class TestClassificationPropagator:
    def test_propagates_to_all_chunks(self):
        from core.chunking.models import Chunk
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id="d", text=f"text {i}")
            for i in range(5)
        ]
        propagated = ClassificationPropagator.propagate(chunks, "CONFIDENTIAL")
        for chunk in propagated:
            assert chunk.metadata["classification"] == "CONFIDENTIAL"

    def test_does_not_override_existing_label(self):
        from core.chunking.models import Chunk
        chunk = Chunk(chunk_id="c1", doc_id="d", text="text",
                      metadata={"classification": "RESTRICTED"})
        ClassificationPropagator.propagate([chunk], "PUBLIC")
        assert chunk.metadata["classification"] == "RESTRICTED"


class TestAccessPolicyEvaluator:
    def test_default_classification_policy(self):
        evaluator = AccessPolicyEvaluator(user_clearance="INTERNAL")
        result = _res(0, classification="CONFIDENTIAL")
        assert evaluator.allows({}, result) is False

    def test_custom_policy_additional_restriction(self):
        def dept_policy(ctx, result):
            return ctx.get("department") == result.document.metadata.get("department")

        evaluator = AccessPolicyEvaluator(
            policy_fn=dept_policy, user_clearance="RESTRICTED"
        )
        result = _res(0, classification="PUBLIC", department="engineering")
        assert evaluator.allows({"department": "engineering"}, result) is True
        assert evaluator.allows({"department": "marketing"}, result) is False

    def test_filter_zero_denied(self):
        """Zero results should be returned for out-of-clearance content."""
        evaluator = AccessPolicyEvaluator(user_clearance="PUBLIC")
        results = [_res(i, classification="RESTRICTED") for i in range(5)]
        filtered = evaluator.filter(results, {})
        assert len(filtered) == 0
