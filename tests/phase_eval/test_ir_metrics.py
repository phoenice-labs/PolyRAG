"""
Tests for core/evaluation/ir_metrics.py and related additions.

Run with:
    pytest tests/phase_eval/test_ir_metrics.py -v
"""
from __future__ import annotations

import math
import pytest
from typing import Any, Dict, List, Optional


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk(text: str = "", source: str = "") -> Dict[str, Any]:
    """Build a dict-style chunk for testing."""
    return {"text": text, "metadata": {"source": source}}


class _ChunkObj:
    """Object-style chunk (mimics pipeline RAGResult)."""
    def __init__(self, text: str = "", source: str = ""):
        self.text = text
        self.metadata = {"source": source}


# ── Task 1: IRMetricsResult ────────────────────────────────────────────────────

class TestIRMetricsResult:
    def test_default_fields(self):
        from core.evaluation.ir_metrics import IRMetricsResult
        r = IRMetricsResult()
        assert r.k == 5
        assert r.mrr is None
        assert r.precision_at_k is None
        assert r.recall_at_k is None
        assert r.ndcg_at_k is None
        assert r.relevant_count == 0
        assert r.total_expected == 0
        assert r.error is None

    def test_as_dict_keys(self):
        from core.evaluation.ir_metrics import IRMetricsResult
        r = IRMetricsResult(k=3, mrr=0.5, precision_at_k=0.33, recall_at_k=1.0,
                            ndcg_at_k=0.75, relevant_count=1, total_expected=1)
        d = r.as_dict()
        expected_keys = {"k", "mrr", "precision_at_k", "recall_at_k",
                         "ndcg_at_k", "relevant_count", "total_expected", "error"}
        assert set(d.keys()) == expected_keys

    def test_as_dict_values(self):
        from core.evaluation.ir_metrics import IRMetricsResult
        r = IRMetricsResult(k=5, mrr=1.0, precision_at_k=0.6, recall_at_k=1.0,
                            ndcg_at_k=0.9, relevant_count=3, total_expected=3)
        d = r.as_dict()
        assert d["k"] == 5
        assert d["mrr"] == 1.0
        assert d["precision_at_k"] == 0.6
        assert d["recall_at_k"] == 1.0
        assert d["ndcg_at_k"] == 0.9
        assert d["relevant_count"] == 3
        assert d["total_expected"] == 3
        assert d["error"] is None

    def test_as_dict_with_error(self):
        from core.evaluation.ir_metrics import IRMetricsResult
        r = IRMetricsResult(error="something went wrong")
        d = r.as_dict()
        assert d["error"] == "something went wrong"


# ── Task 2: _is_relevant ───────────────────────────────────────────────────────

class TestIsRelevant:
    def setup_method(self):
        from core.evaluation.ir_metrics import _is_relevant
        self._is_relevant = _is_relevant

    def test_matches_metadata_source(self):
        chunk = _chunk(text="unrelated", source="policy.pdf")
        assert self._is_relevant(chunk, ["policy.pdf"])

    def test_matches_text_snippet(self):
        chunk = _chunk(text="Source: annual_report.pdf content here", source="")
        assert self._is_relevant(chunk, ["annual_report.pdf"])

    def test_case_insensitive_metadata(self):
        chunk = _chunk(text="", source="Policy.PDF")
        assert self._is_relevant(chunk, ["policy.pdf"])

    def test_case_insensitive_text(self):
        chunk = _chunk(text="Reference: POLICY.PDF", source="")
        assert self._is_relevant(chunk, ["policy.pdf"])

    def test_returns_false_empty_expected(self):
        chunk = _chunk(text="some text", source="doc.pdf")
        assert not self._is_relevant(chunk, [])

    def test_returns_false_no_match(self):
        chunk = _chunk(text="completely unrelated content", source="other.pdf")
        assert not self._is_relevant(chunk, ["policy.pdf"])

    def test_partial_match_within_text(self):
        # "policy" is a substring of "policy_v2.pdf" in text
        chunk = _chunk(text="this references policy_v2.pdf file", source="")
        assert self._is_relevant(chunk, ["policy_v2.pdf"])

    def test_text_truncated_at_200_chars(self):
        # Source mention is past 200 chars — should NOT match via text
        long_text = "x" * 200 + "policy.pdf"
        chunk = _chunk(text=long_text, source="")
        assert not self._is_relevant(chunk, ["policy.pdf"])

    def test_empty_string_in_expected_sources_skipped(self):
        chunk = _chunk(text="", source="")
        # Empty string source should be skipped, no match
        assert not self._is_relevant(chunk, [""])

    def test_missing_metadata_key_handled(self):
        # chunk with no metadata at all
        chunk = {"text": "policy.pdf is here"}
        assert self._is_relevant(chunk, ["policy.pdf"])

    def test_none_metadata_handled(self):
        chunk = {"text": "policy.pdf is here", "metadata": None}
        assert self._is_relevant(chunk, ["policy.pdf"])


# ── Task 3: IRMetricsScorer basic ─────────────────────────────────────────────

class TestIRMetricsScorerBasic:
    @pytest.fixture
    def scorer(self):
        from core.evaluation.ir_metrics import IRMetricsScorer
        return IRMetricsScorer()

    @pytest.fixture
    def relevant_chunks(self):
        return [_chunk(text="", source=f"doc{i}.pdf") for i in range(5)]

    def test_perfect_retrieval(self, scorer):
        chunks = [_chunk(source="doc.pdf") for _ in range(5)]
        result = scorer.score(chunks, expected_sources=["doc.pdf"], k=5)
        assert result.mrr == 1.0
        assert result.precision_at_k == 1.0
        assert result.recall_at_k == 1.0
        assert result.ndcg_at_k == 1.0

    def test_no_relevant_results(self, scorer):
        chunks = [_chunk(source="other.pdf") for _ in range(5)]
        result = scorer.score(chunks, expected_sources=["policy.pdf"], k=5)
        assert result.mrr == 0.0
        assert result.precision_at_k == 0.0
        assert result.recall_at_k == 0.0
        assert result.ndcg_at_k == 0.0

    def test_first_relevant_at_rank_2(self, scorer):
        chunks = [
            _chunk(source="other.pdf"),
            _chunk(source="policy.pdf"),
            _chunk(source="other.pdf"),
        ]
        result = scorer.score(chunks, expected_sources=["policy.pdf"], k=5)
        assert result.mrr == 0.5

    def test_first_relevant_at_rank_3(self, scorer):
        chunks = [
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="policy.pdf"),
        ]
        result = scorer.score(chunks, expected_sources=["policy.pdf"], k=5)
        assert abs(result.mrr - round(1.0 / 3, 4)) < 1e-9

    def test_partial_retrieval(self, scorer):
        # 2 of 5 are relevant, expected has 2 sources
        chunks = [
            _chunk(source="doc1.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="doc2.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
        ]
        result = scorer.score(chunks, expected_sources=["doc1.pdf", "doc2.pdf"], k=5)
        assert result.relevant_count == 2
        assert result.precision_at_k == round(2 / 5, 4)
        assert result.recall_at_k == 1.0  # both expected sources found

    def test_empty_expected_sources(self, scorer):
        chunks = [_chunk(source="doc.pdf")]
        result = scorer.score(chunks, expected_sources=[], k=5)
        assert result.mrr is None
        assert result.precision_at_k is None
        assert result.recall_at_k is None
        assert result.ndcg_at_k is None
        assert result.total_expected == 0

    def test_k_larger_than_chunks(self, scorer):
        chunks = [_chunk(source="doc.pdf")]
        result = scorer.score(chunks, expected_sources=["doc.pdf"], k=10)
        assert result.mrr == 1.0
        assert result.recall_at_k == 1.0

    def test_k_zero(self, scorer):
        chunks = [_chunk(source="doc.pdf")]
        # k=0 → empty top_k, no division errors
        result = scorer.score(chunks, expected_sources=["doc.pdf"], k=0)
        assert result.mrr == 0.0
        assert result.precision_at_k == 0.0
        assert result.relevant_count == 0

    def test_relevant_count(self, scorer):
        chunks = [_chunk(source="a.pdf"), _chunk(source="b.pdf"), _chunk(source="c.pdf")]
        result = scorer.score(chunks, expected_sources=["a.pdf", "b.pdf"], k=5)
        assert result.relevant_count == 2
        assert result.total_expected == 2


# ── Task 4: NDCG ordering tests ───────────────────────────────────────────────

class TestIRMetricsScorerNDCG:
    @pytest.fixture
    def scorer(self):
        from core.evaluation.ir_metrics import IRMetricsScorer
        return IRMetricsScorer()

    def test_ndcg_ideal_order(self, scorer):
        # All relevant docs at top → NDCG = 1.0
        chunks = [
            _chunk(source="doc.pdf"),
            _chunk(source="doc.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
        ]
        result = scorer.score(chunks, expected_sources=["doc.pdf"], k=5)
        assert result.ndcg_at_k == 1.0

    def test_ndcg_worst_order(self, scorer):
        # Relevant doc last → lower NDCG
        chunks = [
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="doc.pdf"),
        ]
        result_worst = scorer.score(chunks, expected_sources=["doc.pdf"], k=5)
        chunks_best = [
            _chunk(source="doc.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
        ]
        result_best = scorer.score(chunks_best, expected_sources=["doc.pdf"], k=5)
        assert result_worst.ndcg_at_k < result_best.ndcg_at_k

    def test_ndcg_partial(self, scorer):
        # 1 of 2 expected sources found at rank 3 (not at the top)
        chunks = [
            _chunk(source="other.pdf"),
            _chunk(source="other.pdf"),
            _chunk(source="doc1.pdf"),  # relevant, but at rank 3
        ]
        result = scorer.score(chunks, expected_sources=["doc1.pdf", "doc2.pdf"], k=5)
        # DCG = 1/log2(4) < IDCG = 1/log2(2), so NDCG < 1.0
        assert 0.0 < result.ndcg_at_k < 1.0

    def test_ndcg_binary_relevance(self, scorer):
        # Manually verify: single relevant doc at rank 1 of 1 expected
        chunks = [_chunk(source="doc.pdf")]
        result = scorer.score(chunks, expected_sources=["doc.pdf"], k=1)
        # DCG = 1/log2(2) = 1.0, IDCG = 1.0, NDCG = 1.0
        assert result.ndcg_at_k == 1.0


# ── Task 5: Batch scoring ─────────────────────────────────────────────────────

class TestIRMetricsScorerBatch:
    @pytest.fixture
    def scorer(self):
        from core.evaluation.ir_metrics import IRMetricsScorer
        return IRMetricsScorer()

    def test_batch_scoring(self, scorer):
        items = [
            {
                "chunks": [_chunk(source="doc1.pdf"), _chunk(source="other.pdf")],
                "expected_sources": ["doc1.pdf"],
            },
            {
                "chunks": [_chunk(source="other.pdf"), _chunk(source="doc2.pdf")],
                "expected_sources": ["doc2.pdf"],
            },
        ]
        results = scorer.score_batch(items, k=5)
        assert len(results) == 2
        assert results[0].mrr == 1.0  # found at rank 1
        assert results[1].mrr == 0.5  # found at rank 2

    def test_batch_empty(self, scorer):
        results = scorer.score_batch([], k=5)
        assert results == []

    def test_batch_item_missing_keys(self, scorer):
        # Items with missing keys should default gracefully
        items = [{}]
        results = scorer.score_batch(items, k=5)
        assert len(results) == 1
        assert results[0].total_expected == 0


# ── Task 6: Singleton ─────────────────────────────────────────────────────────

class TestGetIrScorer:
    def test_singleton(self):
        from core.evaluation.ir_metrics import get_ir_scorer
        s1 = get_ir_scorer()
        s2 = get_ir_scorer()
        assert s1 is s2

    def test_returns_ir_metrics_scorer_instance(self):
        from core.evaluation.ir_metrics import get_ir_scorer, IRMetricsScorer
        scorer = get_ir_scorer()
        assert isinstance(scorer, IRMetricsScorer)


# ── Task 7: Integration with evaluate router ──────────────────────────────────

@pytest.mark.integration
class TestIRMetricsIntegrationWithEvalRouter:
    def test_ir_scores_for_dict_chunks(self):
        from api.routers.evaluate import _ir_scores_for
        chunks = [
            {"text": "policy.pdf content here", "metadata": {"source": "policy.pdf"}},
            {"text": "other content", "metadata": {"source": "other.pdf"}},
        ]
        result = _ir_scores_for(chunks, expected_sources=["policy.pdf"], k=5)
        _assert_ir_dict(result)
        assert result["mrr"] == 1.0

    def test_ir_scores_for_object_chunks(self):
        from api.routers.evaluate import _ir_scores_for
        chunks = [
            _ChunkObj(text="policy.pdf content", source="policy.pdf"),
            _ChunkObj(text="other content", source="other.pdf"),
        ]
        result = _ir_scores_for(chunks, expected_sources=["policy.pdf"], k=5)
        _assert_ir_dict(result)
        assert result["mrr"] == 1.0

    def test_ir_scores_for_empty_expected(self):
        from api.routers.evaluate import _ir_scores_for
        chunks = [_ChunkObj(source="doc.pdf")]
        result = _ir_scores_for(chunks, expected_sources=[], k=5)
        _assert_ir_dict(result)
        assert result["mrr"] is None

    def test_ir_scores_for_all_keys_present(self):
        from api.routers.evaluate import _ir_scores_for
        result = _ir_scores_for([], expected_sources=["x.pdf"], k=5)
        _assert_ir_dict(result)


def _assert_ir_dict(d: dict) -> None:
    """Assert all expected keys are present in an IR metrics dict."""
    required = {"k", "mrr", "precision_at_k", "recall_at_k",
                "ndcg_at_k", "relevant_count", "total_expected", "error"}
    assert required.issubset(set(d.keys())), f"Missing keys: {required - set(d.keys())}"


# ── Task 8: LLMTraceEntry token fields ───────────────────────────────────────

class TestTokenCostFields:
    def test_prompt_tokens_optional_default_none(self):
        from api.schemas import LLMTraceEntry
        import inspect
        fields = LLMTraceEntry.model_fields
        assert "prompt_tokens" in fields
        assert fields["prompt_tokens"].default is None

    def test_completion_tokens_optional_default_none(self):
        from api.schemas import LLMTraceEntry
        fields = LLMTraceEntry.model_fields
        assert "completion_tokens" in fields
        assert fields["completion_tokens"].default is None

    def test_existing_fields_still_present(self):
        from api.schemas import LLMTraceEntry
        fields = LLMTraceEntry.model_fields
        for name in ("method", "system_prompt", "user_message", "response", "latency_ms"):
            assert name in fields, f"Field {name!r} unexpectedly missing"

    def test_backward_compat_without_token_fields(self):
        from api.schemas import LLMTraceEntry
        # Should construct fine without token fields
        entry = LLMTraceEntry(
            method="dense",
            system_prompt="You are a helpful assistant.",
            user_message="What is RAG?",
            response="RAG is Retrieval-Augmented Generation.",
            latency_ms=123.4,
        )
        assert entry.prompt_tokens is None
        assert entry.completion_tokens is None

    def test_construction_with_token_fields(self):
        from api.schemas import LLMTraceEntry
        entry = LLMTraceEntry(
            method="dense",
            system_prompt="sys",
            user_message="user",
            response="resp",
            latency_ms=50.0,
            prompt_tokens=120,
            completion_tokens=80,
        )
        assert entry.prompt_tokens == 120
        assert entry.completion_tokens == 80

    def test_token_fields_type_annotation(self):
        from api.schemas import LLMTraceEntry
        import typing
        hints = typing.get_type_hints(LLMTraceEntry)
        # Optional[int] means Union[int, None]
        for field_name in ("prompt_tokens", "completion_tokens"):
            hint = hints[field_name]
            origin = getattr(hint, "__origin__", None)
            # Could be Union or Optional — both have NoneType in args
            if origin is not None:
                args = hint.__args__
                assert type(None) in args, f"{field_name} should be Optional[int]"
                assert int in args, f"{field_name} should contain int"
            else:
                assert hint in (int, type(None))

    def test_llm_trace_entry_dataclass_has_token_fields(self):
        """The core dataclass (not schema) also has token fields."""
        from core.query.llm_client import LLMTraceEntry
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(LLMTraceEntry)}
        assert "prompt_tokens" in field_names
        assert "completion_tokens" in field_names

    def test_llm_trace_entry_dataclass_defaults(self):
        from core.query.llm_client import LLMTraceEntry
        entry = LLMTraceEntry(
            method="test",
            system_prompt="sys",
            user_message="user",
            response="resp",
            latency_ms=10.0,
        )
        assert entry.prompt_tokens is None
        assert entry.completion_tokens is None
