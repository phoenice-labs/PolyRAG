"""
Eval workbench integrity tests — Phase 1 + Phase 2 coexist without regressions.

Verifies that:
  1. IRMetricsScorer and DatasetRegistry are importable alongside RagasScorer
  2. Public API of core/evaluation/__init__.py is complete
  3. IR metrics produce consistent output (no NaN, correct types)
  4. Dataset registry is independent of vector stores (no pipeline needed)
  5. EvalDatasetItem round-trips cleanly through the registry and to_eval_questions()
  6. Existing schemas (EvaluateRequest, EvaluateQuestionItem) still work unchanged
  7. New schemas coexist with existing ones in api/schemas.py

Run with:
    pytest tests/phase_eval/test_eval_workbench_integrity.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ── 1. Package-level imports ───────────────────────────────────────────────────

class TestPackageImports:
    """All public symbols from core.evaluation are importable."""

    def test_import_ragas_scorer(self):
        from core.evaluation import RagasScorer, RagasResult, get_ragas_scorer
        assert RagasScorer is not None

    def test_import_ir_metrics(self):
        from core.evaluation import IRMetricsScorer, IRMetricsResult, get_ir_scorer
        assert IRMetricsScorer is not None

    def test_import_dataset_registry(self):
        from core.evaluation import (
            DatasetRegistry,
            EvalDataset,
            EvalDatasetItem,
            EvalDatasetMeta,
            get_dataset_registry,
        )
        assert DatasetRegistry is not None

    def test_all_exports_present(self):
        import core.evaluation as ev
        expected = {
            "RagasScorer", "RagasResult", "get_ragas_scorer",
            "IRMetricsScorer", "IRMetricsResult", "get_ir_scorer",
            "DatasetRegistry", "EvalDataset", "EvalDatasetItem",
            "EvalDatasetMeta", "get_dataset_registry",
        }
        assert expected.issubset(set(ev.__all__))


# ── 2. IR metrics — output contract ───────────────────────────────────────────

class TestIRMetricsOutputContract:
    """IRMetricsResult fields are always floats or None — never NaN."""

    def _scorer(self):
        from core.evaluation import get_ir_scorer
        return get_ir_scorer()

    def _chunk(self, text: str = "", source: str = "") -> Dict[str, Any]:
        return {"text": text, "metadata": {"source": source}}

    def test_all_relevant_chunks(self):
        scorer = self._scorer()
        chunks = [self._chunk(source="doc.txt") for _ in range(5)]
        result = scorer.score(chunks, expected_sources=["doc.txt"], k=5)
        assert result.mrr == 1.0
        assert result.precision_at_k == 1.0
        assert result.recall_at_k == 1.0
        assert result.ndcg_at_k == 1.0

    def test_no_relevant_chunks(self):
        scorer = self._scorer()
        chunks = [self._chunk(source="other.txt") for _ in range(5)]
        result = scorer.score(chunks, expected_sources=["expected.txt"], k=5)
        assert result.mrr == 0.0
        assert result.precision_at_k == 0.0
        assert result.ndcg_at_k == 0.0

    def test_empty_sources_returns_none_scores(self):
        scorer = self._scorer()
        result = scorer.score([self._chunk("text", "doc.txt")], expected_sources=[], k=5)
        assert result.mrr is None
        assert result.precision_at_k is None

    def test_result_fields_are_floats_not_nan(self):
        import math
        scorer = self._scorer()
        chunks = [self._chunk(source="doc.txt"), self._chunk(source="other.txt")]
        result = scorer.score(chunks, expected_sources=["doc.txt"], k=2)
        for field_val in [result.mrr, result.precision_at_k, result.recall_at_k, result.ndcg_at_k]:
            if field_val is not None:
                assert isinstance(field_val, float)
                assert not math.isnan(field_val)

    def test_as_dict_is_serialisable(self):
        import json
        scorer = self._scorer()
        chunks = [self._chunk(source="doc.txt")]
        result = scorer.score(chunks, expected_sources=["doc.txt"], k=1)
        # should not raise
        json.dumps(result.as_dict())

    def test_k_zero_handled(self):
        scorer = self._scorer()
        chunks = [self._chunk(source="doc.txt")]
        result = scorer.score(chunks, expected_sources=["doc.txt"], k=0)
        assert result.precision_at_k == 0.0
        assert result.mrr == 0.0


# ── 3. Dataset registry — independence from pipeline ──────────────────────────

class TestDatasetRegistryIndependence:
    """Registry works without any vector store or LLM available."""

    def test_save_load_no_pipeline_needed(self, tmp_path):
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("test", [EvalDatasetItem("Q?", "A.", ["src.txt"])])
        ds = reg.load("test")
        assert ds.items[0].question == "Q?"

    def test_list_no_pipeline_needed(self, tmp_path):
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("alpha", [EvalDatasetItem("Q?", "A.")])
        reg.save("beta", [EvalDatasetItem("Q?", "A.")])
        metas = reg.list_datasets()
        assert len(metas) == 2

    def test_delete_no_pipeline_needed(self, tmp_path):
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("ds", [EvalDatasetItem("Q?", "A.")])
        assert reg.delete("ds") is True
        assert not reg.exists("ds")


# ── 4. to_eval_questions compatibility ────────────────────────────────────────

class TestEvalQuestionsCompatibility:
    """to_eval_questions() output can be fed to EvaluateRequest unchanged."""

    def test_to_eval_questions_schema_compatible(self, tmp_path):
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        from api.schemas import EvaluateQuestionItem

        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("ds", [
            EvalDatasetItem("Q1?", "A1.", ["src1.txt"]),
            EvalDatasetItem("Q2?", "A2."),
        ])
        ds = reg.load("ds")
        questions_dicts = ds.to_eval_questions()

        # Each dict should be constructible as EvaluateQuestionItem
        parsed = [EvaluateQuestionItem(**q) for q in questions_dicts]
        assert len(parsed) == 2
        assert parsed[0].question == "Q1?"
        assert parsed[1].expected_sources == []

    def test_evaluate_request_from_dataset(self, tmp_path):
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        from api.schemas import EvaluateQuestionItem, EvaluateRequest

        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("ds", [EvalDatasetItem("Q?", "A.", ["src.txt"])])
        ds = reg.load("ds")

        eval_req = EvaluateRequest(
            questions=[EvaluateQuestionItem(**q) for q in ds.to_eval_questions()],
            backends=["faiss"],
            collection_name="test",
        )
        assert len(eval_req.questions) == 1
        assert eval_req.backends == ["faiss"]


# ── 5. Existing schemas unchanged ─────────────────────────────────────────────

class TestExistingSchemasUnchanged:
    """Phase 2 additions do not break existing Pydantic models."""

    def test_evaluate_request_still_works(self):
        from api.schemas import EvaluateQuestionItem, EvaluateRequest
        req = EvaluateRequest(
            questions=[EvaluateQuestionItem(question="Q?", expected_answer="A.")],
            backends=["faiss"],
            collection_name="polyrag_docs",
        )
        assert req.collection_name == "polyrag_docs"

    def test_ragas_scores_still_works(self):
        from api.schemas import RagasScores
        rs = RagasScores(faithfulness=0.9, answer_relevancy=0.8)
        assert rs.faithfulness == 0.9

    def test_compare_request_unchanged(self):
        from api.schemas import CompareRequest
        req = CompareRequest(backends=["faiss"], queries=["Q?"])
        assert req.backends == ["faiss"]

    def test_search_request_unchanged(self):
        from api.schemas import SearchRequest
        req = SearchRequest(query="test", backends=["faiss"])
        assert req.query == "test"


# ── 6. New schemas work ────────────────────────────────────────────────────────

class TestNewSchemas:
    def test_dataset_create_valid(self):
        from api.schemas import DatasetCreateRequest, DatasetItem
        req = DatasetCreateRequest(
            name="my-corpus",
            items=[DatasetItem(question="Q?", expected_answer="A.")],
            description="Test",
        )
        assert req.name == "my-corpus"
        assert len(req.items) == 1

    def test_dataset_run_request_defaults(self):
        from api.schemas import DatasetRunRequest
        req = DatasetRunRequest()
        assert "faiss" in req.backends

    def test_dataset_meta_response_serialises(self):
        import json
        from api.schemas import DatasetMetaResponse
        meta = DatasetMetaResponse(
            name="ds", version=2, description="d",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
            item_count=10,
        )
        # model_dump() must work without error
        d = meta.model_dump()
        assert d["version"] == 2
        json.dumps(d)  # must be JSON-serialisable


# ── 7. IR metrics + registry: combined workflow ────────────────────────────────

class TestCombinedWorkflow:
    """Simulates a realistic eval workflow using both Phase 1 and Phase 2."""

    def test_dataset_driven_ir_scoring(self, tmp_path):
        """Store a dataset, load it, convert to IR scorer input, score."""
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        from core.evaluation.ir_metrics import IRMetricsScorer

        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("corpus", [
            EvalDatasetItem("Who is Hamlet?", "Danish prince.", ["hamlet.txt"]),
            EvalDatasetItem("What is the theme?", "Revenge.", ["hamlet.txt"]),
        ])
        ds = reg.load("corpus")

        scorer = IRMetricsScorer()
        # Simulate retrieved chunks (one relevant, one not)
        retrieved = [
            {"text": "Hamlet is a Danish prince.", "metadata": {"source": "hamlet.txt"}},
            {"text": "An unrelated document.", "metadata": {"source": "other.txt"}},
        ]

        for item in ds.items:
            result = scorer.score(retrieved, expected_sources=item.expected_sources, k=2)
            # hamlet.txt appears at rank 1 in retrieved, so MRR must be 1.0
            assert result.mrr == 1.0

    def test_batch_scoring_from_dataset(self, tmp_path):
        from core.evaluation.dataset_registry import DatasetRegistry, EvalDatasetItem
        from core.evaluation.ir_metrics import IRMetricsScorer

        reg = DatasetRegistry(base_dir=tmp_path)
        reg.save("batch-ds", [
            EvalDatasetItem(f"Q{i}?", f"A{i}.", [f"doc{i}.txt"])
            for i in range(5)
        ])
        ds = reg.load("batch-ds")

        scorer = IRMetricsScorer()
        # Build batch items where each query retrieves its own relevant doc
        batch_items = [
            {
                "chunks": [{"text": f"Content of doc{i}", "metadata": {"source": f"doc{i}.txt"}}],
                "expected_sources": ds.items[i].expected_sources,
            }
            for i in range(5)
        ]
        results = scorer.score_batch(batch_items, k=1)
        assert len(results) == 5
        for r in results:
            assert r.mrr == 1.0
