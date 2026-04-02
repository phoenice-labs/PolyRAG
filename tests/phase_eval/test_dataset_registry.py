"""
Tests for core/evaluation/dataset_registry.py and associated Pydantic schemas.

Run with:
    pytest tests/phase_eval/test_dataset_registry.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path):
    from core.evaluation.dataset_registry import DatasetRegistry
    return DatasetRegistry(base_dir=tmp_path)


def _items(n: int = 3):
    from core.evaluation.dataset_registry import EvalDatasetItem
    return [
        EvalDatasetItem(
            question=f"Question {i}?",
            expected_answer=f"Answer {i}.",
            expected_sources=[f"doc{i}.txt"],
        )
        for i in range(1, n + 1)
    ]


# ── EvalDatasetItem ────────────────────────────────────────────────────────────

class TestEvalDatasetItem:
    def test_as_dict_keys(self):
        from core.evaluation.dataset_registry import EvalDatasetItem
        item = EvalDatasetItem("Q?", "A.", ["src1.pdf"])
        d = item.as_dict()
        assert set(d.keys()) == {"question", "expected_answer", "expected_sources"}

    def test_as_dict_values(self):
        from core.evaluation.dataset_registry import EvalDatasetItem
        item = EvalDatasetItem("Q?", "A.", ["s1", "s2"])
        d = item.as_dict()
        assert d["question"] == "Q?"
        assert d["expected_answer"] == "A."
        assert d["expected_sources"] == ["s1", "s2"]

    def test_default_expected_sources_is_empty_list(self):
        from core.evaluation.dataset_registry import EvalDatasetItem
        item = EvalDatasetItem("Q?", "A.")
        assert item.expected_sources == []


# ── EvalDatasetMeta ────────────────────────────────────────────────────────────

class TestEvalDatasetMeta:
    def test_as_dict_contains_all_fields(self):
        from core.evaluation.dataset_registry import EvalDatasetMeta
        meta = EvalDatasetMeta(
            name="test", version=2, description="desc",
            created_at="2025-01-01T00:00:00Z", updated_at="2025-01-02T00:00:00Z",
            item_count=5,
        )
        d = meta.as_dict()
        assert d["name"] == "test"
        assert d["version"] == 2
        assert d["item_count"] == 5

    def test_default_version_is_one(self):
        from core.evaluation.dataset_registry import EvalDatasetMeta
        meta = EvalDatasetMeta(name="x")
        assert meta.version == 1


# ── EvalDataset ────────────────────────────────────────────────────────────────

class TestEvalDataset:
    def test_to_eval_questions_format(self):
        from core.evaluation.dataset_registry import EvalDataset, EvalDatasetItem, EvalDatasetMeta
        items = [
            EvalDatasetItem("Q1?", "A1.", ["s1"]),
            EvalDatasetItem("Q2?", "A2.", []),
        ]
        meta = EvalDatasetMeta(name="x", item_count=2)
        ds = EvalDataset(meta=meta, items=items)
        qs = ds.to_eval_questions()
        assert len(qs) == 2
        assert qs[0]["question"] == "Q1?"
        assert qs[0]["expected_answer"] == "A1."
        assert qs[0]["expected_sources"] == ["s1"]

    def test_to_eval_questions_empty(self):
        from core.evaluation.dataset_registry import EvalDataset, EvalDatasetMeta
        ds = EvalDataset(meta=EvalDatasetMeta(name="x"), items=[])
        assert ds.to_eval_questions() == []


# ── DatasetRegistry: save ─────────────────────────────────────────────────────

class TestDatasetRegistrySave:
    def test_save_creates_file(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("test-ds", _items(2))
        assert (tmp_path / "test-ds.json").exists()

    def test_first_save_version_one(self, tmp_path):
        reg = _make_registry(tmp_path)
        meta = reg.save("ds", _items(1))
        assert meta.version == 1

    def test_second_save_increments_version(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        meta2 = reg.save("ds", _items(2))
        assert meta2.version == 2

    def test_third_save_increments_to_three(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        reg.save("ds", _items(1))
        meta3 = reg.save("ds", _items(1))
        assert meta3.version == 3

    def test_save_updates_item_count(self, tmp_path):
        reg = _make_registry(tmp_path)
        meta = reg.save("ds", _items(5))
        assert meta.item_count == 5

    def test_description_preserved_on_update_when_empty(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1), description="Original desc")
        meta2 = reg.save("ds", _items(1), description="")
        assert meta2.description == "Original desc"

    def test_description_overridden_when_non_empty(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1), description="Old")
        meta2 = reg.save("ds", _items(1), description="New")
        assert meta2.description == "New"

    def test_created_at_preserved_on_update(self, tmp_path):
        reg = _make_registry(tmp_path)
        meta1 = reg.save("ds", _items(1))
        meta2 = reg.save("ds", _items(2))
        assert meta1.created_at == meta2.created_at

    def test_invalid_name_raises_value_error(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(ValueError, match="Invalid dataset name"):
            reg.save("UPPER CASE!", _items(1))

    def test_invalid_name_with_spaces(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(ValueError):
            reg.save("has spaces", _items(1))

    def test_valid_name_with_hyphens(self, tmp_path):
        reg = _make_registry(tmp_path)
        meta = reg.save("my-dataset-v1", _items(1))
        assert meta.name == "my-dataset-v1"

    def test_valid_name_with_underscores(self, tmp_path):
        reg = _make_registry(tmp_path)
        meta = reg.save("my_dataset", _items(1))
        assert meta.name == "my_dataset"

    def test_json_file_content_is_valid(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(2), description="test")
        raw = json.loads((tmp_path / "ds.json").read_text(encoding="utf-8"))
        assert "meta" in raw
        assert "items" in raw
        assert len(raw["items"]) == 2
        assert raw["meta"]["version"] == 1

    def test_empty_items_list_allowed(self, tmp_path):
        reg = _make_registry(tmp_path)
        meta = reg.save("empty-ds", [])
        assert meta.item_count == 0


# ── DatasetRegistry: load ─────────────────────────────────────────────────────

class TestDatasetRegistryLoad:
    def test_load_after_save(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(3))
        dataset = reg.load("ds")
        assert len(dataset.items) == 3

    def test_load_missing_raises_key_error(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            reg.load("nonexistent")

    def test_load_preserves_questions(self, tmp_path):
        from core.evaluation.dataset_registry import EvalDatasetItem
        reg = _make_registry(tmp_path)
        original = [EvalDatasetItem("Who?", "Hamlet.", ["act1.txt"])]
        reg.save("ds", original)
        dataset = reg.load("ds")
        assert dataset.items[0].question == "Who?"
        assert dataset.items[0].expected_answer == "Hamlet."
        assert dataset.items[0].expected_sources == ["act1.txt"]

    def test_load_preserves_version(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        reg.save("ds", _items(2))
        dataset = reg.load("ds")
        assert dataset.meta.version == 2

    def test_load_meta_name_matches(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("my-corpus", _items(1))
        dataset = reg.load("my-corpus")
        assert dataset.meta.name == "my-corpus"


# ── DatasetRegistry: list_datasets ────────────────────────────────────────────

class TestDatasetRegistryList:
    def test_empty_registry_returns_empty_list(self, tmp_path):
        reg = _make_registry(tmp_path)
        assert reg.list_datasets() == []

    def test_lists_all_datasets(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("alpha", _items(1))
        reg.save("beta", _items(2))
        reg.save("gamma", _items(3))
        metas = reg.list_datasets()
        names = [m.name for m in metas]
        assert set(names) == {"alpha", "beta", "gamma"}

    def test_results_sorted_alphabetically(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("zeta", _items(1))
        reg.save("alpha", _items(1))
        reg.save("mu", _items(1))
        names = [m.name for m in reg.list_datasets()]
        assert names == sorted(names)

    def test_item_count_in_listing(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(7))
        metas = reg.list_datasets()
        assert metas[0].item_count == 7

    def test_corrupted_file_skipped_gracefully(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("good", _items(1))
        (tmp_path / "bad.json").write_text("NOT JSON", encoding="utf-8")
        metas = reg.list_datasets()
        names = [m.name for m in metas]
        assert "good" in names
        assert "bad" not in names


# ── DatasetRegistry: delete ───────────────────────────────────────────────────

class TestDatasetRegistryDelete:
    def test_delete_existing_returns_true(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        assert reg.delete("ds") is True

    def test_delete_removes_file(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        reg.delete("ds")
        assert not (tmp_path / "ds.json").exists()

    def test_delete_nonexistent_returns_false(self, tmp_path):
        reg = _make_registry(tmp_path)
        assert reg.delete("phantom") is False

    def test_load_after_delete_raises_key_error(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        reg.delete("ds")
        with pytest.raises(KeyError):
            reg.load("ds")


# ── DatasetRegistry: exists ───────────────────────────────────────────────────

class TestDatasetRegistryExists:
    def test_exists_after_save(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        assert reg.exists("ds") is True

    def test_not_exists_before_save(self, tmp_path):
        reg = _make_registry(tmp_path)
        assert reg.exists("ds") is False

    def test_not_exists_after_delete(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.save("ds", _items(1))
        reg.delete("ds")
        assert reg.exists("ds") is False


# ── _validate_name ─────────────────────────────────────────────────────────────

class TestValidateName:
    def setup_method(self):
        from core.evaluation.dataset_registry import _validate_name
        self._v = _validate_name

    def test_accepts_simple_name(self):
        self._v("hamlet")  # should not raise

    def test_accepts_name_with_digits(self):
        self._v("test123")

    def test_accepts_name_with_hyphens(self):
        self._v("my-test-ds")

    def test_accepts_name_with_underscores(self):
        self._v("my_test_ds")

    def test_rejects_uppercase(self):
        with pytest.raises(ValueError):
            self._v("MyDataset")

    def test_rejects_spaces(self):
        with pytest.raises(ValueError):
            self._v("has space")

    def test_rejects_start_with_hyphen(self):
        with pytest.raises(ValueError):
            self._v("-starts-hyphen")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            self._v("")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            self._v("a" * 65)

    def test_accepts_exactly_64_chars(self):
        self._v("a" * 64)  # should not raise


# ── Round-trip: save → load → to_eval_questions ───────────────────────────────

class TestRoundTrip:
    def test_full_round_trip(self, tmp_path):
        from core.evaluation.dataset_registry import EvalDatasetItem
        reg = _make_registry(tmp_path)
        original_items = [
            EvalDatasetItem("What is Hamlet about?", "Revenge and mortality.", ["hamlet.txt"]),
            EvalDatasetItem("Who is Ophelia?", "Hamlet's love interest.", ["hamlet.txt", "act2.txt"]),
        ]
        reg.save("hamlet-qa", original_items, description="Hamlet test corpus")

        loaded = reg.load("hamlet-qa")
        assert loaded.meta.description == "Hamlet test corpus"
        assert loaded.meta.version == 1
        assert len(loaded.items) == 2

        qs = loaded.to_eval_questions()
        assert qs[0]["question"] == "What is Hamlet about?"
        assert qs[1]["expected_sources"] == ["hamlet.txt", "act2.txt"]

    def test_version_sequence_round_trip(self, tmp_path):
        reg = _make_registry(tmp_path)
        for i in range(1, 5):
            reg.save("ds", _items(i))
        dataset = reg.load("ds")
        assert dataset.meta.version == 4
        assert dataset.meta.item_count == 4
        assert len(dataset.items) == 4


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class TestDatasetSchemas:
    def test_dataset_item_valid(self):
        from api.schemas import DatasetItem
        item = DatasetItem(question="Q?", expected_answer="A.", expected_sources=["s1"])
        assert item.question == "Q?"

    def test_dataset_item_default_sources(self):
        from api.schemas import DatasetItem
        item = DatasetItem(question="Q?", expected_answer="A.")
        assert item.expected_sources == []

    def test_create_request_valid(self):
        from api.schemas import DatasetCreateRequest, DatasetItem
        req = DatasetCreateRequest(
            name="my-dataset",
            items=[DatasetItem(question="Q?", expected_answer="A.")],
        )
        assert req.name == "my-dataset"

    def test_create_request_invalid_name(self):
        from api.schemas import DatasetCreateRequest, DatasetItem
        with pytest.raises(Exception):  # Pydantic ValidationError
            DatasetCreateRequest(
                name="INVALID NAME",
                items=[DatasetItem(question="Q?", expected_answer="A.")],
            )

    def test_create_request_empty_items_raises(self):
        from api.schemas import DatasetCreateRequest
        with pytest.raises(Exception):
            DatasetCreateRequest(name="valid", items=[])

    def test_meta_response_fields(self):
        from api.schemas import DatasetMetaResponse
        meta = DatasetMetaResponse(
            name="ds", version=1, description="d",
            created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T00:00:00Z",
            item_count=3,
        )
        assert meta.version == 1
        assert meta.item_count == 3

    def test_get_response_structure(self):
        from api.schemas import DatasetGetResponse, DatasetItem, DatasetMetaResponse
        resp = DatasetGetResponse(
            meta=DatasetMetaResponse(
                name="ds", version=1, description="",
                created_at="", updated_at="", item_count=1,
            ),
            items=[DatasetItem(question="Q?", expected_answer="A.")],
        )
        assert len(resp.items) == 1

    def test_run_request_defaults(self):
        from api.schemas import DatasetRunRequest
        req = DatasetRunRequest()
        assert req.backends == ["faiss"]
        assert req.collection_name == "polyrag_docs"


# ── get_dataset_registry singleton ────────────────────────────────────────────

class TestSingleton:
    def test_returns_same_instance(self):
        from core.evaluation.dataset_registry import get_dataset_registry
        r1 = get_dataset_registry()
        r2 = get_dataset_registry()
        assert r1 is r2

    def test_is_dataset_registry_instance(self):
        from core.evaluation.dataset_registry import DatasetRegistry, get_dataset_registry
        assert isinstance(get_dataset_registry(), DatasetRegistry)
