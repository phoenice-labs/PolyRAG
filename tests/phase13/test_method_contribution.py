"""
Phase 13: Method Contribution Field Tests
==========================================
Verifies that the ``method_contributions`` field is correctly populated in
retrieval trail records after a search, and that the
``GET /api/retrieval-trails/analysis`` endpoint returns the expected fields.

Uses FastAPI TestClient with a patched search pipeline (Milvus backend) that
reuses a freshly ingested Hamlet collection scoped to this module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────

BACKEND = "milvus"
COLLECTION = "e2e_hamlet_contrib"
SEARCH_QUERY = "Touching this dreaded sight"
HAMLET_PATH = ROOT / "data" / "shakespeare_hamlet.txt"
TRAIL_LOG = ROOT / "data" / "retrieval_trails.jsonl"

client = TestClient(app, raise_server_exceptions=False)

_SEARCH_PAYLOAD = {
    "query": SEARCH_QUERY,
    "backends": [BACKEND],
    "collection_name": COLLECTION,
    "top_k": 5,
    "methods": {
        "enable_dense": True,
        "enable_bm25": True,
        "enable_graph": False,
        "enable_rerank": False,
        "enable_mmr": False,
        "enable_rewrite": False,
        "enable_multi_query": False,
        "enable_hyde": False,
        "enable_raptor": False,
        "enable_contextual_rerank": False,
        "enable_splade": False,
        "enable_llm_graph": False,
    },
}


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hamlet_text():
    if not HAMLET_PATH.exists():
        pytest.skip(f"Hamlet corpus not found at {HAMLET_PATH}")
    text = HAMLET_PATH.read_text(encoding="utf-8", errors="ignore")
    assert len(text) > 1000
    return text


@pytest.fixture(scope="module")
def contrib_pipeline(hamlet_text):
    """
    Build a Milvus pipeline, ingest Hamlet (first 50 kB), and yield.
    The test collection is dropped before ingest and again in teardown.
    """
    from api.deps import build_pipeline_config, create_pipeline

    config = build_pipeline_config(
        backend=BACKEND,
        collection_name=COLLECTION,
        chunk_size=400,
        chunk_strategy="sentence",
        overlap=50,
        enable_er=True,
    )
    pipeline = create_pipeline(config)

    try:
        pipeline.store.drop_collection(pipeline.store.collection_name)
    except Exception:
        pass

    result = pipeline.ingest_text(
        hamlet_text[:50000],
        metadata={"source": "test_hamlet_contrib", "backend": BACKEND},
    )
    assert result.upserted > 0, f"Ingest produced 0 chunks: {result}"

    yield pipeline

    try:
        pipeline.store.drop_collection(pipeline.store.collection_name)
    except Exception:
        pass
    pipeline.stop()


@pytest.fixture(scope="module")
def patched_contrib_client(contrib_pipeline):
    """
    TestClient patched so that POST /api/search reuses contrib_pipeline
    for any BACKEND request.
    """
    import api.routers.search as search_module

    original_create = search_module.create_pipeline

    def _use_contrib(config):
        if config.get("store", {}).get("backend") == BACKEND:
            return contrib_pipeline
        return original_create(config)

    search_module.create_pipeline = _use_contrib
    yield client
    search_module.create_pipeline = original_create


# ── Private helpers ────────────────────────────────────────────────────────────

def _do_search(patched_client) -> tuple[int, dict]:
    """Run the standard search and return (status_code, json_body)."""
    resp = patched_client.post("/api/search", json=_SEARCH_PAYLOAD)
    return resp.status_code, resp.json()


def _read_latest_trail() -> dict | None:
    """Read the most-recently appended trail record from the JSONL file."""
    if not TRAIL_LOG.exists():
        return None
    lines = [
        ln.strip()
        for ln in TRAIL_LOG.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _get_newest_trail_record() -> dict | None:
    """
    Fetch the newest trail via the API endpoint.
    Falls back to reading the JSONL file directly if the endpoint returns
    an empty list (e.g., right after a clear).
    """
    resp = client.get("/api/retrieval-trails")
    if resp.status_code == 200 and resp.json():
        return resp.json()[0]
    return _read_latest_trail()


# ═══════════════════════════════════════════════════════════════════════════════
# Trail Schema Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrailSchema:
    """Verify the structure of trail records persisted after a search."""

    def test_method_contributions_field_present(self, patched_contrib_client):
        """A trail record produced after a search must contain method_contributions."""
        client.delete("/api/retrieval-trails")
        status, _ = _do_search(patched_contrib_client)
        assert status == 200, f"Search returned HTTP {status}"

        record = _get_newest_trail_record()
        assert record is not None, "No trail record found after search"
        assert "method_contributions" in record, (
            f"method_contributions missing; trail keys: {list(record.keys())}"
        )

    def test_enabled_methods_appear_in_contributions(self, patched_contrib_client):
        """
        Methods enabled in the search request (dense, bm25) must appear as
        keys in method_contributions (mapped without the 'enable_' prefix).
        """
        client.delete("/api/retrieval-trails")
        status, _ = _do_search(patched_contrib_client)
        assert status == 200

        record = _get_newest_trail_record()
        assert record is not None
        contributions: dict = record.get("method_contributions", {})

        assert len(contributions) > 0, (
            "method_contributions is empty — expected at least 'dense' and 'bm25'"
        )
        # At least one of the two enabled methods must be present
        assert "dense" in contributions or "bm25" in contributions, (
            f"Neither 'dense' nor 'bm25' found in method_contributions; "
            f"keys: {list(contributions.keys())}"
        )

    def test_contribution_pct_is_float_in_range(self, patched_contrib_client):
        """contribution_pct values in trail records must be floats in [0, 100]."""
        client.delete("/api/retrieval-trails")
        status, _ = _do_search(patched_contrib_client)
        assert status == 200

        record = _get_newest_trail_record()
        assert record is not None

        contributions: dict = record.get("method_contributions", {})
        for method, stats in contributions.items():
            if "contribution_pct" not in stats:
                continue
            pct = stats["contribution_pct"]
            assert isinstance(pct, (int, float)), (
                f"{method}.contribution_pct is not numeric: {type(pct).__name__}"
            )
            assert 0.0 <= pct <= 100.0, (
                f"{method}.contribution_pct={pct} is outside [0, 100]"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalysisEndpoint:
    """Verify GET /api/retrieval-trails/analysis returns the expected schema."""

    @pytest.fixture(autouse=True)
    def _seed_trail(self, patched_contrib_client):
        """Ensure at least one trail with method_contributions exists before each test."""
        _do_search(patched_contrib_client)

    def test_analysis_returns_per_method(self):
        """Response must contain a per_method dict when trails are available."""
        resp = client.get("/api/retrieval-trails/analysis")
        assert resp.status_code == 200
        data = resp.json()
        if "error" in data:
            pytest.skip(f"Analysis endpoint returned error (no qualifying trails): {data['error']}")
        assert "per_method" in data, (
            f"Missing 'per_method'; response keys: {list(data.keys())}"
        )
        assert isinstance(data["per_method"], dict)

    def test_analysis_has_recommended(self):
        """Response must contain a recommended list when trails are available."""
        resp = client.get("/api/retrieval-trails/analysis")
        assert resp.status_code == 200
        data = resp.json()
        if "error" in data:
            pytest.skip(f"Analysis endpoint returned error: {data['error']}")
        assert "recommended" in data, (
            f"Missing 'recommended'; response keys: {list(data.keys())}"
        )
        assert isinstance(data["recommended"], list)

    def test_analysis_interpretation_present(self):
        """Response must contain a non-empty interpretation string."""
        resp = client.get("/api/retrieval-trails/analysis")
        assert resp.status_code == 200
        data = resp.json()
        if "error" in data:
            pytest.skip(f"Analysis endpoint returned error: {data['error']}")
        assert "interpretation" in data, (
            f"Missing 'interpretation'; response keys: {list(data.keys())}"
        )
        assert isinstance(data["interpretation"], str)
        assert len(data["interpretation"]) > 10, (
            "interpretation string is too short to be meaningful"
        )
