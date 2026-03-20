"""
Safety net tests for the unified /api/rag endpoint and profile CRUD.

These tests use FastAPI TestClient — no server, no Docker, no LM Studio needed.
They lock in the contract of api/routers/rag.py so that scalability changes
cannot silently break the endpoint shape or profile lifecycle.

Run with:
    pytest tests/phase13/test_rag_router.py -v
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(name: str | None = None, **overrides) -> dict:
    """Return a minimal valid profile payload."""
    return {
        "name": name or f"test-profile-{uuid.uuid4().hex[:8]}",
        "description": "Created by test_rag_router.py",
        "backend": "faiss",
        "collection_name": "test_collection",
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k": 5,
        "methods": {
            "enable_dense": True,
            "enable_bm25": False,
            "enable_splade": False,
            "enable_graph": False,
            "enable_rerank": False,
            "enable_mmr": False,
            "enable_rewrite": False,
            "enable_multi_query": False,
            "enable_hyde": False,
            "enable_raptor": False,
            "enable_contextual_rerank": False,
            "enable_llm_graph": False,
        },
        "confidence_thresholds": {"high": 0.8, "medium": 0.5, "low": 0.3},
        **overrides,
    }


def _cleanup_profile(profile_id: str) -> None:
    """Delete profile from disk after test."""
    p = Path("data/profiles") / f"{profile_id}.json"
    if p.exists():
        p.unlink()


# ── Profile CRUD Tests ────────────────────────────────────────────────────────


class TestProfileCreate:
    def test_create_returns_201_with_id(self):
        payload = _make_profile()
        r = client.post("/api/rag/profiles", json=payload)
        assert r.status_code == 201
        data = r.json()
        assert "id" in data
        assert data["name"] == payload["name"]
        assert data["backend"] == "faiss"
        _cleanup_profile(data["id"])

    def test_create_auto_generates_id_if_omitted(self):
        payload = _make_profile()
        # No 'id' in payload — should be auto-generated
        assert "id" not in payload
        r = client.post("/api/rag/profiles", json=payload)
        assert r.status_code == 201
        data = r.json()
        assert len(data["id"]) > 0
        _cleanup_profile(data["id"])

    def test_create_with_explicit_id(self):
        pid = f"explicit-{uuid.uuid4().hex[:8]}"
        payload = _make_profile(id=pid)
        r = client.post("/api/rag/profiles", json=payload)
        assert r.status_code == 201
        assert r.json()["id"] == pid
        _cleanup_profile(pid)

    def test_create_duplicate_returns_409(self):
        pid = f"dup-{uuid.uuid4().hex[:8]}"
        payload = _make_profile(id=pid)
        r1 = client.post("/api/rag/profiles", json=payload)
        assert r1.status_code == 201
        r2 = client.post("/api/rag/profiles", json=payload)
        assert r2.status_code == 409
        _cleanup_profile(pid)

    def test_create_sets_created_at_and_updated_at(self):
        payload = _make_profile()
        r = client.post("/api/rag/profiles", json=payload)
        data = r.json()
        assert "created_at" in data
        assert "updated_at" in data
        assert data["created_at"]  # non-empty
        _cleanup_profile(data["id"])

    def test_create_confidence_thresholds_stored(self):
        payload = _make_profile()
        payload["confidence_thresholds"] = {"high": 0.9, "medium": 0.6, "low": 0.4}
        r = client.post("/api/rag/profiles", json=payload)
        data = r.json()
        assert data["confidence_thresholds"]["high"] == 0.9
        assert data["confidence_thresholds"]["medium"] == 0.6
        _cleanup_profile(data["id"])


class TestProfileRead:
    def test_get_existing_profile(self):
        payload = _make_profile()
        create_r = client.post("/api/rag/profiles", json=payload)
        pid = create_r.json()["id"]

        r = client.get(f"/api/rag/profiles/{pid}")
        assert r.status_code == 200
        assert r.json()["id"] == pid
        assert r.json()["name"] == payload["name"]
        _cleanup_profile(pid)

    def test_get_missing_profile_returns_404(self):
        r = client.get("/api/rag/profiles/does-not-exist-xyz")
        assert r.status_code == 404

    def test_list_profiles_returns_list(self):
        r = client.get("/api/rag/profiles")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_includes_newly_created(self):
        payload = _make_profile()
        create_r = client.post("/api/rag/profiles", json=payload)
        pid = create_r.json()["id"]

        r = client.get("/api/rag/profiles")
        ids = [p["id"] for p in r.json()]
        assert pid in ids
        _cleanup_profile(pid)


class TestProfileUpdate:
    def test_update_changes_name(self):
        payload = _make_profile()
        pid = client.post("/api/rag/profiles", json=payload).json()["id"]

        updated = _make_profile(name="updated-name", id=pid)
        r = client.put(f"/api/rag/profiles/{pid}", json=updated)
        assert r.status_code == 200
        assert r.json()["name"] == "updated-name"
        _cleanup_profile(pid)

    def test_update_preserves_id_and_created_at(self):
        payload = _make_profile()
        orig = client.post("/api/rag/profiles", json=payload).json()

        modified = _make_profile(name="modified", id="some-other-id")
        r = client.put(f"/api/rag/profiles/{orig['id']}", json=modified)
        data = r.json()
        # id and created_at must be preserved from original
        assert data["id"] == orig["id"]
        assert data["created_at"] == orig["created_at"]
        _cleanup_profile(orig["id"])

    def test_update_missing_profile_returns_404(self):
        r = client.put("/api/rag/profiles/ghost-id", json=_make_profile())
        assert r.status_code == 404


class TestProfileDelete:
    def test_delete_returns_204(self):
        pid = client.post("/api/rag/profiles", json=_make_profile()).json()["id"]
        r = client.delete(f"/api/rag/profiles/{pid}")
        assert r.status_code == 204

    def test_deleted_profile_returns_404(self):
        pid = client.post("/api/rag/profiles", json=_make_profile()).json()["id"]
        client.delete(f"/api/rag/profiles/{pid}")
        r = client.get(f"/api/rag/profiles/{pid}")
        assert r.status_code == 404

    def test_delete_missing_profile_returns_404(self):
        r = client.delete("/api/rag/profiles/no-such-profile")
        assert r.status_code == 404


# ── /api/rag Query Contract Tests ─────────────────────────────────────────────


class TestRagQueryContract:
    """
    These tests verify the SHAPE of the /api/rag response, not the quality of
    retrieval (which depends on ingested data). They use inline config (no profile)
    with dense-only retrieval against an empty FAISS collection.

    The goal is: if the endpoint returns a response at all, it must always
    conform to the RagAnswer schema — even when the collection is empty.
    """

    def _inline_request(self, query: str = "test query", top_k: int = 3) -> dict:
        return {
            "query": query,
            "backend": "faiss",
            "collection_name": f"test_rag_safety_{uuid.uuid4().hex[:6]}",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": top_k,
            "methods": {
                "enable_dense": True,
                "enable_bm25": False,
                "enable_splade": False,
                "enable_graph": False,
                "enable_rerank": False,
                "enable_mmr": False,
                "enable_rewrite": False,
                "enable_multi_query": False,
                "enable_hyde": False,
                "enable_raptor": False,
                "enable_contextual_rerank": False,
                "enable_llm_graph": False,
            },
        }

    def test_response_has_required_top_level_fields(self):
        r = client.post("/api/rag", json=self._inline_request())
        assert r.status_code == 200
        data = r.json()
        required = {"query", "answer", "answer_confidence", "verdict",
                    "sources", "pipeline_audit", "llm_traces", "timestamp"}
        missing = required - set(data.keys())
        assert not missing, f"Missing fields in RagAnswer: {missing}"

    def test_verdict_is_valid_value(self):
        r = client.post("/api/rag", json=self._inline_request())
        assert r.json()["verdict"] in {"HIGH", "MEDIUM", "LOW", "INSUFFICIENT"}

    def test_answer_confidence_is_float_in_range(self):
        r = client.post("/api/rag", json=self._inline_request())
        conf = r.json()["answer_confidence"]
        assert isinstance(conf, (int, float))
        assert 0.0 <= conf <= 1.0

    def test_sources_is_list(self):
        r = client.post("/api/rag", json=self._inline_request())
        assert isinstance(r.json()["sources"], list)

    def test_pipeline_audit_has_required_fields(self):
        r = client.post("/api/rag", json=self._inline_request())
        audit = r.json()["pipeline_audit"]
        for field in ("backend", "collection", "embedding_model",
                      "methods_active", "funnel", "latency_ms"):
            assert field in audit, f"pipeline_audit missing '{field}'"

    def test_pipeline_audit_backend_matches_request(self):
        r = client.post("/api/rag", json=self._inline_request())
        assert r.json()["pipeline_audit"]["backend"] == "faiss"

    def test_llm_traces_is_list(self):
        r = client.post("/api/rag", json=self._inline_request())
        assert isinstance(r.json()["llm_traces"], list)

    def test_timestamp_is_present_and_non_empty(self):
        r = client.post("/api/rag", json=self._inline_request())
        ts = r.json()["timestamp"]
        assert ts and len(ts) > 10  # ISO 8601 string

    def test_query_echoed_in_response(self):
        q = "what does hamlet say about existence"
        r = client.post("/api/rag", json=self._inline_request(query=q))
        assert r.json()["query"] == q

    def test_profile_id_null_when_no_profile(self):
        r = client.post("/api/rag", json=self._inline_request())
        assert r.json()["profile_id"] is None

    def test_profile_based_request(self):
        """Full round-trip: create profile → query with profile_id."""
        # Create profile
        payload = _make_profile()
        payload["collection_name"] = f"test_rag_profile_{uuid.uuid4().hex[:6]}"
        pid = client.post("/api/rag/profiles", json=payload).json()["id"]

        # Query using profile
        r = client.post("/api/rag", json={"query": "test", "profile_id": pid})
        assert r.status_code == 200
        data = r.json()
        assert data["profile_id"] == pid
        assert data["profile_name"] == payload["name"]
        assert "verdict" in data

        _cleanup_profile(pid)

    def test_unknown_profile_id_returns_404(self):
        r = client.post("/api/rag", json={
            "query": "test",
            "profile_id": "profile-that-does-not-exist"
        })
        assert r.status_code == 404
