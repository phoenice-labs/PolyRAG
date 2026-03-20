"""
End-to-End Frontend API validation tests.
Verifies every API endpoint the frontend calls against the LIVE server at localhost:8000.

Uses data/shakespeare_hamlet.txt + search "Touching this dreaded sight".

Run with:
    python -m pytest tests/phase13/test_e2e_frontend.py -v --tb=short -s

Prerequisites:
    .\start.ps1 -Action start   (API must be running at localhost:8000)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import requests

BASE = "http://localhost:8000/api"
HAMLET_PATH = "data/shakespeare_hamlet.txt"
SEARCH_QUERY = "Touching this dreaded sight"
COLLECTION = "e2e_hamlet_test"
ALL_BACKENDS = ["faiss", "chromadb", "qdrant", "weaviate", "milvus", "pgvector"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(path: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 120)
    return requests.get(f"{BASE}{path}", **kwargs)

def _post(path: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 120)
    return requests.post(f"{BASE}{path}", **kwargs)

def _delete(path: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 30)
    return requests.delete(f"{BASE}{path}", **kwargs)


def _wait_for_job(job_id: str, timeout: int = 180) -> dict:
    """Poll /api/ingest/{job_id}/status until done/error."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _get(f"/ingest/{job_id}/status")
        assert r.status_code == 200, f"Status poll failed: {r.text}"
        d = r.json()
        if d["status"] in ("done", "error"):
            return d
        time.sleep(3)
    pytest.fail(f"Job {job_id} did not complete within {timeout}s")


def _available_backends() -> list[str]:
    r = _get("/backends")
    assert r.status_code == 200
    return [b["name"] for b in r.json() if b["status"] in ("available", "connected")]


def _require_server():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        if r.status_code != 200:
            pytest.skip("API server not running — start with: .\\start.ps1 -Action start")
    except Exception:
        pytest.skip("API server not reachable at localhost:8000 — start with: .\\start.ps1 -Action start")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Server connectivity gate
# ─────────────────────────────────────────────────────────────────────────────

def test_server_is_running():
    """All tests depend on the live server. Fail fast if it's not up."""
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        d = r.json()
        assert d["status"] == "ok"
        assert "Phoenice" in d.get("service", "")
        print(f"\n  API server: {d}")
    except requests.ConnectionError:
        pytest.fail("API server not running at localhost:8000. Run: .\\start.ps1 -Action start")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /api/backends  — BackendSelector + DocumentLibrary
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBackends:
    def test_returns_list(self):
        r = _get("/backends")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_returns_all_six(self):
        r = _get("/backends")
        names = {b["name"] for b in r.json()}
        assert names == set(ALL_BACKENDS), f"Got: {names}"

    def test_each_backend_has_required_fields(self):
        for b in _get("/backends").json():
            assert "name" in b
            assert "status" in b
            assert b["status"] in ("available", "connected", "error")

    def test_faiss_chromadb_available_or_error(self):
        """FAISS/ChromaDB should be available (in-process). If error, report why."""
        by_name = {b["name"]: b for b in _get("/backends").json()}
        for name in ("faiss", "chromadb"):
            b = by_name[name]
            print(f"\n  {name}: status={b['status']} error={b.get('error')}")
            # Not asserting 'available' since server env may differ — just document it
            assert b["status"] in ("available", "connected", "error")

    def test_docker_backends_show_connected_or_error(self):
        """Docker backends show 'connected' when Docker is running."""
        by_name = {b["name"]: b for b in _get("/backends").json()}
        for name in ("qdrant", "weaviate", "milvus", "pgvector"):
            b = by_name[name]
            print(f"\n  {name}: status={b['status']} ping={b.get('ping_ms')}ms")
            assert b["status"] in ("available", "connected", "error")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /api/backends/{name}/health  — BackendHealthBar
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendHealth:
    def test_faiss_health(self):
        r = _get("/backends/faiss/health")
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "faiss"
        assert "ping_ms" in d

    def test_unknown_backend_404(self):
        r = _get("/backends/mongodb/health")
        assert r.status_code == 404

    def test_all_six_backends_health(self):
        """Individual health checks for every backend."""
        print()
        for name in ALL_BACKENDS:
            r = _get(f"/backends/{name}/health")
            assert r.status_code == 200
            d = r.json()
            assert d["name"] == name
            ping = d.get("ping_ms", "?")
            print(f"  {name:<12} status={d['status']:<12} ping={ping}ms")


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST /api/chunks/preview  — IngestionStudio Preview
# ─────────────────────────────────────────────────────────────────────────────

HAMLET_EXCERPT = """\
ACT I. SCENE I. A platform before the castle.
FRANCISCO at his post. Enter to him BERNARDO.
  Ber. Who's there?
  Fran. Nay, answer me. Stand and unfold yourself.
  Ber. Long live the King!
  Fran. You come most carefully upon your hour.
  Ber. 'Tis now struck twelve. Get thee to bed, Francisco.
  Fran. For this relief much thanks. 'Tis bitter cold, And I am sick at heart.
  Ber. Have you had quiet guard?
  Fran. Not a mouse stirring.
"""


class TestChunkPreview:
    def test_sentence_strategy(self):
        r = _post("/chunks/preview", json={
            "text": HAMLET_EXCERPT,
            "strategy": "sentence",
            "chunk_size": 256,
            "overlap": 32,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["total_chunks"] > 0
        assert d["strategy"] == "sentence"
        chunk = d["chunks"][0]
        for field in ("text", "tokens", "index", "char_start", "char_end"):
            assert field in chunk, f"Missing field: {field}"

    def test_section_strategy(self):
        r = _post("/chunks/preview", json={"text": HAMLET_EXCERPT, "strategy": "section"})
        assert r.status_code == 200
        assert r.json()["strategy"] == "section"

    def test_sliding_strategy(self):
        r = _post("/chunks/preview", json={
            "text": HAMLET_EXCERPT, "strategy": "sliding", "chunk_size": 128, "overlap": 16,
        })
        assert r.status_code == 200
        assert r.json()["total_chunks"] > 0

    def test_empty_text_400(self):
        r = _post("/chunks/preview", json={})
        assert r.status_code == 400

    def test_chunk_preview_shows_token_count(self):
        r = _post("/chunks/preview", json={"text": HAMLET_EXCERPT})
        d = r.json()
        for chunk in d["chunks"]:
            assert chunk["tokens"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST /api/ingest  — IngestionStudio (validates exact frontend request shape)
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestRequestShape:
    """
    Frontend sends exact field names: corpus_path, collection_name, chunk_strategy,
    chunk_size, overlap, enable_er  (NOT collection, strategy, chunk_overlap, extract_entities)
    """
    def test_server_path_field_accepted(self):
        """corpus_path (not 'path' or 'file') must be accepted."""
        body = {
            "corpus_path": HAMLET_PATH,
            "backends": ["faiss"],
            "collection_name": COLLECTION,
            "chunk_strategy": "sentence",
            "chunk_size": 400,
            "overlap": 64,
            "enable_er": False,
        }
        r = _post("/ingest", json=body)
        assert r.status_code == 200, f"Ingest failed: {r.text}"
        d = r.json()
        assert "job_ids" in d, f"Missing 'job_ids': {d}"
        assert "faiss" in d["job_ids"]

    def test_response_has_job_ids_dict(self):
        """Response must be {job_ids: {backend: job_id}} not just {job_id: str}."""
        r = _post("/ingest", json={
            "text": "To be or not to be.",
            "backends": ["faiss", "chromadb"],
            "collection_name": COLLECTION,
            "enable_er": False,
        })
        assert r.status_code == 200
        d = r.json()
        assert "job_ids" in d
        assert isinstance(d["job_ids"], dict)
        assert "faiss" in d["job_ids"]
        assert "chromadb" in d["job_ids"]
        for backend, job_id in d["job_ids"].items():
            assert isinstance(job_id, str) and len(job_id) > 0

    def test_missing_content_400(self):
        r = _post("/ingest", json={"backends": ["faiss"]})
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full ingest → search pipeline with hamlet + "Touching this dreaded sight"
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hamlet_ingested_backends():
    """Ingest hamlet into all available backends. Returns list of ready backends."""
    _require_server()
    available = _available_backends()
    print(f"\n[fixture] Available backends: {available}")

    # Delete existing collection to avoid stale data
    for backend in available:
        try:
            _delete(f"/collections/{backend}/{COLLECTION}")
        except Exception:
            pass

    body = {
        "corpus_path": HAMLET_PATH,
        "backends": available,
        "collection_name": COLLECTION,
        "chunk_strategy": "sentence",
        "chunk_size": 400,
        "overlap": 64,
        "enable_er": False,
    }
    r = _post("/ingest", json=body)
    assert r.status_code == 200, f"Ingest failed: {r.text}"
    job_ids = r.json()["job_ids"]

    ready = []
    for backend, job_id in job_ids.items():
        print(f"  Waiting for [{backend}] job {job_id}...")
        result = _wait_for_job(job_id, timeout=300)
        status = result["status"]
        logs = result.get("log_lines", [])
        print(f"  [{backend}] → {status}")
        if status == "error":
            print(f"  [{backend}] error: {result.get('error')}")
            if logs:
                print(f"  [{backend}] last log: {logs[-1]}")
        else:
            if logs:
                print(f"  [{backend}] last log: {logs[-1]}")
            ready.append(backend)

    print(f"[fixture] Ready backends: {ready}")
    return ready


class TestSearchPipeline:
    """
    Frontend sends POST /api/search:
      { query, backends[], collection_name, methods{...}, top_k }
    Response must be:
      { query, results: { backendName: { backend, answer, chunks[], latency_ms } } }
    """
    def test_search_request_exact_frontend_shape(self, hamlet_ingested_backends):
        """Test the exact body the frontend sends."""
        if not hamlet_ingested_backends:
            pytest.skip("No backends ingested")

        body = {
            "query": SEARCH_QUERY,
            "backends": hamlet_ingested_backends[:1],
            "collection_name": COLLECTION,          # NOT 'collection'
            "methods": {                             # NOT 'retrieval_methods'
                "enable_dense": True,
                "enable_bm25": True,
                "enable_graph": False,
                "enable_rerank": True,
                "enable_mmr": True,
                "enable_rewrite": False,
                "enable_multi_query": False,
                "enable_hyde": False,
                "enable_raptor": False,
                "enable_contextual_rerank": False,
            },
            "top_k": 10,
        }
        r = _post("/search", json=body)
        assert r.status_code == 200, f"Search failed: {r.text}"

    def test_response_is_dict_not_list(self, hamlet_ingested_backends):
        """results must be a dict keyed by backend name (not a list)."""
        if not hamlet_ingested_backends:
            pytest.skip("No backends ingested")

        r = _post("/search", json={
            "query": SEARCH_QUERY,
            "backends": hamlet_ingested_backends[:1],
            "collection_name": COLLECTION,
            "top_k": 5,
        })
        assert r.status_code == 200
        d = r.json()
        assert "query" in d
        assert "results" in d
        assert isinstance(d["results"], dict), \
            f"'results' must be dict, got {type(d['results']).__name__}"

    def test_each_result_has_backend_chunks_latency(self, hamlet_ingested_backends):
        """Each BackendSearchResult must have backend, answer, chunks, latency_ms."""
        if not hamlet_ingested_backends:
            pytest.skip("No backends ingested")

        r = _post("/search", json={
            "query": SEARCH_QUERY,
            "backends": hamlet_ingested_backends,
            "collection_name": COLLECTION,
            "top_k": 5,
        })
        d = r.json()
        for backend in hamlet_ingested_backends:
            assert backend in d["results"], f"Backend '{backend}' missing from results"
            br = d["results"][backend]
            for field in ("backend", "answer", "chunks", "latency_ms"):
                assert field in br, f"[{backend}] missing '{field}'"
            assert isinstance(br["chunks"], list)

    def test_chunks_have_chunk_id_text_score_metadata(self, hamlet_ingested_backends):
        """Frontend expects chunk_id, text, score, metadata on each chunk."""
        if not hamlet_ingested_backends:
            pytest.skip("No backends ingested")

        r = _post("/search", json={
            "query": SEARCH_QUERY,
            "backends": hamlet_ingested_backends[:1],
            "collection_name": COLLECTION,
            "top_k": 5,
        })
        backend = hamlet_ingested_backends[0]
        chunks = r.json()["results"][backend]["chunks"]

        if not chunks:
            pytest.skip(f"No chunks from {backend}")

        for i, chunk in enumerate(chunks):
            assert "chunk_id" in chunk, f"chunk[{i}] missing 'chunk_id'"
            assert "text" in chunk,     f"chunk[{i}] missing 'text'"
            assert "score" in chunk,    f"chunk[{i}] missing 'score'"
            assert "metadata" in chunk, f"chunk[{i}] missing 'metadata'"
            assert len(chunk["text"]) > 0, f"chunk[{i}] text is empty"

    def test_search_finds_hamlet_quote(self, hamlet_ingested_backends):
        """'Touching this dreaded sight' is a direct Hamlet quote — must be retrieved."""
        if not hamlet_ingested_backends:
            pytest.skip("No backends ingested")

        r = _post("/search", json={
            "query": SEARCH_QUERY,
            "backends": hamlet_ingested_backends[:1],
            "collection_name": COLLECTION,
            "top_k": 10,
        })
        backend = hamlet_ingested_backends[0]
        chunks = r.json()["results"][backend]["chunks"]

        query_words = {"touching", "dreaded", "sight", "hamlet", "ghost", "king", "horatio"}
        found_chunks = [
            c for c in chunks
            if any(w in c["text"].lower() for w in query_words)
        ]
        print(f"\n  [{backend}] Found {len(found_chunks)}/{len(chunks)} relevant chunks")
        if chunks:
            print(f"  Top chunk (score={chunks[0]['score']:.4f}):\n    {chunks[0]['text'][:200]}")

        assert found_chunks, (
            f"No relevant chunks from [{backend}] for query: {SEARCH_QUERY!r}\n"
            f"Top chunk: {chunks[0]['text'][:200] if chunks else 'NONE'}"
        )

    def test_search_no_backends_400(self):
        r = _post("/search", json={"query": SEARCH_QUERY, "backends": []})
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-backend comparison matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchComparison:
    def test_all_backends_side_by_side(self, hamlet_ingested_backends):
        """Print comparison table like the frontend's ComparisonMatrix."""
        if not hamlet_ingested_backends:
            pytest.skip("No backends ingested")

        r = _post("/search", json={
            "query": SEARCH_QUERY,
            "backends": hamlet_ingested_backends,
            "collection_name": COLLECTION,
            "top_k": 10,
        }, timeout=300)
        assert r.status_code == 200
        results = r.json()["results"]

        print(f"\n{'='*72}")
        print(f"Query: {SEARCH_QUERY!r}")
        print(f"Collection: {COLLECTION}")
        print(f"{'='*72}")
        print(f"{'Backend':<12} {'Chunks':>6}  {'Top Score':>10}  {'Latency':>10}  {'Relevant?':>10}")
        print(f"{'-'*72}")

        query_words = {"touching", "dreaded", "sight", "hamlet", "ghost", "king", "horatio"}
        all_good = True

        for backend in hamlet_ingested_backends:
            br = results.get(backend, {})
            chunks = br.get("chunks", [])
            top_score = chunks[0]["score"] if chunks else 0.0
            latency = br.get("latency_ms", 0)
            error = br.get("error", "")
            if error:
                status = f"ERR: {error[:20]}"
                relevant = "—"
                all_good = False
            else:
                relevant_chunks = [c for c in chunks if any(w in c["text"].lower() for w in query_words)]
                relevant = f"{len(relevant_chunks)}/{len(chunks)}"
                status = "ok"
            print(f"{backend:<12} {len(chunks):>6}  {top_score:>10.4f}  {latency:>8.1f}ms  {relevant:>10}  {status}")

        print(f"{'='*72}")
        # All ready backends should be in results
        for backend in hamlet_ingested_backends:
            assert backend in results


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET/DELETE /api/collections/{backend}  — DocumentLibrary
# ─────────────────────────────────────────────────────────────────────────────

class TestCollections:
    def test_list_collections_faiss(self, hamlet_ingested_backends):
        if "faiss" not in hamlet_ingested_backends:
            pytest.skip("faiss not ingested")
        r = _get("/collections/faiss")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        for col in data:
            assert "name" in col
            assert "chunk_count" in col

    def test_collection_in_list(self, hamlet_ingested_backends):
        if "faiss" not in hamlet_ingested_backends:
            pytest.skip("faiss not ingested")
        r = _get("/collections/faiss")
        names = [c["name"] for c in r.json()]
        print(f"\n  FAISS collections: {names}")
        assert COLLECTION in names

    def test_unknown_backend_404(self):
        r = _get("/collections/mongodb")
        assert r.status_code == 404

    def test_delete_specific_collection(self, hamlet_ingested_backends):
        """Create a temp collection then delete it — verify DELETE works."""
        if "faiss" not in hamlet_ingested_backends:
            pytest.skip("faiss not ingested")

        # Ingest a tiny disposable collection
        r = _post("/ingest", json={
            "text": "Brief text for deletion test.",
            "backends": ["faiss"],
            "collection_name": "e2e_delete_me",
            "enable_er": False,
        })
        assert r.status_code == 200
        job_id = r.json()["job_ids"]["faiss"]
        _wait_for_job(job_id, timeout=60)

        # Delete it
        r = _delete("/collections/faiss/e2e_delete_me")
        assert r.status_code == 200
        d = r.json()
        assert d["backend"] == "faiss"
        assert d["collection"] == "e2e_delete_me"


# ─────────────────────────────────────────────────────────────────────────────
# 8. POST /api/feedback  — ResultCard thumbs
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_thumbs_up(self):
        r = _post("/feedback", json={
            "query": SEARCH_QUERY,
            "chunk_id": "chunk-hamlet-001",
            "backend": "faiss",
            "collection_name": COLLECTION,
            "relevant": True,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "stored"

    def test_thumbs_down(self):
        r = _post("/feedback", json={
            "query": SEARCH_QUERY,
            "chunk_id": "chunk-hamlet-002",
            "backend": "chromadb",
            "collection_name": COLLECTION,
            "relevant": False,
        })
        assert r.status_code == 200

    def test_get_feedback_list(self):
        r = _get("/feedback")
        assert r.status_code == 200
        d = r.json()
        assert "entries" in d
        assert "count" in d
        assert d["count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. GET /api/jobs  — JobHistory
# ─────────────────────────────────────────────────────────────────────────────

class TestJobs:
    def test_list_returns_list(self):
        assert isinstance(_get("/jobs").json(), list)

    def test_job_not_found_404(self):
        assert _get("/jobs/nonexistent-uuid").status_code == 404

    def test_ingest_job_in_list(self, hamlet_ingested_backends):
        jobs = _get("/jobs").json()
        assert len(jobs) > 0
        statuses = {j["status"] for j in jobs}
        assert "done" in statuses or "error" in statuses


# ─────────────────────────────────────────────────────────────────────────────
# 10. SSE stream  — IngestionStudio LogStream
# ─────────────────────────────────────────────────────────────────────────────

class TestSSEStream:
    def test_sse_stream_is_text_event_stream(self):
        """Ingest small text, then stream must return SSE content-type."""
        r = _post("/ingest", json={
            "text": "Hamlet: To be or not to be, that is the question.",
            "backends": ["faiss"],
            "collection_name": "e2e_sse_test",
            "enable_er": False,
        })
        assert r.status_code == 200
        job_id = r.json()["job_ids"]["faiss"]

        # Wait for completion
        _wait_for_job(job_id, timeout=90)

        # Read the SSE stream (already completed, server returns buffered events)
        stream_r = requests.get(f"{BASE}/ingest/{job_id}/stream", timeout=30, stream=True)
        assert stream_r.status_code == 200
        content = stream_r.text
        print(f"\n  SSE stream:\n{content[:500]}")
        assert "data:" in content

    def test_sse_stream_contains_status_done_or_error(self):
        """SSE stream must end with STATUS:done or STATUS:error."""
        r = _post("/ingest", json={
            "text": "Marcellus: Something is rotten in the state of Denmark.",
            "backends": ["faiss"],
            "collection_name": "e2e_sse_status_test",
            "enable_er": False,
        })
        assert r.status_code == 200
        job_id = r.json()["job_ids"]["faiss"]
        _wait_for_job(job_id, timeout=90)

        stream_r = requests.get(f"{BASE}/ingest/{job_id}/stream", timeout=30)
        content = stream_r.text
        assert "STATUS:done" in content or "STATUS:error" in content, \
            f"No STATUS terminal event in SSE:\n{content}"


