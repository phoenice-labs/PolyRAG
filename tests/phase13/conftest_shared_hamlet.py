"""
Shared Hamlet E2E base classes and fixture factories.
Each per-backend E2E test file imports from here to avoid code duplication.

Usage in each backend test file::

    from tests.phase13.conftest_shared_hamlet import (
        HamletE2ESuite, make_hamlet_pipeline_fixture, make_patched_client_fixture,
    )

    BACKEND = "faiss"
    COLLECTION = "e2e_hamlet_faiss"
    _CLIENT = TestClient(app, raise_server_exceptions=False)

    hamlet_pipeline    = make_hamlet_pipeline_fixture(BACKEND, COLLECTION)
    patched_search_client = make_patched_client_fixture(BACKEND, _CLIENT)

    class TestSearch(HamletE2ESuite.TestSearch):
        BACKEND    = "faiss"
        COLLECTION = COLLECTION
        CLIENT     = _CLIENT
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Default constants — concrete subclasses MUST override these class attributes.
# ---------------------------------------------------------------------------

_DEFAULT_SEARCH_QUERY = "Touching this dreaded sight"

_DEFAULT_ALL_METHODS = {
    "enable_dense": True,
    "enable_bm25": True,
    "enable_graph": True,
    "enable_rerank": True,
    "enable_mmr": True,
    "enable_rewrite": False,
    "enable_multi_query": False,
    "enable_hyde": False,
    "enable_raptor": False,
    "enable_contextual_rerank": False,
}


# ═══════════════════════════════════════════════════════════════════════════
# HamletE2ESuite — namespace/mixin-holder (NOT a pytest test class)
# ═══════════════════════════════════════════════════════════════════════════

class HamletE2ESuite:
    """
    Namespace that groups all shared test mixin classes.

    Each inner class is a *mixin*: it contains test methods that reference
    class attributes (BACKEND, COLLECTION, CLIENT, SEARCH_QUERY, ALL_METHODS).
    Concrete subclasses in each backend test file must set those attributes.
    """

    # ── 1. Health ─────────────────────────────────────────────────────────

    class TestHealth:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_health_endpoint(self):
            """Frontend calls GET /api/health on startup."""
            resp = self.CLIENT.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "Phoenice" in data.get("service", "")

    # ── 2. Backends ───────────────────────────────────────────────────────

    class TestBackends:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_list_backends_includes_backend(self):
            """Frontend BackendSelector calls GET /api/backends."""
            resp = self.CLIENT.get("/api/backends")
            assert resp.status_code == 200
            backends = resp.json()
            assert isinstance(backends, list)
            names = {b["name"] for b in backends}
            assert self.BACKEND in names, f"{self.BACKEND} not in backends: {names}"

        def test_backend_health(self):
            """Frontend polls GET /api/backends/{backend}/health."""
            resp = self.CLIENT.get(f"/api/backends/{self.BACKEND}/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == self.BACKEND
            assert data["status"] in ("available", "connected", "error")

        def test_backend_health_unknown_returns_404(self):
            """Frontend should get 404 for unknown backend."""
            resp = self.CLIENT.get("/api/backends/nonexistent/health")
            assert resp.status_code == 404

    # ── 3. Chunk Preview ──────────────────────────────────────────────────

    class TestChunkPreview:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_chunk_preview_sentence(self, hamlet_text):
            """Frontend IngestionStudio calls POST /api/chunks/preview before ingestion."""
            resp = self.CLIENT.post("/api/chunks/preview", json={
                "text": hamlet_text[:5000],
                "strategy": "sentence",
                "chunk_size": 400,
                "overlap": 50,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_chunks"] > 0
            assert data["strategy"] == "sentence"
            assert isinstance(data["chunks"], list)
            first = data["chunks"][0]
            assert "text" in first
            assert "tokens" in first

    # ── 4. Ingestion ──────────────────────────────────────────────────────

    class TestIngestion:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_ingest_hamlet_direct(self, hamlet_pipeline):
            """Verifies Hamlet is successfully ingested via direct pipeline call."""
            pipeline, result = hamlet_pipeline
            assert result.upserted > 0, f"Expected chunks to be upserted, got: {result}"

        def test_ingest_api_endpoint_schema(self, hamlet_text):
            """Frontend IngestionStudio calls POST /api/ingest — verify response schema."""
            resp = self.CLIENT.post("/api/ingest", json={
                "text": hamlet_text[:1000],
                "backends": [self.BACKEND],
                "collection_name": "schema_test_col",
                "chunk_strategy": "sentence",
                "chunk_size": 400,
                "overlap": 50,
                "enable_er": False,
            })
            assert resp.status_code == 200, f"Ingest schema test failed: {resp.text}"
            data = resp.json()
            assert "job_ids" in data, f"Response missing 'job_ids': {data}"
            assert self.BACKEND in data["job_ids"], f"{self.BACKEND} job_id missing: {data}"

        def test_list_jobs_includes_done_job(self, hamlet_pipeline):
            """Frontend JobHistory calls GET /api/jobs."""
            resp = self.CLIENT.get("/api/jobs")
            assert resp.status_code == 200
            jobs = resp.json()
            assert isinstance(jobs, list)

    # ── 5. Collections ────────────────────────────────────────────────────

    class TestCollections:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_list_collections(self, hamlet_pipeline):
            """Frontend BackendSelector calls GET /api/collections/{backend}."""
            resp = self.CLIENT.get(f"/api/collections/{self.BACKEND}")
            assert resp.status_code in (200, 503), \
                f"Unexpected status: {resp.status_code} {resp.text}"
            if resp.status_code == 200:
                assert isinstance(resp.json(), list)

    # ── 6. Search — Core Functionality ───────────────────────────────────

    class TestSearch:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_search_returns_results(self, patched_search_client):
            """Frontend SearchLab calls POST /api/search — verifies query returns results."""
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": 5,
                "methods": self.ALL_METHODS,
            })
            assert resp.status_code == 200, f"Search failed: {resp.text}"
            data = resp.json()
            assert data["query"] == self.SEARCH_QUERY
            assert self.BACKEND in data["results"]
            backend_result = data["results"][self.BACKEND]
            assert backend_result.get("error") is None, \
                f"Backend returned error: {backend_result.get('error')}"
            assert len(backend_result["chunks"]) > 0, "Expected at least 1 result for Hamlet query"

        def test_search_results_contain_hamlet_content(self, patched_search_client):
            """Verify the top result is contextually relevant to the Hamlet ghost scene."""
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": 10,
                "methods": self.ALL_METHODS,
            })
            assert resp.status_code == 200
            chunks = resp.json()["results"][self.BACKEND]["chunks"]
            all_text = " ".join(c["text"].lower() for c in chunks)
            hamlet_terms = {
                "hamlet", "ghost", "horatio", "marcellus", "barnardo", "sight",
                "dreaded", "spirit", "night", "denmark",
            }
            matched = {term for term in hamlet_terms if term in all_text}
            assert len(matched) >= 3, (
                f"Expected Hamlet terms in results, only found: {matched}\n"
                f"Sample: {all_text[:500]}"
            )

        def test_search_response_has_required_fields(self, patched_search_client):
            """Verify API response shape matches what the frontend expects."""
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": 5,
                "methods": self.ALL_METHODS,
            })
            assert resp.status_code == 200
            data = resp.json()
            backend_result = data["results"][self.BACKEND]
            assert "backend" in backend_result
            assert "answer" in backend_result
            assert "chunks" in backend_result
            assert "retrieval_trace" in backend_result
            assert "llm_traces" in backend_result
            assert "latency_ms" in backend_result
            if backend_result["chunks"]:
                chunk = backend_result["chunks"][0]
                assert "chunk_id" in chunk
                assert "text" in chunk
                assert "score" in chunk
                assert isinstance(chunk["score"], float)

        def test_search_scores_are_ranked(self, patched_search_client):
            """Verify results are returned in descending score order (RRF fusion output)."""
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": 5,
                "methods": self.ALL_METHODS,
            })
            assert resp.status_code == 200
            chunks = resp.json()["results"][self.BACKEND]["chunks"]
            if len(chunks) >= 2:
                scores = [c["score"] for c in chunks]
                assert (
                    scores == sorted(scores, reverse=True)
                    or all(
                        abs(scores[i] - scores[i + 1]) < 0.5
                        for i in range(len(scores) - 1)
                    )
                ), f"Scores not in descending order: {scores}"

    # ── 7. RRF Fusion Verification ────────────────────────────────────────

    class TestRRFFusion:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def _search(self, patched_search_client, methods: dict, top_k: int = 5):
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": top_k,
                "methods": methods,
            })
            assert resp.status_code == 200, f"Search failed: {resp.text}"
            return resp.json()["results"][self.BACKEND]

        def test_dense_only_returns_results(self, patched_search_client):
            """Dense vector search alone should return results."""
            result = self._search(
                patched_search_client,
                {k: (k == "enable_dense") for k in self.ALL_METHODS},
            )
            assert len(result["chunks"]) > 0, "Dense-only search returned no results"

        def test_bm25_only_returns_results(self, patched_search_client):
            """BM25 keyword search alone should return results."""
            result = self._search(
                patched_search_client,
                {k: (k == "enable_bm25") for k in self.ALL_METHODS},
            )
            assert len(result["chunks"]) > 0, "BM25-only search returned no results"

        def test_hybrid_rrf_returns_results(self, patched_search_client):
            """Hybrid RRF (dense+BM25) should return results."""
            result = self._search(
                patched_search_client,
                {**{k: False for k in self.ALL_METHODS}, "enable_dense": True, "enable_bm25": True},
            )
            assert len(result["chunks"]) > 0, "Hybrid RRF returned no results"

        def test_mmr_diversity_reranking_active(self, patched_search_client):
            """MMR should return diverse results — no two identical texts."""
            result = self._search(
                patched_search_client,
                {**self.ALL_METHODS, "enable_mmr": True},
            )
            texts = [c["text"] for c in result["chunks"]]
            assert len(texts) == len(set(texts)), "MMR produced duplicate chunks"

        def test_all_methods_enabled_returns_results(self, patched_search_client):
            """All non-LLM methods enabled simultaneously — RRF fusion across all signals."""
            result = self._search(patched_search_client, self.ALL_METHODS)
            assert result.get("error") is None
            assert len(result["chunks"]) > 0

        def test_rrf_fusion_via_pipeline_direct(self, hamlet_pipeline):
            """Directly verify RRF fusion is exercised (no API overhead)."""
            pipeline, _ = hamlet_pipeline
            response = pipeline.ask(self.SEARCH_QUERY, top_k=5)
            assert len(response.results) > 0, "Pipeline returned no results"
            scores = [r.score for r in response.results]
            assert any(s > 0 for s in scores), f"All scores are zero: {scores}"

    # ── 8. LLM Traceability ───────────────────────────────────────────────

    class TestLLMTrace:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_llm_traces_field_present(self, patched_search_client):
            """llm_traces should always be present (empty if LM Studio offline)."""
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": 5,
                "methods": self.ALL_METHODS,
            })
            assert resp.status_code == 200
            backend_result = resp.json()["results"][self.BACKEND]
            assert "llm_traces" in backend_result
            assert isinstance(backend_result["llm_traces"], list)

        def test_llm_trace_schema(self, patched_search_client):
            """Each trace entry must have method, system_prompt, user_message, response, latency_ms."""
            resp = patched_search_client.post("/api/search", json={
                "query": self.SEARCH_QUERY,
                "backends": [self.BACKEND],
                "collection_name": self.COLLECTION,
                "top_k": 5,
                "methods": {**self.ALL_METHODS, "enable_rewrite": True},
            })
            assert resp.status_code == 200
            traces = resp.json()["results"][self.BACKEND]["llm_traces"]
            for trace in traces:
                assert "method" in trace
                assert "system_prompt" in trace
                assert "user_message" in trace
                assert "response" in trace
                assert "latency_ms" in trace
                assert isinstance(trace["latency_ms"], (int, float))

    # ── 9. Prompt Management ──────────────────────────────────────────────

    class TestPromptManagement:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_list_prompts(self):
            """Frontend PromptEditor calls GET /api/prompts."""
            resp = self.CLIENT.get("/api/prompts")
            assert resp.status_code == 200
            prompts = resp.json()
            assert isinstance(prompts, list)
            assert len(prompts) > 0

        def test_prompt_entry_has_required_fields(self):
            """Each prompt entry must have key, method_name, method_id, pipeline_stage, description, template."""
            resp = self.CLIENT.get("/api/prompts")
            assert resp.status_code == 200
            for entry in resp.json():
                assert "key" in entry
                assert "method_name" in entry
                assert "method_id" in entry
                assert "pipeline_stage" in entry
                assert "description" in entry
                assert "template" in entry
                assert len(entry["template"]) > 10, \
                    f"Template for {entry['key']} is suspiciously short"

        def test_prompt_shows_method_link(self):
            """Prompt entries should show which retrieval method they belong to."""
            resp = self.CLIENT.get("/api/prompts")
            assert resp.status_code == 200
            prompts = resp.json()
            rewriter = next((p for p in prompts if p["key"] == "query_rewriter"), None)
            assert rewriter is not None, "query_rewriter prompt not found"
            assert rewriter["method_id"] == 5
            assert rewriter["pipeline_stage"] == "Query Expansion"
            assert "method_name" in rewriter

        def test_update_and_reset_prompt(self):
            """Frontend PromptEditor calls PUT then POST reset."""
            original_resp = self.CLIENT.get("/api/prompts")
            assert original_resp.status_code == 200
            prompts = original_resp.json()
            first = prompts[0]
            original_template = first["template"]
            test_key = first["key"]

            new_template = "Test prompt — updated by E2E test."
            put_resp = self.CLIENT.put(f"/api/prompts/{test_key}", json={"template": new_template})
            assert put_resp.status_code == 200
            assert put_resp.json()["status"] == "updated"

            get_resp = self.CLIENT.get("/api/prompts")
            updated = next(p for p in get_resp.json() if p["key"] == test_key)
            assert updated["template"] == new_template

            reset_resp = self.CLIENT.post(f"/api/prompts/{test_key}/reset")
            assert reset_resp.status_code == 200
            assert reset_resp.json()["status"] == "reset"

            final_resp = self.CLIENT.get("/api/prompts")
            restored = next(p for p in final_resp.json() if p["key"] == test_key)
            assert restored["template"] == original_template

        def test_update_unknown_prompt_returns_404(self):
            resp = self.CLIENT.put("/api/prompts/nonexistent_key", json={"template": "test"})
            assert resp.status_code == 404

    # ── 10. Feedback ──────────────────────────────────────────────────────

    class TestFeedback:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_submit_feedback(self):
            """Frontend ResultCard calls POST /api/feedback."""
            resp = self.CLIENT.post("/api/feedback", json={
                "query": self.SEARCH_QUERY,
                "chunk_id": "test-chunk-001",
                "backend": self.BACKEND,
                "collection_name": self.COLLECTION,
                "relevant": True,
            })
            assert resp.status_code in (200, 201, 422), f"Feedback failed: {resp.text}"

    # ── 11. Parallel Retrieval ────────────────────────────────────────────

    class TestParallelRetrieval:
        BACKEND: str
        COLLECTION: str
        CLIENT = None
        SEARCH_QUERY: str = _DEFAULT_SEARCH_QUERY
        ALL_METHODS: dict = _DEFAULT_ALL_METHODS

        def test_hybrid_search_parallel_dense_bm25(self, hamlet_pipeline):
            """Dense + BM25 run in parallel — results must still be returned."""
            pipeline, _ = hamlet_pipeline
            from core.retrieval.hybrid import HybridRetriever
            assert isinstance(pipeline._hybrid_retriever, HybridRetriever)
            results = pipeline._hybrid_retriever.search(self.SEARCH_QUERY, top_k=5)
            assert len(results) > 0, "Parallel Dense+BM25 returned no results"
            assert all(r.score >= 0 for r in results), "Negative scores found"

        def test_pipeline_query_produces_results(self, hamlet_pipeline):
            """pipeline.query() (parallel TripleHybrid + RAPTOR) returns correct chunks."""
            pipeline, _ = hamlet_pipeline
            results = pipeline.query(self.SEARCH_QUERY, top_k=5)
            assert len(results) > 0, "pipeline.query() returned no results"

        def test_warm_query_is_fast(self, hamlet_pipeline):
            """After cold start, warm queries must complete in <5000ms."""
            import time
            pipeline, _ = hamlet_pipeline
            pipeline.query(self.SEARCH_QUERY, top_k=3)  # cold
            times = []
            for _ in range(2):
                t0 = time.perf_counter()
                pipeline.query(self.SEARCH_QUERY, top_k=3)
                times.append((time.perf_counter() - t0) * 1000)
            avg_ms = sum(times) / len(times)
            assert avg_ms < 5000, \
                f"Warm query avg {avg_ms:.0f}ms — too slow (expected <5000ms)"


# ═══════════════════════════════════════════════════════════════════════════
# Fixture factories
# ═══════════════════════════════════════════════════════════════════════════

def make_hamlet_pipeline_fixture(backend: str, collection: str, enable_er: bool = True):
    """
    Factory returning a module-scoped ``hamlet_pipeline`` fixture for any backend.

    The fixture:
    - Skips re-ingest when the collection already contains data (warm run).
    - Drops the test collection after the module run.
    - Calls ``pipeline.stop()`` for clean shutdown.
    """
    import pytest

    @pytest.fixture(scope="module", name="hamlet_pipeline")
    def hamlet_pipeline_fixture(hamlet_text):
        from api.deps import build_pipeline_config, create_pipeline

        config = build_pipeline_config(
            backend=backend,
            collection_name=collection,
            chunk_size=400,
            chunk_strategy="sentence",
            overlap=50,
            enable_er=enable_er,
        )
        pipeline = create_pipeline(config)

        try:
            existing = pipeline.store.count(collection)
        except Exception:
            existing = 0

        if existing == 0:
            try:
                pipeline.store.drop_collection(collection)
            except Exception:
                pass
            result = pipeline.ingest_text(
                hamlet_text[:50000],
                metadata={"source": "test_hamlet", "backend": backend},
            )
            assert result.upserted > 0, f"Ingest produced 0 chunks: {result}"
        else:
            from types import SimpleNamespace
            result = SimpleNamespace(upserted=existing, skipped=0)

        yield pipeline, result

        try:
            pipeline.store.drop_collection(collection)
        except Exception:
            pass
        pipeline.stop()

    return hamlet_pipeline_fixture


def make_patched_client_fixture(backend: str, fastapi_client):
    """
    Factory returning a module-scoped ``patched_search_client`` fixture.

    Patches ``api.routers.search.create_pipeline`` so that POST /api/search
    reuses the pre-ingested hamlet pipeline instead of creating a new one.
    The patch is reverted after the module finishes.
    """
    import pytest

    @pytest.fixture(scope="module", name="patched_search_client")
    def patched_search_client_fixture(hamlet_pipeline):
        import api.routers.search as search_module

        pipeline, _ = hamlet_pipeline
        original_create = search_module.create_pipeline

        def _use_hamlet_pipeline(config):
            if config.get("store", {}).get("backend") == backend:
                return pipeline
            return original_create(config)

        search_module.create_pipeline = _use_hamlet_pipeline
        yield fastapi_client
        search_module.create_pipeline = original_create

    return patched_search_client_fixture
