"""
Phase 13: Backend Parity Tests
================================
Tests cross-backend functional equivalence for the same Hamlet query.
Uses the **live API at localhost:8000** via the `requests` library.

All tests are skipped when:
  - The API server is not reachable at localhost:8000, OR
  - Fewer than 2 backends are available (Docker service not running).

Conservative thresholds are used (overlap ≥ 20 %, latency < 30 s) to keep
tests non-flaky across different hardware and collection states.
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest
import requests as _requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Test configuration ─────────────────────────────────────────────────────────

ALL_BACKENDS = ["faiss", "chromadb", "qdrant", "milvus", "weaviate", "pgvector"]
DOCKER_PORTS = {"qdrant": 6333, "milvus": 19530, "weaviate": 8088, "pgvector": 5433}

SEARCH_QUERY = "Touching this dreaded sight"
HAMLET_TERMS = {"hamlet", "ghost", "horatio", "ophelia", "sight", "dreaded", "spirit"}

API_BASE = "http://localhost:8000/api"

_COLLECTION_MAP = {
    "chromadb": "e2e_hamlet_chroma",
    "faiss": "e2e_hamlet_faiss",
    "qdrant": "e2e_hamlet_qdrant",
    "milvus": "e2e_hamlet_milvus",
    "weaviate": "e2e_hamlet_weaviate",
    "pgvector": "e2e_hamlet_pgvector",
}

_BASELINE_METHODS = {
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
}

_DENSE_ONLY_METHODS = {k: (k == "enable_dense") for k in _BASELINE_METHODS}


def _collection(backend: str) -> str:
    return _COLLECTION_MAP.get(backend, f"e2e_hamlet_{backend}")


# ── Session fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_api():
    """Skip the entire session if the API server is not reachable."""
    try:
        r = _requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        pytest.skip("API server not running at localhost:8000 — skipping parity tests")


@pytest.fixture(scope="session")
def available_backends(live_api):  # noqa: ARG001
    """
    Return the list of backends that have a reachable service.
    Docker-backed backends (qdrant, milvus, weaviate, pgvector) are included
    only when their port is open. In-process backends (faiss, chromadb) are
    always included.
    """
    backends: list[str] = []
    for b in ALL_BACKENDS:
        if b in DOCKER_PORTS:
            try:
                with socket.create_connection(("localhost", DOCKER_PORTS[b]), timeout=2):
                    backends.append(b)
            except OSError:
                pass
        else:
            backends.append(b)

    if len(backends) < 2:
        pytest.skip(
            f"Need ≥2 backends for parity tests; only found: {backends}"
        )
    return backends


@pytest.fixture(scope="session")
def parity_search_results(available_backends):
    """
    Run the baseline search (dense + BM25) on every available backend and
    collect the results.  Backends that fail or return an error are silently
    omitted from the result dict.
    """
    results: dict = {}
    for backend in available_backends:
        try:
            resp = _requests.post(
                f"{API_BASE}/search",
                json={
                    "query": SEARCH_QUERY,
                    "backends": [backend],
                    "collection_name": _collection(backend),
                    "top_k": 5,
                    "methods": _BASELINE_METHODS,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                backend_result = resp.json().get("results", {}).get(backend, {})
                if not backend_result.get("error"):
                    results[backend] = backend_result
        except Exception as exc:
            print(f"[parity] {backend} search failed: {exc}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultCount:
    """Each available backend must return at least one search result."""

    def test_all_backends_return_results(self, parity_search_results):
        assert parity_search_results, "No backend returned successful results"
        for backend, result in parity_search_results.items():
            chunks = result.get("chunks", [])
            assert len(chunks) >= 1, (
                f"{backend}: returned 0 results for query '{SEARCH_QUERY}'"
            )


class TestHamletRelevance:
    """Each backend must return at least one chunk containing a Hamlet keyword."""

    def test_all_backends_return_hamlet_terms(self, parity_search_results):
        for backend, result in parity_search_results.items():
            chunks = result.get("chunks", [])
            has_term = any(
                any(t in c.get("text", "").lower() for t in HAMLET_TERMS)
                for c in chunks
            )
            first_text = chunks[0]["text"][:200] if chunks else "N/A"
            assert has_term, (
                f"{backend}: none of the {len(chunks)} chunks contain a hamlet term. "
                f"First chunk: {first_text!r}"
            )


class TestScoreNormalization:
    """All backends must return scores in the closed interval [0, 1]."""

    def test_scores_in_unit_interval(self, parity_search_results):
        for backend, result in parity_search_results.items():
            for chunk in result.get("chunks", []):
                score = chunk.get("score", -1)
                assert 0.0 <= score <= 1.0, (
                    f"{backend}: score {score!r} is outside [0, 1]"
                )


class TestLatencySLA:
    """Every available backend must respond within 30 seconds (dense-only)."""

    def test_all_backends_within_sla(self, available_backends):
        for backend in available_backends:
            t0 = time.monotonic()
            try:
                resp = _requests.post(
                    f"{API_BASE}/search",
                    json={
                        "query": SEARCH_QUERY,
                        "backends": [backend],
                        "collection_name": _collection(backend),
                        "top_k": 3,
                        "methods": _DENSE_ONLY_METHODS,
                    },
                    timeout=35,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                assert resp.status_code == 200, (
                    f"{backend}: HTTP {resp.status_code}"
                )
                assert latency_ms < 30_000, (
                    f"{backend}: latency {latency_ms:.0f} ms exceeds 30 s SLA"
                )
            except _requests.exceptions.Timeout:
                pytest.fail(f"{backend}: request timed out (> 30 s)")


class TestSearchResultOverlap:
    """
    For each pair of backends, the top-3 results must share ≥ 20% of tokens.
    This is a loose threshold: different backends may rank differently, but
    they should retrieve from the same Hamlet corpus.
    """

    def test_top3_text_overlap_between_backends(self, parity_search_results):
        backends_with_results = list(parity_search_results.keys())
        if len(backends_with_results) < 2:
            pytest.skip("Need ≥2 backends with results for overlap test")

        for i in range(len(backends_with_results)):
            for j in range(i + 1, len(backends_with_results)):
                b1 = backends_with_results[i]
                b2 = backends_with_results[j]

                def _top3_tokens(backend: str) -> set[str]:
                    texts = " ".join(
                        c.get("text", "")[:200]
                        for c in parity_search_results[backend].get("chunks", [])[:3]
                    )
                    return set(texts.lower().split())

                tokens_1 = _top3_tokens(b1)
                tokens_2 = _top3_tokens(b2)

                if not tokens_1 or not tokens_2:
                    continue

                union = tokens_1 | tokens_2
                overlap = len(tokens_1 & tokens_2) / max(len(union), 1)
                assert overlap >= 0.20, (
                    f"{b1} vs {b2}: token overlap {overlap:.1%} < 20% — "
                    "backends may be querying different or empty collections."
                )
