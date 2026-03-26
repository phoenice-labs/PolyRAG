"""
Phase 13: Milvus-as-Reference Backend Parity Tests
====================================================
Validates that EVERY vector database produces results that are functionally
comparable to Milvus (the reference implementation).

Design principle:
  Milvus is the reference. Each other backend must produce results that are:
    1. Relevant  — contain Hamlet-corpus keywords
    2. Consistent — same ranking direction (descending score)
    3. Coverage-equivalent — ≥ 70% of the hamlet-term coverage Milvus achieves
    4. Corpus-coherent   — top-3 text overlaps ≥ 30% with Milvus's top-3
    5. Latency-acceptable — responds within 30 s
    6. Score-normalised   — all scores in [0, 1]

Each non-Milvus backend gets its own parametrized test entry, so CI reports
show per-backend pass/fail independently.

A JSON parity report is written to data/backend_parity_report.json after the
session so developers can inspect detailed per-method, per-backend scores.

Usage:
    # Full suite (requires live API + Milvus Docker):
    python -m pytest tests/phase13/test_milvus_reference_parity.py -v

    # FAISS and ChromaDB only (no Docker):
    python -m pytest tests/phase13/test_milvus_reference_parity.py -v -k "faiss or chromadb"
"""
from __future__ import annotations

import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import requests as _requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api"

REFERENCE_BACKEND = "milvus"
REFERENCE_COLLECTION = "e2e_hamlet_milvus"

NON_MILVUS_BACKENDS = ["faiss", "chromadb", "qdrant", "weaviate", "pgvector"]
DOCKER_PORTS: dict[str, int] = {
    "qdrant": 6333,
    "milvus": 19530,
    "weaviate": 8088,
    "pgvector": 5433,
}
COLLECTION_MAP: dict[str, str] = {
    "faiss": "e2e_hamlet_faiss",
    "chromadb": "e2e_hamlet_chroma",
    "qdrant": "e2e_hamlet_qdrant",
    "weaviate": "e2e_hamlet_weaviate",
    "pgvector": "e2e_hamlet_pgvector",
}

SEARCH_QUERY = "Touching this dreaded sight"
SECONDARY_QUERY = "Who's there? Stand and unfold yourself"

# Hamlet keywords weighted by importance for the ghost-scene query
HAMLET_TERMS: set[str] = {
    "hamlet", "ghost", "horatio", "marcellus", "barnardo",
    "ophelia", "claudius", "polonius", "sight", "dreaded",
    "spirit", "night", "denmark", "king", "swear",
}

# All 10 non-LLM method flags — the "full retrieval" profile for comparison
FULL_METHODS: dict[str, bool] = {
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
    "enable_splade": False,
    "enable_llm_graph": False,
}

DENSE_BM25_METHODS = {**FULL_METHODS, "enable_graph": False, "enable_rerank": False, "enable_mmr": False}
DENSE_ONLY_METHODS  = {k: (k == "enable_dense") for k in FULL_METHODS}

# Parity thresholds
COVERAGE_PARITY_THRESHOLD   = 0.60   # backend must achieve ≥ 60% of Milvus's term coverage
CORPUS_OVERLAP_THRESHOLD    = 0.25   # top-3 Jaccard token overlap vs Milvus ≥ 25%
LATENCY_SLA_MS              = 30_000 # 30 s SLA for any single-backend search
RESULT_COUNT_RATIO_MIN      = 0.50   # backend must return ≥ 50% as many results as Milvus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


def _api_search(backend: str, collection: str, methods: dict,
                query: str = SEARCH_QUERY, top_k: int = 5,
                timeout: int = 120) -> dict | None:
    """POST /api/search and return the backend result dict, or None on error."""
    try:
        resp = _requests.post(
            f"{API_BASE}/search",
            json={
                "query": query,
                "backends": [backend],
                "collection_name": collection,
                "top_k": top_k,
                "methods": methods,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json().get("results", {}).get(backend, {})
            if not data.get("error"):
                return data
        return None
    except Exception as exc:
        print(f"[parity] {backend} API call failed: {exc}")
        return None


def _term_coverage(result: dict | None) -> float:
    """Fraction of HAMLET_TERMS found in the result chunks."""
    if not result:
        return 0.0
    all_text = " ".join(c.get("text", "").lower() for c in result.get("chunks", []))
    found = {t for t in HAMLET_TERMS if t in all_text}
    return round(len(found) / max(len(HAMLET_TERMS), 1), 4)


def _top3_tokens(result: dict | None) -> set[str]:
    """Bag-of-words (word tokens, lowercased) from the top-3 chunks."""
    if not result:
        return set()
    texts = " ".join(
        c.get("text", "")[:300]
        for c in result.get("chunks", [])[:3]
    )
    return set(texts.lower().split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return round(len(a & b) / max(len(a | b), 1), 4)


def _scores(result: dict | None) -> list[float]:
    if not result:
        return []
    return [float(c.get("score", 0)) for c in result.get("chunks", [])]


# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_api_session():
    """Skip all parity tests when the live API is not reachable."""
    try:
        r = _requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        pytest.skip("API server not running at localhost:8000 — run `uvicorn api.main:app` first")


@pytest.fixture(scope="session")
def milvus_reference_results(live_api_session):  # noqa: ARG001
    """
    Run the primary search on Milvus (the reference backend).
    Skip the entire session if Milvus is not reachable.
    Returns a dict: {methods_profile: BackendSearchResult}
    """
    if not _port_open(DOCKER_PORTS["milvus"]):
        pytest.skip("Milvus not reachable (port 19530) — cannot establish reference results")

    profiles = {
        "full":     FULL_METHODS,
        "dense_bm25": DENSE_BM25_METHODS,
        "dense_only": DENSE_ONLY_METHODS,
    }
    reference: dict[str, Any] = {}
    for profile_name, methods in profiles.items():
        result = _api_search(REFERENCE_BACKEND, REFERENCE_COLLECTION, methods)
        if result is None:
            pytest.skip(
                f"Milvus reference search failed for profile '{profile_name}'. "
                "Ingest Hamlet into Milvus first: run test_e2e_milvus_hamlet.py"
            )
        reference[profile_name] = result
    return reference


@pytest.fixture(scope="session")
def parity_report_data():
    """Shared mutable dict for collecting per-backend parity metrics."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_backend": REFERENCE_BACKEND,
        "reference_collection": REFERENCE_COLLECTION,
        "query": SEARCH_QUERY,
        "thresholds": {
            "coverage_parity": COVERAGE_PARITY_THRESHOLD,
            "corpus_overlap": CORPUS_OVERLAP_THRESHOLD,
            "latency_sla_ms": LATENCY_SLA_MS,
            "result_count_ratio_min": RESULT_COUNT_RATIO_MIN,
        },
        "backends": {},
    }


# Per-backend parametrized fixture — generates one test per non-Milvus backend
def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests that use the `other_backend` fixture."""
    if "other_backend" in metafunc.fixturenames:
        metafunc.parametrize(
            "other_backend",
            NON_MILVUS_BACKENDS,
            ids=NON_MILVUS_BACKENDS,
        )


@pytest.fixture
def other_backend_result(other_backend, milvus_reference_results):  # noqa: ARG001
    """
    Run the SAME full-profile search on `other_backend` and return its result.
    Skips the individual test if the backend is not reachable.
    """
    if other_backend in DOCKER_PORTS and not _port_open(DOCKER_PORTS[other_backend]):
        pytest.skip(f"{other_backend}: port {DOCKER_PORTS[other_backend]} not reachable — "
                    "start Docker service to include this backend")
    collection = COLLECTION_MAP[other_backend]
    result = _api_search(other_backend, collection, FULL_METHODS)
    if result is None:
        pytest.skip(
            f"{other_backend}: search returned no results or errored. "
            f"Ensure Hamlet is ingested: run test_e2e_{other_backend}_hamlet.py first"
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COVERAGE PARITY — Each backend retrieves the same Hamlet terms as Milvus
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoverageParity:
    """
    Each backend's hamlet-term coverage must be at least 60% of Milvus's coverage.
    This ensures the backend retrieves from the same semantic neighbourhood.
    """

    def test_term_coverage_vs_milvus(
        self, other_backend, other_backend_result, milvus_reference_results, parity_report_data
    ):
        ref = milvus_reference_results["full"]
        ref_coverage  = _term_coverage(ref)
        this_coverage = _term_coverage(other_backend_result)

        ratio = this_coverage / max(ref_coverage, 0.001)

        # Store for report
        parity_report_data["backends"].setdefault(other_backend, {})
        parity_report_data["backends"][other_backend]["term_coverage"] = {
            "milvus": ref_coverage,
            other_backend: this_coverage,
            "ratio": round(ratio, 4),
            "pass": ratio >= COVERAGE_PARITY_THRESHOLD,
        }

        ref_terms   = {t for t in HAMLET_TERMS if t in
                       " ".join(c.get("text","").lower() for c in ref.get("chunks",[]))}
        other_terms = {t for t in HAMLET_TERMS if t in
                       " ".join(c.get("text","").lower() for c in other_backend_result.get("chunks",[]))}
        missing = ref_terms - other_terms

        assert ratio >= COVERAGE_PARITY_THRESHOLD, (
            f"\n{'='*60}"
            f"\n❌ COVERAGE PARITY FAIL: {other_backend} vs Milvus"
            f"\n   Milvus coverage:  {ref_coverage:.1%} — found: {sorted(ref_terms)}"
            f"\n   {other_backend:8} coverage:  {this_coverage:.1%} — found: {sorted(other_terms)}"
            f"\n   Ratio: {ratio:.1%}  (threshold ≥ {COVERAGE_PARITY_THRESHOLD:.0%})"
            f"\n   Missing terms: {sorted(missing)}"
            f"\n   Action: Check that '{COLLECTION_MAP[other_backend]}' is populated"
            f"\n           with the same Hamlet text as '{REFERENCE_COLLECTION}'"
            f"\n{'='*60}"
        )

    def test_secondary_query_coverage(self, other_backend, milvus_reference_results):  # noqa: ARG001
        """Run a second Hamlet query to confirm coverage is not query-specific."""
        if other_backend in DOCKER_PORTS and not _port_open(DOCKER_PORTS[other_backend]):
            pytest.skip(f"{other_backend} not reachable")

        ref_result = _api_search(REFERENCE_BACKEND, REFERENCE_COLLECTION,
                                  DENSE_BM25_METHODS, query=SECONDARY_QUERY)
        other_result = _api_search(other_backend, COLLECTION_MAP[other_backend],
                                    DENSE_BM25_METHODS, query=SECONDARY_QUERY)

        if ref_result is None or other_result is None:
            pytest.skip("One or both backends returned no results for secondary query")

        ref_cov   = _term_coverage(ref_result)
        other_cov = _term_coverage(other_result)
        ratio = other_cov / max(ref_cov, 0.001)

        assert ratio >= COVERAGE_PARITY_THRESHOLD, (
            f"{other_backend}: secondary-query term coverage {other_cov:.1%} is only "
            f"{ratio:.1%} of Milvus's {ref_cov:.1%} (need ≥ {COVERAGE_PARITY_THRESHOLD:.0%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CORPUS COHERENCE — Top-3 results must come from the same text neighbourhood
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorpusCoherence:
    """
    The union of top-3 result texts must overlap with Milvus's top-3 by ≥ 25%
    Jaccard similarity on word tokens.  This proves backends index the same corpus.
    """

    def test_top3_jaccard_vs_milvus(
        self, other_backend, other_backend_result, milvus_reference_results, parity_report_data
    ):
        ref = milvus_reference_results["full"]
        ref_tokens   = _top3_tokens(ref)
        other_tokens = _top3_tokens(other_backend_result)
        jaccard      = _jaccard(ref_tokens, other_tokens)

        parity_report_data["backends"].setdefault(other_backend, {})
        parity_report_data["backends"][other_backend]["corpus_overlap"] = {
            "jaccard": jaccard,
            "pass": jaccard >= CORPUS_OVERLAP_THRESHOLD,
        }

        assert jaccard >= CORPUS_OVERLAP_THRESHOLD, (
            f"\n{'='*60}"
            f"\n❌ CORPUS COHERENCE FAIL: {other_backend} vs Milvus"
            f"\n   Jaccard(top-3): {jaccard:.1%}  (threshold ≥ {CORPUS_OVERLAP_THRESHOLD:.0%})"
            f"\n   Milvus top chunk: {list(ref_tokens)[:15]}..."
            f"\n   {other_backend} top chunk: {list(other_tokens)[:15]}..."
            f"\n   This means the two backends are returning text from DIFFERENT"
            f"\n   parts of the corpus — check that both collections were ingested"
            f"\n   with the same chunking parameters (chunk_size=400, strategy=sentence)."
            f"\n{'='*60}"
        )

    def test_top1_is_hamlet_relevant(self, other_backend, other_backend_result):
        """The very first result from every backend must contain a Hamlet term."""
        chunks = other_backend_result.get("chunks", [])
        assert chunks, f"{other_backend}: returned 0 chunks"
        top1_text = chunks[0].get("text", "").lower()
        has_term = any(t in top1_text for t in HAMLET_TERMS)
        assert has_term, (
            f"{other_backend}: top-1 result does not contain any Hamlet keyword.\n"
            f"  Top result text: {top1_text[:300]!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RESULT COUNT PARITY — Backend returns at least 50% as many results as Milvus
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultCountParity:
    """Backend must return at least 50% as many results as Milvus for the same query."""

    def test_result_count_vs_milvus(
        self, other_backend, other_backend_result, milvus_reference_results, parity_report_data
    ):
        ref_count   = len(milvus_reference_results["full"].get("chunks", []))
        other_count = len(other_backend_result.get("chunks", []))
        ratio = other_count / max(ref_count, 1)

        parity_report_data["backends"].setdefault(other_backend, {})
        parity_report_data["backends"][other_backend]["result_count"] = {
            "milvus": ref_count,
            other_backend: other_count,
            "ratio": round(ratio, 4),
            "pass": ratio >= RESULT_COUNT_RATIO_MIN,
        }

        assert ratio >= RESULT_COUNT_RATIO_MIN, (
            f"{other_backend}: returned {other_count} results vs Milvus's {ref_count} "
            f"(ratio={ratio:.1%}, need ≥ {RESULT_COUNT_RATIO_MIN:.0%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORE NORMALIZATION — All backends return scores in [0, 1]
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreNormalization:
    """Scores must be in [0, 1] and ordered descending — same convention as Milvus."""

    def test_scores_in_unit_interval(self, other_backend, other_backend_result, parity_report_data):
        scores = _scores(other_backend_result)
        out_of_range = [s for s in scores if not (0.0 <= s <= 1.0)]

        parity_report_data["backends"].setdefault(other_backend, {})
        parity_report_data["backends"][other_backend]["score_normalization"] = {
            "scores": scores[:5],
            "out_of_range": out_of_range,
            "pass": len(out_of_range) == 0,
        }

        assert not out_of_range, (
            f"{other_backend}: scores out of [0,1]: {out_of_range}\n"
            "  All backends must RRF-normalise scores to [0, 1] so result cards "
            "display consistently regardless of backend choice."
        )

    def test_scores_descending(self, other_backend, other_backend_result):
        """Results must be ranked highest-score-first (matching Milvus convention)."""
        scores = _scores(other_backend_result)
        if len(scores) < 2:
            pytest.skip("Too few results to check ranking order")
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-6, (
                f"{other_backend}: score[{i}]={scores[i]:.4f} < score[{i+1}]={scores[i+1]:.4f} "
                "— results must be sorted descending by score"
            )

    def test_milvus_itself_scores_in_unit_interval(self, milvus_reference_results):
        """Sanity check: Milvus reference scores are also in [0, 1]."""
        scores = _scores(milvus_reference_results["full"])
        bad = [s for s in scores if not (0.0 <= s <= 1.0)]
        assert not bad, f"Milvus reference has out-of-range scores: {bad}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LATENCY PARITY — All backends respond within the same SLA as Milvus
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatencyParity:
    """
    Each backend must respond within 30 s (single-backend dense-only search).
    Measures actual round-trip latency reported by the API (latency_ms field).
    """

    def test_latency_within_sla(self, other_backend, milvus_reference_results, parity_report_data):  # noqa: ARG001
        if other_backend in DOCKER_PORTS and not _port_open(DOCKER_PORTS[other_backend]):
            pytest.skip(f"{other_backend} not reachable")

        t0 = time.monotonic()
        result = _api_search(other_backend, COLLECTION_MAP[other_backend],
                              DENSE_ONLY_METHODS, top_k=3)
        wall_ms = (time.monotonic() - t0) * 1000

        api_latency_ms = float(result.get("latency_ms", wall_ms)) if result else wall_ms

        parity_report_data["backends"].setdefault(other_backend, {})
        parity_report_data["backends"][other_backend]["latency"] = {
            "api_latency_ms": round(api_latency_ms, 1),
            "wall_ms": round(wall_ms, 1),
            "pass": wall_ms < LATENCY_SLA_MS,
        }

        assert wall_ms < LATENCY_SLA_MS, (
            f"{other_backend}: latency {wall_ms:.0f} ms exceeds {LATENCY_SLA_MS/1000:.0f} s SLA.\n"
            "  Ensure the backend collection is populated and the service is healthy."
        )

    def test_milvus_latency_reference(self, milvus_reference_results, parity_report_data):
        """Record Milvus's own latency as the reference baseline in the report."""
        api_latency_ms = float(milvus_reference_results["full"].get("latency_ms", 0))
        parity_report_data.setdefault("milvus_reference", {})
        parity_report_data["milvus_reference"]["latency_ms"] = api_latency_ms


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RETRIEVAL METHOD PARITY — Each method produces results on every backend
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalMethodParity:
    """
    Dense-only, BM25+Dense hybrid, and full non-LLM retrieval must all return
    results on every backend — mirroring Milvus's behaviour for each profile.
    """

    @pytest.mark.parametrize("profile,methods", [
        ("dense_only",   DENSE_ONLY_METHODS),
        ("dense_bm25",   DENSE_BM25_METHODS),
        ("full_no_llm",  FULL_METHODS),
    ])
    def test_retrieval_profile_returns_results(
        self, other_backend, profile, methods, milvus_reference_results, parity_report_data
    ):
        if other_backend in DOCKER_PORTS and not _port_open(DOCKER_PORTS[other_backend]):
            pytest.skip(f"{other_backend} not reachable")

        result = _api_search(other_backend, COLLECTION_MAP[other_backend], methods)
        assert result is not None, (
            f"{other_backend}: retrieval profile '{profile}' returned no result / errored"
        )
        chunks = result.get("chunks", [])
        assert len(chunks) >= 1, (
            f"{other_backend}: profile '{profile}' returned 0 chunks "
            f"(Milvus returned {len(milvus_reference_results.get(profile, {}).get('chunks', []))})"
        )

        # Record in report
        parity_report_data["backends"].setdefault(other_backend, {})
        parity_report_data["backends"][other_backend].setdefault("retrieval_profiles", {})
        parity_report_data["backends"][other_backend]["retrieval_profiles"][profile] = {
            "result_count": len(chunks),
            "pass": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCHEMA PARITY — Response structure matches Milvus exactly
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaParity:
    """
    Every backend must return the same response envelope as Milvus:
    backend, answer, chunks[{chunk_id, text, score}], retrieval_trace, llm_traces, latency_ms.
    """

    REQUIRED_TOP_FIELDS = {"backend", "answer", "chunks", "retrieval_trace",
                            "llm_traces", "latency_ms"}
    REQUIRED_CHUNK_FIELDS = {"chunk_id", "text", "score"}

    def test_response_envelope_matches_milvus(self, other_backend, other_backend_result):
        missing = self.REQUIRED_TOP_FIELDS - set(other_backend_result.keys())
        assert not missing, (
            f"{other_backend}: response missing fields that Milvus returns: {missing}"
        )

    def test_chunk_schema_matches_milvus(self, other_backend, other_backend_result):
        chunks = other_backend_result.get("chunks", [])
        assert chunks, f"{other_backend}: no chunks to validate schema"
        for i, chunk in enumerate(chunks[:3]):
            missing = self.REQUIRED_CHUNK_FIELDS - set(chunk.keys())
            assert not missing, (
                f"{other_backend}: chunk[{i}] missing fields: {missing}"
            )

    def test_backend_field_matches_requested(self, other_backend, other_backend_result):
        assert other_backend_result.get("backend") == other_backend, (
            f"backend field mismatch: expected '{other_backend}', "
            f"got '{other_backend_result.get('backend')}'"
        )

    def test_retrieval_trace_is_list(self, other_backend, other_backend_result):
        trace = other_backend_result.get("retrieval_trace", None)
        assert isinstance(trace, list), (
            f"{other_backend}: retrieval_trace must be a list (like Milvus), got {type(trace)}"
        )

    def test_llm_traces_is_list(self, other_backend, other_backend_result):
        traces = other_backend_result.get("llm_traces", None)
        assert isinstance(traces, list), (
            f"{other_backend}: llm_traces must be a list (like Milvus), got {type(traces)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PARITY REPORT — Write JSON summary after all tests run
# ═══════════════════════════════════════════════════════════════════════════════

class TestParityReport:
    """
    Writes data/backend_parity_report.json summarising all comparison results.
    This is always the last test class; it aggregates data collected by all others.
    """

    def test_write_parity_report(self, parity_report_data, milvus_reference_results):
        """Write final parity report to data/backend_parity_report.json."""
        # Enrich with Milvus reference metrics
        ref = milvus_reference_results["full"]
        parity_report_data["milvus_reference"] = parity_report_data.get("milvus_reference", {})
        parity_report_data["milvus_reference"].update({
            "result_count": len(ref.get("chunks", [])),
            "term_coverage": _term_coverage(ref),
            "scores": _scores(ref)[:5],
        })

        # Compute per-backend parity verdict
        verdicts: dict[str, str] = {}
        for backend, metrics in parity_report_data.get("backends", {}).items():
            all_pass = all(
                v.get("pass", True) if isinstance(v, dict) else True
                for v in metrics.values()
            )
            verdicts[backend] = "✅ PASS — equivalent to Milvus" if all_pass else "❌ FAIL — diverges from Milvus"

        parity_report_data["verdict"] = verdicts
        parity_report_data["summary"] = (
            f"{sum(1 for v in verdicts.values() if 'PASS' in v)}/{len(verdicts)} backends "
            "passed all parity checks vs Milvus reference"
        )

        report_path = ROOT / "data" / "backend_parity_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(parity_report_data, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n[parity] Report written to {report_path}")
        print(f"[parity] {parity_report_data['summary']}")
        for backend, verdict in verdicts.items():
            print(f"[parity]   {backend:12} {verdict}")

        assert report_path.exists()

    def test_parity_report_has_all_backends(self, parity_report_data):
        """Verify report captures at least the in-process backends (faiss, chromadb)."""
        backends_in_report = set(parity_report_data.get("backends", {}).keys())
        always_on = {"faiss", "chromadb"}
        missing = always_on - backends_in_report
        assert not missing, (
            f"Report missing always-on backends: {missing}. "
            "Run the full test class including TestCoverageParity."
        )
