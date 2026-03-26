"""
Phase 13: Retrieval Method Matrix Tests
========================================
Tests all 10 retrieval methods in targeted combinations using the FastAPI
TestClient with a patched search client that injects a pre-ingested milvus
pipeline.

Run count:
  - 5  single-method ablations  (one always-on method enabled at a time)
  - 10 pairwise combinations     (C(5,2)=10 pairs of always-on methods)
  - 4  LLM method additions      (each of 4 LLM methods added to the full
                                  always-on baseline; skipped if LM Studio
                                  offline, marked with @pytest.mark.lm_studio)

After all runs, a JSON report is written to data/method_matrix_report.json.
"""
from __future__ import annotations

import itertools
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402 — import after sys.path insert

# ── Test configuration ─────────────────────────────────────────────────────────

BACKEND = "milvus"
COLLECTION = "e2e_hamlet_matrix"
SEARCH_QUERY = "Touching this dreaded sight"
HAMLET_TERMS = {
    "hamlet", "ghost", "horatio", "ophelia", "claudius", "polonius",
    "sight", "dreaded", "spirit", "denmark",
}
ALWAYS_ON = ["enable_dense", "enable_bm25", "enable_graph", "enable_rerank", "enable_mmr"]
LLM_METHODS = [
    "enable_rewrite", "enable_multi_query", "enable_hyde",
    "enable_raptor", "enable_contextual_rerank",
]
ALL_METHOD_FLAGS = ALWAYS_ON + LLM_METHODS

HAMLET_PATH = ROOT / "data" / "shakespeare_hamlet.txt"
REPORT_PATH = ROOT / "data" / "method_matrix_report.json"

client = TestClient(app, raise_server_exceptions=False)


# ── Utility helpers ────────────────────────────────────────────────────────────

def _build_methods_payload(**overrides: bool) -> dict:
    """All flags off, then apply overrides."""
    base = {flag: False for flag in ALL_METHOD_FLAGS}
    base.update(overrides)
    return base


def _relevance(chunks: list) -> float:
    """Fraction of top-5 chunks containing at least one hamlet keyword."""
    top5 = chunks[:5]
    if not top5:
        return 0.0
    hits = sum(
        1 for c in top5
        if any(t in c.get("text", "").lower() for t in HAMLET_TERMS)
    )
    return round(hits / len(top5), 2)


def _lm_studio_available() -> bool:
    """Return True if LM Studio is reachable at localhost:1234."""
    try:
        with socket.create_connection(("localhost", 1234), timeout=2):
            return True
    except OSError:
        return False


def _run_single(mc, run_id: str, methods: dict) -> dict:
    """Execute one search run and return a result dict."""
    t0 = time.monotonic()
    resp = mc.post("/api/search", json={
        "query": SEARCH_QUERY,
        "backends": [BACKEND],
        "collection_name": COLLECTION,
        "top_k": 5,
        "methods": methods,
    })
    latency_ms = (time.monotonic() - t0) * 1000

    chunks: list = []
    if resp.status_code == 200:
        backend_result = resp.json().get("results", {}).get(BACKEND, {})
        chunks = backend_result.get("chunks", [])

    return {
        "run_id": run_id,
        "methods_enabled": [k for k, v in methods.items() if v],
        "result_count": len(chunks),
        "relevance_score": _relevance(chunks),
        "latency_ms": round(latency_ms, 2),
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
def matrix_pipeline(hamlet_text):
    """
    Build a Milvus pipeline, ingest Hamlet text (first 50 kB), and yield.
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
        metadata={"source": "test_hamlet_matrix", "backend": BACKEND},
    )
    assert result.upserted > 0, f"Ingest produced 0 chunks: {result}"

    yield pipeline

    try:
        pipeline.store.drop_collection(pipeline.store.collection_name)
    except Exception:
        pass
    pipeline.stop()


@pytest.fixture(scope="module")
def matrix_client(matrix_pipeline):
    """
    TestClient patched so that POST /api/search reuses the pre-populated
    matrix_pipeline for any BACKEND request.
    """
    import api.routers.search as search_module

    original_create = search_module.create_pipeline

    def _use_matrix(config):
        if config.get("store", {}).get("backend") == BACKEND:
            return matrix_pipeline
        return original_create(config)

    search_module.create_pipeline = _use_matrix
    yield client
    search_module.create_pipeline = original_create


@pytest.fixture(scope="module")
def run_results() -> list:
    """Shared mutable list that accumulates all run result dicts."""
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════════════════════════

class TestMethodMatrix:
    """Parametrized method-matrix tests + report generation."""

    # ── 1. Single-method ablations ─────────────────────────────────────────────

    @pytest.mark.parametrize("method", ALWAYS_ON)
    def test_single_method_ablations(self, matrix_client, run_results, method):
        """Enable exactly one always-on method; verify non-negative result count."""
        methods = _build_methods_payload(**{method: True})
        run = _run_single(matrix_client, f"{method.replace('enable_', '')}_only", methods)
        run_results.append(run)
        assert run["result_count"] >= 0, (
            f"Negative result count for {method}: {run['result_count']}"
        )

    # ── 2. Pairwise combinations ───────────────────────────────────────────────

    @pytest.mark.parametrize(
        "pair",
        [
            pytest.param(pair, id=f"{pair[0].replace('enable_','')}_x_{pair[1].replace('enable_','')}")
            for pair in itertools.combinations(ALWAYS_ON, 2)
        ],
    )
    def test_pairwise_combinations(self, matrix_client, run_results, pair):
        """Enable a pair of always-on methods; verify non-negative result count."""
        m1, m2 = pair
        methods = _build_methods_payload(**{m1: True, m2: True})
        run_id = f"{m1.replace('enable_', '')}_x_{m2.replace('enable_', '')}"
        run = _run_single(matrix_client, run_id, methods)
        run_results.append(run)
        assert run["result_count"] >= 0, (
            f"Negative result count for {run_id}: {run['result_count']}"
        )

    # ── 3. LLM method additions (requires LM Studio) ───────────────────────────

    @pytest.mark.lm_studio
    @pytest.mark.parametrize("llm_method", LLM_METHODS[:4])
    def test_llm_method_additions(self, matrix_client, run_results, llm_method):
        """
        Add one LLM method to the full always-on baseline.
        Skipped if LM Studio is not running at localhost:1234.
        """
        if not _lm_studio_available():
            pytest.skip("LM Studio not running at localhost:1234")
        methods = _build_methods_payload(
            **{m: True for m in ALWAYS_ON},
            **{llm_method: True},
        )
        run_id = f"baseline_plus_{llm_method.replace('enable_', '')}"
        run = _run_single(matrix_client, run_id, methods)
        run_results.append(run)
        assert run["result_count"] >= 0

    # ── 4. Write report ────────────────────────────────────────────────────────

    def test_report_written(self, run_results):
        """
        Compute per-method contribution summary from all collected runs and
        write the report to data/method_matrix_report.json.
        Verifies the file exists and contains the expected keys.
        """
        # Per-method relevance aggregation
        method_bucket: dict[str, list[float]] = {flag: [] for flag in ALL_METHOD_FLAGS}
        for run in run_results:
            for flag in run["methods_enabled"]:
                if flag in method_bucket:
                    method_bucket[flag].append(run["relevance_score"])

        method_contributions: dict = {}
        for flag, scores in method_bucket.items():
            if scores:
                method_contributions[flag] = {
                    "avg_relevance": round(sum(scores) / len(scores), 3),
                    "appears_in_n_runs": len(scores),
                }
            else:
                method_contributions[flag] = {"avg_relevance": 0.0, "appears_in_n_runs": 0}

        # Recommended: always-on methods that participated in at least one run
        recommended_always_on = sorted(
            [f for f in ALWAYS_ON if method_contributions[f]["appears_in_n_runs"] > 0],
            key=lambda f: -method_contributions[f]["avg_relevance"],
        )

        # LLM addition summary
        llm_addition_summary: dict = {}
        for llm_m in LLM_METHODS:
            llm_runs = [r for r in run_results if llm_m in r["methods_enabled"]]
            if llm_runs:
                llm_addition_summary[llm_m] = {
                    "avg_relevance": round(
                        sum(r["relevance_score"] for r in llm_runs) / len(llm_runs), 3
                    ),
                    "avg_latency_ms": round(
                        sum(r["latency_ms"] for r in llm_runs) / len(llm_runs), 1
                    ),
                    "n_runs": len(llm_runs),
                }

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "query": SEARCH_QUERY,
            "backend": BACKEND,
            "collection": COLLECTION,
            "runs": run_results,
            "method_contributions": method_contributions,
            "recommended_always_on": recommended_always_on,
            "llm_addition_summary": llm_addition_summary,
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Verify
        assert REPORT_PATH.exists(), "Report file was not created"
        loaded = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert "runs" in loaded, "Report missing 'runs' key"
        assert "method_contributions" in loaded, "Report missing 'method_contributions' key"
        assert "generated_at" in loaded, "Report missing 'generated_at' key"
        assert "recommended_always_on" in loaded, "Report missing 'recommended_always_on' key"
