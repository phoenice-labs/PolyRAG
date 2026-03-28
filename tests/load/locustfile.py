"""
PolyRAG Load Test — Locust scenario file.

Tests the scalability fixes shipped in PRs #1–#3 under realistic concurrent load:
  - PR #1: async /api/rag, rate limiting, multi-worker
  - PR #2: LRU pipeline cache, streaming chunker
  - PR #3: system endpoints, ScaleHints

Usage:
  # Install Locust (one-time):
  pip install locust

  # Run headless (CI / quick validation):
  locust -f tests/load/locustfile.py --headless -u 20 -r 5 -t 60s \
         --host http://localhost:8000 --html tests/load/report.html

  # Run with web UI (interactive):
  locust -f tests/load/locustfile.py --host http://localhost:8000
  # Then open http://localhost:8089

  # Or use the convenience script:
  .\scripts\load-test.ps1

Prerequisites:
  1. PolyRAG API running: uvicorn api.main:app --port 8000
  2. Data ingested: POST /api/ingest (or use Shakespeare corpus via pipeline.ingest_gutenberg())
  3. Locust installed: pip install locust>=2.24

Scenarios (weighted):
  70% — POST /api/rag        (primary agentic endpoint — most important to validate)
  15% — POST /api/search     (multi-backend search)
   5% — GET  /api/system/health  (monitoring probe)
   5% — GET  /api/system/cache   (cache inspection)
   5% — GET  /api/health         (liveness probe)

Target thresholds (validated by load-test.ps1):
  p95 latency  < 5s   for /api/rag
  p50 latency  < 2s   for /api/rag
  error rate   < 1%   (excluding 429 rate-limit responses, which are expected)
"""
from __future__ import annotations

import json
import random
from locust import HttpUser, task, between, events


# ── Sample queries drawn from the Shakespeare corpus ──────────────────────────
_QUERIES = [
    "What does Hamlet say about death and mortality?",
    "How does Ophelia go mad in the play?",
    "Describe the relationship between Macbeth and Lady Macbeth.",
    "What is the significance of the ghost in Hamlet?",
    "How does Shakespeare portray jealousy in Othello?",
    "What happens in the final act of King Lear?",
    "Who is Portia in The Merchant of Venice?",
    "Describe the role of the witches in Macbeth.",
    "What themes are explored in A Midsummer Night's Dream?",
    "How does Romeo and Juliet end?",
]

_BACKENDS = ["chromadb", "faiss"]
_COLLECTION = "polyrag_docs_minilm"


def _rand_query() -> str:
    return random.choice(_QUERIES)


# ── User behaviour ────────────────────────────────────────────────────────────

class PolyRAGUser(HttpUser):
    """
    Simulates a developer / agentic AI process querying PolyRAG.

    wait_time: each virtual user waits 1–3s between tasks
    (simulates realistic think time / LLM processing on the client side)
    """
    wait_time = between(1, 3)

    # ── Primary agentic endpoint (70% of traffic) ─────────────────────────────
    @task(70)
    def rag_query(self):
        """POST /api/rag — unified production endpoint."""
        payload = {
            "query": _rand_query(),
            "backend": random.choice(_BACKENDS),
            "collection_name": _COLLECTION,
            "top_k": 5,
            "methods": {
                "enable_dense": True,
                "enable_bm25": True,
                "enable_graph": False,     # keep fast for load test
                "enable_rerank": True,
                "enable_mmr": True,
                "enable_rewrite": False,
                "enable_multi_query": False,
                "enable_hyde": False,
                "enable_raptor": False,
                "enable_contextual_rerank": False,
                "enable_llm_graph": False,
                "enable_splade": False,
            },
        }
        with self.client.post(
            "/api/rag",
            json=payload,
            catch_response=True,
            name="/api/rag",
        ) as resp:
            if resp.status_code == 429:
                # Rate limit — expected under high load, not an error
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"Server error: {resp.status_code} — {resp.text[:200]}")
            else:
                try:
                    body = resp.json()
                    if "answer" not in body:
                        resp.failure(f"Missing 'answer' in response: {list(body.keys())}")
                    else:
                        resp.success()
                except (json.JSONDecodeError, Exception) as exc:
                    resp.failure(f"Non-JSON response: {exc}")

    # ── Multi-backend search (15% of traffic) ─────────────────────────────────
    @task(15)
    def search(self):
        """POST /api/search — parallel multi-backend search."""
        payload = {
            "query": _rand_query(),
            "backends": [random.choice(_BACKENDS)],
            "collection_name": _COLLECTION,
            "top_k": 5,
            "methods": {
                "enable_dense": True,
                "enable_bm25": True,
                "enable_graph": False,
                "enable_rerank": False,
                "enable_mmr": True,
                "enable_rewrite": False,
                "enable_multi_query": False,
                "enable_hyde": False,
                "enable_raptor": False,
                "enable_contextual_rerank": False,
                "enable_llm_graph": False,
                "enable_splade": False,
            },
        }
        with self.client.post(
            "/api/search",
            json=payload,
            catch_response=True,
            name="/api/search",
        ) as resp:
            if resp.status_code == 429:
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"Server error: {resp.status_code}")
            else:
                resp.success()

    # ── System health probe (5% of traffic) ───────────────────────────────────
    @task(5)
    def system_health(self):
        """GET /api/system/health — monitoring probe."""
        with self.client.get(
            "/api/system/health",
            catch_response=True,
            name="/api/system/health",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if body.get("status") != "ok":
                    resp.failure(f"Health status not ok: {body}")
                else:
                    resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    # ── Cache inspection (5% of traffic) ──────────────────────────────────────
    @task(5)
    def cache_info(self):
        """GET /api/system/cache — LRU cache inspection."""
        with self.client.get(
            "/api/system/cache",
            catch_response=True,
            name="/api/system/cache",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Cache endpoint failed: {resp.status_code}")

    # ── Liveness probe (5% of traffic) ────────────────────────────────────────
    @task(5)
    def health(self):
        """GET /api/health — basic liveness."""
        with self.client.get(
            "/api/health",
            catch_response=True,
            name="/api/health",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Liveness failed: {resp.status_code}")


# ── Threshold validation (printed after headless run) ────────────────────────

@events.quitting.add_listener
def check_thresholds(environment, **kwargs):
    """
    Fail the load test if key thresholds are breached.
    Exits with code 1 when run headless (CI mode).
    """
    stats = environment.runner.stats

    failures = []

    rag_stats = stats.get("/api/rag", "POST")
    if rag_stats and rag_stats.num_requests > 0:
        p95 = rag_stats.get_response_time_percentile(0.95)
        p50 = rag_stats.get_response_time_percentile(0.50)
        err_rate = rag_stats.fail_ratio

        print(f"\n── /api/rag thresholds ──────────────────────────")
        print(f"  p50:       {p50:.0f}ms   (target < 2000ms)")
        print(f"  p95:       {p95:.0f}ms   (target < 5000ms)")
        print(f"  error rate: {err_rate*100:.2f}%  (target < 1%)")

        if p95 > 5000:
            failures.append(f"/api/rag p95={p95:.0f}ms exceeds 5000ms target")
        if p50 > 2000:
            failures.append(f"/api/rag p50={p50:.0f}ms exceeds 2000ms target")
        if err_rate > 0.01:
            failures.append(f"/api/rag error rate {err_rate*100:.2f}% exceeds 1% target")

    if failures:
        print("\n❌ LOAD TEST THRESHOLDS BREACHED:")
        for f in failures:
            print(f"   • {f}")
        environment.process_exit_code = 1
    else:
        print("\n✅ All load test thresholds passed")
        environment.process_exit_code = 0
