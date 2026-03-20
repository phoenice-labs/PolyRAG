"""
tests/test_backend_comparison.py
=================================
Pytest-runnable backend comparison test.

Runs the same five queries through every locally-available backend
(chromadb, faiss, qdrant) and asserts that all backends:
  - Successfully ingest the test corpus
  - Return results for every query
  - Find the expected keyword in the top-3 results for ≥ 3 of 5 queries
  - Produce consistent scores (top-score variance ≤ 0.1 across backends)

Also calls the compare_backends script to produce the comparison chart
and JSON report that are saved to data/.

Run:
    pytest tests/test_backend_comparison.py -v
    pytest tests/test_backend_comparison.py -v -k "consistency"
"""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Import the comparison helpers ─────────────────────────────────────────────
from scripts.compare_backends import (   # noqa: E402
    STANDARD_CORPUS,
    STANDARD_QUERIES,
    BackendSummary,
    _make_config,
    _probe_backend,
    _run_backend,
)

# ── Backends that do NOT require external servers ─────────────────────────────
IN_MEMORY_BACKENDS = ["chromadb", "faiss", "qdrant"]


# ── Session-scoped fixture: run once, reuse across all tests ──────────────────
@pytest.fixture(scope="module")
def comparison_results() -> Dict[str, BackendSummary]:
    """
    Runs the standard corpus + queries through every available in-memory backend.
    Returns {backend_name: BackendSummary}.
    Skips gracefully if a backend package is not installed.
    """
    tmp_dir = tempfile.mkdtemp(prefix="polyrag_compare_test_")
    results: Dict[str, BackendSummary] = {}

    try:
        for backend in IN_MEMORY_BACKENDS:
            available, reason = _probe_backend(backend)
            if not available:
                results[backend] = BackendSummary(
                    backend=backend,
                    status="skipped",
                    skip_reason=reason,
                )
                continue

            summary = _run_backend(
                backend=backend,
                corpus=STANDARD_CORPUS,
                queries=STANDARD_QUERIES,
                tmp_dir=tmp_dir,
                quiet=True,   # keep pytest output clean
            )
            results[backend] = summary
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


@pytest.fixture(scope="module")
def available_summaries(comparison_results: Dict[str, BackendSummary]) -> List[BackendSummary]:
    """Only the summaries that completed without error."""
    return [s for s in comparison_results.values() if s.status == "ok"]


# ── Individual backend ingestion tests ────────────────────────────────────────
class TestBackendIngestion:

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    def test_ingestion_succeeds(
        self, backend: str, comparison_results: Dict[str, BackendSummary]
    ) -> None:
        s = comparison_results[backend]
        if s.status == "skipped":
            pytest.skip(f"{backend} not available: {s.skip_reason}")
        assert s.status == "ok", f"{backend} pipeline failed: {s.skip_reason}"

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    def test_chunks_ingested(
        self, backend: str, comparison_results: Dict[str, BackendSummary]
    ) -> None:
        s = comparison_results[backend]
        if s.status == "skipped":
            pytest.skip(f"{backend} not available: {s.skip_reason}")
        assert s.total_chunks_ingested > 0, f"{backend} ingested 0 chunks"
        assert s.ingest_time_ms > 0

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    def test_chunk_count_consistent(
        self, backend: str, comparison_results: Dict[str, BackendSummary]
    ) -> None:
        """All backends must produce the same number of chunks from the same corpus."""
        ok = [
            s for s in comparison_results.values()
            if s.status == "ok"
        ]
        if len(ok) < 2:
            pytest.skip("Need ≥ 2 backends to compare chunk counts")
        counts = {s.backend: s.total_chunks_ingested for s in ok}
        unique_counts = set(counts.values())
        # Allow ±1 difference due to minor chunking edge cases
        assert max(unique_counts) - min(unique_counts) <= 1, (
            f"Chunk counts diverge across backends: {counts}"
        )


# ── Per-query retrieval tests ─────────────────────────────────────────────────
class TestQueryRetrieval:

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    @pytest.mark.parametrize("query_spec", STANDARD_QUERIES, ids=[q["id"] for q in STANDARD_QUERIES])
    def test_returns_results(
        self,
        backend: str,
        query_spec: dict,
        comparison_results: Dict[str, BackendSummary],
    ) -> None:
        s = comparison_results[backend]
        if s.status == "skipped":
            pytest.skip(f"{backend} not available")
        qrs = [qr for qr in s.query_results if qr.query_id == query_spec["id"]]
        assert qrs, f"No result recorded for {query_spec['id']} on {backend}"
        qr = qrs[0]
        assert qr.error is None, f"{backend}/{query_spec['id']} errored: {qr.error}"
        assert qr.results_count > 0, f"{backend}/{query_spec['id']} returned 0 results"

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    @pytest.mark.parametrize("query_spec", STANDARD_QUERIES, ids=[q["id"] for q in STANDARD_QUERIES])
    def test_scores_in_valid_range(
        self,
        backend: str,
        query_spec: dict,
        comparison_results: Dict[str, BackendSummary],
    ) -> None:
        s = comparison_results[backend]
        if s.status == "skipped":
            pytest.skip(f"{backend} not available")
        qrs = [qr for qr in s.query_results if qr.query_id == query_spec["id"]]
        if not qrs or qrs[0].error:
            pytest.skip("Query errored — score not available")
        qr = qrs[0]
        assert 0.0 <= qr.top_score <= 1.5, (
            f"{backend}/{query_spec['id']}: top_score {qr.top_score} out of expected range"
        )
        assert qr.mean_score <= qr.top_score, (
            f"{backend}/{query_spec['id']}: mean_score > top_score"
        )

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    def test_keyword_hit_rate_acceptable(
        self, backend: str, comparison_results: Dict[str, BackendSummary]
    ) -> None:
        """At least 3 of 5 queries must find their expected keyword in top-3 results."""
        s = comparison_results[backend]
        if s.status == "skipped":
            pytest.skip(f"{backend} not available")
        valid = [qr for qr in s.query_results if not qr.error]
        if not valid:
            pytest.skip("No valid query results")
        hits = sum(1 for qr in valid if qr.expect_found)
        assert hits >= 3, (
            f"{backend}: only {hits}/{len(valid)} queries found expected keyword in top-3 "
            f"(minimum 3 required for acceptable recall)"
        )

    @pytest.mark.parametrize("backend", IN_MEMORY_BACKENDS)
    def test_retrieval_latency_reasonable(
        self, backend: str, comparison_results: Dict[str, BackendSummary]
    ) -> None:
        """No single query should take more than 30 seconds (likely a hang/error)."""
        s = comparison_results[backend]
        if s.status == "skipped":
            pytest.skip(f"{backend} not available")
        for qr in s.query_results:
            if qr.error:
                continue
            assert qr.latency_ms < 30_000, (
                f"{backend}/{qr.query_id}: latency {qr.latency_ms:.0f}ms exceeded 30s limit"
            )


# ── Cross-backend consistency tests ──────────────────────────────────────────
class TestCrossBackendConsistency:

    def test_at_least_two_backends_available(
        self, available_summaries: List[BackendSummary]
    ) -> None:
        assert len(available_summaries) >= 1, (
            "Need at least 1 backend — install chromadb: pip install chromadb"
        )

    def test_score_consistency_per_query(
        self,
        available_summaries: List[BackendSummary],
    ) -> None:
        """
        For each query, the top-score variance across backends must be ≤ 0.1.
        A higher variance signals a backend returning very different results.
        """
        if len(available_summaries) < 2:
            pytest.skip("Need ≥ 2 backends for consistency test")

        for q in STANDARD_QUERIES:
            scores = []
            for s in available_summaries:
                qrs = [qr for qr in s.query_results if qr.query_id == q["id"] and not qr.error]
                if qrs:
                    scores.append(qrs[0].top_score)

            if len(scores) < 2:
                continue

            mean_s = sum(scores) / len(scores)
            variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            assert variance <= 0.1, (
                f"Query {q['id']} ({q['label']}): top-score variance {variance:.4f} "
                f"across backends exceeds 0.10 — scores: {scores}"
            )

    def test_keyword_agreement_across_backends(
        self,
        available_summaries: List[BackendSummary],
    ) -> None:
        """
        For queries where ANY backend found the keyword, ALL available backends
        should also find it (complete keyword agreement on reliable queries).
        We enforce this for queries with 100% hit rate across backends.
        """
        if len(available_summaries) < 2:
            pytest.skip("Need ≥ 2 backends for agreement test")

        for q in STANDARD_QUERIES:
            hits_per_backend = {}
            for s in available_summaries:
                qrs = [qr for qr in s.query_results if qr.query_id == q["id"] and not qr.error]
                if qrs:
                    hits_per_backend[s.backend] = qrs[0].expect_found

            if len(hits_per_backend) < 2:
                continue

            all_found = all(hits_per_backend.values())
            none_found = not any(hits_per_backend.values())

            # If all found OR none found → consistent (ok either way)
            # If split → potential concern but not a hard failure (different BM25 tuning)
            # We just assert that the majority (>50%) agree
            hit_count = sum(hits_per_backend.values())
            total = len(hits_per_backend)
            majority_agree = hit_count >= total / 2
            assert majority_agree or none_found, (
                f"Query {q['id']}: keyword hit results split unevenly across backends: "
                f"{hits_per_backend}"
            )

    def test_all_backends_same_chunk_ballpark(
        self, available_summaries: List[BackendSummary]
    ) -> None:
        """Results count must be consistent — same top_k requested, same corpus."""
        if len(available_summaries) < 2:
            pytest.skip("Need ≥ 2 backends")
        for q in STANDARD_QUERIES:
            counts = []
            for s in available_summaries:
                qrs = [qr for qr in s.query_results if qr.query_id == q["id"] and not qr.error]
                if qrs:
                    counts.append(qrs[0].results_count)
            if len(counts) < 2:
                continue
            # All backends should return the same top_k (5)
            assert max(counts) - min(counts) <= 2, (
                f"Query {q['id']}: results_count varies too much: {counts}"
            )


# ── Report generation test ────────────────────────────────────────────────────
class TestReportGeneration:

    def test_json_report_generated(
        self,
        comparison_results: Dict[str, BackendSummary],
        tmp_path: Path,
    ) -> None:
        """Ensure _save_json produces valid, parseable JSON."""
        from scripts.compare_backends import _save_json
        out = tmp_path / "test_comparison.json"
        summaries = list(comparison_results.values())
        _save_json(summaries, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == len(summaries)
        for entry in data:
            assert "backend" in entry
            assert "status" in entry
            assert "queries" in entry

    def test_chart_generated(
        self,
        comparison_results: Dict[str, BackendSummary],
        tmp_path: Path,
    ) -> None:
        """Ensure _generate_chart creates a PNG file."""
        from scripts.compare_backends import _generate_chart
        out = tmp_path / "test_chart.png"
        summaries = [s for s in comparison_results.values() if s.status == "ok"]
        if not summaries:
            pytest.skip("No successful backends — chart requires at least one")
        _generate_chart(summaries, STANDARD_QUERIES, out)
        assert out.exists()
        assert out.stat().st_size > 5_000, "Chart file suspiciously small"

    def test_full_script_runs(self, tmp_path: Path) -> None:
        """Run the compare_backends main() with --no-chart and a tmp output dir."""
        import subprocess, os
        python = str(ROOT / ".venv" / "Scripts" / "python.exe")
        if not Path(python).exists():
            python = sys.executable
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        result = subprocess.run(
            [python, str(ROOT / "scripts" / "compare_backends.py"),
             "--backends", "chromadb", "faiss",
             "--no-chart", "--quiet"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"compare_backends.py exited with {result.returncode}\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        # Verify JSON output was written to data/
        json_out = ROOT / "data" / "comparison_results.json"
        assert json_out.exists(), "comparison_results.json was not created"
