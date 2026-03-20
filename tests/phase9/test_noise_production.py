"""
Phase 9 – Noise Control & Production Hardening Tests
======================================================
Run:  pytest tests/phase9/ -v
"""
from __future__ import annotations

import time

import pytest

from core.store.models import Document, SearchResult
from core.noise.filters import (
    DuplicateDetector,
    QualityScorer,
    ConflictResolver,
    NoiseFilterPipeline,
)
from core.observability.logging import StructuredLogger, PipelineMetrics


def _res(i, text, score=0.8, rank=None):
    return SearchResult(
        document=Document(id=f"d{i}", text=text, embedding=[], metadata={}),
        score=score, rank=rank or i+1,
    )


# ── Duplicate Detector ────────────────────────────────────────────────────────

class TestDuplicateDetector:
    def test_exact_duplicates_removed(self):
        det = DuplicateDetector(similarity_threshold=0.85)
        text = "To be or not to be, that is the question."
        results = [_res(0, text, 0.9), _res(1, text, 0.8), _res(2, "Different content here.", 0.7)]
        unique = det.deduplicate(results)
        assert len(unique) == 2

    def test_unique_results_preserved(self):
        det = DuplicateDetector(similarity_threshold=0.85)
        results = [
            _res(0, "Hamlet ponders existence and mortality in Elsinore."),
            _res(1, "Romeo falls in love with Juliet at the Capulet feast."),
            _res(2, "Macbeth kills King Duncan driven by Lady Macbeth's ambition."),
            _res(3, "Othello is manipulated by Iago into jealous rage."),
        ]
        unique = det.deduplicate(results)
        assert len(unique) == 4

    def test_empty_input(self):
        det = DuplicateDetector()
        assert det.deduplicate([]) == []

    def test_ranks_renumbered(self):
        det = DuplicateDetector()
        text = "same exact text"
        results = [_res(0, text), _res(1, text), _res(2, "other")]
        unique = det.deduplicate(results)
        assert [r.rank for r in unique] == list(range(1, len(unique)+1))

    def test_near_duplicates_caught(self):
        det = DuplicateDetector(similarity_threshold=0.7)
        base = "Hamlet questions the meaning of existence and whether life is worth living."
        near_dup = "Hamlet questions the meaning of existence and whether life is worth living!"
        results = [_res(0, base), _res(1, near_dup), _res(2, "Completely different text here.")]
        unique = det.deduplicate(results)
        assert len(unique) <= 2


# ── Quality Scorer ────────────────────────────────────────────────────────────

class TestQualityScorer:
    def test_good_text_high_score(self):
        scorer = QualityScorer()
        text = (
            "Hamlet, Prince of Denmark, contemplates the nature of existence "
            "and mortality in his famous soliloquy, weighing the merits of action "
            "versus passive suffering."
        )
        score = scorer.score(text)
        assert score >= 0.5

    def test_empty_text_zero_score(self):
        scorer = QualityScorer()
        assert scorer.score("") == 0.0

    def test_short_text_low_score(self):
        scorer = QualityScorer()
        assert scorer.score("Hi") < 0.5

    def test_repetitive_text_low_score(self):
        scorer = QualityScorer()
        repetitive = " ".join(["word"] * 50)
        assert scorer.score(repetitive) < 0.7

    def test_score_in_range(self):
        scorer = QualityScorer()
        for text in ["hello", "test text here", "A" * 500, ""]:
            s = scorer.score(text)
            assert 0.0 <= s <= 1.0

    def test_filter_removes_low_quality(self):
        scorer = QualityScorer(threshold=0.5)
        results = [
            _res(0, "Hi", 0.9),         # too short
            _res(1, "Hamlet is a complex Shakespearean tragedy exploring themes of revenge, "
                    "mortality, and existential despair in Elsinore castle.", 0.8),
            _res(2, "word " * 30, 0.7),  # repetitive
        ]
        filtered = scorer.filter_results(results)
        ids = [r.document.id for r in filtered]
        assert "d1" in ids   # good text kept
        assert "d0" not in ids  # "Hi" removed


# ── Conflict Resolver ─────────────────────────────────────────────────────────

class TestConflictResolver:
    def test_no_conflict_passes_through(self):
        resolver = ConflictResolver()
        results = [
            _res(0, "Hamlet loved Ophelia.", 0.9),
            _res(1, "Hamlet cared deeply for Ophelia.", 0.8),
        ]
        final, conflict, msg = resolver.resolve(results, agreement_score=0.9)
        assert conflict is False
        assert len(final) == 2

    def test_conflict_adds_labels(self):
        resolver = ConflictResolver()
        results = [
            _res(0, "Hamlet loved Ophelia truly.", 0.9),
            _res(1, "Hamlet never loved Ophelia — it was pretense.", 0.8),
        ]
        final, conflict, msg = resolver.resolve(results, agreement_score=0.2)
        assert conflict is True
        labels = [r.document.metadata.get("conflict_label") for r in final]
        assert "PERSPECTIVE_A" in labels
        assert "PERSPECTIVE_B" in labels

    def test_conflict_explanation_non_empty(self):
        resolver = ConflictResolver()
        results = [_res(i, f"passage {i}", 0.8) for i in range(3)]
        _, conflict, msg = resolver.resolve(results, agreement_score=0.1)
        if conflict:
            assert len(msg) > 10


# ── Noise Filter Pipeline ─────────────────────────────────────────────────────

class TestNoiseFilterPipeline:
    def test_pipeline_produces_report(self):
        pipeline = NoiseFilterPipeline(quality_threshold=0.3)
        results = [
            _res(0, "Hamlet, Prince of Denmark, contemplates life and death with great philosophical depth.", 0.9),
            _res(1, "Hi", 0.7),
            _res(2, "Hamlet, Prince of Denmark, contemplates life and death with great philosophical depth.", 0.6),
        ]
        final, report = pipeline.run(results, agreement_score=0.9)
        assert "input_count" in report
        assert report["input_count"] == 3
        assert isinstance(final, list)

    def test_pipeline_deduplicates_and_filters(self):
        pipeline = NoiseFilterPipeline(quality_threshold=0.3)
        dup = "Repeated text content over and over."
        results = [
            _res(0, dup),
            _res(1, dup),
            _res(2, "Unique high quality passage about Hamlet's famous soliloquy on existence."),
        ]
        final, report = pipeline.run(results)
        ids = [r.document.id for r in final]
        # At least one duplicate removed
        assert len(final) <= 2

    def test_pipeline_report_counts(self):
        pipeline = NoiseFilterPipeline(quality_threshold=0.5)
        results = [_res(i, "Good quality text about Hamlet's soliloquy and philosophical musings.") for i in range(5)]
        _, report = pipeline.run(results)
        assert report["input_count"] == 5
        assert report["final_count"] <= 5


# ── Observability ─────────────────────────────────────────────────────────────

class TestStructuredLogger:
    def test_logs_without_error(self, capsys):
        logger = StructuredLogger("test.logger")
        logger.info("test message", key="value")
        captured = capsys.readouterr()
        import json
        entry = json.loads(captured.out.strip())
        assert entry["message"] == "test message"
        assert entry["key"] == "value"
        assert entry["level"] == "INFO"

    def test_correlation_id_in_log(self, capsys):
        logger = StructuredLogger("test.logger2")
        cid = logger.new_correlation_id()
        logger.info("with correlation")
        captured = capsys.readouterr()
        import json
        entry = json.loads(captured.out.strip())
        assert entry.get("correlation_id") == cid

    def test_timed_context_manager(self, capsys):
        logger = StructuredLogger("test.timer")
        with logger.timed("my_operation"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        import json
        entry = json.loads(captured.out.strip())
        assert "duration_ms" in entry
        assert entry["operation"] == "my_operation"


class TestPipelineMetrics:
    def test_increment(self):
        m = PipelineMetrics()
        m.increment("requests")
        m.increment("requests")
        summary = m.summary()
        assert summary["counters"]["requests"] == 2

    def test_record_histogram(self):
        m = PipelineMetrics()
        for v in [100.0, 200.0, 300.0]:
            m.record("latency_ms", v)
        summary = m.summary()
        hist = summary["histograms"]["latency_ms"]
        assert hist["count"] == 3
        assert hist["mean_ms"] == pytest.approx(200.0, abs=1.0)

    def test_reset(self):
        m = PipelineMetrics()
        m.increment("x")
        m.reset()
        assert m.summary()["counters"] == {}
