"""
Phase 7 – Confidence Estimation Tests
========================================
Run:  pytest tests/phase7/ -v
"""
from __future__ import annotations

import pytest

from core.store.models import Document, SearchResult
from core.confidence.signals import (
    RetrievalScoreAnalyser,
    SourceAgreementScorer,
    QuestionCoverageScorer,
    MissingEvidenceFlagger,
    ConflictDetector,
    AnswerConfidenceAggregator,
    ConfidenceReport,
)


def _res(i, text, score):
    return SearchResult(
        document=Document(id=f"d{i}", text=text, embedding=[], metadata={}),
        score=score, rank=i+1,
    )


HIGH_AGREEMENT = [
    _res(0, "Hamlet is a Danish prince who seeks revenge for his father's murder.", 0.9),
    _res(1, "Hamlet, the Prince of Denmark, plots to avenge his father's death.", 0.85),
    _res(2, "The Danish prince Hamlet mourns his father and plans revenge.", 0.8),
]

LOW_AGREEMENT = [
    _res(0, "Hamlet loved Ophelia deeply.", 0.7),
    _res(1, "Hamlet never truly loved Ophelia — it was all pretense.", 0.6),
    _res(2, "Ophelia's love for Hamlet was unrequited.", 0.5),
]


class TestRetrievalScoreAnalyser:
    def test_empty_returns_zeros(self):
        result = RetrievalScoreAnalyser.analyse([])
        assert result["mean"] == 0.0

    def test_correct_statistics(self):
        results = [_res(i, "text", s) for i, s in enumerate([0.9, 0.7, 0.5])]
        stats = RetrievalScoreAnalyser.analyse(results)
        assert abs(stats["mean"] - 0.7) < 0.01
        assert stats["max"] == pytest.approx(0.9)
        assert stats["min"] == pytest.approx(0.5)

    def test_single_result_std_zero(self):
        results = [_res(0, "text", 0.8)]
        stats = RetrievalScoreAnalyser.analyse(results)
        assert stats["std"] == 0.0


class TestSourceAgreementScorer:
    @pytest.fixture(scope="class")
    def embedder(self):
        from core.embedding.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})

    def test_high_agreement_score(self, embedder):
        scorer = SourceAgreementScorer(embedder)
        score = scorer.score(HIGH_AGREEMENT)
        assert 0.0 <= score <= 1.0

    def test_agreement_high_gt_low(self, embedder):
        scorer = SourceAgreementScorer(embedder)
        high = scorer.score(HIGH_AGREEMENT)
        low = scorer.score(LOW_AGREEMENT)
        assert high > low, f"High agreement {high:.3f} should > low {low:.3f}"

    def test_single_result_returns_one(self, embedder):
        scorer = SourceAgreementScorer(embedder)
        assert scorer.score([_res(0, "text", 0.9)]) == 1.0


class TestQuestionCoverageScorer:
    def test_full_coverage(self):
        results = [_res(0, "Hamlet questions existence and death and meaning.", 0.9)]
        score = QuestionCoverageScorer.score("What does Hamlet say about existence?", results)
        assert score >= 0.5

    def test_zero_coverage_no_results(self):
        score = QuestionCoverageScorer.score("What does Hamlet say?", [])
        assert score == 0.0

    def test_stop_words_excluded(self):
        results = [_res(0, "Hamlet the prince of Denmark.", 0.8)]
        score = QuestionCoverageScorer.score("Who is Hamlet?", results)
        assert score > 0.0  # "hamlet" should be found


class TestMissingEvidenceFlagger:
    def test_no_results_is_missing(self):
        flagger = MissingEvidenceFlagger(min_score=0.3)
        assert flagger.is_missing([]) is True

    def test_low_scores_is_missing(self):
        flagger = MissingEvidenceFlagger(min_score=0.5)
        results = [_res(i, "text", 0.1) for i in range(3)]
        assert flagger.is_missing(results) is True

    def test_good_results_not_missing(self):
        flagger = MissingEvidenceFlagger(min_score=0.3)
        results = [_res(i, "relevant text", 0.8) for i in range(3)]
        assert flagger.is_missing(results) is False


class TestConflictDetector:
    def test_low_agreement_detected_as_conflict(self):
        detector = ConflictDetector()
        conflict = detector.detect(LOW_AGREEMENT, agreement_score=0.2)
        assert conflict is True

    def test_high_agreement_no_conflict(self):
        detector = ConflictDetector()
        conflict = detector.detect(HIGH_AGREEMENT, agreement_score=0.85)
        assert conflict is False


class TestAnswerConfidenceAggregator:
    @pytest.fixture(scope="class")
    def embedder(self):
        from core.embedding.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})

    def test_empty_results_insufficient_evidence(self, embedder):
        agg = AnswerConfidenceAggregator(embedder=embedder)
        report = agg.assess("What is Hamlet about?", [])
        assert report.verdict == "INSUFFICIENT_EVIDENCE"
        assert report.missing_evidence is True

    def test_good_results_produce_report(self, embedder):
        agg = AnswerConfidenceAggregator(embedder=embedder)
        report = agg.assess("What is Hamlet about?", HIGH_AGREEMENT)
        assert isinstance(report, ConfidenceReport)
        assert report.composite_score > 0.0
        assert report.verdict in {"HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"}

    def test_composite_score_in_range(self, embedder):
        agg = AnswerConfidenceAggregator(embedder=embedder)
        report = agg.assess("Hamlet existence", HIGH_AGREEMENT)
        assert 0.0 <= report.composite_score <= 1.0

    def test_source_count_correct(self, embedder):
        agg = AnswerConfidenceAggregator(embedder=embedder)
        report = agg.assess("query", HIGH_AGREEMENT)
        assert report.source_count == 3

    def test_explanation_non_empty(self, embedder):
        agg = AnswerConfidenceAggregator(embedder=embedder)
        report = agg.assess("query", HIGH_AGREEMENT)
        assert len(report.explanation) > 0
