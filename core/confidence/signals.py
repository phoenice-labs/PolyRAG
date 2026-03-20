"""
Phase 7: Confidence Estimation — signals, aggregation, and ConfidenceReport.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from core.store.models import SearchResult


@dataclass
class ConfidenceReport:
    """
    All confidence signals for a RAG response + composite score.
    """
    composite_score: float                  # [0.0, 1.0]
    verdict: str                            # HIGH | MEDIUM | LOW | INSUFFICIENT_EVIDENCE

    # Individual signals
    retrieval_score_mean: float = 0.0
    retrieval_score_std: float = 0.0
    retrieval_score_min: float = 0.0
    retrieval_score_max: float = 0.0
    source_agreement_score: float = 0.0    # 1.0 = all sources agree
    question_coverage_score: float = 0.0   # fraction of query terms covered
    missing_evidence: bool = False
    conflict_detected: bool = False
    source_count: int = 0

    explanation: str = ""


class RetrievalScoreAnalyser:
    """Computes statistics over retrieval score distributions."""

    @staticmethod
    def analyse(results: List[SearchResult]) -> dict:
        if not results:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        scores = [r.score for r in results]
        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }


class SourceAgreementScorer:
    """
    Measures semantic agreement between retrieved passages.
    High score → sources say similar things (high confidence).
    Low score → sources disagree (low confidence / conflict risk).
    """

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def score(self, results: List[SearchResult]) -> float:
        if len(results) < 2:
            return 1.0  # Only one source — no disagreement

        texts = [r.document.text[:512] for r in results[:5]]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception:
            return 0.5  # Fallback if embedding fails

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x ** 2 for x in a) ** 0.5
            nb = sum(x ** 2 for x in b) ** 0.5
            return dot / (na * nb + 1e-9)

        pairs = [
            cosine(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        return float(statistics.mean(pairs)) if pairs else 1.0


class QuestionCoverageScorer:
    """
    Estimates what fraction of query terms are covered by the retrieved passages.
    Simple token overlap (sufficient for Phase 7; can be replaced with NLI later).
    """

    @staticmethod
    def score(query: str, results: List[SearchResult]) -> float:
        if not results:
            return 0.0
        import re
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                      "at", "to", "of", "and", "or", "for", "with", "what", "how",
                      "why", "when", "where", "who", "does", "do", "did", "about",
                      "by", "from", "its", "it", "be", "this", "that", "these"}
        query_terms = {
            w.lower()
            for w in re.findall(r'\w+', query)
            if w.lower() not in stop_words and len(w) > 2
        }
        if not query_terms:
            return 1.0

        combined_text = " ".join(r.document.text for r in results).lower()
        covered = sum(1 for t in query_terms if t in combined_text)
        return covered / len(query_terms)


class MissingEvidenceFlagger:
    """Flags when no retrieved chunk meets the minimum quality threshold."""

    def __init__(self, min_score: float = 0.3, min_results: int = 1) -> None:
        self.min_score = min_score
        self.min_results = min_results

    def is_missing(self, results: List[SearchResult]) -> bool:
        if len(results) < self.min_results:
            return True
        return all(r.score < self.min_score for r in results)


class ConflictDetector:
    """
    Detects conflicting statements across retrieved passages.
    Uses simple heuristics: contradictory signal words (not, never, always, etc.)
    combined with low source agreement. Full NLI-based detection added in Phase 9.
    """

    _CONTRADICTION_SIGNALS = {
        "not", "never", "no", "false", "incorrect", "wrong",
        "contrary", "however", "but", "although", "despite", "yet",
    }

    def detect(self, results: List[SearchResult], agreement_score: float) -> bool:
        if agreement_score < 0.3:
            return True  # Very low agreement → likely conflict

        # Heuristic: multiple sources use strong negation words
        negation_docs = 0
        for r in results:
            tokens = set(r.document.text.lower().split())
            if tokens & self._CONTRADICTION_SIGNALS:
                negation_docs += 1

        return negation_docs >= len(results) * 0.6 and agreement_score < 0.5


class AnswerConfidenceAggregator:
    """
    Combines all signals into a single composite confidence score.
    Weights are tunable; defaults are calibrated on Shakespeare retrieval tests.
    """

    def __init__(
        self,
        embedder=None,
        missing_threshold: float = 0.3,
        weights: dict | None = None,
    ) -> None:
        self.embedder = embedder
        self.missing_flagger = MissingEvidenceFlagger(min_score=missing_threshold)
        self.conflict_detector = ConflictDetector()
        self.weights = weights or {
            "score_mean": 0.30,
            "source_agreement": 0.25,
            "question_coverage": 0.25,
            "score_max": 0.20,
        }

    def assess(
        self,
        query: str,
        results: List[SearchResult],
    ) -> ConfidenceReport:
        if not results:
            return ConfidenceReport(
                composite_score=0.0,
                verdict="INSUFFICIENT_EVIDENCE",
                missing_evidence=True,
                explanation="No results retrieved.",
            )

        # Compute signals
        dist = RetrievalScoreAnalyser.analyse(results)
        agreement = (
            SourceAgreementScorer(self.embedder).score(results)
            if self.embedder
            else 0.5
        )
        coverage = QuestionCoverageScorer.score(query, results)
        missing = self.missing_flagger.is_missing(results)
        conflict = self.conflict_detector.detect(results, agreement)

        # Weighted composite
        composite = (
            self.weights["score_mean"] * dist["mean"]
            + self.weights["source_agreement"] * agreement
            + self.weights["question_coverage"] * coverage
            + self.weights["score_max"] * dist["max"]
        )
        composite = max(0.0, min(1.0, composite))

        # Penalise for missing evidence and conflicts
        if missing:
            composite *= 0.3
        elif conflict:
            composite *= 0.7

        # Verdict
        if missing or composite < 0.2:
            verdict = "INSUFFICIENT_EVIDENCE"
        elif composite >= 0.75:
            verdict = "HIGH"
        elif composite >= 0.45:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"

        explanation = (
            f"Score dist: mean={dist['mean']:.2f}, max={dist['max']:.2f} | "
            f"Agreement: {agreement:.2f} | Coverage: {coverage:.2f} | "
            f"Missing: {missing} | Conflict: {conflict}"
        )

        return ConfidenceReport(
            composite_score=round(composite, 4),
            verdict=verdict,
            retrieval_score_mean=dist["mean"],
            retrieval_score_std=dist["std"],
            retrieval_score_min=dist["min"],
            retrieval_score_max=dist["max"],
            source_agreement_score=agreement,
            question_coverage_score=coverage,
            missing_evidence=missing,
            conflict_detected=conflict,
            source_count=len(results),
            explanation=explanation,
        )
