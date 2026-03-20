"""
Phase 9: Noise Control — deduplication, quality scoring, conflict resolution.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

from core.store.models import SearchResult


class DuplicateDetector:
    """
    Near-duplicate detection using MinHash + Jaccard similarity.
    Falls back to exact text-hash dedup if datasketch is not installed.

    Requires (optional): pip install datasketch
    """

    def __init__(self, similarity_threshold: float = 0.85, num_perm: int = 128) -> None:
        self.threshold = similarity_threshold
        self.num_perm = num_perm

    def _shingle(self, text: str, k: int = 5) -> set:
        tokens = text.lower().split()
        return {" ".join(tokens[i: i + k]) for i in range(max(1, len(tokens) - k + 1))}

    def _minhash(self, text: str):
        try:
            from datasketch import MinHash
            mh = MinHash(num_perm=self.num_perm)
            for shingle in self._shingle(text):
                mh.update(shingle.encode("utf-8"))
            return mh
        except ImportError:
            return None

    def deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove near-duplicate results, keeping the highest-scored copy."""
        if not results:
            return []

        # Try MinHash dedup
        minhashes = []
        use_minhash = True
        for r in results:
            mh = self._minhash(r.document.text)
            if mh is None:
                use_minhash = False
                break
            minhashes.append(mh)

        if use_minhash:
            return self._minhash_dedup(results, minhashes)
        else:
            return self._hash_dedup(results)

    def _minhash_dedup(self, results, minhashes) -> List[SearchResult]:
        kept_indices: List[int] = []
        for i, (r, mh_i) in enumerate(zip(results, minhashes)):
            duplicate = False
            for j in kept_indices:
                sim = mh_i.jaccard(minhashes[j])
                if sim >= self.threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept_indices.append(i)
        unique = [results[i] for i in kept_indices]
        for rank, r in enumerate(unique, start=1):
            r.rank = rank
        return unique

    def _hash_dedup(self, results: List[SearchResult]) -> List[SearchResult]:
        seen: set = set()
        unique: List[SearchResult] = []
        for r in results:
            h = hashlib.sha256(r.document.text[:256].encode()).hexdigest()[:16]
            if h not in seen:
                seen.add(h)
                unique.append(r)
        for rank, r in enumerate(unique, start=1):
            r.rank = rank
        return unique


class QualityScorer:
    """
    Assigns a quality score to a chunk based on simple heuristics:
    - Length (too short/long = low quality)
    - Repetition ratio (repeated phrases = low quality)
    - Alphanumeric ratio (mostly symbols = low quality)
    - Sentence completeness (ends with punctuation)
    Score ∈ [0.0, 1.0]; chunks below threshold are filtered.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0

        words = text.split()
        word_count = len(words)

        # Length penalty
        if word_count < 10:
            length_score = word_count / 10
        elif word_count > 800:
            length_score = max(0.1, 1.0 - (word_count - 800) / 1000)
        else:
            length_score = 1.0

        # Repetition ratio (unique / total bigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        rep_score = len(set(bigrams)) / max(1, len(bigrams))

        # Alphanumeric ratio
        alnum = sum(c.isalnum() or c.isspace() for c in text)
        alnum_score = alnum / max(1, len(text))

        # Sentence completeness
        punct_score = 1.0 if text.rstrip()[-1] in ".!?\"'" else 0.7

        return round(
            0.25 * length_score + 0.35 * rep_score + 0.25 * alnum_score + 0.15 * punct_score, 3
        )

    def filter_results(self, results: List[SearchResult]) -> List[SearchResult]:
        kept = [r for r in results if self.score(r.document.text) >= self.threshold]
        for i, r in enumerate(kept, start=1):
            r.rank = i
        return kept

    def filter_low_quality(self, texts: List[str]) -> List[str]:
        return [t for t in texts if self.score(t) >= self.threshold]


class ConflictResolver:
    """
    Surfaces conflicting passages with labels rather than silently discarding.
    When conflict is detected, both perspectives are returned with labels.
    """

    def resolve(
        self,
        results: List[SearchResult],
        agreement_score: float,
    ) -> Tuple[List[SearchResult], bool, str]:
        """
        Returns
        -------
        (results, conflict_detected, explanation)
        Results are labelled 'PERSPECTIVE_A' / 'PERSPECTIVE_B' if conflicting.
        """
        conflict = agreement_score < 0.4 and len(results) >= 2
        if not conflict:
            return results, False, "No conflict detected."

        # Label perspectives
        labelled = []
        for i, r in enumerate(results):
            label = f"PERSPECTIVE_{chr(65 + i % 26)}"
            new_meta = {**r.document.metadata, "conflict_label": label}
            from core.store.models import Document
            doc = Document(
                id=r.document.id,
                text=r.document.text,
                embedding=r.document.embedding,
                metadata=new_meta,
            )
            labelled.append(SearchResult(document=doc, score=r.score, rank=r.rank))

        explanation = (
            f"Conflict detected (agreement={agreement_score:.2f}). "
            f"{len(results)} perspectives surfaced for review."
        )
        return labelled, True, explanation


class NoiseFilterPipeline:
    """
    Composed noise-control pipeline:
      1. Quality filter (remove low-quality chunks)
      2. Deduplication (remove near-duplicates)
      3. Conflict resolution (surface, don't hide conflicts)
    """

    def __init__(
        self,
        quality_threshold: float = 0.3,
        dedup_threshold: float = 0.85,
    ) -> None:
        self.quality = QualityScorer(threshold=quality_threshold)
        self.dedup = DuplicateDetector(similarity_threshold=dedup_threshold)
        self.resolver = ConflictResolver()

    def run(
        self,
        results: List[SearchResult],
        agreement_score: float = 1.0,
    ) -> Tuple[List[SearchResult], dict]:
        # Stage 1: quality filter
        after_quality = self.quality.filter_results(results)

        # Stage 2: dedup
        after_dedup = self.dedup.deduplicate(after_quality)

        # Stage 3: conflict resolution
        final, conflict, explanation = self.resolver.resolve(after_dedup, agreement_score)

        report = {
            "input_count": len(results),
            "after_quality": len(after_quality),
            "after_dedup": len(after_dedup),
            "final_count": len(final),
            "conflict_detected": conflict,
            "conflict_explanation": explanation,
        }
        return final, report
