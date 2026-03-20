"""
ChunkingPipeline — compose multiple chunkers, apply quality gates, dedup.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

from core.chunking.base import ChunkerBase
from core.chunking.models import Chunk, ChunkRegistry


class QualityGate:
    """Filters out chunks that are too short, too long, or duplicates."""

    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 1000,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self._seen_hashes: set = set()

    def passes(self, chunk: Chunk) -> bool:
        if chunk.token_count < self.min_words:
            return False
        if chunk.token_count > self.max_words:
            return False
        if chunk.text_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(chunk.text_hash)
        return True

    def reset(self) -> None:
        self._seen_hashes.clear()


class ChunkingPipeline:
    """
    Runs a primary chunker, optionally followed by fallback chunkers,
    then applies a quality gate and registers all chunks.

    Usage
    -----
        pipeline = ChunkingPipeline(chunker=SectionAwareChunker(...))
        chunks = pipeline.run(text, doc_id="doc_001", metadata={...})
    """

    def __init__(
        self,
        chunker: ChunkerBase,
        registry: Optional[ChunkRegistry] = None,
        min_words: int = 10,
        max_words: int = 1000,
    ) -> None:
        self.chunker = chunker
        self.registry = registry if registry is not None else ChunkRegistry()
        self.gate = QualityGate(min_words=min_words, max_words=max_words)

    def run(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> List[Chunk]:
        """Chunk text, apply quality gate, register, return accepted chunks."""
        self.gate.reset()
        raw = self.chunker.chunk(text, doc_id, metadata)
        accepted = [c for c in raw if self.gate.passes(c)]
        self.registry.register_many(accepted)
        return accepted

    def get_registry(self) -> ChunkRegistry:
        return self.registry
