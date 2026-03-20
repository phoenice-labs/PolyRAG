"""
Semantic boundary chunker — detects topic shifts using embedding cosine similarity.
Chunks at positions where similarity drops below a threshold (breakpoints).
"""
from __future__ import annotations

from typing import List, Optional

from core.chunking.base import ChunkerBase
from core.chunking.models import Chunk


class SemanticBoundaryChunker(ChunkerBase):
    """
    Splits text at semantic breakpoints detected via embedding similarity.

    Algorithm
    ---------
    1. Split text into sentences.
    2. Embed each sentence (or sliding windows of sentences).
    3. Compute cosine similarity between consecutive windows.
    4. Detect breakpoints where similarity < threshold.
    5. Merge sentences within each segment into a chunk.

    Config keys
    -----------
    embedder         : EmbeddingProviderBase instance (injected)
    window_size      : sentences per window for smoothing (default: 3)
    threshold        : similarity drop threshold [0,1] (default: 0.5)
    min_chunk_words  : minimum words before forcing a break (default: 30)
    max_chunk_words  : force a break if chunk exceeds this (default: 300)
    """

    def __init__(
        self,
        embedder,
        window_size: int = 3,
        threshold: float = 0.5,
        min_chunk_words: int = 30,
        max_chunk_words: int = 300,
    ) -> None:
        self.embedder = embedder
        self.window_size = window_size
        self.threshold = threshold
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        import re
        metadata = metadata or {}

        sent_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = [s.strip() for s in sent_re.split(text) if s.strip()]

        if len(sentences) <= 2:
            # Too short to analyse — return as one chunk
            return [Chunk(
                chunk_id=f"{doc_id}::sem::0",
                doc_id=doc_id, text=text.strip(),
                start_char=0, end_char=len(text),
                chunk_index=0, chunk_type="text",
                metadata={"chunker": "semantic_boundary", **metadata},
            )]

        # Build sliding windows and embed
        windows = [
            " ".join(sentences[i: i + self.window_size])
            for i in range(len(sentences))
        ]
        embeddings = self.embedder.embed(windows)

        # Compute cosine similarities between consecutive windows
        def cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x ** 2 for x in a) ** 0.5
            nb = sum(x ** 2 for x in b) ** 0.5
            return dot / (na * nb + 1e-9)

        similarities = [
            cosine(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Detect breakpoints (local minima below threshold)
        breakpoints = set()
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                breakpoints.add(i + 1)  # break AFTER sentence i

        # Build chunks from segments
        chunks: List[Chunk] = []
        segment: List[str] = []
        segment_words = 0
        char_cursor = 0
        idx = 0

        for i, sent in enumerate(sentences):
            words = len(sent.split())
            force_break = segment_words + words > self.max_chunk_words and segment

            if (i in breakpoints and segment_words >= self.min_chunk_words) or force_break:
                chunk_text = " ".join(segment)
                start = text.find(segment[0], char_cursor)
                if start < 0:
                    start = char_cursor
                end = start + len(chunk_text)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::sem::{idx}",
                    doc_id=doc_id, text=chunk_text,
                    start_char=start, end_char=end,
                    chunk_index=idx, chunk_type="text",
                    metadata={"chunker": "semantic_boundary", **metadata},
                ))
                idx += 1
                char_cursor = end
                segment = []
                segment_words = 0

            segment.append(sent)
            segment_words += words

        if segment:
            chunk_text = " ".join(segment)
            start = text.find(segment[0], char_cursor)
            if start < 0:
                start = char_cursor
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::sem::{idx}",
                doc_id=doc_id, text=chunk_text,
                start_char=start, end_char=start + len(chunk_text),
                chunk_index=idx, chunk_type="text",
                metadata={"chunker": "semantic_boundary", **metadata},
            ))

        return chunks
