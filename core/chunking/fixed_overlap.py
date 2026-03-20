"""
Fixed-overlap chunker — character-level sliding window.
Used as a reliable fallback and Phase 1 baseline.
"""
from __future__ import annotations

from typing import List

from core.chunking.base import ChunkerBase
from core.chunking.models import Chunk


class FixedOverlapChunker(ChunkerBase):
    """
    Splits text into fixed-size character windows with overlap.

    Config keys
    -----------
    chunk_size    : target chunk size in characters (default: 512)
    chunk_overlap : overlap between consecutive chunks (default: 64)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        metadata = metadata or {}
        chunks: List[Chunk] = []
        stride = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0

        for i, start in enumerate(range(0, len(text), stride)):
            end = min(start + self.chunk_size, len(text))
            raw = text[start:end]
            snippet = raw.strip()
            if not snippet:
                continue

            # Adjust start_char to account for leading whitespace removed by strip()
            actual_start = start + len(raw) - len(raw.lstrip())
            actual_end = actual_start + len(snippet)

            chunk_id = f"{doc_id}::fixed::{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=snippet,
                    start_char=actual_start,
                    end_char=actual_end,
                    chunk_index=idx,
                    chunk_type="text",
                    metadata={"chunker": "fixed_overlap", **metadata},
                )
            )
            idx += 1

        return chunks
