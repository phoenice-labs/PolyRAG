"""
Paragraph chunker — splits on blank lines (natural paragraph breaks).

Each paragraph is kept intact if it fits within ``max_chars``.
Paragraphs that exceed ``max_chars`` are further split using
FixedOverlapChunker so the size guarantee always holds.
"""
from __future__ import annotations

import re
from typing import List

from core.chunking.base import ChunkerBase
from core.chunking.models import Chunk

_BLANK_LINE = re.compile(r'\n{2,}')


class ParagraphChunker(ChunkerBase):
    """
    Chunks text at blank-line paragraph boundaries.

    Config keys
    -----------
    max_chars    : maximum characters per output chunk (default: 512).
                   Paragraphs larger than this are sub-split with
                   FixedOverlapChunker.
    chunk_overlap: character overlap when sub-splitting (default: 64).
    """

    def __init__(self, max_chars: int = 512, chunk_overlap: int = 64) -> None:
        self.max_chars = max_chars
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        from core.chunking.fixed_overlap import FixedOverlapChunker

        metadata = metadata or {}
        paragraphs = [p.strip() for p in _BLANK_LINE.split(text) if p.strip()]
        splitter = FixedOverlapChunker(self.max_chars, self.chunk_overlap)

        chunks: List[Chunk] = []
        idx = 0
        char_cursor = 0

        for para in paragraphs:
            start = text.find(para, char_cursor)
            if start < 0:
                start = char_cursor

            if len(para) <= self.max_chars:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::para::{idx}",
                    doc_id=doc_id,
                    text=para,
                    start_char=start,
                    end_char=start + len(para),
                    chunk_index=idx,
                    chunk_type="text",
                    metadata={"chunker": "paragraph", **metadata},
                ))
                idx += 1
            else:
                # Paragraph too large — sub-split it
                sub_chunks = splitter.chunk(para, doc_id, metadata)
                for sc in sub_chunks:
                    sc.chunk_id = f"{doc_id}::para::{idx}"
                    sc.start_char += start
                    sc.end_char += start
                    sc.chunk_index = idx
                    sc.metadata["chunker"] = "paragraph_split"
                    chunks.append(sc)
                    idx += 1

            char_cursor = start + len(para)

        return chunks
