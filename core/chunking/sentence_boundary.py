"""
Sentence-boundary chunker — groups complete sentences into chunks.
Uses a simple regex tokeniser (no heavy NLP deps required).
"""
from __future__ import annotations

import re
from typing import List

from core.chunking.base import ChunkerBase
from core.chunking.models import Chunk

# Sentence boundary: period / ! / ? followed by whitespace + capital letter
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> List[str]:
    """Simple regex-based sentence splitter."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _flush(sents, doc_id, idx, text, char_offset, metadata):
    chunk_text = " ".join(sents)
    start = text.find(sents[0], char_offset)
    if start < 0:
        start = char_offset
    return Chunk(
        chunk_id=f"{doc_id}::sent::{idx}",
        doc_id=doc_id,
        text=chunk_text,
        start_char=start,
        end_char=start + len(chunk_text),
        chunk_index=idx,
        chunk_type="text",
        metadata={"chunker": "sentence_boundary", **metadata},
    ), start + len(chunk_text)


class SentenceBoundaryChunker(ChunkerBase):
    """
    Groups whole sentences into chunks that stay within the target word count.

    Config keys
    -----------
    max_words    : max words per chunk (default: 120)
    overlap_sents: number of sentences to overlap (default: 1)
    """

    def __init__(self, max_words: int = 120, overlap_sents: int = 1) -> None:
        self.max_words = max_words
        self.overlap_sents = overlap_sents

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        metadata = metadata or {}
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_words = 0
        char_offset = 0
        idx = 0

        for sent in sentences:
            words = len(sent.split())
            if current_words + words > self.max_words and current_sents:
                chunk, char_offset = _flush(current_sents, doc_id, idx, text, char_offset, metadata)
                chunks.append(chunk)
                idx += 1
                # overlap: keep last N sentences, but only if they don't already exceed limit
                overlap = current_sents[-self.overlap_sents:]
                overlap_words = sum(len(s.split()) for s in overlap)
                if overlap_words < self.max_words:
                    current_sents = overlap
                    current_words = overlap_words
                else:
                    current_sents = []
                    current_words = 0

            current_sents.append(sent)
            current_words += words

        if current_sents:
            chunk, _ = _flush(current_sents, doc_id, idx, text, char_offset, metadata)
            chunks.append(chunk)

        return chunks

