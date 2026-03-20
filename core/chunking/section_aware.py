"""
Section-aware chunker — detects headings (Markdown, plain-text ALL-CAPS, numbered)
and splits on section boundaries. Each section becomes a parent chunk;
its body is further split into child chunks.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from core.chunking.base import ChunkerBase
from core.chunking.models import Chunk

# Heading detection patterns (in priority order)
_HEADING_PATTERNS = [
    re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),                  # Markdown
    re.compile(r'^([A-Z][A-Z\s]{4,})$', re.MULTILINE),             # ALL-CAPS lines
    re.compile(r'^(\d+(?:\.\d+)*\.?\s+[A-Z].{3,})$', re.MULTILINE), # 1.2 Numbered
    re.compile(r'^(ACT [IVX]+|SCENE [IVX]+|CHAPTER \w+)$',          # Drama / book
               re.MULTILINE | re.IGNORECASE),
]


def _find_sections(text: str) -> List[Tuple[int, str, str]]:
    """
    Return list of (start_char, heading_text, body_text) tuples.
    If no headings found, returns the whole text as one unnamed section.
    """
    # Collect all heading matches across all patterns
    matches: List[Tuple[int, str]] = []
    for pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.group(1).strip()))

    if not matches:
        return [(0, "", text)]

    matches.sort(key=lambda x: x[0])
    # Deduplicate overlapping matches (keep first)
    deduped: List[Tuple[int, str]] = []
    last_end = -1
    for start, heading in matches:
        if start > last_end:
            deduped.append((start, heading))
            last_end = start + len(heading)

    sections: List[Tuple[int, str, str]] = []
    for i, (start, heading) in enumerate(deduped):
        next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        # Body starts after the heading line
        body_start = text.find('\n', start)
        body_start = (body_start + 1) if body_start != -1 else start + len(heading)
        body = text[body_start:next_start].strip()
        sections.append((start, heading, body))

    return sections


class SectionAwareChunker(ChunkerBase):
    """
    Splits a document on detected headings.

    Each section is stored as a **parent** chunk.
    The body of each section is further split into **child** chunks
    using fixed-size windows, preserving the parent_id link.

    Config keys
    -----------
    child_size    : target child chunk size in characters (default: 512)
    child_overlap : overlap between child chunks (default: 64)
    min_section_chars : minimum body length to produce children (default: 100)
    """

    def __init__(
        self,
        child_size: int = 512,
        child_overlap: int = 64,
        min_section_chars: int = 100,
    ) -> None:
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.min_section_chars = min_section_chars

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        from core.chunking.fixed_overlap import FixedOverlapChunker

        metadata = metadata or {}
        sections = _find_sections(text)
        all_chunks: List[Chunk] = []
        child_chunker = FixedOverlapChunker(self.child_size, self.child_overlap)
        section_idx = 0

        for start_char, heading, body in sections:
            parent_id = f"{doc_id}::section::{section_idx}"
            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                text=(f"{heading}\n\n{body}" if heading else body).strip(),
                start_char=start_char,
                end_char=start_char + len(heading) + len(body) + 2,
                chunk_index=section_idx,
                chunk_type="heading" if heading else "text",
                section_title=heading or None,
                metadata={"chunker": "section_aware", "is_parent": True, **metadata},
            )
            all_chunks.append(parent_chunk)
            section_idx += 1

            if len(body) >= self.min_section_chars:
                children = child_chunker.chunk(body, doc_id, metadata)
                for j, c in enumerate(children):
                    # Make child IDs globally unique within this document by including section_idx
                    c.chunk_id = f"{doc_id}::section::{section_idx}::child::{j}"
                    c.parent_id = parent_id
                    c.section_title = heading or None
                    c.chunk_type = "text"
                    c.metadata["is_parent"] = False
                    c.metadata["chunker"] = "section_aware_child"
                    c.start_char += start_char
                    c.end_char += start_char
                    all_chunks.append(c)

        return all_chunks
