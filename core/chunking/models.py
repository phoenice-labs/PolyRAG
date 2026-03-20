"""
Chunk data model — the core unit of Phase 2+.
Every chunk carries full provenance back to its parent and source document.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    A semantically coherent fragment of a document.
    Child chunks carry a parent_id pointing to a broader context chunk.
    """

    chunk_id: str
    doc_id: str
    parent_id: Optional[str] = None          # None → this IS a parent / root chunk

    text: str
    start_char: int = 0
    end_char: int = 0

    # Structural metadata
    section_title: Optional[str] = None
    page_num: Optional[int] = None
    chunk_index: int = 0
    chunk_type: str = "text"                 # text | table | heading | list

    # Quality / dedup
    text_hash: str = ""                      # SHA-256[:16] of text
    token_count: int = 0                     # rough word count

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.text_hash:
            self.text_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        if not self.token_count:
            self.token_count = len(self.text.split())
        if not self.end_char:
            self.end_char = self.start_char + len(self.text)


class ChunkRegistry:
    """
    In-memory registry mapping chunk_id → Chunk.
    Used by ParentExpander to fetch parent context from child hits.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Chunk] = {}

    def register(self, chunk: Chunk) -> None:
        self._store[chunk.chunk_id] = chunk

    def register_many(self, chunks: List[Chunk]) -> None:
        for c in chunks:
            self.register(c)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._store.get(chunk_id)

    def get_parent(self, chunk: Chunk) -> Optional[Chunk]:
        if chunk.parent_id:
            return self._store.get(chunk.parent_id)
        return None

    def get_children(self, parent_id: str) -> List[Chunk]:
        return [c for c in self._store.values() if c.parent_id == parent_id]

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
