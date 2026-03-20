"""
Core data models for Phoenice-PolyRAG.
All adapters share these contracts — switching backends requires zero model changes.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A unit of content stored in and retrieved from a vector store."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class SearchResult(BaseModel):
    """A single result returned from a vector store query."""

    document: Document
    score: float          # Higher = more similar (normalised to [0, 1] where possible)
    rank: int             # 1-indexed position in result list

    model_config = {"arbitrary_types_allowed": True}


class CollectionInfo(BaseModel):
    """Metadata about a vector store collection / index."""

    name: str
    count: int
    embedding_dim: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
