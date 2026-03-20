"""
Ingestor — orchestrates the load → chunk → embed → upsert pipeline.
Phase 1: uses naive chunking. Phase 2 will swap in semantic chunking transparently.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.embedding.base import EmbeddingProviderBase
from core.ingestion.loader import naive_chunk_text
from core.store.base import VectorStoreBase
from core.store.models import Document


@dataclass
class IngestionResult:
    """Summary returned after an ingestion run."""

    total_chunks: int
    upserted: int
    skipped: int
    collection: str
    doc_ids: List[str] = field(default_factory=list)


class Ingestor:
    """
    Ingest raw text into a vector store.

    Parameters
    ----------
    store      : Connected VectorStoreBase adapter.
    embedder   : EmbeddingProviderBase instance.
    config     : Ingestion config dict (from config.yaml ingestion section).
    """

    def __init__(
        self,
        store: VectorStoreBase,
        embedder: EmbeddingProviderBase,
        config: dict,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.collection_name: str = config.get("collection_name", "polyrag")
        self._chunk_size: int = config.get("chunk_size", 512)
        self._overlap: int = config.get("chunk_overlap", 64)
        self._batch_size: int = config.get("embed_batch_size", 32)

    # ── Public API ────────────────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if not self.store.collection_exists(self.collection_name):
            self.store.create_collection(
                self.collection_name,
                self.embedder.embedding_dim,
            )

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Chunk, embed, and upsert a raw text string.

        Parameters
        ----------
        text       : Full document text.
        metadata   : Key/value pairs attached to every chunk.
        collection : Override default collection name.
        doc_id     : Parent document ID (auto-generated if None).

        Returns
        -------
        IngestionResult with counts and generated chunk IDs.
        """
        collection = collection or self.collection_name
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        chunks = naive_chunk_text(text, self._chunk_size, self._overlap)
        documents = self._build_documents(chunks, doc_id, metadata)
        self._upsert_batched(collection, documents)

        return IngestionResult(
            total_chunks=len(documents),
            upserted=len(documents),
            skipped=0,
            collection=collection,
            doc_ids=[d.id for d in documents],
        )

    def ingest_file(
        self,
        path: str,
        metadata: Optional[Dict] = None,
        collection: Optional[str] = None,
    ) -> IngestionResult:
        """Load a text file and ingest its contents."""
        from core.ingestion.loader import load_text_file

        text = load_text_file(path)
        base_meta = {"source_file": str(path), **(metadata or {})}
        return self.ingest_text(text, metadata=base_meta, collection=collection)

    def ingest_gutenberg(
        self,
        url: Optional[str] = None,
        strip_boilerplate: bool = True,
        metadata: Optional[Dict] = None,
        collection: Optional[str] = None,
    ) -> IngestionResult:
        """Download and ingest a Project Gutenberg text."""
        from core.ingestion.loader import (
            download_gutenberg,
            strip_gutenberg_header_footer,
            GUTENBERG_SHAKESPEARE_URL,
        )

        url = url or GUTENBERG_SHAKESPEARE_URL
        text = download_gutenberg(url)
        if strip_boilerplate:
            text = strip_gutenberg_header_footer(text)

        base_meta = {"source_url": url, "source": "gutenberg", **(metadata or {})}
        return self.ingest_text(text, metadata=base_meta, collection=collection)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_documents(
        self,
        chunks: List[str],
        doc_id: str,
        metadata: Dict,
    ) -> List[Document]:
        """Create Document objects (without embeddings yet)."""
        docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = _stable_id(doc_id, i, chunk_text)
            docs.append(
                Document(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        **metadata,
                    },
                )
            )
        return docs

    def _upsert_batched(
        self,
        collection: str,
        documents: List[Document],
    ) -> None:
        """Embed in batches then upsert."""
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start: start + self._batch_size]
            texts = [d.text for d in batch]
            embeddings = self.embedder.embed(texts)
            for doc, emb in zip(batch, embeddings):
                doc.embedding = emb
            self.store.upsert(collection, batch)


def _stable_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID — same text at same position → same ID."""
    raw = f"{doc_id}::{chunk_index}::{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    return raw
