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
        enable_rich_formats: bool = True,
    ) -> IngestionResult:
        """
        Load a document (plain text, PDF, or PPTX) and ingest its contents.

        Parameters
        ----------
        path               : Path to the document.
        metadata           : Key/value pairs attached to every chunk.
        collection         : Override default collection name.
        enable_rich_formats: Pass ``False`` to restrict ingestion to plain-text
                             files only (honours the config toggle).
        """
        from core.ingestion.loader import load_document

        text = load_document(path, enable_rich_formats=enable_rich_formats)
        base_meta = {"source_file": str(path), **(metadata or {})}
        return self.ingest_text(text, metadata=base_meta, collection=collection)

    def ingest_file_streaming(
        self,
        path: str,
        metadata: Optional[Dict] = None,
        collection: Optional[str] = None,
        max_doc_size_mb: float = 200.0,
        progress_callback=None,
    ) -> IngestionResult:
        """
        Stream-chunk a large file and ingest it without loading it fully into RAM.

        Safe for files of arbitrary size — only `chunk_size + overlap` characters
        are held in memory at any point. Embeddings are upserted in batches of
        `embed_batch_size` chunks.

        Parameters
        ----------
        path              : Path to the text file.
        metadata          : Key/value pairs attached to every chunk.
        collection        : Override default collection name.
        max_doc_size_mb   : Guard — raises ValueError if file exceeds this.
        progress_callback : Optional callable(chunks_done: int, chunks_est: int)
                            called after each embedding batch for progress reporting.

        Returns
        -------
        IngestionResult with total_chunks upserted.
        """
        from core.ingestion.loader import stream_chunk_file, estimate_chunk_count

        collection = collection or self.collection_name
        doc_id = str(uuid.uuid4())
        metadata = {
            "source_file": str(path),
            "doc_id": doc_id,
            **(metadata or {}),
        }

        chunks_est = estimate_chunk_count(path, self._chunk_size, self._overlap)
        chunks_done = 0
        batch: List[Document] = []
        chunk_index = 0

        for chunk_text in stream_chunk_file(
            path,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            max_doc_size_mb=max_doc_size_mb,
        ):
            chunk_id = _stable_id(doc_id, chunk_index, chunk_text)
            batch.append(Document(
                id=chunk_id,
                text=chunk_text,
                metadata={
                    "chunk_index": chunk_index,
                    **metadata,
                },
            ))
            chunk_index += 1

            if len(batch) >= self._batch_size:
                self._upsert_batched(collection, batch)
                chunks_done += len(batch)
                if progress_callback:
                    progress_callback(chunks_done, chunks_est)
                batch = []

        # Flush remaining batch
        if batch:
            self._upsert_batched(collection, batch)
            chunks_done += len(batch)
            if progress_callback:
                progress_callback(chunks_done, chunks_est)

        return IngestionResult(
            total_chunks=chunks_done,
            upserted=chunks_done,
            skipped=0,
            collection=collection,
            doc_ids=[doc_id],
        )

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
