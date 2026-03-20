"""
Phase 6: Provenance & Traceability — NON-NEGOTIABLE.
Every SearchResult carries a ProvenanceRecord showing exactly where the content came from.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """
    Complete lineage record for a retrieved chunk.
    Attached to every SearchResult so answers are always traceable.
    """

    # Source document
    source_doc_id: str
    source_url: Optional[str] = None
    doc_title: Optional[str] = None
    doc_version: Optional[str] = None
    effective_date: Optional[str] = None   # ISO 8601

    # Chunk lineage
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    section_title: Optional[str] = None
    page_num: Optional[int] = None

    # Text span (exact location in source)
    start_char: int = 0
    end_char: int = 0
    exact_text_span: str = ""

    # Retrieval quality
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None

    # Classification
    classification_label: str = "UNCLASSIFIED"

    # Timestamps
    ingested_at: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_citation(self, style: str = "inline") -> str:
        """Format as a human-readable citation."""
        title = self.doc_title or self.source_doc_id
        version = f" v{self.doc_version}" if self.doc_version else ""
        section = f" § {self.section_title}" if self.section_title else ""
        chars = f" [{self.start_char}:{self.end_char}]"

        if style == "inline":
            return f"[{title}{version}{section}]"
        elif style == "footnote":
            return (
                f"{title}{version}{section}{chars}. "
                f"(Score: {self.retrieval_score:.3f}"
                f"{', Reranked: ' + f'{self.rerank_score:.3f}' if self.rerank_score else ''})"
            )
        elif style == "apa":
            return f"{title}{version}. {self.source_url or ''}"
        return f"[{title}]"


class CitationBuilder:
    """Builds formatted citations from ProvenanceRecords."""

    @staticmethod
    def build_all(
        records: List[ProvenanceRecord],
        style: str = "footnote",
    ) -> List[str]:
        return [r.to_citation(style) for r in records]

    @staticmethod
    def build_unique_sources(records: List[ProvenanceRecord]) -> List[str]:
        """Return deduplicated list of source document identifiers."""
        seen = set()
        sources = []
        for r in records:
            if r.source_doc_id not in seen:
                sources.append(r.source_doc_id)
                seen.add(r.source_doc_id)
        return sources


class SpanHighlighter:
    """Verifies and highlights exact text spans within source documents."""

    @staticmethod
    def verify_span(source_text: str, record: ProvenanceRecord) -> bool:
        """Return True if the text span in the record matches the source."""
        if not source_text:
            return False
        extracted = source_text[record.start_char: record.end_char]
        return extracted.strip() == record.exact_text_span.strip()

    @staticmethod
    def highlight(source_text: str, record: ProvenanceRecord, marker: str = "**") -> str:
        """Wrap the retrieved span in markers for display."""
        before = source_text[: record.start_char]
        span = source_text[record.start_char: record.end_char]
        after = source_text[record.end_char:]
        return f"{before}{marker}{span}{marker}{after}"


class IngestionAuditLog:
    """
    Append-only log of every ingestion event.
    Stored as newline-delimited JSON for simplicity and durability.
    Thread-safe.
    """

    def __init__(self, log_path: str = "./data/ingestion_audit.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        doc_id: str,
        source: str,
        chunk_count: int,
        metadata: Dict[str, Any],
        content_hash: Optional[str] = None,
    ) -> dict:
        entry = {
            "event": "ingestion",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_id": doc_id,
            "source": source,
            "chunk_count": chunk_count,
            "content_hash": content_hash or _sha256_short(source),
            "metadata": metadata,
        }
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

    def read_all(self) -> List[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_doc_history(self, doc_id: str) -> List[dict]:
        return [e for e in self.read_all() if e.get("doc_id") == doc_id]


class DocumentVersionRegistry:
    """
    Tracks all known versions of a document and flags superseded content.
    """

    def __init__(self) -> None:
        self._versions: Dict[str, List[dict]] = {}  # doc_title → versions

    def register(
        self,
        doc_id: str,
        version: str,
        effective_date: Optional[str] = None,
        supersedes: Optional[str] = None,
    ) -> None:
        entry = {
            "doc_id": doc_id,
            "version": version,
            "effective_date": effective_date,
            "supersedes": supersedes,
            "superseded_by": None,
        }
        key = doc_id
        if key not in self._versions:
            self._versions[key] = []
        # Mark old version as superseded
        if supersedes:
            for v in self._versions.get(supersedes, []):
                if v["version"] == supersedes:
                    v["superseded_by"] = doc_id
        self._versions[key].append(entry)

    def is_superseded(self, doc_id: str, version: str) -> bool:
        for v in self._versions.get(doc_id, []):
            if v["version"] == version and v.get("superseded_by"):
                return True
        return False

    def latest_version(self, doc_id: str) -> Optional[dict]:
        versions = self._versions.get(doc_id, [])
        if not versions:
            return None
        active = [v for v in versions if not v.get("superseded_by")]
        return active[-1] if active else versions[-1]

    def all_versions(self, doc_id: str) -> List[dict]:
        return self._versions.get(doc_id, [])


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_provenance(
    result,                      # SearchResult
    chunk_registry=None,         # Optional ChunkRegistry
    version_registry=None,       # Optional DocumentVersionRegistry
) -> ProvenanceRecord:
    """
    Build a ProvenanceRecord from a SearchResult.
    Enriches with chunk registry info and version registry info where available.
    """
    doc = result.document
    meta = doc.metadata or {}
    chunk_id = doc.id
    parent_id = None
    section_title = None
    start_char = 0
    end_char = 0

    if chunk_registry:
        chunk = chunk_registry.get(chunk_id)
        if chunk:
            parent_id = chunk.parent_id
            section_title = chunk.section_title
            start_char = chunk.start_char
            end_char = chunk.end_char

    return ProvenanceRecord(
        source_doc_id=meta.get("doc_id", chunk_id),
        source_url=meta.get("source_url"),
        doc_title=meta.get("doc_title", meta.get("source_file", "")),
        doc_version=meta.get("doc_version"),
        effective_date=meta.get("effective_date"),
        chunk_id=chunk_id,
        parent_chunk_id=parent_id,
        section_title=section_title or meta.get("section_title"),
        start_char=start_char,
        end_char=end_char,
        exact_text_span=doc.text[: min(len(doc.text), end_char - start_char)] if end_char > start_char else doc.text[:500],
        retrieval_score=result.score,
        classification_label=meta.get("classification", "UNCLASSIFIED"),
        ingested_at=meta.get("ingested_at"),
    )
