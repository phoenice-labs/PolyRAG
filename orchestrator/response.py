"""
RAGResponse — the full response envelope returned by the pipeline.
Contains answer, citations, confidence, provenance chain, Knowledge Graph context,
and LLM traces for full observability of every LLM call made during the request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.confidence.signals import ConfidenceReport
from core.graph.models import GraphPath
from core.provenance.models import ProvenanceRecord
from core.query.llm_client import LLMTraceEntry
from core.store.models import SearchResult


@dataclass
class RAGResponse:
    """
    Complete RAG response envelope.
    Every field is populated — no black-box answers.
    """
    query: str
    answer: str                                       # LLM-generated answer (Phase 5+)
    results: List[SearchResult]                       # Retrieved chunks

    # Provenance (Phase 6)
    provenance: List[ProvenanceRecord] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    # Confidence (Phase 7)
    confidence: Optional[ConfidenceReport] = None

    # Knowledge Graph context (Phase 10)
    graph_entities: List[str] = field(default_factory=list)    # entities detected in query
    graph_paths: List[GraphPath] = field(default_factory=list)  # traversal paths used

    # LLM Traces (Phase B): one entry per LLM call made during this request
    llm_traces: List[LLMTraceEntry] = field(default_factory=list)

    # Pipeline metadata
    rewritten_query: Optional[str] = None
    backend: str = ""
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        verdict = self.confidence.verdict if self.confidence else "UNKNOWN"
        score = self.confidence.composite_score if self.confidence else 0.0
        graph_note = f" | Graph paths: {len(self.graph_paths)}" if self.graph_paths else ""
        return (
            f"[{verdict} {score:.2f}] {len(self.results)} sources | "
            f"Citations: {len(self.citations)}{graph_note}"
        )

    def graph_explanation(self) -> str:
        """Human-readable explanation of knowledge graph traversal paths."""
        if not self.graph_paths:
            return "No knowledge graph paths used."
        lines = [f"Knowledge Graph traversal ({len(self.graph_paths)} paths):"]
        for path in self.graph_paths[:5]:
            lines.append(f"  * {path.explanation}  [score={path.relevance_score:.2f}]")
        return "\n".join(lines)
