"""
Phase 11: RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Builds a hierarchical summary tree over leaf chunks using LLM summarization.
Improves recall for broad/conceptual queries that no single chunk fully answers.

Algorithm (Sarthi et al. 2024, arXiv:2401.18059):
  1. Cluster leaf chunks by embedding cosine-similarity (k-means, numpy-only)
  2. Summarize each cluster via LLM → one "summary chunk" per cluster (level 1)
  3. Store summaries in vector store under a companion collection (<name>_raptor)
  4. At query time: search both leaf + summary collections, merge results

Usage:
  # After all ingestion is done:
  n = pipeline.build_raptor_index()
  # Query automatically searches both levels when RAPTOR is enabled.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

import numpy as np

from core.store.models import Document, SearchResult

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
Synthesize the following text passages into a single coherent summary paragraph.
Capture all key entities, concepts, dates, and relationships.
Be concise (3-5 sentences). Include only information present in the passages.

Passages:
{passages}

Summary:"""


class RaptorIndexer:
    """
    Builds the RAPTOR level-1 summary tree at the end of ingestion.

    Call after all documents are ingested:
        n_summaries = pipeline.build_raptor_index()

    Parameters
    ----------
    llm_client      : LMStudioClient (used for summarization)
    embedder        : EmbeddingProvider (embeds summary text)
    n_clusters      : target number of cluster summaries
    max_passage_chars : max characters per passage fed to LLM
    """

    RAPTOR_SUFFIX = "_raptor"

    def __init__(
        self,
        llm_client,
        embedder,
        n_clusters: int = 10,
        max_passage_chars: int = 400,
    ) -> None:
        self.llm = llm_client
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.max_passage_chars = max_passage_chars

    def build(
        self,
        docs: List[Document],
        base_collection: str,
        vector_store,
    ) -> int:
        """
        Build RAPTOR summary index from leaf documents.

        Returns number of summary chunks created (0 if LLM offline).
        """
        if not docs:
            return 0
        if not self.llm.is_available():
            logger.warning("RAPTOR: LM Studio offline — skipping index build")
            return 0

        # Filter docs with valid embeddings
        valid = [d for d in docs if d.embedding]
        if not valid:
            return 0

        emb_matrix = np.array([d.embedding for d in valid], dtype=np.float32)
        k = min(self.n_clusters, len(valid) // 2 + 1)
        if k < 1:
            return 0

        labels = self._kmeans(emb_matrix, k)

        # Group docs by cluster
        clusters: Dict[int, List[Document]] = {}
        for doc, label in zip(valid, labels):
            clusters.setdefault(label, []).append(doc)

        # Ensure raptor collection exists
        raptor_coll = base_collection + self.RAPTOR_SUFFIX
        if not vector_store.collection_exists(raptor_coll):
            vector_store.create_collection(raptor_coll, self.embedder.embedding_dim)

        created = 0
        for cluster_id, cluster_docs in clusters.items():
            summary_doc = self._summarize_cluster(cluster_docs, cluster_id, base_collection)
            if summary_doc:
                vector_store.upsert(raptor_coll, [summary_doc])
                created += 1

        logger.info(
            "RAPTOR: %d summaries from %d leaf chunks (%d clusters)",
            created, len(valid), len(clusters),
        )
        return created

    def _summarize_cluster(
        self, docs: List[Document], cluster_id: int, base_collection: str
    ) -> Optional[Document]:
        passages, total = [], 0
        for doc in docs:
            seg = doc.text[:self.max_passage_chars].strip()
            if not seg:
                continue
            passages.append(f"- {seg}")
            total += len(seg)
            if total > 4000:          # budget guard
                break

        if not passages:
            return None

        # Load prompt from registry (with fallback to hardcoded _SUMMARY_PROMPT)
        try:
            from core.prompt_registry import PromptRegistry
            prompt_template = PromptRegistry.instance().get_prompt("raptor_summarizer")
        except Exception:
            prompt_template = _SUMMARY_PROMPT

        prompt = prompt_template.format(passages="\n".join(passages))
        try:
            summary = self.llm.complete(
                prompt,
                system="You are a concise summarization assistant.",
                max_tokens=300,
                temperature=0.1,
                trace_method="RAPTOR Summarization",
            )
        except Exception as exc:
            logger.warning("RAPTOR cluster %d summarization failed: %s", cluster_id, exc)
            return None

        if not summary or len(summary.strip()) < 20:
            return None

        source_ids = [d.id for d in docs]
        embedding = self.embedder.embed_one(summary)

        return Document(
            id=f"raptor_l1_{cluster_id}_{uuid.uuid4().hex[:8]}",
            text=summary.strip(),
            embedding=embedding,
            metadata={
                "raptor_level": 1,
                "source_chunk_ids": ",".join(source_ids),
                "source_collection": base_collection,
                "cluster_id": str(cluster_id),
            },
        )

    @staticmethod
    def _kmeans(embeddings: np.ndarray, k: int, max_iter: int = 50) -> List[int]:
        """Simple cosine k-means (no scipy dependency)."""
        n = len(embeddings)
        if k >= n:
            return list(range(n))

        # L2-normalise for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        rng = np.random.default_rng(42)
        centroids = normed[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            sims = normed @ centroids.T          # (n, k)
            new_labels = np.argmax(sims, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    c = normed[mask].mean(axis=0)
                    c_norm = np.linalg.norm(c)
                    centroids[j] = c / c_norm if c_norm > 0 else c

        return labels.tolist()


class RaptorRetriever:
    """
    Retrieves from RAPTOR tree: leaf collection + summary collection.

    Summary hits act as semantic "hub" documents that represent a cluster;
    they improve recall for queries that span multiple related leaf chunks.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        summary_weight: float = 0.7,
    ) -> None:
        self.store = vector_store
        self.embedder = embedder
        self.summary_weight = summary_weight

    def retrieve(
        self,
        query: str,
        base_collection: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Retrieve from leaf + RAPTOR summary levels, merge results."""
        q_vec = self.embedder.embed_one(query)

        leaf_results = self.store.query(base_collection, q_vec, top_k=top_k * 2, filters=filters)

        raptor_coll = base_collection + RaptorIndexer.RAPTOR_SUFFIX
        summary_results: List[SearchResult] = []
        if self.store.collection_exists(raptor_coll):
            raw = self.store.query(raptor_coll, q_vec, top_k=top_k, filters=None)
            for sr in raw:
                sr.document.metadata["is_raptor_summary"] = True
                summary_results.append(
                    SearchResult(
                        document=sr.document,
                        score=sr.score * self.summary_weight,
                        rank=sr.rank,
                    )
                )

        # Deduplicate and merge
        seen: set = set()
        merged: List[SearchResult] = []
        for r in sorted(leaf_results + summary_results, key=lambda x: x.score, reverse=True):
            if r.document.id not in seen:
                seen.add(r.document.id)
                merged.append(r)

        for i, r in enumerate(merged[:top_k], start=1):
            r.rank = i
        return merged[:top_k]
