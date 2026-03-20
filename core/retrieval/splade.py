"""
Phase 3b: SPLADE Sparse Neural Retrieval.

SPLADE (Sparse Lexical And Dense Expansion) uses a BERT-based encoder to
produce sparse 30 522-dimensional vectors (one weight per WordPiece vocab
token).  Unlike BM25 (exact term statistics) it *learns* which terms are
important and expands queries with related vocabulary — so it beats BM25
on benchmarks while staying fully lexical and interpretable.

Key properties
──────────────
- Model  : naver/splade-v3   (Apache 2.0, state-of-the-art sparse retrieval)
- Vectors: 30 522-dim sparse float32  (dot-product similarity)
- Search : inverted-index dot-product  (CPU-friendly, no ANN needed)
- Persist: sparse vectors saved to disk on every add() call so the encoder
           is not re-run on server restart

Disk layout  (one folder per collection)
────────────
  data/splade/<collection>/
    docs.json      — [{id, text, metadata}, ...]
    vectors.npz    — CSR-format: term_ids, weights, offsets (one row per doc)

Integration
───────────
    index = SparseNeuralIndex()
    index.add(documents)              # encode + persist
    results = index.search("query")   # inverted-index lookup + dot product
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.store.models import Document, SearchResult

logger = logging.getLogger(__name__)

# Vocabulary size of the BERT/SPLADE tokeniser (WordPiece, uncased).
_VOCAB_SIZE = 30_522


class SparseNeuralIndex:
    """
    In-memory SPLADE index with disk persistence.

    Parameters
    ----------
    model_name  : HuggingFace model ID — must be loadable by
                  ``sentence_transformers.SparseEncoder``.
    persist_dir : Root directory for on-disk storage. One sub-folder per
                  collection is created automatically.
    batch_size  : Documents encoded per forward pass (tune for RAM / speed).
    """

    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        persist_dir: str = "./data/splade",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        self.batch_size = batch_size

        self._docs: List[Document] = []
        # Per-document sparse vectors: list[dict[term_id → weight]]
        self._sparse_vecs: List[Dict[int, float]] = []
        # Inverted index: term_id → [(doc_idx, weight), ...]
        self._inverted: Dict[int, List[Tuple[int, float]]] = {}
        # Lazy-loaded encoder
        self._encoder = None
        # Track which collection this index is bound to (set by load/add)
        self._collection: Optional[str] = None

    # ── Encoder (lazy) ────────────────────────────────────────────────────────

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SparseEncoder  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers >= 3.0 is required for SPLADE. "
                    "Run: pip install 'sentence-transformers>=3.0'"
                ) from exc
            logger.info("Loading SPLADE encoder: %s (first use — model may download)", self.model_name)
            self._encoder = SparseEncoder(self.model_name)
            logger.info("SPLADE encoder ready")
        return self._encoder

    # ── Sparse vector helpers ─────────────────────────────────────────────────

    def _to_sparse_dict(self, vec) -> Dict[int, float]:
        """Convert a dense numpy row [vocab_size] to {term_id: weight} dict."""
        arr = np.asarray(vec, dtype=np.float32)
        nonzero = np.flatnonzero(arr)
        return {int(i): float(arr[i]) for i in nonzero if arr[i] > 0}

    def _encode_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        enc = self._get_encoder()
        vecs = enc.encode_document(texts, show_progress_bar=False, batch_size=self.batch_size)
        return [self._to_sparse_dict(v) for v in vecs]

    def _encode_query(self, query: str) -> Dict[int, float]:
        enc = self._get_encoder()
        vecs = enc.encode_query([query], show_progress_bar=False)
        return self._to_sparse_dict(vecs[0])

    # ── Inverted index helpers ────────────────────────────────────────────────

    def _add_to_inverted(self, doc_idx: int, sparse: Dict[int, float]) -> None:
        for term_id, weight in sparse.items():
            self._inverted.setdefault(term_id, []).append((doc_idx, weight))

    def _rebuild_inverted(self) -> None:
        """Rebuild inverted index from _sparse_vecs (used after disk load)."""
        self._inverted = {}
        for doc_idx, sparse in enumerate(self._sparse_vecs):
            self._add_to_inverted(doc_idx, sparse)

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, documents: List[Document], collection: Optional[str] = None) -> None:
        """
        Encode documents with SPLADE and add to the index.

        Sparse vectors are appended to the on-disk store so the encoder
        need not re-run on the next server restart.
        """
        if not documents:
            return

        if collection:
            self._collection = collection

        start_idx = len(self._docs)
        texts = [d.text for d in documents]

        # Encode in batches
        sparse_vecs: List[Dict[int, float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i: i + self.batch_size]
            sparse_vecs.extend(self._encode_documents(batch_texts))

        for offset, (doc, sparse) in enumerate(zip(documents, sparse_vecs)):
            doc_idx = start_idx + offset
            self._docs.append(doc)
            self._sparse_vecs.append(sparse)
            self._add_to_inverted(doc_idx, sparse)

        # Persist incrementally
        if self._collection:
            self._save(self._collection)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        SPLADE retrieval via inverted-index dot-product.

        Only documents that share at least one non-zero term with the query
        are scored — this keeps search O(|query_terms| × avg_posting_length)
        rather than O(n_docs × vocab_size).
        """
        if not self._docs:
            return []

        query_vec = self._encode_query(query)
        if not query_vec:
            return []

        # Accumulate dot-product scores via posting lists
        scores: Dict[int, float] = {}
        for term_id, q_weight in query_vec.items():
            for doc_idx, doc_weight in self._inverted.get(term_id, []):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + q_weight * doc_weight

        if not scores:
            return []

        # Metadata filtering (AND-logic exact match)
        if filters:
            scores = {
                idx: s for idx, s in scores.items()
                if all(self._docs[idx].metadata.get(k) == v for k, v in filters.items())
            }

        # Sort descending, over-fetch when filtering to avoid empty results
        over_k = top_k * 3 if filters else top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:over_k]

        results = [
            SearchResult(document=self._docs[idx], score=float(score), rank=rank)
            for rank, (idx, score) in enumerate(sorted_items, start=1)
        ]
        return results[:top_k]

    def load(self, collection: str) -> bool:
        """
        Load a previously saved index from disk.

        Returns True if the index was loaded, False if no saved index exists.
        The inverted index is rebuilt in-memory from the stored sparse vectors.
        """
        path = self.persist_dir / collection
        docs_file = path / "docs.json"
        vecs_file = path / "vectors.npz"

        if not docs_file.exists() or not vecs_file.exists():
            return False

        try:
            with open(docs_file, encoding="utf-8") as f:
                docs_data = json.load(f)
            self._docs = [
                Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
                for d in docs_data
            ]

            data = np.load(vecs_file)
            term_ids_flat = data["term_ids"]   # int32 array, all term ids concatenated
            weights_flat  = data["weights"]    # float32, corresponding weights
            offsets       = data["offsets"]    # int32, length = n_docs + 1

            self._sparse_vecs = []
            for i in range(len(self._docs)):
                start, end = int(offsets[i]), int(offsets[i + 1])
                sparse = {
                    int(term_ids_flat[j]): float(weights_flat[j])
                    for j in range(start, end)
                }
                self._sparse_vecs.append(sparse)

            self._rebuild_inverted()
            self._collection = collection
            logger.info(
                "SPLADE index loaded from disk: %d docs, collection=%s",
                len(self._docs), collection,
            )
            return True

        except Exception as exc:
            logger.warning("SPLADE disk load failed (%s), will re-encode on add()", exc)
            self._docs = []
            self._sparse_vecs = []
            self._inverted = {}
            return False

    def clear(self) -> None:
        self._docs = []
        self._sparse_vecs = []
        self._inverted = {}

    def __len__(self) -> int:
        return len(self._docs)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, collection: str) -> None:
        """Persist docs + sparse vectors to disk in CSR-format npz."""
        path = self.persist_dir / collection
        path.mkdir(parents=True, exist_ok=True)

        # Save document metadata
        docs_data = [
            {"id": d.id, "text": d.text, "metadata": d.metadata}
            for d in self._docs
        ]
        with open(path / "docs.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False)

        # Save sparse vectors in CSR format (term_ids, weights, offsets)
        all_term_ids: List[int] = []
        all_weights: List[float] = []
        offsets: List[int] = [0]

        for sparse in self._sparse_vecs:
            for tid, w in sparse.items():
                all_term_ids.append(tid)
                all_weights.append(w)
            offsets.append(len(all_term_ids))

        np.savez_compressed(
            path / "vectors.npz",
            term_ids=np.array(all_term_ids, dtype=np.int32),
            weights=np.array(all_weights, dtype=np.float32),
            offsets=np.array(offsets, dtype=np.int32),
        )
