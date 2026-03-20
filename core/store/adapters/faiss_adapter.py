"""
FAISS adapter — fully in-memory ANN index (no server required).
Supports optional disk persistence via index serialisation.
Requires: pip install faiss-cpu
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.store.base import VectorStoreBase
from core.store.models import CollectionInfo, Document, SearchResult


class _CollectionState:
    """Internal state for one FAISS collection."""

    def __init__(self, dim: int, index_type: str = "Flat") -> None:
        import faiss

        self.dim = dim
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dim)  # Inner-product (cosine after L2-norm)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.id_map: List[str] = []       # faiss int-id → doc_id
        self.doc_store: Dict[str, Document] = {}
        self.deleted: set = set()         # soft-deleted faiss int-ids


class FAISSAdapter(VectorStoreBase):
    """
    FAISS vector store adapter.

    Config keys
    -----------
    mode       : "memory" (default) | "persistent"
    path       : directory for persistent mode (default: ./data/faiss)
    index_type : "Flat" (default) | "HNSW"
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._collections: Dict[str, _CollectionState] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **kwargs) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError("Install faiss: pip install faiss-cpu") from e

        if self.config.get("mode") == "persistent":
            self._load_from_disk()

    def close(self) -> None:
        if self.config.get("mode") == "persistent":
            self._save_to_disk()

    def health_check(self) -> bool:
        return True  # FAISS is always local / in-process

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        if name not in self._collections:
            index_type = self.config.get("index_type", "Flat")
            self._collections[name] = _CollectionState(embedding_dim, index_type)

    def drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        if self.config.get("mode") == "persistent":
            path = Path(self.config.get("path", "./data/faiss"))
            (path / f"{name}.index").unlink(missing_ok=True)
            (path / f"{name}.meta.pkl").unlink(missing_ok=True)

    def delete_collection(self, name: str) -> None:
        """Alias for drop_collection — used by the backends router 'Clear All' action."""
        self.drop_collection(name)

    def list_collections(self) -> list:
        """Return names of all created FAISS collections.

        In persistent mode, scans the index directory on every call so that:
        - Collections deleted by another pipeline instance are not returned.
        - Collections ingested by another pipeline instance are discovered.
        In-memory state is synced with disk (stale entries removed).
        """
        if self.config.get("mode") == "persistent":
            path = Path(self.config.get("path", "./data/faiss"))
            if not path.exists():
                return []
            on_disk = {
                f.stem for f in path.glob("*.index")
                if (path / f"{f.stem}.meta.pkl").exists()
            }
            # Remove in-memory entries whose files are gone
            for name in list(self._collections):
                if name not in on_disk:
                    self._collections.pop(name, None)
            return sorted(on_disk)
        return list(self._collections.keys())

    def collection_exists(self, name: str) -> bool:
        return name in self._collections

    def collection_info(self, name: str) -> CollectionInfo:
        state = self._collections[name]
        return CollectionInfo(
            name=name,
            count=self.count(name),
            embedding_dim=state.dim,
        )

    # ── Data operations ───────────────────────────────────────────────────────

    def upsert(self, collection: str, documents: List[Document]) -> None:
        import faiss

        state = self._collections[collection]
        for doc in documents:
            if doc.id in state.doc_store:
                # Find and soft-delete old vector
                old_faiss_id = state.id_map.index(doc.id)
                state.deleted.add(old_faiss_id)

            vec = np.array([doc.embedding], dtype=np.float32)
            faiss.normalize_L2(vec)
            faiss_id = len(state.id_map)
            state.id_map.append(doc.id)
            state.doc_store[doc.id] = doc
            state.index.add(vec)

        # Auto-persist after each batch in persistent mode
        if self.config.get("mode") == "persistent":
            self._save_to_disk()

    def delete(self, collection: str, ids: List[str]) -> None:
        state = self._collections[collection]
        for doc_id in ids:
            if doc_id in state.doc_store:
                faiss_id = state.id_map.index(doc_id)
                state.deleted.add(faiss_id)
                del state.doc_store[doc_id]

    def count(self, collection: str) -> int:
        state = self._collections[collection]
        return len(state.doc_store)  # active (non-deleted) documents

    # ── Bulk fetch ────────────────────────────────────────────────────────────

    def fetch_all(self, collection: str, limit: int = 1000) -> list:
        """Fetch up to ``limit`` documents (id + text + metadata) without vector search.

        Returns a list of dicts, each with keys: ``id``, ``text``, ``metadata``.
        """
        state = self._collections.get(collection)
        if not state:
            return []
        results = []
        for doc_id, doc in state.doc_store.items():
            results.append({"id": doc_id, "text": doc.text, "metadata": dict(doc.metadata)})
            if len(results) >= limit:
                break
        return results

    # ── Search ────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        import faiss

        state = self._collections[collection]
        if state.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Over-fetch to account for soft-deleted entries and filtering
        fetch_k = min(state.index.ntotal, top_k * 5 + len(state.deleted))
        distances, faiss_ids = state.index.search(query_vec, fetch_k)

        results: List[SearchResult] = []
        for dist, fid in zip(distances[0], faiss_ids[0]):
            if fid < 0 or fid >= len(state.id_map):
                continue
            if fid in state.deleted:
                continue
            doc_id = state.id_map[fid]
            if doc_id not in state.doc_store:
                continue
            doc = state.doc_store[doc_id]
            if filters and not _apply_filters(doc.metadata, filters):
                continue
            score = float(np.clip(dist, 0.0, 1.0))
            results.append(SearchResult(document=doc, score=score, rank=len(results) + 1))
            if len(results) >= top_k:
                break

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        import faiss

        path = Path(self.config.get("path", "./data/faiss"))
        path.mkdir(parents=True, exist_ok=True)
        for name, state in self._collections.items():
            faiss.write_index(state.index, str(path / f"{name}.index"))
            with open(path / f"{name}.meta.pkl", "wb") as f:
                pickle.dump(
                    {"id_map": state.id_map, "doc_store": state.doc_store,
                     "deleted": state.deleted, "dim": state.dim},
                    f,
                )

    def _load_from_disk(self) -> None:
        import faiss

        path = Path(self.config.get("path", "./data/faiss"))
        if not path.exists():
            return
        for index_file in path.glob("*.index"):
            name = index_file.stem
            meta_file = path / f"{name}.meta.pkl"
            if not meta_file.exists():
                continue
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            state = _CollectionState(meta["dim"])
            state.index = faiss.read_index(str(index_file))
            state.id_map = meta["id_map"]
            state.doc_store = meta["doc_store"]
            state.deleted = meta["deleted"]
            self._collections[name] = state


def _apply_filters(metadata: dict, filters: dict) -> bool:
    """Simple key=value metadata filter (AND logic)."""
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True
