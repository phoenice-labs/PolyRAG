"""
ChromaDB adapter — supports in-memory (ephemeral) and persistent modes.
Requires: pip install chromadb
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.store.base import VectorStoreBase
from core.store.models import CollectionInfo, Document, SearchResult


class ChromaDBAdapter(VectorStoreBase):
    """
    ChromaDB vector store adapter.

    Config keys
    -----------
    mode : "memory" (default) | "persistent"
    path : local directory for persistent mode (default: ./data/chromadb)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = None
        self._embedding_dim: Dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **kwargs) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("Install chromadb: pip install chromadb") from e

        mode = self.config.get("mode", "memory")
        if mode == "memory":
            self.client = chromadb.EphemeralClient()
        else:
            path = self.config.get("path", "./data/chromadb")
            self.client = chromadb.PersistentClient(path=path)

    def close(self) -> None:
        self.client = None

    def health_check(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        self._embedding_dim[name] = embedding_dim
        try:
            self.client.get_collection(name)
        except Exception:
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    def drop_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self._embedding_dim.pop(name, None)

    def delete_collection(self, name: str) -> None:
        """Alias for drop_collection — used by the backends router 'Clear All' action."""
        self.drop_collection(name)

    def collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def list_collections(self) -> list:
        """Return names of all collections."""
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception:
            return []

    def collection_info(self, name: str) -> CollectionInfo:
        coll = self.client.get_collection(name)
        return CollectionInfo(
            name=name,
            count=coll.count(),
            embedding_dim=self._embedding_dim.get(name, 0),
        )

    # ── Data operations ───────────────────────────────────────────────────────

    def upsert(self, collection: str, documents: List[Document]) -> None:
        coll = self.client.get_collection(collection)
        coll.upsert(
            ids=[d.id for d in documents],
            embeddings=[d.embedding for d in documents],
            documents=[d.text for d in documents],
            metadatas=[d.metadata for d in documents],
        )

    def delete(self, collection: str, ids: List[str]) -> None:
        coll = self.client.get_collection(collection)
        coll.delete(ids=ids)

    def count(self, collection: str) -> int:
        return self.client.get_collection(collection).count()

    # ── Bulk fetch ────────────────────────────────────────────────────────────

    def fetch_all(self, collection: str, limit: int = 1000) -> list:
        """Fetch up to ``limit`` documents (id + text + metadata) without vector search.

        Returns a list of dicts, each with keys: ``id``, ``text``, ``metadata``.
        """
        coll = self.client.get_collection(collection)
        raw = coll.get(limit=limit, include=["documents", "metadatas"])
        results = []
        for doc_id, text, meta in zip(
            raw.get("ids", []),
            raw.get("documents", []),
            raw.get("metadatas", []) or [{}] * len(raw.get("ids", [])),
        ):
            results.append({"id": doc_id, "text": text, "metadata": dict(meta or {})})
        return results

    # ── Search ────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        coll = self.client.get_collection(collection)
        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, coll.count() or 1),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if filters:
            kwargs["where"] = filters

        raw = coll.query(**kwargs)

        results: List[SearchResult] = []
        for rank, (doc_id, text, meta, dist, emb) in enumerate(
            zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
                raw["embeddings"][0],
            ),
            start=1,
        ):
            # ChromaDB returns cosine *distance* [0,2]; convert to similarity [0,1]
            score = max(0.0, 1.0 - dist / 2.0)
            doc = Document(id=doc_id, text=text, embedding=list(emb), metadata=meta)
            results.append(SearchResult(document=doc, score=score, rank=rank))

        return results
