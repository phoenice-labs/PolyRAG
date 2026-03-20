"""
Qdrant adapter — supports in-memory mode (no server) and remote server mode.
Requires: pip install qdrant-client
"""
from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from core.store.base import VectorStoreBase
from core.store.models import CollectionInfo, Document, SearchResult


def _str_to_uuid(s: str) -> str:
    """Convert an arbitrary string id to a deterministic UUID string."""
    import hashlib
    h = hashlib.md5(s.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class QdrantAdapter(VectorStoreBase):
    """
    Qdrant vector store adapter.

    Config keys
    -----------
    mode    : "memory" (default) | "server"
    url     : Qdrant server URL (server mode, default: http://localhost:6333)
    api_key : optional API key for Qdrant Cloud
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = None
        self._dims: Dict[str, int] = {}
        self._id_map: Dict[str, str] = {}        # doc_id → qdrant uuid
        self._rev_map: Dict[str, str] = {}        # qdrant uuid → doc_id

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **kwargs) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError("Install qdrant-client: pip install qdrant-client") from e

        mode = self.config.get("mode", "memory")
        if mode == "memory":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                url=self.config.get("url", "http://localhost:6333"),
                api_key=self.config.get("api_key"),
            )

    def close(self) -> None:
        if self.client:
            self.client.close()
        self.client = None

    def _reconnect(self) -> None:
        """Re-establish the Qdrant client connection."""
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
        self.connect()

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._dims[name] = embedding_dim
        if not self.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )

    def drop_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self._dims.pop(name, None)

    def delete_collection(self, name: str) -> None:
        """Alias for drop_collection — used by the backends router 'Clear All' action."""
        self.drop_collection(name)

    def list_collections(self) -> list:
        """Return names of all Qdrant collections."""
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception:
            return []

    def collection_exists(self, name: str) -> bool:
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            return name in existing
        except Exception:
            return False

    def collection_info(self, name: str) -> CollectionInfo:
        info = self.client.get_collection(name)
        return CollectionInfo(
            name=name,
            count=info.points_count or 0,
            embedding_dim=self._dims.get(name, 0),
        )

    # ── Data operations ───────────────────────────────────────────────────────

    def upsert(self, collection: str, documents: List[Document]) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for doc in documents:
            uid = _str_to_uuid(doc.id)
            self._id_map[doc.id] = uid
            self._rev_map[uid] = doc.id
            payload = {"text": doc.text, "_orig_id": doc.id, **doc.metadata}
            points.append(PointStruct(id=uid, vector=doc.embedding, payload=payload))

        try:
            self.client.upsert(collection_name=collection, points=points)
        except Exception as exc:
            try:
                self._reconnect()
                self.client.upsert(collection_name=collection, points=points)
            except Exception:
                raise exc

    def delete(self, collection: str, ids: List[str]) -> None:
        from qdrant_client.models import PointIdsList

        uids = [_str_to_uuid(i) for i in ids]
        self.client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=uids),
        )
        for i in ids:
            uid = self._id_map.pop(i, None)
            if uid:
                self._rev_map.pop(uid, None)

    def count(self, collection: str) -> int:
        return self.client.get_collection(collection).points_count or 0

    # ── Bulk fetch ────────────────────────────────────────────────────────────

    def fetch_all(self, collection: str, limit: int = 1000) -> list:
        """Fetch up to ``limit`` documents (id + text + metadata) using Qdrant scroll.

        Returns a list of dicts, each with keys: ``id``, ``text``, ``metadata``.
        Retries once on connection errors.
        """
        def _do_scroll():
            points, _ = self.client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in points:
                payload = dict(point.payload or {})
                orig_id = payload.pop("_orig_id", str(point.id))
                text = payload.pop("text", "")
                results.append({"id": orig_id, "text": text, "metadata": payload})
            return results

        try:
            return _do_scroll()
        except Exception as exc:
            # Attempt a single reconnect on transient connection errors
            try:
                self._reconnect()
                return _do_scroll()
            except Exception:
                raise exc

    # ── Search ────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        def _do_search():
            # qdrant-client ≥1.7 uses query_points(); older used search()
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_embedding,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=True,
                )
                return response.points
            except AttributeError:
                return self.client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=True,
                )

        try:
            hits = _do_search()
        except Exception as exc:
            try:
                self._reconnect()
                hits = _do_search()
            except Exception:
                raise exc

        results: List[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})
            orig_id = payload.pop("_orig_id", str(hit.id))
            text = payload.pop("text", "")
            # vector may be a dict (named vectors) or a list
            raw_vec = hit.vector
            if isinstance(raw_vec, dict):
                raw_vec = raw_vec.get("", list(raw_vec.values())[0] if raw_vec else [])
            vector = list(raw_vec) if raw_vec else []
            doc = Document(id=orig_id, text=text, embedding=vector, metadata=payload)
            results.append(SearchResult(document=doc, score=hit.score, rank=rank))

        return results
