"""
Milvus adapter — supports Milvus Lite (local file / in-memory) and remote server.
Requires: pip install pymilvus>=2.4
Note: Milvus Lite (":memory:" or local .db) is Linux/macOS only.
      On Windows use Docker: docker run -p 19530:19530 milvusdb/milvus:latest
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from core.store.base import VectorStoreBase
from core.store.models import CollectionInfo, Document, SearchResult

_WINDOWS = sys.platform.startswith("win")


class MilvusAdapter(VectorStoreBase):
    """
    Milvus vector store adapter (MilvusClient unified API).

    Config keys
    -----------
    mode : "local" (Milvus Lite, Linux/macOS) | "server"
    uri  : local .db file path or ":memory:" (local mode)
    host : server hostname (server mode, default: localhost)
    port : server gRPC port (server mode, default: 19530)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = None
        self._dims: Dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **kwargs) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError("Install pymilvus: pip install pymilvus>=2.4") from e

        mode = self.config.get("mode", "local")
        if mode == "local":
            uri = self.config.get("uri", ":memory:")
            self.client = MilvusClient(uri=uri)
        else:
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 19530)
            self.client = MilvusClient(uri=f"http://{host}:{port}")

    def close(self) -> None:
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    def _reconnect(self) -> None:
        """Re-establish the Milvus connection (called on closed-channel errors)."""
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
        self.client = None
        self.connect()

    def _is_closed_channel_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "closed channel" in msg or "channel is closed" in msg

    def health_check(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception as e:
            if self._is_closed_channel_error(e):
                try:
                    self._reconnect()
                    self.client.list_collections()
                    return True
                except Exception:
                    pass
            return False

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        from pymilvus import DataType, MilvusClient

        self._dims[name] = embedding_dim
        if self.collection_exists(name):
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=512, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=embedding_dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65_535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",
            metric_type="COSINE",
        )
        try:
            self.client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
            )
        except Exception as e:
            if self._is_closed_channel_error(e):
                self._reconnect()
                self.client.create_collection(
                    collection_name=name,
                    schema=schema,
                    index_params=index_params,
                )
            else:
                raise

    def drop_collection(self, name: str) -> None:
        try:
            self.client.drop_collection(name)
        except Exception as e:
            if self._is_closed_channel_error(e):
                self._reconnect()
                try:
                    self.client.drop_collection(name)
                except Exception:
                    pass
        self._dims.pop(name, None)

    def delete_collection(self, name: str) -> None:
        """Alias for drop_collection — used by the backends router 'Clear All' action."""
        self.drop_collection(name)

    def list_collections(self) -> List[str]:
        try:
            return self.client.list_collections()
        except Exception as e:
            if self._is_closed_channel_error(e):
                self._reconnect()
                return self.client.list_collections()
            raise

    def collection_exists(self, name: str) -> bool:
        return name in self.list_collections()  # uses reconnect-aware list_collections

    def collection_info(self, name: str) -> CollectionInfo:
        stats = self.client.get_collection_stats(name)
        count = int(stats.get("row_count", 0))
        return CollectionInfo(name=name, count=count, embedding_dim=self._dims.get(name, 0))

    # ── Data operations ───────────────────────────────────────────────────────

    def upsert(self, collection: str, documents: List[Document]) -> None:
        data = []
        for doc in documents:
            row: Dict[str, Any] = {
                "id": doc.id,
                "vector": doc.embedding,
                "text": doc.text,
                **{k: str(v) for k, v in doc.metadata.items()},  # dynamic fields
            }
            data.append(row)
        try:
            self.client.upsert(collection_name=collection, data=data)
        except Exception as e:
            if self._is_closed_channel_error(e):
                self._reconnect()
                self.client.upsert(collection_name=collection, data=data)
            else:
                raise

    def delete(self, collection: str, ids: List[str]) -> None:
        self.client.delete(collection_name=collection, ids=ids)

    def count(self, collection: str) -> int:
        try:
            # get_collection_stats can return stale data; query with limit=0 for live count
            result = self.client.query(
                collection_name=collection,
                filter="",
                output_fields=["count(*)"],
            )
            if result and isinstance(result, list) and result:
                return int(result[0].get("count(*)", 0))
        except Exception:
            pass
        # Fallback to stats
        try:
            stats = self.client.get_collection_stats(collection)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    # ── Bulk fetch ────────────────────────────────────────────────────────────

    def fetch_all(self, collection: str, limit: int = 1000) -> list:
        """Fetch up to ``limit`` documents (id + text + metadata) with reconnect on closed-channel errors.

        Returns a list of dicts, each with keys: ``id``, ``text``, ``metadata``.
        The ``metadata`` dict contains all dynamic fields stored at upsert time
        (e.g. parent_id, is_parent, doc_id, start_char, end_char, section_title).
        """
        def _do_query():
            rows = self.client.query(
                collection_name=collection,
                filter="",
                output_fields=["*"],
                limit=limit,
            )
            results = []
            for row in rows:
                row = dict(row)
                doc_id = row.pop("id", "")
                text = row.pop("text", "")
                row.pop("vector", None)   # embeddings are not needed for warm-start
                results.append({"id": doc_id, "text": text, "metadata": row})
            return results

        try:
            return _do_query()
        except Exception as exc:
            if self._is_closed_channel_error(exc):
                self._reconnect()
                return _do_query()
            raise

    # ── Search ────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        expr = None
        if filters:
            parts = [f'{k} == "{v}"' for k, v in filters.items()]
            expr = " and ".join(parts)

        try:
            hits = self.client.search(
                collection_name=collection,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "id", "*"],
                filter=expr,
                search_params={"metric_type": "COSINE"},
            )
        except Exception as e:
            if self._is_closed_channel_error(e):
                self._reconnect()
                hits = self.client.search(
                    collection_name=collection,
                    data=[query_embedding],
                    limit=top_k,
                    output_fields=["text", "id", "*"],
                    filter=expr,
                    search_params={"metric_type": "COSINE"},
                )
            else:
                raise

        results: List[SearchResult] = []
        for rank, hit in enumerate(hits[0], start=1):
            entity = hit.get("entity", {})
            text = entity.pop("text", "")
            doc_id = entity.pop("id", "")
            entity.pop("vector", None)
            score = hit.get("distance", 0.0)
            # Milvus COSINE returns [-1, 1]; normalise to [0, 1]
            score = (score + 1.0) / 2.0
            doc = Document(id=doc_id, text=text, embedding=[], metadata=entity)
            results.append(SearchResult(document=doc, score=score, rank=rank))

        return results
