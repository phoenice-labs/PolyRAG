"""
Weaviate adapter — supports embedded (local, no server) and remote server modes.
Requires: pip install weaviate-client>=4.5
Note: Embedded mode downloads a Weaviate binary on first use (~30 MB).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from core.store.base import VectorStoreBase
from core.store.models import CollectionInfo, Document, SearchResult


def _to_weaviate_class(name: str) -> str:
    """Weaviate class names must start with an uppercase letter."""
    return name[0].upper() + name[1:] if name else name


def _str_to_uuid(s: str) -> str:
    h = hashlib.md5(s.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class WeaviateAdapter(VectorStoreBase):
    """
    Weaviate vector store adapter.

    Config keys
    -----------
    mode     : "embedded" (default, downloads binary once) | "server"
    host     : server hostname (server mode, default: localhost)
    http_port: server HTTP port (default: 8080)
    grpc_port: server gRPC port (default: 50051)
    api_key  : optional API key
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = None
        self._dims: Dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **kwargs) -> None:
        try:
            import weaviate
        except ImportError as e:
            raise ImportError(
                "Install weaviate-client: pip install weaviate-client>=4.5"
            ) from e

        mode = self.config.get("mode", "embedded")
        if mode == "embedded":
            self.client = weaviate.connect_to_embedded()
        else:
            try:
                from weaviate.classes.init import AdditionalConfig, Timeout
                additional = AdditionalConfig(timeout=Timeout(init=5, query=30, insert=60))
            except ImportError:
                additional = None

            kwargs_conn: dict = dict(
                http_host=self.config.get("host", "localhost"),
                http_port=self.config.get("http_port", 8080),
                http_secure=False,
                grpc_host=self.config.get("host", "localhost"),
                grpc_port=self.config.get("grpc_port", 50051),
                grpc_secure=False,
                auth_credentials=(
                    weaviate.auth.AuthApiKey(self.config["api_key"])
                    if self.config.get("api_key")
                    else None
                ),
            )
            if additional is not None:
                kwargs_conn["additional_config"] = additional
            self.client = weaviate.connect_to_custom(**kwargs_conn)

    def close(self) -> None:
        if self.client:
            self.client.close()
        self.client = None

    def health_check(self) -> bool:
        try:
            return self.client.is_ready()
        except Exception:
            return False

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        from weaviate.classes.config import Configure, DataType, Property

        self._dims[name] = embedding_dim
        wname = _to_weaviate_class(name)
        if not self.collection_exists(name):
            self.client.collections.create(
                name=wname,
                vectorizer_config=Configure.Vectorizer.none(),
                # Declare common metadata properties so Weaviate schema handles them cleanly.
                # Additional dynamic properties are auto-created on first upsert (auto_schema=True).
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="orig_id", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="parent_id", data_type=DataType.TEXT),
                    Property(name="is_parent", data_type=DataType.TEXT),
                    Property(name="start_char", data_type=DataType.TEXT),
                    Property(name="end_char", data_type=DataType.TEXT),
                    Property(name="section_title", data_type=DataType.TEXT),
                ],
            )

    def drop_collection(self, name: str) -> None:
        wname = _to_weaviate_class(name)
        try:
            self.client.collections.delete(wname)
        except Exception:
            pass
        self._dims.pop(name, None)

    def delete_collection(self, name: str) -> None:
        """Alias for drop_collection — used by the backends router 'Clear All' action."""
        self.drop_collection(name)

    def list_collections(self) -> list:
        """Return names of all Weaviate collections, converted back to lowercase-first form
        to match the original collection names used by all other adapters."""
        try:
            all_colls = self.client.collections.list_all()
            # list_all() returns a dict {name: config} or a list depending on client version
            if isinstance(all_colls, dict):
                names = list(all_colls.keys())
            else:
                names = [c.name for c in all_colls]
            # Weaviate capitalises the first letter (Polyrag_docs → polyrag_docs).
            # Reverse that so names match what every other backend returns.
            return [n[0].lower() + n[1:] if n else n for n in names]
        except Exception:
            return []

    def collection_exists(self, name: str) -> bool:
        wname = _to_weaviate_class(name)
        return self.client.collections.exists(wname)

    def collection_info(self, name: str) -> CollectionInfo:
        wname = _to_weaviate_class(name)
        coll = self.client.collections.get(wname)
        agg = coll.aggregate.over_all(total_count=True)
        return CollectionInfo(
            name=name,
            count=agg.total_count or 0,
            embedding_dim=self._dims.get(name, 0),
        )

    # ── Data operations ───────────────────────────────────────────────────────

    def _build_props(self, doc: "Document") -> dict:
        """Build a Weaviate-safe property dict for one document.

        All metadata values are coerced to str (Weaviate schema declares every
        metadata field as DataType.TEXT) and None values are dropped to avoid
        type-mismatch errors such as "not a string, but float64".
        """
        props: dict = {"text": doc.text or "", "orig_id": doc.id}
        for k, v in doc.metadata.items():
            if v is not None:
                props[k] = str(v)
        return props

    def upsert(self, collection: str, documents: List[Document]) -> None:
        """Insert or update documents in the collection.

        Uses ``data.insert_many()`` which is thread-safe (unlike ``batch.dynamic()``
        which spawns its own background thread and fails silently when called from an
        asyncio thread-pool worker via ``asyncio.to_thread``).
        """
        from weaviate.classes.data import DataObject

        wname = _to_weaviate_class(collection)
        coll = self.client.collections.get(wname)

        objects = [
            DataObject(
                properties=self._build_props(doc),
                vector=doc.embedding,
                uuid=_str_to_uuid(doc.id),
            )
            for doc in documents
        ]

        response = coll.data.insert_many(objects)

        # insert_many returns a BatchObjectReturn; check for errors.
        if response.has_errors:
            errors = [str(e.message) for e in (response.errors or {}).values()]
            # Reconnect and retry once — handles stale gRPC channels.
            try:
                self.close()
            except Exception:
                pass
            self.connect()
            coll2 = self.client.collections.get(wname)
            response2 = coll2.data.insert_many(objects)
            if response2.has_errors:
                errs2 = [str(e.message) for e in (response2.errors or {}).values()]
                raise RuntimeError(
                    f"Weaviate insert_many into '{wname}' failed after reconnect: "
                    f"{errs2[:3]}"
                )

    def delete(self, collection: str, ids: List[str]) -> None:
        from weaviate.classes.query import Filter

        wname = _to_weaviate_class(collection)
        coll = self.client.collections.get(wname)
        for doc_id in ids:
            coll.data.delete_by_id(_str_to_uuid(doc_id))

    def count(self, collection: str) -> int:
        wname = _to_weaviate_class(collection)
        coll = self.client.collections.get(wname)
        agg = coll.aggregate.over_all(total_count=True)
        return agg.total_count or 0

    # ── Bulk fetch ────────────────────────────────────────────────────────────

    def fetch_all(self, collection: str, limit: int = 1000) -> list:
        """Fetch up to ``limit`` documents (id + text + metadata) without vector search.

        Returns a list of dicts, each with keys: ``id``, ``text``, ``metadata``.
        """
        wname = _to_weaviate_class(collection)
        coll = self.client.collections.get(wname)
        response = coll.query.fetch_objects(limit=limit, include_vector=False)
        results = []
        for obj in response.objects:
            props = dict(obj.properties)
            orig_id = props.pop("orig_id", str(obj.uuid))
            text = props.pop("text", "")
            # Remaining props are metadata (doc_id, parent_id, is_parent, etc.)
            results.append({"id": orig_id, "text": text, "metadata": props})
        return results

    # ── Search ────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        from weaviate.classes.query import MetadataQuery

        wname = _to_weaviate_class(collection)
        coll = self.client.collections.get(wname)

        kwargs: dict = dict(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, score=True),
            include_vector=True,
        )
        # Basic metadata filter support
        if filters:
            from weaviate.classes.query import Filter as WFilter
            flt = None
            for k, v in filters.items():
                condition = WFilter.by_property(k).equal(v)
                flt = condition if flt is None else flt & condition
            kwargs["filters"] = flt

        response = coll.query.near_vector(**kwargs)

        results: List[SearchResult] = []
        for rank, obj in enumerate(response.objects, start=1):
            props = dict(obj.properties)
            orig_id = props.pop("orig_id", str(obj.uuid))
            text = props.pop("text", "")
            vector = list(obj.vector.get("default", [])) if obj.vector else []
            dist = obj.metadata.distance if obj.metadata else 1.0
            score = max(0.0, 1.0 - (dist or 0.0))
            doc = Document(id=orig_id, text=text, embedding=vector, metadata=props)
            results.append(SearchResult(document=doc, score=score, rank=rank))

        return results
