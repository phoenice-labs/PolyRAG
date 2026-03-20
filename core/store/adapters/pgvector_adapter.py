"""
PGVector adapter — PostgreSQL with pgvector extension.
Requires: pip install psycopg2-binary pgvector
          PostgreSQL server with pgvector installed.
Set up with Docker:
    docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.store.base import VectorStoreBase
from core.store.models import CollectionInfo, Document, SearchResult


class PGVectorAdapter(VectorStoreBase):
    """
    PostgreSQL + pgvector adapter.

    Config keys
    -----------
    host     : database host (default: localhost)
    port     : database port (default: 5432)
    database : database name (default: polyrag)
    user     : username (default: postgres)
    password : password (default: postgres)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.conn = None
        self._dims: Dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, **kwargs) -> None:
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError as e:
            raise ImportError(
                "Install: pip install psycopg2-binary pgvector"
            ) from e

        self.conn = psycopg2.connect(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 5432),
            dbname=self.config.get("database", "polyrag"),
            user=self.config.get("user", "postgres"),
            password=self.config.get("password", "postgres"),
        )
        self.conn.autocommit = False
        register_vector(self.conn)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
        self.conn = None

    def _reconnect(self) -> None:
        """Re-open the PostgreSQL connection (handles dropped connections)."""
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
        except Exception:
            pass
        self.connect()

    def health_check(self) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            try:
                self._reconnect()
                return True
            except Exception:
                return False

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, name: str, embedding_dim: int, **kwargs) -> None:
        self._dims[name] = embedding_dim
        safe = _safe_table(name)
        safe_unquoted = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
        idx_name = f"{safe_unquoted}_vec_idx"
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {safe} (
                    id      TEXT PRIMARY KEY,
                    text    TEXT,
                    vector  vector({embedding_dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS {idx_name}
                    ON {safe} USING ivfflat (vector vector_cosine_ops)
                    WITH (lists = 10);
                """
            )
        self.conn.commit()

    def drop_collection(self, name: str) -> None:
        safe = _safe_table(name)
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {safe} CASCADE;")
        self.conn.commit()
        self._dims.pop(name, None)

    def delete_collection(self, name: str) -> None:
        """Alias for drop_collection — used by the backends router 'Clear All' action."""
        self.drop_collection(name)

    def list_collections(self) -> list:
        """Return names of all tables in the public schema (our collections)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"
                )
                return [row[0] for row in cur.fetchall()]
        except Exception:
            return []

    def collection_exists(self, name: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s AND table_schema = 'public';",
                (name.lower(),),
            )
            return cur.fetchone() is not None

    def collection_info(self, name: str) -> CollectionInfo:
        safe = _safe_table(name)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {safe};")
            cnt = cur.fetchone()[0]
        return CollectionInfo(name=name, count=cnt, embedding_dim=self._dims.get(name, 0))

    # ── Data operations ───────────────────────────────────────────────────────

    def upsert(self, collection: str, documents: List[Document]) -> None:
        import json

        safe = _safe_table(collection)
        with self.conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {safe} (id, text, vector, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET text = EXCLUDED.text,
                            vector = EXCLUDED.vector,
                            metadata = EXCLUDED.metadata;
                    """,
                    (doc.id, doc.text, doc.embedding, json.dumps(doc.metadata)),
                )
        self.conn.commit()

    def delete(self, collection: str, ids: List[str]) -> None:
        safe = _safe_table(collection)
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {safe} WHERE id = ANY(%s);", (ids,))
        self.conn.commit()

    def count(self, collection: str) -> int:
        safe = _safe_table(collection)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {safe};")
            return cur.fetchone()[0]

    # ── Bulk fetch ────────────────────────────────────────────────────────────

    def fetch_all(self, collection: str, limit: int = 1000) -> list:
        """Fetch up to ``limit`` documents (id + text + metadata) without vector search.

        Returns a list of dicts, each with keys: ``id``, ``text``, ``metadata``.
        Retries once on connection errors.
        """
        import json

        safe = _safe_table(collection)

        def _do_fetch():
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT id, text, metadata FROM {safe} LIMIT %s;", (limit,))
                rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "text": row[1],
                    "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
                }
                for row in rows
            ]

        try:
            return _do_fetch()
        except Exception as exc:
            try:
                self._reconnect()
                return _do_fetch()
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
        import json

        safe = _safe_table(collection)
        where_clause = ""
        # Start with the two query_embedding params; filter values appended after
        params: list = [query_embedding, query_embedding, top_k]

        if filters:
            conditions = []
            filter_params = []
            for k, v in filters.items():
                # Sanitise key to only alphanumeric + underscore (never user-supplied raw SQL)
                safe_k = "".join(c for c in k if c.isalnum() or c == "_")
                # Use ->> JSONB text extraction; key is sanitised (no parameterisation needed for key name)
                conditions.append("metadata->>" + "'" + safe_k + "'" + " = %s")
                filter_params.append(str(v))
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
                # Filter params go BEFORE the score/ORDER BY embedding params
                params = [query_embedding] + filter_params + [query_embedding, top_k]

        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    f"""
                    SELECT id, text, vector, metadata,
                           1 - (vector <=> %s::vector) AS score
                    FROM {safe}
                    {where_clause}
                    ORDER BY vector <=> %s::vector
                    LIMIT %s;
                    """,
                    params,
                )
                rows = cur.fetchall()
            except Exception as exc:
                # Retry once on connection loss
                self.conn.rollback()
                self._reconnect()
                cur2 = self.conn.cursor()
                cur2.execute(
                    f"""
                    SELECT id, text, vector, metadata,
                           1 - (vector <=> %s::vector) AS score
                    FROM {safe}
                    {where_clause}
                    ORDER BY vector <=> %s::vector
                    LIMIT %s;
                    """,
                    params,
                )
                rows = cur2.fetchall()

        results: List[SearchResult] = []
        for rank, (doc_id, text, vector, meta, score) in enumerate(rows, start=1):
            meta_dict = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            vec_list = list(vector) if vector is not None else []
            doc = Document(id=doc_id, text=text, embedding=vec_list, metadata=meta_dict)
            results.append(SearchResult(document=doc, score=float(score), rank=rank))

        return results


def _safe_table(name: str) -> str:
    """Sanitise collection name for use as a PostgreSQL table identifier."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    return f'"{safe}"'
