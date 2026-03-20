"""
GraphStoreRegistry — factory for all Knowledge Graph backends.
Mirrors AdapterRegistry pattern for vector stores.
"""
from __future__ import annotations

from typing import Any, Dict


_BACKEND_MAP = {
    "networkx": ("core.graph.store_networkx", "NetworkXGraphStore"),
    "kuzu":     ("core.graph.store_kuzu",     "KuzuGraphStore"),
    "neo4j":    ("core.graph.store_neo4j",    "Neo4jGraphStore"),
}


def get_graph_store(backend: str, config: Dict[str, Any]):
    """Instantiate the requested graph store backend from config.

    Parameters
    ----------
    backend : "networkx" | "kuzu" | "neo4j"
    config  : the full graph sub-config dict from config.yaml
    """
    if backend not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown graph backend '{backend}'. "
            f"Choose from: {list(_BACKEND_MAP)}"
        )

    module_path, class_name = _BACKEND_MAP[backend]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if backend == "networkx":
        return cls()

    if backend == "kuzu":
        db_path = config.get("kuzu", {}).get("db_path", "./data/graph.kuzu")
        return cls(db_path=db_path)

    if backend == "neo4j":
        neo4j_cfg = config.get("neo4j", {})
        return cls(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            user=neo4j_cfg.get("user", "neo4j"),
            password=neo4j_cfg.get("password", "password"),
            database=neo4j_cfg.get("database", "neo4j"),
        )
