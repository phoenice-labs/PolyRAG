"""
Phase 1 conftest — adapter fixtures parametrized over all locally-runnable backends.
Weaviate (embedded) and PGVector are tagged as integration tests (require extra setup).
"""
from __future__ import annotations

import pytest

# Small embedding dim for fast unit tests (avoids loading a real model)
EMBEDDING_DIM = 8


# ── Adapter specifications ────────────────────────────────────────────────────
# Each spec drives the parametrized `adapter` fixture below.

LOCAL_ADAPTER_SPECS = [
    pytest.param(
        {"type": "chromadb", "config": {"mode": "memory"}},
        id="chromadb",
    ),
    pytest.param(
        {"type": "faiss", "config": {"mode": "memory", "index_type": "Flat"}},
        id="faiss",
    ),
    pytest.param(
        {"type": "qdrant", "config": {"mode": "memory"}},
        id="qdrant",
    ),
]

# Integration-only adapters (marked, skipped unless --run-integration flag)
INTEGRATION_ADAPTER_SPECS = [
    pytest.param(
        {"type": "weaviate", "config": {"mode": "embedded"}},
        id="weaviate",
        marks=pytest.mark.integration,
    ),
    pytest.param(
        {
            "type": "pgvector",
            "config": {
                "host": "localhost",
                "port": 5432,
                "database": "polyrag_test",
                "user": "postgres",
                "password": "postgres",
            },
        },
        id="pgvector",
        marks=pytest.mark.integration,
    ),
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require external services (deselect with -m 'not integration')",
    )


@pytest.fixture(params=LOCAL_ADAPTER_SPECS)
def adapter(request):
    """
    Parametrized fixture: yields a connected VectorStoreBase with an empty test collection.
    Tears down after each test.
    """
    from core.store.registry import AdapterRegistry

    spec = request.param
    adp = AdapterRegistry.create(spec["type"], spec["config"])
    adp.connect()
    adp.create_collection("test_col", EMBEDDING_DIM)
    yield adp
    try:
        adp.drop_collection("test_col")
    except Exception:
        pass
    try:
        adp.close()
    except Exception:
        pass
