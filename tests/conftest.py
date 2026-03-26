"""
Root conftest — shared fixtures for all phases.

Shared data strategy:
  - shakespeare_hamlet.txt is ingested ONCE per session into all 6 backends
    with Entity Relation extraction enabled and Knowledge Graph (Kuzu).
  - Subsequent runs detect existing collections and skip re-ingestion (fast).
  - LM Studio (localhost:1234) availability is probed once; LLM-dependent
    tests skip gracefully when it is offline.
"""
from __future__ import annotations

import socket
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"
HAMLET_PATH = DATA_DIR / "shakespeare_hamlet.txt"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

# Shared ingest parameters used by every phase
HAMLET_CHUNK_SIZE = 400
HAMLET_CHUNK_STRATEGY = "sentence"
HAMLET_OVERLAP = 50
HAMLET_COLLECTION_PREFIX = "hamlet_shared"  # suffixed per backend

ALL_BACKENDS = ["faiss", "chromadb", "qdrant", "milvus", "weaviate", "pgvector"]
DOCKER_BACKENDS = {
    "qdrant":    6333,
    "milvus":    19530,
    "weaviate":  8088,
    "pgvector":  5433,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _available_backends() -> list[str]:
    """Return backends that are actually reachable right now."""
    available = []
    for b in ALL_BACKENDS:
        if b in DOCKER_BACKENDS:
            if _port_open("localhost", DOCKER_BACKENDS[b]):
                available.append(b)
        else:
            available.append(b)  # faiss and chromadb are always local
    return available


# ── Shakespeare / Hamlet text fixtures ───────────────────────────────────────

@pytest.fixture(scope="session")
def shakespeare_raw() -> str:
    """
    Full raw text of Project Gutenberg's Complete Works of Shakespeare.
    Downloaded once per test session and cached to data/shakespeare.txt.
    """
    if not SHAKESPEARE_PATH.exists():
        import requests
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[conftest] Downloading Shakespeare from {GUTENBERG_URL} ...")
        r = requests.get(GUTENBERG_URL, timeout=120)
        r.raise_for_status()
        SHAKESPEARE_PATH.write_text(r.text, encoding="utf-8")
        print(f"[conftest] Saved {len(r.text):,} chars to {SHAKESPEARE_PATH}")
    return SHAKESPEARE_PATH.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="session")
def shakespeare_clean(shakespeare_raw: str) -> str:
    """Shakespeare text with Gutenberg header/footer stripped."""
    from core.ingestion.loader import strip_gutenberg_header_footer
    return strip_gutenberg_header_footer(shakespeare_raw)


@pytest.fixture(scope="session")
def hamlet_text() -> str:
    """
    Full Hamlet text from data/shakespeare_hamlet.txt.
    Used as the canonical shared dataset across all test phases.
    """
    if not HAMLET_PATH.exists():
        pytest.skip(f"Hamlet corpus not found at {HAMLET_PATH}")
    text = HAMLET_PATH.read_text(encoding="utf-8", errors="ignore")
    assert len(text) > 1000, "Hamlet file appears empty"
    return text


# ── LM Studio availability ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def lm_studio_available() -> bool:
    """
    Returns True if Local LM Studio is reachable at localhost:1234.
    Used by @pytest.mark.lm_studio tests to skip gracefully when offline.
    """
    try:
        import requests
        r = requests.get("http://localhost:1234/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lm_studio: mark test as requiring Local LM Studio at localhost:1234",
    )


# ── Shared hamlet ingestion — all 6 backends, ER enabled ────────────────────

def _collection_name(backend: str) -> str:
    return f"{HAMLET_COLLECTION_PREFIX}_{backend}"


def _ingest_hamlet_into_backend(backend: str, text: str):
    """
    Ingest Hamlet into one backend. Reuses existing collection if already populated.
    Returns (pipeline, already_existed: bool).
    """
    from api.deps import build_pipeline_config, create_pipeline

    collection = _collection_name(backend)
    config = build_pipeline_config(
        backend=backend,
        collection_name=collection,
        chunk_size=HAMLET_CHUNK_SIZE,
        chunk_strategy=HAMLET_CHUNK_STRATEGY,
        overlap=HAMLET_OVERLAP,
        enable_er=True,   # ER + knowledge graph enabled for all backends
    )
    pipeline = create_pipeline(config)

    # Check if collection already has data — skip re-ingest if so
    try:
        existing = pipeline.store.count(collection)
        if existing > 0:
            print(f"\n[conftest] {backend}: reusing {existing} existing chunks in '{collection}'")
            return pipeline, True
    except Exception:
        pass

    # Fresh ingest (first run or empty collection)
    print(f"\n[conftest] {backend}: ingesting Hamlet ({len(text):,} chars) into '{collection}' ...")
    result = pipeline.ingest_text(
        text,
        metadata={"source": "shakespeare_hamlet", "backend": backend},
    )
    print(f"[conftest] {backend}: ingested {result.upserted} chunks")
    assert result.upserted > 0, f"{backend}: ingest produced 0 chunks"
    return pipeline, False


@pytest.fixture(scope="session")
def hamlet_pipelines(hamlet_text) -> Dict[str, object]:
    """
    Session-scoped fixture: ingest Hamlet into ALL available backends with ER enabled.
    Returns dict mapping backend name → RAGPipeline.

    Strategy:
    - faiss and chromadb are always included (local, no Docker)
    - Docker backends (milvus, qdrant, weaviate, pgvector) included only if reachable
    - Collections are NOT dropped after the session (reused on next run)
    """
    pipelines: Dict[str, object] = {}
    for backend in _available_backends():
        try:
            pipeline, _ = _ingest_hamlet_into_backend(backend, hamlet_text)
            pipelines[backend] = pipeline
        except Exception as exc:
            print(f"\n[conftest] WARNING: {backend} ingest failed: {exc}")
    return pipelines


@pytest.fixture(scope="session")
def hamlet_pipeline_faiss(hamlet_pipelines):
    """FAISS hamlet pipeline (always available)."""
    p = hamlet_pipelines.get("faiss")
    if p is None:
        pytest.skip("FAISS hamlet pipeline not available")
    return p


@pytest.fixture(scope="session")
def hamlet_pipeline_chromadb(hamlet_pipelines):
    p = hamlet_pipelines.get("chromadb")
    if p is None:
        pytest.skip("ChromaDB hamlet pipeline not available")
    return p


@pytest.fixture(scope="session")
def hamlet_pipeline_milvus(hamlet_pipelines):
    p = hamlet_pipelines.get("milvus")
    if p is None:
        pytest.skip("Milvus not reachable — start Docker: docker compose up milvus")
    return p


@pytest.fixture(scope="session")
def hamlet_pipeline_qdrant(hamlet_pipelines):
    p = hamlet_pipelines.get("qdrant")
    if p is None:
        pytest.skip("Qdrant not reachable — start Docker: docker compose up qdrant")
    return p


@pytest.fixture(scope="session")
def hamlet_pipeline_weaviate(hamlet_pipelines):
    p = hamlet_pipelines.get("weaviate")
    if p is None:
        pytest.skip("Weaviate not reachable — start Docker: docker compose up weaviate")
    return p


@pytest.fixture(scope="session")
def hamlet_pipeline_pgvector(hamlet_pipelines):
    p = hamlet_pipelines.get("pgvector")
    if p is None:
        pytest.skip("PGVector not reachable — start Docker: docker compose up pgvector")
    return p
