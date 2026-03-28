"""
SPLADE E2E test — FAISS backend.

Verifies the full SPLADE ingestion → search pipeline:
  1. Ingest a small unique-vocabulary corpus into FAISS with SPLADE enabled.
  2. Confirm the SPLADE index was persisted to disk.
  3. Search using ONLY SPLADE (dense=False, bm25=False).
  4. Assert results are returned (SPLADE contributes > 0 candidates).
  5. Assert Method Lineage tags show "SPLADE" contribution.

Run:
    pytest tests/phase13/test_e2e_faiss_splade.py -v
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

# ── Corpus ────────────────────────────────────────────────────────────────────
# Small, distinctive texts with non-overlapping vocabulary so BM25/Dense cannot
# 'accidentally' retrieve them — only SPLADE sparse matching should work.
CORPUS = [
    "The photosynthesis process converts sunlight carbon dioxide water into glucose oxygen.",
    "Mitochondria are the powerhouse of the cell generating adenosine triphosphate energy.",
    "The Fibonacci sequence begins zero one one two three five eight thirteen twenty-one.",
    "Quantum entanglement describes correlated particles regardless of spatial separation.",
    "Tectonic plates drift causing earthquakes volcanic eruptions continental drift phenomena.",
    "Neurons transmit electrochemical signals across synaptic gaps via neurotransmitters.",
    "Blockchain ledgers use cryptographic hashes to ensure immutable transaction records.",
    "The Krebs cycle oxidises acetyl-CoA producing NADH FADH2 carbon dioxide molecules.",
    "Photovoltaic cells convert photons into electric current through semiconductor junctions.",
    "RNA polymerase transcribes DNA templates into messenger RNA during gene expression.",
]

COLLECTION = "test_splade_e2e_faiss"


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def splade_pipeline():
    """Build a fresh FAISS pipeline with SPLADE enabled, ingest corpus, yield."""
    from api.deps import build_pipeline_config, create_pipeline

    config = build_pipeline_config(
        backend="faiss",
        collection_name=COLLECTION,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        overlap=20,
        chunk_strategy="fixed",
        enable_splade=True,
        enable_mmr=False,
        enable_contextual_rerank=False,
        enable_er=False,  # skip KG extraction to keep test fast
    )
    pipeline = create_pipeline(config)

    # ── Ingest corpus ──────────────────────────────────────────────────────────
    corpus_text = "\n\n".join(CORPUS)
    pipeline.ingest_text(corpus_text, metadata={"source_id": "splade_test_corpus"})

    yield pipeline

    # ── Teardown: remove collection + SPLADE index ─────────────────────────────
    # build_pipeline_config appends model slug (e.g. "test_splade_e2e_faiss_minilm")
    scoped = config["ingestion"]["collection_name"]
    try:
        pipeline.store.delete_collection(scoped)
    except Exception:
        pass
    splade_dir = Path("data/splade") / scoped
    if splade_dir.exists():
        shutil.rmtree(splade_dir, ignore_errors=True)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSpladeIngestion:
    def test_splade_index_nonempty(self, splade_pipeline):
        """SPLADE index must have at least one document after ingestion."""
        idx = splade_pipeline.splade_index
        assert idx is not None, "splade_index should be initialised"
        assert len(idx) > 0, (
            f"SPLADE index is empty after ingestion — "
            f"check for tensor encoding errors in splade.py _to_sparse_dict"
        )

    def test_splade_files_persisted(self, splade_pipeline):
        """SPLADE index must be saved to data/splade/<scoped_collection>/."""
        scoped = splade_pipeline.config["ingestion"]["collection_name"]
        splade_dir = Path("data/splade") / scoped
        assert splade_dir.exists(), f"SPLADE persist dir not found: {splade_dir}"
        files = list(splade_dir.iterdir())
        assert len(files) >= 2, (
            f"Expected ≥2 SPLADE files (docs + vecs), got {files}"
        )


class TestSpladeSearch:
    def test_splade_only_returns_results(self, splade_pipeline):
        """Searching with ONLY SPLADE (dense=False, bm25=False) must return results."""
        results = splade_pipeline.query(
            query_text="photosynthesis glucose oxygen sunlight",
            top_k=5,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        assert len(results) > 0, (
            "SPLADE-only search returned 0 results. "
            "Ensure ingestion ran with enable_splade=True and the sparse tensor "
            "bug fix is applied (splade.py _to_sparse_dict)."
        )

    def test_splade_result_contains_expected_text(self, splade_pipeline):
        """Top SPLADE result should be the photosynthesis sentence."""
        results = splade_pipeline.query(
            query_text="photosynthesis glucose oxygen sunlight",
            top_k=3,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        assert results, "No results returned"
        top_text = results[0].document.text.lower()
        assert any(
            kw in top_text
            for kw in ("photosynthesis", "glucose", "oxygen", "sunlight")
        ), f"Top result does not contain expected keywords: {results[0].document.text!r}"

    def test_splade_method_lineage_present(self, splade_pipeline):
        """At least one result should carry SPLADE in its _method_lineage."""
        results = splade_pipeline.query(
            query_text="mitochondria adenosine triphosphate energy",
            top_k=5,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        assert results, "No results returned"
        lineages = [
            entry.get("method", "")
            for r in results
            for entry in r.document.metadata.get("_method_lineage", [])
        ]
        assert any("splade" in m.lower() or "SPLADE" in m for m in lineages), (
            f"No SPLADE lineage found in results. Lineages seen: {lineages}"
        )

    def test_retrieval_trace_initial_retrieval_nonzero(self, splade_pipeline):
        """Pipeline trace 'Initial Retrieval' candidates_after must be > 0."""
        splade_pipeline.query(
            query_text="blockchain cryptographic hashes transaction",
            top_k=5,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        trace = splade_pipeline._last_retrieval_trace
        initial = next(
            (t for t in trace if t.get("method") == "Initial Retrieval"), None
        )
        assert initial is not None, "No 'Initial Retrieval' trace entry found"
        assert initial["candidates_after"] > 0, (
            f"Initial Retrieval shows 0 candidates — SPLADE returned nothing. "
            f"Full trace: {trace}"
        )

    def test_cross_encoder_not_in_trace_when_disabled(self, splade_pipeline):
        """When enable_rerank=False, Cross-Encoder Rerank must NOT appear in trace."""
        splade_pipeline.query(
            query_text="quantum entanglement particles",
            top_k=5,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        trace = splade_pipeline._last_retrieval_trace
        methods_in_trace = [t.get("method") for t in trace]
        assert "Cross-Encoder Rerank" not in methods_in_trace, (
            f"Cross-Encoder Rerank appeared in trace when enable_rerank=False. "
            f"Trace: {methods_in_trace}"
        )

    def test_mmr_not_in_trace_when_disabled(self, splade_pipeline):
        """When pipeline was created with enable_mmr=False, MMR must NOT appear in trace."""
        splade_pipeline.query(
            query_text="tectonic plates earthquakes volcanic",
            top_k=5,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        trace = splade_pipeline._last_retrieval_trace
        methods_in_trace = [t.get("method") for t in trace]
        assert "MMR Diversity" not in methods_in_trace, (
            f"MMR Diversity appeared in trace when pipeline was built with enable_mmr=False. "
            f"Trace: {methods_in_trace}"
        )

    def test_dense_disabled_does_not_retrieve_dense(self, splade_pipeline):
        """Dense retrieval off: no dense-method lineage in results."""
        results = splade_pipeline.query(
            query_text="neurons synaptic neurotransmitters",
            top_k=5,
            enable_dense=False,
            enable_bm25=False,
            enable_splade=True,
            enable_graph=False,
            enable_rerank=False,
        )
        for r in results:
            lineage_methods = [
                e.get("method", "")
                for e in r.document.metadata.get("_method_lineage", [])
            ]
            assert "Dense" not in lineage_methods and "Vector" not in lineage_methods, (
                f"Dense lineage found even though enable_dense=False: {lineage_methods}"
            )
