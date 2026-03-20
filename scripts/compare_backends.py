"""
scripts/compare_backends.py
============================
Side-by-side comparison of all Phoenice-PolyRAG vector store backends.

Runs a standard corpus + standard queries through each available backend,
collects retrieval metrics, and produces:
  • Console table (per-query and aggregate)
  • data/comparison_chart.png  — grouped bar charts
  • data/comparison_results.json — raw data for further analysis

Usage
-----
    python scripts/compare_backends.py

    # Test only specific backends:
    python scripts/compare_backends.py --backends chromadb faiss qdrant

    # Custom corpus / queries file:
    python scripts/compare_backends.py --corpus path/to/doc.txt

    # Limit corpus size (useful for large files like shakespeare.txt):
    python scripts/compare_backends.py --corpus data/shakespeare.txt --corpus-limit 120000

    # Enable all 10 retrieval methods (requires LM Studio running):
    python scripts/compare_backends.py --full-retrieval

    # Side-by-side: 5-method baseline vs all-10-method comparison:
    python scripts/compare_backends.py --compare-modes --backends chromadb faiss

    # Quiet (no per-query output, just final table):
    python scripts/compare_backends.py --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure UTF-8 output on Windows (handles Unicode box-drawing chars in tabulate tables)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Standard test corpus ───────────────────────────────────────────────────────
# Generic enterprise-document corpus — deliberately NOT Shakespeare-specific.
# Covers: policy, personnel, compliance, technical architecture, data handling.
STANDARD_CORPUS = """
## Information Security Policy

### 1. Purpose and Scope
This policy establishes the framework for protecting organisational information assets.
It applies to all employees, contractors, and third-party vendors with access to
company systems or data, regardless of their role or geographic location.

### 2. Data Classification
All data must be classified into one of four tiers:
- PUBLIC: information that can be freely shared without restriction.
- INTERNAL: information intended for use within the organisation only.
- CONFIDENTIAL: sensitive information accessible only to authorised staff.
- RESTRICTED: the highest classification; requires explicit written approval.

Employees must label documents and emails according to their classification level.
Failure to classify data correctly is a disciplinary offence.

### 3. Access Control
Access to information systems must be granted on a least-privilege basis.
Each user receives only the permissions necessary to perform their job function.
Privileged accounts — including administrator and root accounts — must use
multi-factor authentication (MFA) at all times.

Access rights must be reviewed quarterly by the relevant department head.
Dormant accounts unused for 90 days are automatically revoked.

### 4. Data Retention and Disposal
Data must be retained in accordance with the following schedule:
- Financial records: 7 years from the end of the fiscal year.
- HR records: 6 years after an employee leaves the organisation.
- Customer data: 3 years after the last transaction, unless contractually required.
- Security logs: 1 year minimum, or as mandated by regulatory bodies.

Disposal of data must use approved destruction methods:
physical media requires certified shredding; digital data requires cryptographic erasure.

### 5. Incident Response
Any suspected security incident must be reported to the Security Operations Centre
(SOC) within 2 hours of discovery. The SOC is responsible for:
a) Logging and classifying the incident in the ticketing system.
b) Initiating the appropriate response playbook.
c) Notifying senior management and, where legally required, the data protection authority.

A post-incident review must be conducted within 14 days of resolution to identify
root causes and prevent recurrence.

### 6. Roles and Responsibilities
The Chief Information Security Officer (CISO) is accountable for the overall security
programme and reports directly to the Board of Directors.
The Data Protection Officer (DPO) is responsible for compliance with GDPR and
equivalent data privacy regulations.
Line managers are responsible for ensuring their team members complete mandatory
security awareness training annually.

### 7. Consequences of Non-Compliance
Violations of this policy may result in disciplinary action up to and including
termination of employment. Deliberate or reckless breaches may be referred to
law enforcement authorities. Third-party vendors found in breach will have their
contracts terminated without liability to the organisation.

### 8. Policy Review
This policy is reviewed annually by the CISO and Legal Counsel.
Amendments require Board approval and must be communicated to all staff within
30 days of adoption. The current version supersedes all previous editions.
"""

# ── Standard queries — designed to test different retrieval signals ─────────────
STANDARD_QUERIES = [
    {
        "id": "Q1",
        "label": "Factual policy retrieval",
        "query": "What are the data classification levels?",
        "expect_keyword": "CONFIDENTIAL",
    },
    {
        "id": "Q2",
        "label": "Entity / role lookup",
        "query": "Who is responsible for GDPR compliance?",
        "expect_keyword": "Data Protection Officer",
    },
    {
        "id": "Q3",
        "label": "Retention period",
        "query": "How long must financial records be kept?",
        "expect_keyword": "7 years",
    },
    {
        "id": "Q4",
        "label": "Incident procedure",
        "query": "What happens when a security incident is discovered?",
        "expect_keyword": "SOC",
    },
    {
        "id": "Q5",
        "label": "Consequences / enforcement",
        "query": "What are the consequences of violating the security policy?",
        "expect_keyword": "termination",
    },
]

# ── Shakespeare / literary corpus queries ──────────────────────────────────────
# Auto-selected when corpus path contains "shakespeare" (case-insensitive).
# Tests retrieval across entity lookup, semantic themes, keyword, relations, events.
SHAKESPEARE_QUERIES = [
    {
        "id": "Q1",
        "label": "Character lookup",
        "query": "Who is the ghost that appears to Hamlet?",
        "expect_keyword": "Hamlet",
    },
    {
        "id": "Q2",
        "label": "Thematic / semantic",
        "query": "What themes of ambition and power appear in Macbeth?",
        "expect_keyword": "Macbeth",
    },
    {
        "id": "Q3",
        "label": "Famous quotation",
        "query": "To be or not to be — what speech is this from and who says it?",
        "expect_keyword": "Hamlet",
    },
    {
        "id": "Q4",
        "label": "Family relationship",
        "query": "What is the relationship between Romeo and Juliet's families?",
        "expect_keyword": "Montague",
    },
    {
        "id": "Q5",
        "label": "Plot / ending",
        "query": "What tragic event ends the play Romeo and Juliet?",
        "expect_keyword": "Romeo",
    },
]


def _pick_queries(corpus_path: Optional[str]) -> List[Dict]:
    """Return corpus-appropriate query set based on the corpus filename."""
    if corpus_path and "shakespeare" in Path(corpus_path).name.lower():
        return SHAKESPEARE_QUERIES
    return STANDARD_QUERIES
# Docker-based server config (used when Docker containers are detected)
DOCKER_SERVER_CONFIG = {
    "qdrant":   {"mode": "server", "url": "http://localhost:6333"},
    "weaviate": {"mode": "server", "host": "localhost", "http_port": 8088, "grpc_port": 50052},
    "milvus":   {"mode": "server", "host": "localhost", "port": 19530},
    "pgvector": {"host": "localhost", "port": 5433,
                 "database": "polyrag", "user": "postgres", "password": "postgres"},
}


def _detect_docker_backend(backend: str) -> bool:
    """
    Returns True if a Docker server for this backend is reachable.
    Used to automatically switch from in-memory to server mode.
    """
    import socket
    port_map = {
        "qdrant":   6333,
        "weaviate": 8088,
        "milvus":   19530,
        "pgvector": 5433,
    }
    port = port_map.get(backend)
    if not port:
        return False
    try:
        s = socket.create_connection(("localhost", port), timeout=1)
        s.close()
        return True
    except OSError:
        return False


def _make_config(backend: str, tmp_dir: str, full_retrieval: bool = False) -> Dict[str, Any]:
    """
    Build a minimal pipeline config dict for the given backend.
    Automatically selects server mode when Docker containers are running,
    falling back to in-memory mode otherwise.
    The graph backend is set to networkx (in-memory) to avoid file conflicts.

    Parameters
    ----------
    full_retrieval : When True, enables all 10 retrieval methods (requires LM Studio).
                     When False (default), uses the 5 LLM-independent methods only.
    """
    # Auto-detect: use Docker server mode if container is reachable
    use_docker = _detect_docker_backend(backend)

    qdrant_cfg = (DOCKER_SERVER_CONFIG["qdrant"] if use_docker and backend == "qdrant"
                  else {"mode": "memory"})
    weaviate_cfg = (DOCKER_SERVER_CONFIG["weaviate"] if use_docker and backend == "weaviate"
                    else {"mode": "embedded", "host": "localhost", "http_port": 8099, "grpc_port": 50060})
    milvus_cfg = (DOCKER_SERVER_CONFIG["milvus"] if use_docker and backend == "milvus"
                  else {"mode": "local", "uri": str(Path(__file__).resolve().parent.parent / "data" / "milvus_lite.db")})
    pgvector_cfg = (DOCKER_SERVER_CONFIG["pgvector"] if use_docker and backend == "pgvector"
                    else {"host": "localhost", "port": 5432,
                          "database": "polyrag", "user": "postgres", "password": "postgres"})

    base = {
        "store": {
            "backend": backend,
            "chromadb": {"mode": "memory"},
            "faiss":    {"mode": "memory"},
            "qdrant":   qdrant_cfg,
            "weaviate": weaviate_cfg,
            "milvus":   milvus_cfg,
            "pgvector": pgvector_cfg,
        },
        "embedding": {
            "provider": "sentence_transformer",
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "ingestion": {
            "collection_name": f"compare_{backend}",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "embed_batch_size": 32,
        },
        "retrieval": {
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "relevance_threshold": 0.0,
            "recall_multiplier": 3,
        },
        "llm": {
            "base_url": "http://localhost:1234/v1",
            "model": "mistralai/ministral-3b",
            "temperature": 0.2,
            "max_tokens": 256,
            "enable_rewrite": full_retrieval,
            "enable_stepback": False,
            "enable_multi_query": full_retrieval,
            "enable_hyde": full_retrieval,
            "n_paraphrases": 2,
        },
        "access": {"user_clearance": "INTERNAL"},
        "quality": {"min_score": 0.1, "dedup_threshold": 0.85},
        "graph": {
            "enabled": True,
            "backend": "networkx",     # in-memory, no file conflicts
            "spacy_model": "en_core_web_sm",
            "max_hops": 1,
            "graph_weight": 0.5,
            "extract_svo": True,
            "extract_cooccurrence": False,
            "llm_extraction": {"enabled": False},
        },
        "advanced_retrieval": {
            "raptor": {"enabled": full_retrieval, "n_clusters": 3, "max_tokens": 200},
            "contextual_reranker": {"enabled": full_retrieval, "top_k": 5},
            "mmr": {"enabled": True, "diversity_weight": 0.3},
        },
        "audit_log_path": str(Path(tmp_dir) / f"audit_{backend}.jsonl"),
    }
    return base


# ── Availability probe ─────────────────────────────────────────────────────────
def _probe_backend(backend: str) -> tuple[bool, str]:
    """
    Quick check whether a backend is importable and (for servers) reachable.
    Returns (available: bool, reason: str).
    """
    import socket

    def _tcp(host: str, port: int, label: str = "") -> tuple[bool, str]:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True, f"server ok ({label or f'{host}:{port}'})"
        except OSError as e:
            return False, f"Server unreachable at {host}:{port} — {e}"

    try:
        if backend == "chromadb":
            import chromadb  # noqa: F401
            return True, "ok (in-memory)"
        elif backend == "faiss":
            import faiss  # noqa: F401
            return True, "ok (in-memory)"
        elif backend == "qdrant":
            from qdrant_client import QdrantClient  # noqa: F401
            # Prefer Docker server; fallback to in-memory
            ok, msg = _tcp("localhost", 6333, "Docker :6333")
            return True, msg if ok else "ok (in-memory fallback)"
        elif backend == "weaviate":
            import weaviate  # noqa: F401
            ok, msg = _tcp("localhost", 8088, "Docker :8088")
            if not ok:
                return False, "Docker weaviate not reachable (embedded mode has Windows issues)"
            return True, msg
        elif backend == "milvus":
            from pymilvus import MilvusClient  # noqa: F401
            ok, msg = _tcp("localhost", 19530, "Docker :19530")
            return True, msg if ok else "ok (in-memory fallback)"
        elif backend == "pgvector":
            import psycopg2  # noqa: F401
            # Try Docker port 5433 first
            ok, msg = _tcp("localhost", 5433, "Docker :5433")
            if ok:
                return True, msg
            ok2, msg2 = _tcp("localhost", 5432, "localhost:5432")
            if ok2:
                return True, msg2
            return False, f"PGVector not reachable: {msg}"
        return False, f"Unknown backend: {backend}"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, str(e)


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    query_id: str
    query_label: str
    query_text: str
    backend: str
    latency_ms: float
    results_count: int
    top_score: float
    mean_score: float
    min_score: float
    expect_found: bool        # was expected keyword found in top-3 chunks?
    top_chunk_preview: str    # first 120 chars of best chunk
    retrieval_signals: str    # e.g. "vector+bm25+graph"
    error: Optional[str] = None


@dataclass
class BackendSummary:
    backend: str
    status: str               # "ok" | "skipped" | "error"
    skip_reason: str = ""
    ingest_time_ms: float = 0.0
    query_results: List[QueryResult] = field(default_factory=list)
    total_chunks_ingested: int = 0


# ── Core comparison logic ──────────────────────────────────────────────────────
def _run_backend(
    backend: str,
    corpus: str,
    queries: List[Dict],
    tmp_dir: str,
    quiet: bool,
    full_retrieval: bool = False,
    label_suffix: str = "",
) -> BackendSummary:
    """Ingest corpus + run all queries for one backend. Returns BackendSummary."""

    display_backend = f"{backend}{label_suffix}" if label_suffix else backend
    summary = BackendSummary(backend=display_backend, status="ok")

    if not quiet:
        mode_label = "10-method [FULL]" if full_retrieval else "5-method [BASELINE]"
        print(f"\n{'═'*70}")
        print(f"  Backend: {backend.upper()}  |  Retrieval: {mode_label}")
        print(f"{'═'*70}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    try:
        from orchestrator.pipeline import RAGPipeline
        config = _make_config(backend, tmp_dir, full_retrieval=full_retrieval)
        # Give each run a unique collection name to avoid data bleed between modes
        config["ingestion"]["collection_name"] = f"compare_{backend}{label_suffix}".replace("[", "_").replace("]", "")
        pipeline = RAGPipeline(config)
        pipeline.start()
    except Exception as exc:
        summary.status = "error"
        summary.skip_reason = str(exc)
        if not quiet:
            print(f"  ✗ Pipeline start failed: {exc}")
        return summary

    # ── Ingest corpus ─────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        result = pipeline.ingest_text(
            corpus,
            metadata={
                "source": "compare_corpus",
                "title": "Information Security Policy",
                "version": "2024-01",
                "classification": "INTERNAL",
            },
        )
        summary.ingest_time_ms = (time.perf_counter() - t0) * 1000
        summary.total_chunks_ingested = result.upserted
        if not quiet:
            print(f"  ✓ Ingested {result.upserted} chunks in {summary.ingest_time_ms:.0f} ms")
    except Exception as exc:
        summary.status = "error"
        summary.skip_reason = f"Ingest failed: {exc}"
        pipeline.stop()
        if not quiet:
            print(f"  ✗ Ingest failed: {exc}")
        return summary

    # ── Run queries ───────────────────────────────────────────────────────────
    for q in queries:
        qr = QueryResult(
            query_id=q["id"],
            query_label=q["label"],
            query_text=q["query"],
            backend=backend,
            latency_ms=0.0,
            results_count=0,
            top_score=0.0,
            mean_score=0.0,
            min_score=0.0,
            expect_found=False,
            top_chunk_preview="",
            retrieval_signals="",
        )
        try:
            t0 = time.perf_counter()
            results = pipeline.query(q["query"], top_k=5)
            qr.latency_ms = (time.perf_counter() - t0) * 1000

            if results:
                scores = [r.score for r in results]
                qr.results_count = len(results)
                qr.top_score = max(scores)
                qr.mean_score = sum(scores) / len(scores)
                qr.min_score = min(scores)
                qr.top_chunk_preview = results[0].document.text[:120].replace("\n", " ")

                # Check if expected keyword appears in any of top-3 results
                top3_text = " ".join(r.document.text for r in results[:3]).lower()
                qr.expect_found = q["expect_keyword"].lower() in top3_text

                # Retrieval signals (Phase 10 annotation — stored in document metadata)
                signals = set()
                for r in results:
                    sig = r.document.metadata.get("retrieval_signals", "")
                    if sig:
                        signals.add(sig)
                qr.retrieval_signals = "|".join(sorted(signals)) if signals else "vector+bm25"

        except Exception as exc:
            qr.error = str(exc)

        summary.query_results.append(qr)

        if not quiet:
            status_icon = "✓" if qr.error is None else "✗"
            found_icon = "✓" if qr.expect_found else "✗"
            print(
                f"  {status_icon} [{q['id']}] {q['label'][:30]:<30} "
                f"latency={qr.latency_ms:6.0f}ms  "
                f"top={qr.top_score:.3f}  "
                f"found={found_icon}"
            )

    pipeline.stop()
    return summary


# ── Console tables ─────────────────────────────────────────────────────────────
def _print_tables(summaries: List[BackendSummary], queries: List[Dict]) -> None:
    from tabulate import tabulate

    ok = [s for s in summaries if s.status == "ok"]
    if not ok:
        print("\nNo backends completed successfully.")
        return

    backends = [s.backend for s in ok]

    # ── Per-query table ──────────────────────────────────────────────────────
    print(f"\n{'═'*110}")
    print("  PER-QUERY COMPARISON")
    print(f"{'═'*110}")

    for q in queries:
        qid = q["id"]
        rows = []
        for s in ok:
            qr_list = [qr for qr in s.query_results if qr.query_id == qid]
            if not qr_list:
                continue
            qr = qr_list[0]
            rows.append([
                s.backend,
                f"{qr.latency_ms:6.0f}",
                qr.results_count,
                f"{qr.top_score:.4f}",
                f"{qr.mean_score:.4f}",
                "✓" if qr.expect_found else "✗",
                qr.retrieval_signals,
                (qr.error[:40] if qr.error else qr.top_chunk_preview[:60]),
            ])

        print(f"\n  [{qid}] {q['label']} — \"{q['query']}\"")
        print(f"       (expected keyword: \"{q['expect_keyword']}\")")
        print(tabulate(
            rows,
            headers=["Backend", "Latency(ms)", "Results", "Top Score", "Mean Score",
                     "KW Found", "Signals", "Top Chunk Preview"],
            tablefmt="rounded_outline",
            colalign=("left", "right", "right", "right", "right", "center", "left", "left"),
        ))

    # ── Aggregate table ──────────────────────────────────────────────────────
    print(f"\n{'═'*110}")
    print("  AGGREGATE SUMMARY")
    print(f"{'═'*110}\n")

    agg_rows = []
    for s in ok:
        if not s.query_results:
            continue
        valid_qrs = [qr for qr in s.query_results if qr.error is None]
        avg_latency = sum(qr.latency_ms for qr in valid_qrs) / len(valid_qrs) if valid_qrs else 0
        avg_top = sum(qr.top_score for qr in valid_qrs) / len(valid_qrs) if valid_qrs else 0
        avg_mean = sum(qr.mean_score for qr in valid_qrs) / len(valid_qrs) if valid_qrs else 0
        kw_hits = sum(1 for qr in valid_qrs if qr.expect_found)
        kw_total = len(valid_qrs)
        agg_rows.append([
            s.backend,
            f"{s.ingest_time_ms:.0f}",
            s.total_chunks_ingested,
            f"{avg_latency:.0f}",
            f"{avg_top:.4f}",
            f"{avg_mean:.4f}",
            f"{kw_hits}/{kw_total}",
            len([qr for qr in s.query_results if qr.error]),
        ])

    print(tabulate(
        agg_rows,
        headers=["Backend", "IngestTime(ms)", "Chunks", "AvgLatency(ms)",
                 "AvgTopScore", "AvgMeanScore", "KW Hits", "Errors"],
        tablefmt="rounded_outline",
        colalign=("left", "right", "right", "right", "right", "right", "center", "right"),
    ))

    # ── Skipped backends ─────────────────────────────────────────────────────
    skipped = [s for s in summaries if s.status != "ok"]
    if skipped:
        print(f"\n  Skipped / failed backends:")
        for s in skipped:
            print(f"    • {s.backend}: {s.skip_reason}")


# ── Chart generation ───────────────────────────────────────────────────────────
def _generate_chart(
    summaries: List[BackendSummary],
    queries: List[Dict],
    out_path: Path,
    compare_modes: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — works in any shell
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    ok = [s for s in summaries if s.status == "ok"]
    if not ok:
        print("\n[chart] No successful backends — chart not generated.")
        return

    backends = [s.backend for s in ok]
    n_b = len(backends)
    n_q = len(queries)
    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(
        "Phoenice-PolyRAG — Backend Comparison",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Grid: 3 rows × 2 cols
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35)

    # ── Plot 1: Query latency grouped bar ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(n_q)
    width = 0.8 / n_b
    for i, s in enumerate(ok):
        latencies = []
        for q in queries:
            qrs = [qr for qr in s.query_results if qr.query_id == q["id"]]
            latencies.append(qrs[0].latency_ms if qrs and not qrs[0].error else 0)
        bars = ax1.bar(
            x + i * width - (n_b - 1) * width / 2,
            latencies, width * 0.9, label=s.backend, color=colors[i % 10],
        )
        for bar, v in zip(bars, latencies):
            if v > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xlabel("Query")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Query Retrieval Latency per Backend", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{q['id']}\n{q['label']}" for q in queries], fontsize=8
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Plot 2: Average top score per backend ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    avg_top = []
    for s in ok:
        valid = [qr for qr in s.query_results if not qr.error]
        avg_top.append(sum(qr.top_score for qr in valid) / len(valid) if valid else 0)
    bars = ax2.bar(backends, avg_top, color=colors[:n_b], width=0.5, edgecolor="white")
    for bar, v in zip(bars, avg_top):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(avg_top) * 1.2 + 0.05)
    ax2.set_ylabel("Avg Top Score")
    ax2.set_title("Average Top Retrieval Score", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # ── Plot 3: Average mean score per backend ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    avg_mean = []
    for s in ok:
        valid = [qr for qr in s.query_results if not qr.error]
        avg_mean.append(sum(qr.mean_score for qr in valid) / len(valid) if valid else 0)
    bars = ax3.bar(backends, avg_mean, color=colors[:n_b], width=0.5, edgecolor="white")
    for bar, v in zip(bars, avg_mean):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylim(0, max(avg_mean) * 1.2 + 0.05)
    ax3.set_ylabel("Avg Mean Score")
    ax3.set_title("Average Mean Retrieval Score", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # ── Plot 4: Keyword hit rate ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    kw_rates = []
    for s in ok:
        valid = [qr for qr in s.query_results if not qr.error]
        rate = (sum(1 for qr in valid if qr.expect_found) / len(valid) * 100) if valid else 0
        kw_rates.append(rate)
    bar_colors = [("green" if r == 100 else "orange" if r >= 60 else "red")
                  for r in kw_rates]
    bars = ax4.bar(backends, kw_rates, color=bar_colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, kw_rates):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 120)
    ax4.axhline(100, color="green", linewidth=1, linestyle="--", alpha=0.5)
    ax4.set_ylabel("Keyword Hit Rate (%)")
    ax4.set_title("Expected Keyword Found in Top-3 Results", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # ── Plot 5: Ingest time ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ingest_times = [s.ingest_time_ms for s in ok]
    bars = ax5.bar(backends, ingest_times, color=colors[:n_b], width=0.5, edgecolor="white")
    for bar, v in zip(bars, ingest_times):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{v:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax5.set_ylabel("Ingest Time (ms)")
    ax5.set_title("Corpus Ingestion Time", fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Chart saved → {out_path}")


# ── JSON export ────────────────────────────────────────────────────────────────
def _save_json(summaries: List[BackendSummary], out_path: Path) -> None:
    data = []
    for s in summaries:
        entry = {
            "backend": s.backend,
            "status": s.status,
            "skip_reason": s.skip_reason,
            "ingest_time_ms": s.ingest_time_ms,
            "total_chunks": s.total_chunks_ingested,
            "queries": [],
        }
        for qr in s.query_results:
            entry["queries"].append({
                "id": qr.query_id,
                "label": qr.query_label,
                "query": qr.query_text,
                "latency_ms": qr.latency_ms,
                "results_count": qr.results_count,
                "top_score": qr.top_score,
                "mean_score": qr.mean_score,
                "min_score": qr.min_score,
                "expect_found": qr.expect_found,
                "retrieval_signals": qr.retrieval_signals,
                "top_chunk_preview": qr.top_chunk_preview,
                "error": qr.error,
            })
        data.append(entry)

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  ✓ JSON results saved → {out_path}")


# ── Consistency analysis ───────────────────────────────────────────────────────
def _print_consistency(summaries: List[BackendSummary], queries: List[Dict]) -> None:
    """
    Print a cross-backend consistency analysis:
    how well do backends agree on keyword recall and score ordering?
    """
    from tabulate import tabulate

    ok = [s for s in summaries if s.status == "ok"]
    if len(ok) < 2:
        return

    print(f"\n{'═'*110}")
    print("  CROSS-BACKEND CONSISTENCY ANALYSIS")
    print(f"{'═'*110}\n")

    rows = []
    for q in queries:
        qid = q["id"]
        qrs = []
        for s in ok:
            m = [qr for qr in s.query_results if qr.query_id == qid]
            if m and not m[0].error:
                qrs.append((s.backend, m[0]))

        if not qrs:
            continue

        top_scores = [qr.top_score for _, qr in qrs]
        kw_hits = [qr.expect_found for _, qr in qrs]

        score_range = max(top_scores) - min(top_scores) if top_scores else 0
        kw_agreement = "✓ ALL" if all(kw_hits) else ("✗ NONE" if not any(kw_hits) else "PARTIAL")

        # Score variance (lower = more consistent across backends)
        mean_s = sum(top_scores) / len(top_scores) if top_scores else 0
        variance = sum((s - mean_s) ** 2 for s in top_scores) / len(top_scores) if top_scores else 0
        consistency = "HIGH" if variance < 0.005 else ("MEDIUM" if variance < 0.02 else "LOW")

        rows.append([
            qid,
            q["label"],
            f"{min(top_scores):.4f}",
            f"{max(top_scores):.4f}",
            f"{score_range:.4f}",
            f"{variance:.5f}",
            consistency,
            kw_agreement,
        ])

    print(tabulate(
        rows,
        headers=["ID", "Query Label", "Min Score", "Max Score",
                 "Score Range", "Variance", "Consistency", "KW Agreement"],
        tablefmt="rounded_outline",
        colalign=("left", "left", "right", "right", "right", "right", "center", "center"),
    ))

    # Overall summary
    all_qrs_per_backend: Dict[str, List[QueryResult]] = {}
    for s in ok:
        all_qrs_per_backend[s.backend] = [qr for qr in s.query_results if not qr.error]

    print("\n  Overall keyword recall across all queries and backends:")
    for bname, qrs in all_qrs_per_backend.items():
        rate = sum(1 for qr in qrs if qr.expect_found) / len(qrs) * 100 if qrs else 0
        bar = "█" * int(rate / 5)
        print(f"    {bname:<18} {bar:<20} {rate:.0f}%")


# ── Mode delta table (compare-modes) ──────────────────────────────────────────
def _print_mode_delta(summaries: List[BackendSummary], queries: List[Dict]) -> None:
    """
    When --compare-modes is used, print a delta table showing improvement from
    baseline (5 methods) to full (10 methods) for each backend.
    """
    from tabulate import tabulate

    # Pair up baseline [5] and full [10] summaries by base backend name
    base_map: Dict[str, BackendSummary] = {}
    full_map: Dict[str, BackendSummary] = {}
    for s in summaries:
        if s.backend.endswith("[5]"):
            base_map[s.backend[:-3]] = s
        elif s.backend.endswith("[10]"):
            full_map[s.backend[:-4]] = s

    paired = [(b, base_map[b], full_map[b])
              for b in base_map if b in full_map]
    if not paired:
        return

    print(f"\n{'═'*90}")
    print("  RETRIEVAL METHOD IMPROVEMENT: BASELINE (5-method) vs FULL (10-method)")
    print(f"{'═'*90}\n")

    rows = []
    for backend, s_base, s_full in paired:
        base_valid = [qr for qr in s_base.query_results if not qr.error]
        full_valid = [qr for qr in s_full.query_results if not qr.error]

        base_top  = sum(qr.top_score for qr in base_valid)  / len(base_valid)  if base_valid else 0
        full_top  = sum(qr.top_score for qr in full_valid)  / len(full_valid)  if full_valid else 0
        base_mean = sum(qr.mean_score for qr in base_valid) / len(base_valid)  if base_valid else 0
        full_mean = sum(qr.mean_score for qr in full_valid) / len(full_valid)  if full_valid else 0
        base_kw   = sum(1 for qr in base_valid if qr.expect_found)
        full_kw   = sum(1 for qr in full_valid if qr.expect_found)
        base_lat  = sum(qr.latency_ms for qr in base_valid) / len(base_valid)  if base_valid else 0
        full_lat  = sum(qr.latency_ms for qr in full_valid) / len(full_valid)  if full_valid else 0

        top_delta  = full_top  - base_top
        mean_delta = full_mean - base_mean
        kw_delta   = full_kw   - base_kw
        lat_delta  = full_lat  - base_lat

        def _delta(v: float, positive_good: bool = True) -> str:
            arrow = ("↑" if v > 0 else "↓" if v < 0 else "→")
            if not positive_good:
                arrow = ("↑" if v > 0 else "↓" if v < 0 else "→")
            return f"{v:+.4f} {arrow}"

        rows.append([
            backend,
            f"{base_top:.4f}",
            f"{full_top:.4f}",
            _delta(top_delta),
            f"{base_mean:.4f}",
            f"{full_mean:.4f}",
            _delta(mean_delta),
            f"{base_kw}/{len(base_valid)}",
            f"{full_kw}/{len(full_valid)}",
            f"{base_lat:.0f}ms",
            f"{full_lat:.0f}ms",
        ])

    print(tabulate(
        rows,
        headers=["Backend",
                 "Base TopScore", "Full TopScore", "Δ Top Score",
                 "Base MeanScore", "Full MeanScore", "Δ Mean Score",
                 "Base KW", "Full KW",
                 "Base Latency", "Full Latency"],
        tablefmt="rounded_outline",
        colalign=("left",
                  "right", "right", "right",
                  "right", "right", "right",
                  "center", "center",
                  "right", "right"),
    ))


# ── Entry point ────────────────────────────────────────────────────────────────
ALL_BACKENDS = ["chromadb", "faiss", "qdrant", "weaviate", "milvus", "pgvector"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phoenice-PolyRAG — side-by-side backend comparison"
    )
    parser.add_argument(
        "--backends", nargs="+", default=ALL_BACKENDS,
        choices=ALL_BACKENDS, metavar="BACKEND",
        help="Backends to compare (default: all six)",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Path to a plain-text corpus file (default: built-in policy corpus)",
    )
    parser.add_argument(
        "--corpus-limit", type=int, default=None, metavar="CHARS",
        help="Truncate corpus to this many characters (useful for large files like shakespeare.txt)",
    )
    parser.add_argument(
        "--full-retrieval", action="store_true",
        help="Enable all 10 retrieval methods including LLM-dependent ones (requires LM Studio)",
    )
    parser.add_argument(
        "--compare-modes", action="store_true",
        help="Run each backend TWICE: baseline (5 methods) vs full (10 methods) — shows improvement delta",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-query output; show only final tables",
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="Skip chart generation (useful in headless environments)",
    )
    args = parser.parse_args()

    # ── Corpus loading ────────────────────────────────────────────────────────
    corpus = STANDARD_CORPUS
    corpus_path_arg = args.corpus
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"ERROR: corpus file not found: {args.corpus}")
            sys.exit(1)
        corpus = corpus_path.read_text(encoding="utf-8", errors="replace")
        if args.corpus_limit and len(corpus) > args.corpus_limit:
            corpus = corpus[:args.corpus_limit]
            print(f"  Using corpus: {args.corpus} (truncated to {args.corpus_limit:,} chars)")
        else:
            print(f"  Using corpus: {args.corpus} ({len(corpus):,} chars)")
    else:
        print(f"  Using built-in policy corpus ({len(corpus):,} chars)")

    # ── Query set (auto-detect from corpus filename) ──────────────────────────
    queries = _pick_queries(corpus_path_arg)
    query_set_name = "Shakespeare" if queries is SHAKESPEARE_QUERIES else "Enterprise Policy"

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print("  PHOENICE-POLYRAG — BACKEND COMPARISON SUITE")
    print(f"{'═'*70}")
    print(f"  Backends to test: {', '.join(args.backends)}")
    print(f"  Queries        : {len(queries)} ({query_set_name})")
    print(f"  Corpus size    : {len(corpus):,} chars")
    if args.compare_modes:
        print("  Mode           : BASELINE (5 methods) vs FULL (10 methods)")
    elif args.full_retrieval:
        print("  Mode           : FULL (all 10 retrieval methods)")
    else:
        print("  Mode           : BASELINE (5 LLM-independent methods)")

    # ── Probe availability ────────────────────────────────────────────────────
    print(f"\n  Probing backend availability...")
    available: List[str] = []
    probe_results: Dict[str, tuple] = {}
    for b in args.backends:
        ok, reason = _probe_backend(b)
        probe_results[b] = (ok, reason)
        icon = "✓" if ok else "✗"
        print(f"    {icon} {b:<12}  {reason}")
        if ok:
            available.append(b)

    if not available:
        print("\n  No backends available. Install chromadb or faiss-cpu at minimum.")
        sys.exit(1)

    # ── Run comparison ────────────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="polyrag_compare_")
    summaries: List[BackendSummary] = []

    try:
        for b in args.backends:
            if b not in available:
                summaries.append(BackendSummary(
                    backend=b,
                    status="skipped",
                    skip_reason=probe_results[b][1],
                ))
                continue

            if args.compare_modes:
                # Run baseline (5 methods) then full (10 methods)
                s_base = _run_backend(b, corpus, queries, tmp_dir, args.quiet,
                                      full_retrieval=False, label_suffix="[5]")
                summaries.append(s_base)
                s_full = _run_backend(b, corpus, queries, tmp_dir, args.quiet,
                                      full_retrieval=True, label_suffix="[10]")
                summaries.append(s_full)
            else:
                s = _run_backend(b, corpus, queries, tmp_dir, args.quiet,
                                 full_retrieval=args.full_retrieval)
                summaries.append(s)

        # ── Output ────────────────────────────────────────────────────────────
        _print_tables(summaries, queries)
        _print_consistency(summaries, queries)

        if args.compare_modes:
            _print_mode_delta(summaries, queries)

        json_path = data_dir / "comparison_results.json"
        _save_json(summaries, json_path)

        if not args.no_chart:
            chart_path = data_dir / "comparison_chart.png"
            _generate_chart(summaries, queries, chart_path,
                            compare_modes=args.compare_modes)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'═'*70}")
    print("  COMPARISON COMPLETE")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
