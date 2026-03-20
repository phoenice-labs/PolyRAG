"""
Phase 9: Structured observability — JSON logging with correlation IDs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Emits JSON-structured log lines compatible with any log aggregator
    (ELK, CloudWatch, Datadog, etc.).
    """

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(level)
        self._correlation_id: Optional[str] = None

    def set_correlation_id(self, cid: str) -> None:
        self._correlation_id = cid

    def new_correlation_id(self) -> str:
        cid = str(uuid.uuid4())[:8]
        self._correlation_id = cid
        return cid

    def _emit(self, level: str, message: str, **extra: Any) -> None:
        entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "logger": self._logger.name,
            "message": message,
        }
        if self._correlation_id:
            entry["correlation_id"] = self._correlation_id
        entry.update(extra)
        self._logger.info(json.dumps(entry))

    def info(self, message: str, **extra):
        self._emit("INFO", message, **extra)

    def warning(self, message: str, **extra):
        self._emit("WARNING", message, **extra)

    def error(self, message: str, **extra):
        self._emit("ERROR", message, **extra)

    def debug(self, message: str, **extra):
        self._emit("DEBUG", message, **extra)

    @contextmanager
    def timed(self, operation: str, **extra):
        """Context manager that logs operation duration."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            self._emit("INFO", f"{operation} completed",
                       duration_ms=elapsed_ms, operation=operation, **extra)


class PipelineMetrics:
    """
    In-process metrics counters for the RAG pipeline.
    Suitable for export to Prometheus via prometheus_client if installed.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, list] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def record(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)

    def summary(self) -> dict:
        import statistics
        hist_summary = {}
        for name, vals in self._histograms.items():
            if vals:
                hist_summary[name] = {
                    "count": len(vals),
                    "mean_ms": round(statistics.mean(vals), 1),
                    "p99_ms": round(sorted(vals)[int(len(vals) * 0.99)], 1),
                }
        return {"counters": dict(self._counters), "histograms": hist_summary}

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()


# Module-level singletons
pipeline_logger = StructuredLogger("polyrag.pipeline")
metrics = PipelineMetrics()
