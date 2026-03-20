"""
Job store for batch ingestion tracking.

Persistence strategy: write-through to data/jobs.jsonl (one JSON line per job).
On startup, existing jobs are loaded from disk so status survives server restarts.
The in-memory dict is the source of truth for reads; disk is the durability layer.

Thread/async safety: asyncio.Lock guards all mutations.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_JOBS_FILE = Path("data/jobs.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    status: str  # pending | running | done | error
    backend: str
    corpus_path: Optional[str]
    config: Dict[str, Any]
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    log_lines: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _job_to_dict(job: Job) -> Dict[str, Any]:
    return asdict(job)


def _job_from_dict(d: Dict[str, Any]) -> Job:
    return Job(
        id=d["id"],
        status=d.get("status", "unknown"),
        backend=d.get("backend", ""),
        corpus_path=d.get("corpus_path"),
        config=d.get("config", {}),
        created_at=d.get("created_at", _now_iso()),
        updated_at=d.get("updated_at", _now_iso()),
        log_lines=d.get("log_lines", []),
        result=d.get("result"),
        error=d.get("error"),
    )


class JobStore:
    """
    Thread-safe job store with file write-through persistence.

    Jobs survive server restarts — on first access, existing jobs are loaded
    from data/jobs.jsonl. Each mutation rewrites the entire file (safe for
    the expected job counts in a developer/integration context; use Redis
    for high-volume production job queues).
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._loaded = False  # lazy load on first access

    def _ensure_dir(self) -> None:
        _JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _load_from_disk(self) -> None:
        """Load persisted jobs from disk (called once, lazily)."""
        self._ensure_dir()
        if not _JOBS_FILE.exists():
            return
        try:
            for line in _JOBS_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    job = _job_from_dict(d)
                    self._jobs[job.id] = job
                except (json.JSONDecodeError, KeyError):
                    pass  # skip corrupted lines
        except Exception:
            pass  # disk read failure is non-fatal; start with empty store

    def _flush_to_disk(self) -> None:
        """Rewrite the entire jobs file. Caller must hold the lock."""
        self._ensure_dir()
        try:
            lines = [json.dumps(_job_to_dict(j), default=str) for j in self._jobs.values()]
            _JOBS_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except Exception:
            pass  # disk write failure is non-fatal; in-memory state is authoritative

    async def _lazy_load(self) -> None:
        """Load from disk on first call (inside the lock)."""
        if not self._loaded:
            self._load_from_disk()
            self._loaded = True

    async def create_job(
        self,
        backend: str,
        corpus_path: Optional[str],
        config: Dict[str, Any],
    ) -> Job:
        async with self._lock:
            await self._lazy_load()
            job = Job(
                id=str(uuid.uuid4()),
                status="pending",
                backend=backend,
                corpus_path=corpus_path,
                config=config,
            )
            self._jobs[job.id] = job
            self._flush_to_disk()
            return job

    async def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            await self._lazy_load()
            job = self._jobs.get(job_id)
            if not job:
                return
            if status is not None:
                job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = _now_iso()
            self._flush_to_disk()

    async def append_log(self, job_id: str, line: str) -> None:
        async with self._lock:
            await self._lazy_load()
            job = self._jobs.get(job_id)
            if job:
                job.log_lines.append(line)
                job.updated_at = _now_iso()
                self._flush_to_disk()

    async def get_job(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            await self._lazy_load()
            return self._jobs.get(job_id)

    async def list_jobs(self) -> List[Job]:
        async with self._lock:
            await self._lazy_load()
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
