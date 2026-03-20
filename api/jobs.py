"""
In-memory job store for batch ingestion tracking.
Thread-safe using asyncio.Lock.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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


class JobStore:
    """In-memory thread-safe job store."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create_job(
        self,
        backend: str,
        corpus_path: Optional[str],
        config: Dict[str, Any],
    ) -> Job:
        async with self._lock:
            job = Job(
                id=str(uuid.uuid4()),
                status="pending",
                backend=backend,
                corpus_path=corpus_path,
                config=config,
            )
            self._jobs[job.id] = job
            return job

    async def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
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

    async def append_log(self, job_id: str, line: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.log_lines.append(line)
                job.updated_at = _now_iso()

    async def get_job(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_jobs(self) -> List[Job]:
        async with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
