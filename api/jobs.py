"""
Job store for batch ingestion tracking.

Persistence strategy: write-through to data/jobs.jsonl (one JSON line per job).
On startup, existing jobs are loaded from disk so status survives server restarts.
The in-memory dict is the source of truth for reads; disk is the durability layer.

Retention policy (auto-applied on startup and on each new job):
  - Completed/errored jobs are kept for JOB_RETENTION_DAYS (default 7 days).
  - Log lines for completed jobs older than LOG_RETENTION_HOURS (default 24h) are
    stripped from disk to keep the file small; the job metadata (status, result) is kept.
  - Running/pending jobs are never pruned.

Write optimisation:
  - Status changes and results → full file rewrite (infrequent).
  - Log line appends → buffered in memory only; flushed to disk every
    LOG_FLUSH_INTERVAL lines OR when job reaches done/error status.
    This eliminates the O(N*M) rewrite storm during a 459-chunk ingest.

Thread/async safety: asyncio.Lock guards all mutations.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_JOBS_FILE = Path("data/jobs.jsonl")

# Retention / flush configuration
JOB_RETENTION_DAYS: int = 7        # keep done/error job metadata for N days
LOG_RETENTION_HOURS: int = 24      # strip log_lines from done jobs older than N hours
LOG_FLUSH_INTERVAL: int = 20       # flush log lines to disk every N appends


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


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
    collection_name: Optional[str] = None  # scoped collection name (set at create time)


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
        collection_name=d.get("collection_name"),
    )


class JobStore:
    """
    Thread-safe job store with file write-through persistence.

    Jobs survive server restarts — on first access, existing jobs are loaded
    from data/jobs.jsonl. Each status mutation rewrites the entire file (safe for
    the expected job counts in a developer/integration context; use Redis
    for high-volume production job queues).

    Log appends are buffered and only flushed to disk every LOG_FLUSH_INTERVAL
    lines to avoid the full-file-rewrite storm during high-frequency ingestion.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._loaded = False
        self._unflushed_logs: Dict[str, int] = {}  # job_id → unflushed log count

    def _ensure_dir(self) -> None:
        _JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Disk I/O ──────────────────────────────────────────────────────────────

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

    # ── Retention / pruning ───────────────────────────────────────────────────

    def _prune(self) -> None:
        """
        Remove stale jobs and strip old log lines. Caller must hold the lock.

        Rules:
        - done/error jobs older than JOB_RETENTION_DAYS → deleted entirely.
        - done/error jobs older than LOG_RETENTION_HOURS → log_lines cleared
          (metadata kept for history; logs no longer needed after 24 h).
        - pending/running jobs → never pruned.

        Stale lock cleanup: jobs stuck in "running" or "pending" for more than
        STALE_JOB_HOURS (default 2h) are auto-transitioned to "error" so their
        collection lock is released. This handles server crashes mid-ingest.
        """
        STALE_JOB_HOURS = 2
        now = datetime.now(timezone.utc)
        retain_cutoff = now - timedelta(days=JOB_RETENTION_DAYS)
        log_cutoff = now - timedelta(hours=LOG_RETENTION_HOURS)
        stale_cutoff = now - timedelta(hours=STALE_JOB_HOURS)

        to_delete = []
        dirty = False
        for job_id, job in self._jobs.items():
            # Auto-expire stale running/pending jobs (server crash recovery)
            if job.status in ("running", "pending"):
                updated = _parse_iso(job.updated_at)
                if updated < stale_cutoff:
                    job.status = "error"
                    job.error = "Job timed out — likely lost due to server restart"
                    job.updated_at = _now_iso()
                    dirty = True
                continue  # don't prune active or freshly-staled jobs this cycle

            if job.status not in ("done", "error"):
                continue
            created = _parse_iso(job.created_at)
            if created < retain_cutoff:
                to_delete.append(job_id)
            elif created < log_cutoff and job.log_lines:
                job.log_lines = []
                dirty = True

        for job_id in to_delete:
            del self._jobs[job_id]
            self._unflushed_logs.pop(job_id, None)
            dirty = True

        if dirty:
            self._flush_to_disk()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _lazy_load(self) -> None:
        """Load from disk and prune on first call (inside the lock)."""
        if not self._loaded:
            self._load_from_disk()
            self._prune()
            self._loaded = True

    async def create_job(
        self,
        backend: str,
        corpus_path: Optional[str],
        config: Dict[str, Any],
        collection_name: Optional[str] = None,
    ) -> Job:
        async with self._lock:
            await self._lazy_load()
            self._prune()  # prune on every new job to keep file lean
            job = Job(
                id=str(uuid.uuid4()),
                status="pending",
                backend=backend,
                corpus_path=corpus_path,
                config=config,
                collection_name=collection_name,
            )
            self._jobs[job.id] = job
            self._unflushed_logs[job.id] = 0
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
            # Status change always flushes (infrequent — once per job lifecycle)
            self._unflushed_logs[job_id] = 0
            self._flush_to_disk()

    async def append_log(self, job_id: str, line: str) -> None:
        async with self._lock:
            await self._lazy_load()
            job = self._jobs.get(job_id)
            if not job:
                return
            job.log_lines.append(line)
            job.updated_at = _now_iso()
            # Buffered flush: only write to disk every LOG_FLUSH_INTERVAL lines
            count = self._unflushed_logs.get(job_id, 0) + 1
            self._unflushed_logs[job_id] = count
            if count >= LOG_FLUSH_INTERVAL:
                self._unflushed_logs[job_id] = 0
                self._flush_to_disk()

    async def flush_job_logs(self, job_id: str) -> None:
        """Force-flush any buffered log lines for a job (call on done/error)."""
        async with self._lock:
            if self._unflushed_logs.get(job_id, 0) > 0:
                self._unflushed_logs[job_id] = 0
                self._flush_to_disk()

    async def get_job(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            await self._lazy_load()
            return self._jobs.get(job_id)

    async def list_jobs(self) -> List[Job]:
        async with self._lock:
            await self._lazy_load()
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def delete_completed(self) -> int:
        """Delete all done/error jobs. Returns count deleted."""
        async with self._lock:
            await self._lazy_load()
            to_delete = [jid for jid, j in self._jobs.items() if j.status in ("done", "error")]
            for jid in to_delete:
                del self._jobs[jid]
                self._unflushed_logs.pop(jid, None)
            if to_delete:
                self._flush_to_disk()
            return len(to_delete)

    async def get_running_jobs_for_collection(self, collection_name: str) -> List[Job]:
        """
        Return all jobs in running/pending state for the given collection.

        Used for collection lock checking: if this returns any jobs, the collection
        is considered locked and a new ingest should be rejected with 409 Conflict.

        Matches on both the raw `collection_name` field and the scoped name stored
        inside config["ingestion"]["collection_name"], to handle callers that pass
        either form.
        """
        async with self._lock:
            await self._lazy_load()
            result = []
            for job in self._jobs.values():
                if job.status not in ("running", "pending"):
                    continue
                # Match on the explicit field (set by new create_job calls)
                if job.collection_name and (
                    job.collection_name == collection_name
                    or job.collection_name.startswith(collection_name + "_")
                    or collection_name.startswith(job.collection_name + "_")
                ):
                    result.append(job)
                    continue
                # Fallback: match on config dict for jobs created before this field existed
                cfg_coll = job.config.get("ingestion", {}).get("collection_name", "")
                if cfg_coll and (
                    cfg_coll == collection_name
                    or cfg_coll.startswith(collection_name + "_")
                    or collection_name.startswith(cfg_coll + "_")
                ):
                    result.append(job)
            return result

