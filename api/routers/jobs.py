"""
Jobs router: GET /api/jobs, GET /api/jobs/{job_id}
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from api.deps import get_job_store
from api.schemas import JobStatus

router = APIRouter(tags=["jobs"])


def _job_to_schema(job) -> JobStatus:
    return JobStatus(
        job_id=job.id,
        status=job.status,
        backend=job.backend,
        created_at=job.created_at,
        updated_at=job.updated_at,
        log_lines=job.log_lines,
        result=job.result,
        error=job.error,
        collection_name=job.collection_name,
    )


@router.get("/jobs", response_model=List[JobStatus])
async def list_jobs() -> List[JobStatus]:
    """Return all jobs sorted by created_at desc."""
    store = get_job_store()
    jobs = await store.list_jobs()
    return [_job_to_schema(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Return full job details including log lines."""
    store = get_job_store()
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_schema(job)
