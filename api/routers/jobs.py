"""
Job status + Server-Sent Events (SSE) routes.

Once a job is started (by the datasets/training/seed routes), the browser polls
``/jobs/active`` for the busy indicator and opens an ``EventSource`` on
``/jobs/{id}/events`` to watch progress live. ``/jobs/{id}/cancel`` requests
cooperative cancellation.
"""

import asyncio
import json
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_jobs
from api.jobs import Job, JobEvent, JobManager
from api.schemas import JobStatusResponse

router = APIRouter(tags=["jobs"])


def _to_status(job: Job) -> JobStatusResponse:
    """
    Project a Job into the wire DTO, surfacing the latest event's progress/message.

    The point-in-time progress/message come from the most recent event so a poll
    (or the active-indicator) reflects where the job currently is.
    """
    last = job.events[-1] if job.events else None
    return JobStatusResponse(
        id=job.id,
        kind=job.kind.value,
        status=job.status.value,
        progress=last.progress if last else None,
        message=last.message if last else None,
        result=job.result,
        error=job.error,
    )


@router.get(
    "/jobs/active",
    response_model=Optional[JobStatusResponse],
    response_description="The currently-running job, or null when idle.",
)
def active_job(jobs: JobManager = Depends(get_jobs)) -> Optional[JobStatusResponse]:
    """Return the running job (for the UI busy indicator) or null when nothing runs."""
    job = jobs.active()
    return _to_status(job) if job else None


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    response_description="Point-in-time status of a job (poll fallback for SSE).",
)
def job_status(job_id: str, jobs: JobManager = Depends(get_jobs)) -> JobStatusResponse:
    """Return one job's current status; 404 if the id is unknown (mapped from JobNotFoundError)."""
    return _to_status(jobs.get(job_id))


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=JobStatusResponse,
    response_description="Request cooperative cancellation of a running job.",
)
def cancel_job(job_id: str, jobs: JobManager = Depends(get_jobs)) -> JobStatusResponse:
    """
    Ask a running job to stop at its next checkpoint.

    409 if the job has already finished — there is nothing to cancel, and the
    client should treat that as a no-op rather than success.
    """
    job = jobs.get(job_id)
    if job.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' already {job.status.value}; nothing to cancel.",
        )
    return _to_status(jobs.cancel(job_id))


def _sse(event: JobEvent) -> str:
    """Format one event as an SSE ``data:`` frame (JSON payload, blank-line terminated)."""
    return f"data: {json.dumps(asdict(event))}\n\n"


@router.get(
    "/jobs/{job_id}/events",
    response_description="Server-Sent Events stream of a job's progress until it ends.",
)
async def job_events(job_id: str, jobs: JobManager = Depends(get_jobs)) -> StreamingResponse:
    """
    Stream a job's progress as SSE until it reaches a terminal state.

    The generator replays the whole event log from the start (so a client that
    connects slightly late still sees earlier progress), then tails new events,
    sleeping briefly when caught up. It closes once a terminal event has been sent.
    Reads the lock-guarded event log via ``snapshot_events`` so it stays consistent
    with the worker thread appending to it.
    """
    job = jobs.get(job_id)  # raises JobNotFoundError → 404 before streaming starts

    async def event_stream():
        sent = 0
        while True:
            new_events = job.snapshot_events(sent)
            for event in new_events:
                yield _sse(event)
            sent += len(new_events)

            terminal_sent = any(
                e.status in ("succeeded", "failed", "cancelled") for e in job.events[:sent]
            )
            if terminal_sent:
                break
            await asyncio.sleep(0.25)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
