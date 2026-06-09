"""
Background job manager for long-running operations (download / precompute / train / seed).

These take minutes, so the API can't run them inside a request. This module runs
them on a background worker thread, records their progress as an append-only event
log, and lets the routes stream that log to the browser over SSE.

Two deliberate constraints, matching the product decisions:
- **Single-flight**: at most one job runs at a time. A second start is *rejected*
  (``JobBusyError`` → 409), not queued — so the UI can show "a job is running" and
  disable the buttons rather than silently stacking work.
- **Cooperative cancellation**: jobs can't be force-killed (you can't safely kill a
  Python thread). Instead each job carries a ``cancel_event``; the work passes a
  progress callback that raises ``JobCancelled`` when the flag is set, unwinding the
  work at the next loop checkpoint.

State is in-memory and per-process — fine because the API runs as a single uvicorn
worker. A restart forgets jobs (acceptable for a demo).
"""

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from ml.utils.logging import get_logger

logger = get_logger(__name__)


class JobKind(str, Enum):
    """The operations exposed as background jobs (used for display + routing)."""

    DOWNLOAD = "download"
    PRECOMPUTE = "precompute"
    TRAIN = "train"
    SEED = "seed"


class JobStatus(str, Enum):
    """Lifecycle states. The last three are terminal (the SSE stream ends on them)."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobBusyError(RuntimeError):
    """Raised when a job is requested while another is already running (→ 409)."""


class JobNotFoundError(KeyError):
    """Raised when a job id is unknown (→ 404)."""


class JobCancelled(Exception):
    """
    Raised inside a job's progress callback to abort it cooperatively.

    Not an error condition — the runner catches it and marks the job ``cancelled``
    rather than ``failed``.
    """


@dataclass
class JobEvent:
    """One entry in a job's progress log; serialised verbatim into each SSE message."""

    status: str
    message: Optional[str] = None
    progress: Optional[float] = None  # 0..1 when known
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class Job:
    """
    A single background operation and its live state.

    ``events`` is append-only and lock-guarded so the worker thread can write while
    the SSE generator reads. ``cancel_event`` is the cooperative-cancel signal.
    """

    id: str
    kind: JobKind
    params: dict
    status: JobStatus = JobStatus.RUNNING
    events: List[JobEvent] = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: JobEvent) -> None:
        """Append an event under the lock so SSE readers see a consistent log."""
        with self._lock:
            self.events.append(event)

    def snapshot_events(self, start: int) -> List[JobEvent]:
        """Return events from index ``start`` onward — how the SSE generator tails the log."""
        with self._lock:
            return self.events[start:]

    @property
    def is_terminal(self) -> bool:
        """True once the job has finished (succeeded/failed/cancelled)."""
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)

    def cancel(self) -> None:
        """Request cancellation; takes effect at the next progress checkpoint."""
        self.cancel_event.set()

    def raise_if_cancelled(self) -> None:
        """Raise ``JobCancelled`` if cancellation was requested — call from progress callbacks."""
        if self.cancel_event.is_set():
            raise JobCancelled()


# A job body: receives its Job and does the work, emitting progress events and
# returning a result dict (stored as job.result). It should call
# ``job.raise_if_cancelled()`` (typically inside a progress callback) at loop points.
JobBody = Callable[[Job], dict]


class JobManager:
    """
    Owns the single worker thread, the active-job slot, and the job history.

    Constructed once at app startup and shared via dependency injection.
    """

    def __init__(self):
        """Set up a single-worker executor and the lock guarding the active slot."""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reid-job")
        self._jobs: Dict[str, Job] = {}
        self._active: Optional[Job] = None
        self._lock = threading.Lock()

    def active(self) -> Optional[Job]:
        """The currently-running job, or None — drives the UI's busy indicator."""
        with self._lock:
            return self._active if self._active and not self._active.is_terminal else None

    def get(self, job_id: str) -> Job:
        """Look up a job by id (for status + SSE); raises ``JobNotFoundError`` if unknown."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"No job '{job_id}'.")
        return job

    def start(self, kind: JobKind, params: dict, body: JobBody) -> Job:
        """
        Begin a job, or reject with ``JobBusyError`` if one is already running.

        The single-flight check + active-slot assignment happen under the lock so two
        racing requests can't both start. The body runs on the worker thread.
        """
        with self._lock:
            if self._active is not None and not self._active.is_terminal:
                raise JobBusyError(
                    f"A '{self._active.kind.value}' job is already running (id={self._active.id}). "
                    f"Wait for it to finish or cancel it."
                )
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
            self._jobs[job.id] = job
            self._active = job

        job.emit(JobEvent(status=JobStatus.RUNNING.value, message=f"{kind.value} started", progress=0.0))
        self._executor.submit(self._run, job, body)
        logger.info(f"Job {job.id} ({kind.value}) started with params={params}.")
        return job

    def cancel(self, job_id: str) -> Job:
        """Request cancellation of a job; no-op if it already finished."""
        job = self.get(job_id)
        if not job.is_terminal:
            job.cancel()
            logger.info(f"Job {job.id} ({job.kind.value}) cancellation requested.")
        return job

    def _run(self, job: Job, body: JobBody) -> None:
        """
        Execute the job body on the worker thread, translating outcome into status.

        ``JobCancelled`` → cancelled; any other exception → failed (with the message
        surfaced to the client). The active slot is always cleared so the next job
        can start.
        """
        try:
            result = body(job)
            job.result = result if isinstance(result, dict) else None
            job.status = JobStatus.SUCCEEDED
            job.emit(JobEvent(status=JobStatus.SUCCEEDED.value, message="done", progress=1.0, result=job.result))
            logger.info(f"Job {job.id} ({job.kind.value}) succeeded.")
        except JobCancelled:
            job.status = JobStatus.CANCELLED
            job.emit(JobEvent(status=JobStatus.CANCELLED.value, message="cancelled"))
            logger.info(f"Job {job.id} ({job.kind.value}) cancelled.")
        except Exception as error:  # noqa: BLE001 — surface any failure to the client
            job.status = JobStatus.FAILED
            job.error = str(error)
            job.emit(JobEvent(status=JobStatus.FAILED.value, error=str(error)))
            logger.error(f"Job {job.id} ({job.kind.value}) failed: {error}\n{traceback.format_exc()}")
        finally:
            with self._lock:
                if self._active is job:
                    self._active = None
