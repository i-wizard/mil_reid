"""
Enrollment routes: add an individual to the gallery, or bulk-seed from the dataset.

These are the "teach the system who exists" operations. They are grouped together
because both grow the gallery (one from user uploads, one from the trained
dataset's held-out split) and both go through the service's locked, persisted
enroll path.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from api.dependencies import get_jobs, get_service, get_settings
from api.config import ApiSettings
from api.jobs import Job, JobEvent, JobKind, JobManager, JobStatus
from api.schemas import EnrollResponse, JobAccepted
from api.uploads import saved_uploads
from api.service import ReidService

router = APIRouter(tags=["enroll"])


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    response_description="Enroll (or re-enroll) an individual from reference images.",
)
def enroll(
    individual_id: str = Form(..., description="Identifier for the animal being enrolled."),
    files: List[UploadFile] = File(..., description="One or more reference images of this individual."),
    model: Optional[str] = Form(default=None, description="Target model; omit to use the default model."),
    service: ReidService = Depends(get_service),
    settings: ApiSettings = Depends(get_settings),
) -> EnrollResponse:
    """
    Enroll an individual into the selected model's gallery by averaging its images.

    Re-enrolling an existing id overwrites its prototype, so corrections are cheap.
    Uploads are streamed to temp files (validated as images, size-capped) and
    removed once embedded. 404 for an unknown model, 503 if it can't load, 400 for
    an empty id.
    """
    with saved_uploads(uploads=files, max_bytes=settings.max_upload_bytes) as paths:
        enrolled = service.enroll(model=model, individual_id=individual_id, image_paths=paths)
    return EnrollResponse(
        model=model or service.default_model(),
        individual_id=individual_id,
        images_enrolled=enrolled,
        total_individuals=len(service.list_individuals(model=model)),
    )


@router.post(
    "/gallery/seed",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    response_description="Start seeding the gallery from the dataset's held-out split (background job).",
)
def seed(
    model: Optional[str] = Form(default=None, description="Target model; omit to use the default model."),
    service: ReidService = Depends(get_service),
    jobs: JobManager = Depends(get_jobs),
) -> JobAccepted:
    """
    Start a background job that enrolls the selected model's dataset gallery split.

    Seeding embeds every reference image, so it's slow enough to run as a job: it
    returns 202 + a job id and streams per-individual progress over SSE. 409 if a job
    is already running; the underlying seed raises 404/503 (unknown/unready model)
    only once the job starts — surfaced as a failed-job event.
    """

    def body(job: Job) -> dict:
        def on_progress(done: int, total: int) -> None:
            job.raise_if_cancelled()
            job.emit(
                JobEvent(
                    status=JobStatus.RUNNING.value,
                    message=f"Enrolled {done}/{total} individuals",
                    progress=done / max(1, total),
                )
            )

        count = service.seed_from_dataset(model=model, progress_callback=on_progress)
        return {"model": model or service.default_model(), "individuals_enrolled": count}

    job = jobs.start(kind=JobKind.SEED, params={"model": model}, body=body)
    return JobAccepted(job_id=job.id, kind=job.kind.value)
