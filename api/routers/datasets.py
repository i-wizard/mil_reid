"""
Settings view — dataset catalogue + download/precompute jobs.

Lists the curated datasets (with a downloaded flag) and starts the two slow,
per-dataset operations as background jobs: downloading the images and precomputing
the frozen-backbone feature cache. Both return 202 + a job id; progress streams via
the jobs SSE route.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_jobs
from api.jobs import Job, JobEvent, JobKind, JobManager, JobStatus
from api.schemas import DatasetInfo, DatasetsResponse, JobAccepted
from ml.config import get_settings
from ml.data.dataset import CURATED_DATASETS, catalog, download_dataset, is_downloaded, load_dataset
from ml.features.cache import precompute_features

router = APIRouter(tags=["datasets"])


def _require_curated(name: str) -> None:
    """Reject names outside the curated catalogue with 404 (keeps downloads bounded)."""
    if name not in CURATED_DATASETS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not in the curated dataset list: {CURATED_DATASETS}.",
        )


@router.get(
    "/datasets",
    response_model=DatasetsResponse,
    response_description="Curated datasets offered for download, with their local status.",
)
def list_datasets() -> DatasetsResponse:
    """Return the curated catalogue plus, for each, whether it's already downloaded."""
    settings = get_settings()
    return DatasetsResponse(datasets=[DatasetInfo(**item) for item in catalog(settings=settings)])


@router.post(
    "/datasets/{name}/download",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    response_description="Start downloading a dataset in the background.",
)
def download(name: str, jobs: JobManager = Depends(get_jobs)) -> JobAccepted:
    """
    Kick off a background download of a curated dataset.

    404 if the name isn't curated; 409 if a job is already running. The SDK download
    is one opaque call, so progress is coarse (started → succeeded) and cancellation
    only takes effect if the worker hasn't entered the download yet.
    """
    _require_curated(name)

    def body(job: Job) -> dict:
        job.raise_if_cancelled()
        settings = get_settings({"dataset": name})
        job.emit(JobEvent(status=JobStatus.RUNNING.value, message=f"Downloading {name}…", progress=0.1))
        path = download_dataset(settings=settings)
        return {"dataset": name, "path": str(path), "downloaded": True}

    job = jobs.start(kind=JobKind.DOWNLOAD, params={"dataset": name}, body=body)
    return JobAccepted(job_id=job.id, kind=job.kind.value)


@router.post(
    "/datasets/{name}/precompute",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    response_description="Start precomputing the feature cache for a dataset in the background.",
)
def precompute(name: str, jobs: JobManager = Depends(get_jobs)) -> JobAccepted:
    """
    Kick off background feature precomputation for a downloaded dataset.

    404 if not curated; 409 if busy; 422 if the dataset isn't downloaded yet. Streams
    per-image progress and is cancellable at each checkpoint.
    """
    _require_curated(name)
    settings = get_settings({"dataset": name})
    if not is_downloaded(name=name, settings=settings):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset '{name}' is not downloaded yet — download it first.",
        )

    def body(job: Job) -> dict:
        job_settings = get_settings({"dataset": name})
        bundle = load_dataset(settings=job_settings)

        def on_progress(done: int, total: int) -> None:
            job.raise_if_cancelled()
            job.emit(
                JobEvent(
                    status=JobStatus.RUNNING.value,
                    message=f"Embedded {done}/{total} images",
                    progress=done / max(1, total),
                )
            )

        written = precompute_features(df=bundle.df, settings=job_settings, progress_callback=on_progress)
        return {"dataset": name, "images": bundle.num_images, "newly_cached": written}

    job = jobs.start(kind=JobKind.PRECOMPUTE, params={"dataset": name}, body=body)
    return JobAccepted(job_id=job.id, kind=job.kind.value)
