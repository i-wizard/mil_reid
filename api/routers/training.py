"""
Settings view — train a model as a background job.

Starts ``run_training`` for a chosen model name + downloaded dataset, streaming
per-epoch progress over SSE. On success the new model is discoverable via /models
(the registry rescans on demand), so it appears in the UI's model picker without a
restart.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_jobs
from api.jobs import Job, JobEvent, JobKind, JobManager, JobStatus
from api.schemas import JobAccepted, TrainRequest
from ml.config import get_settings
from ml.data.dataset import CURATED_DATASETS, is_downloaded, is_precomputed
from ml.training.train import run_training

router = APIRouter(tags=["training"])


@router.post(
    "/train",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    response_description="Start training a model on a downloaded dataset in the background.",
)
def train(request: TrainRequest, jobs: JobManager = Depends(get_jobs)) -> JobAccepted:
    """
    Kick off a background training run for ``model_name`` on ``dataset``.

    404 if the dataset isn't curated; 422 if it isn't downloaded (and therefore has
    no feature cache to train on); 409 if a job is already running. Streams per-epoch
    train-loss/val-accuracy and is cancellable between epochs (no partial checkpoint
    is left, since saves only happen on an improving epoch).
    """
    if request.dataset not in CURATED_DATASETS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{request.dataset}' is not in the curated dataset list.",
        )
    probe = get_settings({"dataset": request.dataset})
    if not is_downloaded(name=request.dataset, settings=probe):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset '{request.dataset}' is not downloaded — download + precompute it first.",
        )
    # Training reads the feature cache, so it must be precomputed under the current
    # settings. Fail fast here instead of letting the job build the backbone and then
    # crash on the first batch with a per-image FileNotFoundError.
    if not is_precomputed(name=request.dataset, settings=probe):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset '{request.dataset}' has no feature cache for the current settings — precompute it first.",
        )

    def body(job: Job) -> dict:
        settings = get_settings({"model_name": request.model_name, "dataset": request.dataset})

        def on_epoch(epoch: int, total: int, train_loss: float, val_acc: float) -> None:
            job.raise_if_cancelled()
            job.emit(
                JobEvent(
                    status=JobStatus.RUNNING.value,
                    message=f"epoch {epoch}/{total} · loss={train_loss:.3f} · val_acc={val_acc:.3f}",
                    progress=epoch / max(1, total),
                )
            )

        run_training(settings=settings, progress_callback=on_epoch)
        return {"model_name": request.model_name, "dataset": request.dataset}

    job = jobs.start(
        kind=JobKind.TRAIN,
        params={"model_name": request.model_name, "dataset": request.dataset},
        body=body,
    )
    return JobAccepted(job_id=job.id, kind=job.kind.value)
