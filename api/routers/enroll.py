"""
Enrollment routes: add an individual to the gallery, or bulk-seed from the dataset.

These are the "teach the system who exists" operations. They are grouped together
because both grow the gallery (one from user uploads, one from the trained
dataset's held-out split) and both go through the service's locked, persisted
enroll path.
"""

from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependencies import get_service, get_settings
from api.config import ApiSettings
from api.schemas import EnrollResponse, SeedResponse
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
    service: ReidService = Depends(get_service),
    settings: ApiSettings = Depends(get_settings),
) -> EnrollResponse:
    """
    Enroll an individual by averaging the embeddings of its reference images.

    Re-enrolling an existing id overwrites its prototype, so corrections are cheap.
    Uploads are streamed to temp files (validated as images, size-capped) and
    removed once embedded. Returns 503 if no model is loaded, 400 for an empty id.
    """
    with saved_uploads(uploads=files, max_bytes=settings.max_upload_bytes) as paths:
        enrolled = service.enroll(individual_id=individual_id, image_paths=paths)
    return EnrollResponse(
        individual_id=individual_id,
        images_enrolled=enrolled,
        total_individuals=len(service.list_individuals()),
    )


@router.post(
    "/gallery/seed",
    response_model=SeedResponse,
    response_description="Seed the gallery from the trained dataset's held-out gallery split.",
)
def seed(service: ReidService = Depends(get_service)) -> SeedResponse:
    """
    Auto-enroll the dataset's gallery split so identification works immediately.

    Convenience for the demo: rather than enrolling animals by hand, this enrolls
    the same reference set the evaluation uses. Returns 503 if no model is loaded.
    """
    count = service.seed_from_dataset()
    return SeedResponse(individuals_enrolled=count)
