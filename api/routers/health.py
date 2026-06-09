"""
Health/readiness endpoint.

Separate from the inference routes because a client (or a Docker healthcheck)
needs to know whether the model is loaded *before* sending work — the rest of the
API 503s until then, and this is how you find out without triggering that error.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import HealthResponse
from api.service import ReidService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    response_description="Service liveness and model readiness.",
)
def health(service: ReidService = Depends(get_service)) -> HealthResponse:
    """
    Report whether the API is up and whether a trained model is loaded.

    ``model_ready=false`` means the head checkpoint was absent at startup; train
    it and restart, after which enrollment and identification become available.
    """
    return HealthResponse(
        status="ok",
        model_ready=service.is_ready(),
        backbone=service.settings.backbone.value,
        dataset=service.settings.dataset.value,
        num_individuals=len(service.list_individuals()),
    )
