"""
Health endpoint — global liveness, not model-specific.

A client or Docker healthcheck uses this to confirm the app is up and whether
*any* model is trained. Per-model readiness (which the UI banner needs) comes
from /models, since the app now serves several models.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import HealthResponse
from api.service import ReidService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    response_description="Global service liveness and how many models are trained.",
)
def health(service: ReidService = Depends(get_service)) -> HealthResponse:
    """
    Report that the API is up and how many trained models were discovered.

    ``models_available == 0`` means nothing has been trained yet — train a model
    (scripts.train) and it appears here and in /models without an API restart.
    """
    names = service.registry.names()
    return HealthResponse(
        status="ok",
        models_available=len(names),
        default_model=service.default_model() if names else None,
    )
