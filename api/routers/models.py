"""
Model catalogue route.

Multi-model means the client must discover which models exist and which is the
default before it can sensibly pick one. This endpoint backs the UI's model
picker and lets any consumer see each model's readiness and provenance
(dataset/backbone) without loading torch.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import ModelInfo, ModelsResponse
from api.service import ReidService

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    response_description="List the trained models available to serve.",
)
def list_models(service: ReidService = Depends(get_service)) -> ModelsResponse:
    """
    Return every discovered model with its readiness, provenance, and enrolled count.

    Drives the UI's model selector; ``default_model`` is what requests target when
    they omit the ``model`` field.
    """
    return ModelsResponse(
        models=[ModelInfo(**m) for m in service.list_models()],
        default_model=service.default_model() if service.registry.names() else None,
    )
