"""
Gallery management routes: list, delete one, reset all.

These are read/curate operations on the enrolled set. None of them need the model
to be loaded (they only touch the gallery), so unlike enroll/identify they work
even when no head is trained — useful for inspecting or clearing state.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_service
from api.schemas import IndividualsResponse, MessageResponse
from api.service import ReidService

router = APIRouter(tags=["gallery"])


@router.get(
    "/individuals",
    response_model=IndividualsResponse,
    response_description="List the individuals enrolled in a model's gallery.",
)
def list_individuals(
    model: Optional[str] = None,
    service: ReidService = Depends(get_service),
) -> IndividualsResponse:
    """Return the enrolled individual ids and count for the selected model (default if omitted)."""
    individuals = service.list_individuals(model=model)
    return IndividualsResponse(individuals=individuals, count=len(individuals))


@router.delete(
    "/individuals/{individual_id}",
    response_model=MessageResponse,
    response_description="Remove a single enrolled individual from a model's gallery.",
)
def delete_individual(
    individual_id: str,
    model: Optional[str] = None,
    service: ReidService = Depends(get_service),
) -> MessageResponse:
    """
    Delete one enrolled individual from the selected model's gallery.

    Returns 404 when the id was never enrolled, so the client can tell "removed"
    from "nothing to remove" rather than silently succeeding.
    """
    removed = service.delete_individual(model=model, individual_id=individual_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No enrolled individual with id '{individual_id}'.",
        )
    return MessageResponse(message=f"Removed individual '{individual_id}'.")


@router.post(
    "/gallery/reset",
    response_model=MessageResponse,
    response_description="Clear all enrolled individuals from a model's gallery.",
)
def reset_gallery(
    model: Optional[str] = None,
    service: ReidService = Depends(get_service),
) -> MessageResponse:
    """Empty the selected model's gallery entirely and persist the cleared state."""
    service.reset_gallery(model=model)
    return MessageResponse(message="Gallery reset; all individuals removed.")
