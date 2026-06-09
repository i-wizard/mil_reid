"""
Identification routes: open-set match (JSON) and the attention overlay (PNG).

Split into two endpoints by output type. ``/identify`` returns the structured
verdict + ranked candidates a client renders as UI; ``/explain`` returns a ready
-to-display heatmap image. Keeping them separate means each has one clear content
type and the client fetches the overlay only when it wants to show it.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from api.config import ApiSettings
from api.dependencies import get_service, get_settings
from api.schemas import CandidateResponse, IdentifyResponse
from api.service import ReidService
from api.uploads import saved_uploads


def _parse_bbox(bbox: Optional[str]) -> Optional[List[float]]:
    """
    Parse an optional ``x,y,w,h`` bbox form field into floats.

    Accepts a comma-separated string (the natural multipart form encoding) and
    returns None when absent, so the model tiles the whole image. Malformed input
    raises ValueError → mapped to 400 by the app handler.
    """
    if not bbox:
        return None
    parts = [p.strip() for p in bbox.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x,y,w,h' (four comma-separated numbers).")
    return [float(p) for p in parts]


router = APIRouter(tags=["identify"])


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    response_description="Open-set identification: ranked candidates and the unknown verdict.",
)
def identify(
    file: UploadFile = File(..., description="Query image to identify."),
    bbox: Optional[str] = Form(default=None, description="Optional 'x,y,w,h' crop; omit to use the whole image."),
    service: ReidService = Depends(get_service),
    settings: ApiSettings = Depends(get_settings),
) -> IdentifyResponse:
    """
    Identify the individual in the uploaded image against the current gallery.

    Returns the top-k matches (best first) and ``is_unknown=true`` when the best
    score is below the configured threshold. The attention grid is included so a
    client can visualise it; use /explain for a rendered overlay. 503 if no model.
    """
    parsed_bbox = _parse_bbox(bbox)
    with saved_uploads(uploads=[file], max_bytes=settings.max_upload_bytes) as paths:
        result = service.identify(image_path=paths[0], bbox=parsed_bbox)

    grid = result.embed_result.grid
    attention_grid = result.embed_result.attention.reshape(grid, grid).tolist()
    return IdentifyResponse(
        is_unknown=result.is_unknown,
        candidates=[CandidateResponse(individual_id=c.individual_id, score=c.score) for c in result.candidates],
        grid=grid,
        attention_grid=attention_grid,
    )


@router.post(
    "/explain",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Attention heatmap overlay (PNG)."}},
    response_description="PNG overlay of the model's per-patch attention on the query image.",
)
def explain(
    file: UploadFile = File(..., description="Query image to visualise attention for."),
    bbox: Optional[str] = Form(default=None, description="Optional 'x,y,w,h' crop; omit to use the whole image."),
    service: ReidService = Depends(get_service),
    settings: ApiSettings = Depends(get_settings),
) -> Response:
    """
    Render the attention heatmap overlay for the uploaded image as a PNG.

    This is the explainability visual — it shows which patches the model weighted
    when forming the identity embedding. Binary output, so it returns a raw
    ``Response`` (the documented exception to the JSON response_model rule). 503 if
    no model is loaded.
    """
    parsed_bbox = _parse_bbox(bbox)
    with saved_uploads(uploads=[file], max_bytes=settings.max_upload_bytes) as paths:
        png_bytes = service.attention_png(image_path=paths[0], bbox=parsed_bbox)
    return Response(content=png_bytes, media_type="image/png")
