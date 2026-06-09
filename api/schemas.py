"""
Pydantic request/response DTOs for the API.

Every JSON endpoint declares one of these as its ``response_model`` so the
contract is explicit, validated, and self-documenting in the auto-generated
OpenAPI schema. They are intentionally decoupled from the ML core's internal
dataclasses (``IdentifyResult``/``Candidate``) — the service layer maps between
them, so the HTTP contract can stay stable even if the internal types evolve.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness/readiness snapshot — lets a client (or Docker healthcheck) gate calls."""

    status: str = Field(description="'ok' once the service object exists.")
    model_ready: bool = Field(description="True when a trained head is loaded; inference 503s until then.")
    backbone: str = Field(description="The configured frozen backbone identifier.")
    dataset: str = Field(description="The configured dataset name.")
    num_individuals: int = Field(description="How many individuals are currently enrolled in the gallery.")


class CandidateResponse(BaseModel):
    """One ranked gallery match for an identify query."""

    individual_id: str = Field(description="The enrolled individual's id.")
    score: float = Field(description="Cosine similarity to the query embedding (higher = more similar).")


class IdentifyResponse(BaseModel):
    """
    Result of an open-set identification.

    ``is_unknown`` is the open-set verdict (top score below the threshold). The
    attention grid is included so a client can render its own visualisation
    without a second call; the dedicated /explain endpoint returns the rendered
    PNG when an overlay image is wanted instead.
    """

    is_unknown: bool = Field(description="True when the best match is too weak — query is treated as a new individual.")
    candidates: List[CandidateResponse] = Field(description="Top-k matches, best first (empty if the gallery is empty).")
    grid: int = Field(description="Side length of the square attention grid (patch_grid).")
    attention_grid: List[List[float]] = Field(description="grid×grid attention weights over patches (sum to 1).")


class EnrollResponse(BaseModel):
    """Acknowledges an enrollment and reports the resulting gallery size."""

    individual_id: str = Field(description="The individual that was enrolled (or re-enrolled).")
    images_enrolled: int = Field(description="How many reference images were averaged into the prototype.")
    total_individuals: int = Field(description="Gallery size after this enrollment.")


class IndividualsResponse(BaseModel):
    """The current roster of enrolled individuals."""

    individuals: List[str] = Field(description="Enrolled individual ids.")
    count: int = Field(description="Number of enrolled individuals.")


class SeedResponse(BaseModel):
    """Reports how many individuals were auto-enrolled from the dataset's gallery split."""

    individuals_enrolled: int = Field(description="Individuals enrolled from the trained dataset's gallery split.")


class MessageResponse(BaseModel):
    """Generic human-readable acknowledgement for mutations without richer output."""

    message: str = Field(description="What happened, in plain language.")
    detail: Optional[str] = Field(default=None, description="Optional extra context.")
