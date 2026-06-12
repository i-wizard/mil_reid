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
    """
    Global liveness snapshot — lets a client (or Docker healthcheck) confirm the app is up.

    Per-model readiness lives in /models now (since the app serves several models);
    this stays deliberately model-agnostic.
    """

    status: str = Field(description="'ok' once the service is up.")
    models_available: int = Field(description="How many trained models were discovered on disk.")
    default_model: Optional[str] = Field(description="The model used when a request omits one (None if none trained).")


class ModelInfo(BaseModel):
    """One trained model's identity + readiness, for the model picker."""

    name: str = Field(description="Logical model name (its artifacts/models/<name> dir).")
    ready: bool = Field(description="True when the model's checkpoint is present and loadable.")
    dataset: Optional[str] = Field(default=None, description="Dataset the model was trained on (from its checkpoint).")
    backbone: Optional[str] = Field(default=None, description="Backbone the model was trained with.")
    num_individuals: int = Field(description="Individuals currently enrolled in this model's gallery.")


class ModelsResponse(BaseModel):
    """The catalogue of available models + which one is the default."""

    models: List[ModelInfo] = Field(description="All discovered models.")
    default_model: Optional[str] = Field(description="Name used when a request omits 'model'.")


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

    model: str = Field(description="The model this identification ran against.")
    is_unknown: bool = Field(description="True when the best match is too weak — query is treated as a new individual.")
    candidates: List[CandidateResponse] = Field(description="Top-k matches, best first (empty if the gallery is empty).")
    grid: int = Field(description="Side length of the square attention grid (patch_grid).")
    attention_grid: List[List[float]] = Field(description="grid×grid attention weights over patches (sum to 1).")


class EnrollResponse(BaseModel):
    """Acknowledges an enrollment and reports the resulting gallery size."""

    model: str = Field(description="The model whose gallery was updated.")
    individual_id: str = Field(description="The individual that was enrolled (or re-enrolled).")
    images_enrolled: int = Field(description="How many reference images were averaged into the prototype.")
    total_individuals: int = Field(description="Gallery size after this enrollment.")


class IndividualsResponse(BaseModel):
    """The current roster of enrolled individuals."""

    individuals: List[str] = Field(description="Enrolled individual ids.")
    count: int = Field(description="Number of enrolled individuals.")


class MessageResponse(BaseModel):
    """Generic human-readable acknowledgement for mutations without richer output."""

    message: str = Field(description="What happened, in plain language.")
    detail: Optional[str] = Field(default=None, description="Optional extra context.")


# --- Settings view: datasets + background jobs ---------------------------------


class DatasetInfo(BaseModel):
    """One curated dataset with its local status (for the Settings list)."""

    name: str = Field(description="WildlifeDatasets class name.")
    downloaded: bool = Field(description="True when the dataset's files are present locally.")
    precomputed: bool = Field(description="True when its feature cache is built for the current settings (trainable).")


class DatasetsResponse(BaseModel):
    """The curated catalogue offered in the Settings view."""

    datasets: List[DatasetInfo] = Field(description="Curated, demo-friendly datasets and their download status.")


class TrainRequest(BaseModel):
    """Body for starting a training job — minimal: which model name, which dataset."""

    model_config = {"protected_namespaces": ()}  # allow the field name `model_name`

    model_name: str = Field(description="Name to save the trained model under (its own namespace).")
    dataset: str = Field(description="A downloaded dataset to train on (must be in the curated catalogue).")


class JobAccepted(BaseModel):
    """202 response when a background job is started; the client then opens the SSE stream."""

    job_id: str = Field(description="Id to poll (/jobs/{id}) or stream (/jobs/{id}/events).")
    kind: str = Field(description="The kind of job started (download/precompute/train/seed).")


class JobStatusResponse(BaseModel):
    """Point-in-time job state — the SSE event payload and the /jobs/{id} poll shape."""

    id: str = Field(description="Job id.")
    kind: str = Field(description="download | precompute | train | seed.")
    status: str = Field(description="running | succeeded | failed | cancelled.")
    progress: Optional[float] = Field(default=None, description="0..1 completion when known.")
    message: Optional[str] = Field(default=None, description="Latest human-readable progress note.")
    result: Optional[dict] = Field(default=None, description="Result payload on success.")
    error: Optional[str] = Field(default=None, description="Error message on failure.")
