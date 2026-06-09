"""
FastAPI application entry point.

Wires the whole HTTP layer together: builds the shared ``ReidService`` once in a
lifespan handler (so the costly backbone load happens a single time at startup),
enables CORS for the future browser client, and mounts the routers. Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import os

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config import ApiSettings
from api.jobs import JobBusyError, JobManager, JobNotFoundError
from api.routers import datasets, enroll, gallery, health, identify, jobs, models, training
from api.service import ModelNotFoundError, ModelNotReadyError, ReidService
from ml.config import get_settings as get_model_settings
from ml.utils.logging import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build shared state at startup and expose it on ``app.state``.

    The ReidService is created here once (its model registry scan is cheap; the
    heavy per-model backbone load happens lazily on first request). The server
    starts even with zero trained models — readiness is reported per-model via
    /models.
    """
    api_settings = ApiSettings()
    model_settings = get_model_settings()

    app.state.api_settings = api_settings
    app.state.service = ReidService.build(settings=model_settings)
    app.state.jobs = JobManager()
    logger.info(f"API ready. Models discovered: {app.state.service.registry.names() or '(none)'}.")
    yield
    # Nothing to tear down: torch holds no external handles and the gallery is
    # persisted on every mutation, so there is no shutdown work to do.


app = FastAPI(
    title="Animal Re-Identification API",
    description="HTTP layer over the patch-bag MIL re-identification ML core.",
    version="1.0.0",
    lifespan=lifespan,
)

_cors = ApiSettings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ModelNotReadyError)
async def _model_not_ready_handler(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    """
    Map a missing-trained-model error to 503 Service Unavailable.

    Registered once here so every route that needs the model gets the same
    actionable response without each handler repeating a try/except.
    """
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})


@app.exception_handler(ModelNotFoundError)
async def _model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """Map a request for an unknown model name to 404 Not Found (distinct from 503 load failures)."""
    # KeyError's str() wraps the message in quotes; strip them for a clean detail.
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc).strip("'\"")})


@app.exception_handler(JobBusyError)
async def _job_busy_handler(request: Request, exc: JobBusyError) -> JSONResponse:
    """Map a start-while-busy to 409 Conflict — single-flight rejection, not a queue."""
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})


@app.exception_handler(JobNotFoundError)
async def _job_not_found_handler(request: Request, exc: JobNotFoundError) -> JSONResponse:
    """Map an unknown job id to 404 Not Found."""
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc).strip("'\"")})


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Map service-layer validation errors (e.g. empty individual_id) to 400 Bad Request."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


app.include_router(health.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(jobs.router)
app.include_router(enroll.router)
app.include_router(identify.router)
app.include_router(gallery.router)

# Serve the browser client from the same origin. Mounted LAST and at "/"
# so the greedy static mount only handles paths the API routers (and /docs) did
# not claim — same origin means the UI's fetch() calls need no CORS. Guarded so
# the API still imports/runs in environments without the web/ dir (e.g. tests).
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:
    logger.warning(f"web/ dir not found at {_WEB_DIR}; UI not served (API endpoints still active).")
