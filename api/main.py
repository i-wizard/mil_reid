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
from api.routers import enroll, gallery, health, identify
from api.service import ModelNotReadyError, ReidService
from ml.config import get_settings as get_model_settings
from ml.utils.logging import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build shared state at startup and expose it on ``app.state``.

    The model settings and the (load-once) ReidService are created here rather
    than per-request because constructing the frozen backbone is expensive. The
    service tolerates a missing trained head, coming up not-ready so the server
    still starts — readiness is then reported via /health.
    """
    api_settings = ApiSettings()
    model_settings = get_model_settings()

    app.state.api_settings = api_settings
    app.state.service = ReidService.build(settings=model_settings)
    logger.info(
        f"API ready (model_ready={app.state.service.is_ready()}, "
        f"enrolled={len(app.state.service.list_individuals())})."
    )
    yield
    # Nothing to tear down: torch holds no external handles and the gallery is
    # persisted on every mutation, so there is no shutdown work to do.


app = FastAPI(
    title="Animal Re-Identification API",
    description="HTTP layer over the patch-bag MIL re-identification ML core.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is configured from settings at import time using defaults; the demo allows
# all origins so the Part 3 browser client can call the API without extra setup.
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


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Map service-layer validation errors (e.g. empty individual_id) to 400 Bad Request."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


app.include_router(health.router)
app.include_router(enroll.router)
app.include_router(identify.router)
app.include_router(gallery.router)

# Serve the Part 3 browser client from the same origin. Mounted LAST and at "/"
# so the greedy static mount only handles paths the API routers (and /docs) did
# not claim — same origin means the UI's fetch() calls need no CORS. Guarded so
# the API still imports/runs in environments without the web/ dir (e.g. tests).
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:
    logger.warning(f"web/ dir not found at {_WEB_DIR}; UI not served (API endpoints still active).")
