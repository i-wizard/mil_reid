"""
Dependency-injection factories for the API.

Centralises how routes obtain the shared ``ReidService`` and settings, per the
project's FastAPI conventions (DI via ``Depends`` rather than globals). The
service is built once in the app lifespan and stashed on ``app.state``; this
indirection is also the seam the tests use — ``app.dependency_overrides`` swaps
in a stub service so routes can be tested without loading torch.
"""

from fastapi import Depends, Request

from api.config import ApiSettings
from api.service import ReidService


def get_settings(request: Request) -> ApiSettings:
    """Return the ApiSettings built at startup (stored on app.state)."""
    return request.app.state.api_settings


def get_service(request: Request) -> ReidService:
    """
    Return the singleton ``ReidService`` created during app startup.

    Reads it off ``app.state`` rather than constructing per-request, because the
    service owns the expensive, load-once backbone + head.
    """
    return request.app.state.service


# Re-exported as the canonical injection points so routers import one name each.
ServiceDep = Depends(get_service)
SettingsDep = Depends(get_settings)
