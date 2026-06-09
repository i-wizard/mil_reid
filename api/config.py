"""
API-specific configuration.

Kept separate from the ML core's ``ml.config.Settings`` (which owns model/dataset
knobs) so HTTP concerns — bind address, CORS, upload limits — have their own
clearly-scoped settings object. Both are env-overridable; this one uses the
``REID_API_`` prefix so the two namespaces never collide.
"""

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """
    HTTP-layer settings, overridable via ``REID_API_*`` environment variables.

    Defaults are demo-friendly (bind all interfaces, permissive CORS) so the
    container and the Part 3 browser client work out of the box; tighten
    ``cors_origins`` for any real deployment.
    """

    model_config = SettingsConfigDict(env_prefix="REID_API_", protected_namespaces=())

    host: str = Field(default="0.0.0.0", description="Interface uvicorn binds to (0.0.0.0 inside a container).")
    port: int = Field(default=8000, description="Port the API listens on.")
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins for the browser client. '*' is fine for the demo; restrict in production.",
    )
    max_upload_mb: float = Field(
        default=15.0,
        description="Reject uploads larger than this many megabytes — guards against accidental huge files.",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """
        Allow ``REID_API_CORS_ORIGINS`` to be a comma-separated string.

        Env vars are strings, so a list-typed setting is most convenient to
        provide as ``a.com,b.com``; we split it here so callers can still pass a
        real list in code.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def max_upload_bytes(self) -> int:
        """Upload cap in bytes — derived so the byte comparison lives in one place."""
        return int(self.max_upload_mb * 1024 * 1024)
