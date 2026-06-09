"""
API-layer tests using a stub service — fast, no torch, no dataset.

The routes are exercised through FastAPI's ``TestClient`` with the real
``get_service`` dependency overridden by a lightweight fake. This isolates the
HTTP layer (routing, validation, status codes, response schemas, error mapping)
from the heavy ML core, so the whole suite runs in milliseconds and asserts the
contract the browser client depends on.

Run: python -m pytest tests/test_api.py   (or python -m tests.test_api for a quick run)
"""

import io
from dataclasses import dataclass
from typing import List, Optional

from fastapi.testclient import TestClient
from PIL import Image

from api.dependencies import get_service
from api.main import app
from api.service import ModelNotReadyError

# Endpoint paths declared once (per the project's test conventions).
HEALTH_URL = "/health"
ENROLL_URL = "/enroll"
IDENTIFY_URL = "/identify"
EXPLAIN_URL = "/explain"
INDIVIDUALS_URL = "/individuals"
RESET_URL = "/gallery/reset"
SEED_URL = "/gallery/seed"


@dataclass
class _Cand:
    """Mimics ml.inference.identifier.Candidate for the stub's return value."""

    individual_id: str
    score: float


class _EmbedResult:
    """Minimal stand-in for EmbedResult — only the fields the identify route reads."""

    def __init__(self, grid: int = 2):
        import numpy as np

        self.grid = grid
        self.attention = np.full(grid * grid, 1.0 / (grid * grid), dtype="float32")


class _IdentifyResult:
    """Stand-in for IdentifyResult carrying candidates, verdict, and embed_result."""

    def __init__(self, candidates: List[_Cand], is_unknown: bool):
        self.candidates = candidates
        self.is_unknown = is_unknown
        self.embed_result = _EmbedResult()


class StubService:
    """
    In-memory fake of ReidService — deterministic, torch-free.

    Lets tests drive every route and toggle readiness (``ready``) to assert the
    503 gating, without loading a backbone or touching disk.
    """

    def __init__(self, ready: bool = True):
        self._ready = ready
        self._individuals: List[str] = []

        class _S:
            backbone = type("B", (), {"value": "stub-backbone"})()
            dataset = type("D", (), {"value": "StubDataset"})()

        self.settings = _S()

    def is_ready(self) -> bool:
        return self._ready

    def _require_ready(self) -> None:
        if not self._ready:
            raise ModelNotReadyError("No trained model loaded.")

    def enroll(self, individual_id: str, image_paths: List[str]) -> int:
        self._require_ready()
        if not individual_id.strip():
            raise ValueError("individual_id must be non-empty.")
        if individual_id not in self._individuals:
            self._individuals.append(individual_id)
        return len(image_paths)

    def identify(self, image_path: str, bbox: Optional[List[float]] = None) -> _IdentifyResult:
        self._require_ready()
        return _IdentifyResult(candidates=[_Cand("turtle_a", 0.91), _Cand("turtle_b", 0.42)], is_unknown=False)

    def attention_png(self, image_path: str, bbox: Optional[List[float]] = None) -> bytes:
        self._require_ready()
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
        return buffer.getvalue()

    def list_individuals(self) -> List[str]:
        return list(self._individuals)

    def delete_individual(self, individual_id: str) -> bool:
        if individual_id in self._individuals:
            self._individuals.remove(individual_id)
            return True
        return False

    def reset_gallery(self) -> None:
        self._individuals.clear()

    def seed_from_dataset(self) -> int:
        self._require_ready()
        self._individuals = ["seed_1", "seed_2", "seed_3"]
        return len(self._individuals)


def _client(service: StubService) -> TestClient:
    """Build a TestClient with the service dependency overridden by the stub."""
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def _png_bytes() -> bytes:
    """A tiny valid PNG used as a fake image upload."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 90, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_readiness_and_schema() -> None:
    """/health returns the documented fields and reflects the service readiness."""
    client = _client(StubService(ready=True))
    res = client.get(HEALTH_URL)
    assert res.status_code == 200
    body = res.json()
    assert body == {
        "status": "ok",
        "model_ready": True,
        "backbone": "stub-backbone",
        "dataset": "StubDataset",
        "num_individuals": 0,
    }


def test_identify_returns_ranked_candidates_and_verdict() -> None:
    """/identify maps the service result into the IdentifyResponse schema."""
    client = _client(StubService(ready=True))
    res = client.post(IDENTIFY_URL, files={"file": ("q.png", _png_bytes(), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["is_unknown"] is False
    assert [c["individual_id"] for c in body["candidates"]] == ["turtle_a", "turtle_b"]
    assert body["grid"] == 2
    assert len(body["attention_grid"]) == 2 and len(body["attention_grid"][0]) == 2


def test_identify_503_when_model_not_ready() -> None:
    """Inference endpoints surface ModelNotReadyError as 503, not a 500."""
    client = _client(StubService(ready=False))
    res = client.post(IDENTIFY_URL, files={"file": ("q.png", _png_bytes(), "image/png")})
    assert res.status_code == 503
    assert "detail" in res.json()


def test_explain_returns_png() -> None:
    """/explain returns a PNG body with the right content type."""
    client = _client(StubService(ready=True))
    res = client.post(EXPLAIN_URL, files={"file": ("q.png", _png_bytes(), "image/png")})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_enroll_then_list_then_delete_then_reset() -> None:
    """Full gallery lifecycle: enroll grows it, delete shrinks it, reset clears it."""
    service = StubService(ready=True)
    client = _client(service)

    # Enroll
    res = client.post(
        ENROLL_URL,
        data={"individual_id": "turtle_x"},
        files=[("files", ("a.png", _png_bytes(), "image/png")), ("files", ("b.png", _png_bytes(), "image/png"))],
    )
    assert res.status_code == 200
    assert res.json() == {"individual_id": "turtle_x", "images_enrolled": 2, "total_individuals": 1}

    # List
    assert client.get(INDIVIDUALS_URL).json() == {"individuals": ["turtle_x"], "count": 1}

    # Delete present vs absent
    assert client.delete(f"{INDIVIDUALS_URL}/turtle_x").status_code == 200
    assert client.delete(f"{INDIVIDUALS_URL}/turtle_x").status_code == 404  # already gone

    # Reset
    client.post(ENROLL_URL, data={"individual_id": "turtle_y"}, files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert client.post(RESET_URL).status_code == 200
    assert client.get(INDIVIDUALS_URL).json()["count"] == 0


def test_enroll_rejects_non_image_upload() -> None:
    """A non-image upload is refused with 415 before touching the model."""
    client = _client(StubService(ready=True))
    res = client.post(
        ENROLL_URL,
        data={"individual_id": "turtle_x"},
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert res.status_code == 415


def test_seed_enrolls_from_dataset() -> None:
    """/gallery/seed returns the seeded count and populates the gallery."""
    service = StubService(ready=True)
    client = _client(service)
    res = client.post(SEED_URL)
    assert res.status_code == 200
    assert res.json() == {"individuals_enrolled": 3}
    assert client.get(INDIVIDUALS_URL).json()["count"] == 3


def main() -> None:
    """Run all tests directly (mirrors the core-logic suite's runnable style)."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    # Clean up overrides so a direct run doesn't leak state.
    app.dependency_overrides.clear()
    print(f"\nALL {len(tests)} API TESTS PASSED")


if __name__ == "__main__":
    main()
