"""
API-layer tests using a stub service — fast, no torch, no dataset.

The routes are exercised through FastAPI's ``TestClient`` with the real
``get_service`` dependency overridden by a lightweight **multi-model** fake. This
isolates the HTTP layer (routing, validation, status codes, response schemas,
error mapping, model selection) from the heavy ML core, so the whole suite runs in
milliseconds and asserts the contract the browser client depends on.

Run: python -m pytest tests/test_api.py   (or python -m tests.test_api for a quick run)
"""

import io
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi.testclient import TestClient
from PIL import Image

from api.config import ApiSettings
from api.dependencies import get_jobs, get_service, get_settings
from api.jobs import Job, JobBusyError, JobEvent, JobKind, JobManager, JobNotFoundError, JobStatus
from api.main import app
from api.service import ModelNotFoundError, ModelNotReadyError

# Endpoint paths declared once (per the project's test conventions).
HEALTH_URL = "/health"
MODELS_URL = "/models"
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


@dataclass
class _StubModel:
    """One fake model: readiness, provenance, and its isolated set of enrolled ids."""

    ready: bool = True
    dataset: str = "StubDataset"
    backbone: str = "stub-backbone"
    individuals: List[str] = field(default_factory=list)


class _StubRegistry:
    """Tiny registry exposing names(), matching what the health/models routers call."""

    def __init__(self, models: Dict[str, _StubModel]):
        self._models = models

    def names(self) -> List[str]:
        return sorted(self._models.keys())


class StubService:
    """
    In-memory multi-model fake of ReidService — deterministic, torch-free.

    Holds several named models, each with an isolated gallery, so tests can assert
    model routing + isolation, unknown-model 404s, not-ready 503s, and defaulting.
    """

    def __init__(self, models: Optional[Dict[str, _StubModel]] = None):
        self._models: Dict[str, _StubModel] = models if models is not None else {"default": _StubModel()}
        self.registry = _StubRegistry(self._models)

    # --- resolution mirrors the real service ---
    def default_model(self) -> str:
        names = self.registry.names()
        return names[0] if names else "default"

    def _resolve(self, model: Optional[str]) -> _StubModel:
        name = model or self.default_model()
        if name not in self._models:
            raise ModelNotFoundError(f"No model named '{name}'.")
        m = self._models[name]
        if not m.ready:
            raise ModelNotReadyError(f"Model '{name}' failed to load.")
        return m

    def list_models(self) -> List[dict]:
        return [
            {
                "name": name,
                "ready": m.ready,
                "dataset": m.dataset,
                "backbone": m.backbone,
                "num_individuals": len(m.individuals),
            }
            for name, m in sorted(self._models.items())
        ]

    # --- reads ---
    def identify(self, model: Optional[str], image_path: str, bbox=None) -> _IdentifyResult:
        self._resolve(model)
        return _IdentifyResult(candidates=[_Cand("ind_a", 0.91), _Cand("ind_b", 0.42)], is_unknown=False)

    def attention_png(self, model: Optional[str], image_path: str, bbox=None) -> bytes:
        self._resolve(model)
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
        return buffer.getvalue()

    def list_individuals(self, model: Optional[str]) -> List[str]:
        return list(self._resolve(model).individuals)

    # --- mutations ---
    def enroll(self, model: Optional[str], individual_id: str, image_paths: List[str]) -> int:
        if not individual_id.strip():
            raise ValueError("individual_id must be non-empty.")
        m = self._resolve(model)
        if individual_id not in m.individuals:
            m.individuals.append(individual_id)
        return len(image_paths)

    def delete_individual(self, model: Optional[str], individual_id: str) -> bool:
        m = self._resolve(model)
        if individual_id in m.individuals:
            m.individuals.remove(individual_id)
            return True
        return False

    def reset_gallery(self, model: Optional[str]) -> None:
        self._resolve(model).individuals.clear()

    def seed_from_dataset(self, model: Optional[str]) -> int:
        m = self._resolve(model)
        m.individuals = ["seed_1", "seed_2", "seed_3"]
        return len(m.individuals)


class FakeJobs:
    """
    Stand-in JobManager for HTTP tests — records starts without running any ML.

    ``start`` returns a real ``Job`` (so the routes/SSE see the true shape) but never
    executes the body, so dataset/train/seed routes can be tested without torch. A
    ``busy`` flag drives the single-flight 409 path; jobs can be pre-seeded for the
    status/cancel/SSE routes.
    """

    def __init__(self):
        self.busy = False
        self.jobs: Dict[str, Job] = {}
        self.started: List[tuple] = []

    def start(self, kind: JobKind, params: dict, body) -> Job:
        if self.busy:
            raise JobBusyError("a job is already running")
        job = Job(id=f"job{len(self.jobs) + 1}", kind=kind, params=params)
        job.emit(JobEvent(status=JobStatus.RUNNING.value, message="started", progress=0.0))
        self.jobs[job.id] = job
        self.started.append((kind, params))
        return job

    def active(self) -> Optional[Job]:
        for job in self.jobs.values():
            if not job.is_terminal:
                return job
        return None

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise JobNotFoundError(job_id)
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.status = JobStatus.CANCELLED
        return job


def _client(service: StubService, jobs: Optional[FakeJobs] = None) -> TestClient:
    """
    Build a TestClient with the service, settings, and jobs dependencies overridden.

    ``get_settings`` is overridden because TestClient does not run the app lifespan
    (which normally populates ``app.state``); ``get_jobs`` is overridden with a
    ``FakeJobs`` so background-job routes never execute real ML. The fake is attached
    as ``client.fake_jobs`` for tests that drive the single-flight / status paths.
    """
    fake = jobs if jobs is not None else FakeJobs()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: ApiSettings()
    app.dependency_overrides[get_jobs] = lambda: fake
    client = TestClient(app)
    client.fake_jobs = fake
    return client


def _png_bytes() -> bytes:
    """A tiny valid PNG used as a fake image upload."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 90, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _two_model_service() -> StubService:
    """A service with two ready models for routing/isolation tests."""
    return StubService({"turtles": _StubModel(dataset="SeaTurtleIDHeads"), "pandas": _StubModel(dataset="IPanda50")})


def test_health_reports_models_available() -> None:
    """/health is global: it reports how many models exist and the default."""
    client = _client(_two_model_service())
    res = client.get(HEALTH_URL)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["models_available"] == 2
    assert body["default_model"] == "pandas"  # first alphabetically of {pandas, turtles}


def test_models_lists_all_with_metadata() -> None:
    """/models returns each model's readiness + provenance for the picker."""
    client = _client(_two_model_service())
    body = client.get(MODELS_URL).json()
    names = {m["name"] for m in body["models"]}
    assert names == {"turtles", "pandas"}
    assert {m["dataset"] for m in body["models"]} == {"SeaTurtleIDHeads", "IPanda50"}
    assert body["default_model"] in names


def test_identify_returns_ranked_candidates_and_echoes_model() -> None:
    """/identify maps the result into IdentifyResponse and echoes the chosen model."""
    client = _client(_two_model_service())
    res = client.post(
        IDENTIFY_URL, data={"model": "turtles"}, files={"file": ("q.png", _png_bytes(), "image/png")}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "turtles"
    assert body["is_unknown"] is False
    assert [c["individual_id"] for c in body["candidates"]] == ["ind_a", "ind_b"]
    assert body["grid"] == 2 and len(body["attention_grid"]) == 2


def test_unknown_model_is_404() -> None:
    """Selecting a model that doesn't exist returns 404, not 503/500."""
    client = _client(_two_model_service())
    res = client.post(
        IDENTIFY_URL, data={"model": "giraffes"}, files={"file": ("q.png", _png_bytes(), "image/png")}
    )
    assert res.status_code == 404


def test_not_ready_model_is_503() -> None:
    """A known-but-unloadable model returns 503."""
    client = _client(StubService({"turtles": _StubModel(ready=False)}))
    res = client.post(
        IDENTIFY_URL, data={"model": "turtles"}, files={"file": ("q.png", _png_bytes(), "image/png")}
    )
    assert res.status_code == 503


def test_explain_returns_png() -> None:
    """/explain returns a PNG body with the right content type."""
    client = _client(_two_model_service())
    res = client.post(EXPLAIN_URL, data={"model": "pandas"}, files={"file": ("q.png", _png_bytes(), "image/png")})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_enroll_is_isolated_per_model() -> None:
    """Enrolling under one model must not leak into another model's gallery."""
    service = _two_model_service()
    client = _client(service)

    res = client.post(
        ENROLL_URL,
        data={"individual_id": "turtle_x", "model": "turtles"},
        files=[("files", ("a.png", _png_bytes(), "image/png")), ("files", ("b.png", _png_bytes(), "image/png"))],
    )
    assert res.status_code == 200
    assert res.json() == {"model": "turtles", "individual_id": "turtle_x", "images_enrolled": 2, "total_individuals": 1}

    # Present under 'turtles', absent under 'pandas' → isolation holds.
    assert client.get(f"{INDIVIDUALS_URL}?model=turtles").json() == {"individuals": ["turtle_x"], "count": 1}
    assert client.get(f"{INDIVIDUALS_URL}?model=pandas").json() == {"individuals": [], "count": 0}


def test_default_model_used_when_omitted() -> None:
    """Omitting 'model' targets the default model (back-compat with single-model clients)."""
    service = StubService({"default": _StubModel()})
    client = _client(service)
    client.post(ENROLL_URL, data={"individual_id": "x"}, files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert client.get(INDIVIDUALS_URL).json() == {"individuals": ["x"], "count": 1}


def test_delete_and_reset_per_model() -> None:
    """Delete returns 404 when absent; reset clears only the targeted model."""
    service = _two_model_service()
    client = _client(service)
    client.post(ENROLL_URL, data={"individual_id": "p1", "model": "pandas"}, files=[("files", ("a.png", _png_bytes(), "image/png"))])

    assert client.delete(f"{INDIVIDUALS_URL}/p1?model=pandas").status_code == 200
    assert client.delete(f"{INDIVIDUALS_URL}/p1?model=pandas").status_code == 404  # already gone

    client.post(ENROLL_URL, data={"individual_id": "p2", "model": "pandas"}, files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert client.post(RESET_URL, data={"model": "pandas"}).status_code == 200
    assert client.get(f"{INDIVIDUALS_URL}?model=pandas").json()["count"] == 0


def test_seed_starts_a_background_job() -> None:
    """/gallery/seed now returns 202 + a job id (the work runs in the background)."""
    client = _client(_two_model_service())
    res = client.post(SEED_URL, data={"model": "turtles"})
    assert res.status_code == 202
    body = res.json()
    assert body["kind"] == "seed" and body["job_id"]


def test_enroll_rejects_non_image_upload() -> None:
    """A non-image upload is refused with 415 before touching the model."""
    client = _client(_two_model_service())
    res = client.post(
        ENROLL_URL,
        data={"individual_id": "x", "model": "turtles"},
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert res.status_code == 415


# --- Settings: datasets + background jobs --------------------------------------

DATASETS_URL = "/datasets"
TRAIN_URL = "/train"


def test_datasets_lists_curated_with_status_flags() -> None:
    """/datasets returns the curated catalogue, each with downloaded + precomputed bools."""
    from ml.data.dataset import CURATED_DATASETS

    client = _client(_two_model_service())
    body = client.get(DATASETS_URL).json()
    names = [d["name"] for d in body["datasets"]]
    assert names == CURATED_DATASETS
    assert all(isinstance(d["downloaded"], bool) for d in body["datasets"])
    assert all(isinstance(d["precomputed"], bool) for d in body["datasets"])


def test_precompute_marker_signature() -> None:
    """A marker counts as precomputed only while the cache signature matches."""
    import tempfile

    from ml.config import get_settings
    from ml.data.dataset import is_precomputed, mark_precomputed

    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings({"cache_root": tmp})
        assert is_precomputed("DemoSet", settings) is False
        mark_precomputed("DemoSet", settings, count=5)
        assert is_precomputed("DemoSet", settings) is True
        # Changing patch geometry invalidates the marker (stale cache).
        changed = get_settings({"cache_root": tmp, "patch_grid": settings.patch_grid + 1})
        assert is_precomputed("DemoSet", changed) is False


def test_download_starts_job_and_rejects_unknown_dataset() -> None:
    """Download returns 202 for a curated name and 404 for an unknown one."""
    client = _client(_two_model_service())
    ok = client.post(f"{DATASETS_URL}/SeaTurtleIDHeads/download")
    assert ok.status_code == 202 and ok.json()["kind"] == "download"
    assert client.post(f"{DATASETS_URL}/NotARealDataset/download").status_code == 404


def test_single_flight_rejects_second_job_with_409() -> None:
    """While a job is running, starting another is rejected up front with 409."""
    client = _client(_two_model_service())
    client.fake_jobs.busy = True  # simulate a job already in flight
    res = client.post(f"{DATASETS_URL}/SeaTurtleIDHeads/download")
    assert res.status_code == 409


def test_train_validates_dataset() -> None:
    """/train gates on curated → downloaded → precomputed before starting a job."""
    import api.routers.training as training_router

    client = _client(_two_model_service())
    # Non-curated → 404 (checked before any download/precompute probe).
    assert client.post(TRAIN_URL, json={"model_name": "m", "dataset": "NotReal"}).status_code == 404

    # Curated but not downloaded → 422.
    training_router.is_downloaded = lambda name, settings: False
    training_router.is_precomputed = lambda name, settings: False
    assert client.post(TRAIN_URL, json={"model_name": "m", "dataset": "SeaTurtleIDHeads"}).status_code == 422

    # Downloaded but NOT precomputed → 422 (the new guard).
    training_router.is_downloaded = lambda name, settings: True
    training_router.is_precomputed = lambda name, settings: False
    assert client.post(TRAIN_URL, json={"model_name": "m", "dataset": "SeaTurtleIDHeads"}).status_code == 422

    # Downloaded + precomputed → 202 job.
    training_router.is_precomputed = lambda name, settings: True
    r202 = client.post(TRAIN_URL, json={"model_name": "m", "dataset": "SeaTurtleIDHeads"})
    assert r202.status_code == 202 and r202.json()["kind"] == "train"


def test_jobs_active_status_and_cancel() -> None:
    """/jobs/active reflects a running job; cancel flips it; cancel-after-terminal is 409."""
    client = _client(_two_model_service())
    started = client.post(f"{DATASETS_URL}/SeaTurtleIDHeads/download").json()
    job_id = started["job_id"]

    active = client.get("/jobs/active").json()
    assert active and active["id"] == job_id and active["status"] == "running"

    cancelled = client.post(f"/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    # Now terminal → cancelling again is a 409 no-op.
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 409
    # Unknown job → 404.
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_job_events_stream_is_sse() -> None:
    """/jobs/{id}/events streams text/event-stream and closes on a terminal event."""
    client = _client(_two_model_service())
    job_id = client.post(f"{DATASETS_URL}/SeaTurtleIDHeads/download").json()["job_id"]
    # Drive the fake job to a terminal state so the stream ends promptly.
    job = client.fake_jobs.get(job_id)
    job.status = JobStatus.SUCCEEDED
    job.emit(JobEvent(status="succeeded", message="done", progress=1.0))

    res = client.get(f"/jobs/{job_id}/events")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert "succeeded" in res.text


# --- JobManager unit behaviour (real manager, trivial bodies, no ML) -----------


def test_jobmanager_single_flight_and_completion() -> None:
    """Real JobManager runs one job, rejects a concurrent start, then frees the slot."""
    manager = JobManager()
    release = threading.Event()
    entered = threading.Event()

    def slow_body(job):
        entered.set()
        release.wait(timeout=2)
        return {"ok": True}

    job = manager.start(JobKind.SEED, {}, slow_body)
    assert entered.wait(timeout=2)
    try:
        manager.start(JobKind.SEED, {}, slow_body)
        assert False, "expected JobBusyError while a job is running"
    except JobBusyError:
        pass

    release.set()
    for _ in range(200):
        if job.is_terminal:
            break
        time.sleep(0.01)
    assert job.status == JobStatus.SUCCEEDED and job.result == {"ok": True}
    # Slot is free again.
    assert manager.active() is None


def test_jobmanager_cooperative_cancel() -> None:
    """A job that checks raise_if_cancelled aborts to 'cancelled' when cancel() is set."""
    manager = JobManager()
    entered = threading.Event()

    def cancellable_body(job):
        for _ in range(1000):
            job.raise_if_cancelled()
            entered.set()
            time.sleep(0.01)
        return {}

    job = manager.start(JobKind.PRECOMPUTE, {}, cancellable_body)
    assert entered.wait(timeout=2)
    manager.cancel(job.id)
    for _ in range(300):
        if job.is_terminal:
            break
        time.sleep(0.01)
    assert job.status == JobStatus.CANCELLED


def main() -> None:
    """Run all tests directly (mirrors the core-logic suite's runnable style)."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    app.dependency_overrides.clear()
    print(f"\nALL {len(tests)} API TESTS PASSED")


if __name__ == "__main__":
    main()
