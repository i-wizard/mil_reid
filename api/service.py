"""
ReidService — the API's business-logic layer over the ML inference seam.

Routers stay thin by delegating everything here. This service is **multi-model**:
several trained models (one per species/domain) can be served from one running
app. It owns a :class:`ModelRegistry` (what exists on disk) plus a lazy cache of
loaded models (each an expensive backbone + head + its own gallery + write lock).
Models load on first use and stay resident; switching model is just a different
cache key. It is the only place the API touches ``ml.inference`` / ``ml.eval``.

Concurrency: FastAPI runs our sync handlers in a threadpool, so requests overlap.
Each loaded model carries its own lock guarding its gallery; loading itself is
guarded by a registry-level lock so two requests can't race to load the same model.
"""

import io
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from ml.config import Settings
from ml.inference.embedder import Embedder, load_embedder_for
from ml.inference.explain import render_attention_overlay
from ml.inference.gallery import Gallery
from ml.inference.identifier import Identifier, IdentifyResult
from ml.inference.registry import ModelInfo, ModelRegistry
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class ModelNotReadyError(RuntimeError):
    """
    Raised when a selected model exists but its head could not be loaded.

    Routers translate this into a 503 so a transient/load failure is distinct from
    asking for a model that simply isn't there (404).
    """


class ModelNotFoundError(KeyError):
    """Raised when a requested model name is not present on disk (mapped to 404)."""


@dataclass
class LoadedModel:
    """
    One model held in memory: its embedder, gallery, identifier, and write lock.

    Bundled so the cache stores everything a request needs for a given model and
    each model's gallery mutations are serialised independently of other models.
    """

    info: ModelInfo
    embedder: Embedder
    gallery: Gallery
    identifier: Identifier
    lock: threading.Lock


class ReidService:
    """
    Multi-model façade over the re-identification pipeline.

    Construct once at app startup via :meth:`build`. Discovery is cheap and done
    eagerly; the heavy per-model load happens lazily on first use.
    """

    def __init__(self, settings: Settings, registry: ModelRegistry):
        """Hold settings + registry and an empty load cache guarded by a registry lock."""
        self.settings = settings
        self.registry = registry
        self._loaded: Dict[str, LoadedModel] = {}
        self._registry_lock = threading.Lock()

    @classmethod
    def build(cls, settings: Settings) -> "ReidService":
        """
        Construct the service at startup. Never loads a model here.

        Only the (cheap) registry scan runs, so the API starts fast and even with
        zero trained models — readiness is reported per-model via /models.
        """
        registry = ModelRegistry(settings=settings)
        found = registry.names()
        logger.info(f"ReidService up. Discovered models: {found or '(none — train one first)'}.")
        return cls(settings=settings, registry=registry)

    # --- Model resolution / loading ---------------------------------------

    def default_model(self) -> str:
        """
        The model used when a request omits one.

        Prefers the configured ``model_name`` when it exists on disk, else the first
        discovered model, else the configured name (so error messages name it).
        """
        available = self.registry.names()
        if self.settings.model_name in available:
            return self.settings.model_name
        return available[0] if available else self.settings.model_name

    def list_models(self) -> List[dict]:
        """
        Describe every discovered model for /models (no heavy loading).

        ``ready`` reflects whether the checkpoint exists (loadable), and the
        enrolled count is read from whichever gallery is on disk / already loaded.
        """
        out: List[dict] = []
        for name, info in sorted(self.registry.discover().items()):
            out.append(
                {
                    "name": name,
                    "ready": info.checkpoint_path.exists(),
                    "dataset": info.dataset,
                    "backbone": info.backbone,
                    "num_individuals": self._gallery_count(info),
                }
            )
        return out

    def _gallery_count(self, info: ModelInfo) -> int:
        """Enrolled count for a model without forcing a full load (uses cache or disk)."""
        if info.name in self._loaded:
            return len(self._loaded[info.name].gallery.individuals)
        if info.gallery_path.exists():
            return len(Gallery.load(info.gallery_path).individuals)
        return 0

    def _get(self, model: Optional[str]) -> LoadedModel:
        """
        Resolve + lazily load a model by name (defaulting when omitted).

        Raises ModelNotFoundError for an unknown name and ModelNotReadyError if the
        head fails to load. The load is guarded so concurrent first-hits on the same
        model don't both pay the cost or race the cache.
        """
        name = model or self.default_model()
        if name in self._loaded:
            return self._loaded[name]

        with self._registry_lock:
            if name in self._loaded:  # another thread loaded it while we waited
                return self._loaded[name]

            info = self.registry.get(name)
            if info is None:
                raise ModelNotFoundError(
                    f"No model named '{name}'. Available: {self.registry.names() or 'none'}."
                )
            try:
                embedder = load_embedder_for(checkpoint_path=info.checkpoint_path, settings=self.settings)
            except Exception as error:
                raise ModelNotReadyError(f"Model '{name}' failed to load: {error}") from error

            gallery = Gallery.load(info.gallery_path) if info.gallery_path.exists() else Gallery()
            loaded = LoadedModel(
                info=info,
                embedder=embedder,
                gallery=gallery,
                identifier=Identifier(embedder=embedder, gallery=gallery, settings=self.settings),
                lock=threading.Lock(),
            )
            self._loaded[name] = loaded
            logger.info(f"Loaded model '{name}' (enrolled={len(gallery.individuals)}).")
            return loaded
        # NOTE: with many models this cache grows unbounded; an LRU eviction keyed by
        # last-use would go here if memory becomes a concern. For the demo's handful
        # of models, keeping them resident is the right trade-off.

    # --- Reads -------------------------------------------------------------

    def identify(self, model: Optional[str], image_path: str, bbox: Optional[List[float]] = None) -> IdentifyResult:
        """Identify the individual in an image against the selected model's gallery."""
        return self._get(model).identifier.identify(image_path=image_path, bbox=bbox)

    def attention_png(self, model: Optional[str], image_path: str, bbox: Optional[List[float]] = None) -> bytes:
        """Render the attention heatmap overlay (PNG bytes) using the selected model."""
        loaded = self._get(model)
        embed_result = loaded.embedder.embed_path(path=image_path, bbox=bbox)
        overlay = render_attention_overlay(
            image_path=image_path, embed_result=embed_result, settings=loaded.embedder.settings, bbox=bbox
        )
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        return buffer.getvalue()

    def list_individuals(self, model: Optional[str]) -> List[str]:
        """Return the enrolled individual ids for the selected model's gallery."""
        return self._get(model).gallery.individuals

    # --- Mutations (per-model lock + persisted) ----------------------------

    def enroll(self, model: Optional[str], individual_id: str, image_paths: List[str]) -> int:
        """
        Enroll/replace an individual in the selected model's gallery and persist.

        Returns the number of images averaged into the prototype. Uses the model's
        own lock so concurrent enrollments to the same model can't corrupt its
        gallery, while different models proceed independently.
        """
        if not individual_id.strip():
            raise ValueError("individual_id must be non-empty.")
        loaded = self._get(model)
        with loaded.lock:
            loaded.gallery.enroll(individual_id=individual_id, image_paths=image_paths, embedder=loaded.embedder)
            loaded.gallery.save(loaded.info.gallery_path)
        return len(image_paths)

    def delete_individual(self, model: Optional[str], individual_id: str) -> bool:
        """Remove one enrolled individual from the selected model and persist; False if absent."""
        loaded = self._get(model)
        with loaded.lock:
            if individual_id not in loaded.gallery._embeddings:
                return False
            del loaded.gallery._embeddings[individual_id]
            loaded.gallery.save(loaded.info.gallery_path)
        return True

    def reset_gallery(self, model: Optional[str]) -> None:
        """Clear all enrolled individuals for the selected model and persist the empty gallery."""
        loaded = self._get(model)
        with loaded.lock:
            loaded.gallery._embeddings.clear()
            loaded.gallery.save(loaded.info.gallery_path)

    def seed_from_dataset(self, model: Optional[str], progress_callback=None) -> int:
        """
        Auto-enroll the selected model's dataset gallery split so identify works at once.

        Reuses the evaluation/demo gallery construction so the seeded set matches the
        eval protocol. The dataset comes from the model's own training settings (its
        embedder), so seeding enrolls the correct species for that model.
        ``progress_callback(done, total)`` streams per-individual progress for the
        seed job and may raise to cancel cooperatively (no gallery is persisted then,
        since the save happens only after the build completes).
        """
        loaded = self._get(model)
        # Lazy import: pulls in the dataset toolkit, unwanted for plain identify traffic.
        from ml.data.dataset import load_dataset
        from ml.data.splits import make_open_set_split
        from ml.eval.evaluate import _build_gallery

        model_settings = loaded.embedder.settings
        bundle = load_dataset(settings=model_settings)
        split = make_open_set_split(df=bundle.df, settings=model_settings)
        with loaded.lock:
            seeded = _build_gallery(split=split, embedder=loaded.embedder, progress_callback=progress_callback)
            loaded.gallery = seeded
            loaded.identifier = Identifier(embedder=loaded.embedder, gallery=seeded, settings=self.settings)
            seeded.save(loaded.info.gallery_path)
        return len(seeded.individuals)
