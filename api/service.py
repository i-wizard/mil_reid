"""
ReidService — the API's business-logic layer over the ML inference seam.

Routers stay thin by delegating everything here: this class owns the (expensive,
load-once) embedder, the in-memory gallery, the identifier, and the rules around
them — readiness gating, gallery persistence, and thread-safety. It is the only
place the API touches ``ml.inference`` / ``ml.eval``, so swapping or extending the
ML core means changing this file alone.

Concurrency note: FastAPI runs our sync handlers in a threadpool, so several
requests can hit this object at once. Reads (identify) are safe on the immutable
loaded model + a snapshot of the gallery; mutations (enroll/delete/reset/seed)
take a lock and re-persist the gallery so concurrent writers can't corrupt it.
"""

import io
import threading
from typing import List, Optional

from ml.config import Settings
from ml.inference.embedder import Embedder, load_embedder
from ml.inference.explain import render_attention_overlay
from ml.inference.gallery import Gallery
from ml.inference.identifier import Identifier, IdentifyResult
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class ModelNotReadyError(RuntimeError):
    """
    Raised when an operation needs the trained head but none is loaded.

    Routers translate this into a 503 so a reviewer who starts the API before
    training gets an actionable message instead of an opaque crash.
    """


class ReidService:
    """
    Stateful façade over the re-identification inference pipeline.

    Construct once at app startup via :meth:`build`. Holds an optional embedder
    (None until a trained head exists), the gallery, and a write lock.
    """

    def __init__(self, settings: Settings, embedder: Optional[Embedder], gallery: Gallery):
        """Hold the loaded components; ``embedder`` is None when no head is trained yet."""
        self.settings = settings
        self.embedder = embedder
        self.gallery = gallery
        self._lock = threading.Lock()
        # Rebuilt whenever the embedder is available; identify() needs it.
        self.identifier = (
            Identifier(embedder=embedder, gallery=gallery, settings=settings) if embedder else None
        )

    @classmethod
    def build(cls, settings: Settings) -> "ReidService":
        """
        Construct the service at startup, tolerating an untrained model.

        Attempts to load the trained head; if the checkpoint is missing the
        service comes up **not ready** (so the container still starts and /health
        can report it) rather than failing. Loads a previously persisted gallery
        when present so enrollments survive restarts.
        """
        embedder: Optional[Embedder] = None
        try:
            embedder = load_embedder(settings=settings)
        except FileNotFoundError:
            logger.warning(
                f"No trained head at {settings.head_weights_path}; API starts in NOT-READY mode. "
                f"Run training, then restart or call the model will load on next start."
            )

        if settings.gallery_path.exists():
            gallery = Gallery.load(settings.gallery_path)
        else:
            gallery = Gallery()

        return cls(settings=settings, embedder=embedder, gallery=gallery)

    def is_ready(self) -> bool:
        """True when a trained head is loaded — the precondition for any inference."""
        return self.embedder is not None and self.identifier is not None

    def _require_ready(self) -> None:
        """Guard used by every model-dependent op; raises the 503-mapped error if not ready."""
        if not self.is_ready():
            raise ModelNotReadyError(
                "No trained model loaded. Train the head (scripts.train) and restart the API."
            )

    # --- Reads -------------------------------------------------------------

    def identify(self, image_path: str, bbox: Optional[List[float]] = None) -> IdentifyResult:
        """Identify the individual in an image against the current gallery (open-set)."""
        self._require_ready()
        return self.identifier.identify(image_path=image_path, bbox=bbox)

    def attention_png(self, image_path: str, bbox: Optional[List[float]] = None) -> bytes:
        """
        Render the attention heatmap overlay for an image and return PNG bytes.

        Embeds the image (one forward pass) and overlays its attention weights,
        encoding to PNG in-memory so the router can stream it without a temp file.
        """
        self._require_ready()
        embed_result = self.embedder.embed_path(path=image_path, bbox=bbox)
        overlay = render_attention_overlay(
            image_path=image_path, embed_result=embed_result, settings=self.settings, bbox=bbox
        )
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        return buffer.getvalue()

    def list_individuals(self) -> List[str]:
        """Return the enrolled individual ids (no model needed — just the gallery)."""
        return self.gallery.individuals

    # --- Mutations (locked + persisted) ------------------------------------

    def enroll(self, individual_id: str, image_paths: List[str]) -> int:
        """
        Enroll/replace an individual from reference images and persist the gallery.

        Returns the number of images averaged into the prototype. Locked so two
        concurrent enrollments can't race on the gallery dict or its on-disk file.
        """
        self._require_ready()
        if not individual_id.strip():
            raise ValueError("individual_id must be non-empty.")
        with self._lock:
            self.gallery.enroll(individual_id=individual_id, image_paths=image_paths, embedder=self.embedder)
            self.gallery.save(self.settings.gallery_path)
        return len(image_paths)

    def delete_individual(self, individual_id: str) -> bool:
        """
        Remove one enrolled individual and persist. Returns False if it was absent.

        Mutates the gallery's backing dict under the lock; the identifier reads the
        same gallery object, so the removal takes effect on the next query.
        """
        with self._lock:
            if individual_id not in self.gallery._embeddings:
                return False
            del self.gallery._embeddings[individual_id]
            self.gallery.save(self.settings.gallery_path)
        return True

    def reset_gallery(self) -> None:
        """Clear all enrolled individuals and persist the now-empty gallery."""
        with self._lock:
            self.gallery._embeddings.clear()
            self.gallery.save(self.settings.gallery_path)

    def seed_from_dataset(self) -> int:
        """
        Auto-enroll the trained dataset's gallery split so /identify works at once.

        Reuses the exact gallery-construction used by evaluation/demo
        (``make_open_set_split`` + ``_build_gallery``) so the seeded gallery
        matches the evaluation protocol. Returns the number of individuals added.
        """
        self._require_ready()
        # Imported lazily: pulls in the dataset toolkit, which we don't want loaded
        # for plain identify/enroll traffic.
        from ml.data.dataset import load_dataset
        from ml.data.splits import make_open_set_split
        from ml.eval.evaluate import _build_gallery

        bundle = load_dataset(settings=self.settings)
        split = make_open_set_split(df=bundle.df, settings=self.settings)
        with self._lock:
            seeded = _build_gallery(split=split, embedder=self.embedder)
            self.gallery = seeded
            self.identifier = Identifier(embedder=self.embedder, gallery=seeded, settings=self.settings)
            self.gallery.save(self.settings.gallery_path)
        return len(seeded.individuals)
