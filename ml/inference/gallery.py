"""
The gallery: enrolled reference embeddings for known individuals.

Open-set re-identification works by comparison, not classification — so we keep
a gallery of vectors, one (mean) embedding per enrolled individual, and identify
a query by finding its nearest gallery neighbour. The decisive property is that
**enrolling a new individual is just appending a vector**: no retraining, which
is exactly what closed-set classification cannot offer and what makes this
practical for conservation use where new animals appear constantly.

We store a mean embedding per individual (averaged over their reference images
and re-normalised). Averaging denoises pose/lighting variation into a single
stable prototype, which both shrinks the gallery and stabilises the similarity
scores the unknown-threshold is calibrated against.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ml.inference.embedder import Embedder
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class Gallery:
    """
    An in-memory, disk-persistable map of individual id → prototype embedding.

    Backed by a plain dict plus a lazily-built matrix for fast batched cosine
    similarity. Persistence is a single ``.npz`` so the (later) API can load the
    same gallery the offline scripts built.
    """

    def __init__(self):
        """Start empty; individuals are added via :meth:`enroll`."""
        self._embeddings: Dict[str, np.ndarray] = {}

    @property
    def individuals(self) -> List[str]:
        """Enrolled individual ids, in insertion order — used for reporting."""
        return list(self._embeddings.keys())

    def enroll(self, individual_id: str, image_paths: List[str], embedder: Embedder) -> None:
        """
        Add (or replace) an individual from one or more reference images.

        Each image is embedded and the L2-normalised mean is stored as the
        individual's prototype. Taking the mean across several references is why
        a single bad pose does not define the whole identity. Re-enrolling an id
        overwrites it, so corrections are cheap.
        """
        if not image_paths:
            raise ValueError(f"Cannot enroll '{individual_id}' with no images.")

        vectors = [embedder.embed_path(path=p).embedding for p in image_paths]
        prototype = np.mean(np.stack(vectors, axis=0), axis=0)
        prototype = prototype / (np.linalg.norm(prototype) + 1e-12)
        self._embeddings[individual_id] = prototype.astype(np.float32)
        logger.info(f"Enrolled '{individual_id}' from {len(image_paths)} image(s).")

    def as_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Return (ids, matrix) where row i is the prototype for ids[i].

        The identifier uses this for a single vectorised cosine computation
        against all individuals at once, rather than looping per individual.
        """
        ids = list(self._embeddings.keys())
        if not ids:
            return [], np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([self._embeddings[i] for i in ids], axis=0)
        return ids, matrix

    def save(self, path: Path) -> None:
        """
        Persist the gallery to a single ``.npz`` (ids + stacked prototypes).

        One file keeps enrollment reproducible and lets the API load exactly what
        the offline pipeline produced.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ids, matrix = self.as_matrix()
        np.savez(path, ids=np.array(ids, dtype=object), embeddings=matrix)
        logger.info(f"Saved gallery with {len(ids)} individuals to {path}.")

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        """
        Rebuild a gallery from a ``.npz`` written by :meth:`save`.

        A classmethod constructor so callers get a ready-to-query gallery in one
        call without poking at internals.
        """
        gallery = cls()
        data = np.load(path, allow_pickle=True)
        ids = data["ids"].tolist()
        embeddings = data["embeddings"]
        for i, individual_id in enumerate(ids):
            gallery._embeddings[str(individual_id)] = embeddings[i]
        logger.info(f"Loaded gallery with {len(ids)} individuals from {path}.")
        return gallery
