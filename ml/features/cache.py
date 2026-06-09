"""
Precompute and persist per-image patch embeddings.

Because the backbone is frozen, a given image always produces the exact same
bag of patch embeddings. Computing them is the single most expensive step in the
pipeline, so we do it once and write the result to disk keyed by ``image_id``.
Training and evaluation then read these vectors instead of running the backbone,
which is what collapses each training epoch from "run a CNN over thousands of
images" to "load some small arrays" — the reason CPU training is viable.

Cache layout: one ``.npy`` file per image under ``cache_root``, holding a
``[K, feature_dim]`` float32 array (K = patches per bag).
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from ml.config import Settings
from ml.data.dataset import COL_BBOX, COL_IMAGE_ID, COL_PATH
from ml.data.patches import make_bag_from_path
from ml.features.backbone import FrozenBackbone, build_backbone
from ml.utils.logging import get_logger

logger = get_logger(__name__)


def _cache_file(cache_root: Path, image_id: str) -> Path:
    """Map an image id to its cache file path — centralised so readers/writers agree."""
    return cache_root / f"{image_id}.npy"


def precompute_features(
    df: pd.DataFrame,
    settings: Settings,
    backbone: Optional[FrozenBackbone] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Embed every image in ``df`` and write its patch embeddings to the cache.

    Skips images whose cache file already exists, so the (long) precompute step
    is resumable after an interruption and cheap to re-run after adding data.
    Returns the number of newly written files for progress reporting.

    The dataframe is the unit of work (not the whole dataset) so callers can
    precompute only the splits they need. ``progress_callback(done, total)`` is
    invoked periodically (throttled) for callers — e.g. a background job — that
    want to stream progress; it may raise to abort cooperatively (used for job
    cancellation), which simply unwinds this loop.
    """
    settings.cache_root.mkdir(parents=True, exist_ok=True)
    if backbone is None:
        backbone = build_backbone(settings=settings)

    written = 0
    total = len(df)
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        image_id = str(row[COL_IMAGE_ID])
        out_path = _cache_file(cache_root=settings.cache_root, image_id=image_id)
        if not out_path.exists():
            bag = make_bag_from_path(path=row[COL_PATH], bbox=row[COL_BBOX], settings=settings)
            embeddings = backbone.embed_patches(bag.patches).cpu().numpy().astype(np.float32)
            np.save(out_path, embeddings)
            written += 1

        # Report/throttle on every Nth image and the last one. Placed outside the
        # skip-check so progress (and cancellation) advance even when most files
        # are already cached.
        if progress_callback is not None and (position % 25 == 0 or position == total):
            progress_callback(position, total)
        if position % 200 == 0 or position == total:
            logger.info(f"Feature cache: processed {position}/{total} images ({written} new).")

    logger.info(f"Feature cache complete: {written} new files in {settings.cache_root}.")
    return written


def load_cached_features(image_id: str, settings: Settings) -> torch.Tensor:
    """
    Load one image's cached patch embeddings as a tensor.

    Raises a clear error if the cache is missing rather than silently recomputing
    — a missing entry almost always means the precompute step was skipped, and we
    want that surfaced, not hidden behind slow on-the-fly extraction during
    training.
    """
    path = _cache_file(cache_root=settings.cache_root, image_id=image_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features for image_id='{image_id}' at {path}. "
            f"Run precompute_features over this split first."
        )
    return torch.from_numpy(np.load(path))
