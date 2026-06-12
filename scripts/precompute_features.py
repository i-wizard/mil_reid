"""
CLI: build the frozen-backbone patch-embedding cache for the whole dataset.

Run this once after downloading. It populates the cache that training and
evaluation read from, which is what makes those steps fast on CPU. Because it
skips images already cached, it is safe to re-run after adding data or after an
interruption.

Usage:
    python -m scripts.precompute_features
"""

from ml.config import get_settings
from ml.data.dataset import load_dataset, mark_precomputed
from ml.features.cache import precompute_features
from ml.utils.logging import get_logger

logger = get_logger("scripts.precompute_features")


def main() -> None:
    """Cache patch embeddings for every image in the configured dataset."""
    settings = get_settings()
    bundle = load_dataset(settings=settings)
    written = precompute_features(df=bundle.df, settings=settings)
    # Record completion so the API/UI and the train guard recognise this dataset as
    # precomputed (under the current cache signature), matching the job path.
    mark_precomputed(name=settings.dataset, settings=settings, count=bundle.num_images)
    logger.info(f"Done. {written} new cache files written under {settings.cache_root}.")


if __name__ == "__main__":
    main()
