"""
CLI: download the configured dataset via the WildlifeDatasets toolkit.

Separated from feature precomputation because the download is network-bound and
only needs to happen once; keeping it standalone means later steps never
accidentally re-trigger a multi-gigabyte fetch.

Usage:
    python -m scripts.download_data
"""

from ml.config import get_settings
from ml.data.dataset import download_dataset, load_dataset
from ml.utils.logging import get_logger

logger = get_logger("scripts.download_data")


def main() -> None:
    """Download (if absent) and then load the dataset to confirm it parses."""
    settings = get_settings()
    download_dataset(settings=settings)
    bundle = load_dataset(settings=settings)
    logger.info(f"Ready: {bundle.num_images} images across {bundle.num_identities} individuals.")


if __name__ == "__main__":
    main()
