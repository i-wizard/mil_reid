"""
Thin wrapper over the WildlifeDatasets toolkit.

We isolate every call into ``wildlife_datasets`` here, in one small module,
because the toolkit's exact class/attribute names can shift between versions.
Keeping the dependency behind ``load_dataset`` / ``DatasetBundle`` means the rest
of the pipeline depends only on a plain pandas dataframe with a stable schema —
if the toolkit's API changes, this is the single file to fix.

The dataframe we hand onward is guaranteed to expose:
    - ``image_id``  : stable unique id per image (used as the feature-cache key)
    - ``identity``  : the individual-animal label (the thing we re-identify)
    - ``path``      : absolute path to the image file on disk
    - ``bbox``      : optional [x, y, w, h] crop, or None when absent
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ml.config import Settings
from ml.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical column names the rest of the pipeline relies on. Declared once so a
# rename only has to happen in this module.
COL_IMAGE_ID = "image_id"
COL_IDENTITY = "identity"
COL_PATH = "path"
COL_BBOX = "bbox"

# A curated, demo-friendly subset of WildlifeDatasets the Settings UI offers for
# download. Kept small and (mostly) freely auto-downloadable so a reviewer isn't
# pulling tens of GB or hitting Kaggle auth. Declared as plain strings so listing
# the catalogue needs no SDK import — only the actual download resolves the class.
CURATED_DATASETS: List[str] = [
    "SeaTurtleIDHeads",
    "IPanda50",
    "MacaqueFaces",
    "AAUZebraFish",
    "CatIndividualImages",
    "DogFaceNet",
]


def is_downloaded(name: str, settings: Settings) -> bool:
    """
    Whether a dataset's files are already present on disk.

    Mirrors the skip-check in :func:`download_dataset` (directory exists and is
    non-empty) so the UI's "downloaded" badge and the downloader agree on what
    "downloaded" means.
    """
    dataset_dir = settings.data_root / name
    return dataset_dir.is_dir() and any(dataset_dir.iterdir())


def _cache_signature(settings: Settings) -> Dict[str, object]:
    """
    The settings that determine a cached embedding, used to validate precompute markers.

    If any of these change, previously-cached features are stale, so a precompute
    marker recorded under a different signature must not count as "precomputed".
    """
    return {
        "backbone": settings.backbone.value,
        "patch_resolution": settings.patch_resolution.value,
        "patch_grid": settings.patch_grid,
        "image_size": settings.image_size,
        "backbone_input_size": settings.backbone_input_size,
    }


def _marker_path(name: str, settings: Settings) -> Path:
    """Where a dataset's precompute-completion marker lives (under the feature cache)."""
    return settings.cache_root / ".markers" / f"{name}.json"


def mark_precomputed(name: str, settings: Settings, count: int) -> None:
    """
    Record that ``name`` has been fully precomputed under the current cache signature.

    Written by the precompute step (job + CLI) on success so the UI/train guard can
    tell, in O(1), whether a dataset's features are ready — without enumerating the
    flat, image-id-keyed cache or loading the dataset.
    """
    path = _marker_path(name=name, settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": _cache_signature(settings=settings), "count": count}))


def is_precomputed(name: str, settings: Settings) -> bool:
    """
    Whether ``name`` has a precompute marker matching the *current* cache signature.

    A marker from a different backbone/patch geometry reads as not-precomputed, which
    correctly forces a re-run after changing those settings.
    """
    path = _marker_path(name=name, settings=settings)
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text())
    except (ValueError, OSError):
        return False
    return marker.get("signature") == _cache_signature(settings=settings)


def catalog(settings: Settings) -> List[Dict[str, object]]:
    """
    The curated datasets plus each one's downloaded + precomputed status, for Settings.

    No SDK import here — filesystem checks (dir presence + marker file) are enough, so
    rendering the catalogue stays fast and never pulls in the heavy toolkit.
    """
    return [
        {
            "name": name,
            "downloaded": is_downloaded(name=name, settings=settings),
            "precomputed": is_precomputed(name=name, settings=settings),
        }
        for name in CURATED_DATASETS
    ]


@dataclass
class DatasetBundle:
    """
    A loaded dataset reduced to what the pipeline actually needs.

    Bundling the dataframe with its root path keeps path resolution (relative
    paths in the toolkit's dataframe vs. absolute paths we need to open files)
    in one place rather than recomputed at every call site.
    """

    name: str
    root: Path
    df: pd.DataFrame

    @property
    def num_images(self) -> int:
        """Total images available — used for sanity logging and cache sizing."""
        return len(self.df)

    @property
    def num_identities(self) -> int:
        """Distinct individuals present — drives the ArcFace classifier width."""
        return self.df[COL_IDENTITY].nunique()


def download_dataset(settings: Settings) -> Path:
    """
    Ensure the configured dataset is present on disk, downloading if needed.

    Split out from loading so the (slow, network-bound) download can be run once
    as its own script step and then skipped on every subsequent run.
    """
    from wildlife_datasets import datasets

    dataset_dir = settings.data_root / settings.dataset
    dataset_cls = getattr(datasets, settings.dataset)

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        logger.info(f"Dataset '{settings.dataset}' already present at {dataset_dir}; skipping download.")
        return dataset_dir

    logger.info(f"Downloading dataset '{settings.dataset}' into {dataset_dir} ...")
    dataset_cls.get_data(str(dataset_dir))
    return dataset_dir


def load_dataset(settings: Settings) -> DatasetBundle:
    """
    Load the configured dataset's metadata into our normalized schema.

    Returns a ``DatasetBundle`` whose dataframe is guaranteed to carry the four
    canonical columns above with absolute, openable image paths — so downstream
    code never has to know which toolkit dataset it came from.
    """
    from wildlife_datasets import datasets

    dataset_dir = settings.data_root / settings.dataset
    dataset_cls = getattr(datasets, settings.dataset)
    dataset = dataset_cls(str(dataset_dir))

    df = dataset.df.copy()
    df = _normalize_columns(df=df, root=dataset_dir)

    bundle = DatasetBundle(name=settings.dataset, root=dataset_dir, df=df)
    logger.info(
        f"Loaded '{settings.dataset}': {bundle.num_images} images, "
        f"{bundle.num_identities} individuals."
    )
    return bundle


def _normalize_columns(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """
    Coerce the toolkit dataframe into our canonical schema.

    The toolkit stores paths relative to the dataset root and may omit a bbox
    column entirely; we resolve paths to absolute here and synthesize a None
    bbox column when missing so downstream code can treat the schema as fixed.
    """
    if COL_PATH not in df.columns:
        raise ValueError(f"Expected a '{COL_PATH}' column in the dataset dataframe; got {list(df.columns)}")

    # Resolve to absolute paths once, here, so every reader can just open them.
    df[COL_PATH] = df[COL_PATH].apply(lambda p: str((root / p).resolve()) if not Path(p).is_absolute() else p)

    if COL_IMAGE_ID not in df.columns:
        # Fall back to the row index as a stable id when the toolkit omits one.
        df[COL_IMAGE_ID] = df.index.astype(str)

    if COL_BBOX not in df.columns:
        df[COL_BBOX] = None

    return df[[COL_IMAGE_ID, COL_IDENTITY, COL_PATH, COL_BBOX]]
