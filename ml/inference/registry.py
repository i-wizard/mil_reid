"""
Discovery of trained models on disk.

Multi-model support means "which models exist?" must be answered from the
filesystem, not hard-coded. A model is simply a directory under
``artifacts/models/<name>/`` containing a ``mil_head.pt`` checkpoint (and,
optionally, a ``gallery.npz``). This registry scans for those, reads each
checkpoint's saved settings snapshot for display metadata (dataset/backbone), and
resolves the per-model paths the service needs.

It also bridges the *legacy* single-model layout: if an old ``artifacts/mil_head.pt``
exists and no namespaced models do, it is surfaced as the model ``"default"`` so a
previously-trained model keeps working without retraining or moving files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ml.config import Settings
from ml.utils.logging import get_logger

logger = get_logger(__name__)

LEGACY_MODEL_NAME = "default"


@dataclass
class ModelInfo:
    """
    Where a model's files live plus the metadata needed to describe it.

    ``checkpoint_path``/``gallery_path`` are what the service loads; ``dataset`` and
    ``backbone`` are read from the checkpoint snapshot purely for display (so the UI
    can show "turtles · SeaTurtleIDHeads · MegaDescriptor-T" without loading torch).
    """

    name: str
    checkpoint_path: Path
    gallery_path: Path
    dataset: Optional[str] = None
    backbone: Optional[str] = None


class ModelRegistry:
    """
    Filesystem-backed catalogue of trained models.

    Re-scans on demand (``discover``) rather than caching forever, so a model
    trained while the API is running shows up without a restart once re-scanned.
    """

    def __init__(self, settings: Settings):
        """Hold settings so path resolution uses the configured artifacts layout."""
        self.settings = settings

    def discover(self) -> Dict[str, ModelInfo]:
        """
        Scan the artifacts tree and return every available model by name.

        Prefers the namespaced ``artifacts/models/<name>/`` layout; if none are
        found, falls back to surfacing a legacy ``artifacts/mil_head.pt`` as
        ``"default"`` so old setups keep working.
        """
        models: Dict[str, ModelInfo] = {}

        models_root = self.settings.models_root
        if models_root.is_dir():
            for entry in sorted(models_root.iterdir()):
                checkpoint = entry / "mil_head.pt"
                if entry.is_dir() and checkpoint.exists():
                    models[entry.name] = self._describe(name=entry.name, checkpoint_path=checkpoint)

        if not models:
            legacy = self.settings.artifacts_root / "mil_head.pt"
            if legacy.exists():
                logger.info(f"Surfacing legacy checkpoint {legacy} as model '{LEGACY_MODEL_NAME}'.")
                models[LEGACY_MODEL_NAME] = self._describe(
                    name=LEGACY_MODEL_NAME,
                    checkpoint_path=legacy,
                    gallery_path=self.settings.artifacts_root / "gallery.npz",
                )

        return models

    def get(self, name: str) -> Optional[ModelInfo]:
        """Return one model's info, or None if it is not present on disk."""
        return self.discover().get(name)

    def names(self) -> List[str]:
        """List the available model names (sorted) — used for defaults and listing."""
        return sorted(self.discover().keys())

    def _describe(self, name: str, checkpoint_path: Path, gallery_path: Optional[Path] = None) -> ModelInfo:
        """
        Build a ``ModelInfo``, reading dataset/backbone from the checkpoint snapshot.

        The snapshot read is best-effort and metadata-only: a corrupt or
        snapshot-less checkpoint still yields a usable ModelInfo (just without the
        display fields), so discovery never fails because of one bad file.
        """
        gallery_path = gallery_path if gallery_path is not None else self.settings.gallery_path_for(name)
        dataset = backbone = None
        try:
            snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("settings", {})
            dataset = snapshot.get("dataset")
            backbone = snapshot.get("backbone")
        except Exception as error:
            logger.warning(f"Could not read snapshot from {checkpoint_path} for metadata: {error}")
        return ModelInfo(
            name=name,
            checkpoint_path=checkpoint_path,
            gallery_path=gallery_path,
            dataset=dataset,
            backbone=backbone,
        )
