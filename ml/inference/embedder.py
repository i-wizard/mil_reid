"""
Image → identity embedding, the single source of truth for "how an image becomes
a vector".

Both enrollment and querying must embed images *identically* — any difference in
preprocessing, backbone, or head would put gallery and query vectors in
incomparable spaces and silently wreck retrieval. So everything routes through
this one ``Embedder``: it loads the frozen backbone and the trained MIL head from
the training checkpoint and chains them exactly as training did.

It also returns the attention weights and patch coordinates, because the
explainability heatmap must reflect the *same* forward pass that produced the
embedding — recomputing attention separately could drift from what was used.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ml.config import Settings, get_settings
from ml.data.patches import make_bag_from_path
from ml.features.backbone import FrozenBackbone, build_backbone
from ml.models.attention_mil import MILEmbedder
from ml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbedResult:
    """
    Everything one forward pass yields about an image.

    ``embedding`` feeds retrieval; ``attention`` + ``coords`` + ``grid`` feed the
    heatmap. Bundling them guarantees callers use attention that matches the
    returned embedding.
    """

    embedding: np.ndarray  # [embedding_dim], L2-normalized
    attention: np.ndarray  # [K] weights over patches, sums to 1
    coords: List[Tuple[int, int]]  # (row, col) per patch
    grid: int


class Embedder:
    """
    Loads the frozen backbone + trained head and turns image paths into vectors.

    Constructed from a checkpoint so the inference-time geometry (feature_dim,
    embedding_dim, pooling type) is whatever training actually used, not whatever
    the live settings happen to say.
    """

    def __init__(self, backbone: FrozenBackbone, head: MILEmbedder, settings: Settings):
        """Hold the two (eval-mode) components and the settings driving patching."""
        self.backbone = backbone
        self.head = head
        self.settings = settings
        self.head.eval()

    @torch.inference_mode()
    def embed_path(self, path: str, bbox=None) -> EmbedResult:
        """
        Embed an image file into an identity vector plus its attention map.

        Runs the identical patch→backbone→MIL-head route used in training, under
        ``inference_mode`` since nothing here trains. The bbox is optional so the
        API can pass a user-supplied crop or let the model tile the whole image.
        """
        bag = make_bag_from_path(path=path, bbox=bbox, settings=self.settings)
        patch_features = self.backbone.embed_patches(bag.patches)  # [K, feature_dim]
        embedding, attention = self.head(patch_features.unsqueeze(0))  # batch of 1
        return EmbedResult(
            embedding=embedding.squeeze(0).cpu().numpy(),
            attention=attention.squeeze(0).cpu().numpy(),
            coords=bag.coords,
            grid=bag.grid,
        )


def _settings_from_checkpoint(checkpoint: dict, fallback: Settings) -> Settings:
    """
    Rebuild the training-time ``Settings`` from a checkpoint's saved snapshot.

    A model's backbone and patch geometry are properties of *how it was trained*,
    not of the current environment — so we reconstruct them from the snapshot
    ``_save_checkpoint`` wrote. Without this, a model trained with, say, ``resnet50``
    or ``NATIVE`` patches would be loaded with the live env's backbone/geometry and
    produce garbage embeddings. Falls back to ``fallback`` for legacy checkpoints
    that predate the snapshot.
    """
    snapshot = checkpoint.get("settings")
    if not snapshot:
        return fallback
    try:
        return Settings(**snapshot)
    except Exception as error:
        logger.warning(f"Could not rebuild settings from checkpoint snapshot ({error}); using live settings.")
        return fallback


def _load_embedder_from_path(checkpoint_path: Path, fallback_settings: Settings) -> Embedder:
    """
    Build an ``Embedder`` from a specific checkpoint file.

    The backbone + head are constructed from the checkpoint's own settings so the
    embedder is byte-identical to the trained model regardless of the current
    environment. Shared by the active-model and named-model entry points.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained head at {checkpoint_path}. Run training before inference.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_dim = checkpoint["feature_dim"]
    model_settings = _settings_from_checkpoint(checkpoint=checkpoint, fallback=fallback_settings)

    backbone = build_backbone(settings=model_settings)
    head = MILEmbedder(settings=model_settings, feature_dim=feature_dim)
    head.load_state_dict(checkpoint["head_state_dict"])

    logger.info(
        f"Loaded embedder from {checkpoint_path} "
        f"(backbone={model_settings.backbone.value}, feature_dim={feature_dim})."
    )
    return Embedder(backbone=backbone, head=head, settings=model_settings)


def load_embedder(settings: Settings = None) -> Embedder:
    """
    Reconstruct the *active* model's ``Embedder`` (``settings.model_name``).

    Loads from the active model's checkpoint path and rebuilds it from the
    checkpoint's saved settings. Raises ``FileNotFoundError`` when untrained, since
    inference is meaningless without a head.
    """
    settings = settings if settings is not None else get_settings()
    return _load_embedder_from_path(checkpoint_path=settings.head_weights_path, fallback_settings=settings)


def load_embedder_for(checkpoint_path: Path, settings: Settings = None) -> Embedder:
    """
    Reconstruct an ``Embedder`` from an explicit checkpoint path (a named model).

    Used by the multi-model registry/service, which resolves each model's
    checkpoint path itself. The embedder's geometry comes from the checkpoint, so
    ``settings`` is only a fallback for legacy snapshot-less checkpoints.
    """
    settings = settings if settings is not None else get_settings()
    return _load_embedder_from_path(checkpoint_path=checkpoint_path, fallback_settings=settings)
