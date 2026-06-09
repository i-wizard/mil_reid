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


def load_embedder(settings: Settings = None) -> Embedder:
    """
    Reconstruct an ``Embedder`` from the saved training checkpoint.

    Rebuilds the head against the checkpoint's stored ``feature_dim`` and loads
    its weights, so the embedder is byte-identical to the trained model. Raises a
    clear error if no checkpoint exists, since inference is meaningless without a
    trained head.
    """
    settings = settings if settings is not None else get_settings()
    if not settings.head_weights_path.exists():
        raise FileNotFoundError(
            f"No trained head at {settings.head_weights_path}. Run training before inference."
        )

    checkpoint = torch.load(settings.head_weights_path, map_location="cpu", weights_only=False)
    feature_dim = checkpoint["feature_dim"]

    backbone = build_backbone(settings=settings)
    head = MILEmbedder(settings=settings, feature_dim=feature_dim)
    head.load_state_dict(checkpoint["head_state_dict"])

    logger.info(f"Loaded embedder from {settings.head_weights_path} (feature_dim={feature_dim}).")
    return Embedder(backbone=backbone, head=head, settings=settings)
