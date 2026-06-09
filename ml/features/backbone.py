"""
The frozen patch feature extractor.

This backbone is the part we deliberately do *not* train. Freezing a strong
pretrained model and learning only a tiny pooling head on top is what makes the
whole pipeline CPU-trainable and data-efficient — the expensive representation
learning was already paid for by the pretrained weights, and we reuse it.

Each patch crop is pushed through the backbone independently and reduced to one
embedding vector, so an image's bag of K patches becomes K vectors that the MIL
head will later pool.
"""

from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import BackboneName, Settings
from ml.utils.logging import get_logger

logger = get_logger(__name__)


class FrozenBackbone(nn.Module):
    """
    Wraps a pretrained timm/MegaDescriptor model as a frozen patch embedder.

    Exposes a single ``embed_patches`` method returning one vector per patch.
    Kept as an ``nn.Module`` (rather than a bare function) so it participates in
    ``.to(device)`` / ``.eval()`` and so its output width can be queried to
    configure the downstream head.
    """

    def __init__(self, settings: Settings):
        """
        Load the configured backbone with pretrained weights and freeze it.

        We request ``num_classes=0`` so timm returns pooled feature vectors
        rather than classification logits, and immediately disable gradients so
        no optimizer can ever accidentally update these weights.
        """
        super().__init__()
        self.settings = settings
        self.model = timm.create_model(
            settings.backbone.value,
            pretrained=True,
            num_classes=0,  # strip the classifier → emit feature vectors
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self._feature_dim = self.model.num_features
        if self._feature_dim != settings.feature_dim:
            # Surface the mismatch loudly: the head is sized from settings, so a
            # silent disagreement would only fail much later as a shape error.
            logger.warning(
                f"Backbone '{settings.backbone.value}' emits {self._feature_dim}-d features "
                f"but settings.feature_dim={settings.feature_dim}. Using the backbone's {self._feature_dim}."
            )

        # Many strong backbones (Swin/ViT — incl. MegaDescriptor) hard-require a
        # fixed input resolution and will assert if fed anything else. Our patches
        # are image_size/patch_grid on a side (e.g. 56), so we must upsample each
        # patch to the backbone's expected size before the forward pass. Resolve
        # that size from the model's own data config so it can never drift.
        self._input_size = self._resolve_input_size()

    def _resolve_input_size(self) -> int:
        """
        Read the backbone's expected square input side (e.g. 224) from timm.

        Prefers timm's resolved data config and falls back to the pretrained
        config; defaults to 224 if neither is present. Used by ``embed_patches``
        to resize patches so a fixed-input backbone (Swin/ViT) accepts them.
        """
        try:
            data_config = timm.data.resolve_model_data_config(self.model)
            return int(data_config["input_size"][-1])
        except Exception:
            pretrained_cfg = getattr(self.model, "pretrained_cfg", {}) or {}
            input_size = pretrained_cfg.get("input_size")
            if input_size:
                return int(input_size[-1])
            return 224

    @property
    def input_size(self) -> int:
        """The square input side the backbone expects; patches are resized to it."""
        return self._input_size

    @property
    def feature_dim(self) -> int:
        """Backbone output width — the head reads this to size its first layer."""
        return self._feature_dim

    @torch.inference_mode()
    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Embed a bag of patches into one feature vector each.

        ``patches`` is ``[K, 3, px, px]`` and the return is ``[K, feature_dim]``.
        Runs under ``inference_mode`` because the backbone is frozen — there is
        never a backward pass through it, and this avoids building an autograd
        graph we would only throw away.

        Patches are bilinearly resized to the backbone's expected input size when
        they differ, because fixed-input backbones (Swin/ViT, including
        MegaDescriptor) assert on any other resolution. Upsampling a patch adds no
        information but lets the pretrained model embed each region consistently.
        """
        device = next(self.model.parameters()).device
        patches = patches.to(device)
        if patches.shape[-1] != self._input_size or patches.shape[-2] != self._input_size:
            patches = F.interpolate(
                patches,
                size=(self._input_size, self._input_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.model(patches)


def build_backbone(settings: Settings) -> FrozenBackbone:
    """
    Construct the frozen backbone described by ``settings``.

    A factory (rather than direct construction at call sites) gives us one place
    to later add caching of the loaded model or device placement policy.
    """
    backbone = FrozenBackbone(settings=settings)
    logger.info(
        f"Loaded frozen backbone '{settings.backbone.value}' "
        f"(feature_dim={backbone.feature_dim}); weights are not trainable."
    )
    return backbone


def supported_backbones() -> List[str]:
    """Return the configured backbone identifiers — used by CLI help and tests."""
    return [member.value for member in BackboneName]
