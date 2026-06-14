"""
Turning one image into a *bag* of patch instances.

This is where the MIL framing is created. The model never sees a whole image as
a single thing; it sees a set (bag) of equal-sized patches. The attention head
later decides which of those patches carry the animal's identity. We tile on a
regular grid (rather than, say, random crops) for two reasons: it keeps the
mapping from patch index → image region fixed, which is exactly what lets the
attention weights be reshaped back into a heatmap; and a complete, non-redundant
tiling guarantees the discriminative region is captured in *some* patch.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.config import Settings


@dataclass
class PatchBag:
    """
    The bag of patches extracted from a single image.

    ``patches`` is the model input; ``coords`` records each patch's (row, col)
    on the grid so attention weights can be laid back over the image for the
    explainability heatmap; ``grid`` echoes the grid size so the heatmap renderer
    does not have to re-read the settings.
    """

    patches: torch.Tensor  # [K, 3, patch_px, patch_px]
    coords: List[Tuple[int, int]]  # length K, (row, col) per patch
    grid: int


# ImageNet normalisation — the pretrained backbones we use were trained with it,
# so patches must be normalised the same way for their features to be valid.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(image_size: int) -> transforms.Compose:
    """
    Build the deterministic resize→tensor→normalise transform.

    No augmentation here on purpose: tiling and feature extraction must be
    deterministic so cached embeddings exactly match what inference recomputes.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def load_and_crop(path: str, bbox: Optional[List[float]]) -> Image.Image:
    """
    Open an image and crop to its animal bounding box when one is provided.

    Cropping to the bbox first means the patch grid is spent entirely on the
    animal rather than wasting tiles on background — which both sharpens the
    identity signal and keeps the attention heatmap interpretable.
    """
    image = Image.open(path).convert("RGB")
    if bbox is not None and len(bbox) == 4:
        x, y, w, h = bbox
        image = image.crop((int(x), int(y), int(x + w), int(y + h)))
    return image


def make_bag(
    image: Image.Image,
    settings: Settings,
) -> PatchBag:
    """
    Tile a (already-cropped) image into the bag of patch tensors for the model.

    The image is resized to ``effective_image_size`` and split into a
    ``patch_grid``×``patch_grid`` lattice; each cell becomes one normalised patch
    tensor. We resize-then-tile (rather than tile-then-resize) so every patch is
    identical in pixel size regardless of the source image's aspect ratio. The
    resize target and patch size come from the settings' derived geometry, so the
    UPSAMPLED vs NATIVE patch-resolution mode is honoured here without branching.
    """
    transform = _build_transform(image_size=settings.effective_image_size)
    tensor = transform(image)  # [3, H, W]

    grid = settings.patch_grid
    patch_px = settings.patch_pixels

    patches: List[torch.Tensor] = []
    coords: List[Tuple[int, int]] = []
    for row in range(grid):
        for col in range(grid):
            top = row * patch_px
            left = col * patch_px
            patch = tensor[:, top : top + patch_px, left : left + patch_px]
            patches.append(patch)
            coords.append((row, col))

    return PatchBag(patches=torch.stack(patches, dim=0), coords=coords, grid=grid)


def make_bag_from_path(path: str, bbox: Optional[List[float]], settings: Settings) -> PatchBag:
    """
    Convenience: load+crop+tile in one call.

    Used by feature precomputation and by inference so both follow the identical
    path → bag route and can never diverge in preprocessing.
    """
    image = load_and_crop(path=path, bbox=bbox)
    return make_bag(image=image, settings=settings)


def attention_to_grid(attention: np.ndarray, grid: int) -> np.ndarray:
    """
    Reshape a flat per-patch attention vector back into a 2-D grid.

    Inverse of the row-major tiling in :func:`make_bag`; the explain module
    upsamples the result to image size to produce the heatmap overlay.
    """
    return attention.reshape(grid, grid)
