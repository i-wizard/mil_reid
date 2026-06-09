"""
Explainability: turn the MIL attention weights into a heatmap over the image.

The attention weights are not a side artefact — they are *the* evidence for the
model's decision, since the identity embedding is literally their weighted sum of
patches. Visualising them answers "which part of the animal did the model use?",
which is both the key paper figure and the feature that makes the demo
trustworthy to a non-expert viewer.

The attention vector is reshaped to the patch grid, upsampled to image
resolution, and blended over the original image as a warm overlay.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ml.config import Settings
from ml.data.patches import attention_to_grid, load_and_crop
from ml.inference.embedder import EmbedResult


def render_attention_overlay(
    image_path: str,
    embed_result: EmbedResult,
    settings: Settings,
    bbox: Optional[list] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Produce an RGB image with the attention heatmap blended over the animal.

    We re-load and crop the image the same way the embedder did (so the overlay
    aligns with the patches that were actually scored), build a coarse grid
    heatmap from the attention weights, upsample it smoothly to the image size,
    and alpha-blend it. ``alpha`` trades off heatmap visibility against seeing the
    underlying animal.
    """
    base = load_and_crop(path=image_path, bbox=bbox).resize((settings.image_size, settings.image_size))

    grid_heat = attention_to_grid(attention=embed_result.attention, grid=embed_result.grid)
    heat_image = _grid_to_heat_rgb(grid_heat=grid_heat, size=settings.image_size)

    blended = Image.blend(base.convert("RGB"), heat_image, alpha=alpha)
    return blended


def save_attention_overlay(
    image_path: str,
    embed_result: EmbedResult,
    settings: Settings,
    out_path: Path,
    bbox: Optional[list] = None,
) -> Path:
    """
    Render and write the attention overlay to ``out_path``.

    A thin convenience wrapper so scripts and the API can persist a figure in one
    call; returns the path for logging/chaining.
    """
    overlay = render_attention_overlay(
        image_path=image_path, embed_result=embed_result, settings=settings, bbox=bbox
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    return out_path


def _grid_to_heat_rgb(grid_heat: np.ndarray, size: int) -> Image.Image:
    """
    Convert a ``grid×grid`` attention array into a smooth ``size×size`` RGB heatmap.

    Normalises to [0, 1] for contrast, maps intensity to a simple warm ramp
    (low = dark blue, high = red/yellow), and uses bicubic upsampling so the
    coarse patch grid reads as a smooth field rather than hard blocks. We avoid a
    matplotlib colormap dependency here to keep the inference path light.
    """
    normalised = grid_heat - grid_heat.min()
    denominator = normalised.max() if normalised.max() > 0 else 1.0
    normalised = normalised / denominator

    # Warm ramp: red rises early, green rises later, blue fades out — yields a
    # blue→red→yellow progression as attention increases.
    red = np.clip(normalised * 1.5, 0, 1)
    green = np.clip(normalised * 1.5 - 0.5, 0, 1)
    blue = np.clip(0.5 - normalised, 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)

    small = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    return small.resize((size, size), resample=Image.BICUBIC)
