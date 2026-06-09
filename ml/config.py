"""
Central configuration for the re-identification ML core.

Every tunable knob lives here as a single ``Settings`` object so that scripts,
training, and inference all read the *same* values. Keeping configuration in one
typed place (rather than scattered constants) is what lets us, for example,
precompute features under one patch grid and be certain training/inference use
the identical grid — a mismatch there would silently corrupt the embeddings.

Fixed-choice fields are modelled as ``str, Enum`` rather than free strings so an
invalid backbone or pooling name fails loudly at construction time instead of
deep inside a model call.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackboneName(str, Enum):
    """
    Frozen feature-extractor backbones.

    These never train; they only turn patch crops into embedding vectors. The
    MegaDescriptor option is the wildlife-specific foundation model (best
    accuracy); the timm options are lighter fallbacks when the HF download or
    the extra accuracy is not wanted.
    """

    MEGADESCRIPTOR_T = "hf-hub:BVRA/MegaDescriptor-T-224"
    VIT_SMALL = "vit_small_patch16_224"
    RESNET50 = "resnet50"


class PoolingType(str, Enum):
    """
    Bag-pooling strategy.

    ``GATED_ATTENTION`` is the paper's contribution; ``MEAN`` is the ablation
    baseline that the evaluation compares against to quantify what attention
    actually buys us.
    """

    GATED_ATTENTION = "GATED_ATTENTION"
    MEAN = "MEAN"


class PatchResolution(str, Enum):
    """
    How much detail each patch instance carries before the frozen backbone.

    The backbone (e.g. MegaDescriptor/Swin) demands a fixed input side (224), so a
    patch must reach that size one way or another — the choice is *when* the
    detail is fixed:

    - ``UPSAMPLED`` — resize the whole animal to ``image_size`` (e.g. 224), tile
      into small patches (e.g. 56px), then let the backbone bilinearly upsample
      each patch to 224. Cheap, but a patch is a blurry low-detail region.

    - ``NATIVE`` — resize the whole animal to ``patch_grid × backbone_input_size``
      (e.g. 896) so each tile is *natively* the backbone's input size (224) with
      genuine fine detail. Higher fidelity per patch (better for distinguishing
      fur/scale texture), at more compute per image.
    """

    UPSAMPLED = "UPSAMPLED"
    NATIVE = "NATIVE"


class Settings(BaseSettings):
    """
    Single source of truth for paths, dataset choice, and all hyper-parameters.

    Values may be overridden via environment variables (prefix ``REID_``) so the
    same code runs unchanged in a container without editing this file.
    """

    model_config = SettingsConfigDict(env_prefix="REID_", protected_namespaces=())

    # --- Storage layout ---
    data_root: Path = Field(
        default=Path("data"),
        description="Where WildlifeDatasets downloads and extracts raw images.",
    )
    cache_root: Path = Field(
        default=Path("artifacts/feature_cache"),
        description="Where precomputed per-image patch embeddings are persisted.",
    )
    artifacts_root: Path = Field(
        default=Path("artifacts"),
        description="Where trained head weights, the gallery index, and reports live.",
    )

    # --- Model identity (multi-model support) ---
    # Each trained model lives in its own namespace so several species/domains can
    # coexist: training writes to artifacts/models/<model_name>/, and the API
    # selects which model to enroll/identify against. Defaults to "default" so the
    # single-model workflow is unchanged.
    model_name: str = Field(
        default="default",
        description="Logical name of the model being trained/served; namespaces its checkpoint + gallery.",
    )

    # --- Dataset / backbone selection ---
    # `dataset` is the WildlifeDatasets class name (e.g. "SeaTurtleIDHeads"). It is a
    # free string (not an enum) so the Settings view can train on any curated
    # dataset; validity is checked at the API boundary against CURATED_DATASETS,
    # keeping this field — and every Settings() construction — free of any SDK import.
    dataset: str = Field(default="SeaTurtleIDHeads", description="WildlifeDatasets class name to load.")
    backbone: BackboneName = BackboneName.MEGADESCRIPTOR_T
    pooling: PoolingType = PoolingType.GATED_ATTENTION

    # --- Image / patch geometry ---
    # image_size must be divisible by patch_grid so tiling produces equal patches.
    image_size: int = Field(default=224, description="UPSAMPLED-mode resize side for the cropped animal (ignored in NATIVE mode).")
    patch_grid: int = Field(default=4, description="Tiles per side; the bag holds patch_grid**2 patches.")
    patch_resolution: PatchResolution = Field(
        default=PatchResolution.UPSAMPLED,
        description="UPSAMPLED (cheap, patches upsampled by backbone) or NATIVE (each tile is full backbone-resolution detail).",
    )
    backbone_input_size: int = Field(
        default=224,
        description="The square input side the backbone expects (224 for MegaDescriptor/ViT-S). Drives NATIVE-mode tiling.",
    )

    # --- Backbone output / head geometry ---
    feature_dim: int = Field(default=768, description="Backbone embedding width per patch. Must match the chosen backbone.")
    attention_dim: int = Field(default=128, description="Hidden width of the gated-attention scoring MLP.")
    embedding_dim: int = Field(default=256, description="Final L2-normalized identity embedding width.")

    # --- ArcFace (train-time only) ---
    arcface_margin: float = 0.5
    arcface_scale: float = 30.0

    # --- Training ---
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42

    # --- Open-set retrieval ---
    top_k: int = Field(default=5, description="How many ranked candidates identify() returns.")
    unknown_threshold: float = Field(
        default=0.5,
        description="Cosine similarity below which the top match is rejected as an unseen individual.",
    )

    # --- Open-set split proportions ---
    open_set_ratio: float = Field(
        default=0.2,
        description="Fraction of identities held out as never-seen-in-training (used to test the unknown threshold).",
    )

    @property
    def num_patches(self) -> int:
        """Number of instances per bag — derived so it can never drift from patch_grid."""
        return self.patch_grid * self.patch_grid

    @property
    def effective_image_size(self) -> int:
        """
        The side the cropped animal is actually resized to before tiling.

        Derived from the patch-resolution mode so tiling geometry follows the
        chosen fidelity without callers branching: NATIVE yields a large canvas
        (``patch_grid × backbone_input_size``) so each tile is full-resolution;
        UPSAMPLED keeps the small ``image_size`` canvas.
        """
        if self.patch_resolution is PatchResolution.NATIVE:
            return self.patch_grid * self.backbone_input_size
        return self.image_size

    @property
    def patch_pixels(self) -> int:
        """
        Side length of a single tiled patch in pixels.

        In NATIVE mode this equals ``backbone_input_size`` (no upsampling needed);
        in UPSAMPLED mode it is the smaller ``image_size / patch_grid`` and the
        backbone upsamples it. Derived so tiling and the backbone always agree.
        """
        return self.effective_image_size // self.patch_grid

    @property
    def models_root(self) -> Path:
        """Directory holding every model's namespaced subdirectory."""
        return self.artifacts_root / "models"

    def model_dir(self, name: str) -> Path:
        """Per-model directory (checkpoint + gallery) for ``name``."""
        return self.models_root / name

    def head_weights_path_for(self, name: str) -> Path:
        """Trained head checkpoint path for a named model."""
        return self.model_dir(name) / "mil_head.pt"

    def gallery_path_for(self, name: str) -> Path:
        """Persisted gallery path for a named model."""
        return self.model_dir(name) / "gallery.npz"

    @property
    def head_weights_path(self) -> Path:
        """
        Trained head checkpoint for the *active* model (``model_name``).

        Resolves through the per-model layout so the single-model training/inference
        code paths keep working unchanged — they just write/read under
        ``artifacts/models/<model_name>/`` now.
        """
        return self.head_weights_path_for(self.model_name)

    @property
    def gallery_path(self) -> Path:
        """Persisted gallery for the active model (``model_name``)."""
        return self.gallery_path_for(self.model_name)


def get_settings(overrides: Optional[dict] = None) -> Settings:
    """
    Build a ``Settings`` instance, optionally patched with explicit overrides.

    Scripts use this rather than instantiating ``Settings`` directly so a single
    call site applies environment + override precedence consistently, which keeps
    test setups and CLI flags from each re-implementing that merge.
    """
    if overrides:
        return Settings(**overrides)
    return Settings()
