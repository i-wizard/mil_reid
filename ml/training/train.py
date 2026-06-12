"""
The training loop for the MIL head.

What trains here is *only* the gated-attention pooling + projection head plus the
throwaway ArcFace classifier — the backbone stays frozen and is not even invoked
(we read its outputs from the feature cache). The loop therefore optimises a few
small matrices over cached vectors, which is why it runs on CPU.

Validation uses a held-out slice of the training identities and measures nearest
-class accuracy against the ArcFace prototypes — a self-contained proxy for
rank-1 retrieval that needs no gallery, so training can report progress without
depending on the inference layer.

On completion it saves a single checkpoint containing the head weights, the label
encoder, and the geometry needed to rebuild the head — everything inference needs
and nothing it does not (the ArcFace weights are intentionally excluded).
"""

from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from ml.config import Settings
from ml.data.dataset import COL_IDENTITY, load_dataset
from ml.data.splits import make_open_set_split
from ml.features.backbone import build_backbone
from ml.models.arcface import ArcFaceHead
from ml.models.attention_mil import MILEmbedder
from ml.training.dataset import CachedBagDataset, LabelEncoder
from ml.training.losses import identity_loss
from ml.utils.logging import get_logger
from ml.utils.seed import seed_everything

logger = get_logger(__name__)


def run_training(
    settings: Settings,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> None:
    """
    Train the MIL head end to end and persist the inference checkpoint.

    Assumes features for the training split are already cached; it deliberately
    does not extract them here so the slow backbone pass stays a separate,
    resumable step.
    """
    seed_everything(settings.seed)
    settings.artifacts_root.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(settings=settings)
    split = make_open_set_split(df=bundle.df, settings=settings)

    encoder = LabelEncoder(identities=split.train[COL_IDENTITY].tolist())
    logger.info(f"Training over {len(encoder)} known identities.")

    feature_dim = _resolve_feature_dim(settings=settings)
    train_loader, val_loader = _build_loaders(split_train=split, settings=settings, encoder=encoder)

    embedder = MILEmbedder(settings=settings, feature_dim=feature_dim)
    arcface = ArcFaceHead(
        embedding_dim=settings.embedding_dim,
        num_identities=len(encoder),
        margin=settings.arcface_margin,
        scale=settings.arcface_scale,
    )

    # Both the head and the ArcFace prototypes train; the backbone does not, so
    # only these parameters are handed to the optimizer.
    optimizer = torch.optim.AdamW(
        params=list(embedder.parameters()) + list(arcface.parameters()),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    best_val_acc = 0.0
    for epoch in range(1, settings.epochs + 1):
        train_loss = _train_one_epoch(
            embedder=embedder, arcface=arcface, loader=train_loader, optimizer=optimizer
        )
        val_acc = _validate(embedder=embedder, arcface=arcface, loader=val_loader)
        logger.info(f"Epoch {epoch:3d}/{settings.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        # Stream per-epoch progress to callers (e.g. a background job). The callback
        # may raise to abort cooperatively (job cancellation) — checked here, between
        # epochs, so a cancelled run stops at a clean boundary.
        if progress_callback is not None:
            progress_callback(epoch, settings.epochs, train_loss, val_acc)

        # Keep the checkpoint from the best-validating epoch rather than the last,
        # so a late-epoch overfit does not overwrite a better model.
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(embedder=embedder, encoder=encoder, feature_dim=feature_dim, settings=settings)

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}. Head saved to {settings.head_weights_path}.")


def _resolve_feature_dim(settings: Settings) -> int:
    """
    Determine the backbone's true output width.

    We instantiate the backbone briefly (not for inference, just to read
    ``feature_dim``) so the head is sized to what the cache actually contains,
    even if ``settings.feature_dim`` was left at a stale default.
    """
    backbone = build_backbone(settings=settings)
    return backbone.feature_dim


def _build_loaders(
    split_train, settings: Settings, encoder: LabelEncoder
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val dataloaders from the training split.

    The training split is further divided 90/10 into train/val by image, giving
    an in-distribution validation signal during training without touching the
    gallery/query images reserved for final evaluation.
    """
    full = CachedBagDataset(df=split_train.train, encoder=encoder, settings=settings)
    val_size = max(1, int(0.1 * len(full)))
    train_size = len(full) - val_size
    generator = torch.Generator().manual_seed(settings.seed)
    train_ds, val_ds = random_split(full, lengths=[train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    return train_loader, val_loader


def _train_one_epoch(embedder: MILEmbedder, arcface: ArcFaceHead, loader: DataLoader, optimizer) -> float:
    """Run one optimisation pass over the training data; return mean batch loss."""
    embedder.train()
    arcface.train()
    running = 0.0
    batches = 0
    for features, labels in loader:
        optimizer.zero_grad()
        embeddings, _ = embedder(features)
        logits = arcface(embeddings=embeddings, labels=labels)
        loss = identity_loss(logits=logits, labels=labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
        batches += 1
    return running / max(1, batches)


@torch.inference_mode()
def _validate(embedder: MILEmbedder, arcface: ArcFaceHead, loader: DataLoader) -> float:
    """
    Nearest-class accuracy on the validation slice.

    Compares each embedding against the ArcFace prototypes by cosine similarity
    and counts an argmax match as correct — a margin-free proxy for rank-1 that
    tracks whether the embedding space is separating identities.
    """
    embedder.eval()
    arcface.eval()
    prototypes = torch.nn.functional.normalize(arcface.weight, dim=1)

    correct = 0
    total = 0
    for features, labels in loader:
        embeddings, _ = embedder(features)
        cosine = torch.nn.functional.linear(embeddings, prototypes)
        predicted = cosine.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.shape[0]
    return correct / max(1, total)


def _save_checkpoint(
    embedder: MILEmbedder, encoder: LabelEncoder, feature_dim: int, settings: Settings
) -> None:
    """
    Persist exactly what inference needs to rebuild the embedder.

    Stores the head's state dict, the label encoder ordering, the backbone
    feature width, and a snapshot of the settings — but not the ArcFace head,
    which has no role at inference. Saving the geometry alongside the weights
    means inference never has to guess how the checkpoint was shaped.
    """
    # Each model is namespaced under artifacts/models/<model_name>/, which may not
    # exist yet on a first train — create it before saving.
    settings.head_weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": embedder.state_dict(),
            "index_to_identity": encoder.index_to_identity,
            "feature_dim": feature_dim,
            "settings": asdict_safe(settings),
        },
        settings.head_weights_path,
    )


def asdict_safe(settings: Settings) -> dict:
    """
    Serialise settings to a plain dict with enums/paths as primitives.

    Pydantic's ``model_dump`` keeps enums/Paths as objects that ``torch.save``'s
    pickling can choke on across environments; dumping in JSON mode yields plain
    strings so the checkpoint stays portable into the (later) API container.
    """
    return settings.model_dump(mode="json")
