"""
Core-logic verification for the ML pipeline using only synthetic data.

This deliberately avoids datasets, the pretrained backbone, and the network: it
fabricates patch-embedding tensors and tiny dataframes so the *trainable and
algorithmic* parts (MIL pooling, ArcFace, the loss, retrieval/threshold logic,
metrics, splitting, heatmap rendering) can be checked in seconds on CPU. It
exercises the paper's contribution without the slow I/O around it.

Run directly:  python -m tests.test_core_logic
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.config import PoolingType, get_settings
from ml.data.dataset import COL_BBOX, COL_IDENTITY, COL_IMAGE_ID, COL_PATH
from ml.data.patches import attention_to_grid
from ml.data.splits import make_open_set_split
from ml.eval.metrics import mean_average_precision, open_set_auroc, rank_k_accuracy
from ml.inference.explain import _grid_to_heat_rgb
from ml.inference.gallery import Gallery
from ml.inference.identifier import Candidate, IdentifyResult, Identifier
from ml.models.arcface import ArcFaceHead
from ml.models.attention_mil import MILEmbedder
from ml.training.dataset import LabelEncoder
from ml.training.losses import identity_loss

FEATURE_DIM = 64
NUM_PATCHES = 16  # 4x4 grid
BATCH = 8
NUM_IDS = 5


def test_gated_attention_embedder_shapes_and_normalisation() -> None:
    """MIL head must emit unit-norm embeddings and attention that sums to 1 per bag."""
    settings = get_settings({"embedding_dim": 32, "attention_dim": 16, "patch_grid": 4})
    head = MILEmbedder(settings=settings, feature_dim=FEATURE_DIM)

    bags = torch.randn(BATCH, NUM_PATCHES, FEATURE_DIM)
    embedding, attention = head(bags)

    assert embedding.shape == (BATCH, 32), embedding.shape
    assert attention.shape == (BATCH, NUM_PATCHES), attention.shape
    assert torch.allclose(attention.sum(dim=1), torch.ones(BATCH), atol=1e-5), "attention must sum to 1"
    norms = embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), "embeddings must be L2-normalised"
    print("ok: gated-attention embedder shapes + normalisation")


def test_mean_pool_baseline_is_uniform() -> None:
    """The ablation baseline must report uniform attention (interface parity)."""
    settings = get_settings({"embedding_dim": 32, "pooling": PoolingType.MEAN, "patch_grid": 4})
    head = MILEmbedder(settings=settings, feature_dim=FEATURE_DIM)

    bags = torch.randn(BATCH, NUM_PATCHES, FEATURE_DIM)
    _, attention = head(bags)
    expected = torch.full((BATCH, NUM_PATCHES), 1.0 / NUM_PATCHES)
    assert torch.allclose(attention, expected, atol=1e-6), "mean pool must give uniform attention"
    print("ok: mean-pool baseline uniform attention")


def test_arcface_logits_and_training_step_reduces_loss() -> None:
    """ArcFace + loss must produce valid logits and a single step must lower the loss."""
    settings = get_settings({"embedding_dim": 32, "attention_dim": 16, "patch_grid": 4})
    head = MILEmbedder(settings=settings, feature_dim=FEATURE_DIM)
    arc = ArcFaceHead(embedding_dim=32, num_identities=NUM_IDS, margin=0.5, scale=30.0)

    bags = torch.randn(BATCH, NUM_PATCHES, FEATURE_DIM)
    labels = torch.randint(low=0, high=NUM_IDS, size=(BATCH,))

    optimizer = torch.optim.AdamW(list(head.parameters()) + list(arc.parameters()), lr=1e-2)

    emb, _ = head(bags)
    logits = arc(embeddings=emb, labels=labels)
    assert logits.shape == (BATCH, NUM_IDS), logits.shape
    loss_before = identity_loss(logits=logits, labels=labels)

    # Overfit a few steps on the same batch — loss must fall, proving grads flow
    # through the head (and the frozen-backbone assumption doesn't block learning).
    for _ in range(20):
        optimizer.zero_grad()
        emb, _ = head(bags)
        loss = identity_loss(logits=arc(embeddings=emb, labels=labels), labels=labels)
        loss.backward()
        optimizer.step()

    assert loss.item() < loss_before.item(), (loss_before.item(), loss.item())
    print(f"ok: ArcFace training step reduces loss ({loss_before.item():.3f} -> {loss.item():.3f})")


def test_retrieval_metrics_known_values() -> None:
    """Metric functions must match hand-computed values on tiny fixed inputs."""
    # Query A: correct at rank 1; Query B: correct at rank 3.
    ranked = [[True, False, False], [False, False, True]]
    assert rank_k_accuracy(ranked_correct=ranked, k=1) == 0.5
    assert rank_k_accuracy(ranked_correct=ranked, k=5) == 1.0
    # mAP = mean(1/1, 1/3) = (1 + 0.333...) / 2
    assert abs(mean_average_precision(ranked_correct=ranked) - (1.0 + 1.0 / 3.0) / 2.0) < 1e-9
    # Perfectly separable known(high) vs unknown(low) → AUROC 1.0
    assert open_set_auroc(known_top_scores=[0.9, 0.8], unknown_top_scores=[0.2, 0.1]) == 1.0
    print("ok: retrieval + open-set metrics match hand-computed values")


def test_open_set_split_is_disjoint_by_identity() -> None:
    """Unknown identities must never leak into train/gallery/known-query."""
    rows = []
    for ident in range(10):
        for shot in range(4):
            rows.append(
                {COL_IMAGE_ID: f"{ident}_{shot}", COL_IDENTITY: f"id{ident}", COL_PATH: "x", COL_BBOX: None}
            )
    df = pd.DataFrame(rows)
    settings = get_settings({"open_set_ratio": 0.3, "seed": 1})
    split = make_open_set_split(df=df, settings=settings)

    known_ids = set(split.train[COL_IDENTITY]) | set(split.gallery[COL_IDENTITY]) | set(
        split.query_known[COL_IDENTITY]
    )
    unknown_ids = set(split.query_unknown[COL_IDENTITY])
    assert known_ids.isdisjoint(unknown_ids), "open-set leak: an identity is both known and unknown"
    assert len(unknown_ids) >= 1
    print(f"ok: open-set split disjoint ({len(known_ids)} known, {len(unknown_ids)} unknown ids)")


def test_label_encoder_roundtrip() -> None:
    """Encoding then decoding an identity must be the identity function."""
    encoder = LabelEncoder(identities=["zebra_b", "zebra_a", "zebra_a", "zebra_c"])
    assert len(encoder) == 3
    for name in ["zebra_a", "zebra_b", "zebra_c"]:
        assert encoder.decode(encoder.encode(name)) == name
    print("ok: label encoder round-trip")


class _FakeEmbedder:
    """
    Stand-in embedder returning preset unit vectors by path.

    Lets the gallery/identifier retrieval logic be tested without a backbone:
    we control exactly which vectors exist, so the cosine ranking and unknown
    threshold have a known correct answer.
    """

    def __init__(self, vectors):
        self._vectors = vectors

    def embed_path(self, path, bbox=None):
        from ml.inference.embedder import EmbedResult

        v = self._vectors[path]
        return EmbedResult(embedding=v, attention=np.ones(NUM_PATCHES) / NUM_PATCHES, coords=[], grid=4)


def _unit(vec):
    """Normalise a small vector for use as a fake embedding."""
    arr = np.array(vec, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-12)


def test_identifier_ranks_correct_individual_and_flags_unknown() -> None:
    """Identifier must rank the true match first and reject a dissimilar query."""
    a = _unit([1, 0, 0])
    b = _unit([0, 1, 0])
    query_known = _unit([0.95, 0.05, 0])  # close to A
    query_unknown = _unit([0, 0, 1])  # orthogonal to both → should be unknown

    embedder = _FakeEmbedder(
        {"a.jpg": a, "b.jpg": b, "qk.jpg": query_known, "qu.jpg": query_unknown}
    )
    gallery = Gallery()
    gallery.enroll(individual_id="A", image_paths=["a.jpg"], embedder=embedder)
    gallery.enroll(individual_id="B", image_paths=["b.jpg"], embedder=embedder)

    settings = get_settings({"top_k": 2, "unknown_threshold": 0.5})
    identifier = Identifier(embedder=embedder, gallery=gallery, settings=settings)

    known = identifier.identify(image_path="qk.jpg")
    assert known.candidates[0].individual_id == "A", known.candidates
    assert not known.is_unknown

    unknown = identifier.identify(image_path="qu.jpg")
    assert unknown.is_unknown, unknown.candidates
    print("ok: identifier ranks correct id + flags unknown via threshold")


def test_gallery_save_load_roundtrip() -> None:
    """A saved gallery must reload to identical prototypes (API will reload it)."""
    embedder = _FakeEmbedder({"a.jpg": _unit([1, 0, 0]), "b.jpg": _unit([0, 1, 0])})
    gallery = Gallery()
    gallery.enroll(individual_id="A", image_paths=["a.jpg"], embedder=embedder)
    gallery.enroll(individual_id="B", image_paths=["b.jpg"], embedder=embedder)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gallery.npz"
        gallery.save(path)
        reloaded = Gallery.load(path)

    ids_a, mat_a = gallery.as_matrix()
    ids_b, mat_b = reloaded.as_matrix()
    assert ids_a == ids_b
    assert np.allclose(mat_a, mat_b)
    print("ok: gallery save/load round-trip")


def test_attention_heatmap_render() -> None:
    """Attention vector must reshape to the grid and render to a full-size RGB image."""
    attention = np.linspace(0, 1, NUM_PATCHES).astype(np.float32)
    grid = attention_to_grid(attention=attention, grid=4)
    assert grid.shape == (4, 4)

    heat = _grid_to_heat_rgb(grid_heat=grid, size=224)
    assert heat.size == (224, 224)
    assert heat.mode == "RGB"
    print("ok: attention → grid → upsampled heatmap render")


def main() -> None:
    """Run every check and report a single pass/fail summary."""
    tests = [
        test_gated_attention_embedder_shapes_and_normalisation,
        test_mean_pool_baseline_is_uniform,
        test_arcface_logits_and_training_step_reduces_loss,
        test_retrieval_metrics_known_values,
        test_open_set_split_is_disjoint_by_identity,
        test_label_encoder_roundtrip,
        test_identifier_ranks_correct_individual_and_flags_unknown,
        test_gallery_save_load_roundtrip,
        test_attention_heatmap_render,
    ]
    for test in tests:
        test()
    print(f"\nALL {len(tests)} CORE-LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
