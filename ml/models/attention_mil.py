"""
The trainable core: gated-attention MIL pooling that turns a bag of patch
embeddings into one identity embedding.

This module *is* the paper's contribution. A bag holds K patch vectors but only
a few patches actually carry the animal's identity (a stripe junction, a notch
on a fin, a scar); the rest are background or generic texture. Gated attention
(Ilse, Tomczak & Welling, 2018, "Attention-based Deep Multiple Instance
Learning") learns a per-patch weight and pools the patches as a weighted sum —
so the model concentrates on the discriminative patches *without ever being told
which patches those are* (no patch-level labels). The learned weights double as
an explanation: reshaped to the patch grid they show where the model looked.

Only this head trains. With the backbone frozen it is a few small matrices, so
it fits and trains on CPU.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import PoolingType, Settings


class GatedAttentionPool(nn.Module):
    """
    Gated-attention pooling over a bag of instance embeddings.

    Implements the gated variant ``a_k = softmax(w^T (tanh(V h_k) ⊙ sigm(U h_k)))``.
    The sigmoid "gate" lets the network suppress patches that the tanh branch
    alone would over-weight, which empirically gives cleaner, more selective
    attention than plain (ungated) attention — important here because most
    patches are uninformative and we want them driven close to zero weight.
    """

    def __init__(self, feature_dim: int, attention_dim: int):
        """Set up the two attention branches (V, U) and the scoring vector w."""
        super().__init__()
        self.tanh_branch = nn.Linear(feature_dim, attention_dim)
        self.gate_branch = nn.Linear(feature_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool ``[B, K, feature_dim]`` instances into ``[B, feature_dim]``.

        Returns ``(pooled, attention)`` where attention is ``[B, K]`` and sums to
        one over the K patches. We return the attention deliberately: the
        explainability heatmap is just these weights laid back over the image, so
        recomputing them elsewhere would risk drift from what the model used.
        """
        gated = torch.tanh(self.tanh_branch(instances)) * torch.sigmoid(self.gate_branch(instances))
        scores = self.score(gated).squeeze(-1)  # [B, K]
        attention = F.softmax(scores, dim=1)
        pooled = torch.bmm(attention.unsqueeze(1), instances).squeeze(1)  # [B, feature_dim]
        return pooled, attention


class MeanPool(nn.Module):
    """
    Plain mean pooling — the ablation baseline.

    Exists so the evaluation can answer "does attention actually help, or would
    averaging the patches do just as well?" It returns a uniform attention vector
    so it is drop-in compatible with the attention head's interface.
    """

    def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average instances and report uniform attention for interface parity."""
        pooled = instances.mean(dim=1)
        k = instances.shape[1]
        attention = torch.full(
            size=(instances.shape[0], k),
            fill_value=1.0 / k,
            device=instances.device,
        )
        return pooled, attention


class MILEmbedder(nn.Module):
    """
    Full trainable head: pool a bag, then project to the identity embedding.

    Composition (pooling → projection MLP → L2-normalise) is the entire learned
    pipeline that sits on the frozen backbone. The L2 normalisation at the end is
    what makes cosine similarity at retrieval time equivalent to a dot product
    and keeps embedding magnitudes from skewing the unknown threshold.
    """

    def __init__(self, settings: Settings, feature_dim: int):
        """
        Build the head sized to the backbone's ``feature_dim``.

        ``feature_dim`` is passed in (from the backbone) rather than read from
        settings so the head always matches the actual backbone output, even if
        the configured value drifts.
        """
        super().__init__()
        self.settings = settings

        if settings.pooling is PoolingType.GATED_ATTENTION:
            self.pool: nn.Module = GatedAttentionPool(
                feature_dim=feature_dim, attention_dim=settings.attention_dim
            )
        else:
            self.pool = MeanPool()

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, settings.embedding_dim),
        )

    def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map a bag ``[B, K, feature_dim]`` to ``(embedding [B, embedding_dim],
        attention [B, K])``.

        The embedding is L2-normalised so downstream cosine retrieval and the
        ArcFace loss both operate on the unit sphere.
        """
        pooled, attention = self.pool(instances)
        embedding = self.projection(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding, attention
