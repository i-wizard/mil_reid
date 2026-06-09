"""
Training objective.

The primary loss is cross-entropy over ArcFace's margin-adjusted logits. We wrap
it in a named function rather than calling ``F.cross_entropy`` inline so the
training loop reads at the level of intent ("identity loss") and so an alternate
objective (e.g. an added triplet term for ablation) can be slotted in here
without touching the loop.
"""

import torch
import torch.nn.functional as F


def identity_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy between ArcFace logits and the true identity indices.

    Minimising this drives each embedding toward its own identity's class weight
    by more than the angular margin, which is what carves out the well-separated
    embedding space that open-set cosine retrieval depends on.
    """
    return F.cross_entropy(logits, labels)
