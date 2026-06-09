"""
ArcFace margin head — used only while training.

Open-set retrieval needs an embedding space where same-individual images cluster
tightly and different individuals sit far apart. A plain softmax classifier does
not enforce that; ArcFace (Deng et al., 2019) does, by adding an angular margin
between an embedding and its true class weight before the softmax. This pushes
classes apart on the unit sphere, which is exactly the geometry cosine retrieval
exploits at inference. It is the same loss MegaDescriptor trains with.

Critically this head is *discarded* after training: it is tied to the fixed set
of training identities, whereas inference must handle never-seen individuals via
retrieval. So nothing here is saved into the inference path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    Additive angular margin classifier over the training identities.

    Holds one weight vector per training identity. During the forward pass it
    measures the angle between an embedding and each class weight, adds a margin
    to the *true* class angle, and scales the cosines — producing logits for a
    standard cross-entropy loss that, minimised, enforces the angular separation.
    """

    def __init__(self, embedding_dim: int, num_identities: int, margin: float, scale: float):
        """
        Allocate the per-identity weights and store the margin/scale schedule.

        Precomputes the trig constants used to apply the margin so they are not
        recomputed every forward call.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_identities, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        self.scale = scale
        self.margin = margin
        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        # Threshold beyond which adding the margin would leave the monotonic
        # region of cosine; past it we fall back to a linear penalty for stability.
        self._threshold = math.cos(math.pi - margin)
        self._mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Produce margin-adjusted, scaled logits for cross-entropy.

        ``embeddings`` are already L2-normalised by the MIL head; we normalise the
        class weights here so the linear layer computes pure cosine similarities.
        The margin is applied only to the column of each sample's true label.
        """
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) via the angle-addition identity.
        phi = cosine * self._cos_m - sine * self._sin_m
        # Keep things monotonic when theta + m would exceed pi.
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(embeddings.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
