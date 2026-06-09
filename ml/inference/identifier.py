"""
Open-set identification: query image → ranked candidates, or "unknown".

This ties the pieces together into the capability the demo actually shows. A
query image is embedded, compared by cosine similarity against every gallery
prototype, and the top matches are returned ranked. The open-set twist is the
threshold: if even the best match is below ``unknown_threshold``, the query is
reported as an unseen individual rather than forced onto the nearest known one —
which is the behaviour that makes the system honest about animals it has never
encountered.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ml.config import Settings
from ml.inference.embedder import EmbedResult, Embedder
from ml.inference.gallery import Gallery


@dataclass
class Candidate:
    """A single ranked match: which individual and how similar (cosine, 0..1-ish)."""

    individual_id: str
    score: float


@dataclass
class IdentifyResult:
    """
    The full outcome of one identification.

    ``candidates`` is the ranked top-k; ``is_unknown`` is the open-set verdict;
    ``embed_result`` is carried along so the caller can render the attention
    heatmap from the very same forward pass (no recomputation).
    """

    candidates: List[Candidate]
    is_unknown: bool
    embed_result: EmbedResult


class Identifier:
    """
    Runs query embedding + cosine retrieval + the unknown decision.

    Holds a gallery and an embedder; constructed once and reused per query so the
    backbone and head are loaded a single time.
    """

    def __init__(self, embedder: Embedder, gallery: Gallery, settings: Settings):
        """Wire together the embedder, the gallery to search, and the thresholds."""
        self.embedder = embedder
        self.gallery = gallery
        self.settings = settings

    def identify(self, image_path: str, bbox=None) -> IdentifyResult:
        """
        Identify the individual in ``image_path`` against the gallery.

        Returns the top-``k`` candidates by cosine similarity and flags the result
        as unknown when the best score falls below the configured threshold. The
        embedding is L2-normalised and so are the gallery prototypes, so the dot
        product *is* cosine similarity — no extra normalisation needed here.
        """
        embed_result = self.embedder.embed_path(path=image_path, bbox=bbox)
        ids, matrix = self.gallery.as_matrix()

        if not ids:
            # An empty gallery can match nothing — everything is, by definition,
            # an unseen individual.
            return IdentifyResult(candidates=[], is_unknown=True, embed_result=embed_result)

        scores = matrix @ embed_result.embedding  # cosine similarity per individual
        order = np.argsort(scores)[::-1][: self.settings.top_k]
        candidates = [Candidate(individual_id=ids[i], score=float(scores[i])) for i in order]

        is_unknown = candidates[0].score < self.settings.unknown_threshold
        return IdentifyResult(candidates=candidates, is_unknown=is_unknown, embed_result=embed_result)
