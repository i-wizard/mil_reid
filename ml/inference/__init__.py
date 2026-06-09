"""
Inference layer — the public surface of the ML core.

This is the *only* subpackage the FastAPI app (Part 2) should import. It exposes
four capabilities, each in its own module:

    - ``embedder``   : image  → identity embedding (+ attention)
    - ``gallery``    : enroll/persist reference embeddings for known individuals
    - ``identifier`` : open-set retrieval — query image → ranked candidates / unknown
    - ``explain``    : attention weights → heatmap overlay

Keeping the whole inference contract here means no model internals leak into the
API layer; the API depends on these typed functions and nothing deeper.
"""

from ml.inference.embedder import Embedder, EmbedResult, load_embedder
from ml.inference.gallery import Gallery
from ml.inference.identifier import Candidate, IdentifyResult, Identifier

__all__ = [
    "Embedder",
    "EmbedResult",
    "load_embedder",
    "Gallery",
    "Identifier",
    "IdentifyResult",
    "Candidate",
]
