"""
A torch ``Dataset`` that serves *cached* patch embeddings, not raw images.

Training reads from the precomputed feature cache rather than decoding images
and running the backbone every epoch. That is the whole reason CPU training is
feasible: each item is a small ``[K, feature_dim]`` array load plus an integer
label. This class also owns the identity→index encoding, because ArcFace needs
contiguous class indices while the dataset uses arbitrary identity strings.
"""

from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.config import Settings
from ml.data.dataset import COL_IDENTITY, COL_IMAGE_ID
from ml.features.cache import load_cached_features


class LabelEncoder:
    """
    Bidirectional map between identity strings and contiguous class indices.

    Kept as its own object (and saved alongside the trained head) so inference
    and evaluation can translate model class indices back to human identity
    labels without re-deriving the ordering — which must stay fixed or the
    saved ArcFace weights would point at the wrong identities.
    """

    def __init__(self, identities: List[str]):
        """Build the mapping from a list of unique identities in a fixed sorted order."""
        ordered = sorted(set(identities))
        self.identity_to_index: Dict[str, int] = {name: i for i, name in enumerate(ordered)}
        self.index_to_identity: List[str] = ordered

    def __len__(self) -> int:
        """Number of distinct identities — i.e. the ArcFace classifier width."""
        return len(self.index_to_identity)

    def encode(self, identity: str) -> int:
        """Identity string → class index."""
        return self.identity_to_index[identity]

    def decode(self, index: int) -> str:
        """Class index → identity string."""
        return self.index_to_identity[index]


class CachedBagDataset(Dataset):
    """
    Serves ``(bag_features, label_index)`` pairs from the feature cache.

    Only images whose identity is in the encoder are kept; this lets the same
    class serve a training subset whose label space is exactly the ArcFace head's
    output space, with no out-of-range labels sneaking in.
    """

    def __init__(self, df: pd.DataFrame, encoder: LabelEncoder, settings: Settings):
        """Filter the dataframe to known identities and hold rows for indexing."""
        super().__init__()
        self.settings = settings
        self.encoder = encoder
        self.rows = df[df[COL_IDENTITY].isin(encoder.identity_to_index)].reset_index(drop=True)

    def __len__(self) -> int:
        """Number of usable training images."""
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Load one image's cached bag and its encoded label.

        The bag tensor is ``[K, feature_dim]``; the default collate stacks these
        into ``[B, K, feature_dim]`` for the MIL head.
        """
        row = self.rows.iloc[index]
        features = load_cached_features(image_id=str(row[COL_IMAGE_ID]), settings=self.settings)
        label = self.encoder.encode(row[COL_IDENTITY])
        return features, label
