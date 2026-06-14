"""
Open-set partitioning of the dataset.

Open-set re-identification has a property closed-set classification does not:
the system must cope with individuals it never saw during training. To measure
that honestly, the data is partitioned along *two* axes:

    1. By identity — a fraction of individuals (``open_set_ratio``) is held out
       entirely as "unknowns". They never appear in training or the gallery, so
       at test time they probe whether the unknown-threshold correctly rejects
       strangers.

    2. Within the seen identities — their images are split into a training set,
       a gallery (the enrolled references), and a query set (held-out probes of
       known individuals used to measure rank-1/rank-5/mAP).

The split is computed natively here (deterministic given ``settings.seed``)
rather than via an external toolkit, so the open-set protocol is fully under our
control and has no extra dependencies. If you later want to match a published
benchmark's exact partition, this is the one place to swap in that split logic —
the output schema below would stay identical and the rest of the pipeline would
not change.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from ml.config import Settings
from ml.data.dataset import COL_IDENTITY
from ml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OpenSetSplit:
    """
    The four image groups produced by :func:`make_open_set_split`.

    ``train`` shapes the embedding space; ``gallery`` is enrolled as references;
    ``query_known`` measures retrieval on seen individuals; ``query_unknown``
    holds images of never-seen individuals to test rejection. They are disjoint
    by construction and carry the canonical dataframe columns.
    """

    train: pd.DataFrame
    gallery: pd.DataFrame
    query_known: pd.DataFrame
    query_unknown: pd.DataFrame

    def summary(self) -> str:
        """One-line size summary, logged so a run records exactly what it trained on."""
        return (
            f"train={len(self.train)} gallery={len(self.gallery)} "
            f"query_known={len(self.query_known)} query_unknown={len(self.query_unknown)}"
        )


def make_open_set_split(df: pd.DataFrame, settings: Settings) -> OpenSetSplit:
    """
    Partition the metadata dataframe into the open-set protocol described above.

    Deterministic given ``settings.seed`` so the same split (and therefore the
    same reported metrics) is reproduced on every run.
    """
    rng = np.random.default_rng(settings.seed)

    identities = np.array(sorted(df[COL_IDENTITY].unique()))
    rng.shuffle(identities)

    num_unknown = max(1, int(round(len(identities) * settings.open_set_ratio)))
    unknown_ids = set(identities[:num_unknown].tolist())
    known_ids = identities[num_unknown:].tolist()

    query_unknown = df[df[COL_IDENTITY].isin(unknown_ids)].reset_index(drop=True)

    train_rows: List[pd.DataFrame] = []
    gallery_rows: List[pd.DataFrame] = []
    query_known_rows: List[pd.DataFrame] = []

    # Per known individual, reserve one image for the gallery and one as a query
    # probe, and use the remainder for training. We split per-identity (rather
    # than globally) to guarantee every seen individual is represented in both
    # the gallery and the training set — a global split could starve an
    # individual of gallery references and make it un-retrievable for reasons
    # unrelated to model quality.
    for identity in known_ids:
        rows = df[df[COL_IDENTITY] == identity].sample(frac=1.0, random_state=settings.seed).reset_index(drop=True)
        if len(rows) < 3:
            # Too few images to also spare a query probe — keep it purely as a
            # gallery+train reference rather than dropping the individual.
            gallery_rows.append(rows.iloc[:1])
            train_rows.append(rows.iloc[1:])
            continue
        gallery_rows.append(rows.iloc[:1])
        query_known_rows.append(rows.iloc[1:2])
        train_rows.append(rows.iloc[2:])

    split = OpenSetSplit(
        train=_concat(train_rows),
        gallery=_concat(gallery_rows),
        query_known=_concat(query_known_rows),
        query_unknown=query_unknown,
    )
    logger.info(
        f"Open-set split | {len(known_ids)} known + {len(unknown_ids)} unknown identities | {split.summary()}"
    )
    return split


def _concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-identity frames into one, tolerating an empty group list."""
    if not frames:
        return pd.DataFrame(columns=[COL_IDENTITY])
    return pd.concat(frames, ignore_index=True)
