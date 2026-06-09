"""
Metrics for re-identification quality.

Two distinct questions get two distinct metric families, because a model can be
good at one and bad at the other:

    - *Closed-set retrieval quality* (rank-1, rank-5, mAP): when a query really is
      a known individual, does the right one rank at/near the top? This measures
      embedding discriminativeness.

    - *Open-set rejection* (AUROC): can the top-match similarity separate known
      individuals from never-seen ones? This measures whether the unknown
      threshold has anything to work with, independent of where it is set.

Keeping these as small pure functions (numpy in, float out) makes them trivial to
unit-test and reuse from both the evaluation driver and any notebook.
"""

from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score


def rank_k_accuracy(ranked_correct: List[List[bool]], k: int) -> float:
    """
    Fraction of queries whose correct individual appears within the top ``k``.

    ``ranked_correct[i]`` is the per-query list of "is this ranked candidate the
    true individual?" booleans, already ordered best-first. Rank-1 and rank-5 are
    just this with k=1 and k=5.
    """
    if not ranked_correct:
        return 0.0
    hits = sum(1 for row in ranked_correct if any(row[:k]))
    return hits / len(ranked_correct)


def mean_average_precision(ranked_correct: List[List[bool]]) -> float:
    """
    Mean average precision over all known-individual queries.

    Unlike rank-k, mAP rewards getting *all* correct gallery matches high in the
    ranking, not just the first — a fuller picture of retrieval quality when an
    individual has several gallery references. Averages per-query AP over queries
    that have at least one correct match.
    """
    if not ranked_correct:
        return 0.0

    average_precisions: List[float] = []
    for row in ranked_correct:
        num_correct = 0
        precision_sum = 0.0
        for rank, is_correct in enumerate(row, start=1):
            if is_correct:
                num_correct += 1
                precision_sum += num_correct / rank
        if num_correct > 0:
            average_precisions.append(precision_sum / num_correct)

    if not average_precisions:
        return 0.0
    return float(np.mean(average_precisions))


def open_set_auroc(known_top_scores: List[float], unknown_top_scores: List[float]) -> float:
    """
    AUROC for separating known vs unknown queries by their top-match similarity.

    Known queries should score high (they have a real match in the gallery),
    unknown ones low. AUROC summarises that separability across all possible
    thresholds, so it judges the *signal* without committing to a particular
    ``unknown_threshold``. Returns 0.5 (chance) when either group is empty.
    """
    if not known_top_scores or not unknown_top_scores:
        return 0.5
    labels = [1] * len(known_top_scores) + [0] * len(unknown_top_scores)
    scores = known_top_scores + unknown_top_scores
    return float(roc_auc_score(y_true=labels, y_score=scores))
