"""
End-to-end evaluation driver.

Reconstructs the exact open-set protocol used in training, builds the gallery
from the gallery split, then runs every held-out query through the identifier and
scores the results. It reports closed-set retrieval metrics on known-individual
queries and open-set AUROC using the unknown-individual queries — the full
picture the paper needs in one report.

The driver also writes the report to disk so a run's numbers are archived
alongside the checkpoint that produced them.
"""

import json
from dataclasses import asdict, dataclass
from typing import List

from ml.config import Settings
from ml.data.dataset import COL_IDENTITY, load_dataset
from ml.data.splits import make_open_set_split
from ml.eval.metrics import mean_average_precision, open_set_auroc, rank_k_accuracy
from ml.inference.embedder import load_embedder
from ml.inference.gallery import Gallery
from ml.inference.identifier import Identifier
from ml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """
    The headline numbers from one evaluation run.

    Grouped in a dataclass so the same object can be logged, JSON-serialised, and
    (later) returned by an API endpoint without re-shaping.
    """

    pooling: str
    num_gallery_individuals: int
    num_known_queries: int
    num_unknown_queries: int
    rank_1: float
    rank_5: float
    mean_ap: float
    open_set_auroc: float


def run_evaluation(settings: Settings) -> EvaluationReport:
    """
    Evaluate the trained model under the open-set protocol and persist a report.

    Mirrors training's split (same seed → same partition) so gallery/query images
    are exactly those held out of training. Builds the gallery, identifies every
    query, and computes the metric suite.
    """
    bundle = load_dataset(settings=settings)
    split = make_open_set_split(df=bundle.df, settings=settings)

    embedder = load_embedder(settings=settings)
    gallery = _build_gallery(split=split, embedder=embedder)

    identifier = Identifier(embedder=embedder, gallery=gallery, settings=settings)

    ranked_correct, known_top_scores = _score_known_queries(
        split=split, identifier=identifier
    )
    unknown_top_scores = _score_unknown_queries(split=split, identifier=identifier)

    report = EvaluationReport(
        pooling=settings.pooling.value,
        num_gallery_individuals=len(gallery.individuals),
        num_known_queries=len(ranked_correct),
        num_unknown_queries=len(unknown_top_scores),
        rank_1=rank_k_accuracy(ranked_correct=ranked_correct, k=1),
        rank_5=rank_k_accuracy(ranked_correct=ranked_correct, k=5),
        mean_ap=mean_average_precision(ranked_correct=ranked_correct),
        open_set_auroc=open_set_auroc(
            known_top_scores=known_top_scores, unknown_top_scores=unknown_top_scores
        ),
    )

    _persist(report=report, settings=settings)
    logger.info(
        f"Eval [{report.pooling}] rank-1={report.rank_1:.3f} rank-5={report.rank_5:.3f} "
        f"mAP={report.mean_ap:.3f} open-set AUROC={report.open_set_auroc:.3f}"
    )
    return report


def _build_gallery(split, embedder) -> Gallery:
    """
    Enroll every known individual from its gallery-split images.

    One prototype per individual, built from the reference images reserved by the
    split — this is the set queries are matched against.
    """
    gallery = Gallery()
    for identity, rows in split.gallery.groupby(COL_IDENTITY):
        gallery.enroll(individual_id=str(identity), image_paths=rows["path"].tolist(), embedder=embedder)
    return gallery


def _score_known_queries(split, identifier: Identifier):
    """
    Run known-individual queries and collect per-query correctness + top scores.

    For each query we record whether each ranked candidate is the true individual
    (feeding rank-k and mAP) and the top similarity (feeding open-set AUROC as the
    positive class).
    """
    ranked_correct: List[List[bool]] = []
    known_top_scores: List[float] = []

    for _, row in split.query_known.iterrows():
        result = identifier.identify(image_path=row["path"])
        truth = str(row[COL_IDENTITY])
        ranked_correct.append([c.individual_id == truth for c in result.candidates])
        if result.candidates:
            known_top_scores.append(result.candidates[0].score)

    return ranked_correct, known_top_scores


def _score_unknown_queries(split, identifier: Identifier) -> List[float]:
    """
    Run never-seen-individual queries and collect their top similarity scores.

    These form the negative class for open-set AUROC: ideally their best match is
    weak, so they score lower than genuine known queries.
    """
    unknown_top_scores: List[float] = []
    for _, row in split.query_unknown.iterrows():
        result = identifier.identify(image_path=row["path"])
        if result.candidates:
            unknown_top_scores.append(result.candidates[0].score)
    return unknown_top_scores


def _persist(report: EvaluationReport, settings: Settings) -> None:
    """Write the report as JSON next to the checkpoint for archival."""
    out_path = settings.artifacts_root / f"eval_report_{report.pooling.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2))
    logger.info(f"Wrote evaluation report to {out_path}.")
