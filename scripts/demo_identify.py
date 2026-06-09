"""
CLI: an end-to-end inference smoke test / demo of the ML core.

Builds the gallery from the held-out gallery split, then identifies one *known*
query (expecting a correct top-1) and one *unknown* query (expecting an
``is_unknown`` verdict), and writes an attention-heatmap overlay for the known
query. This exercises the exact path the FastAPI layer will call and produces the
visual the demo/paper showcases — without needing the web stack.

Usage:
    python -m scripts.demo_identify
"""

from ml.config import get_settings
from ml.data.dataset import COL_IDENTITY, COL_PATH, load_dataset
from ml.data.splits import make_open_set_split
from ml.eval.evaluate import _build_gallery
from ml.inference.embedder import load_embedder
from ml.inference.explain import save_attention_overlay
from ml.inference.identifier import Identifier
from ml.utils.logging import get_logger

logger = get_logger("scripts.demo_identify")


def main() -> None:
    """Run a known + unknown identification and save a heatmap for the known query."""
    settings = get_settings()
    bundle = load_dataset(settings=settings)
    split = make_open_set_split(df=bundle.df, settings=settings)

    embedder = load_embedder(settings=settings)
    gallery = _build_gallery(split=split, embedder=embedder)
    identifier = Identifier(embedder=embedder, gallery=gallery, settings=settings)

    # Known query: a held-out probe of an individual that IS enrolled.
    known_row = split.query_known.iloc[0]
    known_result = identifier.identify(image_path=known_row[COL_PATH])
    top = known_result.candidates[0] if known_result.candidates else None
    logger.info(
        f"KNOWN query truth='{known_row[COL_IDENTITY]}' -> "
        f"top='{top.individual_id if top else None}' score={top.score if top else None:.3f} "
        f"is_unknown={known_result.is_unknown}"
    )

    overlay_path = settings.artifacts_root / "demo_attention_overlay.png"
    save_attention_overlay(
        image_path=known_row[COL_PATH],
        embed_result=known_result.embed_result,
        settings=settings,
        out_path=overlay_path,
    )
    logger.info(f"Saved attention heatmap to {overlay_path}.")

    # Unknown query: an individual that was held out of training and the gallery.
    if len(split.query_unknown) > 0:
        unknown_row = split.query_unknown.iloc[0]
        unknown_result = identifier.identify(image_path=unknown_row[COL_PATH])
        top_u = unknown_result.candidates[0] if unknown_result.candidates else None
        logger.info(
            f"UNKNOWN query truth='{unknown_row[COL_IDENTITY]}' -> "
            f"top='{top_u.individual_id if top_u else None}' "
            f"score={top_u.score if top_u else None} is_unknown={unknown_result.is_unknown}"
        )


if __name__ == "__main__":
    main()
