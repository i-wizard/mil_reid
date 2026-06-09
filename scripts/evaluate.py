"""
CLI: evaluate the trained model under the open-set protocol.

Prints and archives rank-1/rank-5/mAP and open-set AUROC. Run the same command
with ``REID_POOLING=MEAN`` (after training a mean-pooling model) to produce the
ablation baseline the paper compares against.

Usage:
    python -m scripts.evaluate
"""

from ml.config import get_settings
from ml.eval.evaluate import run_evaluation


def main() -> None:
    """Evaluate with the active settings and write the JSON report."""
    run_evaluation(settings=get_settings())


if __name__ == "__main__":
    main()
