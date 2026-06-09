"""
CLI: train the MIL head on the cached features.

Thin wrapper around ``ml.training.train.run_training`` so the training entrypoint
is a one-liner and all logic stays in the library (where it is testable and
reusable by the API later).

Usage:
    python -m scripts.train
"""

from ml.config import get_settings
from ml.training.train import run_training


def main() -> None:
    """Run training with the active settings (env-overridable via REID_*)."""
    run_training(settings=get_settings())


if __name__ == "__main__":
    main()
