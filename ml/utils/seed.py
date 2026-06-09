"""
Reproducibility helper.

Re-identification results (which identity ranks first, where the unknown
threshold lands) are sensitive to the random train/val/test partition and to
weight initialisation. Seeding every RNG from one place means a reported number
can be reproduced exactly, which matters when the figures go into a paper.
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and Torch RNGs so a run is byte-for-byte repeatable.

    Called at the top of each script rather than implicitly, so it is explicit
    in the logs which seed produced a given checkpoint or evaluation report.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
