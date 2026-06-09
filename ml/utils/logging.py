"""
Logging helper for the ML core.

A single configured logger factory exists so that every module logs in the same
format and the pipeline's flow (downloads, cache hits, epoch metrics, retrieval
decisions) is reconstructable from stdout alone — important because most of this
runs as long, unattended scripts where logs are the only window into progress.
"""

import logging
import sys
from typing import Optional

_CONFIGURED = False


def _configure_root() -> None:
    """
    Attach a stdout handler with a consistent format exactly once.

    Guarded by a module flag because repeated ``get_logger`` calls (one per
    module) would otherwise stack duplicate handlers and print every line N
    times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger("ml")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a namespaced child logger under the shared ``ml`` root.

    Callers pass ``__name__`` so log lines identify their source module, while
    all of them inherit the single handler/format configured here.
    """
    _configure_root()
    return logging.getLogger(name if name else "ml")
