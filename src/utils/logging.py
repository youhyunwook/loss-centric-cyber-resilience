"""
src/utils/logging.py
--------------------
Centralised logger factory so every module logs in a consistent format.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "cyber_resilience", level: str = "INFO") -> logging.Logger:
    """Return a configured logger; idempotent across calls."""
    logger = logging.getLogger(name)
    if logger.handlers:                     # already configured
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
