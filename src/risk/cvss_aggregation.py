"""
src/risk/cvss_aggregation.py
-----------------------------
CVSS aggregation strategies for V(S).

This module is central to the R1 / R2 / R3 V(S) = 0.99 saturation
critique. The original paper used ``max`` (worst-case CVSS within the
scenario CWE pool), which collapses both S1 and S2 to ~0.99 and
removes scenario-level differentiation.

We expose three statistically-distinct strategies so the framework can
be evaluated under each (paper Table 11 is produced by
``experiments/exp15_cvss_live_aggregation_sensitivity.py``):

  * max     — original; conservative / worst-case (NIST SP 800-30)
  * mean    — average severity (smooths outliers)
  * median  — outlier-robust central tendency

The three methods give the manuscript a clean three-row comparison
table where each row has a distinct, defensible statistical meaning.

All strategies return a value in [0, 10]; the caller normalises by
``cvss_normalizer`` (default 10) to get V(S) in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

AVAILABLE_AGGREGATIONS = ("max", "mean", "median")


def aggregate_cvss(
    cvss_pool: pd.Series | list[float] | np.ndarray,
    method: str = "max",
    normalizer: float = 10.0,
) -> float:
    """Aggregate a CVSS pool into a single V(S) in [0, 1].

    Parameters
    ----------
    cvss_pool : array-like of float
        CVSS scores for all CVEs matching the scenario's CWE.
    method : {"max", "mean", "median"}
        Aggregation strategy. See module docstring.
    normalizer : float, default 10.0
        CVSS scores are divided by this to produce V(S) in [0, 1].

    Returns
    -------
    float
        V(S) in [0, 1].
    """
    arr = np.asarray(list(cvss_pool), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        log.warning("Empty CVSS pool — returning 0.0.")
        return 0.0

    method = method.lower()
    if method == "max":
        v = arr.max()
    elif method == "mean":
        v = arr.mean()
    elif method == "median":
        v = float(np.median(arr))
    else:
        raise ValueError(
            f"Unknown CVSS aggregation '{method}'. "
            f"Choose from {AVAILABLE_AGGREGATIONS}."
        )

    return float(min(v / normalizer, 1.0))


def aggregate_cvss_all_methods(
    cvss_pool: pd.Series | list[float] | np.ndarray,
    normalizer: float = 10.0,
) -> dict[str, float]:
    """Convenience: run every available aggregation, return a dict.

    Convenience helper for comparing aggregation strategies on a CVSS pool.
    """
    return {
        m: aggregate_cvss(cvss_pool, method=m, normalizer=normalizer)
        for m in AVAILABLE_AGGREGATIONS
    }
