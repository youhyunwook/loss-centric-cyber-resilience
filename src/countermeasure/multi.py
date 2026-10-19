"""
src/countermeasure/multi.py
----------------------------
Multi-countermeasure composition rules.

Implements the theoretical extension promised in the discussion to
address Reviewers 2 and 3 (single-CM limitation).

Three composition rules are implemented:

  * "independent"     r_combined = prod_i r_i
                      (no overlap, no synergy)

  * "sub_multiplicative" r_combined = prod_i r_i ** alpha   (alpha < 1)
                      (defense-in-depth synergy: combined is better
                      than independent product)

  * "super_additive"  r_combined = max(prod_i r_i, max_i r_i * beta)
                      (overlapping coverage: combined floor at
                      single-best * beta, beta in (0, 1])
                      i.e. diminishing returns

Manuscript Sec. VI (Discussion).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def combine_residual(
    residuals: Iterable[float],
    rule: str = "independent",
    alpha: float = 0.7,
    beta: float = 0.9,
) -> float:
    """Combine multiple residual ratios under a composition rule.

    Parameters
    ----------
    residuals : iterable of float
        Per-countermeasure r in [0, 1].
    rule : {"independent", "sub_multiplicative", "super_additive"}
    alpha : float
        Synergy exponent for ``"sub_multiplicative"`` (0 < alpha < 1
        => stronger combined effect than independent product).
    beta : float
        Overlap parameter for ``"super_additive"``
        (0 < beta <= 1 => diminishing returns vs. independent).

    Returns
    -------
    float
        Combined residual ratio in [0, 1].
    """
    arr = np.asarray(list(residuals), dtype=float)
    if arr.size == 0:
        return 1.0
    arr = np.clip(arr, 0.0, 1.0)
    indep = float(np.prod(arr))

    if rule == "independent":
        return indep
    elif rule == "sub_multiplicative":
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        return float(min(1.0, indep ** alpha))
    elif rule == "super_additive":
        if not (0.0 < beta <= 1.0):
            raise ValueError("beta must be in (0, 1].")
        floor = float(arr.min() * beta)        # best single CM, attenuated
        return float(max(indep, floor))
    else:
        raise ValueError(f"Unknown composition rule '{rule}'.")
