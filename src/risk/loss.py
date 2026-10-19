"""
src/risk/loss.py
----------------
Risk(n) and Loss(s) computation.

  Risk(n) = I(n) * ER(n) * V(S)
  Loss(s) = sum_{n in R(s, K)} Risk(n)

where R(s, K) is the K-hop impact range around scenario s targets.

Manuscript Sec. III.E.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


def compute_risk(
    importance: dict[str, float],
    exposure: dict[str, float],
    vulnerability: float,
    nodes: Iterable[str] | None = None,
) -> dict[str, float]:
    """Per-node Risk(n) = I(n) * ER(n) * V(S).

    Parameters
    ----------
    importance, exposure : dict[str, float]
        Min-max normalised maps node -> [0, 1].
    vulnerability : float
        Scenario V(S) in [0, 1].
    nodes : iterable of str, optional
        Restrict computation to these nodes; defaults to importance.keys().

    Returns
    -------
    dict[str, float]
        Node -> risk score.
    """
    nodes = list(nodes) if nodes is not None else list(importance.keys())
    risk: dict[str, float] = {}
    for n in nodes:
        i = importance.get(n, 0.0)
        e = exposure.get(n, 0.0)
        risk[n] = float(i * e * vulnerability)
    return risk


def compute_loss(
    risk: dict[str, float],
    impact_range: Iterable[str],
) -> float:
    """Loss(s) = sum of Risk(n) over the K-hop impact range.

    Parameters
    ----------
    risk : dict[str, float]
        Output of :func:`compute_risk`.
    impact_range : iterable of str
        Nodes within the K-hop reach of scenario targets.

    Returns
    -------
    float
        Scenario loss.
    """
    return float(sum(risk.get(n, 0.0) for n in impact_range))


def loss_breakdown(
    importance: dict[str, float],
    exposure: dict[str, float],
    vulnerability: float,
    impact_range: Iterable[str],
) -> pd.DataFrame:
    """Return per-node contributions to Loss(s) for diagnostic tables."""
    rows = []
    for n in impact_range:
        i = importance.get(n, 0.0)
        e = exposure.get(n, 0.0)
        rows.append({
            "node": n,
            "I": i,
            "ER": e,
            "V": vulnerability,
            "Risk": i * e * vulnerability,
        })
    df = pd.DataFrame(rows).sort_values("Risk", ascending=False)
    return df
