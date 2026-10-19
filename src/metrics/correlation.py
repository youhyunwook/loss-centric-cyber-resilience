"""
src/metrics/correlation.py
---------------------------
Spearman rho between asset importance I(n) and exposure ER(n).

Per Reviewer 2 #5, this analysis is promoted from the conclusion
into Section V.E and used as evidence supporting the design choice
to compute I and ER independently.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from ..utils.logging import get_logger

log = get_logger(__name__)


def spearman_importance_exposure(
    importance: dict[str, float],
    exposure: dict[str, float],
) -> dict[str, float]:
    """Return Spearman rho, p-value, and the joined dataframe.

    Returns
    -------
    dict
        ``{"rho", "pvalue", "n", "df"}`` — df is the joined DataFrame
        with columns [node, I, ER].
    """
    common = sorted(set(importance) & set(exposure))
    if not common:
        log.warning("No overlapping nodes between importance and exposure.")
        return {"rho": float("nan"), "pvalue": float("nan"),
                "n": 0, "df": pd.DataFrame()}

    df = pd.DataFrame({
        "node": common,
        "I": [importance[n] for n in common],
        "ER": [exposure[n] for n in common],
    })
    rho, pval = stats.spearmanr(df["I"], df["ER"])
    log.info("Spearman rho(I, ER) = %.4f (p=%.4g, n=%d)", rho, pval, len(common))
    return {"rho": float(rho), "pvalue": float(pval), "n": len(common), "df": df}
