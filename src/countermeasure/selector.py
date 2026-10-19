"""
src/countermeasure/selector.py
-------------------------------
Score countermeasures by Score(c) = dLoss(c) / D(c) and pick the
optimum.

  dLoss(c) = Loss_before(s) - Loss_after(s, c)

The scoring uses the operation-based application API
(``apply_operation``) so the manuscript table values for
Loss / dLoss / Score are computed exactly as in notebook cell [50].

Manuscript Sec. IV.D, Tables IX / X.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import pandas as pd

from ..risk.loss import compute_loss, compute_risk
from ..utils.logging import get_logger
from .d3fend_mapper import CountermeasureSpec
from .difficulty import compute_difficulty
from .effectiveness import apply_operation

log = get_logger(__name__)


def score_countermeasures(
    candidates: list[CountermeasureSpec],
    importance: dict[str, float],
    exposure: dict[str, float],
    vulnerability: float,
    impact_range: Iterable[str],
    loss_before: float,
    config: dict,
    target: str | list[str] | None = None,
    G: nx.Graph | nx.DiGraph | None = None,
    perturbation_pct: float = 0.0,
    sample_residuals: bool = False,             # legacy flag, ignored
) -> pd.DataFrame:
    """Score every candidate countermeasure for a scenario using the
    operation-based API.

    Multi-target support
    --------------------
    ``target`` may be a single IP string (S1, S2: one target) or a list
    of IPs (multi-target scenarios). Target-scoped
    operations are applied to every element of the list — see
    :func:`src.countermeasure.effectiveness.apply_operation` for
    semantics.

    Returns
    -------
    DataFrame
        Columns: cid, name, d3fend, tier, op, params, intervention_type,
        D, loss_before, loss_after, dLoss, score
        Sorted by score descending.
    """
    impact_range = list(impact_range)
    diff_cfg = config["countermeasure"]["difficulty"]
    rows: list[dict] = []

    # Normalise target so apply_operation receives a list (or empty list).
    if target is None:
        target_arg: list[str] = []
    elif isinstance(target, str):
        target_arg = [target] if target else []
    else:
        target_arg = list(target)

    for spec in candidates:
        D = compute_difficulty(
            spec,
            components=diff_cfg.get("components"),
            perturbation_pct=perturbation_pct,
            min_value=diff_cfg.get("min_value", 1),
        )

        I_new, E_new, V_new = apply_operation(
            op=spec.op,
            params=spec.params,
            target=target_arg,
            R=impact_range,
            I=importance,
            E=exposure,
            V=vulnerability,
            G=G,
        )
        risk_after = compute_risk(I_new, E_new, V_new, nodes=impact_range)
        loss_after = compute_loss(risk_after, impact_range)
        dLoss = loss_before - loss_after
        score = dLoss / D if D > 0 else 0.0

        rows.append({
            "cid": spec.cid,
            "name": spec.name,
            "d3fend": spec.d3fend,
            "tier": spec.tier,
            "op": spec.op,
            "params": str(spec.params),
            "intervention_type": spec.intervention_type,
            "D": D,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "dLoss": dLoss,
            "score": score,
        })

    score_cfg = config.get("score", {})
    denominator_offset = float(score_cfg.get("difficulty_denominator_offset", 0.0))
    if denominator_offset:
        for row in rows:
            denom = row["D"] + denominator_offset
            row["score"] = row["dLoss"] / denom if denom > 0 else 0.0

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def select_optimal(scored: pd.DataFrame) -> dict | None:
    """Return the top-scored countermeasure as a dict (or None)."""
    if scored.empty:
        return None
    return scored.iloc[0].to_dict()
