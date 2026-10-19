"""
src/countermeasure/effectiveness.py
------------------------------------
Effectiveness modelling.

Two parallel APIs:

  (1) Tier-based residual ratio (high/medium/low) — used by simple
      countermeasures that scale the I, E or V factor uniformly.

  (2) Operation-based application (notebook cells [42], [50]) — used
      when the manuscript's mapping table prescribes a specific
      operation per (scenario, countermeasure):
        - SET_V          : set V to a fixed post-deployment value
        - SCALE_E_TARGET : multiply E(target) by E_scale
        - SCALE_E_KHOP   : multiply E(n) by E_scale for n in K-hop ball around target
        - SCALE_E_RSET   : multiply E(n) by E_scale for n in R(s, K)
        - SET_E_TARGET   : set E(target) to a fixed value (cap, e.g. ACL)
        - SCALE_I_TARGET : multiply I(target) by I_scale (e.g. load balancing)

Manuscript Sec. IV.B and Sec. IV.D.
"""

from __future__ import annotations

import random
from typing import Iterable

import networkx as nx

from ..utils.logging import get_logger
from .d3fend_mapper import CountermeasureSpec

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Residual ratio (legacy, tier-based)
# ---------------------------------------------------------------------------
def residual_ratio(
    spec: CountermeasureSpec,
    tier_config: dict,
    sample_within_band: bool = False,
    rng: random.Random | None = None,
) -> float:
    """Map a countermeasure spec to a residual ratio r.

    Parameters
    ----------
    spec : CountermeasureSpec
    tier_config : dict
        ``config['countermeasure']['effectiveness_tiers']``.
    sample_within_band : bool
        If True, sample r uniformly within the tier's [r_min, r_max] band.
    rng : random.Random, optional
    """
    band = tier_config.get(spec.tier)
    if band is None:
        log.warning("Unknown tier '%s'; defaulting to medium.", spec.tier)
        band = tier_config.get("medium", {"r_default": 0.5, "r_min": 0.4, "r_max": 0.6})
    if sample_within_band:
        rng = rng or random.Random()
        return rng.uniform(band["r_min"], band["r_max"])
    return float(band["r_default"])


def apply_countermeasure(
    spec: CountermeasureSpec,
    r: float,
    importance: dict[str, float],
    exposure: dict[str, float],
    vulnerability: float,
    impact_nodes: Iterable[str],
) -> tuple[dict[str, float], dict[str, float], float]:
    """Legacy single-r intervention (multiply chosen factor by r)."""
    impact_set = set(impact_nodes)
    I_new = dict(importance)
    E_new = dict(exposure)
    V_new = vulnerability

    itype = spec.intervention_type.upper()
    if itype == "I":
        for n in impact_set:
            if n in I_new:
                I_new[n] = I_new[n] * r
    elif itype == "E":
        for n in impact_set:
            if n in E_new:
                E_new[n] = E_new[n] * r
    elif itype == "V":
        V_new = vulnerability * r
    else:
        log.warning("Unknown intervention_type '%s' for %s.", itype, spec.cid)

    return I_new, E_new, V_new


# ---------------------------------------------------------------------------
# Operation-based application (notebook cells [42], [50])
# ---------------------------------------------------------------------------
def _khop_ball(
    G: nx.Graph | nx.DiGraph,
    target: str,
    k: int = 1,
) -> set[str]:
    """K-hop ball around target on the undirected projection."""
    UG = G.to_undirected() if G.is_directed() else G
    if target not in UG:
        return set()
    return set(nx.single_source_shortest_path_length(UG, target, cutoff=k).keys())


def apply_operation(
    op: str,
    params: dict,
    target: str | list[str],
    R: Iterable[str],
    I: dict[str, float],
    E: dict[str, float],
    V: float,
    G: nx.Graph | nx.DiGraph | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Apply one of the notebook's countermeasure operations.

    Multi-target support
    --------------------
    Target-scoped operations (SCALE_E_TARGET, SET_E_TARGET, SCALE_I_TARGET,
    SCALE_E_KHOP) are applied independently to every target IP when
    ``target`` is a list. This is the natural semantics for scenarios
    whose ``targets`` field contains multiple destination IPs: a real
    detection rule or ACL would block the source from reaching ANY of the target hosts,
    not just the first one.

    Parameters
    ----------
    op : str
        One of: SET_V, SCALE_E_TARGET, SCALE_E_KHOP, SCALE_E_RSET,
        SET_E_TARGET, SCALE_I_TARGET.
    params : dict
        Operation-specific parameters (E_scale, V_after, ...).
    target : str or list[str]
        Scenario target IP(s). If a list, target-scoped operations apply
        to every element.
    R : iterable of str
        Impact range R(s, K).
    I, E : dict[str, float]
        Current node-level importance and exposure values.
    V : float
        Current scenario vulnerability.
    G : graph, optional
        Required for SCALE_E_KHOP.

    Returns
    -------
    (I', E', V') : copies after the operation is applied.
    """
    R_set = set(R)
    I_local = {n: float(I.get(n, 0.0)) for n in R_set}
    E_local = {n: float(E.get(n, 0.0)) for n in R_set}
    V_local = float(V)

    # Normalise target into a list for uniform handling.
    if isinstance(target, str):
        target_list: list[str] = [target] if target else []
    else:
        target_list = [str(t) for t in (target or [])]

    op = op.upper()

    if op == "SET_V":
        V_local = float(params["V_after"])

    elif op == "SCALE_V":
        V_local = max(0.0, min(1.0, V_local * float(params["V_scale"])))

    elif op == "SCALE_E_TARGET":
        scale = float(params["E_scale"])
        for t in target_list:
            if t in E_local:
                E_local[t] = max(0.0, min(1.0, E_local[t] * scale))

    elif op == "SCALE_E_RSET":
        scale = float(params["E_scale"])
        for n in R_set:
            E_local[n] = max(0.0, min(1.0, E_local[n] * scale))

    elif op == "SCALE_E_KHOP":
        if G is None:
            raise ValueError("SCALE_E_KHOP requires a graph G.")
        scale = float(params["E_scale"])
        khop = int(params.get("khop", 1))
        affected: set[str] = set()
        for t in target_list:
            affected |= _khop_ball(G, t, k=khop)
        for n in (R_set & affected):
            E_local[n] = max(0.0, min(1.0, E_local[n] * scale))

    elif op == "SET_E_TARGET":
        val = float(params["E_after"])
        for t in target_list:
            if t in E_local:
                # ACL semantics: only lower the value
                E_local[t] = min(E_local[t], max(0.0, min(1.0, val)))

    elif op == "SCALE_I_TARGET":
        scale = float(params["I_scale"])
        for t in target_list:
            if t in I_local:
                I_local[t] = max(0.0, I_local[t] * scale)

    else:
        log.warning("Unknown operation '%s'; returning unchanged state.", op)

    # Merge back so the caller's dicts (outside R) remain intact.
    I_out = dict(I)
    E_out = dict(E)
    for n, v in I_local.items():
        I_out[n] = v
    for n, v in E_local.items():
        E_out[n] = v
    return I_out, E_out, V_local
