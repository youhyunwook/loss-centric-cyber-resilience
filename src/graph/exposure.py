"""
src/graph/exposure.py
---------------------
Attack exposure ER(n) — manuscript Sec. III.C.

Defined as the number of distinct inbound source IPs (un-weighted).
The notebook (cell [35]) applies ``log1p`` followed by min-max
normalisation; we replicate that here. The independence of ER from I
(PageRank) is the basis for the design argument in Sec. V.E
(Reviewer 2 #5).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ..preprocessing.flow_filter import is_in_networks_series
from ..utils.logging import get_logger

log = get_logger(__name__)


def compute_exposure(
    G: nx.DiGraph,
    flows: pd.DataFrame | None = None,
    method: str = "unique_inbound_sources",
    normalize: str = "log1p_minmax",
    src_col: str = "Src IP",
    dst_col: str = "Dst IP",
    internal_networks: list[str] | None = None,
) -> dict[str, float]:
    """Compute exposure ER(n) per node.

    Parameters
    ----------
    G : networkx.DiGraph
        Communication graph (used to enumerate the node set).
    flows : DataFrame, optional
        Raw flow records. If supplied, ER is computed at flow level
        (more faithful to "unique inbound source IPs" than the post-tau
        graph). If None, falls back to graph in-degree.
    method : {"unique_inbound_sources", "internal_indegree_plus_external_sources", "in_degree"}
    normalize : {"log1p_minmax", "minmax", "none"}
        - "log1p_minmax": ``log1p`` then min-max (notebook default)
        - "minmax"      : min-max only
        - "none"        : raw counts

    Returns
    -------
    dict[str, float]
        Node -> exposure score.
    """
    nodes = list(G.nodes())
    log.info(
        "Computing exposure (method=%s, normalize=%s) on %d nodes",
        method, normalize, len(nodes),
    )

    if method == "unique_inbound_sources" and flows is not None:
        grp = flows.groupby(dst_col)[src_col].nunique().to_dict()
        raw = {n: float(grp.get(n, 0)) for n in nodes}
    elif method == "internal_indegree_plus_external_sources" and flows is not None:
        # Original manuscript notebook (Experiment_v5):
        # Eraw(n) = tau-filtered internal graph in-degree
        #           + unique benign external sources to internal dst n.
        dst_internal = flows[dst_col].astype(str).isin(nodes)
        src_internal = is_in_networks_series(flows[src_col], internal_networks)
        external_to_graph_nodes = flows[dst_internal & ~src_internal]
        ext_grp = external_to_graph_nodes.groupby(dst_col)[src_col].nunique().to_dict()
        raw = {
            n: float(G.in_degree(n) + ext_grp.get(n, 0))
            for n in nodes
        }
    elif method == "in_degree" or flows is None:
        raw = {n: float(d) for n, d in G.in_degree()}
    else:
        raise ValueError(f"Unknown exposure method '{method}'.")

    if normalize == "log1p_minmax":
        return _log1p_minmax(raw)
    elif normalize == "minmax":
        return _minmax(raw)
    elif normalize in ("none", None, False):
        return raw
    else:
        raise ValueError(f"Unknown normalize='{normalize}'.")


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


def _log1p_minmax(scores: dict[str, float]) -> dict[str, float]:
    """Notebook cell [35]: ``log1p`` then min-max normalise."""
    if not scores:
        return {}
    vals = np.log1p(np.array(list(scores.values()), dtype=float))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    keys = list(scores.keys())
    out = (vals - lo) / (hi - lo)
    return {k: float(v) for k, v in zip(keys, out)}
