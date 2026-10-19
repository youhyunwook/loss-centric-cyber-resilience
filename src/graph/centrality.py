"""
src/graph/centrality.py
-----------------------
Asset importance I(n).

PageRank is the primary method (manuscript Sec. III.B). Comparator
centralities are exposed for the centrality ablation (paper Table 13 uses
degree and betweenness; ``experiments/exp16_pagerank_degree_betweenness_ablation.py``):

  * degree              — simple in-degree
  * betweenness         — shortest-path bottleneck centrality
  * weighted_frequency  — sum of inbound edge weights

All methods produce a min-max normalised dict[node -> [0, 1]].
"""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

AVAILABLE_CENTRALITIES = ("pagerank", "degree", "betweenness", "weighted_frequency")


# ----------------------------------------------------------------------------
# Individual centrality implementations
# ----------------------------------------------------------------------------
def _pagerank(
    G: nx.DiGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight_transform: str = "log1p",
) -> dict[str, float]:
    """PageRank with optional log1p weight transform (notebook cell [31]).

    The original notebook computed ``w_pr = log1p(raw_count)`` per edge
    before running PageRank, to dampen the heavy-tailed flow-count
    distribution. We replicate that behaviour by default.

    Parameters
    ----------
    weight_transform : {"log1p", "raw", "none"}
        - "log1p": ``log1p(raw_count)`` per edge (notebook default)
        - "raw"  : use ``weight`` (= flow count) directly
        - "none" : equal weights (= unweighted PageRank)
    """
    if weight_transform == "none":
        return nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol, weight=None)

    weight_attr: str = "weight"
    if weight_transform == "log1p":
        # Cache transformed weights on each edge so we can reuse without
        # mutating the caller's graph attributes destructively.
        for _u, _v, data in G.edges(data=True):
            rc = data.get("raw_count", data.get("weight", 1))
            data["w_pr"] = float(np.log1p(int(rc)))
        weight_attr = "w_pr"
    elif weight_transform == "raw":
        weight_attr = "weight"
    else:
        raise ValueError(
            f"Unknown weight_transform '{weight_transform}'. "
            "Choose 'log1p', 'raw', or 'none'."
        )

    return nx.pagerank(
        G, alpha=damping, max_iter=max_iter, tol=tol, weight=weight_attr,
    )


def _degree(G: nx.DiGraph) -> dict[str, float]:
    """In-degree centrality (un-weighted)."""
    return {n: float(d) for n, d in G.in_degree()}


def _betweenness(G: nx.DiGraph) -> dict[str, float]:
    """Shortest-path betweenness centrality (un-weighted)."""
    return nx.betweenness_centrality(G, normalized=True)


def _weighted_frequency(G: nx.DiGraph) -> dict[str, float]:
    """Sum of inbound edge weights (= total inbound flow count)."""
    out: dict[str, float] = {}
    for n in G.nodes():
        out[n] = float(sum(d.get("weight", 1) for _, _, d in G.in_edges(n, data=True)))
    return out


_DISPATCH: dict[str, Callable[..., dict[str, float]]] = {
    "pagerank": _pagerank,
    "degree": _degree,
    "betweenness": _betweenness,
    "weighted_frequency": _weighted_frequency,
}


# ----------------------------------------------------------------------------
# Min-max normalisation
# ----------------------------------------------------------------------------
def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:                         # constant — avoid div0
        return {k: 0.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


def _log1p_minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.log1p(np.array(list(scores.values()), dtype=float))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    keys = list(scores.keys())
    out = (vals - lo) / (hi - lo)
    return {k: float(v) for k, v in zip(keys, out)}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def compute_importance(
    G: nx.DiGraph,
    method: str = "pagerank",
    normalize: str = "minmax",
    **kwargs,
) -> dict[str, float]:
    """Compute asset importance I(n) per node.

    Parameters
    ----------
    G : networkx.DiGraph
    method : {"pagerank", "degree", "betweenness", "weighted_frequency"}
        Centrality algorithm.
    normalize : {"minmax", "log1p_minmax", "none"}
        Post-processing.
    **kwargs : passed to the underlying algorithm
        e.g. ``damping=0.85`` for PageRank.

    Returns
    -------
    dict[str, float]
        Node -> importance score.
    """
    method = method.lower()
    if method not in _DISPATCH:
        raise ValueError(
            f"Unknown centrality '{method}'. "
            f"Choose from {AVAILABLE_CENTRALITIES}."
        )

    log.info("Computing %s centrality on |V|=%d", method, G.number_of_nodes())
    raw = _DISPATCH[method](G, **kwargs) if method == "pagerank" \
        else _DISPATCH[method](G)

    if normalize == "minmax":
        return _minmax(raw)
    elif normalize == "log1p_minmax":
        return _log1p_minmax(raw)
    elif normalize in ("none", None, False):
        return raw
    else:
        raise ValueError(f"Unknown normalize='{normalize}'.")
