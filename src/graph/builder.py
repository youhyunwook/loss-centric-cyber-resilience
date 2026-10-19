"""
src/graph/builder.py
--------------------
Construct the directed communication graph G_normal from benign flows,
and extract the K-hop subgraph around scenario targets for Loss(s).

Manuscript Sec. III.A.

Backbone extraction (notebook cell [29]):
  When the raw benign edge set is dense (~10^7 edges across ~10^5 nodes
  in the Improved CSE-CIC-IDS-2018 dataset), the structural centrality
  signal becomes diluted. The original manuscript therefore applied a
  best-first backbone extraction starting from a seed set (the scenario
  targets and their /24 subnet) and traversing the densest edges within
  ``max_hops`` until ``node_budget`` is reached. This module reproduces
  that traversal so the framework yields the same V/E counts as the
  notebook.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Iterable

import networkx as nx
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backbone extraction (notebook cell [29])
# ---------------------------------------------------------------------------
def extract_backbone(
    seed_nodes: Iterable[str],
    adj: dict[str, list[tuple[str, int]]],
    node_budget: int = 100_000,
    max_hops: int = 4,
) -> tuple[set[str], set[tuple[str, str]], dict[str, int]]:
    """Best-first backbone extraction from seed nodes.

    Re-implements ``extract_backbone`` from the original notebook
    (cell [29]) verbatim. The traversal greedily takes the densest
    outgoing edge from already-selected nodes, expanding outward up to
    ``max_hops`` from any seed and stopping when ``node_budget`` is
    reached.

    Parameters
    ----------
    seed_nodes : iterable of str
        Initial node set (scenario targets + same-/24 hosts).
    adj : dict[str, list[(str, int)]]
        Adjacency map: ``src -> [(dst, weight), ...]``, sorted by weight
        descending (caller's responsibility).
    node_budget : int, default 100_000
        Soft cap on the size of the selected node set. Use a large value
        (e.g. 100_000) to let the natural ``MIN_EDGE_COUNT`` filter
        determine the final size, as in the original manuscript run.
    max_hops : int, default 4
        Cap on hop distance from any seed.

    Returns
    -------
    selected_nodes : set[str]
    selected_edges : set[(str, str)]
    hop_of : dict[str, int]
        Hop distance from the seed set for each selected node.
    """
    selected_nodes: set[str] = set(seed_nodes)
    selected_edges: set[tuple[str, str]] = set()
    pq: list = []
    hop_of: dict[str, int] = {n: 0 for n in seed_nodes}

    def push_from(node: str) -> None:
        h = hop_of.get(node, 0)
        if h >= max_hops:
            return
        for nbr, cnt in adj.get(node, []):
            heapq.heappush(pq, (-cnt, h + 1, node, nbr, cnt))

    for n in list(seed_nodes):
        push_from(n)

    while pq and len(selected_nodes) < node_budget:
        neg_cnt, h, s, d, cnt = heapq.heappop(pq)
        selected_edges.add((s, d))
        if d not in selected_nodes:
            selected_nodes.add(d)
            hop_of[d] = h
            push_from(d)

    return selected_nodes, selected_edges, hop_of


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_communication_graph(
    flows: pd.DataFrame,
    edge_threshold_tau: int = 1,
    src_col: str = "Src IP",
    dst_col: str = "Dst IP",
    keep_largest_wcc: bool = False,
    target_hosts: Iterable[str] | None = None,
    use_backbone: bool = False,
    node_budget: int = 100_000,
    max_hops: int = 4,
    backbone_seed_subnet_prefix: bool = True,
) -> nx.DiGraph:
    """Build a directed graph G = (V, E) from flow records.

    An edge (u, v) is included only if the flow count from u to v is
    >= ``edge_threshold_tau`` (filters scan / one-shot noise; this is
    the notebook's ``MIN_EDGE_COUNT``).

    Parameters
    ----------
    flows : DataFrame
        Benign flows.
    edge_threshold_tau : int, default 1
        Minimum flow count to retain an edge. The original manuscript
        used 1 (no filter); the framework default raised this to 3.
        For the revision we follow the notebook (1).
    keep_largest_wcc : bool, default False
        If True, keep only the largest weakly connected component.
        If ``target_hosts`` is also given, keep the WCC(s) containing them.
    target_hosts : iterable of str, optional
        Used as seeds when ``use_backbone=True`` and / or for WCC restriction.
    use_backbone : bool, default False
        If True, run the notebook's best-first backbone extraction
        starting from ``target_hosts`` (plus same-/24 nodes when
        ``backbone_seed_subnet_prefix`` is True).
    node_budget, max_hops : int
        Backbone-extraction parameters. See :func:`extract_backbone`.
    backbone_seed_subnet_prefix : bool, default True
        If True, the seed set is augmented with all internal nodes whose
        IP shares the targets' /24 prefix (notebook cell [27] behaviour).

    Returns
    -------
    networkx.DiGraph
        ``G.edges[u, v]['weight']`` = flow count (also stored as
        ``raw_count`` for downstream PageRank weighting).
    """
    pairs = list(zip(flows[src_col].astype(str), flows[dst_col].astype(str)))
    counts = Counter(pairs)
    log.info(
        "Edges before threshold: %d unique pairs (tau=%d)",
        len(counts), edge_threshold_tau,
    )

    if use_backbone:
        if not target_hosts:
            raise ValueError(
                "use_backbone=True requires target_hosts to seed extraction."
            )

        # Build adjacency of edges that pass the tau filter.
        adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (u, v), w in counts.items():
            if w >= edge_threshold_tau:
                adj[u].append((v, int(w)))
        for s in adj:
            adj[s].sort(key=lambda x: x[1], reverse=True)

        # Seed: targets + same-/24 internal hosts (notebook cell [27]).
        seed_nodes: set[str] = set(map(str, target_hosts))
        if backbone_seed_subnet_prefix:
            target_set = set(seed_nodes)
            seed_prefixes = {t.rsplit(".", 1)[0] + "." for t in target_set}
            # Find every internal node sharing those /24 prefixes among
            # the (post-tau) edge endpoints.
            for u, neigh in adj.items():
                if any(u.startswith(p) for p in seed_prefixes):
                    seed_nodes.add(u)
                for v, _ in neigh:
                    if any(v.startswith(p) for p in seed_prefixes):
                        seed_nodes.add(v)
        log.info(
            "Backbone seed: %d nodes (targets + /24 subnet)",
            len(seed_nodes),
        )

        selected_nodes, selected_edges, _hop = extract_backbone(
            seed_nodes=seed_nodes,
            adj=adj,
            node_budget=node_budget,
            max_hops=max_hops,
        )
        log.info(
            "Backbone extracted: %d nodes, %d edges (budget=%d, max_hops=%d)",
            len(selected_nodes), len(selected_edges), node_budget, max_hops,
        )

        G = nx.DiGraph()
        for (u, v) in selected_edges:
            w = counts[(u, v)]
            G.add_edge(u, v, weight=w, raw_count=w)
        # Ensure isolated targets remain in the graph
        for t in target_hosts:
            t = str(t)
            if t not in G:
                G.add_node(t)

    else:
        # Plain tau-filtered graph (no backbone).
        G = nx.DiGraph()
        for (u, v), w in counts.items():
            if w >= edge_threshold_tau:
                G.add_edge(u, v, weight=w, raw_count=w)

    log.info(
        "After construction: |V|=%d, |E|=%d",
        G.number_of_nodes(), G.number_of_edges(),
    )

    if keep_largest_wcc and G.number_of_nodes() > 0:
        if target_hosts:
            target_set = set(map(str, target_hosts))
            keep_components = [
                comp for comp in nx.weakly_connected_components(G)
                if comp & target_set
            ]
            if keep_components:
                keep_nodes: set[str] = set().union(*keep_components)
                G = G.subgraph(keep_nodes).copy()
                log.info(
                    "Kept WCC(s) containing targets: |V|=%d, |E|=%d",
                    G.number_of_nodes(), G.number_of_edges(),
                )
            else:
                log.warning("No WCC contained any target host; graph unchanged.")
        else:
            largest = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            log.info(
                "Kept largest WCC: |V|=%d, |E|=%d",
                G.number_of_nodes(), G.number_of_edges(),
            )

    return G


def k_hop_subgraph(
    G: nx.DiGraph,
    targets: Iterable[str],
    K: int = 3,
) -> set[str]:
    """Return the set of nodes reachable within K hops of any target,
    treating the graph as undirected for impact propagation.

    Manuscript: ``R(t, K)`` — impact range of an attack on target t.
    """
    if K < 0:
        raise ValueError("K must be >= 0.")

    targets = [t for t in targets if t in G]
    if not targets:
        log.warning("No targets present in graph; returning empty set.")
        return set()

    H = G.to_undirected()
    reach: set[str] = set()
    for t in targets:
        ego = nx.ego_graph(H, t, radius=K, undirected=True)
        reach.update(ego.nodes())
    log.info(
        "K-hop reach (K=%d) from %d targets: %d nodes",
        K, len(targets), len(reach),
    )
    return reach
