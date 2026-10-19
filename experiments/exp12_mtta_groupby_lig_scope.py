"""
experiments/exp12_mtta_groupby_lig_scope.py
-------------------------------------------
MTTA measurement (paper Table 15), matching the original Experiment_v5
notebook scope, using pandas groupby for graph construction.

The graph-construction step is the notebook-style edge-frequency step:

    df.groupby(["Src IP", "Dst IP"]).size()

Timing excludes CSV ingestion and preprocessing, but includes graph
construction, PageRank I(n), E(n) from cached inbound-source counts,
V(S), R(t,K), Loss, countermeasure scoring, and best selection.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import networkx as nx
import numpy as np
import pandas as pd

from src.countermeasure.d3fend_mapper import D3FENDMapper
from src.countermeasure.selector import score_countermeasures
from src.graph.builder import k_hop_subgraph
from src.graph.centrality import compute_importance
from src.metrics.mtta import mtta_improvement_table
from src.preprocessing.augmentation import augment_multi_hop
from src.preprocessing.data_loader import load_cic_ids_2018
from src.preprocessing.flow_filter import (
    drop_invalid_flows,
    exclude_attempted_flows,
    filter_internal_destination_only,
    filter_internal_only,
    is_in_networks_series,
    separate_benign_attack,
)
from src.risk.loss import compute_loss, compute_risk
from src.utils.logging import get_logger

from ._common import (
    DEFAULT_COUNTERMEASURES,
    load_default_artifacts,
    make_experiment_output,
    resolve_v_s,
)

log = get_logger("exp12")


def _log1p_minmax(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    keys = list(raw.keys())
    vals = np.log1p(np.array([raw[k] for k in keys], dtype=float))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in keys}
    out = (vals - lo) / (hi - lo)
    return {k: float(v) for k, v in zip(keys, out)}


def _prepare_cached_inputs(config: dict, data_path: Path) -> dict:
    """Materialise the pre-timing inputs used by the LIG v5 MTTA scope."""
    raw = load_cic_ids_2018(data_path)
    raw = drop_invalid_flows(
        raw,
        drop_na_subset=config["preprocessing"]["drop_na_subset"],
        drop_duplicates=config["preprocessing"]["drop_duplicates"],
    )
    benign, attack = separate_benign_attack(
        raw,
        benign_label=config["preprocessing"]["benign_label"],
    )
    if config["preprocessing"].get("exclude_attempted", False):
        attack = exclude_attempted_flows(
            attack,
            suffix=config["preprocessing"].get("attempted_suffix", "-Attempted"),
        )

    internal_networks = config["preprocessing"].get("internal_networks")
    exposure_scope = config.get("exposure", {}).get("flow_scope", "graph_benign")
    if exposure_scope == "benign_internal_destination":
        benign_exposure = filter_internal_destination_only(
            benign,
            internal_networks=internal_networks,
        )
    else:
        benign_exposure = benign

    if config["preprocessing"].get("internal_only", False):
        benign_graph = filter_internal_only(
            benign,
            internal_networks=internal_networks,
        )
    else:
        benign_graph = benign

    if config["preprocessing"]["augmentation"]["enabled"]:
        benign_graph = augment_multi_hop(
            benign_graph,
            max_hops=config["preprocessing"]["augmentation"]["max_hops"],
            time_window_seconds=config["preprocessing"]["augmentation"]["time_window_seconds"],
        )

    return {
        "benign_graph": benign_graph,
        "benign_exposure": benign_exposure,
        "attack": attack,
    }


def _cached_external_source_counts(
    benign_exposure: pd.DataFrame,
    config: dict,
    src_col: str = "Src IP",
    dst_col: str = "Dst IP",
) -> dict[str, int]:
    """Cache Eout_raw from preprocessed benign exposure flows."""
    internal_networks = config["preprocessing"].get("internal_networks")
    src_internal = is_in_networks_series(benign_exposure[src_col], internal_networks)
    external_to_internal = benign_exposure[~src_internal]
    return external_to_internal.groupby(dst_col)[src_col].nunique().astype(int).to_dict()


def _target_wcc_subgraph(G: nx.DiGraph, targets: list[str]) -> nx.DiGraph:
    """Keep the weakly connected component(s) containing scenario targets."""
    if not targets or G.number_of_nodes() == 0:
        return G

    target_set = set(map(str, targets))
    keep_components = [
        comp for comp in nx.weakly_connected_components(G)
        if comp & target_set
    ]
    if not keep_components:
        log.warning("No WCC contained any target host; graph unchanged.")
        return G

    keep_nodes: set[str] = set().union(*keep_components)
    return G.subgraph(keep_nodes).copy()


def _build_groupby_graph(
    flows: pd.DataFrame,
    targets: list[str],
    tau: int,
    keep_target_wcc: bool,
    src_col: str = "Src IP",
    dst_col: str = "Dst IP",
) -> nx.DiGraph:
    """Build the notebook-style tau-filtered graph using pandas groupby."""
    edge_freq = flows.groupby([src_col, dst_col]).size().reset_index(name="freq")
    log.info(
        "Edges before threshold: %d unique pairs (tau=%d)",
        len(edge_freq), tau,
    )
    edge_freq = edge_freq[edge_freq["freq"] >= tau]

    G = nx.DiGraph()
    G.add_edges_from(zip(edge_freq[src_col].astype(str), edge_freq[dst_col].astype(str)))
    for target in targets:
        if target not in G:
            G.add_node(str(target))

    log.info(
        "After groupby construction: |V|=%d, |E|=%d",
        G.number_of_nodes(), G.number_of_edges(),
    )

    if keep_target_wcc:
        G = _target_wcc_subgraph(G, targets)
        log.info(
            "Kept WCC(s) containing targets: |V|=%d, |E|=%d",
            G.number_of_nodes(), G.number_of_edges(),
        )

    return G


def _run_groupby_lig_scope_once(
    scenario: dict,
    config: dict,
    cached: dict,
    ext_counts: dict[str, int],
    mapper: D3FENDMapper,
) -> tuple[float, dict]:
    """Run the timed v5-scope pipeline once with groupby graph construction."""
    t0 = time.perf_counter()

    targets = list(scenario.get("targets") or [])
    if not targets:
        label_set = set(scenario["cic_labels"])
        scenario_attack = cached["attack"][cached["attack"]["Label"].isin(label_set)]
        targets = sorted(scenario_attack["Dst IP"].dropna().astype(str).unique().tolist())

    G = _build_groupby_graph(
        cached["benign_graph"],
        targets=targets,
        tau=config["graph"]["edge_threshold_tau"],
        keep_target_wcc=config["graph"].get("keep_target_wcc", False),
    )

    importance = compute_importance(
        G,
        method=config["centrality"]["method"],
        normalize=config["centrality"]["normalize"],
        **(config["centrality"].get("pagerank") or {}),
    )

    nodes = list(G.nodes())
    if config["exposure"]["method"] == "internal_indegree_plus_external_sources":
        e_raw = {n: float(G.in_degree(n) + ext_counts.get(n, 0)) for n in nodes}
        exposure = _log1p_minmax(e_raw)
    else:
        grp = cached["benign_exposure"].groupby("Dst IP")["Src IP"].nunique().to_dict()
        exposure = _log1p_minmax({n: float(grp.get(n, 0)) for n in nodes})

    V_s, pool_size = resolve_v_s(scenario, config)
    R_sK = k_hop_subgraph(G, targets, K=config["graph"]["impact_range_K"])
    risk = compute_risk(importance, exposure, V_s, nodes=R_sK)
    loss_before = compute_loss(risk, R_sK)

    cands = mapper.for_scenario(scenario["id"])
    scored = score_countermeasures(
        candidates=cands,
        importance=importance,
        exposure=exposure,
        vulnerability=V_s,
        impact_range=R_sK,
        loss_before=loss_before,
        config=config,
        target=targets,
        G=G,
    )

    elapsed = time.perf_counter() - t0
    outputs = {
        "n_targets": len(targets),
        "graph_V": G.number_of_nodes(),
        "graph_E": G.number_of_edges(),
        "impact_range_size": len(R_sK),
        "V_s": V_s,
        "cvss_pool_size": pool_size,
        "loss_before": loss_before,
        "n_candidates": len(cands),
        "best_cm": scored.iloc[0]["cid"] if not scored.empty else None,
        "best_score": float(scored.iloc[0]["score"]) if not scored.empty else 0.0,
    }
    return elapsed, outputs


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
@click.option("--n-runs", default=50, show_default=True)
@click.option("--warmup", default=5, show_default=True)
def main(data_path: str, scenario_ids: str, n_runs: int, warmup: int) -> None:
    config, scenarios = load_default_artifacts()
    chosen = set(s.strip() for s in scenario_ids.split(",") if s.strip())
    scenarios = [s for s in scenarios if s["id"] in chosen]

    out = make_experiment_output(
        "exp12_mtta_groupby_lig_scope",
        config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    mapper = D3FENDMapper(str(DEFAULT_COUNTERMEASURES))

    log.info("Preparing cached LIG v5 inputs outside timing scope.")
    cached = _prepare_cached_inputs(config, Path(data_path))
    ext_counts = _cached_external_source_counts(cached["benign_exposure"], config)

    summary_rows = []
    for sc in scenarios:
        log.info("--- groupby LIG v5-scope MTTA: %s ---", sc["id"])
        for _ in range(warmup):
            _run_groupby_lig_scope_once(sc, config, cached, ext_counts, mapper)

        times = []
        last_outputs = {}
        run_rows = []
        for i in range(n_runs):
            elapsed, outputs = _run_groupby_lig_scope_once(
                sc, config, cached, ext_counts, mapper,
            )
            times.append(elapsed)
            last_outputs = outputs
            run_rows.append({
                "scenario_id": sc["id"],
                "run": i + 1,
                "seconds": elapsed,
                **outputs,
            })

        arr = np.array(times, dtype=float)
        timing = {
            "mean_seconds": float(arr.mean()),
            "median_seconds": float(np.median(arr)),
            "std_seconds": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min_seconds": float(arr.min()),
            "max_seconds": float(arr.max()),
            "n_runs": int(len(arr)),
        }

        out.save_table(pd.DataFrame(run_rows), f"mtta_groupby_runs_{sc['id']}.csv")

        comp = mtta_improvement_table(
            measured_seconds=timing["median_seconds"],
            baselines=config["mtta"]["baselines"],
        )
        comp.insert(0, "scenario_id", sc["id"])
        out.save_table(comp, f"mtta_groupby_comparison_{sc['id']}.csv", also_xlsx=True)

        summary_rows.append({
            "scenario_id": sc["id"],
            **last_outputs,
            **timing,
            "improvement_pct_min": float(comp["improvement_pct"].min()),
            "improvement_pct_max": float(comp["improvement_pct"].max()),
            "n_baselines": int(len(comp)),
        })

    out.save_table(pd.DataFrame(summary_rows), "mtta_groupby_summary.csv", also_xlsx=True)
    out.add_summary("measurement_definition", (
        "LIG Experiment_v5 scope with pandas groupby graph construction: "
        "CSV ingestion/preprocessing are excluded. Timing includes graph "
        "construction, PageRank importance, E(n) from cached inbound-source "
        "counts, V(S), K-hop impact range, Loss, countermeasure scoring, "
        "and best selection."
    ))
    out.close()


if __name__ == "__main__":
    main()
