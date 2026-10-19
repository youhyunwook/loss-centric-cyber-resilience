"""
scripts/run_pipeline.py
------------------------
End-to-end pipeline runner (notebook-aligned).

Run:
    python scripts/run_pipeline.py --config config/default.yaml \
        --scenarios config/scenarios.yaml \
        --countermeasures config/countermeasures.yaml \
        --scenario S1 \
        --data D:/.../All_Combined_slim.csv \
        --output results/run_S1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.countermeasure.d3fend_mapper import D3FENDMapper
from src.countermeasure.selector import score_countermeasures
from src.graph.builder import build_communication_graph, k_hop_subgraph
from src.graph.centrality import compute_importance
from src.graph.exposure import compute_exposure
from src.preprocessing.augmentation import augment_multi_hop
from src.preprocessing.data_loader import load_cic_ids_2018, load_scenarios
from src.preprocessing.flow_filter import (
    drop_invalid_flows, exclude_attempted_flows,
    filter_flows, filter_internal_only,
    separate_benign_attack,
)
from src.risk.attack_mapping import build_threat_table, scenario_cwe_pool
from src.risk.cvss_aggregation import aggregate_cvss
from src.risk.loss import compute_loss, compute_risk, loss_breakdown
from src.utils.io import ensure_dir, load_config, save_dataframe
from src.utils.logging import get_logger

log = get_logger("run_pipeline")


def run_one_scenario(
    scenario, config, cic_path, output_dir, mapper,
    benign_cache=None,
):
    """Run pipeline for a single scenario; return summary dict."""
    output_dir = ensure_dir(output_dir)

    if benign_cache and "benign" in benign_cache:
        benign = benign_cache["benign"]
        attack = benign_cache["attack"]
    else:
        raw = load_cic_ids_2018(cic_path)
        raw = drop_invalid_flows(
            raw,
            drop_na_subset=config["preprocessing"]["drop_na_subset"],
            drop_duplicates=config["preprocessing"]["drop_duplicates"],
        )
        benign, attack = separate_benign_attack(
            raw, benign_label=config["preprocessing"]["benign_label"],
        )
        if config["preprocessing"].get("exclude_attempted", False):
            attack = exclude_attempted_flows(
                attack,
                suffix=config["preprocessing"].get("attempted_suffix", "-Attempted"),
            )
        if config["preprocessing"].get("internal_only", False):
            benign = filter_internal_only(benign)
        if config["preprocessing"]["augmentation"]["enabled"]:
            benign = augment_multi_hop(
                benign,
                max_hops=config["preprocessing"]["augmentation"]["max_hops"],
                time_window_seconds=config["preprocessing"]["augmentation"]["time_window_seconds"],
            )
        if benign_cache is not None:
            benign_cache["benign"] = benign
            benign_cache["attack"] = attack

    targets = list(scenario.get("targets") or [])
    if not targets:
        sa = filter_flows(attack, scenario["cic_labels"])
        targets = sorted(sa["Dst IP"].dropna().astype(str).unique().tolist())

    log.info("Scenario %s targets: %s", scenario["id"], targets)

    G = build_communication_graph(
        benign,
        edge_threshold_tau=config["graph"]["edge_threshold_tau"],
        keep_largest_wcc=config["graph"].get("keep_target_wcc", False),
        target_hosts=targets,
        use_backbone=config["graph"].get("use_backbone", False),
        node_budget=config["graph"].get("node_budget", 100000),
        max_hops=config["graph"].get("max_hops", 4),
        backbone_seed_subnet_prefix=config["graph"].get(
            "backbone_seed_subnet_prefix", True,
        ),
    )

    importance = compute_importance(
        G,
        method=config["centrality"]["method"],
        normalize=config["centrality"]["normalize"],
        **(config["centrality"].get("pagerank") or {}),
    )

    exposure = compute_exposure(
        G, flows=benign,
        method=config["exposure"]["method"],
        normalize=config["exposure"]["normalize"],
    )

    if config["risk"].get("use_scenario_cvss_proxy", False) and "cvss_proxy" in scenario:
        V_s = float(scenario["cvss_proxy"]) / float(config["risk"]["cvss_normalizer"])
        V_s = min(V_s, 1.0)
        pool_size = 1
    else:
        threat_table = build_threat_table(
            [scenario],
            cwe_csv=config["paths"]["cwe_csv"],
            cvss_csv=config["paths"]["cvss_csv"],
        )
        pool = scenario_cwe_pool(threat_table, scenario["id"])["cvss"]
        V_s = aggregate_cvss(
            pool,
            method=config["risk"]["cvss_aggregation"],
            normalizer=config["risk"]["cvss_normalizer"],
        )
        pool_size = len(pool)

    log.info("V(S) for %s = %.4f", scenario["id"], V_s)

    R_sK = k_hop_subgraph(G, targets, K=config["graph"]["impact_range_K"])
    risk = compute_risk(importance, exposure, V_s, nodes=R_sK)
    loss_before = compute_loss(risk, R_sK)
    log.info(
        "Loss(s) for %s = %.6f over |R(s,K)|=%d",
        scenario["id"], loss_before, len(R_sK),
    )

    breakdown = loss_breakdown(importance, exposure, V_s, R_sK)
    save_dataframe(breakdown, output_dir / "loss_breakdown.csv")

    cands = mapper.for_scenario(scenario["id"])
    log.info("Scenario %s -> %d candidates", scenario["id"], len(cands))

    t0 = time.perf_counter()
    scored = score_countermeasures(
        candidates=cands,
        importance=importance, exposure=exposure,
        vulnerability=V_s, impact_range=R_sK,
        loss_before=loss_before, config=config,
        target=targets,
        G=G,
    )
    decision_seconds = time.perf_counter() - t0
    save_dataframe(scored, output_dir / "countermeasure_scores.csv")

    summary = {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "n_targets": len(targets),
        "graph_V": G.number_of_nodes(),
        "graph_E": G.number_of_edges(),
        "impact_range_size": len(R_sK),
        "V_s": V_s,
        "cvss_pool_size": pool_size,
        "loss_before": loss_before,
        "best_countermeasure": scored.iloc[0]["cid"] if not scored.empty else None,
        "best_score": float(scored.iloc[0]["score"]) if not scored.empty else 0.0,
        "decision_seconds": decision_seconds,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    log.info("Scenario %s done -> %s", scenario["id"], output_dir)
    return summary


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", show_default=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenarios_path", default="config/scenarios.yaml", show_default=True, type=click.Path(exists=True))
@click.option("--countermeasures", "countermeasures_path", default="config/countermeasures.yaml", show_default=True, type=click.Path(exists=True))
@click.option("--scenario", "scenario_id", default="ALL", show_default=True)
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_root", default="results", show_default=True, type=click.Path())
def main(config_path, scenarios_path, countermeasures_path, scenario_id, data_path, output_root):
    """Run the loss-centric resilience pipeline."""
    config = load_config(config_path)
    scenarios = load_scenarios(scenarios_path)
    if scenario_id != "ALL":
        scenarios = [s for s in scenarios if s["id"] == scenario_id]
        if not scenarios:
            raise SystemExit(f"Scenario '{scenario_id}' not found.")

    mapper = D3FENDMapper(countermeasures_path)
    benign_cache: dict = {}
    summaries = []
    for sc in scenarios:
        out = Path(output_root) / sc["id"]
        s = run_one_scenario(
            scenario=sc, config=config,
            cic_path=Path(data_path),
            output_dir=out, mapper=mapper,
            benign_cache=benign_cache,
        )
        summaries.append(s)

    if summaries:
        pd.DataFrame(summaries).to_csv(
            Path(output_root) / "summary_all.csv", index=False,
        )
        log.info("Written %s", Path(output_root) / "summary_all.csv")


if __name__ == "__main__":
    main()
