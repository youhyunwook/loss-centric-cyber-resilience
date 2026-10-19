"""
experiments/exp05_graph_sensitivity.py
---------------------------------------
Graph parameter robustness (paper Table 14: tau sweep; Table 5: K-hop range).

For each scenario, sweep:
  * edge_threshold tau in {1, 2, 3, 5, 10}   (Table 14)
  * impact_range K   in {1, 2, 3, 4}         (Table 5)

WCC toggle was dropped from the sweep (effect is theoretically self-evident).

Output:
    results/exp05_graph_sensitivity/
        tables/   tau_sweep_<scenario>.csv, K_sweep_<scenario>.csv
        figures/  tau_sweep_<scenario>.png, K_sweep_<scenario>.png
"""

from __future__ import annotations

import copy
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from src.countermeasure.d3fend_mapper import D3FENDMapper
from src.countermeasure.selector import score_countermeasures
from src.risk.loss import compute_loss, compute_risk
from src.utils.logging import get_logger
from src.visualization.plotting import plot_sensitivity_lines

from ._common import (
    DEFAULT_COUNTERMEASURES,
    load_default_artifacts,
    make_experiment_output,
    prepare_scenario_pipeline,
)

log = get_logger("exp05")


def _run_with_config(scenario, cfg, data_path, mapper, benign_cache):
    """Single (tau, K) configuration run."""
    ctx = prepare_scenario_pipeline(scenario, cfg, data_path, benign_cache)
    risk = compute_risk(ctx["importance"], ctx["exposure"], ctx["V_s"], nodes=ctx["R_sK"])
    loss_before = compute_loss(risk, ctx["R_sK"])
    cands = mapper.for_scenario(scenario["id"])
    scored = score_countermeasures(
        candidates=cands,
        importance=ctx["importance"], exposure=ctx["exposure"],
        vulnerability=ctx["V_s"], impact_range=ctx["R_sK"],
        loss_before=loss_before, config=cfg,
        target=ctx["targets"],
        G=ctx["G"],
    )
    return {
        "tau": cfg["graph"]["edge_threshold_tau"],
        "K": cfg["graph"]["impact_range_K"],
        "graph_V": ctx["G"].number_of_nodes(),
        "graph_E": ctx["G"].number_of_edges(),
        "impact_range_size": len(ctx["R_sK"]),
        "loss_before": loss_before,
        "best_cm": scored.iloc[0]["cid"] if not scored.empty else None,
        "best_score": float(scored.iloc[0]["score"]) if not scored.empty else 0.0,
    }


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
def main(data_path: str, scenario_ids: str) -> None:
    config, scenarios = load_default_artifacts()
    chosen = set(scenario_ids.split(","))
    scenarios = [s for s in scenarios if s["id"] in chosen]

    out = make_experiment_output(
        "exp05_graph_sensitivity", config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    mapper = D3FENDMapper(str(DEFAULT_COUNTERMEASURES))

    tau_grid = config["sensitivity"]["graph"]["edge_threshold_tau"]
    K_grid = config["sensitivity"]["graph"]["impact_range_K"]

    for sc in scenarios:
        log.info("--- %s ---", sc["id"])

        # tau sweep — graph rebuild required; cannot reuse benign_cache benign frame
        # (we still cache benign loading to avoid re-reading the CSV)
        rows = []
        for tau in tau_grid:
            cfg = copy.deepcopy(config)
            cfg["graph"]["edge_threshold_tau"] = tau
            try:
                # benign loading reusable, but graph rebuild every time
                benign_cache: dict = {}
                rows.append(_run_with_config(sc, cfg, Path(data_path), mapper, benign_cache))
            except Exception as e:        # noqa: BLE001
                log.error("tau=%s failed: %s", tau, e)
        tau_df = pd.DataFrame(rows)
        out.save_table(tau_df, f"tau_sweep_{sc['id']}.csv")

        if not tau_df.empty:
            fig = plot_sensitivity_lines(
                tau_df, x="tau", y="loss_before",
                title=f"Loss(s) vs tau - {sc['id']}",
                xlabel="Edge threshold tau", ylabel="Loss(s)",
            )
            out.save_figure(fig, f"tau_sweep_{sc['id']}.png", also_pdf=True)
            plt.close(fig)

        # K sweep — graph stays the same, only K changes
        rows = []
        benign_cache_K: dict = {}
        for K in K_grid:
            cfg = copy.deepcopy(config)
            cfg["graph"]["impact_range_K"] = K
            try:
                rows.append(_run_with_config(sc, cfg, Path(data_path), mapper, benign_cache_K))
            except Exception as e:        # noqa: BLE001
                log.error("K=%s failed: %s", K, e)
        K_df = pd.DataFrame(rows)
        out.save_table(K_df, f"K_sweep_{sc['id']}.csv")

        if not K_df.empty:
            fig = plot_sensitivity_lines(
                K_df, x="K", y="loss_before",
                title=f"Loss(s) vs K - {sc['id']}",
                xlabel="Impact range K (hops)", ylabel="Loss(s)",
            )
            out.save_figure(fig, f"K_sweep_{sc['id']}.png", also_pdf=True)
            plt.close(fig)

    out.add_summary("tau_grid", tau_grid)
    out.add_summary("K_grid", K_grid)
    out.close()


if __name__ == "__main__":
    main()
