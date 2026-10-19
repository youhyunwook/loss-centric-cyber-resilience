"""
experiments/exp01_baseline.py
------------------------------
Reproduce the manuscript baseline using the notebook-aligned pipeline.

Output:
    results/exp01_baseline/
        tables/   scenario_summary.csv (+xlsx)
                  S<i>_countermeasure_scores.csv
        figures/  loss_per_scenario.png
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from src.countermeasure.d3fend_mapper import D3FENDMapper
from src.countermeasure.selector import score_countermeasures
from src.risk.loss import compute_loss, compute_risk
from src.utils.logging import get_logger
from src.visualization.plotting import set_paper_style

from ._common import (
    DEFAULT_COUNTERMEASURES,
    load_default_artifacts,
    make_experiment_output,
    prepare_scenario_pipeline,
)

log = get_logger("exp01")


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
def main(data_path: str, scenario_ids: str) -> None:
    config, scenarios = load_default_artifacts()
    chosen = set(scenario_ids.split(","))
    scenarios = [s for s in scenarios if s["id"] in chosen]

    out = make_experiment_output(
        "exp01_baseline", config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    mapper = D3FENDMapper(str(DEFAULT_COUNTERMEASURES))
    benign_cache: dict = {}                  # share benign frame across scenarios

    summary_rows = []
    for sc in scenarios:
        log.info("--- %s (%s) ---", sc["id"], sc["name"])
        ctx = prepare_scenario_pipeline(sc, config, Path(data_path), benign_cache)

        risk = compute_risk(
            ctx["importance"], ctx["exposure"], ctx["V_s"], nodes=ctx["R_sK"],
        )
        loss_before = compute_loss(risk, ctx["R_sK"])

        cands = mapper.for_scenario(sc["id"])
        scored = score_countermeasures(
            candidates=cands,
            importance=ctx["importance"], exposure=ctx["exposure"],
            vulnerability=ctx["V_s"], impact_range=ctx["R_sK"],
            loss_before=loss_before, config=config,
            target=ctx["targets"],
            G=ctx["G"],
        )
        out.save_table(scored, f"{sc['id']}_countermeasure_scores.csv")

        summary_rows.append({
            "scenario_id": sc["id"], "scenario_name": sc["name"],
            "n_targets": len(ctx["targets"]),
            "graph_V": ctx["G"].number_of_nodes(),
            "graph_E": ctx["G"].number_of_edges(),
            "impact_range_size": len(ctx["R_sK"]),
            "V_s": ctx["V_s"],
            "loss_before": loss_before,
            "best_cm": scored.iloc[0]["cid"] if not scored.empty else None,
            "best_score": float(scored.iloc[0]["score"]) if not scored.empty else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    out.save_table(summary_df, "scenario_summary.csv", also_xlsx=True)

    if not summary_df.empty:
        set_paper_style()
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.bar(summary_df["scenario_id"], summary_df["loss_before"],
               color="#1f4e79", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Loss(s)")
        ax.set_title("Baseline Loss(s) per Scenario")
        for i, v in enumerate(summary_df["loss_before"]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out.save_figure(fig, "loss_per_scenario.png", also_pdf=True)
        plt.close(fig)

    out.add_summary("n_scenarios", len(summary_rows))
    out.close()


if __name__ == "__main__":
    main()
