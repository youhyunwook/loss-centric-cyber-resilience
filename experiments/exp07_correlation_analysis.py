"""
experiments/exp07_correlation_analysis.py
------------------------------------------
Reviewer 2 #5: promote Spearman rho between I(n) and ER(n) into Sec V.E.

Output:
    results/exp07_correlation_analysis/
        tables/   correlation_<scenario>.csv, joined_<scenario>.csv,
                  correlation_summary.csv
        figures/  scatter_<scenario>.png
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics.correlation import spearman_importance_exposure
from src.utils.logging import get_logger
from src.visualization.plotting import plot_scatter_correlation

from ._common import (
    load_default_artifacts,
    make_experiment_output,
    prepare_scenario_pipeline,
)

log = get_logger("exp07")


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
def main(data_path: str, scenario_ids: str) -> None:
    config, scenarios = load_default_artifacts()
    chosen = set(scenario_ids.split(","))
    scenarios = [s for s in scenarios if s["id"] in chosen]

    out = make_experiment_output(
        "exp07_correlation_analysis", config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    benign_cache: dict = {}

    summary = []
    for sc in scenarios:
        log.info("--- %s ---", sc["id"])
        ctx = prepare_scenario_pipeline(sc, config, Path(data_path), benign_cache)
        result = spearman_importance_exposure(ctx["importance"], ctx["exposure"])

        out.save_table(result["df"], f"joined_{sc['id']}.csv")
        out.save_table(
            pd.DataFrame([{
                "scenario_id": sc["id"],
                "rho": result["rho"], "pvalue": result["pvalue"],
                "n": result["n"],
            }]),
            f"correlation_{sc['id']}.csv",
        )

        if not result["df"].empty:
            fig = plot_scatter_correlation(
                result["df"],
                rho=result["rho"], pvalue=result["pvalue"],
                title=f"Scenario {sc['id']} ({sc['name']})",
            )
            out.save_figure(fig, f"scatter_{sc['id']}.png", also_pdf=True)
            plt.close(fig)

        summary.append({
            "scenario_id": sc["id"],
            "rho": result["rho"], "pvalue": result["pvalue"],
            "n": result["n"],
        })

    out.save_table(pd.DataFrame(summary), "correlation_summary.csv", also_xlsx=True)
    out.add_summary("scenarios", summary)
    out.close()


if __name__ == "__main__":
    main()
