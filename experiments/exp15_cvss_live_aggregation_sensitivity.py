"""
experiments/exp15_cvss_live_aggregation_sensitivity.py
-------------------------------------------------------
CVSS aggregation sensitivity using the live NVD-derived S1/S2 values.

This experiment keeps the original graph, PageRank, exposure, loss, and
countermeasure-scoring pipeline unchanged. Only V(S) is varied according
to the max/mean/median values measured from the current NVD CWE pools.

Output:
    results/exp15_cvss_live_aggregation_sensitivity/
        tables/ v_s_inputs.csv
                loss_per_method.csv
                countermeasure_scores_per_method.csv
                best_cm_per_method.csv
                sensitivity_pivot.csv
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from src.countermeasure.d3fend_mapper import D3FENDMapper
from src.countermeasure.selector import score_countermeasures
from src.risk.loss import compute_loss, compute_risk
from src.utils.logging import get_logger

from ._common import (
    DEFAULT_COUNTERMEASURES,
    load_default_artifacts,
    make_experiment_output,
    prepare_scenario_pipeline,
)

log = get_logger("exp15")


LIVE_NVD_VS = {
    "S1": {
        "max": {"cvss": 9.9000, "v_s": 0.9900, "pool_size": 325},
        "mean": {"cvss": 7.9960, "v_s": 0.7996, "pool_size": 325},
        "median": {"cvss": 8.1000, "v_s": 0.8100, "pool_size": 325},
    },
    "S2": {
        "max": {"cvss": 9.9000, "v_s": 0.9900, "pool_size": 964},
        "mean": {"cvss": 6.7368, "v_s": 0.6737, "pool_size": 964},
        "median": {"cvss": 7.1000, "v_s": 0.7100, "pool_size": 964},
    },
}


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
def main(data_path: str, scenario_ids: str) -> None:
    config, scenarios = load_default_artifacts()
    chosen = {s.strip() for s in scenario_ids.split(",") if s.strip()}
    scenarios = [s for s in scenarios if s["id"] in chosen and s["id"] in LIVE_NVD_VS]

    out = make_experiment_output(
        "exp15_cvss_live_aggregation_sensitivity",
        config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    mapper = D3FENDMapper(str(DEFAULT_COUNTERMEASURES))
    benign_cache: dict = {}

    input_rows: list[dict] = []
    loss_rows: list[dict] = []
    score_rows: list[dict] = []
    best_rows: list[dict] = []

    for sc in scenarios:
        sc_id = sc["id"]
        log.info("--- %s ---", sc_id)
        ctx = prepare_scenario_pipeline(sc, config, Path(data_path), benign_cache)
        cands = mapper.for_scenario(sc_id)

        for method, values in LIVE_NVD_VS[sc_id].items():
            v_s = float(values["v_s"])
            risk = compute_risk(ctx["importance"], ctx["exposure"], v_s, nodes=ctx["R_sK"])
            loss_before = compute_loss(risk, ctx["R_sK"])
            scored = score_countermeasures(
                candidates=cands,
                importance=ctx["importance"],
                exposure=ctx["exposure"],
                vulnerability=v_s,
                impact_range=ctx["R_sK"],
                loss_before=loss_before,
                config=config,
                target=ctx["targets"],
                G=ctx["G"],
            )

            input_rows.append({
                "scenario_id": sc_id,
                "method": method,
                "cvss_pool_size": int(values["pool_size"]),
                "aggregated_cvss": float(values["cvss"]),
                "V_s": v_s,
            })
            loss_rows.append({
                "scenario_id": sc_id,
                "method": method,
                "V_s": v_s,
                "loss_before": float(loss_before),
                "n_nodes": int(ctx["G"].number_of_nodes()),
                "n_edges": int(ctx["G"].number_of_edges()),
                "impact_range_size": int(len(ctx["R_sK"])),
            })

            if scored.empty:
                best_rows.append({
                    "scenario_id": sc_id,
                    "method": method,
                    "best_cm": None,
                    "best_score": 0.0,
                    "loss_after_best": float(loss_before),
                    "dLoss_best": 0.0,
                })
                continue

            scored = scored.copy()
            scored.insert(0, "scenario_id", sc_id)
            scored.insert(1, "method", method)
            scored.insert(2, "V_s", v_s)
            score_rows.extend(scored.to_dict(orient="records"))

            best = scored.iloc[0]
            best_rows.append({
                "scenario_id": sc_id,
                "method": method,
                "best_cm": best["cid"],
                "best_score": float(best["score"]),
                "loss_after_best": float(best["loss_after"]),
                "dLoss_best": float(best["dLoss"]),
            })

    inputs_df = pd.DataFrame(input_rows)
    loss_df = pd.DataFrame(loss_rows)
    scores_df = pd.DataFrame(score_rows)
    best_df = pd.DataFrame(best_rows)

    out.save_table(inputs_df, "v_s_inputs.csv")
    out.save_table(loss_df, "loss_per_method.csv")
    out.save_table(scores_df, "countermeasure_scores_per_method.csv")
    out.save_table(best_df, "best_cm_per_method.csv")

    if not loss_df.empty:
        pivot = loss_df.pivot(index="scenario_id", columns="method", values="loss_before")
        out.save_table(pivot, "sensitivity_pivot.csv", index=True)

    out.add_summary("v_s_source", "NVD live CWE pool results measured on 2026-05-03")
    out.add_summary("methods", ["max", "mean", "median"])
    out.close()


if __name__ == "__main__":
    main()
