"""
experiments/exp06_countermeasure_parameter_sensitivity.py
---------------------------------------------------------
Countermeasure-parameter sensitivity (paper Table 12).

For each scenario, every countermeasure's residual ratio rr is varied over
five evenly spaced values inside its effectiveness tier
(High: 0.05-0.25, Medium: 0.30-0.50; ``config/default.yaml``), and all
combinations are enumerated (5^4 = 625 per scenario). For each combination
the top-ranked countermeasure is recorded under both ΔLoss and
Score = ΔLoss / (D + 1). D(c) is NOT varied.

Ties are resolved by catalogue order in ``config/countermeasures.yaml``
(first candidate wins), identical to ``np.argmax`` in the original
notebook (``notebooks/Experiment_v5.ipynb``, sensitivity cell).

Output:
    results/exp06_countermeasure_parameter_sensitivity/
        tables/   rank1_<scenario>.csv          (Table 12 rates)
                  combinations_<scenario>.csv   (all 625 combinations)
                  rr_grid_dloss_<scenario>.csv  (ΔLoss / Score per rr value)
                  table12_summary.csv (+xlsx)
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import replace
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.countermeasure.d3fend_mapper import D3FENDMapper, CountermeasureSpec
from src.countermeasure.selector import score_countermeasures
from src.risk.loss import compute_loss, compute_risk
from src.utils.logging import get_logger

from ._common import (
    DEFAULT_COUNTERMEASURES,
    load_default_artifacts,
    make_experiment_output,
    prepare_scenario_pipeline,
)

log = get_logger("exp06")

# Operation parameter that carries the residual ratio rr.
RR_PARAM_KEYS = ("V_scale", "E_scale", "I_scale")


def _rr_key(spec: CountermeasureSpec) -> str:
    for k in RR_PARAM_KEYS:
        if k in spec.params:
            return k
    raise ValueError(f"{spec.cid}: no residual-ratio parameter in {spec.params}")


def _tier_grid(config: dict, tier: str, n_points: int) -> list[float]:
    band = config["countermeasure"]["effectiveness_tiers"][tier]
    return [round(float(v), 10) for v in np.linspace(band["r_min"], band["r_max"], n_points)]


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
def main(data_path: str, scenario_ids: str) -> None:
    config, scenarios = load_default_artifacts()
    chosen = set(scenario_ids.split(","))
    scenarios = [s for s in scenarios if s["id"] in chosen]
    n_points = int(config["sensitivity"]["residual"]["grid_points"])

    out = make_experiment_output(
        "exp06_countermeasure_parameter_sensitivity", config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    mapper = D3FENDMapper(str(DEFAULT_COUNTERMEASURES))
    benign_cache: dict = {}
    summary_rows: list[dict] = []

    for sc in scenarios:
        sid = sc["id"]
        log.info("--- %s ---", sid)
        ctx = prepare_scenario_pipeline(sc, config, Path(data_path), benign_cache)
        risk = compute_risk(ctx["importance"], ctx["exposure"], ctx["V_s"], nodes=ctx["R_sK"])
        loss_before = compute_loss(risk, ctx["R_sK"])
        cands = mapper.for_scenario(sid)

        # ΔLoss / Score of each candidate at each rr value of its tier grid.
        grids: list[list[float]] = []
        dl_table: list[dict[float, float]] = []
        sc_table: list[dict[float, float]] = []
        grid_rows: list[dict] = []
        for spec in cands:
            key = _rr_key(spec)
            grid = _tier_grid(config, spec.tier, n_points)
            grids.append(grid)
            dl_map, sc_map = {}, {}
            for rr in grid:
                variant = replace(spec, params={**spec.params, key: rr})
                row = score_countermeasures(
                    candidates=[variant],
                    importance=ctx["importance"], exposure=ctx["exposure"],
                    vulnerability=ctx["V_s"], impact_range=ctx["R_sK"],
                    loss_before=loss_before, config=config,
                    target=ctx["targets"], G=ctx["G"],
                ).iloc[0]
                dl_map[rr], sc_map[rr] = float(row["dLoss"]), float(row["score"])
                grid_rows.append({"cid": spec.cid, "tier": spec.tier, "rr": rr,
                                  "D": float(row["D"]), "dLoss": dl_map[rr],
                                  "score": sc_map[rr]})
            dl_table.append(dl_map)
            sc_table.append(sc_map)
        out.save_table(pd.DataFrame(grid_rows), f"rr_grid_dloss_{sid}.csv")

        # Enumerate all combinations; first candidate wins ties (np.argmax).
        rank1_dl: Counter = Counter()
        rank1_sc: Counter = Counter()
        combo_rows: list[dict] = []
        for combo in itertools.product(*grids):
            dls = [dl_table[i][rr] for i, rr in enumerate(combo)]
            scs = [sc_table[i][rr] for i, rr in enumerate(combo)]
            best_dl = cands[int(np.argmax(dls))].cid
            best_sc = cands[int(np.argmax(scs))].cid
            rank1_dl[best_dl] += 1
            rank1_sc[best_sc] += 1
            combo_rows.append({**{f"rr_{c.cid}": rr for c, rr in zip(cands, combo)},
                               "top_dLoss": best_dl, "top_score": best_sc})
        total = len(combo_rows)
        out.save_table(pd.DataFrame(combo_rows), f"combinations_{sid}.csv")

        names = {c.cid: c.name for c in cands}
        rank_rows = []
        for criterion, counter in (("dLoss", rank1_dl), ("score", rank1_sc)):
            for cid, cnt in counter.most_common():
                rank_rows.append({"scenario_id": sid, "criterion": criterion,
                                  "cid": cid, "name": names[cid],
                                  "count": cnt, "n_combinations": total,
                                  "rate_pct": 100.0 * cnt / total})
        out.save_table(pd.DataFrame(rank_rows), f"rank1_{sid}.csv")
        summary_rows.extend(rank_rows)
        log.info("%s (%d combos) dLoss: %s | score: %s", sid, total,
                 dict(rank1_dl), dict(rank1_sc))

    out.save_table(pd.DataFrame(summary_rows), "table12_summary.csv", also_xlsx=True)
    out.add_summary("n_grid_points_per_tier", n_points)
    out.add_summary("tie_break", "catalogue order (first candidate wins), as np.argmax")
    out.close()


if __name__ == "__main__":
    main()
