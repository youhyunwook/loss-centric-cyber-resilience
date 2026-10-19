"""
experiments/exp16_pagerank_degree_betweenness_ablation.py
---------------------------------------------------------
PageRank ablation experiment for the revised S1/S2 evaluation.

This script compares the manuscript's default PageRank-based asset
importance I(n) against two simpler alternatives:

  * degree      : in-degree centrality
  * betweenness : shortest-path betweenness centrality

The graph, exposure E(n), vulnerability V(S), impact range R(t,K), loss
formula, and countermeasure scoring are kept identical to the main
pipeline. Only the centrality method used for I(n) is changed.

Output:
    results/exp16_pagerank_degree_betweenness_ablation/
        tables/
            centrality_rho_<scenario>.csv
            top10_overlap_<scenario>.csv
            loss_and_best_cm_<scenario>.csv
            combined_loss_and_best_cm.csv
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from scipy import stats

from src.countermeasure.d3fend_mapper import D3FENDMapper
from src.countermeasure.selector import score_countermeasures
from src.graph.centrality import compute_importance
from src.risk.loss import compute_loss, compute_risk
from src.utils.logging import get_logger

from ._common import (
    DEFAULT_COUNTERMEASURES,
    load_default_artifacts,
    make_experiment_output,
    prepare_scenario_pipeline,
)


log = get_logger("exp16")

CENTRALITIES = ("pagerank", "degree", "betweenness")
TOPK = 10


def _topk_nodes(scores: dict[str, float], k: int = TOPK) -> set[str]:
    return {
        node
        for node, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    }


def _topk_jaccard(a: dict[str, float], b: dict[str, float], k: int = TOPK) -> float:
    top_a = _topk_nodes(a, k)
    top_b = _topk_nodes(b, k)
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0


@click.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--scenarios", "scenario_ids", default="S1,S2", show_default=True)
def main(data_path: str, scenario_ids: str) -> None:
    config, scenarios = load_default_artifacts()
    chosen = {s.strip() for s in scenario_ids.split(",") if s.strip()}
    scenarios = [s for s in scenarios if s["id"] in chosen]

    out = make_experiment_output(
        "exp16_pagerank_degree_betweenness_ablation",
        config=config,
        scenarios=[s["id"] for s in scenarios],
    )
    mapper = D3FENDMapper(str(DEFAULT_COUNTERMEASURES))
    benign_cache: dict = {}
    combined_rows: list[dict] = []

    pr_kwargs = config["centrality"].get("pagerank") or {}

    for sc in scenarios:
        sc_id = sc["id"]
        log.info("--- %s ---", sc_id)
        ctx = prepare_scenario_pipeline(sc, config, Path(data_path), benign_cache)
        G = ctx["G"]

        importance_by_method: dict[str, dict[str, float]] = {}
        for method in CENTRALITIES:
            kwargs = pr_kwargs if method == "pagerank" else {}
            importance_by_method[method] = compute_importance(
                G,
                method=method,
                normalize=config["centrality"]["normalize"],
                **kwargs,
            )

        nodes = sorted(set().union(*(scores.keys() for scores in importance_by_method.values())))

        rho = pd.DataFrame(index=CENTRALITIES, columns=CENTRALITIES, dtype=float)
        for m1 in CENTRALITIES:
            for m2 in CENTRALITIES:
                a = [importance_by_method[m1].get(node, 0.0) for node in nodes]
                b = [importance_by_method[m2].get(node, 0.0) for node in nodes]
                if not a or len(set(a)) <= 1 or len(set(b)) <= 1:
                    rho.loc[m1, m2] = float("nan")
                else:
                    rho.loc[m1, m2] = float(stats.spearmanr(a, b).correlation)
        out.save_table(rho, f"centrality_rho_{sc_id}.csv", index=True)

        overlap = pd.DataFrame(index=CENTRALITIES, columns=CENTRALITIES, dtype=float)
        for m1 in CENTRALITIES:
            for m2 in CENTRALITIES:
                overlap.loc[m1, m2] = _topk_jaccard(
                    importance_by_method[m1],
                    importance_by_method[m2],
                    k=TOPK,
                )
        out.save_table(overlap, f"top10_overlap_{sc_id}.csv", index=True)

        scenario_rows: list[dict] = []
        cands = mapper.for_scenario(sc_id)
        for method, importance in importance_by_method.items():
            risk = compute_risk(
                importance,
                ctx["exposure"],
                ctx["V_s"],
                nodes=ctx["R_sK"],
            )
            loss_before = compute_loss(risk, ctx["R_sK"])
            scored = score_countermeasures(
                candidates=cands,
                importance=importance,
                exposure=ctx["exposure"],
                vulnerability=ctx["V_s"],
                impact_range=ctx["R_sK"],
                loss_before=loss_before,
                config=config,
                target=ctx["targets"],
                G=G,
            )
            best = scored.iloc[0] if not scored.empty else None
            row = {
                "scenario_id": sc_id,
                "centrality": method,
                "n_nodes": int(G.number_of_nodes()),
                "n_edges": int(G.number_of_edges()),
                "impact_range_size": int(len(ctx["R_sK"])),
                "V_s": float(ctx["V_s"]),
                "loss_before": float(loss_before),
                "best_cm": best["cid"] if best is not None else None,
                "best_score": float(best["score"]) if best is not None else 0.0,
                "loss_after_best": float(best["loss_after"]) if best is not None else loss_before,
                "dLoss_best": float(best["dLoss"]) if best is not None else 0.0,
            }
            scenario_rows.append(row)
            combined_rows.append(row)

        out.save_table(pd.DataFrame(scenario_rows), f"loss_and_best_cm_{sc_id}.csv")

    out.save_table(pd.DataFrame(combined_rows), "combined_loss_and_best_cm.csv")
    out.add_summary("centralities", list(CENTRALITIES))
    out.add_summary("topk", TOPK)
    out.close()


if __name__ == "__main__":
    main()
