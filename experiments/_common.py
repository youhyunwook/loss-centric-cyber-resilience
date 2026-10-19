"""
experiments/_common.py
-----------------------
Shared scaffolding so experiments don't reimplement pipeline plumbing.
Notebook-aligned pipeline:
   load -> hygiene -> separate -> exclude_attempted -> internal_only
   -> build graph (with backbone) -> PageRank (log1p) -> ER (log1p+minmax)
   -> V(S) (scenario CVSS proxy or CWE pool)
   -> K-hop impact range
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.builder import build_communication_graph, k_hop_subgraph
from src.graph.centrality import compute_importance
from src.graph.exposure import compute_exposure
from src.preprocessing.augmentation import augment_multi_hop
from src.preprocessing.data_loader import load_cic_ids_2018, load_scenarios
from src.preprocessing.flow_filter import (
    drop_invalid_flows, exclude_attempted_flows,
    filter_flows, filter_internal_destination_only, filter_internal_only,
    separate_benign_attack,
)
from src.risk.attack_mapping import build_threat_table, scenario_cwe_pool
from src.risk.cvss_aggregation import aggregate_cvss
from src.risk.loss import compute_loss, compute_risk
from src.utils.io import load_config
from src.utils.results import ExperimentOutput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_SCENARIOS = PROJECT_ROOT / "config" / "scenarios.yaml"
DEFAULT_COUNTERMEASURES = PROJECT_ROOT / "config" / "countermeasures.yaml"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def load_default_artifacts() -> tuple[dict, list[dict]]:
    """Return (config, active_scenarios)."""
    return load_config(DEFAULT_CONFIG), load_scenarios(DEFAULT_SCENARIOS)


def make_experiment_output(
    name: str,
    config: dict | None = None,
    scenarios: list[str] | None = None,
) -> ExperimentOutput:
    return ExperimentOutput(
        name=name,
        results_root=DEFAULT_RESULTS,
        config=config,
        scenarios=scenarios,
    )


def resolve_v_s(
    scenario: dict,
    config: dict,
) -> tuple[float, int]:
    """Resolve V(S) for a scenario.

    Priority:
      1. If config.risk.use_scenario_cvss_proxy is True and scenario has
         a ``cvss_proxy`` field, use that (notebook LABEL_CVSS / 10.0).
      2. Otherwise aggregate the CWE -> CVSS pool via ``aggregate_cvss``.

    Returns
    -------
    (V_s, pool_size)
    """
    if config["risk"].get("use_scenario_cvss_proxy", False) and "cvss_proxy" in scenario:
        v = float(scenario["cvss_proxy"]) / float(config["risk"]["cvss_normalizer"])
        return min(v, 1.0), 1

    threat_table = build_threat_table(
        [scenario],
        cwe_csv=config["paths"]["cwe_csv"],
        cvss_csv=config["paths"]["cvss_csv"],
    )
    pool = scenario_cwe_pool(threat_table, scenario["id"])["cvss"]
    v = aggregate_cvss(
        pool,
        method=config["risk"]["cvss_aggregation"],
        normalizer=config["risk"]["cvss_normalizer"],
    )
    return v, len(pool)


def prepare_scenario_pipeline(
    scenario: dict,
    config: dict,
    cic_path: Path,
    benign_cache: dict | None = None,
):
    """Run the shared pipeline pre-amble.

    Notebook-aligned:
      1. load CSV
      2. drop_invalid + drop_duplicates
      3. separate benign / attack
      4. exclude *-Attempted (no-op on slim file)
      5. internal-only filter on benign (RFC1918)
      6. build graph with backbone extraction (seeds = scenario targets
         + same-/24 internal hosts)
      7. PageRank (log1p weight)
      8. ER (log1p + min-max)
      9. K-hop impact range from scenario targets

    The optional ``benign_cache`` lets sweep experiments avoid reloading
    the CSV between scenarios.
    """
    if benign_cache and "benign" in benign_cache and "attack" in benign_cache:
        benign = benign_cache["benign"]
        attack = benign_cache["attack"]
        benign_exposure = benign_cache.get("benign_exposure", benign)
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
        benign_exposure = benign
        exposure_scope = config.get("exposure", {}).get("flow_scope", "graph_benign")
        internal_networks = config["preprocessing"].get("internal_networks")
        if exposure_scope == "benign_internal_destination":
            benign_exposure = filter_internal_destination_only(
                benign,
                internal_networks=internal_networks,
            )
        if config["preprocessing"].get("internal_only", False):
            benign = filter_internal_only(
                benign,
                internal_networks=internal_networks,
            )
        if config["preprocessing"]["augmentation"]["enabled"]:
            benign = augment_multi_hop(
                benign,
                max_hops=config["preprocessing"]["augmentation"]["max_hops"],
                time_window_seconds=config["preprocessing"]["augmentation"]["time_window_seconds"],
            )
        if benign_cache is not None:
            benign_cache["benign"] = benign
            benign_cache["attack"] = attack
            benign_cache["benign_exposure"] = benign_exposure

    targets = list(scenario.get("targets") or [])
    if not targets:
        # fall back to data-driven extraction
        scenario_attack = filter_flows(attack, scenario["cic_labels"])
        targets = sorted(scenario_attack["Dst IP"].dropna().astype(str).unique().tolist())

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

    pr_kwargs = config["centrality"].get("pagerank") or {}
    importance = compute_importance(
        G,
        method=config["centrality"]["method"],
        normalize=config["centrality"]["normalize"],
        **pr_kwargs,
    )

    exposure = compute_exposure(
        G, flows=benign_exposure,
        method=config["exposure"]["method"],
        normalize=config["exposure"]["normalize"],
        internal_networks=config["preprocessing"].get("internal_networks"),
    )

    R_sK = k_hop_subgraph(G, targets, K=config["graph"]["impact_range_K"])

    V_s, pool_size = resolve_v_s(scenario, config)

    return {
        "benign": benign, "benign_exposure": benign_exposure,
        "attack": attack, "G": G,
        "importance": importance, "exposure": exposure,
        "targets": targets, "R_sK": R_sK,
        "V_s": V_s, "cvss_pool_size": pool_size,
    }
