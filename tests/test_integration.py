"""
tests/test_integration.py
--------------------------
End-to-end integration test using a synthetic CIC-IDS-2018-style CSV.

PURPOSE: code-integrity verification only. The synthetic data is NOT
used in any manuscript figure, table, or claim. It exists so that
reviewers, CI systems, and future contributors can verify the pipeline
links together correctly without downloading the ~62M-row Improved
CSE-CIC-IDS-2018 dataset.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_synthetic_cic(path: Path) -> None:
    rows = []
    benign_pairs = [
        ("10.0.0.1", "10.0.0.2"), ("10.0.0.1", "10.0.0.3"),
        ("10.0.0.2", "10.0.0.3"), ("10.0.0.3", "10.0.0.4"),
        ("10.0.0.4", "10.0.0.5"), ("10.0.0.5", "10.0.0.1"),
        ("10.0.0.2", "10.0.0.5"), ("10.0.0.6", "10.0.0.3"),
    ]
    for src, dst in benign_pairs:
        for _ in range(5):
            rows.append({
                "Src IP": src, "Src Port": 12345, "Dst IP": dst,
                "Dst Port": 80, "Protocol": 6,
                "Timestamp": "2018-02-14 10:00:00",
                "Label": "BENIGN",
            })
    # External egress flow that internal_only must reject
    for _ in range(5):
        rows.append({
            "Src IP": "10.0.0.1", "Src Port": 22222, "Dst IP": "8.8.8.8",
            "Dst Port": 53, "Protocol": 17,
            "Timestamp": "2018-02-14 10:00:00",
            "Label": "BENIGN",
        })
    # SSH attack
    for i in range(20):
        rows.append({
            "Src IP": f"172.31.0.{i+1}", "Src Port": 54321,
            "Dst IP": "10.0.0.3", "Dst Port": 22, "Protocol": 6,
            "Timestamp": "2018-02-14 11:00:00",
            "Label": "SSH-BruteForce",
        })
    # Improved-dataset Attempted label that must be excluded
    for _ in range(3):
        rows.append({
            "Src IP": "172.31.0.50", "Src Port": 11111,
            "Dst IP": "10.0.0.4", "Dst Port": 21, "Protocol": 6,
            "Timestamp": "2018-02-14 11:30:00",
            "Label": "FTP-BruteForce-Attempted",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_end_to_end_synthetic():
    """Run pipeline end-to-end on synthetic data."""
    from src.countermeasure.d3fend_mapper import D3FENDMapper
    from src.countermeasure.selector import score_countermeasures
    from src.graph.builder import build_communication_graph, k_hop_subgraph
    from src.graph.centrality import compute_importance
    from src.graph.exposure import compute_exposure
    from src.preprocessing.data_loader import load_cic_ids_2018, load_scenarios
    from src.preprocessing.flow_filter import (
        drop_invalid_flows, exclude_attempted_flows,
        filter_internal_only, separate_benign_attack,
    )
    from src.risk.cvss_aggregation import aggregate_cvss
    from src.risk.loss import compute_loss, compute_risk
    from src.utils.io import load_config

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "default.yaml")
    scenarios = load_scenarios(project_root / "config" / "scenarios.yaml")
    s1 = next(s for s in scenarios if s["id"] == "S1")

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "synthetic.csv"
        _write_synthetic_cic(csv_path)

        # Preprocessing
        raw = load_cic_ids_2018(csv_path)
        raw = drop_invalid_flows(raw)
        benign, attack = separate_benign_attack(raw, benign_label="BENIGN")
        assert not benign.empty and not attack.empty

        n_attack_before = len(attack)
        attack = exclude_attempted_flows(attack)
        assert len(attack) < n_attack_before

        n_benign_before = len(benign)
        benign = filter_internal_only(benign)
        assert len(benign) < n_benign_before  # 8.8.8.8 dropped

        # Use the scenario's targets directly (multi-target supported)
        targets = list(s1.get("targets") or ["10.0.0.3"])

        # Backbone-aware graph
        G = build_communication_graph(
            benign, edge_threshold_tau=1,
            target_hosts=targets,
            use_backbone=True, node_budget=100, max_hops=4,
        )
        assert G.number_of_nodes() > 0

        # Centrality + exposure
        importance = compute_importance(G, method="pagerank", weight_transform="log1p")
        exposure = compute_exposure(G, flows=benign, normalize="log1p_minmax")
        assert all(0.0 <= v <= 1.0 for v in importance.values())
        assert all(0.0 <= v <= 1.0 for v in exposure.values())

        # V(S) - use proxy
        V_s = float(s1["cvss_proxy"]) / 10.0

        # Loss
        # Use synthetic targets if real ones aren't in our tiny graph
        targets_in_graph = [t for t in targets if t in G]
        if not targets_in_graph:
            targets_in_graph = ["10.0.0.3"]   # synthetic SSH target
        R_sK = k_hop_subgraph(G, targets_in_graph, K=3)
        risk = compute_risk(importance, exposure, V_s, nodes=R_sK)
        loss = compute_loss(risk, R_sK)
        assert loss >= 0.0

        # CM scoring (use S1 catalogue)
        mapper = D3FENDMapper(str(project_root / "config" / "countermeasures.yaml"))
        cands = mapper.for_scenario("S1")
        assert len(cands) > 0
        scored = score_countermeasures(
            candidates=cands,
            importance=importance, exposure=exposure,
            vulnerability=V_s, impact_range=R_sK,
            loss_before=loss, config=config,
            target=targets_in_graph[0], G=G,
        )
        assert not scored.empty
        assert scored["score"].is_monotonic_decreasing
